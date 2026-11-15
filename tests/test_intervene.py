# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""spectral_trust.interventions on tiny random models built offline (no
download): graph-frequency steering (symmetric and causal), head ablation,
sink-preserving edge removal, singular-value rescaling and patching.

The reference implementations of the spectral-steering rescaling, of the
spectral probe's SNR and of the 0.2.3 branch's weight_snr are copied
verbatim below, so that the library is checked against them and not against
itself."""
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust import GSPConfig, SpectralModel  # noqa: E402
from spectral_trust.interventions import (  # noqa: E402
    PatchSpec, Patcher, attention_adjacency, equal_count_bands, filter_coefficients,
    filter_response, graph_filter, haar_basis, laplacian, path_adjacency,
    remove_edges_from_pattern, select_modes, svd_scale_matrix, weight_snr,
)
from spectral_trust.interventions._hooks import register_site_hook  # noqa: E402

torch.set_num_threads(2)
CFG = GSPConfig(device="cpu", verbose=False, save_plots=False)
V = 40


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=3, n_head=4, n_embd=32, vocab_size=V, n_positions=64,
                     bos_token_id=0, eos_token_id=0, pad_token_id=0)
    return GPT2LMHeadModel(cfg).eval()


@pytest.fixture(scope="module")
def llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1)
    cfg = LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      bos_token_id=0, eos_token_id=0, pad_token_id=0)
    return LlamaForCausalLM(cfg).eval()


def _ids(seed, n=12, batch=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (batch, n), generator=g)


def _logits(model, ids, **kw):
    with torch.no_grad():
        return model(input_ids=ids, **kw).logits


def _eager_logits(model, ids):
    """Logits with eager attention, the implementation Patcher runs."""
    impl = model.config._attn_implementation
    model.set_attn_implementation("eager")
    try:
        return _logits(model, ids)
    finally:
        model.set_attn_implementation(impl)


def _forward_attn(model, ids):
    impl = model.config._attn_implementation
    model.set_attn_implementation("eager")
    try:
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, output_hidden_states=True)
    finally:
        model.set_attn_implementation(impl)
    return out


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _causal_attention(B, H, S, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, H, S, S, generator=g) * 2
    logits[..., 0] += 3.0                                    # a first-token sink, as in LMs
    mask = torch.ones(S, S, dtype=torch.bool).tril()
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def _rayleigh(X, L):
    """Dirichlet energy tr(X^T L X) and its normalised form / ||X||_F^2."""
    X64, L64 = X.double(), L.double()
    E = float((X64 * (L64 @ X64)).sum())
    return E, E / float((X64 * X64).sum())


# ── responses ────────────────────────────────────────────────────────────────

def test_coefficients_bandstop_empty_is_identity_and_kinds_partition():
    c = filter_coefficients("bandstop", band=(0.7, 0.7), order=32)
    assert c[0] == 1.0 and not np.any(c[1:])
    lam = np.linspace(0, 2, 101)
    lo = filter_response(lam, "lowpass", cutoff=0.8)
    hi = filter_response(lam, "highpass", cutoff=0.8)
    np.testing.assert_allclose(lo + hi, 1.0, atol=1e-12)
    assert lo[0] > 0.99 and lo[-1] < 1e-3 and hi[0] < 1e-3 and hi[-1] > 0.99
    bp = filter_response(lam, "bandpass", band=(0.5, 1.5))
    bs = filter_response(lam, "bandstop", band=(0.5, 1.5))
    np.testing.assert_allclose(bp + bs, 1.0, atol=1e-12)
    assert np.all(bp >= -1e-9) and np.all(bp <= 1 + 1e-9)       # Jackson: no Gibbs overshoot
    with pytest.raises(ValueError):
        filter_coefficients("lowpass", band=(0, 1))
    with pytest.raises(ValueError):
        filter_coefficients("bandpass", cutoff=1.0)


# ── graph filter on tensors ──────────────────────────────────────────────────

def _graph_and_signal(S=14, d=6, seed=0):
    A = _causal_attention(1, 3, S, seed)
    X = torch.randn(1, S, d, generator=torch.Generator().manual_seed(seed + 100))
    return A, X


def test_strength_zero_is_identity_and_chebyshev_matches_polynomial():
    A, X = _graph_and_signal()
    assert torch.equal(graph_filter(X, A, kind="lowpass", cutoff=0.6, strength=0.0), X)
    assert torch.equal(graph_filter(X, A, kind="bandstop", band=(0.9, 0.9)), X)
    L = laplacian(attention_adjacency(A, CFG), CFG)[0].double()
    vals, vecs = torch.linalg.eigh(0.5 * (L + L.T))
    for kind, kw in [("lowpass", {"cutoff": 0.6}), ("highpass", {"cutoff": 0.6}),
                     ("bandpass", {"band": (0.4, 1.1)}), ("bandstop", {"band": (0.4, 1.1)})]:
        Y = graph_filter(X, A, kind=kind, order=48, **kw)
        p = torch.from_numpy(filter_response(vals.numpy(), kind, order=48, **kw))
        ref = vecs @ (p.unsqueeze(-1) * (vecs.T @ X[0].double()))
        np.testing.assert_allclose(Y[0].double().numpy(), ref.numpy(), atol=2e-5)
        half = graph_filter(X, A, kind=kind, order=48, strength=0.5, **kw)
        np.testing.assert_allclose(half.numpy(), (0.5 * (X + Y)).numpy(), atol=1e-5)


def test_dense_is_the_ideal_filter_and_chebyshev_converges_to_it():
    A, X = _graph_and_signal(seed=3)
    L = laplacian(attention_adjacency(A, CFG), CFG)[0].double()
    vals, vecs = torch.linalg.eigh(0.5 * (L + L.T))
    gaps = (vals[1:] + vals[:-1]) / 2
    cut = float(gaps[torch.argmax(vals[1:] - vals[:-1])])      # cutoff in the widest gap
    dense = graph_filter(X, A, kind="lowpass", cutoff=cut, method="dense")
    keep = (vals < cut).double()
    ref = vecs @ (keep.unsqueeze(-1) * (vecs.T @ X[0].double()))
    np.testing.assert_allclose(dense[0].double().numpy(), ref.numpy(), atol=1e-5)
    err = [float((graph_filter(X, A, kind="lowpass", cutoff=cut, order=o) - dense).norm())
           for o in (16, 64, 256)]
    assert err[2] < err[0] and err[2] < 0.05 * float(dense.norm())


def test_lowpass_lowers_and_highpass_raises_dirichlet_energy():
    A, X = _graph_and_signal(seed=5)
    L = laplacian(attention_adjacency(A, CFG), CFG)[0]
    E0, R0 = _rayleigh(X[0], L)
    E_lo, R_lo = _rayleigh(graph_filter(X, A, kind="lowpass", cutoff=0.8)[0], L)
    E_hi, R_hi = _rayleigh(graph_filter(X, A, kind="highpass", cutoff=0.8)[0], L)
    assert E_lo < E0 and R_lo < R0
    assert R_hi > R0
    # strength < 0 pushes away from the low-pass signal: raw energy rises too
    E_boost, _ = _rayleigh(graph_filter(X, A, kind="lowpass", cutoff=0.8, strength=-1.0)[0], L)
    assert E_boost > E0


def test_path_graph_callable_graph_and_token_span():
    A, X = _graph_and_signal(seed=7)
    Yp = graph_filter(X, kind="lowpass", cutoff=0.5, graph="path")
    Lp = laplacian(path_adjacency(X.shape[1]), CFG)[0]
    assert _rayleigh(Yp[0], Lp)[1] < _rayleigh(X[0], Lp)[1]
    W = attention_adjacency(A, CFG)
    Yc = graph_filter(X, A, kind="lowpass", cutoff=0.5, graph=lambda a, x: attention_adjacency(a, CFG))
    Yt = graph_filter(X, kind="lowpass", cutoff=0.5, graph=W[0])
    Ya = graph_filter(X, A, kind="lowpass", cutoff=0.5)
    np.testing.assert_allclose(Yc.numpy(), Ya.numpy(), atol=1e-6)
    np.testing.assert_allclose(Yt.numpy(), Ya.numpy(), atol=1e-6)
    Ys = graph_filter(X, A, kind="lowpass", cutoff=0.5, token_span=(1, None))
    assert torch.equal(Ys[:, 0], X[:, 0])
    sub = graph_filter(X[:, 1:], A[:, :, 1:, 1:], kind="lowpass", cutoff=0.5)
    np.testing.assert_allclose(Ys[:, 1:].numpy(), sub.numpy(), atol=1e-6)


def test_causal_filter_never_reads_future_positions():
    S, j = 14, 8
    A, X = _graph_and_signal(S=S, seed=11)
    A2, X2 = A.clone(), X.clone()
    X2[:, j:] += 5.0 * torch.randn(1, S - j, X.shape[-1])
    A2[:, :, j:] = _causal_attention(1, 3, S, 99)[:, :, j:]          # future rows differ
    for method in ("chebyshev", "dense"):
        Y = graph_filter(X, A, kind="lowpass", cutoff=0.7, causal=True, method=method)
        Y2 = graph_filter(X2, A2, kind="lowpass", cutoff=0.7, causal=True, method=method)
        np.testing.assert_allclose(Y[:, :j].numpy(), Y2[:, :j].numpy(), rtol=0, atol=1e-6)
        assert not torch.allclose(Y[:, j:], Y2[:, j:])
    Ys = graph_filter(X, A, kind="lowpass", cutoff=0.7)
    Ys2 = graph_filter(X2, A2, kind="lowpass", cutoff=0.7)
    assert not torch.allclose(Ys[:, :j], Ys2[:, :j], atol=1e-4)       # the symmetric one does


def test_causal_row_is_the_prefix_filter_and_agrees_at_the_last_position():
    S = 12
    A, X = _graph_and_signal(S=S, seed=13)
    Yc = graph_filter(X, A, kind="highpass", cutoff=0.9, causal=True)
    for i in (0, 3, 7, S - 1):
        prefix = graph_filter(X[:, :i + 1], A[:, :, :i + 1, :i + 1], kind="highpass", cutoff=0.9)
        np.testing.assert_allclose(Yc[:, i].numpy(), prefix[:, i].numpy(), atol=1e-5)
    Ys = graph_filter(X, A, kind="highpass", cutoff=0.9)
    np.testing.assert_allclose(Yc[:, -1].numpy(), Ys[:, -1].numpy(), atol=1e-5)


# ── graph filter inside the model ────────────────────────────────────────────

def test_resid_filter_strength_zero_identity_and_restored(gpt2):
    ids = _ids(0)
    before, params = _logits(gpt2, ids), _state(gpt2)
    impl = gpt2.config._attn_implementation
    with st.intervene(gpt2) as iv:
        iv.resid(1).graph_filter("lowpass", cutoff=0.5, strength=0.0)
        iv.resid(2).graph_filter("bandstop", band=(1.0, 1.0))
        assert torch.equal(_logits(gpt2, ids), _logits(gpt2, ids))
        np.testing.assert_allclose(_logits(gpt2, ids).numpy(), before.numpy(), atol=1e-5)
        iv.resid(1).graph_filter("lowpass", cutoff=0.5, strength=1.0)
        assert not torch.allclose(_logits(gpt2, ids), before, atol=1e-3)
    assert gpt2.config._attn_implementation == impl
    assert torch.equal(_logits(gpt2, ids), before)
    after = _state(gpt2)
    assert all(torch.equal(params[k], after[k]) for k in params)


def test_resid_filter_matches_functional_filter_and_lowers_energy(gpt2):
    ids = _ids(1)
    out = _forward_attn(gpt2, ids)
    lay = 1
    X, A = out.hidden_states[lay + 1], out.attentions[lay]
    got = {}
    with st.intervene(gpt2) as iv:
        iv.resid(lay).graph_filter("lowpass", cutoff=0.8)
        iv.hook("resid_post", lay, lambda x: got.__setitem__("x", x.clone()))
        _logits(gpt2, ids)
    ref = graph_filter(X, A, kind="lowpass", cutoff=0.8)
    np.testing.assert_allclose(got["x"].numpy(), ref.numpy(), atol=1e-5)
    L = laplacian(attention_adjacency(A, CFG), CFG)[0]
    assert _rayleigh(got["x"][0], L)[0] < _rayleigh(X[0], L)[0]
    with st.intervene(gpt2) as iv:
        iv.resid(lay).graph_filter("highpass", cutoff=0.8)
        iv.hook("resid_post", lay, lambda x: got.__setitem__("h", x.clone()))
        _logits(gpt2, ids)
    assert _rayleigh(got["h"][0], L)[1] > _rayleigh(X[0], L)[1]


def test_causal_filter_in_model_does_not_read_future_tokens(gpt2):
    ids = _ids(2, n=12)
    ids2 = ids.clone()
    ids2[0, 8:] = (ids2[0, 8:] % (V - 2)) + 1
    for causal, same in ((True, True), (False, False)):
        with st.intervene(gpt2) as iv:
            iv.resid(0).graph_filter("lowpass", cutoff=0.6, causal=causal)
            a, b = _logits(gpt2, ids), _logits(gpt2, ids2)
        assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5) == same


def test_causal_filter_generation_with_kv_cache_matches_recompute(gpt2, llama):
    for model in (gpt2, llama):
        ids = _ids(3, n=7)
        kw = dict(attention_mask=torch.ones_like(ids), max_new_tokens=6, do_sample=False,
                  output_scores=True, return_dict_in_generate=True, pad_token_id=0)
        with st.intervene(model) as iv:
            iv.resid(0).graph_filter("highpass", cutoff=0.7, strength=0.8, causal=True)
            iv.resid(1).graph_filter("lowpass", cutoff=0.5, causal=True, graph="path")
            with torch.no_grad():
                cached = model.generate(ids, use_cache=True, **kw)
                full = model.generate(ids, use_cache=False, **kw)
        assert torch.equal(cached.sequences, full.sequences)
        for s1, s2 in zip(cached.scores, full.scores):
            np.testing.assert_allclose(s1.numpy(), s2.numpy(), atol=1e-4)
        with st.intervene(model) as iv:
            iv.resid(0).graph_filter("lowpass", cutoff=0.6, causal=False)
            with pytest.raises(RuntimeError, match="causal=True"):
                with torch.no_grad():
                    model.generate(ids, use_cache=True, **kw)


def test_filter_ignores_right_padding(gpt2):
    ids = _ids(4, n=9)
    padded = torch.cat([ids, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 9), torch.zeros(1, 3)], dim=1).long()
    with st.intervene(gpt2) as iv:
        iv.resid(1).graph_filter("lowpass", cutoff=0.6)
        a = _logits(gpt2, ids)
        b = _logits(gpt2, padded, attention_mask=mask)
    np.testing.assert_allclose(a.numpy(), b[:, :9].numpy(), atol=1e-4)


def test_spectral_model_wrapper_and_add_steering(gpt2):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = [f"w{i}" for i in range(V)]
    tk = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="w0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="w0", eos_token="w0")
    impl = gpt2.config._attn_implementation
    sm = SpectralModel(gpt2, tok)                    # switches the model to eager for good
    ids = _ids(5)
    vec = torch.randn(32)
    got = {}
    try:
        with st.intervene(sm) as iv:
            iv.hook("resid_post", 0, lambda x: got.__setitem__("pre", x.clone()))
            iv.resid(0).add(vec, scale=2.0, positions=[1, 4])
            iv.hook("resid_post", 0, lambda x: got.__setitem__("post", x.clone()))
            _logits(gpt2, ids)
            assert iv.tokenizer is tok and iv.config is sm.config
    finally:
        gpt2.set_attn_implementation(impl)
    delta = got["post"] - got["pre"]
    assert torch.allclose(delta[0, 1], 2.0 * vec) and torch.allclose(delta[0, 4], 2.0 * vec)
    assert torch.equal(delta[0, 0], torch.zeros(32))


# ── ablation and edge removal ────────────────────────────────────────────────

def _z(model, ids, layer, arch):
    got = {}
    h = register_site_hook(arch, "z", layer, lambda x: got.__setitem__("z", x.clone()))
    try:
        _logits(model, ids)
    finally:
        for x in h:
            x.remove()
    return got["z"]


def test_head_ablation_zero_mean_resample(gpt2):
    from spectral_trust.interventions._arch import Arch
    arch = Arch(gpt2)
    ids, ref = _ids(6), _ids(7, batch=3)
    dh = arch.head_dim(1)
    z_clean = _z(gpt2, ids, 1, arch)
    z_ref = _z(gpt2, ref, 1, arch)
    cases = {
        "zero": lambda z: z.index_fill(-1, torch.arange(2 * dh, 3 * dh), 0.0),
        "mean": lambda z: torch.cat([z[..., :2 * dh],
                                     z_ref[..., 2 * dh:3 * dh].mean(dim=(0, 1)).expand(1, 12, dh),
                                     z[..., 3 * dh:]], -1),
    }
    for mode, manual in cases.items():
        got = {}
        with st.intervene(gpt2) as iv:
            iv.attention(1, 2).ablate(mode, reference=ref if mode == "mean" else None)
            iv.hook("z", 1, lambda x: got.__setitem__("z", x.clone()))
            a = _logits(gpt2, ids)
        np.testing.assert_allclose(got["z"].numpy(), manual(z_clean).numpy(), atol=1e-6)
        with st.intervene(gpt2) as iv:
            iv.hook("z", 1, manual)
            b = _logits(gpt2, ids)
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)
    got = {}
    with st.intervene(gpt2) as iv:
        iv.attention(1, [0, 3]).ablate("resample", reference=ref, seed=4)
        iv.hook("z", 1, lambda x: got.__setitem__("z", x.clone()))
        _logits(gpt2, ids)
    pick = torch.randint(3, (1,), generator=torch.Generator().manual_seed(4))
    for h in (0, 3):
        sl = slice(h * dh, (h + 1) * dh)
        assert torch.allclose(got["z"][0, :, sl], z_ref[pick[0], :12, sl])
    assert torch.allclose(got["z"][0, :, dh:2 * dh], z_clean[0, :, dh:2 * dh])


def test_ablate_all_heads_zero_leaves_only_the_bias(gpt2):
    got = {}
    with st.intervene(gpt2) as iv:
        iv.attention(2).ablate("zero")
        iv.hook("attn_out", 2, lambda x: got.__setitem__("o", x.clone()))
        _logits(gpt2, _ids(8))
    bias = gpt2.transformer.h[2].attn.c_proj.bias
    assert torch.allclose(got["o"], bias.expand_as(got["o"]))


def _pattern(model, ids, layer, arch):
    """Attention weights of one layer, read with eager attention (the
    implementation is put back afterwards)."""
    got = {}
    impl = model.config._attn_implementation
    model.set_attn_implementation("eager")
    h = arch.attention(layer).register_forward_hook(
        lambda m, a, o: got.__setitem__("w", o[1].detach().clone()))
    try:
        _logits(model, ids)
    finally:
        h.remove()
        model.set_attn_implementation(impl)
    return got["w"]


@pytest.mark.parametrize("which", ["gpt2", "llama"])
def test_remove_edges_keeps_sink_mass(which, gpt2, llama):
    from spectral_trust.interventions._arch import Arch
    model = gpt2 if which == "gpt2" else llama
    arch = Arch(model)
    ids = _ids(9, n=10)
    S = ids.shape[1]
    key_mask = torch.zeros(S, S, dtype=torch.bool)
    key_mask[5:, 2] = True
    key_mask[7:, 3:6] = True
    import sys
    modeling = sys.modules[type(arch.attention(1)).__module__]
    orig_fn, impl = modeling.eager_attention_forward, model.config._attn_implementation
    with st.intervene(model) as iv:
        w0 = _pattern(model, ids, 1, arch)
        iv.attention(1, 1).remove_edges(key_mask, keep_sink=True)
        assert modeling.eager_attention_forward is not orig_fn
        w1 = _pattern(model, ids, 1, arch)
        edited = _logits(model, ids)
    assert modeling.eager_attention_forward is orig_fn                # restored on exit
    assert model.config._attn_implementation == impl
    assert torch.equal(w1[:, [0, 2, 3]], w0[:, [0, 2, 3]])           # other heads untouched
    a0, a1 = w0[0, 1], w1[0, 1]
    assert torch.equal(a1[:, 0], a0[:, 0])                            # sink mass held fixed
    assert torch.equal(a1[:5], a0[:5])                                # untouched rows bit for bit
    assert torch.all(a1[key_mask] == 0)
    np.testing.assert_allclose(a1.sum(-1).numpy(), a0.sum(-1).numpy(), atol=1e-6)
    for i in range(5, S):
        kept = (~key_mask[i]) & (torch.arange(S) <= i)
        kept[0] = False
        ratio = a1[i, kept] / a0[i, kept]
        assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5)
        assert ratio[0] > 1
    with st.intervene(model) as iv:                                  # the output uses the edit
        def manual(w, qpos):
            return remove_edges_from_pattern(w, key_mask[qpos][None, None],
                                             torch.zeros(w.shape[0], dtype=torch.long))
        iv.attention(1, 1).edit_pattern(manual)
        np.testing.assert_allclose(_logits(model, ids).numpy(), edited.numpy(), atol=1e-6)
    with st.intervene(model) as iv:
        iv.attention(1, 1).remove_edges(key_mask, keep_sink=False)
        w2 = _pattern(model, ids, 1, arch)[0, 1]
    assert not torch.allclose(w2[5:, 0], a0[5:, 0])                  # plain renormalisation
    np.testing.assert_allclose(w2.sum(-1).numpy(), 1.0, atol=1e-6)
    with st.intervene(model) as iv:
        bad = torch.zeros(S, dtype=torch.bool)
        bad[0] = True
        iv.attention(1, 1).remove_edges(bad, keep_sink=True)
        with pytest.raises(ValueError, match="keep_sink"):
            _logits(model, ids)


def test_remove_edges_sink_rule_on_a_matrix():
    w = torch.tensor([[[[1.0, 0, 0, 0], [0.6, 0.4, 0, 0], [0.5, 0.2, 0.3, 0],
                        [0.4, 0.3, 0.2, 0.1]]]])
    rm = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    rm[0, 0, 3, 1] = True
    rm[0, 0, 2, 1:3] = True                                           # row 2: no non-sink key left
    out = remove_edges_from_pattern(w, rm, torch.zeros(1, dtype=torch.long))
    assert torch.allclose(out[0, 0, 3], torch.tensor([0.4, 0.0, 0.2 * 2, 0.1 * 2]))
    assert torch.allclose(out[0, 0, 2], torch.tensor([0.5, 0.0, 0.0, 0.0]))
    assert torch.equal(out[0, 0, :2], w[0, 0, :2])
    plain = remove_edges_from_pattern(w, rm, torch.zeros(1, dtype=torch.long), keep_sink=False)
    assert torch.allclose(plain[0, 0, 3], torch.tensor([0.4, 0.0, 0.2, 0.1]) / 0.7)


def _no_warnings(fn):
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn()
    assert not [r for r in rec if "keep_sink" in str(r.message)]
    return out


def test_remove_edges_warns_when_keep_sink_amplifies_the_kept_edges():
    # rows: [sink, removed, kept, kept]; the kept non-sink mass is 0.05 of 0.80
    w = torch.tensor([0.2, 0.75, 0.04, 0.01]).repeat(2, 3, 4, 1)          # [2, 3, 4, 4]
    rm = torch.zeros(1, 1, 1, 4, dtype=torch.bool)
    rm[..., 1] = True
    sink = torch.zeros(2, dtype=torch.long)
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = remove_edges_from_pattern(w, rm, sink)
    msgs = [r for r in rec if issubclass(r.category, UserWarning) and "keep_sink" in str(r.message)]
    assert len(msgs) == 1                                   # once per call, not once per row
    assert "24 row(s)" in str(msgs[0].message) and "16" in str(msgs[0].message)
    np.testing.assert_allclose(out[0, 0, 0].numpy(), [0.2, 0.0, 0.64, 0.16], atol=1e-6)
    # below the threshold, with the warning switched off, and for the other choices: silent
    small = torch.tensor([0.2, 0.1, 0.4, 0.3]).view(1, 1, 1, 4)
    _no_warnings(lambda: remove_edges_from_pattern(small, rm, sink[:1]))
    _no_warnings(lambda: remove_edges_from_pattern(w, rm, sink, warn_factor=None))
    _no_warnings(lambda: remove_edges_from_pattern(w, rm, sink, keep_sink=False))
    _no_warnings(lambda: remove_edges_from_pattern(w, rm, sink, renormalize=False))


def test_remove_edges_in_model_warns_once_per_call(gpt2):
    import warnings
    from spectral_trust.interventions._arch import Arch
    arch = Arch(gpt2)
    ids = _ids(4, n=16)
    S = ids.shape[1]
    with st.intervene(gpt2):
        w0 = _pattern(gpt2, ids, 1, arch)[0, 0]
    key_mask = torch.zeros(S, S, dtype=torch.bool)
    key_mask[S - 1, 1:S - 1] = True                    # the last query keeps only itself
    factor = float((1 - w0[S - 1, 0]) / w0[S - 1, S - 1])
    assert factor > 10                                 # the case the warning is for
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with st.intervene(gpt2) as iv:
            iv.attention(1, 0).remove_edges(key_mask)
            _logits(gpt2, ids)
            _logits(gpt2, ids)                         # a second pass under the same call
        with st.intervene(gpt2) as iv:
            iv.attention(1, 0).remove_edges(key_mask, keep_sink=False)
            _logits(gpt2, ids)
    msgs = [str(r.message) for r in rec if "keep_sink" in str(r.message)]
    assert len(msgs) == 1 and "layer 1 (1.0)" in msgs[0]
    with st.intervene(gpt2) as iv:                     # a new call warns again
        iv.attention(1, 0).remove_edges(key_mask)
        with pytest.warns(UserWarning, match="keep_sink=True"):
            _logits(gpt2, ids)
    light = torch.zeros(S, S, dtype=torch.bool)
    light[S - 1, 1] = True                             # one key of fifteen: a small factor
    with st.intervene(gpt2) as iv:
        iv.attention(1, 0).remove_edges(light)
        _no_warnings(lambda: _logits(gpt2, ids))


# ── weights ──────────────────────────────────────────────────────────────────

def _v2_steer(W_param: torch.Tensor, alpha: float) -> torch.Tensor:
    """spectral-steering-v2 scripts/steer.py, get_svd + apply_steering,
    verbatim (weight_svd_full = torch.linalg.svd(W.float(),
    full_matrices=False))."""
    W = W_param.data.float().cpu()
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S_new = S * (1 + alpha * (S - S.mean()) / S.std())
    W_new = U @ torch.diag(S_new) @ Vh
    return W_new.to(W_param.dtype)


def _probe_steer(w: torch.Tensor, alpha: float) -> torch.Tensor:
    """spectral-security probe.py steer, verbatim."""
    u, s, vh = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
    s2 = s * (1 + alpha * (s - s.mean()) / s.std())
    return ((u * s2) @ vh).to(w.dtype)


def _probe_snr(w: torch.Tensor) -> float:
    """spectral-security probe.py svdvals + snr, verbatim."""
    sigma = torch.linalg.svdvals(w.to(torch.float32)).cpu()
    return float(sigma[0] / sigma.median())


def test_svd_scale_and_snr_match_reference_code(llama):
    W = llama.model.layers[1].mlp.down_proj.weight
    for alpha in (-0.5, 0.3):
        ref = _v2_steer(W, alpha)
        assert torch.equal(svd_scale_matrix(W, alpha), ref)
        with st.intervene(llama) as iv:
            iv.weights(1).svd_scale(alpha)
            assert torch.equal(llama.model.layers[1].mlp.down_proj.weight, ref)
        np.testing.assert_allclose(svd_scale_matrix(W, alpha).numpy(),
                                   _probe_steer(W.detach(), alpha).numpy(), atol=1e-6)
    assert weight_snr(llama, 1) == _probe_snr(W.detach())
    assert weight_snr(W) == _probe_snr(W.detach())
    np.testing.assert_array_equal(
        weight_snr(llama), [_probe_snr(l.mlp.down_proj.weight.detach())
                            for l in llama.model.layers])
    top2 = svd_scale_matrix(W, -0.5, k=2)
    s0, s2 = torch.linalg.svdvals(W.detach().float()), torch.linalg.svdvals(top2.float())
    scaled = s0 * (1 - 0.5 * (s0 - s0.mean()) / s0.std())
    expected = torch.sort(torch.cat([scaled[:2], s0[2:]]), descending=True).values
    np.testing.assert_allclose(s2.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
    assert float(s2[0]) < float(s0[0])
    z_max = float(((s0 - s0.mean()) / s0.std()).max())
    with pytest.warns(UserWarning, match="non-positive factor"):
        svd_scale_matrix(W, -1.5 / z_max)                             # sigma_1 flips sign
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        svd_scale_matrix(W, -0.5 / z_max)                             # safe regime: silent


@pytest.mark.filterwarnings("ignore:svd_scale")        # large alphas on purpose
def test_weights_restored_bit_exactly(gpt2, llama):
    for model in (gpt2, llama):
        params, ids = _state(model), _ids(10)
        before = _logits(model, ids)
        with st.intervene(model) as iv:
            iv.weights(0).svd_scale(-0.8)
            iv.weights(0).svd_scale(0.4, k=3)                         # stacked on the same matrix
            iv.weights(1, "mlp.down_proj").svd_scale(1.0)
            iv.attention(0, 0).ablate("zero")
            iv.resid(0).graph_filter("lowpass", cutoff=0.5)
            assert not torch.allclose(_logits(model, ids), before, atol=1e-4)
        after = _state(model)
        assert all(torch.equal(params[k], after[k]) for k in params)
        assert torch.equal(_logits(model, ids), before)


def _numpy_svd_scale(W_lin: np.ndarray, alpha: float) -> np.ndarray:
    """Independent reference: float64 numpy SVD of the [out, in] matrix."""
    U, S, Vt = np.linalg.svd(W_lin.astype(np.float64), full_matrices=False)
    S_new = S * (1 + alpha * (S - S.mean()) / S.std(ddof=1))
    return (U * S_new) @ Vt


@pytest.mark.filterwarnings("ignore:svd_scale")
def test_gpt2_conv1d_matches_an_independent_reference(gpt2):
    mod = gpt2.transformer.h[1].mlp.c_proj                             # Conv1D, weight [in, out]
    W0, b0 = mod.weight.detach().clone(), mod.bias.detach().clone()
    x = torch.randn(5, 128)                                            # Conv1D input width
    for alpha in (-0.5, 0.3):
        ref_lin = _numpy_svd_scale(W0.T.numpy(), alpha)                # [out, in]
        with st.intervene(gpt2) as iv:
            site = iv.weights(1)
            assert site.path == "mlp.c_proj"
            site.svd_scale(alpha)
            np.testing.assert_allclose(mod.weight.detach().T.numpy(), ref_lin, atol=2e-5)
            assert torch.equal(mod.bias, b0)                           # the bias is kept
            with torch.no_grad():
                y = mod(x)
            ref_y = x.double().numpy() @ ref_lin.T + b0.double().numpy()
            np.testing.assert_allclose(y.numpy(), ref_y, atol=1e-4)
    assert torch.equal(mod.weight, W0)
    sig = np.linalg.svd(W0.double().numpy(), compute_uv=False)
    assert abs(weight_snr(gpt2, 1) - sig[0] / np.sort(sig)[(len(sig) - 1) // 2]) < 1e-4


# ── patching ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pair():
    return _ids(20, n=11), _ids(21, n=11)


def test_plain_patch_transformer_lens_semantics(gpt2, pair):
    clean, corrupt = pair
    p = Patcher(gpt2, clean, corrupt)
    assert torch.equal(p.clean_logits, _eager_logits(gpt2, clean))
    assert torch.equal(p.corrupt_logits, _eager_logits(gpt2, corrupt))
    full0 = p.patch(site="resid_pre", layer=0)
    np.testing.assert_allclose(full0.numpy(), p.clean_logits.numpy(), atol=1e-5)
    for site, head, pos in [("resid_pre", None, [3, 7]), ("resid_post", None, [-1]),
                            ("attn_out", None, None), ("mlp_out", None, [2]),
                            ("head", 2, [5, 10]), ("z", [0, 3], None)]:
        got = p.patch(site=site, layer=1, head=head, positions=pos)
        cl = p.activation("clean", site, 1)
        dh = 8
        cols = (torch.arange(32) if head is None else
                torch.cat([torch.arange(h * dh, (h + 1) * dh)
                           for h in ([head] if isinstance(head, int) else head)]))
        m = torch.zeros(11, dtype=torch.bool)
        m[pos if pos is not None else slice(None)] = True

        def manual(x, cl=cl, cols=cols, m=m):
            x = x.clone()
            for i in torch.nonzero(m).flatten():
                x[0, i, cols] = cl[0, i, cols]
            return x
        with st.intervene(gpt2) as iv:
            iv.hook(site, 1, manual)
            ref = _eager_logits(gpt2, corrupt)
        assert torch.equal(got, ref), site
    lone = st.patch(gpt2, clean, corrupt, site="head", layer=1, head=2, positions=[5, 10])
    assert torch.equal(lone, p.patch(site="head", layer=1, head=2, positions=[5, 10]))


@pytest.mark.parametrize("basis", ["clean_attention", "corrupt_attention", "path", "random"])
def test_full_band_equals_plain_patch_and_empty_band_is_identity(gpt2, pair, basis):
    clean, corrupt = pair
    p = Patcher(gpt2, clean, corrupt)
    for site, head in (("resid_pre", None), ("resid_post", None), ("attn_out", None),
                       ("z", 1)):
        plain = p.patch(site=site, layer=1, head=head)
        full = p.patch(site=site, layer=1, head=head, band=(0, None), basis=basis, seed=3)
        np.testing.assert_allclose(full.numpy(), plain.numpy(), atol=1e-4)
        empty = p.patch(site=site, layer=1, head=head, band=(4, 4), basis=basis)
        assert torch.equal(empty, p.corrupt_logits)
    if basis != "random":
        allf = p.patch(site="resid_pre", layer=2, band=(0.0, 2.0), basis=basis)
        np.testing.assert_allclose(allf.numpy(), p.patch(site="resid_pre", layer=2).numpy(),
                                   atol=1e-4)


def test_bands_partition_the_patch_linearly(gpt2, pair):
    clean, corrupt = pair
    p = Patcher(gpt2, clean, corrupt)
    S = clean.shape[1]
    bands = equal_count_bands(S, 4)
    assert [hi - lo for lo, hi in bands] == [3, 3, 3, 2]
    P = sum(p.projector(PatchSpec(layer=1, band=b)) for b in bands)
    np.testing.assert_allclose(P[0].numpy(), np.eye(S), atol=1e-10)
    for b in bands:
        Pb = p.projector(PatchSpec(layer=1, band=b))[0]
        np.testing.assert_allclose((Pb @ Pb).numpy(), Pb.numpy(), atol=1e-10)
        assert abs(float(torch.trace(Pb)) - (b[1] - b[0])) < 1e-10
    lam, U = p.basis("clean_attention", 1)
    A = p.activation("clean", "pattern", 1)
    L = laplacian(attention_adjacency(A, CFG), CFG)[0].double()
    np.testing.assert_allclose((U[0].T @ L @ U[0]).numpy(), np.diag(lam[0]), atol=1e-6)
    many = p.patch_many([PatchSpec("resid_pre", 1, band=b) for b in bands], batch_size=3)
    for k, b in enumerate(bands):
        assert torch.allclose(many[k], p.patch(site="resid_pre", layer=1, band=b), atol=1e-6)


def test_random_basis_and_mode_selection():
    Q = haar_basis(9, seed=2, item=1)
    np.testing.assert_allclose((Q.T @ Q).numpy(), np.eye(9), atol=1e-12)
    assert not torch.allclose(Q, haar_basis(9, seed=3, item=1))
    lam = np.array([0.0, 0.1, 0.5, 0.9, 1.3, 1.9])
    assert list(select_modes(6, (1, 3), lam, None)) == [1, 2]
    assert list(select_modes(6, (0.4, 1.0), lam, None)) == [2, 3]
    assert list(select_modes(6, (1.0, 2.0), lam, None)) == [4, 5]
    assert list(select_modes(6, (0.0, 2.0), lam, 2)) == [0, 1]
    assert list(select_modes(6, None, None, 3)) == [0, 1, 2]
    with pytest.raises(ValueError):
        select_modes(6, (0.1, 0.5), None, None)


def test_patch_batch_inputs_and_metric(gpt2):
    clean, corrupt = _ids(30, n=8, batch=2), _ids(31, n=8, batch=2)
    p = Patcher(gpt2, clean, corrupt)

    def metric(logits):
        return logits[:, -1, 3] - logits[:, -1, 4]
    specs = [PatchSpec("resid_pre", 0), PatchSpec("resid_pre", 2, band=(0, 2)),
             PatchSpec("resid_post", 1, positions=[2, 5]),
             PatchSpec("z", 1, head=[1, 2], band=(3, 6), basis="corrupt_attention"),
             PatchSpec("resid_pre", 1, band=(0.0, 0.8), basis="path"),
             PatchSpec("attn_out", 2, band=(1, 4), basis="random", seed=7)]
    vals = p.patch_many(specs, metric=metric, batch_size=5)
    assert vals.shape == (len(specs), 2)
    np.testing.assert_allclose(vals[0].numpy(), metric(p.clean_logits).numpy(), atol=1e-5)
    for b in range(2):                                   # every spec, against one item at a time
        single = Patcher(gpt2, clean[b], corrupt[b])
        for j, s in enumerate(specs):
            if s.basis == "random":                      # the random basis is seeded per item
                lam, U = p.basis("random", s.layer, s.seed)
                s = PatchSpec(s.site, s.layer, s.head, s.positions, s.band, U[b], s.n_modes)
            ref = single.patch(s, metric=metric)
            np.testing.assert_allclose(float(vals[j, b]), float(ref[0]), atol=1e-5)
    with pytest.raises(ValueError, match="align"):
        Patcher(gpt2, _ids(1, n=8), _ids(2, n=9))


# ── regression tests from the review of 1df529d ──────────────────────────────

def test_subpackage_imports_and_top_level_names():                      # S3
    import importlib
    patching = importlib.import_module("spectral_trust.interventions.patching")
    import spectral_trust.interventions.filters as filters
    assert callable(st.intervene) and st.interventions.intervene is st.intervene
    assert st.patch is patching.patch and st.Patcher is patching.Patcher
    assert st.graph_filter is filters.graph_filter
    assert st.weight_snr is st.interventions.weight_snr

def _gen(model, ids, use_cache, n=5, **extra):
    kw = dict(attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
              output_scores=True, return_dict_in_generate=True, pad_token_id=0)
    kw.update(extra)
    with torch.no_grad():
        return model.generate(ids, use_cache=use_cache, **kw)


def _same_generation(a, b, atol=1e-4):
    assert torch.equal(a.sequences, b.sequences)
    for s1, s2 in zip(a.scores, b.scores):
        np.testing.assert_allclose(s1.numpy(), s2.numpy(), atol=atol)


@pytest.mark.filterwarnings("ignore:svd_scale")
def test_out_of_order_remove_rebuilds_weights(gpt2, llama):            # B1
    for model, W in ((gpt2, lambda: gpt2.transformer.h[0].mlp.c_proj.weight),
                     (llama, lambda: llama.model.layers[0].mlp.down_proj.weight)):
        W0 = W().detach().clone()
        lin = (lambda t: t.T) if model is gpt2 else (lambda t: t)
        only_second = lin(svd_scale_matrix(lin(W0), 0.2))
        with st.intervene(model) as iv:
            e1 = iv.weights(0).svd_scale(-0.3)
            iv.weights(0).svd_scale(0.2)
            e1.remove()
            assert torch.equal(W(), only_second)                       # as if e1 never ran
        assert torch.equal(W(), W0)
        iv1, iv2 = st.intervene(model), st.intervene(model)            # across interventions
        iv1.weights(0).svd_scale(-0.3)
        iv2.weights(0).svd_scale(0.2)
        iv1.close()
        assert torch.equal(W(), only_second)
        iv2.close()
        assert torch.equal(W(), W0)


@pytest.mark.parametrize("which", ["gpt2", "llama"])
def test_causal_filter_follows_beam_search(which, gpt2, llama):        # B2
    model = gpt2 if which == "gpt2" else llama
    ids = _ids(40, n=6)
    assert "generate" not in model.__dict__ and "_reorder_cache" not in model.__dict__
    with st.intervene(model) as iv:
        iv.resid(0).graph_filter("lowpass", cutoff=0.5, strength=0.9, causal=True)
        iv.resid(1).graph_filter("highpass", cutoff=0.8, causal=True, graph="path")
        a = _gen(model, ids, True, n=6, num_beams=3)
        b = _gen(model, ids, False, n=6, num_beams=3)
    assert torch.equal(a.sequences, b.sequences)
    np.testing.assert_allclose(a.sequences_scores.numpy(), b.sequences_scores.numpy(), atol=1e-4)
    assert "generate" not in model.__dict__ and "_reorder_cache" not in model.__dict__


def test_add_uses_absolute_positions_under_generation(gpt2):           # B3a
    ids = _ids(41, n=6)
    vec = torch.randn(32) * 3
    per_pos = torch.randn(20, 32)
    for kw in (dict(positions=[0]), dict(positions=[7, 9]), dict(positions=None)):
        with st.intervene(gpt2) as iv:
            iv.resid(1).add(vec, **kw)
            _same_generation(_gen(gpt2, ids, True), _gen(gpt2, ids, False))
    with st.intervene(gpt2) as iv:
        iv.resid(1, site="pre").add(per_pos)
        _same_generation(_gen(gpt2, ids, True), _gen(gpt2, ids, False))
    with st.intervene(gpt2) as iv:
        iv.resid(1).add(vec, positions=[-1])
        with pytest.raises(RuntimeError, match="absolute positions"):
            _gen(gpt2, ids, True)


def test_mean_and_resample_ablation_under_generation(gpt2, llama):      # B3b, B3c, N5
    ids = _ids(42, n=6)
    for model in (gpt2, llama):
        with st.intervene(model) as iv:
            with pytest.raises(ValueError, match="reference"):
                iv.attention(1, 0).ablate("mean")
            iv.attention(1, [0, 2]).ablate("mean", reference=_ids(43, n=9, batch=3))
            _same_generation(_gen(model, ids, True), _gen(model, ids, False))
        with st.intervene(model) as iv:
            iv.attention(1, 1).ablate("resample", reference=_ids(44, n=16, batch=5), seed=3)
            with torch.no_grad():
                outs = [_logits(model, ids) for _ in range(3)]
            assert all(torch.equal(outs[0], o) for o in outs[1:])       # drawn once
            _same_generation(_gen(model, ids, True), _gen(model, ids, False))


def test_mean_reference_of_different_lengths_has_no_padding(gpt2):     # N5
    from spectral_trust.interventions._arch import Arch
    arch = Arch(gpt2)
    refs = [_ids(45, n=5)[0], _ids(46, n=9)[0]]
    z = [_z(gpt2, r[None], 1, arch)[0].view(-1, 4, 8)[:, 2] for r in refs]
    expected = torch.cat(z).mean(dim=0)
    got = {}
    with st.intervene(gpt2) as iv:
        iv.attention(1, 2).ablate("mean", reference=refs)
        iv.hook("z", 1, lambda x: got.__setitem__("z", x.clone()))
        _logits(gpt2, _ids(47, n=4))
    assert torch.allclose(got["z"][0, :, 16:24], expected.expand(4, 8), atol=1e-6)


def _big_first_token(model_src):
    """A copy of the tiny GPT-2 whose first position embedding is 1000x larger,
    as GPT-2's first-token residual is."""
    import copy
    m = copy.deepcopy(model_src)
    with torch.no_grad():
        m.transformer.wpe.weight[0] *= 1000.0
    return m


def test_float64_filter_with_a_large_norm_first_token(gpt2):           # S1
    S = 14
    A, X = _graph_and_signal(S=S, seed=17)
    X = X.clone()
    X[:, 0] *= 1e4
    Yc = graph_filter(X, A, kind="lowpass", cutoff=0.7, causal=True)
    for i in range(1, S):         # float32 internals miss by ~1e-6 relative; float64 by ~0
        pre = graph_filter(X[:, :i + 1], A[:, :, :i + 1, :i + 1], kind="lowpass", cutoff=0.7)
        rel = float((Yc[:, i] - pre[:, i]).abs().max() / pre[:, i].abs().max())
        assert rel < 1e-7, (i, rel)
    big = _big_first_token(gpt2)
    ids = _ids(48, n=7)
    with st.intervene(big) as iv:
        iv.resid(0).graph_filter("highpass", cutoff=0.8, causal=True)
        iv.resid(1).graph_filter("lowpass", cutoff=0.5, strength=0.7, causal=True)
        _same_generation(_gen(big, ids, True), _gen(big, ids, False), atol=1e-3)


def test_negative_bounds_raise_with_causal(gpt2):                      # S2
    with st.intervene(gpt2) as iv:
        with pytest.raises(ValueError, match="absolute"):
            iv.resid(0).graph_filter("lowpass", cutoff=0.5, causal=True, token_span=(-5, None))
        with pytest.raises(ValueError, match="absolute"):
            iv.resid(0).graph_filter("lowpass", cutoff=0.5, causal=True, positions=[-1])
        iv.resid(0).graph_filter("lowpass", cutoff=0.5, token_span=(-5, None))   # fine if symmetric


def test_unnormalised_laplacian_is_not_skipped():                       # S4
    cfg = GSPConfig(device="cpu", verbose=False, normalization="none")
    S = 16
    A, X = _graph_and_signal(S=S, seed=19)
    from spectral_trust.interventions import laplacian64
    L = laplacian64(attention_adjacency(A, cfg), cfg)[0]
    vals, vecs = torch.linalg.eigh(0.5 * (L + L.T))
    lam_max = float(L.abs().sum(-1).max())
    assert float(vals.max()) > 3.0
    Y = graph_filter(X, A, kind="bandstop", band=(3.0, 100.0), config=cfg, order=48)
    assert not torch.equal(Y, X)
    p = torch.from_numpy(filter_response(vals.numpy(), "bandstop", band=(3.0, 100.0), order=48,
                                         lambda_max=lam_max))
    ref = vecs @ (p.unsqueeze(-1) * (vecs.T @ X[0].double()))
    np.testing.assert_allclose(Y[0].double().numpy(), ref.numpy(), atol=1e-4)
    np.testing.assert_allclose(L.numpy(), laplacian(attention_adjacency(A, cfg), cfg)[0].numpy(),
                               atol=1e-5)


def test_nested_interventions_close_in_any_order(gpt2):                # S5
    import sys
    from spectral_trust.interventions._arch import Arch
    arch = Arch(gpt2)
    modeling = sys.modules[type(arch.attention(1)).__module__]
    orig_fn, impl = modeling.eager_attention_forward, gpt2.config._attn_implementation
    assert impl != "eager"
    ids = _ids(49, n=10)
    S = ids.shape[1]
    km = torch.zeros(S, S, dtype=torch.bool)
    km[5:, 2] = True
    iv1, iv2 = st.intervene(gpt2), st.intervene(gpt2)
    iv1.attention(1, 0).remove_edges(km)
    iv2.attention(2, 0).remove_edges(km)
    iv2.resid(1).graph_filter("lowpass", cutoff=0.5)
    both = _logits(gpt2, ids)
    iv1.close()                                                   # not the last one opened
    assert modeling.eager_attention_forward is not orig_fn
    assert gpt2.config._attn_implementation == "eager"
    only2 = _logits(gpt2, ids)
    assert not torch.allclose(only2, both)
    iv2.close()
    assert modeling.eager_attention_forward is orig_fn
    assert gpt2.config._attn_implementation == impl
    with st.intervene(gpt2) as iv3:                               # iv2's edits alone
        iv3.attention(2, 0).remove_edges(km)
        iv3.resid(1).graph_filter("lowpass", cutoff=0.5)
        np.testing.assert_allclose(_logits(gpt2, ids).numpy(), only2.numpy(), atol=1e-6)


def test_patcher_refuses_a_changed_model_and_captures_lazily(gpt2, pair):   # S6, S7
    clean, corrupt = pair
    p = Patcher(gpt2, clean, corrupt)
    assert p._cache["clean"] == {} and p._cache["corrupt"] == {}
    p.patch(site="resid_pre", layer=1)
    assert set(p._cache["clean"]) == {("resid_pre", 1)}            # no pattern for a plain patch
    p.patch(site="resid_pre", layer=1, band=(0, 3))
    assert ("pattern", 1) in p._cache["clean"]
    with st.intervene(gpt2) as iv:
        iv.weights(2).svd_scale(-0.05)
        with pytest.raises(RuntimeError, match="changed"):
            p.patch(site="resid_pre", layer=0)
        p.refresh()
        np.testing.assert_allclose(p.patch(site="resid_pre", layer=0).numpy(),
                                   _eager_logits(gpt2, clean).numpy(), atol=1e-5)
    with pytest.raises(RuntimeError, match="changed"):
        p.patch(site="resid_pre", layer=0)


def _old_weight_snr(W: torch.Tensor, k: int = 1) -> float:
    """spectral_trust 0.2.3 branch (commit 82316ee) weight_snr, verbatim."""
    W_fp = W.float()
    niter = max(4, k // 4)
    _, S, _ = torch.svd_lowrank(W_fp, q=max(k, 16), niter=niter)
    sigma_1 = S[0].item()
    sigma_median = S[len(S) // 2].item()
    return sigma_1 / sigma_median if sigma_median > 1e-12 else 0.0


def test_spectral_steering_imports_run_on_this_release(llama):         # S8
    W = llama.model.layers[1].mlp.down_proj.weight.detach()
    torch.manual_seed(5)
    ref = _old_weight_snr(W, k=32)
    torch.manual_seed(5)
    assert st.weight_snr(W, k=32) == ref
    torch.manual_seed(5)
    assert st.weight_snr(W, 32) == ref                                  # positional k
    U, S, Vh = st.weight_svd_full(W)
    U2, S2, Vh2 = torch.linalg.svd(W.float(), full_matrices=False)
    assert torch.equal(U, U2) and torch.equal(S, S2) and torch.equal(Vh, Vh2)
    assert st.weight_snr(W) == _probe_snr(W)                            # full spectrum by default


def test_remove_edges_rows_with_only_zero_removals_stay_exact():        # N1
    g = torch.Generator().manual_seed(0)
    S = 32
    lg = torch.randn(1, 4, S, S, generator=g)
    w = torch.softmax(lg.masked_fill(~torch.ones(S, S, dtype=torch.bool).tril(), float("-inf")), -1)
    rm = torch.zeros(S, dtype=torch.bool)
    rm[S - 1] = True                                       # zero in every row but the last
    out = remove_edges_from_pattern(w, rm.view(1, 1, 1, S), torch.zeros(1, dtype=torch.long))
    assert torch.equal(out[..., :S - 1, :], w[..., :S - 1, :])
    assert out[0, 0, S - 1, S - 1] == 0


def test_pattern_editor_requires_module_level_eager_function():         # N2
    from spectral_trust.interventions import _registry

    class Lonely(torch.nn.Module):
        pass
    with pytest.raises(ValueError, match="4.48"):
        _registry.acquire_pattern_editor(Lonely(), object(), lambda w, m: w)


def test_pattern_edits_check_the_transformers_version(gpt2, monkeypatch):   # setup note
    from spectral_trust.interventions import _registry
    monkeypatch.setattr(_registry, "transformers_version", lambda: "4.40.2")
    with st.intervene(gpt2) as iv:
        with pytest.raises(ValueError, match=r"transformers >= 4\.48; found 4\.40\.2"):
            iv.attention(1, 0).remove_edges(torch.zeros(10, dtype=torch.bool))
        iv.resid(1).add(torch.zeros(32))                       # the rest still works
        _logits(gpt2, _ids(61, n=5))


def test_dense_filter_includes_the_top_edge():                          # N3
    from spectral_trust.interventions import ideal_response
    lam = np.array([0.5, 1.999, 2.0])
    np.testing.assert_array_equal(ideal_response(lam, "lowpass", cutoff=2.0), [1, 1, 1])
    np.testing.assert_array_equal(ideal_response(lam, "bandpass", band=(1.0, 2.0)), [0, 1, 1])
    np.testing.assert_array_equal(ideal_response(lam, "bandstop", band=(1.0, 2.0)), [1, 0, 0])
    np.testing.assert_array_equal(ideal_response(lam, "lowpass", cutoff=1.999), [1, 0, 0])
    S = 8
    X = torch.randn(1, S, 3, generator=torch.Generator().manual_seed(2))
    L = laplacian(path_adjacency(S), CFG)[0].double()
    assert abs(float(torch.linalg.eigvalsh(L).max()) - 2.0) < 1e-6     # bipartite: lambda = 2
    for kind, kw in (("lowpass", {"cutoff": 2.0}), ("bandpass", {"band": (0.0, 2.0)})):
        d = graph_filter(X, kind=kind, graph="path", method="dense", **kw)
        np.testing.assert_allclose(d.numpy(), X.numpy(), atol=1e-6)
        c = graph_filter(X, kind=kind, graph="path", order=256, **kw)
        np.testing.assert_allclose(c.numpy(), d.numpy(), atol=1e-3)
    d = graph_filter(X, kind="bandstop", band=(1.5, 2.0), graph="path", method="dense")
    c = graph_filter(X, kind="bandstop", band=(1.5, 2.0), graph="path", order=256)
    np.testing.assert_allclose(c.numpy(), d.numpy(), atol=0.05)


def test_head_outputs_keeps_the_causal_state(gpt2):                     # N6
    ids = _ids(50, n=7)
    nxt = torch.tensor([[5]])

    def step(iv, between):
        with torch.no_grad():
            out = gpt2(input_ids=ids, use_cache=True)
            if between:
                iv.attention(0, 1).head_outputs(_ids(51, n=11))
            return gpt2(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True,
                        attention_mask=torch.ones(1, 8, dtype=torch.long)).logits
    with st.intervene(gpt2) as iv:
        iv.resid(1).graph_filter("lowpass", cutoff=0.5, causal=True)
        plain = step(iv, False)
        mixed = step(iv, True)
    np.testing.assert_allclose(mixed.numpy(), plain.numpy(), atol=1e-6)


# ── regression tests from the re-verification of 7926bb2 ────────────────────

def _hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def _registries_empty():
    from spectral_trust.interventions import _registry as reg
    return not (reg._WEIGHTS or reg._EAGER_FN or reg._EAGER_IMPL or reg._BEAMS
                or reg.PATTERN_EDITS or reg.OPEN)


def test_failed_weight_edit_leaves_no_trace(gpt2):                      # R1
    W = gpt2.transformer.h[0].mlp.c_proj.weight
    W0, hooks, impl = W.detach().clone(), _hook_count(gpt2), gpt2.config._attn_implementation
    with st.intervene(gpt2) as iv:
        with pytest.raises(ValueError, match="k must be"):
            iv.weights(0).svd_scale(0.1, k=-1)
        assert iv.edits == []
        assert torch.equal(W, W0)
    assert _registries_empty()
    with torch.no_grad():
        W.mul_(2.0)                                     # the user changes the weight
    W_user = W.detach().clone()
    try:
        with st.intervene(gpt2) as iv:
            iv.resid(1).graph_filter("lowpass", cutoff=0.5, causal=True)
            iv.weights(0).svd_scale(-0.2)
        assert torch.equal(W, W_user)                   # the user's value, not a stale copy
        assert _hook_count(gpt2) == hooks and gpt2.config._attn_implementation == impl
        assert "generate" not in gpt2.__dict__ and _registries_empty()
    finally:
        with torch.no_grad():
            W.copy_(W0)


def test_close_finishes_after_a_failing_step(gpt2):                    # R1
    W = gpt2.transformer.h[0].mlp.c_proj.weight
    W0, hooks, impl = W.detach().clone(), _hook_count(gpt2), gpt2.config._attn_implementation
    iv = st.intervene(gpt2)
    iv.resid(1).graph_filter("lowpass", cutoff=0.5, causal=True)
    e = iv.weights(0).svd_scale(-0.2)
    iv.attention(1, 0).remove_edges(torch.zeros(10, dtype=torch.bool))

    def boom():
        raise KeyError("undo failed")
    e.undo.insert(0, boom)                              # runs last in the reverse order
    with pytest.raises(KeyError):
        iv.close()
    assert torch.equal(W, W0)
    assert _hook_count(gpt2) == hooks and gpt2.config._attn_implementation == impl
    assert "generate" not in gpt2.__dict__ and _registries_empty()


def test_precision_option_and_placement():                              # R2
    from spectral_trust.interventions import compute_placement
    mps, cuda, cpu = torch.device("mps"), torch.device("cuda"), torch.device("cpu")
    assert compute_placement(mps, "float64") == (torch.float64, cpu)    # no float64 on MPS
    assert compute_placement(mps, "float32") == (torch.float32, mps)
    assert compute_placement(cuda, "float64") == (torch.float64, cuda)
    assert compute_placement(cuda, "float32", "dense") == (torch.float32, cpu)
    A, X = _graph_and_signal(S=14, seed=23)
    with pytest.raises(ValueError, match="precision"):
        graph_filter(X, A, kind="lowpass", cutoff=0.5, precision="float16")
    Y64 = graph_filter(X, A, kind="lowpass", cutoff=0.5, causal=True)
    Y32 = graph_filter(X, A, kind="lowpass", cutoff=0.5, causal=True, precision="float32")
    np.testing.assert_allclose(Y32.numpy(), Y64.numpy(), atol=1e-5)
    assert Y32.dtype == X.dtype


def test_precision_float32_in_the_model(gpt2):                          # R2
    ids = _ids(62, n=8)
    out = {}
    for p in ("float64", "float32"):
        with st.intervene(gpt2) as iv:
            iv.resid(1).graph_filter("highpass", cutoff=0.8, causal=True, precision=p)
            out[p] = _logits(gpt2, ids)
    np.testing.assert_allclose(out["float32"].numpy(), out["float64"].numpy(), atol=1e-4)


def test_generate_wrapper_reads_a_positional_generation_config(gpt2):   # R3
    from transformers import GenerationConfig
    from spectral_trust.interventions import _registry as reg
    ids = _ids(63, n=6)
    gc = GenerationConfig(num_beams=3, max_new_tokens=4, do_sample=False, pad_token_id=0)
    seen = []
    with st.intervene(gpt2) as iv:
        iv.resid(0).graph_filter("lowpass", cutoff=0.5, causal=True)
        h = gpt2.register_forward_pre_hook(lambda m, a: seen.append(reg.generation_beams(gpt2)))
        try:
            with torch.no_grad():
                a = gpt2.generate(ids, gc, attention_mask=torch.ones_like(ids), use_cache=True)
                b = gpt2.generate(ids, gc, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            h.remove()
    assert seen and set(seen) == {3}
    assert torch.equal(a, b)


def test_resample_ablation_under_beam_search(gpt2):                    # R5 (verify3)
    ids, ref = _ids(64, n=6), _ids(65, n=20, batch=5)
    zs = []
    with st.intervene(gpt2) as iv:
        iv.attention(1).ablate("resample", reference=ref, seed=0)
        iv.attention(2).ablate("resample", reference=ref, seed=1)
        iv.hook("z", 1, lambda x: zs.append(x.clone()))       # sees the replaced outputs
        a = _gen(gpt2, ids, True, n=6, num_beams=4)
        b = _gen(gpt2, ids, False, n=6, num_beams=4)
    prefill = zs[0]                                           # [4 beams, 6, 32]: one input
    assert prefill.shape[0] == 4
    assert all(torch.equal(prefill[0], prefill[r]) for r in range(1, 4))   # beams share a draw
    assert torch.equal(a.sequences, b.sequences)
    np.testing.assert_allclose(a.sequences_scores.numpy(), b.sequences_scores.numpy(), atol=1e-4)


def test_head_outputs_keeps_other_interventions_state(gpt2):           # N6
    ids, nxt = _ids(66, n=7), torch.tensor([[5]])

    def step(other):
        with torch.no_grad():
            out = gpt2(input_ids=ids, use_cache=True)
            if other is not None:
                other.attention(0, 1).head_outputs(_ids(67, n=11))
            return gpt2(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True,
                        attention_mask=torch.ones(1, 8, dtype=torch.long)).logits
    with st.intervene(gpt2) as iv1:
        iv1.resid(1).graph_filter("lowpass", cutoff=0.5, causal=True)
        plain = step(None)
        with st.intervene(gpt2) as iv2:
            mixed = step(iv2)
    np.testing.assert_allclose(mixed.numpy(), plain.numpy(), atol=1e-6)
