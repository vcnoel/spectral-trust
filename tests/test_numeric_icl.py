"""spectral_trust.numeric, spectral_trust.audit and
spectral_trust.interventions.edit_with_controls, on tiny random models built
offline and on hand-made tokenizers (no download).

The reference values are computed here from the formulas of Bao et al.
(2026) with numpy and scipy, not with the library's own helpers."""
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust import GSPConfig  # noqa: E402
from spectral_trust.numeric import (  # noqa: E402
    attention_gini, graph_signal_metrics, next_value_distribution, pool_nodes,
    positional_metrics, profile_from_outputs, quantize, repeat_fraction, serialize, value_spans,
)

pytestmark = pytest.mark.filterwarnings("ignore:graph_filter at layer")
torch.set_num_threads(2)
CFG = GSPConfig(device="cpu", verbose=False, save_plots=False)


# ── hand-made tokenizers ─────────────────────────────────────────────────────

class DigitTokenizer:
    """BOS = 0, digits '0'..'9' = 1..10, ',' = 11 (Qwen-like)."""
    all_special_ids = [0]

    def __init__(self, bos=True):
        self.bos = bos

    def _pieces(self, text):
        out, pos = [], 0
        for ch in text:
            out.append((1 + int(ch) if ch.isdigit() else 11, pos, pos + 1))
            pos += 1
        return out

    def encode(self, text, add_special_tokens=True):
        ids = [p[0] for p in self._pieces(text)]
        return ([0] if add_special_tokens and self.bos else []) + ids

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        pieces = self._pieces(text)
        ids = [p[0] for p in pieces]
        offs = [(a, b) for _, a, b in pieces]
        if add_special_tokens and self.bos:
            ids, offs = [0] + ids, [(0, 0)] + offs
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


class TripletTokenizer(DigitTokenizer):
    """BOS = 0, ',' = 11, every 3-digit string one token 12 + v (Llama-3-like)."""

    def _pieces(self, text):
        out, i = [], 0
        while i < len(text):
            if text[i:i + 3].isdigit() and len(text[i:i + 3]) == 3:
                out.append((12 + int(text[i:i + 3]), i, i + 3))
                i += 3
            else:
                out.append((11 if text[i] == "," else 1 + int(text[i]), i, i + 1))
                i += 1
        return out


def _llama(vocab, seed=0, layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, bos_token_id=0, eos_token_id=0, pad_token_id=0,
                      attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def digit_model():
    return _llama(12, seed=3)


@pytest.fixture(scope="module")
def triplet_model():
    return _llama(1012, seed=4)


# ── serialisation ────────────────────────────────────────────────────────────

def test_quantize_and_serialize():
    q = quantize([0.0, 0.5, 1.0])
    assert q.tolist() == [50, 500, 950]
    assert quantize([3.0, 3.0]).tolist() == [500, 500]
    # bounds of the full trajectory fix the mapping of a prefix
    assert quantize([0.5], bounds=(0.0, 1.0)).tolist() == [500]
    assert serialize([87, 173, 642]) == "087,173,642,"
    assert serialize([87, 173], trailing=False) == "087,173"
    with pytest.raises(ValueError):
        serialize([1000])


def test_value_spans_digit_and_triplet():
    text = serialize([87, 173, 5])
    sp = value_spans(DigitTokenizer(), text)
    assert sp.special == [0]
    assert sp.values == [(1, 4), (5, 8), (9, 12)]
    assert sp.separators == [4, 8, 12]
    assert sp.tokens_per_value.tolist() == [3, 3, 3]
    sp3 = value_spans(TripletTokenizer(), text)
    assert sp3.values == [(1, 2), (3, 4), (5, 6)] and sp3.separators == [2, 4, 6]
    assert sp3.numeric_positions == [1, 3, 5]


def test_repeat_fraction():
    assert repeat_fraction([1, 2] * 10) > 0.8
    assert repeat_fraction(list(range(20))) == 0.0


# ── readout ──────────────────────────────────────────────────────────────────

def test_single_token_readout_is_restricted_softmax(triplet_model):
    tok = TripletTokenizer()
    ids = tok.encode(serialize([87, 173, 642]))
    d = next_value_distribution(triplet_model, ids, tok)
    assert d.method == "single"
    with torch.no_grad():
        logits = triplet_model(input_ids=torch.tensor([ids])).logits[0, -1].double()
    ref = torch.softmax(logits[12:1012], -1).numpy()
    assert np.allclose(d.probs, ref, atol=1e-10)
    v = np.arange(1000)
    assert d.mean == pytest.approx(float(v @ ref))
    assert d.entropy == pytest.approx(float(-(ref * np.log(ref)).sum()))


def test_hierarchical_readout_matches_brute_force(digit_model):
    tok = DigitTokenizer()
    ids = tok.encode(serialize([87, 173, 642]))
    d = next_value_distribution(digit_model, ids, tok)
    assert d.method == "hierarchical"
    # brute force: P(d1) P(d2|d1) P(d3|d1 d2) from full sequences
    dig = list(range(1, 11))
    brute = np.zeros(1000)
    with torch.no_grad():
        l1 = torch.log_softmax(digit_model(input_ids=torch.tensor([ids])).logits[0, -1, dig].double(), -1)
        for a in range(10):
            l2 = torch.log_softmax(digit_model(input_ids=torch.tensor([ids + [1 + a]])).logits[0, -1, dig].double(), -1)
            batch = torch.tensor([ids + [1 + a, 1 + b] for b in range(10)])
            l3 = torch.log_softmax(digit_model(input_ids=batch).logits[:, -1][:, dig].double(), -1)
            for b in range(10):
                brute[100 * a + 10 * b:100 * a + 10 * b + 10] = (l1[a] + l2[b] + l3[b]).exp().numpy()
    assert np.allclose(d.probs, brute, atol=1e-6)
    # pruned expansion keeps total mass and stays close
    dp = next_value_distribution(digit_model, ids, tok, min_prob=0.05, chunk=3)
    assert dp.probs.sum() == pytest.approx(1.0)
    assert abs(dp.mean - d.mean) < 60


# ── graph-signal diagnostics ─────────────────────────────────────────────────

def _causal_attention(H, S, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(H, S, S, generator=g, dtype=torch.float64)
    logits = logits.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool)), -1e9)
    return torch.softmax(logits, -1)


def test_profile_matches_paper_formulas():
    from scipy.linalg import eigh
    H, S, d = 3, 12, 5
    A = _causal_attention(H, S, 0)
    X = torch.randn(S, d, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    prof = profile_from_outputs([A.unsqueeze(0)], [X.unsqueeze(0), X.unsqueeze(0)],
                                drop=(0,), renormalize=True)
    # reference: drop BOS, renormalise rows, W = (1/2H) sum (A + A^T)
    a = A.numpy()[:, 1:, 1:]
    a = a / a.sum(-1, keepdims=True)
    W = (a + a.transpose(0, 2, 1)).sum(0) / (2 * H)
    D = np.diag(W.sum(1))
    Dm = np.diag(W.sum(1) ** -0.5)
    lam = eigh(Dm @ (D - W) @ Dm, eigvals_only=True)
    _, U = eigh(D - W)
    Xh = U.T @ X.numpy()[1:]
    p = (Xh ** 2).sum(1)
    p /= p.sum()
    n = S - 1
    assert prof.mean["fiedler"] == pytest.approx(lam[1], abs=1e-9)
    assert prof.mean["hfer"] == pytest.approx(p[n // 2:].sum(), abs=1e-9)
    assert prof.mean["exp_se"] == pytest.approx(np.exp(-(p * np.log(p)).sum()), abs=1e-9)
    assert prof.mean["sink_mass"] == pytest.approx(A.numpy()[:, 1:, 0].mean())
    assert len(prof.per_layer["low_eigenvalues"][0]) == 10


def test_constant_signal_is_lowest_frequency():
    A = _causal_attention(2, 10, 2)
    W = 0.5 * (A + A.transpose(-1, -2)).mean(0)
    m = graph_signal_metrics(W, torch.ones(10, 4, dtype=torch.float64))
    assert m["hfer"] == pytest.approx(0.0, abs=1e-12)
    assert m["exp_se"] == pytest.approx(1.0, abs=1e-6)
    pm = positional_metrics(np.ones((10, 4)))
    assert pm["pos_hfer"] == pytest.approx(0.0, abs=1e-12)


def test_gini_extremes():
    S = 6
    uniform = torch.tril(torch.ones(S, S, dtype=torch.float64))
    uniform = uniform / uniform.sum(-1, keepdim=True)
    assert attention_gini(uniform.unsqueeze(0)) == pytest.approx(0.0, abs=1e-12)
    onehot = torch.zeros(1, S, S, dtype=torch.float64)
    onehot[0, :, 0] = 1.0
    assert attention_gini(onehot) == pytest.approx(1.0, abs=1e-12)


def test_pool_nodes():
    A = _causal_attention(2, 8, 3)
    W = 0.5 * (A + A.transpose(-1, -2)).mean(0)
    X = torch.arange(16, dtype=torch.float64).view(8, 2)
    Wp, Xp = pool_nodes(W, X, [[0, 1, 2], [3], [4, 5, 6], [7]])
    assert Wp.shape == (4, 4) and Wp.sum() == pytest.approx(float(W.sum()))
    assert np.allclose(Xp[0], X[:3].mean(0).numpy())


# ── confound audit ───────────────────────────────────────────────────────────

def test_confound_audit_separates_confounded_from_genuine():
    rng = np.random.default_rng(0)
    n = 300
    c = rng.normal(size=n)
    y = (c + 0.3 * rng.normal(size=n) > 0).astype(int)
    confounded = c + 0.1 * rng.normal(size=n)               # carries only the covariate
    genuine = y + 0.5 * rng.normal(size=n)                   # carries the label beyond it
    res = st.confound_audit({"confounded": confounded, "genuine": genuine}, y,
                            {"c": c, "constant": np.ones(n)}, n_boot=300)
    rows = {r["readout"]: r for r in res.rows}
    assert res.covariates == ["c"]                           # constants are ignored
    assert rows["confounded"]["auroc"] > 0.9
    assert rows["confounded"]["cv_gain_lo"] < 0.01
    assert rows["genuine"]["cv_gain_lo"] > 0
    assert rows["genuine"]["resid_auroc_lo"] > 0.6
    assert "AUROC raw" in res.to_markdown() and "(covariate)" in res.to_markdown()
    assert [r["covariate"] for r in res.covariate_rows] == ["c"]
    assert res.covariate_rows[0]["auroc"] > 0.9


def test_confound_audit_continuous_and_helpers():
    rng = np.random.default_rng(1)
    n = 200
    c = rng.normal(size=n)
    y = c + rng.normal(size=n)
    res = st.confound_audit({"r": c + 0.05 * rng.normal(size=n)}, y, {"c": c},
                            kind="continuous", n_boot=200)
    row = res.rows[0]
    assert row["spearman"] > 0.5 and row["partial_spearman_lo"] < 0.1
    est, lo, hi = st.bootstrap_ci(np.ones(10) * 2.0, n_boot=50)
    assert est == lo == hi == 2.0
    assert st.effective_rank(np.outer(np.arange(1, 6), np.arange(1, 4))) == pytest.approx(1.0)
    assert st.effective_rank(np.eye(4)) == pytest.approx(4.0)
    cov = st.confound_covariates(_causal_attention(2, 9, 0), np.eye(9))
    assert cov["n_tokens"] == 9 and cov["mean_position"] == 4.0


# ── edit with controls ───────────────────────────────────────────────────────

def _last_logits(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids).logits[0, -1]
    return {f"l{i}": float(out[i]) for i in range(5)}


def test_edit_with_controls_target_equals_live_filter(digit_model):
    ids = torch.tensor([[0] + [1 + (i * 7) % 10 for i in range(23)]])
    pos = list(range(1, 24, 2))
    res = st.edit_with_controls(digit_model, [ids[0]], _last_logits, layer=0, kind="lowpass",
                                cutoff=1.0, strengths=(0.0, 0.5), positions=pos, config=CFG)
    with st.intervene(digit_model, config=CFG) as iv:
        iv.resid(0).graph_filter("lowpass", cutoff=1.0, strength=0.5, causal=True,
                                 positions=pos)
        live = _last_logits(digit_model, ids)
    got = res.edited[("target", 0.5)][0]
    assert all(abs(got[k] - live[k]) < 1e-4 for k in live)
    for cond in res.conditions:                  # strength 0 is the identity for every control
        assert res.edited[(cond, 0.0)][0] == pytest.approx(res.baseline[0], abs=1e-5)
    rows = res.summary(n_boot=50)
    assert {r["condition"] for r in rows} == {"target", "permuted", "path", "scaling"}
    # the first token out of the graph, as ResidSite.graph_filter(token_span=(1, None))
    res = st.edit_with_controls(digit_model, [ids[0]], _last_logits, layer=1, kind="lowpass",
                                cutoff=0.8, strengths=(0.6,), positions=pos,
                                token_span=(1, None), config=CFG)
    with st.intervene(digit_model, config=CFG) as iv:
        iv.resid(1).graph_filter("lowpass", cutoff=0.8, strength=0.6, causal=True,
                                 positions=pos, token_span=(1, None))
        live = _last_logits(digit_model, ids)
    got = res.edited[("target", 0.6)][0]
    assert all(abs(got[k] - live[k]) < 1e-4 for k in live)
    with pytest.raises(ValueError):
        st.edit_with_controls(digit_model, [ids[0]], _last_logits, layer=1, positions=[0, 3],
                              token_span=(1, None), config=CFG)


def test_edit_with_controls_norm_matching(digit_model):
    ids = torch.tensor([[0] + [1 + (i * 3) % 10 for i in range(19)]])
    seen = {}

    def readout(model, x):
        return {"x": 0.0}

    from spectral_trust.interventions import controls as C
    orig = C._run_with_delta

    def spy(arch, model, ids, site, layer, rows, delta, readout):
        seen.setdefault(round(float(delta.norm()), 6), 0)
        seen[round(float(delta.norm()), 6)] += 1
        return orig(arch, model, ids, site, layer, rows, delta, readout)

    C._run_with_delta = spy
    try:
        st.edit_with_controls(digit_model, [ids[0]], readout, layer=1, strengths=(0.7,),
                              config=CFG)
    finally:
        C._run_with_delta = orig
    assert len(seen) == 1 and list(seen.values())[0] == 4   # four conditions, one norm


def test_band_patch_matches_patcher(digit_model):
    clean = torch.tensor([[0] + [1 + (i % 4) for i in range(19)]])
    corrupt = torch.tensor([[0] + [1 + (i * 7) % 10 for i in range(19)]])
    res = st.edit_with_controls(digit_model, [corrupt[0]], _last_logits, layer=1,
                                edit="band_patch", source=[clean[0]], modes=(0, 5),
                                strengths=(1.0,), controls=("random", "path", "scaling"),
                                config=CFG)
    p = st.Patcher(digit_model, clean, corrupt, config=CFG)
    ref = p.patch(site="resid_pre", layer=1, band=(0, 5), basis="clean_attention")[0, -1]
    got = res.edited[("target", 1.0)][0]
    assert all(abs(got[f"l{i}"] - float(ref[i])) < 1e-4 for i in range(5))
