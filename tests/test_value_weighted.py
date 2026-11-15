# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Value-weighted attention graphs (spectral_trust.value_weighted).

The acceptance check: for every architecture covered, the per-key
contributions c_ij plus the output bias rebuild the attention module's real
output at every token, to a float32 relative error of 1e-4. Tiny models are
built from configs with every parameter (biases, norms, gates included)
drawn at random, so that no term vanishes by initialisation.

Real checkpoints from the local Hugging Face cache run when
SPECTRAL_TRUST_VALUE_MODELS names them (comma-separated), for example::

    SPECTRAL_TRUST_VALUE_MODELS=gpt2,Qwen/Qwen3-0.6B pytest tests/test_value_weighted.py
"""
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust.interventions import _registry as reg  # noqa: E402
from spectral_trust.value_weighted import ValueCapture  # noqa: E402

torch.set_num_threads(2)
V = 60
TOL = 1e-4


def _randomise(model, seed, scale=0.4):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return model.eval()


def _gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=4, n_embd=32, vocab_size=V,
                                      n_positions=64))


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    return LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=48,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=8, attention_bias=True))


def _qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=48,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=1))


def _qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=V, hidden_size=32, intermediate_size=48,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=16))


def _phi3():
    from transformers import Phi3Config, Phi3ForCausalLM
    return Phi3ForCausalLM(Phi3Config(vocab_size=V, hidden_size=32, intermediate_size=48,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, pad_token_id=0))


def _gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    return Gemma2ForCausalLM(Gemma2Config(vocab_size=V, hidden_size=32, intermediate_size=48,
                                          num_hidden_layers=2, num_attention_heads=4,
                                          num_key_value_heads=2, head_dim=8, sliding_window=4,
                                          query_pre_attn_scalar=8))


def _gpt_neox():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    return GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=V, hidden_size=32, intermediate_size=48,
                                            num_hidden_layers=2, num_attention_heads=4,
                                            rotary_pct=0.5))


def _phi():
    from transformers import PhiConfig, PhiForCausalLM
    return PhiForCausalLM(PhiConfig(vocab_size=V, hidden_size=32, intermediate_size=48,
                                    num_hidden_layers=2, num_attention_heads=4))


def _qwen35():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=V, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "linear_attention", "linear_attention",
                     "full_attention"]))


def _qwen3_next():
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    return Qwen3NextForCausalLM(Qwen3NextConfig(
        vocab_size=V, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, num_experts=2, num_experts_per_tok=1,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"]))


#: name -> (builder, gate layout, grouped-query attention)
ARCHS = {
    "gpt2": (_gpt2, None, False),            # fused Conv1D c_attn, value and output biases
    "llama": (_llama, None, True),           # GQA, attention_bias=True
    "qwen2": (_qwen2, None, True),           # GQA (one KV head), q/k/v biases
    "qwen3": (_qwen3, None, True),           # GQA, q/k norms, head_dim != d / H
    "phi3": (_phi3, None, True),             # fused qkv_proj
    "gemma2": (_gemma2, None, True),         # soft-capping, sliding window
    "gpt_neox": (_gpt_neox, None, False),    # fused, per-head interleaved query_key_value
    "phi": (_phi, None, False),              # dense output projection with bias
    "qwen3_5": (_qwen35, "query_split", True),     # hybrid, gated
    "qwen3_next": (_qwen3_next, "query_split", True),
}


@pytest.fixture(scope="module", params=sorted(ARCHS))
def arch(request):
    builder, gate, gqa = ARCHS[request.param]
    torch.manual_seed(0)
    return request.param, _randomise(builder(), seed=1), gate, gqa


def _ids(seed, n=13, batch=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (batch, n), generator=g)


def _rel(a, b):
    """Largest per-token relative error ||a_i - b_i|| / ||b_i||."""
    a, b = a.double(), b.double()
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-30)).max())


# ── the acceptance criterion ─────────────────────────────────────────────────

def test_contributions_rebuild_the_attention_output(arch):
    name, model, gate, gqa = arch
    vals = st.capture_attention_values(model, _ids(0), keep_output=True)
    assert vals, name
    for lay, av in vals.items():
        assert av.gate_layout == gate, (name, lay)
        assert (av.n_kv_heads < av.n_heads) == gqa, (name, lay)
        # sum over keys of c_ij (no bias) + b_O = the module's output
        c = av.contributions()                                   # [B, S, S, d]
        y = c.sum(dim=2) + (av.out_bias if av.out_bias is not None else 0)
        assert _rel(y, av.output) < TOL, (name, lay, _rel(y, av.output))
        assert _rel(av.reconstruct(), av.output) < TOL
        if av.out_bias is not None:              # the bias is in no contribution
            assert float(av.out_bias.abs().max()) > 0


def test_edge_weights_are_the_norms_of_the_contributions(arch):
    name, model, gate, _ = arch
    for lay, av in st.capture_attention_values(model, _ids(1)).items():
        ref = av.contributions().double().norm(dim=-1)            # [B, S, S]
        w = av.edge_weights()
        assert w.shape == ref.shape and w.dtype == torch.float32
        assert torch.allclose(w.double(), ref, rtol=1e-5, atol=1e-6 * float(ref.max()))
        # per head: the norm of each head's own term, and their sum is the full output
        ph = av.edge_weights(per_head=True)
        assert ph.shape == av.weights.shape
        total = 0
        for h in range(av.n_heads):
            only = torch.zeros_like(av.weights)
            only[:, h] = av.weights[:, h]
            ch = st.attention_contributions(only, av.values, av.out_weight, av.gate)
            total = total + ch
            assert torch.allclose(ph[:, h].double(), ch.double().norm(dim=-1), rtol=1e-5,
                                  atol=1e-6 * float(ref.max())), (name, lay, h)
        assert torch.allclose(total, av.contributions(), rtol=1e-5, atol=1e-6)
        # triangle inequality: the summed norm is at most the sum of per-head norms
        assert bool((w <= ph.sum(dim=1) * (1 + 1e-5) + 1e-6).all())


def test_capture_changes_nothing(arch):
    name, model, _, _ = arch
    ids = _ids(2)
    impl = model.config._attn_implementation
    with torch.no_grad():
        before = model(input_ids=ids, use_cache=False).logits
        model.set_attn_implementation("eager")
        eager = model(input_ids=ids, use_cache=False).logits
        model.set_attn_implementation(impl)
    with ValueCapture(model) as cap:
        assert model.config._attn_implementation == "eager"
        with torch.no_grad():
            during = model(input_ids=ids, use_cache=False).logits
    assert cap.results()
    with torch.no_grad():
        after = model(input_ids=ids, use_cache=False).logits
    assert model.config._attn_implementation == impl
    assert not reg.CAPTURES and not reg._EAGER_FN
    assert torch.equal(eager, during)                 # the capture only reads
    assert torch.equal(before, after)                 # and leaves the model as it was


# ── chunking, batches, layers ───────────────────────────────────────────────

@pytest.mark.parametrize("name", ["llama", "qwen3_5"])
def test_chunked_edges_equal_unchunked(name):
    torch.manual_seed(0)
    model = _randomise(ARCHS[name][0](), seed=2)
    for av in st.capture_attention_values(model, _ids(3, n=17)).values():
        for per_head in (False, True):
            big = av.edge_weights(per_head=per_head)
            small = av.edge_weights(per_head=per_head, chunk_bytes=1)
            assert torch.allclose(big, small, rtol=1e-6, atol=1e-7)
        assert _rel(av.reconstruct(chunk_bytes=1), av.reconstruct()) < 1e-6


def test_functional_api_unbatched_and_float64():
    torch.manual_seed(0)
    model = _randomise(_qwen35(), seed=3)
    av = st.capture_attention_values(model, _ids(4, batch=1))[3]
    w = st.value_weighted_edges(av.weights[0], av.values[0], av.out_weight, av.gate[0])
    assert w.shape == av.weights.shape[-2:]
    assert torch.allclose(w, av.edge_weights()[0])
    w64 = st.value_weighted_edges(av.weights.double(), av.values.double(),
                                  av.out_weight.double(), av.gate.double())
    assert w64.dtype == torch.float64
    c = st.attention_contributions(av.weights[0], av.values[0], av.out_weight, av.gate[0],
                                   queries=[2, -1])
    assert c.shape == (2, av.weights.shape[-1], av.out_weight.shape[0])
    assert torch.allclose(c[1], av.contributions()[0, -1], atol=1e-6)


def test_padded_batch_matches_single_runs():
    torch.manual_seed(0)
    model = _randomise(_llama(), seed=4)
    a, b = [5, 9, 13, 22, 7, 31], [8, 40, 12]
    both = st.capture_attention_values(model, [a, b])
    for lay, av in both.items():
        assert av.lengths == [6, 3]
        one = st.capture_attention_values(model, [b])[lay]
        e_both, e_one = av.edge_weights(), one.edge_weights()
        assert torch.allclose(e_both[1, :3, :3], e_one[0], rtol=1e-5, atol=1e-6)


def test_value_weighted_attention_and_layer_selection():
    torch.manual_seed(0)
    model = _randomise(_qwen35(), seed=5)
    out = st.value_weighted_attention(model, _ids(5), per_head=True)
    assert list(out) == [3] and out[3].shape == (2, 4, 13, 13)
    out = st.value_weighted_attention(model, _ids(5), layers=[-1])
    assert list(out) == [3] and out[3].shape == (2, 13, 13)
    with pytest.raises(ValueError, match="linear_attention"):
        st.value_weighted_attention(model, _ids(5), layers=[0])


# ── what the capture refuses ────────────────────────────────────────────────

def test_unknown_output_transformation_is_refused():
    torch.manual_seed(0)
    model = _randomise(_llama(), seed=6)
    proj = model.model.layers[1].self_attn.o_proj
    handle = proj.register_forward_pre_hook(lambda m, args: (args[0] * 0.5,))
    try:
        with pytest.raises(NotImplementedError, match="does not decompose"):
            st.capture_attention_values(model, _ids(6), layers=[1])
        st.capture_attention_values(model, _ids(6), layers=[0])      # layer 0 is plain
    finally:
        handle.remove()


def test_gpt2_reordered_attention_is_refused():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=V,
                                       reorder_and_upcast_attn=True)).eval()
    with pytest.raises(RuntimeError, match="eager_attention_forward"):
        st.capture_attention_values(model, _ids(7))
    assert not reg.CAPTURES and not reg._EAGER_FN


def test_shape_errors():
    w = torch.rand(1, 4, 5, 5)
    with pytest.raises(ValueError):
        st.value_weighted_edges(w, torch.rand(1, 3, 5, 8), torch.rand(16, 32))   # 4 % 3
    with pytest.raises(ValueError):
        st.value_weighted_edges(w, torch.rand(1, 2, 5, 8), torch.rand(16, 30))
    with pytest.raises(ValueError):
        st.value_weighted_edges(w, torch.rand(1, 2, 5, 8), torch.rand(16, 32),
                                gate=torch.rand(1, 5, 8))


# ── real checkpoints (opt-in) ───────────────────────────────────────────────

_REAL = [m for m in os.environ.get("SPECTRAL_TRUST_VALUE_MODELS", "").split(",") if m.strip()]


@pytest.mark.skipif(not _REAL, reason="set SPECTRAL_TRUST_VALUE_MODELS to run")
@pytest.mark.parametrize("name", _REAL or ["none"])
def test_real_checkpoint_reconstruction(name):
    model = st.SpectralModel.from_pretrained(name.strip(), local_files_only=True)
    text = "The quick brown fox jumps over the lazy dog because it was late."
    vals = st.capture_attention_values(model, text, keep_output=True)
    worst = max(_rel(av.reconstruct(), av.output) for av in vals.values())
    assert worst < TOL, worst
