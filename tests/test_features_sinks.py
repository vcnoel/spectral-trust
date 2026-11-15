"""Sink mass, value norms, the value-weighted sink score, effective rank and
the attention covariates of spectral_trust.features."""
import numpy as np
import pytest
import torch

from spectral_trust import features as F


def _causal(shape, seq, seed, sink=2.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(*shape, seq, seq, generator=g).tril()
    a[..., 0] += sink * a.sum(-1)
    return a / a.sum(-1, keepdim=True)


def test_sink_mass_is_the_mean_over_later_queries():
    a = _causal((2, 3), 6, 0)
    got = F.sink_mass(a)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, a[..., 1:, 0].mean(-1).double().numpy())
    np.testing.assert_allclose(F.sink_mass(a, sink=2), a[..., 3:, 2].mean(-1).double().numpy())
    assert isinstance(F.sink_mass(a[0, 0]), float)
    assert F.sink_mass(torch.ones(1, 1)) == 0.0
    with pytest.raises(ValueError):
        F.sink_mass(torch.ones(3, 4))


def _brute_force_sink_scores(A, V, W, sink=0):
    """Mean over queries after the sink of ||W_O^h (A[h, i, sink] v^h_sink)||."""
    H, S, dh = V.shape
    out = np.zeros(H)
    for h in range(H):
        Wh = W[:, h * dh:(h + 1) * dh]
        norms = [np.linalg.norm(Wh @ (A[h, i, sink] * V[h, sink])) for i in range(sink + 1, S)]
        out[h] = np.mean(norms)
    return out


def test_value_weighted_sink_matches_the_brute_force_contribution():
    H, S, dh, D = 3, 7, 4, 10
    rng = np.random.default_rng(0)
    A = _causal((H,), S, 1).double().numpy()
    V, W = rng.normal(size=(H, S, dh)), rng.normal(size=(D, H * dh))
    np.testing.assert_allclose(F.value_weighted_sink(A, V, W), _brute_force_sink_scores(A, V, W),
                               rtol=1e-10)
    norms = F.value_norms(V, W)
    assert norms.shape == (H, S)
    np.testing.assert_allclose(norms[1, 4], np.linalg.norm(W[:, dh:2 * dh] @ V[1, 4]))


def test_grouped_query_values_are_repeated_to_the_query_heads():
    rng = np.random.default_rng(1)
    Vkv, W = rng.normal(size=(2, 5, 3)), rng.normal(size=(8, 4 * 3))
    full = np.repeat(Vkv, 2, axis=0)
    np.testing.assert_allclose(F.value_norms(Vkv, W), F.value_norms(full, W))
    with pytest.raises(ValueError):
        F.value_norms(rng.normal(size=(3, 5, 3)), W)


def test_value_weighted_sink_on_a_grouped_query_llama_layer():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                                   num_hidden_layers=1, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).eval()
    model.config._attn_implementation = "eager"
    attn = model.model.layers[0].self_attn
    seen = {}
    h1 = attn.v_proj.register_forward_hook(lambda m, i, o: seen.__setitem__("v", o))
    h2 = attn.o_proj.register_forward_pre_hook(lambda m, i: seen.__setitem__("o_in", i[0]))
    with torch.no_grad():
        out = model(torch.randint(0, 64, (1, 9)), output_attentions=True)
    h1.remove(), h2.remove()
    A = out.attentions[0][0].double()                                  # [4, 9, 9]
    V = seen["v"][0].double().reshape(9, 2, 8).transpose(0, 1)         # [2, 9, 8]
    Vq = V.repeat_interleave(2, dim=0)
    o_in = seen["o_in"][0].double().reshape(9, 4, 8).transpose(0, 1)   # [4, 9, 8]
    assert torch.allclose(o_in, A @ Vq, atol=1e-6)                     # the layout is right
    W = attn.o_proj.weight.double()
    want = _brute_force_sink_scores(A.numpy(), Vq.numpy(), W.detach().numpy())
    np.testing.assert_allclose(F.value_weighted_sink(A, V, W), want, rtol=1e-10)


def test_effective_rank_extremes_and_formula():
    assert F.effective_rank(torch.eye(5)) == pytest.approx(5.0)
    assert F.effective_rank(np.outer(np.arange(1, 5), np.ones(3))) == pytest.approx(1.0)
    assert F.effective_rank(np.zeros((3, 3))) == 0.0
    X = np.random.default_rng(2).normal(size=(2, 6, 4))
    s = np.linalg.svd(X[1], compute_uv=False)
    p = s / s.sum()
    got = F.effective_rank(X)
    assert got.shape == (2,)
    assert got[1] == pytest.approx(np.exp(-(p * np.log(p)).sum()), rel=1e-12)


def test_attention_covariates_on_a_hand_map():
    A = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])
    cov = F.attention_covariates(A)
    assert set(cov) == set(F.ATTENTION_COVARIATES) - {"value_weighted_sink"}
    assert cov["sink_mass"] == pytest.approx(0.375)
    assert cov["diagonal_mass"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert cov["previous_token_mass"] == pytest.approx((0.5 + 0.25) / 2)
    h = [0.0, np.log(2), -(0.5 * np.log(0.25) + 0.5 * np.log(0.5))]
    assert cov["row_entropy"] == pytest.approx(np.mean(h))
    assert cov["n_tokens"] == 3
    rng = np.random.default_rng(3)
    a4 = _causal((2, 3), 5, 4)
    cov = F.attention_covariates(a4, rng.normal(size=(2, 3, 5, 2)), rng.normal(size=(4, 6)))
    assert cov["value_weighted_sink"].shape == (2, 3)
    assert cov["attention_effective_rank"].shape == (2, 3)
