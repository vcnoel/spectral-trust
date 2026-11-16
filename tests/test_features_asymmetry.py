"""Asymmetry G and the sweep conductance estimate of Dahlem et al.
(arXiv 2605.04893), checked against their stated properties and against
brute force."""
from itertools import product

import numpy as np
import pytest
import torch

from spectral_trust import features as F


def _causal(n, seed, shape=()):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(*shape, n, n, generator=g).tril() + 1e-3 * torch.eye(n)
    return (a / a.sum(-1, keepdim=True)).double()


def test_transport_operator_matches_definition_and_has_unit_top_singular_value():
    B = _causal(7, 0)
    dq, dk = B.sum(1), B.sum(0)
    M = torch.diag(dq ** -0.5) @ B @ torch.diag(dk ** -0.5)
    assert torch.allclose(F.transport_operator(B), M)
    assert torch.linalg.svdvals(M)[0].item() == pytest.approx(1.0, abs=1e-12)


def test_asymmetry_extremes():
    S = torch.rand(5, 5, generator=torch.Generator().manual_seed(1))
    assert F.asymmetry(S + S.T) == pytest.approx(0.0, abs=1e-15)
    assert F.asymmetry(S - S.T, normalize=None) == pytest.approx(1.0)
    strict = torch.rand(6, 6, generator=torch.Generator().manual_seed(2)).tril(-1)
    assert F.asymmetry(strict) == pytest.approx(2 ** -0.5)
    assert F.asymmetry(strict, normalize=None) == pytest.approx(2 ** -0.5)
    assert F.asymmetry(torch.zeros(3, 3)) == 0.0


def test_asymmetry_formula_transpose_invariance_and_causal_bound():
    B = _causal(8, 3, shape=(2, 3))
    G = F.asymmetry(B)
    assert G.shape == (2, 3)
    M = F.transport_operator(B[1, 2]).numpy()
    want = np.linalg.norm((M - M.T) / 2) / np.linalg.norm(M)
    assert G[1, 2] == pytest.approx(want, rel=1e-12)
    assert F.asymmetry(B[1, 2].T) == pytest.approx(G[1, 2], rel=1e-12)
    assert F.asymmetry(B[0, 0], eps=1.0) < G[0, 0]
    assert np.all(G <= 2 ** -0.5 + 1e-12)
    cov = F.attention_covariates(B)
    np.testing.assert_allclose(cov["asymmetry"], G)


def _conductance(X, S_mask):
    nq = X.shape[0]
    W = np.block([[np.zeros((nq, nq)), X], [X.T, np.zeros((X.shape[1],) * 2)]])
    deg = W.sum(1)
    cut = W[np.ix_(S_mask, ~S_mask)].sum()
    small = min(deg[S_mask].sum(), deg[~S_mask].sum())
    return cut / small if small > 0 else np.inf


def test_sweep_matches_a_naive_sweep_and_bounds_the_exact_conductance():
    B = _causal(4, 4)
    M = F.transport_operator(B).numpy()
    u, s, vt = np.linalg.svd(M)
    order = np.argsort(-np.concatenate([u[:, 1], vt[1]]), kind="stable")
    for graph, X in (("normalized", M), ("raw", B.numpy())):
        naive = []
        for t in range(1, 8):
            mask = np.zeros(8, bool)
            mask[order[:t]] = True
            naive.append(_conductance(X, mask))
        assert F.capacity_phi(B, graph=graph) == pytest.approx(min(naive), rel=1e-12)
        exact = min(_conductance(X, np.array(m, bool)) for m in product([0, 1], repeat=8)
                    if 0 < sum(m) < 8)
        assert F.capacity_phi(B, graph=graph) >= exact - 1e-12
    gap = 1 - s[1]
    assert gap / 2 - 1e-12 <= F.capacity_phi(B, graph="raw")   # Cheeger: on A_bip = H(B)
    assert F.transport_sigma2(B) == pytest.approx(s[1], rel=1e-12)


def test_disconnected_heads_have_zero_conductance():
    B = torch.zeros(6, 6, dtype=torch.float64)
    B[:3, :3] = _causal(3, 5)
    B[3:, 3:] = _causal(3, 6)
    assert F.transport_sigma2(B) == pytest.approx(1.0, abs=1e-12)
    assert F.capacity_phi(B) == pytest.approx(0.0, abs=1e-12)
    assert F.capacity_phi(_causal(5, 7, shape=(2,))).shape == (2,)
    with pytest.raises(ValueError):
        F.capacity_phi(B, graph="other")
