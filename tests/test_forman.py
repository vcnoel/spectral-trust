# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Forman curvature of attention graphs (spectral_trust.forman): closed forms
on small graphs, the standard undirected formula, and the edge weights and
summaries of Kim, Lutzeyer and Guo (2026)."""
import math

import numpy as np
import pytest
import torch

from spectral_trust.forman import (
    FORMAN_EPS, forman_curvature, forman_node_curvature, forman_quantiles,
)

D = torch.float64


def path(n):
    """Causal path: token i attends to i - 1."""
    a = torch.zeros(n, n, dtype=D)
    a[torch.arange(1, n), torch.arange(n - 1)] = 1.0
    return a


def edges(F):
    return {(int(i), int(j)): float(F[i, j]) for i, j in zip(*torch.where(~torch.isnan(F)))}


def test_directed_path():
    assert edges(forman_curvature(path(2), weights="unit")) == {(1, 0): 2.0}
    F = edges(forman_curvature(path(6), weights="unit"))
    # (i, i-1) has one edge into i unless i is last, one out of i-1 unless i-1 is 0.
    assert F == {(1, 0): 1.0, (2, 1): 0.0, (3, 2): 0.0, (4, 3): 0.0, (5, 4): 1.0}


@pytest.mark.parametrize("weights", ["unit", "neg_log", "attention"])
def test_directed_cycle_is_flat(weights):
    n = 7
    a = torch.zeros(n, n, dtype=D)
    a[torch.arange(n), (torch.arange(n) + 1) % n] = 0.5        # equal weights on a cycle
    F = edges(forman_curvature(a, weights=weights))
    assert len(F) == n and all(v == pytest.approx(0.0, abs=1e-12) for v in F.values())


def test_stars():
    n = 6
    sink = torch.zeros(n, n, dtype=D)
    sink[1:, 0] = 1.0                                          # every token reads token 0
    assert set(edges(forman_curvature(sink, weights="unit")).values()) == {2.0}
    hub = sink.T.clone()                                       # token 0 reads every token
    assert set(edges(forman_curvature(hub, weights="unit")).values()) == {2.0}
    star = sink + hub                                          # undirected K_{1,k}, k = n - 1
    F = edges(forman_curvature(star, directed=False, weights="unit"))
    assert set(F.values()) == {3.0 - (n - 1)} and len(F) == 2 * (n - 1)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_complete_graphs(n):
    full = torch.ones(n, n, dtype=D)
    und = edges(forman_curvature(full, directed=False, weights="unit"))
    assert set(und.values()) == {6.0 - 2 * n}                   # 4 - 2(n - 1)
    dirc = edges(forman_curvature(full, directed=True, weights="unit"))
    assert set(dirc.values()) == {4.0 - 2 * n}                  # 2 - (n - 1) - (n - 1)
    causal = edges(forman_curvature(torch.ones(n, n, dtype=D).tril(), weights="unit"))
    for (i, j), v in causal.items():
        assert i > j and v == 2.0 - (n - 1 - i) - j             # in-edges of i, out-edges of j


def sreejith_undirected(a):
    """Forman curvature of a weighted undirected graph with unit vertex
    weights (Sreejith et al. 2016), edge by edge."""
    n = a.shape[0]
    F = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j or a[i, j] <= 0:
                continue
            we = a[i, j]
            s1 = sum(1 / math.sqrt(we * a[i, k]) for k in range(n) if k not in (i, j) and a[i, k] > 0)
            s2 = sum(1 / math.sqrt(we * a[j, k]) for k in range(n) if k not in (i, j) and a[j, k] > 0)
            F[i, j] = we * (1 / we + 1 / we - s1 - s2)
    return F


def test_undirected_matches_the_standard_formula():
    g = torch.Generator().manual_seed(0)
    w = (torch.rand(9, 9, generator=g, dtype=D) + 0.1) * (torch.rand(9, 9, generator=g) < 0.5)
    w = torch.maximum(w, w.T).fill_diagonal_(0)
    F = forman_curvature(w, directed=False, weights="attention").numpy()
    assert np.allclose(F, sreejith_undirected(w.numpy()), equal_nan=True, atol=1e-12)
    assert np.allclose(F, F.T, equal_nan=True)
    unit = forman_curvature(w, directed=False, weights="unit").numpy()
    deg = (w > 0).sum(1).numpy()
    for i, j in zip(*np.where(~np.isnan(unit))):
        assert unit[i, j] == 4 - deg[i] - deg[j]
    with pytest.raises(ValueError, match="symmetric"):
        forman_curvature(torch.rand(4, 4, dtype=D), directed=False)


def test_negative_log_weights_as_in_the_paper():
    # Path 3 -> 2 -> 1 -> 0 with attention 0.2, 0.5, 0.9 on its edges.
    a = torch.zeros(4, 4, dtype=D)
    a[1, 0], a[2, 1], a[3, 2] = 0.9, 0.5, 0.2
    w = {e: max(-math.log(v), FORMAN_EPS) for e, v in {(1, 0): 0.9, (2, 1): 0.5, (3, 2): 0.2}.items()}
    F = edges(forman_curvature(a))                             # default weights="neg_log"
    s = {e: math.sqrt(v) for e, v in w.items()}
    assert F[(1, 0)] == pytest.approx(2 - s[(1, 0)] / s[(2, 1)])
    assert F[(2, 1)] == pytest.approx(2 - s[(2, 1)] * (1 / s[(3, 2)] + 1 / s[(1, 0)]))
    assert F[(3, 2)] == pytest.approx(2 - s[(3, 2)] / s[(2, 1)])
    # Attention 1 gets the floor eps; self-loops and zero entries are no edges;
    # rows are not renormalised.
    b = a.clone()
    b[2, 1], b[2, 2], b[3, 3] = 1.0, 0.7, 0.3
    Fb = edges(forman_curvature(b))
    assert set(Fb) == {(1, 0), (2, 1), (3, 2)}
    assert Fb[(2, 1)] == pytest.approx(2 - math.sqrt(FORMAN_EPS) * (1 / s[(3, 2)] + 1 / s[(1, 0)]))
    assert Fb[(1, 0)] == pytest.approx(2 - s[(1, 0)] / math.sqrt(FORMAN_EPS))
    half = edges(forman_curvature(a * 0.5))
    assert half[(1, 0)] != pytest.approx(F[(1, 0)])
    # Weights "attention" and a callable.
    att = edges(forman_curvature(a, weights="attention"))
    assert att[(1, 0)] == pytest.approx(2 - math.sqrt(0.9) / math.sqrt(0.5))
    cal = forman_curvature(a, weights=lambda x: 2 * torch.ones_like(x))
    assert torch.equal(torch.isnan(cal), torch.isnan(forman_curvature(a, weights="unit")))
    with pytest.raises(ValueError, match="weights"):
        forman_curvature(a, weights="log")
    with pytest.raises(ValueError, match="non-negative"):
        forman_curvature(-a)


def causal_attention(n, heads=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (n, n) if heads is None else (heads, n, n)
    logits = torch.randn(shape, generator=g, dtype=D)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def test_final_token_edge_to_the_first_token_has_curvature_two():
    # Appendix A: on a prefix graph, the final token has no incoming edge and
    # the first token no outgoing edge, so their edge has curvature exactly 2,
    # the maximum quantile of the final token's summary.
    a = causal_attention(12, heads=4, seed=1).mean(0)
    for T in (5, 9, 12):
        F = forman_curvature(a[:T, :T])
        assert float(F[T - 1, 0]) == 2.0
        assert float(np.nanmax(F.numpy())) <= 2.0
        assert float(forman_quantiles(F, tokens=T - 1)[-1]) == 2.0


def test_quantile_summary_pools_outgoing_edges():
    a = causal_attention(10, seed=2)
    F = forman_curvature(a)
    q = (0.0, 0.25, 0.5, 0.75, 1.0)
    pooled = F[4:8].numpy()
    want = np.quantile(pooled[~np.isnan(pooled)], q)           # linear interpolation
    assert np.allclose(forman_quantiles(F, tokens=(4, 8)).numpy(), want)
    assert np.allclose(forman_quantiles(F, tokens=[4, 5, 6, 7]).numpy(), want)
    one = F[6].numpy()
    assert np.allclose(forman_quantiles(F, tokens=6).numpy(), np.quantile(one[~np.isnan(one)], q))
    # The pooled summary is not the mean of the token summaries.
    per_token = torch.stack([forman_quantiles(F, tokens=t) for t in range(4, 8)]).mean(0)
    assert not torch.allclose(per_token, forman_quantiles(F, tokens=(4, 8)))
    # Token 0 has no outgoing edge once self-loops are removed: zeros.
    assert torch.equal(forman_quantiles(F, tokens=0), torch.zeros(5, dtype=D))
    assert torch.equal(forman_quantiles(F, tokens=0, empty=-1.0), -torch.ones(5, dtype=D))
    incoming = F[:, 3].numpy()
    assert np.allclose(forman_quantiles(F, tokens=3, direction="in").numpy(),
                       np.quantile(incoming[~np.isnan(incoming)], q))


def test_node_curvature_and_batches():
    A = causal_attention(8, heads=3, seed=3)
    F = forman_curvature(A)                                     # per head, batched
    assert F.shape == (3, 8, 8)
    for h in range(3):
        assert torch.equal(torch.isnan(F[h]), torch.isnan(forman_curvature(A[h])))
        assert torch.allclose(F[h].nan_to_num(), forman_curvature(A[h]).nan_to_num())
    avg = forman_curvature(A, average_heads=True)
    assert torch.allclose(avg.nan_to_num(), forman_curvature(A.mean(0)).nan_to_num())
    node = forman_node_curvature(avg)
    assert node[0] == 0.0                                       # no outgoing edge
    for t in range(1, 8):
        row = avg[t].numpy()
        assert float(node[t]) == pytest.approx(np.nanmean(row))
        assert float(forman_node_curvature(avg, reduce="min")[t]) == pytest.approx(np.nanmin(row))
    assert forman_node_curvature(avg, direction="in")[7] == 0.0  # nobody reads the last token
    assert forman_quantiles(F).shape == (3, 5)
    with pytest.raises(ValueError, match="reduce"):
        forman_node_curvature(avg, reduce="mode")
    with pytest.raises(ValueError, match="direction"):
        forman_quantiles(avg, direction="both")
