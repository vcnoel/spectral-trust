# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""The graph views and the animations show the graph the metrics were
computed on: the value-weighted graph under edge_weights="value" or
"value_per_head", and a clear error for a directed graph."""
from dataclasses import replace

import numpy as np
import pytest
import torch

import spectral_trust as st
from spectral_trust import viz
from spectral_trust.animation._data import layer_graph

CFG = st.GSPConfig(device="cpu", verbose=False, save_plots=False)
H, S, D = 4, 12, 16


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(1, H, S, S, generator=g).tril()
    A = A / A.sum(-1, keepdim=True)
    X = torch.randn(1, S, D, generator=g)
    E = torch.rand(1, S, S, generator=g).tril() * 5               # value edges, summed heads
    Eh = torch.rand(1, H, S, S, generator=g).tril() * 5           # value edges, per head
    return A, X, E, Eh, [[f"t{i}" for i in range(S)]]


def _cache(cfg, edges=None, span=None, keep=True):
    A, X, _, _, toks = _inputs()
    return st.compute_spectra({0: A}, {0: X}, tokens=toks, config=cfg, keep_patterns=keep,
                              token_span=span, edges=None if edges is None else {0: edges})


@pytest.mark.parametrize("mode", ["value", "value_per_head"])
@pytest.mark.parametrize("span", [None, (2, 10)])
def test_value_graph_view_uses_the_measured_graph(mode, span):
    _, _, E, Eh, _ = _inputs()
    c = _cache(replace(CFG, edge_weights=mode), E if mode == "value" else Eh, span)
    W, vals, u, tokens, offset, vecs = viz._graph_data(c, 0, None, vectors=True)
    assert float(vals[1]) == pytest.approx(float(c["blocks.0.graph.fiedler_value"]), abs=1e-6)
    assert np.allclose(vals, c["blocks.0.graph.eigenvalues"], atol=1e-5)
    # The stored Fiedler vector is an eigenvector of the graph the view builds.
    lap = st.GraphConstructor(c.config).construct_laplacian(torch.as_tensor(W[None]).float())[0]
    fv = c["blocks.0.graph.fiedler_vector"]
    assert np.allclose(lap.double().numpy() @ fv, vals[1] * fv, atol=1e-4)
    assert np.allclose(u, fv)
    s, e = span or (0, S)
    assert len(tokens) == e - s and offset == s
    assert np.array_equal(W, c["blocks.0.graph.adjacency"][s:e, s:e])
    assert isinstance(viz.graph(c, 0), viz.View)
    g = layer_graph(c, 0)                                         # the animations' graph
    assert np.allclose(g.eigenvalues, vals) and np.array_equal(g.W, W)


def test_directed_graph_views_raise_and_heads_still_work():
    c = _cache(replace(CFG, graph_direction="directed"))
    for call in (lambda: viz.graph(c, 0), lambda: viz._graph_data(c, 0, None),
                 lambda: layer_graph(c, 0)):
        with pytest.raises(ValueError, match="directed"):
            call()
    assert isinstance(viz.graph(c, 0, head=1), viz.View)


def test_value_graph_without_kept_patterns_raises():
    _, _, E, _, _ = _inputs()
    c = _cache(replace(CFG, edge_weights="value"), E, keep=False)
    assert "blocks.0.graph.adjacency" not in c
    with pytest.raises(ValueError, match="keep_patterns"):
        viz.graph(c, 0)


def test_default_graph_view_is_unchanged():
    c = _cache(CFG)
    assert "blocks.0.graph.adjacency" not in c                    # no new key on default runs
    A = _inputs()[0]
    gc = st.GraphConstructor(CFG)
    W, vals, u, *_ = viz._graph_data(c, 0, None)
    assert np.array_equal(W, gc.aggregate_heads(gc.symmetrize_attention(A))[0].numpy())
    assert float(vals[1]) == pytest.approx(float(c["blocks.0.graph.fiedler_value"]), abs=1e-6)
