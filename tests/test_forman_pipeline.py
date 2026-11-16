# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""graph.forman_curvature_mean and graph.forman_quantiles in the pipeline, on
plain or value-weighted, symmetric or directed layer graphs."""
from dataclasses import replace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust import GraphConstructor, value_weighted  # noqa: E402
from spectral_trust.cache import resolve_metrics  # noqa: E402
from spectral_trust.directed import directed_operator  # noqa: E402
from spectral_trust.forman import graph_curvature_summary  # noqa: E402

torch.set_num_threads(2)
V = 40
CFG = st.GSPConfig(device="cpu", verbose=False, save_plots=False)
KEY = "graph.forman_curvature_mean"


@pytest.fixture(scope="module")
def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=4, n_embd=32, vocab_size=V, n_positions=64))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.4)
    m.set_attn_implementation("eager")
    return m.eval()


def _ids(seed, n=11):
    return torch.randint(1, V, (n,), generator=torch.Generator().manual_seed(seed))


def _attn(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids[None], output_attentions=True, use_cache=False)
    return [a[0].float() for a in out.attentions]


def _run(model, ids, metrics=KEY, span=None, **opts):
    return st.SpectralModel(model, None, config=replace(CFG, **opts)).run_with_spectra(
        ids, metrics=metrics, token_span=span)


def test_forman_metric_is_opt_in():
    assert KEY not in resolve_metrics(None) and KEY not in resolve_metrics("graph")
    assert resolve_metrics("forman_curvature_mean") == [KEY]
    assert resolve_metrics([KEY], hybrid=False) == [KEY]


@pytest.mark.parametrize("span", [None, (2, 9)])
def test_attention_graphs(gpt2, span):
    ids = _ids(0)
    s, e = span or (0, 11)
    sym, dirc = _run(gpt2, ids, span=span), _run(gpt2, ids, span=span, graph_direction="directed")
    gc = GraphConstructor(CFG)
    for lay, a in enumerate(_attn(gpt2, ids)):
        G = gc.aggregate_heads(gc.symmetrize_attention(a[None]))[0][s:e, s:e]
        mean, quant = graph_curvature_summary(G, directed=False)
        assert sym[f"blocks.{lay}.{KEY}"] == pytest.approx(mean, rel=1e-6)
        assert np.allclose(sym[f"blocks.{lay}.graph.forman_quantiles"], quant, rtol=1e-5)
        P = directed_operator(a[:, s:e, s:e].mean(0))                # the span's own edges
        mean, quant = graph_curvature_summary(P, directed=True)
        assert dirc[f"blocks.{lay}.{KEY}"] == pytest.approx(mean, rel=1e-6)
        assert np.allclose(dirc[f"blocks.{lay}.graph.forman_quantiles"], quant, rtol=1e-5)
        assert dirc[f"blocks.{lay}.graph.forman_quantiles"][-1] == pytest.approx(2.0)


@pytest.mark.parametrize("mode", ["value", "value_per_head"])
@pytest.mark.parametrize("direction", ["symmetric", "directed"])
def test_value_weighted_graphs(gpt2, mode, direction):
    ids = _ids(1)
    cache = _run(gpt2, ids, edge_weights=mode, graph_direction=direction)
    vals = value_weighted.capture_attention_values(gpt2, ids[None])
    for lay in range(2):
        w = vals[lay].edge_weights(per_head=mode == "value_per_head")[0].float()
        P = directed_operator(w.mean(0) if w.dim() == 3 else w)
        G = P if direction == "directed" else 0.5 * (P + P.T)
        mean, _ = graph_curvature_summary(G, directed=direction == "directed")
        assert cache[f"blocks.{lay}.{KEY}"] == pytest.approx(mean, rel=1e-5), (mode, lay)
    plain = _run(gpt2, ids, graph_direction=direction)
    assert cache[f"blocks.1.{KEY}"] != plain[f"blocks.1.{KEY}"]


def test_weights_option_and_errors(gpt2):
    ids = _ids(2)
    unit = _run(gpt2, ids, forman_weights="unit", span=(1, 8))
    for lay in range(2):                                   # complete graph on 7 tokens
        assert unit[f"blocks.{lay}.{KEY}"] == pytest.approx(6 - 2 * 7)
    att = _run(gpt2, ids, forman_weights="attention")
    assert att[f"blocks.0.{KEY}"] != _run(gpt2, ids)[f"blocks.0.{KEY}"]
    with pytest.raises(ValueError, match="forman_weights"):
        _run(gpt2, ids, forman_weights="log")


def test_curvature_leaves_the_other_graph_keys_bit_identical(gpt2):
    ids = _ids(3)
    for opts in ({}, dict(graph_direction="directed"), dict(edge_weights="value")):
        plain = _run(gpt2, ids, metrics="graph", **opts)
        both = _run(gpt2, ids, metrics=["graph", KEY], **opts)
        assert set(both.keys()) - set(plain.keys()) == {
            f"blocks.{lay}.graph.{m}" for lay in range(2)
            for m in ("forman_curvature_mean", "forman_quantiles")}
        for key in plain.keys():
            assert np.array_equal(np.asarray(plain[key]), np.asarray(both[key])), key
    only = _run(gpt2, ids)
    assert set(only.metric_names) == {KEY}
    assert only.to_dataframe().metric.unique().tolist() == [KEY]
