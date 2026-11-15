# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Graph options of the layer graph (GSPConfig.edge_weights,
graph_direction, magnetic_q, sink_grounding) in GraphConstructor,
compute_spectra and SpectralModel.run_with_spectra.

The regression tests pin the default path: every call that does not set the
new options gives bit-identical results to the construction of 0.5.0, copied
verbatim below so that the library is checked against it and not against
itself."""
from dataclasses import replace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer  # noqa: E402
from spectral_trust.directed import analyze_directed_layer  # noqa: E402
from spectral_trust.interventions import _registry as reg  # noqa: E402
from spectral_trust.model import _fiedler_vector  # noqa: E402

torch.set_num_threads(2)
V = 40
CFG = GSPConfig(device="cpu", verbose=False, save_plots=False)


def _randomise(model, seed, scale=0.4):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return model.eval()


@pytest.fixture(scope="module")
def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=3, n_head=4, n_embd=32, vocab_size=V, n_positions=64))
    return _randomise(m, 0)


@pytest.fixture(scope="module")
def qwen35():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=V, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"]))
    return _randomise(m, 1)


def _ids(seed, n=12):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=g)


def _forward(model, ids):
    model.set_attn_implementation("eager")
    with torch.no_grad():
        return model(input_ids=ids[None], output_attentions=True, output_hidden_states=True,
                     use_cache=False)


def _graph_0_5_0(a, x, span, cfg, lay):
    """The graph namespace of compute_spectra in 0.5.0, verbatim."""
    gc = GraphConstructor(cfg)
    analyzer = SpectralAnalyzer(cfg)
    s, e = span
    n = a.shape[-1]
    full = (s, e) == (0, n)
    rec = {}
    sym = gc.symmetrize_attention(a.unsqueeze(0))             # [1, H, S, S]
    adj = gc.aggregate_heads(sym)                             # [1, S, S]
    if full:
        lap = gc.construct_laplacian(adj)[0]
    else:
        lap = gc.subgraph_laplacian(adj, list(range(s, e)))[0]
    sig = x[s:e] if x is not None else torch.zeros(e - s, 1)
    diag = analyzer.analyze_layer(sig, lap, lay)
    for name in st.GRAPH_METRIC_NAMES:
        rec[f"blocks.{lay}.graph.{name}"] = float(getattr(diag, name))
    rec[f"blocks.{lay}.graph.eigenvalues"] = np.asarray(diag.eigenvalues, dtype=np.float32)
    fv = _fiedler_vector(diag.eigenvectors)
    if fv is not None:
        rec[f"blocks.{lay}.graph.fiedler_vector"] = fv
    if x is not None:
        rec[f"blocks.{lay}.graph.spectral_masses"] = np.asarray(diag.spectral_masses,
                                                                dtype=np.float32)
    return rec


# ── regression: the default path is unchanged ───────────────────────────────

@pytest.mark.parametrize("cfg", [CFG, replace(CFG, normalization="rw"),
                                 replace(CFG, normalization="none", symmetrization="row_norm")])
@pytest.mark.parametrize("span", [None, (2, 10)])
def test_default_graph_namespace_is_bit_identical_to_0_5_0(gpt2, cfg, span):
    ids = _ids(0)
    out = _forward(gpt2, ids)
    model = st.SpectralModel(gpt2, None, config=cfg)
    cache = model.run_with_spectra(ids, metrics="graph", token_span=span)
    for lay in range(3):
        a = out.attentions[lay][0].float()
        x = out.hidden_states[lay + 1][0].float()
        ref = _graph_0_5_0(a, x, span or (0, 12), cfg, lay)
        for key, value in ref.items():
            got = cache[key]
            assert np.array_equal(np.asarray(got), np.asarray(value)), key
        keys = {k for k in cache.keys() if k.startswith(f"blocks.{lay}.graph.")}
        assert keys == set(ref)


def test_explicit_defaults_equal_the_plain_config(gpt2):
    ids = _ids(1)
    plain = st.SpectralModel(gpt2, None, config=CFG).run_with_spectra(ids)
    explicit = st.SpectralModel(gpt2, None, config=replace(
        CFG, edge_weights="attention", graph_direction="symmetric", magnetic_q=0.7,
        sink_grounding=None)).run_with_spectra(ids)
    assert plain.keys() == explicit.keys() and plain.metric_names == explicit.metric_names
    for key in plain.keys():
        assert np.array_equal(np.asarray(plain[key]), np.asarray(explicit[key])), key


def test_default_layer_graph_is_the_existing_construction():
    a = torch.rand(2, 3, 7, 7, generator=torch.Generator().manual_seed(2)).tril()
    weighted = replace(CFG, symmetrization="row_norm", head_aggregation="attention_weighted")
    for cfg in (CFG, weighted):
        gc = GraphConstructor(cfg)
        assert torch.equal(gc.layer_graph(a), gc.aggregate_heads(gc.symmetrize_attention(a)))
        mask = torch.ones(2, 7, dtype=torch.bool)
        assert torch.equal(gc.layer_graph(a, attn_mask=mask),
                           gc.aggregate_heads(gc.symmetrize_attention(a), attn_mask=mask))


def test_default_run_does_not_capture_values(gpt2, monkeypatch):
    from spectral_trust import value_weighted
    def refuse(*args, **kwargs):
        raise AssertionError("the default path must not capture values")
    monkeypatch.setattr(value_weighted, "ValueCapture", refuse)
    st.SpectralModel(gpt2, None, config=CFG).run_with_spectra(_ids(3))
    assert not reg.CAPTURES


# ── option checks and the sink-grounding hook ───────────────────────────────

def test_unknown_options_raise(gpt2):
    gc = GraphConstructor(replace(CFG, edge_weights="values"))
    with pytest.raises(ValueError, match="edge_weights"):
        gc.layer_graph(torch.rand(1, 2, 4, 4))
    with pytest.raises(ValueError, match="graph_direction"):
        st.SpectralModel(gpt2, None, config=replace(CFG, graph_direction="causal")
                         ).run_with_spectra(_ids(4))


def test_sink_grounding_is_reserved(gpt2):
    cfg = replace(CFG, sink_grounding="sink")
    with pytest.raises(NotImplementedError, match="sink_grounding"):
        GraphConstructor(cfg).layer_graph(torch.rand(1, 2, 4, 4))
    with pytest.raises(NotImplementedError, match="sink_grounding"):
        st.SpectralModel(gpt2, None, config=cfg).run_with_spectra(_ids(4))
    with pytest.raises(NotImplementedError, match="sink_grounding"):
        st.compute_spectra({0: torch.rand(1, 2, 4, 4)}, tokens=[list("abcd")], config=cfg)


def test_value_edges_need_the_values():
    cfg = replace(CFG, edge_weights="value")
    with pytest.raises(ValueError, match="value states"):
        st.compute_spectra({0: torch.rand(1, 2, 4, 4)}, tokens=[list("abcd")], config=cfg)


def test_symmetric_only_tools_refuse_the_new_options(tmp_path):
    from spectral_trust.interventions import attention_adjacency
    from types import SimpleNamespace
    for cfg in (replace(CFG, edge_weights="value"), replace(CFG, graph_direction="directed")):
        with pytest.raises(ValueError, match="symmetrised attention"):
            attention_adjacency(torch.rand(1, 2, 4, 4), cfg)
        fw = st.GSPDiagnosticsFramework.__new__(st.GSPDiagnosticsFramework)
        fw.config = cfg
        fw.instrumenter = SimpleNamespace(process_text=lambda t: None)
        with pytest.raises(ValueError, match="SpectralModel"):
            fw.analyze_text("x", save_results=False)


def test_directed_metric_needs_a_directed_config(gpt2):
    with pytest.raises(ValueError, match="directed"):
        st.SpectralModel(gpt2, None, config=CFG).run_with_spectra(_ids(5), metrics=["directed_tv"])


# ── value-weighted edges in the pipeline ────────────────────────────────────

@pytest.mark.parametrize("mode", ["value", "value_per_head"])
def test_value_weighted_graph_namespace(gpt2, qwen35, mode):
    for model, layers in ((gpt2, [0, 1, 2]), (qwen35, [3])):
        ids = _ids(6)
        cfg = replace(CFG, edge_weights=mode)
        cache = st.SpectralModel(model, None, config=cfg).run_with_spectra(ids)
        plain = st.SpectralModel(model, None, config=CFG).run_with_spectra(ids)
        assert cache.layers == layers
        edges = st.value_weighted_attention(model, ids, per_head=(mode == "value_per_head"))
        out = _forward(model, ids)
        gc, an = GraphConstructor(cfg), SpectralAnalyzer(cfg)
        for lay in layers:
            x = out.hidden_states[lay + 1][0]
            lap = gc.construct_laplacian(gc.layer_graph(edges[lay]))[0]
            ref = an.analyze_layer(x, lap, lay)
            for name in st.GRAPH_METRIC_NAMES:
                assert cache[f"blocks.{lay}.graph.{name}"] == pytest.approx(
                    float(getattr(ref, name)), rel=1e-5, abs=1e-6), (lay, name)
            # the per-head namespaces still measure each head's attention
            for key in plain.keys():
                if ".attn." in key or ".resid." in key:
                    assert np.array_equal(np.asarray(cache[key]), np.asarray(plain[key])), key
            # value weighting moves the graph
            assert cache[f"blocks.{lay}.graph.fiedler_value"] != plain[
                f"blocks.{lay}.graph.fiedler_value"]


# ── directed graphs in the pipeline ─────────────────────────────────────────

def test_directed_graph_namespace(gpt2):
    ids = _ids(7)
    cfg = replace(CFG, graph_direction="directed")
    cache = st.SpectralModel(gpt2, None, config=cfg).run_with_spectra(ids)
    out = _forward(gpt2, ids)
    gc = GraphConstructor(cfg)
    assert "graph.directed_tv" in cache.metric_names
    assert cache.families["graph.directed_tv"] == "hybrid"
    for lay in range(3):
        a = out.attentions[lay][0].float()
        x = out.hidden_states[lay + 1][0]
        P = gc.layer_graph(a[None])[0]
        assert torch.allclose(P.sum(-1), torch.ones(12), atol=1e-6)
        assert torch.equal(P, P.tril())                          # causal
        diag, extra = analyze_directed_layer(x, P, cfg, lay)
        for name in st.GRAPH_METRIC_NAMES:
            assert cache[f"blocks.{lay}.graph.{name}"] == pytest.approx(
                float(getattr(diag, name)), rel=1e-6, abs=1e-9), (lay, name)
        assert cache[f"blocks.{lay}.graph.directed_tv"] == pytest.approx(
            float(st.directed_total_variation(x, P)), rel=1e-5)
        assert cache[f"blocks.{lay}.graph.phase"].shape == (12,)
        assert f"blocks.{lay}.graph.fiedler_vector" not in cache
        lam = cache[f"blocks.{lay}.graph.eigenvalues"]
        assert lam.min() >= 0 and lam.max() <= 2 + 1e-6


def test_directed_at_q0_matches_the_symmetric_graph(gpt2):
    ids = _ids(8)
    sym = st.SpectralModel(gpt2, None, config=CFG).run_with_spectra(ids, metrics="graph")
    dq0 = st.SpectralModel(gpt2, None, config=replace(CFG, graph_direction="directed",
                                                       magnetic_q=0.0)).run_with_spectra(
        ids, metrics="graph")
    for lay in range(3):
        for name in ("hfer", "energy", "smoothness_index", "spectral_entropy", "fiedler_value",
                     "eig_energy"):
            assert dq0[f"blocks.{lay}.graph.{name}"] == pytest.approx(
                sym[f"blocks.{lay}.graph.{name}"], rel=1e-4, abs=1e-6), (lay, name)


def test_directed_value_weighted_span_and_round_trip(qwen35, tmp_path):
    ids = _ids(9, n=14)
    cfg = replace(CFG, graph_direction="directed", edge_weights="value", magnetic_q=0.1)
    model = st.SpectralModel(qwen35, None, config=cfg)
    cache = model.run_with_spectra(ids, token_span=(3, 14))
    edges = st.value_weighted_attention(qwen35, ids)[3][0]
    P = st.directed_operator(edges[3:, 3:])                       # the span's own edges
    x = _forward(qwen35, ids).hidden_states[4][0, 3:]
    diag, extra = analyze_directed_layer(x, P, cfg, 3)
    assert cache["blocks.3.graph.hfer"] == pytest.approx(diag.hfer, rel=1e-5)
    assert cache["blocks.3.graph.directed_tv"] == pytest.approx(extra["directed_tv"], rel=1e-5)
    back = st.SpectralCache.load(cache.save(tmp_path / "directed"))
    assert back.config.graph_direction == "directed" and back.config.magnetic_q == 0.1
    assert back["blocks.3.graph.directed_tv"] == cache["blocks.3.graph.directed_tv"]
    assert np.array_equal(back["blocks.3.graph.phase"], cache["blocks.3.graph.phase"])
    df = cache.to_dataframe()
    assert "graph.directed_tv" in set(df["metric"])
    no_hybrid = model.run_with_spectra(ids, hybrid=False)
    assert "graph.directed_tv" not in no_hybrid.metric_names
