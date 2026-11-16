"""GSPConfig.subgraph_renormalize and graph.drop_and_renormalize: drop nodes
(the BOS sink) and renormalise the rows before the graph is built."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import spectral_trust as st
from spectral_trust.graph import drop_and_renormalize, renormalized_within


def _causal(heads, seq, seed, sink=3.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(heads, seq, seq, generator=g).tril()
    a[..., 0] += sink * a.sum(-1)
    return a / a.sum(-1, keepdim=True)


def _cfg(**kw):
    return st.GSPConfig(device="cpu", save_plots=False, verbose=False, **kw)


def test_rows_sum_to_one_and_match_the_manual_construction():
    a = _causal(3, 9, 0)
    out = drop_and_renormalize(a)
    manual = a[:, 1:, 1:] / a[:, 1:, 1:].sum(-1, keepdim=True)
    assert out.shape == (3, 8, 8)
    assert torch.allclose(out, manual)
    assert torch.allclose(out.sum(-1), torch.ones(3, 8))
    assert torch.equal(drop_and_renormalize(a, keep=range(1, 9)), out)
    assert torch.equal(drop_and_renormalize(a, drop=[0, -1]),
                       drop_and_renormalize(a, keep=range(1, 8)))


def test_zero_rows_stay_zero():
    a = torch.zeros(4, 4)
    a[:, 0] = 1.0
    out = drop_and_renormalize(a)
    assert torch.equal(out, torch.zeros(3, 3))


def test_renormalized_within_embeds_the_block():
    a = _causal(2, 7, 1)
    full = renormalized_within(a, [1, 2, 4, 6])
    idx = torch.tensor([1, 2, 4, 6])
    assert torch.equal(full.index_select(-2, idx).index_select(-1, idx),
                       drop_and_renormalize(a, keep=[1, 2, 4, 6]))
    assert full.sum() == pytest.approx(2 * 4, rel=1e-6)


@pytest.mark.parametrize("cfg_kw", [{}, {"normalization": "none"},
                                    {"symmetrization": "row_norm",
                                     "head_aggregation": "attention_weighted"}])
def test_compute_spectra_equals_a_fresh_graph_of_the_renormalised_block(cfg_kw):
    S, H, d = 12, 3, 8
    attn = {lay: _causal(H, S, 10 + lay) for lay in (0, 1)}
    g = torch.Generator().manual_seed(5)
    sig = {lay: torch.randn(S, d, generator=g) for lay in (0, 1)}
    toks = [f"t{i}" for i in range(S)]
    got = st.compute_spectra(attn, sig, tokens=[toks], token_span=(1, None),
                             config=_cfg(subgraph_renormalize=True, **cfg_kw))
    ref = st.compute_spectra({k: drop_and_renormalize(v) for k, v in attn.items()},
                             {k: v[1:] for k, v in sig.items()}, tokens=[toks[1:]],
                             config=_cfg(**cfg_kw))
    plain = st.compute_spectra(attn, sig, tokens=[toks], token_span=(1, None),
                               config=_cfg(**cfg_kw))
    keys = [k for k in ref.keys() if "pattern" not in k]
    assert keys
    for key in keys:
        np.testing.assert_allclose(np.asarray(got[key]), np.asarray(ref[key]),
                                   rtol=1e-4, atol=1e-5, err_msg=key)
    assert not np.allclose(got["blocks.1.graph.fiedler_value"],
                           plain["blocks.1.graph.fiedler_value"])


def test_full_span_ignores_the_option():
    attn = {0: _causal(2, 6, 3)}
    toks = [[f"t{i}" for i in range(6)]]
    a = st.compute_spectra(attn, tokens=toks, config=_cfg(subgraph_renormalize=True))
    b = st.compute_spectra(attn, tokens=toks, config=_cfg())
    for key in b.keys():
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]), equal_nan=True)


def test_analyze_text_renormalises_the_subgraph(tmp_path):
    S, H, d = 10, 2, 6
    attns = [_causal(H, S, 20 + k).unsqueeze(0) for k in range(2)]
    g = torch.Generator().manual_seed(2)
    hidden = [torch.randn(1, S, d, generator=g) for _ in range(3)]
    keep = list(range(1, S))
    cfg = _cfg(output_dir=str(tmp_path), subgraph_renormalize=True, normalization="none")
    fw = st.GSPDiagnosticsFramework.__new__(st.GSPDiagnosticsFramework)
    fw.config, fw.graph_constructor, fw.spectral_analyzer = \
        cfg, st.GraphConstructor(cfg), st.SpectralAnalyzer(cfg)
    fw.instrumenter = SimpleNamespace(model=None, process_text=lambda t: {
        "attentions": attns, "hidden_states": hidden, "inputs": {}, "tokens": ["t"] * S})
    diags = fw.analyze_text("x", save_results=False, subgraph_indices=keep)["layer_diagnostics"]
    gc, an = fw.graph_constructor, fw.spectral_analyzer
    for k in range(2):
        block = drop_and_renormalize(attns[k][0], keep=keep)
        adj = gc.aggregate_heads(gc.symmetrize_attention(block.unsqueeze(0)))
        want = an.analyze_layer(hidden[k + 1][0, 1:], gc.construct_laplacian(adj)[0], k)
        assert diags[k].fiedler_value == pytest.approx(want.fiedler_value, rel=1e-5)
        assert diags[k].hfer == pytest.approx(want.hfer, rel=1e-5)
        assert diags[k].energy == pytest.approx(want.energy, rel=1e-5)
