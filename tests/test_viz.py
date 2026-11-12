"""spectral_trust.viz: self-contained SVG views of a SpectralCache, built here
from synthetic attention so that no model is needed."""
import re

import numpy as np
import pytest
import torch

import spectral_trust as st
from spectral_trust import viz

TOKENS = ["The", " cat", " sat", " on", " the", " mat", "."]


def _attn(layers=3, heads=4, seq=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for lay in range(layers):
        a = torch.rand(1, heads, seq, seq, generator=g).tril() + 1e-3 * torch.eye(seq)
        out[lay] = a / a.sum(-1, keepdim=True)
    return out


def _signals(layers=3, seq=7, dim=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {lay: torch.randn(1, seq, dim, generator=g) for lay in range(layers)}


@pytest.fixture(scope="module")
def cache():
    return st.compute_spectra(_attn(), _signals(), tokens=[TOKENS])


def _no_external_resources(page: str):
    urls = re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)""", page)
    assert not [u for u in urls if not u.startswith("#")], urls
    assert "http://" not in page.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in page


@pytest.mark.parametrize("make", [
    lambda c: viz.graph(c, 1),
    lambda c: viz.graph(c, 1, head=2),
    lambda c: viz.heatmap(c, "fiedler_value"),
    lambda c: viz.heatmap(c, "resid.hfer"),
    lambda c: viz.spectrum(c, 1),
    lambda c: viz.spectrum(c, 1, head=0),
])
def test_views_are_self_contained_and_save(cache, make, tmp_path):
    view = make(cache)
    frag = view._repr_html_()
    assert frag.startswith("<div") and "<svg" in frag and "<script>" in frag
    page = view.to_html()
    assert page.startswith("<!doctype html>") and '<meta charset="utf-8">' in page
    _no_external_resources(page)
    path = view.save(tmp_path / "view")
    assert path.suffix == ".html" and path.read_text(encoding="utf-8") == page
    assert view.svg.startswith("<svg") and view.svg.endswith("</svg>")


def test_graph_tooltips_name_tokens_and_values(cache):
    svg = viz.graph(cache, 2, head=1).svg
    assert svg.count("<circle") == len(TOKENS)
    assert "&#x27; cat&#x27; (token 1)" in svg and "Fiedler vector" in svg
    assert svg.count('data-node="') == len(TOKENS)
    assert "<line" in svg and "weight" in svg


def test_graph_colours_follow_the_cached_fiedler_vector(cache):
    svg = viz.graph(cache, 0).svg
    u = cache["blocks.0.graph.fiedler_vector"]
    k = int(np.argmax(u))
    m = re.findall(r'<circle[^>]*fill="(#[0-9A-F]{6})"[^>]*data-node="(\d+)"', svg)
    fills = {int(i): c for c, i in m}
    assert fills[k] == viz.ORANGE           # the largest entry is fully orange
    assert fills[int(np.argmin(u))] != viz.ORANGE


def test_heatmap_cells_and_blank_heads():
    c = st.compute_spectra(_attn(), _signals(), tokens=[TOKENS], heads=[(0, 1), (2, 3)])
    svg = viz.heatmap(c, "fiedler_value").svg
    assert svg.count("<rect") >= 3 * 4
    assert svg.count("not computed") == 3 * 4 - 2
    assert "layer 2, head 3" in svg


def test_heatmap_of_a_graph_metric_spans_texts():
    attn = {lay: torch.cat([a, a.flip(0)]) for lay, a in _attn().items()}
    c = st.compute_spectra(attn, None, tokens=[TOKENS, TOKENS], hybrid=False)
    svg = viz.heatmap(c, "graph.fiedler_value").svg
    assert "layer 1, text 1" in svg


def test_matplotlib_fallback(cache):
    import matplotlib
    matplotlib.use("Agg")
    fig = viz.heatmap(cache, "spectral_radius", backend="matplotlib")
    assert fig.axes and fig.axes[0].images
    import matplotlib.pyplot as plt
    plt.close(fig)
    with pytest.raises(ValueError):
        viz.heatmap(cache, "fiedler_value", backend="plotly")


def test_spectrum_marks_lambda_2(cache):
    svg = viz.spectrum(cache, 1).svg
    lam = cache["blocks.1.graph.eigenvalues"]
    assert f"λ = {lam[1]:.5f} (Fiedler value)" in svg
    assert "HFER modes" in svg
    assert svg.count("<circle") == len(TOKENS)


def test_cache_methods_call_the_views(cache):
    assert isinstance(cache.heatmap("fiedler_value"), viz.View)
    assert isinstance(cache.graph(1, 0), viz.View)
    assert isinstance(cache.spectrum(1), viz.View)
    assert "graph view" in repr(cache.graph(1))


def test_graph_needs_patterns():
    c = st.compute_spectra(_attn(), None, tokens=[TOKENS], keep_patterns=False)
    with pytest.raises(ValueError, match="keep_patterns=True"):
        viz.graph(c, 0)
    viz.spectrum(c, 0)      # the spectrum view reads the cached eigenvalues only


def test_graph_respects_the_token_span():
    c = st.compute_spectra(_attn(), _signals(), tokens=[TOKENS], token_span=(2, None))
    svg = viz.graph(c, 1).svg
    assert svg.count("<circle") == len(TOKENS) - 2
    assert "(token 2)" in svg and "(token 0)" not in svg
