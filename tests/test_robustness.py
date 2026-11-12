"""Edge cases, JSON round trips, and agreement between the fast and dense paths."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from spectral_trust import (
    GSPConfig, GraphConstructor, SpectralAnalyzer, DirectedTopologist,
    METRIC_FAMILIES, calculate_spectral_velocity,
    per_head_metrics, per_head_metrics_all_layers,
    per_head_hybrid_metrics, per_head_hybrid_metrics_all_layers,
    stack_feature_vector, layer_delta_features,
    top_band_hfer, row_dirichlet_energy, band_energies, band_entropy,
)

NORMS = ("sym", "rw", "none")


def _laplacian(attn, norm):
    """[H, S, S] attention -> [H, S, S] Laplacian, as the library builds it."""
    gc = GraphConstructor(GSPConfig(device="cpu", normalization=norm))
    return gc.construct_laplacian(gc.symmetrize_attention(attn.unsqueeze(0))[0])


def _adjacency(attn):
    gc = GraphConstructor(GSPConfig(device="cpu"))
    return gc.symmetrize_attention(attn.unsqueeze(0))[0]


def _graph(kind, heads=3, seq=48, seed=0):
    """random: dense softmax attention. sink: one token takes most of the
    mass. causal: lower-triangular mask with a locality bias."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(heads, seq, seq, generator=g)
    if kind == "sink":
        logits[..., 0] += 5.0
    elif kind == "causal":
        pos = torch.arange(seq, dtype=torch.float32)
        logits = logits - 0.1 * (pos[:, None] - pos[None, :]).abs()
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _signals(seq, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed + 100)
    X = torch.randn(seq, dim, generator=g)
    X[:, 0] += 3.0
    return X


def _eigen_metrics(vals):
    """The five per-head attention metrics from a sorted spectrum."""
    n = vals.size
    p = np.maximum(vals / vals.sum(), 1e-12)
    return [vals[1], vals[1] / vals[-1], -np.sum(p * np.log(p)) / np.log(n),
            vals[n // 2:].sum() / vals.sum(), vals[-1]]


def _energy_from_rows(W, X, norm):
    """tr(X^T L X) from row-anchored energies (identities in the docstring)."""
    W, X = W.double(), X.double()
    deg = W.sum(-1)
    iso = deg <= 1e-8
    if norm == "none":
        return 0.5 * float(row_dirichlet_energy(W, X))
    if norm == "sym":
        s = torch.where(iso, torch.zeros_like(deg), deg.clamp(min=1e-8).rsqrt())
        return 0.5 * float(row_dirichlet_energy(W, s[:, None] * X)) + float((X[iso] ** 2).sum())
    inv = torch.where(iso, torch.zeros_like(deg), 1.0 / deg.clamp(min=1e-8))
    P = inv[:, None] * W
    w = torch.where(iso, torch.ones_like(deg), 0.5 * (1.0 - P.sum(0)))
    return 0.5 * float(row_dirichlet_energy(P, X)) + float((w * (X ** 2).sum(-1)).sum())


# ── Agreement between the fast and dense paths ───────────────────────────────

@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("kind", ["random", "sink", "causal"])
def test_fast_and_dense_paths_agree(kind, norm):
    attn = _graph(kind)
    X = _signals(attn.shape[-1])
    cfg = GSPConfig(device="cpu", normalization=norm)
    dense = SpectralAnalyzer(replace(cfg, eigen_solver="dense"))
    laps = _laplacian(attn, norm)
    adj = _adjacency(attn)
    attention_only = per_head_metrics(attn, config=cfg)
    hybrid = per_head_hybrid_metrics(attn, X, config=cfg)
    for h in range(attn.shape[0]):
        d = dense.analyze_layer(X, laps[h], 0)
        vals = np.sort(np.asarray(d.eigenvalues, dtype=np.float64))
        # per-head attention metrics: the spectrum of the operator itself
        np.testing.assert_allclose(attention_only.values[h], _eigen_metrics(vals),
                                   rtol=1e-4, atol=1e-5)
        # per-head hybrid metrics: the layer analyzer on that head's graph
        np.testing.assert_allclose(
            hybrid.values[h], [d.energy, d.smoothness_index, d.spectral_entropy, d.hfer],
            rtol=2e-4, atol=1e-5)
        # row-anchored energies over all rows reproduce tr(X^T L X)
        assert _energy_from_rows(adj[h], X, norm) == pytest.approx(d.energy, rel=1e-5)
        if norm == "rw":
            with pytest.raises(ValueError):
                top_band_hfer(laps[h], X)
            with pytest.raises(ValueError):
                band_energies(laps[h], X, edges=4)
            continue
        assert top_band_hfer(laps[h], X) == pytest.approx(d.hfer, abs=1e-6)
        lam_max = 2.0 if norm == "sym" else float(vals[-1]) * 1.01
        edges = np.linspace(0.0, lam_max, 5)
        exact, _ = band_energies(laps[h], X, edges=edges, method="dense")
        L64 = laps[h].double().numpy()
        v64, U64 = np.linalg.eigh(0.5 * (L64 + L64.T))
        m64 = ((U64.T @ X.double().numpy()) ** 2).sum(1)
        b = np.clip(np.searchsorted(edges, np.maximum(v64, 0), side="right") - 1, 0, 3)
        np.testing.assert_allclose(exact, np.bincount(b, weights=m64, minlength=4),
                                   rtol=1e-4, atol=1e-4 * m64.sum())
        approx, _ = band_energies(laps[h], X, edges=edges, order=128, lambda_max=lam_max)
        assert approx.sum() == pytest.approx(exact.sum(), rel=1e-5)
        assert np.abs(approx / approx.sum() - exact / exact.sum()).max() < 0.05
    # a projection width >= d is the identity
    wide = SpectralAnalyzer(replace(cfg, eigen_solver="dense", project_dim=X.shape[1]))
    assert wide.analyze_layer(X, laps[0], 0).to_dict() == \
        dense.analyze_layer(X, laps[0], 0).to_dict()


# ── JSON round trips ─────────────────────────────────────────────────────────

def _round_trip(obj):
    back = json.loads(json.dumps(obj))
    assert back == obj
    return back


@pytest.mark.parametrize("cfg", [
    dict(normalization="sym"), dict(normalization="sym", eigen_solver="sparse"),
    dict(normalization="rw"), dict(normalization="none"),
    dict(normalization="sym", project_dim=8),
], ids=["sym", "sym-sparse", "rw", "none", "projected"])
def test_spectral_diagnostics_json_round_trip(cfg):
    attn = _graph("causal", heads=1, seq=64)
    X = _signals(64)
    c = GSPConfig(device="cpu", **cfg)
    d = SpectralAnalyzer(c).analyze_layer(X, _laplacian(attn, c.normalization)[0], 3)
    # directed metrics are attached the way the framework attaches them
    topo = DirectedTopologist(device="cpu")
    directed = topo.get_directed_metrics(topo.compute_directed_laplacian(_adjacency(attn)[0]))
    d.max_imaginary = directed["max_imaginary"]
    d.spectral_radius = directed["spectral_radius"]
    back = _round_trip(d.to_dict())
    assert back["families"] == METRIC_FAMILIES and back["layer"] == 3
    np.testing.assert_array_equal(np.asarray(back["eigenvalues"]), d.eigenvalues)
    np.testing.assert_array_equal(np.asarray(back["spectral_masses"]), d.spectral_masses)
    for name in ("energy", "smoothness_index", "spectral_entropy", "hfer",
                 "fiedler_value", "spectral_radius", "max_imaginary"):
        assert back[name] == getattr(d, name)


def test_per_head_diagnostics_json_round_trip():
    attns = [_graph("sink", heads=4, seq=20, seed=s) for s in range(3)]
    hidden = [_signals(20, seed=s) for s in range(4)]
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, [0, 2]] = True
    objects = (per_head_metrics_all_layers(attns)
               + per_head_metrics_all_layers(attns, heads=mask)
               + per_head_hybrid_metrics_all_layers(attns, hidden)
               + per_head_hybrid_metrics_all_layers(attns, hidden, heads=[(0, 3)]))
    for d in objects:
        back = _round_trip(d.to_dict())
        np.testing.assert_array_equal(
            np.asarray(back["values"], dtype=np.float32).reshape(d.values.shape), d.values)
        assert back["metric_names"] == d.metric_names and back["family"] == d.family
        assert back.get("heads") == d.heads


def test_framework_results_json_round_trip():
    """The structure GSPDiagnosticsFramework._save_results writes."""
    attns = [_graph("causal", heads=2, seq=30, seed=s) for s in range(4)]
    X = _signals(30)
    an = SpectralAnalyzer(GSPConfig(device="cpu"))
    diags = [an.analyze_layer(X, _laplacian(a.mean(0, keepdim=True), "sym")[0], i)
             for i, a in enumerate(attns)]
    vel, max_vel, max_idx = calculate_spectral_velocity(
        torch.tensor([d.fiedler_value for d in diags]))
    _round_trip({"text": "t", "tokens": ["a"] * 30,
                 "diagnostics": [d.to_dict() for d in diags],
                 "velocity_metrics": {"fiedler_velocity": vel.tolist(),
                                      "max_velocity_value": max_vel,
                                      "max_velocity_layer_index": max_idx}})


# ── Edge cases ───────────────────────────────────────────────────────────────

def _disconnected(heads=2, seed=6):
    """Two blocks with random weights and one isolated token."""
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(heads, 9, 9)
    a[:, :5, :5] = torch.rand(heads, 5, 5, generator=g)
    a[:, 5:8, 5:8] = torch.rand(heads, 3, 3, generator=g)
    return a


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("seq", [1, 2])
def test_tiny_graphs(seq, norm):
    attn = torch.softmax(torch.randn(2, seq, seq, generator=torch.Generator().manual_seed(7)), -1)
    X = _signals(seq, dim=4)
    cfg = GSPConfig(device="cpu", normalization=norm)
    d = SpectralAnalyzer(cfg).analyze_layer(X, _laplacian(attn[:1], norm)[0], 0)
    assert d.eigenvalues.shape == (seq,) and d.connectivity
    assert all(np.isfinite(v) for v in (d.energy, d.smoothness_index,
                                        d.spectral_entropy, d.hfer, d.fiedler_value))
    # the attention-only per-head metrics zero out below three tokens
    assert np.array_equal(per_head_metrics(attn, config=cfg).values, np.zeros((2, 5)))
    hyb = per_head_hybrid_metrics(attn, X, config=cfg)
    for h in range(2):
        ref = SpectralAnalyzer(cfg).analyze_layer(X, _laplacian(attn[h:h + 1], norm)[0], 0)
        np.testing.assert_allclose(
            hyb.values[h], [ref.energy, ref.smoothness_index, ref.spectral_entropy, ref.hfer],
            rtol=1e-4, atol=1e-6)
    if norm != "rw":
        L = _laplacian(attn[:1], norm)[0]
        assert top_band_hfer(L, X) == pytest.approx(d.hfer, abs=1e-6)
        E, _ = band_energies(L, X, edges=4, order=32)
        assert E.sum() == pytest.approx(float((X ** 2).sum()), rel=1e-5)
    assert row_dirichlet_energy(attn, X).shape == (2,)


@pytest.mark.parametrize("norm", NORMS)
def test_disconnected_graph(norm):
    attn = _disconnected()
    X = _signals(9, dim=4)
    cfg = GSPConfig(device="cpu", normalization=norm)
    L = _laplacian(attn, norm)
    d = SpectralAnalyzer(cfg).analyze_layer(X, L[0], 0)
    assert not d.connectivity
    vals = np.asarray(d.eigenvalues)
    # two components; the isolated token adds a zero mode only for L = D - W
    # (float32 puts it at ~1e-6 on this scale, hence the relative threshold)
    assert int(np.sum(vals < 1e-5 * vals.max())) == 2 + (norm == "none")
    assert d.fiedler_value == pytest.approx(0.0, abs=1e-6)
    ph = per_head_metrics(attn, config=cfg)
    np.testing.assert_allclose(ph.metric("fiedler_value"), 0.0, atol=1e-5)
    assert np.isfinite(ph.values).all()
    hyb = per_head_hybrid_metrics(attn, X, config=cfg)
    np.testing.assert_allclose(
        hyb.values[0], [d.energy, d.smoothness_index, d.spectral_entropy, d.hfer],
        rtol=2e-4, atol=1e-5)
    if norm != "rw":
        assert top_band_hfer(L[0], X) == pytest.approx(d.hfer, abs=1e-6)
    assert _energy_from_rows(_adjacency(attn)[0], X, norm) == pytest.approx(d.energy, rel=1e-5)


@pytest.mark.parametrize("norm", NORMS)
def test_all_zero_signals(norm):
    attn = _graph("random", heads=2, seq=12)
    X = torch.zeros(12, 5)
    cfg = GSPConfig(device="cpu", normalization=norm)
    L = _laplacian(attn, norm)
    d = SpectralAnalyzer(cfg).analyze_layer(X, L[0], 0)
    assert (d.energy, d.smoothness_index, d.spectral_entropy, d.hfer) == (0.0, 0.0, 0.0, 0.0)
    assert np.array_equal(per_head_hybrid_metrics(attn, X, config=cfg).values, np.zeros((2, 4)))
    assert float(row_dirichlet_energy(attn, X).abs().max()) == 0.0
    if norm != "rw":
        assert top_band_hfer(L[0], X) == 0.0
        E, _ = band_energies(L[0], X, edges=4, order=16)
        assert np.array_equal(E, np.zeros(4)) and band_entropy(E) == 0.0


@pytest.mark.parametrize("norm", NORMS)
def test_float16_inputs_match_their_float32_upcast(norm):
    attn16 = _graph("causal", heads=2, seq=24).half()
    X16 = _signals(24).half()
    cfg = GSPConfig(device="cpu", normalization=norm)
    L16 = _laplacian(attn16[:1].float(), norm)[0].half()
    an = SpectralAnalyzer(cfg)
    assert an.analyze_layer(X16, L16, 0).to_dict() == \
        an.analyze_layer(X16.float(), L16.float(), 0).to_dict()
    np.testing.assert_array_equal(per_head_metrics(attn16, config=cfg).values,
                                  per_head_metrics(attn16.float(), config=cfg).values)
    np.testing.assert_array_equal(per_head_hybrid_metrics(attn16, X16, config=cfg).values,
                                  per_head_hybrid_metrics(attn16.float(), X16.float(),
                                                          config=cfg).values)
    torch.testing.assert_close(row_dirichlet_energy(attn16, X16),
                               row_dirichlet_energy(attn16.float(), X16.float()))
    if norm != "rw":
        assert top_band_hfer(L16, X16) == pytest.approx(
            top_band_hfer(L16.float(), X16.float()), abs=1e-9)
        np.testing.assert_allclose(band_energies(L16, X16, 4, order=16, lambda_max=2.0)[0],
                                   band_energies(L16.float(), X16.float(), 4, order=16,
                                                 lambda_max=2.0)[0], rtol=1e-6)


# ── Head selections: empty and full ──────────────────────────────────────────

def test_empty_head_selection():
    attns = [_graph("random", heads=4, seq=16, seed=s) for s in range(3)]
    hidden = [_signals(16, seed=s) for s in range(4)]
    for heads in ([], np.zeros(4, dtype=bool), np.zeros((3, 4), dtype=bool)):
        att = per_head_metrics_all_layers(attns, heads=heads)
        hyb = per_head_hybrid_metrics_all_layers(attns, hidden, heads=heads)
        for a, b in zip(att, hyb):
            assert a.values.shape == (0, 5) and b.values.shape == (0, 4)
            assert a.heads == [] and b.heads == []
            assert json.loads(json.dumps(a.to_dict()))["heads"] == []
        assert stack_feature_vector(att).shape == (0,)
        assert layer_delta_features(att, metric="fiedler_value").shape == (0,)


def test_full_head_selection_equals_no_selection():
    attns = [_graph("sink", heads=4, seq=16, seed=s) for s in range(3)]
    hidden = [_signals(16, seed=s) for s in range(4)]
    full_att = per_head_metrics_all_layers(attns)
    full_hyb = per_head_hybrid_metrics_all_layers(attns, hidden)
    every = [(l, h) for l in range(3) for h in range(4)]
    for heads in (list(range(4)), np.ones(4, dtype=bool), np.ones((3, 4), dtype=bool), every):
        att = per_head_metrics_all_layers(attns, heads=heads)
        hyb = per_head_hybrid_metrics_all_layers(attns, hidden, heads=heads)
        for f, s in zip(full_att + full_hyb, att + hyb):
            assert s.heads == [0, 1, 2, 3] and f.heads is None
            np.testing.assert_array_equal(s.values, f.values)
            assert s.to_dict()["heads"] == [0, 1, 2, 3]
        np.testing.assert_array_equal(stack_feature_vector(att), stack_feature_vector(full_att))
        np.testing.assert_array_equal(layer_delta_features(att), layer_delta_features(full_att))


@pytest.mark.parametrize("norm", ["sym", "rw", "none"])
@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_analyze_layer_single_token(norm, solver):
    """Regression (0.4.0): a one-token graph crashed analyze_layer, whose
    squeeze() turned the 1 x 1 Laplacian into a 0-d array."""
    attn = torch.ones(1, 1, 1)
    L = _laplacian(attn, norm)[0]
    X = torch.tensor([[1.0, -2.0, 0.5]])
    cfg = GSPConfig(device="cpu", normalization=norm, eigen_solver=solver)
    d = SpectralAnalyzer(cfg).analyze_layer(X, L, 0)
    assert d.eigenvalues.shape == (1,) and d.connectivity
    assert d.fiedler_value == 0.0 and d.energy == pytest.approx(0.0, abs=1e-12)
    assert d.hfer == pytest.approx(1.0)          # the single mode is the top mode
    # a leading batch dimension is still accepted
    d_batched = SpectralAnalyzer(cfg).analyze_layer(X, L.unsqueeze(0), 0)
    assert d_batched.to_dict() == d.to_dict()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_jl_project_half_precision_computes_in_float32(dtype):
    """Regression (0.4.0): jl_project multiplied half-precision signals by a
    half-precision Gaussian matrix, losing about three digits."""
    from spectral_trust import jl_project, gaussian_projection
    X = torch.randn(16, 64, generator=torch.Generator().manual_seed(5)).to(dtype)
    got = jl_project(X, 8, seed=1)
    want = X.float() @ torch.from_numpy(gaussian_projection(64, 8, seed=1))
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    Xn = X.float().numpy().astype(np.float16)
    assert jl_project(Xn, 8, seed=1).dtype == np.float32


def test_band_energies_refuses_rw_and_handles_a_zero_spectrum():
    """Regression (0.4.0): band_energies symmetrized a non-symmetric (rw)
    operator in its dense path and ran the Chebyshev recurrence on it (both
    rely on an orthonormal eigenbasis), and the default Gershgorin range was
    0 for an all-zero Laplacian (one token), a division by zero."""
    from spectral_trust import band_energies
    g = torch.Generator().manual_seed(4)
    attn = torch.softmax(torch.randn(1, 12, 12, generator=g), -1)
    attn[0, :, 0] += 2.0                           # uneven degrees
    L_rw = _laplacian(attn, "rw")[0]
    X = torch.randn(12, 3, generator=g)
    for method in ("dense", "chebyshev"):
        with pytest.raises(ValueError):
            band_energies(L_rw, X, edges=4, order=8, method=method)
    L0 = _laplacian(torch.ones(1, 1, 1), "sym")[0]        # [[0.]]
    X0 = torch.tensor([[3.0, 4.0]])
    for method, tol in (("dense", 1e-6), ("chebyshev", 2e-2)):   # order-8 filter
        E, edges = band_energies(L0, X0, edges=4, order=8, method=method)
        assert E.sum() == pytest.approx(25.0, rel=1e-6)
        assert E[0] == pytest.approx(25.0, rel=tol) and edges[-1] > 0
