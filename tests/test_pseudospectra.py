"""spectral_trust.pseudospectra on matrices with known answers: normal
matrices, Jordan blocks, causal stochastic matrices (Li 2026, Propositions 1
and 2), the triangular path against dense SVD, and its speed."""
import time

import numpy as np
import pytest
import torch

from spectral_trust import pseudospectra as ps


def _normal(n, seed, radius=0.8):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    lam = radius * rng.random(n) * np.exp(2j * np.pi * rng.random(n))
    return Q @ np.diag(lam) @ Q.conj().T, lam


def _causal(n, seed, sink=2.0):
    rng = np.random.default_rng(seed)
    a = np.tril(rng.random((n, n)))
    a[:, 0] += sink * a.sum(1)
    return a / a.sum(1, keepdims=True)


def test_normal_pseudospectra_are_eps_balls():
    N, lam = _normal(8, 0)
    res = ps.pseudospectrum(N, (np.linspace(-1, 1, 15), np.linspace(-1, 1, 13)), eps=0.1)
    dist = np.abs(res.z[..., None] - lam).min(-1)
    assert res.z.shape == (13, 15)
    np.testing.assert_allclose(res.sigma_min, dist, rtol=1e-8)
    assert np.array_equal(res.inside(0.1), dist < 0.1)
    for e in (1e-3, 0.1, 0.5):
        assert ps.pseudospectral_radius(N, e) == pytest.approx(np.abs(lam).max() + e, rel=1e-9)
        assert ps.pseudospectral_abscissa(N, e) == pytest.approx(lam.real.max() + e, rel=1e-9)
    assert ps.departure_from_normality(N) == pytest.approx(0.0, abs=1e-12)
    assert ps.numerical_range(N).radius == pytest.approx(np.abs(lam).max(), rel=1e-6)
    assert ps.kreiss_constant(N, how="resolvent") == pytest.approx(1.0, abs=1e-6)
    assert ps.kreiss_constant(N) == pytest.approx(1.0, abs=1e-3)


def test_jordan_block_follows_the_eps_to_the_one_over_n_law():
    n, eps = 10, 1e-6
    J = np.diag(np.ones(n - 1), -1)
    r = ps.pseudospectral_radius(J, eps)
    # ‖(zI - J)^-1‖ = ‖Σ_k J^k / z^(k+1)‖ lies between |z|^-n and Σ_k |z|^-(k+1)
    upper = next(x for x in np.linspace(eps ** (1 / n), 1, 100000)
                 if np.sum(x ** -np.arange(1, n + 1)) <= 1 / eps)
    assert eps ** (1 / n) <= r <= upper
    assert r == pytest.approx(0.253, abs=0.002)              # Li 2026, appendix A
    z = 0.4 * np.exp(1j * np.linspace(0, 2 * np.pi, 7))
    s = ps.sigma_min(J, z)
    np.testing.assert_allclose(s, s[0], rtol=1e-8)            # rotationally symmetric
    assert ps.departure_from_normality(J) == pytest.approx(np.sqrt(n - 1))
    assert ps.numerical_range(J).radius == pytest.approx(np.cos(np.pi / (n + 1)), rel=1e-6)


def test_kreiss_brackets_the_transient_of_a_jordan_block():
    n = 8
    J = 0.9 * np.eye(n) + np.diag(np.ones(n - 1), -1)
    peak = max(np.linalg.norm(np.linalg.matrix_power(J, k), 2) for k in range(400))
    for how in ("resolvent", "radii"):
        K = ps.kreiss_constant(J, how=how)
        assert K <= peak * (1 + 1e-9) and peak <= np.e * n * K
    assert ps.kreiss_constant(1.1 * np.eye(3)) == float("inf")


def test_causal_stochastic_kreiss_is_sqrt_n_and_deflation_is_exact():
    A = _causal(16, 1)
    assert ps.kreiss_constant(A, how="resolvent") == pytest.approx(4.0, rel=1e-3)
    assert ps.kreiss_constant(A) == pytest.approx(4.0, rel=1e-3)
    layers = np.stack([_causal(12, s) for s in range(4)])
    P = np.zeros((12, 12))
    P[:, 0] = 1.0
    full, dev = ps.depth_propagator(layers), ps.depth_propagator(layers, deflate=True)
    np.testing.assert_allclose(dev, full - P, atol=1e-12)     # Li 2026, Proposition 2
    np.testing.assert_allclose(full[2], layers[2] @ layers[1] @ layers[0])


def test_triangular_path_matches_dense_svd():
    rng = np.random.default_rng(2)
    z = 1.2 * (rng.normal(size=60) + 1j * rng.normal(size=60))
    for A in (_causal(64, 3), _causal(64, 3).T, rng.normal(size=(40, 40))):
        fast, dense = ps.sigma_min(A, z), ps.sigma_min(A, z, method="svd")
        ok = dense > 1e-12 * np.linalg.norm(A, 2)             # float64-resolvable points
        assert ok.sum() > 30
        np.testing.assert_allclose(fast[ok], dense[ok], rtol=1e-7)
    assert ps.sigma_min(np.eye(3) * 0.5, 0.5) == 0.0
    assert ps.sigma_min(torch.eye(2), 3.0) == pytest.approx(2.0)


def test_nonnormality_summaries_and_depth_option():
    A = _causal(10, 4)
    out = ps.nonnormality(A, eps=1e-2, n_angles=24)
    assert set(out) == set(ps.NONNORMALITY_NAMES)
    assert out["spectral_radius"] == pytest.approx(1.0)
    assert out["pseudospectral_radius"] >= 1.01 - 1e-12
    assert out["numerical_radius"] <= out["norm"] + 1e-12
    layers = np.stack([_causal(10, s) for s in range(3)])
    d = ps.nonnormality(layers, depth=True, deflate=True, kreiss=False)
    assert d["norm"].shape == (3,) and "kreiss_constant" not in d
    P = np.zeros((10, 10))
    P[:, 0] = 1.0
    rho = [np.abs(np.diag(layers[l] - P)).max() for l in range(3)]
    np.testing.assert_allclose(d["eigen_heuristic"], np.cumprod(rho))
    assert np.all(d["norm"] <= d["norm_bound"] * (1 + 1e-12))


def test_triangular_path_is_faster_than_dense_svd():
    A = _causal(256, 5)
    z = 0.5 * np.exp(1j * np.linspace(0, 2 * np.pi, 48))
    ps.sigma_min(A, z[:2])
    t0 = time.perf_counter()
    ps.sigma_min(A, z)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    ps.sigma_min(A, z, method="svd")
    dense = time.perf_counter() - t0
    assert fast < dense
