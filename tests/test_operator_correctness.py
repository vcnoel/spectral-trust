"""Regression tests for the eigensolver/operator contract (added in 0.2.2).

The bug fixed in 0.2.2: `compute_eigendecomposition` always used a *symmetric*
solver (eigh/eigsh), which reads only the lower triangle. With the default
normalization="rw" the Laplacian I - D^{-1}W is NOT symmetric, so the library
silently diagonalized a symmetrized surrogate.
"""
import numpy as np
import pytest
import torch

from spectral_trust.config import GSPConfig
from spectral_trust.graph import GraphConstructor
from spectral_trust.spectral import SpectralAnalyzer


def _ring_adjacency(n=6):
    """A weighted ring with unequal weights -> D^{-1}W is genuinely non-symmetric."""
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        W[i, (i + 1) % n] = 1.0 + 0.5 * i          # asymmetric degrees
        W[(i + 1) % n, i] = 1.0 + 0.5 * i          # W itself stays symmetric
    return torch.tensor(W)


def _laplacian(norm, W):
    cfg = GSPConfig(device="cpu", normalization=norm)
    return GraphConstructor(cfg).construct_laplacian(W.unsqueeze(0)).squeeze(0).numpy(), cfg


def test_rw_laplacian_is_actually_diagonalized():
    """The returned (vals, vecs) must satisfy L U = U diag(vals) for the rw operator.

    Before 0.2.2 this residual was O(1e-2); a symmetric solver was diagonalizing a
    different matrix.
    """
    L, cfg = _laplacian("rw", _ring_adjacency())
    assert not np.allclose(L, L.T, atol=1e-6), "rw Laplacian should be non-symmetric here"

    vals, vecs = SpectralAnalyzer(cfg).compute_eigendecomposition(L)
    residual = np.linalg.norm(L @ vecs - vecs @ np.diag(vals)) / np.linalg.norm(L)
    assert residual < 1e-6, f"rw operator not diagonalized (residual={residual:.2e})"


def test_rw_and_sym_share_eigenvalues():
    """L_rw = D^{-1/2} L_sym D^{1/2} are similar, so the spectra must agree."""
    W = _ring_adjacency()
    L_rw, cfg_rw = _laplacian("rw", W)
    L_sym, cfg_sym = _laplacian("sym", W)

    v_rw, _ = SpectralAnalyzer(cfg_rw).compute_eigendecomposition(L_rw)
    v_sym, _ = SpectralAnalyzer(cfg_sym).compute_eigendecomposition(L_sym)
    np.testing.assert_allclose(np.sort(v_rw), np.sort(v_sym), atol=1e-6)


def test_symmetric_path_unchanged_and_orthonormal():
    """Symmetric operators keep the eigh path: orthonormal U, exact diagonalization."""
    L, cfg = _laplacian("sym", _ring_adjacency())
    vals, vecs = SpectralAnalyzer(cfg).compute_eigendecomposition(L)

    np.testing.assert_allclose(vecs.T @ vecs, np.eye(vecs.shape[1]), atol=1e-6)
    residual = np.linalg.norm(L @ vecs - vecs @ np.diag(vals)) / np.linalg.norm(L)
    assert residual < 1e-6
    assert vals.min() >= -1e-9 and vals.max() <= 2.0 + 1e-6      # sym spectrum in [0, 2]


def test_star_graph_spectrum():
    """Golden value: the symmetric normalized Laplacian of a star K_{1,n-1} has
    eigenvalues {0, 1 (x n-2), 2} (Lemma 1 of the attention-graph brief)."""
    n = 6
    W = np.zeros((n, n), dtype=np.float32)
    W[0, 1:] = 1.0
    W[1:, 0] = 1.0
    L, cfg = _laplacian("sym", torch.tensor(W))
    vals, _ = SpectralAnalyzer(cfg).compute_eigendecomposition(L)

    np.testing.assert_allclose(vals[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(vals[-1], 2.0, atol=1e-6)
    np.testing.assert_allclose(vals[1:-1], np.ones(n - 2), atol=1e-6)


def test_is_symmetric_helper():
    assert SpectralAnalyzer._is_symmetric(np.eye(3))
    assert not SpectralAnalyzer._is_symmetric(np.array([[0.0, 1.0], [0.0, 0.0]]))
