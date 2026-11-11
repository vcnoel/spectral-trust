"""Edge cases, JSON round trips, and agreement between the fast and dense paths."""
import json

import numpy as np
import pytest
import torch

from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer


def _laplacian(attn, norm):
    """[H, S, S] attention -> [H, S, S] Laplacian, as the library builds it."""
    gc = GraphConstructor(GSPConfig(device="cpu", normalization=norm))
    return gc.construct_laplacian(gc.symmetrize_attention(attn.unsqueeze(0))[0])


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
