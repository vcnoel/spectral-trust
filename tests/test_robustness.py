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
