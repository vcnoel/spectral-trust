"""Regression tests for the last three 0.4.0 fixes: the connectivity tolerance,
the CLI normalisation default and the packaged probe file."""
import argparse
import importlib.resources
import json

import numpy as np
import pytest
import torch

from spectral_trust import GSPConfig, SpectralAnalyzer


def _heavy_none_laplacian(blocks, size, weight=1000.0, seed=0):
    """Unnormalised L = D - W for `blocks` dense components with large weights."""
    rng = np.random.default_rng(seed)
    n = blocks * size
    W = np.zeros((n, n))
    for b in range(blocks):
        s = slice(b * size, (b + 1) * size)
        block = rng.random((size, size)) * weight
        W[s, s] = (block + block.T) / 2
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(1)) - W
    return torch.tensor(L, dtype=torch.float32)


def test_connectivity_survives_float32_noise_on_unnormalised_graphs():
    L = _heavy_none_laplacian(blocks=1, size=128)
    X = torch.randn(128, 8)
    d = SpectralAnalyzer(GSPConfig(device="cpu", normalization="none")).analyze_layer(X, L, 0)
    vals = np.asarray(d.eigenvalues)
    # the precondition that made the fixed 1e-6 threshold report a disconnected graph
    assert abs(vals[0]) > 1e-6
    assert d.connectivity


def test_two_heavy_components_stay_disconnected():
    L = _heavy_none_laplacian(blocks=2, size=64)
    X = torch.randn(128, 8)
    d = SpectralAnalyzer(GSPConfig(device="cpu", normalization="none")).analyze_layer(X, L, 0)
    assert not d.connectivity


def test_zero_tolerance_keeps_the_old_floor_in_float64():
    vals = np.array([0.0, 0.5, 1.2, 2.0], dtype=np.float64)
    assert SpectralAnalyzer.zero_eigenvalue_tolerance(vals, 4) == pytest.approx(1e-6)
    assert SpectralAnalyzer.is_connected(vals, 4)
    assert not SpectralAnalyzer.is_connected(np.array([0.0, 5e-7, 1.0, 2.0]), 4)


def test_zero_tolerance_scales_with_lambda_max_in_float32():
    vals = np.array([3e-3, 50.0, 6000.0], dtype=np.float32)
    tol = SpectralAnalyzer.zero_eigenvalue_tolerance(vals, 256)
    assert tol == pytest.approx(16 * 16 * np.finfo(np.float32).eps * 6000.0, rel=1e-6)
    assert SpectralAnalyzer.is_connected(vals, 256)


def test_noise_on_the_zero_mode_does_not_disconnect_a_normalised_graph():
    # gpt2 layer 5, float32 sym: the zero mode came out at 1.07e-6, above the old 1e-6 cut
    vals = np.array([1.07e-6, 0.85, 1.1, 1.3], dtype=np.float32)
    assert SpectralAnalyzer.is_connected(vals, 12)


def test_two_numerically_zero_modes_stay_disconnected_in_float32():
    vals = np.array([1e-7, 2e-6, 0.9, 1.3], dtype=np.float32)
    assert not SpectralAnalyzer.is_connected(vals, 12)


def test_single_node_is_connected():
    assert SpectralAnalyzer.is_connected(np.array([0.0]), 1)
    assert not SpectralAnalyzer.is_connected(np.array([]), 0)


def test_gpt2_like_attention_is_connected_end_to_end():
    from spectral_trust import GraphConstructor
    g = torch.Generator().manual_seed(5)
    seq = 12
    a = torch.rand(1, 12, seq, seq, generator=g).tril()
    a = a / a.sum(-1, keepdim=True)
    cfg = GSPConfig(device="cpu", normalization="sym")
    gc = GraphConstructor(cfg)
    sym = gc.symmetrize_attention(a).squeeze(0)
    adj = gc.aggregate_heads(sym.unsqueeze(0), attn_mask=None).squeeze(0)
    lap = gc.construct_laplacian(adj.unsqueeze(0)).squeeze(0)
    d = SpectralAnalyzer(cfg).analyze_layer(torch.randn(seq, 16, generator=g), lap, 0)
    assert d.connectivity


def test_cli_normalization_default_is_sym():
    from spectral_trust.cli import create_model_config
    cfg = create_model_config("example-org/custom-model", argparse.Namespace())
    assert cfg.normalization == "sym"
    assert GSPConfig().normalization == "sym"


def test_cli_parsers_default_to_sym():
    src =importlib.resources.files("spectral_trust.cli").joinpath("__init__.py").read_text(encoding="utf-8")
    assert "default='rw'" not in src
    assert src.count("'--normalization', type=str, default='sym'") == 2


def test_probe_file_is_packaged_and_readable():
    probe = importlib.resources.files("spectral_trust.cli").joinpath("data/probes.json")
    assert probe.is_file()
    json.loads(probe.read_text(encoding="utf-8"))
