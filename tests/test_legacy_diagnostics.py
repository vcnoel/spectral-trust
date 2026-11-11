import tracemalloc

import pytest
import torch
import numpy as np
from spectral_trust.config import GSPConfig
from spectral_trust.spectral import SpectralAnalyzer
from spectral_trust.graph import GraphConstructor


def _random_laplacian(seq, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.softmax(torch.randn(1, 1, seq, seq, generator=g), dim=-1)
    gc = GraphConstructor(GSPConfig(device="cpu"))
    return gc.construct_laplacian(gc.symmetrize_attention(a)[0])[0].numpy()


def test_energy_and_smoothness_never_form_a_d_by_d_matrix():
    """Regression (0.4.0): tr(X^T L X) and tr(X^T X) were taken from the full
    d x d products, O(S d^2) time and d^2 memory (61 MB here, 64 GB at
    d = 131072), and the smoothness index computed the energy a second time.
    Both are now trace-only sums."""
    S, d = 8, 4000
    X = np.random.default_rng(0).standard_normal((S, d)).astype(np.float32)
    L = _random_laplacian(S)
    an = SpectralAnalyzer(GSPConfig(device="cpu"))
    X64, L64 = X.astype(np.float64), L.astype(np.float64)
    want_energy = float(np.sum(X64 * (L64 @ X64)))
    want_smooth = want_energy / float(np.sum(X64 * X64))
    for fn, want in ((lambda: an.compute_dirichlet_energy(X, L), want_energy),
                     (lambda: an.compute_smoothness_index(X, L), want_smooth)):
        tracemalloc.start()
        try:
            got = fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 8 * 2 ** 20, f"peak {peak / 2 ** 20:.1f} MB: a d x d matrix was built"
        assert got == pytest.approx(want, rel=1e-6)


def test_analyze_layer_evaluates_the_energy_once_and_projects_once(monkeypatch):
    """Regression (0.4.0): analyze_layer computed the Dirichlet energy twice
    (directly and inside the smoothness index) and projected the signals onto
    the eigenbasis three times (entropy, HFER, spectral masses)."""
    class CountingBasis(np.ndarray):
        transposes = 0

        @property
        def T(self):
            CountingBasis.transposes += 1
            return np.asarray(self).T

    calls = {"energy": 0}
    real_energy = SpectralAnalyzer.compute_dirichlet_energy
    real_eig = SpectralAnalyzer.compute_eigendecomposition

    def counting_energy(self, signals, laplacian):
        calls["energy"] += 1
        return real_energy(self, signals, laplacian)

    def counting_eig(self, laplacian):
        vals, vecs = real_eig(self, laplacian)
        return vals, vecs.view(CountingBasis)

    monkeypatch.setattr(SpectralAnalyzer, "compute_dirichlet_energy", counting_energy)
    monkeypatch.setattr(SpectralAnalyzer, "compute_eigendecomposition", counting_eig)
    S = 30
    X = torch.randn(S, 16, generator=torch.Generator().manual_seed(1))
    an = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense"))
    diag = an.analyze_layer(X, torch.from_numpy(_random_laplacian(S)), 0)
    assert calls["energy"] == 1
    assert CountingBasis.transposes == 1
    # the single projection still yields consistent masses, entropy and HFER
    m = np.asarray(diag.spectral_masses, dtype=np.float64)
    p = np.maximum(m / m.sum(), 1e-12)
    assert diag.spectral_entropy == pytest.approx(float(-(p * np.log(p)).sum()), rel=1e-5)
    assert diag.hfer == pytest.approx(float(m[int(0.9 * S):].sum() / m.sum()), rel=1e-5)

def test_legacy_metrics_accuracy():
    """Verify standard GSP diagnostics on a fixed symmetric graph."""
    # Use Combinatorial Laplacian for predictable eigenvalues
    config = GSPConfig(device="cpu", normalization="none")
    analyzer = SpectralAnalyzer(config)
    
    # Simple 4-node path graph: 0-1-2-3
    laplacian = torch.tensor([
        [ 1.0, -1.0,  0.0,  0.0],
        [-1.0,  2.0, -1.0,  0.0],
        [ 0.0, -1.0,  2.0, -1.0],
        [ 0.0,  0.0, -1.0,  1.0]
    ]).unsqueeze(0) # Analyzer expects [1, N, N] if coming from framework, but analyze_layer expects [N, N]
    
    # analyze_layer expects [N, D] and [N, N]
    signals = torch.ones((4, 8))
    diag = analyzer.analyze_layer(signals, laplacian.squeeze(0), 0)
    
    # Energy should be 0 for constant signal on combinatorial Laplacian
    assert diag.energy < 1e-6
    assert diag.connectivity == True
    # Fiedler value for path graph N=4 is 2 - sqrt(2) approx 0.5858
    assert abs(diag.fiedler_value - 0.5858) < 1e-3

def test_graph_constructor_symmetry():
    """Ensure GraphConstructor produces symmetric Laplacians for 'sym' and 'none' normalization."""
    # Sym normalization: L = I - D^-1/2 A D^-1/2 is symmetric if A is symmetric.
    # Our GraphConstructor symmetrizes A by default if config.symmetrization == "symmetric"
    config = GSPConfig(device="cpu", normalization="none", symmetrization="symmetric")
    constructor = GraphConstructor(config)
    
    # Random asymmetric adjacency
    adj = torch.rand((1, 1, 4, 4)) # [B, H, Q, K]
    
    # Aggregate and symmetrize
    agg = constructor.aggregate_heads(adj)
    sym_adj = constructor.symmetrize_attention(agg.unsqueeze(1)).squeeze(1)
    
    # Construct Laplacian
    laplacian = constructor.construct_laplacian(sym_adj).squeeze(0)
    
    # Check symmetry
    assert torch.allclose(laplacian, laplacian.T, atol=1e-6)

def test_config_defaults():
    """Pin the documented GSPConfig defaults.

    normalization changed "rw" -> "sym" in v0.3.0: the symmetric normalized
    Laplacian has a spectrum in [0, 2] independent of sequence length, so
    metrics are comparable across inputs of different length, and it always
    takes the deterministic symmetric solver path. Callers reproducing
    pre-0.3.0 numbers must pass normalization="rw" explicitly.

    eigen_solver changed "sparse" -> "dense" in v0.4.0 (regression for the
    truncated default): above 50 tokens the sparse default kept 50 modes, so
    HFER, spectral entropy and spectral masses were normalised over a
    truncated basis, and it was also slower than the dense solve.
    """
    config = GSPConfig()
    assert config.directed == False
    assert config.calc_velocity == False
    assert config.subgraph_indices is None
    assert config.normalization == "sym"
    assert config.eigen_solver == "dense"
    assert config.project_dim is None


def test_default_analysis_uses_the_full_basis_and_sparse_stays_available(caplog):
    S = 80
    L = _random_laplacian(S, seed=2)
    X = torch.randn(S, 8, generator=torch.Generator().manual_seed(2))
    with caplog.at_level("WARNING", logger="spectral_trust.spectral"):
        full = SpectralAnalyzer(GSPConfig(device="cpu")).analyze_layer(
            X, torch.from_numpy(L), 0)
    assert full.eigenvalues.shape == (S,)
    assert not any("Truncated eigenbasis" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING", logger="spectral_trust.spectral"):
        trunc = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="sparse")) \
            .analyze_layer(X, torch.from_numpy(L), 0)
    assert trunc.eigenvalues.shape == (50,)
    assert any("Truncated eigenbasis" in r.message for r in caplog.records)


def test_cli_config_uses_the_dense_default():
    """The CLI has no solver flag; its fallback must follow the library."""
    import argparse
    from spectral_trust.cli import create_model_config
    cfg = create_model_config("example-org/custom-model", argparse.Namespace())
    assert cfg.eigen_solver == "dense"
