import pytest
import torch
import numpy as np
from spectral_trust.config import GSPConfig
from spectral_trust.spectral import SpectralAnalyzer
from spectral_trust.graph import GraphConstructor

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

def test_config_backward_compatibility():
    """Verify GSPConfig defaults maintain v0.1.4 behavior."""
    config = GSPConfig()
    assert config.directed == False
    assert config.calc_velocity == False
    assert config.subgraph_indices is None
    assert config.normalization == "rw"
