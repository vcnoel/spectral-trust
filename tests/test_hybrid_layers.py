"""Hybrid models (linear-attention layers) return attention for some layers only.
analyze_text must pair each attention tensor with its own layer's hidden state."""
from types import SimpleNamespace

import pytest
import torch

from spectral_trust import (
    GSPConfig, GSPDiagnosticsFramework, GraphConstructor, SpectralAnalyzer,
    attention_layer_indices,
)

QWEN35_LIKE = ["linear_attention"] * 3 + ["full_attention"]
QWEN35_LIKE = QWEN35_LIKE * 6  # 24 layers, full attention at 3, 7, ..., 23
FULL_LAYERS = [3, 7, 11, 15, 19, 23]


def test_standard_models_map_tensor_i_to_layer_i():
    assert attention_layer_indices(None, 12, 13) == list(range(12))


def test_sliding_and_full_layers_all_return_attention():
    cfg = SimpleNamespace(layer_types=["sliding_attention", "full_attention"] * 3)
    assert attention_layer_indices(cfg, 6, 7) == list(range(6))


def test_hybrid_layers_are_read_from_layer_types():
    cfg = SimpleNamespace(text_config=SimpleNamespace(layer_types=QWEN35_LIKE))
    assert attention_layer_indices(cfg, 6, 25) == FULL_LAYERS


def test_unknown_layout_raises_and_names_the_override():
    with pytest.raises(ValueError, match="attention_layer_map"):
        attention_layer_indices(SimpleNamespace(), 6, 25)


def test_override_wins_and_is_checked():
    assert attention_layer_indices(None, 2, 25, override=[5, 9]) == [5, 9]
    with pytest.raises(ValueError):
        attention_layer_indices(None, 2, 25, override=[5])


def _causal_attention(seq, heads, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(1, heads, seq, seq, generator=g).tril()
    return a / a.sum(-1, keepdim=True)


def test_analyze_text_pairs_each_attention_with_its_own_layer(tmp_path):
    seq, heads, dim = 10, 2, 16
    g = torch.Generator().manual_seed(0)
    hidden = [torch.randn(1, seq, dim, generator=g) for _ in range(25)]
    attns = [_causal_attention(seq, heads, s) for s in range(6)]
    cfg = GSPConfig(device="cpu", output_dir=str(tmp_path), save_plots=False)

    fw = GSPDiagnosticsFramework.__new__(GSPDiagnosticsFramework)
    fw.config = cfg
    fw.graph_constructor = GraphConstructor(cfg)
    fw.spectral_analyzer = SpectralAnalyzer(cfg)
    fw.instrumenter = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(
            text_config=SimpleNamespace(layer_types=QWEN35_LIKE))),
        process_text=lambda text: {"attentions": attns, "hidden_states": hidden,
                                   "inputs": {}, "tokens": ["t"] * seq})
    out = fw.analyze_text("x", save_results=False)
    diags = out["layer_diagnostics"]
    assert [d.layer for d in diags] == FULL_LAYERS

    gc, an = fw.graph_constructor, fw.spectral_analyzer
    for k, layer in enumerate(FULL_LAYERS):
        sym = gc.symmetrize_attention(attns[k]).squeeze(0)
        adj = gc.aggregate_heads(sym.unsqueeze(0), attn_mask=None).squeeze(0)
        lap = gc.construct_laplacian(adj.unsqueeze(0)).squeeze(0)
        expected = an.analyze_layer(hidden[layer + 1].squeeze(0), lap, layer)
        assert diags[k].energy == pytest.approx(expected.energy, rel=1e-6)
        assert diags[k].hfer == pytest.approx(expected.hfer, rel=1e-6)
