# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""Seeded inputs shared by make_regression_050.py and test_regression_050.py."""
import torch

#: GSPConfig overrides recorded by the fixture (the defaults, and the other
#: normalisations, symmetrisations and head aggregation).
CONFIGS = {
    "default": {},
    "rw": {"normalization": "rw"},
    "none": {"normalization": "none"},
    "rownorm_attnw": {"symmetrization": "row_norm", "head_aggregation": "attention_weighted"},
    "selfloops_cut05": {"remove_self_loops": True, "hfer_cutoff_ratio": 0.5},
}
#: token_span values passed to compute_spectra.
SPANS = {"all": None, "drop_bos": (1, None), "tail": (-8, None)}
#: subgraph_indices passed to analyze_text.
SUBGRAPH = list(range(1, 22))


def causal(batch: int, heads: int, seq: int, gen: torch.Generator, sink: float) -> torch.Tensor:
    """Row-stochastic causal attention with extra weight on token 0."""
    a = torch.rand(batch, heads, seq, seq, generator=gen).tril()
    a[..., 0] += sink * a.sum(-1)
    return a / a.sum(-1, keepdim=True)


def inputs():
    """``(attentions, signals, tokens, lengths)`` for compute_spectra: three
    layers, four heads, two texts of 22 and 17 tokens (right-padded to 22),
    hidden size 12."""
    g = torch.Generator().manual_seed(1234)
    seq, heads, d = 22, 4, 12
    attn, sig = {}, {}
    for lay, sink in zip((0, 1, 2), (0.5, 2.0, 6.0)):
        attn[lay] = causal(2, heads, seq, g, sink)
        sig[lay] = torch.randn(2, seq, d, generator=g)
    lengths = [22, 17]
    tokens = [[f"t{i}" for i in range(n)] for n in lengths]
    return attn, sig, tokens, lengths
