# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Numerical in-context learning: serialised number sequences, the model's
next-value distribution, and the attention-graph diagnostics of Bao et al.
(2026), "A Graph Signal Processing Perspective on Numerical Sequence
Representations in LLM In-Context Learning" (EACL 2027). New after 0.5.0 (unreleased).

- :func:`quantize`, :func:`serialize`: the ``"087,173,642,"`` format.
- :func:`value_spans`: the token span of each value and the separators.
- :func:`repeat_fraction`: how much of a sequence an induction (copy)
  mechanism can continue, a covariate of any complexity ordering.
- :func:`next_value_distribution`: the predictive distribution over the next
  value (one restricted softmax, or digit-wise probabilities composed
  hierarchically), with its mean, standard deviation and entropy.
- :func:`profile_from_outputs`: normalised Fiedler value, HFER and
  ``exp(SE)`` per layer, with the paper's graph construction (BOS dropped,
  rows renormalised, combinatorial Fourier basis, input hidden states) and
  its non-graph controls; :func:`pool_nodes` merges digit tokens into one
  node per value.
"""
from .diagnostics import (
    PROFILE_METRICS, GraphSignalProfile, attention_gini, attention_graph,
    combinatorial_laplacian, graph_signal_metrics, normalized_laplacian, pool_nodes,
    positional_metrics, profile_from_outputs, sink_mass,
)
from .readout import ValueDistribution, digit_token_ids, next_value_distribution, value_token_ids
from .serialize import ValueSpans, quantize, repeat_fraction, serialize, value_spans

__all__ = [
    "quantize", "serialize", "value_spans", "ValueSpans", "repeat_fraction",
    "next_value_distribution", "ValueDistribution", "value_token_ids", "digit_token_ids",
    "profile_from_outputs", "GraphSignalProfile", "graph_signal_metrics", "attention_graph",
    "positional_metrics", "attention_gini", "sink_mass", "pool_nodes", "normalized_laplacian",
    "combinatorial_laplacian", "PROFILE_METRICS",
]
