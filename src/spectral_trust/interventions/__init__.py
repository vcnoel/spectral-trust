# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Interventions on Hugging Face causal LMs, with spectral ones next to the
TransformerLens basics. Added in 0.5.0.

- :func:`intervene`: a context manager of hook-based edits, all undone on
  exit: graph-frequency steering of the residual stream along the token axis
  (:meth:`ResidSite.graph_filter`, symmetric or causal), activation
  steering (:meth:`ResidSite.add`), head ablation (zero, mean, resample),
  attention-edge removal with an explicit choice of where the removed mass
  goes (:meth:`AttentionSite.remove_edges`), and singular-value rescaling of
  a weight matrix (:meth:`WeightSite.svd_scale`).
- :func:`patch` and :class:`Patcher`: activation patching, plain or
  frequency-resolved (one band of a token-axis graph eigenbasis).
- :func:`weight_snr`: ``sigma_1 / median(sigma)`` of a weight matrix, and
  :func:`weight_svd_full`.
- :func:`graph_filter`: the filter on plain tensors.

The top-level package exports ``st.intervene`` (the context manager),
``st.patch``, ``st.Patcher``, ``st.PatchSpec``, ``st.equal_count_bands``,
``st.graph_filter``, ``st.weight_snr`` and ``st.weight_svd_full``.
"""
from .context import (
    KEEP_SINK_WARN_FACTOR, AttentionSite, Edit, Intervention, ResidSite, WeightSite, intervene,
    remove_edges_from_pattern,
)
from .filters import (
    FILTER_KINDS, GRAPH_NAMES, PRECISIONS, FilterSpec, attention_adjacency, compute_placement,
    filter_coefficients, filter_response, graph_filter, ideal_response, laplacian, laplacian64,
    path_adjacency,
)
from .patching import (
    BASES, PatchSpec, Patcher, equal_count_bands, haar_basis, patch, select_modes,
)
from ._hooks import SITES
from .weights import (
    lowrank_snr, matrix_snr, svd_scale_matrix, weight_snr, weight_svd_full,
)

__all__ = [
    "intervene", "Intervention", "Edit", "ResidSite", "AttentionSite", "WeightSite",
    "remove_edges_from_pattern", "KEEP_SINK_WARN_FACTOR",
    "graph_filter", "FilterSpec", "filter_coefficients", "filter_response", "ideal_response",
    "attention_adjacency", "path_adjacency", "laplacian", "laplacian64", "FILTER_KINDS",
    "GRAPH_NAMES", "PRECISIONS", "compute_placement",
    "patch", "Patcher", "PatchSpec", "equal_count_bands", "haar_basis", "select_modes", "BASES",
    "SITES",
    "weight_snr", "weight_svd_full", "matrix_snr", "lowrank_snr", "svd_scale_matrix",
]
