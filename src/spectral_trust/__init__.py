# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

from .config import GSPConfig
from .framework import GSPDiagnosticsFramework, attention_layer_indices
from .spectral import (
    SpectralDiagnostics, SpectralAnalyzer, calculate_spectral_velocity,
    METRIC_FAMILIES,
)
from .graph import GraphConstructor
from .instrumentation import (
    LLMInstrumenter, NonFiniteAttentionError, assert_finite_attention,
)
from .directed_topology import DirectedTopologist
from .per_head import (
    PerHeadDiagnostics, per_head_metrics, per_head_metrics_all_layers,
    stack_feature_vector, layer_delta_features,
    PER_HEAD_METRIC_NAMES, PER_HEAD_METRIC_FAMILY,
    per_head_hybrid_metrics, per_head_hybrid_metrics_all_layers,
    heads_from_scores,
    PER_HEAD_HYBRID_METRIC_NAMES, PER_HEAD_HYBRID_METRIC_FAMILY,
)
from .fast import (
    row_dirichlet_energy, row_value_spread,
    top_band_hfer, num_high_modes,
    gaussian_projection, jl_project,
    band_energies, band_entropy, band_filter_response,
)
from .columns import ColumnDiagnostics, per_column_metrics, COLUMN_METRIC_NAMES
from .cache import SpectralCache, LayerSpectra, HeadSpectra, GRAPH_METRIC_NAMES
from .model import SpectralModel, compute_spectra
from .interop import from_transformer_lens, from_nnsight, from_hf_outputs
from . import viz
from . import interventions
from .interventions import (
    Patcher, PatchSpec, equal_count_bands, graph_filter, intervene, patch, weight_snr,
    weight_svd_full, edit_with_controls, EditControlResult,
)
from . import numeric
from .audit import (
    confound_audit, AuditResult, confound_covariates, covariates_from_cache, effective_rank,
    bootstrap_ci, paired_bootstrap, auroc,
)

__version__ = "0.5.0"

__all__ = [
    "GSPConfig",
    "GSPDiagnosticsFramework",
    "attention_layer_indices",
    "SpectralDiagnostics",
    "SpectralAnalyzer",
    "calculate_spectral_velocity",
    "METRIC_FAMILIES",
    "GraphConstructor",
    "LLMInstrumenter",
    "NonFiniteAttentionError",
    "assert_finite_attention",
    "DirectedTopologist",
    "PerHeadDiagnostics",
    "per_head_metrics",
    "per_head_metrics_all_layers",
    "stack_feature_vector",
    "layer_delta_features",
    "PER_HEAD_METRIC_NAMES",
    "PER_HEAD_METRIC_FAMILY",
    "per_head_hybrid_metrics",
    "per_head_hybrid_metrics_all_layers",
    "heads_from_scores",
    "PER_HEAD_HYBRID_METRIC_NAMES",
    "PER_HEAD_HYBRID_METRIC_FAMILY",
    "row_dirichlet_energy",
    "row_value_spread",
    "top_band_hfer",
    "num_high_modes",
    "gaussian_projection",
    "jl_project",
    "band_energies",
    "band_entropy",
    "band_filter_response",
    "ColumnDiagnostics",
    "per_column_metrics",
    "COLUMN_METRIC_NAMES",
    "SpectralModel",
    "SpectralCache",
    "LayerSpectra",
    "HeadSpectra",
    "GRAPH_METRIC_NAMES",
    "compute_spectra",
    "from_transformer_lens",
    "from_nnsight",
    "from_hf_outputs",
    "viz",
    "interventions",
    "intervene",
    "patch",
    "Patcher",
    "PatchSpec",
    "equal_count_bands",
    "graph_filter",
    "weight_snr",
    "weight_svd_full",
    "edit_with_controls",
    "EditControlResult",
    "numeric",
    "confound_audit",
    "AuditResult",
    "confound_covariates",
    "covariates_from_cache",
    "effective_rank",
    "bootstrap_ci",
    "paired_bootstrap",
    "auroc",
    "__version__",
]
