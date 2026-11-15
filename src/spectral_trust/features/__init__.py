# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""
Features that answer the critiques of spectral attention diagnostics:
attention-sink measures and covariates (:mod:`spectral_trust.features.sinks`)
the orientation and capacity of a head
(:mod:`spectral_trust.features.asymmetry`), and the reference Geometry of
Reason features (:mod:`spectral_trust.features.gor`). New in 0.5.1.
"""
from .sinks import (
    sink_mass, value_norms, value_weighted_sink, effective_rank, attention_covariates,
    ATTENTION_COVARIATES,
)
from .asymmetry import transport_operator, asymmetry, capacity_phi, transport_sigma2
from .gor import geometry_of_reason, geometry_of_reason_text, GOR_FEATURES

__all__ = [
    "sink_mass", "value_norms", "value_weighted_sink", "effective_rank",
    "attention_covariates", "ATTENTION_COVARIATES",
    "transport_operator", "asymmetry", "capacity_phi", "transport_sigma2",
    "geometry_of_reason", "geometry_of_reason_text", "GOR_FEATURES",
]
