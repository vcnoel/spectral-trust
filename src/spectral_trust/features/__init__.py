# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""
Features that answer the critiques of spectral attention diagnostics:
attention-sink measures and covariates (:mod:`spectral_trust.features.sinks`)
and the orientation and capacity of a head
(:mod:`spectral_trust.features.asymmetry`). New in 0.5.1.
"""
from .sinks import (
    sink_mass, value_norms, value_weighted_sink, effective_rank, attention_covariates,
    ATTENTION_COVARIATES,
)
from .asymmetry import transport_operator, asymmetry, capacity_phi, transport_sigma2

__all__ = [
    "sink_mass", "value_norms", "value_weighted_sink", "effective_rank",
    "attention_covariates", "ATTENTION_COVARIATES",
    "transport_operator", "asymmetry", "capacity_phi", "transport_sigma2",
]
