# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

__version__ = "0.4.0rc1"

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
    "__version__",
]
