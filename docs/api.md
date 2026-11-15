# API reference

Every name below is importable from `spectral_trust` (`import spectral_trust
as st`), except where a module path is shown.

## High-level API

::: spectral_trust.SpectralModel

::: spectral_trust.SpectralCache

::: spectral_trust.cache.LayerSpectra

::: spectral_trust.cache.HeadSpectra

::: spectral_trust.compute_spectra

## Adapters

::: spectral_trust.from_transformer_lens

::: spectral_trust.from_nnsight

::: spectral_trust.from_hf_outputs

## Interventions

Added in 0.5.0, in the subpackage `spectral_trust.interventions`. The
context manager `st.intervene` and `st.patch`, `st.Patcher`,
`st.PatchSpec`, `st.equal_count_bands`, `st.graph_filter`, `st.weight_snr`
and `st.weight_svd_full` are also top-level names. See the
[interventions tutorial](tutorials/interventions.md).

::: spectral_trust.interventions.context.intervene

::: spectral_trust.interventions.context.Intervention

::: spectral_trust.interventions.context.ResidSite

::: spectral_trust.interventions.context.AttentionSite

::: spectral_trust.interventions.context.WeightSite

::: spectral_trust.interventions.context.Edit

::: spectral_trust.interventions.context.remove_edges_from_pattern

::: spectral_trust.interventions.patching.patch

::: spectral_trust.interventions.patching.Patcher

::: spectral_trust.interventions.patching.PatchSpec

::: spectral_trust.interventions.patching.equal_count_bands

::: spectral_trust.interventions.patching.select_modes

::: spectral_trust.interventions.patching.haar_basis

::: spectral_trust.interventions.filters.graph_filter

::: spectral_trust.interventions.filters.filter_coefficients

::: spectral_trust.interventions.filters.filter_response

::: spectral_trust.interventions.filters.ideal_response

::: spectral_trust.interventions.filters.attention_adjacency

::: spectral_trust.interventions.filters.path_adjacency

::: spectral_trust.interventions.filters.laplacian64

::: spectral_trust.interventions.filters.compute_placement

::: spectral_trust.interventions.weights.weight_snr

::: spectral_trust.interventions.weights.weight_svd_full

::: spectral_trust.interventions.weights.svd_scale_matrix

::: spectral_trust.interventions.weights.matrix_snr

::: spectral_trust.interventions.weights.lowrank_snr

## Views

`spectral_trust.viz`; also `SpectralCache.graph`, `.heatmap`, `.spectrum` and
`.animate` (see the [animated gallery](gallery.md)).

::: spectral_trust.viz.graph

::: spectral_trust.viz.heatmap

::: spectral_trust.viz.spectrum

::: spectral_trust.viz.View

::: spectral_trust.animation.animate

## Per-head diagnostics

::: spectral_trust.per_head_metrics

::: spectral_trust.per_head_metrics_all_layers

::: spectral_trust.per_head_hybrid_metrics

::: spectral_trust.per_head_hybrid_metrics_all_layers

::: spectral_trust.PerHeadDiagnostics

::: spectral_trust.heads_from_scores

::: spectral_trust.stack_feature_vector

::: spectral_trust.layer_delta_features

## Directed and value-weighted graphs

Added in 0.5.1. Graph options of `GSPConfig` (`edge_weights`,
`graph_direction`, `magnetic_q`) and the functions behind them; see
[Directed and value-weighted graphs](tutorials/directed-graphs.md).

::: spectral_trust.value_weighted_attention

::: spectral_trust.capture_attention_values

::: spectral_trust.AttentionValues

::: spectral_trust.ValueCapture

::: spectral_trust.value_weighted_edges

::: spectral_trust.attention_contributions

::: spectral_trust.directed_operator

::: spectral_trust.directed_total_variation

::: spectral_trust.magnetic_laplacian

::: spectral_trust.magnetic_spectrum

::: spectral_trust.magnetic_hfer

::: spectral_trust.magnetic_phase

::: spectral_trust.hermitian_clustering

::: spectral_trust.flow_imbalance

::: spectral_trust.FlowClusters

::: spectral_trust.directed.analyze_directed_layer

## Per-column diagnostics

::: spectral_trust.per_column_metrics

::: spectral_trust.ColumnDiagnostics

## Graphs and layer diagnostics

::: spectral_trust.GSPConfig

::: spectral_trust.GraphConstructor

::: spectral_trust.SpectralAnalyzer

::: spectral_trust.SpectralDiagnostics

::: spectral_trust.GSPDiagnosticsFramework

::: spectral_trust.attention_layer_indices

## Fast diagnostics

::: spectral_trust.row_dirichlet_energy

::: spectral_trust.row_value_spread

::: spectral_trust.top_band_hfer

::: spectral_trust.num_high_modes

::: spectral_trust.gaussian_projection

::: spectral_trust.jl_project

::: spectral_trust.band_energies

::: spectral_trust.band_entropy

::: spectral_trust.band_filter_response
