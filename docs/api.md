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

## Views

`spectral_trust.viz`; also `SpectralCache.graph`, `.heatmap` and `.spectrum`.

::: spectral_trust.viz.graph

::: spectral_trust.viz.heatmap

::: spectral_trust.viz.spectrum

::: spectral_trust.viz.View

## Per-head diagnostics

::: spectral_trust.per_head_metrics

::: spectral_trust.per_head_metrics_all_layers

::: spectral_trust.per_head_hybrid_metrics

::: spectral_trust.per_head_hybrid_metrics_all_layers

::: spectral_trust.PerHeadDiagnostics

::: spectral_trust.heads_from_scores

::: spectral_trust.stack_feature_vector

::: spectral_trust.layer_delta_features

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
