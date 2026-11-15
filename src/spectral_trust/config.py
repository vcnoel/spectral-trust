# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field, dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class GSPConfig:
    """Configuration for GSP diagnostics"""
    # Model parameters
    model_name: str = "gpt2"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GSP parameters
    head_aggregation: str = "uniform"  # uniform, attention_weighted, learnable
    symmetrization: str = "symmetric"  # symmetric, row_norm, col_norm
    # Default changed rw -> sym in v0.3.0: the symmetric normalized Laplacian
    # has spectrum in [0, 2] independent of sequence length, so metrics are
    # comparable across inputs of different length (the combinatorial and
    # random-walk variants scale with T, a length confound); it is also a
    # symmetric operator, so the deterministic eigh/eigsh path is always used.
    normalization: str = "sym"  # sym (symmetric), rw (random walk), none
    # HFER counts the top n - int((1 - hfer_cutoff_ratio) * n) of the n modes
    # as high frequency (spectral_trust.fast.num_high_modes, the one rule
    # every HFER path uses); must lie in [0, 1].
    hfer_cutoff_ratio: float = 0.1
    
    # Spectral computation
    # Default changed sparse -> dense in v0.4.0. "sparse" (ARPACK) keeps only
    # num_eigenvalues modes once seq_len > 50 (half from each end of the
    # spectrum), so HFER, spectral entropy and spectral masses are normalised
    # over a truncated basis and are not comparable to a full computation (a
    # warning is logged); it was also slower than the dense solve at every
    # size benchmarked. "dense" is the full eigendecomposition.
    num_eigenvalues: int = 50  # Modes kept by eigen_solver="sparse"
    eigen_solver: str = "dense"  # dense (full eigendecomposition), sparse (ARPACK, truncated)
    lanczos_max_iter: int = 1000

    # Random projection of the signals before the hybrid metrics (energy,
    # smoothness_index, spectral_entropy, hfer, spectral_masses): a seeded
    # Gaussian Johnson-Lindenstrauss map from the hidden size d to
    # project_dim. Energy and spectral masses stay unbiased; the ratios carry
    # an O(1/sqrt(project_dim)) error (see spectral_trust.fast). None
    # disables it, and a width >= d leaves the signals unchanged. Attention
    # metrics are not affected.
    project_dim: Optional[int] = None
    project_seed: int = 0
    
    # Evaluation parameters
    batch_size: int = 1
    num_layers_analyze: Optional[int] = None  # None means all layers
    save_attention: bool = True
    save_activations: bool = True
    
    # Output parameters
    output_dir: str = "./gsp_results"
    save_plots: bool = True
    display_plots: bool = False
    save_intermediate: bool = True
    verbose: bool = True

    # Model loading options
    trust_remote_code: bool = False
    torch_dtype: str = "float32"
    device_map: Optional[str] = None
    local_files_only: bool = False
    
    # Multi-run options
    runs: int = 1
    temperature: float = 1.0
    plot_metrics: Optional[list] = None  # None/empty means "all"
    latex_export: bool = False
    
    # Advanced
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    remove_self_loops: bool = False


    # New in v0.2.0
    display_plots: bool = False
    directed: bool = False
    calc_velocity: bool = False
    subgraph_indices: Optional[list] = None
    # Layer index of each attention tensor the model returns. None reads it
    # from the model config (layer_types) when the model returns attention for
    # only some layers, as hybrid linear-attention models do.
    attention_layer_map: Optional[list] = None

    # New in v0.5.1: options of the head-averaged layer graph (the "graph"
    # namespace of SpectralModel.run_with_spectra and compute_spectra, and
    # GraphConstructor.layer_graph). The defaults build the graph that every
    # earlier release built, with identical results.
    # Edge weights: "attention" (the attention weights), "value" (the norm of
    # what token j adds to token i's attention output,
    # ||sum_h a^h_ij W_O^h v^h_j||, Kobayashi et al. 2020) or
    # "value_per_head" (a^h_ij ||W_O^h v^h_j|| per head, then aggregated over
    # heads like attention). See spectral_trust.value_weighted.
    edge_weights: str = "attention"
    # "symmetric" symmetrises the edges before the Laplacian (above);
    # "directed" keeps them as the causal row-normalised operator
    # P = D_out^-1 A, measured through its magnetic Laplacian and the directed
    # total variation (spectral_trust.directed).
    graph_direction: str = "symmetric"

    # New in v0.5.1: charge q of the magnetic Laplacian of a directed graph
    # (spectral_trust.directed). With edge weights in [0, 1] (rows of a
    # row-normalised operator), the phase 2*pi*q*(A_ij - A_ji) stays in
    # [-pi/2, pi/2] for q <= 1/4, and at q = 1/4 a one-way edge of weight 1
    # gets the phase pi/2 (see spectral_trust.directed.magnetic_laplacian).
    magnetic_q: float = 0.25

    # New in v0.5.1: the "operator" namespace of SpectralModel.run_with_spectra
    # and compute_spectra (spectral_trust.operator_spectra), computed only when
    # it is named in metrics= (metrics=None leaves it out, so every earlier
    # call is unchanged). Each layer's attention A is read as an operator
    # L2(mu) -> L2(pi); its rows are renormalised first (the span's own edges
    # under token_span).
    # Measures: "pullback" (pi uniform, mu = A^T pi) or "counting" (A^T A).
    operator_measure: str = "pullback"
    # "per_head": one value per head ([n_heads], heads= applies); "mean": one
    # value per layer, from the head-aggregated A (head_aggregation applies).
    operator_heads: str = "per_head"
    # A: "attention", or the value-weighted edges of spectral_trust.value_weighted
    # with each row divided by its sum: "value_per_head" (a^h_ij ||W_O^h v^h_j||,
    # per head) or "value" (||sum_h a^h_ij W_O^h v^h_j||, one matrix per
    # layer, so operator_heads="mean" only).
    operator_edges: str = "attention"
