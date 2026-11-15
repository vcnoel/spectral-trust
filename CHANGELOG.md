# Changelog

## Known issue in releases before 0.2.2

Releases 0.1.0 to 0.2.1 computed the layer-level spectrum
(`SpectralAnalyzer.compute_eigendecomposition`, used by
`GSPDiagnosticsFramework`) with `scipy.linalg.eigh` (`eigsh` on the sparse
path) for every normalization. Under `normalization="rw"`, the default until
0.3.0, the Laplacian `I − D⁻¹W` is not symmetric: `eigh` reads only one
triangle of it, so it returned the spectrum of a different, symmetric
matrix, and negative eigenvalues were then clipped to 0. The Fiedler value
and the other metrics computed from that eigendecomposition were wrong,
mostly too low: recomputing the 0.1.2 construction on Phi-3-mini-4k-instruct
gave Fiedler values up to 0.36 below the correct ones (and up to 0.07
above). `normalization="sym"` and `"none"` were not affected. 0.2.2
diagonalises non-symmetric Laplacians with a general solver, and 0.3.0 made
`"sym"` the default. Results computed with 0.1.x, 0.2.0 or 0.2.1 under
`normalization="rw"` should be recomputed with a current release. The
per-head functions, added in 0.3.0, had a related issue with `"rw"` that
0.4.0 fixed (see v0.4.0).

## Unreleased: critique pack

Measures that answer the critiques of spectral attention diagnostics. No
result changes: every 0.5.0 call is bit-identical
(`tests/test_regression_050.py`).

- `GSPConfig.subgraph_renormalize`: the nodes kept by `subgraph_indices` or
  `token_span` are renormalised per row before the graph is built (BOS drop
  as in Bao et al.); `graph.drop_and_renormalize`.
- `spectral_trust.features`: `sink_mass`, `value_norms`,
  `value_weighted_sink`, `effective_rank`, `attention_covariates`; the
  asymmetry `G` and conductance `φ̂` of Dahlem et al. (`asymmetry`,
  `capacity_phi`, `transport_sigma2`, `transport_operator`; from their
  equations, no released code); `geometry_of_reason`, the paper's features
  as its code computed them, bit-identical to 0.1.2.
- `spectral_trust.pseudospectra`: `sigma_min`, `pseudospectrum`,
  `pseudospectral_radius`, `pseudospectral_abscissa`, `kreiss_constant`,
  `departure_from_normality`, `numerical_range`, `depth_propagator`,
  `nonnormality`.
- Docs: "What a symmetrised spectrum cannot see" and "Baselines".

## v0.5.0 (2026-11-15)

Interventions: the TransformerLens basics (patching, ablation, steering)
on Hugging Face models, and interventions in the graph-spectral domain. No
result changes for existing calls.

- New subpackage `spectral_trust.interventions` and context manager
  `st.intervene(model)`: hook-based edits, all undone on exit (hooks
  removed, weights restored bit for bit, the attention implementation and
  any wrapped `transformers` function put back). It takes a `SpectralModel`
  or a plain `transformers` causal LM (GPT-2, Llama, Qwen, Mistral, Gemma,
  GPT-NeoX and OPT layouts). Interventions on one model may be nested and
  closed in any order. Positions are absolute, so position-dependent edits
  agree between cached generation and recomputation.
- Public names of `spectral_trust.interventions`:
  - edits: `intervene` (also `st.intervene`), `Intervention` (`resid`,
    `attention`, `weights`, `hook`, `close`), `ResidSite` (`add`,
    `graph_filter`), `AttentionSite` (`ablate`, `head_outputs`,
    `edit_pattern`, `remove_edges`), `WeightSite` (`svd_scale`, `snr`,
    `weight`), `Edit` (`remove`) and `remove_edges_from_pattern`;
  - graph filters: `graph_filter` (also `st.graph_filter`), `FilterSpec`,
    `filter_coefficients`, `filter_response`, `ideal_response`,
    `attention_adjacency`, `path_adjacency`, `laplacian`, `laplacian64`,
    `compute_placement`, `FILTER_KINDS`, `GRAPH_NAMES`, `PRECISIONS`;
  - patching: `patch`, `Patcher`, `PatchSpec`, `equal_count_bands` (all
    four also top-level), `select_modes`, `haar_basis`, `BASES`, `SITES`;
  - weights: `weight_snr`, `weight_svd_full` (both also top-level),
    `matrix_snr`, `lowrank_snr`, `svd_scale_matrix`;
  - warning thresholds: `KEEP_SINK_WARN_FACTOR` (10) and
    `GRAPH_FILTER_WARN_RELATIVE_EDIT` (0.5).
- Graph-frequency steering: `iv.resid(layer).graph_filter(kind, cutoff= |
  band=, strength=, graph=, causal=)` filters the residual stream along the
  token axis on that layer's head-averaged attention graph (built as the
  measurements build it), the chain graph, or a given graph:
  `X' = X + s (g(L) X − X)`. Low-pass, high-pass, band-pass and band-stop
  responses use the Jackson-damped Chebyshev machinery of `band_energies`
  (no eigendecomposition), computed in float64 by default (on the CPU when
  the model is on Apple MPS) or in float32 with `precision="float32"`,
  faster on GPUs; `method="dense"` applies the ideal band indicator. `causal=True` filters each position on the graph
  of its prefix, a lower-triangular operator, so generation with a KV cache
  matches recomputation, beam search included; a cached step under the
  symmetric filter raises.
- Head ablation (`ablate("zero" | "mean" | "resample")` on the per-head
  outputs; mean and resample take a reference, and resample draws its items
  once per input, shared by its beams), activation steering
  (`resid(l).add`), custom hooks (`iv.hook(site, layer, fn)`) and
  attention-pattern edits (`edit_pattern`; these need transformers 4.48 or
  later and say so at run time, while the package keeps requiring 4.30).
- `remove_edges(key_mask, keep_sink=True)` removes attention edges and
  renormalises each edited row over its remaining non-sink keys with the
  first token's entry held fixed; `keep_sink=False` renormalises over every
  remaining key, the first token included, and `renormalize=False` only
  zeroes. None of the three is a pure edge ablation when the removed edges
  carry most of a row: holding the sink fixed then multiplies the edges
  left by a large factor (13 to 181 for the GPT-2 name movers in the IOI
  use case).
- Weight-spectrum steering: `iv.weights(layer, module).svd_scale(alpha, k=)`
  applies the spectral-steering rescaling
  `σ' = σ (1 + α (σ − mean) / std)` and warns when a factor turns
  non-positive. The rebuilt matrix equals spectral-steering's reference code
  bit for bit; the two give the same model for bias-free `nn.Linear`
  projections (the Llama-family `mlp.down_proj`), since the reference swaps
  in a bias-free layer while `svd_scale` edits the weight in place, keeps
  the bias and also handles GPT-2's `Conv1D`. Edits of one matrix stack and
  can be removed in any order.
- `st.weight_snr(model, layer, module)` is `σ₁ / median(σ)` over the full
  spectrum (`layer=None`: every layer). `st.weight_snr(W, k)` on a tensor and
  the new `st.weight_svd_full(W)` are the two functions of the unreleased
  0.2.3 branch that spectral-steering's scripts import, so they run on this
  release.
- Activation patching: `st.patch(model, clean, corrupt, site=, layer=,
  head=, positions=)` and `st.Patcher`, which runs the clean and corrupt
  inputs once, captures activations as patches need them, batches many
  patches, and raises if the model changed since the capture.
  Frequency-resolved patching (`band=`, `basis="clean_attention" |
  "corrupt_attention" | "path" | "random"`) takes one band of a token-axis
  eigenbasis from the clean run; the full band equals plain patching.
  `st.equal_count_bands` splits the modes into equal-count bands.
- Warnings: `remove_edges` with `keep_sink=True` warns, once per call,
  when it multiplies the kept non-sink attention of any edited row by more
  than 10; `graph_filter` on the residual stream warns, once per call, when
  its edit exceeds half the residual norm of the edited positions
  (`‖X' − X‖ / ‖X‖ > 0.5`). `svd_scale` warns when a singular value's factor
  turns non-positive.
- New tutorial: Interventions.
- Use cases (`docs/use-cases/`, with scripts and outputs in
  `examples/use_cases/`): edge ablation and the attention sink on GPT-2 IOI,
  graph-frequency steering against matched controls, frequency-resolved
  patching on IOI, which words split Phi-3-mini's attention graph, the side
  effects of `graph_filter` on GPT-2 and Phi-3-mini, and the design of a
  band-stop injection defence, which has not been run.

## v0.4.2

Animated views. One call writes a video, a GIF or an interactive page of a
`SpectralCache`. No computed value changes.

- New `SpectralCache.animate(kind, out, ...)` and
  `st.viz.animate(cache, kind, out, ...)`. The file suffix picks the format:
  `.mp4` (H.264), `.webm` (VP9), `.gif`, or `.html` (interactive).
- Kinds:
  - `flow`: each token is a tube through depth, from layer 0 to the last.
    Its position at each layer is the attention graph's Fiedler vector
    (sign-matched from layer to layer) and third eigenvector, its colour is
    the Fiedler side, and its thickness is the attention it receives.
    `other=` outlines the tokens missing from a second prompt, for example an
    injected instruction. Drawn with three.js; videos are captured frame by
    frame from the same page in headless Chrome, with no new Python
    dependency, and fall back to matplotlib when no browser is found.
  - `waterfall`: the energy share of each graph-frequency mode as 3D
    ridgelines through depth, with the HFER band marked.
  - `compare`: two prompts on shared token positions, with the tokens present
    in only one outlined, and the Fiedler value and HFER of both traced
    through depth.
  - `graph_movie`: the circular attention graph morphing through the layers.
  - `head_atlas`: every head of a layer as small multiples sorted by a
    per-head metric, sweeping over the layers.
  - `embedding3d`: tokens in 3D by Laplacian eigenvectors 2 to 4.
- Okabe–Ito palette, light and dark themes, and deterministic output for a
  fixed `seed`. Rendering runs on the CPU, and ffmpeg uses two threads by
  default (`threads=`).
- The `viz` extra adds imageio and imageio-ffmpeg. Interactive plotly pages
  embed plotly.js when the `plotly` package is installed, and load a pinned
  CDN copy otherwise. Flow pages embed three.js r160 (MIT; its licence ships
  with the package).
- Docs: an animated gallery page.

## v0.4.1

A new licence, a per-column metric, parallel layers for the per-head
functions, an explicit device for them, and a corrected tolerance
statement. No result changes for existing calls.

- *(licence change)* spectral-trust is released under the Apache License
  2.0 from this version, instead of AGPL-3.0-or-later, so it can be used in
  academic and commercial work, open or closed source. Versions up to 0.4.0
  remain under AGPL-3.0-or-later.
- New `per_column_metrics(signals, laplacian, config)`: energy, smoothness
  index, spectral entropy and HFER of every column of a signal matrix (one
  SAE feature, neuron or probe projection per column) from one
  eigendecomposition. Column `j` gets what `analyze_layer` reports for
  `signals[:, j:j+1]` alone. Returns a `ColumnDiagnostics`.
- `per_head_metrics_all_layers` and `per_head_hybrid_metrics_all_layers`
  take `num_workers`: layers are decomposed at once, each on one CPU
  thread, with identical results in layer order. At 28 layers, 16 heads and
  256 tokens on a 24-thread CPU, 8 workers took the attention metrics from
  0.55 s to 0.15 s (3.7x) and the hybrid metrics from 1.30 s to 0.38 s
  (3.4x).
- `per_head_metrics` and `per_head_hybrid_metrics` (and their `_all_layers`
  versions) take `device`. The default stays the CPU, which is faster for
  many small eigenproblems; `"cuda"` decomposes on the GPU. They never read
  `GSPConfig.device`, and the docstrings now say so (it defaults to
  `"cuda"` when one is available, which suggested otherwise).
- Per-head calls change only their own thread's count. Each call runs its
  eigendecomposition on one intra-op thread and restores the count after.
  `torch.set_num_threads` also sets the count that threads created later
  start with, so in 0.4.0 a thread created during a per-head call started
  with one thread and kept it, and could leave one thread as the default
  for the rest of the process. The calls now put that default back at once,
  from a short-lived helper thread, so threads created meanwhile (including
  the `num_workers` workers) start at the process default, and the default
  is unchanged afterwards. The cost is about 0.4 ms per call.
- The `per_head_hybrid_metrics` docstring said its values match
  `analyze_layer` on each head. They agree to float32 rounding: energy and
  smoothness index to about 1e-6 relative, HFER and spectral entropy, which
  depend on eigenvectors computed by different float32 solvers (torch and
  scipy), up to 8e-5 and 1.2e-3 absolute on Qwen3.5-2B attention with
  one-column signals. Both are equally close to a float64 solve.

## v0.4.0

A TransformerLens-style API (`SpectralModel`, `SpectralCache`), adapters
for TransformerLens, nnsight and Hugging Face outputs, interactive views,
fast diagnostics, a per-head `rw` fix, and a dense default
solver. Four changes alter results for existing callers; they are marked
*(behaviour change)*. With `normalization="sym"` and `eigen_solver="dense"`,
and for the per-head attention metrics, 0.4.0 reproduces 0.3.0 to float32
rounding (`benchmarks/reproduce_v030.md`).

- *(behaviour change)* Default `eigen_solver` `"sparse"` → `"dense"`, in
  `GSPConfig` and in the CLI. Above 50 tokens the sparse default kept 50 of
  the S modes, so `hfer`, `spectral_entropy`, `spectral_masses`, the `eig_*`
  metrics and the length of `eigenvalues` came from a truncated basis (its
  HFER was 4x to 30x smaller than the full-basis value in the benchmark);
  `fiedler_value`, `spectral_radius` and `connectivity` agree up to the
  ARPACK tolerance. The sparse solve was also slower than the dense one at
  every size measured. Callers at 50 tokens or fewer, or who already passed
  `eigen_solver="dense"`, get the same results. Pass `eigen_solver="sparse"`
  to reproduce the old default; it still logs the truncation warning.
- *(behaviour change)* The CLI's `--normalization` default `"rw"` → `"sym"`,
  matching `GSPConfig` since 0.3.0. Pass `--normalization rw` for the old
  CLI results.
- *(behaviour change)* `connectivity` is true when the second-smallest
  Laplacian eigenvalue is at least `max(1e-6, 16 * sqrt(S) * eps * lambda_max)`,
  through `SpectralAnalyzer.zero_eigenvalue_tolerance` and `is_connected`.
  0.3.0 counted eigenvalues below a fixed 1e-6 and required exactly one, so
  float32 noise on the zero mode (1.07e-6 on gpt2 layer 5, and larger on
  unnormalised `"none"` graphs) reported connected graphs as disconnected.
  Float64 results change only where the zero mode itself exceeded 1e-6. For
  float32 normalised graphs the threshold is at most about 2e-4 at 2048 tokens.
- *(behaviour change)* Hybrid models such as Qwen3.5, which return attention
  for their full-attention layers only: `analyze_text` now pairs each
  attention tensor with the output of its own layer and reports the true
  layer index, read from `layer_types` in the model config. 0.3.0 numbered
  the returned tensors 0, 1, 2, ... and paired tensor i with
  `hidden_states[i + 1]`, so the hybrid metrics (`energy`,
  `smoothness_index`, `spectral_entropy`, `hfer`) used another layer's
  hidden states. New: `GSPConfig.attention_layer_map` and
  `attention_layer_indices`. Models that return attention for every layer
  are unchanged.
- **High-level API** (`model.py`, `cache.py`): `SpectralModel.from_pretrained`
  loads a Hugging Face causal LM with eager attention, or wraps a loaded one
  (switching it to eager attention), and `run_with_spectra(text_or_list,
  metrics, layers, heads, token_span, hybrid)` returns a `SpectralCache`. Its
  keys follow TransformerLens: `blocks.{l}.attn.{metric}` (one value per
  head, `per_head_metrics`), `blocks.{l}.resid.{metric}` (per-head hybrid
  metrics) and `blocks.{l}.graph.{metric}` (`analyze_layer` on the
  head-averaged graph of `analyze_text`, which it reproduces exactly). Layer
  indices are the model's own on hybrid models. Lists of texts run as
  right-padded batches and equal one-text runs to float32 rounding. The
  per-head eigendecompositions of several layers share one batched call.
  The cache has `layer(l).head(h)` views, `tokens`, `layers`,
  `metric_names`, `families`, `to_dataframe()` in long format, and
  `save` / `SpectralCache.load` to one `.npz` file with JSON metadata (no
  pickle). `compute_spectra` runs the same computation on any tensors.
- **Adapters** (`interop.py`): `from_transformer_lens` reads
  `blocks.{l}.attn.hook_pattern` and `blocks.{l}.hook_resid_post` from an
  `ActivationCache` (`TransformerBridge` or `HookedTransformer`),
  `from_nnsight` takes attention weights and optionally hidden states saved
  in an nnsight trace, and `from_hf_outputs` a Hugging Face output; each
  crops a right-padded batch with `attention_mask=`. On gpt2 the nnsight and
  Hugging Face paths equal `run_with_spectra`, and `TransformerBridge`, which
  computes attention itself, agrees to 1e-4 relative
  (`tests/test_interop.py`). New extras `[tl]` and `[nnsight]`; the core
  install imports neither.
- **Views** (`viz.py`): `graph`, `heatmap` and `spectrum` render a cache as
  self-contained SVG with tooltips and node highlighting, inline in Jupyter
  and saved as standalone HTML that opens offline; `heatmap(...,
  backend="matplotlib")` returns a figure. Also `cache.graph`,
  `cache.heatmap`, `cache.spectrum`. New extra `[viz]`.
- `per_head_metrics(keep_eigenvalues=True)` keeps each head's spectrum in
  `PerHeadDiagnostics.eigenvalues`.
- Documentation site (mkdocs-material, mkdocstrings) with getting started,
  concepts, three tutorials and the API reference, built on pull requests and
  deployed from `main`. Extras `[docs]` and `[dev]`.
- README: a static hero in light and dark themes and an SVG wordmark, both
  generated by `docs/make_assets.py`; the animated GIF is removed.
- CI: a model-matrix job runs `run_with_spectra` on tiny random checkpoints
  of several architectures, and an interop job runs the adapter tests.
- The CLI's `data/probes.json` is now in the wheel and the sdist
  (`package_data` and `MANIFEST.in`), so `gsp-cli diagnose` works after a
  pip install. It was missing from 0.3.0.
- **Row-anchored energies** (`fast.py`): `row_dirichlet_energy(W, X, rows)`
  and `row_value_spread(A, V, rows)`, batched over heads, in torch on CPU or
  GPU, with no eigendecomposition.
- **Exact top-band HFER**: `top_band_hfer(L, X)` returns the full-basis HFER
  of a symmetric Laplacian from the top `num_high_modes(S, cutoff_ratio)`
  eigenpairs, equal to the dense path.
- **Random projection**: `GSPConfig.project_dim` (default `None`) and
  `project_seed` apply a seeded Gaussian Johnson-Lindenstrauss map to the
  signals before the hybrid metrics; `gaussian_projection` and `jl_project`
  are public.
- **Chebyshev band energies**: `band_energies(L, X, edges, order)` with
  Jackson-damped filters (or `method="dense"`), `band_entropy`, and
  `band_filter_response` to read the filters off a spectrum.
- **Head subsets**: `heads=` in `per_head_metrics` and
  `per_head_metrics_all_layers` (indices, `(layer, head)` pairs, or boolean
  masks), `heads_from_scores` for user-supplied scores, and
  `PerHeadDiagnostics.heads`.
- **Per-head hybrid metrics**: `per_head_hybrid_metrics` and
  `per_head_hybrid_metrics_all_layers` (energy, smoothness, spectral entropy,
  HFER on each head's full eigenbasis; family "hybrid"), with `heads=` and
  `project_dim`.
- *(fix)* `per_head_metrics` with `normalization="rw"` symmetrized
  `I - D^{-1}W` before the eigensolve and so reported the spectrum of a
  different matrix (λ_max 2.42 against the true 1.58 on a sink-dominated
  graph). It now reports the rw operator's spectrum. Results with the
  default `sym` normalization are unchanged.
- *(fix)* One HFER mode-count rule everywhere: `num_high_modes(n, r) =
  n - int((1 - r) n)`, the rule `compute_hfer` has always used (it differs
  from `ceil(r n)` only by floating-point rounding, never at the default
  0.1 below n = 5000). `compute_hfer`, `top_band_hfer` and the per-head
  hybrid path all call it, and a `cutoff_ratio` outside [0, 1] now raises
  `ValueError`; `compute_hfer` used to slice with a negative index and return
  a wrong value.
- *(fix)* Every scalar metric of `SpectralDiagnostics` is a plain Python
  `float` (`fiedler_value`, `spectral_radius` and `smoothness_index` were
  numpy scalars), so the fields serialise with `json` directly.
- *(fix)* `analyze_layer` accepts a one-token graph (its `squeeze()` turned a
  1 x 1 Laplacian into a 0-d array).
- *(fix)* `band_energies` raises `ValueError` for a non-symmetric (`rw`)
  operator instead of symmetrizing it, and handles an all-zero Laplacian.
- *(performance)* The Dirichlet energy and the smoothness index are
  trace-only sums, O(S²d), accumulated in float64; they no longer form the
  d x d matrix `X^T L X` (O(S d²) time, d² memory) or evaluate the energy
  twice. `analyze_layer` projects the signals onto the eigenbasis once
  instead of three times (`compute_spectral_masses`). Energy and smoothness
  move only by float32 rounding (at most 1.6e-7 relative in the
  reproduction check). Under each version's defaults, `analyze_layer` is
  2.4x to 25x faster than in 0.3.0 at S = 128 to 2048, and per-head hybrid
  metrics 6.0x to 67x faster than a 0.3.0 loop over heads
  (`benchmarks/end_to_end.py`).
- Benchmarks: `benchmarks/fast_vs_dense.py`, `end_to_end.py` and
  `reproduce_v030.py`, with results in `benchmarks/results.md` and
  `benchmarks/reproduce_v030.md`.

## v0.3.0

First release since 0.2.2. Three of these changes alter results
for existing callers; they are marked *(behaviour change)*.

- **Per-head spectral diagnostics** (`per_head.py`): `per_head_metrics`,
  `per_head_metrics_all_layers`, `stack_feature_vector`,
  `layer_delta_features`. One Laplacian per head, five eigenvalue-only
  metrics from a single batched eigendecomposition, optional token-span
  restriction, and probe-ready feature vectors plus cross-layer dynamics.
- **Metric provenance** (`METRIC_FAMILIES`, `SpectralDiagnostics.families`):
  every metric declares whether it is computed from attention weights alone
  ("attention") or from residual-stream states projected onto the attention
  eigenbasis ("hybrid"). The energy / smoothness / entropy / HFER metrics are
  hybrid and must not be reported as attention-only.
- **New attention-only metrics**: `eig_energy`, `eig_spectral_entropy`,
  `eig_hfer` — eigenvalue-only counterparts of the hybrid trio.
- *(behaviour change)* Default Laplacian normalization `rw` → `sym`: spectrum
  in [0, 2] independent of sequence length, and always the deterministic
  symmetric solver path. Pass `normalization="rw"` to reproduce pre-0.3.0
  numbers.
- *(behaviour change)* `get_fiedler_value`: exact dense solve for graphs up to
  512 nodes, seeded and null-deflated Lanczos above. Removes an unseeded
  random Lanczos start (non-deterministic results) and a `>1e-6` eigenvalue
  filter that discarded exactly the near-zero λ₂ regime the diagnostic is
  meant to detect.
- *(behaviour change)* Non-finite attention tensors now raise
  `NonFiniteAttentionError` instead of propagating NaNs.
- Documentation rewritten around the metric-family contract; informal
  terminology removed from CLI output and the demo notebook.

## v0.2.x

Directed topology metrics, spectral velocity, subgraph
isolation, GPU Lanczos, sparse solver optimizations.
