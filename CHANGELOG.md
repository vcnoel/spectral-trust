# Changelog

## v0.4.1

A per-column metric, parallel layers for the per-head functions, an
explicit device for them, and a corrected tolerance statement. No result
changes for existing calls.

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
