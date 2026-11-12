# Changelog

## v0.4.0

Fast diagnostics, a per-head `rw` fix, and a dense default
solver. Three changes alter results for existing callers; they are marked
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
- *(behaviour change)* `connectivity` counts an eigenvalue as zero below
  `max(1e-6, sqrt(S) * eps * lambda_max)` instead of a fixed 1e-6, through
  `SpectralAnalyzer.zero_eigenvalue_tolerance` and `is_connected`. Float32
  noise on unnormalised (`"none"`) graphs crossed the fixed threshold and
  reported connected graphs as disconnected. Float64 input keeps the old
  threshold; float32 normalised graphs see it rise to at most about 1e-5.
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
