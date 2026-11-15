# Baselines

A method that spectral-trust offers as a baseline is ported from its authors'
released code and checked numerically against that code, in a committed
test. It is not reimplemented from a paper's text. This page lists the state
of each baseline. Status as of 15 November 2026.

## Geometry of Reason (reference features)

`features.geometry_of_reason(attentions, hidden_states)` returns the five
per-layer features of Noël (arXiv 2601.00791): `fiedler_value`, `energy`,
`smoothness`, `entropy` and `hfer`. It computes them exactly as the paper's
released code did ([vcnoel/geometry-of-reason](https://github.com/vcnoel/geometry-of-reason),
`scripts/core/run_experiment.py`), which ran spectral-trust 0.1.2 with that
release's defaults.

- **Check.** On four stored cases (dense and ARPACK paths, the paper's text
  settings, weighted heads), the output is bit-identical to the original
  0.1.2 code, which is read from this repository's history
  (`tests/test_gor_reference.py`, fixture written by
  `tests/data/make_gor_reference.py`).
- **Settings of the run.** The run used the random-walk Laplacian
  `I − D⁻¹W`, symmetrised attention averaged over heads, ARPACK with 50
  modes above 50 tokens, and an HFER cutoff at the top 10% of modes. The
  0.1.2 solvers were symmetric, so the random-walk spectrum was not exact
  (see the known issue in the [changelog](changelog.md)). Above 50 tokens,
  ARPACK starts from a random vector, so the Fiedler value, entropy and
  HFER vary slightly from run to run; `arpack_seed` fixes the start vector.
- **The paper's text** describes the combinatorial Laplacian and a median
  cutoff instead: `geometry_of_reason(..., normalization="none",
  eigen_solver="dense", hfer_cutoff_ratio=0.5)`.

## LapEigvals

Binkowski, Janiak, Sawczyn, Gabrys and Kajdanowicz, EMNLP 2025
(arXiv 2502.17598).

- **Official code.**
  [graphml-lab-pwr/lapeigvals](https://github.com/graphml-lab-pwr/lapeigvals)
  (commit `74f885c`).
- **Licence.** None declared: no licence file, and none in the README or
  the package metadata.
- **Status.** Without a licence the code cannot be ported, so spectral-trust
  ships no LapEigvals feature and no numerical check.
- **What it computes.** Per layer and head, the diagonal of `L = D − A` with
  `D_jj = (Σ_i A_ij)/(T − j)`, the mean attention that token j receives
  (`hallucinations/features/attention_weights.py`). `L` is lower-triangular,
  so this diagonal is its spectrum. The values are sorted in decreasing
  order and the top k (5 to 100) are fed to a probe: PCA to 512 dimensions,
  then logistic regression on a stratified 80/20 split.

## SinkProbe

Binkowski, Adamczewski and Kajdanowicz, ICML 2026 (arXiv 2604.10697).

- **Official code.**
  [graphml-lab-pwr/sink-probe](https://github.com/graphml-lab-pwr/sink-probe)
  (commit `8108351`).
- **Licence.** None declared.
- **Status.** Not ported, and no numerical check.
- **What it computes.** Per layer and head, the sink score of each token j
  is `(Σ_i A_ij)/(T − j)` (`hallucinations/features/sink_scores.py`). The
  scores are sorted, and the top k (1 to 100) are fed to a logistic
  regression under stratified 5-fold cross-validation. The released feature
  code applies no value-norm weighting; the paper's value-norm result is an
  analysis of the probe.
- **Relation to LapEigvals.** The SinkProbe score of token j is the
  LapEigvals value plus `A_jj`.

spectral-trust's `features.sink_mass` is the mean attention to one token from
the queries after it, without that token's own row. `value_weighted_sink`
multiplies it by `‖W_O^h v^h_sink‖`. Neither is a SinkProbe feature. To
compare with LapEigvals or SinkProbe, run their pipelines unmodified from
their repositories, on the same model outputs.
