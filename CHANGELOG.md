# Changelog

All notable changes to `spectral-trust` are documented here.

## [0.2.2] — 2026-07-16

Correctness release. No API or default changes: `normalization` still defaults to
`"rw"`, `eigen_solver` to `"sparse"`. Existing code runs unchanged.

**If you are reproducing published results, pin `spectral-trust==0.2.1`.** The
random-walk fix below changes `rw` diagnostics (it replaces wrong numbers with
right ones), so figures regenerated under 0.2.2 with `normalization="rw"` will
differ from those produced with 0.2.x.

### Fixed

- **Random-walk Laplacian was not actually diagonalized.**
  `SpectralAnalyzer.compute_eigendecomposition` always used a *symmetric* solver
  (`eigh`/`eigsh`), which reads only the lower triangle of its input. With the
  default `normalization="rw"` the operator `L_rw = I - D⁻¹W` is **not**
  symmetric, so the library silently returned the eigendecomposition of a
  symmetrized surrogate. On a weighted ring the relative residual
  `‖L U − U Λ‖ / ‖L‖` was **1.3e-01**; it is now **< 1e-6**.
  The solver now dispatches on the operator: symmetric Laplacians (`"sym"`,
  `"none"`) keep the `eigh`/`eigsh` path, while non-symmetric ones use a general
  solver (`scipy.linalg.eig`). `L_rw` is similar to the symmetric normalized
  Laplacian (`L_rw = D^{-1/2} L_sym D^{1/2}`), so its eigenvalues are real and now
  match `"sym"` exactly; its eigenvectors exist but are **not orthonormal**
  (Parseval does not hold for the `rw` GFT — use `"sym"` if you need an orthogonal
  transform).

- **`torch_dtype` deprecation / bfloat16 loading.** `from_pretrained` renamed
  `torch_dtype=` to `dtype=` in Transformers 4.56. The keyword is now selected by
  the installed version, so bf16/fp16 loads cleanly on Transformers 5.x without a
  deprecation warning (and still works on older versions).

### Added

- **Truncated-eigenbasis warning.** With `eigen_solver="sparse"` and `N > 50`,
  only `num_eigenvalues` (default 50) modes are computed, so `hfer`,
  `spectral_entropy` and `spectral_masses` are normalized over the retained modes
  only and are not comparable to a full-spectrum computation. This now warns and
  points to `eigen_solver="dense"`. (Silently truncating made energy fractions on
  large graphs meaningless.)

- `SpectralAnalyzer._is_symmetric` helper, and `tests/test_operator_correctness.py`
  covering: rw diagonalization residual, rw/sym eigenvalue agreement (similarity),
  orthonormality and `[0,2]` support of the symmetric path, and the star-graph
  golden spectrum `{0, 1, …, 1, 2}`.

### Notes

- Recommended configuration for GSP work that relies on an orthogonal GFT
  (Parseval, bandlimited sampling): `normalization="sym"`, `eigen_solver="dense"`.
- Changing the defaults to `sym`/`dense` is deferred to a future minor release, as
  it would alter results for existing users without a code change.

## [0.2.1]

- Directed metrics (`DirectedTopologist`), spectral velocity, subgraph Laplacians.
