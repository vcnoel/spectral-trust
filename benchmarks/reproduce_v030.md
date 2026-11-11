# Reproduction of 0.3.0 (7197315) by 0.4.0

37 seeded input sets (random, sink-dominated and causal attention, S in {16, 64, 200}, d in {32, 256}, two seeds, plus a three-layer per-head stack). Layer pipeline: normalization="sym", eigen_solver="dense", full graph and token subgraph. Per-head: attention-only metrics, default configuration, with and without token_span.

**Result: reproduced** - largest relative difference 1.64e-07 (tolerance 1e-06).

| value | values compared | bitwise identical | max relative difference |
|---|---|---|---|
| `deltas` | 8 | 8 | 0.00e+00 |
| `features` | 60 | 60 | 0.00e+00 |
| `layer.connectivity` | 36 | 36 | 0.00e+00 |
| `layer.eig_energy` | 36 | 36 | 0.00e+00 |
| `layer.eig_hfer` | 36 | 36 | 0.00e+00 |
| `layer.eig_spectral_entropy` | 36 | 36 | 0.00e+00 |
| `layer.eigenvalues` | 3360 | 3360 | 0.00e+00 |
| `layer.energy` | 36 | 0 | 1.36e-07 |
| `layer.fiedler_value` | 36 | 36 | 0.00e+00 |
| `layer.hfer` | 36 | 36 | 0.00e+00 |
| `layer.layer` | 36 | 36 | 0.00e+00 |
| `layer.smoothness_index` | 36 | 0 | 1.41e-07 |
| `layer.spectral_entropy` | 36 | 36 | 0.00e+00 |
| `layer.spectral_masses` | 3360 | 3360 | 0.00e+00 |
| `layer.spectral_radius` | 36 | 36 | 0.00e+00 |
| `per_head` | 720 | 720 | 0.00e+00 |
| `per_head_span` | 720 | 720 | 0.00e+00 |
| `subgraph.connectivity` | 36 | 36 | 0.00e+00 |
| `subgraph.eig_energy` | 36 | 36 | 0.00e+00 |
| `subgraph.eig_hfer` | 36 | 36 | 0.00e+00 |
| `subgraph.eig_spectral_entropy` | 36 | 36 | 0.00e+00 |
| `subgraph.eigenvalues` | 2520 | 2520 | 0.00e+00 |
| `subgraph.energy` | 36 | 0 | 1.51e-07 |
| `subgraph.fiedler_value` | 36 | 36 | 0.00e+00 |
| `subgraph.hfer` | 36 | 36 | 0.00e+00 |
| `subgraph.layer` | 36 | 36 | 0.00e+00 |
| `subgraph.smoothness_index` | 36 | 0 | 1.64e-07 |
| `subgraph.spectral_entropy` | 36 | 36 | 0.00e+00 |
| `subgraph.spectral_masses` | 2520 | 2520 | 0.00e+00 |
| `subgraph.spectral_radius` | 36 | 36 | 0.00e+00 |
| metric-family tags | 936 | 936 | 0 |

Relative difference: |old - new| / max(|old|, 1e-7 x the largest |old| of that value in the case), so entries that are zero up to rounding are measured on the scale of their vector.

Against a float64 evaluation of tr(X^T L X) on the same float32 inputs (maximum relative error over all cases):

| value | 0.3.0 | current |
|---|---|---|
| `energy` | 4.19e-07 | 4.86e-07 |
| `smoothness_index` | 3.72e-07 | 4.86e-07 |
