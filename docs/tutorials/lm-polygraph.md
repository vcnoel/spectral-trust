# LM-Polygraph bridge

[LM-Polygraph](https://github.com/IINemo/lm-polygraph) (Fadeeva et al.,
2023; MIT licence) is a benchmark of uncertainty estimators for language
models: datasets, generation, quality metrics and several dozen estimators,
all scored the same way. `spectral_trust.integrations.lm_polygraph` turns
any attention metric of this library into one of its estimators, so that a
spectral score can be compared with LM-Polygraph's baselines on its own
benchmark.

```bash
pip install "spectral_trust[polygraph]"      # lm-polygraph >= 0.7, Python >= 3.10
```

The core package never imports lm-polygraph; only
`spectral_trust.integrations.lm_polygraph` does, and it raises an
`ImportError` that names the extra when lm-polygraph is missing.

## How it plugs in

LM-Polygraph computes statistics once per batch with its stat calculators
and passes them to every estimator; an estimator names the statistics it
needs and returns one value per item, higher meaning more uncertain.
`SpectralTrustEstimator` asks for two of them:

- `forwardpass_attention_weights`: a forward pass over the prompt followed
  by the greedy generation, `[layers, heads, S, S]` per item (NaN-padded
  in a batch);
- `greedy_tokens`, to locate the generated tokens at the end.

For each item it runs `spectral_trust.compute_spectra` on the attention,
reduces the per-head values over heads, then over layers, and multiplies by
`sign`.

```python
from spectral_trust.integrations.lm_polygraph import SpectralTrustEstimator

est = SpectralTrustEstimator(
    "operator.participation_ratio",   # any attention metric, namespaced
    layers=None,                      # every layer; or an index or a list
    head_aggregation="mean",          # mean, median, max, min or sum
    layer_aggregation="mean",
    sign=-1,                          # a lower ratio scores as more uncertain
    span="all",                       # "all", "prompt" or "generated"
)
```

The metric can be any attention metric of the library:
`operator.participation_ratio`, `operator.contraction`,
`graph.fiedler_value`, `graph.forman_curvature_mean`, `attn.hfer`, and so
on, with the graph and operator options of a `GSPConfig` passed as
`config=`. Hybrid metrics (which read hidden states) are refused, since the
statistic holds attention only, and so are value-weighted edges
(`edge_weights`, `operator_edges`). `span="generated"` measures the subgraph
of the generated tokens, the analogue of `token_span`; an empty span scores
NaN, which LM-Polygraph leaves out of its metrics. `str(est)` names the
estimator in LM-Polygraph's tables after its arguments, or `name=`.
`est.score(attention, n_generated)` scores one item without LM-Polygraph's
manager.

The direction of a score is a choice to make before looking at the
results: LM-Polygraph reads higher as more uncertain, and `sign` says which
way the metric is expected to move.
