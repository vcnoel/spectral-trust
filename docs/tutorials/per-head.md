# Per-head analysis

Averaging the heads of a layer mixes heads with different roles into one
graph: a routing pattern carried by a few heads disappears into the
background of the others. Per-head graphs keep that structure.

## Every head, one call

```python
import numpy as np
import spectral_trust as st

model = st.SpectralModel.from_pretrained("Qwen/Qwen3-0.6B")
text = "The cat sat on the mat because it was warm."
cache = model.run_with_spectra(text, metrics=["attn"])
fiedler = cache.metric("fiedler_value")        # [28 layers, 16 heads]
cache.heatmap("fiedler_value")
```

All the heads of a text go through batched symmetric eigendecompositions,
several layers at a time, so the five attention metrics of every head cost
about as much as one. `cache["blocks.12.attn.eigenvalues"]` holds each
head's full spectrum.

## Which heads cut the sentence?

A small Fiedler value means a weak cut: the head's graph almost separates
into two groups of tokens. The Fiedler vector says which.

```python
layer, head = np.unravel_index(np.nanargmin(fiedler), fiedler.shape)
layer = cache.layers[layer]
print(layer, head, fiedler.min())
cache.graph(layer, head=int(head))
```

## A subset of heads

The heads to study often come from elsewhere: an attribution method, a
probe's weights, an ablation study. Pass them as `heads=`:

```python
scores = np.random.default_rng(0).random((28, 16))    # your scores here
top = st.heads_from_scores(scores, k=10)              # [(layer, head), ...]
sub = model.run_with_spectra(text, heads=top)
sub.heads                                             # {layer: [heads computed]}
```

`heads=` also takes head indices (the same heads in every layer) or a boolean
`[heads]` or `[layers, heads]` mask. The per-head namespaces compute only
those heads, the others are NaN, and the `graph` namespace still averages
every head.

## Hybrid metrics per head

`blocks.{l}.resid.*` measures the layer's residual stream on each head's
graph: the Dirichlet energy, the smoothness index, the spectral entropy and
the HFER, each on the head's full eigenbasis. They read the hidden states
(family `hybrid`); `hybrid=False` leaves them out.

```python
cache = model.run_with_spectra(text, metrics=["resid.hfer", "attn.fiedler_value"])
cache.heatmap("resid.hfer")
```

## Features for a probe

The flat layout of the DataFrame turns into probe features directly:

```python
df = cache.to_dataframe()
features = df.pivot_table(index=["layer", "head"], columns="metric", values="value")
```

For many texts, `run_with_spectra(list_of_texts).to_dataframe()` gives one
row per (text, layer, head, metric). The lower-level
`per_head_metrics_all_layers`, `stack_feature_vector` and
`layer_delta_features` work on attention tensors directly.
