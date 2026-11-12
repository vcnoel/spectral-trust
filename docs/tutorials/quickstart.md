# Quickstart

This tutorial follows
[`examples/quickstart.ipynb`](https://github.com/vcnoel/spectral-trust/blob/main/examples/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/vcnoel/spectral-trust/blob/main/examples/quickstart.ipynb)).
Everything runs on a CPU; the first run downloads `Qwen/Qwen3-0.6B` (about
1.5 GB).

## One text

```python
import spectral_trust as st

model = st.SpectralModel.from_pretrained("Qwen/Qwen3-0.6B")
text = "The cat sat on the mat because it was warm."
cache = model.run_with_spectra(text)
print(cache)
print(cache.tokens)
```

The cache holds, for each of the 28 layers, three groups of measurements:

- `blocks.{l}.attn.*`: five attention-only metrics of each head's graph;
- `blocks.{l}.resid.*`: four metrics of the residual stream on each head's
  graph;
- `blocks.{l}.graph.*`: every metric of the head-averaged graph, the one
  `GSPDiagnosticsFramework.analyze_text` builds.

```python
cache.layer(12)                    # the head-averaged metrics and per-head summaries
cache.layer(12).head(3)            # one head
cache.layer(12).head(3)["attn.hfer"], cache.layer(12).head(3)["resid.hfer"]
```

## A table

```python
df = cache.to_dataframe()
df[df.metric == "graph.fiedler_value"][["layer", "value"]]
```

`to_dataframe` returns one row per (layer, head, metric) with the metric's
family; `head` is missing for the head-averaged `graph` metrics.

## Views

```python
cache.graph(12)                    # the head-averaged graph of layer 12
cache.graph(12, head=3)            # one head
cache.heatmap("fiedler_value")     # layers x heads
cache.spectrum(12, head=3)
```

Each view is self-contained SVG with tooltips; `.save("view.html")` writes a
page that opens offline. `cache.heatmap(..., backend="matplotlib")` returns a
matplotlib figure instead.

## Several texts

```python
texts = ["The capital of France is Paris.", "The capital of France is Rome."]
batch = model.run_with_spectra(texts)
batch["blocks.12.graph.fiedler_value"]        # [2]
batch[1].layer(12).head(3)                    # text 1
batch.to_dataframe()                          # with a text_id column
```

Texts run as right-padded batches of `batch_size` (8 by default). In a causal
LM the padding changes neither the attention nor the hidden states of the
real tokens, and each text is cropped to its length before any measurement,
so the values equal one-text runs to float32 rounding.

## Save and reload

```python
path = batch.save("capitals.npz")
again = st.SpectralCache.load(path)
```
