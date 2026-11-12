# Getting started

## Install

```bash
pip install spectral-trust
```

The core install depends on PyTorch, Transformers, NumPy, SciPy, pandas and
matplotlib. Optional extras:

| Extra | Adds |
|---|---|
| `spectral-trust[tl]` | TransformerLens, for `from_transformer_lens` |
| `spectral-trust[nnsight]` | nnsight, for `from_nnsight` |
| `spectral-trust[viz]` | a matplotlib recent enough for the heatmap fallback (the SVG views need nothing) |
| `spectral-trust[docs]` | mkdocs-material and mkdocstrings, to build this site |
| `spectral-trust[dev]` | pytest, build, twine and the notebook runner |

The adapters never import TransformerLens or nnsight themselves, so the core
install works without them.

## Load a model

```python
import spectral_trust as st

model = st.SpectralModel.from_pretrained("Qwen/Qwen3-0.6B")
```

`from_pretrained` takes a Hugging Face model id or path and loads the causal
LM in float32 with eager attention, the implementation that returns attention
weights. Keyword arguments go to `AutoModelForCausalLM.from_pretrained`
(`revision`, `local_files_only`, `device_map`, ...). An already-loaded model
works too: `st.SpectralModel.from_pretrained(hf_model, tokenizer=tok)`
switches it to eager attention if needed and changes nothing else.

The graph construction is set by a `GSPConfig`: `SpectralModel(..., config=
st.GSPConfig(normalization="rw"))`. The defaults are the symmetric
normalised Laplacian (spectrum in [0, 2] at any length), symmetric
symmetrisation and uniform head averaging.

## Run it

```python
text = "The cat sat on the mat because it was warm."
cache = model.run_with_spectra(text)
cache
# SpectralCache('Qwen/Qwen3-0.6B', 11 tokens, layers 0-27, 16 heads,
#               metrics: attn=5, resid=4, graph=10)
```

One forward pass, then for every layer: the five attention-only metrics of
each head's graph, the four hybrid metrics of the residual stream on each
head's graph, and every metric of the head-averaged graph.

## Read the cache

```python
cache["blocks.10.attn.fiedler_value"]     # [16]: one value per head
cache["blocks.10.graph.hfer"]             # float: the head-averaged graph
cache["attn.fiedler_value", 10]           # the same as the first line
cache.layer(10).head(3)                   # every measurement of one head
cache.metric("fiedler_value")             # [layers, heads]
cache.to_dataframe()                      # layer, head, metric, family, value
cache.tokens, cache.layers, cache.metric_names, cache.families
```

A list of texts runs as a batch: values gain a leading text dimension,
`cache[i]` is the cache of text `i`, and `to_dataframe()` has a `text_id`
column.

## Choose what to compute

```python
model.run_with_spectra(text, metrics=["attn"])              # attention only, per head
model.run_with_spectra(text, metrics=["fiedler_value", "graph.hfer"])
model.run_with_spectra(text, hybrid=False)                  # no hidden states read
model.run_with_spectra(text, layers=[-4, -3, -2, -1])       # the last four layers
model.run_with_spectra(text, heads=[(10, 3), (12, 7)])      # two heads
model.run_with_spectra(text, token_span=(-5, None))         # the last five tokens
```

A namespace that is not selected is not computed. Heads that are not
computed are NaN, so a `[n_heads]` array always indexes by head.

## Look at it

```python
cache.graph(10, head=3)          # tokens as nodes, edges = attention
cache.heatmap("fiedler_value")   # layers x heads
cache.spectrum(10)               # the head-averaged graph's spectrum
cache.heatmap("resid.hfer").save("hfer.html")
```

In Jupyter the views render inline; `save` writes a standalone HTML file
that opens offline. Hover a node, an edge, a cell or an eigenvalue for its
values.

## Save it

```python
path = cache.save("run.npz")
cache = st.SpectralCache.load(path)
```

One `.npz` file holds every array and the metadata (tokens, layers, heads,
configuration) as JSON; loading needs no pickle.
