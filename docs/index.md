# spectral-trust

![spectral-trust](assets/logo.svg#only-light){ height="56" }
![spectral-trust](assets/logo-dark.svg#only-dark){ height="56" }

**spectral-trust turns a transformer's attention into graphs over the tokens
and measures them: the Laplacian spectrum of every head, and the frequency
content of the residual stream on those graphs, from one forward pass.**

![Three lines of code and their output: an attention graph, a layers x heads heatmap and a Laplacian spectrum](assets/hero.png#only-light)
![Three lines of code and their output: an attention graph, a layers x heads heatmap and a Laplacian spectrum](assets/hero-dark.png#only-dark)

```python
import spectral_trust as st

model = st.SpectralModel.from_pretrained("Qwen/Qwen3-0.6B")
cache = model.run_with_spectra("The cat sat on the mat because it was warm.")
print(cache["blocks.10.attn.fiedler_value"])     # one value per head
print(cache["blocks.10.graph.hfer"])             # the head-averaged graph
cache.heatmap("fiedler_value").save("fiedler.html")
```

Every measurement is a function of one forward pass: no labels, no probes,
no training. The library reads attention from a Hugging Face model, a
TransformerLens cache or an nnsight trace, and returns a `SpectralCache`
keyed like TransformerLens' `ActivationCache`.

- [Getting started](getting-started.md): install, load a model, read the cache.
- [Concepts](concepts.md): graphs from attention, the two metric families,
  hybrid linear-attention models.
- Tutorials: [quickstart](tutorials/quickstart.md),
  [per-head analysis](tutorials/per-head.md),
  [TransformerLens and nnsight](tutorials/interop.md).
- [API reference](api.md) and [changelog](changelog.md).
