# Interventions

*Added in 0.5.0.*

`spectral_trust.interventions` edits a Hugging Face causal LM with PyTorch
hooks.
It covers the TransformerLens basics (activation patching, head ablation,
activation steering) and adds interventions in the graph-spectral domain:
filtering the residual stream along the token axis on the attention graph,
patching one band of graph frequencies, removing attention edges without
moving the attention sink's mass, and rescaling a weight matrix's singular
values.

Every edit lives inside a context manager, `st.intervene`, and is undone on
exit: hooks are removed, weights are restored bit for bit, and the attention
implementation is put back. Interventions on one model may be nested and
closed in any order.

```python
import torch
import spectral_trust as st

model = st.SpectralModel.from_pretrained("gpt2")
tok = model.tokenizer
ids = tok("When Mary and John went to the store, John gave a drink to",
          return_tensors="pt").input_ids

with st.intervene(model) as iv:
    iv.resid(6).graph_filter("lowpass", cutoff=0.5, strength=0.8)
    iv.attention(9, 6).ablate("zero")
    iv.weights(7).svd_scale(-0.05)
    edited = model(ids).logits
# the model is unchanged here
```

`st.intervene` takes a `SpectralModel` or a plain `transformers` model
(GPT-2, Llama, Qwen, Mistral, Gemma, GPT-NeoX and OPT layouts). Each call
such as `iv.resid(6).graph_filter(...)` returns an `Edit`; `edit.remove()`
undoes that one edit, and `iv.edits` lists the active ones.

Positions are absolute: position i is the i-th token of the sequence,
including tokens already in the KV cache, so position-dependent edits
(`positions=`, per-position vectors, resample ablation) give the same result
in cached generation as in a full recomputation. Negative positions count
from the end of the current sequence, which moves during generation, and
raise at a cached step.

## Graph-frequency steering

The residual stream after block l is a signal on the tokens: one vector per
position. Block l's attention, averaged over heads and symmetrised, is a
graph on the same tokens. The filter keeps or removes graph frequencies of
the signal along the token axis:

`X' = X + s (g(L) X − X)`,

where `L` is the graph Laplacian built exactly as the measurements build it
(`GraphConstructor`, normalisation `"sym"` by default, spectrum in
[0, 2]), `g` the band response and `s` the strength. Low eigenvalues are
directions that vary little across strongly connected tokens; a low-pass
filter smooths the residual stream along the attention graph, a high-pass
filter keeps what distinguishes a token from the tokens it is connected to.

```python
with st.intervene(model) as iv:
    iv.resid(6).graph_filter("lowpass", cutoff=0.5)             # keep eigenvalues < 0.5
    iv.resid(8).graph_filter("highpass", cutoff=1.2, strength=0.5)
    iv.resid(10).graph_filter("bandstop", band=(0.8, 1.0))
    logits = model(ids).logits
```

- `strength=0` is the identity, `1` applies the filter, values in between
  interpolate, negative values push away from the filtered signal
  (`"lowpass"` with `strength=-1` doubles the high frequencies).
- `graph="attention"` (default) uses the same layer's attention from the same
  forward pass. `graph="path"` uses the chain graph over positions, a purely
  positional baseline whose eigenvectors are cosines along the sequence.
  An adjacency tensor or a callable `(attention, x) -> adjacency` works too.
- The filter is a Jackson-damped Chebyshev polynomial of degree `order`
  (default 64) in `L`: no eigendecomposition, `order` products with `L`.
  It computes in float64 by default (on Apple MPS, which has no float64, on
  the CPU). `precision="float32"` keeps it on the GPU in float32, which is
  much faster there but noisier when one token's residual dwarfs the
  others', as GPT-2's first token does. `method="dense"` applies the ideal
  band indicator through an eigendecomposition, for checks.
  `filter_response` shows the polynomial:

```python
import numpy as np
from spectral_trust.interventions import filter_response

lam = np.linspace(0, 2, 9)
print(np.round(filter_response(lam, "lowpass", cutoff=0.5), 3))
```

- In GPT-2 and most LMs the first token's residual has a far larger norm
  than the others. Mixing it into other positions dominates any filter, so
  `token_span=(1, None)` leaves it out of the graph and unedited. A span's
  graph is built from the attention among its own tokens (sub-block first,
  then symmetrised and normalised); with the default symmetrisation and head
  averaging this is the sub-block of the full graph, as `run_with_spectra`
  measures it under `token_span`.

### Symmetric or causal

The symmetrised graph links each token to the tokens after it, so with
`causal=False` (the default) position i reads later positions. That is
right for analysing one forward pass, but it cannot run during generation
with a KV cache: past positions would have to change at every step, and a
cached step raises.

`causal=True` filters position i on the graph of its prefix `0..i` only.
The effective operator is lower triangular, each position's output is fixed
once its prefix is known, and at the last position it equals the symmetric
filter. Generating with a KV cache then gives the same tokens as recomputing
the sequence at every step:

```python
prompt = tok("The capital of France is", return_tensors="pt").input_ids
with st.intervene(model) as iv:
    iv.resid(4).graph_filter("lowpass", cutoff=0.3, strength=0.5,
                             causal=True, token_span=(1, None))
    out = model.model.generate(prompt, attention_mask=torch.ones_like(prompt),
                               max_new_tokens=10, do_sample=False,
                               pad_token_id=tok.eos_token_id)
print(tok.decode(out[0]))
```

The causal filter keeps the residual stream and attention of past tokens,
and follows the cache when beam search reorders it, so
`generate(num_beams=3)` with and without the cache agree too. Its
`token_span` bounds and `positions` must be absolute (non-negative). It costs
about `order * S^3 / 3` operations for S tokens (one polynomial per prefix),
against `order * S^2 * d` for the symmetric one; for a few hundred tokens
both take milliseconds.

### On plain tensors

`graph_filter` applies the same filter to tensors you already have:

```python
out = model.model(ids, output_attentions=True, output_hidden_states=True)
x = out.hidden_states[7]                  # residual stream after block 6
smooth = st.graph_filter(x, out.attentions[6], kind="lowpass", cutoff=0.5)
```

## Head ablation

`iv.attention(layer, head)` selects heads (an index, a list, or None for
all). `ablate` replaces their outputs before the output projection
(TransformerLens `hook_z`), so a head's whole contribution is replaced:

```python
with st.intervene(model) as iv:
    iv.attention(9, [6, 9]).ablate("zero")
    iv.attention(10, 0).ablate("mean", reference=ids)       # mean over the reference
    zeroed = model(ids).logits
```

- `"zero"` sets the outputs to zero.
- `"mean"` uses the mean over a reference, which it requires: every token of
  the reference texts (texts of different lengths are run one at a time, so
  no padding enters the mean), or per position with `per_position=True`.
- `"resample"` uses the head's output on a reference item, at the same
  positions, so the reference must be at least as long as the input. The
  item is drawn once per input (seeded) and kept, and the beams of one input
  share it, so repeated calls, cached generation and beam search agree.

## Removing attention edges

`remove_edges` zeroes chosen (query, key) entries of the selected heads'
attention and renormalises each edited row:

```python
S = ids.shape[1]
key_mask = torch.zeros(S, S, dtype=torch.bool)
key_mask[-1, 1:6] = True                 # the last query stops reading keys 1..5
with st.intervene(model) as iv:
    iv.attention(9, 9).remove_edges(key_mask)                 # keep_sink=True
    cut = model(ids).logits
```

With `keep_sink=True` (the default) the row is renormalised over its
remaining non-sink keys while the first token's entry keeps its value. Most
attention rows put much of their mass on the first token. Plain
renormalisation (`keep_sink=False`) lets that sink mass move: on GPT-2 small
IOI, removing first-token edges at the answer position with plain
renormalisation cost about six times more logit difference per unit of
removed mass than removing random content edges, because the released mass
multiplied each row's content attention. Holding the sink fixed keeps an
edge ablation an edge ablation.

`edit_pattern(fn)` gives the general form: `fn(weights, query_positions)`
returns new attention probabilities, and the head's output is rebuilt as
`weights @ values` (checked once against the model's own output). Pattern
edits need the model's eager attention, which the intervention switches on
and restores; they rely on `transformers`' module-level
`eager_attention_forward` (transformers 4.48 or later).

## Weight-spectrum steering

`svd_scale` rescales the singular values of a weight matrix
`W = U diag(σ) Vᵀ` by `σᵢ' = σᵢ (1 + α (σᵢ − μ) / s)`, with `mu` and `s` the mean and standard deviation of the singular values.
`alpha < 0` compresses the dominant directions, `alpha > 0` amplifies them.
This is the spectral-steering rescaling, computed as its reference code does
(float32 thin SVD on the CPU); the rebuilt matrix equals the reference's bit
for bit. The reference then swaps in a bias-free `nn.Linear`, which is the
same model only for bias-free projections (the Llama-family
`mlp.down_proj`); `svd_scale` edits the weight in place, keeps any bias, and
also handles GPT-2's `Conv1D`. `k` limits the edit to the top `k` singular
values. Edits of one matrix stack: removing any of them rebuilds the matrix
from its original value with the others re-applied.

```python
snr = st.weight_snr(model)               # sigma_1 / median(sigma), every layer
layer = int(snr.argmax())
with st.intervene(model) as iv:
    w = iv.weights(layer)                # "mlp.down_proj", or GPT-2's mlp.c_proj
    print(w.path, round(w.snr(), 2))
    w.svd_scale(-0.05)
    print(round(w.snr(), 2))
```

The factor `1 + α zᵢ` (`zᵢ` the singular value's z-score) must stay
positive: past `α = −1 / z_max` the largest singular value changes sign and
grows instead of shrinking, and `svd_scale` warns. Outlier-heavy matrices
have a large `z_max`, so the safe range of `α` is narrow.

`weight_snr` uses the full singular spectrum in float32. Any 2-D weight of a
block can be named by its path, for example `iv.weights(5, "attn.c_proj")`.
`st.weight_snr(W, k=32)` on a weight tensor gives the fast top-k estimate of
the unreleased 0.2.3 branch, and `st.weight_svd_full(W)` its full SVD, the
two functions spectral-steering's scripts import.

## Activation patching

`st.patch` runs the corrupt input, replaces one activation by its value on
the clean input, and returns the logits (or `metric(logits)`), as a
TransformerLens patching hook does. Clean and corrupt inputs must have the
same length; positions align, as in IOI.

```python
clean = tok("When Mary and John went to the store, John gave a drink to",
            return_tensors="pt").input_ids
corrupt = tok("When Alice and Bob went to the store, Carl gave a drink to",
              return_tensors="pt").input_ids
io, s = tok(" Mary").input_ids[0], tok(" John").input_ids[0]

def logit_diff(logits):
    return logits[:, -1, io] - logits[:, -1, s]

st.patch(model, clean, corrupt, site="resid_pre", layer=8, metric=logit_diff)
st.patch(model, clean, corrupt, site="head", layer=9, head=9, positions=[-1],
         metric=logit_diff)
```

Sites follow TransformerLens: `resid_pre`, `resid_post`, `attn_out`,
`mlp_out`, and `z` (or `head`, with `head=`) for per-head outputs.

### Frequency-resolved patching

With `band=`, only part of the activation's graph spectrum is taken from the
clean run. Both residuals are projected onto an orthonormal basis of the
token positions (`basis=`), the band's coefficients come from the clean run
and the rest from the corrupt run:
`X' = X_corrupt + U_b U_bᵀ (X_clean − X_corrupt)`, with `U_b` the band's
basis vectors.

- `basis="clean_attention"` or `"corrupt_attention"`: eigenvectors of that
  run's attention Laplacian at the patched layer;
- `"path"`: the chain graph, a positional control;
- `"random"`: a seeded Haar-random basis, the null for a subspace of a given
  size.

`band=(lo, hi)` takes integer mode indices in ascending eigenvalue order, or
float eigenvalues; `n_modes` caps the band's size. The full band is plain
patching and the empty band leaves the corrupt run unchanged.

`Patcher` runs the clean and corrupt inputs once and batches many patches:

```python
from spectral_trust.interventions import PatchSpec, equal_count_bands

p = st.Patcher(model, clean, corrupt, forward_kwargs={"logits_to_keep": 1})
S = clean.shape[1]
bands = equal_count_bands(S, 4)
specs = [PatchSpec("resid_pre", layer, band=b, basis=basis)
         for layer in (6, 8) for basis in ("clean_attention", "path") for b in bands]
recovered = p.patch_many(specs, metric=logit_diff)       # [len(specs), batch]
full = logit_diff(p.clean_logits) - logit_diff(p.corrupt_logits)
fraction = (recovered[:, 0] - logit_diff(p.corrupt_logits)) / full
```

Every pass of a `Patcher` uses eager attention, so patched logits differ from
the unpatched ones only through the patch. Activations are captured lazily,
the first time a patch needs them (attention patterns only for attention
bases). They belong to the model as it was then: if its weights, hooks or
attention implementation change (a weight edit, an intervention opened or
closed), the next call raises until `p.refresh()` recaptures. Writes through
`param.data` are not detected.

## TransformerLens correspondence

| TransformerLens | spectral-trust |
|---|---|
| `run_with_hooks` with a hook on `hook_resid_post` | `iv.hook("resid_post", l, fn)` |
| adding a steering vector to the residual stream | `iv.resid(l).add(vec, scale)` |
| zero, mean or resample ablation on `hook_z` | `iv.attention(l, h).ablate(mode)` |
| editing `hook_pattern` | `iv.attention(l, h).edit_pattern(fn)` |
| activation patching with `run_with_cache` and a patching hook | `st.patch`, `st.Patcher` |

## Limits

- Attention-graph filters and pattern edits need eager attention; the
  intervention switches to it and restores the original on exit.
- Padding: pass a 2-D `attention_mask` as a keyword; padded positions are
  left out of the graphs and the sink of a left-padded row is its first real
  token. Patching inputs must be unpadded and of equal length.
- The causal filter follows beam search but raises when the cache is cropped
  or reordered otherwise (assisted decoding, for example).
- Pattern edits (`remove_edges`, `edit_pattern`) need transformers 4.48 or
  later.
- Quantised (bitsandbytes) weights are not supported by `svd_scale`.
