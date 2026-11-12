# TransformerLens and nnsight

TransformerLens and nnsight give access to a model's internals and let you
intervene on them. spectral-trust measures the attention graphs of a run you
already have: the adapters take the other library's tensors, with no second
forward pass, and return the same `SpectralCache` as `run_with_spectra`.

```bash
pip install "spectral-trust[tl,nnsight]"
```

## TransformerLens

```python
from transformer_lens.model_bridge import TransformerBridge
import spectral_trust as st

model = TransformerBridge.boot_transformers("gpt2")
tokens = model.to_tokens("The cat sat on the mat because it was warm.")
_, tl_cache = model.run_with_cache(tokens)
cache = st.from_transformer_lens(tl_cache, model, tokens)
```

The adapter reads `blocks.{l}.attn.hook_pattern` for every layer and, when
present, `blocks.{l}.hook_resid_post` as the signal of the hybrid metrics.
The last layer is paired with `ln_final.hook_out` when the cache holds it,
which is what Hugging Face returns as its last hidden state; pass
`final_norm=False` to pair it with its `hook_resid_post`.
`cache.meta["signal_source"]` records which tensor each layer used.

To keep the cache small, save the patterns only; the adapter then computes
the attention metrics:

```python
_, tl_cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_pattern"))
cache = st.from_transformer_lens(tl_cache, model, tokens)
```

With TransformerLens 2, `HookedTransformer` caches work the same way. Its
default weight processing (`center_writing_weights`) removes the mean of the
residual stream; load with `HookedTransformer.from_pretrained_no_processing`
for hybrid metrics equal to a Hugging Face run. The attention patterns do not
depend on it.

## nnsight

The model must run eager attention, so that its attention modules return
their weights:

```python
from nnsight import LanguageModel
import spectral_trust as st

prompt = "The cat sat on the mat because it was warm."
lm = LanguageModel("gpt2", attn_implementation="eager", dispatch=True)
attn, resid = [], []
with lm.trace(prompt):
    for block in lm.transformer.h:
        attn.append(block.attn.output[1].save())    # attention weights [1, H, S, S]
        resid.append(block.output.save())           # block output [1, S, d]
cache = st.from_nnsight(attn, resid, model=lm, tokens=prompt)
```

Modules must be read in execution order inside a trace, hence one loop over
the blocks. The module paths are those of the Hugging Face model
(`lm.model.layers[i].self_attn` for Llama, Qwen and Mistral). In nnsight 0.8,
`TransformersModel("gpt2", task="text-generation", ...)` works the same way.

The saved tensors can come after any intervention in the same trace, which
is the point: patch, ablate or steer with nnsight, then measure what it did to
the attention graphs. The hidden states are optional; without them
`from_nnsight` computes the attention metrics only. To pair the last layer
with the final norm's output, as Hugging Face and `run_with_spectra` do, also
save `lm.transformer.ln_f.output` and pass it as `final_norm=`.

The model's own output works too:

```python
with lm.trace(prompt, output_attentions=True, output_hidden_states=True):
    out = lm.output.save()
cache = st.from_nnsight(out.attentions, out.hidden_states, model=lm, tokens=prompt)
```

## Hugging Face outputs

```python
out = hf_model(**inputs, output_attentions=True, output_hidden_states=True)
cache = st.from_hf_outputs(out, hf_model.config, inputs["input_ids"], tokenizer=tokenizer)
```

`model_config` locates the attention layers of hybrid models; a right-padded
batch is cropped per item with `attention_mask=`.

## How close are they?

`tests/test_interop.py` runs gpt2 through each adapter and through
`run_with_spectra` and compares every value:

- nnsight's saved tensors are Hugging Face's, and every value agrees to
  float32 rounding.
- `TransformerBridge` computes attention itself: its patterns differ from
  Hugging Face's by about 1e-6 and its residual stream by about 2e-7
  relative. The metrics agree to 1e-4 relative, the per-mode spectral masses
  to 1e-3.
- `connectivity` is a threshold on the smallest eigenvalue, so it can flip
  where that eigenvalue sits at the tolerance; the test skips those layers.
