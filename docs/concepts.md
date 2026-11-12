# Concepts

## Graphs from attention

The attention weights of one head form a matrix `A` whose row `i` says how
much query token `i` attends to each earlier token. spectral-trust reads it
as a weighted graph on the tokens:

1. **Symmetrise.** `W = (A + Aᵀ) / 2` by default (`symmetrization=
   "symmetric"`); `"row_norm"` and `"col_norm"` normalise first.
2. **Average heads, or not.** The `graph` namespace averages the symmetrised
   heads of a layer (`head_aggregation="uniform"`, or `"attention_weighted"`
   by attention mass). The `attn` and `resid` namespaces keep one graph per
   head.
3. **Build the Laplacian.** The symmetric normalised Laplacian
   `L = I − D^{-1/2} W D^{-1/2}` is the default: its spectrum lies in
   [0, 2] whatever the number of tokens, so values compare across prompts.
   `normalization="rw"` gives `I − D^{-1} W` (the same eigenvalues, other
   eigenvectors), `"none"` the combinatorial `D − W`.
4. **Decompose.** The eigenvalues `0 = λ₁ ≤ λ₂ ≤ … ≤ λ_S` describe how the
   graph routes information. The Fiedler value `λ₂` is zero when the graph
   falls into pieces that do not attend to each other, and small when a few
   weak edges hold it together. Its eigenvector, the Fiedler vector, places
   the tokens along the weakest cut, which is what the node colours of
   `cache.graph` show.

The eigenvectors also form a frequency basis on the tokens. Projecting the
residual stream onto them measures how much of the hidden state changes
sharply between tokens that attend to each other: the high-frequency energy
ratio (HFER) is the share of the signal energy in the top 10% of modes.

`token_span=(start, end)` takes the induced subgraph of a span before the
Laplacian is built, so the diagnostic is local to that span (a generated
answer, a tool call) rather than to the whole sequence.

## The metric families

Every metric reads specific tensors, and the difference matters for what you
can claim and for what access you need. Each metric is tagged with its family
(`cache.families`, `spectral_trust.METRIC_FAMILIES`):

| Family | Reads | Metrics |
|---|---|---|
| attention | attention weights only | `attn.*`; `graph.fiedler_value`, `spectral_radius`, `connectivity`, `eig_energy`, `eig_spectral_entropy`, `eig_hfer` |
| hybrid | attention weights and the residual stream | `resid.*`; `graph.energy`, `smoothness_index`, `spectral_entropy`, `hfer` |

Attention metrics are available wherever a serving stack exposes attention
maps. Hybrid metrics project residual-stream states onto the attention
eigenbasis: they reflect the content of the residual stream as much as the
graph, and results with them are fairly compared with hidden-state probes at
equal access, not with attention-only baselines. `hybrid=False` computes the
attention family only and never reads the hidden states.

Two names recur with different meanings. `attn.hfer` is the share of the
eigenvalue mass in the upper half of a head's spectrum (attention only), while
`resid.hfer` and `graph.hfer` are the share of the residual stream's energy in
the top modes (hybrid). `attn.spectral_entropy_norm` is the entropy of the
eigenvalues, `resid.spectral_entropy` that of the signal energy per mode.

The signal paired with layer `l` is `hidden_states[l + 1]` of the Hugging Face
output, the output of that layer (for the last layer, Hugging Face returns it
after the final norm). The TransformerLens adapter pairs each layer with
`blocks.{l}.hook_resid_post`, and the last one with `ln_final.hook_out` when
the cache holds it, so the two agree.

## Hybrid models

Some models mix full-attention layers with linear-attention or state-space
layers, for example Qwen3.5 (three linear-attention layers for each
full-attention layer) or Qwen3-Next. Only the full-attention layers return attention weights,
and Hugging Face returns those tensors as a plain list: Qwen3.5-0.8B gives 6
tensors for 24 layers, without saying which layers they come from.

spectral-trust reads the layer of each tensor from `layer_types` in the model
config (or its `text_config`), reports the true index (3, 7, 11, … for
Qwen3.5-0.8B) and pairs each tensor with the hidden state of its own layer.
`cache.layers` lists the layers that have attention; asking for a
linear-attention layer raises an error that lists them. When a config does not
say, pass `GSPConfig(attention_layer_map=[...])`.
`spectral_trust.attention_layer_indices` exposes the same mapping for other
pipelines.
