# Directed and value-weighted graphs

The layer graph of the rest of the library is built from attention weights
and symmetrised. Two things are lost on the way. An attention weight says
how much query i reads key j, not how much key j changes query i's output,
which also depends on the vector that j's value carries. And causal
attention has a direction (token i reads tokens j ≤ i, and information
flows from j to i), which `(A + Aᵀ) / 2` averages away. Two options of
`GSPConfig` keep them: `edge_weights="value"` and
`graph_direction="directed"`.

## Value-weighted edges

The output of an attention module at token i splits exactly into one
vector per key token, plus the output bias:

```text
y_i = Σ_j c_ij + b_O,      c_ij = Σ_h a^h_ij · W_O^h (g^h_i ⊙ v^h_j)
```

with `v^h_j` the value of token j read by head h, `W_O^h` the columns of
the output projection that read head h, and `g_i` an output gate (1 without
one). The value-weighted edge from j to i is `w_ij = ‖c_ij‖`, the norm of
the sum over heads (Kobayashi et al., 2020). `per_head=True` gives instead
`a^h_ij ‖W_O^h (g^h_i ⊙ v^h_j)‖` for each head; their sum over heads
bounds `w_ij` from above and differs from it where heads cancel.

```python
import spectral_trust as st

model = st.SpectralModel.from_pretrained("gpt2")
text = "When Mary and John went to the store, John gave a drink to"

vals = st.capture_attention_values(model, text)     # every layer, one forward pass
for layer in (0, 5, 9):
    av = vals[layer]
    A = av.weights[0].mean(0)                       # head-averaged attention [S, S]
    W = av.edge_weights()[0]                        # value-weighted edges ||c_ij||
    W = W / W.sum(-1, keepdim=True)                 # rows on the scale of attention
    print(f"layer {layer}: first token gets {A[1:, 0].mean():.3f} of a row (attention), "
          f"{W[1:, 0].mean():.3f} (value-weighted)")
```

```text
layer 0: first token gets 0.235 of a row (attention), 0.186 (value-weighted)
layer 5: first token gets 0.747 of a row (attention), 0.126 (value-weighted)
layer 9: first token gets 0.792 of a row (attention), 0.082 (value-weighted)
```

From layer 5 on, the first token (the attention sink) takes three quarters
of every row of attention, and a tenth of every row once the edges are
weighted by what they carry: the sink's values are small after the output
projection.

```python
import torch

av = vals[9]
W_O = av.out_weight.view(-1, av.n_heads, av.head_dim)          # [d_model, H, d_head]
u = torch.einsum("hkx,dhx->hkd", av.values[0], W_O)            # W_O^h v^h_j
norms = u.norm(dim=-1).mean(0)                                 # mean over heads
print(f"{norms[0]:.1f} against {norms[1:].mean():.1f}")       # 2.4 against 28.1
```

`st.value_weighted_attention(model, text)` returns the edges of every
layer directly, `{layer: [batch, S, S]}`. `AttentionValues.contributions()`
gives the vectors `c_ij` themselves and `reconstruct()` their sum plus the
bias, which equals the module's output (the tests check it to a relative
error of 1e-4 in float32).

What the decomposition covers:

| layout | how it is handled |
|---|---|
| grouped-query attention | head h reads value group `h // (H / H_kv)`, as `repeat_kv` does |
| fused projections (GPT-2 `c_attn`, Phi-3 `qkv_proj`, GPT-NeoX) | values are taken where the attention reads them, inside `eager_attention_forward` |
| value bias | part of `v_j`, as in Kobayashi et al.'s `f(x) = (x W_V + b_V) W_O` |
| output bias | belongs to no token: left out of every `c_ij` and every edge |
| output gate (Qwen3.5, Qwen3-Next; a separate `gate_proj`) | `sigmoid` of the gate projection, applied per query, checked against the input of `o_proj` |
| hybrid models (Qwen3.5) | only the full-attention layers have a graph; other layers raise |
| post-attention norm (Gemma-2) | outside the attention module, not part of the decomposition |

A layout the capture does not recognise (anything between the attention and
its output projection other than a known gate) raises instead of giving
wrong edges. The attention runs in the eager implementation while values
are captured, as for every attention-weight measurement of the library.

## Directed graphs

With `graph_direction="directed"` the heads are averaged without
symmetrisation and each row is divided by its sum: the causal
row-normalised operator `P = D_out⁻¹ A` (`st.directed_operator`). For
attention, `P` is the head-averaged attention itself.

### Transport

`(P x)_i` is what token i receives in one step of attention transport: the
average of the signal over the tokens it reads. The directed total
variation

```text
TV(x) = ‖x − P x‖² / ‖x‖²
```

measures how much that step changes the signal (`st.directed_total_variation`,
with `per_column=True` for one value per feature, and
`st.per_column_metrics(..., operator=P)` for a fifth column next to HFER).
It is zero for a signal that transport leaves unchanged, a constant for
instance. It is the quadratic form of Sandryhaila and Moura (2014) with `P`
as the shift; a row-stochastic `P` has spectral radius 1, so it needs no
further normalisation.

A Laplacian energy cannot do this. For a real signal,
`xᵀ L x = xᵀ ((L + Lᵀ) / 2) x`: every quadratic Laplacian energy reads only
the symmetric part of the matrix, and gives the same value for a graph and
its reverse. Dahlem, Maniloff and Misiura (2026) prove the general limit:
every transpose-invariant spectral diagnostic of an attention operator is
orientation-blind. `TV` is the quadratic form of `(I − P)ᵀ (I − P)`, which
changes when the edges are reversed.

### The magnetic Laplacian

The magnetic Laplacian keeps the direction of each edge in the phase of a
Hermitian matrix (Fanuel et al., 2017; Furutani et al., 2019; Zhang et
al., 2021):

```text
A_s = (A + Aᵀ) / 2,    Θ_ij = 2π q (A_ij − A_ji),
L = D_s − A_s ⊙ exp(iΘ),    L_N = D_s^(−1/2) L D_s^(−1/2)
```

`st.magnetic_laplacian` builds it (normalised by default, in complex128),
`st.magnetic_spectrum` diagonalises it with a Hermitian solver (real
eigenvalues; those of `L_N` lie in [0, 2]), `st.magnetic_hfer` gives the
energy fractions `|Uᴴ x|²` of a signal and `st.magnetic_phase` the phase of
the lowest eigenvector. At `q = 0` the phases vanish and `L` is the
symmetric Laplacian of `A_s`.

The default charge is `q = 1/4`. The rows of `P` lie in [0, 1], so
`A_ij − A_ji` lies in [−1, 1] and the phase in [−2πq, 2πq]. At `q = 1/4`
the phase stays within [−π/2, π/2], where the real part of every entry
stays non-negative and the map from asymmetry to phase stays one to one,
and a one-way edge of weight 1 gets the phase π/2 (`+i`, against `−i` for
its reverse), the directed extreme of MagNet's range `0 ≤ q ≤ 1/4`. At
`q = 1/2` such an edge would get the phase π in both directions. Attention
weights are mostly far below 1, so most phases are small; the strong edges
carry most of them.

**The magnetic HFER of a real signal cannot tell a graph from its
transpose.** Reversing every edge conjugates the magnetic Laplacian, its
eigenvectors become `conj(U)`, and for a real `x`,
`|conj(u)ᴴ x| = |uᴴ x|`. The phases change the energies of a real signal
through `|A_ij − A_ji|`, not through the sign. A complex signal can tell
the two apart. For the direction of a real signal (hidden states, features),
use the directed total variation.

The phase of the lowest eigenvector is a potential of the edge asymmetries:
on a chain it grows by `Θ_ij` from each token to the next one along the
edges; on attention graphs it ranks tokens by the net asymmetry of their
strongest edges, which in a layer dominated by its sink mostly means by how
much they read from the sink.

## The graphs of one sentence

```python
from dataclasses import replace

base = st.GSPConfig(device="cpu", verbose=False, save_plots=False)
configs = {
    "attention, symmetric": base,
    "value, symmetric": replace(base, edge_weights="value"),
    "attention, directed": replace(base, graph_direction="directed"),
    "value, directed": replace(base, edge_weights="value", graph_direction="directed"),
}
for name, cfg in configs.items():
    m = st.SpectralModel(model.model, model.tokenizer, config=cfg)
    cache = m.run_with_spectra(text, metrics="graph", layers=[9])
    tv = cache["blocks.9.graph.directed_tv"] if "graph.directed_tv" in cache.metric_names else None
    print(name, cache["blocks.9.graph.fiedler_value"], cache["blocks.9.graph.hfer"], tv)
```

| layer 9 graph | Fiedler value | HFER | directed TV |
|---|---|---|---|
| attention, symmetric (the default) | 0.849 | 0.455 | |
| value-weighted, symmetric | 0.566 | 0.003 | |
| attention, directed (magnetic, q = 1/4) | 0.844 | 0.507 | 7.62 |
| value-weighted, directed | 0.378 | 0.002 | 0.129 |

Once the sink's edges are weighted by its small values, the hidden states
are smooth on the graph (HFER 0.455 to 0.003) and attention transport
barely changes them (TV 7.62 to 0.129). The large transport value of the
attention graph comes from the first token: its hidden state has a norm of
3076 against about 162 for the others, and an operator that sends 79% of
every row to it changes every other token's signal.

Reversing the edges:

```python
ids = model.tokenizer(text, return_tensors="pt").input_ids
x = model(ids, output_hidden_states=True).hidden_states[10][0].detach()   # paired with layer 9
P = st.directed_operator(vals[9].weights[0].mean(0))
R = st.directed_operator(P.T)                       # every edge reversed
print(st.directed_total_variation(x, P), st.directed_total_variation(x, R))   # 7.62, 0.774
print(st.magnetic_hfer(x, P), st.magnetic_hfer(x, P.T))                       # 0.5069, 0.5069
```

The transport measure separates the two orientations; the magnetic HFER of
the real hidden states is identical for both, to the last digit.

In a `SpectralCache`, directed graphs fill the usual `graph.*` metrics from
the magnetic Laplacian, add `graph.directed_tv` (family "hybrid") and store
`graph.phase` in place of `graph.fiedler_vector`. At `magnetic_q=0` the
metrics equal those of the symmetric graph built from `P`.

## Flow-imbalance clustering

`st.hermitian_clustering(A, k)` clusters the nodes of a directed graph by
the direction of the flow between them (Cucuringu et al., 2020): k-means on
the leading eigenvectors of the Hermitian matrix `i (A − Aᵀ)`, in which a
reciprocated edge cancels. It returns the labels, the flow between clusters
(`flow[a, b]`: how much cluster a reads from cluster b) and the signed cut
imbalance `flow[a, b] / (flow[a, b] + flow[b, a]) − 1/2`. On three planted
blocks with one edge density everywhere, where only the direction of the
edges between blocks differs, it recovers the blocks exactly; symmetrised
spectral clustering cannot see them.

```python
import numpy as np

rng = np.random.default_rng(0)
k, m = 3, 50
labels = np.repeat(np.arange(k), m)
A = np.zeros((k * m, k * m))
for i in range(k * m):
    for j in range(i + 1, k * m):
        if rng.random() < 0.5:                          # one density everywhere
            p_forward = 0.5 if labels[i] == labels[j] else 0.95
            if rng.random() < p_forward:
                A[j, i] = 1.0                           # j reads i: block a -> block b, a < b
            else:
                A[i, j] = 1.0
fc = st.hermitian_clustering(A, k=3)
print((fc.labels == labels).all())                      # True
print(fc.imbalance.round(2))
```

```text
[[ 0.   -0.44 -0.45]
 [ 0.44  0.   -0.44]
 [ 0.45  0.44  0.  ]]
```

Block 1 reads from block 0 and block 2 from both, about 95% one way.

## Options and cost

| `GSPConfig` field | values | default |
|---|---|---|
| `edge_weights` | `"attention"`, `"value"`, `"value_per_head"` | `"attention"` |
| `graph_direction` | `"symmetric"`, `"directed"` | `"symmetric"` |
| `magnetic_q` | charge of the magnetic Laplacian, `≥ 0` | `0.25` |

The options apply to the head-averaged graph: the `graph` namespace of
`SpectralModel.run_with_spectra` and `compute_spectra` (which takes
value-weighted edges as `edges=`), and `GraphConstructor.layer_graph`. The
per-head namespaces `attn` and `resid` always measure each head's
symmetrised attention. Directed graphs use `normalization="sym"` (the
normalised magnetic Laplacian) or `"none"`. Calls that leave the options at
their defaults give results identical, bit for bit, to earlier releases.
`GSPDiagnosticsFramework`, the graph filters and frequency-resolved
patching run on the symmetrised attention graph and refuse the other
options.

Without a gate the summed edges come from the Gram matrices of the per-head
projected values, `w_ij² = Σ_{h,h'} a^h_ij a^h'_ij ⟨u^h_j, u^h'_j⟩`, in
`O(S² H²)` time and no `[S, S, d_model]` tensor. A gate depends on the
query, so gated layers are computed over chunks of queries, each over the
keys it attends to, in `O(S² H d_head d_model)` time.

## References

- G. Kobayashi, T. Kuribayashi, S. Yokoi and K. Inui. Attention is not only
  a weight: analyzing transformers with vector norms. EMNLP 2020.
- A. Sandryhaila and J. M. F. Moura. Discrete signal processing on graphs:
  frequency analysis. IEEE Transactions on Signal Processing, 62(12), 2014.
- M. Fanuel, C. M. Alaíz and J. A. K. Suykens. Magnetic eigenmaps for
  community detection in directed networks. Physical Review E, 95, 022302,
  2017.
- S. Furutani, T. Shibahara, M. Akiyama, K. Hato and M. Aida. Graph signal
  processing for directed graphs based on the Hermitian Laplacian. ECML
  PKDD 2019.
- X. Zhang, Y. He, N. Brugnone, M. Perlmutter and M. Hirn. MagNet: a neural
  network for directed graphs. NeurIPS 2021.
- M. Cucuringu, H. Li, H. Sun and L. Zanetti. Hermitian matrices for
  clustering directed graphs: insights and applications. AISTATS 2020.
- D. Dahlem, D. Maniloff and M. Misiura. Self-attention as transport:
  limits of symmetric spectral diagnostics. arXiv:2605.04893, 2026.
