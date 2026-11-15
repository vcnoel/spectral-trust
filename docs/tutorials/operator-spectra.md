# Operator spectra in the attention geometry

An attention head maps a signal on its keys to a signal on its queries:
`f ↦ A f`, with `A` row-stochastic and causal. The singular values of `A`
measure how much the head stretches signals, but in the Euclidean geometry
they mostly measure sink mass: for the pure sink `A = 1 e₁ᵀ`,
`AᵀA = n e₁e₁ᵀ`, so `σ₁² = n` and all the spectral energy sits on the
first token. Giving the keys and the queries their own measures removes
that bias. `spectral_trust.operator_spectra` computes the resulting
spectrum per head or per layer.

The construction is that of Viganò, Melzi and Ovsjanikov (NeurIPS 2026,
arXiv:2610.06012). This module implements it from the equations of their
paper; it has not been checked against official code, which is not
released as of 15 Nov 2026. Their uncertainty estimator (SAUQ) is not part
of the library.

## The operator

Give the keys a measure `μ` and the queries a measure `π`, with
`⟨f, g⟩_μ = fᵀ Ω_μ g` and `Ω_μ = diag(μ)`. Read as a map
`L²(μ) → L²(π)`, `A` has the adjoint `Ω_μ⁻¹ Aᵀ Ω_π`, and

```text
D = Ω_μ⁻¹ Aᵀ Ω_π A          (self-adjoint and positive semidefinite on L²(μ))
S = Ω_μ^{-1/2} Aᵀ Ω_π A Ω_μ^{-1/2}    (symmetric, similar to D)
```

The eigenvalues of `D` are the squared gains `‖A φ‖²_π / ‖φ‖²_μ` of its
eigenfunctions `φ = Ω_μ^{-1/2} u` (`S u = λ u`), which are orthonormal in
`L²(μ)`. Two choices of measures:

| `measure` | `π` | `μ` | spectrum |
|---|---|---|---|
| `"pullback"` (default) | `1/n` | `Aᵀπ`, the average attention row | `λ₁ = 1` on the constant function; every gain `≤ 1` |
| `"counting"` | `1` | `1` | `D = AᵀA`: the squared singular values of `A` |

A pair `(mu, pi)` sets both measures directly. Under the pullback measure
the constant function carries no token-dependent information (attention
preserves it), and `√λ₂` is the largest gain of a signal of `μ`-mean zero:
the contraction of the routed, centred part of the values.

The spectrum is summarised by the normalised participation ratio and its
centred variant, which leaves out the largest eigenvalue (the constant mode
under the pullback measure):

```text
PR      = (1/n)     (Σ_k λ_k)² / Σ_k λ_k²            from 1/n (one mode) to rank/n
PR_cent = (1/(n-1)) (Σ_{k≥2} λ_k)² / Σ_{k≥2} λ_k²    0 when every centred λ vanishes
```

**Zero-mass keys.** Under the pullback measure `μ_j = 0` exactly when no
query attends to key j, and `Ω_μ⁻¹` does not exist. The library reads it as
the pseudo-inverse (`0` on such keys). The key is in the kernel of `A`, so
it gets the eigenvalue 0 with the indicator `e_j` as eigenfunction; this is
the operator on the keys of positive mass with its spectrum padded by
zeros. The spectrum keeps length n and the ratio keeps its `1/n`. With
measures given as `(mu, pi)`, a zero-mass key that the attention reads has
an unbounded gain and raises.

## One head

```python
import torch
import spectral_trust as st

n, s = 40, 0.77
# A sink-biased head: 0.77 of every row on the first token, the rest on the diagonal.
A = s * torch.ones(n, 1) * torch.eye(n)[0] + (1 - s) * torch.eye(n)

st.operator_metrics(A, "counting")    # participation_ratio 0.029
st.operator_metrics(A)                # participation_ratio 0.789, centred 0.976, contraction 0.480
```

The head still routes every token to itself, but in the counting geometry
its participation ratio is close to that of a pure sink (`1/n = 0.025`):
the sink column dominates `AᵀA`. Under the pullback measure the sink column
becomes the invariant measure instead, and the ratio stays high. This is
the "sink-biased" case of the paper's Figure 2 (a real head of
LLaMA-3.1-8B there: 0.02 against 0.62).

`operator_spectrum` returns the eigenvalues (decreasing), the
eigenfunctions in `L²(μ)` and the measures; `attention_operator` returns
`S` (and `D` with `return_difference=True`); `participation_ratio` takes any
spectrum. All of them take `[..., n, n]`: one head, a stack of heads
(`[heads, n, n]`, one value per head) or a batch, and `average_heads=True`
measures the head-averaged attention instead. For weights that are not
row-stochastic, such as value-weighted edges, `normalize_rows=True`
divides each row by its sum first; under the pullback measure without it,
rows must sum to 1 (or 0).

## In the pipeline

The `operator` namespace of `run_with_spectra` and `compute_spectra` holds
`participation_ratio`, `participation_ratio_centered`, `contraction` and
the array `eigenvalues`, for every measured layer. It is computed only when
`metrics=` names it (`"operator"`, `"operator.contraction"` or a bare name
such as `"participation_ratio"`), so every other call is unchanged.

```python
model = st.SpectralModel.from_pretrained("gpt2")
text = "When Mary and John went to the store, John gave a drink to"

cache = model.run_with_spectra(text, metrics="operator")
cache.metric("operator.participation_ratio")     # [12 layers, 12 heads]
cache["blocks.0.operator.participation_ratio"]  # layer 0: 0.206 0.930 0.269 0.904 ...
cache.layer(5).head(3)["contraction"]
```

On this sentence the mean contraction `√λ₂` over heads falls with depth,
0.753 at layer 0, 0.631 at layer 5 and 0.506 at layer 11.

| `GSPConfig` field | values | default |
|---|---|---|
| `operator_measure` | `"pullback"`, `"counting"` | `"pullback"` |
| `operator_heads` | `"per_head"` (`[n_heads]` per layer, `heads=` applies), `"mean"` (one value per layer, from the head-aggregated attention) | `"per_head"` |
| `operator_edges` | `"attention"`, `"value_per_head"`, `"value"` (one matrix per layer: `"mean"` only) | `"attention"` |

With `operator_edges="value"` or `"value_per_head"` the operator reads the
value-weighted attention of `spectral_trust.value_weighted`, each row
divided by its sum: the edge from key j to query i becomes the share of
`‖Σ_h a^h_ij W_O^h v^h_j‖` (or of its per-head terms) in row i. Rows are
always renormalised before the operator is built: under `token_span` this
gives the operator of the span's own edges, and for plain attention it only
removes rounding. `remove_self_loops` does not apply.

```python
cfg = st.GSPConfig(operator_heads="mean", operator_edges="value")
cache = st.SpectralModel(model.model, model.tokenizer, config=cfg).run_with_spectra(
    text, metrics="operator")
cache["blocks.11.operator.participation_ratio"]   # 0.311 (0.497 at layer 0)
```

## Sinks and output dimensionality on GPT-2

The paper's Figure 5 reports, for the heads of GPT-2, the Spearman
correlation of each diagnostic with the sink score and with the spectral
entropy of the head output `H(AV)`. `examples/operator_spectra_sink.py`
repeats it with this module on 200 WikiText-103 test paragraphs truncated
to 128 tokens. The sink score is a head's mean attention to the first token
and `H(AV)` the entropy of the normalised squared singular values of `A V`;
the paper's inputs and its exact sink score are not specified, so these are
our choices. Correlations over the 28,800 (input, head) points, then over
the 144 heads averaged across inputs, against the values printed in the
paper's figure:

| diagnostic | sink score | paper | `H(AV)` | paper |
|---|---|---|---|---|
| `PR(AᵀA)`, counting | −0.892 / −0.883 | −0.895 | 0.332 / 0.330 | 0.286 |
| `PR(D)`, pullback | −0.431 / −0.415 | −0.414 | 0.734 / 0.748 | 0.714 |
| `PR_cent(D)` | −0.360 / −0.363 | −0.340 | 0.728 / 0.755 | 0.717 |

The pattern reproduces: the Euclidean ratio is almost a function of sink
mass, the measure-aware ratios depend on it about half as much, and they
track the dimensionality of the head output more closely. Each of our
values is within 0.05 of the paper's.

```bash
python examples/operator_spectra_sink.py --items 200 --length 128
```

## Cost

Per head, the operator is one `n × n` product and one symmetric
eigendecomposition in float64; the participation ratio alone would need only
`tr S` and `‖S‖²_F`, but the contraction needs `λ₂`, so the full spectrum is
computed. Time of `compute_spectra` on the attention of all 12 layers of
GPT-2 small (CPU, 4 threads, median of 3 runs), next to the forward pass
and to the existing metrics:

| sequence length | forward pass | `operator`, per head | `operator`, `"mean"` | `graph.fiedler_value` | default metrics |
|---|---|---|---|---|---|
| 128 | 0.12 s | 0.09 s | 0.02 s | 0.06 s | 0.26 s |
| 512 | 0.65 s | 1.75 s | 0.17 s | 0.56 s | 3.57 s |

The value-weighted operator (`operator_edges="value_per_head"`) costs the
same once the edges are computed (1.77 s at 512 tokens), plus the value
capture of the forward pass.

## References

- F. Viganò, S. Melzi and M. Ovsjanikov. Spectral geometry of attention:
  from information routing to uncertainty. NeurIPS 2026, arXiv:2610.06012.
- E. Fadeeva et al. LM-Polygraph: uncertainty estimation for language
  models. EMNLP 2023 (System Demonstrations).
