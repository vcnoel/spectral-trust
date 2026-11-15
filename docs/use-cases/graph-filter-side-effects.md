# Side effects of graph-frequency steering

`graph_filter` edits the residual stream along the token axis:
`X' = X + s (g(L) X − X)`, with `L` the Laplacian of the layer's attention
graph and `g` a band response. Before using it to steer, it helps to know
what an edit of a given size does to the rest of the model: to the next
token, to the next layer's attention graph and to generated text, compared
with edits of the same size that do not use the attention graph.

```python
import spectral_trust as st

model = st.SpectralModel.from_pretrained("gpt2")
ids = model.tokenizer("The history of the city begins in the", return_tensors="pt").input_ids

with st.intervene(model) as iv:
    iv.resid(6).graph_filter("lowpass", cutoff=1.0, strength=1.0,
                             causal=True, token_span=(1, None))
    logits = model(ids).logits
```

## Setup

GPT-2 small (float32) at layers 2, 6 and 10, and Phi-3-mini-4k-instruct
(bfloat16) at layers 2, 12 and 24. The edit is the causal filter above on
tokens 1.. of the sequence, after the block: low-pass and high-pass at cutoff
1.0 and band-pass on [0.5, 1.5), at strengths 0.5 and 1. The size of an edit
is its relative norm `‖X' − X‖ / ‖X‖` over tokens 1.. of a passage (Frobenius
norms), averaged over passages.

Each edit is compared with three controls, each rescaled per passage to the
same edit norm:

- **random**: the same filter on a row-shuffled attention graph;
- **path**: the same filter on the chain graph over positions;
- **scale**: the residual shrunk along itself.

Measured on 100 WikiText-2 test passages of 64 tokens: next-token KL from the
unedited model, averaged over positions, and the change in next-token loss
(ΔNLL); λ2 and sink mass of the next layer's attention graph. Generation:
greedy 30-token continuations of the first 24 tokens of 20 other passages. A
continuation counts as degenerate if its distinct-2 ratio is below 0.5, one
token makes up more than 40% of it, or fewer than half of its non-space
characters are alphanumeric. Unedited, 1 of 20 GPT-2 continuations and 0 of
20 Phi-3 continuations are degenerate. Intervals are 95% percentile bootstrap
intervals over passages (10,000 resamples).

## Results

**At matched size, the low-pass edit costs more next-token KL than the
controls.** On GPT-2 (KL in nats):

| layer, strength | edit size | attention | random | path | scale |
|---|---|---|---|---|---|
| 2, 0.5 | 0.12 | 0.023 | 0.019 | 0.019 | 0.013 |
| 2, 1 | 0.24 | 0.221 | 0.140 | 0.105 | 0.061 |
| 6, 0.5 | 0.14 | 0.034 | 0.026 | 0.025 | 0.011 |
| 6, 1 | 0.27 | 0.298 | 0.155 | 0.152 | 0.054 |
| 10, 0.5 | 0.11 | 0.038 | 0.029 | 0.031 | 0.001 |
| 10, 1 | 0.21 | 0.265 | 0.177 | 0.187 | 0.005 |

The attention graph's KL is above each control's in all six cells, with the
interval of the paired difference above zero. The median ratios are 1.41
(random), 1.40 (path) and 4.5 (scale). On Phi-3 the same holds at layers 12
and 24 at strength 1: KL 0.047 and 0.050 against 0.021 and 0.019 (random),
0.025 and 0.018 (path), 0.010 and 0.007 (scale). At strength 0.5 there the
edits are small (0.06 and 0.04 of the residual) and every condition stays at
or below 0.008. At Phi-3 layer 2 the edits are large (0.29 and 0.57) and the
order changes: at strength 1 the attention filter costs 1.26 against 2.22,
2.20 and 1.82.

**It turns the residual rather than shrinking it.** Over the six low-pass
cells, the scaling control shrinks the norm by the full edit (−0.18 on
GPT-2, −0.19 on Phi-3), the random and path graphs by 0.07 and 0.10 (GPT-2),
and the attention filter by 0.05 on GPT-2, while on Phi-3 it grows the norm
by 0.03. An edit that changes direction costs more than one that changes
size, which fits the gap to scaling. The random and path graphs also turn
the residual, so the gap to them is specific to the attention graph.

**The next layer's graph barely moves, except at Phi-3 layer 2.** On GPT-2
the low-pass edit changes λ2 at the next layer by +0.001 to +0.014 and its
sink mass by +0.014 to +0.026; the random graph moves both by up to +0.080.
At Phi-3 layers 12 and 24 the changes are 0.002 to 0.007, against 0.013 to
0.056 for the random graph. At Phi-3 layer 2 the low-pass edit lowers λ2 at
layer 3 by 0.146 and 0.209 and its sink mass by 0.104 and 0.142, about twice
the random graph's change.

**The first token of the span is never edited by the low-pass filter.**
With `causal=True`, position i is filtered on the graph of its own prefix.
With `token_span=(1, None)`, the prefix of position 1 is a single node whose
only edge is its attention to itself (self-loops are kept by default). Its
normalised Laplacian is 1 − a/a = 0: one eigenvalue, 0, carried by the
constant vector. A low-pass response keeps that eigenvalue, so
`x'₁ = x₁ + s (g(0) − 1) x₁ = x₁` up to the Chebyshev approximation of the
band, for which 1 − g(0) = 7.3 × 10⁻⁶. The measured relative edit at
position 1 is 3.6 × 10⁻⁶ at strength 0.5 and 7.3 × 10⁻⁶ at strength 1, on
both models. High-pass and band-pass responses have g(0) ≈ 0, so they do the
opposite: `x'₁ = (1 − s) x₁`, an edit of 0.50 and 1.00 of that position's
residual. The chain graph's one-node prefix has no edge, and under the
`"sym"` normalisation an isolated node has eigenvalue 1, on the low-pass
cutoff, so the path control does edit position 1 (relative edit 0.23 on
GPT-2 and 0.17 on Phi-3, averaged over the low-pass cells). After position 1
the attention low-pass edit grows along the sequence on GPT-2: 0.12 at
positions 2 to 4, 0.16 in the middle and 0.17 at the last position. Its KL
exceeds the random graph's most at the start: 0.122 at positions 2 to 4,
against 0.024.

**Generation stays clean at steering sizes and breaks under full-strength
high-pass and band-pass edits.** Low-pass edits up to 0.27 of the residual
on GPT-2 and 0.29 on Phi-3 leave the share of degenerate continuations where
it is without an edit. The one larger low-pass edit, Phi-3 layer 2 at
strength 1 (0.57), makes 4 of 20 continuations degenerate. At strength 1,
high-pass and band-pass edits are 0.72 to 0.95 of the residual on GPT-2 and
0.77 to 1.16 on Phi-3:

| model, layer | high-pass: size, KL, degenerate | band-pass: size, KL, degenerate |
|---|---|---|
| GPT-2, 2 | 0.94, 8.58, 70% | 0.75, 4.83, 40% |
| GPT-2, 6 | 0.93, 9.13, 90% | 0.72, 3.41, 30% |
| GPT-2, 10 | 0.95, 5.10, 65% | 0.82, 1.99, 10% |
| Phi-3, 2 | 1.16, 10.27, 50% | 1.09, 9.35, 75% |
| Phi-3, 12 | 0.98, 8.32, 5% | 0.77, 2.73, 10% |
| Phi-3, 24 | 0.99, 12.88, 5% | 0.84, 6.94, 0% |

Degenerate output looks like "Word Word sub in us Word W us us us us us us
…" (Phi-3 layer 2, band-pass, attention graph). The controls break the model
as well, up to 100% degenerate continuations (GPT-2 layer 10, high-pass,
scaling control), and for these filters the attention graph is not worse
than the random graph: over both strengths, its KL is above the random
graph's in 2 of 6 high-pass and 1 of 6 band-pass cells on GPT-2, and below
it in 4 and 5. At strength 0.5 the high-pass and band-pass edits are 0.36 to
0.58 of the residual. They make at most 4 of 20 GPT-2 continuations
degenerate, none at Phi-3 layers 12 and 24, and 9 (high-pass) and 8
(band-pass) of 20 at Phi-3 layer 2.

## Safe range

- Use the low-pass filter, causal for generation, and keep the relative edit
  at or below about 0.3 of the residual. In these runs that range left greedy
  generation as clean as the unedited model on both models. `graph_filter`
  warns, once per call, when an edit exceeds 0.5.
- Set the strength from the edit size, not by value: the edit is linear in
  the strength, and strength 1 gave 0.07 and 0.12 of the residual at Phi-3
  layers 24 and 12 but 0.57 at layer 2.
- Expect more next-token change than from a random edit of the same size, and
  compare with a norm-matched control (another graph, the chain graph, or
  scaling) before reading an effect as specific to the attention graph.
- Treat high-pass and band-pass filters at strength 1 as ablations rather
  than steering: they remove most of the residual.
- The first position of a causal span is left as it is by a low-pass filter
  and erased by a high-pass one. Start the span one token before the tokens
  to edit.

## Controls and limits

- Controls: the same filter on a row-shuffled attention graph and on the
  chain graph, and scaling, each matched in edit norm per passage.
- Two models, three layers each, one cutoff per filter, two strengths and
  greedy decoding only. With 20 continuations per condition, a degenerate
  share moves in steps of 5%, and the degenerate rule is a heuristic.
- One run per condition: attention is deterministic at a fixed dtype, and
  intervals are over passages. Phi-3 ran in bfloat16.
- The threshold of the library's warning (0.5) is set from these runs, not
  tuned on other models.

## Reproduce

`examples/use_cases/filter_side_effects.py --model gpt2` and
`--model microsoft/Phi-3-mini-4k-instruct` with `USE_CASE_DEVICE=cuda` (GPT-2
small took 158 s on one GPU); `filter_side_effects_analyse.py` summarises the
runs. The numbers are in `examples/use_cases/outputs/filter_side_effects.json`
and the continuations in `filter_side_effects_<model>_generations.json`.
