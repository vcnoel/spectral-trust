# Which words split Phi-3's attention graph

In Phi-3-mini-4k-instruct, the Fiedler value λ2 of the head-averaged
attention graph at layers 3 to 5 is close to 0.93 for most short sentences
and falls to about 0.37 for some. Passive sentences are among those that
fall, so the drop looks like a passive-voice effect. This page measures it
on matched pairs, finds the words that trigger it, compares λ2 with plain
attention-sink mass, and tries to undo it.

```python
import spectral_trust as st

model = st.SpectralModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct", dtype="bfloat16")
texts = ["The scientist discovered the particle.",
         "The particle was discovered by the scientist."]
cache = model.run_with_spectra(texts, metrics=["graph.fiedler_value"], layers=[3],
                               hybrid=False, batch_size=1)
print(cache["blocks.3.graph.fiedler_value"])    # about 0.94 and 0.38
```

## Setup

Phi-3-mini-4k-instruct in bfloat16. The graph is the one the library
measures: each head's attention symmetrised, heads averaged, symmetric
normalised Laplacian; λ2 is its second-smallest eigenvalue. Sink mass is the
attention received by position 0, averaged over heads and over queries 1 to
S − 1. Phi-3's tokenizer adds no BOS token, so position 0, the attention
sink, is the first word of the sentence.

Sentences (`examples/use_cases/passive_data.py`):

- 72 matched active/passive pairs from 36 (agent, verb, patient) triples in
  three templates; the two members of a pair start with the same word ("The"
  or "Yesterday"), so the sink token is the same;
- 24 agentless passives ("The cat was chased.") paired with actives;
- 24 actives with a non-agentive "by" ("The cat slept by the door.");
- 24 pairs whose first words differ ("The scientist discovered a new
  particle." against "A new particle was discovered by the scientist.");
- 24 French and 24 German pairs.

Passives are 2.0 tokens longer than their actives on average. The
length-adjusted difference subtracts the slope of λ2 on token count among
the actives times that length difference, with the slope refitted in each
bootstrap resample. Intervals are 95% percentile bootstrap intervals over
pairs (10,000 resamples). AUROC is that of the passive value over the active
one: 0 means that every passive is below every active.

## Results

**The drop is large, and length does not explain it.** On the 72 pairs:

| layer | active λ2 | passive λ2 | difference | length-adjusted | AUROC, λ2 / sink mass |
|---|---|---|---|---|---|
| 2 | 0.416 | 0.276 | −0.139 [−0.150, −0.128] | −0.117 [−0.129, −0.105] | 0.00 / 0.22 |
| 3 | 0.929 | 0.361 | −0.569 [−0.575, −0.562] | −0.556 [−0.562, −0.549] | 0.00 / 0.00 |
| 4 | 0.933 | 0.377 | −0.556 [−0.562, −0.549] | −0.541 [−0.548, −0.535] | 0.00 / 0.00 |
| 5 | 0.926 | 0.515 | −0.411 [−0.417, −0.406] | −0.400 [−0.405, −0.395] | 0.00 / 0.00 |

Ten further pairs in another style ("The router forwarded the packet.") give
0.944 and 0.353 at layer 3.

**It is not the voice.** Three controls separate the passive voice from the
words it brings:

- agentless passives do not drop: +0.003 [−0.000, +0.006] at layer 3;
- actives with a non-agentive "by" drop as much as passives. Against short
  actives, length-adjusted at layer 3, by-actives differ by −0.563
  [−0.570, −0.555] and short passives by −0.574 [−0.578, −0.569];
- pairs whose first words differ drop the same way (−0.586 [−0.591, −0.580]
  at layer 3), so the sink token plays no part.

**Eight prepositions trigger it, four do not.** In actives of the form "The
cat slept by the door.", 8 sentences per word:

| word | λ2 at layer 3 | sink mass at layer 3 |
|---|---|---|
| to, of, for, with, at, by, from, under | 0.370 to 0.379 | 0.669 to 0.679 |
| near, behind, after, beside | 0.928 to 0.931 | 0.903 to 0.911 |
| possessive 's ("The cat's friend slept near the door.") | 0.902 | 0.898 |

The eight words give the same value to within 0.01. Twelve words are too
few to say what sets the two groups apart. In French the sentences that drop
are those with the elided article "l'" after the first word, actives and
passives alike, which leaves the French pairs at −0.148 [−0.273, −0.023];
German pairs barely move (−0.014 [−0.021, −0.007]).

**The Fiedler split finds the tokens after the trigger.** At layer 3, the
smaller side of the split holds a determiner in all 72 passives and the
agent and the period in 71, while the first word, "was" and "by" are on it
in only one. For "The particle was discovered by the scientist." the smaller
side is "the scient ist ." and λ2 is 0.375. The sink loses mass at the same time (−0.225 at layer 3), and among
passives λ2 and sink mass are anti-correlated (Spearman −0.91 at layer 3).

**Sink mass detects it as well as λ2 at layers 3 to 5.** Both separate every
pair there (AUROC 0.00), and so does the attention mass of the most-attended
token. λ2 is ahead only at layer 2 (0.00 against 0.22). What λ2 adds is the
location of the split, not the detection.

**Other models.** On the same 72 pairs:

| model | layer 2, raw | layer 2, length-adjusted | layer 3, raw | layer 3, length-adjusted |
|---|---|---|---|---|
| GPT-2 small | −0.079 [−0.087, −0.072] | +0.007 [−0.000, +0.014] | −0.048 [−0.055, −0.041] | +0.030 [+0.022, +0.039] |
| Qwen3-0.6B | +0.005 [−0.003, +0.012] | +0.023 [+0.015, +0.032] | −0.008 [−0.014, −0.003] | +0.046 [+0.037, +0.053] |
| Llama-3.2-1B-Instruct | +0.005 [+0.001, +0.009] | +0.015 [+0.010, +0.019] | −0.005 [−0.009, −0.002] | +0.012 [+0.007, +0.017] |
| Phi-4-mini-instruct | −0.103 [−0.110, −0.097] | −0.119 [−0.128, −0.108] | −0.081 [−0.087, −0.076] | −0.108 [−0.120, −0.093] |

GPT-2's raw drop is length: among its actives λ2 falls with token count
(Spearman −0.93 at layer 2), the drop vanishes after adjustment, and French
and German pairs show it too (−0.062 and −0.078 at layer 2). Qwen3 and
Llama show no drop. Phi-4-mini shows the same trigger at about a fifth of
Phi-3's size: at layer 2, agentless passives do not drop (−0.003 [−0.013,
+0.007]), and by-actives drop by −0.096 [−0.107, −0.086] against −0.126
[−0.138, −0.113] for passives (length-adjusted). There, λ2 separates every
pair at layer 2 (AUROC 0.00) while sink mass does not (0.76).

**No edit at layer 2 undoes it.** Three edits at layer 2, each with a
matched control, measured on the 72 passives (recovery: the share of the
active-passive gap closed) and on 50 WikiText-2 passages (KL from the
unedited model):

| edit | recovery, layers 3 / 4 / 5 | text KL | control: recovery | control: text KL |
|---|---|---|---|---|
| graph low-pass, strength 0.5 | −0.14 / −0.13 / −0.07 | 0.36 | −0.04 / −0.04 / −0.01 | 0.23 |
| graph low-pass, strength 1 | −0.18 / −0.18 / −0.18 | 1.08 | −0.07 / −0.05 / −0.02 | 1.79 |
| remove the edges into "was" and "by" | +0.01 / +0.01 / +0.03 | 0.002 | −0.00 / −0.00 / +0.00 | 0.003 |
| mean-difference steering, α 0.2 | +0.13 / +0.06 / −0.01 | 1.23 | +0.15 / +0.05 / +0.00 | 3.18 |
| mean-difference steering, α 1 | +0.42 / +0.36 / +0.28 | 6.10 | +0.29 / +0.21 / +0.16 | 11.50 |

- The graph filter (causal low-pass at cutoff 1.0 on tokens 1..; control: the
  same filter on a row-shuffled graph, same edit norm) moves λ2 the wrong
  way, further than its control.
- Removing the edges into "was" and "by" in every head, with the sink held
  fixed (control: as many random keys), recovers 1% to 3%. The tokens after
  "by" leave the sink at layer 3 whether or not layer 2 attended to "by".
- The steering vector is the mean residual after block 2 of 16 actives minus
  16 passives (norm 64.7; control: a random direction of the same norm). At
  α 0.2 it recovers no more than the random direction and lowers the actives'
  λ2 at layer 3 by 0.13. At α 1 it recovers more (0.42 against 0.29 at layer
  3), with the model broken (text loss +5.8 nats) and the actives' λ2 down by
  0.25. Both pull actives and passives towards the middle; neither is a
  repair.

## What to do with this

- In Phi-3-mini, a low λ2 at layers 3 to 5 flags a sentence that contains
  one of a set of frequent function words, and the Fiedler split points at
  the tokens that follow it. It does not flag the passive voice.
- Compare a spectral drop with sink mass before reading it as structure:
  here sink mass detects it as well at layers 3 to 5.
- Before naming an effect after a grammatical category, test the words that
  come with the category (here, "by") on their own.

## Controls and limits

- Controls: agentless passives, by-actives, first-word pairs, French and
  German pairs, a length adjustment, sink mass and top-token mass, four other
  models, and a matched random control for every edit.
- Templated sentences. The probe has 12 words in 8 frames each.
- Phi-3 ran in bfloat16; float32 on the first 8 blocks gives the same means
  at layers 2 to 5 to within 0.003, and the random-walk Laplacian
  (`normalization="rw"`) gives the same λ2 as the symmetric one to within
  1.3 × 10⁻⁶.
- One run per model: attention is deterministic at a fixed dtype.
- Edits at layer 2 only. The edge removal leaves the actives untouched (they
  contain neither "was" nor "by"), so its side effects were measured on
  WikiText-2 only.

## Reproduce

`examples/use_cases/passive_data.py` writes the sentences;
`passive_drop.py --model microsoft/Phi-3-mini-4k-instruct --dtype bfloat16`
measures one model (29 s on one GPU for Phi-3), `passive_trigger_probe.py`
the twelve words, `passive_repair.py` the edits (38 s) and
`passive_reproduce.py` the ten further pairs. The numbers are in
`examples/use_cases/outputs/passive_drop.json`, `passive_trigger_probe.json`,
`passive_repair.json` and `passive_reproduce.json`.
