# Use cases

!!! note "Experimental"
    The interventions are new and experimental, and their API may change in a
    minor release. Default results of existing calls are guarded by
    bit-identity tests across releases. Audits are welcome: open a
    ["wrong number" issue](https://github.com/vcnoel/spectral-trust/issues/new?template=wrong-number.yml).

Each page below applies spectral-trust to a real model and data set,
against controls that could have produced the same result, with bootstrap
intervals over prompts. Each states what was measured, what it was compared
with, and where it stops. The scripts are in `examples/use_cases/`.

| Page | Model and data | What it shows |
|---|---|---|
| [Edge ablation and the sink](sink-ablation.md) | GPT-2 small, 200 IOI prompts | Removing the name movers' END → name edges changes the logit difference by −0.79, −0.12 or +0.08, depending only on where the removed attention mass goes. |
| [Graph-frequency steering](graph-filter-steering.md) | GPT-2 small, IOI and WikiText-2 | Large, layer-specific effects that mostly follow the size of the edit; along the attention graph, smoothing costs 1.4 to 2.5 times more next-token loss than an edit of the same size along positions. |
| [Frequency-resolved patching](frequency-patching.md) | GPT-2 small, 200 IOI prompt pairs | One band of the attention graph recovers 0.51 of the logit difference at layer 10, above positional and random bands; patching the END position alone recovers more. |
| [Which words split Phi-3's attention graph](phi3-prepositions.md) | Phi-3-mini-4k-instruct, 72 active/passive pairs and controls; four other models | λ2 at layer 3 falls from 0.93 to 0.36 on passives, but the trigger is a set of frequent prepositions, not the voice; sink mass detects it as well, the Fiedler split locates it, and no layer-2 edit undoes it. |
| [Graph-filter side effects](graph-filter-side-effects.md) | GPT-2 small and Phi-3-mini, WikiText-2 | At matched size, a low-pass edit along the attention graph costs more next-token KL than random-graph, path and scaling edits on GPT-2 and in Phi-3's middle and late layers; generation stays clean up to about 0.3 of the residual norm, and full-strength high-pass and band-pass edits break the model. |
| [Injection defence (design, not run)](injection-defence.md) | — | A design, not a result: a band-stop filter defence and its attacker-aware evaluation. |

The GPT-2 IOI scripts pin the CPU to two threads; the Phi-3 and side-effect
scripts were run on one GPU (`USE_CASE_DEVICE=cuda`). All read models and
data from the local Hugging Face cache, and each page gives its runtime.
