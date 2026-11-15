# Use cases

Each page below runs one of the interventions of `st.intervene` on a real
model and data set, against controls that could have produced the same
result, with bootstrap intervals over prompts. Each states what was
measured, what it was compared with, and where it stops. The scripts are in
`examples/use_cases/` and run on a CPU.

| Page | Model and data | What it shows |
|---|---|---|
| [Edge ablation and the sink](sink-ablation.md) | GPT-2 small, 200 IOI prompts | Removing the name movers' END → name edges changes the logit difference by −0.79, −0.12 or +0.08, depending only on where the removed attention mass goes. |
| [Graph-frequency steering](graph-filter-steering.md) | GPT-2 small, IOI and WikiText-2 | Large, layer-specific effects that mostly follow the size of the edit; along the attention graph, smoothing costs 1.4 to 2.5 times more next-token loss than an edit of the same size along positions. |
| [Frequency-resolved patching](frequency-patching.md) | GPT-2 small, 200 IOI prompt pairs | One band of the attention graph recovers 0.51 of the logit difference at layer 10, above positional and random bands; patching the END position alone recovers more. |
| [Injection defence (planned)](injection-defence.md) | — | The design of a band-stop filter defence and its attacker-aware evaluation; not run. |

The scripts pin the CPU to two threads and read models and data from the
local Hugging Face cache; each page gives its runtime.
