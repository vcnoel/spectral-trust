# A band-stop filter as an injection defence (planned)

!!! note "Planned, not run"
    This page describes an experiment design. It has not been run and gives
    no results. The run needs GPU models of 7B parameters or more.

## Idea

In a prompt injection, an instruction is hidden in the data an assistant is
asked to process (a document, an email, a tool output). If the tokens of
the injected instruction form their own group in the attention graph, part
of what they contribute to the residual stream lives in a few graph
frequencies. A band-stop filter on those frequencies, applied at the
positions after the data block, could weaken the injection while leaving
the rest of the prompt intact:

```python
with st.intervene(model) as iv:
    iv.resid(layer).graph_filter("bandstop", band=(lo, hi), strength=s,
                                 causal=True, token_span=(1, None))
    out = model.model.generate(ids, max_new_tokens=128, do_sample=False)
```

`causal=True` is required: generation uses a KV cache, and the symmetric
filter would change past positions.

## Design

- **Benchmark.** SEP (Zverev et al., ICLR 2025, *Can LLMs Separate
  Instructions From Data? And What Do We Even Mean By That?*), from the
  authors' release: each item places a probe instruction in the task
  (instruction) part or in the data part of the prompt. The separation score
  is how often the probe is executed when it sits in the instruction part
  and not when it sits in the data part.
- **Models.** Instruction-tuned open models at 7B to 8B parameters, in
  bfloat16 on a GPU.
- **Choosing the band.** On a calibration split only: for each layer, the
  band whose energy at the data positions differs most between items with
  and without a probe. Layer, band and strength are fixed before the test
  split is read.
- **Outcomes.** On the test split: probe execution rate with the probe in
  the data part (lower is better), with the probe in the instruction part
  (it should not fall), and utility on the SEP tasks without probes
  (accuracy against the reference answers).
- **Controls.**
    - the same band-stop filter on the chain graph (`graph="path"`), a purely
      positional filter;
    - a band of the same size chosen at random;
    - a filter on a shuffled attention graph with the edit's norm matched per
      prompt, as in [graph-frequency steering](graph-filter-steering.md);
    - non-spectral baselines that any spectral defence must beat: the
      attention mass that the final positions put on the data span,
      a linear probe on the residual stream, and prompting defences
      (delimiters, spotlighting), each from its authors' code.
- **Attacker-aware evaluation.** Fixed injections are not enough: the
  defence is public, so the attacker knows the layer, band and strength.
  Injections are optimised against the filtered model (the causal filter is
  differentiable, so gradient-based attacks such as GCG apply), and the
  attack success rate is reported for fixed and adaptive attacks, with the
  attack budget stated.
- **Statistics.** Bootstrap intervals over items for every rate; the defence
  counts only if it lowers the data-side execution rate under the adaptive
  attack beyond every control, at a utility loss stated alongside.

## Limits of the idea

- An injection that does not form its own group in the attention graph has
  no band to remove.
- The filter edits every position after the data block; utility costs must
  be measured, not assumed.
- A band chosen on one model, layer or prompt format may not transfer.
