# spectral-trust

**Graph-spectral diagnostics for transformer attention.**

`spectral_trust` builds graphs from a transformer's attention patterns and
computes spectral diagnostics on them (Laplacian eigenvalues, algebraic
connectivity, Dirichlet energy). It is designed for research on internal-signal
analysis of LLMs: hallucination detection, uncertainty quantification, and
layer-wise routing diagnostics.

## Metric families: read this first

Every metric is a function of a specific tensor, and the distinction matters
for both access requirements and interpretation. As of v0.3.0 each metric is
explicitly tagged with its family (`spectral_trust.METRIC_FAMILIES`, also
embedded in every result):

| Family | Input tensors | Metrics | Valid claim |
|--------|--------------|---------|-------------|
| **attention** | attention weights only | `fiedler_value`, `spectral_radius`, `connectivity`, `eigenvalues`, `eig_energy`, `eig_spectral_entropy`, `eig_hfer`, `max_imaginary` | "attention-only": computable from attention maps, no hidden-state access |
| **hybrid** | attention weights **and** residual-stream states | `energy`, `smoothness_index`, `spectral_entropy`, `hfer`, `spectral_masses` | projects residual-stream vectors onto the attention eigenbasis; requires full internals and reflects residual-stream content, **not** attention topology alone |

If you report results as "attention-only", use family-**attention** metrics.
The hybrid metrics remain available and can be strong detectors, but they must
be benchmarked against direct hidden-state probes at equal access, not against
attention-only baselines.

Note on terminology: what Hugging Face returns via `output_hidden_states` are
**residual-stream snapshots** (block outputs). Module-internal activations
(attention-block outputs, MLP activations) are different tensors and are not
consumed by this library.

## Diagnostics

Attention-graph construction: per layer, multi-head attention is aggregated
(uniform or attention-mass-weighted), symmetrized, and turned into a graph
Laplacian. Three normalizations are supported; the **symmetric normalized
Laplacian is the default** since v0.3.0 because its spectrum lies in [0, 2]
independent of sequence length, making metrics comparable across inputs
(the combinatorial and random-walk spectra scale with length).

- **Fiedler value** (λ₂) — algebraic connectivity; low values indicate
  fragmented attention routing.
- **eig_energy / eig_spectral_entropy / eig_hfer** — eigenvalue-only
  counterparts of the classic energy/entropy/HFER diagnostics (v0.3.0).
- **Dirichlet energy / smoothness index** — variation of residual-stream
  signals across attention edges (hybrid).
- **Spectral entropy / HFER / spectral masses** — frequency content of
  residual-stream signals in the attention eigenbasis (hybrid).
- **Spectral velocity** — layer-to-layer derivative of any metric trajectory.
- **Directed diagnostics** — spectral radius and maximum imaginary component
  of the directed random-walk Laplacian.
- **Subgraph isolation** — restrict any diagnostic to a token span (e.g. a
  generated tool call).

## Installation

```bash
pip install spectral_trust
# or from source
pip install -e .
```

## Quickstart

### Python API

```python
from spectral_trust import GSPDiagnosticsFramework, GSPConfig, METRIC_FAMILIES

config = GSPConfig(model_name="meta-llama/Llama-3.2-1B", device="cuda")
with GSPDiagnosticsFramework(config) as framework:
    framework.instrumenter.load_model("meta-llama/Llama-3.2-1B")
    results = framework.analyze_text("The capital of France is Paris.")

    last = results["layer_diagnostics"][-1]
    print("fiedler:", last.fiedler_value)            # attention-only
    print("eig_entropy:", last.eig_spectral_entropy) # attention-only
    print("smoothness:", last.smoothness_index)      # hybrid (needs hidden states)
    print(METRIC_FAMILIES["smoothness_index"])       # -> "hybrid"
```

### CLI

```bash
# analyze one input
gsp-cli analyze --text "The capital of France is Paris." --model llama-3.2-1b

# compare two inputs side by side (overlaid metric plots)
gsp-cli compare --text1 "..." --text2 "..." --model llama-3.2-1b

# multi-run stability analysis with sampling
gsp-cli analyze --text "..." --runs 5 --temperature 0.7

# offline mode (cached models only)
gsp-cli analyze --text "..." --model llama-3.2-1b --offline
```

### Graph construction options

- `normalization`: `"sym"` (default, length-invariant spectrum), `"rw"`,
  `"none"`.
- `symmetrization`: `"symmetric"` (default), `"row_norm"`, `"col_norm"`.
- `head_aggregation`: `"uniform"` (default) or `"attention_weighted"`.
  Head-averaging discards head-level structure; for detection tasks consider
  per-head analysis on top of the raw attentions.
- `remove_self_loops`: drop the diagonal before building the Laplacian
  (standard in spectral graph theory; changes the Fiedler scale).

### Determinism and precision (v0.3.0)

`DirectedTopologist.get_fiedler_value` now uses an **exact dense solve for
graphs up to 512 nodes** (a few milliseconds on GPU) and a **seeded, deflated
Lanczos** beyond that. Two v0.2.x defects are fixed: the unseeded random
Lanczos start (non-deterministic results) and a `>1e-6` eigenvalue filter that
discarded exactly the near-zero λ₂ regime that connectivity-based diagnostics
are meant to detect. Callers who need bit-identical runs should stay on the
exact path or pass a fixed `seed`.

## Repository layout

- `src/spectral_trust/` — package source
  - `graph.py` — attention-graph construction (symmetrization, aggregation, Laplacians)
  - `spectral.py` — eigendecomposition and diagnostics (`SpectralAnalyzer`, `SpectralDiagnostics`, `METRIC_FAMILIES`)
  - `directed_topology.py` — directed Laplacian metrics, GPU Lanczos
  - `framework.py` — end-to-end orchestration (`GSPDiagnosticsFramework`)
  - `instrumentation.py` — model loading and tensor capture
- `examples/` — minimal usage examples (hallucination differential analysis, head-masking ablation)
- `benchmarks/` — latency and precision scaling scripts
- `notebooks/` — demo notebook
- `tests/` — unit tests

## Model compatibility

Works with any Hugging Face causal LM that exposes attention maps
(`output_attentions=True`), including Llama 3.x, Mistral, Qwen 2.x/3.x, Gemma,
and Phi. Hybrid models with linear-attention layers (e.g. Qwen3.5) expose
attention only on their full-attention layers; diagnostics are computed on
those layers.

## Changelog

**v0.3.0**
- Metric family tagging (`METRIC_FAMILIES`, `SpectralDiagnostics.families`):
  every metric declares whether it is attention-only or hybrid.
- New attention-only metrics: `eig_energy`, `eig_spectral_entropy`, `eig_hfer`.
- Default Laplacian normalization changed `rw` → `sym` (length-invariant
  spectrum, deterministic symmetric solver path).
- `get_fiedler_value`: exact solve ≤ 512 nodes; seeded, null-deflated Lanczos
  above; removed the near-zero eigenvalue filter.

**v0.2.3** — fail loudly on non-finite attention tensors.

**v0.2.x** — directed topology metrics, spectral velocity, subgraph isolation,
GPU Lanczos, sparse solver optimizations.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). For commercial use or
closed-source integration, contact the author for a commercial license.
