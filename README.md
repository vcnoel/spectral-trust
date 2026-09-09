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

### Head resolution matters

Most attention-graph diagnostics (including earlier versions of this library)
average attention over heads before building the graph. **Head-averaging is a
signal destroyer.** Attention heads are specialized: the routing anomaly that
accompanies a failure is typically carried by a few heads, and summing heads
mixes it into a near-uniform background. Two independent lines of evidence:

- In our tool-call hallucination experiments, per-head Fiedler values (one λ₂
  per head per layer, no averaging) reached detection AUCs competitive with
  supervised hidden-state probes, while the same metric computed on the
  head-averaged graph was near chance under leakage-controlled evaluation.
- LapEigvals (Binkowski et al., EMNLP 2025), the strongest published
  attention-based hallucination detector, keeps per-head resolution
  throughout — its features are per-layer, per-head top-k Laplacian values.

Practical guidance: use head-averaged diagnostics (`aggregate_heads`) for
qualitative layer-trajectory analysis and visualization, but for any
*detection* task, compute metrics per head and let the probe learn which
heads carry signal. The per-head pattern is: symmetrize each head's attention
matrix separately, build one Laplacian per head, and concatenate per-head
metric values as features. `head_aggregation="attention_weighted"` (mass-
weighted averaging) softens but does not remove the aggregation loss.

### Per-head diagnostics

```python
import torch
from spectral_trust import (
    GSPConfig, per_head_metrics_all_layers,
    stack_feature_vector, layer_delta_features,
)

out = model(input_ids, output_attentions=True)     # attentions: [1, H, T, T] per layer

# one Laplacian per head per layer; all heads decomposed in one batched call
diags = per_head_metrics_all_layers(
    out.attentions,
    config=GSPConfig(normalization="sym"),
    token_span=(prompt_len, seq_len),              # restrict to a span of interest
)

diags[12].metric("fiedler_value")                  # [num_heads] for layer 12
features = stack_feature_vector(diags)             # layers x heads x 5, for a probe
dynamics = layer_delta_features(diags, metric="spectral_radius")
```

Five metrics per head come out of a single eigendecomposition, so extracting
all of them costs the same as extracting one: `fiedler_value` (lambda_2),
`connectivity_ratio` (lambda_2/lambda_max), `spectral_entropy_norm`, `hfer`,
and `spectral_radius`. All are family **attention** (attention weights only).
`token_span` takes the induced subgraph *before* building the Laplacian, so
the diagnostic is local to that span rather than to the whole sequence.

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
  - `per_head.py` — per-head spectral diagnostics and probe feature helpers
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

**v0.3.0** — first release since 0.2.2. Three of these changes alter results
for existing callers; they are marked *(behaviour change)*.

- **Per-head spectral diagnostics** (`per_head.py`): `per_head_metrics`,
  `per_head_metrics_all_layers`, `stack_feature_vector`,
  `layer_delta_features`. One Laplacian per head, five eigenvalue-only
  metrics from a single batched eigendecomposition, optional token-span
  restriction, and probe-ready feature vectors plus cross-layer dynamics.
- **Metric provenance** (`METRIC_FAMILIES`, `SpectralDiagnostics.families`):
  every metric declares whether it is computed from attention weights alone
  ("attention") or from residual-stream states projected onto the attention
  eigenbasis ("hybrid"). The energy / smoothness / entropy / HFER metrics are
  hybrid and must not be reported as attention-only.
- **New attention-only metrics**: `eig_energy`, `eig_spectral_entropy`,
  `eig_hfer` — eigenvalue-only counterparts of the hybrid trio.
- *(behaviour change)* Default Laplacian normalization `rw` → `sym`: spectrum
  in [0, 2] independent of sequence length, and always the deterministic
  symmetric solver path. Pass `normalization="rw"` to reproduce pre-0.3.0
  numbers.
- *(behaviour change)* `get_fiedler_value`: exact dense solve for graphs up to
  512 nodes, seeded and null-deflated Lanczos above. Removes an unseeded
  random Lanczos start (non-deterministic results) and a `>1e-6` eigenvalue
  filter that discarded exactly the near-zero λ₂ regime the diagnostic is
  meant to detect.
- *(behaviour change)* Non-finite attention tensors now raise
  `NonFiniteAttentionError` instead of propagating NaNs.
- Documentation rewritten around the metric-family contract; informal
  terminology removed from CLI output and the demo notebook.

**v0.2.x** — directed topology metrics, spectral velocity, subgraph
isolation, GPU Lanczos, sparse solver optimizations.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). For commercial use or
closed-source integration, contact the author for a commercial license.
