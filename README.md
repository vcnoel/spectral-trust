# Spectral Trust Framework

**A Graph Signal Processing (GSP) framework for measuring the trustworthiness of LLM internal representations.**

[`spectral_trust`](https://github.com/vcnoel/spectral-trust) constructs dynamic graphs from attention patterns and applies spectral analysis (eigenvalues, Dirichlet energy) to detect hallucinations, quantify uncertainty, and map the "smoothness" of reasoning flows.

## What is it?
By treating the transformer's attention mechanism as a **graph** and the hidden states as **signals** on that graph, we can calculate rigorous mathematical metrics:
*   **Dirichlet Energy**: How much the signal varies across connected tokens (proxy for conflict/uncertainty).
*   **Smoothness Index**: Normalized energy indicating how well the representation aligns with the attention structure.
*   **Fiedler Value**: Algebraic connectivity of the attention graph.
*   **HFER (High-Frequency Energy Ratio)**: Energy concentration in high-frequency spectral components.

## Features
- **Plug-and-Play**: Works out-of-the-box with `Llama-3`, `Mistral`, `Qwen`, `Gemma`, and `Phi`.
- **Offline Ready**: `--offline` mode to use cached models without internet access.
- **Spectral Metrics**: Automatically computes Energy, Entropy, Fiedler Value, HFER, and Smoothness.
- **Robustness Tools**: Includes hooks for head ablation and residual patching.

## Structure
- `src/spectral_trust/`: Core package source code.
- `notebooks/`: Tutorials and demos.
- `experiments/`: Reproduction scripts for paper findings (Super Scar, etc.).
- `examples/`: Minimal usage examples.

## Installation

```bash
pip install spectral_trust
# OR install from source
pip install -e .
```

## Usage

### Automated Diagnosis (New!)
Run a full medical report on your model to detect known pathologies (like the "Super Scar"):

```bash
gsp-cli diagnose --model microsoft/phi-4 --verbose
```
*   **scans** for structural anomalies (graph disconnection).
*   **probes** with adversarial inputs (Active vs Passive).
*   **reports** signature matches (e.g., "Synthetic Scar Detected").

### Single-Shot Analysis
**Analyze a sentence** (uses `cuda` if available):
```bash
gsp-cli analyze --text "The capital of France is Paris." --model llama-3.1-8b
```

**Offline Mode** (no internet required):
```bash
gsp-cli analyze --text "Refactoring is fun." --model llama-3.2-1b --offline
```

### Python API

```python
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

config = GSPConfig(model_name="llama-3.2-1b", device="cuda", local_files_only=True)
with GSPDiagnosticsFramework(config) as framework:
    framework.instrumenter.load_model("meta-llama/Llama-3.2-1B")
    results = framework.analyze_text("The capital of France is Paris.")
    
    print(f"Smoothness: {results['layer_diagnostics'][-1].smoothness_index:.4f}")
```

### Compare Two Texts

Compare the spectral properties of two different inputs side-by-side:

```bash
python -m spectral_trust.cli compare \
  --text1 "Total confidence: The capital of France is Paris." \
  --text2 "Low confidence: I think the capital might be Paris." \
  --model llama-3.2-1b
```

This will generate a comparison plot overlaying the metrics for both texts.

### Multi-Run Analysis (Stochastic)

Run the analysis multiple times (useful with sampling enabled) to see metric stability:

```bash
python -m spectral_trust.cli analyze \
  --text "The capital of France is Paris." \
  --runs 5 \
  --temperature 0.7
```

### Advanced GSP Options

For rigorous spectral graph analysis, you may want to exclude self-attention loops (the diagonal) to match standard spectral graph theory (where $A_{ii}=0$). 

*   **Default**: Self-loops kept. Faithful to Transformer mechanics. Fiedler values $\approx 1.0$.
*   **`--remove_self_loops`**: Self-loops removed. Faithful to Graph Signal Processing theory. Fiedler values $\approx 2.0$ (for connected graphs). Better for measuring pure token-to-token mixing.

```bash
gsp-cli analyze --text "..." --remove_self_loops
```

## Scientific Validation

This framework implements the methodologies described in **[Noël, 2026]**.

### Case Study: The Phi-4 "Super Scar"
We used `spectral_trust` to discover a critical vulnerability in the Phi-4 model:
*   **Pathology**: Complete structural attention collapse (Fiedler $\to$ 0.0) when processing "Heavy Agent" passive sentences.
*   **Cause**: Interaction between passive voice syntax and high-complexity noun phrases.
*   **Reproduction**:
    ```bash
    python experiments/reproduce_super_scar.py
    ```
    *(Generates comparative plots for Phi vs. Qwen/Llama baselines)*

It provides the reference implementation for measuring:
*   **Fiedler Drop**: The loss of algebraic connectivity in hallucinating models.
*   **Energy Spikes**: High-frequency noise indicating semantic conflict.

## Model Compatibility & Benchmarks

| Model Family | Status | Tested Version | Precision |
| :--- | :---: | :--- | :---: |
| **Llama-3** | ✅ Passed | `meta-llama/Llama-3.2-1B` | FP16 |
| **Phi-3** | ✅ Passed | `microsoft/Phi-3-mini-4k-instruct` | BF16 |
| **Inference Time** | ⚡ Fast | ~45ms / 128 tokens | Exact Eig |

## Research Tools included
*   `examples/detect_hallucination.py`: Differential spectral analysis of counter-factuals.
*   `examples/ablation_study.py`: Causal intervention via head masking to verify structural load-bearing.
*   `benchmarks/`: Latency and precision scaling scripts.

## License
MIT
