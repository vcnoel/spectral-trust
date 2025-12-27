# Spectral Trust Framework

**A Graph Signal Processing (GSP) framework for measuring the trustworthiness of LLM internal representations.**

[`spectral_trust`](https://github.com/your-username/spectral-trust) constructs dynamic graphs from attention patterns and applies spectral analysis (eigenvalues, Dirichlet energy) to detect hallucinations, quantify uncertainty, and map the "smoothness" of reasoning flows.

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
- `notebooks/`: Jupyter notebooks for demonstration.
- `examples/`: Minimal example scripts.
- `dist/`: Wheel and source distributions.

## Installation

```bash
pip install spectral_trust
# OR install from source
pip install -e .
```

## Usage

### CLI Power Tool

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

## License
MIT
