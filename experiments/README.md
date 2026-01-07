# Spectral Trust Experiments

This directory contains research scripts and reproduction code for the findings presented in our work.

## Key Reproduction Scripts

### 1. The "Super Scar" (Phi-4 Connectivity Collapse)
*   **Script**: `reproduce_super_scar.py` (formerly `investigate_complexity.py`)
*   **Purpose**: Generates the comparative plots showing the "Super Scar" in Phi-4's passive heavy sentences vs. robust baselines (Qwen/Llama).
*   **Usage**: `python experiments/reproduce_super_scar.py`
*   **Output**: `gsp_results/complexity_plots/`

### 2. Adversarial Search
*   **Script**: `adversarial_search.py`
*   **Purpose**: Runs the evolutionary search that discovered the Super Scar triggers.
*   **Usage**: `python experiments/adversarial_search.py -i data/adversarial_gen2.json`

### 3. Hallucination Feature Discovery
*   **Script**: `find_best_layer_metric.py`
*   **Purpose**: Performs the Lasso regression search to find the optimal layer-metric combinations for detecting hallucinations.

## Other Scripts
*   `analyze_scar_severity.py`: Statistical analysis of Fiedler drops on large datasets.
*   `measure_super_scar_delta.py`: Computes the precise numerical Fiedler Delta for the Phi family.
*   `tune_hallucination_metric.py`: Calibration script for the hallucination detector.
