# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
AI Safety Evaluation: Hallucination Detection via Spectral Diagnostics.

This script demonstrates how spectral metrics (Fiedler Value, Smoothness, HFER)
diverge between grounded true statements and hallucinations.

Hypothesis:
- True statements have 'smoother' signal flows (High Smoothness, Low Energy).
- Hallucinations/Confabulations exhibit 'rougher' flows (Low Smoothness, High Energy)
  due to internal conflict between the model's knowledge and the generated token.
"""

import matplotlib.pyplot as plt
import pandas as pd
import argparse
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

def run_hallucination_detection(model_name="gpt2"):
    print(f"Loading {model_name} for hallucination detection...")
    
    config = GSPConfig(
        model_name=model_name,
        device="cuda",  # Auto-fallback to cpu handled in framework usually, or we should be careful.
        local_files_only=False,
        plot_metrics=['energy', 'fiedler', 'smoothness']
    )
    
    # 1. Setup Framework
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_name)
        
        # 2. Define Comparison Cases
        # Case A: Grounded Truth
        text_true = "The capital of France is Paris, a major European city."
        
        # Case B: Hallucination / Counter-factual
        # We force the model to analyze a false statement as if it generated it.
        text_hallucination = "The capital of France is Rome, a major European city."
        
        print("\nAnalyzing True Statement...")
        res_true = framework.analyze_text(text_true, save_results=True)
        
        print("\nAnalyzing Hallucination...")
        res_hallucination = framework.analyze_text(text_hallucination, save_results=True)
        
        # 3. Compare Results
        print("\n--- Spectral Forensic Analysis ---")
        
        # Extract mean metrics across last few layers (reasoning layers)
        def get_avg_metrics(res, start_layer=5):
            diags = res['layer_diagnostics'][start_layer:]
            if not diags: diags = res['layer_diagnostics'] # Fallback if shallow model
            avg_energy = sum(d.energy for d in diags) / len(diags)
            avg_smoothness = sum(d.smoothness_index for d in diags) / len(diags)
            avg_fiedler = sum(d.fiedler_value for d in diags) / len(diags)
            return avg_energy, avg_smoothness, avg_fiedler
        
        # For GPT2-small (12 layers), check last 4 layers
        e_true, s_true, f_true = get_avg_metrics(res_true, start_layer=-4)
        e_hall, s_hall, f_hall = get_avg_metrics(res_hallucination, start_layer=-4)
        
        print(f"True Statement:  Energy={e_true:.4f}, Smoothness={s_true:.4f}, Fiedler={f_true:.4f}")
        print(f"Hallucination:   Energy={e_hall:.4f}, Smoothness={s_hall:.4f}, Fiedler={f_hall:.4f}")
        
        delta_energy = ((e_hall - e_true) / (e_true+1e-9)) * 100
        print(f"\nMetric Delta (Hallucination relative to True):")
        print(f"Dirichlet Energy: {delta_energy:+.2f}% (Expect + increase for hallucinations)")
        
        # 4. Visualize
        framework.visualize_comparison(res_true, res_hallucination)
        print("\nComparison plot generated in output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model to analyze")
    args = parser.parse_args()
    
    run_hallucination_detection(args.model)
