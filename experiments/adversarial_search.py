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


import json
import torch
import numpy as np
import sys
import argparse
from tqdm import tqdm
from spectral_trust.framework import GSPDiagnosticsFramework
from spectral_trust.config import GSPConfig
from transformers import BitsAndBytesConfig

def run_adversarial_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="data/adversarial_gen1.json")
    parser.add_argument("--output", "-o", type=str, default="results_adversarial_gen1.json")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output

    print(f"Loading samples from {input_file}...")
    with open(input_file, 'r') as f:
        samples = json.load(f)

    # Configure 4-bit quantization for Phi-4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    print("Loading Phi-4 (4-bit)...")
    # GSPConfig does not take probe_name in init, it flows contextually if needed or via framework
    config = GSPConfig(
        model_name="microsoft/phi-4",
        model_kwargs={"quantization_config": bnb_config},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Use Framework directly
    results = []
    
    # Layers to analyze (same as Scar detection)
    start_layer = 2
    end_layer = 6 # Exclusive: 2,3,4,5
    
    print("Running Spectral Analysis (Adversarial Search)...")
    try:
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(config.model_name)
            
            for sample in tqdm(samples):
                # Analyze Active
                res_a = framework.analyze_text(sample['active'], save_results=False)
                metrics_a = res_a['layer_diagnostics']
                # Extract Fiedler for target layers (indices match 0-indexed list)
                # metrics_a[i] corresponds to layer i
                active_mean = np.mean([metrics_a[i].fiedler_value for i in range(start_layer, end_layer)])

                # Analyze Passive
                res_p = framework.analyze_text(sample['passive'], save_results=False)
                metrics_p = res_p['layer_diagnostics']
                passive_mean = np.mean([metrics_p[i].fiedler_value for i in range(start_layer, end_layer)])

                # Calculate Delta (Active - Passive)
                delta = active_mean - passive_mean
                
                results.append({
                    "id": sample['id'],
                    "active_text": sample['active'],
                    "passive_text": sample['passive'],
                    "active_mean": float(active_mean),
                    "passive_mean": float(passive_mean),
                    "delta": float(delta)
                })
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return

    # Sort by Delta (Descending)
    results.sort(key=lambda x: x['delta'], reverse=True)

    print("\n\n=== ADVERSARIAL LEADERBOARD (Top 10 Scar Triggers) ===")
    print(f"{'ID':<30} | {'Delta':<10} | {'Passive Mean':<10}")
    print("-" * 60)
    for res in results[:10]:
        print(f"{res['id']:<30} | {res['delta']:<10.4f} | {res['passive_mean']:<10.4f}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    run_adversarial_search()
