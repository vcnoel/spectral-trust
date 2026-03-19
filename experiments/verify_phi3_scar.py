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
from tqdm import tqdm
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def verify_scar():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path = "data/active_passive_200.json"
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    config = GSPConfig(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="float32",
        normalization="sym",
        save_activations=False
    )
    
    deltas = []
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_name)
        
        for item in tqdm(data, desc="Analyzing pairs"):
            active_text = item['active']
            passive_text = item['passive']
            
            # Analyze Active
            res_a = framework.analyze_text(active_text, save_results=False)
            metrics_a = res_a['layer_diagnostics']
            
            # Analyze Passive
            res_p = framework.analyze_text(passive_text, save_results=False)
            metrics_p = res_p['layer_diagnostics']
            
            # Calculate Mean Fiedler in Layers 2-5
            f_a = np.mean([m.fiedler_value for m in metrics_a[2:6]])
            f_p = np.mean([m.fiedler_value for m in metrics_p[2:6]])
            
            delta = f_a - f_p
            deltas.append(delta)
            
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    
    print("\n" + "="*50)
    print("PHI-3 SCAR VERIFICATION (200 PAIRS)")
    print("="*50)
    print(f"Mean Fiedler Delta (Layers 2-5): {mean_delta:.4f}")
    print(f"Std Dev: {std_delta:.4f}")
    print("-" * 50)
    
    if mean_delta > 0.05:
        print("CONCLUSION: Scar CONFIRMED (Passive DROPS connectivity).")
    elif mean_delta < -0.05:
        print("CONCLUSION: Scar REVERSED (Passive SPIKES connectivity).")
    else:
        print("CONCLUSION: No significant aggregate scar found.")

if __name__ == "__main__":
    verify_scar()
