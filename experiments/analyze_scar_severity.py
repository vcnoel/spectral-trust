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
import pandas as pd
from tqdm import tqdm
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def analyze_severity():
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
    
    results = []
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_name)
        
        for item in tqdm(data, desc="Analyzing severity"):
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
            
            delta = f_a - f_p # Positive = Drop in Passive
            
            results.append({
                "active": active_text,
                "passive": passive_text,
                "fiedler_active": f_a,
                "fiedler_passive": f_p,
                "delta": delta
            })
            
    df = pd.DataFrame(results)
    df = df.sort_values(by="delta", ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 SCARRED SENTENCES (Largest Connectivity Drop in Passive)")
    print("="*80)
    
    for i in range(10):
        row = df.iloc[i]
        print(f"Rank {i+1} | Delta: {row['delta']:.4f}")
        print(f"  Active : {row['active']}")
        print(f"  Passive: {row['passive']}")
        print("-" * 80)

    # Save for user inspection
    df.to_csv("experiments/scar_severity_analysis.csv", index=False)
    print("\nFull analysis saved to experiments/scar_severity_analysis.csv")

if __name__ == "__main__":
    analyze_severity()
