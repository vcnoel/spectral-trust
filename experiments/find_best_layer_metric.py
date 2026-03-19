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
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_feature_search(model_name, data_path, output_csv="experiments/feature_search_results.csv"):
    config = GSPConfig(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="float16" if torch.cuda.is_available() and "gpt2" not in model_name else "float32",
        normalization="sym",
        save_activations=False,
        trust_remote_code=True
    )
    
    data = load_data(data_path)
    all_rows = []
    
    print(f"Running comprehensive feature extraction on {len(data)} samples...")
    print(f"Model: {model_name}")
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_name)
        
        for item in tqdm(data):
            text = item['text']
            label = 1 if item['label'] == 'true' else 0
            
            try:
                res = framework.analyze_text(text, save_results=False)
                metrics = res['layer_diagnostics'] # List of SpectralDiagnostics
                
                row = {'text': text, 'label': label}
                
                # Flatten metrics: e.g. energy_L0, energy_L1, ...
                for i, m in enumerate(metrics):
                    row[f'energy_L{i}'] = m.energy
                    row[f'smoothness_L{i}'] = m.smoothness_index
                    row[f'fiedler_L{i}'] = m.fiedler_value
                    row[f'hfer_L{i}'] = m.hfer
                    row[f'entropy_L{i}'] = m.spectral_entropy
                
                all_rows.append(row)
            except Exception as e:
                print(f"Error: {e}")
                
    df = pd.DataFrame(all_rows)
    df.to_csv("experiments/feature_extraction_raw.csv", index=False)
    
    # --- Feature Selection (Lasso) ---
    print("\n--- Sparse Feature Selection (Lasso) ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = [c for c in df.columns if c not in ['text', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # L1 Regularization to force sparsity
    # C=0.1 or C=1.0 depends on data, we'll try a few or just C=1.0
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', max_iter=1000)
    clf.fit(X_scaled, y)
    
    # Extract non-zero weights
    coefs = clf.coef_[0]
    selected_features = []
    
    print("\nBest Combination of Layer/Metrics:")
    for i, col in enumerate(feature_cols):
        weight = coefs[i]
        if abs(weight) > 0.001:
            selected_features.append((col, weight))
            print(f"  {col:20s}: {weight:.4f}")
            
    # Evaluate this combination
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    print(f"\nTraining Accuracy with elected features: {acc:.4f}")
    
    return selected_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--data", type=str, default="data/hallucination_calibration.json")
    args = parser.parse_args()
    
    run_feature_search(args.model, args.data)
