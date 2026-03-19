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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_features(framework, data, desc="Extracting"):
    features = []
    labels = []
    
    for item in tqdm(data, desc=desc):
        text = item['text']
        label = 1 if item['label'] == 'true' else 0
        
        try:
             # Analyze
            res = framework.analyze_text(text, save_results=False)
            metrics = res['layer_diagnostics'] # List of objects
            
            # Aggregate metrics (Focus on middle-to-late layers)
            num_layers = len(metrics)
            start_layer = num_layers // 2
            
            row = {
                'energy': np.mean([m.energy for m in metrics[start_layer:]]),
                'smoothness': np.mean([m.smoothness_index for m in metrics[start_layer:]]),
                'fiedler': np.mean([m.fiedler_value for m in metrics[start_layer:]]),
                'hfer': np.mean([m.hfer for m in metrics[start_layer:]]),
                'entropy': np.mean([m.spectral_entropy for m in metrics[start_layer:]])
            }
            features.append(row)
            labels.append(label)
        except Exception as e:
            print(f"Error processing '{text}': {e}")
            
    return pd.DataFrame(features), pd.Series(labels)

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--train_data", type=str, default="data/hallucination_calibration.json")
    parser.add_argument("--test_data", type=str, default="data/hallucination_validation_500.json")
    args = parser.parse_args()

    config = GSPConfig(
        model_name=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="float16" if torch.cuda.is_available() and "gpt2" not in args.model else "float32",
        normalization="sym",
        save_activations=False,
        trust_remote_code=True
    )
    
    train_data = load_data(args.train_data)
    test_data = load_data(args.test_data)[:500] # Cap at 500
    
    print(f"--- Hallucination Detection Benchmark ---")
    print(f"Model: {args.model}")
    print(f"Train Set: {len(train_data)} samples")
    print(f"Test Set: {len(test_data)} samples")
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(args.model)
        
        print("\n1. Feature Extraction (Train)...")
        X_train, y_train = extract_features(framework, train_data, "Train")
        
        print("\n2. Feature Extraction (Test)...")
        X_test, y_test = extract_features(framework, test_data, "Test")
        
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("\n3. Training Logistic Regression...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_preds = clf.predict(X_test_scaled)
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    print("-" * 40)
    
    weights = dict(zip(X_train.columns, clf.coef_[0]))
    print("Optimization Function Weights:")
    for k, v in weights.items():
        print(f"  {k:12s}: {v:.4f}")
        
    # Save results
    results_df = X_test.copy()
    results_df['label'] = y_test
    results_df['pred'] = test_preds
    results_df['prob'] = test_probs
    results_df.to_csv("experiments/final_benchmark_results.csv", index=False)
    print("\nDetailed results saved to experiments/final_benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
