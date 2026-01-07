
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_experiment(model_name, data_path, output_csv="experiments/tuning_results.csv"):
    config = GSPConfig(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="float16" if torch.cuda.is_available() and "gpt2" not in model_name else "float32",
        normalization="sym",  # Symmetric normalized Laplacian usually best
        save_activations=False,
        trust_remote_code=True
    )
    
    data = load_data(data_path)
    results = []
    
    print(f"Running spectral analysis on {len(data)} samples using {model_name}...")
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(model_name)
        
        for item in tqdm(data):
            text = item['text']
            label = item['label']
            is_true = 1 if label == 'true' else 0
            
            # Analyze
            res = framework.analyze_text(text, save_results=False)
            metrics = res['layer_diagnostics'] # List of objects (one per layer)
            
            # Aggregate metrics (Focus on middle-to-late layers, e.g., last 50%)
            num_layers = len(metrics)
            start_layer = num_layers // 2
            
            # Extract mean metrics across selected layers
            mean_energy = np.mean([m.energy for m in metrics[start_layer:]])
            mean_smoothness = np.mean([m.smoothness_index for m in metrics[start_layer:]])
            mean_fiedler = np.mean([m.fiedler_value for m in metrics[start_layer:]])
            mean_hfer = np.mean([m.hfer for m in metrics[start_layer:]])
            mean_entropy = np.mean([m.spectral_entropy for m in metrics[start_layer:]])
            
            results.append({
                'text': text,
                'label': is_true,
                'energy': mean_energy,
                'smoothness': mean_smoothness,
                'fiedler': mean_fiedler,
                'hfer': mean_hfer,
                'entropy': mean_entropy
            })
            
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("\n--- Raw Data Stats ---")
    print(df.groupby('label').mean(numeric_only=True))
    return df

def tune_metrics(df):
    from sklearn.preprocessing import StandardScaler
    
    X = df[['energy', 'smoothness', 'fiedler', 'hfer', 'entropy']]
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Logistic Regression
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_scaled, y)
    
    preds = clf.predict(X_scaled)
    probs = clf.predict_proba(X_scaled)[:, 1]
    
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    
    print("\n--- Optimization Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    
    weights = dict(zip(X.columns, clf.coef_[0]))
    intercept = clf.intercept_[0]
    
    print("\nBest Linear Combination Coefficients (Scaled):")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    print("\nInterpretation:")
    print("Positive weight -> Higher value indicates TRUE")
    print("Negative weight -> Higher value indicates FAKE/HALLUCINATION")
    
    return weights, intercept

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--data", type=str, default="data/hallucination_calibration.json")
    args = parser.parse_args()
    
    df = run_experiment(args.model, args.data)
    weights, bias = tune_metrics(df)
    
    # Save weights for future use
    with open("experiments/best_weights.json", "w") as f:
        json.dump({"weights": weights, "bias": bias}, f, indent=4)
