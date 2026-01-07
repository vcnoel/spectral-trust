
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_features(framework, data, desc="Extracting"):
    all_rows = []
    labels = []
    
    for item in tqdm(data, desc=desc):
        text = item['text']
        label = 1 if item['label'] == 'true' else 0
        
        try:
            res = framework.analyze_text(text, save_results=False)
            metrics = res['layer_diagnostics']
            
            row = {}
            for i, m in enumerate(metrics):
                row[f'energy_L{i}'] = m.energy
                row[f'smoothness_L{i}'] = m.smoothness_index
                row[f'fiedler_L{i}'] = m.fiedler_value
                row[f'hfer_L{i}'] = m.hfer
                row[f'entropy_L{i}'] = m.spectral_entropy
            
            all_rows.append(row)
            labels.append(label)
        except Exception as e:
            print(f"Error: {e}")
            
    return pd.DataFrame(all_rows), pd.Series(labels)

def run_validation():
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
    
    
    # Load full dataset
    full_data = load_data(args.test_data)
    
    # Organize into pairs to ensure no leakage (True/False of same fact should stay together?)
    # Actually, for "Fact vs Hallucination", leakage is less of an issue if we just want to detect style.
    # But strictly, we should split by "Concept" (Pair).
    # Since data is [True, False, True, False...], we chunk by 2.
    pairs = [full_data[i:i+2] for i in range(0, len(full_data), 2)]
    
    # Shuffle pairs
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    # Split: 400 Pairs (800 samples) Train, 100 Pairs (200 samples) Test
    # If not enough data, just take 80% split
    split_idx = 400
    if len(pairs) < 500:
        print(f"Warning: Only {len(pairs)} pairs found. Using 80/20 split.")
        split_idx = int(len(pairs) * 0.8)
    
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:split_idx+100]
    
    # Flatten
    train_data = [item for sublist in train_pairs for item in sublist]
    test_data = [item for sublist in test_pairs for item in sublist]
    
    print(f"--- Data Split ---")
    print(f"Train: {len(train_data)} samples ({len(train_pairs)} pairs)")
    print(f"Test:  {len(test_data)} samples ({len(test_pairs)} pairs)")
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(args.model)
        
        print("\n1. Feature Extraction (Train)...")
        X_train, y_train = extract_features(framework, train_data, "Train")
        
        print("\n2. Feature Extraction (Test)...")
        X_test, y_test = extract_features(framework, test_data, "Test")
    
    # Train Lasso
    print("\n3. Training Lasso Model on Train Set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    print(f"Train Accuracy: {train_acc:.4f}")
    
    # Identify important features
    coefs = clf.coef_[0]
    feature_importance = []
    for i, col in enumerate(X_train.columns):
        if abs(coefs[i]) > 0.001:
            feature_importance.append((col, abs(coefs[i]), coefs[i]))
            
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop Selected Features:")
    top_features = []
    for name, mag, val in feature_importance[:10]:
        print(f"  {name:20s}: {val:.4f}")
        top_features.append(name)
        
    # Evaluate Full Model
    X_test_scaled = scaler.transform(X_test)
    test_preds = clf.predict(X_test_scaled)
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])
    print(f"\nFull Model Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    
    # --- Refined Model (Top 5) ---
    top_5 = [x[0] for x in feature_importance[:5]]
    print(f"\nTraining Refined Model on Top 5 Features: {top_5}")
    
    X_train_refined = X_train[top_5]
    X_test_refined = X_test[top_5]
    
    scaler_refined = StandardScaler()
    X_train_ref_scaled = scaler_refined.fit_transform(X_train_refined)
    X_test_ref_scaled = scaler_refined.transform(X_test_refined)
    
    clf_refined = LogisticRegression(class_weight='balanced')
    clf_refined.fit(X_train_ref_scaled, y_train)
    
    ref_preds = clf_refined.predict(X_test_ref_scaled)
    ref_probs = clf_refined.predict_proba(X_test_ref_scaled)[:, 1]
    ref_acc = accuracy_score(y_test, ref_preds)
    ref_auc = roc_auc_score(y_test, ref_probs)

    print("\n" + "="*40)
    print("FINAL VALIDATION RESULTS (Inverted Split)")
    print("="*40)
    print(f"Full Model Test Acc:   {accuracy_score(y_test, test_preds):.4f}")
    print(f"Refined Model Test Acc:{ref_acc:.4f}")
    print(f"Refined Model AUC:     {ref_auc:.4f}")
    print("-" * 40)
    
    # Save results
    res_df = pd.DataFrame({'label': y_test, 'pred': ref_preds, 'prob': ref_probs})
    res_df.to_csv("experiments/validation_results_refined.csv", index=False)

if __name__ == "__main__":
    run_validation()
