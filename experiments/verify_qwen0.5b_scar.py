
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def verify_qwen_scar():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path = "data/agent_demotion_100.json"
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Dataset not found! Make sure generation script finished.")
        return
        
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
        
        for item in tqdm(data, desc="Analyzing Qwen 0.5B Agent Demotion"):
            active_text = item['active']
            passive_text = item['passive']
            
            # Analyze Active
            res_a = framework.analyze_text(active_text, save_results=False)
            metrics_a = res_a['layer_diagnostics']
            
            # Analyze Passive
            res_p = framework.analyze_text(passive_text, save_results=False)
            metrics_p = res_p['layer_diagnostics']
            
            # Calculate Mean Fiedler in Layers 2-5 (The Scar Zone)
            f_a = np.mean([m.fiedler_value for m in metrics_a[2:6]])
            f_p = np.mean([m.fiedler_value for m in metrics_p[2:6]])
            
            delta = f_a - f_p # Positive = Drop in Passive
            deltas.append(delta)
            
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    
    print("\n" + "="*50)
    print("QWEN 0.5B AGENT DEMOTION ANALYSIS")
    print("="*50)
    print(f"Mean Fiedler Delta (Layers 2-5): {mean_delta:.4f}")
    print(f"Std Dev: {std_delta:.4f}")
    print("-" * 50)
    
    # Compare with Phi-3 Baseline (~0.12)
    if mean_delta > 0.05:
        print("CONCLUSION: SCARRED. (Similar to Phi-3)")
    elif mean_delta < -0.05:
        print("CONCLUSION: INVERTED SCAR (Spikes).")
    else:
        print("CONCLUSION: ROBUST. (No significant scar detected)")

if __name__ == "__main__":
    verify_qwen_scar()
