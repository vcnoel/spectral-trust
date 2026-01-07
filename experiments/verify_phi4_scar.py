
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from spectral_trust.config import GSPConfig
from spectral_trust.framework import GSPDiagnosticsFramework

def verify_phi4_scar():
    model_name = "microsoft/phi-4"
    dataset_path = "data/agent_demotion_100.json"
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Dataset not found!")
        return
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    config = GSPConfig(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="float16",
        normalization="sym",
        save_activations=False,
        device_map="auto",
        model_kwargs={"quantization_config": bnb_config}
    )
    
    deltas = []
    
    try:
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(model_name)
            
            for item in tqdm(data, desc="Analyzing Phi-4 Agent Demotion"):
                active_text = item['active']
                passive_text = item['passive']
                
                # Analyze Active
                res_a = framework.analyze_text(active_text, save_results=False)
                metrics_a = res_a['layer_diagnostics']
                
                # Analyze Passive
                res_p = framework.analyze_text(passive_text, save_results=False)
                metrics_p = res_p['layer_diagnostics']
                
                # Calculate Mean Fiedler in Layers 2-5
                # Note: Phi-4 is deeper (40 layers), but the signature is usually early.
                # We stick to 2-5 for consistency, but might check broader range if needed.
                f_a = np.mean([m.fiedler_value for m in metrics_a[2:6]])
                f_p = np.mean([m.fiedler_value for m in metrics_p[2:6]])
                
                delta = f_a - f_p 
                deltas.append(delta)
    except Exception as e:
        print(f"Error loading or running Phi-4: {e}")
        return
            
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    
    print("\n" + "="*50)
    print("PHI-4 AGENT DEMOTION ANALYSIS")
    print("="*50)
    print(f"Mean Fiedler Delta (Layers 2-5): {mean_delta:.4f}")
    print(f"Std Dev: {std_delta:.4f}")
    print("-" * 50)
    
    if mean_delta > 0.05:
        print("CONCLUSION: SCAR PERSISTS. (Still vulnerable)")
    elif mean_delta < -0.05:
        print("CONCLUSION: INVERTED SCAR.")
    else:
        print("CONCLUSION: SCAR HEALED. (Robust < 0.05)")

if __name__ == "__main__":
    verify_phi4_scar()
