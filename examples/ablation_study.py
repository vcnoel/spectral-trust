
"""
AI Safety Research: Mechanistic Interpretability via Spectral Ablation.

This script demonstrates the 'ablation_study' capability:
We identify and ablate specific attention heads to observe the collapse 
of spectral smoothness, verifying the head's role in maintaining 
coherent signal propagation.
"""

import argparse
import matplotlib.pyplot as plt
import torch
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

def run_ablation_study(model_name="gpt2"):
    print(f"Loading {model_name} for ablation study...")
    
    config = GSPConfig(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_plots=True
    )
    
    framework = GSPDiagnosticsFramework(config)
    framework.instrumenter.load_model(model_name)
    
    text = "The quick brown fox jumps over the lazy dog."
    
    # 1. Baseline Analysis
    print("\nRunning Baseline Analysis...")
    res_baseline = framework.analyze_text(text, save_results=False)
    
    # Get baseline smoothness for a middle layer
    target_layer = 5 if 'gpt2' in model_name else 10
    baseline_smoothness = res_baseline['layer_diagnostics'][target_layer].smoothness_index
    print(f"Baseline Smoothness (L{target_layer}): {baseline_smoothness:.4f}")
    
    # 2. Ablation Scan (Simple)
    # We ablate heads in the target layer and find the one that causes the biggest drop
    print(f"\nScanning heads in Layer {target_layer}...")
    num_heads = getattr(framework.instrumenter.model.config, 'num_attention_heads', 12)
    
    smoothness_drops = []
    
    # scan first 5 heads for demo speed
    heads_to_scan = range(min(num_heads, 5)) 
    
    for h in heads_to_scan:
        framework.instrumenter.ablate_head(target_layer, h)
        
        # Fast analysis (no saving)
        res_ablated = framework.analyze_text(text, save_results=False)
        s_ablated = res_ablated['layer_diagnostics'][target_layer].smoothness_index
        
        drop = baseline_smoothness - s_ablated
        smoothness_drops.append(drop)
        print(f"  Head {h}: Smoothness = {s_ablated:.4f} (Delta: {drop:+.4f})")
        
        # Reset for next
        framework.instrumenter.reset_ablations()
        
    # 3. Visualization
    plt.figure(figsize=(8, 5))
    plt.bar(heads_to_scan, smoothness_drops, color='red', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Impact of Head Ablation on Spectral Smoothness (Layer {target_layer})")
    plt.xlabel("Head Index")
    plt.ylabel("Smoothness Drop (Positive = Critical Head)")
    
    output_path = "examples/ablation_result.png"
    plt.savefig(output_path)
    print(f"\nAblation study plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()
    
    run_ablation_study(args.model)
