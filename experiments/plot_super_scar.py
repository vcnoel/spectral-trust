
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from spectral_trust.framework import GSPDiagnosticsFramework
from spectral_trust.config import GSPConfig
from transformers import BitsAndBytesConfig

MODELS = [
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-4"
]

# Top Scar Triggers from Gen 2
SCAR_SENTENCES = [
    {
        "id": "heavy_agent_gen2_3",
        "active": "The chaotic, unplanned, and poorly executed migration of the legacy database system caused the outage.",
        "passive": "The outage was caused by the chaotic, unplanned, and poorly executed migration of the legacy database system."
    },
    {
        "id": "heavy_agent_2_mutation_A",
        "active": "The constant, rhythmic, soothing sound of the waves crashing against the shore calmed my mind.",
        "passive": "My mind was calmed by the constant, rhythmic, soothing sound of the waves crashing against the shore."
    }
]

ACTIVE_NATURAL = "The committee reviewed the proposal about the environmental regulations."

def plot_comparative_metrics(model_results, output_dir):
    """
    Plot 4 metrics: Fiedler, HFER, Entropy, Energy
    Comparing: Active Natural vs Active Scar vs Passive Scar
    """
    model_name = model_results['model_name']
    
    # Extract data for each sentence type
    data = {} # type -> {metric -> [values]}
    
    types = ['Active Natural', 'Active Scar', 'Passive Scar']
    input_keys = ['active_natural', 'active_scar', 'passive_scar']
    
    colors = {'Active Natural': 'green', 'Active Scar': 'blue', 'Passive Scar': 'red'}
    linestyles = {'Active Natural': '--', 'Active Scar': '-', 'Passive Scar': '-'}
    
    metrics = {
        'Fiedler Value': 'fiedler_value',
        'HFER': 'hfer',
        'Spectral Entropy': 'spectral_entropy',
        'Dirichlet Energy': 'energy'
    }
    
    # Process each scar sample separatey
    for sample_idx, sample_res in enumerate(model_results['samples']):
        plot_path = output_dir / f"{model_name.replace('/', '_')}_sample_{sample_idx}.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{model_name} - {sample_res['id']}", fontsize=16)
        
        axes = axes.flatten()
        
        for ax, (metric_name, metric_key) in zip(axes, metrics.items()):
            for type_label, res_key in zip(types, input_keys):
                res = sample_res[res_key]
                values = [getattr(d, metric_key) for d in res['layer_diagnostics']]
                layers = range(len(values))
                
                ax.plot(layers, values, 
                        label=type_label, 
                        color=colors[type_label], 
                        linestyle=linestyles[type_label], 
                        linewidth=2 if 'Passive' in type_label else 1.5)
            
            ax.set_title(metric_name)
            ax.set_xlabel("Layer")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Highlight Fiedler Drop
            if metric_key == 'fiedler_value':
                # Check for drop
                pass_f = [getattr(d, 'fiedler_value') for d in sample_res['passive_scar']['layer_diagnostics']]
                act_f = [getattr(d, 'fiedler_value') for d in sample_res['active_scar']['layer_diagnostics']]
                # Simple delta annotation on max diff layer
                deltas = np.array(act_f) - np.array(pass_f)
                max_d_idx = np.argmax(deltas)
                if deltas[max_d_idx] > 0.1:
                    ax.annotate(f"Delta: {deltas[max_d_idx]:.2f}", 
                                xy=(max_d_idx, pass_f[max_d_idx]), 
                                xytext=(max_d_idx, pass_f[max_d_idx]-0.2),
                                arrowprops=dict(facecolor='black', shrink=0.05))

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

def run_cross_model_analysis():
    output_dir = Path("gsp_results/super_scar_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4-bit config for all to be safe/consistent
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    for model_name in MODELS:
        print(f"\nProcessing {model_name}...")
        
        # Phi-1.5 and Phi-2 are small enough for fp16, but consistency is key.
        # Actually 1.5/2 might not support 4bit gracefully or might be fast enough without.
        # Let's use 4-bit for >3B models (Phi-3.5, Phi-4). Phi-3-mini is 3.8B.
        # Let's just use 4-bit for all.
        
        try:
            config = GSPConfig(
                model_name=model_name,
                model_kwargs={"quantization_config": bnb_config},
                device="cuda",
                num_layers_analyze=None # Analyze all layers
            )
            
            model_results = {'model_name': model_name, 'samples': []}
            
            with GSPDiagnosticsFramework(config) as framework:
                framework.instrumenter.load_model(model_name)
                
                # Pre-compute Active Natural baseline
                print("  Analyzing Baseline...")
                res_natural = framework.analyze_text(ACTIVE_NATURAL, save_results=False)
                
                for sample in SCAR_SENTENCES:
                    print(f"  Analyzing Sample: {sample['id']}...")
                    res_active = framework.analyze_text(sample['active'], save_results=False)
                    res_passive = framework.analyze_text(sample['passive'], save_results=False)
                    
                    model_results['samples'].append({
                        "id": sample['id'],
                        "active_natural": res_natural,
                        "active_scar": res_active,
                        "passive_scar": res_passive
                    })
            
            plot_comparative_metrics(model_results, output_dir)
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_cross_model_analysis()
