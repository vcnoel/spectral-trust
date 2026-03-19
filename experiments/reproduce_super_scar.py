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


import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spectral_trust.framework import GSPDiagnosticsFramework
from spectral_trust.config import GSPConfig
from transformers import BitsAndBytesConfig

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct"
]

# The "Super Scar" Trigger
HEAVY_ACTIVE = "The chaotic, unplanned, and poorly executed migration of the legacy database system caused the outage."
HEAVY_PASSIVE = "The outage was caused by the chaotic, unplanned, and poorly executed migration of the legacy database system."

# The "Natural" Control
SIMPLE_ACTIVE = "The man reads a book."
SIMPLE_PASSIVE = "The book is read by the man."

CONDITIONS = {
    'Active Simple': SIMPLE_ACTIVE,
    'Passive Simple': SIMPLE_PASSIVE,
    'Active Heavy': HEAVY_ACTIVE,
    'Passive Heavy': HEAVY_PASSIVE
}

COLORS = {
    'Active Simple': 'green',
    'Passive Simple': 'limegreen', # Dashed
    'Active Heavy': 'blue',
    'Passive Heavy': 'red'     # Dashed
}

STYLES = {
    'Active Simple': '-',
    'Passive Simple': '--',
    'Active Heavy': '-',
    'Passive Heavy': '--'
}

def run_investigation():
    output_dir = Path("gsp_results/complexity_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    for model_name in MODELS:
        print(f"\nAnalyzing {model_name}...")
        try:
            config = GSPConfig(
                model_name=model_name,
                model_kwargs={"quantization_config": bnb_config},
                device="cuda",
                verbose=False
            )
            
            results = {}
            
            with GSPDiagnosticsFramework(config) as framework:
                framework.instrumenter.load_model(model_name)
                
                for label, text in CONDITIONS.items():
                    print(f"  Trace: {label}...")
                    res = framework.analyze_text(text, save_results=False)
                    results[label] = res['layer_diagnostics']
            
            # Plotting
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Complexity vs Structure: {model_name.split('/')[-1]}", fontsize=16)
            axes = axes.flatten()
            
            metrics = {
                'Fiedler Value': 'fiedler_value',
                'HFER': 'hfer',
                'Spectral Entropy': 'spectral_entropy',
                'Smoothness Index': 'smoothness_index' # Swapped Energy for Smoothness
            }
            
            for ax, (metric_name, metric_attr) in zip(axes, metrics.items()):
                for label in CONDITIONS.keys():
                    data = [getattr(d, metric_attr) for d in results[label]]
                    layers = range(len(data))
                    
                    ax.plot(layers, data, 
                            label=label, 
                            color=COLORS[label], 
                            linestyle=STYLES[label], 
                            linewidth=2.5 if 'Heavy' in label else 1.5,
                            alpha=1.0 if 'Heavy' in label else 0.6)
                
                ax.set_title(metric_name)
                ax.set_xlabel("Layer")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            save_path = output_dir / f"{model_name.replace('/', '_')}_complexity.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot to {save_path}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")

if __name__ == "__main__":
    run_investigation()
