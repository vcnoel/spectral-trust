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

SCAR_SENTENCE = {
    "active": "The chaotic, unplanned, and poorly executed migration of the legacy database system caused the outage.",
    "passive": "The outage was caused by the chaotic, unplanned, and poorly executed migration of the legacy database system."
}

def measure_deltas():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    print(f"{'Model':<35} | {'Delta':<10} | {'Passive Fiedler':<15} | {'Status'}")
    print("-" * 80)

    for model_name in MODELS:
        try:
            config = GSPConfig(
                model_name=model_name,
                model_kwargs={"quantization_config": bnb_config},
                device="cuda",
                verbose=False
            )
            
            with GSPDiagnosticsFramework(config) as framework:
                framework.instrumenter.load_model(model_name)
                
                # Active
                res_a = framework.analyze_text(SCAR_SENTENCE['active'], save_results=False)
                # Passive
                res_p = framework.analyze_text(SCAR_SENTENCE['passive'], save_results=False)
                
                # Layers 2-5 (Standard Scar Window)
                # Note: Phi-1.5/2 are shallower (24/32 layers) vs Phi-3/4. 
                # But the scar is usually early-mid.
                # However, if Phi-4 is 40 layers, 2-5 is still very early.
                # Let's inspect 2-5 for consistency.
                
                f_a = [d.fiedler_value for d in res_a['layer_diagnostics']]
                f_p = [d.fiedler_value for d in res_p['layer_diagnostics']]
                
                # Compute avg over 2-5
                # Ensure we have enough layers
                end = min(len(f_a), 6)
                
                mu_a = np.mean(f_a[2:end])
                mu_p = np.mean(f_p[2:end])
                
                delta = mu_a - mu_p
                
                status = "ROBUST"
                if delta > 0.3: status = "SUPER SCAR"
                elif delta > 0.1: status = "SCARRED"
                elif delta > 0.05: status = "MINOR SCAR"
                
                print(f"{model_name:<35} | {delta:<10.4f} | {mu_p:<15.4f} | {status}")
                
        except Exception as e:
            print(f"{model_name:<35} | ERROR: {e}")

if __name__ == "__main__":
    measure_deltas()
