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
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

# --- Monkey Patch for Phi-3 Compatibility ---
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        def get_usable_length(self, input_length, layer_idx=None):
            return 0
        DynamicCache.get_usable_length = get_usable_length
    
    # Patch seen_tokens if missing (needed for some Phi-3 builds)
    if not hasattr(DynamicCache, 'seen_tokens'):
        @property
        def seen_tokens(self):
            return self.get_seq_length() if hasattr(self, 'get_seq_length') else 0
        DynamicCache.seen_tokens = seen_tokens
    
    if not hasattr(DynamicCache, 'get_max_length'):
        def get_max_length(self):
            return 2048 # Default for mini-4k is 4k but safe limit
        DynamicCache.get_max_length = get_max_length
        
except ImportError:
    pass
# ---------------------------------------------

def generate_validation_set(output_file="data/hallucination_validation_500.json"):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name} for generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype="auto"
    )
    
    prompts = [
        "Generate a factual statement about history.",
        "Generate a factual statement about biology.",
        "Generate a factual statement about physics.",
        "Generate a factual statement about geography.",
        "Generate a factual statement about technology."
    ]
    
    dataset = []
    
    # We need 250 pairs (500 total)
    # Strategy: Ask model to generate a fact and a specific lie.
    
    generation_prompt = """You are a dataset generator. Output 5 pairs of (True, False) statements. 
Format:
True: [Statement]
False: [Subtle modification making it false]
---
True: The sun is a star located at the center of the solar system.
False: The sun is a planet located at the center of the solar system.
---
True: Water is composed of two hydrogen atoms and one oxygen atom.
False: Water is composed of two oxygen atoms and one hydrogen atom.
---
"""
    
    # Generate until we have enough valid pairs
    target_count = 500 # 500 pairs = 1000 samples
    dataset = []
    
    pbar = tqdm(total=target_count, desc="Generating valid pairs")
    
    while len(dataset) < target_count * 2:
        input_ids = tokenizer(generation_prompt, return_tensors="pt").input_ids.to(model.device)
        
        try:
            outputs = model.generate(
                input_ids, 
                max_new_tokens=256, 
                temperature=0.9, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(input_ids)
            )
            
            generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Parse output
            lines = generated_text.split('\n')
            current_pair = {}
            for line in lines:
                line = line.strip()
                if line.startswith("True:") and len(line) > 10:
                    current_pair['true'] = line.replace("True:", "").strip()
                elif line.startswith("False:") and len(line) > 10:
                    current_pair['false'] = line.replace("False:", "").strip()
                    
                    if 'true' in current_pair and 'false' in current_pair:
                        t_text = current_pair['true']
                        f_text = current_pair['false']
                        
                        # VALIDITY CHECKS
                        if t_text != f_text and len(t_text) > 20 and len(f_text) > 20:
                            dataset.append({"text": t_text, "label": "true"})
                            dataset.append({"text": f_text, "label": "false"})
                            pbar.update(1)
                            if len(dataset) >= target_count * 2:
                                break
                        current_pair = {}
        except Exception as e:
            print(f"Error: {e}")
            continue

    pbar.close()
    print(f"Generated {len(dataset)} samples.")
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    return dataset

if __name__ == "__main__":
    generate_validation_set()
