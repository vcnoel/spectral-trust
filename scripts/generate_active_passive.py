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

def generate_pairs():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

    prompt = """Generate 10 distinct, complex, diverse English sentences in Active Voice, and convert each to Passive Voice.
Format:
Active: [Sentence]
Passive: [Sentence]
---
Active: The chef prepared a gourmet meal for the distinguished guests.
Passive: A gourmet meal was prepared for the distinguished guests by the chef.
---
"""
    dataset = []
    target = 200
    
    pbar = tqdm(total=target)
    
    while len(dataset) < target:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs.input_ids, max_new_tokens=512, temperature=0.9, do_sample=True)
        text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        pairs = text.split('---')
        for p in pairs:
            if "Active:" in p and "Passive:" in p:
                try:
                    active = p.split("Active:")[1].split("Passive:")[0].strip()
                    passive = p.split("Passive:")[1].strip()
                    if len(active) > 20 and len(passive) > 20:
                        dataset.append({"active": active, "passive": passive})
                        pbar.update(1)
                        if len(dataset) >= target: break
                except:
                    continue
                    
    with open("data/active_passive_200.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    generate_pairs()
