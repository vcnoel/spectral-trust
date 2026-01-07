
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def generate_agent_demotion():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

    # Prompt specifically designed to force "by [Agent]" structure
    prompt = """Generate 10 pairs of sentences demonstrating "Agent Demotion".
1. Active: [Agent] [Action] [Object] [Context]
2. Passive: [Object] [Action] [Context] by [Agent]
Ensure the "by [Agent]" phrase comes at the END of the passive sentence.
Format:
Active: [Sentence]
Passive: [Sentence]
---
Active: The furious storm destroyed the ancient lighthouse on the cliff.
Passive: The ancient lighthouse on the cliff was destroyed by the furious storm.
---
Active: The senior architect designed the innovative bridge spanning the river.
Passive: The innovative bridge spanning the river was designed by the senior architect.
---
"""
    dataset = []
    target = 100
    
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
                    
                    # Verify structure (Agent Demotion check)
                    if "by" in passive.lower() and len(active) > 20:
                        dataset.append({"active": active, "passive": passive})
                        pbar.update(1)
                        if len(dataset) >= target: break
                except:
                    continue
                    
    with open("data/agent_demotion_100.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Done. Generated data/agent_demotion_100.json")

if __name__ == "__main__":
    generate_agent_demotion()
