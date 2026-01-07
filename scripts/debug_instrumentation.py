
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_gpt2_outputs():
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print("\n--- Test 1: model(**inputs, output_attentions=True, return_dict=True) ---")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    print(f"Output Type: {type(outputs)}")
    print(f"Has attentions attribute? {hasattr(outputs, 'attentions')}")
    if hasattr(outputs, 'attentions'):
        print(f"Attentions value is None? {outputs.attentions is None}")
        if outputs.attentions is not None:
            print(f"Attentions type: {type(outputs.attentions)}")
            print(f"Attentions len: {len(outputs.attentions)}")
            
    print("\n--- Test 2: model(**inputs, output_attentions=True, return_dict=False) ---")
    with torch.no_grad():
        outputs_tuple = model(**inputs, output_attentions=True, return_dict=False)
        
    print(f"Output Type: {type(outputs_tuple)}")
    print(f"Tuple output keys/indices:")
    # Expected: (loss), logits, past_key_values, hidden_states, attentions
    # output_attentions=True -> returns attentions at end
    print(f"Length of tuple: {len(outputs_tuple)}")
    
    # Check if we can find attentions (list of tensors) in the tuple
    found_attentions = False
    for i, item in enumerate(outputs_tuple):
        if isinstance(item, (list, tuple)):
            # Attentions is usually a tuple of tensors
            if len(item) > 0 and isinstance(item[0], torch.Tensor) and item[0].ndim == 4:
                print(f"Item {i} looks like attentions: {type(item)} len={len(item)} shape[0]={item[0].shape}")
                found_attentions = True
    
    if not found_attentions:
        print("Could not identify attentions in tuple.")

if __name__ == "__main__":
    debug_gpt2_outputs()
