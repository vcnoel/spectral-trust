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

import logging
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
)
from typing import Dict, Any, List
from .config import GSPConfig

logger = logging.getLogger(__name__)

try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        def get_usable_length(self, input_length, layer_idx=None):
            return 0
        DynamicCache.get_usable_length = get_usable_length
    if not hasattr(DynamicCache, 'from_legacy_cache'):
        @classmethod
        def from_legacy_cache(cls, past_key_values, *args, **kwargs):
            return cls()
        DynamicCache.from_legacy_cache = from_legacy_cache
        logger.info("Applied full DynamicCache compatibility patch.")
except ImportError:
    pass
# -----------------------------------------------------------

class LLMInstrumenter:
    """Instruments LLM to extract attention patterns and activations"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.attention_data = {}
        self.activation_data = {}
        self.hooks = []
        # Hook for head masking (Ablation)
        self.head_mask = None

    def ablate_head(self, layer_idx: int, head_idx: int):
        """
        Ablate a specific attention head by masking it during forward pass.
        This modifies the 'head_mask' passed to the model.
        """
        if self.head_mask is None:
            # Initialize mask with 1s: [num_layers, num_heads]
            cfg = self.model.config
            num_layers = self.config.num_layers_analyze or getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', 32))
            num_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head', 12))
            
            self.head_mask = torch.ones((num_layers, num_heads), device=self.config.device)
            
        # Set mask to 0 for specific head
        self.head_mask[layer_idx, head_idx] = 0.0
        logger.info(f"Ablated Head L{layer_idx}H{head_idx}")

    def reset_ablations(self):
        """Reset all head ablations"""
        self.head_mask = None
        logger.info("Reset all ablations")

    def load_model(self, model_name: str):
        """Load HuggingFace model and tokenizer with support for custom code"""
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer with trust_remote_code if needed
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=getattr(self.config, "trust_remote_code", False),
                local_files_only=getattr(self.config, "local_files_only", False)
            )
        except Exception as e:
            logger.warning(f"Falling back tokenizer load for {model_name}: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=getattr(self.config, "local_files_only", False)
            )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare dtype and device map
        dtype_str = getattr(self.config, "torch_dtype", "float32")
        dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str
        
        device_map = getattr(self.config, "device_map", None)
        if device_map is None and self.config.device != "auto":
             device_map = None # Should fallback to manually moving to device
        elif device_map is None:
             device_map = "auto"

        # Try loading as causal LM first
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation="eager",
                trust_remote_code=getattr(self.config, "trust_remote_code", False),
                local_files_only=getattr(self.config, "local_files_only", False),
                **getattr(self.config, "model_kwargs", {})
            )
        except Exception as e1:
            logger.warning(f"AutoModelForCausalLM failed for {model_name}: {e1}. Trying AutoModel...")
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    attn_implementation="eager",
                    trust_remote_code=getattr(self.config, "trust_remote_code", False),
                    local_files_only=getattr(self.config, "local_files_only", False),
                    **getattr(self.config, "model_kwargs", {})
                )
            except Exception as e2:
                logger.error(f"Failed to load model {model_name}: {e2}")
                raise

        # If not using device_map='auto' or similar offloading, move to device manually
        if device_map is None and self.config.device != "auto" and self.config.device != "cpu":
             self.model.to(self.config.device)
        
        self.model.eval()
        self.model.config.output_attentions = True
        self.model.config.return_dict = True
        try:
            logger.info(f"Model loaded successfully. Device: {next(self.model.parameters()).device}")
        except:
             logger.info(f"Model loaded successfully.")

    def register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        if not self.config.save_activations:
            return
        
        def create_activation_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Handle different output formats
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                self.activation_data[layer_name] = hidden_states.detach()
            return hook
        
        # Register hooks for transformer layers
        if hasattr(self.model, 'transformer'):
            # GPT-style models
            for i, layer in enumerate(self.model.transformer.h):
                hook = layer.register_forward_hook(create_activation_hook(f"layer_{i}"))
                self.hooks.append(hook)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama-style models
            for i, layer in enumerate(self.model.model.layers):
                hook = layer.register_forward_hook(create_activation_hook(f"layer_{i}"))
                self.hooks.append(hook)
        elif hasattr(self.model, 'encoder'):
            # BERT-style models
            for i, layer in enumerate(self.model.encoder.layer):
                hook = layer.register_forward_hook(create_activation_hook(f"layer_{i}"))
                self.hooks.append(hook)
    
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text through model and extract attention/activations
        Args:
            text: Input text string
        Returns:
            Dictionary containing processed results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Clear previous data
        self.attention_data.clear()
        self.activation_data.clear()
        
        # Prepare kwargs
        model_kwargs = {
            'output_attentions': True,
            'output_hidden_states': True,
            'return_dict': True 
        }
        
        # Pass head_mask if active
        if self.head_mask is not None:
            # head_mask needs to be passed. 
            model_kwargs['head_mask'] = self.head_mask
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, **model_kwargs)
        
        # Extract attention patterns
        if hasattr(outputs, 'attentions'):
             attentions = outputs.attentions
             hidden_states = outputs.hidden_states
        else:
             # Fallback for tuple output
             # GPT2 tuple: (loss, logits, past, hidden_states, attentions)
             attentions = outputs[-1]
             hidden_states = outputs[-2]
             attentions = outputs[-1]
             hidden_states = outputs[-2]
             logger.warning("Accessed attentions via tuple index (fallback).")
        
        if attentions is None:
             raise ValueError(f"Model returned None for attentions. Output type: {type(outputs)}. Keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
        
        return {
            'inputs': inputs,
            'attentions': attentions,
            'hidden_states': hidden_states,
            'activations': dict(self.activation_data),
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'text': text
        }
