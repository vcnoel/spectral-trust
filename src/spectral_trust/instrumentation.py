import logging
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
)
from typing import Dict, Any, List
from .config import GSPConfig

logger = logging.getLogger(__name__)

# --- Monkey Patch for Phi-3 / Transformers Compatibility ---
# The remote code for Phi-3 uses 'get_usable_length' which was removed/missing in newer DynamicCache
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        def get_usable_length(self, input_length, layer_idx=None):
            # For this analysis tool, we always do full forward pass with no past cache.
            # Returning 0 ensures the model treats all inputs as new, preventing shape mismatches.
            return 0
        DynamicCache.get_usable_length = get_usable_length
        logger.info("Applied monkey-patch to DynamicCache for Phi-3 compatibility.")
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
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract attention patterns
        attentions = outputs.attentions  # Tuple of [batch, heads, seq_len, seq_len]
        hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim]
        
        return {
            'inputs': inputs,
            'attentions': attentions,
            'hidden_states': hidden_states,
            'activations': dict(self.activation_data),
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'text': text
        }
