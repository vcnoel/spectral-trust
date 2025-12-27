#!/usr/bin/env python3
"""
Command Line Interface for Spectral Trust Framework (formerly GSP LLM Diagnostics)
Provides easy-to-use CLI for running various analysis tasks with modern language models
"""

import argparse
import json
import logging
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import torch

from .framework import GSPDiagnosticsFramework
from .config import GSPConfig
from .spectral import SpectralDiagnostics

# --- UTF-8 console safety on Windows ---
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modern model configurations with optimal settings
MODERN_MODELS = {
    # Llama models
    'llama-3.2-1b': {
        'name': 'meta-llama/Llama-3.2-1B',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.2-3b': {
        'name': 'meta-llama/Llama-3.2-3B',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.2-3b-instruct': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'max_length': 2048,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.1-8b': {
        'name': 'meta-llama/Meta-Llama-3.1-8B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.1-8b-instruct': {
        'name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'llama-3.1-70b': {
        'name': 'meta-llama/Meta-Llama-3.1-70B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    
    # Mistral models
    'mistral-7b': {
        'name': 'mistralai/Mistral-7B-v0.1',
        'max_length': 8192,
        'torch_dtype': 'float16'
    },
    'mistral-7b-instruct': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'max_length': 8192,
        'torch_dtype': 'float16'
    },
    'mixtral-8x7b': {
        'name': 'mistralai/Mixtral-8x7B-v0.1',
        'max_length': 8192,
        'torch_dtype': 'float16',
        'device_map': 'auto'
    },
    
    # Qwen models
    'qwen2-7b': {
        'name': 'Qwen/Qwen2-7B',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    'qwen2.5-7b': {
        'name': 'Qwen/Qwen2.5-7B',
        'max_length': 8192,
        'trust_remote_code': True,
        'torch_dtype': 'float16'
    },
    
    # Gemma models
    'gemma-2b': {
        'name': 'google/gemma-2b',
        'max_length': 2048,
        'torch_dtype': 'float16'
    },
    'gemma-7b': {
        'name': 'google/gemma-7b',
        'max_length': 2048,
        'torch_dtype': 'float16'
    },
    'gemma2-9b': {
        'name': 'google/gemma-2-9b',
        'max_length': 4096,
        'torch_dtype': 'float16'
    },

    # Phi models
    'phi-3-mini': {
        'name': 'microsoft/Phi-3-mini-4k-instruct',
        'max_length': 4096,
        'trust_remote_code': True,
        'torch_dtype': 'float16',
        'model_kwargs': {'attn_implementation': 'eager'}
    },
    
    # Legacy models
    'gpt2': {
        'name': 'gpt2',
        'max_length': 1024,
        'torch_dtype': 'float32'
    },
}

def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model configuration by key or return custom config for unknown models"""
    if model_key in MODERN_MODELS:
        return MODERN_MODELS[model_key]
    else:
        # Assume it's a HuggingFace model path
        return {
            'name': model_key,
            'max_length': 2048,
            'torch_dtype': 'float16',
            'trust_remote_code': False
        }

@dataclass
class HeadSpec:
    layer: int
    heads: List[int]

class PatchResidualAt:
    """
    Forward hook to patch residual slice at a layer and token window.
    Assumes transformer blocks are at model.model.layers[idx].
    """
    def __init__(self, t_lo: int, t_hi: int, donor_hidden: torch.Tensor):
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.donor_hidden = donor_hidden  # (1, seq, dim)

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            hs = out[0]; rest = out[1:]
        else:
            hs = out; rest = None
        L = hs.shape[1]
        a = max(0, min(self.t_lo, L))
        b = max(0, min(self.t_hi, L))
        if a < b:
            hs = hs.clone()
            hs[:, a:b, :] = self.donor_hidden[:, a:b, :].to(hs.device)
        return (hs, *rest) if rest is not None else hs

def parse_heads(spec: Optional[str]) -> List[HeadSpec]:
    if not spec or not spec.strip():
        return []
    out: List[HeadSpec] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        L_str, hs = chunk.split(":")
        L = int(L_str.strip())
        idxs = []
        for tok in hs.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-"); a, b = int(a), int(b)
                idxs.extend(range(a, b + 1))
            else:
                idxs.append(int(tok))
        out.append(HeadSpec(L, sorted(set(idxs))))
    return out

def _attn_module(hf_model, layer_idx: int):
    """
    Return the attention module for a decoder block across families.
    """
    block_container = getattr(hf_model, "model", hf_model)
    block = block_container.layers[layer_idx]
    for name in ("self_attn", "attention", "attn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise RuntimeError(f"Layer {layer_idx} has no attention module")

def _attach_attn_output_head_zero_hook(hf_model, specs: List[HeadSpec]) -> List:
    """
    Forward-hook on the ATTENTION MODULE OUTPUT to zero selected heads.
    """
    handles = []
    nH = hf_model.config.num_attention_heads
    Hd = hf_model.config.hidden_size // nH

    by_layer = {}
    for s in specs:
        valid = [h for h in s.heads if 0 <= h < nH]
        if valid:
            by_layer.setdefault(s.layer, sorted(set(valid)))

    if not by_layer:
        return handles

    for layer_idx, heads in by_layer.items():
        attn = _attn_module(hf_model, layer_idx)
        head_slices = [(h * Hd, (h + 1) * Hd) for h in heads]

        def hook(module, inputs, output, head_slices=head_slices, Hd=Hd):
            y = output
            if isinstance(y, tuple):
                y = y[0]
            if y is None:
                return output
            for a, b in head_slices:
                y[..., a:b] = 0
            return y

        handles.append(attn.register_forward_hook(hook))
    return handles

def setup_model_for_analysis(hf_model):
    """
    Setup model configuration for better compatibility with GSP analysis.
    """
    if hasattr(hf_model, "config"):
        hf_model.config.output_attentions = True
        hf_model.config.output_hidden_states = True
        try:
            hf_model.config.attn_implementation = "eager"
        except Exception:
            pass
    return hf_model

def safe_tokenize_with_fallback(tokenizer, text: str, **kwargs):
    """
    Safely tokenize text with fallbacks.
    """
    enc = tokenizer(text, **kwargs)
    if (enc.get("input_ids", None) is None or 
        enc["input_ids"].numel() == 0 or 
        enc["input_ids"].shape[1] == 0):
        
        candidates = [text.strip(), " " + text.strip(), text.strip() + "."]
        for candidate in candidates:
            if not candidate: continue
            try:
                enc = tokenizer(candidate, **kwargs)
                if (enc.get("input_ids", None) is not None and enc["input_ids"].numel() > 0):
                    return enc, candidate
            except Exception:
                continue
        raise RuntimeError("Failed to tokenize any variant of the input text")
    return enc, text

def generate_response(hf_model, hf_tokenizer, prompt_text, max_new_tokens=20):
    """Generate response with minimal risk of CUDA assertion errors"""
    try:
        inputs = hf_tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
        model_device = next(hf_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = hf_model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=hf_tokenizer.eos_token_id,
                use_cache=False
            )
        
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        logger.warning(f"Generation failed: {e}")
        return ""

def extract_answer(text: str) -> str:
    if not text: return "unknown"
    return text.strip() # Simplified for brevity, original regex logic can be added if needed

def create_model_config(model_key: str, args) -> GSPConfig:
    model_info = get_model_info(model_key)
    if hasattr(args, 'device') and args.device != 'auto':
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_kwargs = {
        "output_attentions": True,
        "output_hidden_states": True,
    }
    # Merge model-specific kwargs
    if 'model_kwargs' in model_info:
        model_kwargs.update(model_info['model_kwargs'])

    return GSPConfig(
        model_name=model_info['name'],
        max_length=getattr(args, 'max_length', model_info['max_length']),
        device=device,
        trust_remote_code=model_info.get('trust_remote_code', False),
        torch_dtype=model_info.get('torch_dtype', 'float16'),
        output_dir=getattr(args, 'output_dir', './gsp_results'),
        save_plots=getattr(args, 'save_plots', False),
        verbose=getattr(args, 'verbose', False),
        num_layers_analyze=getattr(args, 'num_layers', None),
        num_eigenvalues=getattr(args, 'num_eigenvalues', 100),
        eigen_solver=getattr(args, 'eigen_solver', 'sparse'),
        head_aggregation=getattr(args, 'head_aggregation', 'uniform'),
        symmetrization=getattr(args, 'symmetrization', 'symmetric'),
        normalization=getattr(args, 'normalization', 'rw'),
        hfer_cutoff_ratio=getattr(args, 'hfer_cutoff', 0.1),
        remove_self_loops=getattr(args, 'remove_self_loops', False),
        local_files_only=getattr(args, 'offline', False),
        runs=getattr(args, 'runs', 1),
        temperature=getattr(args, 'temperature', 1.0),
        plot_metrics=getattr(args, 'plot', None),
        latex_export=getattr(args, 'latex', False),
        model_kwargs=model_kwargs
    )

def cmd_analyze(args):
    """Analyze (single or batch), generate a model answer, score vs gold, and log per-layer metrics."""
    logger.info(f"Starting analysis with model: {args.model}")
    config = create_model_config(args.model, args)
    
    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding='utf-8')
    
    if not text:
        logger.error("No text provided")
        return

    try:
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(config.model_name)
            framework.instrumenter.register_hooks()
            
            hf_model = setup_model_for_analysis(framework.instrumenter.model)
            hf_tokenizer = framework.instrumenter.tokenizer
            
            # Head ablation hooks
            handles = []
            if args.ablate_heads:
                specs = parse_heads(args.ablate_heads)
                handles = _attach_attn_output_head_zero_hook(hf_model, specs)
            
            # Analyze
            all_run_results = []
            for run_idx in range(config.runs):
                logger.info(f"Running analysis {run_idx+1}/{config.runs}...")
                run_result = framework.analyze_text(text)
                all_run_results.append(run_result)
            
            # Show output (last run)
            if args.emit_text:
                resp = generate_response(hf_model, hf_tokenizer, text)
                logger.info(f"Response: {resp}")
            
            # Print Summary (Mean of runs)
            if all_run_results:
                print("\nResults (Mean across runs):")
                # Compute mean metrics for display
                import numpy as np
                mean_metrics = {}
                num_layers = len(all_run_results[0]['layer_diagnostics'])
                
                headers = f"{'Layer':>5} {'Energy':>10} {'HFER':>10} {'Entropy':>10} {'Fiedler':>10} {'Smoothness':>10}"
                print(headers)
                print("-" * len(headers))
                
                for l in range(num_layers):
                    metrics = {'energy': [], 'hfer': [], 'spectral_entropy': [], 'fiedler_value': [], 'smoothness_index': []}
                    for res in all_run_results:
                        d = res['layer_diagnostics'][l]
                        for k in metrics:
                            metrics[k].append(getattr(d, k))
                    
                    means = {k: np.mean(v) for k, v in metrics.items()}
                    print(f"{l:5d} {means['energy']:10.4f} {means['hfer']:10.4f} {means['spectral_entropy']:10.4f} {means['fiedler_value']:10.4f} {means['smoothness_index']:10.4f}")
                
                # Visualize multiple runs
                if config.save_plots and len(all_run_results) > 0:
                     framework.visualize_multi_run(all_run_results)
            
            for h in handles: h.remove()
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
def cmd_compare(args):
    """Compare two texts or two models."""
    import torch
    import gc
    
    # Logic:
    # 1. Compare text1 vs text2 on model1
    # 2. Compare text1 on model1 vs model2 (text2 defaults to text1 if not set)
    # 3. Compare text1 on model1 vs text2 on model2
    
    # Defaults
    text1 = args.text1
    text2 = args.text2 if args.text2 else text1
    model1_key = args.model
    model2_key = args.model2 if args.model2 else model1_key
    
    if model1_key == model2_key and text1 == text2:
        logger.error("Nothing to compare! Please provide different texts or different models.")
        return

    logger.info(f"Comparing:")
    logger.info(f"  A: Model={model1_key}, Text='{text1[:30]}...'")
    logger.info(f"  B: Model={model2_key}, Text='{text2[:30]}...'")

    try:
        # --- Run Analysis 1 ---
        logger.info(f"Running Analysis A with {model1_key}...")
        config1 = create_model_config(model1_key, args)
        res1 = None
        
        # Use a scope to ensure cleanup
        with GSPDiagnosticsFramework(config1) as framework1:
            framework1.instrumenter.load_model(config1.model_name)
            framework1.instrumenter.register_hooks()
            setup_model_for_analysis(framework1.instrumenter.model)
            res1 = framework1.analyze_text(text1, save_results=False)
            
            # Explicit cleanup to free VRAM
            del framework1.instrumenter.model
            del framework1.instrumenter.tokenizer
            framework1.instrumenter.cleanup_hooks()
            
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # --- Run Analysis 2 ---
        logger.info(f"Running Analysis B with {model2_key}...")
        args.plot = getattr(args, 'plot', ['all']) # Ensure it exists if called programmatically
        config2 = create_model_config(model2_key, args)
        res2 = None
        
        with GSPDiagnosticsFramework(config2) as framework2:
            framework2.instrumenter.load_model(config2.model_name)
            framework2.instrumenter.register_hooks()
            setup_model_for_analysis(framework2.instrumenter.model)
            res2 = framework2.analyze_text(text2, save_results=False)
            
            # Visualize comparison using the second framework (or first, doesn't matter)
            framework2.visualize_comparison(res1, res2)
            
            if args.latex:
                framework2.save_comparison_latex_data(res1, res2)
            
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

def cmd_list_models(args):
    print("Available Models:")
    for k, v in MODERN_MODELS.items():
        print(f"  {k:20} -> {v['name']}")

def main():
    parser = argparse.ArgumentParser(description="Spectral Trust CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Analyze
    p_analyze = subparsers.add_parser('analyze')
    p_analyze.add_argument('--text', type=str)
    p_analyze.add_argument('--text_file', type=str)
    p_analyze.add_argument('--model', type=str, default='llama-3.2-1b')
    p_analyze.add_argument('--device', type=str, default='auto')
    p_analyze.add_argument('--output_dir', type=str, default='./results')
    p_analyze.add_argument('--no-plots', action='store_false', dest='save_plots', help='Disable plotting')
    p_analyze.set_defaults(save_plots=True)
    p_analyze.add_argument('--verbose', action='store_true')
    p_analyze.add_argument('--emit_text', action='store_true')
    p_analyze.add_argument('--ablate_heads', type=str)
    
    # GSP params
    p_analyze.add_argument('--head_aggregation', type=str, default='uniform')
    p_analyze.add_argument('--symmetrization', type=str, default='symmetric')
    p_analyze.add_argument('--normalization', type=str, default='rw')
    p_analyze.add_argument('--hfer_cutoff', type=float, default=0.1)
    p_analyze.add_argument('--remove_self_loops', action='store_true', help='Remove self-loops (diagonal) from adjacency matrix')
    p_analyze.add_argument('--offline', action='store_true', help='Use local files only (no download)')
    p_analyze.add_argument('--runs', type=int, default=1, help='Number of runs')
    p_analyze.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation/sampling')
    p_analyze.add_argument('--plot', nargs='+', default=['all'], help='Metrics to plot: all, energy, hfer, entropy, fiedler, smoothness')
    p_analyze.add_argument('--latex', action='store_true', help='Export metrics to LaTeX-friendly .dat files')
    
    # Compare
    p_compare = subparsers.add_parser('compare')
    p_compare.add_argument('--text1', type=str, required=True)
    p_compare.add_argument('--text2', type=str, help='Second text (optional, defaults to text1)')
    p_compare.add_argument('--model', type=str, default='llama-3.2-1b', help='Primary model')
    p_compare.add_argument('--model2', type=str, help='Secondary model (optional)')
    p_compare.add_argument('--device', type=str, default='auto')
    p_compare.add_argument('--output_dir', type=str, default='./results')
    p_compare.add_argument('--no-plots', action='store_false', dest='save_plots', help='Disable plotting')
    p_compare.set_defaults(save_plots=True)
    p_compare.add_argument('--verbose', action='store_true')
    # GSP params for compare
    p_compare.add_argument('--head_aggregation', type=str, default='uniform')
    p_compare.add_argument('--symmetrization', type=str, default='symmetric')
    p_compare.add_argument('--normalization', type=str, default='rw')
    p_compare.add_argument('--hfer_cutoff', type=float, default=0.1)
    p_compare.add_argument('--remove_self_loops', action='store_true', help='Remove self-loops (diagonal) from adjacency matrix')
    p_compare.add_argument('--plot', nargs='+', default=['all'], choices=['all', 'energy', 'hfer', 'entropy', 'fiedler', 'smoothness'], help='Metrics to plot')
    p_compare.add_argument('--latex', action='store_true', help='Export metrics to LaTeX-friendly .dat files')
    p_compare.add_argument('--offline', action='store_true')
    
    # List models
    subparsers.add_parser('list-models')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'list-models':
        cmd_list_models(args)
    elif args.command == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
