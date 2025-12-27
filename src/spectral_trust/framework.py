import logging
import json
import pickle
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch

from .config import GSPConfig
from .instrumentation import LLMInstrumenter
from .graph import GraphConstructor
from .spectral import SpectralAnalyzer

logger = logging.getLogger(__name__)

class GSPDiagnosticsFramework:
    """Main framework for GSP-based LLM diagnostics"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
        self.instrumenter = LLMInstrumenter(config)
        self.graph_constructor = GraphConstructor(config)
        self.spectral_analyzer = SpectralAnalyzer(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(Path(config.output_dir) / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'instrumenter'):
            self.instrumenter.cleanup_hooks()
    
    def analyze_text(self, text: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Perform complete GSP analysis on input text
        Args:
            text: Input text to analyze
            save_results: Whether to save intermediate results
        Returns:
            Complete analysis results
        """
        logger.info(f"Analyzing text: {text[:100]}...")
        
        # Process text through model
        model_outputs = self.instrumenter.process_text(text)
        attentions = model_outputs['attentions']
        hidden_states = model_outputs['hidden_states']
        
        # Analyze each layer
        layer_diagnostics = []
        num_layers = len(attentions)
        
        if self.config.num_layers_analyze is not None:
            num_layers = min(num_layers, self.config.num_layers_analyze)
        
        for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
            # Get layer attention and hidden states
            attention = attentions[layer_idx]  # [batch, heads, seq_len, seq_len]
            signals = hidden_states[layer_idx + 1]  # [batch, seq_len, hidden_dim]
            
            # Remove batch dimension (assuming batch_size=1)
            attention = attention.squeeze(0)  # [heads, seq_len, seq_len]
            signals = signals.squeeze(0)  # [seq_len, hidden_dim]
            
            attention_sym = self.graph_constructor.symmetrize_attention(
                attention.unsqueeze(0)
            ).squeeze(0)  # [heads, seq_len, seq_len]

            # NEW: pull the model's attention mask (if present)
            inputs_mask = model_outputs['inputs'].get('attention_mask', None)
            if inputs_mask is not None:
                inputs_mask = inputs_mask.to(torch.bool)  # [1, Q]

            adjacency = self.graph_constructor.aggregate_heads(
                attention_sym.unsqueeze(0),        # [1, H, Q, K]
                attn_mask=inputs_mask              # [1, Q] or None
            ).squeeze(0)                           # [Q, K]

            laplacian = self.graph_constructor.construct_laplacian(
                adjacency.unsqueeze(0)
            ).squeeze(0)  # [seq_len, seq_len]
            
            # Perform spectral analysis
            diagnostics = self.spectral_analyzer.analyze_layer(
                signals, laplacian, layer_idx
            )
            
            layer_diagnostics.append(diagnostics)
            
            if self.config.verbose:
                logger.info(f"Layer {layer_idx}: Energy={diagnostics.energy:.4f}, "
                          f"SMI={diagnostics.smoothness_index:.4f}, "
                          f"SE={diagnostics.spectral_entropy:.4f}, "
                          f"HFER={diagnostics.hfer:.4f}")
        
        # Compile results
        results = {
            'text': text,
            'tokens': model_outputs['tokens'],
            'layer_diagnostics': layer_diagnostics,
            'config': asdict(self.config),
            'model_outputs': model_outputs if save_results else None
        }
        
        if save_results:
            self._save_results(results)
            if self.config.save_plots:
                self.create_visualizations(results)
        
        return results
    
    def analyze_dataset(self, texts: List[str], labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze multiple texts and return results as DataFrame
        Args:
            texts: List of input texts
            labels: Optional labels for texts (e.g., 'hallucination', 'factual')
        Returns:
            DataFrame with aggregated results
        """
        all_results = []
        
        for i, text in enumerate(tqdm(texts, desc="Processing dataset")):
            try:
                result = self.analyze_text(text, save_results=False)
                
                # Extract summary statistics per text
                for layer_idx, diag in enumerate(result['layer_diagnostics']):
                    row = {
                        'text_id': i,
                        'text': text,
                        'layer': layer_idx,
                        'energy': diag.energy,
                        'smoothness_index': diag.smoothness_index,
                        'spectral_entropy': diag.spectral_entropy,
                        'hfer': diag.hfer,
                        'fiedler_value': diag.fiedler_value,
                        'connectivity': diag.connectivity,
                        'num_tokens': len(result['tokens'])
                    }
                    
                    if labels is not None:
                        row['label'] = labels[i]
                    
                    all_results.append(row)
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                continue
        
        df = pd.DataFrame(all_results)
        
        # Save dataset results
        output_path = Path(self.config.output_dir) / "dataset_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset results saved to {output_path}")
        
        return df
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to disk"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_path = Path(self.config.output_dir) / f"analysis_{timestamp}.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        
        # Save diagnostics as JSON (more readable)
        diagnostics_data = []
        for diag in results['layer_diagnostics']:
            diagnostics_data.append(diag.to_dict())
        
        json_path = Path(self.config.output_dir) / f"diagnostics_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                'text': results['text'],
                'tokens': results['tokens'],
                'diagnostics': diagnostics_data
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path} and {json_path}")
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create diagnostic visualizations"""
        if not self.config.save_plots:
            return
        
        diagnostics = results['layer_diagnostics']
        num_layers = len(diagnostics)
        
        # Extract metrics for plotting
        layers = list(range(num_layers))
        energies = [d.energy for d in diagnostics]
        smoothness_indices = [d.smoothness_index for d in diagnostics]
        spectral_entropies = [d.spectral_entropy for d in diagnostics]
        hfers = [d.hfer for d in diagnostics]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"GSP Diagnostics: {results['text'][:50]}...", fontsize=14)
        
        # Energy plot
        axes[0, 0].plot(layers, energies, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Dirichlet Energy by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Smoothness Index plot
        axes[0, 1].plot(layers, smoothness_indices, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Smoothness Index by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Smoothness Index')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spectral Entropy plot
        axes[1, 0].plot(layers, spectral_entropies, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Spectral Entropy by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Spectral Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HFER plot
        axes[1, 1].plot(layers, hfers, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].set_title('High-Frequency Energy Ratio by Layer')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('HFER')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(self.config.output_dir) / f"diagnostics_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
