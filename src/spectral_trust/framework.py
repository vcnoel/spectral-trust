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
from .spectral import SpectralAnalyzer, calculate_spectral_velocity
from .directed_topology import DirectedTopologist

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
    
    @torch.no_grad()
    def analyze_text(self, text: str, save_results: bool = True, subgraph_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Perform complete GSP analysis on input text
        Args:
            text: Input text to analyze
            save_results: Whether to save intermediate results
            subgraph_indices: Optional list of indices to isolate a specific subgraph
        Returns:
            Complete analysis results
        """
        logger.info(f"Analyzing text: {text[:100]}...")
        
        # Use config indices if none provided
        target_indices = subgraph_indices or self.config.subgraph_indices
        
        # Initialize directed topologist if needed
        directed_topologist = None
        if self.config.directed:
            directed_topologist = DirectedTopologist(device=self.config.device)
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

            if target_indices:
                laplacian = self.graph_constructor.subgraph_laplacian(
                    adjacency.unsqueeze(0), target_indices
                ).squeeze(0)
            else:
                laplacian = self.graph_constructor.construct_laplacian(
                    adjacency.unsqueeze(0)
                ).squeeze(0)  # [seq_len, seq_len]
            
            # Perform spectral analysis
            if target_indices:
                device = signals.device
                idx_tensor = torch.tensor(target_indices, device=device)
                signals_sub = signals.index_select(0, idx_tensor)
                diagnostics = self.spectral_analyzer.analyze_layer(
                    signals_sub, laplacian, layer_idx
                )
            else:
                diagnostics = self.spectral_analyzer.analyze_layer(
                    signals, laplacian, layer_idx
                )
            
            # New in v0.2.0: Directed Analysis
            if directed_topologist:
                # Compute directed laplacian
                L_dir = directed_topologist.compute_directed_laplacian(adjacency)
                dir_metrics = directed_topologist.get_directed_metrics(L_dir, k=6)
                diagnostics.max_imaginary = dir_metrics['max_imaginary']
                diagnostics.spectral_radius = dir_metrics['spectral_radius']
            
            layer_diagnostics.append(diagnostics)
            
            if self.config.verbose:
                msg = (f"Layer {layer_idx}: Energy={diagnostics.energy:.4f}, "
                       f"SMI={diagnostics.smoothness_index:.4f}, "
                       f"SE={diagnostics.spectral_entropy:.4f}, "
                       f"HFER={diagnostics.hfer:.4f}")
                if directed_topologist:
                    msg += f", MaxImag={diagnostics.max_imaginary:.4f}, SpecRad={diagnostics.spectral_radius:.4f}"
                logger.info(msg)
        
        # New in v0.2.0: Spectral Velocity
        velocity_results = {}
        if self.config.calc_velocity:
            # We compute velocity for Fiedler as the primary indicator
            fiedler_values = torch.tensor([d.fiedler_value for d in layer_diagnostics])
            if self.config.device != "cpu" and torch.cuda.is_available():
                fiedler_values = fiedler_values.cuda()
            
            vel_tensor, max_vel, max_idx = calculate_spectral_velocity(fiedler_values)
            velocity_results = {
                'fiedler_velocity': vel_tensor.cpu().tolist(),
                'max_velocity_value': max_vel,
                'max_velocity_layer_index': max_idx
            }
            logger.info(f"Spectral Velocity: Max={max_vel:.4f} at Layer {max_idx}")
        
        # Compile results
        results = {
            'text': text,
            'tokens': model_outputs['tokens'],
            'layer_diagnostics': layer_diagnostics,
            'velocity_metrics': velocity_results,
            'config': asdict(self.config),
            'model_outputs': model_outputs if save_results else None
        }
        
        if save_results:
            self._save_results(results)
        
        if self.config.save_plots or self.config.display_plots:
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
        
        output_dict = {
            'text': results['text'],
            'tokens': results['tokens'],
            'diagnostics': diagnostics_data
        }
        
        if 'velocity_metrics' in results:
            output_dict['velocity_metrics'] = results['velocity_metrics']
        
        with open(json_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_path} and {json_path}")
        
        if self.config.latex_export:
            self.save_latex_data(results)

    def save_latex_data(self, results: Dict[str, Any]):
        """
        Save analysis results as a LaTeX-friendly .dat file (space-separated)
        Format suitable for pgfplots: \addplot table {file.dat};
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.output_dir) / f"metrics_latex_{timestamp}.dat"
        
        diagnostics = results['layer_diagnostics']
        
        with open(output_path, 'w') as f:
            # Header
            f.write("Layer Energy HFER Entropy Fiedler Smoothness\n")
            
            # Rows
            for i, d in enumerate(diagnostics):
                f.write(f"{i} {d.energy:.6f} {d.hfer:.6f} {d.spectral_entropy:.6f} {d.fiedler_value:.6f} {d.smoothness_index:.6f}\n")
        
        logger.info(f"LaTeX data saved to {output_path}")
        print(f"\n[LaTeX Data] Saved to: {output_path}")
        print("Use with PGFPlots:")
        print(f"\\addplot table [x=Layer, y=Fiedler] {{{output_path.name}}};")

    def save_comparison_latex_data(self, res1: Dict[str, Any], res2: Dict[str, Any]):
        """Save comparison results as .dat file for PGFPlots"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.output_dir) / f"comparison_latex_{timestamp}.dat"
        
        diag1 = res1['layer_diagnostics']
        diag2 = res2['layer_diagnostics']
        
        with open(output_path, 'w') as f:
            # Header
            f.write("Layer Energy1 Energy2 Fiedler1 Fiedler2 Smoothness1 Smoothness2 HFER1 HFER2 Entropy1 Entropy2\n")
            
            for i in range(min(len(diag1), len(diag2))):
                d1 = diag1[i]
                d2 = diag2[i]
                f.write(f"{i} {d1.energy:.6f} {d2.energy:.6f} {d1.fiedler_value:.6f} {d2.fiedler_value:.6f} "
                        f"{d1.smoothness_index:.6f} {d2.smoothness_index:.6f} {d1.hfer:.6f} {d2.hfer:.6f} "
                        f"{d1.spectral_entropy:.6f} {d2.spectral_entropy:.6f}\n")
                        
        logger.info(f"Comparison LaTeX data saved to {output_path}")
        print(f"\n[LaTeX Data] Saved to: {output_path}")
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create legacy symmetric diagnostic visualizations (2x2 grid)"""
        if not self.config.save_plots and not self.config.display_plots:
            return
        
        diagnostics = results['layer_diagnostics']
        num_layers = len(diagnostics)
        
        # Extract metrics for plotting
        layers = list(range(num_layers))
        fiedlers = [d.fiedler_value for d in diagnostics]
        smoothness_indices = [d.smoothness_index for d in diagnostics]
        energies = [d.energy for d in diagnostics]
        hfers = [d.hfer for d in diagnostics]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        model_name = self.config.model_name.split('/')[-1]
        fig.suptitle(f"Legacy Symmetric Proof: {model_name}\n{results['text'][:50]}...", fontsize=14)
        
        # Top-Left: Fiedler Value
        axes[0, 0].plot(layers, fiedlers, 'orange', marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Fiedler Value ($\lambda_2$)')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Spectral Gap')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-Right: Smoothness Index
        axes[0, 1].plot(layers, smoothness_indices, 'red', marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Smoothness Index')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Index')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bottom-Left: Dirichlet Energy
        axes[1, 0].plot(layers, energies, 'blue', marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Dirichlet Energy')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bottom-Right: HFER
        axes[1, 1].plot(layers, hfers, 'magenta', marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_title('HFER (High-Frequency Energy Ratio)')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        if self.config.save_plots:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            plot_path = Path(self.config.output_dir) / f"legacy_proof_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visual Proof saved to {plot_path}")
        
        # Display plot
        if self.config.display_plots:
            plt.show()
            
        plt.close()

    def visualize_multi_run(self, run_results_list: List[Dict[str, Any]]):
        """
        Visualize multiple runs with mean and individual traces.
        According to user request:
          If config.plot_metrics is ['all'] (or None):
             2 subplots:
               Left: HFER, Entropy, Fiedler, Smoothness
               Right: Energy
          Else:
             Plot specific metrics.
        """
        if not run_results_list:
            return

        import numpy as np
        
        # Extract data
        num_runs = len(run_results_list)
        num_layers = len(run_results_list[0]['layer_diagnostics'])
        layers = np.arange(num_layers)
        
        # Metrics map
        # metric_key -> (DisplayName, Color)
        metrics_info = {
            'energy': ('Energy', 'blue'),
            'hfer': ('HFER', 'magenta'),
            'spectral_entropy': ('Entropy', 'green'),
            'fiedler_value': ('Fiedler', 'orange'),
            'smoothness_index': ('Smoothness', 'red')
        }
        
        # Data container: metric -> [runs, layers]
        data = {k: np.zeros((num_runs, num_layers)) for k in metrics_info}
        
        for r_idx, res in enumerate(run_results_list):
            for l_idx, diag in enumerate(res['layer_diagnostics']):
                data['energy'][r_idx, l_idx] = diag.energy
                data['hfer'][r_idx, l_idx] = diag.hfer
                data['spectral_entropy'][r_idx, l_idx] = diag.spectral_entropy
                data['fiedler_value'][r_idx, l_idx] = diag.fiedler_value
                data['smoothness_index'][r_idx, l_idx] = diag.smoothness_index

        plot_selection = self.config.plot_metrics
        if not plot_selection or 'all' in plot_selection:
            # Special 2-figure layout
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            # Right: Energy
            ax_energy = axes[1]
            E = data['energy']
            E_mean = np.mean(E, axis=0)
            E_std = np.std(E, axis=0)
            
            # Plot runs (faint)
            if num_runs > 1:
                for r in range(num_runs):
                    ax_energy.plot(layers, E[r], color='blue', alpha=0.1, linewidth=0.5)
                # Plot STD shading
                ax_energy.fill_between(layers, E_mean - E_std, E_mean + E_std, color='blue', alpha=0.2)
                
            # Plot mean
            ax_energy.plot(layers, E_mean, color='blue', linewidth=2.5, label='Mean Energy')
            model_name = self.config.model_name.split('/')[-1]
            ax_energy.set_title(f"Dirichlet Energy ({model_name})")
            ax_energy.set_xlabel("Layer")
            ax_energy.set_ylabel("Energy")
            ax_energy.grid(True, alpha=0.3)
            ax_energy.legend()
            
            # Left: Others
            ax_others = axes[0]
            # Metrics to plot on left
            left_metrics = ['hfer', 'spectral_entropy', 'fiedler_value', 'smoothness_index']
            colors = ['magenta', 'green', 'orange', 'red']
            
            for m_key, color in zip(left_metrics, colors):
                M = data[m_key]
                display_name = metrics_info[m_key][0]
                M_mean = np.mean(M, axis=0)
                M_std = np.std(M, axis=0)
                
                # Plot runs
                if num_runs > 1:
                    for r in range(num_runs):
                        ax_others.plot(layers, M[r], color=color, alpha=0.1, linewidth=0.5)
                    # Plot STD shading
                    ax_others.fill_between(layers, M_mean - M_std, M_mean + M_std, color=color, alpha=0.2)
                
                # Plot mean
                ax_others.plot(layers, M_mean, color=color, linewidth=2.5, label=f"Mean {display_name}")

            ax_others.set_title(f"Spectral Metrics ({model_name})")
            ax_others.set_xlabel("Layer")
            ax_others.set_ylabel("Value")
            ax_others.grid(True, alpha=0.3)
            ax_others.legend()
            
        else:
            # Specific choice
            n_plots = len(plot_selection)
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
            if n_plots == 1: axes = [axes]
            
            for ax, m_key in zip(axes, plot_selection):
                metric_key_map = {
                    'energy': 'energy', 'hfer': 'hfer', 'entropy': 'spectral_entropy',
                    'fiedler': 'fiedler_value', 'smoothness': 'smoothness_index'
                }
                real_key = metric_key_map.get(m_key, m_key)
                
                if real_key not in data:
                    logger.warning(f"Unknown metric {m_key}, skipping plot.")
                    continue
                    
                M = data[real_key]
                display_name, color = metrics_info.get(real_key, (m_key, 'black'))
                M_mean = np.mean(M, axis=0)
                M_std = np.std(M, axis=0)
                
                if num_runs > 1:
                    for r in range(num_runs):
                        ax.plot(layers, M[r], color=color, alpha=0.1, linewidth=0.5)
                    # Plot STD shading
                    ax.fill_between(layers, M_mean - M_std, M_mean + M_std, color=color, alpha=0.2)
                
                ax.plot(layers, M_mean, color=color, linewidth=2.5, label=f"Mean {display_name}")
                ax.set_title(display_name)
                ax.set_xlabel("Layer")
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(self.config.output_dir) / f"multi_run_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved multi-run plot to {plot_path}")

    def visualize_comparison(self, result1: Dict[str, Any], result2: Dict[str, Any]):
        """
        Compare two analysis results side-by-side with optional metric filtering.
        Output dynamically adjusts rows based on config.plot_metrics.
        """
        import numpy as np
        
        diag1 = result1['layer_diagnostics']
        diag2 = result2['layer_diagnostics']
        num_layers = len(diag1)
        layers = np.arange(num_layers)
        
        metrics_info = {
            'energy': ('Energy', 'blue'),
            'hfer': ('HFER', 'magenta'),
            'spectral_entropy': ('Entropy', 'green'),
            'fiedler_value': ('Fiedler', 'orange'),
            'smoothness_index': ('Smoothness', 'red')
        }
        
        # Helpler to extract metric array
        def get_metric(diags, key):
            return np.array([getattr(d, key) for d in diags])

        # Determine labels
        model1_name = result1['config']['model_name'].split('/')[-1]
        model2_name = result2['config']['model_name'].split('/')[-1]
        text1_short = result1['text'][:20]
        text2_short = result2['text'][:20]
        
        if model1_name == model2_name:
            label1 = f"T1: {text1_short}..."
            label2 = f"T2: {text2_short}..."
        else:
            label1 = f"{model1_name} ({text1_short}...)"
            label2 = f"{model2_name} ({text2_short}...)"

        # Determine metrics to plot from config
        # Use config from result1 as primary source of truth
        plot_metrics = result1['config'].get('plot_metrics', ['all'])
        
        # If 'all' or empty, use standard two-panel layout (Aggregated + Energy)
        if not plot_metrics or 'all' in plot_metrics:
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            # Add main title with model names
            main_title = f"Comparison: {model1_name} vs {model2_name}"
            if model1_name == model2_name:
                main_title = f"Comparison ({model1_name}): {text1_short}... vs {text2_short}..."
            fig.suptitle(main_title, fontsize=14)
            
            # --- Right: Energy ---
            ax_energy = axes[1]
            e1 = get_metric(diag1, 'energy')
            e2 = get_metric(diag2, 'energy')
            
            ax_energy.plot(layers, e1, color='blue', linestyle='-', linewidth=2, label=label1)
            ax_energy.plot(layers[:len(e2)], e2, color='blue', linestyle='--', linewidth=2, label=label2)
            
            ax_energy.set_title("Dirichlet Energy Comparison")
            ax_energy.set_xlabel("Layer")
            ax_energy.set_ylabel("Energy")
            ax_energy.grid(True, alpha=0.3)
            ax_energy.legend()
            
            # --- Left: Others ---
            ax_others = axes[0]
            left_metrics = ['hfer', 'spectral_entropy', 'fiedler_value', 'smoothness_index']
            colors = ['magenta', 'green', 'orange', 'red']
            
            for m_key, color in zip(left_metrics, colors):
                display_name = metrics_info[m_key][0]
                v1 = get_metric(diag1, m_key)
                v2 = get_metric(diag2, m_key)
                
                # Text 1 Solid
                ax_others.plot(layers, v1, color=color, linestyle='-', linewidth=2, label=f"{display_name} ({label1})")
                # Text 2 Dashed
                ax_others.plot(layers[:len(v2)], v2, color=color, linestyle='--', linewidth=2, label=f"{display_name} ({label2})")
                
            ax_others.set_title("Spectral Metrics Comparison")
            ax_others.set_xlabel("Layer")
            ax_others.set_ylabel("Value")
            ax_others.grid(True, alpha=0.3)
            ax_others.legend(fontsize='small', ncol=2)

        else:
            # --- Specific Metrics Mode ---
            # e.g. ['fiedler', 'smoothness']
            metrics_to_plot = []
            for m in plot_metrics:
                # Map potential alias 'fiedler' -> 'fiedler_value'
                key_map = {'fiedler': 'fiedler_value', 'entropy': 'spectral_entropy', 'smoothness': 'smoothness_index'}
                real_key = key_map.get(m, m)
                if real_key in metrics_info:
                    metrics_to_plot.append(real_key)
            
            num_plots = len(metrics_to_plot)
            if num_plots == 0:
                logger.warning(f"No valid metrics found in {plot_metrics}")
                return

            fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
            if num_plots == 1:
                axes = [axes]
            
            for ax, m_key in zip(axes, metrics_to_plot):
                display_name, color = metrics_info[m_key]
                v1 = get_metric(diag1, m_key)
                v2 = get_metric(diag2, m_key)
                
                ax.plot(layers, v1, color=color, linestyle='-', linewidth=2, label=label1)
                ax.plot(layers[:len(v2)], v2, color=color, linestyle='--', linewidth=2, label=label2)
                
                ax.set_title(f"{display_name} Comparison")
                ax.set_xlabel("Layer")
                ax.set_ylabel(display_name)
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(self.config.output_dir) / f"comparison_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot to {plot_path}")
