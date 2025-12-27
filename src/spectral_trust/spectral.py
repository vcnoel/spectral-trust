from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh
import logging
from .config import GSPConfig

logger = logging.getLogger(__name__)

@dataclass
class SpectralDiagnostics:
    """Container for spectral diagnostics results"""
    layer: int
    energy: float
    smoothness_index: float
    spectral_entropy: float
    hfer: float
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray]
    spectral_masses: np.ndarray
    fiedler_value: float
    connectivity: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'layer': int(self.layer),
            'energy': float(self.energy),
            'smoothness_index': float(self.smoothness_index),
            'spectral_entropy': float(self.spectral_entropy),
            'hfer': float(self.hfer),
            'eigenvalues': self.eigenvalues.tolist(),
            'spectral_masses': self.spectral_masses.tolist(),
            'fiedler_value': float(self.fiedler_value),
            'connectivity': bool(self.connectivity)
        }


class SpectralAnalyzer:
    """Performs spectral analysis and computes GSP diagnostics"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
    
    def compute_eigendecomposition(self, laplacian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of graph Laplacian
        Args:
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            eigenvalues, eigenvectors
        """
        seq_len = laplacian.shape[0]
        
        if self.config.eigen_solver == "sparse" and seq_len > 50:
            # Use sparse eigenvalue solver for large matrices
            try:
                # Convert to sparse matrix for efficiency
                laplacian_sparse = csr_matrix(laplacian)
                
                # Compute smallest eigenvalues (including 0) and largest ones
                k_small = min(self.config.num_eigenvalues // 2, seq_len - 2)
                k_large = min(self.config.num_eigenvalues - k_small, seq_len - k_small - 1)
                
                if k_small > 0:
                    eigenvals_small, eigenvecs_small = eigsh(
                        laplacian_sparse, k=k_small, which='SM',
                        maxiter=self.config.lanczos_max_iter, tol=1e-6
                    )
                else:
                    eigenvals_small, eigenvecs_small = np.array([]), np.zeros((seq_len, 0))
                
                if k_large > 0:
                    eigenvals_large, eigenvecs_large = eigsh(
                        laplacian_sparse, k=k_large, which='LM',
                        maxiter=self.config.lanczos_max_iter, tol=1e-6
                    )
                else:
                    eigenvals_large, eigenvecs_large = np.array([]), np.zeros((seq_len, 0))
                
                # Combine and sort
                eigenvals = np.concatenate([eigenvals_small, eigenvals_large])
                eigenvecs = np.concatenate([eigenvecs_small, eigenvecs_large], axis=1)
                
                # Sort by eigenvalue
                sort_idx = np.argsort(eigenvals)
                eigenvals = eigenvals[sort_idx]
                eigenvecs = eigenvecs[:, sort_idx]
                
            except ArpackNoConvergence as e:
                logger.warning(f"ARPACK did not converge, falling back to dense solver: {e}")
                eigenvals, eigenvecs = eigh(laplacian)
        else:
            # Use dense eigenvalue solver
            eigenvals, eigenvecs = eigh(laplacian)
        
        # Ensure eigenvalues are non-negative (numerical precision)
        eigenvals = np.maximum(eigenvals, 0)
        
        return eigenvals, eigenvecs
    
    def compute_dirichlet_energy(self, signals: np.ndarray, laplacian: np.ndarray) -> float:
        """
        Compute Dirichlet energy of signals on graph
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            Total Dirichlet energy
        """
        # Energy = Tr(X^T L X)
        energy_matrix = np.dot(signals.T, np.dot(laplacian, signals))
        energy = np.trace(energy_matrix)
        return float(energy)
    
    def compute_smoothness_index(self, signals: np.ndarray, laplacian: np.ndarray) -> float:
        """
        Compute smoothness index (normalized energy)
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            Smoothness index
        """
        energy = self.compute_dirichlet_energy(signals, laplacian)
        signal_norm = np.trace(np.dot(signals.T, signals))
        
        if signal_norm < 1e-8:
            return 0.0
        
        return energy / signal_norm
    
    def compute_spectral_entropy(self, signals: np.ndarray, eigenvectors: np.ndarray) -> float:
        """
        Compute spectral entropy of signals
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            eigenvectors: [seq_len, num_eigenvectors] eigenvector matrix
        Returns:
            Spectral entropy
        """
        # Project signals onto eigenbasis
        signal_hat = np.dot(eigenvectors.T, signals)  # [num_eigenvectors, embedding_dim]
        
        # Compute spectral energies per frequency
        spectral_energies = np.sum(signal_hat**2, axis=1)  # [num_eigenvectors]
        
        # Normalize to get probability distribution
        total_energy = np.sum(spectral_energies)
        if total_energy < 1e-8:
            return 0.0
        
        spectral_probs = spectral_energies / total_energy
        
        # Compute entropy
        spectral_probs = np.maximum(spectral_probs, 1e-12)  # Avoid log(0)
        entropy = -np.sum(spectral_probs * np.log(spectral_probs))
        
        return float(entropy)
    
    def compute_hfer(self, signals: np.ndarray, eigenvectors: np.ndarray, 
                    eigenvalues: np.ndarray, cutoff_ratio: float) -> float:
        """
        Compute High-Frequency Energy Ratio
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            eigenvectors: [seq_len, num_eigenvectors] eigenvector matrix
            eigenvalues: [num_eigenvectors] eigenvalue array
            cutoff_ratio: Fraction of spectrum to consider as high-frequency
        Returns:
            HFER value
        """
        # Project signals onto eigenbasis
        signal_hat = np.dot(eigenvectors.T, signals)  # [num_eigenvectors, embedding_dim]
        
        # Compute spectral energies per frequency
        spectral_energies = np.sum(signal_hat**2, axis=1)  # [num_eigenvectors]
        
        # Determine cutoff index
        num_eigenvectors = len(eigenvalues)
        cutoff_index = int((1 - cutoff_ratio) * num_eigenvectors)
        
        # Compute HFER
        total_energy = np.sum(spectral_energies)
        if total_energy < 1e-8:
            return 0.0
        
        high_freq_energy = np.sum(spectral_energies[cutoff_index:])
        hfer = high_freq_energy / total_energy
        
        return float(hfer)
    
    def analyze_layer(self, signals: torch.Tensor, laplacian: torch.Tensor, 
                     layer_idx: int) -> SpectralDiagnostics:
        """
        Perform complete spectral analysis for a single layer
        Args:
            signals: [seq_len, embedding_dim] activation tensor
            laplacian: [seq_len, seq_len] Laplacian tensor
            layer_idx: Layer index
        Returns:
            Complete spectral diagnostics
        """
        # Convert to numpy for numerical computations
        signals_np = signals.detach().cpu().numpy().astype(np.float32)
        laplacian_np = laplacian.detach().cpu().numpy().squeeze().astype(np.float32)
        
        # Check connectivity
        connectivity = self._check_connectivity(laplacian_np)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = self.compute_eigendecomposition(laplacian_np)
        
        # Compute diagnostics
        energy = self.compute_dirichlet_energy(signals_np, laplacian_np)
        smoothness_index = self.compute_smoothness_index(signals_np, laplacian_np)
        spectral_entropy = self.compute_spectral_entropy(signals_np, eigenvectors)
        hfer = self.compute_hfer(signals_np, eigenvectors, eigenvalues, 
                               self.config.hfer_cutoff_ratio)
        
        # Compute spectral masses
        signal_hat = np.dot(eigenvectors.T, signals_np)
        spectral_masses = np.sum(signal_hat**2, axis=1)
        
        # Fiedler value (second smallest eigenvalue)
        fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        return SpectralDiagnostics(
            layer=layer_idx,
            energy=energy,
            smoothness_index=smoothness_index,
            spectral_entropy=spectral_entropy,
            hfer=hfer,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors if self.config.save_intermediate else None,
            spectral_masses=spectral_masses,
            fiedler_value=fiedler_value,
            connectivity=connectivity
        )
    
    def _check_connectivity(self, laplacian: np.ndarray) -> bool:
        """Check if the graph is connected by examining the null space of Laplacian"""
        eigenvals, _ = self.compute_eigendecomposition(laplacian)
        # Graph is connected if there's exactly one zero eigenvalue
        zero_eigenvals = np.sum(eigenvals < 1e-6)
        return zero_eigenvals == 1
