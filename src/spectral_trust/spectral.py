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

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh, eig
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
    # New in v0.2.0: Directed metrics
    max_imaginary: Optional[float] = None
    spectral_radius: Optional[float] = None
    gini_sparsity: Optional[float] = None
    attention_gini: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        d = {
            'layer': int(self.layer),
            'energy': float(self.energy),
            'smoothness_index': float(self.smoothness_index),
            'spectral_entropy': float(self.spectral_entropy),
            'hfer': float(self.hfer),
            'eigenvalues': self.eigenvalues.tolist(),
            'spectral_masses': self.spectral_masses.tolist(),
            'fiedler_value': float(self.fiedler_value),
            'connectivity': bool(self.connectivity),
            'spectral_radius': float(self.spectral_radius) if self.spectral_radius is not None else None,
            'max_imaginary': float(self.max_imaginary) if self.max_imaginary is not None else None,
            'gini_sparsity': float(self.gini_sparsity) if self.gini_sparsity is not None else None,
            'attention_gini': float(self.attention_gini) if self.attention_gini is not None else None
        }
        return d


class SpectralAnalyzer:
    """Performs spectral analysis and computes GSP diagnostics"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
    
    @staticmethod
    def _is_symmetric(matrix: np.ndarray, tol: float = 1e-6) -> bool:
        """True if the operator is symmetric to within `tol` (so eigh/eigsh are valid)."""
        return bool(np.allclose(matrix, matrix.T, atol=tol, rtol=0.0))

    def _eig_nonsymmetric(self, laplacian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition of a NON-symmetric operator, e.g. the random-walk
        Laplacian L_rw = I - D^{-1} W selected by normalization="rw".

        L_rw is similar to the symmetric normalized Laplacian
        (L_rw = D^{-1/2} L_sym D^{1/2}), so its eigenvalues are real; its
        eigenvectors exist but are NOT orthogonal. A symmetric solver
        (eigh/eigsh) reads only one triangle and would silently diagonalize a
        symmetrized surrogate instead — a wrong answer with no error raised,
        which is why this path exists.

        Basis-dependent diagnostics (HFER, spectral entropy, and any Parseval
        accounting) assume an orthonormal basis and remain invalid under "rw"
        even with the correct solver; use normalization="sym" for those.
        """
        eigenvals, eigenvecs = eig(laplacian)
        max_imag = float(np.max(np.abs(eigenvals.imag))) if eigenvals.size else 0.0
        if max_imag > 1e-6:
            logger.warning(
                "Non-symmetric operator has complex eigenvalues (max|Im|=%.2e); "
                "taking real parts. Diagnostics assume a real spectrum.", max_imag
            )
        eigenvals = eigenvals.real
        eigenvecs = eigenvecs.real
        sort_idx = np.argsort(eigenvals)
        return eigenvals[sort_idx], eigenvecs[:, sort_idx]

    def compute_eigendecomposition(self, laplacian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of graph Laplacian.

        Dispatches on the operator: symmetric Laplacians (normalization="sym"
        or "none") use a symmetric solver; non-symmetric ones
        (normalization="rw") use a general solver, since eigh/eigsh read only
        the lower triangle and would otherwise diagonalize a different,
        symmetrized matrix.

        Args:
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            eigenvalues, eigenvectors (eigenvectors are orthonormal only for
            symmetric operators)
        """
        seq_len = laplacian.shape[0]

        if not self._is_symmetric(laplacian):
            eigenvals, eigenvecs = self._eig_nonsymmetric(laplacian)
            return np.maximum(eigenvals, 0), eigenvecs

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

    def compute_gini_sparsity(self, signals: np.ndarray) -> float:
        """
        Compute Gini coefficient of signal magnitudes to measure focus/sparsity.
        High Gini = signal concentrated on few eigenvalues/tokens.
        """
        # We look at the magnitude of the signal on the tokens (norm across embedding dim)
        s = np.linalg.norm(signals, axis=1)
        return self._gini(s)

    def compute_attention_gini(self, adjacency: np.ndarray) -> float:
        """
        Compute Gini coefficient of attention weights to measure sparsity.
        High Gini = attention concentrated on few tokens.
        """
        # Flatten adjacency and compute Gini on non-zero weights
        weights = adjacency.flatten()
        weights = weights[weights > 1e-12]
        if len(weights) == 0:
            return 0.0
        return self._gini(weights)

    def _gini(self, x: np.ndarray) -> float:
        """Standard Gini coefficient calculation"""
        n = len(x)
        if n == 0 or np.sum(x) == 0:
            return 0.0
        
        sorted_x = np.sort(x)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_x)) / (n * np.sum(sorted_x))
        return float(gini)
    
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
        seq_len = signals.shape[0]
        if eigenvectors.shape[1] < seq_len:
            logger.warning(
                "Truncated eigenbasis: %d of %d modes (eigen_solver='sparse', "
                "num_eigenvalues=%d). Energy-ratio diagnostics (HFER, spectral "
                "entropy) are normalized over the retained modes only and are "
                "NOT comparable to a full-spectrum computation. Use "
                "eigen_solver='dense' for exact energy fractions.",
                eigenvectors.shape[1], seq_len, self.config.num_eigenvalues
            )

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
                     layer_idx: int, adjacency: Optional[torch.Tensor] = None) -> SpectralDiagnostics:
        """
        Perform complete spectral analysis for a single layer
        Args:
            signals: [seq_len, embedding_dim] activation tensor
            laplacian: [seq_len, seq_len] Laplacian tensor
            layer_idx: Layer index
            adjacency: [seq_len, seq_len] Adjacency/Attention tensor
        Returns:
            Complete spectral diagnostics
        """
        # Convert to numpy for numerical computations.
        # Upcast to float32 first: NumPy has no bfloat16 dtype, so a bf16 model
        # would otherwise raise "Got unsupported ScalarType BFloat16" here. .float()
        # is a no-op for float32 and losslessly widens float16/bfloat16.
        signals_np = signals.detach().cpu().float().numpy()
        laplacian_np = laplacian.detach().cpu().float().numpy().squeeze()

        # Determine adjacency for Gini (if not provided, we just use laplacian diagonal/non-diagonal info if needed,
        # but better to have the actual adjacency)
        if adjacency is not None:
            adjacency_np = adjacency.detach().cpu().float().numpy().squeeze()
        else:
            # Fallback: estimate from laplacian (L = D - A)
            adjacency_np = -laplacian_np
            np.fill_diagonal(adjacency_np, 0)
            adjacency_np = np.maximum(adjacency_np, 0)
        
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
        gini_sparsity = self.compute_gini_sparsity(signals_np)
        attention_gini = self.compute_attention_gini(adjacency_np)
        
        # Compute spectral masses
        signal_hat = np.dot(eigenvectors.T, signals_np)
        spectral_masses = np.sum(signal_hat**2, axis=1)
        
        # Fiedler value (second smallest eigenvalue)
        fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        # Spectral radius (max eigenvalue)
        spectral_radius = eigenvalues[-1] if len(eigenvalues) > 0 else 0.0
        
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
            connectivity=connectivity,
            spectral_radius=spectral_radius,
            gini_sparsity=gini_sparsity,
            attention_gini=attention_gini
        )
    
    def _check_connectivity(self, laplacian: np.ndarray) -> bool:
        """Check if the graph is connected by examining the null space of Laplacian"""
        eigenvals, _ = self.compute_eigendecomposition(laplacian)
        # Graph is connected if there's exactly one zero eigenvalue
        zero_eigenvals = np.sum(eigenvals < 1e-6)
        return zero_eigenvals == 1

def weight_snr(W: torch.Tensor, k: int = 1) -> float:
    """
    Fast SNR estimate for a weight matrix using Lanczos-based randomized SVD.

    SNR = σ₁ / median(σ).  For large matrices, estimating σ₁ with top-k
    Lanczos is O(k·m·n) vs O(min(m,n)²·max(m,n)) for full SVD.

    Args:
        W:  2-D weight tensor, e.g. mlp.down_proj
        k:  number of singular vectors to compute (default 1 for σ₁ only;
            use ≥ 16 for a reliable median estimate)
    Returns:
        SNR as a Python float
    """
    W_fp = W.float()
    # torch.svd_lowrank uses a block-Krylov / randomized Lanczos algorithm
    niter = max(4, k // 4)
    _, S, _ = torch.svd_lowrank(W_fp, q=max(k, 16), niter=niter)
    sigma_1 = S[0].item()
    sigma_median = S[len(S) // 2].item()
    return sigma_1 / sigma_median if sigma_median > 1e-12 else 0.0


def weight_svd_full(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full thin SVD of a weight matrix (W = U Σ Vᴴ).

    Uses torch.linalg.svd which internally picks the fastest LAPACK driver for
    the given shape.  Required when all singular values are needed (e.g. for
    variance-normalised spectral sharpening).

    Args:
        W:  2-D weight tensor
    Returns:
        (U, S, Vh) in float32
    """
    return torch.linalg.svd(W.float(), full_matrices=False)


def calculate_spectral_velocity(metric_array: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """
    Compute discrete derivative of metrics across layers: Delta_m = m_n - m_{n-1}
    Uses vectorized GPU tensor operations.
    Returns:
        velocity_tensor: [num_layers - 1]
        max_velocity: Maximum absolute change
        max_velocity_layer: Index of the layer where max change occurred (n)
    """
    # Ensure tensor
    if not isinstance(metric_array, torch.Tensor):
        metric_array = torch.tensor(metric_array)
    
    # Vectorized discrete derivative (velocity)
    # torch.diff(x) computes x[i+1] - x[i]
    velocity = torch.diff(metric_array)
    
    # Calculate max velocity and its location
    abs_velocity = torch.abs(velocity)
    max_val, max_idx = torch.max(abs_velocity, dim=0)
    
    # The index in the velocity tensor corresponds to the jump from layer i to i+1
    # We return the layer index i+1 as the "max_velocity_layer"
    return velocity, float(max_val), int(max_idx.item()) + 1
