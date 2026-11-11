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
from .fast import gaussian_projection, num_high_modes

logger = logging.getLogger(__name__)

# ── Metric provenance (v0.3.0) ─────────────────────────────────────────────────
#
# Every diagnostic is tagged with the tensor family it consumes. This matters
# because the families have different access requirements and different
# scientific interpretations, and conflating them has led to "attention-only"
# claims about metrics that in fact read the residual stream:
#
#   "attention"  — function of the attention weight matrices alone.
#                  Available when a serving stack exposes attention maps.
#   "hybrid"     — residual-stream states projected onto the attention-graph
#                  eigenbasis. Requires BOTH attentions and hidden states;
#                  results reflect residual-stream content as much as graph
#                  topology. NOT attention-only.
#
# (Signals that read hidden states directly, e.g. supervised probes, are out
# of scope for this library and would form a third family, "residual".)
METRIC_FAMILIES: Dict[str, str] = {
    "fiedler_value": "attention",
    "spectral_radius": "attention",
    "max_imaginary": "attention",
    "connectivity": "attention",
    "eigenvalues": "attention",
    "eig_energy": "attention",
    "eig_spectral_entropy": "attention",
    "eig_hfer": "attention",
    "energy": "hybrid",
    "smoothness_index": "hybrid",
    "spectral_entropy": "hybrid",
    "hfer": "hybrid",
    "spectral_masses": "hybrid",
}


@dataclass
class SpectralDiagnostics:
    """
    Container for spectral diagnostics results.

    Each metric belongs to a tensor family (see METRIC_FAMILIES / the
    `families` field): "attention" metrics are functions of attention
    weights alone; "hybrid" metrics project residual-stream states onto the
    attention eigenbasis and therefore require hidden-state access.
    """
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
    spectral_radius: Optional[float] = None
    max_imaginary: Optional[float] = None
    # New in v0.3.0: attention-only (eigenvalue-only) counterparts of the
    # hybrid energy/entropy/HFER metrics.
    eig_energy: Optional[float] = None
    eig_spectral_entropy: Optional[float] = None
    eig_hfer: Optional[float] = None

    #: metric name -> tensor family ("attention" | "hybrid")
    families: Dict[str, str] = None

    #: Scalar metric fields, stored as plain Python floats (None if unset).
    SCALAR_FIELDS = ("energy", "smoothness_index", "spectral_entropy", "hfer",
                     "fiedler_value", "spectral_radius", "max_imaginary",
                     "eig_energy", "eig_spectral_entropy", "eig_hfer")

    def __post_init__(self):
        if self.families is None:
            self.families = dict(METRIC_FAMILIES)
        # Plain Python types, so that the fields themselves (not only
        # to_dict) serialise with json and compare without numpy scalars.
        self.layer = int(self.layer)
        self.connectivity = bool(self.connectivity)
        for name in self.SCALAR_FIELDS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, float(value))

    def to_dict(self) -> Dict[str, Any]:
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
            'families': dict(self.families),
        }
        if self.spectral_radius is not None:
            d['spectral_radius'] = float(self.spectral_radius)
        if self.max_imaginary is not None:
            d['max_imaginary'] = float(self.max_imaginary)
        for k in ('eig_energy', 'eig_spectral_entropy', 'eig_hfer'):
            v = getattr(self, k)
            if v is not None:
                d[k] = float(v)
        return d


class SpectralAnalyzer:
    """Performs spectral analysis and computes GSP diagnostics"""
    
    def __init__(self, config: GSPConfig):
        self.config = config
        self._projections: Dict[Tuple[int, int, int, str], np.ndarray] = {}

    def project_signals(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply the configured Johnson-Lindenstrauss projection
        (``config.project_dim``) to ``[seq_len, d]`` signals. Returns the
        signals unchanged when projection is off or ``project_dim >= d``.
        The matrix is cached per (d, k, seed, dtype), so every layer of a
        model shares one projection.
        """
        k = getattr(self.config, "project_dim", None)
        if k is None:
            return signals
        if k <= 0:
            raise ValueError(f"project_dim must be positive, got {k}")
        dim = signals.shape[1]
        if k >= dim:
            return signals
        seed = getattr(self.config, "project_seed", 0)
        key = (dim, k, seed, signals.dtype.str)
        R = self._projections.get(key)
        if R is None:
            R = gaussian_projection(dim, k, seed=seed, dtype=signals.dtype)
            self._projections[key] = R
        return signals @ R

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
        eigenvectors exist but are NOT orthogonal. A symmetric solver (eigh/eigsh)
        would silently diagonalize a symmetrized surrogate instead, which is why
        this path exists.
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

        Dispatches on the operator: symmetric Laplacians (normalization="sym" or
        "none") use a symmetric solver; non-symmetric ones (normalization="rw")
        use a general solver, since eigh/eigsh read only the lower triangle and
        would otherwise diagonalize a different, symmetrized matrix.

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

        if eigenvecs.shape[1] < seq_len:
            logger.warning(
                "Truncated eigenbasis: %d of %d modes (eigen_solver='sparse', "
                "num_eigenvalues=%d). Energy-ratio diagnostics (HFER, spectral "
                "entropy, spectral_masses) are normalized over the retained modes "
                "only and are NOT comparable to a full-spectrum computation. Use "
                "eigen_solver='dense' for exact energy fractions.",
                eigenvecs.shape[1], seq_len, self.config.num_eigenvalues
            )

        return eigenvals, eigenvecs
    
    # ── Attention-only (eigenvalue-only) metrics, v0.3.0 ──────────────────────
    # Counterparts of energy / spectral_entropy / hfer that are functions of
    # the Laplacian EIGENVALUES alone — no hidden states enter the
    # computation, so these are valid under attention-only access. When the
    # sparse solver truncates the eigenbasis, they are normalized over the
    # retained modes (the existing truncation warning applies).

    @staticmethod
    def compute_eig_energy(eigenvalues: np.ndarray) -> float:
        """Mean Laplacian eigenvalue (trace / T). Attention-only."""
        if eigenvalues.size == 0:
            return 0.0
        return float(np.mean(eigenvalues))

    @staticmethod
    def compute_eig_spectral_entropy(eigenvalues: np.ndarray) -> float:
        """Entropy of the eigenvalue distribution, normalized to [0, 1] by
        log(#modes). Attention-only."""
        n = eigenvalues.size
        if n < 2:
            return 0.0
        total = float(np.sum(eigenvalues))
        if total <= 1e-12:
            return 0.0
        p = np.maximum(eigenvalues / total, 1e-12)
        return float(-np.sum(p * np.log(p)) / np.log(n))

    @staticmethod
    def compute_eig_hfer(eigenvalues: np.ndarray) -> float:
        """Fraction of eigenvalue mass in the upper half of the spectrum.
        Attention-only."""
        n = eigenvalues.size
        if n < 2:
            return 0.0
        total = float(np.sum(eigenvalues))
        if total <= 1e-12:
            return 0.0
        return float(np.sum(eigenvalues[n // 2:]) / total)

    def compute_dirichlet_energy(self, signals: np.ndarray, laplacian: np.ndarray) -> float:
        """
        Compute Dirichlet energy of signals on graph, ``tr(X^T L X)``.

        Evaluated as ``sum(X * (L X))``: one S x S by S x d product, O(S^2 d)
        time and O(S d) memory. The d x d matrix ``X^T L X`` is never formed
        (before 0.4.0 it was, at O(S d^2) cost). Accumulated in float64.

        Args:
            signals: [seq_len, embedding_dim] signal matrix
            laplacian: [seq_len, seq_len] Laplacian matrix
        Returns:
            Total Dirichlet energy
        """
        return float(np.sum(signals * np.dot(laplacian, signals), dtype=np.float64))

    def compute_smoothness_index(self, signals: np.ndarray, laplacian: np.ndarray,
                                 energy: Optional[float] = None) -> float:
        """
        Compute smoothness index (normalized energy),
        ``tr(X^T L X) / ||X||_F^2``.

        Args:
            signals: [seq_len, embedding_dim] signal matrix
            laplacian: [seq_len, seq_len] Laplacian matrix
            energy: the Dirichlet energy, if already computed (analyze_layer
                passes it so that it is evaluated once)
        Returns:
            Smoothness index
        """
        if energy is None:
            energy = self.compute_dirichlet_energy(signals, laplacian)
        signal_norm = float(np.sum(signals * signals, dtype=np.float64))

        if signal_norm < 1e-8:
            return 0.0

        return energy / signal_norm

    @staticmethod
    def compute_spectral_masses(signals: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        """Energy of the signals in each eigenmode, ``||u_k^T X||^2``: the one
        projection onto the eigenbasis that entropy, HFER and spectral masses
        share."""
        signal_hat = np.dot(eigenvectors.T, signals)  # [num_eigenvectors, embedding_dim]
        return np.sum(signal_hat**2, axis=1)          # [num_eigenvectors]

    def compute_spectral_entropy(self, signals: np.ndarray, eigenvectors: np.ndarray) -> float:
        """
        Compute spectral entropy of signals
        Args:
            signals: [seq_len, embedding_dim] signal matrix
            eigenvectors: [seq_len, num_eigenvectors] eigenvector matrix
        Returns:
            Spectral entropy
        """
        return self.spectral_entropy_from_masses(
            self.compute_spectral_masses(signals, eigenvectors))

    @staticmethod
    def spectral_entropy_from_masses(spectral_energies: np.ndarray) -> float:
        """Spectral entropy (natural log, unnormalised) of per-mode energies."""
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
        return self.hfer_from_masses(
            self.compute_spectral_masses(signals, eigenvectors), cutoff_ratio)

    @staticmethod
    def hfer_from_masses(spectral_energies: np.ndarray, cutoff_ratio: float) -> float:
        """HFER of per-mode energies ordered by ascending eigenvalue: the
        share of the top ``num_high_modes(n, cutoff_ratio)`` modes, the
        library's single mode-count rule (see :func:`fast.num_high_modes`)."""
        # Determine cutoff index
        num_eigenvectors = len(spectral_energies)
        cutoff_index = num_eigenvectors - num_high_modes(num_eigenvectors, cutoff_ratio)
        
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
        signals_np = signals.to(torch.float32).detach().cpu().numpy()
        laplacian_np = laplacian.to(torch.float32).detach().cpu().numpy().squeeze()
        # Optional Johnson-Lindenstrauss projection (config.project_dim): only
        # the hybrid metrics below read the signals.
        signals_np = self.project_signals(signals_np)
        
        # Single eigendecomposition (was previously called twice — once here, once inside _check_connectivity)
        eigenvalues, eigenvectors = self.compute_eigendecomposition(laplacian_np)
        connectivity = bool(np.sum(eigenvalues < 1e-6) == 1)

        # Compute diagnostics: the energy once, the eigenbasis projection once
        energy = self.compute_dirichlet_energy(signals_np, laplacian_np)
        smoothness_index = self.compute_smoothness_index(signals_np, laplacian_np,
                                                         energy=energy)
        spectral_masses = self.compute_spectral_masses(signals_np, eigenvectors)
        spectral_entropy = self.spectral_entropy_from_masses(spectral_masses)
        hfer = self.hfer_from_masses(spectral_masses, self.config.hfer_cutoff_ratio)

        fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        spectral_radius = float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0

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
            eig_energy=self.compute_eig_energy(eigenvalues),
            eig_spectral_entropy=self.compute_eig_spectral_entropy(eigenvalues),
            eig_hfer=self.compute_eig_hfer(eigenvalues),
        )

    def _check_connectivity(self, laplacian: np.ndarray) -> bool:
        """Retained for external callers. analyze_layer no longer calls this."""
        eigenvals, _ = self.compute_eigendecomposition(laplacian)
        return np.sum(eigenvals < 1e-6) == 1


def calculate_spectral_velocity(metric_array: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """
    Vectorized GPU discrete derivative across layers: Δm = m_{n+1} − m_n.
    Returns (velocity_tensor, max_abs_velocity, layer_index_of_max).
    """
    if not isinstance(metric_array, torch.Tensor):
        metric_array = torch.tensor(metric_array)
    velocity = torch.diff(metric_array)
    abs_velocity = torch.abs(velocity)
    max_val, max_idx = torch.max(abs_velocity, dim=0)
    return velocity, float(max_val), int(max_idx.item()) + 1
