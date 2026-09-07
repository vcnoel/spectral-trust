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

import torch
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

def sparse_arnoldi_iteration(A: torch.Tensor, k_steps: int = 20, tol: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Arnoldi iteration to find the Hessenberg matrix H and basis Q.
    A: (N, N) tensor on GPU/CPU
    k_steps: number of Krylov subspace dimensions
    """
    device = A.device
    N = A.shape[0]
    k_steps = min(k_steps, N)
    
    # Cast to float32 for numerical stability on GPU
    A_f32 = A.to(torch.float32)
    Q = torch.zeros((N, k_steps + 1), device=device, dtype=torch.float32)
    H = torch.zeros((k_steps + 1, k_steps), device=device, dtype=torch.float32)
    
    # Random start vector
    q = torch.randn(N, device=device, dtype=torch.float32)
    q = q / torch.norm(q)
    Q[:, 0] = q
    
    for j in range(k_steps):
        # Sparse-friendly matrix-vector product
        v = torch.matmul(A_f32, Q[:, j])
        
        # Modified Gram-Schmidt with re-orthogonalization
        for i in range(j + 1):
            h = torch.dot(Q[:, i], v)
            H[i, j] = H[i, j] + h
            v = v - h * Q[:, i]
            
        # Re-orthogonalization step to maintain precision
        for i in range(j + 1):
            h = torch.dot(Q[:, i], v)
            H[i, j] = H[i, j] + h
            v = v - h * Q[:, i]
        
        H[j+1, j] = torch.norm(v)
        if H[j+1, j] < tol:
            # Happy breakdown: subspace is invariant
            return H[:j+1, :j+1], Q[:, :j+1]
        
        Q[:, j+1] = v / H[j+1, j]
        
    return H[:-1, :], Q[:, :-1]

def sparse_lanczos_iteration(A: torch.Tensor, k_steps: int = 20, tol: float = 1e-9,
                             seed: Optional[int] = 0,
                             deflate: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Perform Lanczos iteration for symmetric matrices.
    A: (N, N) symmetric tensor
    seed: seeds the start vector so results are deterministic (v0.3.0 fix —
          previously the unseeded random start made repeated calls disagree).
    deflate: optional unit vector to project out of the Krylov space (e.g. a
          known Laplacian null vector), so Ritz values approximate the
          spectrum on its orthogonal complement.
    Returns the tridiagonal matrix T.
    """
    device = A.device
    N = A.shape[0]
    k_steps = min(k_steps, N)

    alpha = torch.zeros(k_steps, device=device, dtype=A.dtype)
    beta = torch.zeros(k_steps, device=device, dtype=A.dtype)
    q_prev = torch.zeros(N, device=device, dtype=A.dtype)

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    q = torch.randn(N, generator=gen).to(device=device, dtype=A.dtype)

    def _deflated(v: torch.Tensor) -> torch.Tensor:
        if deflate is not None:
            v = v - torch.dot(deflate, v) * deflate
        return v

    q = _deflated(q)
    q = q / torch.norm(q)

    for j in range(k_steps):
        v = torch.matmul(A, q)
        alpha[j] = torch.dot(q, v)
        v = v - alpha[j] * q - (beta[j-1] * q_prev if j > 0 else 0)
        v = _deflated(v)

        if j < k_steps - 1:
            beta[j] = torch.norm(v)
            if beta[j] < tol:
                break
            q_prev = q
            q = v / beta[j]

    # Construct tridiagonal matrix T
    T = torch.diag(alpha)
    if k_steps > 1:
        T += torch.diag(beta[:k_steps-1], diagonal=1)
        T += torch.diag(beta[:k_steps-1], diagonal=-1)
    return T

class DirectedTopologist:
    """Handles directed graph metrics and asymmetric spectral analysis"""
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
    def compute_directed_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Compute Directed Random Walk Laplacian: L = I - D^-1 A
        Adjacency A is assumed to be asymmetric.
        """
        if torch.isnan(adjacency).any() or torch.isinf(adjacency).any():
            raise ValueError("Input adjacency matrix contains NaN or Inf values.")
            
        A = adjacency.to(self.device)
        N = A.shape[-1]
        
        # Row sums for out-degree
        d_out = A.sum(dim=-1)
        d_inv = torch.where(d_out > 1e-8, 1.0 / d_out, torch.zeros_like(d_out))
        D_inv = torch.diag_embed(d_inv)
        
        I = torch.eye(N, device=self.device, dtype=A.dtype)
        # Random Walk Laplacian L = I - P where P = D^-1 A
        L = I - torch.matmul(D_inv, A)
        return L

    def get_directed_metrics(self, L: torch.Tensor, k: int = 6) -> Dict[str, float]:
        """
        Extract Max Imaginary Component and Spectral Radius using Arnoldi
        """
        # Ensure L is 2D
        if L.dim() > 2:
            L = L.squeeze()
            
        # Use Arnoldi to get eigenvalues of the Hessenberg matrix
        # k_steps should be larger than k to improve convergence
        n_iter = min(L.shape[0], k * 3 + 10)
        
        try:
            H, _ = sparse_arnoldi_iteration(L, k_steps=n_iter)
            # H is already float32 from sparse_arnoldi_iteration
            eigvals = torch.linalg.eigvals(H)
            
            # Max Imaginary Component
            max_imag = torch.max(torch.abs(eigvals.imag)).item()
            
            # Spectral Radius (max magnitude)
            spectral_radius = torch.max(torch.abs(eigvals)).item()
            
            return {
                "max_imaginary": max_imag,
                "spectral_radius": spectral_radius
            }
        except Exception as e:
            logger.warning(f"Eigenvalue computation failed: {e}")
            return {"max_imaginary": 0.0, "spectral_radius": 0.0}

    def get_fiedler_value(self, L_sym: torch.Tensor,
                          exact_threshold: int = 512,
                          seed: int = 0) -> float:
        """
        Fiedler value (second-smallest eigenvalue) of a symmetric Laplacian.

        v0.3.0 fixes two defects of the original implementation:
        1. It filtered out eigenvalues below 1e-6, which erased exactly the
           near-zero lambda_2 regime (fragmented graphs) that Fiedler-based
           hallucination diagnostics are designed to detect.
        2. Its Lanczos start vector was unseeded, so repeated calls on the
           same input returned different values.

        For graphs up to `exact_threshold` nodes the exact dense solve is
        used (a few ms on GPU and exact by construction). Larger graphs use
        seeded Lanczos with the constant vector deflated (the null direction
        of a combinatorial Laplacian), returning the second-smallest Ritz
        value clamped at zero. Lanczos values are approximations; prefer the
        exact path when latency allows.
        """
        L = L_sym.to(torch.float32)
        N = L.shape[-1]
        if N < 2:
            return 0.0

        try:
            if N <= exact_threshold:
                eigvals = torch.linalg.eigvalsh(0.5 * (L + L.transpose(-2, -1)))
                return float(eigvals[1].clamp(min=0.0))

            ones = torch.ones(N, device=L.device, dtype=L.dtype)
            ones = ones / torch.norm(ones)
            n_iter = min(N, 32)
            T = sparse_lanczos_iteration(L, k_steps=n_iter, seed=seed,
                                         deflate=ones)
            eigvals = torch.linalg.eigvalsh(T)
            idx = 1 if len(eigvals) > 1 else 0
            return float(eigvals[idx].clamp(min=0.0))
        except Exception:
            return 0.0
