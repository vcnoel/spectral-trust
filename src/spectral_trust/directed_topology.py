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
    
    Q = torch.zeros((N, k_steps + 1), device=device, dtype=A.dtype)
    H = torch.zeros((k_steps + 1, k_steps), device=device, dtype=A.dtype)
    
    # Random start vector
    q = torch.randn(N, device=device, dtype=A.dtype)
    q = q / torch.norm(q)
    Q[:, 0] = q
    
    for j in range(k_steps):
        # Sparse-friendly matrix-vector product
        v = torch.matmul(A, Q[:, j])
        
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

def sparse_lanczos_iteration(A: torch.Tensor, k_steps: int = 20, tol: float = 1e-9) -> torch.Tensor:
    """
    Perform Lanczos iteration for symmetric matrices.
    A: (N, N) symmetric tensor
    Returns the tridiagonal matrix T.
    """
    device = A.device
    N = A.shape[0]
    k_steps = min(k_steps, N)
    
    alpha = torch.zeros(k_steps, device=device, dtype=A.dtype)
    beta = torch.zeros(k_steps, device=device, dtype=A.dtype)
    q_prev = torch.zeros(N, device=device, dtype=A.dtype)
    
    # Random start vector
    q = torch.randn(N, device=device, dtype=A.dtype)
    q = q / torch.norm(q)
    
    for j in range(k_steps):
        v = torch.matmul(A, q)
        alpha[j] = torch.dot(q, v)
        v = v - alpha[j] * q - (beta[j-1] * q_prev if j > 0 else 0)
        
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
        H, _ = sparse_arnoldi_iteration(L, k_steps=n_iter)
        
        # Find eigenvalues of the small Hessenberg matrix H
        # Since H is small, we use dense eigvals
        try:
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

    def get_fiedler_value(self, L_sym: torch.Tensor) -> float:
        """
        Extract Fiedler value (smallest non-zero eigenvalue) using Lanczos
        """
        # For small non-zero, we look at the tridiagonal matrix
        n_iter = min(L_sym.shape[-1], 20)
        T = sparse_lanczos_iteration(L_sym, k_steps=n_iter)
        
        try:
            # Smallest eigenvalues of T approximate smallest of L_sym
            eigvals = torch.linalg.eigvalsh(T)
            # Filter near-zero (the first eigenvalue of Laplacian is always 0)
            non_zero = eigvals[eigvals > 1e-6]
            if len(non_zero) > 0:
                return torch.min(non_zero).item()
            return 0.0
        except Exception:
            return 0.0
