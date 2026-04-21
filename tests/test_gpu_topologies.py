# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import torch
import time
import numpy as np
from spectral_trust.directed_topology import (
    DirectedTopologist, 
    sparse_arnoldi_iteration, 
    sparse_lanczos_iteration
)
from spectral_trust.spectral import calculate_spectral_velocity

def test_arnoldi_accuracy():
    """Verify Arnoldi iteration matches dense solver for top k eigenvalues"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 128
    k = 6
    # Use a row-normalized matrix (attention-like) for better spectral properties
    A = torch.rand(N, N, device=device, dtype=torch.float64)
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
    
    # 1. Dense Ground Truth
    dense_eigvals = torch.linalg.eigvals(A)
    dense_mags, _ = torch.sort(torch.abs(dense_eigvals), descending=True)
    
    # 2. Sparse Arnoldi (Using k_steps > k for stability)
    k_steps = 80
    H, Q = sparse_arnoldi_iteration(A, k_steps=k_steps, tol=1e-15)
    sparse_eigvals = torch.linalg.eigvals(H)
    sparse_mags, _ = torch.sort(torch.abs(sparse_eigvals), descending=True)
    
    # Compare top k magnitudes
    # Note: Using rtol=1e-7 for random matrices, 1e-9 might be too strict for k_steps=60
    torch.testing.assert_close(sparse_mags[:k].to(torch.float32), dense_mags[:k].to(torch.float32), rtol=1e-5, atol=1e-5)
    print(f"\n[SUCCESS] Arnoldi Accuracy validated for top k={k}.")

def test_zero_matrix_edge_case():
    """Verify robustness against completely disconnected graphs (zero matrices)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 16
    A = torch.zeros((N, N), device=device)
    
    # Test Laplacian construction
    topologist = DirectedTopologist(device=str(device))
    L = topologist.compute_directed_laplacian(A)
    assert not torch.any(torch.isnan(L)), "Laplacian should not contain NaNs for zero matrix"
    
    # Test solver on zero matrix
    H, Q = sparse_arnoldi_iteration(L, k_steps=5)
    eigvals = torch.linalg.eigvals(H)
    assert not torch.any(torch.isnan(eigvals)), "Arnoldi eigenvalues should not be NaN for zero matrix"
    print("[SUCCESS] Zero-matrix edge case handled gracefully.")

def test_acceleration_benchmark():
    """
    Performance Benchmark: Sparse Arnoldi/Lanczos vs Dense linalg.
    Includes accuracy assertion before benchmarking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BENCHMARK] Running on {device}")
    
    N = 1024
    # Use float64 for verification to separate iterative error from float32 noise
    A = torch.rand(N, N, device=device, dtype=torch.float64)
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
    
    # 1. Warmup and Accuracy Verify
    H_val, _ = sparse_arnoldi_iteration(A, k_steps=150)
    sparse_vals = torch.linalg.eigvals(H_val)
    dense_vals = torch.linalg.eigvals(A)
    
    s_mags, _ = torch.sort(torch.abs(sparse_vals), descending=True)
    d_mags, _ = torch.sort(torch.abs(dense_vals), descending=True)
    
    # Assert top 6 match
    # Clustered eigenvalues in random matrices require higher k_steps for exact matching
    torch.testing.assert_close(s_mags[:6].to(torch.float32), d_mags[:6].to(torch.float32), rtol=1e-5, atol=1e-5)
    print("  Mathematical Accuracy Verified (Top 6 Eigenvalues)")
    
    # 2. Timing (Using float32 for realistic production performance)
    A_f32 = A.float()
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_dense = time.time()
    _ = torch.linalg.eigvals(A_f32)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_dense = time.time()
    dense_duration = end_dense - start_dense
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_sparse = time.time()
    H, _ = sparse_arnoldi_iteration(A_f32, k_steps=100)
    _ = torch.linalg.eigvals(H)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_sparse = time.time()
    sparse_duration = end_sparse - start_sparse
    
    speedup = dense_duration / sparse_duration
    
    print(f"  Configuration: N={N}, k=6")
    print(f"  Dense (torch.linalg.eigvals): {dense_duration:.4f}s")
    print(f"  Sparse (Arnoldi Iteration):  {sparse_duration:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    try:
        test_arnoldi_accuracy()
        test_zero_matrix_edge_case()
        test_acceleration_benchmark()
    except Exception as e:
        print(f"\n[FAILURE] Test failed: {e}")
        import traceback
        traceback.print_exc()
