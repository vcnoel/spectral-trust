import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from tests.test_gpu_topologies import (
    test_arnoldi_accuracy, 
    test_zero_matrix_edge_case, 
    test_acceleration_benchmark
)

if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    print("\n--- Running Accuracy Tests ---")
    test_arnoldi_accuracy()
    test_zero_matrix_edge_case()
    
    print("\n--- Running Performance Benchmark ---")
    test_acceleration_benchmark()
