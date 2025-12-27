from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class GSPConfig:
    """Configuration for GSP diagnostics"""
    # Model parameters
    model_name: str = "gpt2"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GSP parameters
    head_aggregation: str = "uniform"  # uniform, attention_weighted, learnable
    symmetrization: str = "symmetric"  # symmetric, row_norm, col_norm
    normalization: str = "rw"  # rw (random walk), sym (symmetric), none
    hfer_cutoff_ratio: float = 0.1  # High frequency cutoff as ratio of total eigenvectors
    
    # Spectral computation
    num_eigenvalues: int = 50  # Number of eigenvalues to compute for spectral analysis
    eigen_solver: str = "sparse"  # sparse (ARPACK), dense (full eigendecomposition)
    lanczos_max_iter: int = 1000
    
    # Evaluation parameters
    batch_size: int = 1
    num_layers_analyze: Optional[int] = None  # None means all layers
    save_attention: bool = True
    save_activations: bool = True
    
    # Output parameters
    output_dir: str = "./gsp_results"
    save_plots: bool = True
    save_intermediate: bool = True
    verbose: bool = True

    # Model loading options
    trust_remote_code: bool = False
    torch_dtype: str = "float32"
    device_map: Optional[str] = None
    local_files_only: bool = False
