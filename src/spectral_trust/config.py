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

from dataclasses import field, dataclass
from typing import Optional, Dict, Any
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
    display_plots: bool = False
    save_intermediate: bool = True
    verbose: bool = True

    # Model loading options
    trust_remote_code: bool = False
    torch_dtype: str = "float32"
    device_map: Optional[str] = None
    local_files_only: bool = False
    
    # Multi-run options
    runs: int = 1
    temperature: float = 1.0
    plot_metrics: Optional[list] = None  # None/empty means "all"
    latex_export: bool = False
    
    # Advanced
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    remove_self_loops: bool = False
    
    # New in v0.2.0
    directed: bool = False
    calc_velocity: bool = False
    subgraph_indices: Optional[list] = None
