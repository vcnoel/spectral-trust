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
    # Default changed rw -> sym in v0.3.0: the symmetric normalized Laplacian
    # has spectrum in [0, 2] independent of sequence length, so metrics are
    # comparable across inputs of different length (the combinatorial and
    # random-walk variants scale with T, a length confound); it is also a
    # symmetric operator, so the deterministic eigh/eigsh path is always used.
    normalization: str = "sym"  # sym (symmetric), rw (random walk), none
    hfer_cutoff_ratio: float = 0.1  # High frequency cutoff as ratio of total eigenvectors
    
    # Spectral computation
    # Default changed sparse -> dense in v0.4.0. "sparse" (ARPACK) keeps only
    # num_eigenvalues modes once seq_len > 50 (half from each end of the
    # spectrum), so HFER, spectral entropy and spectral masses are normalised
    # over a truncated basis and are not comparable to a full computation (a
    # warning is logged); it was also slower than the dense solve at every
    # size benchmarked. "dense" is the full eigendecomposition.
    num_eigenvalues: int = 50  # Modes kept by eigen_solver="sparse"
    eigen_solver: str = "dense"  # dense (full eigendecomposition), sparse (ARPACK, truncated)
    lanczos_max_iter: int = 1000

    # Random projection of the signals before the hybrid metrics (energy,
    # smoothness_index, spectral_entropy, hfer, spectral_masses): a seeded
    # Gaussian Johnson-Lindenstrauss map from the hidden size d to
    # project_dim. Energy and spectral masses stay unbiased; the ratios carry
    # an O(1/sqrt(project_dim)) error (see spectral_trust.fast). None
    # disables it, and a width >= d leaves the signals unchanged. Attention
    # metrics are not affected.
    project_dim: Optional[int] = None
    project_seed: int = 0
    
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
    display_plots: bool = False
    directed: bool = False
    calc_velocity: bool = False
    subgraph_indices: Optional[list] = None
