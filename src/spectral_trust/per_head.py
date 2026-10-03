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

"""
Per-head spectral diagnostics.

Head-averaged attention graphs lose the signal that distinguishes normal from
anomalous routing: attention heads are specialized, so the few heads that
carry an anomaly are diluted by the rest when heads are summed. Keeping one
Laplacian per head preserves that structure, and downstream probes can learn
which heads matter.

Every metric here is a function of the attention weights alone (family
"attention" in ``spectral_trust.METRIC_FAMILIES``): no hidden states enter
the computation, so results are valid under attention-only access.

All heads are decomposed in one batched ``eigvalsh`` call, so extracting the
five metrics costs the same as extracting one.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import GSPConfig
from .graph import GraphConstructor

#: Metric names, in the column order returned by :func:`per_head_metrics`.
PER_HEAD_METRIC_NAMES: List[str] = [
    "fiedler_value",        # lambda_2: algebraic connectivity
    "connectivity_ratio",   # lambda_2 / lambda_max: scale-free connectivity
    "spectral_entropy_norm",  # eigenvalue-distribution entropy / log(S)
    "hfer",                 # eigenvalue mass in the upper half of the spectrum
    "spectral_radius",      # lambda_max
]

#: Tensor family of every metric in this module (see METRIC_FAMILIES).
PER_HEAD_METRIC_FAMILY = "attention"


@dataclass
class PerHeadDiagnostics:
    """Per-head spectral metrics for one layer.

    ``values`` is [num_heads, len(PER_HEAD_METRIC_NAMES)].
    """
    layer: int
    values: np.ndarray
    metric_names: List[str]
    family: str = PER_HEAD_METRIC_FAMILY

    def metric(self, name: str) -> np.ndarray:
        """All heads' values for one metric, as a [num_heads] array."""
        return self.values[:, self.metric_names.index(name)]

    def to_dict(self) -> Dict:
        return {
            "layer": int(self.layer),
            "metric_names": list(self.metric_names),
            "family": self.family,
            "values": self.values.tolist(),
        }


def per_head_metrics(
    attention: torch.Tensor,
    config: Optional[GSPConfig] = None,
    token_span: Optional[Tuple[int, int]] = None,
    layer: int = -1,
) -> PerHeadDiagnostics:
    """
    Compute per-head spectral metrics for a single layer.

    Args:
        attention: attention weights for one layer, ``[heads, seq, seq]`` (a
            leading batch dimension of size 1 is accepted and squeezed).
        config: graph-construction settings (symmetrization, normalization,
            self-loops). Defaults to ``GSPConfig()``, i.e. the symmetric
            normalized Laplacian, whose spectrum lies in [0, 2] independent
            of sequence length.
        token_span: optional ``(start, end)`` half-open token range. The
            induced subgraph is taken *before* the Laplacian is built, which
            is what makes the diagnostic local to a span of interest (e.g.
            a generated tool call) rather than to the whole sequence.
        layer: layer index recorded in the result.

    Returns:
        :class:`PerHeadDiagnostics` with a ``[heads, 5]`` value matrix.
    """
    if attention.dim() == 4:
        if attention.shape[0] != 1:
            raise ValueError(
                f"expected a single layer/example, got batch "
                f"{attention.shape[0]}; index the batch first"
            )
        attention = attention[0]
    if attention.dim() != 3:
        raise ValueError(
            f"attention must be [heads, seq, seq], got {tuple(attention.shape)}"
        )

    cfg = config or GSPConfig()
    gc = GraphConstructor(cfg)

    # Slice BEFORE widening the dtype. A full layer of attention at a few
    # thousand tokens is over a gigabyte in float32, and when the diagnostic
    # is restricted to a span only that submatrix is needed; converting first
    # costs both the memory and the time of the whole matrix.
    if token_span is not None:
        start, end = token_span
        attention = attention[:, start:end, start:end]
    # Decompose on the CPU. The batched symmetric eigensolver on CUDA is
    # very slow for many small matrices (measured 1.9 s against 0.1 s for
    # 32 heads x 16 layers at a 90-token span on one consumer GPU), and the
    # block moved here is a few hundred kilobytes.
    attn = attention.detach().to(device="cpu", dtype=torch.float32)

    num_heads, seq_len, _ = attn.shape
    if seq_len < 3:
        return PerHeadDiagnostics(
            layer=layer,
            values=np.zeros((num_heads, len(PER_HEAD_METRIC_NAMES)),
                            dtype=np.float32),
            metric_names=list(PER_HEAD_METRIC_NAMES),
        )

    # One Laplacian per head (heads act as the batch dimension), then a single
    # batched symmetric eigendecomposition for all of them.
    adjacency = gc.symmetrize_attention(attn.unsqueeze(0))[0]
    laplacian = gc.construct_laplacian(adjacency)
    laplacian = 0.5 * (laplacian + laplacian.transpose(-2, -1))
    # One thread: a batch of small symmetric eigenproblems is slower with more
    # threads, which oversubscribe the cores (measured at 32 heads and a
    # 250-token span: 36 ms on one thread, 1.2 s on 24). The caller's thread
    # count is restored afterwards.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        eigenvalues = torch.linalg.eigvalsh(laplacian).clamp(min=0.0)  # [H, S]
    finally:
        torch.set_num_threads(prev_threads)

    lam2 = eigenvalues[:, 1]
    lam_max = eigenvalues[:, -1]
    total = eigenvalues.sum(dim=-1, keepdim=True)

    probs = (eigenvalues / total.clamp(min=1e-12)).clamp(min=1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1) / float(np.log(seq_len))
    hfer = (eigenvalues[:, seq_len // 2:].sum(dim=-1)
            / total.squeeze(-1).clamp(min=1e-12))

    degenerate = total.squeeze(-1) <= 1e-12
    connectivity_ratio = torch.where(
        lam_max > 1e-12, lam2 / lam_max.clamp(min=1e-12),
        torch.zeros_like(lam2))
    entropy = torch.where(degenerate, torch.zeros_like(entropy), entropy)
    hfer = torch.where(degenerate, torch.zeros_like(hfer), hfer)

    values = torch.stack(
        [lam2, connectivity_ratio, entropy, hfer, lam_max], dim=-1)
    return PerHeadDiagnostics(
        layer=layer,
        values=values.detach().cpu().numpy().astype(np.float32),
        metric_names=list(PER_HEAD_METRIC_NAMES),
    )


def per_head_metrics_all_layers(
    attentions,
    config: Optional[GSPConfig] = None,
    token_span: Optional[Tuple[int, int]] = None,
) -> List[PerHeadDiagnostics]:
    """
    Per-head metrics for every layer.

    Args:
        attentions: sequence of per-layer attention tensors, each
            ``[heads, seq, seq]`` or ``[1, heads, seq, seq]`` — i.e. exactly
            what ``model(..., output_attentions=True).attentions`` returns.
        config, token_span: as in :func:`per_head_metrics`.
    """
    return [
        per_head_metrics(layer_attn, config=config,
                         token_span=token_span, layer=idx)
        for idx, layer_attn in enumerate(attentions)
    ]


def stack_feature_vector(
    diagnostics: List[PerHeadDiagnostics],
    metrics: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Flatten per-layer diagnostics into one feature vector for a probe.

    Args:
        diagnostics: output of :func:`per_head_metrics_all_layers`.
        metrics: metric subset to keep, in order (default: all five).

    Returns:
        1-D array of length ``layers * heads * len(metrics)``, ordered
        layer-major then head then metric.
    """
    if not diagnostics:
        return np.zeros(0, dtype=np.float32)
    names = metrics or diagnostics[0].metric_names
    cols = [diagnostics[0].metric_names.index(n) for n in names]
    return np.concatenate(
        [d.values[:, cols].ravel() for d in diagnostics]
    ).astype(np.float32)


def layer_delta_features(
    diagnostics: List[PerHeadDiagnostics],
    metric: str = "spectral_radius",
) -> np.ndarray:
    """
    Layer-to-layer differences of one per-head metric ("spectral dynamics").

    Captures how each head's spectrum evolves with depth, which flat
    per-layer feature sets do not represent explicitly.

    Returns:
        1-D array of length ``(layers - 1) * heads``.
    """
    if len(diagnostics) < 2:
        return np.zeros(0, dtype=np.float32)
    traj = np.stack([d.metric(metric) for d in diagnostics])  # [L, H]
    return np.diff(traj, axis=0).ravel().astype(np.float32)
