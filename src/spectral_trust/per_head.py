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

:func:`per_head_metrics` is a function of the attention weights alone (family
"attention" in ``spectral_trust.METRIC_FAMILIES``): no hidden states enter
the computation, so results are valid under attention-only access. All heads
are decomposed in one batched ``eigvalsh`` call, so extracting the five
metrics costs the same as extracting one.

:func:`per_head_hybrid_metrics` is the per-head counterpart of the hybrid
metrics (energy, smoothness index, spectral entropy, HFER): the layer's
residual-stream states projected onto each head's Laplacian eigenbasis.
It needs hidden states and is family "hybrid".

Both accept ``heads=`` to compute only a subset of heads, e.g. the heads an
attribution method ranked highest. The attribution stays outside the
library: pass its scores through :func:`heads_from_scores`, or pass a
boolean mask or ``(layer, head)`` pairs directly.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import GSPConfig
from .fast import gaussian_projection
from .graph import GraphConstructor
from .spectral import SpectralAnalyzer

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

#: Metric names of :func:`per_head_hybrid_metrics`, in column order. They
#: are the per-head versions of the SpectralDiagnostics fields of the same
#: name.
PER_HEAD_HYBRID_METRIC_NAMES: List[str] = [
    "energy",               # tr(X^T L_h X)
    "smoothness_index",     # energy / ||X||_F^2
    "spectral_entropy",     # entropy of the per-mode energies (natural log)
    "hfer",                 # energy share of the top cutoff_ratio of modes
]

#: Tensor family of :func:`per_head_hybrid_metrics`.
PER_HEAD_HYBRID_METRIC_FAMILY = "hybrid"

#: Accepted forms of a head selection (see :func:`per_head_metrics`).
HeadSelection = Union[None, Sequence[int], Sequence[Tuple[int, int]],
                      np.ndarray, torch.Tensor]


@dataclass
class PerHeadDiagnostics:
    """Per-head spectral metrics for one layer.

    ``values`` is [num_heads, len(metric_names)]. When only a subset of heads
    was computed (``heads=``), ``heads`` lists their indices in row order;
    it is None when every head was computed.
    """
    layer: int
    values: np.ndarray
    metric_names: List[str]
    family: str = PER_HEAD_METRIC_FAMILY
    heads: Optional[List[int]] = None

    def metric(self, name: str) -> np.ndarray:
        """All heads' values for one metric, as a [num_heads] array."""
        return self.values[:, self.metric_names.index(name)]

    def to_dict(self) -> Dict:
        d = {
            "layer": int(self.layer),
            "metric_names": list(self.metric_names),
            "family": self.family,
            "values": self.values.tolist(),
        }
        if self.heads is not None:
            d["heads"] = list(self.heads)
        return d


def heads_from_scores(scores, k: int) -> List[Tuple[int, int]]:
    """
    The ``k`` highest-scoring heads of a ``[layers, heads]`` score matrix,
    as ``(layer, head)`` pairs sorted by layer then head, ready for
    ``heads=``. NaN scores are never selected; ties keep row-major order.

    The scores come from outside the library (an attribution method, a
    probe's weights, an ablation study); this only ranks them.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"scores must be [layers, heads], got shape {s.shape}")
    order = np.argsort(-s, axis=None, kind="stable")          # NaN sorts last
    order = [i for i in order[: max(int(k), 0)] if not np.isnan(s.flat[i])]
    num_heads = s.shape[1]
    return sorted((int(i // num_heads), int(i % num_heads)) for i in order)


def _resolve_heads(heads: HeadSelection, layer: int,
                   num_heads: int) -> Optional[List[int]]:
    """Normalise a head selection for one layer to sorted head indices."""
    if heads is None:
        return None
    if isinstance(heads, torch.Tensor):
        heads = heads.detach().cpu().numpy()
    mask = None
    if isinstance(heads, np.ndarray) and heads.dtype == bool:
        mask = heads
    elif (isinstance(heads, (list, tuple)) and len(heads) > 0
          and all(isinstance(h, (bool, np.bool_)) for h in heads)):
        mask = np.asarray(heads, dtype=bool)
    if mask is not None:
        if mask.ndim == 2:
            if not 0 <= layer < mask.shape[0]:
                raise ValueError(
                    f"a [layers, heads] mask needs a layer index in "
                    f"[0, {mask.shape[0]}), got layer={layer}")
            mask = mask[layer]
        if mask.ndim != 1 or mask.shape[0] != num_heads:
            raise ValueError(
                f"head mask must be [{num_heads}] or [layers, {num_heads}], "
                f"got {mask.shape}")
        return [int(h) for h in np.flatnonzero(mask)]

    items = heads.tolist() if isinstance(heads, np.ndarray) else list(heads)
    if items and all(isinstance(it, (tuple, list)) and len(it) == 2 for it in items):
        if layer < 0:
            raise ValueError("(layer, head) pairs need the layer index: pass layer=")
        selected = {int(h) for lay, h in items if int(lay) == layer}
    elif all(isinstance(it, (int, np.integer)) and not isinstance(it, bool)
             for it in items):
        selected = {int(h) for h in items}
    else:
        raise TypeError(
            "heads must be None, a boolean mask, head indices, or "
            "(layer, head) pairs")
    bad = sorted(h for h in selected if not 0 <= h < num_heads)
    if bad:
        raise IndexError(f"head indices {bad} out of range for {num_heads} heads")
    return sorted(selected)


def _layer_attention(attention: torch.Tensor) -> torch.Tensor:
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
    return attention


def _select_heads(attention: torch.Tensor,
                  selected: Optional[List[int]]) -> torch.Tensor:
    if selected is None:
        return attention
    idx = torch.as_tensor(selected, dtype=torch.long, device=attention.device)
    return attention.index_select(0, idx)


def per_head_metrics(
    attention: torch.Tensor,
    config: Optional[GSPConfig] = None,
    token_span: Optional[Tuple[int, int]] = None,
    layer: int = -1,
    heads: HeadSelection = None,
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
        heads: compute only these heads. One of: head indices for this layer
            (``[0, 5, 7]``); ``(layer, head)`` pairs, of which those with
            ``layer == layer`` are kept (requires ``layer >= 0``); a boolean
            mask ``[heads]`` or ``[layers, heads]`` (row ``layer`` is used).
            Unselected heads are dropped before any conversion or
            decomposition. Rows of the result follow ascending head index
            and ``result.heads`` lists them.

    Returns:
        :class:`PerHeadDiagnostics` with a ``[heads, 5]`` value matrix.
    """
    attention = _layer_attention(attention)
    selected = _resolve_heads(heads, layer, attention.shape[0])
    attention = _select_heads(attention, selected)

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
    if seq_len < 3 or num_heads == 0:
        return PerHeadDiagnostics(
            layer=layer,
            values=np.zeros((num_heads, len(PER_HEAD_METRIC_NAMES)),
                            dtype=np.float32),
            metric_names=list(PER_HEAD_METRIC_NAMES),
            heads=selected,
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
        heads=selected,
    )


def per_head_metrics_all_layers(
    attentions,
    config: Optional[GSPConfig] = None,
    token_span: Optional[Tuple[int, int]] = None,
    heads: HeadSelection = None,
) -> List[PerHeadDiagnostics]:
    """
    Per-head metrics for every layer.

    Args:
        attentions: sequence of per-layer attention tensors, each
            ``[heads, seq, seq]`` or ``[1, heads, seq, seq]`` — i.e. exactly
            what ``model(..., output_attentions=True).attentions`` returns.
        config, token_span: as in :func:`per_head_metrics`.
        heads: as in :func:`per_head_metrics`, resolved per layer: ``(layer,
            head)`` pairs (e.g. from :func:`heads_from_scores`) or a
            ``[layers, heads]`` mask select different heads in each layer,
            plain indices or a ``[heads]`` mask the same heads everywhere.
    """
    return [
        per_head_metrics(layer_attn, config=config,
                         token_span=token_span, layer=idx, heads=heads)
        for idx, layer_attn in enumerate(attentions)
    ]


@lru_cache(maxsize=8)
def _projection(dim: int, k: int, seed: int) -> torch.Tensor:
    return torch.from_numpy(gaussian_projection(dim, k, seed=seed, dtype=np.float32))


def per_head_hybrid_metrics(
    attention: torch.Tensor,
    signals: torch.Tensor,
    config: Optional[GSPConfig] = None,
    token_span: Optional[Tuple[int, int]] = None,
    layer: int = -1,
    heads: HeadSelection = None,
) -> PerHeadDiagnostics:
    """
    Per-head hybrid metrics for one layer: the layer's residual-stream
    signals measured on each head's attention graph.

    For each selected head h, the head's attention is symmetrized and turned
    into a Laplacian ``L_h`` exactly as in :func:`per_head_metrics`; then
    ``energy = tr(X^T L_h X)``, ``smoothness_index = energy / ||X||_F^2``,
    and ``spectral_entropy`` / ``hfer`` follow
    :class:`~spectral_trust.spectral.SpectralAnalyzer` on the full
    eigenbasis of ``L_h`` (never truncated). Values match
    ``SpectralAnalyzer(eigen_solver="dense").analyze_layer`` run on each
    head's Laplacian. ``config.project_dim`` applies the same
    Johnson-Lindenstrauss projection to ``X`` first.

    Args:
        attention: ``[heads, seq, seq]`` (or ``[1, heads, seq, seq]``).
        signals: ``[seq, d]`` (or ``[1, seq, d]``), the residual-stream
            states paired with this layer (``hidden_states[layer + 1]`` in
            the Hugging Face convention, as in the framework).
        config, token_span, layer, heads: as in :func:`per_head_metrics`.

    Returns:
        :class:`PerHeadDiagnostics` with family ``"hybrid"`` and a
        ``[heads, 4]`` matrix over :data:`PER_HEAD_HYBRID_METRIC_NAMES`.
    """
    attention = _layer_attention(attention)
    if signals.dim() == 3:
        if signals.shape[0] != 1:
            raise ValueError("expected signals for a single example; index the batch first")
        signals = signals[0]
    if signals.dim() != 2 or signals.shape[0] != attention.shape[-1]:
        raise ValueError(
            f"signals must be [seq, d] with seq={attention.shape[-1]}, "
            f"got {tuple(signals.shape)}")
    selected = _resolve_heads(heads, layer, attention.shape[0])
    attention = _select_heads(attention, selected)

    cfg = config or GSPConfig()
    gc = GraphConstructor(cfg)
    if token_span is not None:
        start, end = token_span
        attention = attention[:, start:end, start:end]
        signals = signals[start:end]
    attn = attention.detach().to(device="cpu", dtype=torch.float32)
    X = signals.detach().to(device="cpu", dtype=torch.float32)
    k = cfg.project_dim
    if k is not None:
        if k <= 0:
            raise ValueError(f"project_dim must be positive, got {k}")
        if k < X.shape[1]:
            X = X @ _projection(X.shape[1], int(k), int(cfg.project_seed))

    num_heads, seq_len, _ = attn.shape
    names = list(PER_HEAD_HYBRID_METRIC_NAMES)
    if num_heads == 0 or seq_len == 0:
        return PerHeadDiagnostics(
            layer=layer, values=np.zeros((num_heads, len(names)), dtype=np.float32),
            metric_names=names, family=PER_HEAD_HYBRID_METRIC_FAMILY, heads=selected)

    adjacency = gc.symmetrize_attention(attn.unsqueeze(0))[0]
    laplacian = gc.construct_laplacian(adjacency)                # [H, S, S]

    if cfg.normalization == "rw":
        # Non-symmetric operator: defer to the analyzer's general
        # eigensolver, head by head, exactly as the layer-level path does.
        an = SpectralAnalyzer(GSPConfig(normalization="rw", eigen_solver="dense",
                                        hfer_cutoff_ratio=cfg.hfer_cutoff_ratio,
                                        save_intermediate=False, device="cpu"))
        rows = []
        for h in range(num_heads):
            d = an.analyze_layer(X, laplacian[h], layer)
            rows.append([d.energy, d.smoothness_index, d.spectral_entropy, d.hfer])
        values = np.asarray(rows, dtype=np.float32)
    else:
        energy = (X.unsqueeze(0) * (laplacian @ X)).sum(dim=(-2, -1))   # [H]
        signal_norm = float((X * X).sum())
        smoothness = (energy / signal_norm if signal_norm >= 1e-8
                      else torch.zeros_like(energy))
        sym = 0.5 * (laplacian + laplacian.transpose(-2, -1))
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)     # see per_head_metrics
        try:
            _, vecs = torch.linalg.eigh(sym)                          # [H, S, S]
        finally:
            torch.set_num_threads(prev_threads)
        masses = ((vecs.transpose(-2, -1) @ X) ** 2).sum(dim=-1)      # [H, S]
        total = masses.sum(dim=-1)
        probs = (masses / total.unsqueeze(-1).clamp(min=1e-30)).clamp(min=1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)
        cutoff = int((1 - cfg.hfer_cutoff_ratio) * seq_len)
        hfer = masses[:, cutoff:].sum(dim=-1) / total.clamp(min=1e-30)
        empty = total < 1e-8
        entropy = torch.where(empty, torch.zeros_like(entropy), entropy)
        hfer = torch.where(empty, torch.zeros_like(hfer), hfer)
        values = torch.stack([energy, smoothness, entropy, hfer], dim=-1) \
            .numpy().astype(np.float32)
    return PerHeadDiagnostics(layer=layer, values=values, metric_names=names,
                              family=PER_HEAD_HYBRID_METRIC_FAMILY, heads=selected)


def per_head_hybrid_metrics_all_layers(
    attentions,
    hidden_states,
    config: Optional[GSPConfig] = None,
    token_span: Optional[Tuple[int, int]] = None,
    heads: HeadSelection = None,
) -> List[PerHeadDiagnostics]:
    """
    :func:`per_head_hybrid_metrics` for every layer.

    ``hidden_states`` may have one more entry than ``attentions`` (the Hugging
    Face convention, embeddings first): layer i is then paired with
    ``hidden_states[i + 1]``, as in the framework. With equal lengths layer i
    is paired with ``hidden_states[i]``.
    """
    n, m = len(attentions), len(hidden_states)
    if m == n + 1:
        offset = 1
    elif m == n:
        offset = 0
    else:
        raise ValueError(
            f"need len(hidden_states) == len(attentions) (+1); got {m} and {n}")
    return [
        per_head_hybrid_metrics(layer_attn, hidden_states[idx + offset],
                                config=config, token_span=token_span,
                                layer=idx, heads=heads)
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
        layer-major then head then metric. With head subsets (``heads=``)
        each layer contributes its selected heads, in ascending order.
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
    per-layer feature sets do not represent explicitly. Needs the same
    number of heads in every layer (all heads, or one head mask for all).

    Returns:
        1-D array of length ``(layers - 1) * heads``.
    """
    if len(diagnostics) < 2:
        return np.zeros(0, dtype=np.float32)
    traj = np.stack([d.metric(metric) for d in diagnostics])  # [L, H]
    return np.diff(traj, axis=0).ravel().astype(np.float32)
