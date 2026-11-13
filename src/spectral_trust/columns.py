# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Graph-signal metrics of each column of a signal matrix.

The layer-level hybrid metrics treat the ``d`` columns of ``X`` as one signal
and sum their energies. Many questions need one value per column instead:
how smooth one SAE feature, one neuron or one probe projection is on a token
graph. :func:`per_column_metrics` returns energy, smoothness index, spectral
entropy and HFER for every column from a single eigendecomposition, with the
rules :meth:`SpectralAnalyzer.analyze_layer` applies to that column alone.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import GSPConfig
from .fast import num_high_modes
from .spectral import SpectralAnalyzer

#: Metric names of :func:`per_column_metrics`, in column order of ``values``.
COLUMN_METRIC_NAMES: List[str] = [
    "energy",               # x^T L x
    "smoothness_index",     # energy / ||x||^2
    "spectral_entropy",     # entropy of the per-mode energies (natural log)
    "hfer",                 # energy share of the top cutoff_ratio of modes
]


@dataclass
class ColumnDiagnostics:
    """Per-column metrics of one signal matrix on one graph.

    ``values`` is ``[d, 4]`` over :data:`COLUMN_METRIC_NAMES`, one row per
    column of the signals. ``eigenvalues`` is the graph's spectrum, in
    ascending order.
    """
    values: np.ndarray
    metric_names: List[str]
    eigenvalues: np.ndarray

    def metric(self, name: str) -> np.ndarray:
        """One metric for every column, as a ``[d]`` array."""
        return self.values[:, self.metric_names.index(name)]

    def to_dict(self) -> Dict:
        return {"metric_names": list(self.metric_names),
                "values": self.values.tolist()}


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(x, dtype=np.float32)


def per_column_metrics(signals, laplacian, config: Optional[GSPConfig] = None
                       ) -> ColumnDiagnostics:
    """
    Energy, smoothness index, spectral entropy and HFER of each column of
    ``signals`` on the graph of ``laplacian``.

    Column ``j`` gets exactly what
    ``SpectralAnalyzer(config).analyze_layer(signals[:, j:j+1], laplacian, layer)``
    reports for it (the same eigenbasis, mode-count rule and thresholds: zero
    HFER and entropy below a total energy of 1e-8, zero smoothness below a
    squared norm of 1e-8), from one eigendecomposition for all columns.
    ``config.project_dim`` does not apply, since each column is its own
    one-dimensional signal.

    Example, the attention-graph HFER of every SAE feature on a passage::

        gc = st.GraphConstructor(cfg)
        adj = gc.aggregate_heads(gc.symmetrize_attention(attn[None]))  # [1, S, S]
        L = gc.construct_laplacian(adj)[0]
        cols = st.per_column_metrics(feature_acts, L, cfg)  # [S, n_features]
        cols.metric("hfer")                                  # [n_features]

    Args:
        signals: ``[seq, d]`` (or ``[seq]`` for one column), a torch tensor
            or an array.
        laplacian: ``[seq, seq]`` (a leading dimension of size 1 is
            accepted), e.g. from :meth:`GraphConstructor.construct_laplacian`.
        config: supplies ``eigen_solver`` and ``hfer_cutoff_ratio``.
            Defaults to ``GSPConfig(device="cpu")``.

    Returns:
        :class:`ColumnDiagnostics` with a ``[d, 4]`` value matrix.
    """
    cfg = config or GSPConfig(device="cpu", verbose=False)
    X = _as_numpy(signals)
    if X.ndim == 1:
        X = X[:, None]
    L = _as_numpy(laplacian)
    if L.ndim > 2:
        if int(np.prod(L.shape[:-2])) != 1:
            raise ValueError(f"expected one [S, S] Laplacian, got shape {L.shape}")
        L = L.reshape(L.shape[-2:])
    if X.ndim != 2 or L.ndim != 2 or L.shape[0] != L.shape[1] or X.shape[0] != L.shape[0]:
        raise ValueError(f"signals must be [seq, d] and laplacian [seq, seq], got "
                         f"{X.shape} and {L.shape}")

    eigenvalues, U = SpectralAnalyzer(cfg).compute_eigendecomposition(L)
    energy = np.sum(X * np.dot(L, X), axis=0, dtype=np.float64)           # [d]
    norm = np.sum(X * X, axis=0, dtype=np.float64)
    smoothness = np.where(norm >= 1e-8, energy / np.maximum(norm, 1e-300), 0.0)

    masses = np.dot(U.T, X) ** 2                                          # [k, d]
    total = masses.sum(axis=0)
    k = masses.shape[0]
    cut = k - num_high_modes(k, cfg.hfer_cutoff_ratio)
    live = total >= 1e-8
    safe = np.where(live, total, 1.0)
    hfer = np.where(live, masses[cut:].sum(axis=0) / safe, 0.0)
    p = np.maximum(masses / safe, 1e-12)
    entropy = np.where(live, -(p * np.log(p)).sum(axis=0), 0.0)

    values = np.stack([energy, smoothness, entropy, hfer], axis=1).astype(np.float32)
    return ColumnDiagnostics(values=values, metric_names=list(COLUMN_METRIC_NAMES),
                             eigenvalues=np.asarray(eigenvalues, dtype=np.float32))
