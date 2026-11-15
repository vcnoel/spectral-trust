# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""
Orientation and capacity of an attention head, after Dahlem, Maniloff and
Misiura, "Self-Attention as Transport: Orientation Blindness and the Limits
of Symmetric Spectral Diagnostics" (arXiv 2605.04893, v3).

A head's row-stochastic attention ``B`` is read through the degree-normalised
operator (their Definition 1, eq. 1)::

    M = D_Q^{-1/2} B D_K^{-1/2},   D_Q = diag(B 1),  D_K = diag(Bᵀ 1).

Its Frobenius-orthogonal split ``M = M_sym + M_asym`` separates capacity
(``M_sym``: every transpose-invariant spectral diagnostic lives there, so it
cannot tell ``M`` from ``Mᵀ``) from orientation (``M_asym``).

- :func:`asymmetry` is their coefficient ``G = ‖M_asym‖_F / (‖M‖_F + ε)``
  (eq. 7);
- :func:`capacity_phi` is their spectral-sweep conductance estimate ``φ̂``
  (Sections 3 and 5), and :func:`transport_sigma2` the second singular value
  of ``M`` that the Cheeger inequality ties to it.

Implemented from the paper's equations: the authors have released no code
(checked 15 Nov 2026), so these functions are not checked against theirs.
"""
from typing import Optional

import numpy as np
import torch

from .sinks import _t, _out, _check_square

__all__ = ["transport_operator", "asymmetry", "capacity_phi", "transport_sigma2"]


def transport_operator(attention):
    """``M = D_Q^{-1/2} B D_K^{-1/2}`` for each ``[..., S, S]`` map ``B``
    (row and column sums as degrees; a zero degree gives a zero row or
    column). For a row-stochastic ``B`` its largest singular value is 1
    (Dahlem et al., Theorem 3(i)). Returns a float64 tensor."""
    B = _t(attention)
    dq, dk = B.sum(-1), B.sum(-2)

    def inv_sqrt(d):
        return torch.where(d > 0, 1.0 / torch.sqrt(torch.where(d > 0, d, torch.ones_like(d))),
                           torch.zeros_like(d))
    return inv_sqrt(dq).unsqueeze(-1) * B * inv_sqrt(dk).unsqueeze(-2)


def asymmetry(attention, normalize: Optional[str] = "degree", eps: float = 0.0):
    """Asymmetry coefficient ``G = ‖M_asym‖_F / (‖M‖_F + ε)``, with
    ``M_asym = (M − Mᵀ) / 2`` (Dahlem et al., eq. 7).

    ``G`` lies in ``[0, 1]``: 0 iff ``M`` is symmetric, 1 iff it is skew. A
    causal map has ``G <= 1/√2``, reached by a strictly lower-triangular map;
    ``G`` near 0 in a decoder means the head attends to itself only (their
    "temporal isolation"). ``G(M) = G(Mᵀ)``: it measures how much
    orientation there is, not which way it points.

    Args:
        attention: ``[..., S, S]``.
        normalize: ``"degree"`` (default) measures the degree-normalised
            operator ``M`` of the paper; ``None`` measures the map itself,
            ``‖A_asym‖_F / ‖A‖_F``.
        eps: the stabiliser ``ε`` of eq. 7 (the paper's ``G(M, 0)`` is
            ``eps=0``). A zero map gives 0.

    Returns:
        ``[...]`` (a float for one map).
    """
    if normalize not in ("degree", None):
        raise ValueError(f"normalize must be 'degree' or None, got {normalize!r}")
    A = _t(attention)
    _check_square(A)
    M = transport_operator(A) if normalize == "degree" else A
    skew = 0.5 * (M - M.transpose(-1, -2))
    num = torch.linalg.matrix_norm(skew)
    den = torch.linalg.matrix_norm(M) + eps
    return _out(torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)),
                            torch.zeros_like(num)))


def transport_sigma2(attention):
    """Second singular value ``σ₂`` of ``M`` per map (``[...]``); the
    spectral gap ``1 − σ₂`` bounds the conductance through the Cheeger
    inequality ``(1 − σ₂)/2 <= φ <= sqrt(2 (1 − σ₂))`` (Dahlem et al., eq. 3)."""
    s = torch.linalg.svdvals(transport_operator(attention))
    return _out(s[..., 1] if s.shape[-1] > 1 else torch.zeros(s.shape[:-1], dtype=s.dtype))


def _sweep_conductance(X: np.ndarray, order: np.ndarray) -> float:
    """Minimum conductance over the prefix cuts of ``order`` in the bipartite
    graph ``H(X)`` (queries first, then keys)."""
    nq, nk = X.shape
    W = np.zeros((nq + nk, nq + nk))
    W[:nq, nq:] = X
    W[nq:, :nq] = X.T
    deg = W.sum(1)
    Wp = W[np.ix_(order, order)]
    vol = np.cumsum(deg[order])[:-1]
    within = np.cumsum(np.tril(Wp, -1).sum(1))[:-1]
    cut = vol - 2.0 * within
    small = np.minimum(vol, deg.sum() - vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(small > 0, cut / small, np.inf)
    return float(phi.min()) if phi.size else float("nan")


def capacity_phi(attention, graph: str = "normalized"):
    """Spectral-sweep conductance estimate ``φ̂`` of a head's bipartite
    transport graph (Dahlem et al., Sections 3 and 5).

    The ``S`` queries and ``S`` keys are ranked by the second singular
    vectors of ``M`` (the vector ``(u₂; v₂)``, the second eigenvector of the
    dilation ``H(M)``), and ``φ̂`` is the smallest conductance
    ``w(S, S̄) / min(vol S, vol S̄)`` over the ``2S − 1`` threshold cuts of
    that ranking. Low ``φ̂``: a bottleneck (concentrated routing); high:
    diffuse routing. An estimate: the exact conductance is NP-hard and can
    only be lower.

    Args:
        attention: ``[..., S, S]`` row-stochastic maps.
        graph: ``"normalized"`` (default) measures the cuts on
            ``N_bip = H(M)``, the graph the paper's estimator sweeps;
            ``"raw"`` on ``A_bip = H(B)``, the raw attention weights.

    Returns:
        ``[...]`` (a float for one map). When ``σ₂`` is repeated the ranking,
        and so ``φ̂``, depends on the solver's choice of singular vectors.
    """
    if graph not in ("normalized", "raw"):
        raise ValueError(f"graph must be 'normalized' or 'raw', got {graph!r}")
    A = _t(attention)
    _check_square(A)
    M = transport_operator(A).numpy()
    B = A.numpy()
    lead = M.shape[:-2]
    out = np.empty(int(np.prod(lead)) if lead else 1)
    for k, (m, b) in enumerate(zip(M.reshape(-1, *M.shape[-2:]), B.reshape(-1, *B.shape[-2:]))):
        if m.shape[0] < 2:
            out[k] = float("nan")
            continue
        u, _, vt = np.linalg.svd(m)
        order = np.argsort(-np.concatenate([u[:, 1], vt[1]]), kind="stable")
        out[k] = _sweep_conductance(m if graph == "normalized" else b, order)
    return float(out[0]) if not lead else out.reshape(lead)
