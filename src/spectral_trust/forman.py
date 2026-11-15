# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Forman curvature of attention graphs, per edge, with node and span
summaries.

Edges. An entry ``A[i, j] > 0`` with ``i ≠ j`` is an edge ``e = (i, j)``
from the query i to the key j (self-loops are left out, and the weights are
not renormalised). Its weight is ``w_e = max(-log a_e, ε)`` with
``ε = 1e-9`` (``weights="neg_log"``, the default), so a weak attention
weight is a long edge; ``weights="attention"`` keeps ``w_e = a_e`` and
``weights="unit"`` gives every edge weight 1.

Directed curvature (Sreejith, Jost, Saucan and Samal 2017), with unit vertex
weights:

    F(v₁, v₂) = 2 − √w_e ( Σ_{e' ∈ E_in(v₁)} 1/√w_e' + Σ_{e' ∈ E_out(v₂)} 1/√w_e' ),

where ``E_in(v₁)`` are the edges into the source (queries that attend to
v₁) and ``E_out(v₂)`` the edges out of the target (keys that v₂ attends
to). An edge whose source nobody attends to and whose target attends to
nobody (an edge to the first token, which has no outgoing edge once
self-loops are removed) has curvature exactly 2, the maximum.

Undirected curvature (Forman 2003, Sreejith et al. 2016), for a symmetric
adjacency, with unit vertex weights:

    F(v₁, v₂) = 2 − √w_e ( Σ_{e' ∋ v₁, e' ≠ e} 1/√w_e' + Σ_{e' ∋ v₂, e' ≠ e} 1/√w_e' ),

which for unit weights is the combinatorial ``4 − deg(v₁) − deg(v₂)``.

Summaries. :func:`forman_quantiles` pools the curvatures of the outgoing
edges of a set of tokens and returns their quantiles at
:data:`FORMAN_QUANTILES` (linear interpolation; zeros when the set has no
edge). On the graph of a prefix, the tokens of a step give the step's
summary; one vector per layer, stacked over layers, is the step's curvature
vector. :func:`forman_node_curvature` reduces the edges of each token.

The directed edge curvature, its weights and the quantile summary are
implemented from the equations of Kim, Lutzeyer and Guo, "Attention-Graph
Curvature for the Verification of Mathematical Reasoning" (2026,
arXiv:2610.16709), Section 2.1 and Appendix A; not checked against official
code (theirs is released under AGPL-3.0 and was not read or run). Their
per-step verifiers are not part of this module.
"""
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import torch

#: Quantile levels of the curvature summary.
FORMAN_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Floor of the negative-log edge weight.
FORMAN_EPS = 1e-9

#: Named edge-weight transforms of :func:`forman_curvature`.
FORMAN_WEIGHTS = ("neg_log", "attention", "unit")

Weights = Union[str, Callable[[torch.Tensor], torch.Tensor]]
Tokens = Union[None, int, Tuple[int, int], Sequence[int]]


def _edge_weights(a: torch.Tensor, weights: Weights, eps: float) -> torch.Tensor:
    """Edge weights ``w`` on the edges (``a > 0``); undefined elsewhere."""
    if callable(weights):
        w = torch.as_tensor(weights(a), dtype=torch.float64)
    elif weights == "neg_log":
        w = torch.clamp(-torch.log(torch.where(a > 0, a, torch.ones_like(a))), min=eps)
    elif weights == "attention":
        w = a
    elif weights == "unit":
        w = torch.ones_like(a)
    else:
        raise ValueError(f"weights must be one of {FORMAN_WEIGHTS} or a callable, got {weights!r}")
    edge = a > 0
    if (edge & ~(torch.isfinite(w) & (w > 0))).any():
        raise ValueError("edge weights must be finite and positive on every edge")
    return w


def forman_curvature(adjacency, directed: bool = True, weights: Weights = "neg_log",
                     eps: float = FORMAN_EPS, average_heads: bool = False) -> torch.Tensor:
    r"""Forman curvature of every edge of a weighted graph (module notes).

    Args:
        adjacency: ``[..., S, S]`` non-negative edge weights, row i the query
            (source) and column j the key (target): an attention map, a
            head-averaged attention, value-weighted edges or any graph.
            Entries ``> 0`` off the diagonal are the edges; the diagonal is
            ignored.
        directed: the directed curvature (edges into the source, edges out
            of the target). With False the adjacency must be symmetric and
            the undirected curvature is returned (a symmetric matrix).
        weights: ``"neg_log"`` (``w = max(-log a, eps)``, for weights in
            ``(0, 1]`` such as attention), ``"attention"`` (``w = a``),
            ``"unit"`` (``w = 1``, the combinatorial curvature) or a
            callable mapping the adjacency to positive weights.
        eps: floor of the negative-log weight.
        average_heads: average over dimension -3 (heads) first.

    Returns:
        ``[..., S, S]`` float64: ``F(i, j)`` on each edge, NaN where there is
        no edge (and on the diagonal).
    """
    a = adjacency if isinstance(adjacency, torch.Tensor) else torch.as_tensor(np.asarray(adjacency))
    if a.dim() < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"adjacency must be [..., S, S], got {list(a.shape)}")
    a = a.detach().to(torch.float64)
    if average_heads:
        a = a.mean(dim=-3)
    if not torch.isfinite(a).all() or (a < 0).any():
        raise ValueError("adjacency must be finite and non-negative")
    S = a.shape[-1]
    eye = torch.eye(S, dtype=torch.bool, device=a.device)
    a = a.masked_fill(eye, 0.0)
    if not directed and not torch.allclose(a, a.transpose(-1, -2), rtol=1e-6, atol=1e-12):
        raise ValueError("directed=False needs a symmetric adjacency; symmetrise it first "
                         "(GraphConstructor.symmetrize_attention) or use directed=True")
    if not directed:
        a = 0.5 * (a + a.transpose(-1, -2))       # exact symmetry for a symmetric result
    edge = a > 0
    w = _edge_weights(a, weights, eps)
    sqrt_w = torch.where(edge, w, torch.ones_like(w)).sqrt()
    inv = torch.where(edge, 1.0 / sqrt_w, torch.zeros_like(sqrt_w))     # 1/√w on edges
    if directed:
        into = inv.sum(dim=-2)              # Σ over edges into v (column sums)
        out = inv.sum(dim=-1)               # Σ over edges out of v (row sums)
        total = into.unsqueeze(-1) + out.unsqueeze(-2)                   # [v₁ in] + [v₂ out]
    else:
        deg = inv.sum(dim=-1)               # Σ over edges at v
        total = deg.unsqueeze(-1) + deg.unsqueeze(-2) - 2.0 * inv        # both ends, e left out
    F = 2.0 - sqrt_w * total
    return torch.where(edge, F, torch.full_like(F, float("nan")))


def _token_mask(tokens: Tokens, S: int) -> torch.Tensor:
    mask = torch.zeros(S, dtype=torch.bool)
    if tokens is None:
        mask[:] = True
    elif isinstance(tokens, (int, np.integer)):
        mask[int(tokens)] = True
    elif isinstance(tokens, tuple) and len(tokens) == 2 and not isinstance(tokens[0], bool):
        start, end = tokens
        mask[slice(start, end)] = True
    else:
        mask[torch.as_tensor(list(tokens), dtype=torch.long)] = True
    return mask


def _edges_of(curvature: torch.Tensor, direction: str) -> torch.Tensor:
    """``[..., S, S']`` with row v holding the curvatures of the edges of v."""
    if direction == "out":
        return curvature
    if direction == "in":
        return curvature.transpose(-1, -2)
    if direction == "all":
        return torch.cat([curvature, curvature.transpose(-1, -2)], dim=-1)
    raise ValueError(f"direction must be 'out', 'in' or 'all', got {direction!r}")


def forman_quantiles(curvature, tokens: Tokens = None,
                     quantiles: Sequence[float] = FORMAN_QUANTILES, direction: str = "out",
                     empty: float = 0.0) -> torch.Tensor:
    """Quantiles of the curvatures of the edges of a set of tokens, pooled.

    Args:
        curvature: ``[..., S, S]`` from :func:`forman_curvature`.
        tokens: the tokens whose edges are pooled: None (all), one index, a
            ``(start, end)`` span (slice semantics) or a list of indices.
        quantiles: levels in ``[0, 1]``; linear interpolation between order
            statistics.
        direction: ``"out"`` (edges leaving the tokens, the default),
            ``"in"`` or ``"all"``.
        empty: the value of every quantile when the set has no edge.

    Returns:
        ``[..., len(quantiles)]``.

    For the curvature vector of a step, take the curvature of the graph of
    the prefix that ends with the step, ``forman_curvature(A[..., :T, :T])``,
    and the step's tokens as a span ``(T_prev, T)``; one vector per layer,
    stacked over layers.
    """
    F = torch.as_tensor(curvature, dtype=torch.float64)
    S = F.shape[-1]
    rows = _edges_of(F, direction)[..., _token_mask(tokens, S), :]
    flat = rows.reshape(*rows.shape[:-2], -1)
    q = torch.as_tensor(list(quantiles), dtype=torch.float64)
    has = (~torch.isnan(flat)).any(dim=-1, keepdim=True)
    filled = torch.where(has, flat, torch.zeros_like(flat))         # all-NaN rows → 0 (masked)
    out = torch.nanquantile(filled, q, dim=-1, interpolation="linear").movedim(0, -1)
    return torch.where(has, out, torch.full_like(out, float(empty)))


_REDUCE = {"mean": torch.nanmean, "median": lambda x, dim: torch.nanmedian(x, dim=dim).values,
           "sum": torch.nansum,
           "min": lambda x, dim: torch.where(torch.isnan(x), torch.inf, x).amin(dim=dim),
           "max": lambda x, dim: torch.where(torch.isnan(x), -torch.inf, x).amax(dim=dim)}


def forman_node_curvature(curvature, direction: str = "out", reduce: str = "mean",
                          empty: float = 0.0) -> torch.Tensor:
    """One value per token: ``reduce`` (``"mean"``, ``"median"``, ``"min"``,
    ``"max"`` or ``"sum"``) of the curvatures of its edges (``"out"``,
    ``"in"`` or ``"all"``), and ``empty`` for a token with no such edge.
    ``[..., S, S]`` in, ``[..., S]`` out."""
    if reduce not in _REDUCE:
        raise ValueError(f"reduce must be one of {sorted(_REDUCE)}, got {reduce!r}")
    F = torch.as_tensor(curvature, dtype=torch.float64)
    rows = _edges_of(F, direction)
    has = (~torch.isnan(rows)).any(dim=-1)
    out = _REDUCE[reduce](rows, dim=-1)
    return torch.where(has, out, torch.full_like(out, float(empty)))


__all__ = ["FORMAN_QUANTILES", "FORMAN_EPS", "FORMAN_WEIGHTS", "forman_curvature",
           "forman_quantiles", "forman_node_curvature"]
