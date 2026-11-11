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
Fast graph-signal diagnostics.

The hybrid metrics in :mod:`spectral_trust.spectral` project residual-stream
signals onto a full Laplacian eigenbasis, which costs O(S^3) per layer. The
functions here avoid, shrink or share that cost:

- :func:`row_dirichlet_energy` and :func:`row_value_spread` are row-anchored
  energies for a chosen set of query tokens. They need no eigendecomposition
  and cost O(|rows| * S * d).

Every function accepts batched inputs (a leading head dimension, e.g.
``[H, S, S]``) where noted, and runs in torch on whatever device its inputs
live on.
"""
from typing import Optional, Sequence, Union

import numpy as np
import torch

#: Accepted forms of a row selection: None (all rows), a slice, a sequence or
#: 1-D tensor of indices, or a boolean mask of length S.
Rows = Union[None, slice, Sequence[int], np.ndarray, torch.Tensor]


def _as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))


def _row_index(rows: Rows, size: int, device) -> Optional[torch.Tensor]:
    """Normalise a row selection to a 1-D LongTensor of indices (or None)."""
    if rows is None:
        return None
    if isinstance(rows, slice):
        return torch.arange(size, device=device)[rows]
    idx = torch.as_tensor(np.asarray(rows) if not isinstance(rows, torch.Tensor)
                          else rows, device=device)
    if idx.dtype == torch.bool:
        if idx.dim() != 1 or idx.shape[0] != size:
            raise ValueError(
                f"boolean row mask must have shape [{size}], got {tuple(idx.shape)}")
        return idx.nonzero(as_tuple=True)[0]
    idx = idx.reshape(-1).long()
    if idx.numel() and (int(idx.min()) < -size or int(idx.max()) >= size):
        raise IndexError(f"row index out of range for {size} tokens")
    return torch.remainder(idx, size)


def _reduce(per_row: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return per_row.sum(dim=-1)
    if reduction == "none":
        return per_row
    raise ValueError(f"reduction must be 'sum' or 'none', got {reduction!r}")


def row_dirichlet_energy(
    W,
    X,
    rows: Rows = None,
    reduction: str = "sum",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    r"""
    Row-anchored Dirichlet energy, with no eigendecomposition.

    .. math::
        E_R(W, X) = \sum_{i \in R} \sum_j W_{ij} \, \lVert x_i - x_j \rVert^2

    It measures how far each selected token's signal lies from the signals
    of the tokens it is connected to. With ``W`` a raw attention matrix (row
    i = query i), it is the attention-weighted distance between a query's
    state and the states it attends to.

    Relation to the library's Laplacian normalisations (``W`` symmetric,
    ``E`` taken over all rows, ``D = diag(W 1)``, ``L`` as built by
    :meth:`GraphConstructor.construct_laplacian`):

    - ``normalization="none"`` (``L = D - W``):
      ``tr(X^T L X) = E(W, X) / 2``.
    - ``normalization="sym"`` (``L = I - D^{-1/2} W D^{-1/2}``):
      ``tr(X^T L X) = E(W, D^{-1/2} X) / 2 + sum of ||x_i||^2 over isolated
      tokens (d_i = 0)``, whose rows of ``L`` are the identity.
    - ``normalization="rw"`` (``L = I - P``, ``P = D^{-1} W``):
      ``tr(X^T L X) = E(P, X) / 2 + sum_j (1 - c_j) ||x_j||^2 / 2`` for a graph
      without isolated tokens, where ``c_j = sum_i P_ij`` is a column sum of
      ``P``. The second term vanishes only when ``P`` is doubly stochastic.

    So the row-anchored energy of a subset R is the share of (twice) the
    combinatorial Dirichlet energy carried by the edges leaving R.

    Args:
        W: edge weights, ``[..., S, S]`` (e.g. ``[H, S, S]`` for one graph per
            head). Row i holds the weights of the edges leaving token i.
        X: signals, ``[..., S, d]``, broadcastable against ``W``'s leading
            dimensions (``[S, d]`` shared by every head, or ``[H, S, d]``).
        rows: tokens to anchor on (indices, a slice, or a boolean mask of
            length S). ``None`` takes every row.
        reduction: ``"sum"`` (default) sums over the selected rows;
            ``"none"`` returns one value per selected row, ``[..., |R|]``.
        dtype: computation dtype. The expansion
            ``d_i ||x_i||^2 - 2 x_i . (W X)_i + (W ||x||^2)_i`` is evaluated
            after centring ``X`` (the energy is translation invariant), and
            float64 keeps its cancellation error far below the float32 noise
            of the inputs. Only the selected rows of ``W`` are converted.

    Returns:
        Tensor of shape ``[...]`` (``reduction="sum"``) or ``[..., |R|]``.
    """
    W = _as_tensor(W)
    X = _as_tensor(X).to(W.device)
    S = W.shape[-1]
    if W.shape[-2] != S or X.shape[-2] != S:
        raise ValueError(
            f"W must be [..., S, S] and X [..., S, d]; got {tuple(W.shape)} "
            f"and {tuple(X.shape)}")
    idx = _row_index(rows, S, W.device)
    # Slice before converting: only |R| rows of W are needed.
    W_R = (W if idx is None else W.index_select(-2, idx)).to(dtype)
    Xc = X.to(dtype)
    Xc = Xc - Xc.mean(dim=-2, keepdim=True)
    sq = (Xc * Xc).sum(dim=-1)                                   # [..., S]
    X_R = Xc if idx is None else Xc.index_select(-2, idx)        # [..., R, d]
    sq_R = sq if idx is None else sq.index_select(-1, idx)       # [..., R]
    degree = W_R.sum(dim=-1)                                     # [..., R]
    cross = ((W_R @ Xc) * X_R).sum(dim=-1)                       # [..., R]
    neighbour_sq = (W_R @ sq.unsqueeze(-1)).squeeze(-1)          # [..., R]
    per_row = (degree * sq_R - 2.0 * cross + neighbour_sq).clamp(min=0.0)
    return _reduce(per_row, reduction)


def row_value_spread(
    A,
    V,
    rows: Rows = None,
    reduction: str = "sum",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    r"""
    Attention-weighted spread of the values around each query's output.

    .. math::
        \mathrm{spread}_i = \sum_j A_{ij} \, \lVert v_j - o_i \rVert^2,
        \qquad o_i = \sum_j A_{ij} v_j

    For a row-stochastic ``A`` (softmax attention) this is the variance of
    the value vectors under query i's attention distribution,
    ``sum_j A_ij ||v_j||^2 - ||o_i||^2``: zero when the query attends to a
    single token (or to identical values), large when it averages values
    that disagree. No eigendecomposition is involved.

    Args:
        A: attention weights, ``[..., S, S]`` (e.g. ``[H, S, S]``); row i is
            query i. Rows need not sum to one; the formula above is used as
            written.
        V: value vectors, ``[..., S, d]`` broadcastable against ``A``
            (``[H, S, d_head]`` for per-head values).
        rows: queries to evaluate (as in :func:`row_dirichlet_energy`).
        reduction: ``"sum"`` over the selected rows, or ``"none"``.
        dtype: computation dtype (default float64; only the selected rows of
            ``A`` are converted).

    Returns:
        Tensor of shape ``[...]`` (``reduction="sum"``) or ``[..., |R|]``.
    """
    A = _as_tensor(A)
    V = _as_tensor(V).to(A.device)
    S = A.shape[-1]
    if A.shape[-2] != S or V.shape[-2] != S:
        raise ValueError(
            f"A must be [..., S, S] and V [..., S, d]; got {tuple(A.shape)} "
            f"and {tuple(V.shape)}")
    idx = _row_index(rows, S, A.device)
    A_R = (A if idx is None else A.index_select(-2, idx)).to(dtype)
    Vd = V.to(dtype)
    out = A_R @ Vd                                               # [..., R, d]
    out_sq = (out * out).sum(dim=-1)                             # [..., R]
    mass = A_R.sum(dim=-1)                                       # [..., R]
    second_moment = (A_R @ (Vd * Vd).sum(dim=-1, keepdim=True)).squeeze(-1)
    per_row = (second_moment - 2.0 * out_sq + mass * out_sq).clamp(min=0.0)
    return _reduce(per_row, reduction)
