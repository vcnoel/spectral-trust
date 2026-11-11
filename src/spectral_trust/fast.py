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
- :func:`top_band_hfer` computes the exact full-basis HFER from the top
  ``ceil(cutoff_ratio * S)`` eigenpairs only.
- :func:`gaussian_projection` / :func:`jl_project` shrink the signal width d
  to k with a seeded Johnson-Lindenstrauss map (``GSPConfig.project_dim``).
- :func:`band_energies` and :func:`band_entropy` estimate the signal energy in
  B eigenvalue bands with Jackson-damped Chebyshev filters, from about
  ``order / 2`` products with L and no eigendecomposition.

The row energies accept batched inputs (a leading head dimension, e.g.
``[H, S, S]``) and run in torch on whatever device their inputs live on.
"""
from typing import Optional, Sequence, Union

import numpy as np
import scipy.linalg as sla
import torch
from scipy.sparse.linalg import eigsh

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


# ── Exact HFER from the top band ─────────────────────────────────────────────

#: Solvers accepted by :func:`top_band_hfer`.
HFER_METHODS = ("auto", "subset", "torch", "full", "eigsh")

#: ``method="auto"`` uses the full torch solve up to this size and the
#: LAPACK subset solve above it (crossover measured between 1024 and 2048).
AUTO_TORCH_MAX_SIZE = 1024


def num_high_modes(seq_len: int, cutoff_ratio: float) -> int:
    """
    Number of modes :meth:`SpectralAnalyzer.compute_hfer` counts as high
    frequency: ``seq_len - int((1 - cutoff_ratio) * seq_len)``.

    This equals ``ceil(cutoff_ratio * seq_len)`` except where floating-point
    rounding of ``(1 - cutoff_ratio) * seq_len`` lands just below an integer
    (e.g. ``cutoff_ratio=0.3``, ``seq_len=90``: 28 rather than 27). The rule
    is reproduced exactly so that the fast path selects the same modes.
    """
    return int(seq_len) - int((1 - cutoff_ratio) * seq_len)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        return x.numpy()
    return np.asarray(x)


def top_band_eigenvectors(L: np.ndarray, k: int, method: str = "auto") -> np.ndarray:
    """
    Orthonormal eigenvectors of the ``k`` largest eigenvalues of a symmetric
    matrix, ``[S, k]`` in ascending eigenvalue order.

    ``method`` (timings from two runs of ``benchmarks/fast_vs_dense.py
    --section solvers``, float32, two threads, ``k = 0.1 S``):

    - ``"subset"``: LAPACK ``syevr`` restricted to the top ``k`` indices
      (``scipy.linalg.eigh(subset_by_index=...)``). It still reduces ``L``
      to tridiagonal form, but computes and back-transforms only ``k``
      eigenvectors. Clearly fastest at S = 4096: 2.5-3.6 s against
      4.3-4.5 s for the full torch solve and 6.9-7.4 s for the full scipy
      solve.
    - ``"torch"``: the full ``torch.linalg.eigh`` solve (MKL), then the top
      ``k`` columns. Clearly fastest up to S = 512 (11 ms against 17 ms for
      the subset solve at S = 512). Between S = 1024 and 2048 the two are
      within about 20% and swap places from run to run.
    - ``"auto"``: ``"torch"`` up to :data:`AUTO_TORCH_MAX_SIZE`, then
      ``"subset"``.
    - ``"full"``: the full scipy solve, as in the dense analyzer path.
    - ``"eigsh"``: ARPACK Lanczos (``which="LA"``, machine-precision
      tolerance). Slower than the full scipy solve at ``k = 0.1 S``
      (1.6-2.8 s at S = 2048); only worth it for a handful of modes.

    ``torch.lobpcg`` was measured too (0.5 s at S = 128, 2.2 s at S = 512,
    50 to 1000 times slower than a full solve) and is not offered.
    """
    S = L.shape[0]
    if k <= 0:
        return np.zeros((S, 0), dtype=L.dtype)
    if method == "auto":
        method = "torch" if S <= AUTO_TORCH_MAX_SIZE else "subset"
    if method == "subset" or (method == "eigsh" and k >= S - 1):
        _, vecs = sla.eigh(L, subset_by_index=[S - k, S - 1], driver="evr")
        return vecs
    if method == "torch":
        # torch.linalg.eigh reads the lower triangle, as LAPACK does
        vecs = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(L)))[1]
        return vecs[:, S - k:].numpy()
    if method == "full":
        return sla.eigh(L)[1][:, S - k:]
    if method == "eigsh":
        # LAPACK reads only the lower triangle; give ARPACK the same matrix,
        # since a Laplacian built in float32 is asymmetric at ~1e-8.
        lower = np.tril(L)
        vals, vecs = eigsh(lower + np.tril(L, -1).T, k=k, which="LA", tol=0)
        return vecs[:, np.argsort(vals)]
    raise ValueError(f"method must be one of {HFER_METHODS}, got {method!r}")


def top_band_hfer(
    L,
    X,
    cutoff_ratio: float = 0.1,
    method: str = "auto",
) -> float:
    r"""
    Exact high-frequency energy ratio from the top band of the spectrum.

    For a symmetric Laplacian with full orthonormal eigenbasis ``U`` the
    spectral energies sum to the signal energy,
    ``sum_k ||u_k^T X||^2 = ||U^T X||_F^2 = ||X||_F^2``, so

    .. math::
        \mathrm{HFER} = \frac{\lVert U_{\text{high}}^T X \rVert_F^2}
                             {\lVert X \rVert_F^2}

    where ``U_high`` holds the eigenvectors of the top
    :func:`num_high_modes` eigenvalues. Only those eigenpairs are computed.
    The result equals :meth:`SpectralAnalyzer.compute_hfer` with
    ``eigen_solver="dense"`` (to ~1e-8 in float32, ~1e-15 in float64).

    How it differs from the default configuration: ``eigen_solver="sparse"``
    with ``num_eigenvalues=50`` keeps 25 bottom and 25 top modes once
    S > 50, and divides the energy of the top ``int(0.1 * 50) = 5`` retained
    modes by the energy of the 50 retained modes. That is a different
    quantity (a fixed number of modes, normalised over a truncated basis);
    this function returns the full-basis ratio at any S.

    If the boundary eigenvalue is degenerate (``lambda_{S-k} =
    lambda_{S-k-1}``), the split by index is basis-dependent for every
    solver, dense included; attention graphs with continuous weights do not
    hit this in practice.

    Args:
        L: symmetric Laplacian ``[S, S]`` (``normalization="sym"`` or
            ``"none"``), numpy or torch. The random-walk Laplacian is not
            symmetric, its eigenbasis is not orthonormal, and the identity
            above does not hold; it raises ``ValueError``.
        X: signals ``[S, d]``.
        cutoff_ratio: fraction of modes counted as high frequency
            (``GSPConfig.hfer_cutoff_ratio``, default 0.1).
        method: eigensolver, see :func:`top_band_eigenvectors`.

    Returns:
        HFER in [0, 1]; 0.0 when ``||X||_F^2 < 1e-8`` (the dense path's
        convention).
    """
    L = _to_numpy(L)
    X = _to_numpy(X)
    dtype = np.result_type(L.dtype, X.dtype, np.float32)
    L = L.astype(dtype, copy=False)
    X = X.astype(dtype, copy=False)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or X.shape[0] != L.shape[0]:
        raise ValueError(
            f"L must be [S, S] and X [S, d]; got {L.shape} and {X.shape}")
    if not np.allclose(L, L.T, atol=1e-6, rtol=0.0):
        raise ValueError(
            "top_band_hfer needs a symmetric Laplacian (normalization 'sym' "
            "or 'none'); the 'rw' eigenbasis is not orthonormal")
    total = float(np.sum(X * X))
    if total < 1e-8:
        return 0.0
    k = num_high_modes(L.shape[0], cutoff_ratio)
    U_high = top_band_eigenvectors(L, k, method=method)
    high = float(np.sum((U_high.T @ X) ** 2))
    return high / total


# ── Johnson-Lindenstrauss projection of the signals ──────────────────────────

def gaussian_projection(dim: int, k: int, seed: int = 0,
                        dtype=np.float32) -> np.ndarray:
    r"""
    Seeded Gaussian Johnson-Lindenstrauss matrix ``R`` of shape ``[dim, k]``
    with i.i.d. ``N(0, 1/k)`` entries, so ``E[R R^T] = I``.

    For any PSD matrix ``A`` (here ``X^T L X``, ``X^T X`` or
    ``X^T U_band U_band^T X``) the estimate ``tr(R^T A R)`` is unbiased for
    ``tr(A)`` with relative standard deviation
    ``sqrt(2 ||A||_F^2 / k) / tr(A) <= sqrt(2 / k)``. Energy and spectral
    masses are therefore unbiased; smoothness, HFER and entropy are ratios
    or functions of such estimates and carry an O(1/sqrt(k)) error.

    The matrix is drawn as ``standard_normal((k, dim)).T / sqrt(k)`` in
    float64, then cast: for one seed, the first ``k'`` columns of the
    ``k``-column matrix times ``sqrt(k / k')`` are exactly the ``k'``-column
    matrix.
    """
    if k <= 0 or dim <= 0:
        raise ValueError(f"dim and k must be positive, got dim={dim}, k={k}")
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((k, dim)).T / np.sqrt(k)
    return R.astype(dtype)


def jl_project(X, k: int, seed: int = 0):
    """
    Project signals ``X`` (``[..., dim]``, numpy or torch) to ``[..., k]``
    with :func:`gaussian_projection`. Returns the input type, on the input's
    device. If ``k >= dim`` the signals are returned unchanged.
    """
    dim = X.shape[-1]
    if k >= dim:
        return X
    if isinstance(X, torch.Tensor):
        dtype = X.dtype if X.is_floating_point() else torch.float32
        R = torch.from_numpy(gaussian_projection(dim, k, seed, np.float64))
        return X.to(dtype) @ R.to(device=X.device, dtype=dtype)
    X = np.asarray(X)
    dtype = np.result_type(X.dtype, np.float32)
    return X.astype(dtype, copy=False) @ gaussian_projection(dim, k, seed, dtype)


# ── Chebyshev band energies ──────────────────────────────────────────────────

def jackson_damping(order: int) -> np.ndarray:
    """
    Jackson kernel factors ``g_0..g_order`` (Weisse et al., Rev. Mod. Phys.
    78, 275, 2006). Damping the Chebyshev series of a band indicator with
    them turns it into a convolution with a positive kernel, so the filter
    stays in [0, 1] (no Gibbs overshoot) at the price of a smoothed edge of
    width about ``pi * lambda_max / (2 * order)``.
    """
    M = order + 1
    m = np.arange(M)
    a = np.pi / (M + 1)
    return ((M - m + 1) * np.cos(a * m) + np.sin(a * m) / np.tan(a)) / (M + 1)


def _resolve_edges(edges, lambda_max: float) -> np.ndarray:
    if np.isscalar(edges):
        B = int(edges)
        if B < 1:
            raise ValueError(f"number of bands must be >= 1, got {B}")
        return np.linspace(0.0, lambda_max, B + 1)
    e = np.asarray(edges, dtype=np.float64).reshape(-1)
    if e.size < 2 or np.any(np.diff(e) <= 0):
        raise ValueError("edges must be a strictly increasing sequence of >= 2 values")
    return e


#: The expansion interval starts this fraction of lambda_max below zero. A
#: Laplacian built in float32 has its null eigenvalues at about -1e-8, and
#: just outside [-1, 1] the Chebyshev polynomials grow as T_m(-1 - eps) ~
#: (-1)^m (1 + m^2 eps), which at order 100+ would leak the (often large)
#: near-zero mass. The first band is extended down to the interval's end.
CHEBYSHEV_LOWER_MARGIN = 1e-4


def _to_unit_interval(lam, lambda_max: float):
    """Affine map of [-margin * lambda_max, lambda_max] onto [-1, 1]."""
    lo = -CHEBYSHEV_LOWER_MARGIN * lambda_max
    return 2.0 * (np.asarray(lam, dtype=np.float64) - lo) / (lambda_max - lo) - 1.0


def band_chebyshev_coefficients(edges, order: int, lambda_max: float) -> np.ndarray:
    """
    Jackson-damped Chebyshev coefficients ``[B, order + 1]`` of the
    indicators of the bands ``[edges[b], edges[b+1])`` on ``[0, lambda_max]``,
    in the variable ``t`` that maps the expansion interval
    ``[-CHEBYSHEV_LOWER_MARGIN * lambda_max, lambda_max]`` onto ``[-1, 1]``.
    A first edge at or below zero is placed at ``t = -1``.
    """
    e = _resolve_edges(edges, lambda_max)
    t = np.clip(_to_unit_interval(e, lambda_max), -1.0, 1.0)
    if e[0] <= 0.0:
        t[0] = -1.0
    theta = np.arccos(t)                           # decreasing in lambda
    lo, hi = theta[:-1, None], theta[1:, None]     # [B, 1]
    m = np.arange(1, order + 1)[None, :]
    coeffs = np.empty((e.size - 1, order + 1))
    coeffs[:, 0] = (lo - hi)[:, 0] / np.pi
    coeffs[:, 1:] = 2.0 * (np.sin(m * lo) - np.sin(m * hi)) / (m * np.pi)
    return coeffs * jackson_damping(order)[None, :]


def band_filter_response(lam, edges, order: int, lambda_max: float) -> np.ndarray:
    """
    The damped polynomial filters evaluated at eigenvalues ``lam``:
    ``[B, len(lam)]``. :func:`band_energies` with ``method="chebyshev"``
    returns exactly ``response @ spectral_masses``, which is how its error
    against the true band indicators can be read off a dense spectrum.
    """
    t = _to_unit_interval(np.reshape(lam, -1), lambda_max)
    T = np.polynomial.chebyshev.chebvander(t, order).T     # valid for any t
    return band_chebyshev_coefficients(edges, order, lambda_max) @ T


def gershgorin_bound(L) -> float:
    """Upper bound on the largest eigenvalue of a symmetric ``L``:
    ``max_i sum_j |L_ij|``. O(S^2)."""
    L = _as_tensor(L)
    return float(L.abs().sum(dim=-1).max())


def chebyshev_moments(L, X, order: int, lambda_max: float) -> np.ndarray:
    r"""
    ``mu_m = tr(X^T T_m(L~) X)`` for ``m = 0..order``, where ``L~`` maps the
    expansion interval ``[-CHEBYSHEV_LOWER_MARGIN * lambda_max, lambda_max]``
    onto ``[-1, 1]``.

    Uses ``T_m^2 = (T_{2m} + I)/2`` and ``T_{m+1} T_m = (T_{2m+1} + T_1)/2``,
    so only ``Y_m = T_m(L~) X`` for ``m <= ceil(order / 2)`` are formed:
    ``ceil(order / 2)`` products with L, each O(S^2 d).
    """
    L = _as_tensor(L)
    Y_prev = _as_tensor(X).to(device=L.device, dtype=L.dtype)
    lo = -CHEBYSHEV_LOWER_MARGIN * float(lambda_max)
    scale = 2.0 / (float(lambda_max) - lo)
    shift = -scale * lo - 1.0

    def apply(V):
        return scale * (L @ V) + shift * V

    def dot(A, B):
        return float((A * B).sum(dtype=torch.float64))

    mu = np.zeros(order + 1)
    mu[0] = dot(Y_prev, Y_prev)
    if order == 0:
        return mu
    Y_cur = apply(Y_prev)
    mu[1] = dot(Y_prev, Y_cur)
    n = (order + 1) // 2
    for m in range(1, n + 1):
        if 2 * m <= order:
            mu[2 * m] = 2.0 * dot(Y_cur, Y_cur) - mu[0]
        if m >= 2 and 2 * m - 1 <= order:
            mu[2 * m - 1] = 2.0 * dot(Y_cur, Y_prev) - mu[1]
        if m < n:
            Y_prev, Y_cur = Y_cur, 2.0 * apply(Y_cur) - Y_prev
    return mu


def band_energies(
    L,
    X,
    edges=8,
    order: int = 64,
    lambda_max: Optional[float] = None,
    method: str = "chebyshev",
    project_dim: Optional[int] = None,
    seed: int = 0,
):
    r"""
    Signal energy in eigenvalue bands of a symmetric Laplacian.

    The exact band energy is ``E_b = sum_{k: lambda_k in band b}
    ||u_k^T X||^2 = tr(X^T 1_b(L) X)``. ``method="chebyshev"`` replaces the
    indicator ``1_b`` by its Jackson-damped Chebyshev expansion of degree
    ``order`` on ``[0, lambda_max]``:

    .. math::
        \hat E_b = \sum_{m=0}^{N} g_m c_m^{(b)} \,
                   \mathrm{tr}\big(X^T T_m(\tilde L) X\big)

    All bands share the moments (:func:`chebyshev_moments`), so the cost is
    ``ceil(order / 2)`` products ``L @ Y`` (O(order S^2 d)) against O(S^3)
    for a dense eigendecomposition. It wins when ``order * d`` is small next
    to S; pass ``project_dim`` to shrink d first (energies stay unbiased,
    see :func:`gaussian_projection`).

    Properties of the estimate: each ``\hat E_b >= 0``; when the bands
    partition ``[0, lambda_max]`` they sum to ``||X||_F^2`` exactly; and
    ``\hat E_b = sum_k p_b(lambda_k) ||u_k^T X||^2`` with ``p_b`` given by
    :func:`band_filter_response`. Modes within about
    ``pi * lambda_max / (2 * order)`` of an interior edge are shared between
    the two neighbouring bands, which is the whole approximation error.

    Args:
        L: symmetric Laplacian ``[S, S]`` (numpy or torch, any device).
        X: signals ``[S, d]``.
        edges: number of uniform bands on ``[0, lambda_max]`` (default 8), or
            explicit increasing edges in eigenvalue units. Band b is
            ``[edges[b], edges[b+1])``; the last band includes its right edge.
        order: Chebyshev degree N (``method="chebyshev"``).
        lambda_max: an upper bound on the spectrum. Pass ``2.0`` for
            ``normalization="sym"`` (its spectrum lies in [0, 2]) so that
            uniform bands mean the same thing for every input. Default: the
            Gershgorin bound of ``L``, which depends on the input.
        method: ``"chebyshev"`` or ``"dense"`` (exact, ``torch.linalg.eigh``).
        project_dim, seed: optional Johnson-Lindenstrauss projection of X.

    Returns:
        ``(energies, edges)``: numpy arrays ``[B]`` and ``[B + 1]``.
    """
    L = _as_tensor(L)
    if not L.is_floating_point() or L.dtype in (torch.float16, torch.bfloat16):
        L = L.to(torch.float32)
    X = _as_tensor(X).to(device=L.device)
    if L.dim() != 2 or L.shape[0] != L.shape[1] or X.shape[0] != L.shape[0]:
        raise ValueError(
            f"L must be [S, S] and X [S, d]; got {tuple(L.shape)} and {tuple(X.shape)}")
    if project_dim is not None:
        X = jl_project(X, project_dim, seed=seed)
    X = X.to(L.dtype)
    lam_max = float(lambda_max) if lambda_max is not None else gershgorin_bound(L)
    e = _resolve_edges(edges, lam_max)
    if method == "chebyshev":
        mu = chebyshev_moments(L, X, order, lam_max)
        energies = band_chebyshev_coefficients(e, order, lam_max) @ mu
        return np.clip(energies, 0.0, None), e
    if method == "dense":
        Ls = 0.5 * (L + L.T)
        vals, vecs = torch.linalg.eigh(Ls)
        masses = ((vecs.T @ X) ** 2).sum(dim=-1, dtype=torch.float64).cpu().numpy()
        lam = np.maximum(vals.double().cpu().numpy(), 0.0)
        band = np.searchsorted(e, lam, side="right") - 1
        band[lam == e[-1]] = e.size - 2
        keep = (band >= 0) & (band < e.size - 1)
        energies = np.bincount(band[keep], weights=masses[keep], minlength=e.size - 1)
        return energies, e
    raise ValueError(f"method must be 'chebyshev' or 'dense', got {method!r}")


def band_entropy(energies, normalize: bool = False) -> float:
    r"""
    Entropy of the band-energy distribution, ``-sum_b p_b log p_b`` with
    ``p_b = E_b / sum E``, natural log (the convention of
    :meth:`SpectralAnalyzer.compute_spectral_entropy`). ``normalize=True``
    divides by ``log B``.

    This is not the hybrid ``spectral_entropy``. That one is the entropy of
    the per-mode distribution over all S modes; this one coarse-grains the
    modes into B eigenvalue bands. With exact band energies the grouping
    inequality gives ``band_entropy <= spectral_entropy``, with equality only
    when each band holds the mass of a single mode; the two are on different
    scales (at most ``log B`` against ``log S``) and should not be compared
    or mixed across analyses. With Chebyshev energies the edge smoothing
    adds a further, small difference.
    """
    E = np.clip(np.asarray(energies, dtype=np.float64), 0.0, None)
    total = E.sum()
    if E.size < 2 or total < 1e-8:
        return 0.0
    p = np.maximum(E / total, 1e-12)
    H = float(-np.sum(p * np.log(p)))
    return H / np.log(E.size) if normalize else H
