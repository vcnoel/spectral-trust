# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""
Pseudospectra and other measures of non-normality (Trefethen and Embree,
"Spectra and Pseudospectra", Princeton 2005).

Causal attention is lower-triangular, hence non-normal: its eigenvalues are
its diagonal and govern only the limit of its powers and products, while
finite depth is governed by resolvent quantities (Li 2026, arXiv 2607.09279
and 2607.06621). For a matrix ``A`` and ``ε > 0``:

- the ε-pseudospectrum ``Λ_ε(A) = {z : σ_min(zI − A) < ε}``
  (:func:`pseudospectrum`, :func:`sigma_min`);
- its radius ``ρ_ε`` and abscissa ``α_ε`` (:func:`pseudospectral_radius`,
  :func:`pseudospectral_abscissa`);
- the Kreiss constant ``K(A) = sup_{|z|>1} (|z| − 1) ‖(zI − A)⁻¹‖ =
  sup_ε (ρ_ε − 1)/ε``, which brackets the transient,
  ``K <= sup_k ‖A^k‖ <= e n K`` (:func:`kreiss_constant`);
- Henrici's departure from normality (:func:`departure_from_normality`);
- the numerical range and radius (:func:`numerical_range`);
- all of them at once, for one matrix or for the products of a stack of
  layers, the depth propagator (:func:`nonnormality`,
  :func:`depth_propagator`).

``σ_min(zI − A)`` is computed on a triangular form: the matrix itself when
it is triangular (causal attention), otherwise its complex Schur form
``A = Q T Qᴴ`` (``σ_min`` is unitarily invariant). Each point then costs
O(n²) per inverse-iteration step, by two triangular solves, against O(n³)
for a dense SVD. All computations are in complex128.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import numpy as np
import scipy.linalg
import torch

__all__ = ["sigma_min", "pseudospectrum", "Pseudospectrum", "pseudospectral_radius",
           "pseudospectral_abscissa", "kreiss_constant", "departure_from_normality",
           "numerical_range", "NumericalRange", "depth_propagator", "nonnormality",
           "NONNORMALITY_NAMES"]

_CHUNK_BYTES = 128 * 2 ** 20


def _matrix(A) -> np.ndarray:
    a = A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square matrix [n, n], got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("the matrix has non-finite entries")
    return a.astype(np.complex128)


class _Triangular:
    """``A`` as a triangular matrix ``T`` with the same singular values of
    ``zI − T`` as of ``zI − A``: ``A`` itself when triangular, else its
    complex Schur factor."""

    def __init__(self, A, method: str = "auto"):
        a = _matrix(A)
        self.n = a.shape[0]
        self.norm2 = float(np.linalg.norm(a, 2)) if self.n else 0.0
        if np.all(np.triu(a, 1) == 0):
            self.T, self.upper, self.schur = a, False, False
        elif np.all(np.tril(a, -1) == 0):
            self.T, self.upper, self.schur = a, True, False
        else:
            self.T, _ = scipy.linalg.schur(a, output="complex")
            self.upper, self.schur = True, True
        self.eigenvalues = np.diag(self.T).copy()
        if method not in ("auto", "svd"):
            raise ValueError(f"method must be 'auto' or 'svd', got {method!r}")
        self.method = method

    def _start(self, p: int, k: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(self.n, k, generator=g, dtype=torch.float64).to(torch.complex128)
        x[:, 0] += 1.0
        return torch.linalg.qr(x.expand(p, self.n, k))[0]

    def sigma_min(self, z: np.ndarray, x0: Optional[torch.Tensor] = None,
                  tol: float = 1e-11, max_iter: int = 300):
        """``(σ_min(zI − T) for each z, last iterate)``; ``z`` is 1-D.

        Block inverse iteration on ``(MᴴM)⁻¹``, ``M = zI − T``, with a block
        of up to three vectors and a Rayleigh-Ritz estimate: two triangular
        solves per step, O(n²) each. The estimate approaches ``σ_min`` from
        above; it stops when its relative change is below ``tol``."""
        z = np.asarray(z, dtype=np.complex128).ravel()
        if self.method == "svd" or self.n == 0:
            I = np.eye(self.n)
            sig = np.array([np.linalg.svd(zk * I - self.T, compute_uv=False)[-1]
                            if self.n else 0.0 for zk in z])
            return sig, None
        T = torch.from_numpy(self.T)
        I = torch.eye(self.n, dtype=torch.complex128)
        k = min(3, self.n)
        step = max(1, _CHUNK_BYTES // (16 * self.n * self.n))
        sig = np.empty(len(z))
        xs = torch.empty(len(z), self.n, dtype=torch.complex128)
        for s in range(0, len(z), step):
            zc = torch.from_numpy(z[s:s + step])
            exact = (zc[:, None] == torch.from_numpy(self.eigenvalues)[None, :]).any(-1)
            M = torch.where(exact, zc + 1.0, zc)[:, None, None] * I - T   # singular: σ = 0
            MH = M.conj().transpose(-1, -2)
            X = self._start(len(zc), k)
            if x0 is not None:
                X = X.clone()
                X[..., 0] = x0[s:s + step]
                X = torch.linalg.qr(X)[0]
            est = torch.full((len(zc),), np.inf, dtype=torch.float64)
            for _ in range(max_iter):
                W = torch.linalg.solve_triangular(MH, X, upper=not self.upper)  # M⁻ᴴ X
                lost = ~torch.isfinite(W).all(-1).all(-1)                   # σ below 1e-300
                W = torch.where(lost[:, None, None], X, W)
                # Ritz values of (MᴴM)⁻¹ on span(X): squared singular values of W
                _, S, Vh = torch.linalg.svd(W, full_matrices=False)
                new = torch.where(lost, torch.zeros_like(S[..., 0]), 1.0 / S[..., 0])
                scale = S[..., :1, None].clamp_min(1e-300)
                Y = torch.linalg.solve_triangular(M, (W / scale) @ Vh.conj().transpose(-1, -2),
                                                  upper=self.upper)
                good = torch.isfinite(Y).all(-1).all(-1) & torch.isfinite(S).all(-1)
                if bool(good.all()):
                    X = torch.linalg.qr(Y)[0]                         # top Ritz vector first
                done = (est - new).abs() <= tol * new.abs()
                est = new
                if bool(done.all()) or not bool(good.all()):
                    break
            sig[s:s + step] = torch.where(exact, torch.zeros_like(est), est).numpy()
            xs[s:s + step] = X[..., 0]
        return sig, xs


def _grid(grid, tri: _Triangular, pad: float, n_points: int) -> np.ndarray:
    if grid is None:
        R = tri.norm2 + pad
        axis = np.linspace(-R, R, n_points)
        return axis[None, :] + 1j * axis[:, None]
    if isinstance(grid, tuple) and len(grid) == 2:
        re, im = (np.asarray(g, dtype=np.float64).ravel() for g in grid)
        return re[None, :] + 1j * im[:, None]
    return np.asarray(grid, dtype=np.complex128)


def sigma_min(A, z, *, method: str = "auto", tol: float = 1e-11, max_iter: int = 300):
    """``σ_min(zI − A)`` at the points ``z`` (any shape), the distance from
    ``A`` to the nearest matrix with ``z`` as an eigenvalue.

    Args:
        A: ``[n, n]`` matrix (tensor or array).
        z: complex points.
        method: ``"auto"`` (inverse iteration with triangular solves, O(n²)
            per step and point) or ``"svd"`` (a dense SVD per point, O(n³)).
        tol, max_iter: stopping rule of the inverse iteration (relative
            change of the estimate). The estimate approaches ``σ_min`` from
            above.
    """
    tri = _Triangular(A, method)
    z = np.asarray(z, dtype=np.complex128)
    sig, _ = tri.sigma_min(z.ravel(), tol=tol, max_iter=max_iter)
    return float(sig[0]) if z.ndim == 0 else sig.reshape(z.shape)


@dataclass
class Pseudospectrum:
    """``σ_min(zI − A)`` on a grid of points ``z``.

    Attributes:
        z: complex grid (``[len(im), len(re)]`` for a ``(re, im)`` grid).
        sigma_min: same shape as ``z``.
        eigenvalues: eigenvalues of ``A``.
        eps: the levels asked for, or ``()``.
    """
    z: np.ndarray
    sigma_min: np.ndarray
    eigenvalues: np.ndarray
    eps: tuple = ()

    def inside(self, eps: float) -> np.ndarray:
        """Boolean mask of the grid points in the ε-pseudospectrum."""
        return self.sigma_min < eps


def pseudospectrum(A, grid=None, eps=None, *, n_points: int = 100, method: str = "auto",
                   tol: float = 1e-11, max_iter: int = 300) -> Pseudospectrum:
    """``σ_min(zI − A)`` on a complex grid; its level sets are the boundaries
    of the ε-pseudospectra ``Λ_ε(A) = {z : σ_min(zI − A) < ε}``.

    Args:
        A: ``[n, n]`` matrix. A triangular matrix (causal attention) is used
            as it is; any other is reduced to its complex Schur form first.
        grid: complex points of any shape, or a pair ``(re, im)`` of 1-D
            arrays (a mesh with ``re`` along columns), or None for a square
            of ``n_points`` × ``n_points`` covering ``|z| <= ‖A‖₂ + max(eps)``.
        eps: optional level or levels (kept in the result; see
            :meth:`Pseudospectrum.inside`).
        method, tol, max_iter: as in :func:`sigma_min`.

    Values below about ``1e-15 ‖A‖`` are not resolved by float64 and are
    reported as they come (the triangular solves resolve them further than
    a dense SVD does).
    """
    levels = () if eps is None else tuple(np.atleast_1d(np.asarray(eps, dtype=float)))
    tri = _Triangular(A, method)
    Z = _grid(grid, tri, max(levels) if levels else 0.0, n_points)
    sig, _ = tri.sigma_min(Z.ravel(), tol=tol, max_iter=max_iter)
    return Pseudospectrum(Z, sig.reshape(Z.shape), tri.eigenvalues, levels)


def _outermost(tri: _Triangular, eps: float, base: np.ndarray, direction: np.ndarray,
               t_hi: float, t_floor: float, tol: float, n_steps: int = 64) -> np.ndarray:
    """For each ray ``z(t) = base + t·direction`` (``|direction| = 1``), the
    largest ``t <= t_hi`` with ``σ_min(z(t)) <= ε``, or -inf when there is
    none above ``t_floor``. ``σ_min`` is 1-Lipschitz in ``z``, so a step of
    ``σ_min − ε`` never skips a crossing; steps are at least
    ``(t_hi − t_floor) / n_steps``, so a crossing region narrower than that
    can be missed. The bracketing step is refined by bisection to ``tol``."""
    m = len(base)
    h = max((t_hi - t_floor) / n_steps, tol)
    t = np.full(m, float(t_hi))
    prev, lo = t.copy(), np.full(m, -np.inf)
    found, alive = np.zeros(m, bool), np.ones(m, bool)
    x = torch.zeros(m, tri.n, dtype=torch.complex128)
    have = np.zeros(m, bool)

    def evaluate(idx, tt):
        warm = x[idx] if have[idx].all() else None
        sig, xs = tri.sigma_min(base[idx] + tt * direction[idx], x0=warm)
        if xs is not None:
            x[idx] = xs
            have[idx] = True
        return sig
    for _ in range(n_steps + 2):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        f = evaluate(idx, t[idx]) - eps
        hit = f <= 0
        found[idx[hit]], lo[idx[hit]], alive[idx[hit]] = True, t[idx[hit]], False
        keep = idx[~hit]
        alive[keep[t[keep] <= t_floor]] = False          # the floor itself was outside
        keep = keep[t[keep] > t_floor]
        prev[keep] = t[keep]
        t[keep] = np.maximum(t[keep] - np.maximum(f[~hit][np.isin(idx[~hit], keep)], h), t_floor)
    idx = np.flatnonzero(found & (lo < prev))
    a, b = lo[idx], prev[idx]
    for _ in range(80):
        if idx.size == 0 or np.max(b - a) <= tol:
            break
        mid = 0.5 * (a + b)
        inside = evaluate(idx, mid) <= eps
        a, b = np.where(inside, mid, a), np.where(inside, b, mid)
    lo[idx] = a
    return np.where(found, lo, -np.inf)


def _outermost_refined(tri, eps, make_rays, params, width, t_hi, t_floor, tol, refine):
    """Max over rays of :func:`_outermost`, with ``refine`` rounds of finer
    rays around the best one (``params`` -> ``(base, direction)``)."""
    best_t, best_p = -np.inf, None
    for r in range(refine + 1):
        base, direction = make_rays(params)
        t = _outermost(tri, eps, base, direction, t_hi, t_floor, tol)
        k = int(np.argmax(t))
        if t[k] > best_t:
            best_t, best_p = t[k], params[k]
        if best_p is None:
            break
        width = width / 10.0
        params = best_p + np.linspace(-10 * width, 10 * width, 21)
    return max(best_t, t_floor)


def pseudospectral_radius(A, eps: float, *, n_angles: int = 180, refine: int = 2,
                          tol: Optional[float] = None, method: str = "auto") -> float:
    """ε-pseudospectral radius ``ρ_ε(A) = max{|z| : σ_min(zI − A) <= ε}``.

    Each of ``n_angles`` rays from the origin is marched inward from
    ``‖A‖₂ + ε`` to its outermost crossing (never skipping one, since
    ``σ_min`` is 1-Lipschitz), then the best direction is refined
    ``refine`` times. The result is at least ``ρ(A) + ε``, which always
    holds, and approaches ``ρ_ε`` from below as the rays refine.
    """
    tri = _Triangular(A, method)
    rho = float(np.abs(tri.eigenvalues).max()) if tri.n else 0.0
    t_hi = tri.norm2 + eps
    tol = tol if tol is not None else 1e-12 * max(1.0, t_hi)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    def rays(theta):
        return np.zeros(len(theta), complex), np.exp(1j * np.asarray(theta))
    return float(_outermost_refined(tri, eps, rays, angles, 2 * np.pi / n_angles, t_hi,
                                    rho + eps, tol, refine))


def pseudospectral_abscissa(A, eps: float, *, n_lines: int = 181, refine: int = 2,
                            tol: Optional[float] = None, method: str = "auto") -> float:
    """ε-pseudospectral abscissa ``α_ε(A) = max{Re z : σ_min(zI − A) <= ε}``,
    by the same inward march along ``n_lines`` horizontal lines. At least
    ``max Re λ + ε``."""
    tri = _Triangular(A, method)
    alpha = float(tri.eigenvalues.real.max()) if tri.n else 0.0
    t_hi = tri.norm2 + eps
    tol = tol if tol is not None else 1e-12 * max(1.0, t_hi)
    ys = np.linspace(-t_hi, t_hi, n_lines)

    def lines(y):
        y = np.asarray(y, dtype=float)
        return 1j * y, np.ones(len(y), complex)
    return float(_outermost_refined(tri, eps, lines, ys, 2 * t_hi / (n_lines - 1), t_hi,
                                    alpha + eps, tol, refine))


def kreiss_constant(A, eps_grid: Optional[Sequence[float]] = None, *, how: str = "radii",
                    n_angles: int = 36, refine: int = 1, method: str = "auto") -> float:
    """Kreiss constant ``K(A) = sup_{ε>0} (ρ_ε(A) − 1)/ε
    = sup_{|z|>1} (|z| − 1) ‖(zI − A)⁻¹‖₂``. It brackets the transient
    growth of the powers: ``K <= sup_k ‖A^k‖₂ <= e·n·K``.

    Args:
        A: ``[n, n]`` matrix.
        eps_grid: levels for ``how="radii"`` (default ``logspace(-8, 8, 9)``).
        how: ``"radii"`` (default) estimates it from
            :func:`pseudospectral_radius` on ``eps_grid``, refined by a
            golden-section search in ``log ε``; ``"resolvent"`` maximises
            ``(|z| − 1)/σ_min(zI − A)`` over ``|z| = 1 + 10^s``,
            ``s ∈ [-10, 8]``, on a polar grid refined around its best point.
            Both are lower bounds that converge as their grids refine.
        n_angles, refine: rays and refinement rounds.

    Returns:
        ``inf`` when ``ρ(A) > 1``. For a causal row-stochastic matrix ``K``
        is ``√n`` (Li 2026, Proposition 1), approached as ``ε → 0``.
    """
    if how not in ("radii", "resolvent"):
        raise ValueError(f"how must be 'radii' or 'resolvent', got {how!r}")
    tri = _Triangular(A, method)
    if tri.n == 0:
        return 0.0
    if np.abs(tri.eigenvalues).max() > 1 + 1e-12:
        return float("inf")
    if how == "resolvent":
        return _kreiss_resolvent(tri, n_angles, refine)
    grid = np.logspace(-8, 8, 9) if eps_grid is None else np.asarray(eps_grid, float)

    def ratio(e):
        r = pseudospectral_radius(A, e, n_angles=n_angles, refine=refine, method=method,
                                  tol=max(1e-15, 1e-9 * e) * max(1.0, tri.norm2))
        return (r - 1.0) / e
    vals = np.array([ratio(e) for e in grid])
    k = int(np.argmax(vals))
    a, b = np.log(grid[max(k - 1, 0)]), np.log(grid[min(k + 1, len(grid) - 1)])
    g, best = (np.sqrt(5) - 1) / 2, vals[k]
    for _ in range(10 if b > a else 0):
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = ratio(np.exp(c)), ratio(np.exp(d))
        best = max(best, fc, fd)
        a, b = (a, d) if fc >= fd else (c, b)
    return float(best)


def _kreiss_resolvent(tri: _Triangular, n_angles: int, refine: int) -> float:
    s = np.linspace(-10, 8, 73)
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    ds, dth = s[1] - s[0], th[1] - th[0]
    best = 0.0
    for _ in range(refine + 2):
        S, TH = np.meshgrid(s, th)
        z = (1 + 10.0 ** S) * np.exp(1j * TH)
        sig, _ = tri.sigma_min(z.ravel())
        with np.errstate(divide="ignore"):
            vals = np.where(sig > 0, 10.0 ** S.ravel() / sig, np.inf)
        k = int(np.argmax(vals))
        best = max(best, float(vals[k]))
        s0, t0 = S.ravel()[k], TH.ravel()[k]
        s = np.linspace(s0 - ds, s0 + ds, 11)
        th = np.linspace(t0 - dth, t0 + dth, 11)
        ds, dth = ds / 5, dth / 5
    return best


def departure_from_normality(A, relative: bool = False):
    """Henrici's departure from normality ``dep_F(A) = ‖N‖_F``, where
    ``A = Q (D + N) Qᴴ`` is a Schur decomposition with ``N`` strictly
    triangular; equivalently ``sqrt(‖A‖_F² − Σ|λ_i|²)``, 0 iff ``A`` is
    normal. For a triangular matrix it is the norm of its off-diagonal part,
    with no factorisation.

    Args:
        A: ``[..., n, n]``.
        relative: divide by ``‖A‖_F`` (a number in ``[0, 1]``).

    Returns:
        ``[...]`` (a float for one matrix).
    """
    a = A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)
    lead = a.shape[:-2]
    out = []
    for m in a.reshape(-1, *a.shape[-2:]):
        tri = _Triangular(m)
        off = np.tril(tri.T, -1) if not tri.upper else np.triu(tri.T, 1)
        d = float(np.linalg.norm(off))
        if relative:
            nf = float(np.linalg.norm(m))
            d = d / nf if nf > 0 else 0.0
        out.append(d)
    return out[0] if not lead else np.asarray(out).reshape(lead)


@dataclass
class NumericalRange:
    """Field of values ``W(A) = {xᴴAx : ‖x‖ = 1}`` (a convex set).

    Attributes:
        boundary: points of its boundary, one per angle.
        radius: numerical radius ``w(A) = max |z|`` over ``W(A)``, with
            ``‖A‖₂/2 <= w(A) <= ‖A‖₂``.
        abscissa: numerical abscissa ``max Re z`` over ``W(A)``, the
            initial growth rate ``d/dt ‖e^{tA}‖`` at ``t = 0``.
    """
    boundary: np.ndarray
    radius: float
    abscissa: float


def numerical_range(A, n_angles: int = 180, refine: int = 2) -> NumericalRange:
    """Numerical range of ``A`` from its rotated Hermitian parts
    ``H_θ = (e^{iθ}A + e^{-iθ}Aᴴ)/2`` (Johnson's algorithm): the top
    eigenvector ``x_θ`` of ``H_θ`` gives the boundary point ``x_θᴴ A x_θ``,
    and ``w(A) = max_θ λ_max(H_θ)``, refined ``refine`` times around the best
    angle."""
    a = _matrix(A)
    AH = a.conj().T

    def top(theta):
        H = 0.5 * (np.exp(1j * theta) * a + np.exp(-1j * theta) * AH)
        w, v = np.linalg.eigh(H)
        return w[-1], v[:, -1]
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    pts, lam = [], []
    for t in th:
        w, v = top(t)
        lam.append(w)
        pts.append(np.vdot(v, a @ v))
    best = int(np.argmax(lam))
    radius, t0, d = lam[best], th[best], 2 * np.pi / n_angles
    for _ in range(refine):
        for t in np.linspace(t0 - d, t0 + d, 21):
            w, _ = top(t)
            if w > radius:
                radius, t0 = w, t
        d /= 10
    return NumericalRange(np.asarray(pts), float(radius), float(top(0.0)[0]))


def _stack(layers) -> np.ndarray:
    if torch.is_tensor(layers):
        a = layers.detach().cpu().numpy()
    elif isinstance(layers, np.ndarray):
        a = layers
    else:
        a = np.stack([x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
                      for x in layers])
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"expected layers [L, n, n], got shape {a.shape}")
    return a.astype(np.complex128 if np.iscomplexobj(a) else np.float64)


def _deflate(a: np.ndarray, sink: int) -> np.ndarray:
    P = np.zeros_like(a[0])
    P[:, sink] = 1.0
    return a - P


def depth_propagator(layers, deflate: bool = False, sink: int = 0) -> np.ndarray:
    """Products of a stack of layer operators over depth:
    ``Π_L = A_L ⋯ A_2 A_1`` for ``L = 1..len(layers)``, with layer 1 applied
    first (for attention, one head-averaged map per layer).

    With ``deflate=True`` each factor is ``A_l − P``, ``P = 1 e_sinkᵀ``. For
    causal row-stochastic maps, which all share that Perron projector,
    ``Π_L(A) − P = Π_L(A − P)`` (Li 2026, Proposition 2): the deflated
    products carry the deviation from the stationary (all-to-sink) state.

    Returns:
        ``[L, n, n]``.
    """
    a = _stack(layers)
    if deflate:
        a = _deflate(a, sink)
    out = np.empty_like(a)
    acc = np.eye(a.shape[1], dtype=a.dtype)
    for l in range(len(a)):
        acc = a[l] @ acc
        out[l] = acc
    return out


#: Names returned by :func:`nonnormality` for one matrix.
NONNORMALITY_NAMES = ("norm", "spectral_radius", "spectral_abscissa", "pseudospectral_radius",
                      "pseudospectral_abscissa", "kreiss_constant", "henrici",
                      "henrici_relative", "numerical_radius", "numerical_abscissa")


def nonnormality(A, eps: float = 1e-2, *, depth: bool = False, deflate: bool = False,
                 sink: int = 0, kreiss: bool = True, **kwargs) -> Dict[str, object]:
    """The non-normality summaries of one matrix, or of every depth product.

    Args:
        A: ``[n, n]``; with ``depth=True`` a stack ``[L, n, n]`` whose
            products (:func:`depth_propagator`) are summarised.
        eps: level of the pseudospectral radius and abscissa.
        depth, deflate, sink: as in :func:`depth_propagator`.
        kreiss: include the Kreiss constant (the most expensive summary).
        kwargs: passed to :func:`kreiss_constant` (``how``, ``n_angles``).

    Returns:
        name -> value (``[L]`` arrays with ``depth=True``): ``norm`` (2-norm),
        ``spectral_radius``, ``spectral_abscissa``, ``pseudospectral_radius``
        and ``pseudospectral_abscissa`` at ``eps``, ``kreiss_constant``,
        ``henrici`` and ``henrici_relative``, ``numerical_radius`` and
        ``numerical_abscissa``. With ``depth=True`` also Li's references for
        the propagator: ``eigen_heuristic`` ``Π_l ρ(A_l)`` and ``norm_bound``
        ``Π_l ‖A_l‖₂`` (of the deflated factors with ``deflate=True``), to
        compare with ``norm``.
    """
    def one(m):
        tri = _Triangular(m)
        lam = tri.eigenvalues
        r = {"norm": tri.norm2,
             "spectral_radius": float(np.abs(lam).max()),
             "spectral_abscissa": float(lam.real.max()),
             "pseudospectral_radius": pseudospectral_radius(m, eps),
             "pseudospectral_abscissa": pseudospectral_abscissa(m, eps),
             "henrici": departure_from_normality(m),
             "henrici_relative": departure_from_normality(m, relative=True)}
        if kreiss:
            r["kreiss_constant"] = kreiss_constant(m, **kwargs)
        nr = numerical_range(m)
        r["numerical_radius"], r["numerical_abscissa"] = nr.radius, nr.abscissa
        return r
    if not depth:
        return one(_matrix(A))
    a = _stack(A)
    factors = _deflate(a, sink) if deflate else a
    rows = [one(p) for p in depth_propagator(a, deflate=deflate, sink=sink)]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    rho = [float(np.abs(_Triangular(f).eigenvalues).max()) for f in factors]
    out["eigen_heuristic"] = np.cumprod(rho)
    out["norm_bound"] = np.cumprod([float(np.linalg.norm(f, 2)) for f in factors])
    return out
