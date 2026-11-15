# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Measure-aware spectra of an attention map, seen as an operator between two
weighted function spaces on the tokens.

An attention matrix ``A`` (row i the query, row-stochastic) maps a signal
``f`` on the keys to ``A f`` on the queries. Give the keys a measure ``μ``
and the queries a measure ``π``, with inner products
``⟨f, g⟩_μ = fᵀ Ω_μ g`` and ``Ω_μ = diag(μ)``. As a map
``L²(μ) → L²(π)``, ``A`` has the adjoint ``Ω_μ⁻¹ Aᵀ Ω_π`` and the operator

    D = Ω_μ⁻¹ Aᵀ Ω_π A                                     (on L²(μ))

is self-adjoint and positive semidefinite in ``L²(μ)``. Its eigenvalues are
the squared gains ``‖A f‖²_π / ‖f‖²_μ`` of the eigenfunctions. ``D`` is
similar to the symmetric matrix

    S = Ω_μ^{-1/2} Aᵀ Ω_π A Ω_μ^{-1/2},

which is what this module diagonalises. Two choices of measures:

``"pullback"`` (the default, the intrinsic geometry)
    ``π = 1/n`` (uniform over the queries) and ``μ = Aᵀ π``, the average
    attention row. Then ``D 1 = 1`` and every gain is at most 1, so
    ``λ₁ = 1`` with the constant eigenfunction, and ``√λ₂`` is the largest
    gain of a signal with ``μ``-mean zero (:func:`operator_metrics`,
    ``contraction``).
``"counting"`` (the Euclidean geometry)
    ``μ = π = 1``, so ``D = S = Aᵀ A``, whose eigenvalues are the squared
    singular values of ``A``. Sink columns dominate this spectrum: for the
    pure sink ``A = 1 e₁ᵀ``, ``λ₁ = n``.

A pair ``(mu, pi)`` sets both measures directly.

The spectrum is summarised by the normalised participation ratio
``PR = (1/n) (Σ λ)² / Σ λ²`` and its centred variant, which leaves out the
largest eigenvalue (the constant mode under the pullback measure)
(:func:`participation_ratio`).

Zero-mass keys. Under the pullback measure ``μ_j = 0`` exactly when no
query attends to key j (column j of ``A`` is zero), and ``Ω_μ⁻¹`` does not
exist. ``Ω_μ⁻¹`` is then read as its pseudo-inverse (``1/μ_j`` where
``μ_j > 0``, 0 where ``μ_j = 0``). Such a key lies in the kernel of ``A``,
so the rule gives it the eigenvalue 0 with the indicator ``e_j`` as
eigenfunction (a function of zero ``μ``-norm); this equals computing the
operator on the keys of positive mass and padding the spectrum with zeros.
The spectrum keeps length n and the participation ratio keeps its ``1/n``
normalisation, so a zero-mass key counts as a direction that carries
nothing. With measures given as ``(mu, pi)``, a key of zero ``μ``-mass that
the attention does read (``Σ_i π_i A_ij² > 0``) has an unbounded gain, and
a ``ValueError`` is raised.

Implemented from the equations of Viganò, Melzi and Ovsjanikov, "Spectral
Geometry of Attention" (NeurIPS 2026, arXiv:2610.06012), Section 3 and
Appendix A; not checked against official code, which is not released as of
15 Nov 2026. Their uncertainty estimator is not part of this module.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

#: Named measures accepted by :func:`attention_operator`.
MEASURES = ("pullback", "counting")

#: Scalar metrics of :func:`operator_metrics`, in order (the ``operator``
#: namespace of :meth:`~spectral_trust.SpectralModel.run_with_spectra`).
OPERATOR_METRIC_NAMES = ["participation_ratio", "participation_ratio_centered", "contraction"]

#: Default tolerance on the row sums of ``A`` under the pullback measure.
ROW_TOL = 1e-2

Measure = Union[str, Tuple[object, object]]


@dataclass
class OperatorSpectrum:
    """Spectrum of the operator ``D`` of :func:`attention_operator`.

    Attributes:
        eigenvalues: ``[..., n]`` in decreasing order, float64, rounding
            below zero clipped to 0.
        eigenfunctions: ``[..., n, n]``, column k the eigenfunction of
            ``eigenvalues[..., k]``, orthonormal in ``L²(μ)`` on the keys of
            positive mass (None when not asked for). A zero-mass key has the
            indicator ``e_j`` (see the module notes).
        mu: ``[..., n]`` measure on the keys.
        pi: ``[..., n]`` measure on the queries.
    """
    eigenvalues: torch.Tensor
    eigenfunctions: Optional[torch.Tensor]
    mu: torch.Tensor
    pi: torch.Tensor


def _as_attention(A) -> torch.Tensor:
    t = A if isinstance(A, torch.Tensor) else torch.as_tensor(np.asarray(A))
    if t.dim() < 2 or t.shape[-1] != t.shape[-2]:
        raise ValueError(f"A must be [..., n, n], got {list(t.shape)}")
    if t.is_complex():
        raise ValueError("A must be real")
    t = t.detach().to(torch.float64)
    if not torch.isfinite(t).all():
        raise ValueError("A has non-finite entries")
    return t


def _normalise_rows(A: torch.Tensor) -> torch.Tensor:
    rows = A.sum(dim=-1, keepdim=True)
    return torch.where(rows > 0, A / torch.where(rows > 0, rows, torch.ones_like(rows)),
                       torch.zeros_like(A))


def _prepare(A, measure: Measure, normalize_rows: bool, average_heads: bool,
             row_tol: float) -> torch.Tensor:
    A = _as_attention(A)
    if average_heads:
        if A.dim() < 3:
            raise ValueError("average_heads needs A of shape [..., heads, n, n]")
        A = A.mean(dim=-3)
    if normalize_rows:
        if (A < 0).any():
            raise ValueError("normalize_rows needs non-negative weights")
        A = _normalise_rows(A)
    if isinstance(measure, str) and measure == "pullback":
        if (A < 0).any():
            raise ValueError("the pullback measure needs a non-negative A")
        if not normalize_rows:
            rows = A.sum(dim=-1)
            bad = (rows != 0) & ((rows - 1).abs() > row_tol)
            if bad.any():
                worst = float(rows[bad].flatten()[0])
                raise ValueError(
                    f"rows of A must sum to 1 (or 0) under the pullback measure; found a row "
                    f"summing to {worst:.4g}. Pass normalize_rows=True for unnormalised "
                    "weights such as value-weighted edges")
    return A


def operator_measures(A, measure: Measure = "pullback", *, normalize_rows: bool = False,
                      average_heads: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """The measures ``(mu, pi)`` on the keys and the queries, ``[..., n]``
    each, float64: ``π = 1/n`` and ``μ = Aᵀ π`` for ``"pullback"``, ones for
    ``"counting"``, or the given pair broadcast to ``A``'s batch shape.
    ``normalize_rows`` and ``average_heads`` as in :func:`attention_operator`."""
    A = _prepare(A, measure, normalize_rows, average_heads, float("inf"))
    return _measures(A, measure)


def _measures(A: torch.Tensor, measure: Measure) -> Tuple[torch.Tensor, torch.Tensor]:
    n = A.shape[-1]
    shape = A.shape[:-1]
    if isinstance(measure, str):
        if measure == "pullback":
            pi = torch.full(shape, 1.0 / n, dtype=torch.float64, device=A.device)
            mu = (pi.unsqueeze(-1) * A).sum(dim=-2)
            return mu, pi
        if measure == "counting":
            ones = torch.ones(shape, dtype=torch.float64, device=A.device)
            return ones, ones.clone()
        raise ValueError(f"measure must be one of {MEASURES} or a (mu, pi) pair, got {measure!r}")
    try:
        mu, pi = measure
    except (TypeError, ValueError):
        raise ValueError(f"measure must be one of {MEASURES} or a (mu, pi) pair") from None
    out = []
    for name, m in (("mu", mu), ("pi", pi)):
        m = m if isinstance(m, torch.Tensor) else torch.as_tensor(np.asarray(m))
        m = m.detach().to(dtype=torch.float64, device=A.device)
        if m.shape[-1:] != (n,):
            raise ValueError(f"{name} must have n = {n} entries in its last dimension, "
                             f"got {list(m.shape)}")
        if not torch.isfinite(m).all() or (m < 0).any():
            raise ValueError(f"{name} must be finite and non-negative")
        out.append(torch.broadcast_to(m, shape).clone())
    return out[0], out[1]


def _factor(A: torch.Tensor, mu: torch.Tensor, pi: torch.Tensor, custom: bool):
    """``B = Ω_π^{1/2} A Ω_μ^{+1/2}`` (so ``S = Bᵀ B``) and the mask of keys
    of positive mass."""
    pos = mu > 0
    inv_sqrt = torch.where(pos, torch.where(pos, mu, torch.ones_like(mu)).rsqrt(),
                           torch.zeros_like(mu))
    if custom and (~pos).any():
        read = (pi.unsqueeze(-1) * A * A).sum(dim=-2)          # Σ_i π_i A_ij²
        if ((~pos) & (read > 0)).any():
            raise ValueError(
                "mu gives zero mass to a key that the attention reads (sum_i pi_i A_ij^2 > 0): "
                "its gain is unbounded and L2(mu) has no adjoint for A")
    B = pi.sqrt().unsqueeze(-1) * A * inv_sqrt.unsqueeze(-2)
    return B, pos


def attention_operator(A, measure: Measure = "pullback", *, normalize_rows: bool = False,
                       average_heads: bool = False, return_difference: bool = False,
                       row_tol: float = ROW_TOL):
    r"""The symmetric matrix ``S = Ω_μ^{-1/2} Aᵀ Ω_π A Ω_μ^{-1/2}`` of an
    attention map, similar to the self-adjoint operator
    ``D = Ω_μ⁻¹ Aᵀ Ω_π A`` on ``L²(μ)`` (see the module notes).

    Args:
        A: ``[..., n, n]`` attention, row i the query (one head, a stack of
            heads, or any batch). It need not be causal.
        measure: ``"pullback"`` (``π = 1/n``, ``μ = Aᵀπ``), ``"counting"``
            (``μ = π = 1``, ``S = D = AᵀA``) or a pair ``(mu, pi)`` of
            non-negative ``[..., n]`` measures on the keys and the queries,
            used as given (not normalised).
        normalize_rows: divide each row of ``A`` by its sum first (a zero
            row stays zero), for weights that are not row-stochastic, such
            as value-weighted edges
            (:func:`~spectral_trust.value_weighted.value_weighted_edges`).
        average_heads: average ``A`` over its heads (dimension -3) first,
            for the operator of the head-averaged attention; the rows are
            normalised after the average.
        return_difference: also return ``D``.
        row_tol: under the pullback measure without ``normalize_rows``,
            every row must sum to 0 or to 1 within this tolerance (the
            pullback construction assumes a row-stochastic ``A``; rows that
            sum to 0 are queries that read nothing). Rows are not
            renormalised, so ``λ₁ = 1`` holds up to the rounding of the
            row sums.

    Returns:
        ``S`` (``[..., n, n]``, float64, exactly symmetric), or ``(S, D)``.
        ``Ω_μ⁻¹`` is the pseudo-inverse on zero-mass keys.

    Implemented from the equations of Viganò, Melzi and Ovsjanikov
    (NeurIPS 2026); not checked against official code, which is not
    released as of 15 Nov 2026.
    """
    A = _prepare(A, measure, normalize_rows, average_heads, row_tol)
    mu, pi = _measures(A, measure)
    B, pos = _factor(A, mu, pi, custom=not isinstance(measure, str))
    S = B.transpose(-1, -2) @ B
    S = 0.5 * (S + S.transpose(-1, -2))
    if not return_difference:
        return S
    inv = torch.where(pos, 1.0 / torch.where(pos, mu, torch.ones_like(mu)), torch.zeros_like(mu))
    D = inv.unsqueeze(-1) * (A.transpose(-1, -2) @ (pi.unsqueeze(-1) * A))
    return S, D


def operator_spectrum(A, measure: Measure = "pullback", *, normalize_rows: bool = False,
                      average_heads: bool = False, eigenfunctions: bool = True,
                      row_tol: float = ROW_TOL) -> OperatorSpectrum:
    r"""Eigenvalues and eigenfunctions of ``D = Ω_μ⁻¹ Aᵀ Ω_π A`` on
    ``L²(μ)``, from the symmetric matrix ``S`` of :func:`attention_operator`
    (same arguments).

    If ``S u = λ u``, then ``φ = Ω_μ^{-1/2} u`` satisfies ``D φ = λ φ``, and
    the ``φ`` are orthonormal in ``L²(μ)``. The eigenvalues are the squared
    gains ``‖Aφ‖²_π / ‖φ‖²_μ``: under the pullback measure the first is 1,
    with a constant eigenfunction; under the counting measure they are the
    squared singular values of ``A``.

    Returns:
        :class:`OperatorSpectrum`, eigenvalues in decreasing order.

    Implemented from the equations of Viganò, Melzi and Ovsjanikov
    (NeurIPS 2026); not checked against official code, which is not
    released as of 15 Nov 2026.
    """
    A = _prepare(A, measure, normalize_rows, average_heads, row_tol)
    mu, pi = _measures(A, measure)
    B, pos = _factor(A, mu, pi, custom=not isinstance(measure, str))
    S = B.transpose(-1, -2) @ B
    S = 0.5 * (S + S.transpose(-1, -2))
    if not eigenfunctions:
        lam = torch.linalg.eigvalsh(S).flip(-1).clamp_min(0.0)
        return OperatorSpectrum(lam, None, mu, pi)
    lam, U = torch.linalg.eigh(S)
    lam, U = lam.flip(-1).clamp_min(0.0), U.flip(-1)
    # φ = Ω_μ^{-1/2} u on the keys of positive mass; a zero-mass key keeps
    # its coordinate (its mode is the indicator e_j, of eigenvalue 0).
    scale = torch.where(pos, torch.where(pos, mu, torch.ones_like(mu)).rsqrt(),
                        torch.ones_like(mu))
    return OperatorSpectrum(lam, scale.unsqueeze(-1) * U, mu, pi)


def participation_ratio(eigenvalues, centered: bool = False, normalized: bool = True):
    r"""Participation ratio of a spectrum, ``(Σ λ)² / Σ λ²``, over the last
    dimension.

    ``normalized=True`` divides by the number of eigenvalues n, which gives
    ``PR = (1/n) (Σλ)² / Σλ²``: ``1/n`` for one active mode, ``rank/n`` for a
    flat non-zero spectrum, invariant to a global rescaling.
    ``centered=True`` leaves out the largest eigenvalue (the constant mode
    under the pullback measure, where it equals 1) and normalises by
    ``n - 1``: ``PR_cent = (1/(n-1)) (Σ_{k≥2} λ_k)² / Σ_{k≥2} λ_k²``. A
    spectrum whose (remaining) eigenvalues all vanish has ratio 0.

    Args:
        eigenvalues: ``[..., n]`` (tensor, array or list), in any order.

    Returns:
        ``[...]``, a tensor for a tensor input, a NumPy value otherwise.
    """
    is_tensor = isinstance(eigenvalues, torch.Tensor)
    lam = eigenvalues.detach() if is_tensor else torch.as_tensor(np.asarray(eigenvalues))
    lam = lam.to(torch.float64)
    if lam.dim() == 0:
        raise ValueError("eigenvalues must have at least one dimension")
    lam = torch.sort(lam, dim=-1, descending=True).values
    if centered:
        lam = lam[..., 1:]
    m = lam.shape[-1]
    num = lam.sum(dim=-1) ** 2
    den = (lam * lam).sum(dim=-1)
    pr = torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)),
                     torch.zeros_like(den))
    if normalized:
        pr = pr / m if m > 0 else torch.zeros_like(pr)
    if is_tensor:
        return pr
    out = pr.numpy()
    return out if out.ndim else np.float64(out)


def spectrum_metrics(eigenvalues) -> Dict[str, torch.Tensor]:
    """The scalars of :data:`OPERATOR_METRIC_NAMES` from a spectrum
    ``[..., n]`` in decreasing order: the participation ratio, its centred
    variant, and the contraction ``√λ₂`` (the largest gain on signals of
    ``μ``-mean zero under the pullback measure; 0 when n = 1)."""
    lam = torch.as_tensor(eigenvalues, dtype=torch.float64)
    if lam.shape[-1] > 1:
        contraction = lam[..., 1].clamp_min(0.0).sqrt()
    else:
        contraction = torch.zeros(lam.shape[:-1], dtype=torch.float64)
    return {
        "participation_ratio": participation_ratio(lam),
        "participation_ratio_centered": participation_ratio(lam, centered=True),
        "contraction": contraction,
    }


def operator_metrics(A, measure: Measure = "pullback", *, normalize_rows: bool = False,
                     average_heads: bool = False, row_tol: float = ROW_TOL,
                     return_eigenvalues: bool = False) -> Dict[str, torch.Tensor]:
    """Participation ratio, centred participation ratio and contraction
    ``√λ₂`` of the operator of ``A`` (arguments as in
    :func:`attention_operator`), one value per matrix: ``[...]`` for ``A`` of
    shape ``[..., n, n]``. For one value per head pass ``[heads, n, n]``;
    for the operator of the head-averaged attention add
    ``average_heads=True``. ``return_eigenvalues`` adds the spectrum
    (``"eigenvalues"``, ``[..., n]``, decreasing).

    Implemented from the equations of Viganò, Melzi and Ovsjanikov
    (NeurIPS 2026); not checked against official code, which is not
    released as of 15 Nov 2026.
    """
    spec = operator_spectrum(A, measure, normalize_rows=normalize_rows,
                             average_heads=average_heads, eigenfunctions=False,
                             row_tol=row_tol)
    out = spectrum_metrics(spec.eigenvalues)
    if return_eigenvalues:
        out["eigenvalues"] = spec.eigenvalues
    return out


__all__ = [
    "MEASURES", "OPERATOR_METRIC_NAMES", "OperatorSpectrum", "attention_operator",
    "operator_measures", "operator_spectrum", "participation_ratio", "spectrum_metrics",
    "operator_metrics",
]
