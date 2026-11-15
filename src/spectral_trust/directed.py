# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Directed attention graphs. Added in 0.5.1.

Causal attention is directed: row i of a layer's attention says how much
query i reads each key j <= i, so information flows from token j to token i
and never back. The graphs of the rest of the library are symmetrised
(``(A + Aᵀ) / 2``). This module keeps the direction:

- :func:`directed_operator`: the row-normalised operator ``P = D_out⁻¹ A``.
  Its rows sum to one; for attention it is the head-averaged attention.
- :func:`directed_total_variation`: ``TV(x) = ‖x − P x‖² / ‖x‖²``, how much
  one step of attention transport changes a signal.
- :func:`magnetic_laplacian`, :func:`magnetic_spectrum`,
  :func:`magnetic_hfer` and :func:`magnetic_phase`: the magnetic Laplacian,
  a Hermitian matrix that keeps the direction of each edge in the phase of
  its entry.
- :func:`hermitian_clustering`: clusters that differ by the direction of
  the flow between them rather than by edge density.

Why a directed measure is needed
--------------------------------
A quadratic Laplacian energy ``xᵀ L x`` of a real signal equals
``xᵀ ((L + Lᵀ) / 2) x``: it reads only the symmetric part of ``L``, so a
graph and its reverse give the same energy, and it cannot see which way
information flows. Dahlem, Maniloff and Misiura (2026) prove the general
form of this limit: every transpose-invariant spectral diagnostic of an
attention operator is orientation-blind, unable to tell the operator from
its transpose. The directed total variation escapes it. It is the quadratic
form of ``(I − P)ᵀ (I − P)``, which is not the symmetric part of ``I − P``
and changes when the edges are reversed. The magnetic Laplacian keeps the
direction in its phases, but its energy fractions of a real signal still
cannot tell a graph from its transpose (see :func:`magnetic_hfer`); a
complex signal can.

Orientation convention: ``adjacency[i, j]`` is the weight with which row i
reads column j, as in an attention matrix (query i, key j). Information
flows from j to i.

References:
    A. Sandryhaila and J. M. F. Moura. Discrete signal processing on
    graphs: frequency analysis. IEEE Transactions on Signal Processing,
    62(12), 2014.

    M. Fanuel, C. M. Alaíz and J. A. K. Suykens. Magnetic eigenmaps for
    community detection in directed networks. Physical Review E, 95,
    022302, 2017.

    S. Furutani, T. Shibahara, M. Akiyama, K. Hato and M. Aida. Graph
    signal processing for directed graphs based on the Hermitian Laplacian.
    ECML PKDD 2019.

    X. Zhang, Y. He, N. Brugnone, M. Perlmutter and M. Hirn. MagNet: a
    neural network for directed graphs. NeurIPS 2021.

    M. Cucuringu, H. Li, H. Sun and L. Zanetti. Hermitian matrices for
    clustering directed graphs: insights and applications. AISTATS 2020.

    D. Dahlem, D. Maniloff and M. Misiura. Self-attention as transport:
    limits of symmetric spectral diagnostics. arXiv:2605.04893, 2026.
"""
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .config import GSPConfig
from .fast import num_high_modes
from .spectral import SpectralAnalyzer, SpectralDiagnostics

#: Default charge of the magnetic Laplacian (see :func:`magnetic_laplacian`).
DEFAULT_MAGNETIC_Q = 0.25

#: Metrics of the ``graph`` namespace that exist only for directed graphs,
#: with their tensor family.
DIRECTED_METRIC_FAMILIES: Dict[str, str] = {
    "directed_tv": "hybrid",
}

ArrayLike = Union[torch.Tensor, np.ndarray]


def _as_float64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float64)
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def _as_signal(x) -> torch.Tensor:
    """Signals as float64, or complex128 when they are complex."""
    t = x.detach() if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return t.to(torch.complex128 if t.is_complex() else torch.float64)


def _square(a: torch.Tensor, what: str) -> None:
    if a.dim() < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"{what} must be [..., S, S], got shape {tuple(a.shape)}")


# ── the directed operator and its transport measure ──────────────────────────

def directed_operator(adjacency, remove_self_loops: bool = False) -> torch.Tensor:
    r"""
    The row-normalised operator ``P = D_out⁻¹ A`` of a directed graph.

    Row i of ``P`` is row i of ``A`` divided by its sum, so ``(P x)_i`` is
    the ``A``-weighted average of the signal over the tokens that i reads:
    one step of transport along the edges, from j to i. Negative weights are
    clipped to zero (as in :meth:`GraphConstructor.construct_laplacian`), and
    a row with no weight stays zero.

    For attention, ``A`` is already row-stochastic and ``P`` equals it up to
    rounding. For value-weighted edges (raw norms, see
    :mod:`spectral_trust.value_weighted`) the division puts every row on the
    scale of attention.

    Args:
        adjacency: ``[..., S, S]`` edge weights, row i reading column j.
        remove_self_loops: zero the diagonal first
            (``GSPConfig.remove_self_loops``).

    Returns:
        ``P`` in the input's floating dtype (float32 for integer input).
    """
    A = adjacency if isinstance(adjacency, torch.Tensor) else torch.as_tensor(np.asarray(adjacency))
    _square(A, "adjacency")
    if not A.is_floating_point():
        A = A.to(torch.float32)
    A = torch.clamp(A, min=0)
    if remove_self_loops:
        eye = torch.eye(A.shape[-1], dtype=torch.bool, device=A.device)
        A = A.masked_fill(eye, 0.0)
    rows = A.sum(dim=-1, keepdim=True)
    safe = torch.where(rows > 0, rows, torch.ones_like(rows))
    return torch.where(rows > 0, A / safe, torch.zeros_like(A))


def directed_total_variation(signals, operator, per_column: bool = False) -> torch.Tensor:
    r"""
    Directed total variation of a signal on a directed graph.

    .. math::
        \mathrm{TV}(x) = \frac{\lVert x - P x \rVert^2}{\lVert x \rVert^2}

    with ``P`` the row-normalised operator (:func:`directed_operator`). It
    measures how much one step of attention transport changes the signal:
    ``(P x)_i`` is what token i would receive by averaging the signal over
    the tokens it reads, and ``x_i − (P x)_i`` is how far its own value lies
    from that. It is zero for a signal that one step of transport leaves
    unchanged, a constant signal for instance (``P 1 = 1``), and it changes
    when the edges are reversed, unlike any quadratic Laplacian energy (see
    the module notes).

    This is the quadratic form of Sandryhaila and Moura (2014) with ``P`` as
    the shift: their total variation is ``‖x − A_norm x‖₁`` with
    ``A_norm = A / |λ_max(A)|``, and their 2-Dirichlet form is
    ``S₂(x) = ½ ‖x − A_norm x‖²``. A row-stochastic ``P`` has
    ``|λ_max| = 1``, so ``P`` is its own normalised shift and
    ``TV(x) = 2 S₂(x) / ‖x‖²``. The division by ``‖x‖²`` makes the value
    independent of the scale of ``x``, as for the smoothness index.

    Args:
        signals: ``[..., S, d]`` (or ``[S]``), one column per feature; real
            or complex.
        operator: ``[..., S, S]`` row-normalised operator ``P``. It is not
            renormalised here.
        per_column: False (default) treats the ``d`` columns as one signal,
            ``‖X − P X‖_F² / ‖X‖_F²``, as the layer-level HFER does; True
            gives one value per column, ``[..., d]``, as
            :func:`~spectral_trust.per_column_metrics` does.

    Returns:
        float64 tensor ``[...]`` or ``[..., d]``. A signal (or column) whose
        squared norm is below 1e-8 gets 0, the threshold of the smoothness
        index.
    """
    X = _as_signal(signals)
    if X.dim() == 1:
        X = X[:, None]
    P = _as_float64(operator)
    _square(P, "operator")
    if X.shape[-2] != P.shape[-1]:
        raise ValueError(f"signals have {X.shape[-2]} tokens and the operator {P.shape[-1]}")
    R = X - torch.matmul(P.to(X.dtype), X)
    num = (R.abs() ** 2).sum(dim=-2)
    den = (X.abs() ** 2).sum(dim=-2)
    if not per_column:
        num, den = num.sum(dim=-1), den.sum(dim=-1)
    live = den >= 1e-8
    return torch.where(live, num / torch.where(live, den, torch.ones_like(den)),
                       torch.zeros_like(num))


# ── the magnetic Laplacian ───────────────────────────────────────────────────

def magnetic_laplacian(adjacency, q: float = DEFAULT_MAGNETIC_Q, normalized: bool = True,
                       remove_self_loops: bool = False,
                       dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    r"""
    The magnetic Laplacian of a directed graph, a Hermitian matrix.

    .. math::
        A_s = \tfrac12 (A + A^\top), \qquad
        \Theta^{(q)}_{ij} = 2\pi q \, (A_{ij} - A_{ji}), \qquad
        L^{(q)} = D_s - A_s \odot e^{i \Theta^{(q)}}

    with ``D_s`` the degrees of ``A_s``, and the normalised form
    ``L_N = D_s^{-1/2} L^{(q)} D_s^{-1/2} = I − (D_s^{-1/2} A_s D_s^{-1/2}) ⊙
    e^{iΘ}`` (Fanuel et al. 2017; Furutani et al. 2019; MagNet, Zhang et al.
    2021). The magnitude of each entry is the symmetrised weight; the
    direction of the edge sits in its phase, and ``Θ`` is antisymmetric, so
    ``L`` is Hermitian. The eigenvalues are real and non-negative, and those
    of ``L_N`` lie in ``[0, 2]``. At ``q = 0`` the phases vanish and ``L`` is
    the symmetric Laplacian of ``A_s``
    (:meth:`GraphConstructor.construct_laplacian` with ``"none"`` or
    ``"sym"``). Isolated nodes (zero degree) get an identity row in the
    normalised form, as there.

    The charge ``q``: for weights in ``[0, 1]`` (the rows of a
    row-normalised operator), ``A_ij − A_ji`` lies in ``[−1, 1]`` and the
    phase in ``[−2πq, 2πq]``. ``q = 1/4`` (the default) is the largest charge
    for which the phase stays in ``[−π/2, π/2]``: the real part
    ``A_s cos Θ`` of every entry stays non-negative, and the map from
    ``A_ij − A_ji`` to the phase stays one to one. At ``q = 1/4`` a one-way
    edge of weight 1 gets the phase ``π/2`` (``+i`` against ``−i`` for its
    reverse), the setting MagNet uses as its directed extreme (it studies
    ``0 <= q <= 1/4``). At ``q = 1/2`` such an edge would get the phase
    ``π``, the same in both directions, and the direction would be lost.
    Attention weights are mostly far below 1, so their phases are small;
    the strong edges (to the attention sink, for instance) carry most of the
    phase.

    Args:
        adjacency: ``[..., S, S]`` edge weights, row i reading column j;
            negative weights are clipped to zero. Use the row-normalised
            operator (:func:`directed_operator`) for edges outside ``[0, 1]``.
        q: the charge, ``>= 0``.
        normalized: ``L_N`` (default) or the unnormalised ``L^{(q)}``.
        remove_self_loops: zero the diagonal first.
        dtype: ``torch.complex128`` (default) or ``torch.complex64``. The
            matrix is built in float64 and cast.

    Returns:
        ``[..., S, S]`` complex Hermitian tensor.
    """
    if q < 0:
        raise ValueError(f"q must be >= 0, got {q}")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError(f"dtype must be torch.complex64 or torch.complex128, got {dtype}")
    A = torch.clamp(_as_float64(adjacency), min=0)
    _square(A, "adjacency")
    if remove_self_loops:
        eye = torch.eye(A.shape[-1], dtype=torch.bool, device=A.device)
        A = A.masked_fill(eye, 0.0)
    At = A.transpose(-2, -1)
    A_s = 0.5 * (A + At)
    theta = (2.0 * math.pi * float(q)) * (A - At)
    deg = A_s.sum(dim=-1)
    n = A.shape[-1]
    eye = torch.eye(n, dtype=torch.float64, device=A.device)
    if normalized:
        inv = torch.where(deg > 1e-8, 1.0 / torch.sqrt(deg.clamp(min=1e-8)), torch.zeros_like(deg))
        W = inv.unsqueeze(-1) * A_s * inv.unsqueeze(-2)
        L = eye - torch.polar(W, theta)
    else:
        L = torch.diag_embed(deg).to(torch.complex128) - torch.polar(A_s, theta)
    # Exactly Hermitian: Θ is antisymmetric to the last bit, but the product
    # with A_s rounds on each side separately.
    L = 0.5 * (L + L.transpose(-2, -1).conj())
    return L.to(dtype)


def magnetic_spectrum(adjacency, q: float = DEFAULT_MAGNETIC_Q, normalized: bool = True,
                      remove_self_loops: bool = False,
                      dtype: torch.dtype = torch.complex128
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues (ascending, real) and unit-norm complex eigenvectors (the
    columns) of :func:`magnetic_laplacian`, from a Hermitian eigensolver
    (``torch.linalg.eigh``) in ``dtype``. Arguments as there."""
    L = magnetic_laplacian(adjacency, q=q, normalized=normalized,
                           remove_self_loops=remove_self_loops, dtype=dtype)
    return torch.linalg.eigh(L)


def _energy_fractions(masses: torch.Tensor, cutoff_ratio: float) -> torch.Tensor:
    """Energy in the top ``num_high_modes`` modes and in all modes, from
    per-mode energies ``[..., k, d]`` (modes ascending)."""
    k = masses.shape[-2]
    cut = k - num_high_modes(k, cutoff_ratio)
    return masses[..., cut:, :].sum(dim=-2), masses.sum(dim=-2)


def magnetic_hfer(signals, adjacency=None, q: float = DEFAULT_MAGNETIC_Q,
                  normalized: bool = True, cutoff_ratio: float = 0.1,
                  per_column: bool = False, remove_self_loops: bool = False,
                  eigenvectors: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    High-frequency energy ratio of a signal on the magnetic Laplacian.

    The energy of the signal in mode k is ``|u_kᴴ x|²`` (``U`` unitary, from
    :func:`magnetic_spectrum`), and the HFER is the share of the top
    :func:`~spectral_trust.fast.num_high_modes` modes, the rule of every
    other HFER in the library. At ``q = 0`` it is the HFER on the symmetric
    Laplacian of ``(A + Aᵀ) / 2``.

    **Orientation-blind for real signals.** Reversing every edge
    (``A → Aᵀ``) conjugates the magnetic Laplacian, so its eigenvectors
    become ``conj(U)`` with the same eigenvalues, and for a real ``x``
    ``|conj(u_k)ᴴ x| = |u_kᴴ x|``. The magnetic HFER of a real signal is
    therefore identical for a graph and its transpose: the phases change
    which energies a real signal has (they depend on ``|A_ij − A_ji|``) but
    not in a way that reveals which way the edges point. A complex signal
    can tell the two apart. Use :func:`directed_total_variation` for a
    direction-aware measure of a real signal.

    Args:
        signals: ``[S, d]`` or ``[S]``, real or complex.
        adjacency: ``[S, S]`` edge weights (see :func:`magnetic_laplacian`);
            not needed when ``eigenvectors`` are given.
        q, normalized, remove_self_loops: as in :func:`magnetic_laplacian`.
        cutoff_ratio: as ``GSPConfig.hfer_cutoff_ratio``.
        per_column: one value per column (``[d]``) instead of the columns
            taken together.
        eigenvectors: the ``[S, S]`` eigenvectors of
            :func:`magnetic_spectrum`, ascending, to reuse them.

    Returns:
        float64 tensor, a scalar or ``[d]``. A signal with total energy below
        1e-8 gets 0, as in :meth:`SpectralAnalyzer.hfer_from_masses`.
    """
    X = _as_signal(signals)
    if X.dim() == 1:
        X = X[:, None]
    if eigenvectors is None:
        if adjacency is None:
            raise ValueError("pass the adjacency or its eigenvectors")
        _, U = magnetic_spectrum(adjacency, q=q, normalized=normalized,
                                 remove_self_loops=remove_self_loops)
    else:
        U = eigenvectors.to(torch.complex128)
    masses = (torch.matmul(U.conj().transpose(-2, -1), X.to(torch.complex128)).abs() ** 2)
    high, total = _energy_fractions(masses, cutoff_ratio)
    if not per_column:
        high, total = high.sum(dim=-1), total.sum(dim=-1)
    live = total >= 1e-8
    return torch.where(live, high / torch.where(live, total, torch.ones_like(total)),
                       torch.zeros_like(high))


def _gauge_fixed_phase(u: torch.Tensor) -> torch.Tensor:
    """Angles of a complex vector after the global phase is fixed so that
    its entry of largest modulus is real and positive."""
    ref = int(torch.argmax(u.abs()))
    z = u * torch.exp(-1j * torch.angle(u[ref]))
    return torch.angle(z)


def magnetic_phase(adjacency=None, q: float = DEFAULT_MAGNETIC_Q, normalized: bool = True,
                   mode: int = 0, remove_self_loops: bool = False,
                   eigenvectors: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Phase of a magnetic eigenvector, as an ordering of the nodes.

    At ``q = 0`` the lowest eigenvector is the trivial mode (a constant
    phase). For ``q > 0`` its phase is the first readout that is not
    trivial: a potential of the edge asymmetries. On a graph without cycles
    the magnetic Laplacian equals the symmetric one up to a diagonal unitary
    change of basis, and the lowest eigenvector has the phases ``φ`` with
    ``φ_i − φ_j = Θ_ij`` on every edge, so the phase grows along the
    direction of the edges, from the tokens that are read to the tokens that
    read them (a chain is ordered exactly). With cycles, as in attention,
    the lowest mode fits ``φ_i − φ_j ≈ Θ_ij`` with the weights ``A_s``
    (an angular synchronisation of the asymmetries): the phase then ranks
    tokens by the net asymmetry of their strongest edges, not by position.
    On a layer dominated by its sink, it mostly follows how much each token
    reads from the sink.

    The global phase is arbitrary; it is fixed so that the entry of largest
    modulus (on attention graphs usually the sink) has phase 0. Angles lie
    in ``(−π, π]``; apply ``numpy.unwrap`` along an ordering if they wrap,
    which happens when ``q`` times the summed asymmetry along a path exceeds
    one half.

    Args:
        adjacency: ``[S, S]`` edge weights (see :func:`magnetic_laplacian`);
            not needed when ``eigenvectors`` are given.
        q, normalized, remove_self_loops: as in :func:`magnetic_laplacian`.
        mode: which eigenvector, in ascending order of eigenvalue (default
            0, the lowest).
        eigenvectors: the eigenvectors of :func:`magnetic_spectrum`, to
            reuse them.

    Returns:
        float64 tensor ``[S]`` of angles.
    """
    if eigenvectors is None:
        if adjacency is None:
            raise ValueError("pass the adjacency or its eigenvectors")
        _, eigenvectors = magnetic_spectrum(adjacency, q=q, normalized=normalized,
                                            remove_self_loops=remove_self_loops)
    if eigenvectors.dim() != 2:
        raise ValueError("magnetic_phase takes one graph ([S, S])")
    return _gauge_fixed_phase(eigenvectors[:, mode].to(torch.complex128))


# ── Hermitian (flow-imbalance) clustering ────────────────────────────────────

@dataclass
class FlowClusters:
    """Result of :func:`hermitian_clustering`.

    Attributes:
        labels: ``[n]`` cluster of each node, ``0 .. k-1``, numbered in order
            of first appearance along the node index.
        flow: ``[k, k]``, ``flow[a, b] = sum of A[i, j]`` over rows i in
            cluster a and columns j in cluster b: how much cluster a reads
            from cluster b (information flows from b to a).
        imbalance: ``[k, k]``,
            ``flow[a, b] / (flow[a, b] + flow[b, a]) − 1/2`` (0 when both
            are 0), in ``[−1/2, 1/2]`` and antisymmetric. Positive when a
            reads more from b than b reads from a. Its absolute value is the
            cut imbalance ratio of Cucuringu et al. (2020).
        eigenvalues: eigenvalues of the Hermitian matrix whose eigenvectors
            were clustered, by decreasing absolute value.
        embedding: ``[n, 2 l]`` real and imaginary parts of the ``l``
            eigenvectors, the rows k-means clustered.
    """
    labels: np.ndarray
    flow: np.ndarray
    imbalance: np.ndarray
    eigenvalues: np.ndarray
    embedding: np.ndarray

    def to_dict(self) -> Dict:
        return {"labels": self.labels.tolist(), "flow": self.flow.tolist(),
                "imbalance": self.imbalance.tolist(),
                "eigenvalues": self.eigenvalues.tolist()}


def _kmeans(X: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    """Best of ``n_init`` seeded k-means++ runs (scipy's kmeans2)."""
    from scipy.cluster.vq import kmeans2

    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(max(int(n_init), 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centres, labels = kmeans2(X, k, iter=100, minit="++",
                                      seed=np.random.default_rng(rng.integers(2 ** 32)))
        cost = float(((X - centres[labels]) ** 2).sum())
        if cost < best_cost - 1e-12:
            best, best_cost = labels, cost
    return best


def flow_imbalance(adjacency, labels, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The ``flow`` and ``imbalance`` matrices of :class:`FlowClusters` for
    any labelling of the nodes (``adjacency[i, j]``: row i reads column j)."""
    A = np.clip(_as_float64(adjacency).cpu().numpy(), 0.0, None)
    labels = np.asarray(labels, dtype=np.int64)
    k = int(labels.max()) + 1 if k is None else int(k)
    onehot = np.zeros((A.shape[0], k))
    onehot[np.arange(A.shape[0]), labels] = 1.0
    flow = onehot.T @ A @ onehot
    both = flow + flow.T
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(both > 0, flow / np.where(both > 0, both, 1.0) - 0.5, 0.0)
    np.fill_diagonal(imb, 0.0)
    return flow, imb


def hermitian_clustering(adjacency, k: int, n_vectors: Optional[int] = None,
                         normalized: bool = False, seed: int = 0,
                         n_init: int = 10) -> FlowClusters:
    r"""
    Clusters of a directed graph that differ by the direction of the flow
    between them (Cucuringu, Li, Sun and Zanetti, AISTATS 2020).

    The Hermitian adjacency ``H = i (A − Aᵀ)`` puts ``+i`` weight on an edge
    read one way and ``−i`` on the reverse, so a reciprocated edge cancels
    and only the net direction remains. Its eigenvalues are real and come in
    pairs ``±λ``. The rows of its ``n_vectors`` eigenvectors of largest
    ``|λ|`` (as real and imaginary parts) are clustered by k-means. Two
    nodes land together when they send and receive their net flow from the
    same clusters, even when the symmetrised graph has the same density
    everywhere. The k-means rows equal, up to a fixed rotation, the rows of
    the projection ``Σ g gᴴ`` onto those eigenvectors, so the clustering is
    that of the projection.

    Args:
        adjacency: ``[n, n]`` edge weights, row i reading column j (attention
            orientation). Negative weights are clipped; the diagonal carries
            no direction and is ignored.
        k: number of clusters.
        n_vectors: eigenvectors kept, by decreasing ``|λ|``. Default 2, the
            leading pair ``±λ``: the strongest pattern of net flow, which
            separates the planted blocks of an ordered or cyclic flow
            between up to four clusters in the tests. A larger even number
            (up to ``2 ⌊k / 2⌋``, the largest rank of the ``k × k``
            antisymmetric matrix of net flows between clusters) can add
            structure, or noise when the flow pattern has a lower rank.
        normalized: use ``D^{-1/2} H D^{-1/2}`` with ``D`` the degrees of
            ``A + Aᵀ``, which keeps high-degree nodes from dominating the
            leading eigenvectors.
        seed: seeds k-means (best of ``n_init`` k-means++ runs).

    Returns:
        :class:`FlowClusters`: labels, the flow and imbalance matrices
        between clusters, the eigenvalues, the embedding.
    """
    A = np.clip(_as_float64(adjacency).cpu().numpy(), 0.0, None)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency must be [n, n], got {A.shape}")
    n = A.shape[0]
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}], got {k}")
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    H = 1j * (A - A.T)
    if normalized:
        deg = (A + A.T).sum(axis=1)
        inv = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
        H = inv[:, None] * H * inv[None, :]
    lam, U = np.linalg.eigh(H)
    order = np.argsort(-np.abs(lam), kind="stable")
    l = int(n_vectors) if n_vectors is not None else 2
    l = min(max(l, 1), n)
    G = U[:, order[:l]]
    emb = np.concatenate([G.real, G.imag], axis=1)
    if k == 1:
        raw = np.zeros(n, dtype=np.int64)
    else:
        raw = _kmeans(emb, k, seed, n_init)
    # Number the clusters by first appearance along the node index.
    remap, labels = {}, np.empty(n, dtype=np.int64)
    for i, c in enumerate(raw):
        labels[i] = remap.setdefault(int(c), len(remap))
    flow, imb = flow_imbalance(A, labels, k=max(len(remap), 1))
    return FlowClusters(labels=labels, flow=flow, imbalance=imb,
                        eigenvalues=lam[order], embedding=emb)


# ── layer diagnostics on a directed graph ────────────────────────────────────

def magnetic_normalized(normalization: str) -> bool:
    """Whether ``GSPConfig.normalization`` selects the normalised magnetic
    Laplacian (``"sym"``) or the unnormalised one (``"none"``). The
    random-walk form is not Hermitian and has no magnetic counterpart here."""
    if normalization == "sym":
        return True
    if normalization == "none":
        return False
    raise ValueError(
        f"directed graphs are measured through the magnetic Laplacian, which is Hermitian "
        f"for normalization='sym' or 'none'; got {normalization!r}")


def analyze_directed_layer(signals, operator, config: Optional[GSPConfig] = None,
                           layer: int = 0) -> Tuple[SpectralDiagnostics, Dict[str, object]]:
    """
    Layer diagnostics of a directed graph: the magnetic counterparts of
    :meth:`SpectralAnalyzer.analyze_layer`, plus the directed total
    variation.

    The spectral metrics come from the magnetic Laplacian of ``operator``
    (charge ``config.magnetic_q``, normalised for ``normalization="sym"``,
    unnormalised for ``"none"``), with the definitions of ``analyze_layer``:
    ``energy`` is ``Σ_c x_cᴴ L x_c`` (real), ``spectral_masses`` are
    ``|U^H X|²`` summed over columns, ``hfer`` and ``spectral_entropy`` come
    from them, ``fiedler_value`` is the second-smallest eigenvalue and
    ``connectivity`` tests it against the zero tolerance. At ``q = 0`` they
    equal the symmetric metrics of ``(P + Pᵀ) / 2``. For ``q > 0`` the
    smallest eigenvalue can be positive (it is zero only when the phases
    around every cycle cancel), and the energy fractions of the real hidden
    states do not depend on the direction of the edges (see
    :func:`magnetic_hfer`).

    Args:
        signals: ``[S, d]`` hidden states (``config.project_dim`` applies, as
            in ``analyze_layer``).
        operator: ``[S, S]`` row-normalised operator (:func:`directed_operator`).
        config: graph settings; default ``GSPConfig(device="cpu")``.
        layer: layer index stored in the diagnostics.

    Returns:
        ``(diagnostics, extras)``: a :class:`SpectralDiagnostics` (its
        eigenvectors are complex) and ``{"directed_tv": float, "phase":
        [S] array}`` (:func:`directed_total_variation`, :func:`magnetic_phase`).
    """
    cfg = config or GSPConfig(device="cpu", verbose=False)
    an = SpectralAnalyzer(cfg)
    X = signals.detach().to(device="cpu", dtype=torch.float32).numpy() \
        if isinstance(signals, torch.Tensor) else np.asarray(signals, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    X = an.project_signals(X).astype(np.float64)
    P = _as_float64(operator).cpu()
    if P.dim() > 2:
        if int(np.prod(P.shape[:-2])) != 1:
            raise ValueError(f"expected one [S, S] operator, got shape {tuple(P.shape)}")
        P = P.reshape(P.shape[-2:])
    n = P.shape[-1]
    L = magnetic_laplacian(P, q=float(getattr(cfg, "magnetic_q", DEFAULT_MAGNETIC_Q)),
                           normalized=magnetic_normalized(cfg.normalization),
                           remove_self_loops=cfg.remove_self_loops)
    lam_t, U_t = torch.linalg.eigh(L)
    lam = np.maximum(lam_t.numpy(), 0.0)
    U = U_t.numpy()
    Xt = torch.from_numpy(X)
    energy = float((Xt * (L.real @ Xt)).sum())
    norm = float((Xt * Xt).sum())
    smooth = energy / norm if norm >= 1e-8 else 0.0
    masses = (np.abs(U.conj().T @ X) ** 2).sum(axis=1)
    diag = SpectralDiagnostics(
        layer=layer,
        energy=energy,
        smoothness_index=smooth,
        spectral_entropy=an.spectral_entropy_from_masses(masses),
        hfer=an.hfer_from_masses(masses, cfg.hfer_cutoff_ratio),
        eigenvalues=lam,
        eigenvectors=U if cfg.save_intermediate else None,
        spectral_masses=masses,
        fiedler_value=float(lam[1]) if n > 1 else 0.0,
        connectivity=an.is_connected(lam, n),
        spectral_radius=float(lam[-1]) if n > 0 else 0.0,
        eig_energy=an.compute_eig_energy(lam),
        eig_spectral_entropy=an.compute_eig_spectral_entropy(lam),
        eig_hfer=an.compute_eig_hfer(lam),
    )
    extras = {
        "directed_tv": float(directed_total_variation(Xt, P)),
        "phase": _gauge_fixed_phase(U_t[:, 0]).numpy() if n > 0 else np.zeros(0),
    }
    return diag, extras


__all__ = [
    "directed_operator", "directed_total_variation", "magnetic_laplacian",
    "magnetic_spectrum", "magnetic_hfer", "magnetic_phase", "hermitian_clustering",
    "flow_imbalance", "FlowClusters", "analyze_directed_layer", "magnetic_normalized",
    "DEFAULT_MAGNETIC_Q", "DIRECTED_METRIC_FAMILIES",
]
