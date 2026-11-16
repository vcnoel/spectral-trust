# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Graph-frequency filters along the token axis.

A signal ``X`` (``[S, d]``: one row per token, for example the residual
stream) is filtered on a graph over the S tokens:

.. math::
    X' = X + s \, (g(L) X - X)

where ``L`` is the graph's Laplacian, ``g`` a band response (low-pass,
high-pass, band-pass or band-stop) and ``s`` the strength. ``s = 0`` is the
identity, ``s = 1`` applies the filter, ``0 < s < 1`` interpolates and
``s < 0`` pushes away from the filtered signal (a low-pass filter with
``s = -1`` doubles the high frequencies). Each feature column is filtered
independently.

``g(L) X`` is evaluated with the Jackson-damped Chebyshev expansion of the
band indicator that :func:`~spectral_trust.fast.band_energies` uses, from
``order`` products with ``L`` and no eigendecomposition. ``method="dense"``
applies the ideal indicator through a full eigendecomposition, for checks.
By default (``precision="float64"``) the graph, the Laplacian and the filter
are computed in float64 and the result is cast back to the signal's dtype:
residual streams often hold one token (the first) whose norm is a thousand
times the others', and float32 rounding of that token alone would otherwise
reach the other positions (on GPT-2, cached and recomputed generation then
differ by about 0.2 in logits instead of 1e-3). Apple MPS has no float64, so
there the filter runs on the CPU. ``precision="float32"`` keeps the
computation on the signal's device in float32: much faster on GPUs, where
float64 runs 32 to 64 times slower, and noisier on such models.

The graph is built as the measurements build it
(:class:`~spectral_trust.graph.GraphConstructor`: symmetrisation, head
aggregation, normalisation from the ``GSPConfig``), from the attention among
the tokens that form the graph. For a ``token_span`` or a prefix
(``causal=True``) the attention sub-block is taken first and then
symmetrised, aggregated and normalised. With the default symmetrisation
(``"symmetric"``) and head aggregation (``"uniform"``) this equals the
sub-block of the full sequence's adjacency, which is what
:meth:`~spectral_trust.SpectralModel.run_with_spectra` measures under
``token_span``; under ``"row_norm"``, ``"col_norm"`` or
``"attention_weighted"`` the two differ, and the sub-block-first order is the
one a causal filter can use.

Causal filtering
----------------
The symmetrised graph links each token to the tokens after it, so the
symmetric filter at position i reads future positions. ``causal=True``
filters position i on the graph of the prefix that ends at i:

.. math::
    x'_i = x_i + s \, \big( [g(L_{\le i}) X_{\le i}]_i - x_i \big)

where ``L_{\le i}`` is built by the same construction from the attention
among positions ``0..i`` only. The effective operator is lower triangular
(row i is row i of the prefix filter), each position's output is fixed once
its prefix is known, and it equals what the symmetric filter would give at
that position if the sequence ended there. At the last position the two
filters agree. Generation with a KV cache therefore gives the same result as
recomputing the whole sequence at every step. The price is one polynomial
per prefix: ``order * S^3 / 3`` operations for a sequence of S tokens,
against ``order * S^2 * d`` for the symmetric filter.
"""
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..config import GSPConfig
from ..fast import CHEBYSHEV_LOWER_MARGIN, _to_unit_interval, band_chebyshev_coefficients
from ..graph import GraphConstructor, default_graph

FILTER_KINDS = ("lowpass", "highpass", "bandpass", "bandstop")

#: Graphs accepted by name: the layer's head-averaged attention, or the chain
#: graph over positions (a purely positional baseline, DCT-like basis).
GRAPH_NAMES = ("attention", "path")

GraphSpec = Union[str, torch.Tensor, Callable]

# Prefix filters are computed in chunks of prefixes holding at most this many
# bytes of (per-head) float64 adjacency.
_PREFIX_BUDGET_BYTES = 128 * 2 ** 20

_F64 = torch.float64

#: Compute precisions of the filters. float64 (the default) keeps float32
#: rounding of a massive first-token residual from reaching other positions;
#: float32 is much faster on GPUs (consumer GPUs run float64 32 to 64 times
#: slower) and noisier on real models.
PRECISIONS = ("float64", "float32")


# ── responses ────────────────────────────────────────────────────────────────

def _check_spec(kind: str, cutoff, band):
    if kind not in FILTER_KINDS:
        raise ValueError(f"kind must be one of {FILTER_KINDS}, got {kind!r}")
    if kind in ("lowpass", "highpass"):
        if cutoff is None or band is not None:
            raise ValueError(f"kind={kind!r} takes cutoff= (an eigenvalue), not band=")
        return float(cutoff)
    if band is None or cutoff is not None:
        raise ValueError(f"kind={kind!r} takes band=(lo, hi) in eigenvalue units, not cutoff=")
    lo, hi = (float(v) for v in band)
    if hi < lo:
        raise ValueError(f"band must satisfy lo <= hi, got {band}")
    return lo, hi


def _band_coefficients(lo: float, hi: float, order: int, lambda_max: float) -> np.ndarray:
    lo, hi = max(lo, 0.0), min(hi, lambda_max)
    if hi <= lo:
        return np.zeros(order + 1)
    return band_chebyshev_coefficients(np.array([lo, hi]), order, lambda_max)[0]


def filter_coefficients(kind: str, cutoff: Optional[float] = None,
                        band: Optional[Tuple[float, float]] = None, order: int = 64,
                        lambda_max: float = 2.0) -> np.ndarray:
    """Chebyshev coefficients ``[order + 1]`` of a filter on
    ``[0, lambda_max]`` (Jackson-damped band indicators, as in
    :func:`~spectral_trust.fast.band_chebyshev_coefficients`).

    - ``"lowpass"``: the band ``[0, cutoff)``;
    - ``"highpass"``: ``[cutoff, lambda_max]``;
    - ``"bandpass"``: ``[lo, hi)``;
    - ``"bandstop"``: one minus the band-pass. An empty band
      (``lo == hi``) gives exactly the identity.

    A band edge at or above ``lambda_max`` includes the top of the spectrum.
    """
    spec = _check_spec(kind, cutoff, band)
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    lam = float(lambda_max)
    if kind == "lowpass":
        return _band_coefficients(0.0, spec, order, lam)
    if kind == "highpass":
        return _band_coefficients(spec, lam, order, lam)
    c = _band_coefficients(spec[0], spec[1], order, lam)
    if kind == "bandpass":
        return c
    out = -c
    out[0] += 1.0
    return out


def filter_response(lam, kind: str, cutoff: Optional[float] = None,
                    band: Optional[Tuple[float, float]] = None, order: int = 64,
                    lambda_max: float = 2.0) -> np.ndarray:
    """The polynomial response ``g(lambda)`` the Chebyshev filter applies at
    eigenvalues ``lam`` (what ``method="chebyshev"`` does to each mode)."""
    c = filter_coefficients(kind, cutoff, band, order, lambda_max)
    t = _to_unit_interval(np.reshape(lam, -1), float(lambda_max))
    return np.polynomial.chebyshev.chebval(t, c)


def ideal_response(lam, kind: str, cutoff: Optional[float] = None,
                   band: Optional[Tuple[float, float]] = None,
                   lambda_max: float = 2.0) -> np.ndarray:
    """The ideal band indicator at eigenvalues ``lam`` (``method="dense"``).

    Eigenvalues are clipped to ``[0, lambda_max]`` (float rounding of the
    null modes and of the top of the spectrum). Bands are half-open,
    ``[lo, hi)``, except that an upper edge at or above ``lambda_max``
    includes the top, as in the Chebyshev filter and
    :func:`~spectral_trust.fast.band_energies`."""
    spec = _check_spec(kind, cutoff, band)
    top = float(lambda_max)
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, top)

    def below(hi):
        return lam <= hi if hi >= top else lam < hi

    if kind == "lowpass":
        return below(spec).astype(np.float64)
    if kind == "highpass":
        return (lam >= spec).astype(np.float64)
    inside = (lam >= spec[0]) & below(spec[1]) if spec[1] > spec[0] else np.zeros_like(lam, bool)
    return (inside if kind == "bandpass" else ~inside).astype(np.float64)


# ── graphs ───────────────────────────────────────────────────────────────────

def attention_adjacency(attention: torch.Tensor, config: GSPConfig,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Head-aggregated, symmetrised adjacency ``[B, S, S]`` from attention
    ``[B, H, S, S]``, as the measurements build it (in ``dtype``). Filters
    need a symmetric graph built from attention, so the config's
    ``edge_weights`` and ``graph_direction`` must be the defaults."""
    if not default_graph(config):
        raise ValueError(
            "graph filters and frequency-resolved patching run on the symmetrised attention "
            "graph; edge_weights='attention' and graph_direction='symmetric' are required")
    gc = GraphConstructor(config)
    return gc.aggregate_heads(gc.symmetrize_attention(attention.to(dtype)))


def path_adjacency(n: int, batch: int = 1, device=None,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Chain graph over ``n`` positions (edges i -- i+1 of weight 1),
    ``[batch, n, n]``. Its Laplacian eigenvectors are cosines along the
    sequence (exactly the DCT-II basis for the combinatorial Laplacian)."""
    W = torch.zeros(batch, n, n, dtype=dtype, device=device)
    if n > 1:
        i = torch.arange(n - 1, device=device)
        W[:, i, i + 1] = 1.0
        W[:, i + 1, i] = 1.0
    return W


def laplacian(adjacency: torch.Tensor, config: GSPConfig) -> torch.Tensor:
    """Laplacian ``[B, S, S]`` of an adjacency with the config's
    normalisation (:meth:`GraphConstructor.construct_laplacian`, float32).
    Positions with no edges have identity rows under ``"sym"`` and
    ``"rw"``."""
    return GraphConstructor(config).construct_laplacian(adjacency)


def laplacian64(adjacency: torch.Tensor, config: GSPConfig,
                dtype: torch.dtype = _F64) -> torch.Tensor:
    """:meth:`GraphConstructor.construct_laplacian` computed in float64 (or
    ``dtype``), with the same formulas and the same handling of self-loops
    and isolated positions."""
    W = torch.clamp(adjacency.to(dtype), min=0)
    n = W.shape[-1]
    eye = torch.eye(n, dtype=dtype, device=W.device)
    if config.remove_self_loops:
        W = W.masked_fill(eye.bool(), 0.0)
    deg = W.sum(dim=-1)
    if config.normalization == "rw":
        inv = torch.where(deg > 1e-8, 1.0 / deg.clamp(min=1e-8), torch.zeros_like(deg))
        return eye - inv.unsqueeze(-1) * W
    if config.normalization == "sym":
        inv = torch.where(deg > 1e-8, 1.0 / torch.sqrt(deg.clamp(min=1e-8)), torch.zeros_like(deg))
        return eye - inv.unsqueeze(-1) * W * inv.unsqueeze(-2)
    if config.normalization == "none":
        return torch.diag_embed(deg) - W
    raise ValueError(f"Unknown normalization method: {config.normalization}")


def spectrum_bound(L: torch.Tensor, config: GSPConfig) -> torch.Tensor:
    """Upper end of the expansion interval, per item ``[B]``: 2 for the
    normalised Laplacians (spectrum in [0, 2]), the Gershgorin bound for
    ``normalization="none"``."""
    if config.normalization in ("sym", "rw"):
        return torch.full(L.shape[:-2], 2.0, dtype=_F64)
    bound = L.abs().sum(dim=-1).amax(dim=-1).to(_F64).cpu()
    return torch.where(bound > 0, bound, torch.ones_like(bound))


def _resolve_adjacency(graph: GraphSpec, attention, x, config) -> torch.Tensor:
    """Adjacency ``[B, S, S]`` in ``x``'s (compute) dtype."""
    B, S = x.shape[0], x.shape[1]
    dtype = x.dtype
    if isinstance(graph, str):
        if graph == "attention":
            if attention is None:
                raise ValueError("graph='attention' needs the attention weights")
            return attention_adjacency(attention.to(x.device), config, dtype=dtype)
        if graph == "path":
            return path_adjacency(S, B, device=x.device, dtype=dtype)
        raise ValueError(f"graph must be one of {GRAPH_NAMES}, a tensor or a callable; got {graph!r}")
    W = graph(attention, x) if callable(graph) else graph
    W = torch.as_tensor(W).to(device=x.device, dtype=dtype)
    if W.dim() == 2:
        W = W.unsqueeze(0)
    if W.shape[-2:] != (S, S):
        raise ValueError(f"the adjacency must be [S, S] or [B, S, S] with S={S}, got {tuple(W.shape)}")
    return W.expand(B, S, S)


# ── application ──────────────────────────────────────────────────────────────

def _coefficients(kind, cutoff, band, order, lam: torch.Tensor) -> torch.Tensor:
    """Coefficients per item ``[N, order + 1]`` (float64) for bounds ``lam``."""
    lam_np = lam.numpy()
    uniq, inv = np.unique(lam_np, return_inverse=True)
    table = np.stack([filter_coefficients(kind, cutoff, band, order, float(u)) for u in uniq])
    return torch.from_numpy(table[inv.reshape(-1)])


def _chebyshev(L: torch.Tensor, V: torch.Tensor, coeffs: torch.Tensor,
               lam: torch.Tensor) -> torch.Tensor:
    """``sum_k c_k T_k(L~) V`` for ``L [N, S, S]``, ``V [N, S, d]``,
    ``coeffs [N, K + 1]`` and expansion bounds ``lam [N]``, in ``V``'s
    dtype."""
    dtype = V.dtype
    L = L.to(dtype)
    lam = lam.to(_F64)
    lo = -CHEBYSHEV_LOWER_MARGIN * lam
    scale = (2.0 / (lam - lo)).to(device=V.device, dtype=dtype).view(-1, 1, 1)
    shift = (-(2.0 / (lam - lo)) * lo - 1.0).to(device=V.device, dtype=dtype).view(-1, 1, 1)
    c = coeffs.to(device=V.device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)   # [N, K+1, 1, 1]
    K = coeffs.shape[1] - 1

    def apply(M):
        return scale * (L @ M) + shift * M

    T0 = V
    Y = c[:, 0] * T0
    if K == 0:
        return Y
    T1 = apply(T0)
    Y = Y + c[:, 1] * T1
    for k in range(2, K + 1):
        T0, T1 = T1, 2.0 * apply(T1) - T0
        Y = Y + c[:, k] * T1
    return Y


def _dense(L: torch.Tensor, V: torch.Tensor, kind, cutoff, band,
           lam_max: torch.Tensor) -> torch.Tensor:
    """Ideal filter ``U 1_band(Lambda) U^T V`` from a full eigendecomposition
    (float64 on the CPU; returned in ``V``'s dtype and device)."""
    if not bool(((L - L.mT).abs() <= 1e-5).all()):
        raise ValueError("method='dense' needs a symmetric Laplacian (normalization 'sym' or "
                         "'none'); the 'rw' eigenbasis is not orthonormal")
    Ls = (0.5 * (L + L.mT)).to(device="cpu", dtype=_F64)
    vals, vecs = torch.linalg.eigh(Ls)
    r = torch.stack([torch.from_numpy(ideal_response(vals[i].numpy(), kind, cutoff, band,
                                                     float(lam_max[i])))
                     for i in range(vals.shape[0])]).to(vecs)
    out = vecs @ (r.unsqueeze(-1) * (vecs.mT @ V.to(device="cpu", dtype=_F64)))
    return out.to(device=V.device, dtype=V.dtype)


class FilterSpec:
    """A filter and its graph, applied by :func:`apply_filter`."""

    def __init__(self, kind="lowpass", cutoff=None, band=None, strength=1.0,
                 graph: GraphSpec = "attention", causal=False, order=64,
                 method="chebyshev", config: Optional[GSPConfig] = None,
                 precision: str = "float64"):
        _check_spec(kind, cutoff, band)
        if method not in ("chebyshev", "dense"):
            raise ValueError(f"method must be 'chebyshev' or 'dense', got {method!r}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        self.precision = precision
        if isinstance(graph, str) and graph not in GRAPH_NAMES:
            raise ValueError(f"graph must be one of {GRAPH_NAMES}, a tensor or a callable; "
                             f"got {graph!r}")
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        from ..model import default_config
        self.config = config or default_config()
        if self.config.normalization not in ("sym", "rw", "none"):
            raise ValueError(f"unknown normalization {self.config.normalization!r}")
        self.kind, self.cutoff, self.band = kind, cutoff, band
        self.strength = float(strength)
        self.graph, self.causal = graph, bool(causal)
        self.order, self.method = int(order), method

    @property
    def needs_attention(self) -> bool:
        return isinstance(self.graph, str) and self.graph == "attention" or callable(self.graph)

    def identity(self) -> bool:
        """True only for ``strength == 0``. A band that is empty on one
        graph's spectrum need not be empty on another's (the Gershgorin bound
        of ``normalization="none"`` depends on the input), so no other
        shortcut is taken; an empty band-stop still returns its input exactly,
        since its coefficients are exactly the identity's."""
        return self.strength == 0.0

    def __repr__(self):
        what = f"cutoff={self.cutoff}" if self.cutoff is not None else f"band={self.band}"
        g = self.graph if isinstance(self.graph, str) else type(self.graph).__name__
        return (f"FilterSpec({self.kind!r}, {what}, strength={self.strength}, graph={g!r}, "
                f"causal={self.causal}, order={self.order}, method={self.method!r}, "
                f"precision={self.precision!r})")


def compute_placement(device: torch.device, precision: str, method: str = "chebyshev"):
    """``(dtype, device)`` the filter computes in, for a signal on
    ``device``. float64 runs where the signal is, except on Apple MPS, which
    has no float64: there it runs on the CPU. ``method="dense"`` (an
    eigendecomposition, for checks) always runs on the CPU."""
    dtype = _F64 if precision == "float64" else torch.float32
    device = torch.device(device)
    if method == "dense" or (dtype == _F64 and device.type == "mps"):
        return dtype, torch.device("cpu")
    return dtype, device


def _filtered_symmetric(spec: FilterSpec, x, attention, valid) -> torch.Tensor:
    """``g(L) X`` on the whole sequence, ``[B, S, d]``, in ``x``'s dtype."""
    W = _resolve_adjacency(spec.graph, attention, x, spec.config)
    if valid is not None:
        W = W * (valid.unsqueeze(-1) & valid.unsqueeze(-2)).to(W.dtype)
    L = laplacian64(W, spec.config, dtype=x.dtype)
    lam = spectrum_bound(L, spec.config)
    if spec.method == "dense":
        return _dense(L, x, spec.kind, spec.cutoff, spec.band, lam)
    coeffs = _coefficients(spec.kind, spec.cutoff, spec.band, spec.order, lam)
    return _chebyshev(L, x, coeffs, lam)


def _prefix_adjacency(spec: FilterSpec, attention, x, ends: Sequence[int], m: int,
                      valid) -> torch.Tensor:
    """Adjacency of the prefixes ending at each of ``ends``, zero-padded to
    ``m`` positions: ``[B * len(ends), m, m]`` (batch-major), in ``x``'s dtype."""
    B = x.shape[0]
    n = len(ends)
    dtype = x.dtype
    pos = torch.arange(m, device=x.device)
    keep = pos.unsqueeze(0) <= torch.as_tensor(list(ends), device=x.device).unsqueeze(1)  # [n, m]
    keep = keep.unsqueeze(0).expand(B, n, m)
    if valid is not None:
        keep = keep & valid[:, :m].unsqueeze(1)
    pair = (keep.unsqueeze(-1) & keep.unsqueeze(-2)).to(dtype)                      # [B, n, m, m]
    g = spec.graph
    if isinstance(g, str) and g == "attention":
        A = attention[:, :, :m, :m].to(device=x.device, dtype=dtype)                 # [B, H, m, m]
        A = A.unsqueeze(1) * pair.unsqueeze(2)                                       # [B, n, H, m, m]
        return attention_adjacency(A.reshape(B * n, *A.shape[2:]), spec.config, dtype=dtype)
    if isinstance(g, str) and g == "path":
        W = path_adjacency(m, 1, device=x.device, dtype=dtype).unsqueeze(1) * pair
        return W.reshape(B * n, m, m)
    if callable(g):
        out = torch.zeros(B, n, m, m, dtype=dtype, device=x.device)
        for j, i in enumerate(ends):
            a = attention[:, :, :i + 1, :i + 1] if attention is not None else None
            out[:, j, :i + 1, :i + 1] = _resolve_adjacency(g, a, x[:, :i + 1], spec.config)
        return (out * pair).reshape(B * n, m, m)
    W = _resolve_adjacency(g, attention, x, spec.config)[:, :m, :m]
    return (W.unsqueeze(1) * pair).reshape(B * n, m, m)


def _filtered_prefix_rows(spec: FilterSpec, x, attention, rows: Sequence[int],
                          valid) -> torch.Tensor:
    """Row i of ``g(L_{<=i}) X_{<=i}`` for each i in ``rows``:
    ``[B, len(rows), d]``, in ``x``'s dtype."""
    B, S, d = x.shape
    rows = list(rows)
    dtype = x.dtype
    out = torch.zeros(B, len(rows), d, dtype=dtype, device=x.device)
    if not rows:
        return out
    H = attention.shape[1] if attention is not None else 1
    xs = x
    start = 0
    while start < len(rows):
        m = rows[start] + 1
        stop = start + 1
        while stop < len(rows):
            m_next = max(m, rows[stop] + 1)
            if B * (stop + 1 - start) * (H + 4) * m_next * m_next * 8 > _PREFIX_BUDGET_BYTES:
                break
            m = m_next
            stop += 1
        ends = rows[start:stop]
        n = len(ends)
        W = _prefix_adjacency(spec, attention, x, ends, m, valid)                  # [B*n, m, m]
        L = laplacian64(W, spec.config, dtype=dtype)
        E = torch.zeros(B * n, m, 1, dtype=dtype, device=x.device)
        idx = torch.as_tensor(ends, device=x.device).repeat(B)
        E[torch.arange(B * n, device=x.device), idx, 0] = 1.0
        Lt = L.mT                                     # row i of p(L) is (p(L^T) e_i)^T
        lam = spectrum_bound(L, spec.config)
        if spec.method == "dense":
            R = _dense(Lt, E, spec.kind, spec.cutoff, spec.band, lam)
        else:
            coeffs = _coefficients(spec.kind, spec.cutoff, spec.band, spec.order, lam)
            R = _chebyshev(Lt, E, coeffs, lam)
        R = R.reshape(B, n, m)
        out[:, start:stop] = R @ xs[:, :m]
        start = stop
    return out


def apply_filter(spec: FilterSpec, x: torch.Tensor, attention: Optional[torch.Tensor] = None,
                 *, valid: Optional[torch.Tensor] = None, rows: Optional[Sequence[int]] = None,
                 positions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply ``spec`` to ``x [B, S, d]`` (graph from ``attention
    [B, H, S, S]`` when it needs one). Returns the edited rows ``rows``
    (default all), ``[B, len(rows), d]``, in ``x``'s dtype and on its
    device. The computation runs in ``spec.precision`` where
    :func:`compute_placement` puts it.

    ``valid`` (``[B, S]`` bool) drops padded positions from the graph; their
    rows are returned unchanged, as are rows outside ``positions``
    (``[S]`` bool).
    """
    B, S, _ = x.shape
    rows = list(range(S)) if rows is None else [int(r) for r in rows]
    xs = x[:, rows]
    if spec.identity() or not rows:
        return xs.clone()
    dtype, dev = compute_placement(x.device, spec.precision, spec.method)
    work = x.to(device=dev, dtype=dtype)
    att = attention.to(dev) if attention is not None else None
    val = valid.to(dev) if valid is not None else None
    if spec.causal:
        f = _filtered_prefix_rows(spec, work, att, rows, val)
    else:
        if len(rows) != S:
            raise ValueError("the symmetric filter edits every row; rows= is for causal=True")
        f = _filtered_symmetric(spec, work, att, val)
    y = work[:, rows] + spec.strength * (f - work[:, rows])
    keep = torch.ones(B, len(rows), dtype=torch.bool, device=dev)
    if val is not None:
        keep &= val[:, rows]
    if positions is not None:
        keep &= positions[rows].to(dev).unsqueeze(0)
    y = torch.where(keep.unsqueeze(-1), y, work[:, rows])
    return y.to(device=x.device, dtype=x.dtype)


def graph_filter(
    x: torch.Tensor,
    attention: Optional[torch.Tensor] = None,
    *,
    kind: str = "lowpass",
    cutoff: Optional[float] = None,
    band: Optional[Tuple[float, float]] = None,
    strength: float = 1.0,
    graph: GraphSpec = "attention",
    causal: bool = False,
    order: int = 64,
    method: str = "chebyshev",
    config: Optional[GSPConfig] = None,
    token_span=None,
    precision: str = "float64",
) -> torch.Tensor:
    """Filter a signal along the token axis on a graph (functional form of
    :meth:`ResidSite.graph_filter`).

    Args:
        x: signal ``[S, d]`` or ``[B, S, d]``.
        attention: attention weights ``[H, S, S]`` or ``[B, H, S, S]`` (row
            i the query), needed by ``graph="attention"``.
        kind: ``"lowpass"``, ``"highpass"``, ``"bandpass"`` or ``"bandstop"``.
        cutoff: eigenvalue cutoff for low-pass and high-pass filters (the
            ``"sym"`` Laplacian's spectrum lies in [0, 2]).
        band: ``(lo, hi)`` in eigenvalue units for band-pass and band-stop.
        strength: ``s`` in ``X + s (g(L) X - X)``.
        graph: ``"attention"`` (head-averaged attention, symmetrised and
            normalised as in :class:`~spectral_trust.GraphConstructor`),
            ``"path"`` (chain graph over positions), an adjacency tensor
            ``[S, S]`` / ``[B, S, S]``, or a callable ``(attention, x) ->
            adjacency``.
        causal: filter each position on the graph of its prefix (see the
            module notes).
        order: Chebyshev degree.
        method: ``"chebyshev"`` (no eigendecomposition) or ``"dense"`` (ideal
            band indicator from a full eigendecomposition).
        config: graph construction; default the library defaults.
        token_span: ``(start, end)`` with slice semantics: only these tokens
            form the graph and are edited; the others are left unchanged and
            are not read. The graph is built from the attention sub-block of
            the span (see the module notes).
        precision: ``"float64"`` (default; on Apple MPS computed on the CPU)
            or ``"float32"`` (faster on GPUs, noisier when one token's norm
            dwarfs the others'); see the module notes.

    Returns:
        The filtered signal, same shape, dtype and device as ``x``.
    """
    single = x.dim() == 2
    xb = x.unsqueeze(0) if single else x
    a = attention
    if a is not None and a.dim() == 3:
        a = a.unsqueeze(0)
    spec = FilterSpec(kind, cutoff, band, strength, graph, causal, order, method, config,
                      precision)
    S = xb.shape[1]
    s, e = (0, S) if token_span is None else slice(*token_span).indices(S)[:2]
    out = xb.clone()
    sub_a = a[:, :, s:e, s:e] if a is not None else None
    if torch.is_tensor(spec.graph):
        spec.graph = spec.graph[..., s:e, s:e]
    out[:, s:e] = apply_filter(spec, xb[:, s:e], sub_a)
    return out[0] if single else out
