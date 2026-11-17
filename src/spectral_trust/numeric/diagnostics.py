# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Layer diagnostics of an attention graph and the hidden states on it, with
the construction of Bao et al. (2026) available option by option.

Per layer:

1. the attention ``A [H, S, S]`` loses the positions in ``drop`` (special
   tokens such as BOS) and, with ``renormalize=True``, each row is
   renormalised over the remaining keys;
2. ``W = (1 / 2H) sum_h (A_h + A_h^T)``;
3. ``lambda_2`` (``fiedler``) and the low end of the spectrum come from the
   normalised Laplacian ``I - D^{-1/2} W D^{-1/2}`` (spectrum in [0, 2]);
4. the graph Fourier basis is the eigenbasis of the combinatorial Laplacian
   ``D - W`` (``basis="combinatorial"``, as in the paper) or of the
   normalised one (``basis="normalized"``, the library default elsewhere);
   with ``p_m`` the share of the signal's energy in mode m (ascending
   eigenvalue), ``hfer`` is the share of the top
   ``N - floor((1 - hfer_cutoff) N)`` modes (with 0.5: modes above
   ``floor(N/2)``), ``spectral_entropy = -sum p log p`` (nats) and
   ``exp_se = exp(spectral_entropy)`` its effective number of modes;
5. controls without the graph: ``pos_hfer`` and ``pos_exp_se`` in the
   orthonormal DCT-II basis along positions (the path graph's basis),
   ``gini``, the attention concentration of each causal row normalised by
   its maximum ``1 - 1/i`` and averaged over heads and rows 2..N, and
   ``sink_mass``, the attention to the first position before it is
   dropped.

:func:`profile_from_outputs` runs this over the layers of one forward pass
(``output_attentions=True, output_hidden_states=True``) and averages over
layers, as the paper reports.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

#: Scalar metrics per layer, in order.
PROFILE_METRICS = ("fiedler", "hfer", "spectral_entropy", "exp_se", "pos_hfer", "pos_exp_se",
                   "gini", "sink_mass", "signal_energy")


#: Accepted ``edge_weights`` of :func:`profile_from_outputs` (as ``GSPConfig.edge_weights``).
EDGE_WEIGHTS = ("attention", "value", "value_per_head")


def _layer_edges(edges, layer: int, edge_weights: str, item: int):
    """The ``[H, S, S]`` (``[1, S, S]`` for ``"value"``) edges of one item."""
    if layer not in edges:
        raise ValueError(f"edges has no layer {layer} (layers {sorted(edges)})")
    e = edges[layer]
    e = e if torch.is_tensor(e) else torch.as_tensor(np.asarray(e))
    full = 3 if edge_weights == "value" else 4
    if e.dim() == full:
        e = e[item]
    return e if edge_weights == "value_per_head" else e.reshape(1, *e.shape[-2:])


def _center_of(h: np.ndarray, keep: Sequence[int], positions) -> np.ndarray:
    """Mean of ``h [S, d]`` over ``positions`` (default ``keep``)."""
    S = h.shape[0]
    pos = list(keep) if positions is None else sorted({int(p) % S for p in positions})
    if not pos:
        raise ValueError("center_positions selects no position")
    return h[pos].mean(0, keepdims=True)


def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().to(device="cpu", dtype=torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def attention_graph(attention, drop: Sequence[int] = (), renormalize: bool = True):
    """``(W, A)``: the head-averaged symmetrised adjacency ``[n, n]`` and the
    processed attention ``[H, n, n]`` after dropping ``drop`` and (optionally)
    renormalising each row over the remaining keys."""
    A = _np(attention)
    if A.ndim == 2:
        A = A[None]
    S = A.shape[-1]
    keep = np.array([i for i in range(S) if i not in set(int(d) % S for d in drop)])
    A = A[:, keep][:, :, keep]
    if renormalize:
        rs = A.sum(-1, keepdims=True)
        A = np.where(rs > 0, A / np.where(rs > 0, rs, 1.0), 0.0)
    W = 0.5 * (A + np.swapaxes(A, -1, -2)).mean(0)
    return W, A


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = W.sum(-1)
    inv = np.where(deg > 1e-12, 1.0 / np.sqrt(np.where(deg > 1e-12, deg, 1.0)), 0.0)
    return np.eye(len(W)) - inv[:, None] * W * inv[None, :]


def combinatorial_laplacian(W: np.ndarray) -> np.ndarray:
    return np.diag(W.sum(-1)) - W


def _modal(p_energy: np.ndarray, hfer_cutoff: float) -> Dict[str, float]:
    from ..fast import num_high_modes
    total = p_energy.sum()
    n = len(p_energy)
    if total <= 0:
        return {"hfer": 0.0, "spectral_entropy": 0.0, "exp_se": 1.0}
    p = p_energy / total
    k = n - num_high_modes(n, hfer_cutoff)
    nz = p[p > 0]
    se = float(-(nz * np.log(nz)).sum())
    return {"hfer": float(p[k:].sum()), "spectral_entropy": se, "exp_se": float(np.exp(se))}


def graph_signal_metrics(adjacency, signal=None, *, hfer_cutoff: float = 0.5, n_low: int = 10,
                         basis: str = "combinatorial", return_masses: bool = False) -> Dict:
    """Diagnostics of one graph ``[n, n]`` and an optional signal ``[n, d]``:
    ``fiedler``, ``low_eigenvalues`` (``lambda_2 .. lambda_{n_low + 1}`` of
    the normalised Laplacian) and, with a signal, ``hfer``,
    ``spectral_entropy``, ``exp_se`` and ``signal_energy`` in the ``basis``
    (``"combinatorial"`` or ``"normalized"``) eigenbasis.
    ``return_masses`` adds the per-mode energies ``masses``."""
    W = _np(adjacency)
    lam = np.linalg.eigvalsh(normalized_laplacian(W))
    out: Dict = {"fiedler": float(lam[1]) if len(lam) > 1 else 0.0,
                 "low_eigenvalues": lam[1:1 + int(n_low)].tolist()}
    if signal is not None:
        X = _np(signal)
        if basis == "combinatorial":
            L = combinatorial_laplacian(W)
        elif basis == "normalized":
            L = normalized_laplacian(W)
        else:
            raise ValueError("basis must be 'combinatorial' or 'normalized'")
        _, U = np.linalg.eigh(0.5 * (L + L.T))
        masses = ((U.T @ X) ** 2).sum(-1)
        out.update(_modal(masses, hfer_cutoff))
        out["signal_energy"] = float(masses.sum())
        if return_masses:
            out["masses"] = masses
    return out


def positional_metrics(signal, hfer_cutoff: float = 0.5) -> Dict[str, float]:
    """``pos_hfer``, ``pos_spectral_entropy`` and ``pos_exp_se`` of a signal
    ``[n, d]`` in the orthonormal DCT-II basis along positions."""
    from scipy.fft import dct
    X = _np(signal)
    masses = (dct(X, type=2, norm="ortho", axis=0) ** 2).sum(-1)
    m = _modal(masses, hfer_cutoff)
    return {"pos_hfer": m["hfer"], "pos_spectral_entropy": m["spectral_entropy"],
            "pos_exp_se": m["exp_se"]}


def attention_gini(attention) -> float:
    """Normalised attention Gini of causal rows (Hurley and Rickard 2009, as
    in Bao et al. 2026, App. H): for head h and row i >= 2 (1-based), the
    Gini coefficient of ``A[h, i, :i]`` divided by its maximum ``1 - 1/i``,
    averaged over heads and rows. 0 is uniform attention, 1 one key."""
    A = torch.as_tensor(_np(attention))
    if A.dim() == 2:
        A = A.unsqueeze(0)
    H, S, _ = A.shape
    if S < 2:
        return 0.0
    tri = torch.tril(torch.ones(S, S, dtype=torch.bool))
    Z = torch.where(tri, A.clamp_min(0), torch.full_like(A, -1.0))
    Z, _ = torch.sort(Z, dim=-1)                       # masked entries first
    m = torch.arange(1, S + 1, dtype=torch.float64).view(1, S, 1)   # keys per row
    idx = torch.arange(S, dtype=torch.float64).view(1, 1, S)
    rank = idx - (S - m) + 1                           # 1-based rank among available keys
    valid = rank >= 1
    Zv = torch.where(valid, Z, torch.zeros_like(Z))
    tot = Zv.sum(-1)
    g = (2 * (rank.clamp_min(0) * Zv).sum(-1) / (m.squeeze(-1) * tot.clamp_min(1e-30))
         - (m.squeeze(-1) + 1) / m.squeeze(-1))
    g = g / (1 - 1 / m.squeeze(-1)).clamp_min(1e-30)
    g = torch.where(tot > 0, g, torch.zeros_like(g))
    return float(g[:, 1:].mean())


def sink_mass(attention, sink: int = 0) -> float:
    """Mean attention to position ``sink`` over heads and the queries after
    it."""
    A = _np(attention)
    if A.ndim == 2:
        A = A[None]
    return float(A[:, sink + 1:, sink].mean()) if A.shape[-1] > sink + 1 else 0.0


def pool_nodes(adjacency, signal, groups: Sequence[Sequence[int]]):
    """Merge node groups into single nodes: edge weights are summed between
    groups, signals averaged within each group. Positions in no group are
    dropped. Returns ``(W [G, G], X [G, d] or None)``. Used to give a
    digit-level tokenizer one node per value, the aggregation Bao et al.
    (2026) leave to future work."""
    W = _np(adjacency)
    G = len(groups)
    M = np.zeros((G, W.shape[0]))
    for g, idx in enumerate(groups):
        M[g, list(idx)] = 1.0
    Wp = M @ W @ M.T
    Xp = None
    if signal is not None:
        X = _np(signal)
        sizes = M.sum(1, keepdims=True)
        Xp = (M @ X) / np.where(sizes > 0, sizes, 1.0)
    return Wp, Xp


@dataclass
class GraphSignalProfile:
    """Per-layer diagnostics of one input. ``per_layer[name]`` is ``[L]``
    (``low_eigenvalues`` is ``[L, n_low]``); ``mean`` the layer averages."""
    layers: List[int]
    per_layer: Dict[str, np.ndarray]
    masses: Optional[Dict[int, np.ndarray]] = field(default=None, repr=False)

    @property
    def mean(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.per_layer.items()
                if np.ndim(v) == 1}

    def to_dict(self) -> Dict:
        return {"layers": list(self.layers), "mean": self.mean,
                "per_layer": {k: np.asarray(v).tolist() for k, v in self.per_layer.items()}}


def profile_from_outputs(attentions, hidden_states=None, *, layers: Optional[Sequence[int]] = None,
                         pairing: str = "input", drop: Sequence[int] = (0,),
                         renormalize: bool = True, hfer_cutoff: float = 0.5, n_low: int = 10,
                         basis: str = "combinatorial", groups: Optional[Sequence[Sequence[int]]] = None,
                         positional: bool = True, gini: bool = True,
                         keep_masses: bool = False, item: int = 0,
                         edge_weights: str = "attention",
                         edges: Optional[Mapping[int, object]] = None, center: bool = False,
                         center_positions: Optional[Sequence[int]] = None
                         ) -> GraphSignalProfile:
    """Diagnostics of every layer of one forward pass.

    Args:
        attentions: per-layer attention, a sequence of ``[B, H, S, S]`` (or
            ``[H, S, S]``) tensors as Hugging Face returns them.
        hidden_states: ``L + 1`` hidden-state tensors ``[B, S, d]``; None for
            the attention-only metrics.
        layers: layer indices (default all).
        pairing: ``"input"`` pairs layer l with ``hidden_states[l]``, the
            states entering the block that produced the attention (Bao et
            al.); ``"output"`` with ``hidden_states[l + 1]``, as
            :meth:`~spectral_trust.SpectralModel.run_with_spectra` does.
        drop: positions removed from the graph and the signal (default the
            first, BOS); ``()`` keeps every token.
        renormalize: renormalise attention rows after dropping.
        hfer_cutoff, n_low, basis: see :func:`graph_signal_metrics`.
        groups: optional node groups (positions in the original sequence)
            merged by :func:`pool_nodes` before the graph metrics, e.g. one
            group per value and one per separator.
        positional, gini: compute the non-graph controls.
        keep_masses: keep the per-mode energies of each layer.
        item: batch item to read.
        edge_weights: the graph's edges, as ``GSPConfig.edge_weights``:
            ``"attention"`` (default, the paper's graph), ``"value"`` (the
            norm of what token j adds to token i's attention output, summed
            over heads inside the norm) or ``"value_per_head"``
            (``a^h_ij ||W_O^h v^h_j||`` per head, then averaged like
            attention). Value-weighted edges give little weight to a sink
            whose value vectors are small (Kobayashi et al. 2020).
        edges: with value edges, ``{layer: tensor}`` as returned by
            :func:`~spectral_trust.value_weighted_attention` (``[B, S, S]``
            for ``"value"``, ``[B, H, S, S]`` for ``"value_per_head"``,
            ``per_head=True``), from the same input; :func:`profile` computes
            them. Dropping and renormalising apply to them as to attention;
            ``gini`` and ``sink_mass`` always read the attention.
        center: subtract, in each layer, the mean hidden state over
            ``center_positions`` from the signal of every node before the
            graph Fourier transform (and the positional DCT), which removes
            the shared (DC) component; an ablation of Bao et al.'s updated
            paper. Off by default (results unchanged). To be aligned with
            the authors' code once it is public.
        center_positions: positions (in the original sequence) whose mean is
            subtracted, e.g. ``value_spans(...).numeric_positions`` for the
            value tokens; default every node of the graph (after ``drop``).
    """
    if edge_weights not in EDGE_WEIGHTS:
        raise ValueError(f"edge_weights must be one of {EDGE_WEIGHTS}, got {edge_weights!r}")
    if edge_weights != "attention" and edges is None:
        raise ValueError(f"edge_weights={edge_weights!r} needs edges= (from "
                         f"st.value_weighted_attention(model, ids, per_head="
                         f"{edge_weights == 'value_per_head'})) or numeric.profile")
    if pairing not in ("input", "output"):
        raise ValueError("pairing must be 'input' or 'output'")
    n = len(attentions)
    layers = list(range(n)) if layers is None else [int(l) % n for l in layers]
    rows: Dict[str, List] = {}
    masses = {} if keep_masses else None
    for l in layers:
        a = attentions[l]
        a = a[item] if a.dim() == 4 else a
        S = a.shape[-1]
        dropped = sorted({int(d) % S for d in drop})
        keep = [i for i in range(S) if i not in set(dropped)]
        W, Ap = attention_graph(a, dropped, renormalize)
        if edge_weights != "attention":
            W, _ = attention_graph(_layer_edges(edges, l, edge_weights, item), dropped,
                                   renormalize)
        X = None
        if hidden_states is not None:
            h = hidden_states[l if pairing == "input" else l + 1]
            h = h[item] if h.dim() == 3 else h
            X = _np(h)[keep]
            if center:
                X = X - _center_of(_np(h), keep, center_positions)
        Wg, Xg = W, X
        if groups is not None:
            pos = {p: j for j, p in enumerate(keep)}
            Wg, Xg = pool_nodes(W, X, [[pos[p] for p in g if p in pos] for g in groups])
        m = graph_signal_metrics(Wg, Xg, hfer_cutoff=hfer_cutoff, n_low=n_low, basis=basis,
                                 return_masses=keep_masses)
        if keep_masses and "masses" in m:
            masses[l] = m.pop("masses")
        if positional and Xg is not None:
            m.update(positional_metrics(Xg, hfer_cutoff))
        if gini:
            m["gini"] = attention_gini(Ap)
        m["sink_mass"] = sink_mass(a, 0)
        for k, v in m.items():
            rows.setdefault(k, []).append(v)
    per_layer = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    return GraphSignalProfile(layers, per_layer, masses)


@torch.no_grad()
def profile(model, input_ids, *, edge_weights: str = "attention", tokenizer=None,
            **kwargs) -> GraphSignalProfile:
    """:func:`profile_from_outputs` of one input, forward pass included.

    Args:
        model: a Hugging Face causal LM or a :class:`~spectral_trust.SpectralModel`,
            loaded with ``attn_implementation="eager"`` (attention weights are
            needed).
        input_ids: ``[S]`` or ``[1, S]``.
        edge_weights: ``"attention"``, ``"value"`` or ``"value_per_head"``;
            value edges come from
            :func:`~spectral_trust.value_weighted_attention` on the same input.
        **kwargs: passed to :func:`profile_from_outputs` (``drop``,
            ``renormalize``, ``basis``, ``layers``, ...).
    """
    from ..interventions._arch import unwrap
    if edge_weights not in EDGE_WEIGHTS:
        raise ValueError(f"edge_weights must be one of {EDGE_WEIGHTS}, got {edge_weights!r}")
    hf = unwrap(model)
    ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(1, -1)
    ids = ids.to(next(hf.parameters()).device)
    out = hf(input_ids=ids, output_attentions=True, output_hidden_states=True, use_cache=False)
    edges = None
    if edge_weights != "attention":
        from ..value_weighted import value_weighted_attention
        edges = value_weighted_attention(model, ids, per_head=edge_weights == "value_per_head",
                                         tokenizer=tokenizer)
    return profile_from_outputs(out.attentions, out.hidden_states, edge_weights=edge_weights,
                                edges=edges, **kwargs)
