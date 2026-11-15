# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
An attention-graph edit next to its matched controls.

:func:`edit_with_controls` applies one edit along the token axis and, at the
same strengths, controls whose edit has the same Frobenius norm on every
item, then reads the model's output with a user readout and reports the
readout's change with bootstrap intervals.

Edits:

- ``"graph_filter"``: :func:`~spectral_trust.interventions.graph_filter` on
  the residual stream after block ``layer`` (``resid_post``), on the graph of
  that layer's attention: ``X' = X + s (g(L) X - X)`` at ``positions``;
- ``"band_patch"``: frequency-resolved patching at ``resid_pre`` of block
  ``layer``: ``X' = X + s U_b U_b^T (X_source - X)``, ``U_b`` a band of
  ``modes`` of the source run's attention-graph eigenbasis (as
  :class:`~spectral_trust.interventions.Patcher` does at ``s = 1``).

Controls, each rescaled per item to the target edit's norm:

- ``"permuted"``: the same operation on the attention graph with its nodes
  permuted at random (position 0, usually the sink, kept in place): same
  edge weights, unrelated token structure;
- ``"path"``: the same operation on the chain graph over positions (a
  DCT-like, purely positional basis);
- ``"random"`` (``band_patch`` only): a Haar-random subspace of the band's
  dimension;
- ``"scaling"``: no graph at all: ``-X`` (shrinking the residual) for a
  filter, ``X_source - X`` (plain patching, all positions) for a patch.

Every edit is computed from the unedited activation of the item (one capture
pass with eager attention), which is what a live hook at that site would
read, and added in the edited passes only when the call has the prompt's
length, so readouts that continue the prompt with a key-value cache (as
:func:`~spectral_trust.numeric.next_value_distribution` does) are edited on
the prompt and not on their continuation steps. New after 0.5.0
(unreleased).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..config import GSPConfig
from ._arch import Arch, unwrap
from ._hooks import register_site_hook
from .filters import FilterSpec, apply_filter, attention_adjacency, laplacian, path_adjacency
from .patching import haar_basis

EDITS = ("graph_filter", "band_patch")
CONTROLS = ("permuted", "path", "random", "scaling")

Readout = Callable[[object, torch.Tensor], Dict[str, float]]


@dataclass
class EditControlResult:
    """Readouts of :func:`edit_with_controls`.

    Attributes:
        conditions: ``"target"`` and the controls, in order.
        strengths: the strengths run.
        baseline: per-item readout of the unedited model.
        edited: ``(condition, strength) -> per-item readouts``.
        relative_edit: ``strength -> per-item ||Delta|| / ||X||`` over the
            edited positions (equal across conditions by construction).
        skipped: strengths left out by ``max_relative_edit``.
    """
    conditions: List[str]
    strengths: List[float]
    baseline: List[Dict[str, float]]
    edited: Dict[Tuple[str, float], List[Dict[str, float]]] = field(default_factory=dict)
    relative_edit: Dict[float, List[float]] = field(default_factory=dict)
    skipped: List[float] = field(default_factory=list)

    def values(self, metric: str, condition: Optional[str] = None,
               strength: Optional[float] = None) -> np.ndarray:
        """Per-item readout ``metric`` (baseline when ``condition`` is None)."""
        rows = self.baseline if condition is None else self.edited[(condition, strength)]
        return np.array([r[metric] for r in rows], dtype=np.float64)

    def deltas(self, metric: str, condition: str, strength: float) -> np.ndarray:
        return self.values(metric, condition, strength) - self.values(metric)

    def summary(self, metrics: Optional[Sequence[str]] = None, n_boot: int = 1000,
                ci: float = 0.95, seed: int = 0) -> List[Dict[str, float]]:
        """One row per (condition, strength, metric): mean change from the
        baseline with its interval, and for controls the paired difference
        target minus control with its interval."""
        from ..audit import bootstrap_ci
        metrics = list(metrics or self.baseline[0].keys())
        out = []
        for s in self.strengths:
            for cond in self.conditions:
                for m in metrics:
                    d = self.deltas(m, cond, s)
                    est, lo, hi = bootstrap_ci(d, n_boot=n_boot, ci=ci, seed=seed)
                    row = {"condition": cond, "strength": s, "metric": m, "n": len(d),
                           "delta": est, "delta_lo": lo, "delta_hi": hi,
                           "relative_edit": float(np.mean(self.relative_edit[s]))}
                    if cond != "target":
                        diff = self.deltas(m, "target", s) - d
                        e2, l2, h2 = bootstrap_ci(diff, n_boot=n_boot, ci=ci, seed=seed)
                        row.update({"target_minus": e2, "target_minus_lo": l2,
                                    "target_minus_hi": h2})
                    out.append(row)
        return out

    def to_dict(self) -> Dict:
        return {"conditions": self.conditions, "strengths": self.strengths,
                "baseline": self.baseline, "skipped": self.skipped,
                "edited": {f"{c}@{s}": v for (c, s), v in self.edited.items()},
                "relative_edit": {str(s): v for s, v in self.relative_edit.items()}}


def _ids(x) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.long)
    return t.unsqueeze(0) if t.dim() == 1 else t


def _capture(arch: Arch, model, ids: torch.Tensor, site: str, layer: int):
    """``(X [1, S, d], A [1, H, S, S])`` at ``site`` and the layer's
    attention, from one eager forward pass."""
    store = {}
    orig = arch.attn_implementation()
    if orig not in (None, "eager"):
        arch.set_attn_implementation("eager")

    def grab_w(module, args, out):
        w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        if w is None:
            raise RuntimeError("the attention returned no weights (eager attention is needed)")
        store["A"] = w.detach()

    def grab_x(x):
        store["X"] = x.detach().clone()

    handles = [arch.attention(layer).register_forward_hook(grab_w)]
    handles += register_site_hook(arch, site, layer, grab_x)
    try:
        with torch.no_grad():
            model(input_ids=ids.to(next(model.parameters()).device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        if orig not in (None, "eager"):
            arch.set_attn_implementation(orig)
    return store["X"], store["A"]


def _permutation(S: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(S - 1, generator=g)])


def _permuted_adjacency(A: torch.Tensor, config: GSPConfig, seed: int) -> torch.Tensor:
    W = attention_adjacency(A.to(torch.float64), config, dtype=torch.float64)[0]
    p = _permutation(W.shape[-1], seed)
    return W[p][:, p]


def _band_basis(W: torch.Tensor, config: GSPConfig) -> torch.Tensor:
    L = laplacian(W.unsqueeze(0).to(torch.float32), config)[0].to(torch.float64)
    _, U = torch.linalg.eigh(0.5 * (L + L.T))
    return U


def _unit_edits(edit, conds, X, A, rows, *, kind, cutoff, band, causal, order, config, seed,
                Xs=None, As=None, modes=None) -> Dict[str, torch.Tensor]:
    """Unit-strength edit of each condition at ``rows``: ``[len(rows), d]``
    (float64)."""
    X64 = X.to(torch.float64)
    S = X.shape[1]
    out = {}
    if edit == "graph_filter":
        graphs = {"target": "attention", "path": "path"}
        if "permuted" in conds:
            graphs["permuted"] = _permuted_adjacency(A, config, seed)
        for c in ["target"] + [c for c in conds if c in ("permuted", "path")]:
            spec = FilterSpec(kind, cutoff, band, 1.0, graphs[c], causal, order, "chebyshev",
                              config, "float64")
            f = apply_filter(spec, X64, A.to(torch.float64), rows=rows if causal else None)
            f = f[0] if causal else f[0, rows]
            out[c] = f - X64[0, rows]
        if "scaling" in conds:
            out["scaling"] = -X64[0, rows]
        return out
    # band patch
    diff = (Xs.to(torch.float64) - X64)[0]                    # [S, d]
    lo, hi = modes
    bases = {"target": _band_basis(attention_adjacency(As.to(torch.float64), config,
                                                       dtype=torch.float64)[0], config)}
    if "permuted" in conds:
        bases["permuted"] = _band_basis(_permuted_adjacency(As, config, seed), config)
    if "path" in conds:
        bases["path"] = _band_basis(path_adjacency(S, 1, dtype=torch.float64)[0], config)
    if "random" in conds:
        bases["random"] = haar_basis(S, seed)
    for c, U in bases.items():
        Ub = U[:, lo:hi] if c != "random" else U[:, :hi - lo]
        out[c] = (Ub @ (Ub.T @ diff))[rows]
    if "scaling" in conds:
        out["scaling"] = diff[rows]
    return out


def edit_with_controls(model, inputs: Sequence, readout: Readout, *, layer: int,
                       edit: str = "graph_filter", kind: str = "lowpass",
                       cutoff: Optional[float] = 1.0, band=None, causal: bool = True,
                       order: int = 64, strengths: Sequence[float] = (0.25, 0.5, 1.0),
                       controls: Sequence[str] = ("permuted", "path", "scaling"),
                       positions: Union[None, Sequence[int], Callable[[int], Sequence[int]],
                                        Sequence[Sequence[int]]] = None,
                       source: Optional[Sequence] = None, modes: Optional[Tuple[int, int]] = None,
                       token_span: Optional[Tuple[Optional[int], Optional[int]]] = None,
                       config: Optional[GSPConfig] = None, seed: int = 0,
                       max_relative_edit: Optional[float] = None,
                       progress: bool = False) -> EditControlResult:
    """Run an edit and its norm-matched controls over ``inputs``.

    Args:
        model: a Hugging Face causal LM or a :class:`~spectral_trust.SpectralModel`.
        inputs: token-id sequences, run one at a time.
        readout: ``readout(model, ids [1, S]) -> {name: value}``, called with
            the edit installed (for example the summary of
            :func:`~spectral_trust.numeric.next_value_distribution`).
        layer: block index.
        edit: ``"graph_filter"`` or ``"band_patch"``.
        kind, cutoff, band, causal, order: the filter (``graph_filter``), as
            in :meth:`ResidSite.graph_filter`.
        strengths: ``s`` values; 0 is the identity.
        controls: any of ``"permuted"``, ``"path"``, ``"random"`` (band
            patch only) and ``"scaling"``.
        positions: positions edited (others are read by the filter but left
            unchanged): one list for every item, a list per item, or a
            function of the item index. Default all positions.
        source: ``band_patch``: the source (clean) sequences, each of the
            same length as its input.
        modes: ``band_patch``: ``(lo, hi)`` mode indices of the band, in
            ascending eigenvalue order.
        token_span: ``(start, end)`` (slice semantics, non-negative): only
            these tokens form the graph, and positions outside it are
            neither read nor edited. ``(1, None)`` leaves out the first
            token, whose residual norm is often orders of magnitude above
            the others' (an attention sink); without it a low-pass filter
            spreads that token into every position.
        config: graph construction (default the SpectralModel's or the
            library defaults; the band patch needs ``"sym"`` or ``"none"``).
        seed: seed of the permutation and the random subspace (the item index
            is added).
        max_relative_edit: strengths whose mean relative edit exceeds this
            are skipped (for example 0.3, the side-effect guard of the
            graph-filter use cases).
    """
    from ..model import SpectralModel, default_config
    if edit not in EDITS:
        raise ValueError(f"edit must be one of {EDITS}, got {edit!r}")
    bad = [c for c in controls if c not in CONTROLS]
    if bad:
        raise ValueError(f"unknown controls {bad}; use {CONTROLS}")
    if edit == "graph_filter" and "random" in controls:
        raise ValueError("the 'random' control is a subspace; it applies to band_patch")
    if edit == "band_patch" and (source is None or modes is None):
        raise ValueError("band_patch needs source= and modes=(lo, hi)")
    if config is None:
        config = model.config if isinstance(model, SpectralModel) else default_config()
    if edit == "band_patch" and config.normalization not in ("sym", "none"):
        raise ValueError("band_patch needs a symmetric Laplacian (normalization 'sym' or 'none')")
    hf = unwrap(model)
    arch = Arch(model)
    site = "resid_post" if edit == "graph_filter" else "resid_pre"
    conds = ["target"] + [c for c in controls]
    strengths = [float(s) for s in strengths]
    res = EditControlResult(conds, strengths, [])
    per_item: List[Dict[str, torch.Tensor]] = []
    per_rows: List[List[int]] = []
    norms: List[float] = []
    items = [_ids(x) for x in inputs]
    for i, ids in enumerate(items):
        S = ids.shape[1]
        if positions is None:
            rows = list(range(S))
        elif callable(positions):
            rows = list(positions(i))
        elif len(positions) and isinstance(positions[0], (list, tuple, np.ndarray)):
            rows = list(positions[i])
        else:
            rows = list(positions)
        rows = sorted(int(r) % S for r in rows)
        s0, e0 = (0, S) if token_span is None else slice(*token_span).indices(S)[:2]
        if any(not s0 <= r < e0 for r in rows):
            raise ValueError(f"item {i}: edited positions must lie in token_span {(s0, e0)}")
        X, A = _capture(arch, hf, ids, site, layer)
        Xs = As = None
        if edit == "band_patch":
            sids = _ids(source[i])
            if sids.shape != ids.shape:
                raise ValueError(f"source {i} has shape {tuple(sids.shape)}, input "
                                 f"{tuple(ids.shape)}")
            Xs, As = _capture(arch, hf, sids, site, layer)
        sub = slice(s0, e0)
        unit = _unit_edits(edit, conds, X[:, sub].cpu(), A[:, :, sub, sub].cpu(),
                           [r - s0 for r in rows], kind=kind, cutoff=cutoff,
                           band=band, causal=causal, order=order, config=config,
                           seed=seed + i, Xs=None if Xs is None else Xs[:, sub].cpu(),
                           As=None if As is None else As[:, :, sub, sub].cpu(), modes=modes)
        per_item.append(unit)
        per_rows.append(rows)
        norms.append(float(X[0, rows].to(torch.float64).norm()))
        with torch.no_grad():
            res.baseline.append(readout(model, ids))
        if progress:
            print(f"item {i + 1}/{len(items)} captured", flush=True)
    for s in strengths:
        rel = [s * float(u["target"].norm()) / max(n, 1e-30) for u, n in zip(per_item, norms)]
        if max_relative_edit is not None and np.mean(rel) > max_relative_edit:
            res.skipped.append(s)
            continue
        res.relative_edit[s] = rel
        for cond in conds:
            rows_out = []
            for i, ids in enumerate(items):
                u = per_item[i]
                target_norm = s * float(u["target"].norm())
                d = u[cond]
                dn = float(d.norm())
                delta = d * (target_norm / dn if dn > 0 else 0.0) if cond != "target" else s * d
                rows_out.append(_run_with_delta(arch, model, ids, site, layer, per_rows[i],
                                                delta, readout))
            res.edited[(cond, s)] = rows_out
            if progress:
                print(f"strength {s} {cond} done", flush=True)
    res.strengths = [s for s in strengths if s not in res.skipped]
    return res


def _run_with_delta(arch, model, ids, site, layer, rows, delta, readout):
    S = ids.shape[1]
    idx = torch.as_tensor(rows, dtype=torch.long)

    def fn(x):
        if x.shape[1] != S or x.shape[0] != 1:
            return None
        y = x.clone()
        y[0, idx.to(x.device)] += delta.to(device=x.device, dtype=x.dtype)
        return y

    handles = register_site_hook(arch, site, layer, fn)
    try:
        with torch.no_grad():
            return readout(model, ids)
    finally:
        for h in handles:
            h.remove()
