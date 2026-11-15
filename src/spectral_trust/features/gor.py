# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""
Reference implementation of the features of "Geometry of Reason: Spectral
Signatures of Valid Reasoning in Transformers" (Noël, arXiv 2601.00791),
as its released code computed them.

The paper's code (github.com/vcnoel/geometry-of-reason,
``scripts/core/run_experiment.py``) ran ``GSPDiagnosticsFramework`` of
spectral-trust 0.1.2 with the ``GSPConfig`` defaults of that release, and
kept five values per layer: ``fiedler_value``, ``energy``, ``smoothness``,
``entropy`` (the hybrid spectral entropy of the hidden states on the graph)
and ``hfer``. :func:`geometry_of_reason` freezes that computation, so that
other work can reproduce the baseline exactly whatever later releases
change.

The defaults are those of the run (its saved ``gsp_results/config.json``):
``normalization="rw"``, ``eigen_solver="sparse"`` with
``num_eigenvalues=50``, ``hfer_cutoff_ratio=0.1``, symmetrised attention
averaged uniformly over heads. Two properties of these settings carry over:

- the random-walk Laplacian ``I − D⁻¹W`` is not symmetric, and 0.1.2
  diagonalised it with symmetric solvers (``scipy.linalg.eigh``, which reads
  one triangle, and ARPACK ``eigsh``); see the known-issue note in the
  changelog;
- above 50 tokens ``eigsh`` keeps 25 modes from each end of the spectrum
  and starts from a random vector, so the Fiedler value, entropy and HFER
  vary from run to run (energy and smoothness do not). ``arpack_seed`` fixes
  the start vector.

The paper's text describes the combinatorial Laplacian ``D − W`` and a
median HFER cutoff; that variant is ``normalization="none",
eigen_solver="dense", hfer_cutoff_ratio=0.5``.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

__all__ = ["geometry_of_reason", "geometry_of_reason_text", "GOR_FEATURES"]

#: The five features per layer, named as in the paper's results files.
GOR_FEATURES = ("fiedler_value", "energy", "smoothness", "entropy", "hfer")


def _symmetrize(a: torch.Tensor, method: str) -> torch.Tensor:
    if method == "symmetric":
        return 0.5 * (a + a.transpose(-2, -1))
    if method == "row_norm":
        n = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
    elif method == "col_norm":
        n = a / (a.sum(dim=-2, keepdim=True) + 1e-8)
    else:
        raise ValueError(f"Unknown symmetrization method: {method}")
    return 0.5 * (n + n.transpose(-2, -1))


def _aggregate(a: torch.Tensor, method: str, mask: Optional[torch.Tensor]) -> torch.Tensor:
    B, H = a.shape[:2]
    if method in ("uniform", "learnable"):          # 0.1.2: "learnable" fell back to uniform
        w = a.new_full((B, H), 1.0 / H)
    elif method == "attention_weighted":
        A = a
        if mask is not None:
            keep = mask.unsqueeze(1).unsqueeze(-1) & mask.unsqueeze(1).unsqueeze(2) \
                if mask.dim() == 2 else mask
            A = A.masked_fill(~keep, 0.0)
        m = A.sum(dim=(2, 3))
        w = m / (m.sum(dim=1, keepdim=True) + 1e-8)
    else:
        raise ValueError(f"Unknown head aggregation method: {method}")
    return (a * w.view(B, H, 1, 1)).sum(dim=1)


def _laplacian(adj: torch.Tensor, normalization: str, remove_self_loops: bool) -> torch.Tensor:
    adj = torch.clamp(adj, min=0)
    if remove_self_loops:
        adj = adj.masked_fill(torch.eye(adj.shape[-1], device=adj.device).bool(), 0.0)
    deg = adj.sum(dim=-1)
    eye = torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
    if normalization == "rw":
        inv = torch.where(deg > 1e-8, 1.0 / deg, torch.zeros_like(deg))
        return eye - torch.matmul(torch.diag_embed(inv), adj)
    if normalization == "sym":
        inv = torch.where(deg > 1e-8, 1.0 / torch.sqrt(deg), torch.zeros_like(deg))
        d = torch.diag_embed(inv)
        return eye - torch.matmul(torch.matmul(d, adj), d)
    if normalization == "none":
        return torch.diag_embed(deg) - adj
    raise ValueError(f"Unknown normalization method: {normalization}")


def _eig(L: np.ndarray, solver: str, k: int, max_iter: int, v0: Optional[np.ndarray]):
    """0.1.2 ``SpectralAnalyzer.compute_eigendecomposition``."""
    n = L.shape[0]
    if solver == "sparse" and n > 50:
        try:
            Ls = csr_matrix(L)
            k_small = min(k // 2, n - 2)
            k_large = min(k - k_small, n - k_small - 1)
            kw = {} if v0 is None else {"v0": v0}
            if k_small > 0:
                ws, vs = eigsh(Ls, k=k_small, which="SM", maxiter=max_iter, tol=1e-6, **kw)
            else:
                ws, vs = np.array([]), np.zeros((n, 0))
            if k_large > 0:
                wl, vl = eigsh(Ls, k=k_large, which="LM", maxiter=max_iter, tol=1e-6, **kw)
            else:
                wl, vl = np.array([]), np.zeros((n, 0))
            w, v = np.concatenate([ws, wl]), np.concatenate([vs, vl], axis=1)
            order = np.argsort(w)
            w, v = w[order], v[:, order]
        except ArpackNoConvergence:
            w, v = eigh(L)
    else:
        w, v = eigh(L)
    return np.maximum(w, 0), v


def _layer(signals: np.ndarray, L: np.ndarray, cutoff: float, solver: str, k: int,
           max_iter: int, v0: Optional[np.ndarray]) -> Dict[str, float]:
    """0.1.2 ``SpectralAnalyzer.analyze_layer``, for the five features."""
    _eig(L, solver, k, max_iter, v0)          # 0.1.2 ran it once more for its connectivity flag
    w, v = _eig(L, solver, k, max_iter, v0)
    energy = float(np.trace(np.dot(signals.T, np.dot(L, signals))))
    norm = np.trace(np.dot(signals.T, signals))
    smooth = 0.0 if norm < 1e-8 else energy / norm
    e = np.sum(np.dot(v.T, signals) ** 2, axis=1)
    total = np.sum(e)
    if total < 1e-8:
        entropy, hfer = 0.0, 0.0
    else:
        p = np.maximum(e / total, 1e-12)
        entropy = float(-np.sum(p * np.log(p)))
        hfer = float(np.sum(e[int((1 - cutoff) * len(w)):]) / total)
    return {"fiedler_value": float(w[1]) if len(w) > 1 else 0.0, "energy": energy,
            "smoothness": float(smooth), "entropy": entropy, "hfer": hfer}


def geometry_of_reason(attentions: Sequence, hidden_states: Sequence, *,
                       normalization: str = "rw", hfer_cutoff_ratio: float = 0.1,
                       eigen_solver: str = "sparse", num_eigenvalues: int = 50,
                       lanczos_max_iter: int = 1000, symmetrization: str = "symmetric",
                       head_aggregation: str = "uniform", remove_self_loops: bool = False,
                       attention_mask=None, arpack_seed: Optional[int] = None
                       ) -> Dict[str, np.ndarray]:
    """The Geometry of Reason features of one text, per layer, computed as
    the paper's released code computed them (spectral-trust 0.1.2,
    ``GSPDiagnosticsFramework.analyze_text`` with its default ``GSPConfig``).

    Args:
        attentions: one ``[1, heads, S, S]`` (or ``[heads, S, S]``) tensor
            per layer, as a Hugging Face model returns with
            ``output_attentions=True``. Their dtype is kept up to the
            Laplacian, as in 0.1.2.
        hidden_states: the ``layers + 1`` states ``[1, S, d]`` (or
            ``[S, d]``) that ``output_hidden_states=True`` returns, embeddings
            first; layer ``l`` is measured on ``hidden_states[l + 1]``.
        normalization, hfer_cutoff_ratio, eigen_solver, num_eigenvalues,
        lanczos_max_iter, symmetrization, head_aggregation,
        remove_self_loops: the 0.1.2 ``GSPConfig`` fields, at the values of
            the paper's run.
        attention_mask: the tokenizer's ``[1, S]`` mask; it only matters for
            ``head_aggregation="attention_weighted"``.
        arpack_seed: None reproduces 0.1.2, whose ARPACK calls (sparse
            solver, more than 50 tokens) start from a random vector; an int
            fixes that start vector, so repeated runs agree.

    Returns:
        ``fiedler_value``, ``energy``, ``smoothness``, ``entropy`` and
        ``hfer`` (:data:`GOR_FEATURES`), one float64 array ``[layers]`` each.
    """
    if len(hidden_states) != len(attentions) + 1:
        raise ValueError(f"{len(attentions)} attention tensors need {len(attentions) + 1} "
                         f"hidden states (embeddings first), got {len(hidden_states)}")
    mask = None if attention_mask is None else torch.as_tensor(attention_mask).to(torch.bool)
    out = {k: np.empty(len(attentions)) for k in GOR_FEATURES}
    for l, att in enumerate(attentions):
        a = torch.as_tensor(att).detach()
        a = a.squeeze(0) if a.dim() == 4 else a                        # [H, S, S]
        x = torch.as_tensor(hidden_states[l + 1]).detach()
        x = x.squeeze(0) if x.dim() == 3 else x                        # [S, d]
        sym = _symmetrize(a.unsqueeze(0), symmetrization).squeeze(0)
        adj = _aggregate(sym.unsqueeze(0), head_aggregation, mask).squeeze(0)
        lap = _laplacian(adj.unsqueeze(0), normalization, remove_self_loops).squeeze(0)
        L = lap.cpu().numpy().squeeze().astype(np.float32)
        X = x.cpu().numpy().astype(np.float32)
        v0 = None if arpack_seed is None else \
            np.random.default_rng(arpack_seed).uniform(-1, 1, L.shape[0]).astype(L.dtype)
        row = _layer(X, L, hfer_cutoff_ratio, eigen_solver, num_eigenvalues,
                     lanczos_max_iter, v0)
        for k in GOR_FEATURES:
            out[k][l] = row[k]
    return out


@torch.no_grad()
def geometry_of_reason_text(model, tokenizer, text: str, max_length: int = 512,
                            **kwargs) -> Dict[str, np.ndarray]:
    """:func:`geometry_of_reason` of ``text``, tokenised and run as 0.1.2's
    ``LLMInstrumenter.process_text`` did (special tokens, truncation to
    ``max_length`` tokens). Load the model with
    ``attn_implementation="eager"`` so that it returns attention weights."""
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True,
                    padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_attentions=True, output_hidden_states=True)
    return geometry_of_reason(out.attentions, out.hidden_states,
                              attention_mask=enc.get("attention_mask"), **kwargs)
