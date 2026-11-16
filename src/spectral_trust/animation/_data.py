# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
What the animated views read from a :class:`~spectral_trust.cache.SpectralCache`:
each layer's attention graph and its Laplacian eigenpairs (built exactly as
:func:`spectral_trust.viz.graph` builds them), Laplacian eigenmaps aligned
across layers, and the per-layer spectral masses.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class LayerGraph:
    layer: int
    W: np.ndarray            # [S, S] the measured layer graph, or one head's symmetrised attention
    eigenvalues: np.ndarray  # [S] ascending
    vectors: np.ndarray      # [S, S] eigenvectors, column k for eigenvalue k
    fiedler: np.ndarray      # [S] Fiedler vector (library sign convention)
    tokens: List[str]
    offset: int

    @property
    def lambda2(self) -> float:
        return float(self.eigenvalues[1]) if self.eigenvalues.size > 1 else 0.0

    def normalized(self) -> np.ndarray:
        """``D^-1/2 W D^-1/2`` without the diagonal: the edge weights the
        normalised Laplacian sees, so a sink does not swamp every edge."""
        W = self.W.copy()
        np.fill_diagonal(W, 0.0)
        d = W.sum(1)
        inv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
        return W * inv[:, None] * inv[None, :]


def single(cache, text: int = 0):
    return cache.text(text) if cache.is_batch else cache


def pick_layers(cache, layers) -> List[int]:
    have = cache.layers
    if layers is None:
        return list(have)
    if isinstance(layers, (int, np.integer)):
        layers = [int(layers)]
    if isinstance(layers, slice):
        return list(have[layers])
    n = cache.meta.get("n_layers") or (have[-1] + 1 if have else 0)
    out = []
    for lay in layers:
        lay = int(lay)
        if lay < 0:
            lay += int(n)
        if lay not in have:
            raise KeyError(f"layer {lay} is not in the cache; layers present: {have}")
        if lay not in out:
            out.append(lay)
    return sorted(out)


def layer_graph(cache, layer: int, head: Optional[int] = None) -> LayerGraph:
    from ..viz import _graph_data

    W, eig, u, tokens, offset, vecs = _graph_data(cache, layer, head, vectors=True)
    return LayerGraph(layer, W, eig, vecs, np.asarray(u, dtype=np.float64), list(tokens), offset)


def layer_graphs(cache, layers: Sequence[int], head: Optional[int] = None) -> List[LayerGraph]:
    return [layer_graph(cache, lay, head) for lay in layers]


# ── eigenmaps and alignment ─────────────────────────────────────────────────

def eigenmap(g: LayerGraph, dims: int = 3) -> np.ndarray:
    """Tokens in ``dims`` dimensions by Laplacian eigenvectors 2 to
    ``dims + 1``, centred and scaled to unit RMS radius."""
    V = g.vectors[:, 1:1 + dims]
    if V.shape[1] < dims:
        V = np.concatenate([V, np.zeros((V.shape[0], dims - V.shape[1]))], axis=1)
    V = V - V.mean(0)
    rms = np.sqrt(np.mean(np.sum(V ** 2, axis=1)))
    return V / rms if rms > 0 else V


def procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The orthogonal ``R`` minimising ``||X R - Y||_F``."""
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return U @ Vt


def fix_signs(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip each column of ``X`` that points away from the same column of ``ref``."""
    s = np.sign(np.sum(X * ref, axis=0))
    s[s == 0] = 1.0
    return X * s


def align_sequence(Ys: Sequence[np.ndarray], mode: str = "procrustes") -> List[np.ndarray]:
    """Align each layer's eigenmap to the previous (already aligned) one.

    ``"procrustes"``: column signs fixed, then the orthogonal Procrustes
    rotation; ``"sign"``: column signs only; ``"none"``: as computed.
    Eigenvectors are defined up to sign (and up to rotation inside a
    repeated eigenvalue), so without alignment successive layers jump.
    """
    if mode not in ("procrustes", "sign", "none"):
        raise ValueError("align must be 'procrustes', 'sign' or 'none'")
    out = [np.asarray(Ys[0], dtype=float)]
    for Y in Ys[1:]:
        Y = np.asarray(Y, dtype=float)
        if mode != "none":
            Y = fix_signs(Y, out[-1])
        if mode == "procrustes":
            Y = Y @ procrustes(Y, out[-1])
        out.append(Y)
    return out


def frame_displacement(Ys: Sequence[np.ndarray]) -> float:
    """Mean distance a token travels between successive layers."""
    if len(Ys) < 2:
        return 0.0
    return float(np.mean([np.mean(np.linalg.norm(b - a, axis=1))
                          for a, b in zip(Ys[:-1], Ys[1:])]))


def principal_frame(Ys: Sequence[np.ndarray]) -> np.ndarray:
    """One rotation for every layer that puts the largest spread of the
    whole sequence on x and the smallest on z (wide frames, a flat floor)."""
    P = np.concatenate(list(Ys), axis=0)
    P = P - P.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


def match_signs(vectors: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Flip each vector to agree with the previous one: the Fiedler split of
    a layer is the same with either sign, so this keeps a token's colour
    from swapping when only the arbitrary sign changed."""
    out = [np.asarray(vectors[0], dtype=float)]
    for v in vectors[1:]:
        v = np.asarray(v, dtype=float)
        out.append(-v if float(v @ out[-1]) < 0 else v)
    return out


# ── signals ─────────────────────────────────────────────────────────────────

def spectral_masses(cache, layers: Sequence[int]) -> np.ndarray:
    """Signal energy per graph-frequency mode (ascending eigenvalue),
    normalised to sum to one in each layer: ``[layers, modes]``."""
    rows = []
    for lay in layers:
        key = f"blocks.{lay}.graph.spectral_masses"
        if key not in cache:
            raise KeyError(f"{key!r} is not in the cache; the waterfall needs the hybrid "
                           "graph metrics (run_with_spectra(..., hybrid=True))")
        m = np.asarray(cache[key], dtype=np.float64)
        s = m.sum()
        rows.append(m / s if s > 0 else m)
    return np.stack(rows)


def graph_series(cache, name: str, layers: Sequence[int]) -> np.ndarray:
    out = []
    for lay in layers:
        v = cache.get(f"blocks.{lay}.graph.{name}")
        out.append(np.nan if v is None else float(np.asarray(v)))
    return np.asarray(out, dtype=float)


def model_label(cache) -> str:
    name = cache.model_name or ""
    return name.split("/")[-1] if name else "model"


def prompt_label(cache, limit: int = 90) -> str:
    t = cache.texts if not cache.is_batch else cache.texts[0]
    if not t:
        t = "".join(cache.tokens if not cache.is_batch else cache.tokens[0])
    t = " ".join(str(t).split())
    return t if len(t) <= limit else t[:limit - 1] + "…"
