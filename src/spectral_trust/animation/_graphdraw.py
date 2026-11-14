# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Token graphs drawn flat: circle positions, edges by weight, nodes by the
Fiedler vector and labels around the circle (shared by ``compare``,
``graph_movie`` and ``head_atlas``)."""
import math
from typing import Optional, Sequence

import numpy as np

from ._core import Canvas, Theme, hex_rgb, ramp, rgba


def circle_positions(n: int, cx: float, cy: float, R: float) -> np.ndarray:
    """``n`` points clockwise from the top (the layout of ``viz.graph``)."""
    th = np.pi / 2 + np.pi / max(n, 1) - 2 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([cx + R * np.cos(th), cy - R * np.sin(th)], axis=1)


def fiedler_colours(th: Theme, u: np.ndarray) -> np.ndarray:
    """Node RGB on the library's diverging scale: negative side, neutral,
    positive side, relative to the largest entry."""
    m = float(np.max(np.abs(u))) if u.size else 1.0
    return ramp([th.neg, th.node_mid, th.pos], 0.5 + 0.5 * u / (m or 1.0))


def draw_edges(cv: Canvas, th: Theme, pos: np.ndarray, W: np.ndarray, *,
               idx: Optional[Sequence[int]] = None, alert: Optional[np.ndarray] = None,
               min_rel: float = 0.05, width: float = 3.6, alpha: float = 0.75,
               progress: Optional[np.ndarray] = None, z: float = 3.0, scale_ref=None):
    """Chords between nodes, opacity and width by weight relative to the
    largest (or to ``scale_ref``), weakest first. ``alert[i, j]`` draws an
    edge in the alert colour; ``progress[k]`` in [0, 1] draws the k-th
    weakest edge partly, as it is being drawn in."""
    n = W.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    iu, ju = np.triu_indices(n, 1)
    w = W[iu, ju]
    ref = float(scale_ref if scale_ref is not None else (w.max() if w.size else 0)) or 1.0
    rel = np.clip(w / ref, 0, 1)
    keep = rel > min_rel
    order = np.argsort(rel[keep], kind="stable")
    a, b, r = iu[keep][order], ju[keep][order], rel[keep][order]
    if not len(a):
        return
    p0, p1 = pos[idx[a]], pos[idx[b]]
    if progress is not None:
        pr = np.asarray(progress)[:len(a)]
        show = pr > 0
        p1 = p0 + (p1 - p0) * pr[:, None]
        a, b, r, p0, p1 = a[show], b[show], r[show], p0[show], p1[show]
        if not len(a):
            return
    base = hex_rgb(th.edge)
    cols = np.repeat(base[None], len(a), 0)
    if alert is not None:
        hot = alert[a, b] | alert[b, a]
        cols[hot] = hex_rgb(th.alert)
        r = np.where(hot, np.maximum(r, 0.25), r)
    segs = np.stack([p0, p1], 1)
    cv.lines(segs, rgba(cols, (0.07 + alpha * r ** 1.15)),
             cv.px(0.6, 0.45) + cv.px(width, 1.6) * r, z=z)


def draw_nodes(cv: Canvas, th: Theme, pos: np.ndarray, colours: np.ndarray, r: float, *,
               ring: Optional[np.ndarray] = None, ghost: Optional[np.ndarray] = None,
               z: float = 5.0):
    """Filled circles with the theme's outline; ``ring`` marks nodes with an
    alert-coloured ring, ``ghost`` draws hollow placeholders."""
    n = len(pos)
    ghost = np.zeros(n, bool) if ghost is None else np.asarray(ghost, bool)
    if np.any(ghost):
        cv.circles(pos[ghost], r * 0.55, [th.bg], edge=th.rule, lw=cv.px(1.2, 0.8), z=z)
    real = ~ghost
    if ring is not None and np.any(ring & real):
        m = ring & real
        cv.circles(pos[m], r + cv.px(4, 2.5), [th.bg], edge=th.alert, lw=cv.px(2.4, 1.6), z=z)
    if np.any(real):
        cv.circles(pos[real], r, rgba(colours[real], 1.0), edge=th.node_line,
                   lw=cv.px(1.3, 0.9), z=z + 0.1)


def draw_circle_labels(cv: Canvas, th: Theme, pos: np.ndarray, centre, R: float, r: float,
                       tokens: Sequence[str], size: float, *, colours=None, alphas=None,
                       radial: Optional[bool] = None, weight: str = "medium", z: float = 20):
    """Labels just outside the circle: horizontal for short sequences,
    along the radius (reading left to right) for long ones, so that they
    never overlap one another or a node."""
    n = len(tokens)
    if radial is None:
        radial = n > 16
    cx, cy = centre
    gap = r + cv.px(7, 4)
    for i in range(n):
        if alphas is not None and alphas[i] <= 0.01:
            continue
        dx, dy = (pos[i, 0] - cx) / R, (pos[i, 1] - cy) / R
        x, y = cx + (R + gap) * dx, cy + (R + gap) * dy
        col = colours[i] if colours is not None else th.ink2
        a = 1.0 if alphas is None else float(alphas[i])
        if radial:
            ang = math.degrees(math.atan2(-dy, dx))
            if dx < -1e-9:
                cv.text(x, y, tokens[i], size, col, ha="right", va="center", rotation=ang + 180,
                        weight=weight, alpha=a, halo=cv.px(2.5, 1.5), z=z)
            else:
                cv.text(x, y, tokens[i], size, col, ha="left", va="center", rotation=ang,
                        weight=weight, alpha=a, halo=cv.px(2.5, 1.5), z=z)
        else:
            ha = "left" if dx > 0.25 else "right" if dx < -0.25 else "center"
            va = "top" if dy > 0.3 else "bottom" if dy < -0.3 else "center"
            cv.text(x, y, tokens[i], size, col, ha=ha, va=va, weight=weight, alpha=a,
                    halo=cv.px(2.5, 1.5), z=z)


def label_room(cv: Canvas, tokens: Sequence[str], size: float, radial: bool) -> float:
    """Pixels the labels need outside the circle."""
    if not tokens:
        return 0.0
    if radial:
        return max(cv.text_size(t, size, "medium")[0] for t in tokens) + cv.px(10, 6)
    w = max(cv.text_size(t, size, "medium")[0] for t in tokens)
    return max(w * 0.85, size * 2) + cv.px(10, 6)
