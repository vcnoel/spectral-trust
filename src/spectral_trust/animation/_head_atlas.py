# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""``head_atlas``: every head's graph in one layer as small multiples,
sorted by a metric, swept over the layers (tiles slide to their new rank)."""
import math
from typing import Optional

import numpy as np

from ._core import (
    Canvas, display_token, hex_rgb, layer_positions, plotly_layout, plotly_player,
    split_position, write_plotly_html,
)
from ._data import layer_graph, match_signs, model_label, pick_layers, prompt_label, single
from ._graphdraw import circle_positions, draw_edges, draw_nodes, fiedler_colours
from ._scene import Scene

SHORT = {"fiedler_value": "λ₂", "hfer": "HFER", "spectral_entropy_norm": "entropy",
         "connectivity_ratio": "connectivity", "spectral_radius": "ρ", "energy": "energy",
         "smoothness_index": "smoothness", "spectral_entropy": "entropy"}


def _qualify(cache, metric: str) -> str:
    from ..cache import NAMESPACE_METRICS

    if "." in metric:
        q = metric
    elif metric in NAMESPACE_METRICS["attn"]:
        q = f"attn.{metric}"
    elif metric in NAMESPACE_METRICS["resid"]:
        q = f"resid.{metric}"
    else:
        raise ValueError(f"head_atlas sorts by a per-head metric (attn.* or resid.*), "
                         f"got {metric!r}")
    if not q.startswith(("attn.", "resid.")):
        raise ValueError("head_atlas sorts by a per-head metric (attn.* or resid.*)")
    if q not in cache.metric_names:
        raise KeyError(f"{q!r} is not in the cache; metrics present: "
                       f"{', '.join(cache.metric_names)}")
    return q


class HeadAtlasData:
    def __init__(self, cache, layers=None, text: int = 0, metric: str = "fiedler_value",
                 descending: bool = False):
        c = single(cache, text)
        self.cache = c
        self.layers = pick_layers(c, layers)
        self.metric = _qualify(c, metric)
        self.short = SHORT.get(self.metric.split(".", 1)[1], self.metric)
        self.descending = bool(descending)
        H = c.n_heads
        self.H = H
        computed = c.heads
        self.values = np.stack([np.asarray(c[f"blocks.{lay}.{self.metric}"], dtype=float)
                                for lay in self.layers])                 # [L, H]
        W, U, lam = [], [], []
        for lay in self.layers:
            heads = computed.get(lay, list(range(H)))
            Wl, Ul, ll = [], [], []
            for h in range(H):
                if h in heads:
                    g = layer_graph(c, lay, h)
                    Wl.append(g.normalized())
                    Ul.append(g.fiedler)
                    ll.append(g.lambda2)
                else:
                    S = len(c.tokens)
                    Wl.append(np.zeros((S, S)))
                    Ul.append(np.zeros(S))
                    ll.append(np.nan)
            W.append(np.stack(Wl))
            U.append(np.stack(Ul))
            lam.append(ll)
        self.W = np.stack(W)                                             # [L, H, S, S]
        U = np.stack(U)                                                  # [L, H, S]
        for h in range(H):
            U[:, h] = np.stack(match_signs(list(U[:, h])))
        self.U = U
        self.tokens = [display_token(t) for t in single(cache, text).tokens]
        key = np.where(np.isfinite(self.values), self.values, np.inf if not descending
                       else -np.inf)
        order = np.argsort(-key if descending else key, axis=1, kind="stable")
        self.order = order                                               # [L, H]: slot -> head
        self.slot = np.argsort(order, axis=1)                            # [L, H]: head -> slot


def grid_shape(n: int, aw: float, ah: float):
    """Columns and rows that give ``n`` square tiles the largest size."""
    best = (1, n, 0.0)
    for c in range(1, n + 1):
        r = math.ceil(n / c)
        s = min(aw / c, ah / r)
        if s > best[2]:
            best = (c, r, s)
    return best


class HeadAtlas(Scene):
    kind = "head_atlas"

    def __init__(self, cache, canvas: Canvas, *, fps: float, duration: Optional[float] = None,
                 layers=None, text: int = 0, metric: str = "fiedler_value",
                 descending: bool = False, layer=None, title: Optional[str] = None,
                 subtitle: Optional[str] = None, hold: float = 0.45, per_layer: float = 0.7,
                 wordmark: bool = False, seed: int = 0):
        if layer is not None and layers is None:
            layers = [layer]
        self.data = d = HeadAtlasData(cache, layers, text, metric, descending)
        L = len(d.layers)
        if duration is None:
            duration = 1.0 + per_layer * max(L - 1, 0) + 1.5 if L > 1 else 3.0
        n = int(round(duration * fps))
        super().__init__(canvas, fps, n, wordmark)
        self.pos_t = layer_positions(n, L, int(round(1.0 * fps)), int(round(1.5 * fps)), hold)
        self.title = title
        way = "high to low" if descending else "low to high"
        self.subtitle = subtitle if subtitle is not None else (
            f"{model_label(d.cache)} reading “{prompt_label(d.cache, 50)}” · {d.H} heads sorted "
            f"by {d.short} ({way}), left to right")
        cv = self.cv
        aw = cv.W - 2 * self.margin
        ah = self.bottom - self.top - cv.px(6, 4)
        self.cols, self.rows, self.tile = grid_shape(d.H, aw, ah)
        gx = (aw - self.cols * self.tile) / 2
        self.x0 = self.margin + gx
        self.y0 = self.top + cv.px(2, 1)
        self.cap_fs = cv.fs(12, 9.5) if self.tile > cv.px(90, 60) else max(cv.fs(10, 8.5), 8.5)
        self.cap_h = self.cap_fs * 1.6
        self.R = max((self.tile - self.cap_h) * 0.36, 6.0)
        S = len(d.tokens)
        self.r_node = float(np.clip(self.R * 0.11 * min(1.0, 11 / max(S, 1)) ** 0.5, 1.8,
                                    cv.px(7, 4)))
        self.unit = circle_positions(S, 0.0, 0.0, 1.0)

    def slot_xy(self, s: float):
        r, c = divmod(int(s), self.cols)
        return (self.x0 + c * self.tile + self.tile / 2,
                self.y0 + r * self.tile + (self.tile - self.cap_h) / 2)

    def frame(self, f: int) -> np.ndarray:
        """Graphs morph in place; the new order arrives by a short crossfade
        of the two arrangements (sliding sixteen tiles at once is a jumble)."""
        d = self.data
        i, j, e = split_position(self.pos_t[f], len(d.layers))
        lo, hi = 0.42, 0.58
        if lo < e < hi and not np.array_equal(d.slot[i], d.slot[j]):
            from ._core import smoothstep
            a = float(smoothstep((e - lo) / (hi - lo)))
            img0 = self._render(f, i)
            img1 = self._render(f, j)
            return (img0 * (1 - a) + img1 * a + 0.5).astype(np.uint8)
        return self._render(f, i if e < 0.5 else j)

    def _render(self, f: int, order_layer: int) -> np.ndarray:
        self.cv.begin()
        self.draw(f, order_layer)
        self.draw_wordmark()
        return self.cv.grab()

    def draw(self, f: int, order_layer: Optional[int] = None):
        cv, th, d = self.cv, self.th, self.data
        p = self.pos_t[f]
        i, j, e = split_position(p, len(d.layers))
        if order_layer is None:
            order_layer = i if e < 0.5 else j
        lay = d.layers[i] if e < 0.5 else d.layers[j]
        pad = self.tile * 0.05
        for h in range(d.H):
            rank = int(d.slot[order_layer, h])
            cx, cy = self.slot_xy(rank)
            x, y = cx - self.tile / 2 + pad, cy - (self.tile - self.cap_h) / 2 + pad
            w = self.tile - 2 * pad
            cv.rect(x, y, w, w, th.surface, edge=th.faint, lw=cv.px(1, 0.7), z=1,
                    radius=cv.px(8, 4))
            W = (1 - e) * d.W[i, h] + e * d.W[j, h]
            u = (1 - e) * d.U[i, h] + e * d.U[j, h]
            pos = np.array([cx, cy]) + self.R * self.unit
            draw_edges(cv, th, pos, W, width=2.4 * self.R / cv.px(90, 40), alpha=0.75, z=2)
            draw_nodes(cv, th, pos, fiedler_colours(th, u), self.r_node, z=3)
            v = (1 - e) * d.values[i, h] + e * d.values[j, h]
            ty = y + w - pad * 0.4
            cv.text(x + pad * 0.6, ty, f"head {h}", self.cap_fs, th.ink, weight="semibold",
                    va="bottom", z=4)
            vt = f"{d.short} {v:.3f}" if np.isfinite(v) else "not computed"
            cv.text(x + w - pad * 0.6, ty, vt, self.cap_fs, th.muted, ha="right", va="bottom",
                    z=4)
            if not cv.compact and self.tile > cv.px(110, 60):
                cv.text(x + pad * 0.6, y + pad * 0.5, f"#{rank + 1}", self.cap_fs * 0.92,
                        th.muted, va="top", z=4)
        self.header(self.title or f"Every head of layer {lay}", self.subtitle)
        if len(d.layers) > 1:
            self.counter(str(lay), "layer")
            self.progress(p, d.layers)


def head_atlas_html(cache, path, theme, *, layers=None, layer=None, text: int = 0,
                    metric: str = "fiedler_value", descending: bool = False,
                    plotlyjs: str = "auto", title: Optional[str] = None, **_ignored):
    """Interactive version: each head's attention pattern as a heatmap,
    tiles sorted by the metric, with a layer slider; hover a cell for the
    query and key tokens and the weight."""
    if layer is not None and layers is None:
        layers = [layer]
    d = HeadAtlasData(cache, layers, text, metric, descending)
    c = d.cache
    H, L = d.H, len(d.layers)
    cols, rows, _ = grid_shape(H, 16.0, 9.0)
    gap = 0.025
    cw, rh = (1 - gap * (cols - 1)) / cols, (0.92 - gap * (rows - 1)) / rows
    toks = d.tokens
    S = len(toks)
    hover = [[f"query {q}: {toks[q]}<br>key {k}: {toks[k]}" for k in range(S)] for q in range(S)]
    scale = [[0, theme.surface], [0.06, hexa_or(theme.pos, 0.25)], [0.35, theme.pos],
             [1, "#7A3C00" if theme.name == "light" else "#FFD27A"]]

    def tile_data(li):
        out = []
        pat = np.asarray(c[f"blocks.{d.layers[li]}.attn.pattern"], dtype=float)
        sp = c.token_span
        if sp is not None:
            pat = pat[:, sp[0]:sp[1], sp[0]:sp[1]]
        for s in range(H):
            h = int(d.order[li, s])
            out.append({"type": "heatmap", "z": np.round(pat[h], 4).tolist(), "zmin": 0,
                        "zmax": 1, "colorscale": scale, "showscale": False,
                        "xaxis": f"x{s + 1}", "yaxis": f"y{s + 1}", "text": hover,
                        "hovertemplate": f"head {h}<br>%{{text}}<br>weight %{{z:.3f}}"
                                         "<extra></extra>"})
        return out

    def titles(li):
        out = []
        for s in range(H):
            h = int(d.order[li, s])
            r, col = divmod(s, cols)
            v = d.values[li, h]
            out.append({"text": f"<b>head {h}</b> · {d.short} "
                                + (f"{v:.3f}" if np.isfinite(v) else "n/a"),
                        "x": col * (cw + gap), "y": 0.92 - r * (rh + gap) + 0.004,
                        "xref": "paper", "yref": "paper", "xanchor": "left", "yanchor": "bottom",
                        "showarrow": False, "font": {"size": 11, "color": theme.ink2}})
        return out

    axes = {}
    for s in range(H):
        r, col = divmod(s, cols)
        x0 = col * (cw + gap)
        y1 = 0.92 - r * (rh + gap)
        k = "" if s == 0 else str(s + 1)
        axes[f"xaxis{k}"] = {"domain": [x0, x0 + cw], "visible": False, "anchor": f"y{s + 1}"}
        axes[f"yaxis{k}"] = {"domain": [y1 - rh, y1], "visible": False, "autorange": "reversed",
                             "anchor": f"x{s + 1}"}
    frames = [{"name": f"L{lay}", "data": tile_data(li), "layout": {"annotations": titles(li)}}
              for li, lay in enumerate(d.layers)]
    menus, sliders = plotly_player([f["name"] for f in frames], [str(x) for x in d.layers],
                                   theme, frame_ms=600, redraw=True)
    way = "high to low" if descending else "low to high"
    layout = plotly_layout(
        theme, title or "Every head, layer by layer",
        f"{model_label(c)} · attention pattern of each head (rows: queries), sorted by "
        f"{d.short} ({way})", annotations=titles(0), **axes)
    if L > 1:
        layout["updatemenus"], layout["sliders"] = menus, sliders
        for s in layout["sliders"]:
            s["y"] = -0.02
        for m in layout["updatemenus"]:
            m["y"] = -0.02
    fig = {"data": tile_data(0), "layout": layout, "frames": frames if L > 1 else []}
    return write_plotly_html(path, fig, f"head_atlas · {model_label(c)}", theme, plotlyjs)


def hexa_or(c: str, a: float) -> str:
    """``c`` blended into white by ``1 - a`` (an opaque colour scale stop)."""
    rgb = hex_rgb(c) * a + (1 - a)
    return "#" + "".join(f"{int(round(v * 255)):02X}" for v in rgb)
