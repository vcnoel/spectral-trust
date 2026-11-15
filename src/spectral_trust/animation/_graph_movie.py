# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""``graph_movie``: the token graph of each layer, edges by weight and
nodes by the Fiedler vector, morphing from layer to layer."""
from typing import List, Optional

import numpy as np

from ._core import (
    Canvas, LabelPlacer, hex_rgb, layer_positions, plotly_layout, plotly_player, r4, ramp,
    rgba, smootherstep, split_position, write_plotly_html,
)
from ._data import (
    align_sequence, eigenmap, layer_graphs, match_signs, model_label, pick_layers, prompt_label,
    single,
)
from ._graphdraw import (
    circle_positions, draw_circle_labels, draw_edges, draw_nodes, fiedler_colours, label_room,
)
from ._scene import Scene

LAYOUTS = ("circle", "force", "spectral")


def force_layouts(Ws: List[np.ndarray], seed: int = 0, first_iter: int = 300,
                  iters: int = 80) -> List[np.ndarray]:
    """Fruchterman-Reingold layouts of a sequence of graphs, each started
    from the previous one so that successive layouts stay close, then
    Procrustes-aligned and scaled to unit RMS radius. Deterministic for a
    fixed ``seed``."""
    rng = np.random.default_rng(seed)
    n = Ws[0].shape[0]
    pos = rng.normal(size=(n, 2))
    out = []
    for li, W in enumerate(Ws):
        A = W / (W.max() or 1.0)
        k = 1.0 / np.sqrt(max(n, 1))
        steps = first_iter if li == 0 else iters
        t = 0.1 if li == 0 else 0.03
        for s in range(steps):
            d = pos[:, None, :] - pos[None, :, :]
            dist = np.maximum(np.linalg.norm(d, axis=-1), 1e-3)
            rep = (k * k / dist ** 2)[..., None] * d
            att = (A * dist / k)[..., None] * d
            disp = (rep - att).sum(1)
            ln = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9)
            pos = pos + disp / ln * np.minimum(ln, t)
            t *= 0.985
        pos = pos - pos.mean(0)
        out.append(pos / (np.sqrt(np.mean(np.sum(pos ** 2, 1))) or 1.0))
    return align_sequence(out, "procrustes")


class GraphMovieData:
    def __init__(self, cache, layers=None, text: int = 0, layout: str = "circle", seed: int = 0):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        c = single(cache, text)
        self.cache = c
        self.layers = pick_layers(c, layers)
        gs = layer_graphs(c, self.layers)
        self.tokens = gs[0].tokens
        self.W = np.stack([g.normalized() for g in gs])
        self.U = np.stack(match_signs([g.fiedler for g in gs]))
        self.lam2 = np.array([g.lambda2 for g in gs])
        self.layout = layout
        S = len(self.tokens)
        if layout == "circle":
            P = circle_positions(S, 0.0, 0.0, 1.0)
            self.P = np.repeat(P[None], len(gs), 0)
        elif layout == "force":
            self.P = np.stack(force_layouts(list(self.W), seed))
        else:
            self.P = np.stack(align_sequence([eigenmap(g, 2) for g in gs], "procrustes"))
        if layout != "circle":
            # one frame for the film: radius 1 holds every layout
            self.P = self.P / (np.max(np.linalg.norm(self.P, axis=2)) or 1.0)
            self.P[..., 1] *= -1                       # screen y points down

    def at(self, p: float):
        i, j, e = split_position(p, len(self.layers))
        return i, j, e, (1 - e) * self.P[i] + e * self.P[j], (1 - e) * self.W[i] + e * self.W[j], \
            (1 - e) * self.U[i] + e * self.U[j]


class GraphMovie(Scene):
    kind = "graph_movie"

    def __init__(self, cache, canvas: Canvas, *, fps: float, duration: Optional[float] = None,
                 layers=None, text: int = 0, layout: str = "circle", intro: bool = True,
                 title: Optional[str] = None, subtitle: Optional[str] = None, hold: float = 0.35,
                 per_layer: float = 0.5, wordmark: bool = False, seed: int = 0):
        from ._core import display_token

        self.data = d = GraphMovieData(cache, layers, text, layout, seed)
        L = len(d.layers)
        self.intro_s = 2.4 if intro else 0.0
        if duration is None:
            duration = self.intro_s + 0.6 + per_layer * max(L - 1, 1) + 1.5
        n = int(round(duration * fps))
        super().__init__(canvas, fps, n, wordmark)
        lead = int(round((self.intro_s + 0.6) * fps))
        self.pos_t = layer_positions(n, L, lead, int(round(1.5 * fps)), hold)
        self.tokens = [display_token(t) for t in d.tokens]
        self.title = title if title is not None else "The attention graph, layer by layer"
        self.subtitle = subtitle if subtitle is not None else (
            f"{model_label(d.cache)} reading “{prompt_label(d.cache, 70)}”")
        cv = self.cv
        S = len(self.tokens)
        self.radial = layout == "circle" and S > 16
        self.label_fs = cv.fs(17, 12) if S <= 16 else cv.fs(13, 10.5)
        room = label_room(cv, self.tokens, self.label_fs, self.radial)
        y0, y1 = self.top, self.bottom - cv.px(20, 12)
        self.centre = (cv.W / 2, (y0 + y1) / 2)
        self.R = max(min((y1 - y0) / 2 - room * (1.0 if self.radial else 0.7),
                         cv.W / 2 - self.margin - room), cv.px(60, 30))
        self.r_node = float(np.clip(self.R * 0.075 * min(1.0, 11 / max(S, 1)) ** 0.5,
                                    cv.px(5, 3.5), cv.px(16, 9)))
        if layout != "circle":
            sizes = [cv.text_size(t, self.label_fs, "medium") for t in self.tokens]
            self.placer = LabelPlacer(sizes, cv.px(5, 3),
                                      (self.margin, self.top, cv.W - self.margin, self.bottom))
        # edge order of the draw-in, weakest first (as on the project site)
        W0 = d.W[0]
        iu, ju = np.triu_indices(S, 1)
        w = W0[iu, ju]
        self.n_edges = int(np.sum(w / (w.max() or 1.0) > 0.05))

    def _intro_state(self, t: float):
        """Edge progress (weakest first) and node fill in [0, 1] at time t."""
        ne = max(self.n_edges, 1)
        stagger = min(0.04, 1.0 / ne)
        edge = smootherstep((t - 0.2 - np.arange(ne) * stagger) / 0.9)
        t_nodes = 0.2 + ne * stagger + 0.5
        S = len(self.tokens)
        fill = smootherstep((t - t_nodes - np.arange(S) * min(0.07, 0.8 / S)) / 0.6)
        return edge, fill

    def draw(self, f: int):
        cv, th, d = self.cv, self.th, self.data
        p = self.pos_t[f]
        i, j, e, P, W, u = d.at(p)
        pos = np.asarray(self.centre) + self.R * P
        t = f / self.fps
        edge_prog, fill = (None, None)
        if t < self.intro_s:
            edge_prog, fill = self._intro_state(t)
        draw_edges(cv, th, pos, W, width=5.0, alpha=0.78, progress=edge_prog)
        cols = fiedler_colours(th, u)
        if fill is not None:
            empty = hex_rgb(th.surface2)
            cols = empty * (1 - fill[:, None]) + cols * fill[:, None]
        draw_nodes(cv, th, pos, cols, self.r_node)
        if d.layout == "circle":
            draw_circle_labels(cv, th, pos, self.centre, self.R, self.r_node, self.tokens,
                               self.label_fs, radial=self.radial)
        else:
            order = np.argsort(-np.abs(u))
            centres, _ = self.placer.place(pos, self.r_node + cv.px(1.5, 1), order, self.centre)
            for k, tok in enumerate(self.tokens):
                a = float(self.placer.alpha[k])
                if a > 0.01:
                    cv.text(centres[k, 0], centres[k, 1], tok, self.label_fs, th.ink2,
                            ha="center", va="center", weight="medium", alpha=a,
                            halo=cv.px(3, 2), z=20)
        self.header(self.title, self.subtitle)
        lay = d.layers[i] if e < 0.5 else d.layers[j]
        lam = (1 - e) * d.lam2[i] + e * d.lam2[j]
        self.counter(str(lay), "layer", f"λ₂ = {lam:.3f}")
        self._ramp_legend()
        self.progress(p, d.layers)

    def _ramp_legend(self):
        cv, th = self.cv, self.th
        x, y = self.margin, self.bottom + cv.px(12, 7)
        w, h = cv.px(120, 70), cv.px(9, 6)
        xs = np.linspace(0, 1, 61)
        cols = ramp([th.neg, th.node_mid, th.pos], (xs[:-1] + xs[1:]) / 2)
        step = w / (len(xs) - 1)
        polys = [[(x + w * a, y - h / 2), (x + w * a + step * 1.6, y - h / 2),
                  (x + w * a + step * 1.6, y + h / 2), (x + w * a, y + h / 2)] for a in xs[:-1]]
        cv.polygons(polys, rgba(cols, 1.0), z=50)
        cv.rect(x, y - h / 2, w + step * 0.6, h, "none", edge=th.rule, lw=cv.px(0.8, 0.6), z=50.5)
        fs = cv.fs(12.5, 10)
        cv.text(x + w + cv.px(10, 6), y, "Fiedler vector, − to +", fs, th.muted,
                va="center_baseline", z=50)


def graph_movie_html(cache, path, theme, *, layers=None, text: int = 0, layout: str = "circle",
                     plotlyjs: str = "auto", title: Optional[str] = None, seed: int = 0,
                     **_ignored):
    """Interactive version: the graph of each layer with a slider, hover a
    node for its token, Fiedler entry and degree."""
    from ._core import display_token

    d = GraphMovieData(cache, layers, text, layout, seed)
    toks = [display_token(t) for t in d.tokens]
    S = len(toks)

    def traces(li):
        P = d.P[li].copy()
        P[:, 1] *= -1
        W, u = d.W[li], d.U[li]
        iu, ju = np.triu_indices(S, 1)
        w = W[iu, ju]
        rel = w / (w.max() or 1.0)
        out = []
        for lo, hi, a, width in ((0.05, 0.25, 0.15, 1.0), (0.25, 0.55, 0.35, 2.4),
                                 (0.55, 1.01, 0.7, 4.5)):
            sel = (rel > lo) & (rel <= hi)
            xs, ys = [], []
            for a_, b_ in zip(iu[sel], ju[sel]):
                xs += [round(float(P[a_, 0]), 4), round(float(P[b_, 0]), 4), None]
                ys += [round(float(P[a_, 1]), 4), round(float(P[b_, 1]), 4), None]
            r, g, b = (int(v * 255) for v in hex_rgb(theme.edge))
            out.append({"type": "scatter", "mode": "lines", "x": xs, "y": ys,
                        "hoverinfo": "skip", "line": {"color": f"rgba({r},{g},{b},{a})",
                                                      "width": width}})
        cols = fiedler_colours(theme, u)
        deg = W.sum(1)
        out.append({"type": "scatter", "mode": "markers+text", "x": r4(P[:, 0]),
                    "y": r4(P[:, 1]), "text": toks,
                    "textposition": ["middle right" if x > 0.2 else "middle left" if x < -0.2
                                     else ("top center" if y > 0 else "bottom center")
                                     for x, y in P],
                    "hovertext": [f"<b>{t}</b><br>Fiedler {u[k]:+.3f}<br>degree {deg[k]:.3f}"
                                  for k, t in enumerate(toks)],
                    "hoverinfo": "text", "textfont": {"color": theme.ink2, "size": 14},
                    "marker": {"size": 18, "color": ["rgb({},{},{})".format(
                        *(int(round(v * 255)) for v in c)) for c in cols],
                        "line": {"color": theme.node_line, "width": 1.5}}})
        return out

    frames = [{"name": f"L{lay}", "data": traces(li)} for li, lay in enumerate(d.layers)]
    menus, sliders = plotly_player([f["name"] for f in frames], [str(x) for x in d.layers],
                                   theme, frame_ms=400, redraw=False)
    ax = {"visible": False, "range": [-1.5, 1.5], "fixedrange": True}
    layout_ = plotly_layout(
        theme, title or "The attention graph, layer by layer",
        f"{model_label(d.cache)} · edges: normalised attention; node colour: the Fiedler vector "
        "(− blue, + orange)",
        xaxis=ax, yaxis=dict(ax, scaleanchor="x"), updatemenus=menus, sliders=sliders)
    for s in layout_["sliders"]:
        s["y"] = -0.02
    for m in layout_["updatemenus"]:
        m["y"] = -0.02
    fig = {"data": frames[0]["data"], "layout": layout_, "frames": frames}
    return write_plotly_html(path, fig, f"graph_movie · {model_label(d.cache)}", theme, plotlyjs)
