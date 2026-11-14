# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""``embedding3d``: tokens placed in 3D by Laplacian eigenvectors 2 to 4 of
each layer's head-averaged attention graph, morphing from layer to layer."""
import math
from typing import Optional

import numpy as np

from ._core import (
    Camera, Canvas, LabelPlacer, display_token, hex_rgb, hexa, layer_positions, mix, plotly_layout,
    plotly_player, r4, rgba, smooth_track, smoothstep, split_position, write_plotly_html,
)
from ._data import (
    align_sequence, eigenmap, layer_graphs, match_signs, model_label, pick_layers,
    principal_frame, prompt_label, single,
)
from ._scene import Scene


class EmbeddingData:
    """Aligned eigenmaps, Fiedler vectors and edge weights of every layer."""

    def __init__(self, cache, layers=None, text: int = 0, align: str = "procrustes"):
        c = single(cache, text)
        self.cache = c
        self.layers = pick_layers(c, layers)
        gs = layer_graphs(c, self.layers)
        self.raw_tokens = gs[0].tokens
        self.tokens = [display_token(t) for t in gs[0].tokens]
        self.offset = gs[0].offset
        self.raw = [eigenmap(g) for g in gs]
        aligned = align_sequence(self.raw, align)
        R = principal_frame(aligned)
        self.P = np.stack([a @ R for a in aligned])              # [L, S, 3]
        self.U = np.stack(match_signs([g.fiedler for g in gs]))   # [L, S]
        self.Wn = np.stack([g.normalized() for g in gs])          # [L, S, S]
        self.lam2 = np.array([g.lambda2 for g in gs])

    def at(self, p: float):
        """Positions, Fiedler vector and edge weights at fractional layer ``p``."""
        i, j, e = split_position(p, len(self.layers))
        X = (1 - e) * self.P[i] + e * self.P[j]
        rms = math.sqrt(float(np.mean(np.sum(X ** 2, axis=1)))) or 1.0
        return X / rms, i, j, e

    def colours(self, th, i: int, j: int, e: float) -> np.ndarray:
        """Node RGB: the side of the Fiedler split, faded toward neutral for
        entries too small for their sign to mean much."""
        def side(u):
            m = float(np.max(np.abs(u))) or 1.0
            strength = np.clip(np.abs(u) / (0.15 * m), 0, 1)
            base = np.where(u[:, None] < 0, hex_rgb(th.neg), hex_rgb(th.pos))
            mid = hex_rgb(th.node_mid)
            w = (0.35 + 0.65 * strength)[:, None]
            return mid * (1 - w) + base * w
        return (1 - e) * side(self.U[i]) + e * side(self.U[j])


class Embedding3D(Scene):
    kind = "embedding3d"

    def __init__(self, cache, canvas: Canvas, *, fps: float, duration: Optional[float] = None,
                 layers=None, text: int = 0, orbit: float = 120.0, azim: float = -58.0,
                 elev: float = 16.0, align: str = "procrustes", edges: bool = True,
                 labels: bool = True, title: Optional[str] = None,
                 subtitle: Optional[str] = None, hold: float = 0.2, per_layer: float = 0.6,
                 wordmark: bool = False,
                 seed: int = 0):
        self.data = d = EmbeddingData(cache, layers, text, align)
        L = len(d.layers)
        if duration is None:
            duration = 1.0 + per_layer * max(L - 1, 1) + 1.2
        n = int(round(duration * fps))
        super().__init__(canvas, fps, n, wordmark)
        intro = int(round(min(1.0, 0.08 * duration) * fps))
        outro = int(round(min(1.2, 0.1 * duration) * fps))
        self.pos = layer_positions(n, L, intro, outro, hold, ease=smoothstep)
        self.orbit, self.azim0, self.elev0 = float(orbit), float(azim), float(elev)
        self.edges, self.show_labels = bool(edges), bool(labels)
        self.title = title if title is not None else "Attention as geometry"
        self.subtitle = subtitle if subtitle is not None else (
            f"{model_label(d.cache)} reading “{prompt_label(d.cache, 70)}”")
        self.note = "position: Laplacian eigenvectors 2–4 of the layer's attention graph"
        cv = self.cv
        self.label_fs = cv.fs(15, 11)
        sizes = [cv.text_size(t, self.label_fs, "medium") for t in d.tokens]
        self.placer = LabelPlacer(sizes, cv.px(5, 3),
                                  (self.margin, self.top, cv.W - self.margin,
                                   self.bottom - cv.px(26, 18)))
        self.dist = 5.5
        self._plan()

    # camera -----------------------------------------------------------------
    def _camera_angles(self, f: int):
        t = f / max(self.n_frames - 1, 1)
        return self.azim0 + self.orbit * t, self.elev0 + 5.0 * math.sin(2 * math.pi * t)

    def _plan(self):
        """Framing of every frame. The floor sits just under the current
        cloud and its ring just around it; the zoom and the centre fit the
        cloud, the ring and room for labels, then a running minimum and a
        Gaussian make the camera move smoothly without ever overflowing."""
        cv, d, n = self.cv, self.data, self.n_frames
        lab_w = max(cv.text_size(t, self.label_fs, "medium")[0] for t in d.tokens)
        lab_h = cv.text_size("Ag", self.label_fs, "medium")[1]
        x0, x1 = self.margin + 0.75 * lab_w, cv.W - self.margin - 0.75 * lab_w
        y0, y1 = self.top + 1.8 * lab_h, self.bottom - cv.px(34, 22) - 1.4 * lab_h
        floors, rings = np.zeros(n), np.zeros(n)
        for f in range(n):
            X, *_ = d.at(self.pos[f])
            floors[f] = float(X[:, 2].min()) - 0.2
            rings[f] = float(np.max(np.hypot(X[:, 0], X[:, 1]))) * 0.85
        sig = 0.6 * self.fps
        self.floors = smooth_track(floors, sig, low=True)
        self.rings = smooth_track(rings, sig, low=False)
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        scale, cx, cy = np.zeros(n), np.zeros(n), np.zeros(n)
        for f in range(n):
            X, *_ = d.at(self.pos[f])
            az, el = self._camera_angles(f)
            cam = Camera(az, el, self.dist, 1.0, (0.0, 0.0))
            ring = np.stack([self.rings[f] * np.cos(th), self.rings[f] * np.sin(th),
                             np.full_like(th, self.floors[f])], 1)
            q, _, _ = cam.project(np.concatenate([X, ring]))
            lo, hi = q.min(0), q.max(0)
            mid = (lo + hi) / 2
            scale[f] = min((x1 - x0) / max(hi[0] - lo[0], 1e-9),
                           (y1 - y0) / max(hi[1] - lo[1], 1e-9))
            cx[f], cy[f] = mid
        self.scales = smooth_track(scale, sig, low=True)
        # the centre is in world-projected units: screen = c - scale * mid
        self.mids = np.stack([smooth_track(cx, sig), smooth_track(cy, sig)], 1)
        self.box_center = ((x0 + x1) / 2, (y0 + y1) / 2)

    def camera(self, f: int) -> Camera:
        az, el = self._camera_angles(f)
        s = float(self.scales[f])
        bx, by = self.box_center
        mx, my = self.mids[f]
        return Camera(az, el, self.dist, s, (bx - s * mx, by - s * my))

    # drawing ----------------------------------------------------------------
    def draw(self, f: int):
        cv, th, d = self.cv, self.th, self.data
        p = self.pos[f]
        X, i, j, e = d.at(p)
        cam = self.camera(f)
        xy, depth, k = cam.project(X)
        S = len(X)
        dn = (depth - depth.min()) / (np.ptp(depth) or 1.0)
        fog = 1.0 - 0.45 * dn

        # floor: two rings, a drop line and a shadow per token
        th_ = np.linspace(0, 2 * np.pi, 97)
        segs = []
        floor, rr0 = float(self.floors[f]), float(self.rings[f])
        for rr in (rr0,):
            ring = np.stack([rr * np.cos(th_), rr * np.sin(th_), np.full_like(th_, floor)], 1)
            segs.append(cam.project(ring)[0])
        cv.lines(segs, [th.rule], [cv.px(1.1, 0.8)], z=1)
        foot = X.copy()
        foot[:, 2] = floor
        fxy, _, fk = cam.project(foot)
        drops = [[tuple(a), tuple(b)] for a, b in zip(xy, fxy)]
        cv.lines(drops, rgba(hex_rgb(th.rule), 0.9 * fog), cv.px(1.0, 0.7), z=1.5)
        r_node = cv.px(9.5, 5.5) * k
        sh = rgba(hex_rgb(th.rule), 0.95 * fog)
        cv.ellipses(fxy, 1.5 * r_node * fk / k, 1.5 * r_node * fk / k * max(math.sin(cam.elev), 0.15),
                    sh, z=2)

        cols = d.colours(th, i, j, e)
        # edges: the normalised weights the Laplacian sees, strongest on top
        if self.edges:
            W = (1 - e) * d.Wn[i] + e * d.Wn[j]
            iu, ju = np.triu_indices(S, 1)
            w = W[iu, ju]
            wmax = float(w.max()) if w.size and w.max() > 0 else 1.0
            rel = w / wmax
            keep = rel > 0.05
            order = np.argsort(rel[keep])
            a_, b_, r_ = iu[keep][order], ju[keep][order], rel[keep][order]
            if len(a_):
                segs = [[tuple(xy[a]), tuple(xy[b])] for a, b in zip(a_, b_)]
                ec = (cols[a_] + cols[b_]) / 2
                ec = ec * 0.55 + hex_rgb(th.edge) * 0.45
                alpha = (0.10 + 0.72 * r_ ** 1.2) * (fog[a_] + fog[b_]) / 2
                cv.lines(segs, rgba(ec, alpha), cv.px(0.8, 0.5) + cv.px(3.4, 1.8) * r_, z=3)

        # nodes, far first
        order = np.argsort(-depth)
        bg = hex_rgb(th.bg)
        face = cols * fog[:, None] + bg * (1 - fog[:, None])
        glow = rgba(cols[order], (0.16 if th.name == "light" else 0.24) * fog[order])
        cv.circles(xy[order], r_node[order] * 1.9, glow, z=4)
        cv.circles(xy[order], r_node[order], rgba(face[order], 1.0), edge=th.node_line,
                   lw=cv.px(1.2, 0.8), z=5)

        # labels, nearest first
        if self.show_labels:
            near = np.argsort(depth)
            centres, ring = self.placer.place(xy, r_node + cv.px(1.5, 1), near,
                                              tuple(xy.mean(0)))
            for t in range(S):
                a = float(0.55 + 0.45 * fog[t]) * float(self.placer.alpha[t])
                if a <= 0.01:
                    continue
                if ring[t]:
                    w_, h_ = self.placer.sizes[t]
                    cx, cy = centres[t]
                    bx = np.clip(xy[t, 0], cx - w_ / 2, cx + w_ / 2)
                    by = np.clip(xy[t, 1], cy - h_ / 2, cy + h_ / 2)
                    v = np.array([bx, by]) - xy[t]
                    nv = np.linalg.norm(v) or 1.0
                    s0 = xy[t] + v / nv * r_node[t]
                    cv.lines([[tuple(s0), (bx, by)]], rgba(hex_rgb(th.muted), 0.7 * a),
                             cv.px(1.0, 0.7), z=19)
                cv.text(centres[t, 0], centres[t, 1], d.tokens[t], self.label_fs,
                        th.ink2, ha="center",
                        va="center", weight="medium", alpha=a, halo=cv.px(3, 2), z=20)

        # overlays
        self.header(self.title, self.subtitle)
        lay = d.layers[i] if e < 0.5 else d.layers[j]
        lam = (1 - e) * d.lam2[i] + e * d.lam2[j]
        self.counter(str(lay), "layer", f"λ₂ = {lam:.3f}")
        self.legend_split(self.margin, self.bottom + cv.px(14, 8), note=self.note)
        self.progress(p, d.layers)


def embedding3d_html(cache, path, theme, *, layers=None, text: int = 0,
                     align: str = "procrustes", plotlyjs: str = "cdn", steps: int = 6,
                     title: Optional[str] = None, azim: float = -58.0, elev: float = 20.0,
                     frame_ms: int = 70, **_ignored):
    """Interactive version: rotate, zoom and hover; play morphs the layers."""
    d = EmbeddingData(cache, layers, text, align)
    L = len(d.layers)
    S = len(d.tokens)
    R = float(np.max(np.abs(d.P))) * 1.15
    edge_bins = [(0.08, 0.3, 0.18, 1.5), (0.3, 0.6, 0.38, 2.5), (0.6, 1.01, 0.65, 4.0)]

    def traces(p):
        X, i, j, e = d.at(p)
        cols = d.colours(theme, i, j, e)
        W = (1 - e) * d.Wn[i] + e * d.Wn[j]
        iu, ju = np.triu_indices(S, 1)
        w = W[iu, ju]
        rel = w / (w.max() or 1.0)
        out = []
        for lo, hi, a, width in edge_bins:
            sel = (rel > lo) & (rel <= hi)
            xs, ys, zs = [], [], []
            for a_, b_ in zip(iu[sel], ju[sel]):
                xs += [X[a_, 0], X[b_, 0], None]
                ys += [X[a_, 1], X[b_, 1], None]
                zs += [X[a_, 2], X[b_, 2], None]
            out.append({"type": "scatter3d", "mode": "lines", "x": _r(xs), "y": _r(ys),
                        "z": _r(zs), "hoverinfo": "skip",
                        "line": {"color": hexa(theme.edge, a), "width": width}})
        u = (1 - e) * d.U[i] + e * d.U[j]
        lay = d.layers[i] if e < 0.5 else d.layers[j]
        hover = [f"<b>{_esc(t)}</b><br>token {k + d.offset}<br>Fiedler {u[k]:+.3f}"
                 f"<br>layer {lay}" for k, t in enumerate(d.tokens)]
        out.append({"type": "scatter3d", "mode": "markers+text", "x": r4(X[:, 0]),
                    "y": r4(X[:, 1]), "z": r4(X[:, 2]), "text": d.tokens,
                    "textposition": "top center", "hovertext": hover, "hoverinfo": "text",
                    "textfont": {"color": theme.ink2, "size": 14},
                    "marker": {"size": 9, "color": ["rgb({},{},{})".format(
                        *(int(round(v * 255)) for v in c)) for c in cols],
                        "line": {"color": theme.node_line, "width": 1}}})
        return out

    frames, names, labels = [], [], []
    for li in range(L):
        subs = [0.0] if li == L - 1 else [s / steps for s in range(steps)]
        for s in subs:
            p = li + s
            name = f"L{d.layers[li]}" if s == 0 else f"L{d.layers[li]}+{s:.2f}"
            frames.append({"name": name, "data": traces(p)})
            if s == 0:
                names.append(name)
                labels.append(str(d.layers[li]))
    a, e_ = math.radians(azim), math.radians(elev)
    eye = {"x": 1.9 * math.cos(e_) * math.cos(a), "y": 1.9 * math.cos(e_) * math.sin(a),
           "z": 1.9 * math.sin(e_)}
    axis = {"visible": False, "range": [-R, R], "showspikes": False}
    menus, sliders = plotly_player(names, labels, theme, frame_ms=frame_ms, redraw=True)
    layout = plotly_layout(
        theme, title or "Attention as geometry",
        f"{model_label(d.cache)} · tokens placed by Laplacian eigenvectors 2–4 of each "
        "layer's attention graph · drag to rotate, play to move through the layers",
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "cube",
               "bgcolor": theme.bg, "camera": {"eye": eye}, "dragmode": "orbit"},
        updatemenus=menus, sliders=sliders, uirevision="keep",
        annotations=[{"text": f"<span style='color:{theme.neg}'>●</span> Fiedler −   "
                              f"<span style='color:{theme.pos}'>●</span> Fiedler +",
                      "x": 1.0, "y": 1.06, "xref": "paper", "yref": "paper",
                      "xanchor": "right", "showarrow": False,
                      "font": {"color": theme.ink2, "size": 13}}])
    fig = {"data": frames[0]["data"], "layout": layout, "frames": frames}
    return write_plotly_html(path, fig, f"embedding3d · {model_label(d.cache)}", theme, plotlyjs)


def _r(xs):
    return [None if v is None else round(float(v), 4) for v in xs]


def _esc(s: str) -> str:
    import html
    return html.escape(s)
