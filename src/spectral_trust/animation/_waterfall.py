# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""``waterfall``: residual energy per graph-frequency mode through depth, as
ridgelines in 3D (layer x mode x energy share), built layer by layer."""
import math
from typing import Optional

import numpy as np

from ._core import (
    Camera, Canvas, hex_rgb, plotly_layout, r4, rgba, smooth_track,
    smoothstep, write_plotly_html,
)
from ._data import graph_series, model_label, pick_layers, prompt_label, single, spectral_masses
from ._scene import Scene

SCALES = ("linear", "sqrt", "log")


class WaterfallData:
    def __init__(self, cache, layers=None, text: int = 0, scale: str = "linear",
                 log_floor: float = 1e-4):
        from ..fast import num_high_modes

        if scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        c = single(cache, text)
        self.cache = c
        self.layers = pick_layers(c, layers)
        self.share = spectral_masses(c, self.layers)          # [L, S], rows sum to 1
        self.S = self.share.shape[1]
        self.hfer = graph_series(c, "hfer", self.layers)
        self.n_high = num_high_modes(self.S, c.config.hfer_cutoff_ratio)
        self.cut = self.S - self.n_high                       # first mode of the HFER band
        self.scale = scale
        if scale == "linear":
            v = self.share
        elif scale == "sqrt":
            v = np.sqrt(self.share)
        else:
            v = (np.log10(np.maximum(self.share, log_floor)) - math.log10(log_floor)) \
                / -math.log10(log_floor)
        self.value = v / (float(v.max()) or 1.0)              # [L, S] in [0, 1]

    def curve(self, l: int, samples: int = 8):
        """Mode coordinate and height of a smooth ridge through the modes
        (monotone cubic, so it never overshoots the measured values)."""
        from scipy.interpolate import PchipInterpolator

        k = np.arange(self.S, dtype=float)
        if self.S < 2:
            return k, self.value[l]
        kk = np.linspace(0, self.S - 1, samples * (self.S - 1) + 1)
        v = PchipInterpolator(k, self.value[l])(kk)
        return kk, np.clip(v, 0, None)


class Waterfall(Scene):
    kind = "waterfall"

    def __init__(self, cache, canvas: Canvas, *, fps: float, duration: Optional[float] = None,
                 layers=None, text: int = 0, scale: str = "linear", title: Optional[str] = None,
                 subtitle: Optional[str] = None, azim=(-62.0, -104.0), elev=(22.0, 30.0),
                 wordmark: bool = False, seed: int = 0):
        self.data = d = WaterfallData(cache, layers, text, scale)
        L = len(d.layers)
        if duration is None:
            duration = 1.0 + 0.32 * L + 2.5
        n = int(round(duration * fps))
        super().__init__(canvas, fps, n, wordmark)
        self.t_intro = min(1.0, 0.1 * duration)
        self.t_build = max(duration - self.t_intro - 2.0, 0.5 * duration)
        self.grow = min(0.6, self.t_build / max(L, 1) * 2.5)
        self.azim, self.elev = azim, elev
        self.title = title if title is not None else "Where the energy sits"
        self.subtitle = subtitle if subtitle is not None else (
            f"{model_label(d.cache)} reading “{prompt_label(d.cache, 70)}”")
        self.note = ("residual-stream energy per graph-frequency mode, as a share of each "
                     "layer's total" + ("" if scale == "linear" else f" ({scale} scale)"))
        # world: x = mode (smooth to rough), y = layer (front to back), z = energy
        self.X0, self.X1 = -1.3, 1.3
        self.Y0, self.Y1 = -1.25, 1.25
        self.H = 1.15
        # modes sit half a step inside the floor, so the end modes do not look like its edges
        self.xk = lambda k: self.X0 + (self.X1 - self.X0) * (np.asarray(k) + 0.5) / max(d.S, 1)
        self.yl = lambda l: self.Y0 + (self.Y1 - self.Y0) * np.asarray(l) / max(L - 1, 1)
        self.curves = [d.curve(l) for l in range(L)]
        self.dist = 7.0
        self._plan()

    # timing and camera --------------------------------------------------------
    def growth(self, f: int) -> np.ndarray:
        """Height factor in [0, 1] of every ridge at frame ``f``."""
        t = f / self.fps
        L = len(self.data.layers)
        starts = self.t_intro + np.arange(L) * (self.t_build - self.grow) / max(L - 1, 1)
        return smoothstep((t - starts) / self.grow)

    def cursor(self, f: int) -> float:
        g = self.growth(f)
        return float(np.clip(np.sum(g) - 1, 0, len(g) - 1))

    def angles(self, f: int):
        t = smoothstep(f / max(self.n_frames - 1, 1))
        return (self.azim[0] + (self.azim[1] - self.azim[0]) * t,
                self.elev[0] + (self.elev[1] - self.elev[0]) * t)

    def _box(self):
        x = [self.X0, self.X1]
        y = [self.Y0, self.Y1]
        z = [0.0, self.H]
        return np.array([[a, b, c] for a in x for b in y for c in z])

    def _plan(self):
        cv = self.cv
        x0, x1 = self.margin + cv.px(70, 30), cv.W - self.margin - cv.px(250, 40)
        y0, y1 = self.top + cv.px(20, 10), self.bottom - cv.px(56, 34)
        box = self._box()
        sc, mids = [], []
        for f in range(self.n_frames):
            az, el = self.angles(f)
            q, _, _ = Camera(az, el, self.dist, 1.0, (0, 0)).project(box)
            lo, hi = q.min(0), q.max(0)
            sc.append(min((x1 - x0) / (hi[0] - lo[0]), (y1 - y0) / (hi[1] - lo[1])))
            mids.append((lo + hi) / 2)
        self.scale = float(np.min(sc))
        mids = np.asarray(mids)
        sig = 0.5 * self.fps
        self.mids = np.stack([smooth_track(mids[:, 0], sig), smooth_track(mids[:, 1], sig)], 1)
        self.box_center = ((x0 + x1) / 2, (y0 + y1) / 2)

    def camera(self, f: int) -> Camera:
        az, el = self.angles(f)
        bx, by = self.box_center
        mx, my = self.mids[f]
        return Camera(az, el, self.dist, self.scale, (bx - self.scale * mx, by - self.scale * my))

    # drawing ----------------------------------------------------------------
    def draw(self, f: int):
        cv, th, d = self.cv, self.th, self.data
        cam = self.camera(f)
        L = len(d.layers)
        g = self.growth(f)
        cur = self.cursor(f)
        neg, pos, bg = hex_rgb(th.neg), hex_rgb(th.pos), hex_rgb(th.bg)
        xc = (self.xk(d.cut - 0.5))

        # floor: outline, the HFER band, its cutoff
        def P(x, y, z=0.0):
            return np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                                np.asarray(z, float)), -1)
        floor = cam.project(P([self.X0, self.X1, self.X1, self.X0],
                              [self.Y0, self.Y0, self.Y1, self.Y1]))[0]
        cv.polygons([floor], [rgba(hex_rgb(th.surface2), 0.55)[0]], edge=th.rule,
                    lw=cv.px(1, 0.7), z=1)
        band = cam.project(P([xc, self.X1, self.X1, xc], [self.Y0, self.Y0, self.Y1, self.Y1]))[0]
        cv.polygons([band], [rgba(pos, 0.13)[0]], z=1.2)
        cut = cam.project(P([xc, xc], [self.Y0, self.Y1]))[0]
        cv.lines([cut], [rgba(pos, 0.9)[0]], [cv.px(1.4, 1)], z=1.3, linestyles=(0, (4, 3)))

        # ridges, far first
        depth = cam.project(P(0.0, self.yl(np.arange(L)), 0.3))[1]
        for l in np.argsort(-depth):
            if g[l] <= 0:
                continue
            kk, v = self.curves[l]
            xs, ys = self.xk(kk), np.full(kk.shape, float(self.yl(l)))
            zs = v * self.H * g[l]
            top, _, _ = cam.project(P(xs, ys, zs))
            base, _, _ = cam.project(P(xs, ys, 0.0))
            poly = np.concatenate([top, base[::-1]])
            fresh = abs(l - cur) < 0.5
            cv.polygons([poly], [rgba(bg, 1.0)[0]], z=2)
            edge = d.cut - 0.5
            for mask, col in ((kk <= edge, neg), (kk >= edge, pos)):
                idx = np.flatnonzero(mask)
                if idx.size < 2:
                    continue
                seg = np.concatenate([top[idx], base[idx][::-1]])
                cv.polygons([seg], [rgba(col, 0.20 if th.name == "light" else 0.26)[0]], z=2)
            segs = np.stack([top[:-1], top[1:]], 1)
            mid = (kk[:-1] + kk[1:]) / 2
            cols = np.where((mid >= edge)[:, None], pos, neg)
            w = cv.px(2.6 if fresh else 1.6, 1.0)
            cv.lines(segs, rgba(cols, 1.0 if fresh else 0.85), [w], z=2, capstyle="butt")
            cv.lines([base], [rgba(hex_rgb(th.rule), 0.9)[0]], [cv.px(0.8, 0.6)], z=2)

        self._axes(cam, f)
        self.header(self.title, self.subtitle)
        li = int(round(cur))
        self.counter(str(d.layers[li]), "layer", f"HFER = {d.hfer[li]:.3f}")
        self._spark(cur)
        self._legend()
        self.progress(cur, d.layers)

    def _axes(self, cam: Camera, f: int):
        cv, th, d = self.cv, self.th, self.data
        fs = cv.fs(13, 10.5)
        L = len(d.layers)
        pad = 0.16

        def at(x, y, z=0.0):
            return cam.project(np.array([[x, y, z]]))[0][0]

        x, y = at(self.X0, self.Y0 - pad)
        cv.text(x, y, "smooth modes", fs, th.neg, ha="left", va="top", weight="semibold",
                halo=cv.px(2, 1.5), z=30)
        x, y = at(self.X1, self.Y0 - pad)
        cv.text(x, y, "rough modes · HFER band", fs, th.pos, ha="right", va="top",
                weight="semibold", halo=cv.px(2, 1.5), z=30)
        if not cv.compact:
            x, y = at(0.5 * (self.X0 + self.X1) - 0.15, self.Y0 - pad)
            cv.text(x, y + fs * 1.5, "graph frequency →", fs, th.muted, ha="center",
                    va="top", halo=cv.px(2, 1.5), z=30)
        step = max(1, int(math.ceil(L / 6)))
        ticks = sorted(set(list(range(0, L, step)) + [L - 1]))
        if len(ticks) > 1 and ticks[-1] - ticks[-2] < step / 2:
            ticks.pop(-2)
        for l in ticks:
            x, y = at(self.X0 - pad, float(self.yl(l)))
            cv.text(x, y, str(d.layers[l]), cv.fs(12, 10), th.muted, ha="right",
                    va="center", z=30)
        x, y = at(self.X0 - pad, self.Y1 + 0.35)
        cv.text(x, y, "layer", cv.fs(12, 10), th.muted, ha="right", va="center", z=30)

    def _spark(self, cur: float):
        """HFER by layer, drawn up to the current layer, under the counter."""
        cv, th, d = self.cv, self.th, self.data
        if cv.compact:
            return
        w, h = cv.px(200, 120), cv.px(64, 40)
        x1 = cv.W - self.margin
        x0 = x1 - w
        y0 = self.top + cv.px(40, 20)
        L = len(d.layers)
        hv = np.nan_to_num(d.hfer)
        lo, hi = 0.0, max(float(hv.max()) * 1.1, 1e-9)
        xs = x0 + w * np.arange(L) / max(L - 1, 1)
        ys = y0 + h - h * (hv - lo) / (hi - lo)
        cv.text(x0, y0 - cv.px(8, 5), "HFER by layer", cv.fs(12, 10), th.muted, z=30)
        cv.lines([[(x0, y0 + h), (x1, y0 + h)]], [th.rule], [cv.px(1, 0.7)], z=30)
        cv.lines([np.stack([xs, ys], 1)], [rgba(hex_rgb(th.rule), 1.0)[0]], [cv.px(1.4, 1)], z=30)
        k = int(math.floor(cur))
        e = cur - k
        pts = np.stack([xs[:k + 1], ys[:k + 1]], 1)
        if k + 1 < L:
            pts = np.concatenate([pts, [[xs[k] + e * (xs[k + 1] - xs[k]),
                                         ys[k] + e * (ys[k + 1] - ys[k])]]])
        if len(pts) > 1:
            cv.lines([pts], [th.pos], [cv.px(2.2, 1.4)], z=31)
        cv.circles([pts[-1]], cv.px(4.5, 3), [th.pos], edge=th.bg, lw=cv.px(1.5, 1), z=32)

    def _legend(self):
        cv, th = self.cv, self.th
        y = self.bottom + cv.px(14, 8)
        x = self.margin
        fs = cv.fs(13, 10.5)
        r = cv.px(6, 4)
        cv.circles([(x + r, y)], r, [th.neg], z=50)
        t1 = "below the HFER cutoff"
        cv.text(x + 2 * r + cv.px(7, 4), y, t1, fs, th.ink2, va="center_baseline", z=50)
        w, _ = cv.text_size(t1, fs)
        x2 = x + 2 * r + cv.px(7, 4) + w + cv.px(20, 12)
        cv.circles([(x2 + r, y)], r, [th.pos], z=50)
        t2 = "HFER band (top modes)"
        cv.text(x2 + 2 * r + cv.px(7, 4), y, t2, fs, th.ink2, va="center_baseline", z=50)
        if not cv.compact:
            w2, _ = cv.text_size(t2, fs)
            cv.text(x2 + 2 * r + cv.px(7, 4) + w2 + cv.px(24, 12), y, self.note, fs, th.muted,
                    va="center_baseline", z=50)


def waterfall_html(cache, path, theme, *, layers=None, text: int = 0, scale: str = "linear",
                   plotlyjs: str = "cdn", title: Optional[str] = None, **_ignored):
    """Interactive surface: layer x mode x energy share, HFER band in orange."""
    d = WaterfallData(cache, layers, text, scale)
    L, S = len(d.layers), d.S
    k = np.arange(1, S + 1)
    band = (np.arange(S) >= d.cut).astype(float)
    custom = np.round(d.share * 100, 3)
    surf = {"type": "surface", "x": k.tolist(), "y": list(d.layers), "z": r4(d.value),
            "surfacecolor": [band.tolist()] * L, "cmin": 0, "cmax": 1,
            "colorscale": [[0, theme.neg], [0.5, theme.neg], [0.5, theme.pos], [1, theme.pos]],
            "showscale": False, "opacity": 0.92, "customdata": custom.tolist(),
            "hovertemplate": "layer %{y} · mode %{x}<br>%{customdata:.2f}% of the energy"
                             "<extra></extra>",
            "contours": {"y": {"show": True, "color": theme.ink2, "width": 1,
                               "start": d.layers[0], "end": d.layers[-1], "size": 1}},
            "lighting": {"ambient": 0.85, "diffuse": 0.3, "specular": 0.05}}
    xc = d.cut + 0.5
    plane = {"type": "surface", "x": [[xc, xc], [xc, xc]],
             "y": [[d.layers[0], d.layers[-1]]] * 2, "z": [[0, 0], [1.05, 1.05]],
             "surfacecolor": [[1, 1], [1, 1]], "colorscale": [[0, theme.pos], [1, theme.pos]],
             "showscale": False, "opacity": 0.18, "hoverinfo": "skip"}
    hf = {"type": "scatter3d", "mode": "lines", "x": [S + 0.6] * L, "y": list(d.layers),
          "z": r4(d.hfer), "line": {"color": theme.pos, "width": 5},
          "hovertemplate": "layer %{y}<br>HFER %{z:.3f}<extra></extra>"}
    ax = {"color": theme.muted, "gridcolor": theme.faint, "zerolinecolor": theme.rule,
          "showbackground": False}
    zt = {"linear": "energy share", "sqrt": "√ energy share", "log": "log energy share"}[scale]
    layout = plotly_layout(
        theme, title or "Where the energy sits",
        f"{model_label(d.cache)} · residual energy per graph-frequency mode as a share of each "
        "layer's total; orange: the HFER band; the orange line is HFER by layer",
        scene={"xaxis": dict(ax, title={"text": "mode (ascending eigenvalue)"}),
               "yaxis": dict(ax, title={"text": "layer"}),
               "zaxis": dict(ax, title={"text": zt}),
               "bgcolor": theme.bg, "aspectmode": "manual",
               "aspectratio": {"x": 1.3, "y": 1.5, "z": 0.7},
               "camera": {"eye": {"x": 1.5, "y": -1.6, "z": 0.9}}})
    layout["margin"]["b"] = 30
    fig = {"data": [surf, plane, hf], "layout": layout}
    return write_plotly_html(path, fig, f"waterfall · {model_label(d.cache)}", theme, plotlyjs)
