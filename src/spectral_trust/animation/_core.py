# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Machinery shared by the animated views: themes, a pixel-space drawing
canvas on matplotlib's Agg renderer, a perspective camera, the layer
timeline, temporally stable label placement, and the video and HTML writers.

Everything is drawn in pixel coordinates (origin top left, y down) on one
full-figure axes, so the layout is exact at any size and frames are
deterministic for fixed inputs.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── palette and themes ──────────────────────────────────────────────────────

#: Okabe-Ito colours.
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
}

#: Sans-serif fonts tried in order (matplotlib falls back glyph by glyph).
FONT_FAMILY = ["Inter", "Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


@dataclass(frozen=True)
class Theme:
    """Colours of one theme. ``neg`` and ``pos`` are the two sides of the
    Fiedler split; ``alert`` marks an injected span."""
    name: str
    bg: str
    surface: str
    surface2: str
    ink: str
    ink2: str
    muted: str
    faint: str
    rule: str
    edge: str
    node_mid: str
    node_line: str
    neg: str
    pos: str
    alert: str
    accent: str


THEMES: Dict[str, Theme] = {
    "light": Theme(
        "light", bg="#FBFAF7", surface="#FFFFFF", surface2="#F3F0E9", ink="#16140F",
        ink2="#3D3830", muted="#6B645A", faint="#E9E5DD", rule="#D6D0C5", edge="#3D3830",
        node_mid="#FFFFFF", node_line="#16140F", neg=OKABE_ITO["blue"],
        pos=OKABE_ITO["orange"], alert=OKABE_ITO["vermillion"], accent=OKABE_ITO["green"]),
    # The dark theme lifts the Fiedler blue to Okabe-Ito sky, as the
    # project site does, so that it keeps its contrast on a dark ground.
    "dark": Theme(
        "dark", bg="#0F1115", surface="#161A20", surface2="#1C2129", ink="#ECEBE7",
        ink2="#C8C5BD", muted="#9C978D", faint="#20252D", rule="#353B45", edge="#C8C5BD",
        node_mid="#C8C5BD", node_line="#0F1115", neg=OKABE_ITO["sky"],
        pos=OKABE_ITO["orange"], alert="#E8661A", accent=OKABE_ITO["green"]),
}


def get_theme(theme) -> Theme:
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[str(theme)]
    except KeyError:
        raise ValueError(f"theme must be one of {sorted(THEMES)}, got {theme!r}") from None


def hex_rgb(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def mix(a: str, b: str, t) -> np.ndarray:
    """RGB of ``a`` blended toward ``b`` by ``t`` (scalar or array)."""
    t = np.asarray(t, dtype=float)[..., None]
    return hex_rgb(a) * (1 - t) + hex_rgb(b) * t


def ramp(stops: Sequence[str], t) -> np.ndarray:
    """RGB at ``t`` in [0, 1] on a piecewise-linear ramp (vectorised)."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    rgb = np.stack([hex_rgb(s) for s in stops])
    return rgb[i] * (1 - f) + rgb[i + 1] * f


def to_hex(rgb) -> str:
    r = np.clip(np.asarray(rgb, dtype=float), 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02X}" for v in r[:3])


def rgba(rgb, alpha) -> np.ndarray:
    """``[n, 4]`` RGBA from ``[n, 3]`` (or one) RGB and ``[n]`` (or one) alpha."""
    rgb = np.asarray(rgb, dtype=float)
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    if rgb.ndim == 1:
        rgb = np.broadcast_to(rgb, (alpha.shape[0], 3))
    alpha = np.broadcast_to(alpha, (rgb.shape[0],))
    return np.concatenate([rgb, np.clip(alpha, 0, 1)[:, None]], axis=1)


# ── easing and timeline ─────────────────────────────────────────────────────

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def layer_positions(n_frames: int, n_layers: int, intro: int = 0, outro: int = 0,
                    hold: float = 0.3, ease=None) -> np.ndarray:
    """Fractional layer position of every frame: ``intro`` frames on the
    first layer, then each transition holds on its layer for ``hold`` of
    its time and eases into the next, then ``outro`` frames on the last."""
    pos = np.zeros(n_frames)
    if n_layers <= 1 or n_frames <= 0:
        return pos
    moving = max(n_frames - intro - outro, 1)
    for f in range(n_frames):
        if f < intro:
            pos[f] = 0.0
        elif f >= n_frames - outro:
            pos[f] = n_layers - 1
        else:
            u = (f - intro) / moving * (n_layers - 1)
            i = min(int(u), n_layers - 2)
            frac = u - i
            pos[f] = i + float((ease or smootherstep)((frac - hold) / (1 - hold)))
    return pos


def smooth_track(x, sigma: float, low: Optional[bool] = None) -> np.ndarray:
    """Gaussian-smoothed copy of a per-frame series. ``low=True`` first takes
    a running minimum over three sigmas (the result stays at or below the
    series, as a zoom that must never overflow needs); ``low=False`` a
    running maximum."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 or sigma <= 0:
        return x.copy()
    if low is not None:
        w = int(math.ceil(3 * sigma))
        pad = np.pad(x, w, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
        x = win.min(1) if low else win.max(1)
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def split_position(p: float, n: int) -> Tuple[int, int, float]:
    """``(i, j, e)``: the two layers around position ``p`` and the blend."""
    i = int(min(max(math.floor(p), 0), n - 1))
    j = min(i + 1, n - 1)
    return i, j, float(p - i) if j != i else 0.0


# ── camera ──────────────────────────────────────────────────────────────────

class Camera:
    """A perspective camera orbiting the origin; world z is up.

    ``azim`` and ``elev`` in degrees; ``dist`` is the eye distance in world
    units; ``scale`` maps world units to pixels at the origin's depth.
    """

    def __init__(self, azim: float, elev: float, dist: float, scale: float,
                 center: Tuple[float, float]):
        a, e = math.radians(azim), math.radians(elev)
        self.d = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        z = np.array([0.0, 0.0, 1.0])
        r = np.cross(z, self.d)
        self.r = r / (np.linalg.norm(r) or 1.0)
        self.u = np.cross(self.r, -self.d)
        self.dist, self.scale, self.center = float(dist), float(scale), center
        self.elev = e

    def project(self, P: np.ndarray):
        """Screen ``[n, 2]`` pixels, depth ``[n]`` (larger is farther) and
        the perspective factor ``[n]`` of world points ``[n, 3]``."""
        P = np.atleast_2d(np.asarray(P, dtype=float))
        xc, yc, zc = P @ self.r, P @ self.u, P @ self.d
        k = self.dist / np.maximum(self.dist - zc, 1e-3)
        cx, cy = self.center
        xy = np.stack([cx + self.scale * k * xc, cy - self.scale * k * yc], axis=1)
        return xy, -zc, k


# ── canvas ──────────────────────────────────────────────────────────────────

def installed_fonts(families) -> List[str]:
    """The families of ``families`` that matplotlib can find, in order,
    ending with DejaVu Sans (matplotlib's own) for missing glyphs."""
    from matplotlib import font_manager

    if isinstance(families, str):
        families = [families]
    have = {f.name for f in font_manager.fontManager.ttflist}
    out = [f for f in families if f in have]
    if "DejaVu Sans" not in out:
        out.append("DejaVu Sans")
    return out


class Canvas:
    """One frame-sized matplotlib figure drawn in pixel coordinates."""

    def __init__(self, width: int, height: int, dpi: int, theme: Theme,
                 font: Optional[Sequence[str]] = None):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.W, self.H, self.dpi = int(width), int(height), int(dpi)
        self.theme = theme
        self.font = installed_fonts(font or FONT_FAMILY)
        self.fig = Figure(figsize=(self.W / self.dpi, self.H / self.dpi), dpi=self.dpi,
                          facecolor=theme.bg)
        self.agg = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        #: Layout scale relative to a 1280 x 720 frame.
        self.s = min(self.W / 1280.0, self.H / 720.0)
        self.compact = self.W < 900
        self._measure: Dict[Tuple[str, float, str], Tuple[float, float]] = {}
        self.begin()

    # sizes ------------------------------------------------------------------
    def px(self, base: float, floor: float = 0.0) -> float:
        """A length of ``base`` pixels at 1280 x 720, scaled to this frame,
        never below ``floor``."""
        return max(base * self.s, floor)

    def fs(self, base: float, floor: float = 11.0) -> float:
        """Font size in pixels: scaled, but kept legible on small frames."""
        if self.compact:
            floor = max(floor, 11.5)
        return max(base * self.s, floor)

    def pt(self, px: float) -> float:
        return px * 72.0 / self.dpi

    # frame ------------------------------------------------------------------
    def begin(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)
        ax.set_axis_off()
        ax.set_facecolor(self.theme.bg)
        ax.patch.set_visible(False)

    def grab(self) -> np.ndarray:
        self.agg.draw()
        buf = np.asarray(self.agg.buffer_rgba())
        return np.ascontiguousarray(buf[:, :, :3])

    def close(self):
        self.fig.clear()

    # primitives -------------------------------------------------------------
    def text_size(self, s: str, size_px: float, weight: str = "normal") -> Tuple[float, float]:
        """Width and height in pixels of a single-line string."""
        key = (s, round(size_px, 2), weight)
        if key not in self._measure:
            from matplotlib.font_manager import FontProperties
            fp = FontProperties(family=self.font, size=self.pt(size_px), weight=weight)
            r = self.agg.get_renderer()
            w, h, d = r.get_text_width_height_descent(s, fp, ismath=False)
            self._measure[key] = (float(w), float(h))
        return self._measure[key]

    def text(self, x, y, s, size, color=None, ha="left", va="baseline", weight="normal",
             alpha=1.0, halo: float = 0.0, z=10, rotation=0.0, family=None):
        import matplotlib.patheffects as pe

        kw = {}
        if halo > 0:
            kw["path_effects"] = [pe.withStroke(linewidth=self.pt(halo) * 2,
                                                foreground=self.theme.bg)]
        return self.ax.text(x, y, s, fontsize=self.pt(size), color=color or self.theme.ink,
                            ha=ha, va=va, fontweight=weight, alpha=alpha, zorder=z,
                            rotation=rotation, rotation_mode="anchor",
                            fontfamily=family or self.font, clip_on=False, **kw)

    def circles(self, xy, r, face, edge=None, lw=0.0, z=5):
        """Filled circles of pixel radius ``r`` at ``xy`` (drawn in order)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        if xy.shape[0] == 0:
            return None
        r = np.broadcast_to(np.asarray(r, dtype=float), (xy.shape[0],))
        return self.ellipses(xy, 2 * r, 2 * r, face, edge, lw, z)

    def ellipses(self, xy, w, h, face, edge=None, lw=0.0, z=5):
        from matplotlib.collections import EllipseCollection

        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        n = xy.shape[0]
        coll = EllipseCollection(np.broadcast_to(w, (n,)), np.broadcast_to(h, (n,)),
                                 np.zeros(n), units="xy", offsets=xy,
                                 offset_transform=self.ax.transData,
                                 facecolors=face, edgecolors=edge if edge is not None else "none",
                                 linewidths=self.pt(lw) if lw else 0.0, zorder=z)
        self.ax.add_collection(coll)
        return coll

    def lines(self, segments, colors, widths, z=3, capstyle="round", linestyles="solid"):
        from matplotlib.collections import LineCollection

        if len(segments) == 0:
            return None
        widths = np.broadcast_to(np.asarray(widths, dtype=float), (len(segments),))
        coll = LineCollection(segments, colors=colors, linewidths=self.pt(1) * widths,
                              zorder=z, capstyle=capstyle, joinstyle="round",
                              linestyles=linestyles)
        self.ax.add_collection(coll)
        return coll

    def polygons(self, polys, face, edge="none", lw=0.0, z=2):
        from matplotlib.collections import PolyCollection

        if len(polys) == 0:
            return None
        coll = PolyCollection(polys, facecolors=face, edgecolors=edge,
                              linewidths=self.pt(lw) if lw else 0.0, zorder=z)
        self.ax.add_collection(coll)
        return coll

    def rect(self, x, y, w, h, face, edge="none", lw=0.0, z=1, radius=0.0, alpha=1.0):
        from matplotlib.patches import FancyBboxPatch

        style = f"round,pad=0,rounding_size={radius}" if radius else "square,pad=0"
        p = FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=face, edgecolor=edge,
                           linewidth=self.pt(lw) if lw else 0.0, zorder=z, alpha=alpha,
                           mutation_aspect=1)
        self.ax.add_patch(p)
        return p


# ── labels ──────────────────────────────────────────────────────────────────

_DIRS = np.array([[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]],
                 dtype=float)


class LabelPlacer:
    """Places one text label beside each point so that labels overlap
    neither one another nor any point, and stay put from frame to frame.

    Each label picks the cheapest of 16 candidate boxes (8 directions, 2
    distances) around its point, greedily in priority order; the cost adds
    overlaps with labels already placed, with every point and with the
    frame edge, a small preference for pointing away from the centre, and a
    hysteresis term for leaving last frame's choice. The drawn offset eases
    toward the chosen one, so a label that has to move glides.
    """

    def __init__(self, sizes: Sequence[Tuple[float, float]], gap: float,
                 bounds: Tuple[float, float, float, float], ease: float = 0.35):
        self.sizes = np.asarray(sizes, dtype=float).reshape(-1, 2)
        self.gap = float(gap)
        self.bounds = bounds
        self.ease = float(ease)
        self.choice: Optional[np.ndarray] = None
        self.offset: Optional[np.ndarray] = None

    def _candidates(self, i, p, r):
        w, h = self.sizes[i]
        out = []
        for ring in (0, 1, 2):
            extra = ring * (h * 1.15 + self.gap)
            for d in _DIRS:
                dx, dy = d / (np.linalg.norm(d) or 1)
                cx = p[0] + dx * (r + self.gap + extra) + d[0] * w / 2
                cy = p[1] + dy * (r + self.gap + extra) + d[1] * h / 2
                out.append((cx, cy))
        return np.asarray(out)

    def place(self, pts: np.ndarray, radii: np.ndarray, order: Sequence[int],
              center: Optional[Tuple[float, float]] = None, visible=None):
        """Label centres ``[n, 2]`` and whether each one sits on the outer
        ring (draw a leader line)."""
        n = len(pts)
        pts = np.asarray(pts, dtype=float)
        radii = np.broadcast_to(np.asarray(radii, dtype=float), (n,))
        visible = np.ones(n, bool) if visible is None else np.asarray(visible, bool)
        if self.choice is None:
            self.choice = np.full(n, -1)
        c0 = np.asarray(center if center is not None else pts.mean(0))
        x0, y0, x1, y1 = self.bounds
        placed: List[Tuple[float, float, float, float]] = []
        target = np.zeros((n, 2))
        ring = np.zeros(n, bool)
        for i in order:
            if not visible[i]:
                continue
            w, h = self.sizes[i]
            cand = self._candidates(i, pts[i], radii[i])
            out = pts[i] - c0
            out = out / (np.linalg.norm(out) or 1)
            best, best_cost = 0, np.inf
            pad = 0.4 * h
            for k, (cx, cy) in enumerate(cand):
                bx0, by0, bx1, by1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                cost = 0.0
                for (qx0, qy0, qx1, qy1) in placed:
                    ox = min(bx1 + pad, qx1) - max(bx0 - pad, qx0)
                    oy = min(by1 + pad, qy1) - max(by0 - pad, qy0)
                    if ox > 0 and oy > 0:
                        cost += 60.0 + 400.0 * ox * oy / (w * h)
                # points: nearest point of the box to each circle centre
                nx = np.clip(pts[visible, 0], bx0, bx1)
                ny = np.clip(pts[visible, 1], by0, by1)
                dd = np.hypot(nx - pts[visible, 0], ny - pts[visible, 1])
                hit = radii[visible] + 1.5 - dd
                if np.any(hit > 0):
                    cost += 80.0 + 40.0 * float(np.sum(np.clip(hit, 0, None)))
                spill = (max(x0 - bx0, 0) + max(bx1 - x1, 0) + max(y0 - by0, 0)
                         + max(by1 - y1, 0))
                cost += 30.0 * spill
                d = _DIRS[k % 8] / np.linalg.norm(_DIRS[k % 8])
                cost += 1.2 * (1 - float(d @ out))
                cost += 4.0 * (k // 8)
                if self.choice[i] >= 0 and k != self.choice[i]:
                    cost += 6.0
                if cost < best_cost:
                    best, best_cost = k, cost
            self.choice[i] = best
            target[i] = cand[best]
            ring[i] = best >= 8
            cx, cy = cand[best]
            placed.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        off = target - pts
        if self.offset is None:
            self.offset = off
        else:
            eased = self.offset + self.ease * (off - self.offset)
            # an eased label that would collide jumps straight to its target
            bad = self.collisions(pts + eased, pts, radii, order, visible)
            eased[bad] = off[bad]
            self.offset = eased
        centres = pts + self.offset
        # In a dense cluster the label of lower priority (farther away) is
        # hidden rather than allowed to touch another or cover a point; it
        # returns once it has been free for a few frames, fading in.
        bad = self.collisions(centres, pts, radii, order, visible)
        if self.alpha is None:
            self.alpha = np.where(bad, 0.0, 1.0)
            self.free = np.where(bad, 0, self.REAPPEAR)
        else:
            self.free = np.where(bad, 0, self.free + 1)
            grow = np.minimum(self.alpha + 0.25, 1.0)
            self.alpha = np.where(bad, 0.0, np.where((self.alpha > 0) |
                                                     (self.free >= self.REAPPEAR), grow, 0.0))
        self.alpha = np.where(visible, self.alpha, 0.0)
        return centres, ring

    alpha: Optional[np.ndarray] = None
    free: Optional[np.ndarray] = None
    #: Frames a hidden label must be free before it fades back in.
    REAPPEAR = 3

    def collisions(self, centres, pts, radii, order, visible=None,
                   gap: Optional[float] = None) -> np.ndarray:
        """Labels that overlap a point, or come within ``gap`` pixels (default
        a fifth of the label height) of a label of higher priority."""
        n = len(centres)
        visible = np.ones(n, bool) if visible is None else np.asarray(visible, bool)
        radii = np.broadcast_to(np.asarray(radii, dtype=float), (n,))
        half = self.sizes / 2
        lo, hi = centres - half, centres + half
        g = 0.2 * float(np.median(self.sizes[:, 1])) if gap is None else float(gap)
        bad = np.zeros(n, bool)
        kept: List[int] = []
        for i in order:
            if not visible[i]:
                continue
            nx = np.clip(pts[visible, 0], lo[i, 0], hi[i, 0])
            ny = np.clip(pts[visible, 1], lo[i, 1], hi[i, 1])
            hit = bool(np.any(np.hypot(nx - pts[visible, 0], ny - pts[visible, 1])
                              < radii[visible]))
            for k in kept:
                if (min(hi[i, 0], hi[k, 0]) + g > max(lo[i, 0], lo[k, 0])
                        and min(hi[i, 1], hi[k, 1]) + g > max(lo[i, 1], lo[k, 1])):
                    hit = True
                    break
            bad[i] = hit
            if not hit:
                kept.append(i)
        return bad


def display_token(t: str, limit: int = 14) -> str:
    """Token text for a label: stripped, with visible marks for whitespace
    tokens and an ellipsis past ``limit`` characters."""
    s = str(t).strip()
    if not s:
        s = "↵" if "\n" in str(t) else "␣"
    s = s.replace("\n", "↵")
    return s if len(s) <= limit else s[:limit - 1] + "…"


# ── writers ─────────────────────────────────────────────────────────────────

VIDEO_SUFFIXES = (".mp4", ".webm", ".gif")


def need_imageio():
    """imageio's v2 API, after checking that imageio-ffmpeg is installed too."""
    from importlib.util import find_spec

    try:
        import imageio.v2 as iio
        if find_spec("imageio_ffmpeg") is None:
            raise ImportError("imageio_ffmpeg")
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "writing .mp4, .webm or .gif needs imageio and imageio-ffmpeg: "
            "pip install 'spectral_trust[viz]'") from exc
    return iio


class FrameWriter:
    """Streams RGB frames to ffmpeg: ``.mp4`` (H.264) and ``.webm`` (VP9)
    through imageio's ffmpeg writer, ``.gif`` through imageio-ffmpeg's frame
    writer (imageio's refuses GIF) with a palette computed over all frames.
    The encoder runs on at most ``threads`` threads."""

    def __init__(self, path: Path, fps: float, threads: int = 2, gif_colors: int = 128,
                 crf: Optional[int] = None, gif_dither: str = "bayer:bayer_scale=4"):
        iio = need_imageio()
        self.path = Path(path)
        ext = self.path.suffix.lower()
        t = str(max(int(threads), 1))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._w = self._gif = None
        self._fps, self._colors, self._t = fps, int(gif_colors), t
        self._dither = gif_dither
        if ext == ".mp4":
            kw = dict(codec="libx264", pixelformat="yuv420p", output_params=[
                "-crf", str(18 if crf is None else crf), "-preset", "medium",
                "-tune", "animation", "-movflags", "+faststart", "-threads", t])
        elif ext == ".webm":
            kw = dict(codec="libvpx-vp9", pixelformat="yuv420p", output_params=[
                "-crf", str(32 if crf is None else crf), "-b:v", "0", "-deadline", "good",
                "-cpu-used", "2", "-row-mt", "1", "-threads", t])
        elif ext == ".gif":
            return                                 # opened on the first frame (needs its size)
        else:
            raise ValueError(f"unsupported video suffix {ext!r}; use one of {VIDEO_SUFFIXES}")
        self._w = iio.get_writer(str(self.path), format="FFMPEG", mode="I", fps=fps,
                                 quality=None, macro_block_size=1,
                                 ffmpeg_log_level="error", **kw)

    def _open_gif(self, h: int, w: int):
        import imageio_ffmpeg

        vf = (f"split[a][b];[a]palettegen=max_colors={self._colors}:stats_mode=full[p];"
              f"[b][p]paletteuse=dither={self._dither}:diff_mode=rectangle")
        self._gif = imageio_ffmpeg.write_frames(
            str(self.path), (w, h), pix_fmt_in="rgb24", pix_fmt_out="pal8", fps=self._fps,
            quality=None, codec="gif", macro_block_size=1, ffmpeg_log_level="error",
            output_params=["-vf", vf, "-loop", "0", "-threads", self._t])
        self._gif.send(None)

    def append(self, frame: np.ndarray):
        if self._w is not None:
            self._w.append_data(frame)
            return
        if self._gif is None:
            self._open_gif(*frame.shape[:2])
        self._gif.send(np.ascontiguousarray(frame, dtype=np.uint8))

    def close(self):
        if self._w is not None:
            self._w.close()
        if self._gif is not None:
            self._gif.close()


#: plotly.js version of the ``plotlyjs="cdn"`` script tag.
PLOTLY_JS_VERSION = "3.1.0"
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js"


def _plotly_available() -> bool:
    from importlib.util import find_spec

    return find_spec("plotly") is not None


def _plotly_script(plotlyjs: str) -> str:
    if plotlyjs == "auto":
        plotlyjs = "embed" if _plotly_available() else "cdn"
    if plotlyjs == "cdn":
        return f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
    if plotlyjs == "embed":
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as exc:
            raise ImportError("plotlyjs='embed' copies plotly.js from the plotly package: "
                              "pip install plotly, or use plotlyjs='cdn'") from exc
        return f"<script>{get_plotlyjs()}</script>"
    if plotlyjs.startswith(("http://", "https://")):
        return f'<script src="{plotlyjs}" charset="utf-8"></script>'
    p = Path(plotlyjs)
    if p.is_file():
        return f"<script>{p.read_text(encoding='utf-8')}</script>"
    raise ValueError("plotlyjs must be 'auto', 'cdn', 'embed', a URL or the path of "
                     "plotly.min.js")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(type(o).__name__)


def write_plotly_html(path: Path, fig: dict, title: str, theme: Theme, plotlyjs: str = "auto",
                      note: str = "") -> Path:
    """A standalone page that draws ``fig`` (a plotly figure as a dict with
    ``data``, ``layout`` and optional ``frames``) with plotly.js."""
    import html

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(fig, default=_json_default, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("</", "<\\/")
    font = ", ".join(f"'{f}'" for f in FONT_FAMILY) + ", sans-serif"
    page = (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title>"
        f"<style>html,body{{margin:0;height:100%;background:{theme.bg};color:{theme.ink};"
        f"font-family:{font}}}#st-plot{{width:100%;height:100vh}}"
        f".st-note{{position:fixed;left:16px;bottom:10px;font-size:12px;color:{theme.muted};"
        "pointer-events:none}</style>"
        f"{_plotly_script(plotlyjs)}</head><body><div id=\"st-plot\"></div>"
        + (f"<div class=\"st-note\">{html.escape(note)}</div>" if note else "")
        + "<script>(function(){var fig=" + payload + ";"
        "var el=document.getElementById('st-plot');"
        "Plotly.newPlot(el,fig.data,fig.layout,{responsive:true,displaylogo:false,"
        "modeBarButtonsToRemove:['sendDataToCloud']}).then(function(){"
        "if(fig.frames&&fig.frames.length){Plotly.addFrames(el,fig.frames);}});})();"
        "</script></body></html>\n")
    path.write_text(page, encoding="utf-8")
    return path


def plotly_player(frame_names: Sequence[str], labels: Sequence[str], theme: Theme,
                  frame_ms: int = 60, redraw: bool = True, prefix: str = "layer ",
                  y: float = 0.0) -> Tuple[list, list]:
    """``updatemenus`` (play and pause) and ``sliders`` for a plotly figure
    whose frames are named ``frame_names``; ``labels`` are the slider ticks."""
    anim = {"frame": {"duration": frame_ms, "redraw": redraw}, "fromcurrent": True,
            "transition": {"duration": 0}, "mode": "immediate"}
    font = {"color": theme.ink2, "size": 12}
    menus = [{
        "type": "buttons", "direction": "left", "showactive": False, "x": 0.0, "y": y,
        "xanchor": "left", "yanchor": "top", "pad": {"t": 4, "r": 8},
        "bgcolor": theme.surface2, "bordercolor": theme.rule, "font": font,
        "buttons": [
            {"label": "▶ play", "method": "animate", "args": [None, anim]},
            {"label": "❚❚", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": redraw},
                               "mode": "immediate", "transition": {"duration": 0}}]},
        ]}]
    steps = [{"label": lab, "method": "animate",
              "args": [[name], {"frame": {"duration": 0, "redraw": redraw},
                                "mode": "immediate", "transition": {"duration": 0}}]}
             for name, lab in zip(frame_names, labels)]
    sliders = [{"active": 0, "x": 0.12, "y": y, "len": 0.86, "xanchor": "left",
                "yanchor": "top", "pad": {"t": 0, "b": 0}, "steps": steps,
                "currentvalue": {"prefix": prefix, "font": {"color": theme.ink, "size": 14}},
                "font": font, "bgcolor": theme.surface2, "bordercolor": theme.rule,
                "activebgcolor": theme.pos, "tickcolor": theme.rule}]
    return menus, sliders


def plotly_layout(theme: Theme, title: str, subtitle: str = "", **extra) -> dict:
    font = {"family": ", ".join(FONT_FAMILY), "color": theme.ink, "size": 13}
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:13px;color:{theme.muted}'>{subtitle}</span>"
    lay = {"paper_bgcolor": theme.bg, "plot_bgcolor": theme.bg, "font": font,
           "title": {"text": text, "x": 0.02, "xanchor": "left", "y": 0.97},
           "margin": {"l": 24, "r": 24, "t": 76, "b": 90}, "showlegend": False,
           "hoverlabel": {"bgcolor": theme.surface, "bordercolor": theme.rule,
                          "font": {"color": theme.ink, "family": font["family"]}}}
    lay.update(extra)
    return lay


def hexa(c: str, a: float) -> str:
    """CSS ``rgba()`` of a hex colour."""
    r, g, b = (int(v * 255) for v in hex_rgb(c))
    return f"rgba({r},{g},{b},{a:.3f})"


def r4(a, nd: int = 4):
    """Rounded nested lists (smaller HTML)."""
    return np.round(np.asarray(a, dtype=float), nd).tolist()


__all__ = [
    "Camera", "Canvas", "FrameWriter", "LabelPlacer", "OKABE_ITO", "THEMES", "Theme",
    "display_token", "get_theme", "layer_positions", "mix", "ramp", "rgba", "smootherstep",
    "smoothstep", "split_position", "to_hex", "write_plotly_html", "plotly_player",
    "plotly_layout", "hexa", "r4",
]
