# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
``flow``: each token as a glowing tube through depth. The tube runs along
the layer axis; its cross-section position at a layer is the token's
spectral coordinates there (Laplacian eigenvectors 2 and 3 of the layer's
head-averaged attention graph, sign-fixed and Procrustes-aligned from layer
to layer, joined by a Catmull-Rom spline); its colour is the token's side of
the Fiedler split at that layer; its thickness is the attention the token
receives, so an attention sink becomes a trunk.

The scene is drawn by three.js (r160, shipped with the package) both in the
interactive ``.html`` page and, for video, in headless Chrome, frame by frame
at a fixed time step; a matplotlib renderer of the same scene and camera is
the fallback when no browser is available.
"""
import difflib
import json
import math
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ._core import FrameWriter, Theme, display_token, hex_rgb
from ._data import align_sequence, layer_graphs, match_signs, model_label, pick_layers, single

ASSETS = Path(__file__).resolve().parent / "assets" / "three"
THREE_VERSION = "0.160.0"
THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"
_MODULES = {
    "three": "three.module.min.js",
    "three/addons/controls/OrbitControls.js": "addons/controls/OrbitControls.js",
    "three/addons/postprocessing/EffectComposer.js": "addons/postprocessing/EffectComposer.js",
    "three/addons/postprocessing/RenderPass.js": "addons/postprocessing/RenderPass.js",
    "three/addons/postprocessing/ShaderPass.js": "addons/postprocessing/ShaderPass.js",
    "three/addons/postprocessing/MaskPass.js": "addons/postprocessing/MaskPass.js",
    "three/addons/postprocessing/Pass.js": "addons/postprocessing/Pass.js",
    "three/addons/postprocessing/UnrealBloomPass.js": "addons/postprocessing/UnrealBloomPass.js",
    "three/addons/shaders/CopyShader.js": "addons/shaders/CopyShader.js",
    "three/addons/shaders/LuminosityHighPassShader.js": "addons/shaders/LuminosityHighPassShader.js",
}


# ── data ────────────────────────────────────────────────────────────────────

class FlowData:
    """Per-token tracks through the layers, in world units: x is the layer,
    (y, z) the aligned spectral coordinates."""

    def __init__(self, cache, layers=None, text: Optional[int] = None, other=None,
                 highlight=None, axes: str = "fiedler", align: str = "procrustes",
                 smooth: float = 0.8, bundle: float = 0.45, spread: float = 2.6,
                 step: float = 1.0, labels="auto"):
        if text is None:
            text = (cache.n_texts - 1) if (cache.is_batch and other is None) else 0
        c = single(cache, text)
        self.cache = c
        self.layers = pick_layers(c, layers)
        L = len(self.layers)
        gs = layer_graphs(c, self.layers)
        self.raw_tokens = gs[0].tokens
        self.tokens = [display_token(t) for t in self.raw_tokens]
        S = len(self.tokens)
        if axes not in ("fiedler", "procrustes"):
            raise ValueError("axes must be 'fiedler' or 'procrustes'")
        U = np.stack(match_signs([g.fiedler for g in gs]))            # [L, S]
        m = np.max(np.abs(U), axis=1, keepdims=True)
        self.u = U / np.where(m > 0, m, 1.0)                          # in [-1, 1]
        maps = []
        if axes == "procrustes":
            # eigenvectors 2 and 3, sign-fixed and rotated onto the previous layer
            for g in gs:
                V = g.vectors[:, 1:3]
                if V.shape[1] < 2:
                    V = np.concatenate([V, np.zeros((S, 2 - V.shape[1]))], 1)
                V = V - V.mean(0)
                rms = math.sqrt(float(np.mean(np.sum(V ** 2, 1)))) or 1.0
                maps.append(V / rms)
            A = np.stack(align_sequence(maps, align))
        else:
            # the Fiedler vector upright (its sign matched layer to layer), so
            # the split runs above and below the axis; eigenvector 3 in depth
            third = [g.vectors[:, 2] if g.vectors.shape[1] > 2 else np.zeros(S) for g in gs]
            third = match_signs(third)
            for l in range(L):
                y = np.sign(self.u[l]) * np.abs(self.u[l]) ** 0.6
                z = third[l] / (np.max(np.abs(third[l])) or 1.0)
                z = np.sign(z) * np.abs(z) ** 0.6
                V = np.stack([y, z], 1)
                rms = math.sqrt(float(np.mean(np.sum(V ** 2, 1)))) or 1.0
                maps.append(V / rms)
            A = np.stack(maps)
        if bundle > 0:
            # display only: pull each token toward the mean of its side of the
            # split at that layer, in proportion to how clearly it is on it
            strength = np.clip(np.abs(self.u) / 0.2, 0, 1)
            for l in range(L):
                side = self.u[l] >= 0
                for sd in (True, False):
                    mk = side == sd
                    if mk.sum() > 1:
                        centre = A[l, mk].mean(0)
                        b = (bundle * strength[l, mk])[:, None]
                        A[l, mk] = (1 - b) * A[l, mk] + b * centre
        if smooth > 0 and L > 2:
            A = _gauss_layers(A, smooth)
        self.A = A                                                    # [L, S, 2]
        recv = []
        span = c.token_span
        for lay in self.layers:
            pat = np.asarray(c[f"blocks.{lay}.attn.pattern"], dtype=np.float64)
            if span is not None:
                pat = pat[:, span[0]:span[1], span[0]:span[1]]
            recv.append(pat.mean(0).sum(0))                           # received, per key
        self.recv = np.stack(recv)                                    # [L, S]
        self.radius = 0.035 + 0.17 * np.sqrt(self.recv / (self.recv.max() or 1.0))
        self.lam2 = np.array([g.lambda2 for g in gs])
        self.step, self.spread = float(step), float(spread)
        self.X = np.arange(L) * self.step
        P = np.zeros((S, L, 3))
        P[:, :, 0] = self.X[None, :]
        P[:, :, 1] = (A[:, :, 0] * spread).T
        P[:, :, 2] = (A[:, :, 1] * spread).T
        self.P = P                                                    # [S, L, 3]
        self.R = float(np.max(np.linalg.norm(P[:, :, 1:], axis=2))) + 0.3
        self.highlight = self._highlight(highlight, other)
        self.labels = self._labels(labels)

    def _highlight(self, highlight, other) -> List[int]:
        S = len(self.tokens)
        if highlight is not None:
            if isinstance(highlight, slice):
                return list(range(S))[highlight]
            if isinstance(highlight, tuple) and len(highlight) == 2:
                return list(range(*highlight))
            return [int(i) for i in highlight]
        if other is not None:
            o = single(other, 0)
            sm = difflib.SequenceMatcher(a=list(o.tokens), b=list(self.raw_tokens),
                                         autojunk=False)
            out = []
            for tag, _, _, j1, j2 in sm.get_opcodes():
                if tag in ("insert", "replace"):
                    out += list(range(j1, j2))
            return out
        return []

    def _labels(self, labels) -> List[int]:
        if labels in (None, "none", False):
            return []
        if labels != "auto":
            return sorted({int(i) for i in labels})
        S = len(self.tokens)
        mean_recv = self.recv.mean(0)
        keep = {int(np.argmax(mean_recv)), 0, S - 1}
        if self.highlight:
            keep |= {self.highlight[0], self.highlight[-1]}
        # a few strongly attended content tokens, spread along the text
        order = [int(i) for i in np.argsort(-mean_recv) if self.tokens[int(i)].isalnum()]
        for i in order:
            if len(keep) >= 7 + (2 if self.highlight else 0):
                break
            if all(abs(i - k) > max(2, S // 12) for k in keep):
                keep.add(i)
        return sorted(keep)


def _gauss_layers(A: np.ndarray, sigma: float) -> np.ndarray:
    """Light smoothing along the layer axis (edges held)."""
    L = A.shape[0]
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(A[:1], r, 0), A, np.repeat(A[-1:], r, 0)])
    out = np.zeros_like(A)
    for i in range(L):
        out[i] = np.tensordot(k, pad[i:i + 2 * r + 1], axes=(0, 0))
    return out


# ── camera ──────────────────────────────────────────────────────────────────

def camera_path(d: FlowData, n_frames: int, aspect: float, fov: float = 40.0):
    """Eased camera through the braid: close at layer 0, a dolly along the
    depth axis while turning about it, a pull back to the whole braid and a
    slow turn. Keyframes are joined by a cubic Hermite spline (zero speed at
    both ends). Returns positions, targets ``[n, 3]`` and the overview
    distance."""
    X1 = float(d.X[-1]) if len(d.X) else 0.0
    R = d.R
    mid = X1 / 2
    vf = math.radians(fov)
    hf = 2 * math.atan(math.tan(vf / 2) * aspect)
    D = max((X1 / 2 + 1.2 * R) / math.tan(hf / 2), (1.6 * R) / math.tan(vf / 2)) * 0.98

    def around(x, ang, rad):
        a = math.radians(ang)
        return [x, rad * math.sin(a), rad * math.cos(a)]

    def turn(ang):
        a = math.radians(ang)
        off = np.array([0.0, 0.42 * D, D])
        rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        return list(np.array([mid, 0.0, 0.0]) + rot @ off)

    look = min(6.0, X1 * 0.3 + 1)
    keys = [
        (0.00, [-3.6, 0.8 * R, 1.25 * R], [look, 0, 0]),
        (0.10, around(0.6, 35, 1.25 * R), [look + 0.6, 0, 0]),
        (0.30, around(0.33 * X1, 5, 1.35 * R), [0.33 * X1 + look, 0, 0]),
        (0.50, around(0.66 * X1, -35, 1.45 * R), [0.66 * X1 + look, 0, 0]),
        (0.64, around(0.92 * X1, -55, 1.9 * R), [X1, 0, 0]),
        (0.80, turn(30), [mid, 0, 0]),
        (1.00, turn(-22), [mid, 0, 0]),
    ]
    ts = np.array([k[0] for k in keys])
    pos = np.array([k[1] for k in keys], dtype=float)
    tgt = np.array([k[2] for k in keys], dtype=float)
    t = np.linspace(0, 1, max(n_frames, 1))
    look_at = _hermite(ts, tgt, t)
    # the camera always looks at a point of the braid (short models included)
    look_at[:, 0] = np.clip(look_at[:, 0], 0.0, X1)
    look_at[:, 1:] = np.clip(look_at[:, 1:], -R, R)
    return _hermite(ts, pos, t), look_at, D


def _hermite(ts, P, t):
    n = len(ts)
    m = np.zeros_like(P)
    for k in range(1, n - 1):
        m[k] = (P[k + 1] - P[k - 1]) / (ts[k + 1] - ts[k - 1])
    out = np.zeros((len(t), P.shape[1]))
    for q, tq in enumerate(t):
        k = int(np.clip(np.searchsorted(ts, tq, side="right") - 1, 0, n - 2))
        h = ts[k + 1] - ts[k]
        s = (tq - ts[k]) / h
        h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
        h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        out[q] = h00 * P[k] + h10 * h * m[k] + h01 * P[k + 1] + h11 * h * m[k + 1]
    return out


def scene_bounds(d: FlowData, D: float):
    """Box of the braid, and the box the camera may move in."""
    lo = np.array([d.X[0] - 0.5, -d.R, -d.R])
    hi = np.array([d.X[-1] + 0.5, d.R, d.R])
    pad = np.array([D * 0.4, D * 0.6, D * 1.15])
    return (lo, hi), (lo - pad, hi + pad)


# ── page ────────────────────────────────────────────────────────────────────

def _module_source(spec: str) -> str:
    """A vendored module with its relative imports turned into the bare
    specifiers of the import map."""
    rel = _MODULES[spec]
    src = (ASSETS / rel).read_text(encoding="utf-8")
    base = rel.rsplit("/", 1)[0] if "/" in rel else ""

    def fix(m):
        target = m.group(2)
        parts = (base.split("/") if base else []) + target.split("/")
        out = []
        for p in parts:
            if p == "..":
                out.pop()
            elif p != ".":
                out.append(p)
        return f"{m.group(1)}'three/{'/'.join(out)}'"
    return re.sub(r"(from\s*)['\"](\.{1,2}/[^'\"]+)['\"]", fix, src)


def _import_map(three: str) -> str:
    import base64

    if three == "cdn":
        imports = {"three": f"{THREE_CDN}/build/three.module.min.js",
                   "three/addons/": f"{THREE_CDN}/examples/jsm/"}
    else:
        imports = {spec: "data:text/javascript;base64,"
                   + base64.b64encode(_module_source(spec).encode()).decode()
                   for spec in _MODULES}
    return json.dumps({"imports": imports})


def flow_page(d: FlowData, theme: Theme, *, mode: str, n_frames: int, fps: float,
              size=(1280, 720), title: Optional[str] = None, subtitle: Optional[str] = None,
              three: str = "embed", bloom: Optional[bool] = None, fov: float = 40.0) -> str:
    W, H = size
    pos, tgt, D = camera_path(d, n_frames, W / H, fov)
    dist = np.linalg.norm(pos - tgt, axis=1)
    L = len(d.layers)
    readout = []
    for p in pos:
        # inside the braid: the layer the camera is passing; outside: all of them
        if np.hypot(p[1], p[2]) < 2.6 * d.R and -1.0 <= p[0] <= d.X[-1] + 0.5:
            li = int(np.clip(round(max(p[0], 0.0) / d.step), 0, L - 1))
            readout.append(str(d.layers[li]))
        else:
            readout.append(f"{d.layers[0]}–{d.layers[-1]}")
    data = {
        "mode": mode, "fps": fps, "nFrames": n_frames, "size": [W, H], "fov": fov,
        "tokens": d.tokens, "layers": d.layers,
        "P": np.round(d.P, 4).tolist(), "u": np.round(d.u.T, 4).tolist(),
        "r": np.round(d.radius.T, 4).tolist(), "recv": np.round(d.recv.T, 3).tolist(),
        "R": d.R, "X1": float(d.X[-1]),
        "step": d.step, "labels": d.labels, "highlight": d.highlight,
        "cam": {"pos": np.round(pos, 4).tolist(), "tgt": np.round(tgt, 4).tolist(),
                "fogNear": np.round(dist * 0.55, 3).tolist(),
                "fogFar": np.round(dist * 2.1 + d.X[-1] * 0.25, 3).tolist(),
                "readout": readout, "D": D},
        "theme": {"name": theme.name, "bg": theme.bg, "ink": theme.ink, "ink2": theme.ink2,
                  "muted": theme.muted, "rule": theme.rule, "neg": theme.neg,
                  "pos": theme.pos, "mid": theme.node_mid, "alert": theme.alert,
                  "edge": theme.edge},
        "bloom": theme.name == "dark" if bloom is None else bool(bloom),
        "title": title or "Tokens flowing through depth",
        "subtitle": subtitle if subtitle is not None else (
            f"{model_label(d.cache)} · each tube is one token, layer 0 to {d.layers[-1]}"),
    }
    from ._core import FONT_FAMILY

    font = ", ".join(f"'{f}'" for f in FONT_FAMILY) + ", sans-serif"
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    css = _CSS.replace("__BG__", theme.bg).replace("__INK__", theme.ink) \
        .replace("__INK2__", theme.ink2).replace("__MUTED__", theme.muted) \
        .replace("__FONT__", font).replace("__RULE__", theme.rule) \
        .replace("__NEG__", theme.neg).replace("__POS__", theme.pos) \
        .replace("__ALERT__", theme.alert)
    legend_inj = ('<span class="sw ring"></span>injected tokens' if d.highlight else "")
    body = _BODY.replace("__LEGEND_INJ__", legend_inj)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{_esc(data['title'])}</title><style>{css}</style>"
            f"<script type=\"importmap\">{_import_map(three)}</script></head><body>{body}"
            f"<script>window.__ST_DATA={payload};</script>"
            f"<script type=\"module\">{_JS}</script></body></html>\n")


def _esc(s: str) -> str:
    import html
    return html.escape(s)


#: Options of :class:`FlowData` that the entry points pass through.
DATA_KEYS = ("layers", "text", "other", "highlight", "axes", "align", "smooth", "bundle",
             "spread", "labels")


def flow_data(cache, **kw) -> FlowData:
    return FlowData(cache, **{k: kw[k] for k in DATA_KEYS if k in kw})


def flow_html(cache, path, theme, *, plotlyjs: str = "auto", three: Optional[str] = None,
              title=None, subtitle=None, duration: Optional[float] = None, fps: float = 30,
              bloom=None, **kw):
    """Interactive page: the fly-through plays on load, then drag to orbit,
    hover a tube for its token."""
    d = flow_data(cache, **kw)
    if three is None:
        three = "cdn" if plotlyjs == "cdn" else "embed"
    n = int(round((duration or default_duration(d)) * fps))
    page = flow_page(d, theme, mode="interactive", n_frames=n, fps=fps, three=three,
                     title=title, subtitle=subtitle, bloom=bloom)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def default_duration(d: FlowData) -> float:
    return float(np.clip(8.0 + 0.3 * len(d.layers), 9.0, 18.0))


def render_flow_video(cache, out: Path, theme: Theme, *, size=(1280, 720), fps: float = 30,
                      duration: Optional[float] = None, title=None, subtitle=None,
                      bloom=None, engine: str = "auto", threads: int = 2,
                      browser: Optional[str] = None, progress: bool = False, dpi: int = 100,
                      font=None, wordmark: bool = False, crf: Optional[int] = None,
                      segment=None, **kw) -> dict:
    """Encode the flow film to ``out`` with the browser (three.js) or, as a
    fallback, matplotlib. ``segment=(a, b)`` keeps that fraction of the film;
    a GIF keeps the final pull-back and turn by default, with a 96-colour
    palette and no dither (the fly-through changes every pixel of every
    frame, which GIF stores badly). Returns the engine used and timings."""
    from ._browser import BrowserUnavailable, find_browser

    d = flow_data(cache, **kw)
    n = int(round((duration or default_duration(d)) * fps))
    gif = Path(out).suffix.lower() == ".gif"
    if segment is None:
        segment = (0.62, 1.0) if gif else (0.0, 1.0)
    first = int(np.clip(round(segment[0] * n), 0, n - 1))
    last = int(np.clip(round(segment[1] * n), first + 1, n))
    wopts = dict(threads=threads, crf=crf)
    if gif:
        wopts.update(gif_colors=96, gif_dither="none")
    if engine == "auto":
        try:
            find_browser(browser)
            engine = "browser"
        except BrowserUnavailable:
            engine = "matplotlib"
    t0 = time.perf_counter()
    if engine == "matplotlib":
        from ._core import Canvas
        sc = FlowScene(d, Canvas(size[0], size[1], dpi, theme, font), fps=fps, n_frames=n,
                       title=title, subtitle=subtitle, wordmark=wordmark)
        sc.render(out, progress=progress, frames=range(first, last), **wopts)
        return {"engine": "matplotlib", "frames": last - first,
                "seconds": time.perf_counter() - t0}
    if engine != "browser":
        raise ValueError("engine must be 'auto', 'browser' or 'matplotlib'")
    from ._browser import Browser

    page = flow_page(d, theme, mode="capture", n_frames=n, fps=fps, size=size, title=title,
                     subtitle=subtitle, bloom=bloom, three="embed")
    with tempfile.TemporaryDirectory(prefix="st-flow-") as tmp:
        html_path = Path(tmp) / "flow.html"
        html_path.write_text(page, encoding="utf-8")
        w = FrameWriter(out, fps, **wopts)
        try:
            with Browser(size[0], size[1], threads=threads, executable=browser) as b:
                b.open(html_path)
                for f in range(first, last):
                    b.evaluate(f"window.__st.frame({f})")
                    w.append(b.screenshot())
                    if progress and (f % 25 == 0 or f == last - 1):
                        print(f"\rflow: frame {f + 1}/{last}", end="", flush=True)
        finally:
            w.close()
    if progress:
        print()
    return {"engine": "browser", "frames": last - first, "seconds": time.perf_counter() - t0}


def render_flow_frames(cache, theme: Theme, frames: Sequence[int], *, size=(1280, 720),
                       fps: float = 30, duration=None, engine: str = "browser",
                       threads: int = 2, browser=None, **kw) -> List[np.ndarray]:
    """Selected frames as arrays (for stills and contact sheets)."""
    d = flow_data(cache, **kw)
    n = int(round((duration or default_duration(d)) * fps))
    if engine == "matplotlib":
        from ._core import Canvas
        sc = FlowScene(d, Canvas(size[0], size[1], 100, theme), fps=fps, n_frames=n,
                       title=kw.get("title"), subtitle=kw.get("subtitle"))
        return [sc.frame(int(f)) for f in frames]
    from ._browser import Browser

    page = flow_page(d, theme, mode="capture", n_frames=n, fps=fps, size=size,
                     title=kw.get("title"), subtitle=kw.get("subtitle"),
                     bloom=kw.get("bloom"), three="embed")
    out = []
    with tempfile.TemporaryDirectory(prefix="st-flow-") as tmp:
        html_path = Path(tmp) / "flow.html"
        html_path.write_text(page, encoding="utf-8")
        with Browser(size[0], size[1], threads=threads, executable=browser) as b:
            b.open(html_path)
            for f in frames:
                b.evaluate(f"window.__st.frame({int(f)})")
                out.append(b.screenshot())
    return out


# ── matplotlib fallback ─────────────────────────────────────────────────────

class LookAt:
    """A perspective camera from a position, a target and a vertical field
    of view (world y is up), projecting to pixels."""

    def __init__(self, pos, tgt, fov: float, W: int, H: int):
        self.pos = np.asarray(pos, float)
        f = np.asarray(tgt, float) - self.pos
        f /= np.linalg.norm(f) or 1.0
        r = np.cross(f, [0.0, 1.0, 0.0])
        r /= np.linalg.norm(r) or 1.0
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.k = (H / 2) / math.tan(math.radians(fov) / 2)
        self.W, self.H = W, H

    def project(self, P):
        Q = np.asarray(P, float) - self.pos
        z = Q @ self.f
        zc = np.maximum(z, 1e-3)
        x = self.W / 2 + self.k * (Q @ self.r) / zc
        y = self.H / 2 - self.k * (Q @ self.u) / zc
        return np.stack([x, y], -1), z


class FlowScene:
    """The flow film drawn with matplotlib: tubes as depth-sorted strokes
    whose width follows perspective, with a soft glow on dark ground."""

    kind = "flow"

    def __init__(self, d: FlowData, canvas, *, fps: float, n_frames: int, title=None,
                 subtitle=None, wordmark: bool = False, fov: float = 40.0):
        from ._scene import Scene

        self._scene = Scene(canvas, fps, n_frames, wordmark)
        self.cv, self.th, self.d = canvas, canvas.theme, d
        self.fps, self.n_frames = fps, n_frames
        self.fov = fov
        self.pos, self.tgt, self.D = camera_path(d, n_frames, canvas.W / canvas.H, fov)
        self.title = title or "Tokens flowing through depth"
        self.subtitle = subtitle if subtitle is not None else (
            f"{model_label(d.cache)} · each tube is one token, layer 0 to {d.layers[-1]}")
        # dense samples along each track (Catmull-Rom through the layers)
        self.samples = 6
        self.tracks, self.uu, self.rr = self._sample()

    def _sample(self):
        d = self.d
        S, L, _ = d.P.shape
        k = self.samples
        t = np.linspace(0, L - 1, (L - 1) * k + 1)
        tracks = np.stack([_catmull(d.P[s], t) for s in range(S)])
        uu = np.stack([np.interp(t, np.arange(L), d.u[:, s]) for s in range(S)])
        rr = np.stack([np.interp(t, np.arange(L), d.radius[:, s]) for s in range(S)])
        return tracks, uu, rr

    def frame(self, f: int) -> np.ndarray:
        cv, th, d = self.cv, self.th, self.d
        cv.begin()
        cam = LookAt(self.pos[f], self.tgt[f], self.fov, cv.W, cv.H)
        dist = float(np.linalg.norm(self.pos[f] - self.tgt[f]))
        near, far = dist * 0.55, dist * 2.1 + d.X[-1] * 0.25
        segs, cols, widths, depth = [], [], [], []
        neg, pos, mid = hex_rgb(th.neg), hex_rgb(th.pos), hex_rgb(th.node_mid)
        bg = hex_rgb(th.bg)
        for s in range(self.tracks.shape[0]):
            xy, z = cam.project(self.tracks[s])
            ok = z > 0.2
            a = np.stack([xy[:-1], xy[1:]], 1)
            zz = (z[:-1] + z[1:]) / 2
            keep = ok[:-1] & ok[1:]
            u = (self.uu[s, :-1] + self.uu[s, 1:]) / 2
            w = np.clip((u + 0.15) / 0.3, 0, 1)[:, None]
            c = np.where(np.abs(u)[:, None] < 0.05, mid, neg * (1 - w) + pos * w)
            fog = np.clip((zz - near) / max(far - near, 1e-6), 0, 0.9)[:, None]
            c = c * (1 - fog) + bg * fog
            r = (self.rr[s, :-1] + self.rr[s, 1:]) / 2
            wpx = 2 * r * cam.k / np.maximum(zz, 0.2)
            segs.append(a[keep])
            cols.append(c[keep])
            widths.append(wpx[keep])
            depth.append(zz[keep])
        segs = np.concatenate(segs)
        cols = np.concatenate(cols)
        widths = np.concatenate(widths)
        depth = np.concatenate(depth)
        order = np.argsort(-depth)
        if th.name == "dark":                     # a soft glow under each stroke
            cv.lines(segs[order], np.concatenate([cols[order], np.full((len(order), 1), 0.16)], 1),
                     np.clip(widths[order] * 3.0, 1, 60), z=2, capstyle="round")
        cv.lines(segs[order], np.concatenate([cols[order], np.ones((len(order), 1))], 1),
                 np.clip(widths[order], 0.6, 40), z=3, capstyle="round")
        sc = self._scene
        sc.header(self.title, self.subtitle)
        L = len(d.layers)
        li = int(np.clip(round((self.pos[f][0] + 0.5) / d.step), 0, L - 1))
        sc.counter(str(d.layers[li]), "layer")
        sc.draw_wordmark()
        return cv.grab()

    def render(self, path, threads: int = 2, progress: bool = False, frames=None, **writer):
        w = FrameWriter(path, self.fps, threads=threads, **writer)
        frames = range(self.n_frames) if frames is None else frames
        try:
            for f in frames:
                w.append(self.frame(f))
                if progress and (f % 25 == 0 or f == self.n_frames - 1):
                    print(f"\rflow: frame {f + 1}/{self.n_frames}", end="", flush=True)
        finally:
            w.close()
            self.cv.close()


def _catmull(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Uniform Catmull-Rom through the rows of ``P`` at parameters ``t``."""
    n = len(P)
    if n == 1:
        return np.repeat(P, len(t), 0)
    ext = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    out = np.zeros((len(t), P.shape[1]))
    for q, tq in enumerate(t):
        i = int(min(max(math.floor(tq), 0), n - 2))
        s = tq - i
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out[q] = 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3)
    return out


# ── page source ─────────────────────────────────────────────────────────────

_CSS = """html,body{margin:0;height:100%;background:__BG__;overflow:hidden;font-family:__FONT__}
#c{position:fixed;inset:0;display:block}#ov{position:fixed;inset:0;pointer-events:none}
#hd{position:fixed;left:40px;top:28px;pointer-events:none}
#hd h1{margin:0;font-size:26px;font-weight:600;color:__INK__;letter-spacing:-0.01em}
#hd p{margin:6px 0 0;font-size:15px;color:__MUTED__}
#ly{position:fixed;right:40px;top:26px;text-align:right;pointer-events:none;color:__INK__}
#ly b{font-size:40px;font-weight:600}#ly span{font-size:14px;color:__MUTED__;margin-right:8px}
#lg{position:fixed;left:40px;bottom:26px;font-size:13px;color:__INK2__;pointer-events:none}
#lg .sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 6px -1px 16px}
#lg .sw:first-child{margin-left:0}#lg .n{background:__NEG__}#lg .p{background:__POS__}
#lg .ring{border:2px solid __ALERT__;width:9px;height:9px}
#lg .m{color:__MUTED__;margin-left:18px}
#ui{position:fixed;right:40px;bottom:22px;display:none;gap:8px}
#ui button{font:13px __FONT__;color:__INK2__;background:transparent;border:1px solid __RULE__;
border-radius:6px;padding:6px 12px;cursor:pointer}#ui button:hover{color:__INK__}
#tip{position:fixed;display:none;pointer-events:none;font:12px __FONT__;color:__BG__;
background:__INK__;padding:4px 7px;border-radius:4px}
@media (max-width:700px){#hd{left:16px;top:14px}#hd h1{font-size:18px}#hd p{font-size:12px}
#ly{right:16px;top:12px}#ly b{font-size:26px}#lg{left:16px;bottom:12px;font-size:11px}
#lg .m{display:none}#ui{right:16px;bottom:10px}}"""

_BODY = """<canvas id="c"></canvas><canvas id="ov"></canvas>
<div id="hd"><h1 id="t"></h1><p id="s"></p></div>
<div id="ly"><span id="lt">layer</span><b id="lv">0</b></div>
<div id="lg"><span class="sw n"></span>Fiedler −<span class="sw p"></span>Fiedler +
__LEGEND_INJ__<span class="m">thickness: attention received · position: Laplacian
eigenvectors 2–3 of each layer's graph</span></div>
<div id="ui"><button id="fly">▶ fly through</button><button id="ovw">overview</button></div>
<div id="tip"></div>"""

_JS = r"""
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// Colours pass straight through as display values, in every frame and with
// or without bloom: three.js colour management converted the composer's
// output a varying number of times (a grey haze, then a black ground).
THREE.ColorManagement.enabled = false;

const D = window.__ST_DATA, T = D.theme, capture = D.mode === 'capture';
document.getElementById('t').textContent = D.title;
document.getElementById('s').textContent = D.subtitle;
const canvas = document.getElementById('c'), ov = document.getElementById('ov');
const ctx = ov.getContext('2d');
let W = capture ? D.size[0] : innerWidth, H = capture ? D.size[1] : innerHeight;
const renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true});
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
ov.width = W; ov.height = H; ov.style.width = W + 'px'; ov.style.height = H + 'px';
renderer.setClearColor(T.bg, 1);
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(T.bg, 10, 60);
const camera = new THREE.PerspectiveCamera(D.fov, W / H, 0.05, 400);

// lights: soft fill plus a key light that rides with the camera
scene.add(new THREE.HemisphereLight(0xffffff, T.name === 'dark' ? 0x1a2230 : 0xb8b0a0,
                                    T.name === 'dark' ? 1.15 : 1.7));
const key = new THREE.DirectionalLight(0xffffff, T.name === 'dark' ? 1.6 : 1.3);
scene.add(key);

const cNeg = new THREE.Color(T.neg), cPos = new THREE.Color(T.pos), cMid = new THREE.Color(T.mid);
function sideColour(u, out) {
  // sharp sides with a short neutral band: the Fiedler split, not a gradient
  if (Math.abs(u) < 0.05) return out.copy(cMid);
  const w = Math.min(1, Math.max(0, (u + 0.15) / 0.3));
  return out.copy(cNeg).lerp(cPos, w);
}

const S = D.P.length, L = D.layers.length, SEG = 10, RAD = 12;
const meshes = [];
function lerpArr(a, x) {
  const i = Math.min(Math.floor(x), a.length - 2), f = x - i;
  return a.length === 1 ? a[0] : a[i] * (1 - f) + a[i + 1] * f;
}
function tube(s, scale, material) {
  const pts = D.P[s].map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.5);
  const n = Math.max(2, (L - 1) * SEG);
  const pos = [], nor = [], col = [], idx = [];
  let normal = new THREE.Vector3(0, 1, 0), prevT = null;
  const P = new THREE.Vector3(), Tg = new THREE.Vector3(), B = new THREE.Vector3();
  const c = new THREE.Color();
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    curve.getPoint(t, P);
    curve.getTangent(t, Tg).normalize();
    if (prevT === null) {
      if (Math.abs(Tg.y) > 0.9) normal.set(1, 0, 0);
      normal.sub(Tg.clone().multiplyScalar(normal.dot(Tg))).normalize();
    } else {
      // parallel transport of the normal along the curve
      const axis = new THREE.Vector3().crossVectors(prevT, Tg);
      const ang = Math.asin(Math.min(1, axis.length()));
      if (axis.lengthSq() > 1e-12) normal.applyAxisAngle(axis.normalize(), ang);
      normal.sub(Tg.clone().multiplyScalar(normal.dot(Tg))).normalize();
    }
    prevT = Tg.clone();
    B.crossVectors(Tg, normal);
    const lx = t * (L - 1);
    const r = lerpArr(D.r[s], lx) * scale;
    sideColour(lerpArr(D.u[s], lx), c);
    for (let j = 0; j <= RAD; j++) {
      const a = j / RAD * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
      const nx = normal.x * ca + B.x * sa, ny = normal.y * ca + B.y * sa, nz = normal.z * ca + B.z * sa;
      pos.push(P.x + r * nx, P.y + r * ny, P.z + r * nz);
      nor.push(nx, ny, nz);
      col.push(c.r, c.g, c.b);
    }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < RAD; j++) {
    const a = i * (RAD + 1) + j, b = a + RAD + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1);
  }
  // end caps, so a tube seen end-on is closed
  for (const [t, ring, sign] of [[0, 0, -1], [1, n * (RAD + 1), 1]]) {
    curve.getPoint(t, P);
    curve.getTangent(t, Tg).normalize();
    sideColour(lerpArr(D.u[s], t * (L - 1)), c);
    const centre = pos.length / 3;
    pos.push(P.x, P.y, P.z);
    nor.push(sign * Tg.x, sign * Tg.y, sign * Tg.z);
    col.push(c.r, c.g, c.b);
    for (let j = 0; j < RAD; j++)
      if (sign < 0) idx.push(centre, ring + j + 1, ring + j);
      else idx.push(centre, ring + j, ring + j + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  const m = new THREE.Mesh(g, material);
  m.userData.token = s;
  return m;
}
const mat = T.name === 'dark'
  ? new THREE.MeshStandardMaterial({vertexColors: true, roughness: 0.35, metalness: 0.0,
                                    emissive: 0x000000})
  : new THREE.MeshStandardMaterial({vertexColors: true, roughness: 0.55, metalness: 0.0});
if (T.name === 'dark') {
  // glow: the vertex colour also lights the surface from inside
  mat.onBeforeCompile = sh => {
    sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>',
      '#include <emissivemap_fragment>\n totalEmissiveRadiance += vColor.rgb * 0.45;');
  };
}
const ring = new THREE.MeshBasicMaterial({color: T.alert, transparent: true, opacity: 0.9,
                                         side: THREE.BackSide, depthWrite: false});
const hl = new Set(D.highlight);
for (let s = 0; s < S; s++) {
  const m = tube(s, 1, mat);
  scene.add(m); meshes.push(m);
  if (hl.has(s)) scene.add(tube(s, 1.45, ring));
}

// layer gates: faint rings the camera flies through, every fifth stronger
const gateR = D.R * 1.08;
for (let l = 0; l < L; l++) {
  const g = new THREE.BufferGeometry().setFromPoints(
    Array.from({length: 97}, (_, k) => {
      const a = k / 96 * Math.PI * 2;
      return new THREE.Vector3(l * D.step, gateR * Math.cos(a), gateR * Math.sin(a));
    }));
  const major = (D.layers[l] % 5 === 0) || l === L - 1;
  const lm = new THREE.LineBasicMaterial({color: T.name === 'dark' ? T.ink2 : T.muted,
    transparent: true, opacity: major ? 0.26 : 0.09, depthWrite: false});
  scene.add(new THREE.Line(g, lm));
}

let composer = null;
if (D.bloom) {
  composer = new EffectComposer(renderer);
  composer.setPixelRatio(1);
  composer.setSize(W, H);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), 0.55, 0.45, 0.62));
}
function draw() { composer ? composer.render() : renderer.render(scene, camera); overlay(); }

// labels: key tokens at both ends of their tube, and gate numbers
const V = new THREE.Vector3();
function overlay() {
  ctx.clearRect(0, 0, W, H);
  const small = W < 700, fs = small ? 11 : 14;
  ctx.font = `500 ${fs}px ${getComputedStyle(document.body).fontFamily}`;
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  const boxes = [];
  const put = (txt, x, y, colour, align) => {
    const w = ctx.measureText(txt).width + 6, h = fs + 4;
    let bx = align === 'right' ? x - w : x;
    let by = y - h / 2;
    for (let k = 0; k < 8; k++) {
      const hit = boxes.find(b => bx < b[0] + b[2] && bx + w > b[0] && by < b[1] + b[3] && by + h > b[1]);
      if (!hit) break;
      by = hit[1] + hit[3] + 1;
    }
    if (bx < 8 || bx + w > W - 8 || by < 70 || by + h > H - 40) return;
    boxes.push([bx, by, w, h]);
    ctx.lineWidth = 4; ctx.strokeStyle = T.bg; ctx.fillStyle = colour;
    ctx.textAlign = 'left';
    ctx.strokeText(txt, bx + 3, by + h / 2); ctx.fillText(txt, bx + 3, by + h / 2);
  };
  const front = (p) => {
    V.set(p[0], p[1], p[2]);
    const dz = V.clone().sub(camera.position).dot(camera.getWorldDirection(new THREE.Vector3()));
    if (dz < 0.3) return null;
    V.project(camera);
    return [(V.x + 1) / 2 * W, (1 - V.y) / 2 * H, dz];
  };
  for (const s of D.labels) {
    const colour = hl.has(s) ? T.alert : T.ink2;
    const a = front(D.P[s][0]), b = front(D.P[s][L - 1]);
    if (a && a[2] < D.cam.D * 0.75) put(D.tokens[s], a[0] - 10, a[1], colour, 'right');
    if (b) put(D.tokens[s], b[0] + 10, b[1], colour, 'left');
  }
  ctx.font = `${small ? 10 : 12}px ${getComputedStyle(document.body).fontFamily}`;
  for (let l = 0; l < L; l++) {
    if (!((D.layers[l] % 5 === 0) || l === L - 1)) continue;
    const g = front([l * D.step, -gateR, 0]);
    if (!g) continue;
    ctx.fillStyle = T.muted; ctx.textAlign = 'center';
    if (g[0] > 30 && g[0] < W - 30 && g[1] < H - 50 && g[1] > 80)
      ctx.fillText(String(D.layers[l]), g[0], g[1] + 14);
  }
}

const lv = document.getElementById('lv'), lt = document.getElementById('lt');
function setFrame(i) {
  i = Math.max(0, Math.min(D.nFrames - 1, i));
  const p = D.cam.pos[i], q = D.cam.tgt[i];
  camera.position.set(p[0], p[1], p[2]);
  camera.lookAt(q[0], q[1], q[2]);
  camera.updateProjectionMatrix();
  key.position.copy(camera.position).add(new THREE.Vector3(0, 4, 0));
  scene.fog.near = D.cam.fogNear[i]; scene.fog.far = D.cam.fogFar[i];
  lv.textContent = D.cam.readout[i];
  lt.textContent = D.cam.readout[i].includes('–') ? 'layers' : 'layer';
}
const raf = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

if (capture) {
  window.__st = {
    ready: Promise.resolve(true),
    frame: async (i) => { setFrame(i); draw(); await raf(); return true; },
    cameraPath: D.cam,
  };
} else {
  document.getElementById('ui').style.display = 'flex';
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  let playing = true, start = performance.now();
  const stop = () => {
    playing = false;
    controls.target.set(...D.cam.tgt[D.nFrames - 1]);
    scene.fog.near = D.cam.fogNear[D.nFrames - 1]; scene.fog.far = D.cam.fogFar[D.nFrames - 1] * 1.5;
  };
  controls.addEventListener('start', () => { if (playing) stop(); });
  document.getElementById('fly').onclick = () => { playing = true; start = performance.now(); };
  document.getElementById('ovw').onclick = () => { setFrame(D.nFrames - 1); stop(); };
  addEventListener('resize', () => {
    W = innerWidth; H = innerHeight;
    renderer.setSize(W, H, false); canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ov.width = W; ov.height = H; ov.style.width = W + 'px'; ov.style.height = H + 'px';
    if (composer) composer.setSize(W, H);
    camera.aspect = W / H; camera.updateProjectionMatrix();
  });
  const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = document.getElementById('tip');
  canvas.addEventListener('pointermove', e => {
    mouse.set(e.clientX / W * 2 - 1, -(e.clientY / H) * 2 + 1);
    ray.setFromCamera(mouse, camera);
    const hit = ray.intersectObjects(meshes)[0];
    if (hit) {
      const s = hit.object.userData.token, lx = Math.round(hit.point.x / D.step);
      const l = Math.max(0, Math.min(L - 1, lx));
      const u = D.u[s][l];
      tip.textContent = `${D.tokens[s]} · token ${s} · layer ${D.layers[l]} · ` +
        `Fiedler ${u >= 0 ? '+' : '−'} · attention received ${(D.recv[s][l]).toFixed(2)}`;
      tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
      tip.style.display = 'block';
    } else tip.style.display = 'none';
  });
  const loop = () => {
    if (playing) {
      const i = Math.floor((performance.now() - start) / 1000 * D.fps);
      if (i >= D.nFrames) { setFrame(D.nFrames - 1); stop(); } else setFrame(i);
    } else {
      controls.update();
    }
    draw();
    requestAnimationFrame(loop);
  };
  setFrame(0);
  loop();
}
"""
