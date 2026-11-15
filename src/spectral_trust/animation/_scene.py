# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""The frame loop and the overlays every animated view shares."""
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ._core import Canvas, FrameWriter, Theme


class Scene:
    """One animation: ``n_frames`` frames, each drawn by :meth:`draw` on a
    :class:`Canvas`. Subclasses fill :meth:`draw`; the helpers below draw
    the header, layer counter, progress track, legend and wordmark."""

    kind = "scene"

    def __init__(self, canvas: Canvas, fps: float, n_frames: int, wordmark: bool = False):
        self.cv = canvas
        self.th: Theme = canvas.theme
        self.fps = float(fps)
        self.n_frames = int(max(n_frames, 1))
        self.wordmark = bool(wordmark)

    # layout -----------------------------------------------------------------
    @property
    def margin(self) -> float:
        return self.cv.px(40, 16)

    @property
    def top(self) -> float:
        """Bottom edge of the header band."""
        return self.cv.px(96, 70) if not self.cv.compact else self.cv.px(96, 62)

    @property
    def bottom(self) -> float:
        """Top edge of the footer band."""
        return self.cv.H - (self.cv.px(78, 52) if not self.cv.compact else self.cv.px(78, 44))

    # overlays ---------------------------------------------------------------
    def header(self, title: str, subtitle: str = ""):
        cv, th, m = self.cv, self.th, self.margin
        ts = cv.fs(26, 15)
        y = cv.px(52, 26) if not cv.compact else cv.px(52, 24)
        cv.text(m, y, title, ts, th.ink, weight="semibold", z=50)
        if subtitle:
            ss = cv.fs(15, 10.5) if not cv.compact else 10.5
            limit = int((cv.W - 2 * m - cv.px(220, 120)) / (0.52 * ss))
            if len(subtitle) > limit:
                subtitle = subtitle[:max(limit - 1, 8)] + "…"
            cv.text(m, y + ts * 0.62 + ss * 1.25, subtitle, ss, th.muted, z=50)

    def counter(self, value: str, label: str = "layer", extra: str = ""):
        cv, th = self.cv, self.th
        x = cv.W - self.margin
        big = cv.fs(40, 22)
        y = cv.px(64, 34)
        cv.text(x, y, value, big, th.ink, ha="right", weight="semibold", z=50)
        w, _ = cv.text_size(value, big, "semibold")
        cv.text(x - w - cv.px(8, 5), y, label, cv.fs(14, 10.5), th.muted, ha="right", z=50)
        if extra:
            cv.text(x, y + cv.fs(20, 14), extra, cv.fs(14, 10.5), th.ink2, ha="right", z=50)

    def progress(self, p: float, labels: Sequence, color: Optional[str] = None,
                 caption: str = "layer"):
        """A track along the bottom with one dot per layer; ``p`` is the
        fractional position."""
        cv, th, m = self.cv, self.th, self.margin
        n = len(labels)
        y = cv.H - cv.px(30, 18)
        fs = cv.fs(12.5, 10)
        lw0, _ = cv.text_size(f"{caption} {labels[0]}", fs)
        lw1, _ = cv.text_size(str(labels[-1]), fs)
        x0 = m + lw0 + cv.px(14, 8)
        x1 = cv.W - m - lw1 - cv.px(14, 8) - (cv.px(130, 0) if self.wordmark else 0)
        cv.text(m, y, f"{caption} {labels[0]}", fs, th.muted, va="center_baseline", z=50)
        cv.text(x1 + cv.px(14, 8), y, str(labels[-1]), fs, th.muted, va="center_baseline", z=50)
        xs = np.linspace(x0, x1, max(n, 2))[:n] if n > 1 else np.array([x0])
        xp = np.interp(p, np.arange(n), xs) if n > 1 else x0
        lw = cv.px(2.2, 1.5)
        cv.lines([[(x0, y), (x1, y)]], [th.rule], [lw], z=40)
        if xp > x0:
            cv.lines([[(x0, y), (xp, y)]], [th.ink2], [lw], z=41)
        r = cv.px(2.6, 1.6)
        done = xs <= xp + 1e-6
        cols = [th.ink2 if d else th.rule for d in done]
        cv.circles(np.stack([xs, np.full(n, y)], 1), r, cols, z=42)
        cv.circles([(xp, y)], cv.px(7, 4.5), [color or th.pos], edge=th.bg,
                   lw=cv.px(2.5, 1.5), z=43)

    def legend_split(self, x: float, y: float, neg: str = "Fiedler −", pos: str = "Fiedler +",
                     note: str = ""):
        """Two dots for the sides of the Fiedler split, then an optional
        note in the muted colour (dropped on compact frames)."""
        cv, th = self.cv, self.th
        fs = cv.fs(13, 10.5)
        r = cv.px(6, 4)
        gap = cv.px(7, 4)
        cv.circles([(x + r, y)], r, [th.neg], z=50)
        cv.text(x + 2 * r + gap, y, neg, fs, th.ink2, va="center_baseline", z=50)
        w, _ = cv.text_size(neg, fs)
        x2 = x + 2 * r + gap + w + cv.px(20, 12)
        cv.circles([(x2 + r, y)], r, [th.pos], z=50)
        cv.text(x2 + 2 * r + gap, y, pos, fs, th.ink2, va="center_baseline", z=50)
        if note and not cv.compact:
            w2, _ = cv.text_size(pos, fs)
            cv.text(x2 + 2 * r + gap + w2 + cv.px(24, 12), y, note, fs, th.muted,
                    va="center_baseline", z=50)

    def draw_wordmark(self):
        if not self.wordmark:
            return
        cv, th = self.cv, self.th
        cv.text(cv.W - self.margin, cv.H - cv.px(30, 18), "spectral-trust", cv.fs(13, 10.5),
                th.muted, ha="right", va="center_baseline", weight="semibold", z=60)

    # frames -----------------------------------------------------------------
    def draw(self, f: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def frame(self, f: int) -> np.ndarray:
        self.cv.begin()
        self.draw(f)
        self.draw_wordmark()
        return self.cv.grab()

    def render(self, path: Path, threads: int = 2, progress: bool = False, **writer) -> dict:
        """Encode every frame to ``path``; returns timing information."""
        t0 = time.perf_counter()
        w = FrameWriter(path, self.fps, threads=threads, **writer)
        draw_s = 0.0
        try:
            for f in range(self.n_frames):
                t = time.perf_counter()
                img = self.frame(f)
                draw_s += time.perf_counter() - t
                w.append(img)
                if progress and (f % 25 == 0 or f == self.n_frames - 1):
                    print(f"\r{self.kind}: frame {f + 1}/{self.n_frames}", end="", flush=True)
        finally:
            w.close()
            self.cv.close()
        if progress:
            print()
        return {"frames": self.n_frames, "seconds": time.perf_counter() - t0,
                "draw_seconds": draw_s}
