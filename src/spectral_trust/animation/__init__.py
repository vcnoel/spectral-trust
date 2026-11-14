# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Animated and interactive views of a :class:`~spectral_trust.cache.SpectralCache`,
written to a file in one call (added in 0.4.2)::

    cache = model.run_with_spectra("The cat sat on the mat because it was warm.")
    cache.animate("embedding3d", "hero.mp4")
    st.viz.animate(cache, "waterfall", "waterfall.gif", theme="dark")
    cache.animate("embedding3d", "hero.html")          # interactive 3D

The suffix picks the format: ``.mp4`` (H.264), ``.webm`` (VP9) and ``.gif``
are encoded by ffmpeg through imageio and imageio-ffmpeg (``pip install
'spectral_trust[viz]'``); ``.html`` is an interactive plotly.js page.

Kinds:

``embedding3d``
    tokens placed in 3D by Laplacian eigenvectors 2 to 4 of each layer's
    head-averaged attention graph, coloured by the sign of the Fiedler
    vector, morphing from layer to layer (Procrustes-aligned) under a slow
    camera orbit.
``waterfall``
    residual energy per graph-frequency mode (``spectral_masses``, as a
    share of each layer's total) as ridgelines in 3D, built layer by layer,
    with the HFER band marked.
``compare``
    two prompts side by side (for example clean and injected): synced
    layer-by-layer graphs on shared token positions, the injected tokens and
    their edges highlighted, and the Fiedler value and HFER of both.
``graph_movie``
    the token graph of each layer on a circle (or a force or spectral
    layout), edges by weight, nodes by the Fiedler vector, morphing across
    layers.
``head_atlas``
    every head's graph in one layer as small multiples, sorted by a metric,
    swept over the layers.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

from ._core import THEMES, Canvas, Theme, get_theme

KINDS = ("embedding3d", "waterfall", "compare", "graph_movie", "head_atlas")
FORMATS = (".mp4", ".webm", ".gif", ".html")

#: Default frame size and rate per format: the GIF is sized for a phone.
DEFAULT_SIZE = {".mp4": (1280, 720), ".webm": (1280, 720), ".gif": (640, 360)}
DEFAULT_FPS = {".mp4": 30, ".webm": 30, ".gif": 15}


def _scene_class(kind: str):
    if kind == "embedding3d":
        from ._embedding import Embedding3D
        return Embedding3D
    if kind == "waterfall":
        from ._waterfall import Waterfall
        return Waterfall
    if kind == "compare":
        from ._compare import Compare
        return Compare
    if kind == "graph_movie":
        from ._graph_movie import GraphMovie
        return GraphMovie
    if kind == "head_atlas":
        from ._head_atlas import HeadAtlas
        return HeadAtlas
    raise ValueError(f"unknown kind {kind!r}; kinds: {', '.join(KINDS)}")


def _html_builder(kind: str):
    if kind == "embedding3d":
        from ._embedding import embedding3d_html
        return embedding3d_html
    if kind == "waterfall":
        from ._waterfall import waterfall_html
        return waterfall_html
    if kind == "compare":
        from ._compare import compare_html
        return compare_html
    if kind == "graph_movie":
        from ._graph_movie import graph_movie_html
        return graph_movie_html
    if kind == "head_atlas":
        from ._head_atlas import head_atlas_html
        return head_atlas_html
    raise ValueError(f"unknown kind {kind!r}; kinds: {', '.join(KINDS)}")


def make_scene(cache, kind: str, *, size: Tuple[int, int] = (1280, 720), dpi: int = 100,
               fps: float = 30, theme: Union[str, Theme] = "light", font=None,
               wordmark: bool = False, **options):
    """The :class:`~spectral_trust.animation._scene.Scene` behind
    :func:`animate`, for drawing single frames (``scene.frame(i)`` returns
    an ``[H, W, 3]`` uint8 array)."""
    w, h = (int(v) for v in size)
    w, h = w - w % 2, h - h % 2           # yuv420p needs even sizes
    canvas = Canvas(w, h, dpi, get_theme(theme), font)
    return _scene_class(kind)(cache, canvas, fps=fps, wordmark=wordmark, **options)


def animate(cache, kind: str = "embedding3d", out: Union[str, Path] = "spectral.mp4", *,
            fps: Optional[float] = None, duration: Optional[float] = None,
            theme: Union[str, Theme] = "light", size: Optional[Tuple[int, int]] = None,
            dpi: int = 100, seed: int = 0, wordmark: bool = False, plotlyjs: str = "auto",
            threads: int = 2, font=None, progress: bool = False, **options) -> Path:
    """Write an animated (``.mp4``, ``.webm``, ``.gif``) or interactive
    (``.html``) view of ``cache`` to ``out`` and return its path.

    Args:
        cache: a :class:`~spectral_trust.cache.SpectralCache` holding the
            attention patterns (``keep_patterns``, kept by default for short
            texts). ``waterfall`` also needs the hybrid metrics
            (``hybrid=True``, the default).
        kind: one of ``embedding3d``, ``waterfall``, ``compare``,
            ``graph_movie``, ``head_atlas``.
        out: output path; its suffix picks the format.
        fps: frames per second (default 30 for video, 15 for GIF).
        duration: length in seconds (default: set by the number of layers).
        theme: ``"light"`` or ``"dark"``.
        size: frame size in pixels (default 1280 x 720 for video, 640 x 360
            for GIF).
        dpi: resolution of the text and line widths (pixels per inch).
        seed: seed of the layouts that use one (``graph_movie`` force
            layout); output is deterministic for a fixed seed.
        wordmark: draw a small "spectral-trust" wordmark in a corner.
        plotlyjs: for ``.html``: ``"auto"`` (embed when the plotly package
            is installed, else the CDN), ``"embed"`` (plotly.js copied into
            the page from the plotly package: one self-contained file that
            works offline, about 5 MB), ``"cdn"`` (a pinned plotly.js from
            cdn.plot.ly, a small page that needs the network), a URL, or
            the path of a local plotly.min.js.
        threads: encoder threads (ffmpeg).
        font: font family list (default Inter, Segoe UI, Helvetica, Arial).
        progress: print frame progress.
        **options: options of the kind, e.g. ``layers``, ``text``,
            ``title``, ``orbit`` (embedding3d), ``scale`` (waterfall),
            ``other`` and ``labels`` (compare), ``layout`` (graph_movie),
            ``layer`` and ``metric`` (head_atlas).

    Examples:
        >>> cache.animate("embedding3d", "hero.mp4")
        >>> st.viz.animate(cache, "waterfall", "waterfall.gif", theme="dark")
        >>> st.viz.animate(clean, "compare", "compare.mp4", other=injected)
    """
    out = Path(out)
    ext = out.suffix.lower()
    if ext not in FORMATS:
        raise ValueError(f"unsupported output {out.name!r}; use one of {', '.join(FORMATS)}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; kinds: {', '.join(KINDS)}")
    th = get_theme(theme)
    if ext == ".html":
        return Path(_html_builder(kind)(cache, out, th, plotlyjs=plotlyjs, seed=seed, **options))
    from ._core import need_imageio
    need_imageio()
    fps = float(fps or DEFAULT_FPS[ext])
    size = size or DEFAULT_SIZE[ext]
    scene = make_scene(cache, kind, size=size, dpi=dpi, fps=fps, theme=th, font=font,
                       wordmark=wordmark, duration=duration, seed=seed, **options)
    scene.render(out, threads=threads, progress=progress)
    return out


__all__ = ["FORMATS", "KINDS", "THEMES", "animate", "make_scene"]
