# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Interactive views of a :class:`~spectral_trust.cache.SpectralCache`, as
self-contained SVG with a few lines of inline script: no external files, no
CDN, no dependency beyond numpy, so they work offline and in any notebook.

- :func:`graph`: the attention graph of one layer or head, tokens as nodes,
  edge opacity the symmetrised attention, node colour the Fiedler vector.
- :func:`heatmap`: one metric over layers x heads (``backend="matplotlib"``
  returns a matplotlib figure instead).
- :func:`spectrum`: the Laplacian spectrum of one layer or head.

Each returns a :class:`View`, which renders inline in Jupyter
(``_repr_html_``) and saves as a standalone ``.html`` file (``save``).
Hovering an element shows its token text and values; where scripts are
stripped (some notebook viewers), the same text appears as a native tooltip.
"""
import html
import itertools
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Palette shared with the README figures (docs/make_assets.py).
INK = "#1A1A1A"
MID = "#7F7F7F"
GREY = "#9E9E9E"
BAND = "#D9D9D9"
GRID = "#EBEBEB"
ORANGE = "#E69F00"
ORANGE_DARK = "#B35900"
BLUE = "#0072B2"
NAN_FILL = "#F4F4F4"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace"

#: Sequential scale for heatmaps (white, orange, dark brown) and the diverging
#: scale of the Fiedler vector (blue, white, orange).
HEAT_STOPS = ["#FFFFFF", ORANGE, ORANGE_DARK, "#3D1F00"]
FIEDLER_STOPS = [BLUE, "#FFFFFF", ORANGE]

_ids = itertools.count()


def _hex(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _ramp(stops: Sequence[str], t: float) -> str:
    """Colour at ``t`` in [0, 1] on a piecewise-linear ramp."""
    if not np.isfinite(t):
        return NAN_FILL
    t = float(np.clip(t, 0.0, 1.0)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    rgb = _hex(stops[i]) * (1 - (t - i)) + _hex(stops[i + 1]) * (t - i)
    return "#" + "".join(f"{int(round(v)):02X}" for v in rgb)


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _token_label(t: str) -> str:
    s = t.strip()
    return s if s else repr(t)[1:-1] or "␣"


def _num(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e4):
        return f"{v:.3e}"
    return f"{v:.4f}"


class View:
    """An interactive figure: SVG plus a small inline script.

    Displays itself in Jupyter, and ``save("view.html")`` writes a standalone
    page that opens offline. ``svg`` holds the bare SVG markup.
    """

    def __init__(self, svg: str, title: str, kind: str):
        self.svg = svg
        self.title = title
        self.kind = kind
        self._uid = f"st-view-{next(_ids)}"

    def _fragment(self) -> str:
        uid = self._uid
        style = (
            f"#{uid}{{position:relative;display:inline-block;max-width:100%;background:#fff;"
            f"color:{INK};font-family:{FONT};border-radius:6px;padding:6px}}"
            f"#{uid} svg{{max-width:100%;height:auto;display:block}}"
            f"#{uid} .st-tip{{position:absolute;pointer-events:none;display:none;"
            f"background:{INK};color:#fff;font:12px/1.35 {MONO};padding:5px 7px;"
            f"border-radius:4px;white-space:pre;z-index:10;max-width:420px}}"
            f"#{uid} [data-tip]{{cursor:default}}"
            f"#{uid} .st-dim{{opacity:0.06!important}}"
            f"#{uid} .st-hot{{stroke:{INK}!important;stroke-width:2.2px!important}}"
        )
        script = _SCRIPT.replace("__UID__", uid)
        return (f'<div id="{uid}" class="st-view"><style>{style}</style>{self.svg}'
                f'<div class="st-tip"></div><script>{script}</script></div>')

    def _repr_html_(self) -> str:
        return self._fragment()

    def to_html(self) -> str:
        """A standalone HTML page holding the view."""
        return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                f"<title>{_esc(self.title)}</title>"
                f"<style>body{{margin:24px;background:#fff;color:{INK};font-family:{FONT}}}"
                "@media (prefers-color-scheme: dark){body{background:#1e1e1e}}</style>"
                f"</head><body>{self._fragment()}</body></html>\n")

    def save(self, path) -> Path:
        """Write :meth:`to_html` to ``path`` (``.html`` is appended if the
        name has no suffix) and return the path."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def __repr__(self) -> str:
        return f"<{self.kind} view: {self.title}>"


# Tooltips from data-tip (or <title>, which is moved into data-tip so that the
# native tooltip does not double the custom one), and neighbourhood
# highlighting for graph nodes.
_SCRIPT = r"""(function(){var root=document.getElementById('__UID__');if(!root)return;
var tip=root.querySelector('.st-tip');
root.querySelectorAll('title').forEach(function(t){var p=t.parentNode;
if(!p.getAttribute('data-tip'))p.setAttribute('data-tip',t.textContent);p.removeChild(t);});
function show(e,txt){tip.textContent=txt;tip.style.display='block';
var r=root.getBoundingClientRect();var x=e.clientX-r.left+14,y=e.clientY-r.top+14;
if(x+tip.offsetWidth>r.width)x=Math.max(0,e.clientX-r.left-tip.offsetWidth-10);
tip.style.left=x+'px';tip.style.top=y+'px';}
root.querySelectorAll('[data-tip]').forEach(function(el){
el.addEventListener('mousemove',function(e){show(e,el.getAttribute('data-tip'));});
el.addEventListener('mouseleave',function(){tip.style.display='none';});});
var edges=root.querySelectorAll('[data-a]');
root.querySelectorAll('[data-node]').forEach(function(n){var i=n.getAttribute('data-node');
n.addEventListener('mouseenter',function(){edges.forEach(function(ed){
if(ed.getAttribute('data-a')!==i&&ed.getAttribute('data-b')!==i)ed.classList.add('st-dim');});
n.classList.add('st-hot');});
n.addEventListener('mouseleave',function(){edges.forEach(function(ed){ed.classList.remove('st-dim');});
n.classList.remove('st-hot');});});})();"""


def _svg_open(width: float, height: float, label: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'width="{width:.0f}" height="{height:.0f}" role="img" aria-label="{_esc(label)}" '
            f'font-family="{FONT}" font-size="12" fill="{INK}">'
            f'<rect width="100%" height="100%" fill="#fff"/>')


def _text(x, y, s, size=12, anchor="start", color=INK, weight="normal", mono=False,
          baseline="auto") -> str:
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}" dominant-baseline="{baseline}"{fam}>'
            f'{_esc(s)}</text>')


# ── data access ─────────────────────────────────────────────────────────────

def _single(cache, text: int):
    return cache.text(text) if cache.is_batch else cache


def _span(cache) -> Optional[Tuple[int, int]]:
    sp = cache.token_span
    return tuple(sp) if sp is not None else None


def _graph_data(cache, layer: int, head: Optional[int]):
    """Adjacency, Laplacian eigenpairs and tokens of the graph the metrics
    were computed on (span applied)."""
    import torch

    from .graph import GraphConstructor
    from .spectral import SpectralAnalyzer

    key = f"blocks.{layer}.attn.pattern"
    if key not in cache:
        raise ValueError(
            f"layer {layer}'s attention pattern is not in the cache; run "
            "run_with_spectra(..., keep_patterns=True) for the graph view")
    a = torch.as_tensor(np.asarray(cache[key], dtype=np.float32))     # [H, S, S]
    cfg = cache.config
    gc = GraphConstructor(cfg)
    sym = gc.symmetrize_attention(a.unsqueeze(0))                       # [1, H, S, S]
    if head is None:
        adj = gc.aggregate_heads(sym)[0]
    else:
        if not 0 <= head < a.shape[0]:
            raise IndexError(f"head {head} out of range for {a.shape[0]} heads")
        adj = sym[0, head]
    tokens = list(cache.tokens)
    span = _span(cache)
    offset = 0
    if span is not None:
        s, e = span
        adj = adj[s:e, s:e]
        tokens = tokens[s:e]
        offset = s
    lap = gc.construct_laplacian(adj.unsqueeze(0))[0].numpy()
    vals, vecs = SpectralAnalyzer(cfg).compute_eigendecomposition(lap)
    u = vecs[:, 1].astype(np.float64) if vecs.shape[1] > 1 else np.zeros(len(tokens))
    if head is None and f"blocks.{layer}.graph.fiedler_vector" in cache:
        u = np.asarray(cache[f"blocks.{layer}.graph.fiedler_vector"], dtype=np.float64)
    elif u.size and u[np.argmax(np.abs(u))] < 0:
        u = -u
    return adj.numpy().astype(np.float64), np.asarray(vals, float), u, tokens, offset


# ── graph ───────────────────────────────────────────────────────────────────

def graph(cache, layer: int, head: Optional[int] = None, text: int = 0,
          max_edges: int = 400, min_weight: float = 0.02, size: int = 520) -> View:
    """The attention graph of one layer (head-averaged) or one head.

    Tokens are nodes on a circle, clockwise from the top. Edge opacity and
    width follow the symmetrised attention weight, relative to the largest
    weight off the diagonal; node colour is the graph's Fiedler vector (blue
    negative, orange positive), the eigenvector of lambda_2 that splits the
    graph along its weakest cut. Hover a node for its token, Fiedler entry
    and degree, and to highlight its edges; hover an edge for its weight.

    Args:
        cache: a :class:`~spectral_trust.cache.SpectralCache` holding the
            attention patterns (``keep_patterns``).
        layer: true layer index.
        head: one head, or None for the head-averaged graph.
        text: which text of a batch.
        max_edges: draw at most this many edges, the strongest.
        min_weight: skip edges below this fraction of the largest weight.
        size: width of the drawing in pixels.
    """
    c = _single(cache, text)
    W, eig, u, tokens, offset = _graph_data(c, layer, head)
    n = len(tokens)
    width, top = float(size), 58.0
    height = size * 0.86 + top
    cx, cy = width / 2, top + (height - top) / 2 - 6
    radius = min(width, height - top) * 0.34
    theta = np.pi / 2 + np.pi / max(n, 1) - 2 * np.pi * np.arange(n) / max(n, 1)
    pos = np.stack([cx + radius * np.cos(theta), cy - radius * np.sin(theta)], axis=1)

    off = W.copy()
    np.fill_diagonal(off, 0.0)
    wmax = float(off.max()) if off.size and off.max() > 0 else 1.0
    pairs = [(off[i, j] / wmax, i, j) for i in range(n) for j in range(i + 1, n)
             if off[i, j] / wmax >= min_weight]
    pairs.sort()
    pairs = pairs[-max_edges:] if max_edges else pairs
    deg = W.sum(1)
    umax = float(np.max(np.abs(u))) if u.size and np.max(np.abs(u)) > 0 else 1.0
    lam2 = float(eig[1]) if eig.size > 1 else 0.0

    what = "head-averaged graph" if head is None else f"head {head}"
    title = f"layer {layer}, {what}"
    out = [_svg_open(width, height, f"Attention graph, {title}")]
    out.append(_text(14, 24, f"layer {layer}", 16, weight="600"))
    out.append(_text(14 + 9.5 * len(f"layer {layer}"), 24, f"  {what}", 13, color=MID))
    out.append(_text(14, 44, f"λ₂ = {lam2:.4f}   {n} tokens"
                     + (f" (span {offset}-{offset + n})" if offset or _span(c) else ""),
                     12, color=ORANGE_DARK))
    out.append('<g stroke-linecap="round">')
    for w, i, j in pairs:
        (x1, y1), (x2, y2) = pos[i], pos[j]
        tip = (f"{tokens[i]!r} ↔ {tokens[j]!r}\ntokens {i + offset} and {j + offset}\n"
               f"weight {W[i, j]:.4f} ({w:.0%} of the largest)")
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                   f'stroke="{INK}" stroke-opacity="{0.06 + 0.8 * w:.3f}" '
                   f'stroke-width="{0.5 + 3.5 * w:.2f}" data-a="{i}" data-b="{j}">'
                   f'<title>{_esc(tip)}</title></line>')
    out.append("</g>")
    r_node = max(5.0, min(15.0, 260.0 / max(n, 1)))
    font = max(8, min(13, int(300 / max(n, 1)) + 6))
    for i in range(n):
        x, y = pos[i]
        col = _ramp(FIEDLER_STOPS, 0.5 + 0.5 * u[i] / umax)
        tip = (f"{tokens[i]!r} (token {i + offset})\nFiedler vector {u[i]:+.4f}\n"
               f"degree {deg[i]:.4f}")
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_node:.1f}" fill="{col}" '
                   f'stroke="{INK}" stroke-width="0.8" data-node="{i}">'
                   f'<title>{_esc(tip)}</title></circle>')
        dx, dy = (x - cx) / radius, (y - cy) / radius
        lx, ly = cx + (radius + r_node + 6) * dx, cy + (radius + r_node + 6) * dy
        anchor = "start" if dx > 0.25 else "end" if dx < -0.25 else "middle"
        base = "hanging" if dy > 0.3 else "auto" if dy < -0.3 else "middle"
        out.append(_text(lx, ly, _token_label(tokens[i]), font, anchor=anchor,
                         baseline=base))
    # colour legend of the Fiedler vector
    lx, ly, lw = width - 170, 18, 120
    stops = "".join(f'<stop offset="{k / 2:.1f}" stop-color="{s}"/>'
                    for k, s in enumerate(FIEDLER_STOPS))
    gid = f"fg{next(_ids)}"
    out.append(f'<defs><linearGradient id="{gid}">{stops}</linearGradient></defs>')
    out.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="9" fill="url(#{gid})" '
               f'stroke="{GREY}" stroke-width="0.5"/>')
    out.append(_text(lx, ly + 22, "−", 11, color=MID))
    out.append(_text(lx + lw / 2, ly + 22, "Fiedler vector", 11, anchor="middle", color=MID))
    out.append(_text(lx + lw, ly + 22, "+", 11, anchor="end", color=MID))
    out.append("</svg>")
    return View("".join(out), f"Attention graph: {title}", "graph")


# ── heatmap ─────────────────────────────────────────────────────────────────

def _heat_grid(cache, metric: str, text: int):
    """[rows, cols] values, row labels, column labels, column axis name and
    the qualified metric name."""
    q = metric if "." in metric else None
    if q is None:
        from .cache import NAMESPACE_METRICS
        q = (f"attn.{metric}" if metric in NAMESPACE_METRICS["attn"] else
             f"resid.{metric}" if metric in NAMESPACE_METRICS["resid"] else f"graph.{metric}")
    if q not in cache.metric_names:
        raise KeyError(f"{q!r} is not in the cache; metrics present: "
                       f"{', '.join(cache.metric_names)}")
    layers = cache.layers
    if q.startswith("graph."):
        cols = cache.n_texts
        grid = np.array([[float(np.asarray(cache.text(t)[f"blocks.{lay}.{q}"]))
                          for t in range(cols)] for lay in layers])
        return grid, layers, list(range(cols)), "text", q
    c = _single(cache, text)
    grid = c.metric(q)
    return grid, layers, list(range(grid.shape[1])), "head", q


def heatmap(cache, metric: str = "fiedler_value", text: int = 0,
            backend: str = "svg", cell: Optional[float] = None, robust: bool = True):
    """One metric over layers x heads, one cell per (layer, head).

    A bare per-head name means the ``attn`` namespace; pass ``"resid.hfer"``
    for a hybrid one. A ``graph`` metric is drawn layers x texts. Heads that
    were not computed are left blank.

    Args:
        cache: a :class:`~spectral_trust.cache.SpectralCache`.
        metric: metric name, e.g. ``"fiedler_value"`` or ``"resid.energy"``.
        text: which text of a batch (per-head metrics).
        backend: ``"svg"`` (interactive :class:`View`) or ``"matplotlib"``
            (a ``matplotlib.figure.Figure``).
        cell: cell size in pixels (default: fit about 620 px).
        robust: colour scale from the 1st to the 99th percentile, so one
            extreme head does not wash out the rest.
    """
    grid, rows, cols, col_name, q = _heat_grid(cache, metric, text)
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    elif robust:
        lo, hi = (float(v) for v in np.percentile(finite, [1, 99]))
    else:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1e-12
    if backend == "matplotlib":
        return _heatmap_matplotlib(grid, rows, cols, col_name, q, lo, hi)
    if backend != "svg":
        raise ValueError("backend must be 'svg' or 'matplotlib'")

    nr, nc = grid.shape
    if cell is None:
        cell = float(np.clip(min(560 / max(nc, 1), 520 / max(nr, 1)), 6, 28))
    left, top = 52.0, 50.0
    width = left + nc * cell + 110
    height = top + nr * cell + 44
    out = [_svg_open(width, height, f"Heatmap of {q} over layers and {col_name}s")]
    out.append(_text(left, 22, q, 15, weight="600", mono=True))
    out.append(_text(left, 40, f"{'per head, ' if col_name == 'head' else ''}"
                     f"{nr} layers x {nc} {col_name}s", 11, color=MID))
    for r, lay in enumerate(rows):
        for k, col in enumerate(cols):
            v = grid[r, k]
            t = (v - lo) / (hi - lo) if np.isfinite(v) else np.nan
            fill = _ramp(HEAT_STOPS, t)
            tip = (f"layer {lay}, {col_name} {col}\n{q} {_num(v)}" if np.isfinite(v)
                   else f"layer {lay}, {col_name} {col}\nnot computed")
            out.append(f'<rect x="{left + k * cell:.2f}" y="{top + r * cell:.2f}" '
                       f'width="{cell + 0.05:.2f}" height="{cell + 0.05:.2f}" fill="{fill}">'
                       f'<title>{_esc(tip)}</title></rect>')
    out.append(f'<rect x="{left}" y="{top}" width="{nc * cell:.2f}" height="{nr * cell:.2f}" '
               f'fill="none" stroke="{INK}" stroke-width="0.6"/>')
    step_r = max(1, int(np.ceil(nr / 8)))
    for r in range(0, nr, step_r):
        out.append(_text(left - 6, top + (r + 0.5) * cell, str(rows[r]), 10, anchor="end",
                         baseline="middle", color=MID))
    step_c = max(1, int(np.ceil(nc / 10)))
    for k in range(0, nc, step_c):
        out.append(_text(left + (k + 0.5) * cell, top + nr * cell + 14, str(cols[k]), 10,
                         anchor="middle", color=MID))
    out.append(_text(left + nc * cell / 2, top + nr * cell + 32, col_name, 11,
                     anchor="middle", color=MID))
    out.append(f'<text x="{left - 34}" y="{top + nr * cell / 2:.1f}" font-size="11" '
               f'fill="{MID}" text-anchor="middle" transform="rotate(-90 {left - 34} '
               f'{top + nr * cell / 2:.1f})">layer</text>')
    # colour bar
    bx, by, bh = left + nc * cell + 22, top, min(nr * cell, 220)
    gid = f"hg{next(_ids)}"
    # top of the bar is the high end; SVG stop offsets must increase
    top_down = HEAT_STOPS[::-1]
    stops = "".join(f'<stop offset="{k / (len(top_down) - 1):.3f}" stop-color="{s}"/>'
                    for k, s in enumerate(top_down))
    out.append(f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">{stops}'
               f'</linearGradient></defs>')
    out.append(f'<rect x="{bx}" y="{by}" width="11" height="{bh:.1f}" fill="url(#{gid})" '
               f'stroke="{GREY}" stroke-width="0.5"/>')
    out.append(_text(bx + 16, by + 4, _num(hi), 10, baseline="middle", color=MID))
    out.append(_text(bx + 16, by + bh - 2, _num(lo), 10, baseline="middle", color=MID))
    out.append("</svg>")
    return View("".join(out), f"Heatmap: {q}", "heatmap")


def _heatmap_matplotlib(grid, rows, cols, col_name, q, lo, hi):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("spectral_trust_heat", HEAT_STOPS) \
        .with_extremes(bad=NAN_FILL)
    nr, nc = grid.shape
    fig, ax = plt.subplots(figsize=(min(1.6 + 0.28 * nc, 10), min(1.2 + 0.2 * nr, 9)))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=Normalize(lo, hi),
                   aspect="auto", interpolation="nearest")
    ax.set_xlabel(col_name)
    ax.set_ylabel("layer")
    step_r = max(1, int(np.ceil(nr / 8)))
    ax.set_yticks(range(0, nr, step_r))
    ax.set_yticklabels([str(rows[r]) for r in range(0, nr, step_r)])
    step_c = max(1, int(np.ceil(nc / 10)))
    ax.set_xticks(range(0, nc, step_c))
    ax.set_xticklabels([str(cols[k]) for k in range(0, nc, step_c)])
    ax.set_title(q, loc="left", fontsize=11, family="monospace")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    fig.tight_layout()
    return fig


# ── spectrum ────────────────────────────────────────────────────────────────

def spectrum(cache, layer: int, head: Optional[int] = None, text: int = 0,
             width: int = 560) -> View:
    """The Laplacian spectrum of one layer (head-averaged graph) or head.

    Eigenvalues are drawn in ascending order, lambda_2 (the Fiedler value)
    in orange. The shaded band marks the top ``num_high_modes`` modes, whose
    share of the signal energy is the hybrid HFER. Hover a point for its
    value.
    """
    from .fast import num_high_modes

    c = _single(cache, text)
    if head is None:
        key = f"blocks.{layer}.graph.eigenvalues"
        if key not in c:
            raise KeyError(f"{key!r} is not in the cache; compute a graph metric "
                           "(metrics=None or 'graph') for the head-averaged spectrum")
        lam = np.asarray(c[key], dtype=np.float64)
        what = "head-averaged graph"
    else:
        key = f"blocks.{layer}.attn.eigenvalues"
        if key not in c:
            raise KeyError(f"{key!r} is not in the cache; compute an attn metric")
        lam = np.asarray(c[key], dtype=np.float64)[head]
        if not np.all(np.isfinite(lam)):
            raise KeyError(f"head {head} of layer {layer} was not computed")
        what = f"head {head}"
    n = lam.size
    height, left, top, right, bottom = 300.0, 48.0, 54.0, 18.0, 40.0
    pw, ph = width - left - right, height - top - bottom
    ymax = 2.0 if c.config.normalization in ("sym", "rw") and lam.max(initial=0) <= 2 + 1e-6 \
        else float(max(lam.max(initial=1.0), 1e-9)) * 1.05

    def X(k):          # k is 1-based mode index
        return left + (k - 0.5) / max(n, 1) * pw

    def Y(v):
        return top + ph - v / ymax * ph

    out = [_svg_open(width, height, f"Laplacian spectrum, layer {layer}, {what}")]
    out.append(_text(14, 24, f"layer {layer}", 16, weight="600"))
    out.append(_text(14 + 9.5 * len(f"layer {layer}"), 24, f"  {what}", 13, color=MID))
    lam2 = lam[1] if n > 1 else 0.0
    out.append(_text(14, 44, f"λ₂ = {lam2:.4f}   λ_max = {lam[-1]:.4f}   "
                     f"{n} modes", 12, color=ORANGE_DARK))
    hi = num_high_modes(n, c.config.hfer_cutoff_ratio) if n else 0
    if hi:
        x0 = left + (n - hi) / n * pw
        out.append(f'<rect x="{x0:.1f}" y="{top}" width="{left + pw - x0:.1f}" '
                   f'height="{ph}" fill="{GRID}"><title>top {hi} modes: the high-frequency '
                   f'band of the hybrid HFER</title></rect>')
        out.append(_text(left + pw - 4, top + 12, "HFER modes", 10, anchor="end", color=MID))
    for v in ([0, 1, 2] if ymax == 2.0 else list(np.linspace(0, ymax, 3))):
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                   f'stroke="{GRID if v else INK}" stroke-width="{0.6 if v else 0.8}"/>')
        out.append(_text(left - 6, Y(v), f"{v:g}", 10, anchor="end", baseline="middle",
                         color=MID))
    pts = " ".join(f"{X(k + 1):.1f},{Y(v):.1f}" for k, v in enumerate(lam))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{GREY}" stroke-width="1"/>')
    r = max(1.6, min(4.0, 220.0 / max(n, 1)))
    for k, v in enumerate(lam):
        is2 = k == 1
        out.append(f'<circle cx="{X(k + 1):.1f}" cy="{Y(v):.1f}" r="{r * (1.5 if is2 else 1):.1f}" '
                   f'fill="{ORANGE if is2 else INK}" stroke="{INK}" stroke-width="{0.6 if is2 else 0}" '
                   f'><title>mode {k + 1}: λ = {v:.5f}'
                   f'{" (Fiedler value)" if is2 else ""}</title></circle>')
    step = max(1, int(np.ceil(n / 8)))
    for k in range(0, n, step):
        out.append(_text(X(k + 1), top + ph + 14, str(k + 1), 10, anchor="middle", color=MID))
    out.append(_text(left + pw / 2, top + ph + 32, "mode k (ascending eigenvalue)", 11,
                     anchor="middle", color=MID))
    out.append("</svg>")
    return View("".join(out), f"Spectrum: layer {layer}, {what}", "spectrum")


__all__ = ["View", "graph", "heatmap", "spectrum"]
