"""Regenerate the README figures in docs/assets/ from the library's own output.

    python docs/make_assets.py                  # every asset
    python docs/make_assets.py hero logo        # a subset of: hero, logo, heads, speedup
    python docs/make_assets.py --no-tex         # matplotlib's STIX type instead of LaTeX

hero      hero.png and hero-dark.png (light and dark themes): three lines of code and
          their output on Qwen/Qwen3-0.6B, from SpectralModel.run_with_spectra. One
          head's attention graph (node colour: Fiedler vector), the per-head
          fiedler_value of every layer and head, and that head's Laplacian spectrum.
          The head is, among those with a Fiedler value below 0.1, the one whose
          Fiedler vector splits the tokens most evenly.
logo      logo.svg and logo-dark.svg: a six-token graph split by its Fiedler vector,
          and the name, with the text converted to outlines.
heads     heads.png: per-head attention-only metrics on one prompt, one cell per
          (layer, head), from per_head_metrics_all_layers.
speedup   speedup.png: time per call under the 0.3.0 and 0.4.0 defaults, read from
          benchmarks/results.json (written by benchmarks/end_to_end.py).

The model runs on the CPU only (CUDA_VISIBLE_DEVICES=-1, at most 4 threads) and is
read from the local Hugging Face cache (HF_HUB_OFFLINE=1). Run once with
HF_HUB_OFFLINE=0 to download it. Type is LaTeX with newtx when latex, dvipng and
newtx are installed, and matplotlib's STIX otherwise; both are Times designs. Code
is set in DejaVu Sans Mono. Needs matplotlib and Pillow, both dependencies of the
package.
"""
import os

# Before torch is imported: CPU only, 4 threads, local files only.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
RESULTS = ROOT / "benchmarks" / "results.json"

MODEL = "Qwen/Qwen3-0.6B"
HERO_PROMPT = "The cat sat on the mat because it was warm."
HEAD_PROMPT = ("The cat sat on the mat because it was warm. Later the dog came in, lay down "
               "next to it, and both of them slept until the sun went down.")
PROMPTS = (HERO_PROMPT, HEAD_PROMPT)    # one model load for every asset

# Palette: orange the measured series, blue the second series or a reference,
# greys for context; INK for text and axes.
INK, GREY, BAND, GRID = "#1A1A1A", "#9E9E9E", "#D9D9D9", "#EBEBEB"
ORANGE, ORANGE_DARK, BLUE, GREEN, MID = "#E69F00", "#B35900", "#0072B2", "#009E73", "#7F7F7F"
CTRL = dict(marker="o", s=34, facecolor="white", edgecolor=INK, linewidths=0.9)
EFFECT = dict(marker="o", s=36, facecolor=ORANGE, edgecolor=INK, linewidths=0.6)

PREAMBLE = (r"\usepackage[T1]{fontenc}\usepackage{amsmath}"
            r"\usepackage{newtxtext}\usepackage{newtxmath}")
RENDER_DPI = 200          # frames are drawn at 2x and downsampled to 900 px
TEX = False


# ── style ────────────────────────────────────────────────────────────────────

def tex_available() -> bool:
    if not (shutil.which("latex") and shutil.which("dvipng") and shutil.which("kpsewhich")):
        return False
    try:
        out = subprocess.run(["kpsewhich", "newtxtext.sty"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def setup_style(use_tex: bool) -> None:
    global TEX
    TEX = bool(use_tex and tex_available())
    rc = {
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 10, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3,
        "ytick.major.size": 3, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "axes.labelpad": 3.0, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 1.2, "lines.solid_capstyle": "round",
    }
    if TEX:
        rc.update({"text.usetex": True, "text.latex.preamble": PREAMBLE,
                   "font.family": "serif"})
    else:
        rc.update({"text.usetex": False, "font.family": "serif",
                   "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
    plt.rcParams.update(rc)


_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}"}


def esc(s: str) -> str:
    """Plain text that renders literally under either text engine."""
    if TEX:
        return "".join(_TEX_ESCAPES.get(c, c) for c in s)
    return s.replace("$", r"\$")


def code(s: str) -> str:
    """An identifier from the library, in typewriter type under LaTeX."""
    return r"\texttt{" + esc(s) + "}" if TEX else esc(s)


def axes_in(fig, left, bottom, width, height):
    """Axes placed in inches from the lower left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def text_in(fig, x, y, s, **kw):
    W, H = fig.get_size_inches()
    return fig.text(x / W, y / H, s, **kw)


def repel(ys, gap, floor=-np.inf, ceil=np.inf):
    """Spread label positions so that neighbours are at least `gap` apart."""
    order = np.argsort(ys)
    out = np.clip(np.array(ys, float), floor, ceil)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    if out[order[-1]] > ceil:
        out[order[-1]] = ceil
        for a, b in zip(order[::-1][:-1], order[::-1][1:]):
            if out[a] - out[b] < gap:
                out[b] = out[a] - gap
    return out


def render(fig) -> Image.Image:
    """Draw a figure and return it as an RGB image, 900 px wide."""
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    w, h = img.size
    return img.resize((900, round(h * 900 / w)), Image.LANCZOS)


def save_png(fig, path: Path) -> None:
    """Still figures are saved at 2x (1800 px) for sharp display at 900 px."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=RENDER_DPI)
    plt.close(fig)
    with Image.open(path) as im:
        im.convert("RGB").save(path, optimize=True)


# ── model output ─────────────────────────────────────────────────────────────

_RUNS = {}


def run_model(texts=PROMPTS):
    """analyze_text on each prompt with GSPConfig defaults, on the CPU."""
    key = tuple(texts)
    if key in _RUNS:
        return _RUNS[key]
    import transformers
    from spectral_trust import GSPConfig, GSPDiagnosticsFramework

    transformers.logging.set_verbosity_error()
    torch.set_num_threads(4)
    offline = os.environ.get("HF_HUB_OFFLINE", "1") not in ("0", "false", "False")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = GSPConfig(model_name=MODEL, device="cpu", save_plots=False, verbose=False,
                        output_dir=tmp, local_files_only=offline)
        with GSPDiagnosticsFramework(cfg) as fw:
            fw.instrumenter.load_model(MODEL)
            tok = fw.instrumenter.tokenizer
            runs = []
            for text in texts:
                res = fw.analyze_text(text, save_results=False)
                ids = res["model_outputs"]["inputs"]["input_ids"][0].tolist()
                res["token_text"] = [tok.decode([i]) for i in ids]
                runs.append(res)
    _RUNS[key] = (cfg, runs)
    return cfg, runs


# ── static hero: three lines of code and their output ───────────────────────

HERO_CODE = [
    "import spectral_trust as st",
    f'model = st.SpectralModel.from_pretrained("{MODEL}")',
    f'cache = model.run_with_spectra("{HERO_PROMPT}")',
]

# Light and dark themes of the hero and the logo. The dark one follows GitHub's
# dark background; the data colours (orange, blue) are the same in both.
THEMES = {
    "light": dict(bg="#FFFFFF", ink=INK, mid=MID, grid=GRID, band=BAND, code_bg="#F6F8FA",
                  code_edge="#D0D7DE", string=BLUE, node_edge=INK,
                  heat=["#FFFFFF", ORANGE, ORANGE_DARK, "#3D1F00"]),
    "dark": dict(bg="#0D1117", ink="#E6EDF3", mid="#8B949E", grid="#21262D", band="#30363D",
                 code_bg="#161B22", code_edge="#30363D", string="#79C0FF", node_edge="#E6EDF3",
                 heat=["#161B22", ORANGE_DARK, ORANGE, "#FFE7B3"]),
}


def hero_cache():
    """run_with_spectra on the hero prompt, on the CPU, from the local cache."""
    import transformers
    from spectral_trust import SpectralModel

    transformers.logging.set_verbosity_error()
    torch.set_num_threads(4)
    offline = os.environ.get("HF_HUB_OFFLINE", "1") not in ("0", "false", "False")
    model = SpectralModel.from_pretrained(MODEL, local_files_only=offline)
    return model.run_with_spectra(HERO_PROMPT)


def hero_head(cache, weak_cut: float = 0.1):
    """The head drawn in the hero: among the heads whose Fiedler value is
    below ``weak_cut`` (a weak two-way cut), the one whose Fiedler vector
    splits the signal mass most evenly between its two signs."""
    best, choice = -1.0, None
    for layer in cache.layers:
        for head in np.flatnonzero(cache[f"blocks.{layer}.attn.fiedler_value"] < weak_cut):
            _, _, u = _head_graph(cache, layer, int(head))
            balance = min((u[u > 0] ** 2).sum(), (u[u < 0] ** 2).sum())
            if balance > best:
                best, choice = balance, (layer, int(head))
    return choice


def _head_graph(cache, layer, head):
    """Symmetrised attention of one head and the Fiedler vector of its
    Laplacian, built exactly as the library builds it (GSPConfig defaults)."""
    from spectral_trust import GraphConstructor

    gc = GraphConstructor(cache.config)
    pattern = torch.as_tensor(cache[f"blocks.{layer}.attn.pattern"])
    W = gc.symmetrize_attention(pattern.unsqueeze(0))[0, head]
    L = gc.construct_laplacian(W.unsqueeze(0))[0].numpy().astype(np.float64)
    lam, vec = np.linalg.eigh(0.5 * (L + L.T))
    lam = np.maximum(lam, 0.0)
    # the drawn spectrum is the one the cache holds
    assert np.allclose(lam, cache[f"blocks.{layer}.attn.eigenvalues"][head], atol=1e-4)
    u = vec[:, 1]
    u = -u if u[np.argmax(np.abs(u))] < 0 else u
    return W.numpy().astype(np.float64), lam, u


def draw_hero(cache, theme: str):
    from matplotlib.patches import FancyBboxPatch, Rectangle

    t = THEMES[theme]
    layer, head = hero_head(cache)
    W, lam, u = _head_graph(cache, layer, head)
    tokens = [s.strip() or s for s in cache.tokens]
    n = len(tokens)
    fig = plt.figure(figsize=(10.0, 4.75), dpi=RENDER_DPI)
    fig.patch.set_facecolor(t["bg"])

    # the code, typeset as code (never through LaTeX)
    box = FancyBboxPatch((0.25 / 10, 3.62 / 4.75), 9.5 / 10, 0.98 / 4.75,
                         boxstyle="round,pad=0.0,rounding_size=0.012", transform=fig.transFigure,
                         facecolor=t["code_bg"], edgecolor=t["code_edge"], linewidth=0.8)
    fig.patches.append(box)
    for k, line in enumerate(HERO_CODE):
        y = 4.39 - 0.27 * k
        x = 0.45
        parts = line.split('"')
        for j, part in enumerate(parts):
            piece = part if j % 2 == 0 else f'"{part}"'
            colour = t["string"] if j % 2 else t["ink"]
            txt = text_in(fig, x, y, piece, fontsize=10.5, family="DejaVu Sans Mono",
                          color=colour, ha="left", va="baseline", usetex=False)
            fig.canvas.draw()
            bb = txt.get_window_extent()
            x += bb.width / fig.dpi

    def title(ax_left, s):
        text_in(fig, ax_left, 3.22, s, fontsize=9.5, family="DejaVu Sans Mono", color=t["mid"],
                ha="left", va="baseline", usetex=False)

    # 1. one head's attention graph
    title(0.3, f"cache.graph({layer}, head={head})")
    ax = axes_in(fig, 0.25, 0.3, 3.15, 2.75)
    ax.set_facecolor(t["bg"])
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    theta = np.pi / 2 + np.pi / n - 2 * np.pi * np.arange(n) / n
    pos = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    off = W.copy()
    np.fill_diagonal(off, 0.0)
    wmax = float(off.max())
    for w, a, b in sorted((off[a, b] / wmax, a, b) for a in range(n) for b in range(a + 1, n)):
        ax.plot(*pos[[a, b]].T, color=t["ink"], lw=0.3 + 4.5 * w, alpha=0.06 + 0.74 * w,
                solid_capstyle="round", zorder=1)
    cmap = LinearSegmentedColormap.from_list("fiedler", [BLUE, "#FFFFFF", ORANGE])
    umax = float(np.max(np.abs(u)))
    ax.scatter(pos[:, 0], pos[:, 1], s=250, c=u, cmap=cmap, norm=Normalize(-umax, umax),
               edgecolors=t["node_edge"], linewidths=0.6, zorder=3)
    for (x, y), tok in zip(pos, tokens):
        ha = "left" if x > 0.3 else "right" if x < -0.3 else "center"
        va = "bottom" if y > 0.3 else "top" if y < -0.3 else "center"
        ax.text(1.2 * x, 1.16 * y, esc(tok), ha=ha, va=va, fontsize=9.5, color=t["ink"],
                zorder=4)
    text_in(fig, 0.3, 0.12, "node colour: Fiedler vector,  edges: attention", fontsize=8,
            color=t["mid"], ha="left", va="baseline")

    # 2. every head, layers x heads
    title(3.85, 'cache.heatmap("fiedler_value")')
    grid = cache.metric("attn.fiedler_value")
    n_layers, n_heads = grid.shape
    ax2 = axes_in(fig, 4.25, 0.42, 1.85, 2.62)
    lo, hi = np.percentile(grid, [1, 99])
    heat = LinearSegmentedColormap.from_list("heat_" + theme, t["heat"])
    im = ax2.imshow(grid, cmap=heat, norm=Normalize(lo, hi), aspect="auto",
                    interpolation="nearest", origin="upper")
    row = cache.layers.index(layer)
    ax2.add_patch(Rectangle((head - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=t["string"],
                            linewidth=1.4, zorder=5))
    ax2.set_xticks(sorted({0, n_heads // 2, n_heads - 1}))
    ax2.set_yticks(sorted({0, (n_layers - 1) // 3, 2 * (n_layers - 1) // 3, n_layers - 1}))
    ax2.set_xlabel("head", labelpad=1.5, color=t["ink"])
    ax2.set_ylabel("layer", labelpad=1.5, color=t["ink"])
    for side in ("top", "right"):
        ax2.spines[side].set_visible(True)
    cax = axes_in(fig, 6.2, 0.42, 0.09, 2.62)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(t["ink"])
    cb.ax.tick_params(labelsize=7.5, length=2, pad=1.5, colors=t["ink"])
    cb.locator = matplotlib.ticker.MaxNLocator(4)
    cb.update_ticks()

    # 3. that head's Laplacian spectrum
    title(6.95, f"cache.spectrum({layer}, head={head})")
    ax3 = axes_in(fig, 7.4, 0.42, 2.35, 2.62)
    k = np.arange(1, n + 1)
    ax3.plot(k, lam, color=t["band"], lw=0.9, zorder=1)
    ax3.scatter(k, lam, s=13, color=t["ink"], zorder=2)
    ax3.scatter([2], [lam[1]], s=40, facecolor=ORANGE, edgecolor=t["node_edge"],
                linewidths=0.6, zorder=3)
    ax3.annotate(rf"$\lambda_2 = {lam[1]:.3f}$", xy=(2, lam[1]), xytext=(2.8, lam[1] + 0.42),
                 color=ORANGE if theme == "dark" else ORANGE_DARK, fontsize=10, ha="left",
                 va="bottom", arrowprops=dict(arrowstyle="-", color=t["mid"], lw=0.6,
                                              shrinkA=1, shrinkB=4))
    ax3.set_xlim(0.4, n + 0.6)
    ax3.set_ylim(0, 2.0)
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticks([1] + list(range(4, n + 1, 4)))
    ax3.set_xlabel("mode $k$", labelpad=1.5, color=t["ink"])
    ax3.set_ylabel(r"eigenvalue $\lambda_k$", labelpad=1.5, color=t["ink"])
    ax3.grid(axis="y", color=t["grid"], lw=0.6)
    ax3.set_axisbelow(True)

    for a in (ax2, ax3):
        a.set_facecolor(t["bg"])
        a.tick_params(colors=t["ink"])
        for s in a.spines.values():
            s.set_edgecolor(t["ink"])
    return fig


def make_hero():
    cache = hero_cache()
    layer, head = hero_head(cache)
    for theme, name in (("light", "hero.png"), ("dark", "hero-dark.png")):
        fig = draw_hero(cache, theme)
        path = ASSETS / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=RENDER_DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        with Image.open(path) as im:
            im.convert("RGB").save(path, optimize=True)
            size = im.size
        print(f"  {name}: {size[0]}x{size[1]} px, layer {layer} head {head}")


# ── logo: a graph glyph and the name, as SVG paths ───────────────────────────

def _svg_path(path, dx: float, dy: float, scale: float) -> str:
    """SVG path data of a matplotlib Path, y flipped, scaled and shifted."""
    from matplotlib.path import Path as MPath

    out = []
    for verts, code in path.iter_segments(simplify=False, curves=True):
        pts = [(dx + scale * x, dy - scale * y) for x, y in np.asarray(verts).reshape(-1, 2)]
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        if code == MPath.MOVETO:
            out.append(f"M{p}")
        elif code == MPath.LINETO:
            out.append(f"L{p}")
        elif code == MPath.CURVE3:
            out.append(f"Q{p}")
        elif code == MPath.CURVE4:
            out.append(f"C{p}")
        elif code == MPath.CLOSEPOLY:
            out.append("Z")
    return "".join(out)


def logo_svg(theme: str) -> str:
    """Two tightly connected triples of tokens joined by one weak edge,
    coloured by the sign of the Fiedler vector (the cut it finds), then the
    name. Text is converted to outlines (DejaVu Sans Bold, shipped with
    matplotlib), so the file renders the same everywhere."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    t = THEMES[theme]
    nodes = np.array([[14, 16], [9, 40], [30, 30], [54, 30], [75, 40], [70, 16]], float)
    strong = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    W = np.zeros((6, 6))
    for a, b in strong:
        W[a, b] = W[b, a] = 1.0
    W[2, 3] = W[3, 2] = 0.15
    d = W.sum(1)
    L = np.eye(6) - W / np.sqrt(np.outer(d, d))
    u = np.linalg.eigh(L)[1][:, 1]
    u = u if u[0] < 0 else -u                   # left group blue, right group orange
    parts = []
    for a, b in strong:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["ink"]}" '
                     f'stroke-width="2.4" stroke-linecap="round"/>')
    (x1, y1), (x2, y2) = nodes[2], nodes[3]
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["ink"]}" '
                 f'stroke-width="1.2" stroke-dasharray="2.5 2.5" stroke-linecap="round"/>')
    for (x, y), v in zip(nodes, u):
        colour = BLUE if v < 0 else ORANGE
        parts.append(f'<circle cx="{x}" cy="{y}" r="6.2" fill="{colour}" '
                     f'stroke="{t["ink"]}" stroke-width="1.6"/>')
    font = FontProperties(family="DejaVu Sans", weight="bold")
    name = TextPath((0, 0), "spectral", size=30, prop=font)
    rest = TextPath((0, 0), "-trust", size=30, prop=font)
    x0 = 96.0
    width_name = name.get_extents().x1
    parts.append(f'<path d="{_svg_path(name, x0, 39.5, 1.0)}" fill="{t["ink"]}"/>')
    parts.append(f'<path d="{_svg_path(rest, x0 + width_name + 1.5, 39.5, 1.0)}" '
                 f'fill="{ORANGE if theme == "light" else ORANGE}"/>')
    total = x0 + width_name + 1.5 + rest.get_extents().x1 + 4
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total:.0f} 56" '
            f'width="{total:.0f}" height="56" role="img" aria-label="spectral-trust">'
            f'<title>spectral-trust</title>{"".join(parts)}</svg>\n')


def make_logo():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for theme, name in (("light", "logo.svg"), ("dark", "logo-dark.svg")):
        (ASSETS / name).write_text(logo_svg(theme), encoding="utf-8")
        print(f"  {name}")


# ── per-head metrics, layers x heads ─────────────────────────────────────────

HEAD_METRICS = ("fiedler_value", "spectral_radius", "hfer")
HEAT = LinearSegmentedColormap.from_list("heat", ["#FFFFFF", ORANGE, ORANGE_DARK, "#3D1F00"])


def make_heads():
    from spectral_trust import PER_HEAD_METRIC_FAMILY, PER_HEAD_METRIC_NAMES, per_head_metrics_all_layers

    cfg, runs = run_model()
    res = runs[PROMPTS.index(HEAD_PROMPT)]
    attns = [a.to(torch.float32) for a in res["model_outputs"]["attentions"]]
    per_layer = per_head_metrics_all_layers(attns, config=cfg)
    n_layers, n_heads = len(per_layer), per_layer[0].values.shape[0]
    fig = plt.figure(figsize=(9.0, 4.3), dpi=RENDER_DPI)
    w, h, gap, left, bottom = 2.25, 3.05, 0.62, 0.62, 0.78
    for j, name in enumerate(HEAD_METRICS):
        assert name in PER_HEAD_METRIC_NAMES and PER_HEAD_METRIC_FAMILY == "attention", name
        grid = np.stack([d.metric(name) for d in per_layer])          # [layers, heads]
        ax = axes_in(fig, left + j * (w + gap), bottom, w, h)
        lo, hi = np.percentile(grid, [1, 99])
        im = ax.imshow(grid, cmap=HEAT, norm=Normalize(lo, hi), aspect="auto",
                       interpolation="nearest", origin="upper")
        ax.set_xticks(sorted({0, n_heads // 3, 2 * n_heads // 3, n_heads - 1}))
        ax.set_yticks(sorted({0, (n_layers - 1) // 3, 2 * (n_layers - 1) // 3, n_layers - 1}))
        ax.set_xlabel("head", labelpad=1.5)
        if j == 0:
            ax.set_ylabel("layer")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        ax.text(0.0, 1.03, code(name), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10)
        cax = axes_in(fig, left + j * (w + gap), 0.22, w, 0.09)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=8, length=2, pad=1.5)
        cb.locator = matplotlib.ticker.MaxNLocator(4)
        cb.update_ticks()
    save_png(fig, ASSETS / "heads.png")
    print(f"  heads.png ({n_layers} layers x {n_heads} heads)")


# ── 0.3.0 against 0.4.0 ──────────────────────────────────────────────────────

def make_speedup():
    e2e = json.loads(RESULTS.read_text(encoding="utf-8"))["end_to_end"]
    rows = sorted(e2e.values(), key=lambda r: (r["S"], r["d"]))
    old_v, new_v = rows[0]["old_version"], rows[0]["new_version"]
    fig = plt.figure(figsize=(9.0, 2.9), dpi=RENDER_DPI)
    panels = (
        (1.45, "analyze_layer", "one head-averaged layer"),
        (5.55, "per_head", f"per-head hybrid metrics, {rows[0]['heads']} heads"),
    )
    ys = np.arange(len(rows))[::-1].astype(float)
    for p, (left, key, what) in enumerate(panels):
        ax = axes_in(fig, left, 0.55, 2.75, 1.95)
        for y, r in zip(ys, rows):
            old, new = r[f"old_{key}_s"], r[f"new_{key}_s"]
            ax.plot([new, old], [y, y], color=BAND, lw=2.6, zorder=1, solid_capstyle="butt")
            ax.scatter([old], [y], **CTRL, zorder=3)
            ax.scatter([new], [y], **EFFECT, zorder=3)
            ax.text(1.04, y, f"{r[f'{key}_speedup']:.1f}" + r"$\times$", transform=
                    blended_transform_factory(ax.transAxes, ax.transData), fontsize=9.5,
                    ha="left", va="center", clip_on=False)
        top = rows[0]
        ax.text(top[f"old_{key}_s"], ys[0] + 0.48, old_v, ha="center", va="bottom", fontsize=9)
        ax.text(top[f"new_{key}_s"], ys[0] + 0.48, new_v, ha="center", va="bottom", fontsize=9,
                color=ORANGE_DARK)
        ax.text(1.04, ys[0] + 0.48, "speedup", transform=blended_transform_factory(
            ax.transAxes, ax.transData), ha="left", va="bottom", fontsize=8.5, color=MID,
            clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(1.5e-3, 80)
        ticks = [1e-2, 1e-1, 1, 10]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["10 ms", "100 ms", "1 s", "10 s"])
        ax.minorticks_off()
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.set_yticks(ys)
        if p == 0:
            ax.set_yticklabels([f"$S = {r['S']}$, $d = {r['d']}$" for r in rows])
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, pad=4)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        label = (code("analyze_layer") + f", {what}") if key == "analyze_layer" else what
        ax.set_xlabel(label + ": time per call")
    save_png(fig, ASSETS / "speedup.png")
    print("  speedup.png")


# ── entry point ──────────────────────────────────────────────────────────────

ASSET_MAKERS = {"hero": make_hero, "logo": make_logo, "heads": make_heads,
                "speedup": make_speedup}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("assets", nargs="*", choices=[[], *ASSET_MAKERS], default=[],
                    help="assets to rebuild (default: all)")
    ap.add_argument("--no-tex", action="store_true", help="use STIX type instead of LaTeX")
    args = ap.parse_args()
    setup_style(not args.no_tex)
    print(f"type: {'LaTeX (newtx)' if TEX else 'STIX'}")
    for name in args.assets or ASSET_MAKERS:
        ASSET_MAKERS[name]()
    print(f"wrote {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
