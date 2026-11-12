"""Regenerate the README figures in docs/assets/ from the library's own output.

    python docs/make_assets.py                  # every asset
    python docs/make_assets.py hero heads       # a subset of: hero, heads, speedup
    python docs/make_assets.py --no-tex         # matplotlib's STIX type instead of LaTeX

hero      hero.gif, and hero.png as a still fallback. Qwen/Qwen3-0.6B reads one prompt;
          for each layer, the head-averaged attention graph that analyze_text builds
          (one node per token, edge width = symmetrised attention), its Fiedler vector
          as node colour, the layer's Laplacian spectrum, and fiedler_value and hfer
          up to that layer.
heads     heads.png: per-head attention-only metrics on one prompt, one cell per
          (layer, head), from per_head_metrics_all_layers.
speedup   speedup.png: time per call under the 0.3.0 and 0.4.0 defaults, read from
          benchmarks/results.json (written by benchmarks/end_to_end.py).

The model runs on the CPU only (CUDA_VISIBLE_DEVICES=-1, at most 4 threads) and is
read from the local Hugging Face cache (HF_HUB_OFFLINE=1). Run once with
HF_HUB_OFFLINE=0 to download it. Type is LaTeX with newtx when latex, dvipng and
newtx are installed, and matplotlib's STIX otherwise; both are Times designs.
Needs matplotlib and Pillow, both dependencies of the package.
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


# ── hero animation ───────────────────────────────────────────────────────────

def hero_frames(cfg, run):
    """Per layer: the adjacency analyze_text built, and the library's spectrum."""
    from spectral_trust import GraphConstructor

    gc = GraphConstructor(cfg)
    frames, prev = [], None
    for diag, att in zip(run["layer_diagnostics"], run["model_outputs"]["attentions"]):
        W = gc.symmetrize_attention(att)                  # [1, H, S, S]
        A = gc.aggregate_heads(W)                         # [1, S, S], head average
        L = gc.construct_laplacian(A)[0].numpy()
        # The drawn graph is the one the library analysed.
        lam = np.maximum(np.linalg.eigvalsh(0.5 * (L + L.T)), 0.0)
        assert np.allclose(lam, diag.eigenvalues, atol=1e-4), diag.layer
        u = np.asarray(diag.eigenvectors[:, 1], dtype=float)
        # An eigenvector's sign is arbitrary; keep it continuous across layers.
        flip = (u[np.argmax(np.abs(u))] < 0) if prev is None else (u @ prev < 0)
        u = -u if flip else u
        prev = u
        frames.append(dict(layer=diag.layer, A=A[0].numpy(), u=u, eig=np.asarray(diag.eigenvalues),
                           fiedler=diag.fiedler_value, hfer=diag.hfer))
    return frames


def draw_hero(frames, i, tokens, cutoff_ratio, umax):
    from spectral_trust import num_high_modes

    f = frames[i]
    n = len(tokens)
    L = len(frames)
    fig = plt.figure(figsize=(9.0, 4.0), dpi=RENDER_DPI)

    # graph: tokens clockwise from the top left
    ax = axes_in(fig, 0.25, 0.45, 3.9, 3.35)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    ax.axis("off")
    theta = np.pi / 2 + np.pi / n - 2 * np.pi * np.arange(n) / n
    pos = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    A = f["A"]
    off = A[~np.eye(n, dtype=bool)]
    wmax = float(off.max()) if off.size else 1.0
    edges = sorted(((A[a, b] / wmax, a, b) for a in range(n) for b in range(a + 1, n)))
    for w, a, b in edges:
        ax.plot(*pos[[a, b]].T, color=INK, lw=0.3 + 6.5 * w, alpha=0.10 + 0.75 * w,
                solid_capstyle="round", zorder=1)
    cmap = LinearSegmentedColormap.from_list("fiedler", [BLUE, "#FFFFFF", ORANGE])
    norm = Normalize(-umax, umax)
    ax.scatter(pos[:, 0], pos[:, 1], s=430, c=f["u"], cmap=cmap, norm=norm,
               edgecolors=INK, linewidths=0.7, zorder=3)
    for (x, y), t in zip(pos, tokens):
        ha = "left" if x > 0.3 else "right" if x < -0.3 else "center"
        va = "bottom" if y > 0.3 else "top" if y < -0.3 else "center"
        ax.text(1.2 * x, 1.17 * y, esc(t.strip() or t), ha=ha, va=va, fontsize=11, zorder=4)

    text_in(fig, 0.2, 3.66, f"layer {f['layer']}", fontsize=14, ha="left", va="baseline")
    text_in(fig, 0.2, 3.44, esc(f"{MODEL}, {L} layers"), fontsize=8.5, color=MID,
            ha="left", va="baseline")

    # colour scale of the Fiedler vector
    cax = axes_in(fig, 1.45, 0.30, 1.5, 0.08)
    cax.imshow(np.linspace(-1, 1, 256)[None, :], aspect="auto", cmap=cmap,
               extent=(-umax, umax, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-umax, 0, umax])
    cax.set_xticklabels(["$-$", "0", "$+$"])
    cax.tick_params(length=2, labelsize=8, pad=1.5)
    for s in cax.spines.values():
        s.set_visible(False)
    text_in(fig, 2.2, 0.06, "Fiedler vector entry (node colour)", fontsize=8.5, color=MID,
            ha="center", va="baseline")

    # the layer's Laplacian spectrum
    ax2 = axes_in(fig, 5.05, 2.45, 2.55, 1.25)
    k = np.arange(1, n + 1)
    hi = num_high_modes(n, cutoff_ratio)
    ax2.axvspan(n - hi + 0.5, n + 0.5, color=GRID, lw=0, zorder=0)
    ax2.text(n - hi / 2 + 0.5, 1.93, "HFER\nmodes", fontsize=7.5, color=MID, ha="center",
             va="top", linespacing=1.0)
    ax2.plot(k, f["eig"], color=GREY, lw=0.8, zorder=1)
    ax2.scatter(k, f["eig"], s=14, color=INK, zorder=2)
    ax2.scatter([2], [f["eig"][1]], **EFFECT, zorder=3)
    # Upper left stays empty: the sym eigenvalues average 1, so the low modes sit below 1.
    ax2.text(0.04, 0.96, rf"$\lambda_2 = {f['fiedler']:.3f}$", transform=ax2.transAxes,
             color=ORANGE_DARK, fontsize=9.5, ha="left", va="top")
    ax2.set_xlim(0.5, n + 0.5)
    ax2.set_ylim(0, 2.0)
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticks([1, 2] + list(range(4, n + 1, 2)) if n > 3 else list(k))
    ax2.set_xlabel("mode $k$", labelpad=1.5)
    ax2.set_ylabel(r"eigenvalue $\lambda_k$")

    # fiedler_value and hfer, up to this layer
    ax3 = axes_in(fig, 5.05, 0.45, 2.55, 1.3)
    lay = np.arange(L)
    fied = np.array([g["fiedler"] for g in frames])
    hfer = np.array([g["hfer"] for g in frames])
    ax3.axvline(i, color=GRID, lw=1.0, zorder=0)
    for series, col in ((fied, ORANGE), (hfer, BLUE)):
        ax3.plot(lay, series, color=BAND, lw=1.0, zorder=1)
        ax3.plot(lay[: i + 1], series[: i + 1], color=col, lw=1.6, zorder=2)
        ax3.scatter([i], [series[i]], s=22, facecolor=col, edgecolor=INK, linewidths=0.5, zorder=3)
    ax3.set_xlim(-0.8, L - 0.2)
    ax3.set_ylim(0, 1.0)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(["0", "0.5", "1"])
    ax3.set_xticks(sorted({0, (L - 1) // 3, 2 * (L - 1) // 3, L - 1}))
    ax3.set_xlabel("layer", labelpad=1.5)
    tr = blended_transform_factory(ax3.transAxes, ax3.transData)
    ys = repel([fied[i], hfer[i]], 0.13, floor=0.06, ceil=0.97)
    for (name, v, col), y in zip((("fiedler_value", fied[i], ORANGE_DARK),
                                  ("hfer", hfer[i], BLUE)), ys):
        ax3.text(1.03, y, code(name) + f" {v:.2f}", transform=tr, color=col, fontsize=9.5,
                 ha="left", va="center", clip_on=False)
    return fig


def make_hero():
    cfg, runs = run_model()
    run = runs[PROMPTS.index(HERO_PROMPT)]
    frames = hero_frames(cfg, run)
    tokens = run["token_text"]
    umax = max(float(np.max(np.abs(f["u"]))) for f in frames)
    images = [render(draw_hero(frames, i, tokens, cfg.hfer_cutoff_ratio, umax))
              for i in range(len(frames))]

    ASSETS.mkdir(parents=True, exist_ok=True)
    still = len(frames) // 2
    images[still].save(ASSETS / "hero.png", optimize=True)

    # One shared palette, so unchanged regions stay identical between frames.
    # Swatches of the figure colours make sure each of them gets its own entry.
    w, h = images[0].size
    keys = [INK, GREY, BAND, GRID, ORANGE, ORANGE_DARK, BLUE, MID]
    sample = Image.new("RGB", (w, h * 4 + 40 * len(keys)), "white")
    for j, idx in enumerate(np.linspace(0, len(images) - 1, 4).astype(int)):
        sample.paste(images[idx], (0, j * h))
    for j, colour in enumerate(keys):
        sample.paste(colour, (0, h * 4 + 40 * j, w, h * 4 + 40 * (j + 1)))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    durations = [700] * len(gif)               # about 1.4 frames per second
    durations[-1] = 2000                       # hold the last layer
    path = ASSETS / "hero.gif"
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=durations, loop=0,
                optimize=False, disposal=1)
    size = path.stat().st_size
    print(f"  hero.gif: {len(gif)} frames, {images[0].width}x{images[0].height} px, "
          f"{size / 1e6:.2f} MB; hero.png is layer {frames[still]['layer']}")
    assert size < 3e6, "hero.gif must stay under 3 MB"


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

ASSET_MAKERS = {"hero": make_hero, "heads": make_heads, "speedup": make_speedup}


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
