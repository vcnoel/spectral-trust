# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the use-case scripts: CPU settings, the IOI prompts, the
bootstrap and the figure style.

Import this module before torch: it pins the run to the CPU with two threads
and to the local Hugging Face cache. ``USE_CASE_DEVICE=cuda`` leaves the GPU
visible and ``USE_CASE_THREADS`` sets the thread count.

IOI data: the official dataset code of Wang et al. (2022), vendored unmodified
in ``third_party/ioi_official/ioi_dataset.py`` (MIT licence, from
github.com/redwoodresearch/Easy-Transformer, ``easy_transformer/ioi_dataset.py``
at commit e168301aa05154a14d52e5b6f98b6518200f5b1f, sha256
94a72de90e426382322fd690669024a0d2d4bd72a45ed9ce820a99ae9173d231).
"""
import os

if os.environ.get("USE_CASE_DEVICE", "cpu") == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = os.environ.get("USE_CASE_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUTS = HERE / "outputs"
FIGURES = ROOT / "docs" / "use-cases" / "figures"


def check_disk(min_gb: float = 1.0) -> float:
    """Free space on the drive of the repository, in GB; stop below ``min_gb``."""
    free = shutil.disk_usage(ROOT).free / 2 ** 30
    if free < min_gb:
        raise SystemExit(f"only {free:.2f} GB free on disk; stopping")
    return free


def library_info() -> dict:
    """Where spectral_trust was imported from (relative to the repository, or
    ``"installed"``), and its version."""
    import spectral_trust as st
    where = Path(st.__file__).resolve().parent
    try:
        where = where.relative_to(ROOT).as_posix()
    except ValueError:
        where = "installed"
    return {"spectral_trust": st.__version__ if hasattr(st, "__version__") else None,
            "path": where, "torch": torch.__version__, "threads": torch.get_num_threads()}


def save_json(name: str, obj) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_to_builtin)
    return path


def _to_builtin(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(type(o))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return round(time.time() - self.t0, 1)


# ── IOI ──────────────────────────────────────────────────────────────────────

def ioi_pairs(n_per_template: int = 100, seed: int = 0):
    """The IOI prompts of the official dataset code, cropped to [0, END].

    ``random``, ``numpy`` and ``torch`` are seeded once, then ``ABBA`` and
    ``BABA`` datasets of ``n_per_template`` prompts each are built
    (``prepend_bos=False``), then their ABC counterparts (the IO name, both S
    names, then S1 alone replaced by random names, the official
    ``gen_flipped_prompts`` chain). With the defaults these are the 200
    prompts of the IOI analyses in this project.

    Returns a list of dicts: ``clean`` and ``corrupt`` token ids ``[T]``
    (T = END + 1), ``end``, ``io`` and ``s`` (token ids of the answer and the
    subject), ``pos_io``, ``pos_s1``, ``pos_s2``, ``kind`` and ``text``.
    Pairs whose corrupt prompt does not align token for token are dropped.
    """
    sys.path.insert(0, str(HERE / "third_party" / "ioi_official"))
    from ioi_dataset import IOIDataset  # official code, unmodified

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    parts = [IOIDataset(prompt_type=t, N=n_per_template, prepend_bos=False)
             for t in ("ABBA", "BABA")]
    flipped = [ds.gen_flipped_prompts(("IO", "RAND")).gen_flipped_prompts(("S", "RAND"))
               .gen_flipped_prompts(("S1", "RAND")) for ds in parts]
    pairs = []
    for ds, fl in zip(parts, flipped):
        ends, fends = ds.word_idx["end"], fl.word_idx["end"]
        for i in range(ds.N):
            e = int(ends[i])
            toks, ftoks = ds.toks[i].long(), fl.toks[i].long()
            assert int(toks[e + 1]) == ds.io_tokenIDs[i], "token after END is not the IO name"
            clean, corrupt = toks[: e + 1].clone(), ftoks[: e + 1].clone()
            pos = {k: int(ds.word_idx[k][i]) for k in ("IO", "S", "S2")}
            diff = set(torch.nonzero(clean != corrupt).flatten().tolist())
            if int(fends[i]) != e or not diff <= set(pos.values()):
                continue
            pairs.append(dict(kind=ds.templates_by_prompt[i], text=ds.sentences[i],
                              clean=clean, corrupt=corrupt, end=e,
                              io=int(ds.io_tokenIDs[i]), s=int(ds.s_tokenIDs[i]),
                              pos_io=pos["IO"], pos_s1=pos["S"], pos_s2=pos["S2"]))
    return pairs


def by_length(items, key=lambda p: len(p["clean"])):
    """Group item indices by sequence length: {T: [indices]}."""
    groups = {}
    for i, p in enumerate(items):
        groups.setdefault(int(key(p)), []).append(i)
    return dict(sorted(groups.items()))


# ── statistics ───────────────────────────────────────────────────────────────

N_BOOT = 10_000


def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Resampling indices ``[n_boot, n]`` (the same for every quantity of a script)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def mean_ci(x, idx=None, level: float = 0.95):
    """Mean and percentile bootstrap interval over items (axis 0)."""
    x = np.asarray(x, dtype=np.float64)
    idx = boot_idx(len(x)) if idx is None else idx
    bs = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def ratio_ci(num, den, idx=None, level: float = 0.95):
    """Ratio of means ``mean(num) / mean(den)`` with a paired percentile bootstrap."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    idx = boot_idx(len(num)) if idx is None else idx
    bs = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    a = (1 - level) / 2
    return (float(num.mean() / den.mean()), float(np.quantile(bs, a)),
            float(np.quantile(bs, 1 - a)))


def fmt(ci, digits: int = 2) -> str:
    m, lo, hi = ci
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ── figure style (docs/make_assets.py, Okabe-Ito colours) ─────────────────────

INK, GREY, BAND, GRID = "#1A1A1A", "#9E9E9E", "#D9D9D9", "#EBEBEB"
ORANGE, ORANGE_DARK, BLUE, GREEN, MID = "#E69F00", "#B35900", "#0072B2", "#009E73", "#7F7F7F"
SKY, VERMILION, PURPLE = "#56B4E9", "#D55E00", "#CC79A7"


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3,
        "ytick.major.size": 3, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "axes.labelpad": 3.0, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 1.2, "lines.solid_capstyle": "round", "text.usetex": False,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "legend.frameon": False, "axes.titlesize": 10,
    })
    return plt


def save_figure(fig, name: str) -> Path:
    """Save at 200 dpi in docs/use-cases/figures/."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.convert("RGB").save(path, optimize=True)
    except ImportError:  # pragma: no cover
        pass
    return path


def wait_cool(limit: int = 80, poll: float = 10.0) -> None:
    """On the GPU (``USE_CASE_DEVICE=cuda``), wait while its temperature is
    above ``limit`` degrees C (read with nvidia-smi)."""
    if os.environ.get("USE_CASE_DEVICE", "cpu") == "cpu":
        return
    import subprocess
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                                  "--format=csv,noheader"], capture_output=True, text=True)
            t = int(out.stdout.split()[0])
        except (OSError, ValueError, IndexError):
            return
        if t <= limit:
            return
        print(f"GPU at {t} C; waiting", flush=True)
        time.sleep(poll)
