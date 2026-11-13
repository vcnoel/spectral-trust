# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end timing under each version's defaults: 0.3.0 against the current code.

Two workloads, each with ``GSPConfig(device="cpu")``, i.e. the defaults of the
version that runs it (0.3.0: eigen_solver="sparse"; 0.4.0: "dense"):

- ``analyze_layer`` on one layer's head-averaged sym Laplacian;
- per-head hybrid metrics for ``--heads`` heads: ``per_head_hybrid_metrics``
  in the current code; in 0.3.0, which has no per-head hybrid path, a loop of
  ``analyze_layer`` over the heads' Laplacians.

The 0.3.0 tree is extracted with ``git archive`` (``--ref``, default 7197315)
and each version runs in its own worker process. Rows are merged into
``results.json``; rerun ``fast_vs_dense.py --section table`` to refresh
``results.md``.

    python benchmarks/end_to_end.py --sizes 128,512 --dims 1024,4096
    python benchmarks/end_to_end.py --sizes 2048 --dims 1024 --version old
    python benchmarks/end_to_end.py --sizes 2048 --dims 1024 --version new

``--version old|new`` runs one tree, so that each run stays short; rows are
paired in ``results.json`` once both versions have run a case.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--sizes", default="128,512")
parser.add_argument("--dims", default="1024,4096")
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--ref", default="7197315")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--version", choices=["old", "new", "both"], default="both")
parser.add_argument("--out", default=os.path.join(HERE, "results.json"))
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--src", help=argparse.SUPPRESS)
parser.add_argument("--result", help=argparse.SUPPRESS)
args = parser.parse_args()

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = str(args.threads)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def worker():
    sys.path.insert(0, args.src)
    import logging
    import numpy as np
    import torch
    import spectral_trust as st
    from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
    logging.disable(logging.WARNING)
    torch.set_num_threads(args.threads)
    loaded = os.path.normcase(os.path.abspath(st.__file__))
    assert loaded.startswith(os.path.normcase(os.path.abspath(args.src))), loaded
    per_head_hybrid = getattr(st, "per_head_hybrid_metrics", None)

    def timed(fn, reps):
        best, out = float("inf"), None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        return best, out

    rows = {}
    for S in [int(s) for s in args.sizes.split(",")]:
        g = torch.Generator().manual_seed(0)
        logits = 2.0 * torch.randn(args.heads, S, S, generator=g)
        pos = torch.arange(S, dtype=torch.float32)
        logits -= 0.05 * (pos[:, None] - pos[None, :]).abs()
        logits[..., 0] += 3.0
        logits = logits.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool)),
                                    float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        cfg = GSPConfig(device="cpu")                    # this version's defaults
        gc = GraphConstructor(cfg)
        sym = gc.symmetrize_attention(attn.unsqueeze(0))[0]               # [H, S, S]
        L_layer = gc.construct_laplacian(sym.mean(0, keepdim=True))[0]
        L_heads = gc.construct_laplacian(sym)
        for d in [int(x) for x in args.dims.split(",")]:
            rng = np.random.default_rng(S + d)
            X = rng.standard_normal((S, d)) * np.arange(1, d + 1) ** -0.5
            X += 3.0 * rng.standard_normal(d)[None, :]
            X[0, :4] += 50.0
            X = torch.from_numpy(X.astype(np.float32))
            reps = 3 if S <= 512 else 1
            an = SpectralAnalyzer(cfg)
            t_layer, diag = timed(lambda: an.analyze_layer(X, L_layer, 0), reps)
            if per_head_hybrid is not None:
                t_heads, _ = timed(lambda: per_head_hybrid(attn, X, config=cfg), reps)
                head_path = "per_head_hybrid_metrics"
            else:
                t_heads, _ = timed(lambda: [an.analyze_layer(X, L_heads[h], 0)
                                            for h in range(args.heads)], reps)
                head_path = "analyze_layer per head"
            rows[f"S={S},d={d}"] = {
                "S": S, "d": d, "heads": args.heads, "eigen_solver": cfg.eigen_solver,
                "analyze_layer_s": t_layer, "per_head_hybrid_s": t_heads,
                "per_head_path": head_path, "hfer": float(diag.hfer),
                "modes": int(len(diag.eigenvalues))}
            print(st.__version__, rows[f"S={S},d={d}"], flush=True)
    with open(args.result, "w") as f:
        json.dump({"version": st.__version__, "rows": rows}, f)


def extract(ref, dest):
    blob = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", ref,
                           "src/spectral_trust"], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return os.path.join(dest, "src")


def run(src, result):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--src", src,
           "--result", result, "--sizes", args.sizes, "--dims", args.dims,
           "--heads", str(args.heads), "--threads", str(args.threads)]
    subprocess.run(cmd, check=True, cwd=tempfile.gettempdir())
    with open(result) as f:
        return json.load(f)


def main():
    data = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            data = json.load(f)
    raw = data.setdefault("end_to_end_raw", {"old": {}, "new": {}})
    with tempfile.TemporaryDirectory() as tmp:
        if args.version in ("old", "both"):
            old = run(extract(args.ref, os.path.join(tmp, "v030")),
                      os.path.join(tmp, "old.json"))
            for key, row in old["rows"].items():
                raw["old"][key] = dict(row, version=old["version"])
        if args.version in ("new", "both"):
            new = run(os.path.join(REPO, "src"), os.path.join(tmp, "new.json"))
            for key, row in new["rows"].items():
                raw["new"][key] = dict(row, version=new["version"])
    section = data.setdefault("end_to_end", {})
    for key in sorted(set(raw["old"]) & set(raw["new"])):
        o, n = raw["old"][key], raw["new"][key]
        old, new = {"version": o["version"]}, {"version": n["version"]}
        section[key] = {
            "S": o["S"], "d": o["d"], "heads": o["heads"],
            "old_version": old["version"], "new_version": new["version"],
            "old_eigen_solver": o["eigen_solver"], "new_eigen_solver": n["eigen_solver"],
            "old_analyze_layer_s": o["analyze_layer_s"],
            "new_analyze_layer_s": n["analyze_layer_s"],
            "analyze_layer_speedup": o["analyze_layer_s"] / n["analyze_layer_s"],
            "old_per_head_s": o["per_head_hybrid_s"], "new_per_head_s": n["per_head_hybrid_s"],
            "per_head_speedup": o["per_head_hybrid_s"] / n["per_head_hybrid_s"],
            "old_per_head_path": o["per_head_path"], "new_per_head_path": n["per_head_path"],
            "old_hfer": o["hfer"], "new_hfer": n["hfer"],
            "old_modes": o["modes"], "new_modes": n["modes"]}
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    if args.worker:
        worker()
    else:
        main()
