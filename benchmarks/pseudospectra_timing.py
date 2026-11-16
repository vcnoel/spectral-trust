# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""Timings of spectral_trust.pseudospectra on a causal, sink-heavy
row-stochastic matrix (the shape of an attention map).

    PYTHONPATH=src python benchmarks/pseudospectra_timing.py --n 128 512

Writes benchmarks/pseudospectra_timing.json. CPU, float64/complex128.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from spectral_trust import pseudospectra as ps


def causal(n, seed=0, sink=2.0):
    rng = np.random.default_rng(seed)
    a = np.tril(rng.random((n, n)))
    a[:, 0] += sink * a.sum(1)
    return a / a.sum(1, keepdims=True)


def clock(fn, *args, **kw):
    t = time.perf_counter()
    out = fn(*args, **kw)
    return time.perf_counter() - t, out


def run(n: int, kreiss: bool = True) -> dict:
    A = causal(n)
    z = 1.05 * np.exp(1j * np.linspace(0, 2 * np.pi, 64, endpoint=False))
    row = {"n": n}
    t_fast, s_fast = clock(ps.sigma_min, A, z)
    k = 16 if n >= 512 else 64
    t_svd, s_svd = clock(ps.sigma_min, A, z[:k], method="svd")
    row["sigma_min_ms_per_point_triangular"] = 1e3 * t_fast / len(z)
    row["sigma_min_ms_per_point_dense_svd"] = 1e3 * t_svd / k
    row["sigma_min_max_rel_diff"] = float(np.max(np.abs(s_fast[:k] - s_svd) / s_svd))
    grid = (np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    row["pseudospectrum_50x50_s"], _ = clock(ps.pseudospectrum, A, grid)
    row["pseudospectral_radius_s"], r = clock(ps.pseudospectral_radius, A, 1e-2)
    row["pseudospectral_radius"] = r
    row["pseudospectral_abscissa_s"], a = clock(ps.pseudospectral_abscissa, A, 1e-2)
    row["pseudospectral_abscissa"] = a
    if kreiss:
        row["kreiss_resolvent_s"], kr = clock(ps.kreiss_constant, A, how="resolvent")
        row["kreiss_resolvent"] = kr
        row["kreiss_radii_s"], kd = clock(ps.kreiss_constant, A)
        row["kreiss_radii"] = kd
    row["sqrt_n"] = float(np.sqrt(n))
    row["henrici_s"], _ = clock(ps.departure_from_normality, A)
    row["numerical_range_s"], _ = clock(ps.numerical_range, A)
    print(json.dumps({k: (float("%.4g" % v) if isinstance(v, float) else v) for k, v in row.items()}))
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[128, 512])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-kreiss", action="store_true", help="skip the Kreiss constant")
    p.add_argument("--out", default=str(Path(__file__).with_name("pseudospectra_timing.json")))
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rows = [run(n, kreiss=not args.no_kreiss) for n in args.n]
    meta = {"platform": platform.platform(), "processor": platform.processor(),
            "threads": args.threads, "torch": torch.__version__, "numpy": np.__version__}
    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
