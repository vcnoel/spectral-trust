# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Fast diagnostics against the dense path, on synthetic attention graphs.

Each section runs separately so that one run stays within a time budget on a
few CPU threads. Every run merges its rows into ``benchmarks/results.json``
and rewrites ``benchmarks/results.md`` from the whole file.

    python benchmarks/fast_vs_dense.py --section solvers
    python benchmarks/fast_vs_dense.py --section hfer --sizes 128,512,2048
    python benchmarks/fast_vs_dense.py --section hfer --sizes 4096 --dims 1024
    python benchmarks/fast_vs_dense.py --section jl
    python benchmarks/fast_vs_dense.py --section chebyshev --sizes 256,1024
    python benchmarks/fast_vs_dense.py --section chebyshev --sizes 4096
    python benchmarks/fast_vs_dense.py --section heads
    python benchmarks/fast_vs_dense.py --section table

A case whose predicted cost (calibrated at start-up) would overrun
``--budget`` seconds is skipped and listed in the results.
"""
import argparse
import os
import time

START = time.perf_counter()      # the budget includes the imports below

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--section", required=True,
                    choices=["solvers", "hfer", "jl", "chebyshev", "heads", "table"])
parser.add_argument("--sizes", default=None, help="comma-separated S values")
parser.add_argument("--dims", default=None, help="comma-separated d values")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--budget", type=float, default=100.0,
                    help="seconds; cases predicted to overrun it are skipped")
parser.add_argument("--trials", type=int, default=200, help="JL seeds per config")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
args = parser.parse_args()

# Thread pools are sized when numpy and torch load, so set the limits first.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = str(args.threads)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # CPU only

import json                      # noqa: E402
import logging                   # noqa: E402
import platform                  # noqa: E402

import numpy as np               # noqa: E402
import scipy                     # noqa: E402
import scipy.linalg as sla       # noqa: E402
import torch                     # noqa: E402

from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer   # noqa: E402
from spectral_trust import per_head_metrics, per_head_hybrid_metrics        # noqa: E402
from spectral_trust.fast import (                                           # noqa: E402
    top_band_hfer, top_band_eigenvectors, row_dirichlet_energy,
    band_energies, band_entropy, gaussian_projection, num_high_modes,
)

torch.set_num_threads(args.threads)
logging.disable(logging.WARNING)     # the truncation warning is expected here


# ── Synthetic inputs ─────────────────────────────────────────────────────────

def attention(S, seed=0, heads=1):
    """Causal softmax attention with a sink token and a locality bias."""
    g = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn(heads, S, S, generator=g)
    pos = torch.arange(S, dtype=torch.float32)
    logits -= 0.05 * (pos[:, None] - pos[None, :]).abs()
    logits[..., 0] += 3.0
    logits = logits.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool)), float("-inf"))
    return torch.softmax(logits, dim=-1)


def graph(S, seed=0):
    """(adjacency W, sym-normalised Laplacian L), float32 tensors."""
    gc = GraphConstructor(GSPConfig(device="cpu"))
    W = gc.symmetrize_attention(attention(S, seed).unsqueeze(0))[0]
    return W[0], gc.construct_laplacian(W)[0]


def signals(S, d, kind, seed=0):
    """random: i.i.d. Gaussian. structured: residual-stream-like, a large
    shared mean, a few position-smooth components, power-law noise, and one
    massive-activation token."""
    rng = np.random.default_rng(seed + 7)
    if kind == "random":
        return rng.standard_normal((S, d)).astype(np.float32)
    X = rng.standard_normal((S, d)) * np.arange(1, d + 1) ** -0.5
    pos = np.linspace(0.0, 1.0, S)[:, None]
    for r in range(1, 5):
        X += np.cos(np.pi * r * pos) * rng.standard_normal(d)[None, :] / r
    X += 3.0 * rng.standard_normal(d)[None, :]
    X[0, :4] += 50.0
    return X.astype(np.float32)


# ── Timing and budget ────────────────────────────────────────────────────────

def timed(fn, reps=1):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def calibrate():
    A = np.asarray(graph(1024)[1])
    B = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32)
    t_eig, _ = timed(lambda: sla.eigh(A), 2)
    t_mm, _ = timed(lambda: A @ B, 3)
    return t_eig, t_mm


T_EIG, T_MM = calibrate()
#: LAPACK eigh grows more slowly than S^3 over this range (measured 2.45 to
#: 2.6 between S = 512 and 4096 on two threads); 2.6 keeps predictions high.
EIG_EXPONENT = 2.6
ATTEMPTED = []      # (section, case) pairs this run tried, run or skipped


def predicted(S, d, eig_units, mm_units):
    """Seconds, from scipy eigh at S = 1024 and a 1024^3 product at start-up.
    eig_units count dense eigensolves at size S; mm_units count S x S by
    S x d products."""
    return 1.3 * (T_EIG * (S / 1024) ** EIG_EXPONENT * eig_units
                  + T_MM * (S / 1024) ** 2 * (d / 1024) * mm_units)


def over_budget(section, case, cost, skipped):
    ATTEMPTED.append((section, case))
    if time.perf_counter() - START + cost > args.budget:
        skipped.append({"section": section, "case": case, "predicted_s": round(cost, 1)})
        print("skip", section, case, f"(predicted {cost:.0f} s)", flush=True)
        return True
    return False


def rel(a, b):
    return float(abs(float(a) - float(b)) / max(abs(float(b)), 1e-30))


# ── Sections ─────────────────────────────────────────────────────────────────

def section_solvers(sizes):
    """Time to the top num_high_modes(S, 0.1) eigenvectors, per solver."""
    rows, skipped = {}, []
    for S in sizes:
        key = f"S={S}"
        if over_budget("solvers", key, predicted(S, 0, 4 + (4 if S <= 2048 else 0), 0),
                       skipped):
            continue
        reps = 3 if S <= 512 else 1
        _, L = graph(S)
        Lnp = L.numpy()
        k = num_high_modes(S, 0.1)
        row = {"S": S, "k": k}
        row["subset_evr_s"] = timed(lambda: top_band_eigenvectors(Lnp, k, "subset"), reps)[0]
        row["subset_evx_s"] = timed(lambda: sla.eigh(Lnp, subset_by_index=[S - k, S - 1],
                                                     driver="evx"), reps)[0]
        row["full_scipy_s"] = timed(lambda: top_band_eigenvectors(Lnp, k, "full"), reps)[0]
        row["full_torch_s"] = timed(lambda: top_band_eigenvectors(Lnp, k, "torch"), reps)[0]
        if S <= 2048:
            row["eigsh_s"] = timed(lambda: top_band_eigenvectors(Lnp, k, "eigsh"))[0]
        if S <= 512:
            X0 = torch.randn(S, k, generator=torch.Generator().manual_seed(0))
            row["lobpcg_s"] = timed(lambda: torch.lobpcg(L, k=k, X=X0, largest=True,
                                                         tol=1e-6, niter=500))[0]
        rows[key] = row
        print(key, {n: round(v, 4) for n, v in row.items()}, flush=True)
    return {"solvers": rows}, skipped


def section_hfer(sizes, dims):
    rows, skipped = {}, []
    for S in sizes:
        for d in dims:
            key = f"S={S},d={d}"
            # eig units: dense HFER, subset, 3 analyze_layer calls, sparse default
            # mm units: projections plus the d x d products of the energy path
            cost = predicted(S, d, 4.4 + (5 if S <= 2048 else 0),
                             12 + 6 * d / S)
            if over_budget("hfer", key, cost, skipped):
                continue
            reps = 3 if S <= 512 else 1
            W, L = graph(S)
            Lnp = L.numpy()
            X = signals(S, d, "structured")
            Xt = torch.from_numpy(X)
            dense = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense"))

            def dense_hfer():
                vals, vecs = dense.compute_eigendecomposition(Lnp)
                return dense.compute_hfer(X, vecs, vals, 0.1)
            t_dense, h_dense = timed(dense_hfer, reps)
            t_fast, h_fast = timed(lambda: top_band_hfer(Lnp, X), reps)
            row = {"S": S, "d": d, "n_high": num_high_modes(S, 0.1),
                   "hfer_dense_s": t_dense, "hfer_fast_s": t_fast,
                   "hfer_speedup": t_dense / t_fast,
                   "hfer_dense": h_dense, "hfer_abs_err": abs(h_fast - h_dense)}
            if S <= 2048:
                sparse = SpectralAnalyzer(GSPConfig(device="cpu"))   # the default

                def sparse_hfer():
                    vals, vecs = sparse.compute_eigendecomposition(Lnp)
                    return sparse.compute_hfer(X, vecs, vals, 0.1)
                t_sp, h_sp = timed(sparse_hfer)
                row.update(sparse_default_s=t_sp, sparse_default_hfer=h_sp,
                           sparse_default_abs_err=abs(h_sp - h_dense))
            # energies: the library's energy, a trace-only evaluation, 20 rows
            t_energy, e_lib = timed(lambda: dense.compute_dirichlet_energy(X, Lnp), reps)
            t_trace, e_tr = timed(lambda: float(np.sum(X * (Lnp @ X), dtype=np.float64)), reps)
            rows_sel = list(range(S - 20, S))
            t_rows, _ = timed(lambda: row_dirichlet_energy(W, Xt, rows=rows_sel), reps)
            row.update(energy_lib_s=t_energy, energy_trace_s=t_trace,
                       energy_trace_rel_diff=rel(e_tr, e_lib), row_energy_20_s=t_rows)
            # Johnson-Lindenstrauss projection inside analyze_layer
            t_full, d_full = timed(lambda: dense.analyze_layer(Xt, L, 0))
            row["analyze_dense_s"] = t_full
            for k in (64, 256):
                an = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense",
                                                project_dim=k, project_seed=0))
                t_k, d_k = timed(lambda: an.analyze_layer(Xt, L, 0))
                row[f"analyze_jl{k}_s"] = t_k
                row[f"analyze_jl{k}_speedup"] = t_full / t_k
                for m in ("energy", "smoothness_index", "hfer", "spectral_entropy"):
                    row[f"jl{k}_rel_err_{m}"] = rel(getattr(d_k, m), getattr(d_full, m))
            rows[key] = row
            print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                  flush=True)
    return {"hfer": rows}, skipped


def section_jl(sizes, dims, trials):
    """Relative error of the four hybrid metrics under projection, from the
    eigenbasis identity: with U^T X precomputed in float64, each trial needs
    only (U^T X) R. One 256-column draw per seed serves every k (the draws
    are nested, see gaussian_projection)."""
    ks = (32, 64, 128, 256)
    kinds = ("random", "structured")
    exact, errs, skipped = {}, {}, []
    bases = {}
    for S in sizes:
        _, L = graph(S, seed=1)
        vals, vecs = np.linalg.eigh(L.double().numpy())
        bases[S] = (np.maximum(vals, 0.0), vecs)
    for d in dims:
        # per seed: one d x 256 draw, then S x d x 256 products per config
        per_seed = 0.02 * d / 4096 + sum(T_MM * (S / 1024) * (d / 1024) * 0.25 * 2
                                         for S in sizes) * len(kinds)
        if over_budget("jl", f"d={d}", 1.5 * trials * per_seed, skipped):
            continue
        UX = {}
        for S in sizes:
            lam, U = bases[S]
            cut = S - num_high_modes(S, 0.1)
            for kind in kinds:
                X = signals(S, d, kind, seed=S).astype(np.float64)
                ux = U.T @ X
                m = (ux ** 2).sum(1)
                key = (S, d, kind)
                UX[key] = (lam, ux, cut)
                exact[key] = _metrics(lam, m, cut)
                errs[key] = {k: {name: [] for name in exact[key]} for k in ks if k < d}
        for seed in range(trials):
            R = gaussian_projection(d, 256, seed=seed, dtype=np.float64)
            for key, (lam, ux, cut) in UX.items():
                Z = ux @ R
                for k in errs[key]:
                    mk = ((Z[:, :k] * np.sqrt(256.0 / k)) ** 2).sum(1)
                    got = _metrics(lam, mk, cut)
                    for name, v in got.items():
                        errs[key][k][name].append(rel(v, exact[key][name]))
        print("jl d", d, "done", round(time.perf_counter() - START, 1), "s", flush=True)
    configs, pooled = {}, {}
    for (S, d, kind), by_k in errs.items():
        for k, by_m in by_k.items():
            for name, e in by_m.items():
                e = np.asarray(e)
                configs[f"S={S},d={d},{kind},k={k},{name}"] = {
                    "median": float(np.median(e)), "p99": float(np.quantile(e, 0.99))}
                pooled.setdefault(f"{kind},k={k},{name}", []).append(e)
    pooled = {key: {"median": float(np.median(np.concatenate(v))),
                    "p99": float(np.quantile(np.concatenate(v), 0.99)),
                    "configs": len(v)} for key, v in pooled.items()}
    return {"jl_error": {"sizes": sizes, "dims": dims, "trials": trials,
                         "configs": configs, "pooled": pooled}}, skipped


def _metrics(lam, masses, cut):
    total = masses.sum()
    p = np.maximum(masses / total, 1e-12)
    energy = float(lam @ masses)
    return {"energy": energy, "smoothness_index": energy / total,
            "hfer": float(masses[cut:].sum() / total),
            "spectral_entropy": float(-(p * np.log(p)).sum())}


def section_chebyshev(sizes, d=1024, bands=8):
    rows, skipped = {}, []
    orders = (32, 64, 128)
    for S in sizes:
        key = f"S={S},d={d}"
        # eig units: scipy and torch dense paths; mm units: the Chebyshev
        # products (order / 2 each) plus 32 products at width 128
        cost = predicted(S, d, 2.5, 2 + sum(orders) / 2 + 32 * 128 / d)
        if over_budget("chebyshev", key, cost, skipped):
            continue
        reps = 3 if S <= 512 else 1
        _, L = graph(S, seed=2)
        Lnp = L.numpy()
        X = signals(S, d, "structured", seed=2)
        dense = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense"))
        edges = np.linspace(0.0, 2.0, bands + 1)

        def dense_lib():
            vals, vecs = dense.compute_eigendecomposition(Lnp)
            m = ((vecs.T @ X) ** 2).sum(1)
            b = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, bands - 1)
            return np.bincount(b, weights=m, minlength=bands)
        t_lib, _ = timed(dense_lib, reps)
        t_torch, (exact, _) = timed(lambda: band_energies(L, X, bands, lambda_max=2.0,
                                                         method="dense"), reps)
        q, H = exact / exact.sum(), band_entropy(exact)
        row = {"S": S, "d": d, "bands": bands, "dense_lib_s": t_lib, "dense_torch_s": t_torch}
        variants = [(f"cheb{n}", dict(order=n)) for n in orders]
        variants.append(("cheb64_jl128", dict(order=64, project_dim=128)))
        for name, kw in variants:
            t, (approx, _) = timed(lambda: band_energies(L, X, bands, lambda_max=2.0, **kw), reps)
            p = approx / approx.sum()
            row.update({f"{name}_s": t, f"{name}_speedup_vs_lib": t_lib / t,
                        f"{name}_speedup_vs_torch": t_torch / t,
                        f"{name}_max_abs_dp": float(np.abs(p - q).max()),
                        f"{name}_entropy_rel_err": rel(band_entropy(approx), H)})
        rows[key] = row
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
    return {"chebyshev": rows}, skipped


def section_heads(S=512, H=32, d=1024, k=20):
    attn = attention(S, seed=3, heads=H)
    X = torch.from_numpy(signals(S, d, "structured", seed=3))
    top = sorted(np.random.default_rng(0).permutation(H)[:k].tolist())   # stand-in scores
    t_all, full = timed(lambda: per_head_metrics(attn), 2)
    t_sub, sub = timed(lambda: per_head_metrics(attn, heads=top), 2)
    t_hall, hfull = timed(lambda: per_head_hybrid_metrics(attn, X))
    t_hsub, hsub = timed(lambda: per_head_hybrid_metrics(attn, X, heads=top))
    row = {"S": S, "H": H, "d": d, "k": k,
           "attention_all_s": t_all, "attention_subset_s": t_sub,
           "attention_speedup": t_all / t_sub,
           "attention_max_abs_diff": float(np.abs(sub.values - full.values[top]).max()),
           "hybrid_all_s": t_hall, "hybrid_subset_s": t_hsub,
           "hybrid_speedup": t_hall / t_hsub,
           "hybrid_max_rel_diff": float(np.max(np.abs(hsub.values - hfull.values[top])
                                               / np.maximum(np.abs(hfull.values[top]), 1e-12)))}
    print(row, flush=True)
    return {"heads": {f"S={S},H={H},d={d},k={k}": row}}, []


# ── Results ──────────────────────────────────────────────────────────────────

def merge(path, update, skipped):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    for section, rows in update.items():
        if section == "jl_error":
            data[section] = rows
        else:
            data.setdefault(section, {}).update(rows)
    attempted = set(ATTEMPTED)
    data["skipped"] = [s for s in data.get("skipped", [])
                       if (s["section"], s["case"]) not in attempted] + skipped
    data["meta"] = {
        "threads": args.threads, "device": "cpu",
        "processor": platform.processor(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "timing": "best of 3 runs for S <= 512, single run above; seconds. "
                  "Measured on a laptop under concurrent load: repeat runs "
                  "differed by up to about 30% at S >= 1024",
        "graph": "causal softmax attention with sink and locality bias, "
                 "symmetrised, sym-normalised Laplacian",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return data


def fmt_t(x):
    return f"{x * 1e3:.0f} ms" if x < 1 else f"{x:.2f} s"


def write_markdown(data, path):
    out = ["# Fast diagnostics against the dense path", "",
           f"CPU only, {data['meta']['threads']} threads; "
           f"{data['meta']['graph']}. Timing: {data['meta']['timing']}. "
           "Generated by `fast_vs_dense.py`.", ""]
    solvers = sorted(data.get("solvers", {}).values(), key=lambda r: r["S"])
    if solvers:
        out += ["## Top-band eigensolvers", "",
                "Time to the top k = num_high_modes(S, 0.1) eigenvectors, float32. "
                "`method=\"auto\"` takes the full torch solve up to S = 1024 and the "
                "subset evr solve above.", "",
                "| S | k | subset evr | subset evx | full scipy | full torch "
                "| eigsh | lobpcg |", "|---|---|---|---|---|---|---|---|"]
        for r in solvers:
            cells = [fmt_t(r[c]) if c in r else "not run" for c in
                     ("subset_evr_s", "subset_evx_s", "full_scipy_s", "full_torch_s",
                      "eigsh_s", "lobpcg_s")]
            out.append(f"| {r['S']} | {r['k']} | " + " | ".join(cells) + " |")
        out.append("")
    hfer = sorted(data.get("hfer", {}).values(), key=lambda r: (r["S"], r["d"]))
    if hfer:
        out += ["## Exact HFER and projected hybrid metrics", "",
                "| S | d | HFER dense | top-band | speedup | abs err | default sparse err "
                "| analyze dense | JL k=256 | speedup | max rel err k=256 |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in hfer:
            sp = (f"{r['sparse_default_abs_err']:.1e} ({fmt_t(r['sparse_default_s'])})"
                  if "sparse_default_abs_err" in r else "not run")
            jl_err = max(r[f"jl256_rel_err_{m}"] for m in
                         ("energy", "smoothness_index", "hfer", "spectral_entropy"))
            out.append(
                f"| {r['S']} | {r['d']} | {fmt_t(r['hfer_dense_s'])} | {fmt_t(r['hfer_fast_s'])} "
                f"| {r['hfer_speedup']:.1f}x | {r['hfer_abs_err']:.1e} | {sp} "
                f"| {fmt_t(r['analyze_dense_s'])} | {fmt_t(r['analyze_jl256_s'])} "
                f"| {r['analyze_jl256_speedup']:.1f}x | {jl_err:.1e} |")
        out += ["", "HFER error is against the dense path (both float32). \"analyze\" is "
                "`SpectralAnalyzer.analyze_layer` with the dense solver, without and with "
                "`project_dim=256`; its error column is one seed (see the JL section for "
                "distributions). At small S most of that speedup comes from the d x d "
                "products in the library's energy and smoothness evaluation (next table), "
                "which a trace-only evaluation would remove exactly.",
                "", "Energy evaluation (same rows): library `compute_dirichlet_energy` against "
                "a trace-only `sum(X * (L @ X))`, and the row-anchored energy of 20 rows.", "",
                "| S | d | library energy | trace-only | 20-row energy |", "|---|---|---|---|---|"]
        for r in hfer:
            out.append(f"| {r['S']} | {r['d']} | {fmt_t(r['energy_lib_s'])} "
                       f"| {fmt_t(r['energy_trace_s'])} | {fmt_t(r['row_energy_20_s'])} |")
        out.append("")
    jl = data.get("jl_error")
    if jl:
        out += ["## Johnson-Lindenstrauss relative error",
                "", f"Median and 99th percentile of |relative error| over {jl['trials']} seeds, "
                f"pooled over S in {jl['sizes']} and d in {jl['dims']}.", "",
                "| signals | k | energy | smoothness | HFER | entropy |", "|---|---|---|---|---|---|"]
        for kind in ("random", "structured"):
            for k in (32, 64, 128, 256):
                cells = []
                for m in ("energy", "smoothness_index", "hfer", "spectral_entropy"):
                    p = jl["pooled"].get(f"{kind},k={k},{m}")
                    cells.append(f"{p['median']:.3f} / {p['p99']:.3f}" if p else "-")
                out.append(f"| {kind} | {k} | " + " | ".join(cells) + " |")
        out.append("")
    cheb = sorted(data.get("chebyshev", {}).values(), key=lambda r: r["S"])
    if cheb:
        out += ["## Chebyshev band energies (8 bands on [0, 2])", "",
                "Speedup against the library's dense solver (scipy `eigh`), "
                "then against `torch.linalg.eigh`; error as max |band fraction error| "
                "and band-entropy relative error.", "",
                "| S | d | dense scipy | dense torch | variant | time | speedup | max dp | dH/H |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in cheb:
            for v in ("cheb32", "cheb64", "cheb128", "cheb64_jl128"):
                out.append(
                    f"| {r['S']} | {r['d']} | {fmt_t(r['dense_lib_s'])} | {fmt_t(r['dense_torch_s'])} "
                    f"| {v} | {fmt_t(r[v + '_s'])} | {r[v + '_speedup_vs_lib']:.2f}x / "
                    f"{r[v + '_speedup_vs_torch']:.2f}x | {r[v + '_max_abs_dp']:.1e} "
                    f"| {r[v + '_entropy_rel_err']:.1e} |")
        out.append("")
    for r in data.get("heads", {}).values():
        out += ["## Head subsets", "",
                f"{r['k']} of {r['H']} heads at S = {r['S']}: attention metrics "
                f"{fmt_t(r['attention_all_s'])} -> {fmt_t(r['attention_subset_s'])} "
                f"({r['attention_speedup']:.1f}x, max diff {r['attention_max_abs_diff']:.1e}); "
                f"hybrid metrics (d = {r['d']}) {fmt_t(r['hybrid_all_s'])} -> "
                f"{fmt_t(r['hybrid_subset_s'])} ({r['hybrid_speedup']:.1f}x, max rel diff "
                f"{r['hybrid_max_rel_diff']:.1e}).", ""]
    if data.get("skipped"):
        out += ["## Skipped (predicted to exceed the time budget)", ""]
        out += [f"- {s['section']} {s['case']}: predicted {s['predicted_s']} s"
                for s in data["skipped"]]
        out.append("")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
    dims = [int(s) for s in args.dims.split(",")] if args.dims else None
    print(f"calibration: eigh(1024) {T_EIG * 1e3:.0f} ms, 1024^3 product "
          f"{T_MM * 1e3:.1f} ms; {time.perf_counter() - START:.1f} s since start",
          flush=True)
    if args.section == "solvers":
        update, skipped = section_solvers(sizes or [128, 512, 1024, 2048, 4096])
    elif args.section == "hfer":
        update, skipped = section_hfer(sizes or [128, 512, 2048], dims or [1024, 4096])
    elif args.section == "jl":
        update, skipped = section_jl(sizes or [64, 256, 1024], dims or [256, 1024, 4096],
                                     args.trials)
    elif args.section == "chebyshev":
        update, skipped = section_chebyshev(sizes or [256, 1024, 4096])
    elif args.section == "heads":
        update, skipped = section_heads()
    else:
        update, skipped = {}, []
    data = merge(args.out, update, skipped)
    write_markdown(data, os.path.splitext(args.out)[0] + ".md")
    print(f"done in {time.perf_counter() - START:.1f} s", flush=True)


if __name__ == "__main__":
    main()
