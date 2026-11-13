# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Check that the current code reproduces the 0.3.0 values.

Configurations compared (those used in published work):

- the layer pipeline as GSPDiagnosticsFramework runs it (per-head
  symmetrization, uniform head aggregation, Laplacian, analyze_layer) with
  normalization="sym" and eigen_solver="dense", on the whole graph and on a
  token subgraph;
- the per-head attention-only metrics (per_head_metrics with the default
  configuration, with and without token_span, and the all-layers feature
  helpers).

The 0.3.0 source is extracted from git (``--ref``, default 7197315) with
``git archive``; each tree runs in its own worker process on the same seeded
inputs. Writes ``reproduce_v030.json`` and ``reproduce_v030.md`` next to this
script.

    python benchmarks/reproduce_v030.py
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOLERANCE = 1e-6

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--ref", default="7197315", help="git ref of the 0.3.0 tree")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--src", help=argparse.SUPPRESS)
parser.add_argument("--out", help=argparse.SUPPRESS)
args = parser.parse_args()

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = str(args.threads)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ── Worker: compute every value under one source tree ────────────────────────

def worker(src, out):
    sys.path.insert(0, src)
    import numpy as np
    import torch
    import spectral_trust as st
    from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
    from spectral_trust import (per_head_metrics, per_head_metrics_all_layers,
                                stack_feature_vector, layer_delta_features)
    import logging
    logging.disable(logging.WARNING)
    torch.set_num_threads(args.threads)
    loaded = os.path.normcase(os.path.abspath(st.__file__))
    assert loaded.startswith(os.path.normcase(os.path.abspath(src))), loaded

    def attention(kind, heads, seq, seed):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(heads, seq, seq, generator=g)
        if kind == "sink":
            logits[..., 0] += 5.0
        elif kind == "causal":
            pos = torch.arange(seq, dtype=torch.float32)
            logits = logits - 0.1 * (pos[:, None] - pos[None, :]).abs()
            mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
            logits = logits.masked_fill(~mask, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def hidden(seq, dim, seed):
        g = torch.Generator().manual_seed(seed + 1000)
        X = torch.randn(seq, dim, generator=g) * torch.arange(1, dim + 1) ** -0.5
        X = X + 2.0 * torch.randn(1, dim, generator=g)
        X[0, :4] += 20.0
        return X

    layer_cfg = GSPConfig(device="cpu", normalization="sym", eigen_solver="dense")
    head_cfg = GSPConfig(device="cpu")      # default per-head configuration
    gc = GraphConstructor(layer_cfg)
    an = SpectralAnalyzer(layer_cfg)
    cases = {}
    for kind in ("random", "sink", "causal"):
        for seq in (16, 64, 200):
            for dim in (32, 256):
                for seed in (0, 1):
                    key = f"{kind},S={seq},d={dim},seed={seed}"
                    attn = attention(kind, 4, seq, seed)
                    X = hidden(seq, dim, seed)
                    # the framework's layer pipeline
                    sym = gc.symmetrize_attention(attn.unsqueeze(0)).squeeze(0)
                    adj = gc.aggregate_heads(sym.unsqueeze(0)).squeeze(0)
                    L = gc.construct_laplacian(adj.unsqueeze(0)).squeeze(0)
                    layer = an.analyze_layer(X, L, 0).to_dict()
                    span = list(range(seq // 4, seq))
                    L_sub = gc.subgraph_laplacian(adj.unsqueeze(0), span).squeeze(0)
                    X_sub = X.index_select(0, torch.tensor(span))
                    sub = an.analyze_layer(X_sub, L_sub, 0).to_dict()
                    # float64 reference energies (independent of the library)
                    L64 = L.double().numpy()
                    X64 = X.double().numpy()
                    ref_energy = float(np.sum(X64 * (L64 @ X64)))
                    ref_smooth = ref_energy / float(np.sum(X64 * X64))
                    heads = per_head_metrics(attn, config=head_cfg).values.tolist()
                    heads_span = per_head_metrics(attn, config=head_cfg,
                                                  token_span=(seq // 4, seq)).values.tolist()
                    cases[key] = {"layer": layer, "subgraph": sub,
                                  "reference": {"energy": ref_energy,
                                                "smoothness_index": ref_smooth},
                                  "per_head": heads, "per_head_span": heads_span}
    stack = [attention("causal", 4, 64, s) for s in range(3)]
    diags = per_head_metrics_all_layers(stack, config=head_cfg)
    cases["all_layers"] = {
        "features": stack_feature_vector(diags).tolist(),
        "deltas": layer_delta_features(diags, metric="spectral_radius").tolist(),
    }
    with open(out, "w") as f:
        json.dump({"version": st.__version__, "cases": cases}, f)


# ── Driver: run both trees and compare ───────────────────────────────────────

def extract(ref, dest):
    blob = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", ref,
                           "src/spectral_trust"], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return os.path.join(dest, "src")


def run_worker(src, out):
    subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                    "--src", src, "--out", out, "--threads", str(args.threads)],
                   check=True, cwd=tempfile.gettempdir())
    with open(out) as f:
        return json.load(f)


def flatten(prefix, obj, out):
    """Leaf values of nested dicts/lists, keyed by metric name (list indices dropped)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten(prefix, v, out)
    else:
        out.setdefault(prefix, []).append(obj)


def compare(old, new):
    stats = {}
    for case, old_vals in old["cases"].items():
        a, b = {}, {}
        flatten("", {k: v for k, v in old_vals.items() if k != "reference"}, a)
        flatten("", {k: v for k, v in new["cases"][case].items() if k != "reference"}, b)
        assert a.keys() == b.keys(), (case, a.keys() ^ b.keys())
        for metric in a:
            x, y = a[metric], b[metric]
            assert len(x) == len(y), (case, metric)
            s = stats.setdefault(metric, {"values": 0, "identical": 0, "max_rel_diff": 0.0,
                                          "worst_case": None})
            if x and isinstance(x[0], (bool, str)) or (x and x[0] is None):
                s["values"] += len(x)
                s["identical"] += sum(p == q for p, q in zip(x, y))
                if any(p != q for p, q in zip(x, y)):
                    s["max_rel_diff"] = float("inf")
                    s["worst_case"] = case
                continue
            scale = max([abs(v) for v in x] + [1e-30])
            for p, q in zip(x, y):
                s["values"] += 1
                if p == q:
                    s["identical"] += 1
                    continue
                rel = abs(p - q) / max(abs(p), 1e-7 * scale)
                if rel > s["max_rel_diff"]:
                    s["max_rel_diff"], s["worst_case"] = rel, case
    # where 0.3.0 and the current code differ, which is closer to float64?
    attribution = {}
    for metric in ("energy", "smoothness_index"):
        e_old = e_new = 0.0
        for case, vals in old["cases"].items():
            if "reference" not in vals:
                continue
            ref = vals["reference"][metric]
            e_old = max(e_old, abs(vals["layer"][metric] - ref) / abs(ref))
            e_new = max(e_new, abs(new["cases"][case]["layer"][metric] - ref) / abs(ref))
        attribution[metric] = {"v030_max_rel_err_vs_float64": e_old,
                               "current_max_rel_err_vs_float64": e_new}
    return stats, attribution


def main():
    with tempfile.TemporaryDirectory() as tmp:
        old_src = extract(args.ref, os.path.join(tmp, "v030"))
        old = run_worker(old_src, os.path.join(tmp, "old.json"))
        new = run_worker(os.path.join(REPO, "src"), os.path.join(tmp, "new.json"))
    stats, attribution = compare(old, new)
    worst = max(s["max_rel_diff"] for s in stats.values())
    result = {"reference_ref": args.ref, "reference_version": old["version"],
              "current_version": new["version"], "tolerance": TOLERANCE,
              "reproduced": worst < TOLERANCE, "max_rel_diff": worst,
              "cases": len(old["cases"]), "metrics": stats,
              "float64_attribution": attribution}
    with open(os.path.join(HERE, "reproduce_v030.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    lines = [f"# Reproduction of {old['version']} ({args.ref}) by {new['version']}", "",
             f"{len(old['cases'])} seeded input sets (random, sink-dominated and causal "
             "attention, S in {16, 64, 200}, d in {32, 256}, two seeds, plus a three-layer "
             "per-head stack). Layer pipeline: normalization=\"sym\", eigen_solver=\"dense\", "
             "full graph and token subgraph. Per-head: attention-only metrics, default "
             "configuration, with and without token_span.", "",
             f"**Result: {'reproduced' if worst < TOLERANCE else 'NOT reproduced'}** - "
             f"largest relative difference {worst:.2e} (tolerance {TOLERANCE:.0e}).", "",
             "| value | values compared | bitwise identical | max relative difference |",
             "|---|---|---|---|"]
    families = {"values": 0, "identical": 0}
    for name in sorted(stats):
        s = stats[name]
        if ".families." in name:              # metric-family tags (strings)
            families["values"] += s["values"]
            families["identical"] += s["identical"]
            continue
        lines.append(f"| `{name}` | {s['values']} | {s['identical']} | {s['max_rel_diff']:.2e} |")
    lines.append(f"| metric-family tags | {families['values']} | {families['identical']} "
                 f"| {'0' if families['values'] == families['identical'] else 'differ'} |")
    lines += ["", "Relative difference: |old - new| / max(|old|, 1e-7 x the largest |old| of "
              "that value in the case), so entries that are zero up to rounding are measured "
              "on the scale of their vector.", "",
              "Against a float64 evaluation of tr(X^T L X) on the same float32 inputs "
              "(maximum relative error over all cases):", "",
              "| value | 0.3.0 | current |", "|---|---|---|"]
    for metric, a in attribution.items():
        lines.append(f"| `{metric}` | {a['v030_max_rel_err_vs_float64']:.2e} "
                     f"| {a['current_max_rel_err_vs_float64']:.2e} |")
    with open(os.path.join(HERE, "reproduce_v030.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if worst < TOLERANCE else 1


if __name__ == "__main__":
    if args.worker:
        worker(args.src, args.out)
    else:
        sys.exit(main())
