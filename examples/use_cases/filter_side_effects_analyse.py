# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Analysis of filter_side_effects.py: one row per (layer, filter,
strength, condition), the attention-minus-control contrasts and a summary,
in ``outputs/filter_side_effects.json``.

    python examples/use_cases/filter_side_effects_analyse.py
"""
import json

import _common as C
import numpy as np

CONDS = ("attention", "random", "path", "scale")


def band(x):
    """Mean over position groups of ``x [..., S]`` (S positions 0..S-1):
    first edited (1), early (2-4), middle (5..S-2), last."""
    return {"pos1": float(x[..., 1].mean()), "pos2_4": float(x[..., 2:5].mean()),
            "middle": float(x[..., 5:-1].mean()), "last": float(x[..., -1].mean())}


def cell(d, g, key, l, idx):
    w0, k = "clean|", f"{key}|"
    kl = d[k + "wiki_kl"][:, 1:].mean(1)
    dn = d[k + "wiki_nll"] - d[w0 + "wiki_nll"]
    lam = d[k + "wiki_lam"] - d[w0 + "wiki_lam"]
    sink = d[k + "wiki_sink"] - d[w0 + "wiki_sink"]
    pos = np.concatenate([np.zeros((d[k + "edit_pos"].shape[0], 1)), d[k + "edit_pos"]], 1)
    gen = g[key]
    return {"edit_size": float(d[k + "edit_size"].mean()),
            "norm_change": float(d[k + "edit_normchg"].mean()),
            "wiki_kl": C.mean_ci(kl, idx), "wiki_top1_disagree": float(1 - d[k + "wiki_top"][:, 1:].mean()),
            "wiki_dnll": C.mean_ci(dn, idx),
            "pas_kl": float(d[k + "pas_kl"].mean()),
            "pas_dnll": float((d[k + "pas_nll"] - d[w0 + "pas_nll"]).mean()),
            "dlam2_next": float(lam[:, l + 1].mean()) if l + 1 < lam.shape[1] else None,
            "dlam2_later": float(lam[:, l + 1:].mean()) if l + 1 < lam.shape[1] else None,
            "dsink_next": float(sink[:, l + 1].mean()) if l + 1 < sink.shape[1] else None,
            "dsink_later": float(sink[:, l + 1:].mean()) if l + 1 < sink.shape[1] else None,
            "edit_by_position": band(pos), "kl_by_position": band(d[k + "wiki_kl"]),
            "final_hidden_change_by_position": band(d[k + "wiki_hid"]),
            "gen_degenerate": float(np.mean([x["degenerate"] for x in gen])),
            "gen_rep": float(np.mean([x["rep"] for x in gen])),
            "gen_distinct2": float(np.mean([x["distinct2"] for x in gen])),
            "gen_changed": float(np.mean([x["first_diff"] < 30 for x in gen])),
            "gen_first_diff": float(np.mean([x["first_diff"] for x in gen]))}


SUMMARY_KEYS = ("edit_size", "norm_change", "wiki_top1_disagree", "pas_kl", "pas_dnll",
                "dlam2_next", "dlam2_later", "dsink_next", "dsink_later", "gen_degenerate",
                "gen_rep", "gen_distinct2", "gen_changed")


def analyse_model(tag):
    d = dict(np.load(C.OUTPUTS / f"filter_side_effects_{tag}.npz"))
    g = json.load(open(C.OUTPUTS / f"filter_side_effects_{tag}_generations.json", encoding="utf-8"))
    meta = json.load(open(C.OUTPUTS / f"filter_side_effects_{tag}.meta.json"))
    idx = C.boot_idx(len(d["clean|wiki_nll"]), seed=3)
    rows = []
    for l in meta["layers"]:
        for f in meta["filters"]:
            for s in meta["strengths"]:
                row = {"layer": l, "filter": f, "strength": s}
                for c in CONDS:
                    row[c] = cell(d, g, f"{l}|{f}|{s}|{c}", l, idx)
                for c in CONDS[1:]:
                    a, b = f"{l}|{f}|{s}|attention|", f"{l}|{f}|{s}|{c}|"
                    row[f"attention_minus_{c}"] = {
                        "wiki_kl": C.mean_ci(d[a + "wiki_kl"][:, 1:].mean(1)
                                             - d[b + "wiki_kl"][:, 1:].mean(1), idx),
                        "wiki_dnll": C.mean_ci(d[a + "wiki_nll"] - d[b + "wiki_nll"], idx)}
                rows.append(row)
    summ = {}
    for c in CONDS:
        summ[c] = {k: float(np.mean([r[c][k] for r in rows if r[c][k] is not None]))
                   for k in SUMMARY_KEYS}
        summ[c]["wiki_kl"] = float(np.mean([r[c]["wiki_kl"][0] for r in rows]))
        summ[c]["wiki_dnll"] = float(np.mean([r[c]["wiki_dnll"][0] for r in rows]))
        for part in ("edit_by_position", "kl_by_position", "final_hidden_change_by_position"):
            summ[c][part] = {p: float(np.mean([r[c][part][p] for r in rows]))
                             for p in ("pos1", "pos2_4", "middle", "last")}
    counts = {}
    for c in CONDS[1:]:
        for m in ("wiki_kl", "wiki_dnll"):
            ci = [r[f"attention_minus_{c}"][m] for r in rows]
            counts[f"{m}_attention_above_{c}"] = sum(x[1] > 0 for x in ci)
            counts[f"{m}_attention_below_{c}"] = sum(x[2] < 0 for x in ci)
    by_filter = {f: {c: {k: float(np.mean([r[c][k] if not isinstance(r[c][k], list)
                                           else r[c][k][0] for r in rows if r["filter"] == f
                                           and r[c][k] is not None]))
                         for k in ("wiki_kl", "wiki_dnll", "gen_degenerate", "dlam2_next",
                                   "dsink_next")} for c in CONDS} for f in meta["filters"]}
    clean = g["clean"]
    examples = {"clean_degenerate_share": float(np.mean([x["degenerate"] for x in clean])),
                "clean": [x["text"] for x in clean[:3]]}
    worst = max((k for k in g if k != "clean"),
                key=lambda k: np.mean([x["degenerate"] for x in g[k]]))
    examples["most_degenerate_condition"] = worst
    examples["most_degenerate_share"] = float(np.mean([x["degenerate"] for x in g[worst]]))
    examples["most_degenerate_examples"] = [x["text"] for x in g[worst] if x["degenerate"]][:3]
    return {"meta": meta, "check_library_vs_hook": float(d["check_library_vs_hook"]),
            "runtime_s": float(d["runtime_s"]), "n_cells": len(rows),
            "clean_wiki_nll": C.mean_ci(d["clean|wiki_nll"], idx), "summary": summ,
            "contrast_counts": counts, "by_filter": by_filter, "generation": examples,
            "cells": rows}


if __name__ == "__main__":
    out = {}
    for p in sorted(C.OUTPUTS.glob("filter_side_effects_*.meta.json")):
        tag = p.name.replace("filter_side_effects_", "").replace(".meta.json", "")
        out[tag] = analyse_model(tag)
        print("analysed", tag, flush=True)
    C.save_json("filter_side_effects.json", out)
