# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Pilot: the passive-voice drop of the attention graph's Fiedler value.

An earlier measurement, made with spectral-trust 0.1.2, reported that in
Phi-3-mini-4k-instruct English passive sentences drop the head-averaged
Fiedler value at layers 3 to 5 from about 0.94 (actives) to 0.08, on three
sentences per voice. This
script measures it on the sentence set of ``passive_data.py`` (72 matched
English pairs and the controls) and asks whether attention-sink mass
explains it.

Per sentence and layer, from the layer's attention ``A [H, S, S]``:

- ``lam2``: the library's Fiedler value (``run_with_spectra``,
  ``graph.fiedler_value``: per-head symmetrisation, uniform head average,
  symmetric normalised Laplacian, dense eigendecomposition);
- ``lam2_f64``: the same computed here in float64 (a check);
- ``lam2_arch``: the construction of spectral-trust 0.1.2 (``lam2_v012``): the
  random-walk Laplacian ``I - D^-1 W`` in float32 passed to
  ``scipy.linalg.eigh``, which reads only its lower triangle, with
  eigenvalues clipped at 0;
- ``lam2_nosink``: the float64 Fiedler value of the graph without position 0;
- ``sink``: attention received by position 0, the mean over queries 1..S-1
  of the head-averaged weight; ``top``: the largest such received mass over
  all keys, ``top_ns`` over keys 1..S-1, and their positions;
- ``fw``: received mass of the passive's function words ("was"/"were",
  "by"; French "été", "par"; German "wurde", "vom", "von"), 0 when absent;
- ``small_side``: the Fiedler split, the tokens on the smaller side of the
  sign split of the float64 Fiedler vector (ties: the side without
  position 0).

    python examples/use_cases/passive_drop.py --model microsoft/Phi-3-mini-4k-instruct --dtype bfloat16
    python examples/use_cases/passive_drop.py --model microsoft/Phi-3-mini-4k-instruct --dtype float32 --n-layers 8
    python examples/use_cases/passive_drop.py --analyse

Set ``USE_CASE_DEVICE=cuda`` to run on the GPU. Each run is saved to
``outputs/passive_drop_<tag>.npz`` and skipped when that file exists.
"""
import argparse
import gc
import json
import re

import _common as C
import numpy as np
import scipy.linalg
import torch

import spectral_trust as st
from passive_data import load as load_items

FUNCTION_WORDS = {"en": {"was", "were", "by"}, "fr": {"été", "par"},
                  "de": {"wurde", "vom", "von"}}
AUX = {"was", "were", "été", "wurde"}
BYW = {"by", "par", "vom", "von"}
DETS = {"the", "a", "an", "le", "la", "l", "les", "der", "die", "das", "den", "dem"}
ROLES = ("first", "det", "agent", "patient", "verb", "aux", "by", "period", "other")
FOCUS_LAYERS = (2, 3, 4, 5)


def tag_of(model: str, dtype: str, n_layers) -> str:
    t = model.split("/")[-1] + "_" + dtype
    return t + (f"_first{n_layers}" if n_layers else "")


# ── per-sentence measurements ────────────────────────────────────────────────

def token_roles(text: str, item: dict, offsets) -> list:
    """Role of each token (``ROLES``), from the word its offsets overlap."""
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
    agent = set(item.get("agent", "").split())
    patient = set(item.get("patient", "").split())
    verb = item.get("verb")
    roles = []
    for t, (a, b) in enumerate(offsets):
        if t == 0:
            roles.append("first")
            continue
        w = next((w for w, s, e in words if s < b and a < e), None)
        lw = (w or "").lower()
        if w is None:
            r = "other"
        elif lw in AUX:
            r = "aux"
        elif lw in BYW:
            r = "by"
        elif w == ".":
            r = "period"
        elif lw in DETS:
            r = "det"
        elif lw in agent:
            r = "agent"
        elif lw in patient:
            r = "patient"
        elif verb is not None and lw == verb:
            r = "verb"
        else:
            r = "other"
        roles.append(r)
    return roles


def lam2_v012(A: np.ndarray) -> float:
    """Fiedler value as spectral-trust 0.1.2 computed it (GSPConfig defaults:
    symmetric, uniform, normalization="rw", dense ``scipy.linalg.eigh``)."""
    A32 = A.astype(np.float32)
    W = (0.5 * (A32 + A32.transpose(0, 2, 1))).mean(0)
    d = W.sum(-1)
    inv = np.where(d > 1e-8, 1.0 / np.maximum(d, 1e-8), 0.0).astype(np.float32)
    L = np.eye(len(d), dtype=np.float32) - inv[:, None] * W
    ev = np.maximum(scipy.linalg.eigh(L, eigvals_only=True), 0)
    return float(ev[1])


def sym_spectrum(W: np.ndarray):
    d = W.sum(-1)
    inv = np.where(d > 1e-12, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    L = np.eye(len(d)) - inv[:, None] * W * inv[None, :]
    return np.linalg.eigh(L)


def measure(A: np.ndarray, fw_pos) -> dict:
    """``A [H, S, S]`` float64 -> the measurements of one layer."""
    S = A.shape[-1]
    Ab = A.mean(0)
    W = 0.5 * (Ab + Ab.T)
    ev, U = sym_spectrum(W)
    u = U[:, 1]
    pos = u > 0
    n_pos = int(pos.sum())
    if n_pos * 2 < S or (n_pos * 2 == S and not pos[0]):
        small = pos
    else:
        small = ~pos
    ev_ns, _ = sym_spectrum(W[1:, 1:]) if S > 2 else (np.array([0.0, 0.0]), None)
    recv = Ab[1:].mean(0)
    return dict(lam2_f64=float(ev[1]), lam2_arch=lam2_v012(A),
                lam2_nosink=float(ev_ns[1]), sink=float(recv[0]),
                top=float(recv.max()), top_pos=int(recv.argmax()),
                top_ns=float(recv[1:].max()), top_ns_pos=int(1 + recv[1:].argmax()),
                fw=float(recv[fw_pos].sum()) if fw_pos else 0.0,
                small_side=small.astype(bool))


# ── run ──────────────────────────────────────────────────────────────────────

def load_model(name: str, dtype: str, device: str, n_layers=None):
    """A SpectralModel with eager attention; with ``n_layers`` only the
    first blocks are kept (loaded in bfloat16, truncated, then cast), which
    leaves the attention of those layers exactly as in the full model."""
    if n_layers:
        sm = st.SpectralModel.from_pretrained(name, dtype="bfloat16", device_map=device)
        base = sm.model.model
        base.layers = base.layers[:n_layers]
        sm.model.config.num_hidden_layers = n_layers
        if getattr(sm.model.config, "layer_types", None):
            sm.model.config.layer_types = sm.model.config.layer_types[:n_layers]
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        sm.model.to(getattr(torch, dtype))
    else:
        sm = st.SpectralModel.from_pretrained(name, dtype=dtype, device_map=device)
    sm.model.eval()
    return sm


def run(model: str, dtype: str, device: str, n_layers=None) -> None:
    tag = tag_of(model, dtype, n_layers)
    path = C.OUTPUTS / f"passive_drop_{tag}.npz"
    if path.exists():
        print(f"{path} exists; skipped", flush=True)
        return
    timer = C.Timer()
    items = load_items()
    sm = load_model(model, dtype, device, n_layers)
    tok = sm.tokenizer
    print(C.library_info(), f"device {device} dtype {dtype}", f"disk {C.check_disk():.1f} GB",
          f"load {timer()} s", flush=True)

    # the library's measurement, batched (right padding is cropped)
    texts = [it["text"] for it in items]
    cache = sm.run_with_spectra(texts, metrics=["graph.fiedler_value"], hybrid=False,
                                batch_size=16, keep_patterns=False)
    layers = cache.layers
    lam2_lib = np.stack([np.asarray(cache[f"blocks.{l}.graph.fiedler_value"], np.float64)
                         for l in layers], axis=1)                       # [N, L]
    print(f"run_with_spectra done ({timer()} s)", flush=True)

    keys = ("lam2_f64", "lam2_arch", "lam2_nosink", "sink", "top", "top_pos", "top_ns",
            "top_ns_pos", "fw")
    N, L = len(items), len(layers)
    out = {k: np.full((N, L), np.nan) for k in keys}
    meta = []
    small_sides = []
    for n, it in enumerate(items):
        enc = tok(it["text"], return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        toks = [tok.decode([i]) for i in ids]
        roles = token_roles(it["text"], it, offs)
        fws = FUNCTION_WORDS[it["lang"]]
        words = [(m.group(0).lower(), m.start(), m.end())
                 for m in re.finditer(r"\w+|[^\w\s]", it["text"])]
        fw_pos = [t for t, (a, b) in enumerate(offs) if t > 0 and any(
            w in fws and s < b and a < e for w, s, e in words)]
        with torch.no_grad():
            o = sm.model(input_ids=torch.tensor([ids], device=sm.device), output_attentions=True,
                         use_cache=False)
        sides = []
        for j, l in enumerate(layers):
            A = o.attentions[j][0].double().cpu().numpy()
            m = measure(A, fw_pos)
            for k in keys:
                out[k][n, j] = m[k]
            sides.append(m["small_side"])
        small_sides.append(np.stack(sides))
        meta.append(dict(id=it["id"], tokens=toks, roles=roles, fw_pos=fw_pos))
        if n % 50 == 0:
            print(f"{n}/{N} ({timer()} s)", flush=True)
    check = float(np.nanmax(np.abs(lam2_lib - out["lam2_f64"])))
    print(f"max |library - float64| = {check:.2e}", flush=True)
    S_max = max(s.shape[1] for s in small_sides)
    sides = np.zeros((N, L, S_max), dtype=bool)
    for n, s in enumerate(small_sides):
        sides[n, :, :s.shape[1]] = s
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, lam2=lam2_lib, layers=np.array(layers), small_side=sides,
                        check_lib_vs_f64=check, runtime_s=timer(), **out)
    with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(model=model, dtype=dtype, device=device, n_layers=n_layers,
                       library=C.library_info(), items=meta,
                       gpu=torch.cuda.get_device_name(0) if device == "cuda" else None),
                  f, ensure_ascii=False)
    print(f"saved {path} ({timer()} s)", flush=True)
    del sm
    gc.collect()


# ── analysis ─────────────────────────────────────────────────────────────────

def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie)."""
    x = np.concatenate([neg, pos])
    from scipy.stats import rankdata
    r = rankdata(x)
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def paired(items, group, a_voice="active", b_voice="passive"):
    """Indices (a, b) of the pairs of a group."""
    by = {}
    for i, it in enumerate(items):
        if it["group"] == group:
            by.setdefault(it["pair"], {})[it["voice"]] = i
    pairs = [(v[a_voice], v[b_voice]) for v in by.values() if a_voice in v and b_voice in v]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def analyse_run(d: dict, meta: dict, items: list) -> dict:
    layers = list(d["layers"])
    res = {"model": meta["model"], "dtype": meta["dtype"], "device": meta["device"],
           "gpu": meta.get("gpu"), "n_layers_run": meta["n_layers"],
           "check_lib_vs_f64": float(d["check_lib_vs_f64"]), "runtime_s": float(d["runtime_s"]),
           "layers": layers}
    lengths = np.array([len(m["tokens"]) for m in meta["items"]], dtype=np.float64)
    d = dict(d, length=np.repeat(lengths[:, None], len(layers), axis=1))
    feats = ("lam2", "lam2_arch", "lam2_nosink", "sink", "top", "top_ns", "fw", "length")
    groups = {"en_pair": ("active", "passive"), "en_firstword": ("active", "passive"),
              "en_agentless": ("active", "agentless"), "fr_pair": ("active", "passive"),
              "de_pair": ("active", "passive")}
    res["groups"] = {}
    for gi, (g, (va, vb)) in enumerate(groups.items()):
        ia, ib = paired(items, g, va, vb)
        if len(ia) == 0:
            continue
        idx = C.boot_idx(len(ia), seed=gi)
        gr = {"n_pairs": int(len(ia)), "voices": [va, vb]}
        for f in feats:
            a, b = d[f][ia], d[f][ib]
            gr[f] = {"mean_a": a.mean(0).tolist(), "mean_b": b.mean(0).tolist(),
                     "diff_b_minus_a": [C.mean_ci(b[:, j] - a[:, j], idx) for j in range(len(layers))]}
        # AUROC b (passive) vs a (active) at the focus layers, with pair bootstrap
        au = {}
        for f in feats:
            au[f] = {}
            for l in FOCUS_LAYERS:
                if l not in layers:
                    continue
                j = layers.index(l)
                a, b = d[f][ia, j], d[f][ib, j]
                point = auroc(a, b)
                bs = np.array([auroc(a[s], b[s]) for s in idx[:2000]])
                au[f][l] = [point, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        gr["auroc_b_over_a"] = au
        # length-adjusted paired difference: the slope of the feature on token
        # length among this group's first members (actives), refitted in each
        # resample, times the length difference, is subtracted
        dl = lengths[ib] - lengths[ia]
        gr["length_diff_mean"] = float(dl.mean())
        adj = {}
        for f in ("lam2", "sink", "lam2_arch"):
            adj[f] = []
            for j in range(len(layers)):
                ya, yb = d[f][ia, j], d[f][ib, j]

                def adjusted(s):
                    x = lengths[ia][s]
                    beta = np.polyfit(x, ya[s], 1)[0] if np.ptp(x) > 0 else 0.0
                    return float(np.mean((yb[s] - ya[s]) - beta * dl[s])), float(beta)
                point, beta = adjusted(np.arange(len(ia)))
                bs = np.array([adjusted(s)[0] for s in idx[:2000]])
                adj[f].append({"adjusted_diff": [point, float(np.quantile(bs, 0.025)),
                                                 float(np.quantile(bs, 0.975))],
                               "slope_per_token": beta})
        gr["length_adjusted"] = adj
        # paired: share of pairs in which the passive is below the active
        gr["share_b_below_a"] = {f: [float((d[f][ib, j] < d[f][ia, j]).mean())
                                     for j in range(len(layers))] for f in ("lam2", "sink")}
        res["groups"][g] = gr
    # unpaired: by-actives vs the short actives of en_pair (T1), and vs T1 passives
    t1a = np.array([i for i, it in enumerate(items) if it["pair"].startswith("en_t1_")
                    and it["voice"] == "active"])
    t1p = np.array([i for i, it in enumerate(items) if it["pair"].startswith("en_t1_")
                    and it["voice"] == "passive"])
    by = np.array([i for i, it in enumerate(items) if it["group"] == "en_by_active"])
    res["by_active"] = {"mean_length": {"by_active": float(lengths[by].mean()),
                                        "t1_active": float(lengths[t1a].mean()),
                                        "t1_passive": float(lengths[t1p].mean())}}
    # contrasts by-active - T1 active, T1 passive - T1 active, by-active - T1
    # passive: raw, and length-adjusted (each value minus slope * length, the
    # slope fitted on the en_pair actives); unpaired bootstrap, 2000 draws
    ea, _ = paired(items, "en_pair")
    rng = np.random.default_rng(7)
    draws = [(rng.integers(0, len(by), len(by)), rng.integers(0, len(t1a), len(t1a)),
              rng.integers(0, len(t1p), len(t1p)), rng.integers(0, len(ea), len(ea)))
             for _ in range(2000)]
    names = ("by_minus_t1_active", "t1_passive_minus_t1_active", "by_minus_t1_passive")
    for f in ("lam2", "sink", "lam2_arch"):
        out_f = {"mean_by_active": d[f][by].mean(0).tolist(),
                 "mean_t1_active": d[f][t1a].mean(0).tolist(),
                 "mean_t1_passive": d[f][t1p].mean(0).tolist(), "raw": {}, "adjusted": {}}
        for j in range(len(layers)):
            def contrasts(sb, sa, sp, sx, adjust):
                beta = np.polyfit(lengths[ea][sx], d[f][ea, j][sx], 1)[0] if adjust else 0.0
                m = [float(np.mean(d[f][g, j] - beta * lengths[g])) for g in
                     (by[sb], t1a[sa], t1p[sp])]
                return [m[0] - m[1], m[2] - m[1], m[0] - m[2]]
            full = (np.arange(len(by)), np.arange(len(t1a)), np.arange(len(t1p)),
                    np.arange(len(ea)))
            for mode, adjust in (("raw", False), ("adjusted", True)):
                pt = contrasts(*full, adjust)
                bs = np.array([contrasts(*dr, adjust) for dr in draws])
                out_f[mode][j] = {n: [pt[k], float(np.quantile(bs[:, k], 0.025)),
                                      float(np.quantile(bs[:, k], 0.975))]
                                  for k, n in enumerate(names)}
        res["by_active"][f] = out_f
    # original pair, per layer
    oa, op = paired(items, "original")
    res["original"] = {f: {"active": d[f][oa[0]].tolist(), "passive": d[f][op[0]].tolist()}
                       for f in ("lam2", "lam2_arch", "sink", "top")}
    # Fiedler split at the focus layers: how often each role is on the small side,
    # for en_pair passives and actives
    ia, ib = paired(items, "en_pair")
    split = {}
    for voice, ids in (("active", ia), ("passive", ib)):
        split[voice] = {}
        for l in FOCUS_LAYERS:
            if l not in layers:
                continue
            j = layers.index(l)
            counts = {r: [0, 0] for r in ROLES}       # [on small side, present]
            sizes = []
            for i in ids:
                roles = meta["items"][i]["roles"]
                side = d["small_side"][i, j, :len(roles)]
                sizes.append(int(side.sum()))
                for r in set(roles):
                    counts[r][1] += 1
                    if any(side[t] for t, rr in enumerate(roles) if rr == r):
                        counts[r][0] += 1
            split[voice][l] = {"mean_small_side_size": float(np.mean(sizes)),
                               "share_with_role_on_small_side": {
                                   r: (c[0] / c[1] if c[1] else None) for r, c in counts.items()}}
    res["fiedler_split_en_pair"] = split
    # examples of the split
    ex = []
    for i in list(ib[:3]) + list(ia[:2]):
        toks = meta["items"][i]["tokens"]
        row = {"text": items[i]["text"]}
        for l in FOCUS_LAYERS:
            if l in layers:
                j = layers.index(l)
                row[f"layer{l}"] = {"lam2": float(d["lam2"][i, j]),
                                    "small_side": [t for t, s in zip(toks, d["small_side"][i, j])
                                                   if s]}
        ex.append(row)
    res["split_examples"] = ex
    # within-voice correlation of lam2 and sink, and of lam2 and length
    from scipy.stats import spearmanr
    cor = {}
    for l in FOCUS_LAYERS:
        if l in layers:
            j = layers.index(l)
            both = np.concatenate([ia, ib])
            cor[l] = {"lam2_vs_sink_all_en_pair": float(spearmanr(d["lam2"][both, j],
                                                                   d["sink"][both, j])[0]),
                      "lam2_vs_sink_passives": float(spearmanr(d["lam2"][ib, j], d["sink"][ib, j])[0]),
                      "lam2_vs_length_actives": float(spearmanr(d["lam2"][ia, j], lengths[ia])[0])}
    res["spearman"] = cor
    return res


def analyse() -> dict:
    items = load_items()
    out = {"runs": {}}
    for path in sorted(C.OUTPUTS.glob("passive_drop_*.npz")):
        meta = json.load(open(path.with_suffix(".meta.json"), encoding="utf-8"))
        d = dict(np.load(path))
        tag = path.stem.replace("passive_drop_", "")
        out["runs"][tag] = analyse_run(d, meta, items)
        print(f"analysed {tag}", flush=True)
    C.save_json("passive_drop.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        analyse()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run(args.model, args.dtype, device, args.n_layers)
