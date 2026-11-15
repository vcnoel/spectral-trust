# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Use case 1: the renormalisation choice of an attention-edge ablation.

GPT-2 small, the 200 IOI prompts of the official dataset code. At the END
position (the token before the answer), the edges from END to the IO and S
tokens (IO, S1, S2) are removed in one or more heads, with three ways of
treating the row afterwards:

- ``keep_sink``: ``remove_edges(mask)``, the default: the first token's entry
  keeps its value and the removed mass goes back to the other content keys;
- ``plain``: ``remove_edges(mask, keep_sink=False)``: the row is renormalised
  over every remaining key, the first token included (what a pattern hook that
  divides by the row sum does);
- ``zero``: ``remove_edges(mask, renormalize=False)``: the entries are set to
  zero and the row sums to less than one.

Head sets: each of the 144 heads alone; the name movers 9.6, 9.9 and 10.0
together; 20 random triples of other heads (seed 0). A check of the sink edge
itself: the END -> first-token edge (removable only with ``keep_sink=False``)
against 3 random content edges per prompt, in the same head sets.

Metric: LD = logit(IO) - logit(S) at END. Effect: LD(ablated) - LD(clean),
per prompt; 95% percentile bootstrap intervals over prompts (10,000
resamples, the same resamples for every quantity). Removed mass: the
attention the removed edges carried in the clean run, at END, averaged over
the heads of the set.

    python examples/use_cases/sink_ablation.py            # run and plot (about 45 min, 2 threads)
    python examples/use_cases/sink_ablation.py --plot     # plot from outputs/sink_ablation.npz
"""
import argparse

import _common as C  # noqa: F401  (CPU, 2 threads, offline; before torch)
import numpy as np
import torch

import spectral_trust as st

NAME_MOVERS = [(9, 6), (9, 9), (10, 0)]
MODES = {"keep_sink": dict(keep_sink=True), "plain": dict(keep_sink=False),
         "zero": dict(keep_sink=True, renormalize=False)}
N_RANDOM_SETS = 20
NPZ = C.OUTPUTS / "sink_ablation.npz"


# ── model runs ───────────────────────────────────────────────────────────────

class Runner:
    def __init__(self):
        self.pairs = C.ioi_pairs()
        self.sm = st.SpectralModel.from_pretrained("gpt2")
        self.hf = self.sm.model.eval()
        self.cfg = self.hf.config
        self.groups = C.by_length(self.pairs)
        self.io = torch.tensor([p["io"] for p in self.pairs])
        self.s = torch.tensor([p["s"] for p in self.pairs])
        rng_pos = []
        for k, p in enumerate(self.pairs):
            names = {p["pos_io"], p["pos_s1"], p["pos_s2"]}
            pool = [j for j in range(1, p["end"]) if j not in names]
            rng = np.random.default_rng([0, k])
            rng_pos.append(sorted(rng.choice(pool, size=3, replace=False).tolist()))
        self.content_pos = rng_pos

    def keys(self, which, k):
        p = self.pairs[k]
        if which == "names":
            return [p["pos_io"], p["pos_s1"], p["pos_s2"]]
        if which == "sink":
            return [0]
        if which == "content":
            return self.content_pos[k]
        raise ValueError(which)

    def ld(self, heads=None, which="names", mode=None) -> np.ndarray:
        """LD per prompt with the END row's edges ``which`` removed in ``heads``
        (a list of (layer, head)); clean LD when ``heads`` is None."""
        out = np.zeros(len(self.pairs))
        for T, idx in self.groups.items():
            ids = torch.stack([self.pairs[i]["clean"] for i in idx])
            with torch.no_grad():
                if heads is None:
                    logits = self.hf(input_ids=ids, logits_to_keep=1).logits[:, -1]
                else:
                    km = torch.zeros(len(idx), T, T, dtype=torch.bool)
                    for j, i in enumerate(idx):
                        km[j, self.pairs[i]["end"], self.keys(which, i)] = True
                    by_layer = {}
                    for layer, h in heads:
                        by_layer.setdefault(layer, []).append(h)
                    with st.intervene(self.hf) as iv:
                        for layer, hs in by_layer.items():
                            iv.attention(layer, hs).remove_edges(km, **MODES[mode])
                        logits = self.hf(input_ids=ids, logits_to_keep=1).logits[:, -1]
            r = torch.arange(len(idx))
            ii = torch.as_tensor(idx)
            out[idx] = (logits[r, self.io[ii]] - logits[r, self.s[ii]]).numpy()
        return out

    def end_rows(self) -> np.ndarray:
        """Clean attention mass at END, ``[prompts, layers, heads, 3]``: on the
        names, on the first token, on the 3 random content keys."""
        L, H = self.cfg.n_layer, self.cfg.n_head
        out = np.zeros((len(self.pairs), L, H, 3))
        for T, idx in self.groups.items():
            ids = torch.stack([self.pairs[i]["clean"] for i in idx])
            with torch.no_grad():
                att = self.hf(input_ids=ids, output_attentions=True).attentions
            A = torch.stack(att, dim=1)                                 # [B, L, H, T, T]
            for j, i in enumerate(idx):
                row = A[j, :, :, self.pairs[i]["end"], :]               # [L, H, T]
                for c, which in enumerate(("names", "sink", "content")):
                    out[i, :, :, c] = row[:, :, self.keys(which, i)].sum(-1).numpy()
        return out


def random_sets(n_layers=12, n_heads=12, n=N_RANDOM_SETS, seed=0):
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in NAME_MOVERS]
    rng = np.random.default_rng(seed)
    return [sorted(pool[i] for i in rng.choice(len(pool), size=3, replace=False))
            for _ in range(n)]


def run(test: bool = False) -> dict:
    timer = C.Timer()
    R = Runner()
    L, H = R.cfg.n_layer, R.cfg.n_head
    if test:                          # code check: 2 prompts per length, layer 9 only
        R.groups = {T: idx[:2] for T, idx in R.groups.items()}
    print(C.library_info(), f"disk {C.check_disk():.1f} GB")
    clean = R.ld()
    print(f"clean LD {clean.mean():.3f}, {np.mean(clean > 0):.0%} > 0 ({timer()} s)")
    mass = R.end_rows()
    per_head = np.zeros((len(MODES), L, H, len(R.pairs)))
    for m, mode in enumerate(MODES):
        for l in range(L):
            if test and l != 9:
                continue
            for h in range(H):
                per_head[m, l, h] = R.ld([(l, h)], "names", mode) - clean
            print(f"per head, {mode}, layer {l} done ({timer()} s)")
    C.check_disk()
    sets = [NAME_MOVERS] + random_sets(L, H)[: 2 if test else None]
    # set effects: [set, condition, prompt]
    conditions = [("names", "keep_sink"), ("names", "plain"), ("names", "zero"),
                  ("sink", "plain"), ("content", "keep_sink"), ("content", "plain")]
    per_set = np.zeros((len(sets), len(conditions), len(R.pairs)))
    for k, hs in enumerate(sets):
        for c, (which, mode) in enumerate(conditions):
            per_set[k, c] = R.ld(hs, which, mode) - clean
        print(f"set {k} {hs} done ({timer()} s)")
    np.savez_compressed(NPZ if not test else NPZ.with_suffix(".test.npz"), clean=clean, mass=mass, per_head=per_head, per_set=per_set,
                        sets=np.array(sets), modes=np.array(list(MODES)),
                        conditions=np.array([f"{a}/{b}" for a, b in conditions]),
                        runtime_s=timer())
    path = NPZ if not test else NPZ.with_suffix(".test.npz")
    print(f"saved {path} ({timer()} s)")
    return dict(np.load(path, allow_pickle=False))


# ── analysis ─────────────────────────────────────────────────────────────────

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(d) -> dict:
    clean, mass, per_head, per_set = d["clean"], d["mass"], d["per_head"], d["per_set"]
    sets = [list(map(tuple, s)) for s in d["sets"]]
    modes = list(d["modes"])
    n = len(clean)
    idx = C.boot_idx(n)
    L, H = per_head.shape[1:3]
    out = {"n_prompts": n, "clean_ld": C.mean_ci(clean, idx), "runtime_s": float(d["runtime_s"])}

    # per head
    eff = per_head.mean(-1)                                            # [mode, L, H]
    boot = per_head[..., idx].mean(-1)                                 # [mode, L, H, n_boot]
    lo, hi = np.quantile(boot, 0.025, axis=-1), np.quantile(boot, 0.975, axis=-1)
    sig = (lo > 0) | (hi < 0)
    m_names = mass[..., 0].mean(0)                                     # [L, H]
    heads = [(l, h) for l in range(L) for h in range(H)]
    flat = {m: eff[i].reshape(-1) for i, m in enumerate(modes)}
    top = {m: [heads[j] for j in np.argsort(flat[m])[:10]] for m in modes}   # most negative
    ks, pl = modes.index("keep_sink"), modes.index("plain")
    opposite = [(l, h) for l, h in heads
                if sig[ks, l, h] and sig[pl, l, h] and np.sign(eff[ks, l, h]) != np.sign(eff[pl, l, h])]
    only_one = [(l, h) for l, h in heads if sig[ks, l, h] != sig[pl, l, h]]
    big = [(l, h) for l, h in heads if max(abs(eff[ks, l, h]), abs(eff[pl, l, h])) >= 0.1]
    ratio = [eff[pl, l, h] / eff[ks, l, h] for l, h in big if abs(eff[ks, l, h]) > 1e-9]
    out["per_head"] = {
        "effect": {m: eff[i].round(4).tolist() for i, m in enumerate(modes)},
        "ci_low": {m: lo[i].round(4).tolist() for i, m in enumerate(modes)},
        "ci_high": {m: hi[i].round(4).tolist() for i, m in enumerate(modes)},
        "removed_mass": m_names.round(4).tolist(),
        "spearman": {f"{a}~{b}": spearman(flat[a], flat[b])
                     for a, b in (("keep_sink", "plain"), ("keep_sink", "zero"), ("plain", "zero"))},
        "top10_most_negative": {m: top[m] for m in modes},
        "top10_overlap_keep_sink_plain": len(set(top["keep_sink"]) & set(top["plain"])),
        "n_significant": {m: int(sig[i].sum()) for i, m in enumerate(modes)},
        "opposite_sign_both_significant": opposite,
        "significant_in_one_only": len(only_one),
        "heads_with_effect_ge_0.1": len(big),
        "plain_over_keep_sink_median": float(np.median(ratio)) if ratio else None,
    }

    # head sets
    conds = list(d["conditions"])
    m_set = np.zeros((len(sets), n, 3))                                # mean over the set's heads
    for k, hs in enumerate(sets):
        m_set[k] = np.mean([mass[:, l, h] for l, h in hs], axis=0)
    col = {"names": 0, "sink": 1, "content": 2}
    res = {}
    for c, name in enumerate(conds):
        which = name.split("/")[0]
        nm = per_set[0, c]
        rnd = per_set[1:, c].mean(0)                                   # mean over random sets
        mnm, mrnd = m_set[0, :, col[which]], m_set[1:, :, col[which]].mean(0)
        res[name] = {
            "name_movers": {"effect": C.mean_ci(nm, idx), "mass": C.mean_ci(mnm, idx),
                            "per_unit_mass": C.ratio_ci(nm, mnm, idx)},
            "random_sets": {"effect": C.mean_ci(rnd, idx), "mass": C.mean_ci(mrnd, idx),
                            "per_unit_mass": C.ratio_ci(rnd, mrnd, idx),
                            "effect_by_set": per_set[1:, c].mean(1).round(4).tolist()},
        }
    # paired differences between the renormalisation choices (name movers)
    i_ks, i_pl, i_z = conds.index("names/keep_sink"), conds.index("names/plain"), conds.index("names/zero")
    res["name_movers_plain_minus_keep_sink"] = C.mean_ci(per_set[0, i_pl] - per_set[0, i_ks], idx)
    res["name_movers_zero_minus_keep_sink"] = C.mean_ci(per_set[0, i_z] - per_set[0, i_ks], idx)
    # sink edge against content edges, per unit mass, in the random sets
    i_sk, i_cp = conds.index("sink/plain"), conds.index("content/plain")
    rs_sink = per_set[1:, i_sk].mean(0)
    rs_cont = per_set[1:, i_cp].mean(0)
    ms, mc = m_set[1:, :, 1].mean(0), m_set[1:, :, 2].mean(0)
    bs = (rs_sink[idx].mean(1) / ms[idx].mean(1)) / (rs_cont[idx].mean(1) / mc[idx].mean(1))
    res["random_sets_sink_over_content_per_unit_mass"] = (
        float((rs_sink.mean() / ms.mean()) / (rs_cont.mean() / mc.mean())),
        float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    out["sets"] = {"name_movers": NAME_MOVERS, "random": sets[1:], "results": res}

    # where the removed mass goes: the name movers' END rows (mean over heads and prompts)
    rows = {}
    m = np.stack([mass[:, l, h, 0] for l, h in NAME_MOVERS])          # names
    s = np.stack([mass[:, l, h, 1] for l, h in NAME_MOVERS])          # first token
    c = 1.0 - m - s                                                    # other keys
    rows["clean"] = [m.mean(), s.mean(), c.mean()]
    rows["keep_sink"] = [0.0, s.mean(), (1 - s).mean()]
    rows["plain"] = [0.0, (s / (1 - m)).mean(), (c / (1 - m)).mean()]
    rows["zero"] = [0.0, s.mean(), c.mean()]
    out["name_mover_rows"] = {k: [float(v) for v in vals] for k, vals in rows.items()}
    out["name_mover_keep_sink_factor_median"] = {
        f"{l}.{h}": float(np.median((1 - mass[:, l, h, 1]) / (1 - mass[:, l, h, 0] - mass[:, l, h, 1])))
        for l, h in NAME_MOVERS}
    out["name_mover_single_heads"] = {
        f"{l}.{h}": {mo: [float(eff[k, l, h]), float(lo[k, l, h]), float(hi[k, l, h])]
                     for k, mo in enumerate(modes)} for l, h in NAME_MOVERS}
    return out


# ── figure ───────────────────────────────────────────────────────────────────

def plot(d, a) -> None:
    plt = C.setup_style()
    modes = list(d["modes"])
    eff = np.array([a["per_head"]["effect"][m] for m in modes])         # [mode, L, H]
    ks, pl = modes.index("keep_sink"), modes.index("plain")
    fig = plt.figure(figsize=(9.0, 3.6))
    ax1 = fig.add_axes([0.065, 0.22, 0.25, 0.68])
    x, y = eff[ks].reshape(-1), eff[pl].reshape(-1)
    lim = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax1.plot(lim, lim, color=GREY_LINE, lw=0.8, zorder=0)
    ax1.axhline(0, color=C.GRID, lw=0.8, zorder=0)
    ax1.axvline(0, color=C.GRID, lw=0.8, zorder=0)
    nm = [l * 12 + h for l, h in NAME_MOVERS]
    other = [i for i in range(len(x)) if i not in nm]
    ax1.scatter(x[other], y[other], s=14, facecolor="white", edgecolor=C.INK, linewidths=0.6)
    ax1.scatter(x[nm], y[nm], s=30, facecolor=C.ORANGE, edgecolor=C.INK, linewidths=0.6)
    for (l, h), i in zip(NAME_MOVERS, nm):
        ax1.annotate(f"{l}.{h}", (x[i], y[i]), xytext=(4, -3), textcoords="offset points",
                     fontsize=8)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel("ΔLD, keep_sink=True")
    ax1.set_ylabel("ΔLD, keep_sink=False")
    ax1.set_title("(a) one head at a time", loc="left")

    ax2 = fig.add_axes([0.40, 0.22, 0.28, 0.68])
    res = a["sets"]["results"]
    rows = [("names/keep_sink", "keep_sink"), ("names/plain", "plain"), ("names/zero", "zero")]
    colors = [C.ORANGE, C.BLUE, C.GREY]
    width = 0.25
    for g, grp in enumerate(("name_movers", "random_sets")):
        for r, (key, label) in enumerate(rows):
            m, lo, hi = res[key][grp]["per_unit_mass"]
            xpos = g + (r - 1) * width
            ax2.bar(xpos, m, width * 0.9, color=colors[r], edgecolor=C.INK, linewidth=0.5,
                    label=label if g == 0 else None)
            ax2.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], color=C.INK, lw=0.8, capsize=2)
    ax2.axhline(0, color=C.INK, lw=0.6)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["name movers", f"{N_RANDOM_SETS} random triples"])
    ax2.set_ylabel("ΔLD per unit of removed mass")
    ax2.set_title("(b) three heads at once", loc="left")
    ax2.legend(loc="lower right", fontsize=8)

    ax3 = fig.add_axes([0.80, 0.22, 0.18, 0.68])
    comp = a["name_mover_rows"]
    order = ["clean", "keep_sink", "plain", "zero"]
    seg_colors = [C.PURPLE, C.MID, C.SKY]
    seg_labels = ["IO, S1, S2", "first token", "other keys"]
    for i, k in enumerate(order):
        left = 0.0
        for v, col, lab in zip(comp[k], seg_colors, seg_labels):
            ax3.barh(i, v, left=left, color=col, edgecolor=C.INK, linewidth=0.4, height=0.6,
                     label=lab if i == 0 else None)
            left += v
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels(order)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("END row, name movers")
    ax3.set_title("(c) where the mass goes", loc="left")
    h3, l3 = ax3.get_legend_handles_labels()
    fig.legend(h3, l3, loc="lower right", bbox_to_anchor=(0.995, 0.0), fontsize=8, ncol=3,
               handlelength=1.0, columnspacing=1.0)
    C.save_figure(fig, "sink_ablation.png")


GREY_LINE = "#BDBDBD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="only analyse and plot saved outputs")
    ap.add_argument("--test", action="store_true", help="code check on a few prompts")
    args = ap.parse_args()
    d = dict(np.load(NPZ, allow_pickle=False)) if args.plot else run(args.test)
    a = analyse(d)
    C.save_json("sink_ablation.json", a)
    plot(d, a)
    ph, res = a["per_head"], a["sets"]["results"]
    print("clean LD", C.fmt(a["clean_ld"]))
    print("spearman", {k: round(v, 3) for k, v in ph["spearman"].items()})
    print("top10", ph["top10_most_negative"])
    print("top10 overlap keep_sink/plain", ph["top10_overlap_keep_sink_plain"])
    print("significant", ph["n_significant"], "opposite", ph["opposite_sign_both_significant"],
          "one only", ph["significant_in_one_only"], "plain/keep median",
          ph["plain_over_keep_sink_median"])
    for k, v in res.items():
        if isinstance(v, dict):
            for g in ("name_movers", "random_sets"):
                print(k, g, "effect", C.fmt(v[g]["effect"]), "mass", C.fmt(v[g]["mass"], 3),
                      "per mass", C.fmt(v[g]["per_unit_mass"]))
        else:
            print(k, C.fmt(v))


if __name__ == "__main__":
    main()
