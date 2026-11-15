# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Use case 2: graph-frequency steering, layer by layer, against matched controls.

GPT-2 small. At the residual stream after block l (l = 0..11) a low-pass or a
high-pass graph filter (cutoff 1.0 on the ``"sym"`` Laplacian, spectrum in
[0, 2]) is applied along the token axis on that layer's attention graph, at
strengths 0.25, 0.5 and 1:

    iv.resid(l).graph_filter(kind, cutoff=1.0, strength=s, causal=True, token_span=(1, None))

``causal=True``: each position is filtered on the graph of its own prefix, so
no position reads a later token (a symmetric filter would leak the next
token into the next-token loss). ``token_span=(1, None)``: the first token,
whose residual norm in GPT-2 is far larger than the others', is left out of
the graph and unedited.

Controls, per item and per condition, all causal and on the same tokens:

- ``path``: the same filter on the chain graph over positions (a purely
  positional basis, DCT-like), strength set per item so that the edit has the
  same Frobenius norm as the attention-graph edit;
- ``shuffled``: the same filter on a shuffled attention graph, in which each
  query row's attention weights are permuted at random among the keys of its
  own prefix (the row's weights are kept, which key gets which weight is not;
  seeded per item), symmetrised and averaged over heads as the attention graph
  is; norm-matched in the same way;
- ``mean`` (low-pass only): each position moved towards the mean of its
  prefix, ``x_i + s (mean_{j<=i} x_j - x_i)``, at the same strength (not
  norm-matched), the baseline for a low-pass on a graph dominated by one mode.

A difference between the attention graph and the norm-matched controls is due
to which tokens the graph links, not to the size of the edit.

Metrics: the IOI logit difference LD = logit(IO) - logit(S) at END (200
prompts of the official dataset code) as the effect, and the mean next-token
loss of 200 WikiText-2 test passages of 32 tokens (the first 200 lines that
are not headings and have at least 32 GPT-2 tokens) as the side effect.
Differences from the unedited model and between conditions, with 95%
percentile bootstrap intervals over prompts and passages (10,000 resamples).

    python examples/use_cases/graph_filter_steering.py           # run and plot (about 1.5 h, 2 threads)
    python examples/use_cases/graph_filter_steering.py --plot    # from outputs/graph_filter_steering.npz
"""
import argparse

import _common as C  # noqa: F401  (CPU, 2 threads, offline; before torch)
import numpy as np
import torch

import spectral_trust as st

KINDS = ("lowpass", "highpass")
STRENGTHS = (0.25, 0.5, 1.0)
CONDITIONS = ("attention", "path", "shuffled", "mean")
CUTOFF = 1.0
MIN_SIZE = 0.01             # cells whose attention edit is smaller are left out of the counts
N_WIKI, WIKI_LEN = 200, 32
WIKI_BATCH = 50
NPZ = C.OUTPUTS / "graph_filter_steering.npz"


def wikitext_passages(tok, n=N_WIKI, length=WIKI_LEN) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    rows = []
    for line in ds["text"]:
        s = line.strip()
        if not s or s.startswith("="):
            continue
        ids = tok(s)["input_ids"]
        if len(ids) >= length:
            rows.append(ids[:length])
        if len(rows) == n:
            break
    return torch.tensor(rows, dtype=torch.long)


def row_shuffle_index(n_items: int, S: int, seed_items) -> torch.Tensor:
    """``[B, S, S]`` gather index: row r holds a random permutation of the keys
    0..r (and the identity beyond r), seeded per item."""
    P = torch.arange(S).repeat(n_items, S, 1)
    for b, item in enumerate(seed_items):
        rng = np.random.default_rng([1, int(item)])
        for r in range(1, S):
            P[b, r, : r + 1] = torch.from_numpy(rng.permutation(r + 1))
    return P


def prefix_mean(x: torch.Tensor) -> torch.Tensor:
    """``m_i = mean_{j <= i} x_j`` along dim 1."""
    n = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device).view(1, -1, 1)
    return x.cumsum(dim=1) / n


class Steering:
    def __init__(self):
        self.sm = st.SpectralModel.from_pretrained("gpt2")
        self.hf = self.sm.model.eval()
        self.tok = self.sm.tokenizer
        self.gc = st.GraphConstructor(self.sm.config)
        self.pairs = C.ioi_pairs()
        self.groups = C.by_length(self.pairs)
        self.wiki = wikitext_passages(self.tok)
        self.io = torch.tensor([p["io"] for p in self.pairs])
        self.s = torch.tensor([p["s"] for p in self.pairs])
        # batches: (kind, ids, item indices, row-shuffle index of the control graph)
        self.batches = []
        groups = [("ioi", torch.stack([self.pairs[i]["clean"] for i in idx]), idx)
                  for idx in self.groups.values()]
        for a in range(0, len(self.wiki), WIKI_BATCH):
            idx = list(range(a, min(a + WIKI_BATCH, len(self.wiki))))
            groups.append(("wiki", self.wiki[idx], idx))
        for kind, ids, idx in groups:
            offset = 0 if kind == "ioi" else 10_000
            perm = row_shuffle_index(len(idx), ids.shape[1] - 1, [offset + i for i in idx])
            self.batches.append((kind, ids, idx, perm))

    def _metrics(self, kind, ids, idx, logits):
        if kind == "ioi":
            r = torch.arange(len(idx))
            ii = torch.as_tensor(idx)
            lg = logits[:, -1]
            return (lg[r, self.io[ii]] - lg[r, self.s[ii]]).numpy()
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return nll.mean(-1).numpy()

    def evaluate(self, setup=None):
        """``(LD [200], loss [200])``; ``setup(iv, batch)`` registers the edits
        and returns raw torch hook handles to remove afterwards."""
        ld, loss = np.zeros(len(self.pairs)), np.zeros(len(self.wiki))
        for batch in self.batches:
            kind, ids, idx, _ = batch
            kw = {"logits_to_keep": 1} if kind == "ioi" else {}
            with torch.no_grad():
                if setup is None:
                    logits = self.hf(input_ids=ids, **kw).logits
                else:
                    with st.intervene(self.hf) as iv:
                        handles = setup(iv, batch) or []
                        try:
                            logits = self.hf(input_ids=ids, **kw).logits
                        finally:
                            for h in handles:
                                h.remove()
            (ld if kind == "ioi" else loss)[idx] = self._metrics(kind, ids, idx, logits)
        return ld, loss

    def shuffled_graph(self, batch):
        P = batch[3]
        gc = self.gc

        def graph(att, x):
            m = att.shape[-1]
            idx = P[:, :m, :m].unsqueeze(1).expand(-1, att.shape[1], -1, -1)
            A = att.to(torch.float64).gather(-1, idx)
            return gc.aggregate_heads(gc.symmetrize_attention(A))
        return graph

    def dirichlet(self, X, A):
        """Dirichlet energy tr(X^T L X) / tr(X^T X) per item on the symmetric
        attention graph of ``A`` (sym Laplacian), float64."""
        W = self.gc.aggregate_heads(self.gc.symmetrize_attention(A.double()))
        L = self.gc.construct_laplacian(W).double()
        X = X.double()
        return torch.einsum("bsd,bst,btd->b", X, L, X) / (X * X).sum((1, 2))

    def hook_setup(self, layer, kind, strength, condition, record=None):
        """A control condition (``"path"``, ``"shuffled"``, ``"mean"``), or
        ``"attention"`` computed in the hook, to check it against the library
        edit."""
        attn = self.hf.transformer.h[layer].attn
        steer = self

        def setup(iv, batch):
            got = {}

            def grab(module, args, out):
                got["A"] = out[1].detach()

            def fn(x):
                A = got.pop("A")
                xs, As = x[:, 1:].double(), A[:, :, 1:, 1:]
                d_att = st.graph_filter(xs, As, kind=kind, cutoff=CUTOFF, causal=True) - xs
                na = d_att.flatten(1).norm(dim=1)
                if condition == "attention":
                    d, scale = d_att, torch.ones_like(na)
                elif condition == "mean":
                    d = prefix_mean(xs) - xs
                    scale = torch.ones_like(na)
                else:
                    graph = "path" if condition == "path" else steer.shuffled_graph(batch)
                    d = st.graph_filter(xs, As, kind=kind, cutoff=CUTOFF, causal=True,
                                        graph=graph) - xs
                    scale = na / d.flatten(1).norm(dim=1).clamp_min(1e-12)
                y = xs + strength * scale.view(-1, 1, 1) * d
                if record is not None:
                    e0 = steer.dirichlet(xs, As)
                    record.append(dict(
                        rel_size=(strength * scale * d.flatten(1).norm(dim=1)
                                  / xs.flatten(1).norm(dim=1)).tolist(),
                        rel_size_att=(strength * na / xs.flatten(1).norm(dim=1)).tolist(),
                        energy=(steer.dirichlet(y, As) / e0).tolist(),
                        energy_att=(steer.dirichlet(xs + strength * d_att, As) / e0).tolist(),
                        scale=scale.tolist()))
                out = x.clone()
                out[:, 1:] = y.to(x.dtype)
                return out
            handle = attn.register_forward_hook(grab)
            iv.hook("resid_post", layer, fn)
            return [handle]
        return setup

    def library_setup(self, layer, kind, strength):
        def setup(iv, batch):
            iv.resid(layer).graph_filter(kind, cutoff=CUTOFF, strength=strength, causal=True,
                                         token_span=(1, None))
        return setup

    def low_mode_fraction(self):
        """Fraction of the graph's modes with eigenvalue below the cutoff, per
        layer (symmetric attention graph over tokens 1.., unedited runs)."""
        L = self.hf.config.n_layer
        frac = {"ioi": [[] for _ in range(L)], "wiki": [[] for _ in range(L)]}
        for kind, ids, idx, _ in self.batches:
            with torch.no_grad():
                att = self.hf(input_ids=ids, output_attentions=True, logits_to_keep=1).attentions
            for l in range(L):
                W = self.gc.aggregate_heads(self.gc.symmetrize_attention(att[l][:, :, 1:, 1:].double()))
                ev = torch.linalg.eigvalsh(self.gc.construct_laplacian(W).double())
                frac[kind][l] += (ev < CUTOFF).double().mean(-1).tolist()
        return {k: np.array(v).mean(1) for k, v in frac.items()}


def run(test=False) -> dict:
    timer = C.Timer()
    S = Steering()
    if test:
        S.batches = [b for b in S.batches if b[0] == "ioi"][:1] + [b for b in S.batches
                                                                     if b[0] == "wiki"][:1]
    print(C.library_info(), f"disk {C.check_disk():.1f} GB", flush=True)
    L = S.hf.config.n_layer
    layers = [6] if test else list(range(L))
    ld0, loss0 = S.evaluate()
    print(f"clean LD {ld0[ld0 != 0].mean():.3f}, loss {loss0[loss0 != 0].mean():.3f} ({timer()} s)",
          flush=True)
    frac = S.low_mode_fraction()
    print("fraction of modes below cutoff", {k: v.round(2).tolist() for k, v in frac.items()},
          flush=True)

    # check: the hook's attention path equals the library edit
    a = S.evaluate(S.library_setup(6, "lowpass", 0.5))
    b = S.evaluate(S.hook_setup(6, "lowpass", 0.5, "attention"))
    check = max(float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max()))
    print(f"library edit vs hook, max |diff| = {check:.2e}", flush=True)
    shape = (L, len(KINDS), len(STRENGTHS), len(CONDITIONS))
    ld = np.full(shape + (len(S.pairs),), np.nan)
    loss = np.full(shape + (len(S.wiki),), np.nan)
    rel_size, energy, scale = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    for l in layers:
        for j, kind in enumerate(KINDS):
            for i, s in enumerate(STRENGTHS):
                for c, cond in enumerate(CONDITIONS):
                    if cond == "mean" and kind != "lowpass":
                        continue
                    rec = []
                    setup = (S.library_setup(l, kind, s) if cond == "attention"
                             else S.hook_setup(l, kind, s, cond, record=rec))
                    ld[l, j, i, c], loss[l, j, i, c] = S.evaluate(setup)
                    if rec:
                        rel_size[l, j, i, c] = np.mean(sum((r["rel_size"] for r in rec), []))
                        energy[l, j, i, c] = np.median(sum((r["energy"] for r in rec), []))
                        scale[l, j, i, c] = np.median(sum((r["scale"] for r in rec), []))
                        if cond == "path":           # the attention edit, recorded alongside
                            rel_size[l, j, i, 0] = np.mean(sum((r["rel_size_att"] for r in rec), []))
                            energy[l, j, i, 0] = np.median(sum((r["energy_att"] for r in rec), []))
        print(f"layer {l} done ({timer()} s)", flush=True)
        C.check_disk()
    path = NPZ if not test else NPZ.with_suffix(".test.npz")
    np.savez_compressed(path, ld0=ld0, loss0=loss0, ld=ld, loss=loss, rel_size=rel_size,
                        energy=energy, scale=scale, frac_ioi=frac["ioi"], frac_wiki=frac["wiki"],
                        check=check, runtime_s=timer())
    print(f"saved {path} ({timer()} s)", flush=True)
    return dict(np.load(path))


# ── analysis ─────────────────────────────────────────────────────────────────

def analyse(d) -> dict:
    out = {"cutoff": CUTOFF, "strengths": STRENGTHS, "kinds": KINDS, "conditions": CONDITIONS,
           "check_library_vs_hook": float(d["check"]), "runtime_s": float(d["runtime_s"])}
    idx_ld, idx_loss = C.boot_idx(len(d["ld0"]), seed=0), C.boot_idx(len(d["loss0"]), seed=1)
    out["clean_ld"] = C.mean_ci(d["ld0"], idx_ld)
    out["clean_loss"] = C.mean_ci(d["loss0"], idx_loss)
    out["fraction_of_modes_below_cutoff"] = {"ioi": d["frac_ioi"].round(3).tolist(),
                                             "wiki": d["frac_wiki"].round(3).tolist()}
    L = d["ld"].shape[0]
    cells = []
    for l in range(L):
        for j, kind in enumerate(KINDS):
            for i, s in enumerate(STRENGTHS):
                if np.isnan(d["ld"][l, j, i, 0]).all():
                    continue
                row = {"layer": l, "kind": kind, "strength": s}
                for c, cond in enumerate(CONDITIONS):
                    if np.isnan(d["ld"][l, j, i, c]).all():
                        continue
                    row[cond] = {"rel_size": float(d["rel_size"][l, j, i, c]),
                                 "energy_ratio": float(d["energy"][l, j, i, c]),
                                 "scale": float(d["scale"][l, j, i, c])}
                    for metric, base, idx in (("ld", d["ld0"], idx_ld),
                                              ("loss", d["loss0"], idx_loss)):
                        x = d[metric][l, j, i, c]
                        row[cond][f"d{metric}"] = C.mean_ci(x - base, idx)
                        if c:
                            row[cond][f"attention_minus_this_{metric}"] = C.mean_ci(
                                d[metric][l, j, i, 0] - x, idx)
                cells.append(row)
    out["cells"] = cells

    # cells where the attention edit is not close to the identity (relative size >= 0.01;
    # the low-pass at layers 0 and 1 keeps every mode and changes nothing)
    live = [c for c in cells if c["attention"]["rel_size"] >= MIN_SIZE]

    def sign(c, cond, metric):
        lo, hi = c[cond][f"attention_minus_this_{metric}"][1:]
        return 1 if lo > 0 else (-1 if hi < 0 else 0)

    def key(c):
        return (c["layer"], c["kind"], c["strength"])
    out["n_cells"] = len(cells)
    out["n_live_cells"] = len(live)
    out["min_size"] = MIN_SIZE
    for metric in ("ld", "loss"):
        both = [c for c in live if sign(c, "path", metric) and sign(c, "shuffled", metric)]
        out[f"beyond_both_{metric}"] = [key(c) for c in both]
        out[f"beyond_path_{metric}"] = sum(bool(sign(c, "path", metric)) for c in live)
        out[f"beyond_shuffled_{metric}"] = sum(bool(sign(c, "shuffled", metric)) for c in live)
        out[f"above_both_{metric}"] = [key(c) for c in both
                                       if sign(c, "path", metric) > 0 and sign(c, "shuffled", metric) > 0]
        out[f"below_both_{metric}"] = [key(c) for c in both
                                       if sign(c, "path", metric) < 0 and sign(c, "shuffled", metric) < 0]
    return out


# ── figure ───────────────────────────────────────────────────────────────────

def plot(a) -> None:
    plt = C.setup_style()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("bo", [C.BLUE, "#FFFFFF", C.ORANGE])
    cells = {(c["layer"], c["kind"], c["strength"]): c for c in a["cells"]}
    L = 1 + max(c["layer"] for c in a["cells"])
    layers = list(range(L))
    colours = {"attention": C.ORANGE, "path": C.BLUE, "shuffled": C.GREY, "mean": C.GREEN}
    marks = {"attention": "o", "path": "s", "shuffled": "D", "mean": "^"}
    beyond = {tuple(k) for k in a["beyond_both_ld"]}
    fig = plt.figure(figsize=(9.0, 6.2))

    # (a), (b): effect on IOI, attention graph, layer x strength
    vals = np.array([[[cells[(l, k, s)]["attention"]["dld"][0] for s in STRENGTHS]
                      for k in KINDS] for l in layers])
    vmax = float(np.abs(vals).max())
    for j, kind in enumerate(KINDS):
        ax = fig.add_axes([0.065 + j * 0.17, 0.58, 0.115, 0.36])
        im = ax.imshow(vals[:, j, :], aspect="auto", origin="lower", cmap=cmap,
                       norm=TwoSlopeNorm(0.0, -vmax, vmax))
        for l in layers:
            for i, s in enumerate(STRENGTHS):
                if (l, kind, s) in beyond:
                    ax.plot(i, l, marker="o", ms=2.6, color=C.INK)
        ax.set_xticks(range(len(STRENGTHS)))
        ax.set_xticklabels([f"{s:g}" for s in STRENGTHS])
        ax.set_yticks(range(0, L, 2))
        ax.set_xlabel("strength")
        if j == 0:
            ax.set_ylabel("layer")
        ax.set_title(f"({'ab'[j]}) {kind}: ΔLD", loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = fig.add_axes([0.065 + 0.17 + 0.12, 0.58, 0.007, 0.36])
    fig.colorbar(im, cax=cax)

    # (c), (d): side effect per layer against the norm-matched controls
    for j, (kind, s) in enumerate((("lowpass", 1.0), ("highpass", 0.5))):
        ax = fig.add_axes([0.065 + j * 0.27, 0.08, 0.19, 0.36])
        for cond in ("attention", "path", "shuffled"):
            v = np.array([cells[(l, kind, s)][cond]["dloss"] for l in layers])
            ax.fill_between(layers, v[:, 1], v[:, 2], color=colours[cond], alpha=0.25, lw=0)
            ax.plot(layers, v[:, 0], color=colours[cond], marker=marks[cond], ms=3, lw=1.0,
                    markeredgecolor=C.INK, markeredgewidth=0.4, label=cond)
        ax.axhline(0, color=C.INK, lw=0.5)
        ax.set_xticks(range(0, L, 2))
        ax.set_xlabel("layer")
        if j == 0:
            ax.set_ylabel("Δ loss, WikiText (nats)")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("Δ loss (nats, log scale)")
        ax.set_title(f"({'cd'[j]}) {kind}, strength {s:g}: side effect", loc="left",
                     fontsize=9)
        if j == 0:
            ax.legend(loc="upper left", fontsize=7.5)

    # (e), (f): effect against side effect, strength 0 -> 0.25 -> 0.5 (-> 1)
    for j, (l, kind, strengths) in enumerate(((5, "lowpass", STRENGTHS),
                                              (11, "highpass", STRENGTHS[:2]))):
        ax = fig.add_axes([0.62 + j * 0.0, 0.58 - j * 0.5, 0.36, 0.36])
        for cond in CONDITIONS:
            if cond not in cells[(l, kind, strengths[0])]:
                continue
            pts = np.array([(0.0, 0.0)] + [(cells[(l, kind, s)][cond]["dloss"][0],
                                            cells[(l, kind, s)][cond]["dld"][0])
                                           for s in strengths])
            if cond == "mean":
                pts = pts[:3]                          # strength 1 is off the scale
            ax.plot(pts[:, 0], pts[:, 1], color=colours[cond], lw=1.0, marker=marks[cond],
                    ms=4, markeredgecolor=C.INK, markeredgewidth=0.4, label=cond)
        ax.axhline(0, color=C.INK, lw=0.5)
        ax.axvline(0, color=C.INK, lw=0.5)
        ax.set_xlabel("Δ loss, WikiText (side effect)")
        ax.set_ylabel("ΔLD (effect)")
        span = "strength 0 to 1 (mean: to 0.5)" if kind == "lowpass" else "strength 0 to 0.5"
        ax.set_title(f"({'ef'[j]}) layer {l} {kind}, {span}", loc="left", fontsize=9)
        ax.legend(loc="best", fontsize=7.5)
    C.save_figure(fig, "graph_filter_steering.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    d = dict(np.load(NPZ)) if args.plot else run(args.test)
    a = analyse(d)
    C.save_json("graph_filter_steering.json" if not args.test else "graph_filter_steering.test.json", a)
    if not args.test:
        plot(a)
    print("clean LD", C.fmt(a["clean_ld"]), "loss", C.fmt(a["clean_loss"], 3))
    print("check", a["check_library_vs_hook"])
    print("fraction below cutoff", a["fraction_of_modes_below_cutoff"])
    for c in a["cells"]:
        line = [f"L{c['layer']} {c['kind']} s={c['strength']}"]
        for cond in CONDITIONS:
            if cond not in c:
                continue
            e = c[cond]
            line.append(f"{cond}: size {e['rel_size']:.3f} E {e['energy_ratio']:.2f} "
                        f"dLD {C.fmt(e['dld'])} dloss {C.fmt(e['dloss'], 3)}")
        print(" | ".join(line))
    for m in ("ld", "loss"):
        print(m, "beyond both", len(a[f"beyond_both_{m}"]), "beyond path", a[f"beyond_path_{m}"],
              "beyond shuffled", a[f"beyond_shuffled_{m}"], "of", a["n_live_cells"],
              "| attention above both", len(a[f"above_both_{m}"]), "below both",
              len(a[f"below_both_{m}"]))


if __name__ == "__main__":
    main()
