# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Use case 4: frequency-resolved activation patching on IOI.

GPT-2 small, the 200 IOI prompts of the official dataset code and their ABC
counterparts (the IO name, both S names, then S1 alone replaced by random
names). For each layer l, the residual stream at the input of block l is
patched from the clean run into the corrupt run, one band of graph
frequencies at a time:

    X' = X_corrupt + U_b U_b^T (X_clean - X_corrupt)

with U the eigenvectors of a Laplacian over the prompt's tokens (ascending
eigenvalue) and U_b one of K = 8 equal-count bands. Bases: the clean run's
attention at layer l (``"clean_attention"``) and the chain graph over
positions (``"path"``, a positional control); optionally Haar-random bases
(``"random"``, the null for a subspace of the same size). Two positional
references: plain patching of every position (recovery 1 by construction)
and of the END position only.

Recovery of a patch: [mean LD_patched - mean LD_corrupt] / [mean LD_clean -
mean LD_corrupt], LD = logit(IO) - logit(S) at END with the clean prompt's
names; 95% percentile bootstrap intervals over prompts (10,000 resamples).

This script recomputes the clean-attention and path-graph maps of the
registered pilot reported on the docs page (which also ran the random
control and the corrupt-attention basis); with the defaults it is a check of
the library, not a new result.

    python examples/use_cases/frequency_patching.py                 # about 20 min, 2 threads
    python examples/use_cases/frequency_patching.py --n-random 10   # adds random bases
    python examples/use_cases/frequency_patching.py --plot
"""
import argparse

import _common as C  # noqa: F401  (CPU, 2 threads, offline; before torch)
import numpy as np
import torch

import spectral_trust as st

K = 8                       # bands per basis
NPZ = C.OUTPUTS / "frequency_patching.npz"


def specs_for(layer: int, T: int, n_random: int):
    """Patch specs of one layer, with their names (basis, band index, seed)."""
    bands = st.equal_count_bands(T, K)
    names, specs = [], []
    for basis in ("clean_attention", "path"):
        for b, band in enumerate(bands):
            names.append((basis, b, -1))
            specs.append(st.PatchSpec("resid_pre", layer, band=band, basis=basis))
    for seed in range(n_random):
        for b, band in enumerate(bands):
            names.append(("random", b, seed))
            specs.append(st.PatchSpec("resid_pre", layer, band=band, basis="random", seed=seed))
    names.append(("full", -1, -1))
    specs.append(st.PatchSpec("resid_pre", layer))
    names.append(("end_only", -1, -1))
    specs.append(st.PatchSpec("resid_pre", layer, positions=[T - 1]))
    return names, specs


def run(n_random: int) -> dict:
    timer = C.Timer()
    print(C.library_info(), f"disk {C.check_disk():.1f} GB")
    pairs = C.ioi_pairs()
    sm = st.SpectralModel.from_pretrained("gpt2")
    hf = sm.model.eval()
    n_layers = hf.config.n_layer
    n = len(pairs)
    ld_clean, ld_corrupt = np.zeros(n), np.zeros(n)
    patched = np.zeros((n_layers, 2 * K + n_random * K + 2, n))
    names_ref = None
    for T, idx in C.by_length(pairs).items():
        clean = torch.stack([pairs[i]["clean"] for i in idx])
        corrupt = torch.stack([pairs[i]["corrupt"] for i in idx])
        io = torch.tensor([pairs[i]["io"] for i in idx])
        s = torch.tensor([pairs[i]["s"] for i in idx])
        r = torch.arange(len(idx))

        def ld(logits):
            return logits[r, -1, io] - logits[r, -1, s]
        p = st.Patcher(hf, clean, corrupt, forward_kwargs={"logits_to_keep": 1})
        ld_clean[idx] = ld(p.clean_logits).numpy()
        ld_corrupt[idx] = ld(p.corrupt_logits).numpy()
        for layer in range(n_layers):
            names, specs = specs_for(layer, T, n_random)
            names_ref = names_ref or names
            patched[layer][:, idx] = p.patch_many(specs, metric=ld).numpy()
        print(f"length {T}: {len(idx)} prompts ({timer()} s)", flush=True)
    np.savez_compressed(NPZ, ld_clean=ld_clean, ld_corrupt=ld_corrupt, patched=patched,
                        names=np.array([f"{a}|{b}|{c}" for a, b, c in names_ref]),
                        n_random=n_random, runtime_s=timer())
    print(f"saved {NPZ} ({timer()} s)")
    return dict(np.load(NPZ))


def analyse(d) -> dict:
    names = [(a, int(b), int(c)) for a, b, c in (x.split("|") for x in d["names"])]
    clean, corrupt, patched = d["ld_clean"], d["ld_corrupt"], d["patched"]
    n_random = int(d["n_random"])
    n_layers = patched.shape[0]
    idx = C.boot_idx(len(clean))
    den = clean[idx].mean(1) - corrupt[idx].mean(1)

    def rec(x):
        m = (x.mean() - corrupt.mean()) / (clean.mean() - corrupt.mean())
        bs = (x[idx].mean(1) - corrupt[idx].mean(1)) / den
        return float(m), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    def row(name, band=-1, seed=-1):
        return names.index((name, band, seed))
    out = {"n_prompts": len(clean), "K": K, "n_random": n_random,
           "ld_clean": C.mean_ci(clean, idx), "ld_corrupt": C.mean_ci(corrupt, idx),
           "runtime_s": float(d["runtime_s"]), "layers": []}
    for l in range(n_layers):
        e = {"layer": l, "full": rec(patched[l, row("full")]),
             "end_only": rec(patched[l, row("end_only")])}
        for basis in ("clean_attention", "path"):
            e[basis] = [rec(patched[l, row(basis, b)]) for b in range(K)]
        e["sum_of_attention_bands"] = float(sum(x[0] for x in e["clean_attention"]))
        if n_random:
            rnd = np.array([[rec(patched[l, row("random", b, sd)])[0] for sd in range(n_random)]
                            for b in range(K)])
            e["random_p95"] = np.percentile(rnd, 95, axis=1).tolist()
        out["layers"].append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="only analyse saved outputs")
    ap.add_argument("--n-random", type=int, default=0)
    args = ap.parse_args()
    d = dict(np.load(NPZ)) if args.plot else run(args.n_random)
    a = analyse(d)
    C.save_json("frequency_patching.json", a)
    print("LD clean", C.fmt(a["ld_clean"]), "corrupt", C.fmt(a["ld_corrupt"]))
    for e in a["layers"]:
        print(e["layer"], "full", round(e["full"][0], 4), "END", C.fmt(e["end_only"]),
              "attn", [round(x[0], 2) for x in e["clean_attention"]],
              "path", [round(x[0], 2) for x in e["path"]],
              "sum attn", round(e["sum_of_attention_bands"], 2))


if __name__ == "__main__":
    main()
