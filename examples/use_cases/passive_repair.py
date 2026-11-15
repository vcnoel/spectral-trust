# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Pilot: does a layer-2 intervention bring the passive sentences' λ2 at
layers 3-5 of Phi-3-mini back to the active level, and at what cost?

Interventions at layer 2, each with a matched random control:

- ``filter``: ``iv.resid(2).graph_filter("lowpass", cutoff=1.0, strength=s,
  causal=True, token_span=(1, None))`` (via ``_filter_edit.Edit``); control:
  the same filter on a row-shuffled attention graph, same edit norm.
- ``edges``: ``iv.attention(2).remove_edges(mask, keep_sink=True)`` on every
  head, removing the edges into the tokens "was", "were" and "by"; control:
  the same number of random keys (not position 0) per sequence.
- ``steer``: a mean-difference steering vector: ``alpha * v`` added to the
  residual after block 2, ``v`` the mean hidden state of 16 active minus 16
  passive calibration sentences (``CALIBRATION``); control: a random
  Gaussian direction of the same norm.

Measured on the 72 en_pair passives and actives (λ2 at layers 3-5 with the
library construction, mean next-token NLL, KL from the unedited run) and on
50 WikiText-2 passages of 64 tokens (KL, NLL).

    USE_CASE_DEVICE=cuda python examples/use_cases/passive_repair.py
"""
import _common as C
import numpy as np
import torch

import spectral_trust as st
import _filter_metrics as M
from _filter_edit import Edit, shuffle_index
from passive_data import load as load_items

MODEL = "microsoft/Phi-3-mini-4k-instruct"
LAYER, MEASURE = 2, (3, 4, 5)
CALIBRATION = [
    ("The system processed the data.", "The data was processed by the system."),
    ("The server hosted the website.", "The website was hosted by the server."),
    ("The code compiled the program.", "The program was compiled by the code."),
    ("The layer encoded the input.", "The input was encoded by the layer."),
    ("The model predicted the token.", "The token was predicted by the model."),
    ("The node routed the packet.", "The packet was routed by the node."),
    ("The script executed the command.", "The command was executed by the script."),
    ("The database stored the record.", "The record was stored by the database."),
    ("The program calculated the sum.", "The sum was calculated by the program."),
    ("The analyzer parsed the text.", "The text was parsed by the analyzer."),
    ("The sensor detected the motion.", "The motion was detected by the sensor."),
    ("The filter blocked the spam.", "The spam was blocked by the filter."),
    ("The api returned the response.", "The response was returned by the api."),
    ("The drive saved the file.", "The file was saved by the drive."),
    ("The logic validated the user.", "The user was validated by the logic."),
    ("The screen displayed the image.", "The image was displayed by the screen."),
]
CONDITIONS = [("filter", 0.5), ("filter", 1.0), ("edges", None), ("steer", 0.2), ("steer", 1.0)]
N_WIKI, WLEN, BATCH = 50, 64, 25


def by_length(enc):
    g = {}
    for i, e in enumerate(enc):
        g.setdefault(len(e), []).append(i)
    return [(idx, torch.tensor([enc[i] for i in idx])) for _, idx in sorted(g.items())]


class Repair:
    def __init__(self, device):
        sm = st.SpectralModel.from_pretrained(MODEL, dtype="bfloat16", device_map=device)
        self.m, self.tok, self.dev = sm.model.eval(), sm.tokenizer, sm.device
        items = [it for it in load_items() if it["group"] == "en_pair"]
        self.sets = {v: by_length([self.tok(it["text"])["input_ids"] for it in items
                                   if it["voice"] == v]) for v in ("active", "passive")}
        rows = M.wiki_rows(self.tok, N_WIKI, WLEN)
        self.sets["wiki"] = [(list(range(a, a + BATCH)), rows[a:a + BATCH])
                             for a in range(0, N_WIKI, BATCH)]
        self.n = {k: sum(len(i) for i, _ in v) for k, v in self.sets.items()}
        self.fw = {i for i in range(len(self.tok))
                   if self.tok.decode([i]).strip().lower() in ("was", "were", "by")}
        self.v = self.steer_vector()
        self.lp = {}
        self.base = self.measure(None)

    @torch.no_grad()
    def steer_vector(self):
        """Mean hidden state after block 2 (padding masked), actives minus
        passives."""
        def mean_h(texts):
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.dev)
            h = self.m(**enc, output_hidden_states=True).hidden_states[LAYER + 1].float()
            mk = enc.attention_mask.unsqueeze(-1).float()
            return ((h * mk).sum(1) / mk.sum(1)).mean(0)
        return (mean_h([a for a, _ in CALIBRATION]) - mean_h([p for _, p in CALIBRATION])).cpu()

    def install(self, stack, cond, s, control, ids, seeds):
        if cond == "filter":
            e = stack.enter_context(Edit(self.m, LAYER, "lowpass", s,
                                         "random" if control else "attention"))
            e.P = shuffle_index(ids.shape[0], ids.shape[1] - 1, seeds)
            return
        iv = stack.enter_context(st.intervene(self.m))
        if cond == "steer":
            v = self.v
            if control:
                g = torch.Generator().manual_seed(11)
                r = torch.randn(v.shape, generator=g)
                v = r * v.norm() / r.norm()
            iv.resid(LAYER).add(v, scale=s)
            return
        B, S = ids.shape
        mask = torch.zeros(B, S, S, dtype=torch.bool)
        for b in range(B):
            keys = [k for k in range(1, S) if int(ids[b, k]) in self.fw]
            if control and keys:
                rng = np.random.default_rng([3, int(seeds[b])])
                keys = list(rng.choice(np.arange(1, S), size=len(keys), replace=False))
            mask[b, :, keys] = True
        iv.attention(LAYER).remove_edges(mask, keep_sink=True)

    @torch.no_grad()
    def measure(self, cond, s=None, control=False):
        """Per set: NLL [n], KL from the unedited run [n] and λ2 at MEASURE [n, 3]."""
        from contextlib import ExitStack
        out = {}
        for name, batches in self.sets.items():
            nll, kl = np.zeros(self.n[name]), np.zeros(self.n[name])
            lam = np.zeros((self.n[name], len(MEASURE)))
            for b, (idx, ids) in enumerate(batches):
                seeds = [{"active": 0, "passive": 1000, "wiki": 2000}[name] + i for i in idx]
                with ExitStack() as stack:
                    if cond is not None:
                        self.install(stack, cond, s, control, ids, seeds)
                    o = self.m(input_ids=ids.to(self.dev), output_attentions=True, use_cache=False)
                la, _ = M.lam2_sink([o.attentions[l] for l in MEASURE])
                lam[idx] = la
                lp = o.logits.float().log_softmax(-1).cpu()
                tgt = ids[:, 1:].unsqueeze(-1)
                nll[idx] = -lp[:, :-1].gather(-1, tgt).squeeze(-1).mean(-1).numpy()
                if cond is None:
                    self.lp[(name, b)] = lp
                else:
                    ref = self.lp[(name, b)]
                    kl[idx] = (ref.exp() * (ref - lp)).sum(-1)[:, 1:].mean(-1).numpy()
            out[name] = dict(nll=nll, kl=kl, lam=lam)
        return out


def summarise(R, r, idx):
    base = R.base
    act, pas = base["active"]["lam"].mean(0), base["passive"]["lam"].mean(0)
    edited = r["passive"]["lam"].mean(0)
    return {"passive_lam2": edited.tolist(),
            "recovery_fraction": ((edited - pas) / (act - pas)).tolist(),
            "passive_dnll": C.mean_ci(r["passive"]["nll"] - base["passive"]["nll"], idx["passive"]),
            "passive_kl": float(r["passive"]["kl"].mean()),
            "active_dlam2": (r["active"]["lam"].mean(0) - act).tolist(),
            "active_dnll": C.mean_ci(r["active"]["nll"] - base["active"]["nll"], idx["active"]),
            "active_kl": float(r["active"]["kl"].mean()),
            "wiki_dnll": C.mean_ci(r["wiki"]["nll"] - base["wiki"]["nll"], idx["wiki"]),
            "wiki_kl": C.mean_ci(r["wiki"]["kl"], idx["wiki"])}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timer = C.Timer()
    R = Repair(device)
    idx = {k: C.boot_idx(n, seed=5) for k, n in R.n.items()}
    b = R.base
    out = {"model": MODEL, "device": device, "dtype": "bfloat16", "library": C.library_info(),
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "layer": LAYER, "measured_layers": MEASURE, "n": R.n,
           "steer_vector_norm": float(R.v.norm()),
           "base": {"active_lam2": b["active"]["lam"].mean(0).tolist(),
                    "passive_lam2": b["passive"]["lam"].mean(0).tolist(),
                    "passive_minus_active_lam2": [C.mean_ci(b["passive"]["lam"][:, j]
                                                            - b["active"]["lam"][:, j], idx["passive"])
                                                  for j in range(len(MEASURE))],
                    "active_nll": float(b["active"]["nll"].mean()),
                    "passive_nll": float(b["passive"]["nll"].mean()),
                    "wiki_nll": float(b["wiki"]["nll"].mean())},
           "conditions": {}}
    for cond, s in CONDITIONS:
        for control in (False, True):
            key = f"{cond}|{s}|{'control' if control else 'edit'}"
            C.wait_cool()
            out["conditions"][key] = summarise(R, R.measure(cond, s, control), idx)
            print(key, {k: (round(v[0], 3) if isinstance(v[0], float) else v)
                        for k, v in out["conditions"][key].items() if isinstance(v, list)},
                  f"({timer()} s)", flush=True)
    out["runtime_s"] = timer()
    C.save_json("passive_repair.json", out)


if __name__ == "__main__":
    main()
