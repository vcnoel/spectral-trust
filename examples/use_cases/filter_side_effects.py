# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Pilot: side effects of ``st.graph_filter`` against matched controls.

Edit (``_filter_edit.Edit``): ``X' = X + s (g(L) X - X)`` on the residual
stream after block l, causal, tokens 1.. (the library's
``iv.resid(l).graph_filter(kind, ..., causal=True, token_span=(1, None))``,
equal to it bit for bit). Filters: low-pass and high-pass at cutoff 1.0,
band-pass [0.5, 1.5); strengths 0.5 and 1. Controls of the same size per
item (Frobenius norm of the edit over tokens 1..): the same filter on a
row-shuffled attention graph (``random``), on the chain graph (``path``),
and the residual shrunk (``scale``, ``-x``).

Measured per condition: on 100 WikiText-2 test passages of 64 tokens,
next-token KL from the unedited run at every position, top-1 agreement,
NLL, the change of the final hidden state per position, and λ2 and sink
mass at every layer; on the 144 sentences of the 72 English active/passive
pairs, KL, top-1 and NLL; greedy 30-token continuations of 20 other
passages' first 24 tokens (``_filter_metrics.gen_stats``).

    USE_CASE_DEVICE=cuda python examples/use_cases/filter_side_effects.py --model gpt2
"""
import argparse
import json

import _common as C
import numpy as np
import torch

import spectral_trust as st
import _filter_metrics as M
from _filter_edit import CONDITIONS, FILTERS, Edit, shuffle_index
from passive_data import load as load_items

SETUP = {"gpt2": ("float32", (2, 6, 10)),
         "microsoft/Phi-3-mini-4k-instruct": ("bfloat16", (2, 12, 24))}
STRENGTHS = (0.5, 1.0)
N_WIKI, WLEN, N_GEN, PLEN, NEW, BATCH = 100, 64, 20, 24, 30, 25


class Runner:
    def __init__(self, name, device, path=None):
        dtype = SETUP[name][0]
        sm = st.SpectralModel.from_pretrained(path or name, dtype=dtype, device_map=device)
        self.m, self.tok, self.dev = sm.model.eval(), sm.tokenizer, sm.device
        rows = M.wiki_rows(self.tok, N_WIKI + N_GEN, WLEN)
        self.prompts = rows[N_WIKI:, :PLEN]
        items = [it for it in load_items() if it["group"] == "en_pair"]
        enc = [self.tok(it["text"])["input_ids"] for it in items]
        groups = {}
        for i, e in enumerate(enc):
            groups.setdefault(len(e), []).append(i)
        self.n_pas = len(enc)
        self.batches = [("wiki", list(range(a, a + BATCH)), rows[a:a + BATCH])
                        for a in range(0, N_WIKI, BATCH)]
        self.batches += [("pas", idx, torch.tensor([enc[i] for i in idx]))
                         for _, idx in sorted(groups.items())]
        self.lp, self.hid = {}, {}
        self.base = self.evaluate(None)
        self.gen0 = M.greedy(self.m, self.prompts.to(self.dev), NEW)

    @torch.no_grad()
    def evaluate(self, edit):
        r = {"wiki_nll": np.zeros(N_WIKI), "pas_nll": np.zeros(self.n_pas),
             "wiki_kl": np.zeros((N_WIKI, WLEN)), "wiki_top": np.zeros((N_WIKI, WLEN)),
             "wiki_hid": np.zeros((N_WIKI, WLEN)), "pas_kl": np.zeros(self.n_pas),
             "pas_top": np.zeros(self.n_pas)}
        lam, sink, rec = [], [], []
        for b, (kind, idx, ids) in enumerate(self.batches):
            wiki = kind == "wiki"
            if edit is not None:
                seeds = [(0 if wiki else 10_000) + i for i in idx]
                edit.P = shuffle_index(len(idx), ids.shape[1] - 1, seeds)
                edit.rec = [] if wiki else None
            o = self.m(input_ids=ids.to(self.dev), output_attentions=wiki,
                       output_hidden_states=wiki, use_cache=False)
            if wiki:
                la, si = M.lam2_sink(o.attentions)
                lam.append(la)
                sink.append(si)
            h = o.hidden_states[-1].float() if wiki else None
            if edit is None:
                self.lp[b] = o.logits.float().log_softmax(-1).cpu()
                tgt = ids[:, 1:].unsqueeze(-1)
                nll = -self.lp[b][:, :-1].gather(-1, tgt).squeeze(-1).mean(-1).numpy()
                if wiki:
                    self.hid[b] = h.cpu()
                r[f"{kind}_nll"][idx] = nll
                continue
            kl, top, _, ne = M.token_metrics(self.lp[b], o.logits, ids)
            r[f"{kind}_nll"][idx] = ne
            if wiki:
                r["wiki_kl"][idx], r["wiki_top"][idx] = kl, top
                hc = self.hid[b]
                r["wiki_hid"][idx] = ((h.cpu() - hc).norm(dim=-1) / hc.norm(dim=-1)).numpy()
                rec += edit.rec
            else:
                r["pas_kl"][idx], r["pas_top"][idx] = kl[:, 1:].mean(1), top[:, 1:].mean(1)
        r["wiki_lam"], r["wiki_sink"] = np.concatenate(lam), np.concatenate(sink)
        if rec:
            for k in ("pos", "size", "normchg"):
                r[f"edit_{k}"] = np.concatenate([x[k] for x in rec])
        return r


def run(name, device, path=None):
    tag = name.split("/")[-1]
    npz = C.OUTPUTS / f"filter_side_effects_{tag}.npz"
    gfile = C.OUTPUTS / f"filter_side_effects_{tag}_generations.json"
    done = dict(np.load(npz)) if npz.exists() else {}
    gens = json.load(open(gfile, encoding="utf-8")) if gfile.exists() else {}
    timer = C.Timer()
    R = Runner(name, device, path)
    print(C.library_info(), device, SETUP[name][0], f"disk {C.check_disk():.1f} GB",
          f"setup {timer()} s", flush=True)
    for k, v in R.base.items():
        done[f"clean|{k}"] = v
    gens["clean"] = M.gen_stats(R.tok, R.gen0, R.gen0)
    layers = SETUP[name][1]
    ids = R.batches[0][2].to(R.dev)
    with torch.no_grad(), st.intervene(R.m) as iv:
        iv.resid(layers[1]).graph_filter("lowpass", cutoff=1.0, strength=0.5, causal=True,
                                         token_span=(1, None))
        a = R.m(input_ids=ids).logits
    with torch.no_grad(), Edit(R.m, layers[1], "lowpass", 0.5, "attention"):
        b = R.m(input_ids=ids).logits
    done["check_library_vs_hook"] = np.array(float((a - b).abs().max()))
    print("library vs hook", float(done["check_library_vs_hook"]), flush=True)
    for l in layers:
        for f in FILTERS:
            for s in STRENGTHS:
                for c in CONDITIONS:
                    key = f"{l}|{f}|{s}|{c}"
                    if f"{key}|wiki_kl" in done and key in gens:
                        continue
                    C.wait_cool()
                    with Edit(R.m, l, f, s, c) as e:
                        r = R.evaluate(e)
                        e.P, e.rec = shuffle_index(N_GEN, PLEN + NEW - 1,
                                                   [20_000 + i for i in range(N_GEN)]), None
                        g = M.greedy(R.m, R.prompts.to(R.dev), NEW)
                    for k, v in r.items():
                        done[f"{key}|{k}"] = v
                    gens[key] = M.gen_stats(R.tok, g, R.gen0)
                    done["runtime_s"] = np.array(timer())
                    np.savez(npz, **done)
                    with open(gfile, "w", encoding="utf-8") as fh:
                        json.dump(gens, fh, ensure_ascii=False)
            print(f"layer {l} {f} done ({timer()} s)", flush=True)
    meta = dict(model=name, device=device, dtype=SETUP[name][0], layers=layers,
                filters={k: {kk: list(vv) if isinstance(vv, tuple) else vv
                             for kk, vv in v.items()} for k, v in FILTERS.items()},
                strengths=STRENGTHS, conditions=CONDITIONS, library=C.library_info(),
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                n_wiki=N_WIKI, wiki_len=WLEN, n_gen=N_GEN, prompt_len=PLEN, new_tokens=NEW)
    with open(npz.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(f"saved {npz} ({timer()} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--path", default=None, help="local folder of the weights")
    args = ap.parse_args()
    run(args.model, "cuda" if torch.cuda.is_available() else "cpu", args.path)
