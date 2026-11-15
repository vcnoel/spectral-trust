# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Sink dependence and output dimensionality of operator spectra on GPT-2.

Reproduces, qualitatively, the GPT-2 panels of Figure 5 of Viganò, Melzi and
Ovsjanikov (NeurIPS 2026): across the 144 heads of GPT-2 small, the Spearman
correlation of three diagnostics with the sink score and with the spectral
entropy of the head output H(AV):

- PR(AᵀA), the participation ratio in the counting geometry;
- PR(D), the participation ratio under the pullback measure;
- PR_cent(D), its centred variant.

The sink score of a head is its mean attention to the first token; H(AV) is
the entropy of the normalised squared singular values of A V, with V the
head's value states. Inputs: WikiText-103 test paragraphs of at least
--length tokens, truncated to --length.

    python examples/operator_spectra_sink.py --items 200 --length 128
"""
import argparse
import json

import pandas as pd
import torch
from scipy.stats import spearmanr

import spectral_trust as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--length", type=int, default=128)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]
    texts = [t.strip() for t in texts if not t.strip().startswith("=")]
    ids = [x for x in (tok(t).input_ids for t in texts) if len(x) >= args.length][:args.items]

    rows = []
    for item, x in enumerate(ids):
        vals = st.capture_attention_values(model, torch.tensor(x[:args.length])[None])
        for layer, av in sorted(vals.items()):
            A, V = av.weights[0].double(), av.values[0].double()      # [H, S, S], [H, S, d]
            sv = torch.linalg.svdvals(A @ V) ** 2
            p = sv / sv.sum(-1, keepdim=True)
            h_out = -(p * p.clamp_min(1e-300).log()).sum(-1)
            counting = st.operator_metrics(A, "counting")
            pullback = st.operator_metrics(A, normalize_rows=True)
            for h in range(A.shape[0]):
                rows.append(dict(item=item, layer=layer, head=h, sink=float(A[h, :, 0].mean()),
                                 h_out=float(h_out[h]),
                                 pr_count=float(counting["participation_ratio"][h]),
                                 pr=float(pullback["participation_ratio"][h]),
                                 pr_cent=float(pullback["participation_ratio_centered"][h])))
    df = pd.DataFrame(rows)
    heads = df.groupby(["layer", "head"]).mean(numeric_only=True)
    out = {}
    for name in ("pr_count", "pr", "pr_cent"):
        out[name] = {f"{target}_{level}": round(float(spearmanr(d[name], d[target])[0]), 3)
                     for target in ("sink", "h_out")
                     for level, d in (("points", df), ("heads", heads))}
    print(json.dumps({"items": len(ids), "length": args.length, **out}, indent=1))


if __name__ == "__main__":
    main()
