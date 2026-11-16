# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""Check two findings of Li (2026, arXiv 2607.09279) on GPT-2 small with
spectral_trust.pseudospectra (not with the author's code, which does not
cover that paper).

1. Proposition 1: the Kreiss constant of a causal row-stochastic matrix is
   pinned at sqrt(n) by the mask (11.314 at n = 128).
2. Table 2: on the deflated head-mean depth stack, the eigenvalue heuristic
   E_L = prod_l rho(A_l - P) underestimates the surviving deviation
   D_L = ||prod_l (A_l - P)||_2 by many orders of magnitude (GPT-2, depth 12:
   D = 5.1e-4, E = 1.8e-11, D/E = 2.9e7, medians over sequences).

Inputs follow the paper: 128-token rows, BOS-prefixed, float64 with rows
renormalised. The corpus is wikitext-103 test (the paper used pile-10k).

    PYTHONPATH=src python benchmarks/li_depth_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectral_trust import pseudospectra as ps

N, N_SEQ = 128, 8


def sequences(tok):
    root = Path(snapshot_download("Salesforce/wikitext", repo_type="dataset",
                                  local_files_only=True))
    text = pd.read_parquet(root / "wikitext-103-raw-v1" / "test-00000-of-00001.parquet")["text"]
    ids = tok(" ".join(t.strip() for t in text[:400] if t.strip()))["input_ids"]
    return [[tok.bos_token_id] + ids[i * (N - 1):(i + 1) * (N - 1)] for i in range(N_SEQ)]


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True,
                                                 attn_implementation="eager").eval()
    P = np.zeros((N, N))
    P[:, 0] = 1.0
    D, E, NB, kreiss = [], [], [], {}
    for k, seq in enumerate(sequences(tok)):
        with torch.no_grad():
            att = model(torch.tensor([seq]), output_attentions=True).attentions
        A = np.stack([a[0].double().numpy() for a in att])            # [L, H, n, n]
        A = A / A.sum(-1, keepdims=True)
        stack = A.mean(1)                                             # head-mean per layer
        prod = ps.depth_propagator(stack, deflate=True)
        D.append(np.linalg.norm(prod[-1], 2))
        dev = stack - P
        E.append(np.prod([np.abs(np.diag(m)).max() for m in dev]))
        NB.append(np.prod([np.linalg.norm(m, 2) for m in dev]))
        if k == 0:
            for (l, h) in ((0, 0), (5, 1), (11, 8)):
                kreiss[f"L{l}H{h}"] = ps.kreiss_constant(A[l, h], how="resolvent")
            full = ps.depth_propagator(stack)
            kreiss["prop2_max_abs_error"] = float(np.abs(full - P - prod).max())
    D, E, NB = map(np.asarray, (D, E, NB))
    out = {"n": N, "sequences": N_SEQ, "corpus": "wikitext-103-raw-v1 test",
           "D12_median": float(np.median(D)), "E12_median": float(np.median(E)),
           "D_over_E_median": float(np.median(D / E)), "N12_median": float(np.median(NB)),
           "li_table2_gpt2": {"D": 5.1e-4, "E": 1.8e-11, "D_over_E": 2.9e7},
           "kreiss_raw_heads": kreiss, "sqrt_n": float(np.sqrt(N))}
    print(json.dumps(out, indent=1))
    Path(__file__).with_name("li_depth_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
