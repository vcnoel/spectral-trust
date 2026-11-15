# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Pilot: which words make Phi-3-mini's later tokens leave the sink star?

Actives "The {subject} {verb} {prep} the {object}." for 12 prepositions in
8 frames, and the English possessive ("the {subject}'s {object}"). Per
sentence: λ2 at layers 3-5 (library construction, ``run_with_spectra``)
and sink mass (attention to position 0, mean over queries 1..).

    USE_CASE_DEVICE=cuda python examples/use_cases/passive_trigger_probe.py
"""
import _common as C
import numpy as np
import torch

import spectral_trust as st

MODEL = "microsoft/Phi-3-mini-4k-instruct"
PREPS = ("by", "near", "with", "from", "for", "to", "under", "behind", "after", "of",
         "beside", "at")
FRAMES = [("cat", "slept", "door"), ("man", "waited", "window"), ("dog", "sat", "fire"),
          ("woman", "stood", "gate"), ("boat", "drifted", "shore"), ("girl", "read", "lamp"),
          ("soldier", "rested", "wall"), ("bird", "landed", "pond")]
LAYERS = (3, 4, 5)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm = st.SpectralModel.from_pretrained(MODEL, dtype="bfloat16", device_map=device)
    texts, labels = [], []
    for p in PREPS:
        for s, v, o in FRAMES:
            texts.append(f"The {s} {v} {p} the {o}.")
            labels.append(p)
    for s, v, o in FRAMES:
        texts.append(f"The {s}'s friend {v} near the {o}.")
        labels.append("possessive 's")
    cache = sm.run_with_spectra(texts, metrics=["graph.fiedler_value"], hybrid=False,
                                layers=list(LAYERS), batch_size=1)
    lam = np.stack([np.asarray(cache[f"blocks.{l}.graph.fiedler_value"]) for l in LAYERS], 1)
    sink = []
    for t in texts:
        ids = torch.tensor([sm.tokenizer(t)["input_ids"]], device=sm.device)
        with torch.no_grad():
            att = sm.model(input_ids=ids, output_attentions=True).attentions
        sink.append([float(att[l][0].double().mean(0)[1:, 0].mean()) for l in LAYERS])
    sink = np.array(sink)
    out = {"model": MODEL, "device": device, "dtype": "bfloat16", "layers": LAYERS,
           "texts": texts, "labels": labels, "lam2": lam, "sink": sink, "by_word": {}}
    for lab in dict.fromkeys(labels):
        k = [i for i, x in enumerate(labels) if x == lab]
        out["by_word"][lab] = {"lam2_mean": lam[k].mean(0).tolist(),
                               "lam2_min": lam[k].min(0).tolist(),
                               "sink_mean": sink[k].mean(0).tolist()}
        print(f"{lab:14s}", np.round(lam[k].mean(0), 3), np.round(sink[k].mean(0), 3), flush=True)
    C.save_json("passive_trigger_probe.json", out)


if __name__ == "__main__":
    main()
