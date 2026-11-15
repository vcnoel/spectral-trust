# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Pilot: the Fiedler value of spectral-trust 0.1.2 against the current
library, on the passive-voice drop of Phi-3-mini-4k-instruct.

spectral-trust 0.1.2 (``GSPDiagnosticsFramework.analyze_text`` with the
``GSPConfig`` defaults of that version) built the graph by per-head
symmetrisation ``(A + A^T) / 2`` and a uniform head average, formed the
random-walk Laplacian ``I - D^-1 W`` in float32 and passed it to
``scipy.linalg.eigh``. ``eigh`` reads only the lower triangle, so for this
non-symmetric matrix it returns the eigenvalues of a different, symmetrised
matrix; eigenvalues were then clipped at 0. Phi-3 adds no BOS token
(``add_bos_token: false``).

Here, on a reference pair (``ORIGINAL``, whose first words differ) and ten
further test pairs (``TEST_PAIRS``):

- ``lam2_arch``: that construction (``passive_drop.lam2_v012``, equal to the
  0.1.2 code to 3e-6 on GPT-2);
- ``lam2``: the current library (``SpectralModel.run_with_spectra``,
  ``graph.fiedler_value``: symmetric normalised Laplacian, ``"sym"``);
- ``lam2_rw``: the current library with ``normalization="rw"``, which now
  uses a general eigensolver (the random-walk and symmetric Laplacians are
  similar matrices, so this equals ``lam2``);
- ``sink``: attention received by position 0 (mean over queries 1..S-1).

Two precisions: float32 on the first 8 blocks (attention at layer l depends
only on blocks 0..l), and bfloat16 on the full model.

    USE_CASE_DEVICE=cuda python examples/use_cases/passive_reproduce.py
"""
import gc
from dataclasses import replace

import _common as C
import numpy as np
import torch

import spectral_trust as st
from passive_drop import lam2_v012, load_model

MODEL = "microsoft/Phi-3-mini-4k-instruct"
ORIGINAL = ("The scientist discovered a new particle.",
            "A new particle was discovered by the scientist.")
TEST_PAIRS = [
    ("The man ate the apple.", "The apple was eaten by the man."),
    ("The boy threw the ball.", "The ball was thrown by the boy."),
    ("The woman drove the car.", "The car was driven by the woman."),
    ("The dog chased the cat.", "The cat was chased by the dog."),
    ("The bird built the nest.", "The nest was built by the bird."),
    ("The router forwarded the packet.", "The packet was forwarded by the router."),
    ("The compiler optimized the code.", "The code was optimized by the compiler."),
    ("The server validated the request.", "The request was validated by the server."),
    ("The function returned the value.", "The value was returned by the function."),
    ("The browser rendered the page.", "The page was rendered by the browser."),
]
N_FIRST = 8


def measure(sm, texts, layers) -> dict:
    tok = sm.tokenizer
    res = {"lam2_arch": [], "sink": [], "tokens": []}
    for t in texts:
        ids = tok(t)["input_ids"]
        with torch.no_grad():
            o = sm.model(input_ids=torch.tensor([ids], device=sm.device), output_attentions=True,
                         use_cache=False)
        res["tokens"].append([tok.decode([i]) for i in ids])
        res["lam2_arch"].append([lam2_v012(o.attentions[l][0].double().cpu().numpy())
                                 for l in layers])
        res["sink"].append([float(o.attentions[l][0].double().mean(0)[1:, 0].mean())
                            for l in layers])
    cache = sm.run_with_spectra(texts, metrics=["graph.fiedler_value"], hybrid=False,
                                layers=layers, batch_size=1)
    res["lam2"] = np.stack([np.asarray(cache[f"blocks.{l}.graph.fiedler_value"]) for l in layers],
                           1).tolist()
    sm_rw = st.SpectralModel(sm.model, sm.tokenizer,
                             config=replace(sm.config, normalization="rw"))
    cache = sm_rw.run_with_spectra(texts, metrics=["graph.fiedler_value"], hybrid=False,
                                   layers=layers, batch_size=1)
    res["lam2_rw"] = np.stack([np.asarray(cache[f"blocks.{l}.graph.fiedler_value"])
                               for l in layers], 1).tolist()
    return res


def run(device: str, model: str = MODEL) -> dict:
    timer = C.Timer()
    pairs = [ORIGINAL] + TEST_PAIRS
    texts = [a for a, _ in pairs] + [p for _, p in pairs]
    out = {"model": model, "device": device, "library": C.library_info(),
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "texts": texts, "n_pairs": len(pairs), "layers": list(range(N_FIRST)),
           "runs": {}}
    for dtype, n_layers in (("float32", N_FIRST), ("bfloat16", None)):
        sm = load_model(model, dtype, device, n_layers)
        print(f"loaded {dtype} ({timer()} s)", flush=True)
        r = measure(sm, texts, list(range(N_FIRST)))
        r.update(dtype=dtype, n_blocks_loaded=n_layers or sm.n_layers)
        out["runs"][dtype] = r
        del sm
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    # summaries: means over actives / passives, and the original pair
    n = len(pairs)
    summ = {}
    for dtype, r in out["runs"].items():
        for key in ("lam2_arch", "lam2", "lam2_rw", "sink"):
            x = np.asarray(r[key])
            summ[f"{dtype}.{key}.original_active"] = x[0].tolist()
            summ[f"{dtype}.{key}.original_passive"] = x[n].tolist()
            summ[f"{dtype}.{key}.repair_pairs_active_mean"] = x[1:n].mean(0).tolist()
            summ[f"{dtype}.{key}.repair_pairs_passive_mean"] = x[n + 1:].mean(0).tolist()
        r["max_abs_lam2_minus_lam2_rw"] = float(np.abs(np.asarray(r["lam2"])
                                                       - np.asarray(r["lam2_rw"])).max())
    out["summary"] = summ
    out["runtime_s"] = timer()
    name = "passive_reproduce.json" if model == MODEL else \
        f"passive_reproduce_{model.split('/')[-1]}.json"
    C.save_json(name, out)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    out = run("cuda" if torch.cuda.is_available() else "cpu", args.model)
    s = out["summary"]
    for k, v in s.items():
        print(f"{k:55s}", " ".join(f"{x:.3f}" for x in v))
