# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""Write tests/data/regression_050.npz: outputs of the 0.5.0 code paths on
seeded synthetic inputs, so tests/test_regression_050.py can check that
later releases leave every existing call unchanged.

Run from the repository root with the 0.5.0 code (PYTHONPATH=src):
    python tests/data/make_regression_050.py
"""
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_inputs import inputs, CONFIGS, SPANS, SUBGRAPH  # noqa: E402

import spectral_trust as st  # noqa: E402


def collect() -> dict:
    """Every recorded output, as name -> float64 array."""
    attn, sig, tokens, lengths = inputs()
    out = {}
    for cname, kw in CONFIGS.items():
        cfg = st.GSPConfig(device="cpu", save_plots=False, verbose=False, **kw)
        for sname, span in SPANS.items():
            cache = st.compute_spectra(attn, sig, tokens=tokens, lengths=lengths, config=cfg,
                                       token_span=span, keep_patterns=False)
            for b in range(len(tokens)):
                rec = cache.text(b) if cache.is_batch else cache
                for key in sorted(rec.keys()):
                    out[f"cs/{cname}/{sname}/{b}/{key}"] = np.asarray(rec[key], np.float64)
        a0 = attn[1][0]
        ph = st.per_head_metrics(a0, config=cfg, keep_eigenvalues=True)
        out[f"ph/{cname}/values"] = np.asarray(ph.values, np.float64)
        out[f"ph/{cname}/eigenvalues"] = np.asarray(ph.eigenvalues, np.float64)
        hy = st.per_head_hybrid_metrics(a0, sig[1][0], config=cfg, token_span=(2, 20))
        out[f"phh/{cname}/values"] = np.asarray(hy.values, np.float64)
        for sub in (None, SUBGRAPH):
            tag = "full" if sub is None else "sub"
            for k, d in enumerate(analyze_text(cfg, attn, sig, sub)):
                for f in ("energy", "smoothness_index", "spectral_entropy", "hfer",
                          "fiedler_value"):
                    out[f"at/{cname}/{tag}/{k}/{f}"] = np.asarray(getattr(d, f), np.float64)
                out[f"at/{cname}/{tag}/{k}/eigenvalues"] = np.asarray(d.eigenvalues, np.float64)
    return out


def analyze_text(cfg, attn, sig, sub):
    """GSPDiagnosticsFramework.analyze_text on stored tensors (no model)."""
    layers = sorted(attn)
    hidden = [sig[layers[0]][:1]] + [sig[lay][:1] for lay in layers]
    fw = st.GSPDiagnosticsFramework.__new__(st.GSPDiagnosticsFramework)
    fw.config = cfg
    fw.graph_constructor = st.GraphConstructor(cfg)
    fw.spectral_analyzer = st.SpectralAnalyzer(cfg)
    fw.instrumenter = SimpleNamespace(
        model=None,
        process_text=lambda text: {"attentions": [attn[lay][:1] for lay in layers],
                                   "hidden_states": hidden, "inputs": {},
                                   "tokens": ["t"] * attn[layers[0]].shape[-1]})
    return fw.analyze_text("x", save_results=False, subgraph_indices=sub)["layer_diagnostics"]


if __name__ == "__main__":
    torch.set_num_threads(4)
    data = collect()
    env = f"{platform.platform()}|py{platform.python_version()}|torch{torch.__version__}|" \
          f"numpy{np.__version__}|scipy{scipy.__version__}"
    path = Path(__file__).with_name("regression_050.npz")
    np.savez_compressed(path, __env__=np.array(env), **data)
    print(f"{len(data)} arrays -> {path} ({path.stat().st_size // 1024} KB)")
