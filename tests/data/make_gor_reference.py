# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""Write tests/data/gor_reference_012.npz: the Geometry of Reason features
computed by the original code, spectral-trust 0.1.2 (commit 98d3921, the
release the paper's experiments ran on), on seeded synthetic inputs.

The 0.1.2 ``config``, ``graph`` and ``spectral`` modules are read from this
repository's history and run as ``analyze_text`` ran them. Above 50 tokens
0.1.2 calls ARPACK with a random start vector; here ``eigsh`` gets the start
vector that ``geometry_of_reason(..., arpack_seed=0)`` uses, so the two
can be compared exactly.

    python tests/data/make_gor_reference.py      (from the repository root)
"""
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
import torch

COMMIT = "98d3921"
ROOT = Path(__file__).resolve().parents[2]
#: name -> (sequence length, 0.1.2 GSPConfig overrides, ARPACK seed or None)
CASES = {
    "dense_rw": (24, {}, None),
    "sparse_rw": (72, {}, 0),
    "text_variant": (24, {"normalization": "none", "eigen_solver": "dense",
                          "hfer_cutoff_ratio": 0.5}, None),
    "weighted_rownorm_sym": (30, {"normalization": "sym", "symmetrization": "row_norm",
                                  "head_aggregation": "attention_weighted"}, None),
}


def load_012(tmp: Path):
    pkg = tmp / "st012"
    pkg.mkdir()
    for m in ("config", "graph", "spectral"):
        src = subprocess.run(["git", "show", f"{COMMIT}:src/spectral_trust/{m}.py"], cwd=ROOT,
                             check=True, capture_output=True, encoding="utf-8").stdout
        (pkg / f"{m}.py").write_text(src, encoding="utf-8")
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, str(tmp))
    import st012.config, st012.graph, st012.spectral  # noqa: E401
    return st012


def inputs(n: int, seed: int, layers: int = 3, heads: int = 4, d: int = 16):
    g = torch.Generator().manual_seed(seed)
    att = []
    for l in range(layers):
        a = torch.rand(1, heads, n, n, generator=g).tril()
        a[..., 0] += (1.0 + 2 * l) * a.sum(-1)
        att.append(a / a.sum(-1, keepdim=True))
    hid = [torch.randn(1, n, d, generator=g) for _ in range(layers + 1)]
    return att, hid


def run_012(st, att, hid, overrides, seed):
    if seed is not None:
        orig = st.spectral.eigsh

        def eigsh(A, *args, **kw):
            kw["v0"] = np.random.default_rng(seed).uniform(-1, 1, A.shape[0]).astype(A.dtype)
            return orig(A, *args, **kw)
        st.spectral.eigsh = eigsh
    cfg = st.config.GSPConfig(device="cpu", **overrides)
    gc, an = st.graph.GraphConstructor(cfg), st.spectral.SpectralAnalyzer(cfg)
    mask = torch.ones(1, att[0].shape[-1], dtype=torch.bool)
    rows = []
    for l, a in enumerate(att):                           # as 0.1.2 analyze_text
        sym = gc.symmetrize_attention(a.squeeze(0).unsqueeze(0)).squeeze(0)
        adj = gc.aggregate_heads(sym.unsqueeze(0), attn_mask=mask).squeeze(0)
        lap = gc.construct_laplacian(adj.unsqueeze(0)).squeeze(0)
        d = an.analyze_layer(hid[l + 1].squeeze(0), lap, l)
        rows.append([d.fiedler_value, d.energy, d.smoothness_index, d.spectral_entropy, d.hfer])
    if seed is not None:
        st.spectral.eigsh = orig
    return np.asarray(rows, dtype=np.float64).T          # [5, layers]


if __name__ == "__main__":
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        st = load_012(Path(tmp))
        for i, (name, (n, over, seed)) in enumerate(CASES.items()):
            att, hid = inputs(n, 100 + i)
            out[f"{name}/attentions"] = torch.stack(att).numpy()
            out[f"{name}/hidden_states"] = torch.stack(hid).numpy()
            out[f"{name}/features"] = run_012(st, att, hid, over, seed)
    env = f"{platform.platform()}|py{platform.python_version()}|torch{torch.__version__}|" \
          f"numpy{np.__version__}|scipy{scipy.__version__}"
    path = Path(__file__).with_name("gor_reference_012.npz")
    np.savez_compressed(path, __env__=np.array(env), __commit__=np.array(COMMIT), **out)
    print(f"{len(out)} arrays -> {path} ({path.stat().st_size // 1024} KB)")
