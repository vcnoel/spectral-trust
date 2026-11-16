"""Every 0.5.0 call gives the result it gave in 0.5.0.

tests/data/regression_050.npz holds the outputs of compute_spectra (three
token spans, five graph configurations, every metric namespace),
per_head_metrics, per_head_hybrid_metrics and analyze_text (with and without
subgraph_indices) on seeded inputs, written by
tests/data/make_regression_050.py from the 0.5.0 code. Results must be
bit-identical in the environment that wrote the fixture; elsewhere (another
BLAS or library version) they must agree to float32 rounding.
"""
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
import torch

DATA = Path(__file__).with_name("data")
sys.path.insert(0, str(DATA))
import make_regression_050 as gen  # noqa: E402


@pytest.fixture(scope="module")
def recorded():
    with np.load(DATA / "regression_050.npz") as f:
        return {k: f[k] for k in f.files}


def _env() -> str:
    return f"{platform.platform()}|py{platform.python_version()}|torch{torch.__version__}|" \
           f"numpy{np.__version__}|scipy{scipy.__version__}"


def test_every_050_output_is_unchanged(recorded):
    env = str(recorded.pop("__env__"))
    now = gen.collect()
    assert sorted(now) == sorted(recorded), "the set of recorded outputs changed"
    exact = env == _env()
    bad = []
    for key, ref in recorded.items():
        got = now[key]
        same = np.array_equal(got, ref, equal_nan=True) if exact else \
            np.allclose(got, ref, rtol=1e-5, atol=1e-6, equal_nan=True)
        if not same:
            bad.append(key)
    assert not bad, f"{len(bad)} outputs changed, e.g. {bad[:5]}"
