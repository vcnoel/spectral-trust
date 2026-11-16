"""Every 0.5.0 call gives the result it gave in 0.5.0.

tests/data/regression_050.npz holds the outputs of compute_spectra (three
token spans, five graph configurations, every metric namespace),
per_head_metrics, per_head_hybrid_metrics and analyze_text (with and without
subgraph_indices) on seeded inputs, written by
tests/data/make_regression_050.py from the 0.5.0 code. Results must be
bit-identical in the environment that wrote the fixture; elsewhere (another
BLAS or library version) they must agree to float32 rounding, except the
outputs that depend on the eigenvector basis (the Fiedler vector's sign and
the split of spectral mass within repeated eigenvalues), which another
eigensolver may legitimately return differently, and a few HFER values whose
cutoff falls between nearly tied eigenvalues.
"""
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
import torch

DATA = Path(__file__).with_name("data")
BASIS_DEPENDENT = (".fiedler_vector", ".spectral_masses")
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
        if not exact and key.endswith(BASIS_DEPENDENT):
            continue
        got = now[key]
        same = np.array_equal(got, ref, equal_nan=True) if exact else \
            np.allclose(got, ref, rtol=1e-5, atol=1e-6, equal_nan=True)
        if not same:
            bad.append(key)
    if not exact:
        # HFER splits the signal's energy at a cutoff index; where two eigenvalues
        # nearly tie at the cutoff, another eigensolver may split their energy
        # differently. Allow at most 5% of HFER outputs to differ across platforms.
        n_hfer = sum(k.endswith("hfer") for k in recorded)
        hfer_bad = [k for k in bad if k.endswith("hfer")]
        assert len(hfer_bad) <= 0.05 * n_hfer, f"{len(hfer_bad)} of {n_hfer} HFER outputs changed"
        bad = [k for k in bad if not k.endswith("hfer")]
    assert not bad, f"{len(bad)} outputs changed, e.g. {bad[:5]}"
