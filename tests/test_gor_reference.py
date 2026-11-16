"""features.geometry_of_reason against the original code of the paper
(spectral-trust 0.1.2), stored in tests/data/gor_reference_012.npz by
tests/data/make_gor_reference.py."""
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
import torch

from spectral_trust import features as F

DATA = Path(__file__).with_name("data")
sys.path.insert(0, str(DATA))
from make_gor_reference import CASES  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    with np.load(DATA / "gor_reference_012.npz") as f:
        return {k: f[k] for k in f.files}


def _run(ref, name, seed="case"):
    n, over, case_seed = CASES[name]
    att = [torch.from_numpy(a) for a in ref[f"{name}/attentions"]]
    hid = [torch.from_numpy(h) for h in ref[f"{name}/hidden_states"]]
    kw = {"normalization": "rw", **over}  # the stored fixture is the first released run
    out = F.geometry_of_reason(att, hid, arpack_seed=case_seed if seed == "case" else seed,
                               **kw)
    return np.stack([out[k] for k in F.GOR_FEATURES])


@pytest.mark.parametrize("name", list(CASES))
def test_matches_the_original_code(ref, name):
    env = f"{platform.platform()}|py{platform.python_version()}|torch{torch.__version__}|" \
          f"numpy{np.__version__}|scipy{scipy.__version__}"
    got, want = _run(ref, name), ref[f"{name}/features"]
    if str(ref["__env__"]) == env:
        assert np.array_equal(got, want)
    elif name.startswith("sparse"):
        # ARPACK's iterations differ across platforms even from a fixed start vector
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-4)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_unseeded_sparse_path_keeps_energy_and_smoothness(ref):
    got, want = _run(ref, "sparse_rw", seed=None), ref["sparse_rw/features"]
    names = list(F.GOR_FEATURES)
    for k in ("energy", "smoothness"):
        np.testing.assert_allclose(got[names.index(k)], want[names.index(k)], rtol=1e-6)
    assert np.all(np.isfinite(got))


def test_text_helper_runs_a_model_as_0_1_2_did():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=50, n_positions=64, n_embd=16, n_layer=2, n_head=2)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    model.config._attn_implementation = "eager"

    class Tok:
        def __call__(self, text, **kw):
            ids = torch.tensor([[ord(c) % 50 for c in text]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    out = F.geometry_of_reason_text(model, Tok(), "valid proofs are smooth")
    with torch.no_grad():
        ids = Tok()("valid proofs are smooth")
        o = model(**ids, output_attentions=True, output_hidden_states=True)
    want = F.geometry_of_reason(o.attentions, o.hidden_states)
    for k in F.GOR_FEATURES:
        assert out[k].shape == (2,)
        np.testing.assert_array_equal(out[k], want[k])
    with pytest.raises(ValueError):
        F.geometry_of_reason(o.attentions, o.hidden_states[1:])
