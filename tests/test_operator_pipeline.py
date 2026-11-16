# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""The opt-in ``operator`` namespace of SpectralModel.run_with_spectra and
compute_spectra (GSPConfig.operator_measure, operator_heads, operator_edges)."""
from dataclasses import replace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust import value_weighted  # noqa: E402
from spectral_trust.cache import resolve_metrics  # noqa: E402
from spectral_trust.operator_spectra import operator_metrics  # noqa: E402

torch.set_num_threads(2)
V = 40
CFG = st.GSPConfig(device="cpu", verbose=False, save_plots=False)
NAMES = st.OPERATOR_METRIC_NAMES


@pytest.fixture(scope="module")
def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=3, n_head=4, n_embd=32, vocab_size=V, n_positions=64))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.4)
    m.set_attn_implementation("eager")
    return m.eval()


def _ids(seed, n=12):
    return torch.randint(1, V, (n,), generator=torch.Generator().manual_seed(seed))


def _attn(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids[None], output_attentions=True, use_cache=False)
    return [a[0].float() for a in out.attentions]


def _run(model, ids, metrics="operator", **opts):
    return st.SpectralModel(model, None, config=replace(CFG, **opts)).run_with_spectra(
        ids, metrics=metrics)


def test_operator_is_opt_in():
    assert not any(q.startswith("operator.") for q in resolve_metrics(None))
    assert not any(q.startswith("operator.") for q in resolve_metrics(None, hybrid=False))
    assert resolve_metrics("operator") == [f"operator.{n}" for n in NAMES]
    assert resolve_metrics(["participation_ratio"]) == ["operator.participation_ratio"]
    assert resolve_metrics(["graph", "contraction"], hybrid=False)[-1] == "operator.contraction"
    assert "operator.contraction" not in resolve_metrics("graph")
    with pytest.raises(ValueError, match="operator"):
        resolve_metrics("operator.pr")


def test_per_head_values_equal_the_operator_of_each_head(gpt2):
    ids = _ids(0)
    cache = _run(gpt2, ids)
    for lay, a in enumerate(_attn(gpt2, ids)):
        ref = operator_metrics(a.double(), normalize_rows=True, return_eigenvalues=True)
        for name in NAMES:
            got = cache[f"blocks.{lay}.operator.{name}"]
            assert got.shape == (4,) and got.dtype == np.float32
            assert np.allclose(got, ref[name].numpy(), atol=1e-6), (lay, name)
        assert np.allclose(cache[f"blocks.{lay}.operator.eigenvalues"],
                           ref["eigenvalues"].numpy(), atol=1e-6)
        assert np.allclose(cache[f"blocks.{lay}.operator.eigenvalues"][:, 0], 1.0, atol=1e-6)
    assert cache.metric("operator.participation_ratio").shape == (3, 4)
    assert set(cache.metric_names) == {f"operator.{n}" for n in NAMES}


@pytest.mark.parametrize("measure", ["pullback", "counting"])
def test_head_average_span_and_heads(gpt2, measure):
    ids = _ids(1)
    attn = _attn(gpt2, ids)
    mean = _run(gpt2, ids, operator_heads="mean", operator_measure=measure)
    model = st.SpectralModel(gpt2, None, config=replace(CFG, operator_measure=measure))
    spanned = model.run_with_spectra(ids, metrics="operator", token_span=(3, 10))
    picked = model.run_with_spectra(ids, metrics="operator", heads=[1, 3])
    for lay, a in enumerate(attn):
        ref = operator_metrics(a.double().mean(0), measure, normalize_rows=True)
        sub = operator_metrics(a.double()[:, 3:10, 3:10], measure, normalize_rows=True)
        full = operator_metrics(a.double(), measure, normalize_rows=True)
        for name in NAMES:
            assert isinstance(mean[f"blocks.{lay}.operator.{name}"], float)
            assert mean[f"blocks.{lay}.operator.{name}"] == pytest.approx(float(ref[name]),
                                                                           abs=1e-9)
            assert np.allclose(spanned[f"blocks.{lay}.operator.{name}"], sub[name], atol=1e-6)
            got = picked[f"blocks.{lay}.operator.{name}"]
            assert np.isnan(got[[0, 2]]).all()
            assert np.allclose(got[[1, 3]], full[name].numpy()[[1, 3]], atol=1e-6)
        assert mean[f"blocks.{lay}.operator.eigenvalues"].shape == (12,)
        assert spanned[f"blocks.{lay}.operator.eigenvalues"].shape == (4, 7)


@pytest.mark.parametrize("heads_mode", ["per_head", "mean"])
def test_value_weighted_operator(gpt2, heads_mode):
    ids = _ids(2)
    vals = value_weighted.capture_attention_values(gpt2, ids[None])
    for mode in (["value_per_head"] if heads_mode == "per_head" else ["value_per_head", "value"]):
        cache = _run(gpt2, ids, operator_edges=mode, operator_heads=heads_mode)
        for lay in range(3):
            w = vals[lay].edge_weights(per_head=mode == "value_per_head")[0].double()
            if heads_mode == "mean" and w.dim() == 3:
                w = w.mean(0)
            ref = operator_metrics(w, normalize_rows=True)
            for name in NAMES:
                assert np.allclose(cache[f"blocks.{lay}.operator.{name}"], ref[name].numpy(),
                                   atol=1e-5), (mode, lay, name)
        plain = _run(gpt2, ids, operator_heads=heads_mode)
        assert not np.allclose(cache["blocks.1.operator.participation_ratio"],
                               plain["blocks.1.operator.participation_ratio"])


def test_option_errors(gpt2):
    with pytest.raises(ValueError, match="operator_heads='mean'"):
        _run(gpt2, _ids(3), operator_edges="value")
    for field, bad in (("operator_measure", "uniform"), ("operator_heads", "avg"),
                       ("operator_edges", "values")):
        with pytest.raises(ValueError, match=field):
            _run(gpt2, _ids(3), **{field: bad})
    a = {0: torch.softmax(torch.randn(1, 2, 5, 5), -1)}
    with pytest.raises(ValueError, match="operator_edges"):
        st.compute_spectra(a, tokens=[list("abcde")], metrics="operator",
                           config=replace(CFG, operator_edges="value_per_head"))
    # Without an operator metric the option is not used and nothing is captured.
    st.compute_spectra(a, tokens=[list("abcde")], metrics="attn",
                       config=replace(CFG, operator_edges="value_per_head"))


def test_operator_needs_no_value_capture_unless_asked(gpt2, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no value capture expected")
    monkeypatch.setattr(value_weighted, "ValueCapture", refuse)
    _run(gpt2, _ids(4))                                           # attention operator
    _run(gpt2, _ids(4), metrics="attn", operator_edges="value_per_head")


def test_cache_views_save_load_and_dataframe(gpt2, tmp_path):
    ids = _ids(5)
    cache = _run(gpt2, ids, metrics=["attn.hfer", "operator"])
    layer = cache.layer(2)
    assert set(layer.operator) == set(NAMES) and "operator.contraction" in layer.to_dict()
    head = layer.head(1)
    assert head["participation_ratio"] == pytest.approx(
        float(cache["blocks.2.operator.participation_ratio"][1]))
    assert head["operator.contraction"] == pytest.approx(
        float(cache["blocks.2.operator.contraction"][1]))
    assert "operator.participation_ratio" in repr(layer)
    assert np.array_equal(cache["participation_ratio", 2],
                          cache["blocks.2.operator.participation_ratio"])
    df = cache.to_dataframe()
    rows = df[df.metric == "operator.contraction"]
    assert len(rows) == 12 and set(rows.family) == {"attention"} and rows["head"].notna().all()
    path = cache.save(tmp_path / "op.npz")
    back = st.SpectralCache.load(path)
    assert back.keys() == cache.keys()
    for key in cache.keys():
        assert np.array_equal(np.asarray(back[key]), np.asarray(cache[key])), key
    mean = _run(gpt2, ids, operator_heads="mean")
    assert mean.layer(0).head(0).operator == {}
    assert isinstance(mean.layer(0).operator["contraction"], float)
    assert mean.metric("operator.contraction").shape == (3,)
    assert "operator.contraction" in repr(mean.layer(0))


def test_batched_run_matches_single_runs(gpt2):
    texts = [_ids(6, 9), _ids(7, 12)]
    model = st.SpectralModel(gpt2, None, config=CFG)
    batch = model.run_with_spectra([t.tolist() for t in texts], metrics="operator")
    for i, ids in enumerate(texts):
        one = model.run_with_spectra(ids, metrics="operator")
        for name in NAMES:
            assert np.allclose(batch[i][f"blocks.1.operator.{name}"],
                               one[f"blocks.1.operator.{name}"], atol=1e-5)


def test_from_hf_outputs_has_the_operator_namespace(gpt2):
    ids = _ids(8)
    with torch.no_grad():
        out = gpt2(input_ids=ids[None], output_attentions=True, output_hidden_states=True,
                   use_cache=False)
    cache = st.from_hf_outputs(out, metrics="operator", config=CFG)
    direct = _run(gpt2, ids)
    for key in direct.keys():
        assert np.allclose(np.asarray(cache[key]), np.asarray(direct[key]), atol=1e-6), key
