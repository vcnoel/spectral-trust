# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""spectral_trust.integrations.lm_polygraph. Uses the real lm-polygraph when
it is installed, and otherwise a minimal stub of its 0.7.0 Estimator
contract (Estimator(stats_dependencies, level), __str__, __call__(stats))."""
import abc
import importlib
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust.operator_spectra import operator_metrics  # noqa: E402

torch.set_num_threads(2)
BRIDGE = "spectral_trust.integrations.lm_polygraph"
V = 40


def _stub_modules():
    class Estimator(abc.ABC):
        def __init__(self, stats_dependencies, level):
            assert level in ["sequence", "claim", "token"]
            self.level = level
            self.stats_dependencies = stats_dependencies

        @abc.abstractmethod
        def __str__(self):
            ...

        @abc.abstractmethod
        def __call__(self, stats):
            ...

    root = types.ModuleType("lm_polygraph")
    est_pkg = types.ModuleType("lm_polygraph.estimators")
    est_mod = types.ModuleType("lm_polygraph.estimators.estimator")
    est_mod.Estimator = Estimator
    root.estimators, est_pkg.estimator = est_pkg, est_mod
    return {"lm_polygraph": root, "lm_polygraph.estimators": est_pkg,
            "lm_polygraph.estimators.estimator": est_mod}


@pytest.fixture
def bridge(monkeypatch):
    try:
        importlib.import_module("lm_polygraph.estimators.estimator")
    except ImportError:
        for name, mod in _stub_modules().items():
            monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, BRIDGE, raising=False)
    mod = importlib.import_module(BRIDGE)
    yield mod
    sys.modules.pop(BRIDGE, None)


@pytest.fixture(scope="module")
def stats():
    """The two statistics, as AttentionForwardPassCalculator and the greedy
    calculator lay them out: [B, L, H, S, S] padded with NaN, and the
    generated token ids of each item."""
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=3, n_head=4, n_embd=32, vocab_size=V,
                                   n_positions=64)).eval()
    m.set_attn_implementation("eager")
    items, greedy = [], []
    for seed, (n_prompt, n_gen) in enumerate([(9, 5), (6, 3)]):
        ids = torch.randint(1, V, (n_prompt + n_gen,), generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            att = m(input_ids=ids[None], output_attentions=True).attentions
        items.append(torch.cat(att).float().numpy())                   # [L, H, S, S]
        greedy.append(ids[n_prompt:].tolist())
    S = max(a.shape[-1] for a in items)
    padded = np.stack([np.pad(a, ((0, 0), (0, 0), (0, S - a.shape[-1]), (0, S - a.shape[-1])),
                              constant_values=np.nan) for a in items])
    return {"forwardpass_attention_weights": padded, "greedy_tokens": greedy, "raw": items}


def test_core_package_never_imports_lm_polygraph():
    code = ("import sys, spectral_trust; "
            "assert 'lm_polygraph' not in sys.modules; "
            f"assert {BRIDGE!r} not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_extra_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_polygraph", None)
    monkeypatch.setitem(sys.modules, "lm_polygraph.estimators", None)
    monkeypatch.setitem(sys.modules, "lm_polygraph.estimators.estimator", None)
    monkeypatch.delitem(sys.modules, BRIDGE, raising=False)
    with pytest.raises(ImportError, match=r"spectral_trust\[polygraph\]"):
        importlib.import_module(BRIDGE)
    sys.modules.pop(BRIDGE, None)


def test_estimator_contract(bridge):
    from lm_polygraph.estimators.estimator import Estimator
    est = bridge.SpectralTrustEstimator("participation_ratio", sign=-1)
    assert isinstance(est, Estimator)
    assert est.level == "sequence" and est.stats_dependencies == bridge.STATS
    assert est.metric == "operator.participation_ratio"
    names = {str(est), str(bridge.SpectralTrustEstimator("participation_ratio")),
             str(bridge.SpectralTrustEstimator("participation_ratio", layers=[-1], sign=-1)),
             str(bridge.SpectralTrustEstimator("participation_ratio", span="generated", sign=-1))}
    assert len(names) == 4
    assert str(bridge.SpectralTrustEstimator("contraction", name="PR")) == "PR"
    for bad in (dict(metric="resid.energy"), dict(metric="hfer"),
                dict(layer_aggregation="last"), dict(span="answer")):
        with pytest.raises(ValueError):
            bridge.SpectralTrustEstimator(**bad)
    for metric, field in (("graph.fiedler_value", "edge_weights"),
                          ("operator.contraction", "operator_edges")):
        cfg = st.GSPConfig(device="cpu", verbose=False, **{field: "value_per_head"})
        with pytest.raises(ValueError, match=field):
            bridge.SpectralTrustEstimator(metric, config=cfg)
    cfg = st.GSPConfig(device="cpu", verbose=False, edge_weights="value_per_head")
    bridge.SpectralTrustEstimator("operator.contraction", config=cfg)    # graph option unused


def test_scores_are_finite_and_match_the_library(bridge, stats):
    est = bridge.SpectralTrustEstimator("operator.participation_ratio", sign=-1)
    scores = est(stats)
    assert scores.shape == (2,) and np.isfinite(scores).all()
    for k, raw in enumerate(stats["raw"]):
        ref = np.mean([operator_metrics(torch.from_numpy(raw[lay]).double(),
                                        normalize_rows=True)["participation_ratio"].mean()
                       for lay in range(3)])
        assert scores[k] == pytest.approx(-ref, abs=1e-6)
    # The generated span and one layer, with the max over heads.
    gen = bridge.SpectralTrustEstimator("operator.contraction", span="generated", layers=-1,
                                        head_aggregation="max")
    for k, raw in enumerate(stats["raw"]):
        g = len(stats["greedy_tokens"][k])
        a = torch.from_numpy(raw[2][:, -g:, -g:]).double()
        want = operator_metrics(a, normalize_rows=True)["contraction"].max()
        assert gen(stats)[k] == pytest.approx(float(want), abs=1e-6)
    # A per-layer metric of the head-averaged graph.
    fied = bridge.SpectralTrustEstimator("graph.fiedler_value", layer_aggregation="min")(stats)
    assert np.isfinite(fied).all()
    direct = st.compute_spectra({lay: torch.from_numpy(stats["raw"][1][lay]) for lay in range(3)},
                                tokens=[[""] * 9], metrics="graph.fiedler_value", hybrid=False)
    assert fied[1] == pytest.approx(min(direct[f"blocks.{lay}.graph.fiedler_value"]
                                        for lay in range(3)), abs=1e-7)


def test_empty_span_is_nan(bridge, stats):
    est = bridge.SpectralTrustEstimator("operator.participation_ratio", span="generated")
    assert np.isnan(est.score(stats["raw"][0], n_generated=0))
    assert np.isfinite(est.score(stats["raw"][0], n_generated=2))
