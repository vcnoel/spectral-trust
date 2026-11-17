"""spectral_trust.numeric.dynamics, the token-inventory controls and the
multi-step rollout, on tiny random models built offline and on the hand-made
tokenizers of test_numeric_icl (no download)."""
from collections import Counter

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust import GSPConfig, numeric  # noqa: E402
from spectral_trust.numeric import (  # noqa: E402
    Rollout, RolloutReadout, damped_oscillator, generate, inventory_covariates, logistic,
    lorenz, next_value_distribution, pendulum, periodic, rollout, serialize, sine,
    token_preserving_shuffle,
)

from .test_numeric_icl import DigitTokenizer, TripletTokenizer, _llama  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:graph_filter at layer")
torch.set_num_threads(2)
CFG = GSPConfig(device="cpu", verbose=False, save_plots=False)


@pytest.fixture(scope="module")
def digit_model():
    return _llama(12, seed=5)


@pytest.fixture(scope="module")
def triplet_model():
    return _llama(1012, seed=6)


@pytest.fixture(scope="module")
def sharp_model():
    """Digit model with scaled-up logits: peaked distributions that an edit moves."""
    m = _llama(12, seed=7)
    with torch.no_grad():
        m.lm_head.weight.mul_(40.0)
    return m


# ── generators ───────────────────────────────────────────────────────────────

ALL = [("periodic", {"p": 8}), ("logistic", {"r": 3.9, "burn_in": 100}),
       ("lorenz", {"coord": "y"}), ("sine", {}), ("damped_oscillator", {}),
       ("pendulum", {})]


@pytest.mark.parametrize("family,params", ALL)
def test_generators_are_seeded_and_quantised(family, params):
    a = generate(family, 120, seed=3, **params)
    b = generate(family, 120, seed=3, **params)
    c = generate(family, 120, seed=4, **params)
    assert a.family == family and a.params["seed"] == 3 and len(a) == 120
    assert np.array_equal(a.values, b.values) and np.array_equal(a.ints, b.ints)
    assert not np.array_equal(a.values, c.values)
    assert a.ints.dtype == np.int64 and a.ints.min() == 50 and a.ints.max() == 950
    assert np.array_equal(a.ints, numeric.quantize(a.values))
    ctx, cont = a.split(100, 10)
    assert len(ctx) == 100 and np.array_equal(cont, a.ints[100:110])
    assert a.text(3) == serialize(a.ints[:3])


def test_periodic_and_logistic():
    t = periodic(8, 100, seed=2)
    assert np.array_equal(t.ints[8:], t.ints[:-8]) and len(np.unique(t.ints)) == 8
    assert len(np.unique(periodic(1, 10).ints)) == 1
    with pytest.raises(ValueError):
        periodic(0)
    t = logistic(3.9, 50, seed=1)
    x = t.values
    assert np.allclose(x[1:], 3.9 * x[:-1] * (1 - x[:-1]))
    assert 0.3 <= t.params["x0"] <= 0.7 and x[0] == t.params["x0"]
    # after burn-in, r = 3.5 is on its period-4 orbit and r = 3.55 on period 8
    for r, p in ((3.5, 4), (3.55, 8)):
        q = logistic(r, 64, seed=0, burn_in=5000).ints
        assert np.array_equal(q[p:], q[:-p]) and len(np.unique(q)) == p


def test_lorenz_coordinates_and_continuity():
    full = lorenz(30, seed=1, burn_in=0)
    later = lorenz(20, seed=1, burn_in=10)
    assert np.allclose(full.values[10:], later.values)          # burn-in drops samples
    z = lorenz(200, seed=1, coord="z")
    assert z.values.min() > 0 and z.params["coord"] == "z"
    assert np.array_equal(lorenz(50, seed=1, coord=2).values, lorenz(50, seed=1, coord="z").values)
    # the system's derivative: dx/dt = sigma (y - x), checked by central differences
    x, y = lorenz(400, seed=2, dt=0.002, substeps=2).values, lorenz(
        400, seed=2, coord="y", dt=0.002, substeps=2).values
    dx = (x[2:] - x[:-2]) / 0.004
    assert np.allclose(dx, 10.0 * (y[1:-1] - x[1:-1]), rtol=0.02, atol=0.5)
    with pytest.raises(ValueError):
        lorenz(10, coord="w")


def test_smooth_families():
    s = sine(256, frequency=1 / 32, phase=0.3)
    assert np.allclose(s.values, np.sin(2 * np.pi * np.arange(256) / 32 + 0.3))
    assert np.argmax(np.abs(np.fft.rfft(s.values))) == 256 // 32
    d = damped_oscillator(200, frequency=1 / 20, damping=0.0, phase=0.0)
    assert np.allclose(d.values, np.cos(2 * np.pi * np.arange(200) / 20))
    d = damped_oscillator(400, damping=0.05)
    assert np.abs(d.values[-40:]).max() < 0.2 * np.abs(d.values[:40]).max()
    with pytest.raises(ValueError):
        damped_oscillator(10, damping=1.0)
    # small swings: the pendulum is the harmonic oscillator cos(omega t)
    w = 2 * np.pi / 40
    p = pendulum(200, theta0=0.01, omega=w)
    assert np.allclose(p.values, 0.01 * np.cos(w * np.arange(200)), atol=2e-5)
    # large swings: symplectic, so the energy stays put over long runs
    p = pendulum(5000, seed=1)
    assert p.params["energy_drift"] < 1e-3 and 0.5 <= p.params["theta0"] <= 2.5
    m = pendulum(100, seed=1, coord="momentum")
    assert m.values[0] == 0.0
    with pytest.raises(ValueError):
        generate("tent", 10)


# ── token-inventory controls ─────────────────────────────────────────────────

def test_token_preserving_shuffle_keeps_tokens():
    q = logistic(3.9, 80, seed=1).ints
    s = token_preserving_shuffle(q, seed=7)
    assert np.array_equal(s, token_preserving_shuffle(q, seed=7))
    assert not np.array_equal(s, q) and sorted(s) == sorted(q)
    for tok in (DigitTokenizer(), TripletTokenizer()):
        assert Counter(tok.encode(serialize(s))) == Counter(tok.encode(serialize(q)))
    # shuffling raw values gives the shuffled integers
    t = sine(64, seed=2)
    assert np.array_equal(numeric.quantize(token_preserving_shuffle(t.values, 3)),
                          token_preserving_shuffle(t.ints, 3))
    with pytest.raises(ValueError):
        token_preserving_shuffle(np.zeros((2, 3)))


def test_inventory_covariates():
    one = inventory_covariates([500, 500, 120, 500])
    assert one["n_unique"] == 2 and one["repeat_fraction"] == pytest.approx(0.5)
    assert one["value_entropy"] == pytest.approx(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
    assert set(one) == set(numeric.INVENTORY_COVARIATES)
    seqs = [periodic(4, 40, seed=0).ints, logistic(3.9, 60, seed=0).ints, sine(50).ints]
    many = inventory_covariates(seqs)
    assert many["n_unique"].shape == (3,) and many["n_unique"][0] == 4
    shuffled = inventory_covariates([token_preserving_shuffle(s, 1) for s in seqs])
    for k in many:                                   # unchanged by the shuffle
        assert np.allclose(many[k], shuffled[k])
    # token ids: one per value plus the separator, or at most ten digits plus it
    tri = inventory_covariates(seqs, tokenizer=TripletTokenizer())
    assert np.array_equal(tri["n_unique_tokens"], many["n_unique"] + 1)
    dig = inventory_covariates(seqs, tokenizer=DigitTokenizer())
    assert (dig["n_unique_tokens"] <= 11).all()
    # straight into a confound audit
    rng = np.random.default_rng(0)
    seqs = [generate("logistic", 60, seed=i, r=r, burn_in=2000).ints
            for i in range(6) for r in (3.5, 3.9)]
    y = np.array([0, 1] * 6)
    readout = inventory_covariates(seqs)["n_unique"] + rng.normal(0, 1, 12)
    res = st.confound_audit({"readout": readout}, y, inventory_covariates(seqs), n_boot=50)
    names = {r["covariate"] for r in res.covariate_rows}
    assert names == {"n_unique", "repeat_fraction", "value_entropy"}


# ── rollout ──────────────────────────────────────────────────────────────────

def _manual(model, tok, ctx, n_steps, predict="mode"):
    """Reference: serialise the growing context and read each step afresh."""
    vals, out = list(ctx), []
    for _ in range(n_steps):
        ids = torch.tensor([tok(serialize(vals))["input_ids"]])
        d = next_value_distribution(model, ids, tok)
        v = d.mode if predict == "mode" else int(round(d.mean))
        out.append((v, d.entropy))
        vals.append(v)
    return out


@pytest.mark.parametrize("which", ["digit", "triplet"])
def test_rollout_matches_step_by_step_reference(which, digit_model, triplet_model):
    model, tok = ((digit_model, DigitTokenizer()) if which == "digit"
                  else (triplet_model, TripletTokenizer()))
    t = periodic(4, 30, seed=1)
    ctx, cont = t.split(20, 5)
    ref = _manual(model, tok, ctx, 5)
    for use_cache in (True, False):
        r = rollout(model, tok, ctx, 5, target=cont, use_cache=use_cache)
        assert r.method == ("hierarchical" if which == "digit" else "single")
        assert r.predicted.tolist() == [v for v, _ in ref]
        assert np.allclose(r.entropy, [e for _, e in ref], atol=1e-6)
    ids = tok(serialize(ctx))["input_ids"]
    again = rollout(model, tok, None, 5, input_ids=ids, target=cont)
    assert np.array_equal(again.predicted, r.predicted)
    mean = rollout(model, tok, ctx, 3, predict="mean")
    assert mean.predicted.tolist() == [v for v, _ in _manual(model, tok, ctx, 3, "mean")]
    assert mean.abs_error is None and set(mean.summary()) == {"mean_entropy"}


def test_rollout_errors():
    r = Rollout(np.array([100, 130, 300, 200]), np.array([110, 120, 200, 400]),
                np.ones(4), np.ones(4), "single", tol=50)
    assert r.abs_error.tolist() == [10, 10, 100, 200]
    assert r.cum_error.tolist() == [10, 20, 120, 320]
    assert r.valid_steps == 2
    s = r.summary()
    assert s["mae"] == 80 and s["cum_error"] == 320 and s["final_error"] == 200
    assert s["first_error"] == 10 and s["valid_steps"] == 2
    with pytest.raises(ValueError):
        rollout(None, TripletTokenizer(), [1, 2], 3, target=[1, 2], predict="median")


def test_rollout_readout_with_edit_with_controls(sharp_model):
    digit_model = sharp_model
    tok = DigitTokenizer()
    trajs = [sine(40, seed=i, frequency=1 / 8) for i in range(2)]
    ctx = [t.ints[:24] for t in trajs]
    cont = [t.ints[24:] for t in trajs]
    ro = RolloutReadout(tok, ctx, cont, n_steps=3)
    assert ro.positions(0) == [p for p in range(1, len(ro.inputs[0])) if p % 4 != 0]
    res = st.edit_with_controls(digit_model, ro.inputs, ro, layer=0, kind="lowpass",
                                cutoff=0.5, strengths=(0.0, 1.0), positions=ro.positions,
                                token_span=(1, None), controls=("permuted", "scaling"),
                                config=CFG)
    assert set(res.baseline[0]) == {"mean_entropy", "mae", "first_error", "final_error",
                                    "cum_error", "valid_steps"}
    for cond in res.conditions:                  # strength 0 is the identity
        for got, base in zip(res.edited[(cond, 0.0)], res.baseline):
            assert got == pytest.approx(base, abs=1e-6)
    assert not np.allclose(res.values("mean_entropy", "target", 1.0), res.values("mean_entropy"))
    base = rollout(digit_model, tok, ctx[0], 3, target=cont[0])
    assert res.baseline[0] == pytest.approx(base.summary())
    rows = res.summary(["mae", "cum_error"], n_boot=20)
    assert {r["condition"] for r in rows} == {"target", "permuted", "scaling"}
    # the edit reaches every step through the prompt's cache: rerunning the
    # sequence (use_cache=False) edits the first step only
    from spectral_trust.interventions import controls as C
    from spectral_trust.interventions._arch import Arch
    ids = ro.inputs[0][None]
    delta = torch.randn(ids.shape[1] - 1, 32) * 3.0
    rows_ = list(range(1, ids.shape[1]))
    out = {}
    for use_cache in (True, False):
        r1 = RolloutReadout(tok, ctx[:1], cont[:1], n_steps=2, use_cache=use_cache)
        C._run_with_delta(Arch(digit_model), digit_model, ids, "resid_post", 0, rows_, delta,
                          r1)
        out[use_cache] = r1.rollouts[0]
    assert out[True].entropy[0] == pytest.approx(out[False].entropy[0], abs=1e-6)
    assert out[True].entropy[0] != pytest.approx(base.entropy[0], abs=1e-3)
    if out[True].predicted[0] == out[False].predicted[0]:
        assert out[True].entropy[1] != pytest.approx(out[False].entropy[1], abs=1e-4)
    with pytest.raises(KeyError):
        ro(digit_model, torch.tensor([[0, 1, 2, 11]]))
    with pytest.raises(ValueError):
        RolloutReadout(tok, [ctx[0], ctx[0]], [cont[0], cont[1]], n_steps=2)


# ── value-weighted graphs in the diagnostics ─────────────────────────────────

@pytest.mark.parametrize("mode", ["attention", "value", "value_per_head"])
def test_profile_on_value_weighted_graphs_matches_compute_spectra(mode):
    from spectral_trust.numeric import profile, profile_from_outputs
    m = _llama(12, seed=3, layers=3)
    ids = torch.tensor([[0] + [1 + (i * 7) % 10 for i in range(23)]])
    cfg = GSPConfig(device="cpu", verbose=False, save_plots=False, edge_weights=mode)
    with torch.no_grad():
        out = m(input_ids=ids, output_attentions=True, output_hidden_states=True)
    edges = (None if mode == "attention"
             else st.value_weighted_attention(m, ids, per_head=mode == "value_per_head"))
    ref = st.compute_spectra({lay: out.attentions[lay] for lay in range(3)},
                             {lay: out.hidden_states[lay + 1] for lay in range(3)},
                             tokens=[[str(i) for i in ids[0].tolist()]], config=cfg, edges=edges)
    # compute_spectra's construction: no drop, no renormalisation, normalised basis,
    # states after the block, HFER over the top hfer_cutoff_ratio of the modes
    p = profile(m, ids, edge_weights=mode, drop=(), renormalize=False, basis="normalized",
                pairing="output", hfer_cutoff=cfg.hfer_cutoff_ratio)
    for lay in range(3):
        for mine, theirs in (("fiedler", "fiedler_value"), ("hfer", "hfer"),
                             ("spectral_entropy", "spectral_entropy")):
            assert p.per_layer[mine][lay] == pytest.approx(
                ref[f"blocks.{lay}.graph.{theirs}"], rel=1e-4, abs=1e-5), (mode, lay, mine)
    # the default is bit-identical to a call without the new arguments
    plain = profile_from_outputs(out.attentions, out.hidden_states)
    again = profile_from_outputs(out.attentions, out.hidden_states, edge_weights="attention")
    for k in plain.per_layer:
        assert np.array_equal(plain.per_layer[k], again.per_layer[k])
    if mode != "attention":
        pv = profile_from_outputs(out.attentions, out.hidden_states, edge_weights=mode,
                                  edges=edges)
        assert not np.allclose(pv.per_layer["fiedler"], plain.per_layer["fiedler"])
        assert np.array_equal(pv.per_layer["sink_mass"], plain.per_layer["sink_mass"])
        with pytest.raises(ValueError):
            profile_from_outputs(out.attentions, out.hidden_states, edge_weights=mode)
    with pytest.raises(ValueError):
        profile_from_outputs(out.attentions, edge_weights="values")


def test_centering_removes_the_dc_component():
    from spectral_trust.numeric import profile_from_outputs
    m = _llama(12, seed=3, layers=2)
    ids = torch.tensor([[0] + [1 + (i * 3) % 10 for i in range(19)]])
    with torch.no_grad():
        out = m(input_ids=ids, output_attentions=True, output_hidden_states=True)
    plain = profile_from_outputs(out.attentions, out.hidden_states, keep_masses=True)
    off = profile_from_outputs(out.attentions, out.hidden_states, keep_masses=True,
                               center=False, center_positions=[1, 2])
    for k in plain.per_layer:                         # the default is unchanged
        assert np.array_equal(plain.per_layer[k], off.per_layer[k])
    c = profile_from_outputs(out.attentions, out.hidden_states, keep_masses=True, center=True)
    for lay in (0, 1):
        # mode 0 of the combinatorial Laplacian of a connected graph is constant
        assert plain.masses[lay][0] > 1e-3 * plain.masses[lay].sum()
        assert c.masses[lay][0] < 1e-10 * c.masses[lay].sum()
    # by hand: the same as centering the hidden states first
    hs = [h - h[:, 1:].mean(1, keepdim=True) for h in out.hidden_states]
    ref = profile_from_outputs(out.attentions, hs)
    for k in ("hfer", "exp_se", "pos_hfer"):
        assert np.allclose(c.per_layer[k], ref.per_layer[k])
    # centring on a subset (e.g. the value tokens) uses that subset's mean
    sub = profile_from_outputs(out.attentions, out.hidden_states, center=True,
                               center_positions=list(range(1, 20, 2)))
    hs = [h - h[:, 1:20:2].mean(1, keepdim=True) for h in out.hidden_states]
    assert np.allclose(sub.per_layer["hfer"],
                       profile_from_outputs(out.attentions, hs).per_layer["hfer"])
