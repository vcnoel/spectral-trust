"""The TransformerLens and nnsight adapters against run_with_spectra on the
same model (gpt2) and prompt: equal up to float32 rounding.

Skipped when transformer_lens or nnsight is not installed, or when gpt2 is
neither in the local Hugging Face cache nor downloadable (the offline test
job). Install the extras with ``pip install spectral-trust[tl,nnsight]``.
"""
import numpy as np
import pytest
import torch

import spectral_trust as st

MODEL = "gpt2"
PROMPT = "The cat sat on the mat because it was warm."
N_LAYERS = 12


def _close(a, b, key):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    # Per-mode energies of nearly degenerate modes move most with float32
    # differences in the residual stream (up to 2e-4 relative on gpt2 through
    # TransformerBridge, whose residual stream differs from Hugging Face's by
    # about 2e-7 relative).
    rtol = 1e-3 if key.endswith("spectral_masses") else 1e-4
    np.testing.assert_allclose(a, b, rtol=rtol, atol=2e-5, err_msg=key)


def _hybrid_key(key: str) -> bool:
    name = key.split(".", 2)[-1]
    return (".resid." in key or name in ("graph.energy", "graph.smoothness_index",
                                         "graph.spectral_entropy", "graph.hfer",
                                         "graph.spectral_masses"))


def _ill_conditioned_connectivity(key, ref) -> bool:
    """``connectivity`` is a threshold on the smallest eigenvalue. Where that
    eigenvalue's float32 noise sits within a factor 2 of the tolerance, a
    1e-6 change in the attention flips it, so it is not compared there."""
    if not key.endswith(".graph.connectivity"):
        return False
    lam = np.asarray(ref[key.replace("connectivity", "eigenvalues")])
    tol = st.SpectralAnalyzer.zero_eigenvalue_tolerance(lam, lam.size)
    return abs(float(lam[0]) - tol) < tol


def _compare(got, ref, skip=lambda key: False):
    assert got.layers == ref.layers
    assert got.tokens == ref.tokens
    assert set(got.metric_names) <= set(ref.metric_names)
    checked = 0
    for key in got.keys():
        if key.endswith(".pattern") or skip(key) or _ill_conditioned_connectivity(key, ref):
            continue
        _close(got[key], ref[key], key)
        checked += 1
    return checked


@pytest.fixture(scope="module")
def reference():
    try:
        model = st.SpectralModel.from_pretrained(MODEL)
    except OSError as exc:  # not cached and offline
        pytest.skip(f"{MODEL} is not available: {exc}")
    ids = model.tokenizer(PROMPT)["input_ids"]
    return model, ids, model.run_with_spectra(PROMPT)


# ── TransformerLens ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bridge(reference):
    pytest.importorskip("transformer_lens")
    try:
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError:
        pytest.skip("TransformerBridge needs TransformerLens 3 or later")
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


def test_transformer_lens_bridge_matches_run_with_spectra(reference, bridge):
    _, ids, ref = reference
    tokens = torch.tensor([ids])
    _, tl_cache = bridge.run_with_cache(tokens)
    got = st.from_transformer_lens(tl_cache, bridge, tokens)
    assert got.meta["signal_source"][str(N_LAYERS - 1)] == "ln_final.hook_out"
    assert _compare(got, ref) > 200


def test_transformer_lens_resid_post_for_the_last_layer(reference, bridge):
    _, ids, ref = reference
    tokens = torch.tensor([ids])
    _, tl_cache = bridge.run_with_cache(tokens)
    got = st.from_transformer_lens(tl_cache, bridge, tokens, final_norm=False)
    last = N_LAYERS - 1
    # every layer but the last is unchanged
    _compare(got, ref, skip=lambda key: key.startswith(f"blocks.{last}.") and _hybrid_key(key))
    # the last layer now reads the residual stream before the final norm
    hy = st.per_head_hybrid_metrics(tl_cache[f"blocks.{last}.attn.hook_pattern"],
                                    tl_cache[f"blocks.{last}.hook_resid_post"], layer=last)
    _close(got[f"blocks.{last}.resid.energy"], hy.metric("energy"), "resid_post energy")
    assert not np.allclose(got[f"blocks.{last}.resid.energy"], ref[f"blocks.{last}.resid.energy"])


def test_transformer_lens_attention_only_and_text_tokens(reference, bridge):
    _, ids, ref = reference
    _, tl_cache = bridge.run_with_cache(torch.tensor([ids]),
                                        names_filter=lambda name: name.endswith("hook_pattern"))
    got = st.from_transformer_lens(tl_cache, bridge, PROMPT)
    assert not got.meta["hybrid"]
    assert all(f == "attention" for f in got.families.values())
    _compare(got, ref, skip=_hybrid_key)


def test_hooked_transformer_matches_run_with_spectra(reference):
    tl = pytest.importorskip("transformer_lens")
    try:
        HookedTransformer = tl.HookedTransformer
    except (ImportError, AttributeError):
        pytest.skip("HookedTransformer was removed in TransformerLens 4")
    _, ids, ref = reference
    model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cpu")
    tokens = torch.tensor([ids])
    _, tl_cache = model.run_with_cache(tokens)
    got = st.from_transformer_lens(tl_cache, model, tokens)
    # HookedTransformer has no ln_final.hook_out: its last layer reads resid_post
    last = N_LAYERS - 1
    _compare(got, ref, skip=lambda key: key.startswith(f"blocks.{last}.") and _hybrid_key(key))


# ── nnsight ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", params=["LanguageModel", "TransformersModel"])
def lm(request, reference):
    import warnings

    nnsight = pytest.importorskip("nnsight")
    try:
        cls = getattr(nnsight, request.param)
    except (AttributeError, ImportError):
        pytest.skip(f"this nnsight version has no {request.param}")
    kwargs = {"task": "text-generation"} if request.param == "TransformersModel" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # LanguageModel in nnsight 0.8
        return cls(MODEL, attn_implementation="eager", dispatch=True, **kwargs)


def _trace_blocks(lm):
    attn, resid = [], []
    with lm.trace(PROMPT):
        for block in lm.transformer.h:
            attn.append(block.attn.output[1].save())
            resid.append(block.output.save())
        final = lm.transformer.ln_f.output.save()
    return attn, resid, final


def test_nnsight_module_saves_match_run_with_spectra(reference, lm):
    _, _, ref = reference
    attn, resid, final = _trace_blocks(lm)
    got = st.from_nnsight(attn, resid, model=lm, tokens=PROMPT, final_norm=final)
    assert got.layers == list(range(N_LAYERS))
    assert _compare(got, ref) > 200
    # without the final norm only the last layer's hybrid metrics move
    plain = st.from_nnsight(attn, resid, model=lm, tokens=PROMPT)
    last = N_LAYERS - 1
    _compare(plain, ref, skip=lambda key: key.startswith(f"blocks.{last}.") and _hybrid_key(key))


def test_nnsight_attention_only(reference, lm):
    _, _, ref = reference
    attn, _, _ = _trace_blocks(lm)
    got = st.from_nnsight(attn, model=lm, tokens=PROMPT)
    assert all(f == "attention" for f in got.families.values())
    _compare(got, ref, skip=_hybrid_key)


def test_nnsight_model_output_matches_run_with_spectra(reference, lm):
    _, _, ref = reference
    with lm.trace(PROMPT, output_attentions=True, output_hidden_states=True):
        out = lm.output.save()
    got = st.from_nnsight(out.attentions, out.hidden_states, model=lm, tokens=PROMPT)
    assert _compare(got, ref) > 200
