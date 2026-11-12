"""run_with_spectra on tiny random checkpoints of several architectures.

Needs the network (or the checkpoints in the local Hugging Face cache), so it
runs only when SPECTRAL_TRUST_MODELS names the models, comma-separated, or is
"all" for the list below. The model-matrix job in
.github/workflows/models.yml runs one model per job:

    SPECTRAL_TRUST_MODELS=hf-internal-testing/tiny-random-gpt2 pytest tests/test_model_matrix.py
"""
import os

import numpy as np
import pytest
import torch

import spectral_trust as st

#: Tiny random checkpoints, one per architecture. hf-internal-testing has no
#: Qwen2 checkpoint; the TRL test organisation's is used instead.
MODELS = [
    "hf-internal-testing/tiny-random-gpt2",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    "hf-internal-testing/tiny-random-PhiForCausalLM",
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    "trl-internal-testing/tiny-Qwen3ForCausalLM",
]

_REQUESTED = os.environ.get("SPECTRAL_TRUST_MODELS", "").strip()
SELECTED = MODELS if _REQUESTED == "all" else [m for m in _REQUESTED.split(",") if m]

pytestmark = pytest.mark.skipif(not SELECTED, reason="set SPECTRAL_TRUST_MODELS to run")

TEXTS = ["The cat sat on the mat because it was warm.", "Paris is the capital of France."]


@pytest.fixture(scope="module", params=SELECTED or ["none"])
def model(request):
    torch.manual_seed(0)
    return st.SpectralModel.from_pretrained(request.param)


def test_run_with_spectra(model, tmp_path):
    cache = model.run_with_spectra(TEXTS[0])
    n_layers = model.n_layers
    assert cache.layers and set(cache.layers) <= set(range(n_layers))
    assert cache.n_heads == model.n_heads
    for key in cache.keys():
        if not key.endswith(".pattern"):
            assert np.all(np.isfinite(np.asarray(cache[key], dtype=np.float64))), key
    lay = cache.layers[-1]
    lam = cache[f"blocks.{lay}.attn.eigenvalues"]
    assert np.all(lam >= -1e-6) and np.all(lam <= 2 + 1e-5)       # sym spectrum
    df = cache.to_dataframe()
    assert set(df["layer"]) == set(cache.layers)
    cache.heatmap("fiedler_value").save(tmp_path / "heat.html")
    cache.graph(lay, head=0).save(tmp_path / "graph.html")
    back = st.SpectralCache.load(cache.save(tmp_path / "cache"))
    np.testing.assert_array_equal(back[f"blocks.{lay}.attn.fiedler_value"],
                                  cache[f"blocks.{lay}.attn.fiedler_value"])


def test_graph_namespace_matches_analyze_layer(model):
    cache = model.run_with_spectra(TEXTS[0])
    ids = torch.tensor([model.tokenizer(TEXTS[0])["input_ids"]])
    with torch.no_grad():
        out = model.model(input_ids=ids, output_attentions=True, output_hidden_states=True)
    cfg = st.GSPConfig(device="cpu")
    gc, an = st.GraphConstructor(cfg), st.SpectralAnalyzer(cfg)
    k = len(out.attentions) - 1
    lay = cache.layers[-1]
    att = out.attentions[k].float()
    lap = gc.construct_laplacian(gc.aggregate_heads(gc.symmetrize_attention(att)))[0]
    d = an.analyze_layer(out.hidden_states[lay + 1][0].float(), lap, lay)
    assert cache[f"blocks.{lay}.graph.fiedler_value"] == pytest.approx(d.fiedler_value,
                                                                     rel=1e-5, abs=1e-6)
    assert cache[f"blocks.{lay}.graph.hfer"] == pytest.approx(d.hfer, rel=1e-5, abs=1e-6)


def test_batch_equals_single_runs(model):
    batch = model.run_with_spectra(TEXTS)
    for i, text in enumerate(TEXTS):
        one = model.run_with_spectra(text)
        for key in one.keys():
            a = np.asarray(one[key], np.float64)
            b = np.asarray(batch[i][key], np.float64)
            np.testing.assert_allclose(b, a, rtol=1e-3, atol=1e-4, err_msg=key)
