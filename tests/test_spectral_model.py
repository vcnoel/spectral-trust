"""SpectralModel.run_with_spectra, SpectralCache and from_hf_outputs on tiny
random models built offline (no download). Every number must equal the
library's existing functions: per_head_metrics, per_head_hybrid_metrics and
analyze_layer on the graph analyze_text builds."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust import (  # noqa: E402
    GSPConfig, GraphConstructor, SpectralAnalyzer, SpectralCache, SpectralModel,
    per_head_hybrid_metrics, per_head_metrics,
)

WORDS = ["<unk>", "<eos>", "the", "cat", "sat", "on", "mat", "because", "it", "was",
         "warm", ".", "paris", "is", "capital", "of", "france", "a", "dog", "slept"]
TEXT = "the cat sat on the mat because it was warm ."
OTHER = "paris is the capital of france ."


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)},
                                    unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")


@pytest.fixture(scope="module")
def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=3, n_head=4, n_embd=32, vocab_size=len(WORDS), n_positions=64,
                     bos_token_id=1, eos_token_id=1)
    model = GPT2LMHeadModel(cfg).eval()
    return SpectralModel.from_pretrained(model, tokenizer=_tokenizer())


@pytest.fixture(scope="module")
def tiny_hybrid():
    cls = getattr(transformers, "Qwen3NextForCausalLM", None)
    if cls is None:
        pytest.skip("this transformers version has no Qwen3-Next")
    cfg = transformers.Qwen3NextConfig(
        vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_value_heads=2,
        linear_num_key_heads=1, linear_key_head_dim=16, linear_value_head_dim=16,
        num_experts=2, num_experts_per_tok=1, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, decoder_sparse_step=1, full_attention_interval=2)
    torch.manual_seed(0)
    model = cls(cfg).eval()
    return SpectralModel.from_pretrained(model, tokenizer=_tokenizer())


def _forward(sm, text):
    ids = torch.tensor([sm.tokenizer(text)["input_ids"]])
    with torch.no_grad():
        return sm.model(input_ids=ids, output_attentions=True, output_hidden_states=True)


def _close(a, b, rtol=1e-4, atol=1e-5):
    np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64),
                               rtol=rtol, atol=atol)


# ── the cache layout ─────────────────────────────────────────────────────────

def test_keys_shapes_and_properties(tiny_gpt2):
    cache = tiny_gpt2.run_with_spectra(TEXT)
    assert cache.layers == [0, 1, 2]
    assert cache.tokens == TEXT.split()
    assert cache.n_heads == 4
    assert cache["blocks.1.attn.fiedler_value"].shape == (4,)
    assert isinstance(cache["blocks.1.graph.hfer"], float)
    assert cache["blocks.1.attn.eigenvalues"].shape == (4, 11)
    assert cache["blocks.1.graph.fiedler_vector"].shape == (11,)
    assert cache["blocks.1.attn.pattern"].shape == (4, 11, 11)
    assert "attn.fiedler_value" in cache.metric_names and "graph.hfer" in cache.metric_names
    assert cache.families["attn.hfer"] == "attention"
    assert cache.families["resid.hfer"] == "hybrid"
    assert cache.families["graph.hfer"] == "hybrid"
    assert cache.families["graph.fiedler_value"] == "attention"
    np.testing.assert_array_equal(cache["fiedler_value", 1], cache["blocks.1.attn.fiedler_value"])
    assert "blocks.1.attn.fiedler_value" in cache and "blocks.9.attn.hfer" not in cache
    assert len(cache) == len(cache.keys())
    text = repr(cache)
    assert "SpectralCache" in text and "11 tokens" in text and "attn=5" in text
    with pytest.raises(KeyError, match="Did you mean"):
        cache["blocks.1.attn.fiedler_valu"]


def test_matches_per_head_functions_and_analyze_layer(tiny_gpt2):
    cache = tiny_gpt2.run_with_spectra(TEXT)
    out = _forward(tiny_gpt2, TEXT)
    cfg = GSPConfig(device="cpu")
    gc, an = GraphConstructor(cfg), SpectralAnalyzer(cfg)
    for lay, att in enumerate(out.attentions):
        ph = per_head_metrics(att, config=cfg, layer=lay)
        for j, name in enumerate(ph.metric_names):
            _close(cache[f"blocks.{lay}.attn.{name}"], ph.values[:, j], rtol=1e-6, atol=1e-7)
        hy = per_head_hybrid_metrics(att, out.hidden_states[lay + 1], config=cfg, layer=lay)
        for j, name in enumerate(hy.metric_names):
            _close(cache[f"blocks.{lay}.resid.{name}"], hy.values[:, j], rtol=1e-6, atol=1e-7)
        sym = gc.symmetrize_attention(att)
        lap = gc.construct_laplacian(gc.aggregate_heads(sym))[0]
        d = an.analyze_layer(out.hidden_states[lay + 1][0], lap, lay)
        for name in st.GRAPH_METRIC_NAMES:
            assert cache[f"blocks.{lay}.graph.{name}"] == pytest.approx(float(getattr(d, name)),
                                                                     rel=1e-6, abs=1e-9)
        _close(cache[f"blocks.{lay}.graph.eigenvalues"], d.eigenvalues, atol=1e-6)


def test_layer_and_head_views(tiny_gpt2):
    cache = tiny_gpt2.run_with_spectra(TEXT)
    layer = cache.layer(2)
    assert layer["hfer"] == cache["blocks.2.graph.hfer"]
    np.testing.assert_array_equal(layer.attn["fiedler_value"], cache["blocks.2.attn.fiedler_value"])
    head = layer.head(3)
    assert head["fiedler_value"] == pytest.approx(float(cache["blocks.2.attn.fiedler_value"][3]))
    assert head["resid.hfer"] == pytest.approx(float(cache["blocks.2.resid.hfer"][3]))
    with pytest.raises(KeyError, match="ambiguous"):
        head["hfer"]
    assert head.eigenvalues.shape == (11,)
    assert cache.layer(-1).index == 2
    assert "HeadSpectra(layer 2, head 3)" in repr(head)
    assert "graph.fiedler_value" in repr(layer)
    with pytest.raises(KeyError):
        cache.layer(7)


def test_batch_equals_single_runs(tiny_gpt2):
    batch = tiny_gpt2.run_with_spectra([TEXT, OTHER, "a dog"], batch_size=3)
    assert batch.n_texts == 3 and batch.is_batch
    assert batch.tokens[2] == ["a", "dog"]
    assert batch["blocks.0.attn.fiedler_value"].shape == (3, 4)
    assert isinstance(batch["blocks.0.graph.eigenvalues"], list)   # ragged lengths
    for i, text in enumerate([TEXT, OTHER, "a dog"]):
        one = tiny_gpt2.run_with_spectra(text)
        for key in one.keys():
            a, b = np.asarray(one[key], float), np.asarray(batch[i][key], float)
            if a.size and np.isfinite(a).all():
                _close(b, a, rtol=1e-4, atol=1e-5)
    small = tiny_gpt2.run_with_spectra([TEXT, OTHER], batch_size=1)
    _close(small["blocks.2.resid.energy"], batch["blocks.2.resid.energy"][:2], rtol=1e-4)


def test_token_ids_input(tiny_gpt2):
    ids = tiny_gpt2.tokenizer(TEXT)["input_ids"]
    a = tiny_gpt2.run_with_spectra(TEXT)
    b = tiny_gpt2.run_with_spectra(torch.tensor(ids))
    c = tiny_gpt2.run_with_spectra(ids)
    _close(a["blocks.1.graph.energy"], b["blocks.1.graph.energy"], rtol=1e-6)
    _close(a["blocks.1.attn.fiedler_value"], c["blocks.1.attn.fiedler_value"], rtol=1e-6)
    assert not c.is_batch
    assert tiny_gpt2.run_with_spectra(torch.tensor([ids, ids])).n_texts == 2


def test_head_subset_and_layers(tiny_gpt2):
    full = tiny_gpt2.run_with_spectra(TEXT)
    sub = tiny_gpt2.run_with_spectra(TEXT, heads=[(1, 2), (2, 0), (2, 3)], layers=[-2, 2])
    assert sub.layers == [1, 2]
    assert sub.heads == {1: [2], 2: [0, 3]}
    v = sub["blocks.2.attn.fiedler_value"]
    assert np.isnan(v[[1, 2]]).all()
    _close(v[[0, 3]], full["blocks.2.attn.fiedler_value"][[0, 3]], rtol=1e-6)
    _close(sub["blocks.1.resid.hfer"][2], full["blocks.1.resid.hfer"][2], rtol=1e-6)
    # the graph namespace averages every head regardless of heads=
    assert sub["blocks.2.graph.hfer"] == pytest.approx(full["blocks.2.graph.hfer"], rel=1e-6)
    with pytest.raises(KeyError, match="not computed"):
        sub.layer(2).head(1)
    with pytest.raises(ValueError, match="layers with attention"):
        tiny_gpt2.run_with_spectra(TEXT, layers=[5])


def test_token_span_is_the_induced_subgraph(tiny_gpt2):
    cache = tiny_gpt2.run_with_spectra(TEXT, token_span=(-4, None))
    assert cache.token_span == (7, 11)
    out = _forward(tiny_gpt2, TEXT)
    cfg = GSPConfig(device="cpu")
    ph = per_head_metrics(out.attentions[1], config=cfg, token_span=(7, 11), layer=1)
    _close(cache["blocks.1.attn.fiedler_value"], ph.metric("fiedler_value"), rtol=1e-6)
    gc, an = GraphConstructor(cfg), SpectralAnalyzer(cfg)
    adj = gc.aggregate_heads(gc.symmetrize_attention(out.attentions[1]))
    lap = gc.subgraph_laplacian(adj, list(range(7, 11)))[0]
    d = an.analyze_layer(out.hidden_states[2][0][7:11], lap, 1)
    assert cache["blocks.1.graph.hfer"] == pytest.approx(d.hfer, rel=1e-6)
    assert cache["blocks.1.attn.eigenvalues"].shape == (4, 4)


def test_metric_selection_and_attention_only(tiny_gpt2):
    cache = tiny_gpt2.run_with_spectra(TEXT, metrics=["fiedler_value", "graph.hfer"])
    assert cache.metric_names == ["attn.fiedler_value", "graph.fiedler_value", "graph.hfer"]
    assert "blocks.0.resid.energy" not in cache
    att = tiny_gpt2.run_with_spectra(TEXT, hybrid=False)
    assert all(f == "attention" for f in att.families.values())
    assert not any(q.startswith("resid.") for q in att.metric_names)
    assert "graph.hfer" not in att.metric_names and "graph.fiedler_value" in att.metric_names
    full = tiny_gpt2.run_with_spectra(TEXT)
    assert att["blocks.1.graph.fiedler_value"] == pytest.approx(full["blocks.1.graph.fiedler_value"])
    with pytest.raises(ValueError, match="hybrid=True"):
        tiny_gpt2.run_with_spectra(TEXT, metrics=["resid.hfer"], hybrid=False)
    with pytest.raises(ValueError, match="unknown metric"):
        tiny_gpt2.run_with_spectra(TEXT, metrics=["fidler"])


def test_dataframe_long_format(tiny_gpt2):
    cache = tiny_gpt2.run_with_spectra(TEXT)
    df = cache.to_dataframe()
    assert list(df.columns) == ["layer", "head", "metric", "family", "value"]
    per_head = sum(q.split(".")[0] in ("attn", "resid") for q in cache.metric_names)
    graph = sum(q.startswith("graph.") for q in cache.metric_names)
    assert len(df) == 3 * (4 * per_head + graph)
    assert df[df.metric == "graph.hfer"]["head"].isna().all()
    row = df[(df["layer"] == 1) & (df["head"] == 2) & (df["metric"] == "attn.fiedler_value")]
    assert row.value.item() == pytest.approx(float(cache["blocks.1.attn.fiedler_value"][2]))
    batch = tiny_gpt2.run_with_spectra([TEXT, OTHER])
    bdf = batch.to_dataframe(metrics=["attn"])
    assert list(bdf.columns)[0] == "text_id" and set(bdf.text_id) == {0, 1}
    sub = tiny_gpt2.run_with_spectra(TEXT, heads=[0])
    assert set(sub.to_dataframe(metrics=["attn"])["head"].dropna()) == {0}


def test_save_and_load_round_trip(tiny_gpt2, tmp_path):
    batch = tiny_gpt2.run_with_spectra([TEXT, OTHER], heads=[1, 3], token_span=(1, None))
    path = batch.save(tmp_path / "spectra")
    assert path.suffix == ".npz"
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["__metadata__"]).decode())
    assert meta["format"] == "spectral-trust.SpectralCache"
    back = SpectralCache.load(path)
    assert back.tokens == batch.tokens and back.layers == batch.layers
    assert back.heads == batch.heads and back.token_span == batch.token_span
    assert back.metric_names == batch.metric_names
    assert back.config.normalization == batch.config.normalization
    for key in batch.keys():
        a, b = batch[key], back[key]
        if isinstance(a, list):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_array_equal(a, b)
    assert back.to_dataframe().equals(batch.to_dataframe())


def test_from_pretrained_switches_to_eager():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=len(WORDS),
                                       n_positions=32))
    model.config._attn_implementation = "sdpa"
    sm = SpectralModel.from_pretrained(model, tokenizer=_tokenizer())
    assert sm.model.config._attn_implementation == "eager"
    assert sm.run_with_spectra(TEXT).layers == [0]
    assert "SpectralModel(" in repr(sm)


def test_other_normalisations_run(tiny_gpt2):
    for norm in ("rw", "none"):
        sm = SpectralModel(tiny_gpt2.model, tiny_gpt2.tokenizer,
                           config=GSPConfig(device="cpu", normalization=norm))
        cache = sm.run_with_spectra(TEXT)
        assert np.isfinite(cache["blocks.1.attn.fiedler_value"]).all()
        assert np.isfinite(cache["blocks.1.graph.energy"])


# ── hybrid models: true layer indices ────────────────────────────────────────

def test_hybrid_model_reports_true_layers(tiny_hybrid):
    cache = tiny_hybrid.run_with_spectra(TEXT)
    assert cache.layers == [1, 3]
    out = _forward(tiny_hybrid, TEXT)
    cfg = GSPConfig(device="cpu")
    gc, an = GraphConstructor(cfg), SpectralAnalyzer(cfg)
    for k, lay in enumerate([1, 3]):
        lap = gc.construct_laplacian(gc.aggregate_heads(gc.symmetrize_attention(out.attentions[k])))[0]
        d = an.analyze_layer(out.hidden_states[lay + 1][0], lap, lay)
        assert cache[f"blocks.{lay}.graph.energy"] == pytest.approx(d.energy, rel=1e-6)
        hy = per_head_hybrid_metrics(out.attentions[k], out.hidden_states[lay + 1], config=cfg)
        _close(cache[f"blocks.{lay}.resid.hfer"], hy.metric("hfer"), rtol=1e-6)
    with pytest.raises(ValueError, match=r"layers with attention: \[1, 3\]"):
        tiny_hybrid.run_with_spectra(TEXT, layers=[0])
    sub = tiny_hybrid.run_with_spectra(TEXT, layers=[-1], heads=[(3, 1)])
    assert sub.layers == [3] and sub.heads == {3: [1]}
    batch = tiny_hybrid.run_with_spectra([TEXT, OTHER])
    _close(batch[0]["blocks.3.graph.energy"], cache["blocks.3.graph.energy"], rtol=1e-4)


# ── from_hf_outputs ──────────────────────────────────────────────────────────

def test_from_hf_outputs_matches_run_with_spectra(tiny_gpt2, tiny_hybrid):
    for sm in (tiny_gpt2, tiny_hybrid):
        ref = sm.run_with_spectra(TEXT)
        out = _forward(sm, TEXT)
        ids = torch.tensor([sm.tokenizer(TEXT)["input_ids"]])
        got = st.from_hf_outputs(out, sm.model.config, ids, tokenizer=sm.tokenizer)
        assert got.layers == ref.layers and got.tokens == ref.tokens
        for key in ref.keys():
            _close(got[key], ref[key], rtol=1e-6, atol=1e-7)


def test_from_hf_outputs_crops_a_padded_batch(tiny_gpt2):
    tok = tiny_gpt2.tokenizer
    ids = [tok(t)["input_ids"] for t in (TEXT, OTHER)]
    width = max(map(len, ids))
    inp = torch.full((2, width), tok.eos_token_id)
    mask = torch.zeros(2, width, dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    with torch.no_grad():
        out = tiny_gpt2.model(input_ids=inp, attention_mask=mask, output_attentions=True,
                              output_hidden_states=True)
    got = st.from_hf_outputs(out, tiny_gpt2.model.config, inp, tokenizer=tok,
                             attention_mask=mask)
    ref = tiny_gpt2.run_with_spectra([TEXT, OTHER])
    assert got.tokens == ref.tokens
    # A padded batch changes the matmul shapes, so float32 sums accumulate in another
    # order: on Linux CI the hybrid energies differ by up to 1.7e-5 relative. Same
    # tolerance as the other padded-batch comparisons in this file.
    for key in ref.keys():
        for a, b in zip(got[key], ref[key]):
            _close(a, b, rtol=1e-4, atol=1e-5)


def test_a_left_padded_mask_is_refused(tiny_gpt2):
    out = _forward(tiny_gpt2, TEXT)
    mask = torch.ones(1, 11, dtype=torch.long)
    mask[0, :2] = 0
    with pytest.raises(ValueError, match="left-padded"):
        st.from_hf_outputs(out, tiny_gpt2.model.config, attention_mask=mask)


def test_attention_only_outputs():
    a = torch.rand(1, 2, 6, 6).tril()
    a = a / a.sum(-1, keepdim=True)
    cache = st.from_hf_outputs({"attentions": (a, a)})
    assert cache.layers == [0, 1]
    assert not any(q.startswith("resid.") for q in cache.metric_names)
    with pytest.raises(ValueError):
        st.from_hf_outputs({"attentions": (a,)}, hybrid=True)


# ── the core install stays light ─────────────────────────────────────────────

def test_import_does_not_load_interop_libraries():
    code = ("import sys, spectral_trust; "
            "print(any(m in sys.modules for m in ('transformer_lens', 'nnsight')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
