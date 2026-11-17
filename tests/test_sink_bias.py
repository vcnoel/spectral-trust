"""spectral_trust.interventions.sink_bias: a constant logit added to every
query's attention score toward the hub key, on tiny random Llama models
built offline (no download)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

import spectral_trust as st  # noqa: E402
from spectral_trust.interventions import _registry as reg, sink_bias  # noqa: E402

torch.set_num_threads(2)
V = 40


def _llama(seed=0, layers=3, impl="eager"):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                      attn_implementation=impl)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def model():
    return _llama(1)


def _ids(n=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.tensor([1]), torch.randint(3, V, (n - 1,), generator=g)])[None]


def _logits(model, ids, **kw):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False, **kw).logits


def _reference_bias(model, ids, layers, b, hub=0):
    """Reference: the score bias written into each attention module's forward
    by hand (a pre-hook that rebuilds the mask), independent of the library."""
    from transformers.models.llama import modeling_llama as ml
    orig = ml.eager_attention_forward
    attn = {id(model.model.layers[lay].self_attn) for lay in layers}

    def patched(module, q, k, v, mask, *a, **kw):
        if id(module) in attn:             # hub = 0: every query sees it
            mask = mask.clone()
            mask[..., hub] += b
        return orig(module, q, k, v, mask, *a, **kw)
    ml.eager_attention_forward = patched
    try:
        return _logits(model, ids)
    finally:
        ml.eager_attention_forward = orig


def test_zero_bias_is_the_identity(model):
    ids = _ids()
    ref = _logits(model, ids)
    with sink_bias(model, [0, 1, 2], 0.0) as iv:
        assert len(iv.edits) == 1 and not reg.MASK_EDITS
        assert torch.equal(_logits(model, ids), ref)
    # and on an SDPA model, which b = 0 leaves on SDPA
    sdpa = _llama(1, impl="sdpa")
    ref = _logits(sdpa, ids)
    with sink_bias(sdpa, 1, 0.0):
        assert torch.equal(_logits(sdpa, ids), ref)


def test_bias_matches_a_hand_written_score_bias_and_is_undone(model):
    ids = _ids()
    plain = _logits(model, ids)
    for b in (-3.0, 2.0):
        ref = _reference_bias(model, ids, [1], b)
        with sink_bias(model, [1], b):
            got = _logits(model, ids)
        assert torch.allclose(got, ref, atol=1e-5)
        assert not torch.allclose(got, plain, atol=1e-3)
    assert torch.equal(_logits(model, ids), plain)            # undone on exit
    assert not reg.MASK_EDITS
    with st.intervene(model) as iv:                           # composes with intervene
        iv.sink_bias(-1, -3.0)
        a = _logits(model, ids)
    assert torch.allclose(a, _reference_bias(model, ids, [2], -3.0), atol=1e-5)


def test_hub_mass_falls_monotonically_with_b(model):
    ids = _ids(20, seed=3)
    masses = []
    for b in (4.0, 2.0, 1.0, 0.0, -1.0, -2.0, -4.0, -8.0):
        with sink_bias(model, [0, 1, 2], b), torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=False)
        masses.append([float(a[0, :, 1:, 0].mean()) for a in out.attentions])
    for lay in range(3):
        col = [m[lay] for m in masses]
        assert all(x > y for x, y in zip(col, col[1:])), (lay, col)
    # a later hub: the first token keeps its mass at the edited layer's input
    with sink_bias(model, [0], -6.0, hub=3), torch.no_grad():
        out = model(input_ids=ids, output_attentions=True, use_cache=False)
    assert float(out.attentions[0][0, :, 4:, 3].max()) < 0.05


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)])
def test_cached_decoding_matches_full_pass_with_left_padding(dtype, tol):
    m = _llama(2).to(dtype)
    short, long_ = _ids(9, seed=4), _ids(13, seed=5)
    ids = torch.cat([torch.cat([torch.zeros(1, 4, dtype=torch.long), short], 1), long_])
    mask = torch.ones_like(ids)
    mask[0, :4] = 0
    with sink_bias(m, [0, 2], -2.5), torch.no_grad():
        gen = m.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                         output_logits=True, return_dict_in_generate=True, pad_token_id=0)
        full_mask = torch.cat([mask, torch.ones(2, 6, dtype=torch.long)], 1)
        full = m(input_ids=gen.sequences, attention_mask=full_mask, use_cache=False).logits
        alone = m(input_ids=torch.cat([short, gen.sequences[:1, 13:]], 1), use_cache=False).logits
    S = ids.shape[1]
    for t, step in enumerate(gen.logits):                     # step t predicts token S + t
        ref = full[:, S - 1 + t].float()
        scale = float(ref.abs().max())
        assert torch.allclose(step.float(), ref, atol=tol * scale, rtol=tol), t
    # the hub is the first real token of the padded row: same as without padding
    assert torch.allclose(full[0, 4:].float(), alone[0].float(),
                          atol=tol * float(alone.abs().max()), rtol=tol)


def test_negative_hub_raises(model):
    with pytest.raises(ValueError):
        sink_bias(model, [0], -1.0, hub=-1)
