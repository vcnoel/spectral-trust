# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Adapters: spectral measurements from tensors that another library already
produced, with no second forward pass.

- :func:`from_transformer_lens` reads a TransformerLens ``ActivationCache``
  (``HookedTransformer`` or ``TransformerBridge``).
- :func:`from_nnsight` takes attention weights saved inside an nnsight
  trace, and optionally hidden states.
- :func:`from_hf_outputs` takes the output of a Hugging Face model called
  with ``output_attentions=True``.

Each returns the same :class:`~spectral_trust.cache.SpectralCache` as
:meth:`SpectralModel.run_with_spectra`, through the same computation, so the
values agree with it to float32 rounding when the tensors agree. Neither
TransformerLens nor nnsight is imported by this module: the adapters read
tensors and duck-typed attributes only.
"""
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .cache import SpectralCache
from .config import GSPConfig
from .framework import attention_layer_indices
from .model import _as_tensor, _layer_map, _num_layers, compute_spectra
from .per_head import HeadSelection

_PATTERN = re.compile(r"^blocks\.(\d+)\.attn\.hook_pattern$")


def _tokens_from(tokens, tokenizer, batch: int, seq: int, to_str_tokens=None,
                 lengths: Optional[List[int]] = None
                 ) -> Tuple[List[List[str]], Optional[List[List[int]]]]:
    """Token strings (and ids, when known) for ``batch`` sequences, cropped
    to ``lengths`` (the real length of each item of a right-padded batch).

    ``tokens`` is None (tokens labelled by position), text or a list of
    texts (tokenised with ``tokenizer``), token strings for one sequence or
    one list per sequence, or ids (``[seq]``, ``[batch, seq]``).
    """
    want = list(lengths) if lengths is not None else [seq] * batch
    if tokens is None:
        return [[f"[{i}]" for i in range(n)] for n in want], None
    if isinstance(tokens, str):
        tokens = [tokens]
    from_text = False
    if torch.is_tensor(tokens):
        t = tokens if tokens.dim() == 2 else tokens.unsqueeze(0)
        ids = [[int(i) for i in row] for row in t.tolist()]
    elif all(isinstance(t, str) for t in tokens):
        if batch == 1 and len(tokens) == want[0] and len(tokens) > 1:
            return [list(tokens)], None                      # token strings
        if tokenizer is None:
            raise ValueError("text input needs the model's tokenizer; pass token "
                             "strings or ids instead")
        if len(tokens) != batch:
            raise ValueError(f"{len(tokens)} texts for a batch of {batch}")
        ids = [_encode_to_length(tokenizer, t, n) for t, n in zip(tokens, want)]
        from_text = True
    elif all(isinstance(t, int) for t in tokens):
        ids = [[int(i) for i in tokens]]
    elif isinstance(tokens[0], (list, tuple)) and tokens[0] and isinstance(tokens[0][0], str):
        return [list(t)[:n] for t, n in zip(tokens, want)], None
    else:
        ids = [[int(i) for i in row] for row in tokens]
    if len(ids) != batch:
        raise ValueError(f"{len(ids)} token sequences for a batch of {batch}")
    if not from_text:
        if any(len(row) != seq for row in ids):
            raise ValueError(f"the token ids have {[len(r) for r in ids]} tokens but the "
                             f"attention covers {seq}; pass the ids that were run")
        ids = [row[:n] for row, n in zip(ids, want)]
    if to_str_tokens is not None:
        strs = [list(to_str_tokens(torch.tensor(row))) for row in ids]
    elif tokenizer is not None:
        strs = [[tokenizer.decode([i]) for i in row] for row in ids]
    else:
        strs = [[str(i) for i in row] for row in ids]
    return strs, ids


def _encode_to_length(tokenizer, text: str, seq: int) -> List[int]:
    """Token ids of ``text`` as they were run: with the tokenizer's special
    tokens, without them, or with a BOS token prepended (TransformerLens'
    ``prepend_bos``), whichever gives ``seq`` tokens."""
    full = list(tokenizer(text)["input_ids"])
    plain = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    bos = getattr(tokenizer, "bos_token_id", None)
    candidates = [full, plain] + ([[bos] + plain] if bos is not None else [])
    for ids in candidates:
        if len(ids) == seq:
            return ids
    raise ValueError(f"the text has {len(plain)} tokens ({len(full)} with special tokens) "
                     f"but the attention covers {seq}; pass the token ids that were run")


def _lengths_from_mask(attention_mask) -> Optional[List[int]]:
    """Real length of each item of a right-padded batch. A left-padded mask
    raises: cropping keeps the first ``n`` tokens of each item."""
    if attention_mask is None:
        return None
    mask = torch.as_tensor(_as_tensor(attention_mask)).to(torch.bool)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    lengths = mask.sum(-1).tolist()
    for row, n in zip(mask, lengths):
        if not bool(row[:n].all()):
            raise ValueError("the batch is left-padded; pad on the right (tokenizer."
                             "padding_side = 'right') or pass one sequence at a time")
    return [int(n) for n in lengths]


def _cache_get(cache, key):
    try:
        return cache[key]
    except (KeyError, IndexError):
        return None


def from_transformer_lens(
    cache,
    model=None,
    tokens=None,
    *,
    final_norm: bool = True,
    attention_mask=None,
    config: Optional[GSPConfig] = None,
    metrics=None,
    layers: Optional[Sequence[int]] = None,
    heads: HeadSelection = None,
    token_span=None,
    hybrid: Optional[bool] = None,
    keep_patterns: Optional[bool] = None,
) -> SpectralCache:
    """Spectral measurements from a TransformerLens ``ActivationCache``.

    Reads ``blocks.{l}.attn.hook_pattern`` (``[batch, heads, query, key]``)
    for every layer, and ``blocks.{l}.hook_resid_post`` when present, which
    is the signal paired with layer l for the hybrid metrics.

    Examples:
        >>> logits, tl_cache = model.run_with_cache(tokens)
        >>> cache = st.from_transformer_lens(tl_cache, model, tokens)

    Args:
        cache: the ``ActivationCache`` of ``run_with_cache`` (any mapping
            with these keys works).
        model: the ``HookedTransformer`` or ``TransformerBridge``, used for
            the token strings (``to_str_tokens``), the model name and the
            number of layers. Defaults to ``cache.model`` when the cache has
            one.
        tokens: the tokens that were run: ids (``[batch, seq]`` or
            ``[seq]``), text (tokenised with the model's tokenizer), or token
            strings. Without it, tokens are labelled by position.
        final_norm: pair the last layer with ``ln_final.hook_out`` when the
            cache holds it (``TransformerBridge`` caches do). That is the
            tensor Hugging Face returns as its last hidden state, so the
            hybrid metrics equal :meth:`SpectralModel.run_with_spectra` on
            every layer. With False, or when the hook is absent, the last
            layer is paired with its ``hook_resid_post``, the residual stream
            before the final norm. ``cache.meta["signal_source"]`` records
            the choice.
        attention_mask: ``[batch, seq]`` mask of a right-padded batch; each
            item is cropped to its real length. Without it every item is
            measured over the whole sequence.
        config, metrics, layers, heads, token_span, keep_patterns: as in
            :meth:`SpectralModel.run_with_spectra`.
        hybrid: compute the hybrid metrics; None (default) does when the
            cache holds ``hook_resid_post``.

    Notes:
        Attention patterns do not depend on TransformerLens' weight
        processing, so the attention metrics match a Hugging Face run of the
        same model. The residual stream does: ``center_writing_weights``
        (on by default in ``HookedTransformer.from_pretrained``) removes its
        mean. For hybrid metrics equal to a Hugging Face run, load with
        ``HookedTransformer.from_pretrained_no_processing`` or use
        ``TransformerBridge`` without compatibility mode.
    """
    if model is None:
        model = getattr(cache, "model", None)
    keys = list(cache.keys())
    layer_ids = sorted(int(m.group(1)) for k in keys if (m := _PATTERN.match(k)))
    if not layer_ids:
        raise ValueError("the cache holds no blocks.{l}.attn.hook_pattern; run_with_cache "
                         "must keep the attention patterns")
    attentions = {lay: _as_tensor(cache[f"blocks.{lay}.attn.hook_pattern"]) for lay in layer_ids}
    cfg = getattr(model, "cfg", None)
    n_layers = getattr(cfg, "n_layers", None) or (layer_ids[-1] + 1)

    resid = {lay: _cache_get(cache, f"blocks.{lay}.hook_resid_post") for lay in layer_ids}
    have_resid = all(v is not None for v in resid.values())
    if hybrid is None:
        hybrid = have_resid
    signals = None
    sources: Dict[int, str] = {}
    if hybrid:
        if not have_resid:
            missing = [lay for lay, v in resid.items() if v is None]
            raise ValueError(f"no blocks.{{l}}.hook_resid_post for layers {missing}; "
                             "pass hybrid=False for the attention metrics only")
        signals = {lay: _as_tensor(v) for lay, v in resid.items()}
        sources = {lay: f"blocks.{lay}.hook_resid_post" for lay in layer_ids}
        last = n_layers - 1
        final = _cache_get(cache, "ln_final.hook_out") if final_norm else None
        if final is not None and last in signals:
            signals[last] = _as_tensor(final)
            sources[last] = "ln_final.hook_out"

    first = attentions[layer_ids[0]]
    batch = first.shape[0] if first.dim() == 4 else 1
    seq = first.shape[-1]
    tokenizer = getattr(model, "tokenizer", None)
    to_str = getattr(model, "to_str_tokens", None)
    lengths = _lengths_from_mask(attention_mask)
    strs, ids = _tokens_from(tokens, tokenizer, batch, seq,
                             to_str_tokens=to_str if callable(to_str) else None,
                             lengths=lengths)
    name = getattr(cfg, "model_name", None) or getattr(cfg, "tokenizer_name", None)
    meta = {"model_name": name, "source": "from_transformer_lens",
            "token_ids": ids if ids is not None else [None] * batch,
            "texts": [None] * batch,
            "signal_source": {str(k): v for k, v in sources.items()} or None}
    return compute_spectra(attentions, signals, tokens=strs, lengths=lengths,
                           n_layers=n_layers, config=config, metrics=metrics, layers=layers,
                           heads=heads, token_span=token_span, hybrid=hybrid,
                           keep_patterns=keep_patterns, meta=meta)


def from_nnsight(
    attentions: Union[Sequence[Any], Mapping[int, Any]],
    hidden_states: Optional[Union[Sequence[Any], Mapping[int, Any]]] = None,
    *,
    model=None,
    tokens=None,
    layer_indices: Optional[Sequence[int]] = None,
    final_norm=None,
    attention_mask=None,
    config: Optional[GSPConfig] = None,
    metrics=None,
    layers: Optional[Sequence[int]] = None,
    heads: HeadSelection = None,
    token_span=None,
    hybrid: Optional[bool] = None,
    keep_patterns: Optional[bool] = None,
) -> SpectralCache:
    """Spectral measurements from tensors saved in an nnsight trace.

    The model must run eager attention, so that its attention modules return
    their weights. For GPT-2:

    Examples:
        >>> from nnsight import LanguageModel
        >>> lm = LanguageModel("gpt2", attn_implementation="eager", dispatch=True)
        >>> attn, resid = [], []
        >>> with lm.trace(prompt):
        ...     for block in lm.transformer.h:
        ...         attn.append(block.attn.output[1].save())   # weights [1, H, S, S]
        ...         resid.append(block.output.save())          # block output [1, S, d]
        >>> cache = st.from_nnsight(attn, resid, model=lm, tokens=prompt)

    The module paths are the Hugging Face ones (``model.layers[i].self_attn``
    for Llama, Qwen and Mistral). Modules must be read in execution order
    inside the trace, hence one loop over the blocks. ``lm.output`` with
    ``output_attentions=True, output_hidden_states=True`` works as well:
    pass its ``.attentions`` and ``.hidden_states``.

    Args:
        attentions: one saved attention tensor per attention layer,
            ``[batch, heads, seq, seq]``, as a list (layer order) or a
            ``{layer: tensor}`` dict. Saved values of any nnsight version are
            accepted (tensors, or proxies with ``.value``).
        hidden_states: optional signals for the hybrid metrics: one per
            attention layer (block outputs, paired with the same layer), or
            ``n_layers + 1`` of them in the Hugging Face convention
            (embeddings first; layer l is paired with entry l + 1), or a
            ``{layer: tensor}`` dict. Tuples (module outputs) are reduced to
            their first element.
        model: the nnsight ``LanguageModel`` (or the Hugging Face model): its
            config gives the layer layout and the name, its tokenizer the
            token strings.
        tokens: the input that was traced (text, ids or token strings).
        layer_indices: true layer index of each entry of ``attentions`` when
            they are a list; default from the model config, as in
            :func:`~spectral_trust.framework.attention_layer_indices`.
        final_norm: optional saved output of the final norm
            (``lm.transformer.ln_f.output``), paired with the last layer in
            place of its block output. That is what Hugging Face returns as
            the last hidden state, so the hybrid metrics of the last layer
            then equal :meth:`SpectralModel.run_with_spectra`.
        attention_mask: ``[batch, seq]`` mask of a right-padded batch; each
            item is cropped to its real length.
        config, metrics, layers, heads, token_span, keep_patterns: as in
            :meth:`SpectralModel.run_with_spectra`.
        hybrid: compute the hybrid metrics; None (default) does when
            ``hidden_states`` are given.
    """
    hf_model = getattr(model, "_model", None) or model
    model_config = getattr(hf_model, "config", None) or getattr(model, "config", None)
    n_layers = _num_layers(model_config) if model_config is not None else None

    if isinstance(attentions, Mapping):
        attn_by_layer = {int(k): _as_tensor(v) for k, v in attentions.items()}
    else:
        attn_list = [None if a is None else _as_tensor(a) for a in attentions]
        if layer_indices is not None:
            idx = [int(i) for i in layer_indices]
            if len(idx) != len(attn_list):
                raise ValueError(f"layer_indices has {len(idx)} entries for "
                                 f"{len(attn_list)} attention tensors")
        else:
            n = n_layers if n_layers is not None else len(attn_list)
            cfg = config or GSPConfig()
            idx = _layer_map(model_config, attn_list, [None] * (n + 1), cfg)
        attn_by_layer = {i: a for i, a in zip(idx, attn_list) if a is not None}
    order = sorted(attn_by_layer)
    if n_layers is None:
        n_layers = order[-1] + 1

    signals = None
    sources: Dict[int, str] = {}
    if hidden_states is not None:
        if isinstance(hidden_states, Mapping):
            signals = {int(k): _as_tensor(v) for k, v in hidden_states.items()}
            sources = {k: "hidden_states[layer]" for k in signals}
        else:
            hs = [_as_tensor(h) for h in hidden_states]
            if len(hs) == n_layers + 1:
                signals = {lay: hs[lay + 1] for lay in order}
                sources = {lay: "hidden_states[layer + 1]" for lay in order}
            elif len(hs) == len(order):
                signals = dict(zip(order, hs))
                sources = {lay: "block output" for lay in order}
            elif len(hs) == n_layers:
                signals = {lay: hs[lay] for lay in order}
                sources = {lay: "block output" for lay in order}
            else:
                raise ValueError(
                    f"{len(hs)} hidden states for {len(order)} attention layers of a "
                    f"{n_layers}-layer model: pass one per attention layer, one per layer, "
                    f"or n_layers + 1 (Hugging Face convention)")
        if final_norm is not None and (n_layers - 1) in signals:
            signals[n_layers - 1] = _as_tensor(final_norm)
            sources[n_layers - 1] = "final norm output"
    if hybrid is None:
        hybrid = signals is not None

    first = attn_by_layer[order[0]]
    batch = first.shape[0] if first.dim() == 4 else 1
    seq = first.shape[-1]
    tokenizer = getattr(model, "tokenizer", None)
    lengths = _lengths_from_mask(attention_mask)
    strs, ids = _tokens_from(tokens, tokenizer, batch, seq, lengths=lengths)
    name = getattr(model_config, "_name_or_path", None)
    meta = {"model_name": name, "source": "from_nnsight",
            "token_ids": ids if ids is not None else [None] * batch,
            "texts": [tokens] if isinstance(tokens, str) else [None] * batch,
            "signal_source": {str(k): v for k, v in sources.items()} or None}
    return compute_spectra(attn_by_layer, signals if hybrid else None, tokens=strs,
                           lengths=lengths, n_layers=n_layers, config=config,
                           metrics=metrics, layers=layers, heads=heads,
                           token_span=token_span, hybrid=hybrid,
                           keep_patterns=keep_patterns, meta=meta)


def from_hf_outputs(
    outputs,
    model_config=None,
    tokens=None,
    *,
    tokenizer=None,
    attention_mask=None,
    config: Optional[GSPConfig] = None,
    metrics=None,
    layers: Optional[Sequence[int]] = None,
    heads: HeadSelection = None,
    token_span=None,
    hybrid: Optional[bool] = None,
    keep_patterns: Optional[bool] = None,
) -> SpectralCache:
    """Spectral measurements from a Hugging Face model output.

    Examples:
        >>> out = model(**inputs, output_attentions=True, output_hidden_states=True)
        >>> cache = st.from_hf_outputs(out, model.config, inputs["input_ids"], tokenizer=tok)

    Args:
        outputs: a ``ModelOutput`` (or a dict) with ``attentions`` and,
            for the hybrid metrics, ``hidden_states``.
        model_config: the model config, which locates the attention layers
            of hybrid models (``layer_types``); see
            :func:`~spectral_trust.framework.attention_layer_indices`.
            ``config.attention_layer_map`` overrides it.
        tokens: the input ids (or text, or token strings).
        tokenizer: decodes ``tokens`` into token strings.
        attention_mask: ``[batch, seq]`` mask of a right-padded batch; each
            item is cropped to its real length.
        config, metrics, layers, heads, token_span, keep_patterns: as in
            :meth:`SpectralModel.run_with_spectra`.
        hybrid: compute the hybrid metrics; None (default) does when the
            output holds hidden states.
    """
    get = outputs.get if isinstance(outputs, Mapping) else (lambda k: getattr(outputs, k, None))
    attn = list(get("attentions") or [])
    hidden = list(get("hidden_states") or [])
    if not attn or all(a is None for a in attn):
        raise ValueError("the output holds no attention weights; call the model with "
                         "output_attentions=True and eager attention")
    cfg = config or GSPConfig()
    if hidden:
        layer_map = _layer_map(model_config, attn, hidden, cfg)
    else:
        n = _num_layers(model_config) if model_config is not None else None
        layer_map = _layer_map(model_config, attn, [None] * ((n or len(attn)) + 1), cfg)
    attentions = {layer_map[k]: a for k, a in enumerate(attn) if a is not None}
    if hybrid is None:
        hybrid = bool(hidden)
    signals = {lay: hidden[lay + 1] for lay in attentions} if hybrid and hidden else None
    if hybrid and signals is None:
        raise ValueError("the hybrid metrics need output_hidden_states=True")
    first = _as_tensor(attn[min(k for k, a in enumerate(attn) if a is not None)])
    batch = first.shape[0] if first.dim() == 4 else 1
    seq = first.shape[-1]
    lengths = _lengths_from_mask(attention_mask)
    strs, ids = _tokens_from(tokens, tokenizer, batch, seq, lengths=lengths)
    n_layers = (len(hidden) - 1) if hidden else (_num_layers(model_config)
                                                if model_config is not None else None)
    meta = {"model_name": getattr(model_config, "_name_or_path", None),
            "source": "from_hf_outputs",
            "token_ids": ids if ids is not None else [None] * batch,
            "texts": [tokens] if isinstance(tokens, str) else [None] * batch,
            "signal_source": "hidden_states[layer + 1]" if signals else None}
    return compute_spectra(attentions, signals, tokens=strs, lengths=lengths,
                           n_layers=n_layers, config=config, metrics=metrics, layers=layers,
                           heads=heads, token_span=token_span, hybrid=hybrid,
                           keep_patterns=keep_patterns, meta=meta)


__all__ = ["from_transformer_lens", "from_nnsight", "from_hf_outputs",
           "attention_layer_indices"]
