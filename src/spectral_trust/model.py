# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
High-level API: :class:`SpectralModel` and the computation shared by
:meth:`SpectralModel.run_with_spectra` and the adapters in
:mod:`spectral_trust.interop`.

>>> import spectral_trust as st
>>> model = st.SpectralModel.from_pretrained("gpt2")
>>> cache = model.run_with_spectra("The cat sat on the mat.")
>>> cache["blocks.10.attn.fiedler_value"]      # one value per head
>>> cache["blocks.10.graph.hfer"]              # head-averaged graph

Every number comes from the library's existing functions:
:func:`~spectral_trust.per_head.per_head_metrics` (namespace ``attn``),
:func:`~spectral_trust.per_head.per_head_hybrid_metrics` (``resid``) and
:meth:`~spectral_trust.spectral.SpectralAnalyzer.analyze_layer` on the
head-averaged graph that ``GSPDiagnosticsFramework.analyze_text`` builds
(``graph``). Layer l is paired with ``hidden_states[l + 1]``, as in
``analyze_text``, and layer indices are the model's own, also on hybrid
models (:func:`~spectral_trust.framework.attention_layer_indices`).
"""
import logging
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .cache import SpectralCache, resolve_metrics
from .config import GSPConfig
from .framework import attention_layer_indices
from .graph import GraphConstructor
from .instrumentation import assert_finite_attention
from .per_head import (
    HeadSelection, _resolve_heads, per_head_hybrid_metrics, per_head_metrics,
)
from .spectral import SpectralAnalyzer

logger = logging.getLogger(__name__)

#: Patterns are kept in the cache (for the graph view) when they take at
#: most this many bytes in float32; see ``keep_patterns``.
PATTERN_BUDGET_BYTES = 64 * 2 ** 20

# Per-head eigendecompositions of several layers are stacked into one
# batched call up to this many float32 bytes of Laplacians.
_STACK_BUDGET_BYTES = 128 * 2 ** 20

TextInput = Union[str, Sequence[str], torch.Tensor, Sequence[Sequence[int]]]


def default_config() -> GSPConfig:
    """``GSPConfig`` defaults for the high-level API, on the CPU, with no
    plots or files."""
    return GSPConfig(device="cpu", save_plots=False, verbose=False,
                     save_intermediate=True)


def _resolve_span(token_span, n: int) -> Tuple[int, int]:
    """``(start, end)`` with Python slice semantics (negative values and
    None allowed), clipped to a sequence of ``n`` tokens."""
    if token_span is None:
        return 0, n
    if isinstance(token_span, slice):
        start, end = token_span.start, token_span.stop
    else:
        start, end = token_span
    s, e, _ = slice(start, end).indices(n)
    if e <= s:
        raise ValueError(f"token_span {token_span} is empty for a sequence of {n} tokens")
    return s, e


def _as_tensor(x) -> torch.Tensor:
    value = getattr(x, "value", x)            # nnsight < 0.4 proxies
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        value = value[0]                      # module outputs (hidden, ...)
    if not torch.is_tensor(value):
        value = torch.as_tensor(np.asarray(value))
    return value


def _batched(t: torch.Tensor, dims: int) -> torch.Tensor:
    """Add the batch dimension to a ``dims - 1``-dimensional tensor."""
    t = _as_tensor(t)
    if t.dim() == dims - 1:
        t = t.unsqueeze(0)
    if t.dim() != dims:
        raise ValueError(f"expected a {dims}-dimensional tensor (batch first), "
                         f"got shape {tuple(t.shape)}")
    return t


def _fiedler_vector(eigenvectors) -> Optional[np.ndarray]:
    if eigenvectors is None or eigenvectors.shape[1] < 2:
        return None
    u = np.asarray(eigenvectors[:, 1], dtype=np.float64)
    if u[np.argmax(np.abs(u))] < 0:           # sign convention: largest entry positive
        u = -u
    return u.astype(np.float32)


def compute_spectra(
    attentions: Mapping[int, torch.Tensor],
    signals: Optional[Mapping[int, torch.Tensor]] = None,
    *,
    tokens: Sequence[Sequence[str]],
    lengths: Optional[Sequence[int]] = None,
    n_layers: Optional[int] = None,
    config: Optional[GSPConfig] = None,
    metrics=None,
    layers: Optional[Sequence[int]] = None,
    heads: HeadSelection = None,
    token_span=None,
    hybrid: Optional[bool] = None,
    keep_patterns: Optional[bool] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> SpectralCache:
    """
    Spectral measurements from attention tensors (and optionally the signals
    paired with each layer), as a :class:`SpectralCache`.

    This is the computation behind :meth:`SpectralModel.run_with_spectra` and
    every adapter; call it directly for tensors from any other source.

    Args:
        attentions: true layer index -> attention weights ``[batch, heads,
            seq, seq]`` (or ``[heads, seq, seq]``), row i the query.
        signals: true layer index -> the residual-stream states paired with
            that layer, ``[batch, seq, d]``. Needed for the hybrid metrics.
        tokens: one list of token strings per batch item.
        lengths: real length of each batch item (right padding is cropped);
            default the full sequence.
        n_layers: number of layers of the model, for negative ``layers``.
        config: graph construction (``GSPConfig``); default
            :func:`default_config`.
        metrics, layers, heads, token_span, hybrid, keep_patterns: as in
            :meth:`SpectralModel.run_with_spectra`. ``hybrid=None`` computes
            the hybrid metrics when ``signals`` are given.
        meta: extra metadata stored in the cache.
    """
    cfg = config or default_config()
    if not cfg.save_intermediate:
        cfg = replace(cfg, save_intermediate=True)   # the graph view needs eigenvectors
    attentions = {int(k): _batched(v, 4) for k, v in attentions.items() if v is not None}
    if not attentions:
        raise ValueError("no attention tensors were given")
    if signals is not None:
        signals = {int(k): _batched(v, 3) for k, v in signals.items() if v is not None}
    if hybrid is None:
        hybrid = bool(signals)
    if hybrid and not signals:
        raise ValueError("the hybrid metrics need hidden states; pass them, or hybrid=False")

    available = sorted(attentions)
    n_layers = int(n_layers) if n_layers is not None else available[-1] + 1
    if layers is None:
        chosen = available
    else:
        if isinstance(layers, (int, np.integer)):
            layers = [int(layers)]
        chosen = []
        for lay in layers:
            lay = int(lay)
            if lay < 0:
                lay += n_layers
            if lay not in attentions:
                raise ValueError(
                    f"layer {lay} has no attention weights; layers with attention: "
                    f"{available}")
            if lay not in chosen:
                chosen.append(lay)
        chosen.sort()
    if hybrid:
        missing = [lay for lay in chosen if lay not in signals]
        if missing:
            raise ValueError(f"no hidden states for layers {missing}")

    selected = resolve_metrics(metrics, hybrid=hybrid)
    attn_names = [q.split(".", 1)[1] for q in selected if q.startswith("attn.")]
    resid_names = [q.split(".", 1)[1] for q in selected if q.startswith("resid.")]
    graph_names = [q.split(".", 1)[1] for q in selected if q.startswith("graph.")]

    first = attentions[available[0]]
    batch, n_heads = int(first.shape[0]), int(first.shape[1])
    if len(tokens) != batch:
        raise ValueError(f"{len(tokens)} token lists for a batch of {batch}")
    lengths = [int(n) for n in lengths] if lengths is not None else [int(first.shape[-1])] * batch

    if keep_patterns is None:
        need = sum(4 * n_heads * n * n for n in lengths) * len(chosen)
        keep_patterns = need <= PATTERN_BUDGET_BYTES
        if not keep_patterns:
            logger.info("attention patterns (%.0f MB) are not kept in the cache; pass "
                        "keep_patterns=True for the graph view", need / 2 ** 20)

    head_lists = {lay: _resolve_heads(heads, lay, n_heads) for lay in chosen}
    gc = GraphConstructor(cfg)
    analyzer = SpectralAnalyzer(cfg)

    records: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for b in range(batch):
        n = lengths[b]
        s, e = _resolve_span(token_span, n)
        spans.append((s, e))
        full = (s, e) == (0, n)
        rec: Dict[str, Any] = {}
        layer_attn: Dict[int, torch.Tensor] = {}
        for lay in chosen:
            a = attentions[lay][b, :, :n, :n].detach().to(device="cpu", dtype=torch.float32)
            assert_finite_attention([a])
            layer_attn[lay] = a
            if keep_patterns:
                rec[f"blocks.{lay}.attn.pattern"] = a.numpy().copy()

        if attn_names:
            _per_head_attention(rec, layer_attn, chosen, head_lists, n_heads, (s, e),
                                cfg, attn_names)

        for lay in chosen:
            a = layer_attn[lay]
            x = None
            if hybrid:
                x = signals[lay][b, :n].detach().to(device="cpu", dtype=torch.float32)
            if resid_names:
                ph = per_head_hybrid_metrics(a, x, config=cfg,
                                             token_span=None if full else (s, e),
                                             layer=lay, heads=head_lists[lay])
                rows = ph.heads if ph.heads is not None else list(range(n_heads))
                for name in resid_names:
                    col = ph.metric_names.index(name)
                    v = np.full(n_heads, np.nan, dtype=np.float32)
                    v[rows] = ph.values[:, col]
                    rec[f"blocks.{lay}.resid.{name}"] = v
            if graph_names:
                sym = gc.symmetrize_attention(a.unsqueeze(0))             # [1, H, S, S]
                adj = gc.aggregate_heads(sym)                             # [1, S, S]
                if full:
                    lap = gc.construct_laplacian(adj)[0]
                else:
                    lap = gc.subgraph_laplacian(adj, list(range(s, e)))[0]
                sig = x[s:e] if x is not None else torch.zeros(e - s, 1)
                diag = analyzer.analyze_layer(sig, lap, lay)
                for name in graph_names:
                    rec[f"blocks.{lay}.graph.{name}"] = float(getattr(diag, name))
                rec[f"blocks.{lay}.graph.eigenvalues"] = np.asarray(diag.eigenvalues,
                                                                     dtype=np.float32)
                fv = _fiedler_vector(diag.eigenvectors)
                if fv is not None:
                    rec[f"blocks.{lay}.graph.fiedler_vector"] = fv
                if hybrid:
                    rec[f"blocks.{lay}.graph.spectral_masses"] = np.asarray(
                        diag.spectral_masses, dtype=np.float32)
        records.append(rec)

    all_heads = all(v is None for v in head_lists.values())
    info = dict(meta or {})
    info.update({
        "heads": None if all_heads else {lay: (v if v is not None else list(range(n_heads)))
                                         for lay, v in head_lists.items()},
        "token_span": None if token_span is None else [list(sp) for sp in spans],
        "n_layers": n_layers,
        "hybrid": bool(hybrid),
        "patterns_kept": bool(keep_patterns),
    })
    info.setdefault("batched", batch > 1)
    return SpectralCache(records, tokens, chosen, n_heads, selected, cfg, info)


def _per_head_attention(rec, layer_attn, chosen, head_lists, n_heads, span, cfg, names):
    """Per-head attention metrics of every chosen layer, with the heads of
    several layers stacked into one batched eigendecomposition."""
    s, e = span
    blocks, owners = [], []
    for lay in chosen:
        sel = head_lists[lay]
        a = layer_attn[lay][:, s:e, s:e]
        if sel is not None:
            a = a.index_select(0, torch.as_tensor(sel, dtype=torch.long))
        blocks.append(a)
        owners.append((lay, sel if sel is not None else list(range(n_heads))))
    size = (e - s) ** 2 * 4
    start = 0
    while start < len(blocks):
        stop, used = start, 0
        while stop < len(blocks) and (stop == start or
                                      used + blocks[stop].shape[0] * size <= _STACK_BUDGET_BYTES):
            used += blocks[stop].shape[0] * size
            stop += 1
        stacked = torch.cat(blocks[start:stop], dim=0)
        ph = per_head_metrics(stacked, config=cfg, keep_eigenvalues=True)
        row = 0
        for lay, rows in owners[start:stop]:
            k = len(rows)
            vals = ph.values[row:row + k]
            for name in names:
                v = np.full(n_heads, np.nan, dtype=np.float32)
                v[rows] = vals[:, ph.metric_names.index(name)]
                rec[f"blocks.{lay}.attn.{name}"] = v
            eig = np.full((n_heads, e - s), np.nan, dtype=np.float32)
            eig[rows] = ph.eigenvalues[row:row + k]
            rec[f"blocks.{lay}.attn.eigenvalues"] = eig
            row += k
        start = stop


def merge_caches(parts: Sequence[SpectralCache], batched: bool) -> SpectralCache:
    """Concatenate the texts of several caches computed with the same
    settings (as :meth:`SpectralModel.run_with_spectra` does per batch)."""
    if len(parts) == 1 and not batched:
        return parts[0]
    first = parts[0]
    records, tokens = [], []
    meta = dict(first.meta)
    for k in ("texts", "token_ids"):
        meta[k] = [x for p in parts for x in p.meta.get(k, [None] * p.n_texts)]
    spans = [p.meta.get("token_span") for p in parts]
    meta["token_span"] = None if spans[0] is None else [s for sp in spans for s in sp]
    for p in parts:
        records.extend(p._records)
        tokens.extend(p._tokens)
    meta["batched"] = batched
    return SpectralCache(records, tokens, first.layers, first.n_heads, first.metric_names,
                         first.config, meta)


def _num_heads(config) -> Optional[int]:
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for name in ("num_attention_heads", "n_head", "n_heads"):
            v = getattr(cfg, name, None)
            if isinstance(v, int):
                return v
    return None


def _num_layers(config) -> Optional[int]:
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for name in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
            v = getattr(cfg, name, None)
            if isinstance(v, int):
                return v
    return None


class SpectralModel:
    """A Hugging Face causal LM that returns spectral measurements of its
    attention graphs, in the style of TransformerLens' ``HookedTransformer``.

    Examples:
        >>> model = SpectralModel.from_pretrained("Qwen/Qwen3-0.6B")
        >>> cache = model.run_with_spectra("The cat sat on the mat because it was warm.")
        >>> cache.layer(10).head(3)

    The model is used as it is: no weights are changed and no hooks are
    added; it is put in eval mode. Its attention must be the eager
    implementation, the only one that returns attention weights;
    :meth:`from_pretrained` loads it that way and switches an already-loaded
    model to it.

    Attributes:
        model: the wrapped ``PreTrainedModel``.
        tokenizer: its tokenizer.
        config: the ``GSPConfig`` used for graph construction (symmetrisation,
            normalisation, HFER cutoff, ...).
    """

    def __init__(self, model, tokenizer=None, config: Optional[GSPConfig] = None,
                 name: Optional[str] = None):
        if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            raise ValueError("SpectralModel supports decoder-only causal LMs")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or default_config()
        self.name = name or getattr(getattr(model, "config", None), "_name_or_path", None) \
            or type(model).__name__
        _ensure_eager(model)
        model.eval()

    @classmethod
    def from_pretrained(cls, name_or_model, tokenizer=None, config: Optional[GSPConfig] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        dtype: Union[str, torch.dtype, None] = "float32",
                        **hf_kwargs) -> "SpectralModel":
        """Load a Hugging Face causal LM with eager attention, or wrap one.

        Args:
            name_or_model: a model id or local path, or an already-loaded
                ``PreTrainedModel`` (then pass its ``tokenizer``; when it is
                omitted it is loaded from the model's ``name_or_path``).
            tokenizer: the tokenizer of an already-loaded model.
            config: graph-construction settings; default
                :func:`default_config`.
            device: where to put a model loaded here (``"cpu"``, ``"cuda"``);
                default: where ``from_pretrained`` leaves it.
            dtype: weights dtype for a model loaded here (default float32).
            **hf_kwargs: passed to ``AutoModelForCausalLM.from_pretrained``
                (for example ``revision``, ``local_files_only``,
                ``trust_remote_code``, ``device_map``); the ones that apply
                to tokenizers are passed to ``AutoTokenizer`` as well.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .instrumentation import _DTYPE_KW

        tok_keys = ("revision", "local_files_only", "trust_remote_code", "cache_dir",
                    "token", "subfolder")
        tok_kwargs = {k: hf_kwargs[k] for k in tok_keys if k in hf_kwargs}
        if isinstance(name_or_model, str):
            name = name_or_model
            hf_kwargs.setdefault("attn_implementation", "eager")
            if dtype is not None and "torch_dtype" not in hf_kwargs and "dtype" not in hf_kwargs:
                hf_kwargs[_DTYPE_KW] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            model = AutoModelForCausalLM.from_pretrained(name, **hf_kwargs)
            if device is not None:
                model.to(device)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(name, **tok_kwargs)
        else:
            model = name_or_model
            name = getattr(getattr(model, "config", None), "_name_or_path", None)
            if tokenizer is None:
                if not name:
                    raise ValueError("pass the tokenizer of an already-loaded model")
                tokenizer = AutoTokenizer.from_pretrained(name, **tok_kwargs)
            if device is not None:
                model.to(device)
        return cls(model, tokenizer, config=config, name=name)

    # ── model facts ─────────────────────────────────────────────────────────

    @property
    def n_layers(self) -> Optional[int]:
        """Number of decoder layers, from the model config."""
        return _num_layers(self.model.config)

    @property
    def n_heads(self) -> Optional[int]:
        """Attention heads per layer, from the model config."""
        return _num_heads(self.model.config)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    def __call__(self, *args, **kwargs):
        """Run the wrapped model."""
        return self.model(*args, **kwargs)

    def to_str_tokens(self, text: str) -> List[str]:
        """The token strings of ``text``, as :attr:`SpectralCache.tokens`
        holds them (each id decoded on its own)."""
        ids = self.tokenizer(text)["input_ids"]
        return [self.tokenizer.decode([i]) for i in ids]

    def __repr__(self) -> str:
        return (f"SpectralModel({self.name!r}, {self.n_layers} layers, {self.n_heads} heads, "
                f"normalization={self.config.normalization!r})")

    # ── the main call ───────────────────────────────────────────────────────

    def _encode(self, items) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[List[int]]]:
        ids: List[List[int]] = []
        for it in items:
            if isinstance(it, str):
                if self.tokenizer is None:
                    raise ValueError("text input needs a tokenizer; pass token ids instead")
                ids.append(list(self.tokenizer(it)["input_ids"]))
            else:
                ids.append([int(i) for i in (it.tolist() if torch.is_tensor(it) else it)])
        if any(len(x) == 0 for x in ids):
            raise ValueError("empty input")
        lengths = [len(x) for x in ids]
        width = max(lengths)
        pad = None
        if self.tokenizer is not None:
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id
        pad = 0 if pad is None else int(pad)
        input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):       # right padding: real tokens keep positions 0..n-1
            input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            mask[i, :len(x)] = 1
        return input_ids, mask, lengths, ids

    @torch.no_grad()
    def run_with_spectra(
        self,
        text_or_list: TextInput,
        metrics=None,
        layers: Optional[Sequence[int]] = None,
        heads: HeadSelection = None,
        token_span=None,
        hybrid: bool = True,
        batch_size: int = 8,
        keep_patterns: Optional[bool] = None,
    ) -> SpectralCache:
        """Run the model once and return its spectral measurements.

        Args:
            text_or_list: a string, a list of strings, or token ids
                (``[seq]`` or ``[batch, seq]`` tensor, or a list of id
                lists). Strings are tokenised with the model's tokenizer as
                it is configured (special tokens included), with no
                truncation.
            metrics: which metrics to keep. None keeps all. Entries may be a
                namespace (``"attn"``, ``"resid"``, ``"graph"``), a
                namespaced name (``"graph.hfer"``) or a bare name, which
                selects it in every namespace (``"fiedler_value"`` gives
                ``attn.fiedler_value`` and ``graph.fiedler_value``). A
                namespace that is not selected is not computed.
            layers: true layer indices to measure (negative values count
                from the end); default every layer that returns attention.
            heads: heads for the per-head namespaces ``attn`` and ``resid``,
                as in :func:`~spectral_trust.per_head.per_head_metrics`:
                indices, ``(layer, head)`` pairs, or a boolean ``[heads]`` or
                ``[layers, heads]`` mask (true layer indices). Heads not
                computed are NaN. The ``graph`` namespace always averages
                every head.
            token_span: ``(start, end)`` tokens to measure, with slice
                semantics (``(-5, None)`` is the last five tokens). The
                induced subgraph is taken before the Laplacian is built.
            hybrid: compute the metrics that read the hidden states
                (``resid.*`` and the hybrid ``graph.*`` metrics). With False
                only attention-only metrics are computed.
            batch_size: texts per forward pass. Batches are right-padded,
                which leaves every real token's attention and hidden states
                unchanged in a causal LM (the padding is cropped before any
                measurement); results equal one-text runs to float32
                rounding. Use 1 to run texts one at a time.
            keep_patterns: keep the raw attention in the cache, for
                :func:`spectral_trust.viz.graph`. None keeps it when it
                takes at most 64 MB.

        Returns:
            :class:`SpectralCache`. Keys look like
            ``"blocks.10.attn.fiedler_value"`` (one value per head) and
            ``"blocks.10.graph.hfer"`` (head-averaged).
        """
        single, items = _normalise_inputs(text_or_list)
        if self.tokenizer is None and any(isinstance(it, str) for it in items):
            raise ValueError("text input needs a tokenizer")
        batch_size = max(int(batch_size), 1)
        parts = []
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            input_ids, mask, lengths, ids = self._encode(chunk)
            out = self.model(input_ids=input_ids.to(self.device),
                             attention_mask=mask.to(self.device),
                             output_attentions=True, output_hidden_states=True,
                             use_cache=False, return_dict=True)
            attn = list(out.attentions or [])
            hidden = list(out.hidden_states or [])
            if not attn or all(a is None for a in attn):
                raise ValueError(
                    "the model returned no attention weights. Load it with "
                    "attn_implementation='eager' (SpectralModel.from_pretrained does).")
            layer_map = _layer_map(self.model.config, attn, hidden, self.config)
            attentions = {layer_map[k]: a for k, a in enumerate(attn) if a is not None}
            signals = {lay: hidden[lay + 1] for lay in attentions} if hybrid else None
            tokens = [[self.tokenizer.decode([i]) for i in x] if self.tokenizer is not None
                      else [str(i) for i in x] for x in ids]
            texts = [it if isinstance(it, str) else None for it in chunk]
            meta = {"texts": texts, "token_ids": ids, "model_name": self.name,
                    "source": "SpectralModel.run_with_spectra",
                    "signal_source": "hidden_states[layer + 1]"}
            parts.append(compute_spectra(
                attentions, signals, tokens=tokens, lengths=lengths,
                n_layers=self.n_layers or len(hidden) - 1, config=self.config,
                metrics=metrics, layers=layers, heads=heads, token_span=token_span,
                hybrid=hybrid, keep_patterns=keep_patterns, meta=meta))
            del out, attn, hidden
        return merge_caches(parts, batched=not single)


def _layer_map(model_config, attn, hidden, config: GSPConfig) -> List[int]:
    """True layer index of each returned attention tensor."""
    n = len(attn)
    override = getattr(config, "attention_layer_map", None)
    n_layers = len(hidden) - 1 if hidden else _num_layers(model_config)
    if override is None and n_layers is not None and n == n_layers \
            and any(a is None for a in attn):
        return list(range(n))      # None placeholders for layers without weights
    return attention_layer_indices(model_config, n, (n_layers or n) + 1, override)


def _normalise_inputs(x) -> Tuple[bool, list]:
    """(single, items): items are strings or id sequences."""
    if isinstance(x, str):
        return True, [x]
    if torch.is_tensor(x):
        if x.dim() == 1:
            return True, [x]
        if x.dim() == 2:
            return False, [row for row in x]
        raise ValueError(f"token ids must be [seq] or [batch, seq], got {tuple(x.shape)}")
    items = list(x)
    if not items:
        raise ValueError("no input")
    if all(isinstance(i, (int, np.integer)) for i in items):
        return True, [items]
    return False, items


def _ensure_eager(model) -> None:
    cfg = getattr(model, "config", None)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl in (None, "eager"):
        return
    setter = getattr(model, "set_attn_implementation", None)
    try:
        if callable(setter):
            setter("eager")
        else:
            cfg._attn_implementation = "eager"
        logger.info("switched the model's attention from %r to 'eager', the "
                    "implementation that returns attention weights", impl)
    except Exception as exc:  # pragma: no cover - depends on the model class
        logger.warning("could not switch attention from %r to 'eager' (%s); attention "
                       "weights may be missing", impl, exc)
