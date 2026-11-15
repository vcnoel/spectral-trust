# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Value-weighted attention graphs. Added in 0.5.1.

An attention weight ``a^h_ij`` says how much query i reads key j in head h.
How much key j changes query i's output also depends on the vector that j's
value carries through the output projection, and attention to the sink (the
first token) often carries a large weight with a small value
(Kobayashi et al. 2020). The output of an attention module at token i splits
exactly into one vector per key token plus the output bias:

.. math::
    y_i = \sum_j c_{ij} + b_O, \qquad
    c_{ij} = \sum_h a^h_{ij} \, W_O^h \, (g^h_i \odot v^h_j)

with ``v^h_j`` the value of token j that head h reads (shared by the heads of
a group under grouped-query attention), ``W_O^h`` the columns of the output
projection that read head h, ``b_O`` its bias and ``g_i`` an output gate
(``g = 1`` without one). The value-weighted edge from j to i is

.. math::
    w_{ij} = \lVert c_{ij} \rVert
    \quad\text{or, per head,}\quad
    w^h_{ij} = \lVert a^h_{ij} \, W_O^h (g^h_i \odot v^h_j) \rVert ,

the first being Kobayashi et al.'s norm of the sum over heads (their
``‖Σ_h α^h f^h(x_j)‖``), the second the norm of each head's term.

Layouts:

- **Biases.** The value bias (GPT-2's ``c_attn``, Qwen2's ``v_proj``) is part
  of ``v_j``, as in Kobayashi et al.'s ``f(x) = (x W_V + b_V) W_O``. The output
  bias belongs to no token: it is left out of every ``c_ij`` (and of the edge
  weights), and the decomposition reconstructs ``y_i − b_O``.
- **Grouped-query attention.** The values are read as the attention read them
  (``[B, H_kv, S, d_head]``) and head h uses group ``h // (H / H_kv)``, as
  ``transformers``' ``repeat_kv`` does.
- **Fused projections** (GPT-2 ``c_attn``, Phi-3 ``qkv_proj``, GPT-NeoX
  ``query_key_value``) need nothing special: the values are taken where the
  attention reads them, inside ``eager_attention_forward``, after any split,
  reshape or norm.
- **Output gates.** In Qwen3.5 and Qwen3-Next, ``q_proj`` produces a query and
  a gate per head, and ``sigmoid(gate_i)`` multiplies the head outputs of
  query i before ``o_proj``; AFMoE has a separate ``gate_proj``. The gate is
  rebuilt from the projection's output and checked against the input of the
  output projection on every call.
- **Hybrid models** (Qwen3.5: linear-attention and full-attention layers):
  only the full-attention layers have attention weights, and asking for
  another layer raises.

The contributions are to the attention module's output, the output of its
output projection. A norm the block applies after it (Gemma-2's
``post_attention_layernorm``) is not part of the decomposition.

Memory: the summed edge weights use the Gram matrices of the per-head
projected values ``u^h_j = W_O^h v^h_j``, ``G_j[h, h'] = u^hᵀ_j u^h'_j``, so
that ``w_ij² = Σ_{h,h'} a^h_ij a^h'_ij G_j[h, h']``: ``O(S d H²)`` memory
and ``O(S² H²)`` time, never ``[S, S, d_model]``. A gate depends on the
query, so gated layers are computed over chunks of queries (and only over
the keys a chunk attends to), at ``O(S² H d_head d_model)`` time.

Reference:
    G. Kobayashi, T. Kuribayashi, S. Yokoi and K. Inui. Attention is not
    only a weight: analyzing transformers with vector norms. EMNLP 2020.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Working memory of one chunk of the edge computation.
CHUNK_BYTES = 64 * 2 ** 20

#: Output-gate layouts that are recognised: ``"query_split"`` (``q_proj``
#: emits ``[query, gate]`` per head, Qwen3.5 and Qwen3-Next) and
#: ``"gate_proj"`` (a separate projection, AFMoE).
GATE_LAYOUTS = ("query_split", "gate_proj")

Queries = Union[None, int, slice, Sequence[int], torch.Tensor]


# ── tensors ──────────────────────────────────────────────────────────────────

def _prepare(weights, values, gate):
    """Batch every input and check the shapes. Returns (weights, values,
    gate, unbatched)."""
    w, v = torch.as_tensor(weights), torch.as_tensor(values)
    unbatched = w.dim() == 3
    if unbatched:
        w, v = w.unsqueeze(0), v.unsqueeze(0)
        if gate is not None:
            gate = torch.as_tensor(gate).unsqueeze(0)
    if w.dim() != 4 or v.dim() != 4:
        raise ValueError(f"weights must be [B, H, S, S] and values [B, H_kv, S, d_head], got "
                         f"{tuple(w.shape)} and {tuple(v.shape)}")
    B, H, _, K = w.shape
    if v.shape[0] != B or v.shape[2] != K:
        raise ValueError(f"values {tuple(v.shape)} do not match weights {tuple(w.shape)}")
    if H % v.shape[1]:
        raise ValueError(f"{H} heads are not a multiple of {v.shape[1]} value heads")
    if gate is not None:
        gate = torch.as_tensor(gate)
        if gate.shape != (B, w.shape[2], H * v.shape[-1]):
            raise ValueError(f"gate must be [B, S, H * d_head] = "
                             f"{(B, w.shape[2], H * v.shape[-1])}, got {tuple(gate.shape)}")
    return w, v, gate, unbatched


def _out_heads(out_weight: torch.Tensor, H: int, hd: int) -> torch.Tensor:
    """``W_O`` ``[d_model, H * d_head]`` as ``[d_model, H, d_head]``."""
    if out_weight.dim() != 2 or out_weight.shape[1] != H * hd:
        raise ValueError(f"out_weight must be [d_model, H * d_head] = [d_model, {H * hd}], got "
                         f"{tuple(out_weight.shape)}")
    return out_weight.reshape(out_weight.shape[0], H, hd)


def _work_dtype(*ts) -> torch.dtype:
    return torch.float64 if any(t is not None and t.dtype == torch.float64 for t in ts) \
        else torch.float32


def _query_index(queries: Queries, S: int, device) -> torch.Tensor:
    if queries is None:
        return torch.arange(S, device=device)
    if isinstance(queries, slice):
        return torch.arange(S, device=device)[queries]
    if isinstance(queries, int):
        queries = [queries]
    idx = torch.as_tensor(queries, device=device).reshape(-1).long()
    return torch.remainder(idx, S)


def _key_extent(w: torch.Tensor) -> int:
    """One past the last key column with any weight in ``w`` ``[B, H, q, K]``."""
    nz = (w != 0).flatten(0, 2).any(dim=0)
    hits = torch.nonzero(nz)
    return int(hits[-1]) + 1 if hits.numel() else 0


def attention_contributions(weights, values, out_weight, gate=None,
                            queries: Queries = None) -> torch.Tensor:
    r"""
    What each key token adds to each query token's attention output.

    ``c[b, i, j] = Σ_h a^h_ij W_O^h (g^h_i ⊙ v^h_j)``, so that
    ``c[b, i].sum(0) + b_O`` is the attention module's output at token i.
    The tensor is ``[B, Q, S, d_model]``: pass ``queries`` for long
    sequences, or use :func:`value_weighted_edges` for the norms alone.

    Args:
        weights: ``[B, H, S, S]`` attention weights (or ``[H, S, S]``).
        values: ``[B, H_kv, S, d_head]`` value states as the attention read
            them (or ``[H_kv, S, d_head]``).
        out_weight: ``[d_model, H * d_head]`` output projection, as a
            ``torch.nn.Linear`` stores it (the transpose of a GPT-2
            ``Conv1D`` weight).
        gate: ``[B, S, H * d_head]`` output gate after its nonlinearity, or
            None.
        queries: query tokens (index, indices, slice); default all.

    Returns:
        ``[B, Q, S, d_model]`` (``[Q, S, d_model]`` for unbatched input), in
        float32, or float64 when an input is float64.
    """
    w, v, g, unbatched = _prepare(weights, values, gate)
    dt = _work_dtype(w, v, out_weight, g)
    B, H, S, K = w.shape
    hd = v.shape[-1]
    Wo = _out_heads(out_weight.to(dt), H, hd)
    q = _query_index(queries, S, w.device)
    a = w[:, :, q, :].to(dt)                                       # [B, H, Q, K]
    vv = v.to(dt).repeat_interleave(H // v.shape[1], dim=1)        # [B, H, K, hd]
    if g is None:
        u = torch.einsum("bhkx,dhx->bhkd", vv, Wo)                  # [B, H, K, d]
        c = torch.einsum("bhqk,bhkd->bqkd", a, u)
    else:
        gq = g[:, q].to(dt).reshape(B, len(q), H, hd)               # [B, Q, H, hd]
        y = a.permute(0, 2, 3, 1)[..., None] * gq[:, :, None] * vv.permute(0, 2, 1, 3)[:, None]
        c = torch.einsum("bqkhx,dhx->bqkd", y, Wo)                  # [B, Q, K, d]
    return c[0] if unbatched else c


def value_weighted_edges(weights, values, out_weight, gate=None, per_head: bool = False,
                         chunk_bytes: int = CHUNK_BYTES) -> torch.Tensor:
    r"""
    Value-weighted edge weights of one attention layer.

    ``per_head=False`` (default): ``w_ij = ‖Σ_h a^h_ij W_O^h (g^h_i ⊙
    v^h_j)‖``, the norm of what token j adds to token i's attention output
    (Kobayashi et al. 2020), ``[B, S, S]``. ``per_head=True``:
    ``w^h_ij = a^h_ij ‖W_O^h (g^h_i ⊙ v^h_j)‖``, ``[B, H, S, S]``, which
    :class:`~spectral_trust.GraphConstructor` aggregates over heads like
    attention. By the triangle inequality the sum over heads of the per-head
    weights bounds the summed weight from above; they differ where heads
    cancel. The output bias is in neither (it belongs to no token).

    Without a gate the summed weights come from the Gram matrices of
    ``u^h_j = W_O^h v^h_j`` in float64 (``O(S² H²)``); with a gate they are
    computed over chunks of queries, each over the keys it attends to,
    within ``chunk_bytes`` of working memory. No ``[S, S, d_model]`` tensor
    is formed.

    Args:
        weights, values, out_weight, gate: as in
            :func:`attention_contributions`.
        per_head: one graph per head instead of the norm of the sum.
        chunk_bytes: working memory of one chunk.

    Returns:
        ``[B, S, S]`` or ``[B, H, S, S]`` (no batch dimension for unbatched
        input), float32, or float64 when an input is float64. Row i is the
        query, as for attention.
    """
    w, v, g, unbatched = _prepare(weights, values, gate)
    out_dt = _work_dtype(w, v, out_weight, g)
    B, H, S, K = w.shape
    hd = v.shape[-1]
    rep = H // v.shape[1]
    budget = max(int(chunk_bytes), 1)
    if g is None:
        Wo = _out_heads(out_weight.to(torch.float64), H, hd)
        d = Wo.shape[0]
        G = torch.empty(B, K, H, H, dtype=torch.float64, device=w.device)
        step = max(1, budget // max(1, B * H * d * 8 * 2))
        for k0 in range(0, K, step):
            vk = v[:, :, k0:k0 + step].to(torch.float64).repeat_interleave(rep, dim=1)
            u = torch.einsum("bhkx,dhx->bkhd", vk, Wo)              # [B, k, H, d]
            G[:, k0:k0 + step] = torch.matmul(u, u.transpose(-2, -1))
        if per_head:
            norms = torch.diagonal(G, dim1=-2, dim2=-1).clamp(min=0).sqrt()  # [B, K, H]
            out = w.to(torch.float64) * norms.permute(0, 2, 1)[:, :, None, :]
        else:
            out = torch.zeros(B, S, K, dtype=torch.float64, device=w.device)
            step = max(1, budget // max(1, B * K * H * 8 * 3))
            for q0 in range(0, S, step):
                wq = w[:, :, q0:q0 + step]
                kmax = _key_extent(wq)
                if kmax == 0:
                    continue
                a = wq[..., :kmax].to(torch.float64).permute(0, 2, 3, 1)   # [B, q, k, H]
                t = torch.einsum("bqkh,bkhg->bqkg", a, G[:, :kmax])
                out[:, q0:q0 + step, :kmax] = (t * a).sum(-1).clamp(min=0).sqrt()
        out = out.to(out_dt)
        return out[0] if unbatched else out

    dt = out_dt
    Wo = _out_heads(out_weight.to(dt), H, hd)
    d = Wo.shape[0]
    vv = v.to(dt).repeat_interleave(rep, dim=1)                     # [B, H, K, hd]
    gg = g.to(dt).reshape(B, S, H, hd)
    item = dt.itemsize if hasattr(dt, "itemsize") else 4
    if per_head:
        out = torch.zeros(B, H, S, K, dtype=dt, device=w.device)
        step = max(1, budget // max(1, B * K * (hd + d) * item * 2))
        for q0 in range(0, S, step):
            wq = w[:, :, q0:q0 + step]
            kmax = _key_extent(wq)
            if kmax == 0:
                continue
            for h in range(H):
                y = gg[:, q0:q0 + step, h, None, :] * vv[:, h, None, :kmax, :]   # [B, q, k, hd]
                n = torch.matmul(y, Wo[:, h].transpose(0, 1)).norm(dim=-1)        # [B, q, k]
                out[:, h, q0:q0 + step, :kmax] = wq[:, h, :, :kmax].to(dt) * n
    else:
        out = torch.zeros(B, S, K, dtype=dt, device=w.device)
        Wflat = Wo.reshape(d, H * hd).transpose(0, 1)                # [H * hd, d]
        step = max(1, budget // max(1, B * K * (H * hd + d) * item * 2))
        for q0 in range(0, S, step):
            wq = w[:, :, q0:q0 + step]
            kmax = _key_extent(wq)
            if kmax == 0:
                continue
            a = wq[..., :kmax].to(dt).permute(0, 2, 3, 1)[..., None]       # [B, q, k, H, 1]
            y = a * gg[:, q0:q0 + step, None] * vv[:, :, :kmax].permute(0, 2, 1, 3)[:, None]
            c = torch.matmul(y.flatten(-2), Wflat)                          # [B, q, k, d]
            out[:, q0:q0 + step, :kmax] = c.norm(dim=-1)
    return out[0] if unbatched else out


# ── one layer's captured tensors ─────────────────────────────────────────────

@dataclass
class AttentionValues:
    """What one attention layer computed on one batch: enough to split its
    output into one vector per key token (see the module notes).

    Attributes:
        layer: true layer index.
        weights: ``[B, H, S, S]`` attention weights (row i the query).
        values: ``[B, H_kv, S, d_head]`` value states, as the attention read
            them.
        out_weight: ``[d_model, H * d_head]`` output projection
            (``torch.nn.Linear`` layout).
        out_bias: ``[d_model]`` output bias, or None. It is in no
            contribution.
        gate: ``[B, S, H * d_head]`` output gate after its sigmoid, or None.
        gate_layout: ``"query_split"``, ``"gate_proj"`` or None.
        output: ``[B, S, d_model]`` the module's output, when kept.
        lengths: real length of each batch item (right padding follows).
    """
    layer: int
    weights: torch.Tensor
    values: torch.Tensor
    out_weight: torch.Tensor
    out_bias: Optional[torch.Tensor] = None
    gate: Optional[torch.Tensor] = None
    gate_layout: Optional[str] = None
    output: Optional[torch.Tensor] = None
    lengths: Optional[List[int]] = field(default=None)

    @property
    def n_heads(self) -> int:
        return int(self.weights.shape[1])

    @property
    def n_kv_heads(self) -> int:
        return int(self.values.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.values.shape[-1])

    def contributions(self, queries: Queries = None) -> torch.Tensor:
        """``[B, Q, S, d_model]`` per-key contributions
        (:func:`attention_contributions`)."""
        return attention_contributions(self.weights, self.values, self.out_weight,
                                       self.gate, queries)

    def edge_weights(self, per_head: bool = False,
                     chunk_bytes: int = CHUNK_BYTES) -> torch.Tensor:
        """Value-weighted edges, ``[B, S, S]`` or ``[B, H, S, S]``
        (:func:`value_weighted_edges`)."""
        return value_weighted_edges(self.weights, self.values, self.out_weight, self.gate,
                                    per_head=per_head, chunk_bytes=chunk_bytes)

    def reconstruct(self, chunk_bytes: int = CHUNK_BYTES) -> torch.Tensor:
        """``Σ_j c_ij + b_O`` for every query, ``[B, S, d_model]``: the
        attention output rebuilt from the per-key contributions (computed in
        chunks of queries)."""
        B, H, S, K = self.weights.shape
        d = self.out_weight.shape[0]
        dt = _work_dtype(self.weights, self.values, self.out_weight, self.gate)
        per_query = B * K * (d + H * self.head_dim) * 8
        step = max(1, int(chunk_bytes) // max(per_query, 1))
        parts = [self.contributions(slice(q0, q0 + step)).sum(dim=2)
                 for q0 in range(0, S, step)]
        y = torch.cat(parts, dim=1)
        if self.out_bias is not None:
            y = y + self.out_bias.to(dt)
        return y


# ── capture from a model ─────────────────────────────────────────────────────

def _proj_weight(proj: nn.Module) -> torch.Tensor:
    """Weight of a Linear (``[out, in]``) or of a GPT-2 Conv1D (``[in, out]``),
    in the Linear layout."""
    w = proj.weight
    if hasattr(proj, "nf") and not isinstance(proj, nn.Linear):     # transformers Conv1D
        return w.t()
    return w


def _gate_layout(attn: nn.Module, n_heads: int, head_dim: int):
    """The output-gate layout of an attention module and the projection
    whose output carries the gate, or (None, None)."""
    q = getattr(attn, "q_proj", None)
    if isinstance(q, nn.Linear) and q.out_features == 2 * n_heads * head_dim:
        return "query_split", q
    gp = getattr(attn, "gate_proj", None)
    if isinstance(gp, nn.Linear) and gp.out_features == n_heads * head_dim:
        return "gate_proj", gp
    return None, None


def _gate_from(layout: str, t: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    if layout == "query_split":
        part = t.view(*t.shape[:-1], n_heads, 2 * head_dim)[..., head_dim:]
        return torch.sigmoid(part).reshape(*t.shape[:-1], n_heads * head_dim)
    return torch.sigmoid(t)


def _layer_types(config) -> Optional[List[str]]:
    for cfg in (config, getattr(config, "text_config", None)):
        types = getattr(cfg, "layer_types", None) if cfg is not None else None
        if types:
            return [str(t) for t in types]
    return None


class ValueCapture:
    """Context manager that records, during forward passes of a Hugging Face
    causal LM, what :class:`AttentionValues` needs for each layer: the
    attention weights and value states (inside ``eager_attention_forward``),
    the input of the output projection, the gate projection's output and,
    optionally, the module output. Nothing in the computation changes; the
    attention is switched to the eager implementation while the capture is
    open (and switched back on exit).

    Examples:
        >>> with ValueCapture(model, layers=[3, 7]) as cap:
        ...     model(input_ids)
        >>> vals = cap.results()          # {3: AttentionValues, 7: ...}

    Args:
        model: a ``SpectralModel`` or a ``transformers`` causal LM.
        layers: true layer indices; default every layer with softmax
            attention (the full-attention layers of a hybrid model).
        keep_output: also keep each attention module's output.
    """

    def __init__(self, model, layers: Optional[Sequence[int]] = None,
                 keep_output: bool = False):
        from .interventions._arch import Arch
        self.arch = Arch(model)
        self.keep_output = bool(keep_output)
        self.layers = self._resolve_layers(layers)
        self._store: Dict[int, dict] = {}
        self._undo: List[Callable[[], None]] = []
        self._open = False

    def _has_attention(self, layer: int) -> bool:
        try:
            self.arch.attention(layer)
            return True
        except ValueError:
            return False

    def _resolve_layers(self, layers) -> List[int]:
        n = self.arch.n_layers
        with_attn = [lay for lay in range(n) if self._has_attention(lay)]
        if layers is None:
            if not with_attn:
                raise ValueError("the model has no softmax attention layer")
            return with_attn
        if isinstance(layers, int):
            layers = [layers]
        out = []
        types = _layer_types(self.arch.config)
        for lay in layers:
            lay = self.arch.layer_index(lay)
            if lay not in with_attn:
                kind = f" (layer_types[{lay}] = {types[lay]!r})" if types and lay < len(types) \
                    else ""
                raise ValueError(
                    f"layer {lay}{kind} has no softmax attention, so it has no attention "
                    f"weights and no value-weighted graph; layers with one: {with_attn}")
            if lay not in out:
                out.append(lay)
        return sorted(out)

    # ── hooks ──
    def __enter__(self) -> "ValueCapture":
        from .interventions import _registry as reg
        if self._open:
            raise RuntimeError("this ValueCapture is already open")
        self._open = True
        self._store = {}
        try:
            reg.acquire_eager(self.arch, self)
            self._undo.append(lambda: reg.release_eager(self.arch.model, self))
            H = self.arch.n_heads
            for lay in self.layers:
                attn = self.arch.attention(lay)
                proj = self.arch.attention_output(lay)
                slot = self._store.setdefault(lay, {})
                hd = self.arch.head_dim(lay)
                slot["meta"] = (attn, proj, H, hd)

                def on_attn(module, value, weights, out, slot=slot):
                    slot["value"] = value.detach()
                    slot["weights"] = None if weights is None else weights.detach()
                    slot["attn_out"] = out.detach()

                self._undo.append(reg.acquire_capture(attn, self, on_attn))

                def on_proj(module, args, kwargs, slot=slot):
                    x = args[0] if args else next(iter(kwargs.values()))
                    slot["z"] = x.detach()

                self._undo.append(proj.register_forward_pre_hook(on_proj, with_kwargs=True).remove)
                layout, gsrc = _gate_layout(attn, H, hd)
                if layout is not None:
                    slot["gate_layout"] = layout

                    def on_gate(module, args, out, slot=slot, layout=layout, H=H, hd=hd):
                        slot["gate"] = _gate_from(layout, out.detach(), H, hd)

                    self._undo.append(gsrc.register_forward_hook(on_gate).remove)
                if self.keep_output:
                    def on_out(module, args, out, slot=slot):
                        y = out[0] if isinstance(out, (tuple, list)) else out
                        slot["output"] = y.detach()

                    self._undo.append(attn.register_forward_hook(on_out).remove)
        except Exception:
            self._close()
            raise
        return self

    def _close(self) -> None:
        while self._undo:
            fn = self._undo.pop()
            try:
                fn()
            except Exception:  # pragma: no cover - best effort on exit
                logger.exception("could not undo a value-capture hook")
        self._open = False

    def __exit__(self, *exc) -> None:
        self._close()

    # ── results ──
    def results(self, lengths: Optional[Sequence[int]] = None) -> Dict[int, AttentionValues]:
        """:class:`AttentionValues` of every captured layer, from the last
        forward pass. Checks, per layer, that the attention output is
        ``weights @ values`` and that the input of the output projection is
        that output (times the gate, when there is one); raises otherwise."""
        out: Dict[int, AttentionValues] = {}
        for lay in self.layers:
            slot = self._store.get(lay, {})
            attn, proj, H, hd = slot["meta"]
            if "value" not in slot or "z" not in slot:
                raise RuntimeError(
                    f"layer {lay}'s attention did not go through eager_attention_forward, so its "
                    f"values were not captured (GPT-2 with reorder_and_upcast_attn=True takes "
                    f"another path); run a forward pass inside the capture")
            w, v = slot["weights"], slot["value"]
            if w is None:
                raise RuntimeError(f"layer {lay}'s eager attention returned no weights")
            if w.shape[1] != H or v.shape[-1] != hd:
                raise RuntimeError(
                    f"layer {lay}: weights {tuple(w.shape)} and values {tuple(v.shape)} do not "
                    f"match {H} heads of width {hd}")
            o = slot["attn_out"].reshape(*slot["attn_out"].shape[:2], -1)
            check = torch.matmul(w.to(v.dtype), v.repeat_interleave(H // v.shape[1], dim=1))
            check = check.transpose(1, 2).reshape(o.shape)
            tol = 1e-4 * float(o.abs().max()) + 1e-6
            if float((check - o).abs().max()) > tol:
                raise RuntimeError(
                    f"layer {lay}: the attention output is not weights @ values, so it cannot "
                    f"be split by key token")
            z = slot["z"]
            gate = slot.get("gate")
            layout = slot.get("gate_layout") if gate is not None else None
            if gate is not None:
                ref = gate * o
            else:
                ref = o
            ztol = 1e-5 * float(z.abs().max()) + 1e-7
            if ref.shape != z.shape or float((ref - z).abs().max()) > ztol:
                what = (f"its {layout} gate times the attention output" if gate is not None
                        else "the attention output")
                raise NotImplementedError(
                    f"layer {lay}: the input of {type(proj).__name__} is not {what}; the "
                    f"attention has an output transformation (a gate?) this library does not "
                    f"decompose")
            bias = getattr(proj, "bias", None)
            out[lay] = AttentionValues(
                layer=lay, weights=w, values=v, out_weight=_proj_weight(proj).detach(),
                out_bias=None if bias is None else bias.detach(), gate=gate,
                gate_layout=layout, output=slot.get("output"),
                lengths=None if lengths is None else [int(n) for n in lengths])
        return out


def _encode_batch(inputs, tokenizer):
    """Right-padded ids, attention mask and lengths."""
    from .interventions.context import _encode_list
    rows = _encode_list(inputs, tokenizer)
    if any(len(r) == 0 for r in rows):
        raise ValueError("empty input")
    lengths = [len(r) for r in rows]
    width = max(lengths)
    pad = 0
    if tokenizer is not None:
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
            else (tokenizer.eos_token_id or 0)
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, :len(r)] = 1
    return ids, mask, lengths


@torch.no_grad()
def capture_attention_values(model, inputs, layers: Optional[Sequence[int]] = None,
                             tokenizer=None, keep_output: bool = False
                             ) -> Dict[int, AttentionValues]:
    """Run the model once and return, per layer, the tensors that split its
    attention output by key token (:class:`AttentionValues`).

    Args:
        model: a ``SpectralModel`` or a ``transformers`` causal LM.
        inputs: a string, a list of strings, or token ids (``[S]``,
            ``[B, S]`` or lists). Texts of different lengths are right-padded.
        layers: true layer indices (default every softmax-attention layer).
        tokenizer: for text input when ``model`` is not a ``SpectralModel``.
        keep_output: also keep each attention module's output.

    Returns:
        ``{layer: AttentionValues}``.
    """
    from .interventions._arch import unwrap
    tok = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    hf = unwrap(model)
    ids, mask, lengths = _encode_batch(inputs, tok)
    device = next(hf.parameters()).device
    with ValueCapture(model, layers=layers, keep_output=keep_output) as cap:
        hf(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    return cap.results(lengths)


@torch.no_grad()
def value_weighted_attention(model, inputs, layers: Optional[Sequence[int]] = None,
                             per_head: bool = False, tokenizer=None
                             ) -> Dict[int, torch.Tensor]:
    """Value-weighted attention graphs of a model on some input.

    ``{layer: edges}`` with edges ``[B, S, S]`` (``per_head=False``: the norm
    of what token j adds to token i's attention output, summed over heads
    inside the norm) or ``[B, H, S, S]`` (``per_head=True``). See
    :func:`value_weighted_edges` and the module notes for the layouts
    covered.

    Examples:
        >>> model = st.SpectralModel.from_pretrained("gpt2")
        >>> w = st.value_weighted_attention(model, "The cat sat on the mat.")
        >>> w[5].shape        # [1, S, S]
    """
    vals = capture_attention_values(model, inputs, layers=layers, tokenizer=tokenizer)
    return {lay: av.edge_weights(per_head=per_head) for lay, av in vals.items()}


__all__ = [
    "attention_contributions", "value_weighted_edges", "AttentionValues", "ValueCapture",
    "capture_attention_values", "value_weighted_attention", "GATE_LAYOUTS", "CHUNK_BYTES",
]
