# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
:func:`intervene`, a context manager that edits a Hugging Face model with
PyTorch hooks and undoes every edit on exit.

>>> with st.intervene(model) as iv:
...     iv.resid(6).graph_filter("lowpass", cutoff=0.5, strength=0.8)
...     iv.attention(9, 6).ablate("zero")
...     iv.attention(9, 9).remove_edges(key_mask, keep_sink=True)
...     iv.weights(7).svd_scale(-0.05)
...     logits = model(ids).logits

Positions are absolute: position i is the i-th token of the sequence,
including tokens already in the KV cache, so every position-dependent edit
gives the same result in cached generation as in a full recomputation.
Negative positions count from the end of the current sequence, which moves
during generation; they raise at a cached generation step.
"""
import logging
import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from ..config import GSPConfig
from . import _registry as reg
from ._arch import Arch, first_hidden, replace_first, unwrap
from ._hooks import canonical_site, register_site_hook
from .filters import FilterSpec, GraphSpec, apply_filter
from .weights import linear_weight, matrix_snr, svd_scale_matrix

logger = logging.getLogger(__name__)

#: :meth:`AttentionSite.remove_edges` with ``keep_sink=True`` warns when it
#: multiplies the remaining non-sink attention of a row by more than this.
KEEP_SINK_WARN_FACTOR = 10.0

Heads = Union[None, int, Sequence[int]]
Positions = Union[None, int, slice, Sequence[int], torch.Tensor]


def _has_negative(positions: Positions) -> bool:
    if positions is None:
        return False
    if isinstance(positions, slice):
        return any(v is not None and v < 0 for v in (positions.start, positions.stop))
    p = torch.as_tensor(positions if not isinstance(positions, int) else [positions])
    return p.dtype != torch.bool and bool((p < 0).any())


def _position_mask(positions: Positions, n: int, device=None) -> Optional[torch.Tensor]:
    """Boolean mask ``[n]`` over absolute positions (None: every position).
    Negative values count from ``n``."""
    if positions is None:
        return None
    if isinstance(positions, slice):
        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[positions] = True
        return m
    p = torch.as_tensor(positions if not isinstance(positions, int) else [positions],
                        device=device)
    if p.dtype == torch.bool:
        if p.numel() < n:                                    # shorter mask: later positions off
            p = torch.cat([p, torch.zeros(n - p.numel(), dtype=torch.bool, device=device)])
        return p[:n]
    m = torch.zeros(n, dtype=torch.bool, device=device)
    p = p.long().reshape(-1)
    p = torch.where(p < 0, p + n, p)
    m[p[(p >= 0) & (p < n)]] = True
    return m


class Edit:
    """One registered intervention. :meth:`remove` undoes it (closing the
    :class:`Intervention` undoes all of them)."""

    def __init__(self, owner: "Intervention", description: str):
        self.owner = owner
        self.description = description
        self.handles: List = []
        self.undo: List[Callable[[], None]] = []
        self.active = True

    def remove(self) -> None:
        """Undo the edit. Every hook is removed and every undo step runs even
        if one of them fails; the first error is raised afterwards."""
        if not self.active:
            return
        errors = []
        for h in self.handles:
            try:
                h.remove()
            except Exception as exc:  # pragma: no cover - torch handles do not raise
                errors.append(exc)
        for fn in reversed(self.undo):
            try:
                fn()
            except Exception as exc:
                errors.append(exc)
        self.handles.clear()
        self.undo.clear()
        self.active = False
        self.owner.edits[:] = [e for e in self.owner.edits if e is not self]
        if errors:
            raise errors[0]

    def __repr__(self) -> str:
        return f"Edit({self.description}{'' if self.active else ', removed'})"


class Intervention:
    """Edits of one model, undone on :meth:`close` (or on leaving the
    ``with`` block). Create it with :func:`intervene`.

    Interventions on the same model may be nested and closed in any order.

    Attributes:
        model: the Hugging Face model being edited.
        edits: the active :class:`Edit` objects, in registration order.
        config: graph construction for the graph filters (the
            :class:`~spectral_trust.SpectralModel`'s config, or the library
            defaults).
    """

    def __init__(self, model, config: Optional[GSPConfig] = None, tokenizer=None):
        from ..model import SpectralModel, default_config
        self.model = unwrap(model)
        self.arch = Arch(model)
        if config is None:
            config = model.config if isinstance(model, SpectralModel) else default_config()
        self.config = config
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.edits: List[Edit] = []
        self._open = True
        self._eager = False
        self._causal_states: List[dict] = []
        self._fwd = {"mask": None, "positions": None, "cached": False}
        self._model_hook = self.model.register_forward_pre_hook(self._on_forward,
                                                                with_kwargs=True)
        reg.register_open(self.model, self)

    # ── context management ─────────────────────────────────────────────────

    def __enter__(self) -> "Intervention":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Undo every edit: remove the hooks, restore the weights bit for
        bit (other open interventions' weight edits are re-applied), and
        release the shared changes (eager attention, the wrapped attention
        function, the generation hooks) once no intervention holds them."""
        if not self._open:
            return
        self._open = False
        errors = []
        steps = [edit.remove for edit in reversed(self.edits)]
        steps += [self._model_hook.remove,
                  lambda: reg.release_beam_hooks(self.model, self),
                  lambda: reg.unregister_open(self.model, self)]
        if self._eager:
            steps.append(lambda: reg.release_eager(self.model, self))
        for step in steps:           # every step runs, even after a failure
            try:
                step()
            except Exception as exc:
                errors.append(exc)
        self._eager = False
        self.edits.clear()
        if errors:
            raise errors[0]

    def __repr__(self) -> str:
        lines = [f"Intervention({type(self.model).__name__}, {len(self.edits)} edits)"]
        lines += [f"  {e.description}" for e in self.edits]
        return "\n".join(lines)

    # ── shared machinery ───────────────────────────────────────────────────

    def _check_open(self):
        if not self._open:
            raise RuntimeError("this intervention is closed")

    def _on_forward(self, module, args, kwargs):
        """Record, for the current call: the 2-D attention mask, the
        absolute positions of its tokens, and whether past tokens are
        cached."""
        mask = kwargs.get("attention_mask")
        self._fwd["mask"] = mask if torch.is_tensor(mask) and mask.dim() == 2 else None
        ids = kwargs.get("input_ids")
        if ids is None and args and torch.is_tensor(args[0]):
            ids = args[0]
        emb = kwargs.get("inputs_embeds")
        q = ids.shape[-1] if ids is not None else (emb.shape[1] if emb is not None else None)
        cp = kwargs.get("cache_position")
        past = kwargs.get("past_key_values")
        offset = 0
        if past is not None:
            if hasattr(past, "get_seq_length"):
                offset = int(past.get_seq_length())
            elif isinstance(past, (tuple, list)) and past and past[0] is not None:
                offset = int(past[0][0].shape[-2])
        if torch.is_tensor(cp) and cp.dim() == 1 and (q is None or cp.numel() == q):
            positions = cp.long()
        elif q is not None:
            positions = torch.arange(offset, offset + q)
        else:
            positions = None
        self._fwd["positions"] = positions
        self._fwd["cached"] = bool(positions is not None and positions.numel()
                                   and int(positions[0]) > 0)
        return None

    def _positions(self, q: int, k: Optional[int] = None, device=None) -> torch.Tensor:
        """Absolute positions of the ``q`` tokens of the current call."""
        p = self._fwd["positions"]
        if torch.is_tensor(p) and p.numel() == q:
            return p.to(device)
        start = (k - q) if k is not None else 0
        return torch.arange(start, start + q, device=device)

    def _check_negative(self, positions: Positions, what: str) -> None:
        if self._fwd["cached"] and _has_negative(positions):
            raise RuntimeError(
                f"{what}: negative positions count from the end of the current sequence, which "
                f"moves during cached generation; pass absolute positions")

    def _valid(self, batch: int, k: int, device) -> Optional[torch.Tensor]:
        """``[B, k]`` mask of real (non-padding) keys, from the 2-D
        ``attention_mask`` keyword of the current call, or None."""
        m = self._fwd["mask"]
        if m is None or m.shape[0] != batch or m.shape[-1] < k:
            return None
        m = m[:, :k].to(device=device).bool()
        return None if bool(m.all()) else m

    def _require_eager(self) -> None:
        if not self._eager:
            reg.acquire_eager(self.arch, self)
            self._eager = True

    def _new_edit(self, description: str) -> Edit:
        self._check_open()
        e = Edit(self, description)
        self.edits.append(e)
        return e

    def _on_reorder(self, beam_idx: torch.Tensor) -> None:
        for st in self._causal_states:
            if st.get("X") is not None:
                idx = beam_idx.to(st["X"].device)
                st["X"] = st["X"].index_select(0, idx)
                st["A"] = st["A"].index_select(0, idx)
            st["reordered"] = True

    def _install_pattern_editor(self, edit: Edit, layer: int,
                                fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """Run ``fn(weights [B, H, q, k], query_positions [q]) -> weights``
        on layer ``layer``'s attention probabilities before they meet the
        values, through the model's ``eager_attention_forward``."""
        self._require_eager()
        attn = self.arch.attention(layer)
        key = id(attn)

        def run(weights, module):
            q, k = weights.shape[-2], weights.shape[-1]
            return fn(weights, self._positions(q, k, weights.device))
        release = reg.acquire_pattern_editor(attn, self, run)

        def pre(module, args, kwargs):
            reg.PATTERN_CALLED[key] = False

        def post(module, args, out):
            if reg.PATTERN_EDITS.get(key) and not reg.PATTERN_CALLED.get(key, False):
                raise RuntimeError(
                    f"layer {layer}'s attention did not go through eager_attention_forward, so "
                    f"its pattern could not be edited (for GPT-2, reorder_and_upcast_attn must "
                    f"be off)")
        edit.handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
        edit.handles.append(attn.register_forward_hook(post))
        edit.undo.append(release)

    # ── sites ──────────────────────────────────────────────────────────────

    def resid(self, layer: int, site: str = "post") -> "ResidSite":
        """The residual stream after block ``layer`` (``site="post"``, the
        library's pairing: ``hidden_states[layer + 1]``) or before it
        (``site="pre"``)."""
        self._check_open()
        if site not in ("pre", "post"):
            raise ValueError(f"site must be 'pre' or 'post', got {site!r}")
        return ResidSite(self, self.arch.layer_index(layer), site)

    def attention(self, layer: int, head: Heads = None) -> "AttentionSite":
        """Attention heads of block ``layer`` (one head, a list, or None for
        all)."""
        self._check_open()
        return AttentionSite(self, self.arch.layer_index(layer), head)

    def weights(self, layer: int, module: str = "mlp.down_proj") -> "WeightSite":
        """A weight matrix of block ``layer`` by dotted path. The default
        ``"mlp.down_proj"`` resolves to the MLP output projection of
        architectures that name it differently (GPT-2 ``mlp.c_proj``)."""
        self._check_open()
        return WeightSite(self, self.arch.layer_index(layer), module)

    def hook(self, site: str, layer: int,
             fn: Callable[[torch.Tensor], Optional[torch.Tensor]]) -> Edit:
        """A custom edit, ``fn(x) -> new x`` (or None to keep ``x``), at a
        site: ``"resid_pre"``, ``"resid_post"``, ``"attn_out"``,
        ``"mlp_out"`` or ``"z"``, the equivalent of a TransformerLens forward
        hook. ``x`` holds the tokens of the current call only; under cached
        generation that is the new token."""
        site = canonical_site(site)
        lay = self.arch.layer_index(layer)
        edit = self._new_edit(f"hook {site} layer {lay}")
        edit.handles.extend(register_site_hook(self.arch, site, lay, fn))
        return edit


class ResidSite:
    """The residual stream at one layer; see :meth:`Intervention.resid`."""

    def __init__(self, iv: Intervention, layer: int, site: str):
        self.iv, self.layer, self.site = iv, layer, site

    def _register(self, fn, description: str) -> Edit:
        edit = self.iv._new_edit(description)
        hook_site = "resid_post" if self.site == "post" else "resid_pre"
        edit.handles.extend(register_site_hook(self.iv.arch, hook_site, self.layer, fn))
        return edit

    def add(self, vector, scale: float = 1.0, positions: Positions = None) -> Edit:
        """Activation steering: add ``scale * vector`` to the residual stream
        at ``positions`` (absolute; default every position). ``vector`` is
        ``[d]``, or ``[S, d]`` with one row per absolute position."""
        vec = torch.as_tensor(vector)
        iv = self.iv

        def fn(x):
            iv._check_negative(positions, "add")
            q = x.shape[1]
            pos = iv._positions(q, device=x.device)
            v = vec.to(device=x.device, dtype=x.dtype)
            if v.dim() == 2:
                if int(pos.max()) >= v.shape[0]:
                    raise ValueError(f"the vector has {v.shape[0]} rows for position "
                                     f"{int(pos.max())}")
                v = v[pos.cpu()]
            delta = scale * v
            if positions is None:
                return x + delta
            m = _position_mask(positions, int(pos.max()) + 1, x.device)[pos]
            return x + delta * m.view(1, -1, 1).to(x.dtype)
        return self._register(fn, f"add vector at resid_{self.site} layer {self.layer}")

    def graph_filter(
        self,
        kind: str = "lowpass",
        cutoff: Optional[float] = None,
        band=None,
        strength: float = 1.0,
        graph: GraphSpec = "attention",
        causal: bool = False,
        order: int = 64,
        method: str = "chebyshev",
        token_span=None,
        positions: Positions = None,
        precision: str = "float64",
    ) -> Edit:
        r"""Graph-frequency steering along the token axis:
        ``X' = X + s (g(L) X - X)`` on the residual stream after this block.

        The graph is this layer's head-averaged attention from the same
        forward pass, symmetrised and normalised as in
        :class:`~spectral_trust.GraphConstructor` (``graph="attention"``),
        the chain graph over positions (``"path"``, a positional baseline),
        an adjacency tensor, or a callable ``(attention, x) -> adjacency``.
        The filter is computed in float64.

        Args:
            kind: ``"lowpass"`` (keep eigenvalues below ``cutoff``),
                ``"highpass"``, ``"bandpass"`` or ``"bandstop"`` (with
                ``band=(lo, hi)``), in eigenvalue units (the ``"sym"``
                Laplacian's spectrum lies in [0, 2]).
            strength: ``s``; 0 is the identity, 1 applies the filter,
                negative values push away from it.
            causal: False filters each position on the symmetric graph of the
                whole sequence, so position i reads later positions; use it
                for single forward passes. True filters position i on the
                graph of its prefix ``0..i`` only (a lower-triangular
                operator), so generation with a KV cache gives the same
                output as recomputing the sequence, also under beam search
                (the filter's state follows the cache's reordering); at the
                last position the two filters agree. A cached generation step
                with ``causal=False`` raises, as does a cache cropped or
                reordered by anything other than beam search (assisted
                decoding, for example).
            order: Chebyshev degree (no eigendecomposition).
            method: ``"dense"`` applies the ideal band indicator through an
                eigendecomposition, for checks.
            token_span: ``(start, end)`` tokens that form the graph and are
                edited (slice semantics); others are neither edited nor read.
                For example ``(1, None)`` leaves the first token, whose
                residual norm is often far larger than the others', out of
                the graph. The graph is built from the attention sub-block of
                the span (see :mod:`~spectral_trust.interventions.filters`).
                With ``causal=True`` the bounds must be absolute (non-negative
                or None).
            positions: edit only these absolute positions (the filter still
                reads every position of the graph); non-negative with
                ``causal=True``.
            precision: ``"float64"`` (default; computed on the CPU when the
                model is on Apple MPS, which has no float64) or ``"float32"``,
                much faster on GPUs but noisier when the first token's
                residual dwarfs the others', as in GPT-2 (cached and
                recomputed generation then drift apart by about 0.2 logits).

        Padded positions (from a 2-D ``attention_mask`` passed as a keyword)
        are left out of the graph. Attention is switched to the eager
        implementation for the duration of the intervention.
        """
        if self.site != "post":
            raise ValueError("graph_filter acts on the residual stream after the block, paired "
                             "with that block's attention; use resid(layer - 1) for the input "
                             "of block layer")
        if causal:
            if token_span is not None and any(v is not None and v < 0 for v in token_span):
                raise ValueError("with causal=True, token_span bounds must be absolute (>= 0 or "
                                 "None): negative bounds move with the sequence length")
            if _has_negative(positions):
                raise ValueError("with causal=True, positions must be absolute (>= 0)")
        spec = FilterSpec(kind, cutoff, band, strength, graph, causal, order, method,
                          self.iv.config, precision)
        iv, arch, layer = self.iv, self.iv.arch, self.layer
        iv._require_eager()
        attn, block = arch.attention(layer), arch.block(layer)
        state = {"w": None, "X": None, "A": None, "reordered": False}
        edit = iv._new_edit(f"graph_filter at resid_post layer {layer}: {spec}")
        if causal:
            iv._causal_states.append(state)
            reg.acquire_beam_hooks(iv.model, iv, iv._on_reorder)

            def forget():        # by identity: dicts of tensors do not compare by value
                iv._causal_states[:] = [s for s in iv._causal_states if s is not state]
            edit.undo.append(forget)

        def on_attn(module, args, out):
            w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
            if w is None:
                raise RuntimeError(f"layer {layer}'s attention returned no weights; the graph "
                                   f"filter needs the eager attention implementation")
            state["w"] = w

        def run(Xf, A, valid, rows):
            S = Xf.shape[1]
            s, e = (0, S) if token_span is None else slice(*token_span).indices(S)[:2]
            pmask = _position_mask(positions, S, Xf.device)
            out = Xf[:, rows].clone()
            inside = [j for j, r in enumerate(rows) if s <= r < e]
            if not inside or e <= s:
                return out
            sub_rows = [rows[j] - s for j in inside]
            vs = valid[:, s:e] if valid is not None else None
            ps = pmask[s:e] if pmask is not None else None
            sub = apply_filter(spec, Xf[:, s:e], A[:, :, s:e, s:e] if A is not None else None,
                               valid=vs, rows=sub_rows if spec.causal else None, positions=ps)
            if not spec.causal:
                sub = sub[:, sub_rows]
            out[:, inside] = sub
            return out

        def on_block(module, args, out):
            X = first_hidden(out)
            w, state["w"] = state["w"], None
            if w is None:
                raise RuntimeError(f"layer {layer}'s attention weights were not captured")
            B, q, _ = X.shape
            k = w.shape[-1]
            valid = iv._valid(B, k, X.device)
            if q == k:
                Y = run(X, w, valid, list(range(q)))
                if spec.causal:
                    state["X"], state["A"] = X.detach(), w.detach()
                    state["reordered"] = False
                return replace_first(out, Y)
            if not spec.causal:
                raise RuntimeError(
                    "a cached generation step reached a symmetric graph filter, which would "
                    "have to change past positions; use causal=True, or generate with "
                    "use_cache=False")
            t = k - q
            Xp, Ap = state["X"], state["A"]
            if Xp is None or Xp.shape[0] != B or Xp.shape[1] != t:
                raise RuntimeError("the cached filter state does not match this generation "
                                   "step (a cache cropped, or reordered other than by beam "
                                   "search, is not supported)")
            if reg.generation_beams(iv.model) > 1 and not state["reordered"]:
                raise RuntimeError("beam search ran without the cache reordering this filter "
                                   "follows (unsupported transformers version); generate with "
                                   "use_cache=False")
            state["reordered"] = False
            Xf = torch.cat([Xp, X.detach()], dim=1)
            A = w.new_zeros(B, w.shape[1], k, k)
            A[:, :, :t, :t] = Ap
            A[:, :, t:, :] = w.detach()
            state["X"], state["A"] = Xf, A
            Y = run(torch.cat([Xp, X], dim=1), A, valid, list(range(t, k)))
            return replace_first(out, Y)

        edit.handles.append(attn.register_forward_hook(on_attn))
        edit.handles.append(block.register_forward_hook(on_block))
        return edit


class AttentionSite:
    """Attention heads at one layer; see :meth:`Intervention.attention`."""

    def __init__(self, iv: Intervention, layer: int, head: Heads):
        self.iv, self.layer = iv, layer
        H = iv.arch.n_heads
        if head is None:
            heads = list(range(H))
        elif isinstance(head, (int, np.integer)):
            heads = [int(head)]
        else:
            heads = [int(h) for h in head]
        for h in heads:
            if not 0 <= h < H:
                raise IndexError(f"head {h} out of range for {H} heads")
        self.heads = heads
        self.n_heads = H

    def _label(self) -> str:
        hs = "all heads" if len(self.heads) == self.n_heads else \
            ", ".join(f"{self.layer}.{h}" for h in self.heads)
        return f"layer {self.layer} ({hs})"

    # ── head outputs ───────────────────────────────────────────────────────

    def head_outputs(self, inputs) -> torch.Tensor:
        """Per-head outputs ``z`` of the selected heads on ``inputs`` (text,
        texts or ids), ``[B, S, len(heads), d_head]``, with this
        intervention's current edits active. Inputs of different lengths are
        run one at a time and returned as a list. The state of causal
        filters (and of the current call) is kept across this forward pass."""
        arch, layer, iv = self.iv.arch, self.layer, self.iv
        rows = _encode_list(inputs, iv.tokenizer)
        dh = arch.head_dim(layer)
        got = []

        def grab(x):
            got.append(x.detach().view(*x.shape[:-1], self.n_heads, dh)[..., self.heads, :].clone())
        # this forward pass runs through every open intervention on the model:
        # keep all of their causal-filter states and per-call records
        others = reg.open_interventions(iv.model)
        saved = [([dict(s) for s in o._causal_states], dict(o._fwd)) for o in others]
        handles = register_site_hook(arch, "z", layer, grab)
        try:
            with torch.no_grad():
                if len({len(r) for r in rows}) == 1:
                    iv.model(input_ids=torch.tensor(rows).to(_device(iv.model)))
                else:
                    for r in rows:
                        iv.model(input_ids=torch.tensor([r]).to(_device(iv.model)))
        finally:
            for h in handles:
                h.remove()
            for o, (states, fwd) in zip(others, saved):
                for s, old in zip(o._causal_states, states):
                    s.clear()
                    s.update(old)
                o._fwd.update(fwd)
        return got[0] if len(got) == 1 else [g[0] for g in got]

    def ablate(self, mode: str = "zero", reference=None, positions: Positions = None,
               seed: int = 0, per_position: bool = False) -> Edit:
        """Ablate the selected heads' outputs (``z``, before the output
        projection, so a head's whole contribution is replaced).

        Args:
            mode: ``"zero"``; ``"mean"``: the mean of the head's output over
                every token of ``reference`` (or per absolute position with
                ``per_position=True``); ``"resample"``: the head's output on
                a ``reference`` item, at the same absolute position. The item
                is drawn once per input item and kept for the life of the
                edit (seeded); the beams of one input share it. Repeated
                calls, cached generation and beam search therefore agree with
                recomputation. The reference must be at least as long as the
                inputs.
            reference: text, texts or ids to compute the replacement on (run
                once, now, with the edits registered so far; texts of
                different lengths are run one at a time, so no padding enters
                the mean), or a tensor of head outputs ``[R, S, len(heads),
                d_head]`` (or ``[len(heads), d_head]`` for ``"mean"``).
                Required by ``"mean"`` and ``"resample"``.
            positions: ablate only at these absolute positions.
        """
        if mode not in ("zero", "mean", "resample"):
            raise ValueError(f"mode must be 'zero', 'mean' or 'resample', got {mode!r}")
        if mode in ("mean", "resample") and reference is None:
            raise ValueError(f"mode={mode!r} needs a reference: the mean of the current call "
                             f"would change with the batch and be empty at cached steps")
        iv, arch, layer, heads = self.iv, self.iv.arch, self.layer, self.heads
        dh, H = arch.head_dim(layer), self.n_heads
        ref = None
        if mode in ("mean", "resample"):
            if torch.is_tensor(reference) and reference.is_floating_point():
                ref = reference.detach()
            else:
                ref = self.head_outputs(reference)
            if isinstance(ref, list):                       # inputs of different lengths
                if mode == "resample" or per_position:
                    raise ValueError("resample and per-position means need reference inputs of "
                                     "equal length")
                ref = torch.cat([r.reshape(-1, len(heads), dh) for r in ref]).mean(dim=0)
            elif mode == "mean" and ref.dim() == 4:
                ref = ref.mean(dim=0) if per_position else ref.mean(dim=(0, 1))
            if mode == "resample" and ref.dim() != 4:
                raise ValueError("resample needs head outputs [R, S, heads, d_head]")
        gen = torch.Generator().manual_seed(int(seed))
        picks: List[int] = []
        edit = iv._new_edit(f"ablate ({mode}) {self._label()}")
        if mode == "resample":                 # to know the beam width during generate
            reg.acquire_beam_hooks(iv.model, iv, iv._on_reorder)
        idx = torch.as_tensor(heads)

        def fn(z):
            iv._check_negative(positions, "ablate")
            B, q, _ = z.shape
            zh = z.view(B, q, H, dh)
            new = zh.clone()
            qpos = iv._positions(q, device=z.device)
            if mode == "zero":
                rep = torch.zeros(B, q, len(heads), dh, dtype=z.dtype, device=z.device)
            elif mode == "mean":
                r = ref.to(device=z.device, dtype=z.dtype)
                if r.dim() == 3:                                   # per position
                    if int(qpos.max()) >= r.shape[0]:
                        raise ValueError("the reference is shorter than the input")
                    rep = r[qpos].unsqueeze(0).expand(B, -1, -1, -1)
                else:
                    rep = r.view(1, 1, len(heads), dh).expand(B, q, -1, -1)
            else:
                if int(qpos.max()) >= ref.shape[1]:
                    raise ValueError("resample: the reference is shorter than the input")
                # one draw per input item, shared by its beams: generate expands
                # each item into `beams` consecutive rows and beam search only
                # reorders rows within an item, so cached and recomputed agree
                item = torch.arange(B) // max(1, reg.generation_beams(iv.model))
                while len(picks) <= int(item.max()):
                    picks.append(int(torch.randint(ref.shape[0], (1,), generator=gen)))
                pick = torch.as_tensor(picks)[item]
                rep = ref[pick][:, qpos.cpu()].to(device=z.device, dtype=z.dtype)
            m = _position_mask(positions, int(qpos.max()) + 1, z.device)
            if m is not None:
                sel = m[qpos].view(1, q, 1, 1)
                rep = torch.where(sel, rep, zh[:, :, idx])
            new[:, :, idx] = rep
            return new.view(B, q, H * dh)
        edit.handles.extend(register_site_hook(arch, "z", layer, fn))
        return edit

    # ── attention patterns ─────────────────────────────────────────────────

    def edit_pattern(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Edit:
        """Edit the attention probabilities of the selected heads before they
        meet the values: ``fn(weights [B, len(heads), q, k], query_positions
        [q]) -> weights`` (TransformerLens ``hook_pattern``). The output is
        rebuilt as ``weights @ values``; this is checked against the model's
        own output once per module. Needs transformers >= 4.48 (the
        module-level ``eager_attention_forward``)."""
        iv, heads, H = self.iv, self.heads, self.n_heads
        idx = torch.as_tensor(heads)
        edit = iv._new_edit(f"edit_pattern {self._label()}")

        def run(w, qpos):
            if len(heads) == H:
                return fn(w, qpos)
            new = w.clone()
            new[:, idx] = fn(w[:, idx], qpos)
            return new
        iv._install_pattern_editor(edit, self.layer, run)
        return edit

    def remove_edges(self, key_mask, keep_sink: bool = True, queries: Positions = None,
                     renormalize: bool = True, sink: int = 0) -> Edit:
        """Remove attention edges of the selected heads.

        Removed entries are set to zero. With ``keep_sink=True`` (the
        default) each edited row is then renormalised over its remaining
        non-sink keys while the sink entry (the first token, ``sink=0``)
        keeps its value, so the row's sink mass is held fixed and the
        removed mass goes back to the content keys only. With
        ``keep_sink=False`` the row is renormalised over every remaining key,
        and with ``renormalize=False`` the entries are only zeroed.

        The three choices send the removed mass to different places, and none
        of them is a pure edge ablation when the removed edges carry much of
        the row. Plain renormalisation hands most of it to the sink, which in
        trained models already holds a large share of every row. Holding the
        sink fixed hands it to the content keys that are left: each is
        multiplied by the row's non-sink mass before the edit divided by the
        non-sink mass kept. When the removed edges held most of that mass,
        the factor is large and the edit amplifies the few edges left. In the
        GPT-2 small IOI use case (removing the name movers' END edges to the
        names, 0.77 of those rows), the median factor on the kept edges was
        13 (head 10.0), 55 (9.9) and 181 (9.6), and plain renormalisation
        raised the first token's share of the rows from 0.20 to 0.87. A
        ``UserWarning`` is issued, once per ``remove_edges`` call, when the
        factor of any edited row exceeds ``KEEP_SINK_WARN_FACTOR`` (10). In
        that regime, report the removed mass and compare the three choices.

        Args:
            key_mask: which edges to remove (True = remove): ``[k]`` keys
                removed for every query, ``[S, S]`` (query, key) or
                ``[B, S, S]`` over absolute positions, or a callable
                ``(query_positions [q], n_keys) -> mask [q, k]`` or
                ``[B, q, k]``.
            keep_sink: hold the sink entry fixed (see above). Removing the
                sink itself then raises; pass ``keep_sink=False`` for that.
            queries: edit only these absolute query positions (default every
                query).
            renormalize: False only zeroes the entries (rows then sum to
                less than one).
            sink: position of the sink, counted from the first real token
                (left padding, read from a 2-D ``attention_mask`` keyword,
                is skipped).

        A row whose non-sink keys are all removed keeps its sink entry and
        sums to less than one. A row in which every entry to remove is
        already zero is returned bit for bit.
        """
        iv = self.iv
        km = key_mask if callable(key_mask) else torch.as_tensor(key_mask).bool()
        warned = [False]                       # one warning per remove_edges call

        def fn(w, qpos):
            iv._check_negative(queries, "remove_edges")
            B, Hs, q, k = w.shape
            if callable(km):
                rm = torch.as_tensor(km(qpos, k)).to(device=w.device).bool()
            elif km.dim() == 1:
                rm = torch.zeros(k, dtype=torch.bool, device=w.device)
                rm[: min(k, km.numel())] = km[:k].to(w.device)
                rm = rm.view(1, k).expand(q, k)
            else:
                if int(qpos.max()) >= km.shape[-2]:
                    raise ValueError("key_mask has fewer query rows than the input")
                rm = km.to(w.device)[..., qpos, :][..., :k]
                if rm.shape[-1] < k:
                    rm = torch.cat([rm, rm.new_zeros(*rm.shape[:-1], k - rm.shape[-1])], -1)
            if rm.dim() == 2:
                rm = rm.unsqueeze(0)
            rm = rm.unsqueeze(1)                                   # [B|1, 1, q, k]
            qm = _position_mask(queries, int(qpos.max()) + 1, w.device)
            if qm is not None:
                rm = rm & qm[qpos].view(1, 1, q, 1)
            valid = iv._valid(B, k, w.device)
            first = valid.float().argmax(dim=1) if valid is not None else \
                torch.zeros(B, dtype=torch.long, device=w.device)
            sink_pos = (first + int(sink)).clamp(max=k - 1)
            out, factor = _remove_edges(w, rm, sink_pos, keep_sink, renormalize)
            if not warned[0]:
                warned[0] = _warn_keep_sink_factor(factor, KEEP_SINK_WARN_FACTOR,
                                                   self._label())
            return out

        edit = self.edit_pattern(fn)
        edit.description = (f"remove_edges {self._label()} (keep_sink={keep_sink}, "
                            f"renormalize={renormalize})")
        return edit


def remove_edges_from_pattern(w: torch.Tensor, remove: torch.Tensor, sink: torch.Tensor,
                              keep_sink: bool = True, renormalize: bool = True,
                              warn_factor: Optional[float] = KEEP_SINK_WARN_FACTOR
                              ) -> torch.Tensor:
    """Remove edges from attention probabilities ``w [B, H, q, k]``.

    ``remove`` is a boolean mask broadcastable to ``w``; ``sink [B]`` the
    sink key of each item. See :meth:`AttentionSite.remove_edges`. Rows in
    which no nonzero entry is removed are returned bit for bit. With
    ``keep_sink=True``, one ``UserWarning`` is issued when the kept non-sink
    entries of any edited row are multiplied by more than ``warn_factor``
    (``None``: never).
    """
    out, factor = _remove_edges(w, remove, sink, keep_sink, renormalize)
    if warn_factor is not None:
        _warn_keep_sink_factor(factor, warn_factor)
    return out


def _remove_edges(w: torch.Tensor, remove: torch.Tensor, sink: torch.Tensor,
                  keep_sink: bool, renormalize: bool):
    """The edit and, for ``keep_sink=True``, the factor applied to the kept
    non-sink entries of each row (``[B, H, q, 1]``, 0 on rows not edited;
    ``None`` otherwise)."""
    B, H, q, k = w.shape
    remove = remove.expand(B, H, q, k)
    sink_hot = torch.zeros(B, 1, 1, k, dtype=torch.bool, device=w.device)
    sink_hot[torch.arange(B, device=w.device), 0, 0, sink.to(w.device)] = True
    if keep_sink and bool((remove & sink_hot).any()):
        raise ValueError("keep_sink=True holds the sink entry fixed, but the mask removes it; "
                         "pass keep_sink=False to remove sink edges")
    kept = w.masked_fill(remove, 0.0)
    if not renormalize:
        return kept, None
    changed = (remove & (w != 0)).any(dim=-1, keepdim=True)
    row = w.sum(dim=-1, keepdim=True)
    tiny = torch.finfo(w.dtype).tiny
    if keep_sink:
        s = (w * sink_hot).sum(dim=-1, keepdim=True)
        others = kept.masked_fill(sink_hot, 0.0)
        tot = others.sum(dim=-1, keepdim=True)
        scale = torch.where(tot > 0, (row - s) / tot.clamp_min(tiny), torch.zeros_like(tot))
        new = torch.where(sink_hot, w, others * scale)
        return torch.where(changed, new, w), torch.where(changed, scale, torch.zeros_like(scale))
    tot = kept.sum(dim=-1, keepdim=True)
    new = torch.where(tot > 0, kept * (row / tot.clamp_min(tiny)), kept)
    return torch.where(changed, new, w), None


def _warn_keep_sink_factor(factor: Optional[torch.Tensor], limit: float,
                           where: str = "") -> bool:
    """Warn once if any row's keep-sink factor exceeds ``limit``; returns
    whether it warned."""
    if factor is None or factor.numel() == 0:
        return False
    big = factor.detach().float() > limit
    if not bool(big.any()):
        return False
    top = float(factor.detach().float().max())
    at = f" at {where}" if where else ""
    warnings.warn(
        f"remove_edges(keep_sink=True){at} multiplied the remaining non-sink attention of "
        f"{int(big.sum())} row(s) by more than {limit:g} (largest factor {top:.3g}): the "
        f"removed edges held most of those rows' non-sink mass, so the edit amplifies the "
        f"edges left as well as removing the others. Report the removed mass and compare "
        f"keep_sink=False and renormalize=False.", UserWarning, stacklevel=3)
    return True


class WeightSite:
    """A weight matrix of one block; see :meth:`Intervention.weights`."""

    def __init__(self, iv: Intervention, layer: int, module: str):
        self.iv, self.layer = iv, layer
        self.path, self.module = iv.arch.module(layer, module)

    @property
    def weight(self) -> torch.Tensor:
        """The weight in ``nn.Linear`` orientation ``[out, in]`` (a view)."""
        return linear_weight(self.module)

    def snr(self) -> float:
        """``sigma_1 / median(sigma)`` of the current weight
        (:func:`~spectral_trust.interventions.weight_snr`)."""
        return matrix_snr(self.weight)

    def svd_scale(self, alpha: float, k: Optional[int] = None) -> Edit:
        """Rescale the singular values, ``sigma' = sigma (1 + alpha (sigma -
        mean) / std)`` (:func:`~spectral_trust.interventions.weights.svd_scale_matrix`).
        ``k`` limits the edit to the top ``k`` singular values; the bias, if
        any, is kept.

        Edits of one parameter stack, in any number of interventions: each
        applies to the weight as the edits before it left it. Removing one
        rebuilds the parameter from its original value with the others
        re-applied in order, and removing the last one restores the original
        bit for bit."""
        param = self.module.weight
        edit = self.iv._new_edit(f"svd_scale(alpha={alpha}, k={k}) layer {self.layer} "
                                 f"{self.path}")
        module = self.module

        def apply():
            with torch.no_grad():
                W = linear_weight(module)
                W.copy_(svd_scale_matrix(W, alpha, k))
        try:
            reg.push_weight_op(param, self.iv, edit, apply)
        except Exception:
            edit.active = False                       # nothing was applied or recorded
            self.iv.edits[:] = [e for e in self.iv.edits if e is not edit]
            raise
        edit.undo.append(lambda: reg.drop_weight_ops(param, [edit]))
        return edit


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def _encode_list(inputs, tokenizer) -> List[List[int]]:
    """Token id lists from text, texts, or ids (lengths may differ)."""
    if torch.is_tensor(inputs):
        ids = inputs.long()
        ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
        return ids.tolist()
    if isinstance(inputs, str):
        inputs = [inputs]
    items = list(inputs)
    if items and all(isinstance(i, (int, np.integer)) for i in items):
        return [[int(i) for i in items]]
    rows = []
    for it in items:
        if isinstance(it, str):
            if tokenizer is None:
                raise ValueError("text input needs a tokenizer; pass token ids or tokenizer=")
            rows.append(list(tokenizer(it)["input_ids"]))
        else:
            rows.append([int(i) for i in (it.tolist() if torch.is_tensor(it) else it)])
    return rows


def _encode(inputs, tokenizer) -> torch.Tensor:
    """Token ids ``[B, S]`` from text, texts, or ids (all of equal length)."""
    rows = _encode_list(inputs, tokenizer)
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError(f"inputs have different lengths {sorted(lengths)}; positions must "
                         f"align, so pass inputs of equal length (or one at a time)")
    return torch.tensor(rows, dtype=torch.long)


def intervene(model, config: Optional[GSPConfig] = None, tokenizer=None) -> Intervention:
    """Edit a model with hooks; every edit is undone on exit.

    Works on a :class:`~spectral_trust.SpectralModel` or a plain Hugging Face
    causal LM (GPT-2, Llama, Qwen, Mistral, Gemma, GPT-NeoX, OPT layouts).
    Interventions on one model may be nested and closed in any order.

    Examples:
        >>> with st.intervene(model) as iv:
        ...     iv.resid(6).graph_filter("lowpass", cutoff=0.5, strength=0.8, causal=True)
        ...     iv.attention(9, 6).ablate("zero")
        ...     iv.weights(7).svd_scale(-0.05)
        ...     out = model.generate(ids, max_new_tokens=20)

    Args:
        model: the model to edit.
        config: graph construction for the graph filters (default: the
            SpectralModel's config, or the library defaults).
        tokenizer: for text references in :meth:`AttentionSite.ablate`
            (default: the SpectralModel's tokenizer).

    Added in 0.5.0.
    """
    return Intervention(model, config=config, tokenizer=tokenizer)
