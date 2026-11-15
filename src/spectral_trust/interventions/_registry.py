# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Shared, reference-counted changes to a model and to ``transformers``.

Several interventions may be open on one model, nested or not, and closed in
any order. Everything they share lives here, keyed by the object it changes,
with the set of interventions that hold it:

- the attention implementation of a model (switched to eager while any
  holder needs attention weights, then put back);
- the module-level ``eager_attention_forward`` of a modeling module (wrapped
  while any pattern edit exists, then put back);
- the instance attributes ``generate`` and ``_reorder_cache`` of a model,
  through which the causal graph filters follow beam search;
- weight edits, as a stack of operations per parameter on top of a pristine
  copy, so that removing any edit rebuilds the others and removing the last
  one restores the parameter bit for bit.
"""
import sys
import threading
from typing import Callable, Dict, List

import torch

_LOCK = threading.RLock()

# ── attention implementation ─────────────────────────────────────────────────

_EAGER_IMPL: Dict[int, dict] = {}


def acquire_eager(arch, owner) -> None:
    key = id(arch.model)
    with _LOCK:
        st = _EAGER_IMPL.get(key)
        if st is None:
            impl = arch.attn_implementation()
            st = {"orig": impl, "owners": set(), "arch": arch}
            _EAGER_IMPL[key] = st
            if impl not in (None, "eager"):
                arch.set_attn_implementation("eager")
        st["owners"].add(id(owner))


def release_eager(model, owner) -> None:
    with _LOCK:
        st = _EAGER_IMPL.get(id(model))
        if st is None:
            return
        st["owners"].discard(id(owner))
        if not st["owners"]:
            if st["orig"] not in (None, "eager"):
                st["arch"].set_attn_implementation(st["orig"])
            del _EAGER_IMPL[id(model)]


# ── attention-pattern edits through eager_attention_forward ──────────────────

#: id(attention module) -> [(owner id, fn(weights, module) -> weights)]
PATTERN_EDITS: Dict[int, List] = {}
#: id(attention module) -> whether the wrapper ran for it in the current call
PATTERN_CALLED: Dict[int, bool] = {}
_VERIFIED = set()
_EAGER_FN: Dict[str, dict] = {}


def _make_wrapper(orig):
    def eager_attention_forward(module, query, key, value, attention_mask, *args, **kwargs):
        out, weights = orig(module, query, key, value, attention_mask, *args, **kwargs)
        edits = PATTERN_EDITS.get(id(module))
        if not edits:
            return out, weights
        PATTERN_CALLED[id(module)] = True
        if weights is None:
            raise RuntimeError("the eager attention returned no weights")
        v = value
        n_rep = weights.shape[1] // v.shape[1]
        if n_rep > 1:
            v = v.repeat_interleave(n_rep, dim=1)
        if id(module) not in _VERIFIED:
            with torch.no_grad():
                check = torch.matmul(weights.detach().to(v.dtype), v.detach()).transpose(1, 2)
                ref = out.detach()
                tol = 1e-4 * float(ref.abs().max()) + 1e-5
                bad = check.shape != ref.shape or float((check - ref).abs().max()) > tol
            if bad:
                raise RuntimeError(
                    f"cannot rebuild {type(module).__name__}'s output from its attention weights "
                    f"(weights @ values); pattern edits are not supported for it")
            _VERIFIED.add(id(module))
        new = weights
        for _, fn in list(edits):
            new = fn(new, module)
        out = torch.matmul(new.to(v.dtype), v).transpose(1, 2).contiguous()
        return out, new
    return eager_attention_forward


#: Pattern edits rebuild attention outputs inside the module-level
#: ``eager_attention_forward`` that transformers introduced in 4.48.
PATTERN_EDIT_MIN_TRANSFORMERS = (4, 48)


def _version_tuple(v: str):
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (2 - len(parts)))


def transformers_version() -> str:
    """The installed transformers version string."""
    import transformers
    return str(transformers.__version__)


def check_pattern_edit_support() -> None:
    """Raise a clear error when the installed transformers is older than the
    first version with a module-level ``eager_attention_forward``."""
    found = transformers_version()
    if _version_tuple(found) < PATTERN_EDIT_MIN_TRANSFORMERS:
        need = ".".join(str(v) for v in PATTERN_EDIT_MIN_TRANSFORMERS)
        raise ValueError(
            f"attention-pattern edits (remove_edges, edit_pattern) need transformers >= "
            f"{need}; found {found}. The rest of spectral_trust.interventions works with "
            f"older versions.")


def acquire_pattern_editor(attn_module, owner, fn) -> Callable[[], None]:
    """Route ``attn_module``'s eager attention through ``fn``. Returns the
    function that undoes it."""
    check_pattern_edit_support()
    modname = type(attn_module).__module__
    mod = sys.modules.get(modname)
    if mod is None or not callable(getattr(mod, "eager_attention_forward", None)):
        import transformers
        raise ValueError(
            f"{type(attn_module).__name__} has no module-level eager_attention_forward "
            f"(transformers {transformers.__version__}); attention-pattern edits need "
            f"transformers >= 4.48 and a model that uses it")
    entry = (id(owner), fn)
    with _LOCK:
        st = _EAGER_FN.get(modname)
        if st is None:
            st = {"orig": mod.eager_attention_forward, "mod": mod, "count": 0}
            _EAGER_FN[modname] = st
            mod.eager_attention_forward = _make_wrapper(st["orig"])
        st["count"] += 1
        PATTERN_EDITS.setdefault(id(attn_module), []).append(entry)

    def release():
        with _LOCK:
            lst = PATTERN_EDITS.get(id(attn_module), [])
            if entry in lst:
                lst.remove(entry)
            if not lst:
                PATTERN_EDITS.pop(id(attn_module), None)
                _VERIFIED.discard(id(attn_module))
            s = _EAGER_FN.get(modname)
            if s is not None:
                s["count"] -= 1
                if s["count"] <= 0:
                    s["mod"].eager_attention_forward = s["orig"]
                    del _EAGER_FN[modname]
    return release


# ── beam search: generate and _reorder_cache on the model instance ──────────

_BEAMS: Dict[int, dict] = {}


def acquire_beam_hooks(model, owner, on_reorder: Callable[[torch.Tensor], None]) -> None:
    """Call ``on_reorder(beam_idx)`` whenever generation reorders the cache,
    and keep ``generation_beams(model)`` up to date."""
    key = id(model)
    with _LOCK:
        st = _BEAMS.get(key)
        if st is None:
            st = {"callbacks": {}, "beams": 1, "model": model,
                  "had_reorder": "_reorder_cache" in model.__dict__,
                  "had_generate": "generate" in model.__dict__}
            orig_reorder = getattr(model, "_reorder_cache", None)
            orig_generate = model.generate

            def _reorder_cache(past_key_values, beam_idx):
                for cb in list(st["callbacks"].values()):
                    cb(beam_idx)
                if orig_reorder is not None:
                    return orig_reorder(past_key_values, beam_idx)
                past_key_values.reorder_cache(beam_idx)
                return past_key_values

            def generate(*args, **kwargs):
                # generate(inputs, generation_config, ...): the config may come positionally
                gc = kwargs.get("generation_config")
                if gc is None and len(args) > 1 and hasattr(args[1], "num_beams"):
                    gc = args[1]
                if gc is None:
                    gc = getattr(model, "generation_config", None)
                beams = kwargs.get("num_beams")
                if beams is None:
                    beams = getattr(gc, "num_beams", 1) or 1
                prev, st["beams"] = st["beams"], int(beams)
                try:
                    return orig_generate(*args, **kwargs)
                finally:
                    st["beams"] = prev

            st["orig_reorder"], st["orig_generate"] = orig_reorder, orig_generate
            model._reorder_cache = _reorder_cache
            model.generate = generate
            _BEAMS[key] = st
        st["callbacks"][id(owner)] = on_reorder


def release_beam_hooks(model, owner) -> None:
    with _LOCK:
        st = _BEAMS.get(id(model))
        if st is None:
            return
        st["callbacks"].pop(id(owner), None)
        if not st["callbacks"]:
            for name, had, orig in (("_reorder_cache", st["had_reorder"], st["orig_reorder"]),
                                    ("generate", st["had_generate"], st["orig_generate"])):
                if had:
                    setattr(model, name, orig)
                else:
                    model.__dict__.pop(name, None)
            del _BEAMS[id(model)]


def generation_beams(model) -> int:
    st = _BEAMS.get(id(model))
    return st["beams"] if st is not None else 1


# ── weight edits ─────────────────────────────────────────────────────────────

#: id(parameter) -> {"param", "pristine", "ops": [(owner id, edit, apply())]}
_WEIGHTS: Dict[int, dict] = {}


def push_weight_op(param: torch.nn.Parameter, owner, edit, apply: Callable[[], None]) -> None:
    """Apply an edit of ``param`` on top of its current state, and record it
    only if it succeeded. A failed edit leaves no trace: no operation in the
    stack, and no pristine copy if it would have been the first edit."""
    with _LOCK:
        st = _WEIGHTS.get(id(param))
        pristine = param.detach().clone() if st is None else None
        apply()                                     # raises before anything is recorded
        if st is None:
            st = {"param": param, "pristine": pristine, "ops": []}
            _WEIGHTS[id(param)] = st
        st["ops"].append((id(owner), edit, apply))


def drop_weight_ops(param: torch.nn.Parameter, edits) -> None:
    """Remove ``edits`` from ``param``'s stack: the parameter is rebuilt from
    its pristine copy with the remaining edits applied in order, or restored
    bit for bit when none remain."""
    with _LOCK:
        st = _WEIGHTS.get(id(param))
        if st is None:
            return
        drop = {id(e) for e in edits}
        st["ops"] = [op for op in st["ops"] if id(op[1]) not in drop]
        with torch.no_grad():
            param.copy_(st["pristine"])
        if not st["ops"]:
            del _WEIGHTS[id(param)]
            return
        try:
            for _, _, apply in st["ops"]:
                apply()
        except Exception:
            # leave the parameter at its original value rather than half rebuilt;
            # the remaining edits stay registered, so removing them restores it too
            with torch.no_grad():
                param.copy_(st["pristine"])
            raise


# ── open interventions, per model ────────────────────────────────────────────

#: id(model) -> [open Intervention objects]
OPEN: Dict[int, list] = {}


def register_open(model, iv) -> None:
    with _LOCK:
        OPEN.setdefault(id(model), []).append(iv)


def unregister_open(model, iv) -> None:
    with _LOCK:
        lst = OPEN.get(id(model), [])
        lst[:] = [x for x in lst if x is not iv]
        if not lst:
            OPEN.pop(id(model), None)


def open_interventions(model) -> list:
    return list(OPEN.get(id(model), []))
