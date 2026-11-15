# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Hook points, named after TransformerLens' hooks:

==============  ===========================================  ======================
site            tensor                                       TransformerLens hook
==============  ===========================================  ======================
``resid_pre``   input of block l, ``[B, S, d]``              ``hook_resid_pre``
``resid_post``  output of block l, ``[B, S, d]``             ``hook_resid_post``
``attn_out``    attention output (after its projection)      ``hook_attn_out``
``mlp_out``     MLP output                                   ``hook_mlp_out``
``z``           per-head attention outputs, input of the     ``attn.hook_z``
                output projection, ``[B, S, H * d_head]``
==============  ===========================================  ======================

``resid_post`` of layer l is ``hidden_states[l + 1]`` of a Hugging Face
model for every layer but the last (Hugging Face applies the final norm to
its last hidden state; ``resid_post`` is taken before it). Attention
patterns are read and edited separately (``pattern``).
"""
from typing import Callable, List, Optional

import torch

from ._arch import Arch, first_hidden, replace_first

SITES = ("resid_pre", "resid_post", "attn_out", "mlp_out", "z")
_ALIASES = {"head": "z", "hook_z": "z", "resid": "resid_post", "hook_resid_pre": "resid_pre",
            "hook_resid_post": "resid_post", "hook_attn_out": "attn_out",
            "hook_mlp_out": "mlp_out"}


def canonical_site(site: str) -> str:
    s = _ALIASES.get(site, site)
    if s not in SITES:
        raise ValueError(f"site must be one of {SITES} (or 'head'), got {site!r}")
    return s


def register_site_hook(arch: Arch, site: str, layer: int,
                       fn: Callable[[torch.Tensor], Optional[torch.Tensor]]) -> List:
    """Register ``fn(x) -> new x (or None to keep x)`` at a site. Returns the
    torch hook handles."""
    site = canonical_site(site)
    if site == "resid_pre":
        def pre(module, args, kwargs):
            if args:
                y = fn(args[0])
                return None if y is None else ((y,) + tuple(args[1:]), kwargs)
            y = fn(kwargs["hidden_states"])
            return None if y is None else (args, dict(kwargs, hidden_states=y))
        return [arch.block(layer).register_forward_pre_hook(pre, with_kwargs=True)]
    if site == "z":
        def zpre(module, args, kwargs):
            y = fn(args[0])
            return None if y is None else ((y,) + tuple(args[1:]), kwargs)
        return [arch.attention_output(layer).register_forward_pre_hook(zpre, with_kwargs=True)]
    module = {"resid_post": arch.block, "attn_out": arch.attention, "mlp_out": arch.mlp}[site](layer)

    def post(mod, args, out):
        y = fn(first_hidden(out))
        return None if y is None else replace_first(out, y)
    return [module.register_forward_hook(post)]
