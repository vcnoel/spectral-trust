# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Where the pieces of a Hugging Face decoder live: the blocks, each block's
attention module and its output projection, the MLP and its output
projection. Interventions hook these modules, so the same code serves
GPT-2 (``transformer.h``, ``attn.c_proj``, ``mlp.c_proj``), the Llama,
Qwen, Mistral and Gemma families (``model.layers``, ``self_attn.o_proj``,
``mlp.down_proj``), GPT-NeoX (``attention.dense``, ``mlp.dense_4h_to_h``)
and OPT (``self_attn.out_proj``, ``fc2``).
"""
import logging
from typing import List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

_BLOCK_PATHS = (
    "transformer.h", "model.layers", "model.language_model.layers",
    "language_model.model.layers", "language_model.layers", "gpt_neox.layers",
    "model.decoder.layers", "transformer.blocks", "transformer.layers", "layers",
)
_ATTN_NAMES = ("self_attn", "attn", "attention", "self_attention")
_ATTN_OUT_NAMES = ("o_proj", "c_proj", "dense", "out_proj", "wo")
_MLP_NAMES = ("mlp", "feed_forward", "ffn")
_MLP_OUT_NAMES = ("down_proj", "c_proj", "dense_4h_to_h", "fc2", "w2")


def unwrap(model):
    """The Hugging Face model behind a :class:`~spectral_trust.SpectralModel`
    (or the model itself)."""
    from ..model import SpectralModel
    return model.model if isinstance(model, SpectralModel) else model


def get_path(root: nn.Module, path: str):
    obj = root
    for part in path.split("."):
        if part.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential, list, tuple)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _first_attr(module: nn.Module, names) -> Tuple[Optional[str], Optional[nn.Module]]:
    for n in names:
        sub = getattr(module, n, None)
        if isinstance(sub, nn.Module):
            return n, sub
    return None, None


def _in_features(proj: nn.Module) -> int:
    """Input width of a Linear or a GPT-2 Conv1D (weight ``[in, out]``)."""
    if isinstance(proj, nn.Linear):
        return proj.in_features
    w = getattr(proj, "weight", None)
    if w is not None and hasattr(proj, "nf"):          # transformers Conv1D
        return int(w.shape[0])
    if w is not None and w.dim() == 2:
        return int(w.shape[1])
    raise ValueError(f"cannot read the input width of {type(proj).__name__}")


class Arch:
    """Module lookup for one decoder-only causal LM."""

    def __init__(self, model):
        self.model = unwrap(model)
        self.config = getattr(self.model, "config", None)
        self.blocks_path, self.blocks = self._find_blocks()
        self.n_layers = len(self.blocks)

    def _find_blocks(self) -> Tuple[str, nn.ModuleList]:
        from ..model import _num_layers
        n = _num_layers(self.config) if self.config is not None else None
        for path in _BLOCK_PATHS:
            try:
                obj = get_path(self.model, path)
            except AttributeError:
                continue
            if isinstance(obj, nn.ModuleList) and len(obj) and (n is None or len(obj) == n):
                return path, obj
        best = None
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) and (n is None or len(mod) == n):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        if best is None:
            raise ValueError(f"cannot find the decoder blocks of {type(self.model).__name__}")
        return best

    def layer_index(self, layer: int) -> int:
        lay = int(layer)
        if lay < 0:
            lay += self.n_layers
        if not 0 <= lay < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.n_layers} layers")
        return lay

    def block(self, layer: int) -> nn.Module:
        return self.blocks[self.layer_index(layer)]

    def attention(self, layer: int) -> nn.Module:
        name, mod = _first_attr(self.block(layer), _ATTN_NAMES)
        if mod is None:
            raise ValueError(
                f"layer {layer} has no softmax attention module (looked for "
                f"{', '.join(_ATTN_NAMES)}); on hybrid models only the full-attention "
                f"layers can be used")
        return mod

    def attention_output(self, layer: int) -> nn.Module:
        """The attention's output projection, whose input is the
        concatenation of the per-head outputs (TransformerLens ``hook_z``)."""
        attn = self.attention(layer)
        name, mod = _first_attr(attn, _ATTN_OUT_NAMES)
        if mod is None:
            raise ValueError(f"cannot find the output projection of layer {layer}'s attention")
        return mod

    def mlp(self, layer: int) -> nn.Module:
        name, mod = _first_attr(self.block(layer), _MLP_NAMES)
        if mod is None:
            raise ValueError(f"layer {layer} has no MLP module (looked for {', '.join(_MLP_NAMES)})")
        return mod

    def module(self, layer: int, path: str) -> Tuple[str, nn.Module]:
        """A submodule of block ``layer`` by dotted path. ``"mlp.down_proj"``
        falls back to the MLP output projection of architectures that name it
        differently (GPT-2 ``mlp.c_proj``, GPT-NeoX ``mlp.dense_4h_to_h``)."""
        block = self.block(layer)
        try:
            return path, get_path(block, path)
        except (AttributeError, IndexError):
            pass
        if path in ("mlp.down_proj", "mlp.out"):
            mlp_name, mlp = _first_attr(block, _MLP_NAMES)
            if mlp is not None:
                name, mod = _first_attr(mlp, _MLP_OUT_NAMES)
                if mod is not None:
                    resolved = f"{mlp_name}.{name}"
                    logger.info("layer %d: %r resolved to %r", layer, path, resolved)
                    return resolved, mod
        raise ValueError(f"block {layer} has no submodule {path!r}")

    @property
    def n_heads(self) -> int:
        from ..model import _num_heads
        n = _num_heads(self.config)
        if n is None:
            raise ValueError("cannot read the number of attention heads from the config")
        return n

    def head_dim(self, layer: int) -> int:
        width = _in_features(self.attention_output(layer))
        if width % self.n_heads:
            raise ValueError(f"the attention output width {width} is not a multiple of "
                             f"{self.n_heads} heads")
        return width // self.n_heads

    def attn_implementation(self) -> Optional[str]:
        return getattr(self.config, "_attn_implementation", None)

    def set_attn_implementation(self, impl: str) -> None:
        setter = getattr(self.model, "set_attn_implementation", None)
        if callable(setter):
            setter(impl)
        else:
            self.config._attn_implementation = impl


def first_hidden(x):
    """The hidden-state tensor of a module input or output (a tensor, or a
    tuple whose first element is the tensor)."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return x[0]
    raise TypeError(f"expected a tensor or a tuple starting with one, got {type(x).__name__}")


def replace_first(x, new):
    if torch.is_tensor(x):
        return new
    return (new,) + tuple(x[1:]) if isinstance(x, tuple) else [new] + list(x[1:])


def layer_list(layers, n_layers: int) -> List[int]:
    if isinstance(layers, int):
        layers = [layers]
    out = []
    for lay in layers:
        lay = int(lay)
        out.append(lay + n_layers if lay < 0 else lay)
    return out
