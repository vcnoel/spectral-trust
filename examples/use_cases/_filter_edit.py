# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""The edit of filter_side_effects.py: a causal graph filter on the residual
stream after one block, on tokens 1.., or a control of the same size."""
import numpy as np
import torch

import spectral_trust as st
from spectral_trust.interventions._arch import Arch, first_hidden, replace_first

FILTERS = {"lowpass": dict(kind="lowpass", cutoff=1.0),
           "highpass": dict(kind="highpass", cutoff=1.0),
           "bandpass": dict(kind="bandpass", band=(0.5, 1.5))}
CONDITIONS = ("attention", "random", "path", "scale")


def shuffle_index(n_items, S, seeds):
    """``[B, S, S]``: row r a seeded permutation of keys 0..r."""
    P = torch.arange(S).repeat(n_items, S, 1)
    for b, s in enumerate(seeds):
        rng = np.random.default_rng([1, int(s)])
        for r in range(1, S):
            P[b, r, :r + 1] = torch.from_numpy(rng.permutation(r + 1))
    return P


class Edit:
    """Context manager. ``cond``: ``attention`` (the graph filter on the
    layer's attention graph), ``random`` (the same filter on a row-shuffled
    attention graph), ``path`` (on the chain graph), ``scale`` (the residual
    shrunk, ``-x``). Controls are rescaled per item to the Frobenius norm of
    the attention edit. ``self.P`` holds the shuffle index of the batch."""

    def __init__(self, model, layer, filt, strength, cond):
        self.arch, self.layer = Arch(model), layer
        self.filt, self.s, self.cond = FILTERS[filt], float(strength), cond
        self.P, self.got, self.rec = None, {}, None
        self.gc = st.GraphConstructor(st.GSPConfig())

    def _graph(self):
        if self.cond == "path":
            return "path"
        P, gc = self.P, self.gc

        def g(att, x):
            m = att.shape[-1]
            idx = P[:, :m, :m].to(att.device).unsqueeze(1).expand(-1, att.shape[1], -1, -1)
            return gc.aggregate_heads(gc.symmetrize_attention(att.double().gather(-1, idx)))
        return g

    def _edit(self, module, args, out):
        X = first_hidden(out)
        A = self.got.pop("A")
        xs, As = X[:, 1:].double(), A[:, :, 1:, 1:]
        d_att = st.graph_filter(xs, As, causal=True, **self.filt) - xs
        na = d_att.flatten(1).norm(dim=1)
        xn = xs.flatten(1).norm(dim=1)
        if self.cond == "attention":
            d, k = d_att, torch.ones_like(na)
        elif self.cond == "scale":
            d, k = -xs, na / xn
        else:
            d = st.graph_filter(xs, As, causal=True, graph=self._graph(), **self.filt) - xs
            k = na / d.flatten(1).norm(dim=1).clamp_min(1e-12)
        delta = self.s * k.view(-1, 1, 1) * d
        y = xs + delta
        if self.rec is not None:
            self.rec.append(dict(
                pos=(delta.norm(dim=-1) / xs.norm(dim=-1)).cpu().numpy(),
                size=(delta.flatten(1).norm(dim=1) / xn).cpu().numpy(),
                normchg=(y.flatten(1).norm(dim=1) / xn - 1).cpu().numpy()))
        Y = X.clone()
        Y[:, 1:] = y.to(X.dtype)
        return replace_first(out, Y)

    def __enter__(self):
        def grab(module, args, out):
            self.got["A"] = out[1].detach()
        self.handles = [self.arch.attention(self.layer).register_forward_hook(grab),
                        self.arch.block(self.layer).register_forward_hook(self._edit)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
