# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0
"""
Attention-sink measures and attention-family covariates.

A spectral readout of attention can track how much attention flows into the
sink (usually the first token) rather than how information is routed
(SinkProbe, Binkowski et al. 2026; Viganò et al. 2026), and can track length
or the rank of the states (Dev and Le 2026). These measures let an analysis
report the sink and the other attention-family covariates next to every
spectral feature.

- :func:`sink_mass`: attention the later queries give to the sink, per head;
- :func:`value_norms` and :func:`value_weighted_sink`: the norm of what the
  sink writes, ``‖W_O^h v^h_sink‖`` (Kobayashi et al. 2020), times its mass;
- :func:`effective_rank`: ``exp`` of the entropy of the normalised singular
  values (Roy and Vetterli 2007), of hidden states or attention maps;
- :func:`attention_covariates`: these and the other per-head attention
  covariates in one call.

Every function accepts torch tensors or NumPy arrays with any leading
dimensions and computes in float64.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch

__all__ = ["sink_mass", "value_norms", "value_weighted_sink", "effective_rank",
           "attention_covariates", "ATTENTION_COVARIATES"]

#: Names returned by :func:`attention_covariates` (``value_weighted_sink``
#: only when values are given).
ATTENTION_COVARIATES = ("sink_mass", "value_weighted_sink", "diagonal_mass",
                        "previous_token_mass", "row_entropy", "attention_effective_rank",
                        "asymmetry", "n_tokens")


def _t(x) -> torch.Tensor:
    t = x.detach() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    return t.to(device="cpu", dtype=torch.float64)


def _out(t: torch.Tensor):
    """NumPy array of the leading shape, or a float for a single item."""
    a = t.numpy()
    return float(a) if a.ndim == 0 else a


def _check_square(A: torch.Tensor) -> None:
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"expected attention [..., S, S], got shape {tuple(A.shape)}")


def sink_mass(attention, sink: int = 0):
    """Mean attention that the queries after the sink give to it.

    ``mean_{i > sink} A[i, sink]``, per head and layer.

    Args:
        attention: ``[..., S, S]`` attention weights, row i the query (for
            example ``[layers, heads, S, S]``).
        sink: index of the sink token (default the first).

    Returns:
        ``[...]`` (a float for one ``[S, S]`` map). Zero when no query
        follows the sink.
    """
    A = _t(attention)
    _check_square(A)
    if A.shape[-1] <= sink + 1:
        return _out(torch.zeros(A.shape[:-2], dtype=torch.float64))
    return _out(A[..., sink + 1:, sink].mean(dim=-1))


def _per_head_values(values: torch.Tensor, out_weight: torch.Tensor):
    """``(V [..., H, S, dh], W [H, d_model, dh])`` with grouped-query value
    heads repeated to the query heads (query head h reads value head
    ``h // (H / H_kv)``, the Hugging Face convention)."""
    if values.dim() < 3:
        raise ValueError(f"expected values [..., heads, S, head_dim], got {tuple(values.shape)}")
    d_model, inner = out_weight.shape
    dh = values.shape[-1]
    if inner % dh:
        raise ValueError(f"out_weight has {inner} input features, not a multiple of the "
                         f"head size {dh}; pass the [d_model, heads * head_dim] weight")
    H, Hkv = inner // dh, values.shape[-3]
    if H % Hkv:
        raise ValueError(f"{Hkv} value heads cannot serve {H} query heads")
    if Hkv != H:
        values = values.repeat_interleave(H // Hkv, dim=-3)
    W = out_weight.reshape(d_model, H, dh).permute(1, 0, 2)
    return values, W


def value_norms(values, out_weight):
    """Norm of what each token writes through each head: ``‖W_O^h v^h_j‖``.

    Args:
        values: ``[..., heads, S, head_dim]`` value vectors per head (after
            the value projection and its bias). Grouped-query layouts with
            fewer value heads are repeated to the query heads.
        out_weight: output projection as an ``nn.Linear`` weight,
            ``[d_model, heads * head_dim]`` (for GPT-2's ``Conv1D``, pass
            ``c_proj.weight.T``). Head h owns columns
            ``h * head_dim:(h + 1) * head_dim``.

    Returns:
        ``[..., heads, S]``. ``A[h, i, j]`` times entry ``[h, j]`` is the norm
        of token j's contribution to head h's output at token i (Kobayashi
        et al. 2020); output gates, where a model has them, are not applied.
    """
    V, W = _per_head_values(_t(values), _t(out_weight))
    gram = torch.einsum("hmd,hme->hde", W, W)                  # W_h^T W_h
    sq = torch.einsum("...hsd,hde,...hse->...hs", V, gram, V)
    return _out(sq.clamp_min(0).sqrt())


def value_weighted_sink(attention, values, out_weight, sink: int = 0):
    """Sink score weighted by what the sink writes:
    ``‖W_O^h v^h_sink‖ · sink_mass^h``.

    It is the mean, over the queries after the sink, of the norm of the
    sink's contribution to head h's output. A head can pour most of its
    attention into a sink whose value vector is near zero (a no-op); this
    score is small for such a head and large when the sink writes.

    Args:
        attention: ``[..., heads, S, S]``.
        values, out_weight: as in :func:`value_norms`, with the same leading
            dimensions as ``attention``.
        sink: index of the sink token.

    Returns:
        ``[..., heads]``.
    """
    norms = _t(value_norms(values, out_weight))[..., sink]
    return _out(_t(sink_mass(attention, sink)) * norms)


def effective_rank(x):
    """Effective rank (Roy and Vetterli 2007): ``exp(-Σ p_k log p_k)`` with
    ``p = s / Σ s`` from the singular values ``s``.

    Args:
        x: ``[..., n, d]``: hidden states ``[S, d]``, attention maps
            ``[heads, S, S]``, or any batch of matrices.

    Returns:
        ``[...]`` (a float for one matrix), between 1 and ``min(n, d)``; 0
        for a zero matrix.
    """
    s = torch.linalg.svdvals(_t(x))
    total = s.sum(dim=-1, keepdim=True)
    p = torch.where(total > 0, s / torch.where(total > 0, total, torch.ones_like(total)), s)
    h = -(torch.where(p > 0, p * torch.log(torch.where(p > 0, p, torch.ones_like(p))), p)).sum(-1)
    return _out(torch.where(total[..., 0] > 0, torch.exp(h), torch.zeros_like(h)))


def attention_covariates(attention, values=None, out_weight=None, sink: int = 0
                         ) -> Dict[str, object]:
    """Per-head attention-family covariates, to report next to (or regress
    out of) any spectral feature of the same heads.

    Args:
        attention: ``[..., heads, S, S]`` (for example ``[layers, heads, S, S]``).
        values, out_weight: optional, as in :func:`value_weighted_sink`; with
            them ``value_weighted_sink`` is included.
        sink: index of the sink token.

    Returns:
        name -> ``[..., heads]`` (``n_tokens`` is an int). ``sink_mass`` is
        :func:`sink_mass`, ``value_weighted_sink`` :func:`value_weighted_sink`,
        ``diagonal_mass`` the mean of ``A[i, i]`` over queries,
        ``previous_token_mass`` the mean of ``A[i, i - 1]`` over queries
        ``i >= 1``, ``row_entropy`` the mean over queries of
        ``-Σ_j A[i, j] log A[i, j]`` (nats), ``attention_effective_rank``
        the :func:`effective_rank` of each map, ``asymmetry`` the coefficient
        ``G`` of Dahlem et al. (:func:`spectral_trust.features.asymmetry`)
        and ``n_tokens`` is ``S``.
    """
    A = _t(attention)
    _check_square(A)
    S = A.shape[-1]
    out: Dict[str, object] = {"sink_mass": _t(sink_mass(A, sink))}
    if values is not None:
        if out_weight is None:
            raise ValueError("value_weighted_sink needs out_weight with values")
        out["value_weighted_sink"] = _t(value_weighted_sink(A, values, out_weight, sink))
    out["diagonal_mass"] = A.diagonal(dim1=-2, dim2=-1).mean(-1)
    out["previous_token_mass"] = (A.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1) if S > 1
                                  else torch.zeros(A.shape[:-2], dtype=torch.float64))
    logs = torch.log(torch.where(A > 0, A, torch.ones_like(A)))
    out["row_entropy"] = -(A * logs).sum(-1).mean(-1)
    out["attention_effective_rank"] = _t(effective_rank(A))
    from .asymmetry import asymmetry
    out["asymmetry"] = _t(asymmetry(A))
    result = {k: _out(v) for k, v in out.items()}
    result["n_tokens"] = int(S)
    return result
