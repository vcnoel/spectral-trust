# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Multi-step autoregressive extrapolation of a serialised sequence.

:func:`rollout` predicts the next value with
:func:`~spectral_trust.numeric.next_value_distribution` (its mode by
default, the greedy value), writes it back into the context in the same
format and repeats, so each step conditions on the model's own predictions.
Against the true continuation it reports the absolute error per step and its
running sum. A one-step error can be small while the trajectory drifts away
over many steps; this is what the per-step errors show.

With ``use_cache=True`` (the default) the prompt runs once and every step
extends its key-value cache with the predicted value's tokens. The prompt's
pass then has the prompt's length, so an edit installed by
:func:`~spectral_trust.interventions.edit_with_controls` (which edits the
pass of the prompt's length) changes the prompt's keys and values, and every
step of the rollout reads the edited prompt; the predicted tokens themselves
are not edited. :class:`RolloutReadout` packages a rollout as the readout of
``edit_with_controls``, so an edit's effect on multi-step error is measured
against its controls. With ``use_cache=False`` every step reruns the whole
growing sequence through ``next_value_distribution`` (the same numbers up to
float error, slower); an edit of the prompt's length then changes the first
step only.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .readout import (ValueDistribution, _can_cache, _hierarchical_from, digit_token_ids,
                      next_value_distribution, value_token_ids)
from .serialize import serialize, value_spans


@dataclass
class Rollout:
    """A multi-step extrapolation.

    Attributes:
        predicted: ``[n_steps]`` predicted values.
        target: ``[n_steps]`` true continuation, or None.
        entropy: ``[n_steps]`` entropy (nats) of each step's distribution.
        std: ``[n_steps]`` standard deviation of each step's distribution.
        method: ``"single"`` or ``"hierarchical"``.
        tol: the error that ends ``valid_steps``.
    """
    predicted: np.ndarray
    target: Optional[np.ndarray]
    entropy: np.ndarray
    std: np.ndarray
    method: str
    tol: float = 50.0

    @property
    def abs_error(self) -> Optional[np.ndarray]:
        """``|predicted - target|`` per step."""
        if self.target is None:
            return None
        return np.abs(self.predicted.astype(np.float64) - self.target.astype(np.float64))

    @property
    def cum_error(self) -> Optional[np.ndarray]:
        """Running sum of the absolute error."""
        e = self.abs_error
        return None if e is None else np.cumsum(e)

    @property
    def valid_steps(self) -> Optional[int]:
        """Steps before the first absolute error above ``tol``."""
        e = self.abs_error
        if e is None:
            return None
        bad = np.flatnonzero(e > self.tol)
        return int(bad[0]) if len(bad) else len(e)

    def summary(self) -> Dict[str, float]:
        """Scalars for a readout: ``mean_entropy`` and, with a target,
        ``mae``, ``first_error``, ``final_error``, ``cum_error`` (the total)
        and ``valid_steps``."""
        out = {"mean_entropy": float(self.entropy.mean())}
        e = self.abs_error
        if e is not None:
            out.update({"mae": float(e.mean()), "first_error": float(e[0]),
                        "final_error": float(e[-1]), "cum_error": float(e.sum()),
                        "valid_steps": float(self.valid_steps)})
        return out


def _pick(dist: ValueDistribution, predict: str) -> int:
    if predict == "mode":
        return dist.mode
    v = int(round(dist.mean))
    return min(max(v, 0), len(dist.probs) - 1)


def _restore(cache, S: int, device) -> None:
    """Back to the first row and the first S positions."""
    extra = cache.get_seq_length() - S
    if extra > 0:
        cache.crop(-extra)
    cache.batch_select_indices(torch.zeros(1, dtype=torch.long, device=device))


@torch.no_grad()
def rollout(model, tokenizer, values: Optional[Sequence[int]], n_steps: int, *,
            target: Optional[Sequence[int]] = None, input_ids=None, digits: int = 3,
            sep: str = ",", method: str = "auto", predict: str = "mode",
            min_prob: float = 0.0, chunk: int = 16, use_cache: bool = True,
            tol: float = 50.0) -> Rollout:
    """Extrapolate ``n_steps`` values, feeding back each prediction.

    Args:
        model: a Hugging Face causal LM or a :class:`~spectral_trust.SpectralModel`.
        tokenizer: default the SpectralModel's.
        values: the context, integers in ``[0, 10**digits)`` (for example
            ``traj.ints[:n]``); serialised with :func:`serialize`.
        n_steps: values to predict.
        target: the true continuation (at least ``n_steps`` values); gives the
            errors.
        input_ids: the prompt as token ids (``[S]`` or ``[1, S]``, ending
            after a separator) instead of ``values``.
        digits, sep: the format.
        method, min_prob, chunk: as in :func:`next_value_distribution`.
        predict: ``"mode"`` (the most probable value: greedy decoding over
            values) or ``"mean"`` (the rounded expected value).
        use_cache: extend the prompt's key-value cache step by step (see the
            module docstring); False reruns the whole sequence each step.
        tol: absolute error that ends :attr:`Rollout.valid_steps`.
    """
    if predict not in ("mode", "mean"):
        raise ValueError(f"predict must be 'mode' or 'mean', got {predict!r}")
    tok = tokenizer or getattr(model, "tokenizer", None)
    if tok is None:
        raise ValueError("pass the tokenizer")
    hf = getattr(model, "model", None) if hasattr(model, "run_with_spectra") else model
    device = next(hf.parameters()).device
    if input_ids is None:
        if values is None:
            raise ValueError("pass values or input_ids")
        input_ids = tok(serialize(values, digits, sep))["input_ids"]
    ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(1, -1).to(device)
    n_steps = int(n_steps)
    tgt = None
    if target is not None:
        tgt = np.asarray(target, dtype=np.int64)[:n_steps]
        if len(tgt) < n_steps:
            raise ValueError(f"target has {len(tgt)} values for {n_steps} steps")
    if method == "auto":
        method = "single" if value_token_ids(tok, digits) is not None else "hierarchical"
    if method not in ("single", "hierarchical"):
        raise ValueError(f"method must be 'auto', 'single' or 'hierarchical', got {method!r}")
    vids = value_token_ids(tok, digits) if method == "single" else None
    dig = digit_token_ids(tok) if method == "hierarchical" else None
    cache = logits = None
    if use_cache:
        out = hf(input_ids=ids, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
        if cache is None or (method == "hierarchical" and not _can_cache(cache)):
            use_cache = False                   # the cache cannot be reused: rerun
    preds, ents, stds = [], [], []
    for _ in range(n_steps):
        if not use_cache:
            dist = next_value_distribution(hf, ids, tok, digits=digits, method=method,
                                           min_prob=min_prob, chunk=chunk)
        elif method == "single":
            lp = torch.log_softmax(logits.to(torch.float64)[torch.as_tensor(
                vids, device=logits.device)], dim=-1)
            dist = ValueDistribution(lp.exp().cpu().numpy(), "single")
        else:
            probs = _hierarchical_from(hf, ids, logits, cache, dig, digits, float(min_prob),
                                       max(1, int(chunk)))
            dist = ValueDistribution(probs, "hierarchical")
            _restore(cache, ids.shape[1], device)
        v = _pick(dist, predict)
        preds.append(v)
        ents.append(dist.entropy)
        stds.append(dist.std)
        new = torch.as_tensor(tok.encode(serialize([v], digits, sep), add_special_tokens=False),
                              dtype=torch.long, device=device).reshape(1, -1)
        if use_cache and len(preds) < n_steps:
            out = hf(input_ids=new, past_key_values=cache, use_cache=True)
            cache, logits = out.past_key_values, out.logits[0, -1]
        ids = torch.cat([ids, new], dim=1)
    return Rollout(np.array(preds, dtype=np.int64), tgt, np.array(ents), np.array(stds),
                   method, float(tol))


class RolloutReadout:
    """A rollout as the readout of
    :func:`~spectral_trust.interventions.edit_with_controls`.

    Holds the prompts (``inputs``) and their true continuations; called as
    ``readout(model, ids)``, it finds the continuation of ``ids`` and returns
    :meth:`Rollout.summary` (``mae``, ``cum_error``, ``final_error``, ...)::

        ro = RolloutReadout(tok, contexts, continuations, n_steps=20)
        res = st.edit_with_controls(model, ro.inputs, ro, layer=8,
                                    positions=ro.positions, token_span=(1, None))
        res.summary(["mae", "cum_error"])

    Args:
        tokenizer: the model's tokenizer.
        contexts: the integer contexts, one per item.
        targets: the true continuations (at least ``n_steps`` values each).
        n_steps: values to predict per item.
        **kwargs: passed to :func:`rollout` (``predict``, ``method``,
            ``min_prob``, ``tol``, ...).

    ``rollouts`` keeps every :class:`Rollout` computed, in call order.
    """

    def __init__(self, tokenizer, contexts: Sequence[Sequence[int]],
                 targets: Sequence[Sequence[int]], n_steps: int, *, digits: int = 3,
                 sep: str = ",", **kwargs):
        if len(contexts) != len(targets):
            raise ValueError(f"{len(contexts)} contexts for {len(targets)} targets")
        self.tokenizer, self.n_steps, self.digits, self.sep = tokenizer, int(n_steps), digits, sep
        self.kwargs = kwargs
        self.texts = [serialize(c, digits, sep) for c in contexts]
        self.inputs = [torch.tensor(tokenizer(t)["input_ids"], dtype=torch.long)
                       for t in self.texts]
        self._targets: Dict[tuple, np.ndarray] = {}
        for ids, t in zip(self.inputs, targets):
            key, t = tuple(ids.tolist()), np.asarray(t, dtype=np.int64)
            if key in self._targets and not np.array_equal(self._targets[key], t):
                raise ValueError("two identical contexts have different continuations")
            self._targets[key] = t
        self.rollouts = []

    def positions(self, i: int):
        """Token positions of the values of item ``i`` (for ``positions=``)."""
        return value_spans(self.tokenizer, self.texts[i], sep=self.sep).numeric_positions

    def __call__(self, model, ids) -> Dict[str, float]:
        key = tuple(torch.as_tensor(ids).reshape(-1).tolist())
        if key not in self._targets:
            raise KeyError("these ids are not one of the readout's inputs")
        r = rollout(model, self.tokenizer, None, self.n_steps, input_ids=ids,
                    target=self._targets[key], digits=self.digits, sep=self.sep,
                    **self.kwargs)
        self.rollouts.append(r)
        return r.summary()
