# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
The model's predictive distribution over the next value of a serialised
sequence.

Two cases, chosen by :func:`next_value_distribution` (``method="auto"``):

- every value ``000``-``999`` is one token (Llama 3 tokenizers): one softmax
  restricted to those tokens at the last position, as in Bao et al. (2026);
- every digit is one token (Qwen, Gemma, SmolLM): the digit-wise
  probabilities are composed hierarchically, as in Liu et al. (2024),
  ``P(v) = P(d1) P(d2 | d1) P(d3 | d1 d2)``, each digit's softmax restricted
  to the ten digit tokens. Prefix ``d1 ... dk`` is scored by feeding it on
  top of the prompt's key-value cache, so the prompt runs once.

Hooks active on the model during the call act on every forward pass; the
prompt pass has the prompt's length and the digit passes have one to
``digits - 1`` tokens, so an edit that applies only at the prompt's length
(as :func:`~spectral_trust.interventions.edit_with_controls` installs)
edits the prompt and leaves the digit passes alone.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class ValueDistribution:
    """A distribution over the integers ``0 .. 10**digits - 1``.

    Attributes:
        probs: ``[10**digits]`` probabilities (sum 1).
        method: ``"single"`` or ``"hierarchical"``.
    """
    probs: np.ndarray
    method: str

    @property
    def values(self) -> np.ndarray:
        return np.arange(len(self.probs), dtype=np.float64)

    @property
    def mean(self) -> float:
        return float(self.values @ self.probs)

    @property
    def var(self) -> float:
        m = self.mean
        return float(((self.values - m) ** 2) @ self.probs)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 0.0)))

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats."""
        p = self.probs[self.probs > 0]
        return float(-(p * np.log(p)).sum())

    @property
    def mode(self) -> int:
        return int(np.argmax(self.probs))

    def logprob(self, value: int) -> float:
        return float(np.log(max(self.probs[int(value)], 1e-300)))

    def summary(self) -> Dict[str, float]:
        """``mean``, ``std``, ``var``, ``entropy`` and ``mode``."""
        return {"mean": self.mean, "std": self.std, "var": self.var,
                "entropy": self.entropy, "mode": float(self.mode)}


def value_token_ids(tokenizer, digits: int = 3) -> Optional[List[int]]:
    """Ids of the single tokens that spell each zero-padded value, or None
    when some value takes more than one token."""
    out = []
    for v in range(10 ** digits):
        ids = tokenizer.encode(f"{v:0{digits}d}", add_special_tokens=False)
        if len(ids) != 1:
            return None
        out.append(ids[0])
    return out


def digit_token_ids(tokenizer) -> List[int]:
    """Ids of the ten digit tokens; raises if a digit is not one token."""
    out = []
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"digit {d} is not a single token ({ids})")
        out.append(ids[0])
    return out


def _ids(input_ids, device) -> torch.Tensor:
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise ValueError("next_value_distribution reads one sequence at a time")
    return ids.to(device)


@torch.no_grad()
def next_value_distribution(model, input_ids, tokenizer=None, *, digits: int = 3,
                            method: str = "auto", min_prob: float = 0.0,
                            chunk: int = 16) -> ValueDistribution:
    """Predictive distribution over the next ``digits``-digit value.

    Args:
        model: a Hugging Face causal LM or a :class:`~spectral_trust.SpectralModel`.
        input_ids: the prompt, ``[S]`` or ``[1, S]``, ending where the next
            value starts (after the separator).
        tokenizer: default the SpectralModel's.
        digits: digits per value.
        method: ``"single"`` (one token per value), ``"hierarchical"`` (one
            token per digit) or ``"auto"``.
        min_prob: hierarchical only: prefixes with probability below this are
            not expanded; their mass is spread uniformly over the values they
            start (0, the default, expands every prefix: exact).
        chunk: hierarchical only: prefixes scored per forward pass (bounds
            the memory of the repeated cache).

    The hierarchical path needs a key-value cache that can be cropped and
    repeated (``DynamicCache``: Llama, Qwen 2/3, Gemma 2/3, Phi); models
    whose cache cannot are scored by re-running prompt and prefix.
    """
    tok = tokenizer or getattr(model, "tokenizer", None)
    if tok is None:
        raise ValueError("pass the tokenizer")
    hf = getattr(model, "model", None) if hasattr(model, "run_with_spectra") else model
    device = next(hf.parameters()).device
    ids = _ids(input_ids, device)
    if method == "auto":
        method = "single" if value_token_ids(tok, digits) is not None else "hierarchical"
    if method == "single":
        vids = value_token_ids(tok, digits)
        if vids is None:
            raise ValueError("some value is not a single token; use method='hierarchical'")
        logits = hf(input_ids=ids, use_cache=False).logits[0, -1].to(torch.float64)
        lp = torch.log_softmax(logits[torch.as_tensor(vids, device=logits.device)], dim=-1)
        return ValueDistribution(lp.exp().cpu().numpy(), "single")
    if method != "hierarchical":
        raise ValueError(f"method must be 'auto', 'single' or 'hierarchical', got {method!r}")
    return ValueDistribution(_hierarchical(hf, ids, digit_token_ids(tok), digits,
                                           float(min_prob), max(1, int(chunk))), "hierarchical")


def _hierarchical(hf, ids, dig: List[int], digits: int, min_prob: float, chunk: int):
    S = ids.shape[1]
    dig_t = torch.as_tensor(dig, device=ids.device)
    out = hf(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    can_cache = all(hasattr(cache, m) for m in ("crop", "batch_repeat_interleave",
                                                 "batch_select_indices"))
    probs = np.zeros(10 ** digits)
    # frontier: (prefix digits tuple, log prob of the prefix, log probs of the next digit [10])
    first = torch.log_softmax(out.logits[0, -1].to(torch.float64)[dig_t], -1).cpu().numpy()
    frontier = [((), 0.0, first)]
    for level in range(digits):
        rest = digits - level - 1
        children = []
        for prefix, lp, nxt in frontier:
            for d in range(10):
                lj = lp + float(nxt[d])
                pj = float(np.exp(lj))
                if rest == 0:
                    probs[_value(prefix + (d,))] += pj
                elif pj >= min_prob and pj > 0:
                    children.append((prefix + (d,), lj))
                elif pj > 0:                      # not expanded: spread uniformly
                    base = _value(prefix + (d,)) * 10 ** rest
                    probs[base:base + 10 ** rest] += pj / 10 ** rest
        if rest == 0 or not children:
            break
        frontier = []
        for start in range(0, len(children), chunk):
            part = children[start:start + chunk]
            new = torch.as_tensor([[dig[d] for d in p] for p, _ in part], device=ids.device)
            if can_cache:
                cache.crop(S)
                cache.batch_select_indices(torch.zeros(1, dtype=torch.long, device=ids.device))
                cache.batch_repeat_interleave(len(part))
                logits = hf(input_ids=new, past_key_values=cache, use_cache=True).logits
            else:
                full = torch.cat([ids.expand(len(part), S), new], dim=1)
                logits = hf(input_ids=full, use_cache=False).logits
            nxt = torch.log_softmax(logits[:, -1].to(torch.float64)[:, dig_t], -1).cpu().numpy()
            frontier.extend((p, lp, nxt[j]) for j, (p, lp) in enumerate(part))
    total = probs.sum()
    return probs / total if total > 0 else probs


def _value(prefix) -> int:
    v = 0
    for d in prefix:
        v = 10 * v + int(d)
    return v
