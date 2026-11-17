# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Controls for the token inventory of a serialised sequence.

A diagnostic that orders sequences by dynamical complexity may also follow
how many distinct values (hence tokens) a sequence holds. Two tools separate
the two:

- :func:`token_preserving_shuffle`: the same values in a random order, so
  the exact multiset of tokens is kept and the temporal structure is not;
- :func:`inventory_covariates`: the number of distinct values, the share of
  values that repeat an earlier one and the entropy of the value
  distribution (optionally the number of distinct token ids), one value per
  sequence, to pass to :func:`~spectral_trust.confound_audit` as
  covariates. All three are unchanged by the shuffle.
"""
from typing import Dict

import numpy as np

from .serialize import serialize

#: Keys of :func:`inventory_covariates` (``n_unique_tokens`` with a tokenizer).
INVENTORY_COVARIATES = ("n_unique", "repeat_fraction", "value_entropy")


def token_preserving_shuffle(values, seed: int = 0) -> np.ndarray:
    """``values`` in a random order (``seed``): every value, hence every
    token of its serialisation, is kept with its multiplicity; only the
    order changes. Pass the quantised integers (shuffling raw values and
    quantising afterwards gives the same integers, since the bounds are
    unchanged)."""
    v = np.asarray(values)
    if v.ndim != 1:
        raise ValueError("token_preserving_shuffle takes one sequence (1-D)")
    return v[np.random.default_rng(seed).permutation(len(v))]


def _one(values, tokenizer, digits: int) -> Dict[str, float]:
    v = np.asarray(values)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("each sequence must be 1-D and non-empty")
    _, counts = np.unique(v, return_counts=True)
    p = counts / counts.sum()
    out = {"n_unique": float(len(counts)),
           "repeat_fraction": float(1.0 - len(counts) / len(v)),
           "value_entropy": float(-(p * np.log(p)).sum())}
    if tokenizer is not None:
        ids = tokenizer.encode(serialize(v, digits=digits), add_special_tokens=False)
        out["n_unique_tokens"] = float(len(set(ids)))
    return out


def inventory_covariates(values, tokenizer=None, digits: int = 3
                         ) -> Dict[str, object]:
    """Inventory of one sequence, or of each of several.

    Args:
        values: one sequence (1-D) or a list of sequences (lengths may differ).
        tokenizer: if given, also ``n_unique_tokens``, the distinct token ids
            of the serialised sequence (separators included, special tokens
            not): equal to ``n_unique + 1`` with one token per value, at most
            11 with one token per digit.
        digits: digits per value of the serialisation.

    Returns:
        ``{name: float}`` for one sequence; ``{name: np.ndarray}`` (one entry
        per sequence, ready for ``st.confound_audit(..., covariates=...)``)
        for several. ``repeat_fraction`` here is ``1 - n_unique / n``, the
        share of values seen before, which ignores order;
        :func:`~spectral_trust.numeric.repeat_fraction` counts repeated
        contexts instead and changes under a shuffle. ``value_entropy`` is
        in nats.
    """
    if len(values) and np.ndim(values[0]) == 1:
        rows = [_one(v, tokenizer, digits) for v in values]
        return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    return _one(values, tokenizer, digits)
