# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Serialisation of numerical sequences for in-context learning, and the token
spans of each value.

The default format is the one of Gruver et al. (2023), Liu et al. (2024) and
Bao et al. (2026): values rescaled to ``[50, 950]``, rounded, written as
zero-padded three-digit integers and followed by a comma (the last one
included), e.g. ``"087,173,642,"``.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


def quantize(values, low: int = 50, high: int = 950,
             bounds: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Rescale ``values`` linearly to ``[low, high]`` and round to integers.

    The bounds are the minimum and maximum of ``values`` unless ``bounds``
    gives them (pass the bounds of the full trajectory when serialising a
    prefix, so that the mapping does not depend on the context length). A
    constant input maps to the midpoint ``round((low + high) / 2)``.
    """
    u = np.asarray(values, dtype=np.float64)
    lo, hi = (float(u.min()), float(u.max())) if bounds is None else map(float, bounds)
    if hi == lo:
        return np.full(u.shape, int(round((low + high) / 2)), dtype=np.int64)
    q = np.rint(low + (high - low) * (u - lo) / (hi - lo))
    return np.clip(q, low, high).astype(np.int64)


def serialize(values: Sequence[int], digits: int = 3, sep: str = ",",
              trailing: bool = True) -> str:
    """Write non-negative integers as zero-padded ``digits``-digit strings
    joined by ``sep`` (and followed by it when ``trailing``)."""
    ints = [int(v) for v in values]
    top = 10 ** int(digits)
    bad = [v for v in ints if not 0 <= v < top]
    if bad:
        raise ValueError(f"values must lie in [0, {top}) for {digits} digits, got {bad[:3]}")
    body = sep.join(f"{v:0{digits}d}" for v in ints)
    return body + sep if trailing and ints else body


@dataclass
class ValueSpans:
    """Token positions of a serialised numerical sequence.

    Attributes:
        input_ids: the token ids (special tokens included).
        values: ``[start, end)`` token span of each value, in order.
        separators: positions of the separator tokens.
        special: positions of special tokens (BOS and the like).
        mixed: positions of tokens that hold digits and the separator
            together (none for Llama 3 or Qwen 3 tokenizers); they are
            counted in the value they start.
    """
    input_ids: List[int]
    values: List[Tuple[int, int]]
    separators: List[int]
    special: List[int]
    mixed: List[int] = field(default_factory=list)

    @property
    def numeric_positions(self) -> List[int]:
        """Every token position inside a value span."""
        return [p for s, e in self.values for p in range(s, e)]

    @property
    def tokens_per_value(self) -> np.ndarray:
        return np.array([e - s for s, e in self.values], dtype=np.int64)

    @property
    def content_positions(self) -> List[int]:
        """Every position that is not a special token."""
        sp = set(self.special)
        return [p for p in range(len(self.input_ids)) if p not in sp]


def value_spans(tokenizer, text: str, sep: str = ",",
                add_special_tokens: bool = True) -> ValueSpans:
    """Tokenise ``text`` and locate each value's tokens and each separator.

    Needs a fast tokenizer (offset mapping). Values are maximal runs of
    digits (with an optional leading minus sign or decimal point inside the
    run); every other non-space character is expected to be ``sep``.
    """
    enc = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise ValueError("value_spans needs a fast tokenizer (return_offsets_mapping)")
    # character spans of the values
    char_value = np.full(len(text), -1, dtype=np.int64)
    k, i = 0, 0
    while i < len(text):
        if text[i].isdigit() or (text[i] in "-." and i + 1 < len(text) and text[i + 1].isdigit()):
            j = i
            while j < len(text) and (text[j].isdigit() or text[j] == "."):
                j += 1
            if text[i] == "-":
                j = max(j, i + 1)
            char_value[i:j] = k
            k += 1
            i = j
        else:
            i += 1
    n_values = k
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    first: List[Optional[int]] = [None] * n_values
    last: List[Optional[int]] = [None] * n_values
    seps, special, mixed = [], [], []
    for pos, (a, b) in enumerate(offsets):
        a, b = int(a), int(b)
        if b <= a:
            if ids[pos] in special_ids or b == a:
                special.append(pos)
            continue
        vals = sorted({int(v) for v in char_value[a:b] if v >= 0})
        has_sep = sep in text[a:b]
        if vals:
            v = vals[0]
            first[v] = pos if first[v] is None else first[v]
            for v2 in vals:
                first[v2] = pos if first[v2] is None else first[v2]
                last[v2] = pos
            if has_sep:
                mixed.append(pos)
        elif has_sep:
            seps.append(pos)
        elif ids[pos] in special_ids:
            special.append(pos)
    spans = []
    for v in range(n_values):
        if first[v] is None:
            raise ValueError(f"value {v} has no token")
        spans.append((first[v], last[v] + 1))
    return ValueSpans(ids, spans, seps, special, mixed)


def repeat_fraction(values: Sequence[int], order: int = 2) -> float:
    """Share of positions t >= order whose preceding ``order`` values, as a
    tuple, already occurred earlier followed by some value: the part of the
    sequence an induction (copy) mechanism can continue. 0 for sequences
    without repeated contexts, close to 1 for short periods."""
    v = [int(x) for x in values]
    seen = set()
    hits, total = 0, 0
    for t in range(order, len(v)):
        ctx = tuple(v[t - order:t])
        total += 1
        if ctx in seen:
            hits += 1
        seen.add(ctx)
    return hits / total if total else 0.0
