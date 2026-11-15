# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
LM-Polygraph bridge: any attention metric of spectral-trust as an
LM-Polygraph uncertainty estimator, so that it can be scored on the
LM-Polygraph benchmark next to its baselines.

LM-Polygraph (Fadeeva et al. 2023, https://github.com/IINemo/lm-polygraph,
MIT licence) computes statistics once per batch with its stat calculators
and passes them to each estimator. :class:`SpectralTrustEstimator` asks for
``forwardpass_attention_weights`` (a forward pass of the prompt followed by
the greedy generation, ``[layers, heads, S, S]`` per item, NaN-padded in a
batch) and ``greedy_tokens``, measures each item with
:func:`spectral_trust.compute_spectra`, and reduces the result to one number
per sequence: heads first (per-head metrics), then layers, times ``sign``.
LM-Polygraph reads a higher value as more uncertain, so a metric that falls
with uncertainty needs ``sign=-1``.

Needs ``lm-polygraph`` (``pip install spectral_trust[polygraph]``, Python >=
3.10); the core package never imports it. Written against lm-polygraph
0.7.0 (``Estimator(stats_dependencies, level)``, ``__call__(stats) ->
np.ndarray``).

>>> from lm_polygraph import WhiteboxModel
>>> from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
>>> from spectral_trust.integrations.lm_polygraph import SpectralTrustEstimator
>>> model = WhiteboxModel.from_pretrained("gpt2", device="cpu")
>>> est = SpectralTrustEstimator("operator.participation_ratio", sign=-1)
>>> estimate_uncertainty(model, est, input_text="Who wrote Hamlet?")
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

try:
    from lm_polygraph.estimators.estimator import Estimator
except ImportError as exc:                        # pragma: no cover - depends on the extra
    raise ImportError(
        "spectral_trust.integrations.lm_polygraph needs lm-polygraph (Python >= 3.10): "
        "pip install 'spectral_trust[polygraph]'") from exc

from ..cache import metric_family, resolve_metrics
from ..config import GSPConfig
from ..model import compute_spectra

#: Reductions over heads and over layers (NaN entries are skipped).
AGGREGATIONS = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax,
                "min": np.nanmin, "sum": np.nansum}

#: Token spans an estimator can measure.
SPANS = ("all", "prompt", "generated")

#: Statistics the estimator asks LM-Polygraph for.
STATS = ["forwardpass_attention_weights", "greedy_tokens"]


def _unpad(attention) -> np.ndarray:
    """``[layers, heads, S, S]`` with the NaN padding of a batch removed."""
    a = np.asarray(attention, dtype=np.float32)
    if a.ndim == 5 and a.shape[1] == 1:           # visual models: (layers, 1, heads, S, S)
        a = a[:, 0]
    if a.ndim != 4 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"expected attention of shape [layers, heads, S, S], got {a.shape}")
    real = ~np.isnan(a[0, 0, :, 0])               # padded rows are NaN, real rows read key 0
    n = int(real.sum())
    return a[..., :n, :n]


class SpectralTrustEstimator(Estimator):
    """A spectral-trust metric as a sequence-level LM-Polygraph estimator.

    Args:
        metric: one attention metric, namespaced (``"operator.participation_ratio"``,
            ``"graph.fiedler_value"``, ``"attn.hfer"``) or bare when only one
            namespace has it. Hybrid metrics (which read hidden states) are
            refused: the forward-pass statistic holds attention only.
        layers: layer indices to measure (negative values count from the
            end); default every layer.
        layer_aggregation, head_aggregation: ``"mean"``, ``"median"``,
            ``"max"``, ``"min"`` or ``"sum"``. Heads are reduced first, for
            the per-head metrics (``attn``, ``operator`` per head).
        sign: the score is ``sign`` times the aggregated metric (higher means
            more uncertain in LM-Polygraph).
        span: ``"all"`` (prompt and generation), ``"prompt"`` or
            ``"generated"`` (the greedy tokens): the induced subgraph, as
            ``token_span`` of :func:`~spectral_trust.compute_spectra`.
        config: graph and operator options (:class:`~spectral_trust.GSPConfig`);
            default the library defaults on CPU.
        name: the estimator's name in LM-Polygraph tables; default one built
            from the arguments.
    """

    def __init__(self, metric: str = "operator.participation_ratio", *,
                 layers: Optional[Union[int, Sequence[int]]] = None,
                 layer_aggregation: str = "mean", head_aggregation: str = "mean",
                 sign: float = 1.0, span: str = "all", config: Optional[GSPConfig] = None,
                 name: Optional[str] = None):
        resolved = resolve_metrics([metric], hybrid=True,
                                   directed=getattr(config, "graph_direction", "") == "directed")
        if len(resolved) != 1:
            raise ValueError(f"{metric!r} names {resolved}; write one namespaced metric")
        self.metric = resolved[0]
        if metric_family(self.metric) != "attention":
            raise ValueError(f"{self.metric} reads hidden states; LM-Polygraph's forward-pass "
                             "statistic has attention only")
        for what, agg in (("layer_aggregation", layer_aggregation),
                          ("head_aggregation", head_aggregation)):
            if agg not in AGGREGATIONS:
                raise ValueError(f"{what} must be one of {sorted(AGGREGATIONS)}, got {agg!r}")
        if span not in SPANS:
            raise ValueError(f"span must be one of {SPANS}, got {span!r}")
        self.layers = [layers] if isinstance(layers, (int, np.integer)) else \
            (list(layers) if layers is not None else None)
        self.layer_aggregation, self.head_aggregation = layer_aggregation, head_aggregation
        self.sign, self.span, self.name = float(sign), span, name
        self.config = config or GSPConfig(device="cpu", verbose=False, save_plots=False)
        super().__init__(list(STATS), "sequence")

    def __str__(self) -> str:
        if self.name:
            return self.name
        layers = "all" if self.layers is None else ",".join(str(x) for x in self.layers)
        return (f"SpectralTrust[{self.metric}|layers={layers}|{self.layer_aggregation}"
                f"|heads={self.head_aggregation}|{self.span}|sign={self.sign:+g}]")

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """One score per item of the batch (higher: more uncertain)."""
        attention = stats["forwardpass_attention_weights"]
        greedy = stats["greedy_tokens"]
        return np.asarray([self.score(a, len(greedy[k])) for k, a in enumerate(attention)],
                          dtype=np.float64)

    def score(self, attention, n_generated: int = 0) -> float:
        """The score of one item, from its ``[layers, heads, S, S]`` attention
        (NaN padding allowed) and the number of generated tokens at its end.
        Usable without LM-Polygraph's manager; NaN for an empty span."""
        a = _unpad(attention)
        S = a.shape[-1]
        g = min(max(int(n_generated), 0), S)
        start, end = {"all": (0, S), "prompt": (0, S - g), "generated": (S - g, S)}[self.span]
        if end - start < 1:
            return float("nan")
        n_layers = a.shape[0]
        chosen = range(n_layers) if self.layers is None else \
            sorted({x + n_layers if x < 0 else x for x in self.layers})
        cache = compute_spectra(
            {lay: torch.from_numpy(a[lay]) for lay in chosen}, tokens=[[""] * S],
            n_layers=n_layers, config=self.config, metrics=[self.metric], hybrid=False,
            keep_patterns=False, token_span=None if (start, end) == (0, S) else (start, end))
        reduce_heads = AGGREGATIONS[self.head_aggregation]
        per_layer = [float(reduce_heads(np.asarray(cache[f"blocks.{lay}.{self.metric}"],
                                                   dtype=np.float64)))
                     for lay in chosen]
        return self.sign * float(AGGREGATIONS[self.layer_aggregation](per_layer))


__all__ = ["SpectralTrustEstimator", "AGGREGATIONS", "SPANS", "STATS"]
