# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
:class:`SpectralCache`, the result of :meth:`SpectralModel.run_with_spectra`
and of the adapters in :mod:`spectral_trust.interop`.

Keys follow the TransformerLens naming scheme, ``blocks.{layer}.{namespace}.
{metric}``, with the true layer index of the model (also on hybrid models
whose linear-attention layers return no attention). Three namespaces:

``attn``
    one value per head, computed from that head's attention graph alone
    (family "attention"): ``fiedler_value``, ``connectivity_ratio``,
    ``spectral_entropy_norm``, ``hfer`` (eigenvalue mass in the upper half of
    the spectrum) and ``spectral_radius``. Shape ``[n_heads]``.
``resid``
    one value per head, the residual stream paired with the layer measured on
    that head's graph (family "hybrid"): ``energy``, ``smoothness_index``,
    ``spectral_entropy`` and ``hfer`` (signal energy in the top modes).
    Shape ``[n_heads]``.
``graph``
    one value per layer, from the head-averaged graph that
    ``GSPDiagnosticsFramework.analyze_text`` builds, through
    ``SpectralAnalyzer.analyze_layer``: every scalar of
    :class:`~spectral_trust.spectral.SpectralDiagnostics`, each with its
    family from ``METRIC_FAMILIES``.

Heads that were not computed (``heads=``) hold NaN, so ``[n_heads]`` arrays
always index by head. Arrays that are not scalar metrics are stored next to
them: ``blocks.{l}.attn.eigenvalues`` (``[n_heads, seq]``),
``blocks.{l}.graph.eigenvalues``, ``blocks.{l}.graph.fiedler_vector``,
``blocks.{l}.graph.spectral_masses`` (``[seq]``) and, when kept,
``blocks.{l}.attn.pattern`` (``[n_heads, seq, seq]``, the raw attention).
"""
import difflib
import json
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import GSPConfig
from .per_head import (
    PER_HEAD_HYBRID_METRIC_NAMES, PER_HEAD_METRIC_NAMES,
)
from .spectral import METRIC_FAMILIES

#: Scalar metrics of the head-averaged graph (namespace ``graph``), in order.
GRAPH_METRIC_NAMES: List[str] = [
    "fiedler_value", "spectral_radius", "connectivity",
    "eig_energy", "eig_spectral_entropy", "eig_hfer",
    "energy", "smoothness_index", "spectral_entropy", "hfer",
]

#: Scalar metric names of each namespace.
NAMESPACE_METRICS: Dict[str, List[str]] = {
    "attn": list(PER_HEAD_METRIC_NAMES),
    "resid": list(PER_HEAD_HYBRID_METRIC_NAMES),
    "graph": list(GRAPH_METRIC_NAMES),
}

#: Arrays stored next to the scalar metrics (not listed in ``metric_names``).
ARRAY_NAMES: Dict[str, List[str]] = {
    "attn": ["eigenvalues", "pattern"],
    "resid": [],
    "graph": ["eigenvalues", "fiedler_vector", "spectral_masses"],
}

PER_HEAD_NAMESPACES = ("attn", "resid")
_KEY = re.compile(r"^blocks\.(-?\d+)\.(attn|resid|graph)\.([A-Za-z0-9_]+)$")
_FORMAT_VERSION = 1


def metric_family(qualified: str) -> str:
    """Tensor family ("attention" or "hybrid") of a namespaced metric name
    such as ``"attn.hfer"``, ``"resid.energy"`` or ``"graph.hfer"``."""
    namespace, _, name = qualified.partition(".")
    if namespace == "attn":
        return "attention"
    if namespace == "resid":
        return "hybrid"
    if name == "fiedler_vector":
        return "attention"
    return METRIC_FAMILIES.get(name, "hybrid" if name == "spectral_masses" else "attention")


def resolve_metrics(metrics, hybrid: bool = True) -> List[str]:
    """Normalise a metric selection to namespaced names (``"attn.hfer"``).

    ``metrics`` is None (every metric), or a list whose entries are a
    namespace (``"attn"``, ``"resid"``, ``"graph"``), a namespaced name
    (``"graph.hfer"``), or a bare name, which selects it in every namespace
    that has it (``"hfer"`` gives ``attn.hfer``, ``resid.hfer`` and
    ``graph.hfer``). With ``hybrid=False`` the hybrid metrics are left out of
    ``None`` and of namespaces, and naming one explicitly raises.
    """
    def allowed(q):
        return hybrid or metric_family(q) == "attention"

    every = [f"{ns}.{m}" for ns, names in NAMESPACE_METRICS.items() for m in names]
    if metrics is None:
        return [q for q in every if allowed(q)]
    if isinstance(metrics, str):
        metrics = [metrics]
    out: List[str] = []
    for entry in metrics:
        entry = str(entry).strip()
        if entry.endswith(".*"):
            entry = entry[:-2]
        if entry in NAMESPACE_METRICS:
            chosen = [f"{entry}.{m}" for m in NAMESPACE_METRICS[entry]]
            chosen = [q for q in chosen if allowed(q)]
        elif "." in entry:
            ns, _, name = entry.partition(".")
            if ns not in NAMESPACE_METRICS or name not in NAMESPACE_METRICS[ns]:
                raise ValueError(_unknown_metric(entry, every))
            chosen = [entry]
        else:
            chosen = [q for q in every if q.split(".", 1)[1] == entry]
            if not chosen:
                raise ValueError(_unknown_metric(entry, every))
            if not hybrid:
                chosen = [q for q in chosen if allowed(q)] or chosen
        for q in chosen:
            if not allowed(q):
                raise ValueError(
                    f"{q} reads the hidden states (family 'hybrid'); pass hybrid=True "
                    "or choose an attention metric")
            if q not in out:
                out.append(q)
    return [q for q in every if q in out]


def _unknown_metric(entry: str, every: Sequence[str]) -> str:
    bare = sorted({q.split(".", 1)[1] for q in every})
    near = difflib.get_close_matches(entry, list(every) + bare, n=3)
    hint = f" Did you mean {', '.join(near)}?" if near else ""
    return (f"unknown metric {entry!r}.{hint} Namespaces: attn, resid, graph; "
            f"metrics: {', '.join(bare)}")


def _config_to_dict(config: GSPConfig) -> Dict[str, Any]:
    d = asdict(config)
    return json.loads(json.dumps(d, default=str))


def _config_from_dict(d: Dict[str, Any]) -> GSPConfig:
    known = {f.name for f in fields(GSPConfig)}
    return GSPConfig(**{k: v for k, v in d.items() if k in known})


class SpectralCache:
    """Spectral measurements of one text or a batch of texts, keyed like a
    TransformerLens ``ActivationCache``.

    ``cache["blocks.10.attn.fiedler_value"]`` is a ``[n_heads]`` array,
    ``cache["blocks.10.graph.hfer"]`` a float. For a batch, values gain a
    leading text dimension (stacked when the shapes agree, a list otherwise),
    and ``cache[i]`` is the cache of text ``i``. ``cache["attn.fiedler_value",
    10]`` is the same as ``cache["blocks.10.attn.fiedler_value"]``.

    Iteration and ``len`` follow the keys, as for a dict. ``to_dataframe``
    gives one row per (text, layer, head, metric); ``save`` and ``load`` keep
    everything in one ``.npz`` file with its metadata as JSON.
    """

    def __init__(self, records: List[Dict[str, Any]], tokens: List[List[str]],
                 layers: Sequence[int], n_heads: int, metric_names: Sequence[str],
                 config: Optional[GSPConfig] = None, meta: Optional[Dict[str, Any]] = None):
        if len(records) != len(tokens):
            raise ValueError("one token list per record is needed")
        self._records = records
        self._tokens = [list(t) for t in tokens]
        self._layers = sorted(int(x) for x in layers)
        self.n_heads = int(n_heads)
        self._metric_names = list(metric_names)
        self.config = config or GSPConfig(device="cpu")
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.setdefault("texts", [None] * len(records))
        self.meta.setdefault("token_ids", [None] * len(records))
        self.meta.setdefault("heads", None)
        self.meta.setdefault("token_span", None)
        self.meta.setdefault("model_name", None)
        self.meta.setdefault("source", None)
        self.meta.setdefault("batched", len(records) > 1)

    # ── basic properties ────────────────────────────────────────────────────

    @property
    def n_texts(self) -> int:
        """Number of texts in the cache."""
        return len(self._records)

    @property
    def is_batch(self) -> bool:
        """True when the cache was built from a list of texts."""
        return bool(self.meta.get("batched")) or self.n_texts > 1

    @property
    def tokens(self) -> Union[List[str], List[List[str]]]:
        """Token strings: a list for one text, a list of lists for a batch."""
        return self._tokens if self.is_batch else self._tokens[0]

    @property
    def texts(self) -> Union[Optional[str], List[Optional[str]]]:
        """The input strings (None where the input was token ids)."""
        t = self.meta["texts"]
        return t if self.is_batch else t[0]

    @property
    def layers(self) -> List[int]:
        """True layer indices present in the cache, ascending."""
        return list(self._layers)

    @property
    def metric_names(self) -> List[str]:
        """Namespaced scalar metrics present, e.g. ``"attn.fiedler_value"``."""
        return list(self._metric_names)

    @property
    def families(self) -> Dict[str, str]:
        """Tensor family of every metric in :attr:`metric_names`."""
        return {q: metric_family(q) for q in self._metric_names}

    @property
    def heads(self) -> Dict[int, List[int]]:
        """Heads computed in each layer (every head when ``heads=None``)."""
        sel = self.meta.get("heads")
        if sel is None:
            return {lay: list(range(self.n_heads)) for lay in self._layers}
        return {int(k): list(v) for k, v in sel.items()}

    @property
    def token_span(self) -> Optional[Tuple[int, int]]:
        """The ``(start, end)`` span of each text that the metrics cover
        (None for the whole sequence; per text for a batch)."""
        spans = self.meta.get("token_span")
        if spans is None:
            return None
        spans = [tuple(s) if s is not None else None for s in spans]
        return spans if self.is_batch else spans[0]

    @property
    def model_name(self) -> Optional[str]:
        return self.meta.get("model_name")

    def keys(self) -> List[str]:
        """Every key present, in layer then namespace order."""
        seen: Dict[str, None] = {}
        for rec in self._records:
            for k in rec:
                seen.setdefault(k, None)
        return sorted(seen, key=_key_order)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key) -> bool:
        try:
            key = self._normalise_key(key)
        except (KeyError, TypeError):
            return False
        return any(key in rec for rec in self._records)

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]

    # ── indexing ────────────────────────────────────────────────────────────

    def _normalise_key(self, key) -> str:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise KeyError(f"tuple keys are (metric, layer), got {key!r}")
            name, layer = key
            name = str(name)
            if "." not in name:
                ns = "attn" if name in NAMESPACE_METRICS["attn"] + ARRAY_NAMES["attn"] \
                    else "resid" if name in NAMESPACE_METRICS["resid"] else "graph"
                name = f"{ns}.{name}"
            return f"blocks.{int(layer)}.{name}"
        if not isinstance(key, str):
            raise TypeError(f"keys are strings or (metric, layer) tuples, got {type(key).__name__}")
        return key

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            return self.text(int(key))
        key = self._normalise_key(key)
        vals = []
        for rec in self._records:
            if key not in rec:
                raise KeyError(self._missing(key))
            vals.append(rec[key])
        if not self.is_batch:
            return _plain(vals[0])
        return _stack(vals)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _missing(self, key: str) -> str:
        m = _KEY.match(key)
        msg = f"{key!r} is not in the cache."
        if m and int(m.group(1)) not in self._layers:
            msg += f" Layers present: {_compact(self._layers)}."
        near = difflib.get_close_matches(key, self.keys(), n=3)
        if near:
            msg += f" Did you mean {', '.join(near)}?"
        return msg

    def text(self, i: int) -> "SpectralCache":
        """The cache of text ``i`` of a batch, as a single-text cache."""
        n = self.n_texts
        if not -n <= i < n:
            raise IndexError(f"text index {i} out of range for {n} texts")
        i %= n
        meta = dict(self.meta)
        for k in ("texts", "token_ids", "token_span"):
            if meta.get(k) is not None:
                meta[k] = [meta[k][i]]
        meta["batched"] = False
        meta["text_id"] = i
        return SpectralCache([self._records[i]], [self._tokens[i]], self._layers,
                             self.n_heads, self._metric_names, self.config, meta)

    def layer(self, layer: int) -> "LayerSpectra":
        """Every measurement of one layer: ``cache.layer(10).head(3)``."""
        if layer < 0 and self.meta.get("n_layers"):
            layer += int(self.meta["n_layers"])
        if layer not in self._layers:
            raise KeyError(f"layer {layer} is not in the cache; layers present: "
                           f"{_compact(self._layers)}")
        return LayerSpectra(self, int(layer))

    def metric(self, name: str, text: int = 0) -> np.ndarray:
        """One metric over every layer of one text: ``[layers, n_heads]`` for
        the per-head namespaces, ``[layers]`` for ``graph``. Rows follow
        :attr:`layers`. A bare per-head name means ``attn``."""
        q = name if "." in name else (
            f"attn.{name}" if name in NAMESPACE_METRICS["attn"] else
            f"resid.{name}" if name in NAMESPACE_METRICS["resid"] else f"graph.{name}")
        if q not in self._metric_names:
            raise KeyError(f"{q!r} was not computed; metrics present: "
                           f"{', '.join(self._metric_names)}")
        rec = self._records[text]
        rows = [np.asarray(rec[f"blocks.{lay}.{q}"], dtype=np.float64) for lay in self._layers]
        return np.stack(rows)

    # ── export ──────────────────────────────────────────────────────────────

    def to_dataframe(self, metrics: Optional[Sequence[str]] = None):
        """Long-format table: one row per (layer, head, metric), with a
        ``text_id`` column first for a batch. ``head`` is missing (``<NA>``)
        for the head-averaged ``graph`` metrics, and heads that were not
        computed are left out. Columns: [text_id], layer, head, metric,
        family, value."""
        import pandas as pd

        wanted = self._metric_names if metrics is None else resolve_metrics(metrics)
        wanted = [q for q in wanted if q in self._metric_names]
        rows: Dict[str, list] = {"text_id": [], "layer": [], "head": [], "metric": [],
                                 "family": [], "value": []}
        for tid, rec in enumerate(self._records):
            for lay in self._layers:
                for q in wanted:
                    key = f"blocks.{lay}.{q}"
                    if key not in rec:
                        continue
                    v = np.asarray(rec[key], dtype=np.float64)
                    fam = metric_family(q)
                    if v.ndim == 0:
                        rows["text_id"].append(tid)
                        rows["layer"].append(lay)
                        rows["head"].append(None)
                        rows["metric"].append(q)
                        rows["family"].append(fam)
                        rows["value"].append(float(v))
                        continue
                    for h, x in enumerate(v.reshape(-1)):
                        if np.isnan(x):
                            continue
                        rows["text_id"].append(tid)
                        rows["layer"].append(lay)
                        rows["head"].append(h)
                        rows["metric"].append(q)
                        rows["family"].append(fam)
                        rows["value"].append(float(x))
        df = pd.DataFrame(rows)
        df["head"] = pd.array(df["head"].tolist(), dtype="Int64")
        df["layer"] = df["layer"].astype("int64")
        df["text_id"] = df["text_id"].astype("int64")
        if not self.is_batch:
            df = df.drop(columns="text_id")
        return df

    def save(self, path) -> Path:
        """Write the cache to one ``.npz`` file: every array, plus the
        metadata (tokens, layers, heads, config) as JSON. Loading needs no
        pickle. Returns the path written (``.npz`` is appended if missing)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        arrays: Dict[str, np.ndarray] = {}
        index: List[List[str]] = []
        for tid, rec in enumerate(self._records):
            keys = []
            for key, value in rec.items():
                arrays[f"{tid}::{key}"] = np.asarray(value)
                keys.append(key)
            index.append(keys)
        meta = {
            "format": "spectral-trust.SpectralCache",
            "format_version": _FORMAT_VERSION,
            "tokens": self._tokens,
            "layers": self._layers,
            "n_heads": self.n_heads,
            "metric_names": self._metric_names,
            "config": _config_to_dict(self.config),
            "meta": json.loads(json.dumps(self.meta, default=_json_default)),
            "index": index,
        }
        try:
            from . import __version__ as version
        except ImportError:  # pragma: no cover
            version = None
        meta["library_version"] = version
        arrays["__metadata__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path) -> "SpectralCache":
        """Read a cache written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(bytes(data["__metadata__"]).decode("utf-8"))
            if meta.get("format") != "spectral-trust.SpectralCache":
                raise ValueError(f"{path} is not a SpectralCache file")
            records = []
            for tid, keys in enumerate(meta["index"]):
                rec = {}
                for key in keys:
                    arr = data[f"{tid}::{key}"]
                    rec[key] = float(arr) if arr.ndim == 0 else arr
                records.append(rec)
        inner = meta.get("meta", {})
        if inner.get("heads") is not None:
            inner["heads"] = {int(k): v for k, v in inner["heads"].items()}
        return cls(records, meta["tokens"], meta["layers"], meta["n_heads"],
                   meta["metric_names"], _config_from_dict(meta["config"]), inner)

    # ── views ───────────────────────────────────────────────────────────────

    def heatmap(self, metric: str = "fiedler_value", **kwargs):
        """Layers x heads heatmap of one metric (see :func:`spectral_trust.viz.heatmap`)."""
        from . import viz
        return viz.heatmap(self, metric, **kwargs)

    def graph(self, layer: int, head: Optional[int] = None, **kwargs):
        """Attention graph of one layer or head (see :func:`spectral_trust.viz.graph`)."""
        from . import viz
        return viz.graph(self, layer, head, **kwargs)

    def spectrum(self, layer: int, head: Optional[int] = None, **kwargs):
        """Laplacian spectrum of one layer or head (see :func:`spectral_trust.viz.spectrum`)."""
        from . import viz
        return viz.spectrum(self, layer, head, **kwargs)

    def __repr__(self) -> str:
        counts = {ns: sum(q.startswith(ns + ".") for q in self._metric_names)
                  for ns in NAMESPACE_METRICS}
        ns_txt = ", ".join(f"{ns}={n}" for ns, n in counts.items() if n)
        lens = [len(t) for t in self._tokens]
        if not self.is_batch:
            tok = f"{lens[0]} tokens"
        elif min(lens) == max(lens):
            tok = f"{self.n_texts} texts of {lens[0]} tokens"
        else:
            tok = f"{self.n_texts} texts, {min(lens)}-{max(lens)} tokens"
        name = f"{self.model_name!r}, " if self.model_name else ""
        heads = f"{self.n_heads} heads"
        if self.meta.get("heads") is not None:
            n_sel = sum(len(v) for v in self.meta["heads"].values())
            heads += f" ({n_sel} computed)"
        return (f"SpectralCache({name}{tok}, layers {_compact(self._layers)}, {heads}, "
                f"metrics: {ns_txt})")

    def _repr_html_(self) -> str:
        import html
        return f"<pre>{html.escape(repr(self))}</pre>"


class LayerSpectra:
    """Measurements of one layer of a :class:`SpectralCache`.

    ``layer["hfer"]`` is the head-averaged (``graph``) metric;
    ``layer.attn["fiedler_value"]`` and ``layer.resid["hfer"]`` are
    ``[n_heads]`` arrays; ``layer["attn.fiedler_value"]`` names the namespace
    explicitly. ``layer.head(3)`` selects one head.
    """

    def __init__(self, cache: SpectralCache, layer: int):
        self.cache = cache
        self.index = layer

    def _namespace(self, ns: str) -> Dict[str, Any]:
        out = {}
        for q in self.cache.metric_names:
            if q.startswith(ns + "."):
                out[q.split(".", 1)[1]] = self.cache[f"blocks.{self.index}.{q}"]
        return out

    @property
    def attn(self) -> Dict[str, Any]:
        """Per-head attention-only metrics, name -> ``[n_heads]``."""
        return self._namespace("attn")

    @property
    def resid(self) -> Dict[str, Any]:
        """Per-head hybrid metrics, name -> ``[n_heads]``."""
        return self._namespace("resid")

    @property
    def graph(self) -> Dict[str, Any]:
        """Head-averaged graph metrics, name -> value."""
        return self._namespace("graph")

    @property
    def eigenvalues(self):
        """Spectrum of the head-averaged graph."""
        return self.cache.get(f"blocks.{self.index}.graph.eigenvalues")

    @property
    def fiedler_vector(self):
        """Fiedler vector of the head-averaged graph (largest entry positive)."""
        return self.cache.get(f"blocks.{self.index}.graph.fiedler_vector")

    def __getitem__(self, name: str):
        q = name if "." in name else f"graph.{name}"
        return self.cache[f"blocks.{self.index}.{q}"]

    def head(self, head: int) -> "HeadSpectra":
        n = self.cache.n_heads
        if not -n <= head < n:
            raise IndexError(f"head {head} out of range for {n} heads")
        head %= n
        if head not in self.cache.heads.get(self.index, []):
            raise KeyError(f"head {head} of layer {self.index} was not computed; "
                           f"heads computed: {self.cache.heads.get(self.index, [])}")
        return HeadSpectra(self, head)

    def to_dict(self) -> Dict[str, Any]:
        d = {f"graph.{k}": v for k, v in self.graph.items()}
        d.update({f"attn.{k}": v for k, v in self.attn.items()})
        d.update({f"resid.{k}": v for k, v in self.resid.items()})
        return d

    def __repr__(self) -> str:
        lines = [f"LayerSpectra(layer {self.index})"]
        for k, v in self.graph.items():
            lines.append(f"  graph.{k:<22} {_fmt(v)}")
        for ns in PER_HEAD_NAMESPACES:
            for k, v in self._namespace(ns).items():
                lines.append(f"  {ns}.{k:<{27 - len(ns)}} {_fmt_heads(v)}")
        return "\n".join(lines)


class HeadSpectra:
    """Measurements of one head: ``cache.layer(10).head(3)``.

    ``head["fiedler_value"]`` resolves a bare name in ``attn`` or ``resid``
    when only one of them has it; ``hfer`` exists in both, so write
    ``head["attn.hfer"]`` or ``head["resid.hfer"]``.
    """

    def __init__(self, layer: LayerSpectra, head: int):
        self.layer = layer
        self.index = head

    def _namespace(self, ns: str) -> Dict[str, Any]:
        batch = self.layer.cache.is_batch
        return {k: _take_head(v, self.index, batch)
                for k, v in self.layer._namespace(ns).items()}

    @property
    def attn(self) -> Dict[str, Any]:
        return self._namespace("attn")

    @property
    def resid(self) -> Dict[str, Any]:
        return self._namespace("resid")

    @property
    def eigenvalues(self):
        """This head's Laplacian spectrum, ascending."""
        v = self.layer.cache.get(f"blocks.{self.layer.index}.attn.eigenvalues")
        return None if v is None else _take_head(v, self.index, self.layer.cache.is_batch)

    def __getitem__(self, name: str):
        if "." in name:
            ns, _, bare = name.partition(".")
            values = self._namespace(ns)
            if bare not in values:
                raise KeyError(f"{name!r} is not in the cache")
            return values[bare]
        hits = [ns for ns in PER_HEAD_NAMESPACES if name in self._namespace(ns)]
        if len(hits) == 1:
            return self._namespace(hits[0])[name]
        if not hits:
            raise KeyError(f"{name!r} is not a per-head metric in the cache")
        raise KeyError(f"{name!r} is ambiguous: write 'attn.{name}' (attention only) "
                       f"or 'resid.{name}' (hybrid)")

    def to_dict(self) -> Dict[str, Any]:
        d = {f"attn.{k}": v for k, v in self.attn.items()}
        d.update({f"resid.{k}": v for k, v in self.resid.items()})
        return d

    def __repr__(self) -> str:
        lines = [f"HeadSpectra(layer {self.layer.index}, head {self.index})"]
        for k, v in self.to_dict().items():
            lines.append(f"  {k:<28} {_fmt(v)}")
        return "\n".join(lines)


# ── helpers ─────────────────────────────────────────────────────────────────

def _key_order(key: str):
    m = _KEY.match(key)
    if not m:
        return (10 ** 9, 9, key)
    ns_rank = {"attn": 0, "resid": 1, "graph": 2}[m.group(2)]
    return (int(m.group(1)), ns_rank, m.group(3))


def _plain(v):
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return float(v)
    return v


def _stack(vals):
    if all(np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0) for v in vals):
        return np.asarray([float(v) for v in vals])
    arrays = [np.asarray(v) for v in vals]
    if all(a.shape == arrays[0].shape for a in arrays):
        return np.stack(arrays)
    return arrays


def _take_head(v, head: int, batch: bool):
    """Row ``head`` of a per-head array: axis 0 for one text, axis 1 for a
    stacked batch, and each entry of a ragged batch list."""
    if isinstance(v, list):
        return [_plain(np.asarray(x)[head]) for x in v]
    a = np.asarray(v)
    return _plain(a[:, head] if batch else a[head])


def _fmt(v) -> str:
    if isinstance(v, (float, int, np.floating)):
        return f"{float(v):.4f}"
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        return f"{float(a):.4f}"
    return np.array2string(a, precision=4, max_line_width=72, threshold=8)


def _fmt_heads(v) -> str:
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 1:
        ok = a[~np.isnan(a)]
        if ok.size == 0:
            return "[not computed]"
        return (f"[{a.shape[0]} heads] min {ok.min():.4f}  mean {ok.mean():.4f}  "
                f"max {ok.max():.4f}")
    return f"array{list(a.shape)}"


def _compact(xs: Sequence[int]) -> str:
    xs = list(xs)
    if not xs:
        return "[]"
    if xs == list(range(xs[0], xs[-1] + 1)) and len(xs) > 2:
        return f"{xs[0]}-{xs[-1]}"
    if len(xs) > 8:
        return "[" + ", ".join(map(str, xs[:4])) + ", ..., " + ", ".join(map(str, xs[-2:])) + "]"
    return "[" + ", ".join(map(str, xs)) + "]"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)
