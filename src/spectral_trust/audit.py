# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Confound audit of spectral readouts: does a readout's association with a
label survive the covariates that could produce it?

:func:`confound_audit` takes per-item readouts (Fiedler value, HFER, ...), a
target (a binary label or a continuous score) and covariates (sequence
length in tokens, sink mass, effective rank of the hidden states, mean
position, and any others), and reports for each readout:

- binary target: the raw AUROC; the AUROC of the readout after its linear
  dependence on the covariates is removed (``resid_auroc``); the partial
  correlation with the label; and the gain in out-of-fold AUROC when the
  readout is added to a logistic regression on the covariates
  (``cv_gain``), all with bootstrap intervals over items;
- continuous target: the raw Spearman correlation and the partial Spearman
  correlation given the covariates, with bootstrap intervals.

:func:`confound_covariates` and :func:`covariates_from_cache` compute the
standard covariates; :func:`bootstrap_ci` and :func:`paired_bootstrap` are
the interval helpers the rest of the library uses. New after 0.5.0
(unreleased).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

#: Covariates computed by :func:`confound_covariates`.
STANDARD_COVARIATES = ("n_tokens", "sink_mass", "effective_rank", "mean_position")


# ── interval helpers ─────────────────────────────────────────────────────────

def bootstrap_ci(values, stat: Callable = np.mean, n_boot: int = 1000, ci: float = 0.95,
                 seed: int = 0, groups=None) -> Tuple[float, float, float]:
    """``(estimate, lo, hi)``: percentile bootstrap interval of ``stat``
    over items (or over ``groups`` of items, resampled whole)."""
    v = np.asarray(values, dtype=np.float64)
    est = float(stat(v))
    if len(v) < 2:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_sets = _resample_index(len(v), n_boot, rng, groups)
    boots = np.array([stat(v[ix]) for ix in idx_sets])
    a = (1 - ci) / 2
    return est, float(np.nanquantile(boots, a)), float(np.nanquantile(boots, 1 - a))


def paired_bootstrap(a, b, stat: Callable = np.mean, **kwargs) -> Tuple[float, float, float]:
    """Bootstrap interval of ``stat(a - b)`` over paired items."""
    return bootstrap_ci(np.asarray(a, np.float64) - np.asarray(b, np.float64), stat, **kwargs)


def _resample_index(n: int, n_boot: int, rng, groups=None) -> List[np.ndarray]:
    if groups is None:
        return [rng.integers(0, n, n) for _ in range(n_boot)]
    g = np.asarray(groups)
    uniq = np.unique(g)
    members = {u: np.nonzero(g == u)[0] for u in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        out.append(np.concatenate([members[u] for u in pick]))
    return out


# ── covariates ───────────────────────────────────────────────────────────────

def effective_rank(signal) -> float:
    """Effective rank of a signal ``[n, d]`` (Roy and Vetterli 2007):
    ``exp`` of the entropy of its normalised singular values."""
    import torch
    X = signal.detach().double().cpu().numpy() if torch.is_tensor(signal) else \
        np.asarray(signal, np.float64)
    s = np.linalg.svd(X, compute_uv=False)
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def confound_covariates(attention=None, signal=None, *, positions: Optional[Sequence[int]] = None,
                        sink: int = 0) -> Dict[str, float]:
    """The standard covariates of one item: ``n_tokens``, ``sink_mass``
    (mean attention to position ``sink`` over heads and later queries, from
    ``attention [H, S, S]``), ``effective_rank`` of ``signal [S, d]`` and
    ``mean_position`` of the analysed ``positions`` (default all)."""
    from .numeric.diagnostics import sink_mass
    S = None
    out: Dict[str, float] = {}
    if attention is not None:
        S = int(np.shape(attention)[-1])
        out["sink_mass"] = sink_mass(attention, sink)
    if signal is not None:
        S = S or int(np.shape(signal)[0])
        out["effective_rank"] = effective_rank(signal)
    if S is None and positions is None:
        raise ValueError("pass attention, signal or positions")
    pos = np.arange(S) if positions is None else np.asarray(positions, np.float64)
    out["n_tokens"] = float(len(pos))
    out["mean_position"] = float(np.mean(pos))
    return out


def covariates_from_cache(cache, *, layers: Optional[Sequence[int]] = None,
                          sink: int = 0) -> Dict[str, np.ndarray]:
    """Per-text covariates from a :class:`~spectral_trust.SpectralCache`:
    ``n_tokens``, ``mean_position`` and, when the attention patterns were
    kept, ``sink_mass`` averaged over ``layers`` (default all)."""
    n = cache.n_texts
    layers = list(cache.layers) if layers is None else list(layers)
    toks = cache.tokens if cache.is_batch else [cache.tokens]
    out = {"n_tokens": np.array([len(t) for t in toks], np.float64),
           "mean_position": np.array([(len(t) - 1) / 2 for t in toks], np.float64)}
    if cache.meta.get("patterns_kept"):
        sm = np.zeros(n)
        for i in range(n):
            c = cache.text(i) if cache.is_batch else cache
            vals = [float(np.asarray(c[f"blocks.{l}.attn.pattern"])[:, sink + 1:, sink].mean())
                    for l in layers]
            sm[i] = np.mean(vals)
        out["sink_mass"] = sm
    return out


# ── models ───────────────────────────────────────────────────────────────────

def auroc(scores, labels) -> float:
    """Area under the ROC curve (Mann-Whitney), ties counted half."""
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _standardize(Z: np.ndarray, ref: Optional[np.ndarray] = None):
    ref = Z if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (Z - mu) / sd


def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    """L2-penalised logistic regression by Newton steps; returns weights
    with the intercept first."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    pen = np.full(Xb.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + pen * w
        Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(pen + 1e-9)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _oof_scores(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold logistic scores with stratified folds."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), dtype=int)
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    out = np.zeros(len(y))
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        Xtr = _standardize(X[tr])
        Xte = _standardize(X[te], X[tr])
        w = _logistic_fit(Xtr, y[tr].astype(float))
        out[te] = np.hstack([np.ones((te.sum(), 1)), Xte]) @ w
    return out


def _residualize(v: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
    if C is None or C.shape[1] == 0:
        return v - v.mean()
    Cb = np.hstack([np.ones((len(v), 1)), _standardize(C)])
    beta, *_ = np.linalg.lstsq(Cb, v, rcond=None)
    return v - Cb @ beta


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


# ── the audit ────────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    """Rows of :func:`confound_audit`, one per readout.

    Each row holds the readout name, ``n`` and, for a binary target,
    ``auroc``, ``resid_auroc``, ``partial_r``, ``cv_auroc_covariates``,
    ``cv_auroc_full`` and ``cv_gain``; for a continuous target ``spearman``
    and ``partial_spearman``. Each statistic ``x`` comes with ``x_lo`` and
    ``x_hi``.
    """
    kind: str
    covariates: List[str]
    rows: List[Dict[str, float]] = field(default_factory=list)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows)

    def to_dict(self) -> Dict:
        return {"kind": self.kind, "covariates": self.covariates, "rows": self.rows}

    def to_markdown(self, digits: int = 2) -> str:
        if self.kind == "binary":
            cols = [("auroc", "AUROC raw"), ("resid_auroc", "AUROC after covariates"),
                    ("partial_r", "partial r"), ("cv_gain", "CV AUROC gain")]
        else:
            cols = [("spearman", "Spearman raw"), ("partial_spearman", "partial Spearman")]
        head = "| readout | " + " | ".join(c[1] for c in cols) + " |"
        sep = "|---|" + "---|" * len(cols)
        lines = [head, sep]
        f = f"{{:.{digits}f}}"
        for r in self.rows:
            cells = [f"{f.format(r[k])} [{f.format(r[k + '_lo'])}, {f.format(r[k + '_hi'])}]"
                     for k, _ in cols]
            lines.append(f"| {r['readout']} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def confound_audit(readouts: Mapping[str, Sequence[float]], target: Sequence[float],
                   covariates: Optional[Mapping[str, Sequence[float]]] = None, *,
                   cache=None, kind: str = "auto", n_folds: int = 5, n_boot: int = 1000,
                   ci: float = 0.95, seed: int = 0, groups=None) -> AuditResult:
    """Effect of each readout on ``target``, raw and given the covariates.

    Args:
        readouts: name -> per-item values.
        target: per-item label (binary) or score (continuous).
        covariates: name -> per-item values (length, sink mass, effective
            rank, mean position, any others). Constant covariates are
            ignored.
        cache: a :class:`~spectral_trust.SpectralCache` with one text per
            item; :func:`covariates_from_cache` fills the standard
            covariates it can, without overriding ``covariates``.
        kind: ``"binary"``, ``"continuous"`` or ``"auto"`` (binary when the
            target has two values).
        n_folds: folds of the cross-validated logistic regression.
        n_boot, ci, seed: bootstrap over items (``groups``: resample groups
            of items whole, e.g. realisations shared across conditions).
    """
    y = np.asarray(target, np.float64)
    n = len(y)
    cov = {}
    if cache is not None:
        for k, v in covariates_from_cache(cache).items():
            cov[k] = np.asarray(v, np.float64)
    for k, v in (covariates or {}).items():
        cov[k] = np.asarray(v, np.float64)
    for k, v in cov.items():
        if len(v) != n:
            raise ValueError(f"covariate {k!r} has {len(v)} values for {n} items")
    used = [k for k, v in cov.items() if np.nanstd(v) > 1e-12]
    C = np.column_stack([cov[k] for k in used]) if used else np.zeros((n, 0))
    if kind == "auto":
        kind = "binary" if len(np.unique(y)) == 2 else "continuous"
    rng = np.random.default_rng(seed)
    boots = _resample_index(n, n_boot, rng, groups)
    a = (1 - ci) / 2
    result = AuditResult(kind, used)

    def interval(fn, est):
        vals = np.array([fn(ix) for ix in boots], dtype=np.float64)
        return est, float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))

    if kind == "binary":
        yb = (y == np.max(y)).astype(int)
        oof_c = _oof_scores(C, yb, n_folds, seed) if C.shape[1] else np.zeros(n)
    for name, vals in readouts.items():
        r = np.asarray(vals, np.float64)
        if len(r) != n:
            raise ValueError(f"readout {name!r} has {len(r)} values for {n} items")
        row: Dict[str, float] = {"readout": name, "n": n}
        if kind == "binary":
            rr = _residualize(r, C)
            yr = _residualize(yb.astype(float), C)
            oof_f = _oof_scores(np.column_stack([C, r]), yb, n_folds, seed)
            stats = {
                "auroc": lambda ix: auroc(r[ix], yb[ix]),
                "resid_auroc": lambda ix: auroc(rr[ix], yb[ix]),
                "partial_r": lambda ix: _corr(rr[ix], yr[ix]),
                "cv_auroc_covariates": lambda ix: auroc(oof_c[ix], yb[ix]) if C.shape[1]
                else 0.5,
                "cv_auroc_full": lambda ix: auroc(oof_f[ix], yb[ix]),
                "cv_gain": lambda ix: auroc(oof_f[ix], yb[ix]) - (
                    auroc(oof_c[ix], yb[ix]) if C.shape[1] else 0.5),
            }
        else:
            rk, yk = rankdata(r), rankdata(y)
            Ck = np.column_stack([rankdata(c) for c in C.T]) if C.shape[1] else C
            rr, yr = _residualize(rk, Ck), _residualize(yk, Ck)
            stats = {
                "spearman": lambda ix: _corr(rankdata(r[ix]), rankdata(y[ix])),
                "partial_spearman": lambda ix: _corr(rr[ix], yr[ix]),
            }
        for key, fn in stats.items():
            est, lo, hi = interval(fn, fn(np.arange(n)))
            row[key], row[key + "_lo"], row[key + "_hi"] = est, lo, hi
        result.rows.append(row)
    return result
