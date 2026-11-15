# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

r"""
Activation patching, plain and frequency-resolved.

Plain patching (TransformerLens' ``run_with_hooks`` with a patching hook)
runs the corrupt input and replaces one activation, at chosen positions, by
its value on the clean input. Clean and corrupt inputs must have the same
length, and positions align (as in IOI, where the corrupt prompt swaps
names token for token).

Frequency-resolved patching replaces only a band of graph frequencies
along the token axis. With ``U`` an orthonormal basis of the S positions
(the eigenvectors of a graph Laplacian, ascending eigenvalue) and ``U_b``
the band's columns,

.. math::
    X' = X_{corrupt} + U_b U_b^T (X_{clean} - X_{corrupt})

i.e. both activations are projected onto the basis, the band's coefficients
are taken from the clean run and the rest from the corrupt run, and the
result is transformed back. The full band is plain patching (to float
rounding); the empty band leaves the corrupt run unchanged. Bases:

- ``"clean_attention"`` / ``"corrupt_attention"``: the Laplacian of the
  layer's head-averaged attention on that run, built as the measurements
  build it (:class:`~spectral_trust.GraphConstructor`);
- ``"path"``: the chain graph over positions (DCT-like, purely positional);
- ``"random"``: a Haar-random orthonormal basis (seeded), the null for a
  subspace of a given size;
- an orthonormal tensor ``[S, S]`` or ``[B, S, S]`` (columns are modes).

The projector ``U_b U_b^T`` does not depend on the signs of the
eigenvectors; it is basis-dependent only when a band edge splits a
degenerate eigenvalue.
"""
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..config import GSPConfig
from ._arch import Arch, unwrap
from ._hooks import canonical_site, register_site_hook
from .context import Positions, _device, _encode, _position_mask
from .filters import attention_adjacency, laplacian, path_adjacency

BASES = ("clean_attention", "corrupt_attention", "path", "random")
CAPTURE_SITES = ("resid_pre", "resid_post", "attn_out", "mlp_out", "z", "pattern")

Band = Optional[Tuple[Union[int, float, None], Union[int, float, None]]]


def equal_count_bands(n_modes: int, k: int) -> List[Tuple[int, int]]:
    """Split ``n_modes`` modes (ascending eigenvalue) into ``k`` contiguous
    bands of equal count (the first ``n_modes % k`` bands hold one more), as
    ``(lo, hi)`` index pairs. Bands are empty when ``k > n_modes``."""
    sizes = [len(a) for a in np.array_split(np.arange(n_modes), k)]
    out, lo = [], 0
    for s in sizes:
        out.append((lo, lo + s))
        lo += s
    return out


def haar_basis(n: int, seed: int, item: int = 0) -> torch.Tensor:
    """Haar-random orthonormal ``[n, n]`` (QR of a Gaussian with the signs
    of R's diagonal fixed), seeded by ``(seed, item)``."""
    rng = np.random.default_rng([int(seed), int(item)])
    G = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(G)
    Q = Q * np.sign(np.where(np.diag(R) == 0, 1.0, np.diag(R)))
    return torch.from_numpy(Q)


def select_modes(n: int, band: Band, eigenvalues: Optional[np.ndarray],
                 n_modes: Optional[int]) -> np.ndarray:
    """Indices of the band's modes among ``n`` modes in ascending eigenvalue
    order. Integer bounds are mode indices with slice semantics; float
    bounds are eigenvalues (``lo <= lambda < hi``, the top mode included
    when ``hi`` is at or above it). ``n_modes`` keeps the first (lowest)
    ``n_modes`` of them."""
    if band is None:
        idx = np.arange(n)
    else:
        lo, hi = band
        if all(v is None or isinstance(v, (int, np.integer)) for v in (lo, hi)):
            idx = np.arange(n)[slice(lo, hi)]
        else:
            if eigenvalues is None:
                raise ValueError("an eigenvalue band needs an eigenbasis; the random basis "
                                 "takes integer bands or n_modes")
            lam = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
            lo = -np.inf if lo is None else float(lo)
            hi = np.inf if hi is None else float(hi)
            inside = (lam >= lo) & (lam < hi)
            if hi >= lam.max():
                inside |= lam >= lo
            idx = np.nonzero(inside)[0]
    if n_modes is not None:
        idx = idx[: int(n_modes)]
    return idx


@dataclass
class PatchSpec:
    """One patch: where, and which part of the activation.

    Attributes:
        site: ``"resid_pre"``, ``"resid_post"``, ``"attn_out"``,
            ``"mlp_out"``, or ``"z"``/``"head"`` (per-head outputs).
        layer: block index.
        head: head or heads for ``site="z"`` (None: all heads).
        positions: positions to patch (default all). With a band, the
            band-projected difference is added at these positions only.
        band: None for plain patching, or ``(lo, hi)``: integer mode
            indices (ascending eigenvalue, slice semantics) or float
            eigenvalues.
        basis: one of :data:`BASES`, or an orthonormal tensor.
        n_modes: keep the first ``n_modes`` modes of the band (lowest
            eigenvalue first); with ``basis="random"`` and no band, the
            subspace size.
        seed: seed of the random basis.
    """
    site: str = "resid_pre"
    layer: int = 0
    head: Union[None, int, Sequence[int]] = None
    positions: Positions = None
    band: Band = None
    basis: Union[str, torch.Tensor] = "clean_attention"
    n_modes: Optional[int] = None
    seed: int = 0

    @property
    def frequency_resolved(self) -> bool:
        return self.band is not None or self.n_modes is not None


class Patcher:
    """Runs a clean and a corrupt input once, caches their activations, and
    patches clean activations into the corrupt run.

    Examples:
        >>> p = st.Patcher(model, clean_ids, corrupt_ids)
        >>> logits = p.patch(site="resid_pre", layer=6)                 # plain
        >>> logits = p.patch(site="resid_pre", layer=6, band=(0, 3))    # lowest 3 modes
        >>> ld = p.patch_many([st.PatchSpec("resid_pre", l, band=b)
        ...                    for l in range(12) for b in st.equal_count_bands(16, 8)],
        ...                   metric=logit_diff)

    Args:
        model: a :class:`~spectral_trust.SpectralModel` or a Hugging Face
            causal LM.
        clean, corrupt: text, texts or ids (``[S]`` or ``[B, S]``) of the
            same shape. Batches must not be padded.
        tokenizer: for text inputs (default: the SpectralModel's).
        config: graph construction for the attention bases (default: the
            SpectralModel's config, or the library defaults). Its
            normalisation must be symmetric (``"sym"`` or ``"none"``).
        layers, sites: activations to cache up front (``sites`` from
            :data:`CAPTURE_SITES`, ``"pattern"`` for attention weights
            ``[B, H, S, S]``). By default nothing is cached up front: each
            call of :meth:`patch_many` captures what its patches need that is
            not cached yet, in one more pair of forward passes.
        forward_kwargs: extra keyword arguments of every forward call (for
            example ``{"logits_to_keep": 1}``).

    The cached activations belong to the model as it was when they were
    captured. Every call checks a token of the model's state (each
    parameter's storage and in-place version counter, the forward hooks and
    the attention implementation) and raises if it changed, for example
    under a weight edit or inside an :func:`~spectral_trust.intervene` block
    that was opened or closed since; :meth:`refresh` recaptures. Writes
    through ``param.data`` bypass the version counter and are not detected.

    Attributes:
        clean_logits, corrupt_logits: logits of the two unpatched runs.

    Added in 0.5.0.
    """

    def __init__(self, model, clean, corrupt, *, tokenizer=None,
                 config: Optional[GSPConfig] = None, layers: Optional[Sequence[int]] = None,
                 sites: Optional[Sequence[str]] = None, forward_kwargs: Optional[dict] = None):
        from ..model import SpectralModel, default_config
        self.model = unwrap(model)
        self.arch = Arch(model)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if config is None:
            config = model.config if isinstance(model, SpectralModel) else default_config()
        self.config = config
        self.clean_ids = _encode(clean, self.tokenizer)
        self.corrupt_ids = _encode(corrupt, self.tokenizer)
        if self.clean_ids.shape != self.corrupt_ids.shape:
            raise ValueError(
                f"clean {tuple(self.clean_ids.shape)} and corrupt "
                f"{tuple(self.corrupt_ids.shape)} differ in shape; positions must align")
        self.batch, self.seq_len = self.clean_ids.shape
        self.forward_kwargs = dict(forward_kwargs or {})
        n = self.arch.n_layers
        layers = list(range(n)) if layers is None else [self.arch.layer_index(l) for l in layers]
        sites = [] if sites is None else [s if s == "pattern" else canonical_site(s)
                                          for s in sites]
        self._prefetch = [(s, l) for s in sites for l in layers]
        self.refresh()

    # ── model state ────────────────────────────────────────────────────────

    def _state_token(self):
        params = tuple((id(p), p.data_ptr(), p._version) for p in self.model.parameters())
        hooks = tuple((id(m), tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                      for m in self.model.modules()
                      if m._forward_hooks or m._forward_pre_hooks)
        return params, hooks, self.arch.attn_implementation()

    def _check_state(self) -> None:
        if self._state_token() != self._token:
            raise RuntimeError(
                "the model changed since the clean and corrupt runs were cached (weights, hooks "
                "or attention implementation); call refresh() or create a new Patcher")

    def refresh(self) -> None:
        """Drop every cached activation and basis, and rerun the clean and
        corrupt inputs on the model as it is now."""
        self._cache: Dict[str, Dict[Tuple[str, int], torch.Tensor]] = {"clean": {}, "corrupt": {}}
        self._bases: Dict = {}
        self._token = self._state_token()
        self.clean_logits = self._capture("clean", self._prefetch)
        self.corrupt_logits = self._capture("corrupt", self._prefetch)

    def _ensure(self, items) -> None:
        """Capture, in one pair of passes, every ``(site, layer)`` of
        ``items`` that is not cached yet."""
        missing = sorted({it for it in items if it not in self._cache["clean"]})
        if missing:
            self._capture("clean", missing)
            self._capture("corrupt", missing)

    # ── capture ────────────────────────────────────────────────────────────

    def _forward(self, ids: torch.Tensor):
        """One forward pass with eager attention. Every pass of the patcher
        (clean, corrupt, patched) uses the same implementation, so patched
        logits differ from the unpatched ones only through the patch (eager
        and SDPA attention differ at float rounding)."""
        orig_impl = self.arch.attn_implementation()
        if orig_impl not in (None, "eager"):
            self.arch.set_attn_implementation("eager")
        try:
            with torch.no_grad():
                out = self.model(input_ids=ids.to(_device(self.model)), use_cache=False,
                                 **self.forward_kwargs)
        finally:
            if orig_impl not in (None, "eager"):
                self.arch.set_attn_implementation(orig_impl)
        return out.logits if hasattr(out, "logits") else out[0]

    def _capture(self, run: str, items) -> torch.Tensor:
        self._check_state()
        store = self._cache[run]
        handles = []
        try:
            for site, layer in items:
                key = (site, layer)
                if site == "pattern":
                    def grab_w(module, args, out, key=key):
                        w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                        if w is None:
                            raise RuntimeError("the attention returned no weights (eager "
                                               "attention is needed)")
                        store[key] = w.detach().clone()
                    handles.append(self.arch.attention(layer).register_forward_hook(grab_w))
                else:
                    def grab(x, key=key):
                        store[key] = x.detach().clone()
                    handles.extend(register_site_hook(self.arch, site, layer, grab))
            ids = self.clean_ids if run == "clean" else self.corrupt_ids
            return self._forward(ids).detach()
        finally:
            for h in handles:
                h.remove()

    def activation(self, run: str, site: str, layer: int) -> torch.Tensor:
        """A cached activation of the ``"clean"`` or ``"corrupt"`` run (or
        ``site="pattern"``: attention weights ``[B, H, S, S]``)."""
        if run not in self._cache:
            raise ValueError("run must be 'clean' or 'corrupt'")
        site = site if site == "pattern" else canonical_site(site)
        key = (site, self.arch.layer_index(layer))
        if key not in self._cache[run]:
            self._ensure([key])
        return self._cache[run][key]

    # ── bases ──────────────────────────────────────────────────────────────

    def basis(self, basis: Union[str, torch.Tensor], layer: int,
              seed: int = 0) -> Tuple[Optional[np.ndarray], torch.Tensor]:
        """``(eigenvalues [B, S] or None, U [B, S, S])``, columns in ascending
        eigenvalue order (float64)."""
        S, B = self.seq_len, self.batch
        if torch.is_tensor(basis):
            U = basis.to(torch.float64)
            U = U.unsqueeze(0).expand(B, S, S) if U.dim() == 2 else U
            if U.shape != (B, S, S):
                raise ValueError(f"a basis tensor must be [S, S] or [B, S, S], got "
                                 f"{tuple(basis.shape)}")
            return None, U
        key = (basis, self.arch.layer_index(layer), int(seed) if basis == "random" else 0)
        if key in self._bases:
            return self._bases[key]
        if basis == "random":
            U = torch.stack([haar_basis(S, seed, b) for b in range(B)])
            result = (None, U)
        else:
            if self.config.normalization not in ("sym", "none"):
                raise ValueError("frequency-resolved patching needs a symmetric Laplacian "
                                 "(normalization 'sym' or 'none'); the 'rw' eigenbasis is not "
                                 "orthonormal")
            if basis in ("clean_attention", "corrupt_attention"):
                A = self.activation(basis.split("_")[0], "pattern", layer)
                W = attention_adjacency(A.cpu(), self.config)
            elif basis == "path":
                W = path_adjacency(S, B)
            else:
                raise ValueError(f"basis must be one of {BASES} or a tensor, got {basis!r}")
            L = laplacian(W, self.config).to(torch.float64)
            vals, vecs = torch.linalg.eigh(0.5 * (L + L.mT))
            result = (vals.numpy(), vecs)
        self._bases[key] = result
        return result

    def projector(self, spec: PatchSpec) -> Optional[torch.Tensor]:
        """``U_b U_b^T`` per batch item, ``[B, S, S]`` (None for plain
        patching)."""
        if not spec.frequency_resolved:
            return None
        lam, U = self.basis(spec.basis, spec.layer, spec.seed)
        P = []
        for b in range(self.batch):
            idx = select_modes(self.seq_len, spec.band, None if lam is None else lam[b],
                               spec.n_modes)
            Ub = U[b][:, torch.as_tensor(idx, dtype=torch.long)]
            P.append(Ub @ Ub.T)
        return torch.stack(P)

    # ── patching ───────────────────────────────────────────────────────────

    def _head_slices(self, spec: PatchSpec):
        if canonical_site(spec.site) != "z":
            return None
        H, dh = self.arch.n_heads, self.arch.head_dim(spec.layer)
        heads = range(H) if spec.head is None else (
            [spec.head] if isinstance(spec.head, (int, np.integer)) else list(spec.head))
        cols = torch.cat([torch.arange(int(h) * dh, (int(h) + 1) * dh) for h in heads])
        return cols

    def _patched_rows(self, x: torch.Tensor, rows: List[Tuple[int, int, PatchSpec]],
                      prepared) -> torch.Tensor:
        """Apply the patches of ``rows`` (``(row in x, input item, spec)``)."""
        out = x.clone()
        for r, b, spec in rows:
            clean, P, cols, pmask = prepared[id(spec)]
            xc = clean[b].to(device=x.device, dtype=x.dtype)
            cur = out[r] if cols is None else out[r][:, cols]
            src = xc if cols is None else xc[:, cols]
            if P is None:
                new = src.clone()
            else:
                Pb = P[b].to(device=x.device, dtype=torch.float64)
                delta = Pb @ (src - cur).to(torch.float64)
                new = (cur.to(torch.float64) + delta).to(x.dtype)
            if pmask is not None:
                new = torch.where(pmask.view(-1, 1).to(x.device), new, cur)
            if cols is None:
                out[r] = new
            else:
                out[r][:, cols] = new
        return out

    def _prepare(self, spec: PatchSpec):
        site = canonical_site(spec.site)
        clean = self.activation("clean", site, spec.layer)
        P = self.projector(spec)
        cols = self._head_slices(spec)
        pmask = _position_mask(spec.positions, self.seq_len)
        return clean, P, cols, pmask

    def patch_many(self, specs: Sequence[PatchSpec],
                   metric: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                   batch_size: int = 32):
        """Run many patches in batched forward passes of the corrupt input.

        Returns ``metric`` values ``[len(specs), B]`` when ``metric`` (a
        function of logits ``[B, ...]`` returning ``[B]``) is given, else
        the patched logits ``[len(specs), B, ...]``.
        """
        specs = [replace(s, layer=self.arch.layer_index(s.layer)) for s in specs]
        self._check_state()
        needed = {(canonical_site(s.site), s.layer) for s in specs}
        needed |= {("pattern", s.layer) for s in specs if s.frequency_resolved
                   and isinstance(s.basis, str) and s.basis.endswith("_attention")}
        self._ensure(needed)
        prepared = {id(s): self._prepare(s) for s in specs}
        jobs = [(j, b) for j in range(len(specs)) for b in range(self.batch)]
        outs: List[torch.Tensor] = [None] * len(jobs)
        for start in range(0, len(jobs), max(1, int(batch_size))):
            chunk = jobs[start:start + int(batch_size)]
            ids = torch.stack([self.corrupt_ids[b] for _, b in chunk])
            groups: Dict[Tuple[str, int], List] = {}
            for r, (j, b) in enumerate(chunk):
                s = specs[j]
                groups.setdefault((canonical_site(s.site), s.layer), []).append((r, b, s))
            handles = []
            try:
                for (site, layer), rows in groups.items():
                    def fn(x, rows=rows):
                        return self._patched_rows(x, rows, prepared)
                    handles.extend(register_site_hook(self.arch, site, layer, fn))
                logits = self._forward(ids)
            finally:
                for h in handles:
                    h.remove()
            for r in range(len(chunk)):
                outs[start + r] = logits[r]
        logits = torch.stack(outs).view(len(specs), self.batch, *outs[0].shape)
        if metric is None:
            return logits
        return torch.stack([torch.as_tensor(metric(logits[j])) for j in range(len(specs))])

    def patch(self, spec: Optional[PatchSpec] = None, *, metric=None, **kwargs):
        """One patch (a :class:`PatchSpec`, or its fields as keywords).
        Returns the patched logits ``[B, ...]``, or ``metric(logits)``."""
        spec = spec or PatchSpec(**kwargs)
        out = self.patch_many([spec], metric=metric, batch_size=max(1, self.batch))
        return out[0]


def patch(
    model,
    clean,
    corrupt,
    site: str = "resid_pre",
    layer: int = 0,
    head: Union[None, int, Sequence[int]] = None,
    positions: Positions = None,
    *,
    band: Band = None,
    basis: Union[str, torch.Tensor] = "clean_attention",
    n_modes: Optional[int] = None,
    seed: int = 0,
    metric: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    tokenizer=None,
    config: Optional[GSPConfig] = None,
    forward_kwargs: Optional[dict] = None,
):
    """Patch one clean activation into the corrupt run.

    Plain patching (``band=None``) replaces the activation at ``positions``
    (default all) by its clean value, as a TransformerLens patching hook
    does. With ``band``, only that band of graph frequencies along the token
    axis is taken from the clean run (see :mod:`spectral_trust.intervene.patching`):
    ``band=(lo, hi)`` integer mode indices in ascending eigenvalue order, or
    float eigenvalues; ``basis`` one of ``"clean_attention"``,
    ``"corrupt_attention"``, ``"path"``, ``"random"`` or a tensor;
    ``n_modes`` caps the band's size.

    Examples:
        >>> st.patch(model, clean, corrupt, site="resid_pre", layer=8)
        >>> st.patch(model, clean, corrupt, site="head", layer=9, head=6, positions=[-1])
        >>> st.patch(model, clean, corrupt, site="resid_pre", layer=8, band=(0, 4))

    Returns:
        Patched logits ``[B, ...]`` (``[1, ...]`` for one input), or
        ``metric(logits)``. For many patches, use :class:`Patcher`, which
        runs the clean and corrupt inputs once.

    Added in 0.5.0.
    """
    spec = PatchSpec(site=site, layer=layer, head=head, positions=positions, band=band,
                     basis=basis, n_modes=n_modes, seed=seed)
    site_c = canonical_site(site)
    sites = [site_c]
    if spec.frequency_resolved and isinstance(basis, str) and basis.endswith("_attention"):
        sites.append("pattern")
    p = Patcher(model, clean, corrupt, tokenizer=tokenizer, config=config, layers=[layer],
                sites=sites, forward_kwargs=forward_kwargs)
    return p.patch(spec, metric=metric)
