# What a symmetrised spectrum cannot see

By default spectral-trust builds an undirected graph: each head's attention
`A` is symmetrised, `(A + Aᵀ)/2`, and its Laplacian spectrum is read. Recent
work names four things that this spectrum misses or confounds. Each section
below states the point, and ends with the call that measures it. The
functions are new in 0.5.1, in `spectral_trust.features` and
`spectral_trust.pseudospectra`.

## Orientation

Dahlem, Maniloff and Misiura (arXiv 2605.04893) read a head as a transport
operator, `M = D_Q^{-1/2} A D_K^{-1/2}`, and split it into a symmetric and an
antisymmetric part. Every diagnostic that is unchanged when `M` is transposed
(eigenvalues of a symmetrised graph, singular values, spectral entropy, the
spectral gap) depends on the symmetric part and on the norm of the
antisymmetric part, never on its sign. Such a diagnostic cannot tell
attention from its transpose, so it cannot tell which way information flows.
Their asymmetry coefficient `G = ‖M_asym‖_F / ‖M‖_F` measures how much
orientation there is: 0 for a symmetric head, at most `1/√2` for a causal one,
near 0 for a head that attends only to itself. Their conductance estimate
`φ̂` measures the transport capacity.

```python
from spectral_trust import features

G = features.asymmetry(attn)            # [..., heads], their eq. 7
phi = features.capacity_phi(attn)       # spectral-sweep conductance
s2 = features.transport_sigma2(attn)    # Cheeger: (1 - s2)/2 <= phi
```

The authors have released no code, so these follow the paper's equations and
are not checked against an original. A directed graph keeps the orientation:
see `GSPConfig.directed` and `spectral_trust.DirectedTopologist`.

## Non-normality

Causal attention is lower-triangular, so it is non-normal: its eigenvalues
are its diagonal, and they describe only the limit of its powers. The
behaviour at finite depth is described by the resolvent, through the
ε-pseudospectra `Λ_ε(A) = {z : σ_min(zI − A) < ε}` and the Kreiss constant
`K(A) = sup_{|z|>1} (|z| − 1) ‖(zI − A)⁻¹‖`, which brackets the transient,
`K <= sup_k ‖A^k‖ <= e·n·K` (Trefethen and Embree 2005). Li (arXiv
2607.09279 and 2607.06621) shows that the mask alone pins the Kreiss
constant of any causal row-stochastic matrix at `√n`. He also shows that,
over the depth products of GPT-2, Pythia and Llama-3-8B, the eigenvalue
prediction of the surviving deviation is too small by seven to eleven orders
of magnitude.

```python
from spectral_trust import pseudospectra as ps

ps.pseudospectrum(A, grid=(re, im), eps=[1e-1, 1e-2])
ps.pseudospectral_radius(A, 1e-2), ps.pseudospectral_abscissa(A, 1e-2)
ps.kreiss_constant(A)                   # sqrt(n) for raw causal attention
ps.departure_from_normality(A)          # Henrici, exact
ps.numerical_range(A).radius
ps.nonnormality(stack, depth=True, deflate=True)   # the depth propagator
```

`σ_min` is computed with triangular solves, O(n²) per grid point and
iteration, because causal attention is already triangular; other matrices
are reduced to Schur form first. `benchmarks/li_depth_check.py` reproduces
Li's two propositions and his depth-product gap on GPT-2 (the eigenvalue
heuristic falls short of the measured deviation by a factor of 8.6·10⁶,
against his 2.9·10⁷).

## Sink bias

Most heads send much of their attention to the first token, the attention
sink. Viganò, Melzi and Ovsjanikov (arXiv 2610.06012) show that Euclidean
spectra of attention are biased by sinks: a large eigenvalue can come from
mass piled into one column rather than from routing among the other tokens.
Their remedy is a spectrum in the probability geometry that attention
induces (spectral-trust's measure-aware operator spectra follow it). A
simpler check is to drop the sink and renormalise: each query then spreads
unit mass over the remaining tokens, which is how Bao et al. (2026) build
their graphs. `GSPConfig.subgraph_indices` and `token_span` drop nodes
without renormalising; `subgraph_renormalize=True` renormalises them.

```python
import spectral_trust as st
from spectral_trust.graph import drop_and_renormalize

cfg = st.GSPConfig(subgraph_renormalize=True)
cache = st.SpectralModel.from_pretrained(name, config=cfg).run_with_spectra(
    text, token_span=(1, None))                    # BOS dropped, rows renormalised
A_wo_sink = drop_and_renormalize(attn, drop=0)     # the same on a tensor
```

A metric that moves a lot between the two runs was reading the sink.

## Sink entanglement

Binkowski, Adamczewski and Kajdanowicz (SinkProbe, arXiv 2604.10697) find
that hallucination detection from attention is entangled with the sinks.
Their probe relies most on sinks whose value vectors have large norms, and
earlier attention-map detectors, LapEigvals among them, implicitly depend on
sink scores (see [Baselines](baselines.md)). A spectral feature is then
informative only if it adds something beyond the sink. Report the sink
next to it, weighted by what the sink writes. A head can pour most of its
mass into a sink whose value vector is near zero, which is a no-op.

```python
from spectral_trust import features

features.sink_mass(attn)                               # [..., heads]
features.value_weighted_sink(attn, values, W_O)        # ‖W_O^h v^h_sink‖ · mass
features.effective_rank(hidden)                        # Roy and Vetterli
features.attention_covariates(attn, values, W_O)       # all per-head covariates
```

`attention_covariates` returns sink mass, the value-weighted sink, diagonal
and previous-token mass, row entropy, the effective rank of each map, the
asymmetry `G` and the length. Use them as covariates, so that a reported
effect of a spectral feature is its effect beyond these.
