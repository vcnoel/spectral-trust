"""Fast diagnostics: exactness against brute force and dense references."""
import numpy as np
import pytest
import torch

from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
from spectral_trust.fast import (
    row_dirichlet_energy, row_value_spread,
    top_band_hfer, num_high_modes, HFER_METHODS,
    gaussian_projection, jl_project,
)


def _attention(num_heads, seq, seed=0, causal=True):
    """Softmax attention with a sink token and a locality bias."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_heads, seq, seq, generator=g, dtype=torch.float64)
    pos = torch.arange(seq, dtype=torch.float64)
    logits = logits - 0.05 * (pos[:, None] - pos[None, :]).abs()
    logits[..., 0] += 3.0
    if causal:
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _signals(seq, dim, seed=0, batch=()):
    g = torch.Generator().manual_seed(seed + 1000)
    return torch.randn(*batch, seq, dim, generator=g, dtype=torch.float64)


# ── Row-anchored energies ────────────────────────────────────────────────────

def _brute_row_energy(W, X, rows):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    return sum(W[i, j] * np.sum((X[i] - X[j]) ** 2)
               for i in rows for j in range(W.shape[1]))


def _brute_value_spread(A, V, rows):
    A = np.asarray(A, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    total = 0.0
    for i in rows:
        o = A[i] @ V
        total += sum(A[i, j] * np.sum((V[j] - o) ** 2) for j in range(A.shape[1]))
    return total


def test_row_dirichlet_energy_matches_brute_force_batched():
    A = _attention(3, 17, seed=1)
    X = _signals(17, 5, seed=1)
    rows = [0, 4, 16]
    got = row_dirichlet_energy(A, X, rows=rows)
    assert got.shape == (3,)
    for h in range(3):
        want = _brute_row_energy(A[h], X, rows)
        assert abs(float(got[h]) - want) <= 1e-10 * max(1.0, want)
    # per-row output and the boolean-mask form agree with the index form
    per_row = row_dirichlet_energy(A, X, rows=rows, reduction="none")
    assert per_row.shape == (3, 3)
    mask = torch.zeros(17, dtype=torch.bool)
    mask[rows] = True
    assert torch.allclose(row_dirichlet_energy(A, X, rows=mask), got, rtol=1e-12)
    assert torch.allclose(per_row.sum(-1), got, rtol=1e-12)


def test_row_dirichlet_energy_per_head_signals_and_float32():
    A = _attention(2, 12, seed=2)
    X = _signals(12, 4, seed=2, batch=(2,))
    got = row_dirichlet_energy(A.float(), X.float(), rows=slice(3, 9))
    for h in range(2):
        want = _brute_row_energy(A[h], X[h], range(3, 9))
        assert abs(float(got[h]) - want) <= 1e-5 * want


@pytest.mark.parametrize("norm", ["none", "sym", "rw"])
def test_row_dirichlet_energy_relates_to_library_energy(norm):
    """The documented identities between the row-anchored energy summed over
    all rows and tr(X^T L X) for each of the library's normalisations."""
    attn = _attention(1, 20, seed=3)[0]
    W = 0.5 * (attn + attn.T)
    W[7, :] = 0.0                       # one isolated token
    W[:, 7] = 0.0
    X = _signals(20, 6, seed=3)
    cfg = GSPConfig(device="cpu", normalization=norm)
    L = GraphConstructor(cfg).construct_laplacian(W.unsqueeze(0))[0].double()
    energy = SpectralAnalyzer(cfg).compute_dirichlet_energy(X.numpy(), L.numpy())
    deg = W.sum(-1)
    isolated = deg <= 1e-8
    if norm == "none":
        want = 0.5 * float(row_dirichlet_energy(W, X))
    elif norm == "sym":
        inv_sqrt = torch.where(isolated, torch.zeros_like(deg),
                               deg.clamp(min=1e-8).rsqrt())
        want = (0.5 * float(row_dirichlet_energy(W, inv_sqrt[:, None] * X))
                + float((X[isolated] ** 2).sum()))
    else:
        inv = torch.where(isolated, torch.zeros_like(deg), 1.0 / deg.clamp(min=1e-8))
        P = inv[:, None] * W
        col = P.sum(0)
        sq = (X ** 2).sum(-1)
        # (2 - r_j - c_j)/2 with r_j = 1 for connected and 0 for isolated tokens
        weight = torch.where(isolated, torch.ones_like(col), 0.5 * (1.0 - col))
        want = 0.5 * float(row_dirichlet_energy(P, X)) + float((weight * sq).sum())
    # construct_laplacian works in float32, so L carries ~1e-7 relative error
    assert abs(energy - want) <= 1e-6 * max(1.0, abs(energy))


def test_row_value_spread_matches_brute_force_batched():
    A = _attention(4, 15, seed=4)
    V = _signals(15, 8, seed=4, batch=(4,))
    rows = [1, 2, 14]
    got = row_value_spread(A, V, rows=rows)
    assert got.shape == (4,)
    for h in range(4):
        want = _brute_value_spread(A[h], V[h], rows)
        assert abs(float(got[h]) - want) <= 1e-10 * max(1.0, want)
    # softmax rows: the spread is the attention-weighted variance
    o = A @ V
    var = (A @ (V ** 2).sum(-1, keepdim=True)).squeeze(-1) - (o ** 2).sum(-1)
    per_row = row_value_spread(A, V, reduction="none")
    assert torch.allclose(per_row, var.clamp(min=0), atol=1e-10)


def test_row_value_spread_one_hot_attention_is_zero():
    S = 9
    A = torch.eye(S, dtype=torch.float64).unsqueeze(0)
    V = _signals(S, 3, seed=5, batch=(1,))
    assert float(row_value_spread(A, V)) == pytest.approx(0.0, abs=1e-12)


def test_row_selection_validation():
    A = _attention(1, 6)
    X = _signals(6, 2)
    with pytest.raises(IndexError):
        row_dirichlet_energy(A, X, rows=[6])
    with pytest.raises(ValueError):
        row_dirichlet_energy(A, X, rows=torch.ones(5, dtype=torch.bool))
    with pytest.raises(ValueError):
        row_value_spread(A, X, reduction="mean")
    assert torch.allclose(row_dirichlet_energy(A, X, rows=[-1]),
                          row_dirichlet_energy(A, X, rows=[5]))


# ── Top-band HFER ────────────────────────────────────────────────────────────

def _sym_laplacian(seq, seed=0, norm="sym"):
    attn = _attention(1, seq, seed=seed)
    cfg = GSPConfig(device="cpu", normalization=norm)
    gc = GraphConstructor(cfg)
    W = gc.symmetrize_attention(attn.unsqueeze(0))[0]
    return gc.construct_laplacian(W)[0].double().numpy(), cfg


def _dense_hfer(L, X, ratio):
    cfg = GSPConfig(device="cpu", eigen_solver="dense", hfer_cutoff_ratio=ratio)
    an = SpectralAnalyzer(cfg)
    vals, vecs = an.compute_eigendecomposition(L)
    return an.compute_hfer(X, vecs, vals, ratio)


@pytest.mark.parametrize("seq", [40, 97, 300])
@pytest.mark.parametrize("ratio", [0.1, 0.05, 0.3])
def test_top_band_hfer_matches_dense_float64(seq, ratio):
    L, _ = _sym_laplacian(seq, seed=seq)
    X = _signals(seq, 64, seed=seq).numpy()
    X[:, 0] += 5.0                      # a dominant direction, as in residual streams
    want = _dense_hfer(L, X, ratio)
    for method in HFER_METHODS:
        got = top_band_hfer(L, X, cutoff_ratio=ratio, method=method)
        assert abs(got - want) <= 1e-6, (method, got, want)
        assert abs(got - want) <= 1e-10, (method, got, want)


@pytest.mark.parametrize("norm", ["sym", "none"])
def test_top_band_hfer_matches_analyze_layer_float32(norm):
    """Against the library's own dense hybrid path, in its float32 precision."""
    seq = 256
    L, cfg = _sym_laplacian(seq, seed=7, norm=norm)
    X = torch.from_numpy(_signals(seq, 128, seed=7).numpy()).float()
    dense_cfg = GSPConfig(device="cpu", normalization=norm, eigen_solver="dense")
    diag = SpectralAnalyzer(dense_cfg).analyze_layer(X, torch.from_numpy(L).float(), 0)
    got = top_band_hfer(torch.from_numpy(L).float(), X)
    assert abs(got - diag.hfer) <= 1e-6, (got, diag.hfer)


def test_top_band_hfer_uses_the_dense_cutoff_rule():
    # (1 - 0.3) * 90 rounds to 62.999...; the dense path keeps 28 high modes
    assert num_high_modes(90, 0.3) == 28
    assert num_high_modes(128, 0.1) == 13
    assert num_high_modes(50, 0.1) == 5


def test_top_band_hfer_rejects_rw_and_handles_zero_signal():
    L, _ = _sym_laplacian(30, norm="rw")
    with pytest.raises(ValueError):
        top_band_hfer(L, np.ones((30, 4)))
    L, _ = _sym_laplacian(30)
    assert top_band_hfer(L, np.zeros((30, 4))) == 0.0
    with pytest.raises(ValueError):
        top_band_hfer(L, np.ones((30, 4)), method="lanczos")


# ── Johnson-Lindenstrauss projection ─────────────────────────────────────────

def _structured_signals(seq, dim, seed):
    """Residual-stream-like: a large shared direction plus a decaying spectrum."""
    rng = np.random.default_rng(seed)
    scales = 1.0 / np.arange(1, dim + 1)
    X = rng.standard_normal((seq, dim)) * scales
    X[:, 0] += 3.0
    return X


def _jl_trials(L64, X, k, seeds):
    cfg = dict(device="cpu", eigen_solver="dense", project_dim=k)
    out = {m: [] for m in ("energy", "smoothness_index", "hfer", "spectral_entropy")}
    for s in seeds:
        an = SpectralAnalyzer(GSPConfig(project_seed=s, **cfg))
        d = an.analyze_layer(torch.from_numpy(X), torch.from_numpy(L64), 0)
        for m in out:
            out[m].append(getattr(d, m))
    return {m: np.array(v) for m, v in out.items()}


def _lm_bounds(A, k, t):
    """Laurent-Massart bounds on tr(R^T A R)/tr(A) - 1 at level exp(-t) per side."""
    w = np.clip(np.linalg.eigvalsh(A), 0, None)
    w = w / w.sum()
    upper = 2 * np.sqrt((w ** 2).sum() * t / k) + 2 * w.max() * t / k
    lower = 2 * np.sqrt((w ** 2).sum() * t / k)
    return lower, upper


def test_projection_plumbing_is_exact():
    seq, dim, k = 70, 96, 16
    L, _ = _sym_laplacian(seq, seed=11)
    X = _structured_signals(seq, dim, 11).astype(np.float32)
    proj = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense",
                                      project_dim=k, project_seed=5))
    plain = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense"))
    Lt = torch.from_numpy(L).float()
    d_proj = proj.analyze_layer(torch.from_numpy(X), Lt, 0)
    XR = X @ gaussian_projection(dim, k, seed=5)
    d_ref = plain.analyze_layer(torch.from_numpy(XR), Lt, 0)
    for m in ("energy", "smoothness_index", "hfer", "spectral_entropy"):
        assert getattr(d_proj, m) == pytest.approx(getattr(d_ref, m), rel=1e-6)
    # attention-family metrics do not read the signals
    d_plain = plain.analyze_layer(torch.from_numpy(X), Lt, 0)
    assert d_proj.fiedler_value == d_plain.fiedler_value
    np.testing.assert_array_equal(d_proj.eigenvalues, d_plain.eigenvalues)
    # a width >= d is the identity
    wide = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense",
                                      project_dim=dim))
    assert wide.analyze_layer(torch.from_numpy(X), Lt, 0).energy == d_plain.energy
    # nested draws, and the torch path uses the same matrix
    R8 = gaussian_projection(dim, 8, seed=5, dtype=np.float64)
    R16 = gaussian_projection(dim, 16, seed=5, dtype=np.float64)
    np.testing.assert_allclose(R16[:, :8] * np.sqrt(16 / 8), R8, rtol=1e-12)
    Xt = torch.from_numpy(X).double()
    np.testing.assert_allclose(jl_project(Xt, 16, seed=5).numpy(),
                               X.astype(np.float64) @ R16, rtol=1e-10)


def test_jl_energy_is_unbiased_with_the_predicted_spread():
    seq, dim, k, n = 64, 256, 64, 400
    L, _ = _sym_laplacian(seq, seed=12)
    X = _structured_signals(seq, dim, 12)
    A = X.T @ L @ X
    exact = float(np.trace(A))
    trials = _jl_trials(L, X, k, range(n))
    err = trials["energy"] / exact - 1.0
    predicted_sd = np.sqrt(2.0 * np.sum(A * A) / k) / exact
    assert predicted_sd <= np.sqrt(2.0 / k) + 1e-12
    # unbiased: mean within 4 standard errors of zero
    assert abs(err.mean()) <= 4 * predicted_sd / np.sqrt(n)
    # spread: sample sd within 15% of the closed form (SE of an sd ~ 4% here)
    assert abs(err.std(ddof=1) / predicted_sd - 1.0) <= 0.15
    # tail: Laurent-Massart bound at delta = 0.05 holds empirically
    delta = 0.05
    lower, upper = _lm_bounds(A, k, np.log(2 / delta))
    outside = np.mean((err > upper) | (err < -lower))
    assert outside <= delta + 3 * np.sqrt(delta * (1 - delta) / n)


@pytest.mark.parametrize("metric", ["smoothness_index", "hfer"])
def test_jl_ratio_metrics_follow_the_delta_method_and_union_bound(metric):
    seq, dim, k, n = 64, 256, 64, 400
    L, _ = _sym_laplacian(seq, seed=13)
    X = _structured_signals(seq, dim, 13)
    vals, vecs = np.linalg.eigh(L)
    if metric == "smoothness_index":
        num = X.T @ L @ X
    else:
        U_high = vecs[:, seq - num_high_modes(seq, 0.1):]
        num = X.T @ U_high @ U_high.T @ X
    den = X.T @ X
    exact = np.trace(num) / np.trace(den)
    err = _jl_trials(L, X, k, range(n))[metric] / exact - 1.0
    # delta method: Var(Q_num - Q_den), Var(Q_A) = 2 tr(A^2)/(k tr(A)^2)
    tn, td = np.trace(num), np.trace(den)
    var = (2.0 / k) * (np.sum(num * num) / tn ** 2 + np.sum(den * den) / td ** 2
                       - 2.0 * np.sum(num * den) / (tn * td))
    # first-order prediction; second-order terms add ~10% at k = 64
    sd_ratio = err.std(ddof=1) / np.sqrt(var)
    assert 0.85 <= sd_ratio <= 1.3, sd_ratio
    # union bound: numerator and denominator each inside their delta/2 band
    delta = 0.05
    t = np.log(4 / delta)
    ln, un = _lm_bounds(num, k, t)
    ld, ud = _lm_bounds(den, k, t)
    lo, hi = (1 - ln) / (1 + ud) - 1.0, (1 + un) / (1 - ld) - 1.0
    outside = np.mean((err < lo) | (err > hi))
    assert outside <= delta + 3 * np.sqrt(delta * (1 - delta) / n)


def test_jl_entropy_error_shrinks_with_width():
    seq, dim = 64, 256
    L, _ = _sym_laplacian(seq, seed=14)
    X = _structured_signals(seq, dim, 14)
    exact = SpectralAnalyzer(GSPConfig(device="cpu", eigen_solver="dense")) \
        .analyze_layer(torch.from_numpy(X), torch.from_numpy(L), 0).spectral_entropy
    med = {}
    for k in (16, 128):
        ent = _jl_trials(L, X, k, range(100))["spectral_entropy"]
        med[k] = np.median(np.abs(ent / exact - 1.0))
    assert med[128] < med[16] / 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_row_energies_on_cuda():  # pragma: no cover - CPU-only CI
    A = _attention(2, 10).cuda()
    X = _signals(10, 4).cuda()
    got = row_dirichlet_energy(A, X, rows=[0, 3])
    assert got.device.type == "cuda"
    want = row_dirichlet_energy(A.cpu(), X.cpu(), rows=[0, 3])
    assert torch.allclose(got.cpu(), want, rtol=1e-10)
