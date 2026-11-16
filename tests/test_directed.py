# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""spectral_trust.directed: the row-normalised operator, the directed total
variation, the magnetic Laplacian and Hermitian (flow-imbalance) clustering."""
import itertools

import numpy as np
import pytest
import torch

import spectral_trust as st
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
from spectral_trust.directed import analyze_directed_layer


def _causal(S, seed, sink=0.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(S, S, generator=g, dtype=torch.float64).tril()
    a[:, 0] += sink * a.sum(-1)
    return a / a.sum(-1, keepdim=True)


def _signal(S, d, seed, complex_=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(S, d, generator=g, dtype=torch.float64)
    if complex_:
        x = x + 1j * torch.randn(S, d, generator=g, dtype=torch.float64)
    return x


# ── the operator and the transport measure ──────────────────────────────────

def test_directed_operator_is_row_stochastic_and_keeps_empty_rows():
    A = torch.tensor([[0.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, -3.0, 3.0]])
    P = st.directed_operator(A)
    assert torch.allclose(P[0], torch.tensor([0.0, 0.5, 0.5]))
    assert torch.equal(P[1], torch.zeros(3))                       # no weight: stays zero
    assert torch.allclose(P[2], torch.tensor([0.25, 0.0, 0.75]))   # negatives clipped
    P2 = st.directed_operator(A, remove_self_loops=True)
    assert torch.allclose(P2[2], torch.tensor([1.0, 0.0, 0.0]))


def test_directed_operator_of_attention_is_the_attention():
    a = _causal(12, 0, sink=2.0)
    assert torch.allclose(st.directed_operator(a), a, atol=1e-15)


def test_directed_tv_definition_and_reductions():
    P = _causal(10, 1)
    X = _signal(10, 6, 2)
    total = st.directed_total_variation(X, P)
    R = X - P @ X
    assert float(total) == pytest.approx(float((R ** 2).sum() / (X ** 2).sum()), rel=1e-12)
    cols = st.directed_total_variation(X, P, per_column=True)
    for c in range(6):
        x = X[:, c]
        ref = float(((x - P @ x) ** 2).sum() / (x ** 2).sum())
        assert float(cols[c]) == pytest.approx(ref, rel=1e-12)
    # a quadratic form of (I - P)^T (I - P)
    M = (torch.eye(10, dtype=torch.float64) - P)
    q = float(X[:, 0] @ (M.T @ M) @ X[:, 0] / (X[:, 0] @ X[:, 0]))
    assert float(cols[0]) == pytest.approx(q, rel=1e-10)


def test_directed_tv_is_zero_on_constants_and_on_vanishing_signals():
    P = _causal(9, 3)
    ones = torch.ones(9, 2, dtype=torch.float64)
    assert float(st.directed_total_variation(ones, P)) < 1e-24
    tiny = torch.zeros(9, 3, dtype=torch.float64)
    tiny[:, 1] = 1e-6
    assert torch.equal(st.directed_total_variation(tiny, P, per_column=True),
                       torch.zeros(3, dtype=torch.float64))


def test_quadratic_energy_is_orientation_blind_and_directed_tv_is_not():
    """x^T L x reads only the symmetric part of L: I - P and I - P^T give
    the same energy. The transport measure tells P from its reverse."""
    P = _causal(14, 4, sink=1.0)
    Prev = st.directed_operator(P.T)                 # every edge reversed
    I = torch.eye(14, dtype=torch.float64)
    x = _signal(14, 1, 5)[:, 0]
    assert float(x @ (I - P) @ x) == pytest.approx(float(x @ (I - P.T) @ x), rel=1e-12)
    tv, tv_rev = st.directed_total_variation(x, P), st.directed_total_variation(x, Prev)
    assert abs(float(tv) - float(tv_rev)) > 0.05 * float(tv)


# ── the magnetic Laplacian ──────────────────────────────────────────────────

@pytest.mark.parametrize("normalized", [True, False])
def test_magnetic_laplacian_is_hermitian(normalized):
    A = torch.rand(11, 11, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    L = st.magnetic_laplacian(A, q=0.25, normalized=normalized)
    assert L.dtype == torch.complex128
    assert torch.equal(L, L.conj().transpose(0, 1))
    assert float(L.imag.abs().max()) > 1e-3                    # the phases are there


@pytest.mark.parametrize("norm", ["sym", "none"])
def test_magnetic_laplacian_at_q0_is_the_symmetric_laplacian(norm):
    A = _causal(13, 7, sink=1.5)
    L0 = st.magnetic_laplacian(A, q=0.0, normalized=(norm == "sym"))
    assert torch.equal(L0.imag, torch.zeros_like(L0.imag))
    gc = GraphConstructor(GSPConfig(device="cpu", normalization=norm))
    ref = gc.construct_laplacian((0.5 * (A + A.T)).unsqueeze(0))[0].double()  # float32 path
    assert torch.allclose(L0.real, ref, atol=2e-6)
    # and through the GraphConstructor method, which reads the config
    assert torch.equal(gc.magnetic_laplacian(A, q=0.0), L0)


def test_normalized_magnetic_eigenvalues_lie_in_0_2():
    g = torch.Generator().manual_seed(8)
    for trial in range(5):
        A = torch.rand(15, 15, generator=g, dtype=torch.float64)
        A = A * (torch.rand(15, 15, generator=g, dtype=torch.float64) < 0.4)
        A[3] = 0.0
        A[:, 3] = 0.0                                          # an isolated node
        for q in (0.05, 0.25, 0.5):
            lam, U = st.magnetic_spectrum(A, q=q)
            assert float(lam.min()) >= -1e-12 and float(lam.max()) <= 2 + 1e-12
            assert torch.allclose(U.conj().T @ U, torch.eye(15, dtype=torch.complex128),
                                  atol=1e-10)
        lam_u, _ = st.magnetic_spectrum(A, q=0.25, normalized=False)
        assert float(lam_u.min()) >= -1e-12                     # positive semidefinite


def test_magnetic_laplacian_complex64():
    A = _causal(6, 9)
    L = st.magnetic_laplacian(A, dtype=torch.complex64)
    assert L.dtype == torch.complex64
    with pytest.raises(ValueError):
        st.magnetic_laplacian(A, dtype=torch.float32)
    with pytest.raises(ValueError):
        st.magnetic_laplacian(A, q=-0.1)


def test_magnetic_hfer_is_orientation_blind_for_real_signals_only():
    """Transposing the graph conjugates L, so |U^H x|^2 is unchanged for a
    real x: the magnetic HFER cannot tell A from A^T. A complex signal can."""
    A = _causal(16, 10, sink=2.0)
    x = _signal(16, 4, 11)
    for q in (0.1, 0.25):
        h = st.magnetic_hfer(x, A, q=q, cutoff_ratio=0.3)
        h_t = st.magnetic_hfer(x, A.T, q=q, cutoff_ratio=0.3)
        assert float(h) == pytest.approx(float(h_t), rel=1e-10, abs=1e-14)
        cols, cols_t = (st.magnetic_hfer(x, B, q=q, cutoff_ratio=0.3, per_column=True)
                        for B in (A, A.T))
        assert torch.allclose(cols, cols_t, rtol=1e-10, atol=1e-14)
    z = _signal(16, 4, 12, complex_=True)
    hz = st.magnetic_hfer(z, A, q=0.25, cutoff_ratio=0.3)
    hz_t = st.magnetic_hfer(z, A.T, q=0.25, cutoff_ratio=0.3)
    assert abs(float(hz) - float(hz_t)) > 1e-3


def test_magnetic_hfer_at_q0_matches_the_symmetric_hfer():
    A = _causal(12, 13, sink=1.0)
    x = _signal(12, 5, 14)
    cfg = GSPConfig(device="cpu", hfer_cutoff_ratio=0.25)
    L = GraphConstructor(cfg).construct_laplacian((0.5 * (A + A.T)).unsqueeze(0))[0]
    ref = st.per_column_metrics(x, L, cfg).metric("hfer")
    got = st.magnetic_hfer(x, A, q=0.0, cutoff_ratio=0.25, per_column=True)
    assert np.allclose(got.numpy(), ref, atol=1e-5)


def test_magnetic_phase_orders_a_chain():
    n = 8
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(1, n):
        A[i, i - 1] = 1.0                                      # token i reads token i - 1
    q = 0.05
    ph = st.magnetic_phase(A, q=q)
    step = np.diff(ph.numpy())
    assert np.all(step > 0)                                    # grows along the flow
    assert np.allclose(step, 2 * np.pi * q, atol=1e-8)          # phi_i - phi_j = Theta_ij
    assert float(st.magnetic_phase(A, q=0.0).abs().max()) < 1e-8  # trivial at q = 0
    # the largest-modulus entry is the gauge reference
    _, U = st.magnetic_spectrum(A, q=q)
    ref = int(torch.argmax(U[:, 0].abs()))
    assert abs(float(ph[ref])) < 1e-12


# ── Hermitian clustering ────────────────────────────────────────────────────

def _dsbm(k, m, p, eta, seed, cyclic=False):
    """Directed block model: one density p everywhere; within a block each
    edge points either way, between blocks it follows the meta-graph with
    probability eta. A[i, j] = 1 when i reads j (j -> i)."""
    rng = np.random.default_rng(seed)
    n = k * m
    lab = np.repeat(np.arange(k), m)
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(iu.size) < p
    iu, ju = iu[keep], ju[keep]
    coin = rng.random(iu.size)
    meta = (lab[ju] - lab[iu]) % k == 1 if cyclic else lab[iu] < lab[ju]
    fwd = np.where(lab[iu] == lab[ju], coin < 0.5, (coin < eta) == meta)
    A = np.zeros((n, n))
    A[ju[fwd], iu[fwd]] = 1.0                                  # iu -> ju
    A[iu[~fwd], ju[~fwd]] = 1.0                                # ju -> iu
    return A, lab


def _same_partition(a, b) -> bool:
    return len(set(zip(a.tolist(), b.tolist()))) == len(set(b.tolist())) == len(set(a.tolist()))


@pytest.mark.parametrize("k,cyclic", [(3, False), (3, True), (4, True)])
def test_hermitian_clustering_recovers_planted_blocks(k, cyclic):
    for seed in range(3):
        A, lab = _dsbm(k, 50, 0.5, 0.95, seed, cyclic=cyclic)
        fc = st.hermitian_clustering(A, k, seed=seed)
        assert _same_partition(fc.labels, lab)
        assert np.array_equal(fc.labels, lab)                   # numbered by first appearance
        assert fc.embedding.shape == (k * 50, 4)
        assert np.all(np.diff(np.abs(fc.eigenvalues)) <= 1e-9)


def test_symmetrised_spectral_clustering_cannot_see_these_blocks():
    """Same density inside and between blocks: only the direction differs."""
    from scipy.cluster.vq import kmeans2
    k = 3
    A, lab = _dsbm(k, 50, 0.5, 0.95, 0)
    W = A + A.T
    d = W.sum(1)
    _, U = np.linalg.eigh(np.eye(len(d)) - W / np.sqrt(np.outer(d, d)))
    _, sym = kmeans2(U[:, :k], k, minit="++", seed=np.random.default_rng(0))
    acc = max(np.mean(np.array(p)[sym] == lab) for p in itertools.permutations(range(k)))
    assert acc < 0.6
    assert np.array_equal(st.hermitian_clustering(A, k).labels, lab)


def test_flow_imbalance_signs_follow_the_planted_order():
    A, lab = _dsbm(3, 50, 0.5, 0.95, 1)
    fc = st.hermitian_clustering(A, 3)
    F = fc.imbalance
    assert np.allclose(F, -F.T)
    assert np.all(np.abs(F) <= 0.5)
    # block a reads block b (b -> a) when b < a, so a reads more from b than b from a
    for a in range(3):
        for b in range(3):
            if a > b:
                assert F[a, b] > 0.35
    flow, imb = st.flow_imbalance(A, lab)
    assert np.allclose(imb, F) and np.allclose(flow, fc.flow)
    assert flow.sum() == pytest.approx(A.sum() - np.trace(A))


def test_hermitian_clustering_normalized_and_edge_cases():
    A, lab = _dsbm(3, 50, 0.5, 0.95, 2)
    assert _same_partition(st.hermitian_clustering(A, 3, normalized=True).labels, lab)
    one = st.hermitian_clustering(A, 1)
    assert np.all(one.labels == 0) and one.flow.shape == (1, 1)
    with pytest.raises(ValueError):
        st.hermitian_clustering(A, 0)
    assert set(st.hermitian_clustering(A, 3).to_dict()) == {"labels", "flow", "imbalance",
                                                             "eigenvalues"}


# ── per-column variant and layer diagnostics ────────────────────────────────

def test_per_column_metrics_with_operator_adds_directed_tv():
    a = _causal(10, 15, sink=1.0).float()
    x = _signal(10, 7, 16).float()
    cfg = GSPConfig(device="cpu")
    gc = GraphConstructor(cfg)
    L = gc.construct_laplacian(gc.aggregate_heads(gc.symmetrize_attention(a[None, None])))[0]
    plain = st.per_column_metrics(x, L, cfg)
    both = st.per_column_metrics(x, L, cfg, operator=st.directed_operator(a))
    assert plain.metric_names == st.COLUMN_METRIC_NAMES
    assert both.metric_names == st.COLUMN_METRIC_NAMES + ["directed_tv"]
    assert np.array_equal(both.values[:, :4], plain.values)
    ref = st.directed_total_variation(x, st.directed_operator(a), per_column=True).numpy()
    assert np.allclose(both.metric("directed_tv"), ref, rtol=1e-6)
    with pytest.raises(ValueError):
        st.per_column_metrics(x, L, cfg, operator=torch.eye(4))


def test_analyze_directed_layer_at_q0_matches_analyze_layer():
    P = _causal(12, 17, sink=1.0)
    x = _signal(12, 6, 18)
    cfg = GSPConfig(device="cpu", magnetic_q=0.0)
    diag, extra = analyze_directed_layer(x, P, cfg, layer=4)
    gc = GraphConstructor(cfg)
    L = gc.construct_laplacian((0.5 * (P + P.T)).unsqueeze(0))[0]
    ref = SpectralAnalyzer(cfg).analyze_layer(x, L, 4)
    for name in ("energy", "smoothness_index", "spectral_entropy", "hfer", "fiedler_value",
                 "spectral_radius", "eig_energy", "eig_hfer"):
        assert getattr(diag, name) == pytest.approx(getattr(ref, name), rel=1e-4, abs=1e-6), name
    assert diag.connectivity == ref.connectivity and diag.layer == 4
    assert extra["directed_tv"] == pytest.approx(float(st.directed_total_variation(x, P)),
                                                 rel=1e-6)
    assert extra["phase"].shape == (12,)
    with pytest.raises(ValueError, match="Hermitian"):
        analyze_directed_layer(x, P, GSPConfig(device="cpu", normalization="rw"))
