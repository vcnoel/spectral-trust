# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Measure-aware operator spectra (spectral_trust.operator_spectra), checked
against the equations of Viganò, Melzi and Ovsjanikov (NeurIPS 2026)."""
import numpy as np
import pytest
import torch

from spectral_trust.operator_spectra import (
    attention_operator, operator_measures, operator_metrics, operator_spectrum,
    participation_ratio,
)

torch.set_num_threads(2)


def causal_attention(n, heads=None, seed=0, temp=1.0):
    g = torch.Generator().manual_seed(seed)
    shape = (n, n) if heads is None else (heads, n, n)
    logits = torch.randn(shape, generator=g, dtype=torch.float64) / temp
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def sink_biased(n, s):
    """Mass s on the first token, the rest on the diagonal."""
    return s * torch.ones(n, 1, dtype=torch.float64) * torch.eye(n, dtype=torch.float64)[0] \
        + (1 - s) * torch.eye(n, dtype=torch.float64)


MEASURES = ["pullback", "counting"]


@pytest.mark.parametrize("measure", MEASURES + ["custom"])
def test_difference_operator_is_self_adjoint_in_L2_mu(measure):
    A = causal_attention(9, seed=1)
    if measure == "custom":
        g = torch.Generator().manual_seed(5)
        measure = (torch.rand(9, generator=g, dtype=torch.float64) + 0.1,
                   torch.rand(9, generator=g, dtype=torch.float64) + 0.1)
    S, D = attention_operator(A, measure, return_difference=True)
    mu, _ = operator_measures(A, measure)
    M = torch.diag(mu) @ D                                   # Ω_μ D
    assert torch.allclose(M, M.T, atol=1e-13)
    g = torch.Generator().manual_seed(3)
    f, h = torch.randn(9, generator=g, dtype=torch.float64), torch.randn(
        9, generator=g, dtype=torch.float64)
    left, right = (D @ f) @ (mu * h), f @ (mu * (D @ h))       # ⟨Df, h⟩_μ, ⟨f, Dh⟩_μ
    assert abs(float(left - right)) < 1e-12
    assert torch.equal(S, S.T)


@pytest.mark.parametrize("measure", MEASURES)
def test_D_and_S_have_the_same_eigenvalues(measure):
    A = causal_attention(12, seed=2)
    S, D = attention_operator(A, measure, return_difference=True)
    eig_d = torch.sort(torch.linalg.eigvals(D).real, descending=True).values
    eig_s = torch.sort(torch.linalg.eigvalsh(S), descending=True).values
    assert torch.allclose(eig_d, eig_s, atol=1e-10)
    assert torch.linalg.eigvals(D).imag.abs().max() < 1e-10


def test_pullback_top_eigenvalue_is_one_with_a_constant_eigenfunction():
    for seed in range(4):
        A = causal_attention(15, seed=seed, temp=0.5)
        spec = operator_spectrum(A)
        assert abs(float(spec.eigenvalues[0]) - 1.0) < 1e-12
        phi = spec.eigenfunctions[:, 0]
        assert torch.allclose(phi / phi[0], torch.ones(15, dtype=torch.float64), atol=1e-9)
        assert abs(float((spec.mu * phi * phi).sum()) - 1.0) < 1e-12   # unit μ-norm
        S, D = attention_operator(A, return_difference=True)
        one = torch.ones(15, dtype=torch.float64)
        assert torch.allclose(D @ one, one, atol=1e-12)                 # D 1 = 1
        assert torch.allclose(spec.mu @ D, spec.mu, atol=1e-12)         # μᵀ D = μᵀ
        assert abs(float(spec.mu.sum()) - 1.0) < 1e-12
        assert torch.allclose(spec.pi, torch.full((15,), 1 / 15, dtype=torch.float64))


def test_counting_measure_gives_the_squared_singular_values():
    A = causal_attention(11, heads=3, seed=4)
    lam = operator_spectrum(A, "counting", eigenfunctions=False).eigenvalues
    sv = torch.linalg.svdvals(A)
    assert torch.allclose(lam, sv ** 2, atol=1e-12)
    S, D = attention_operator(A, "counting", return_difference=True)
    assert torch.allclose(S, A.transpose(-1, -2) @ A, atol=1e-14) and torch.allclose(S, D)
    # Proposition 1: σ₁ ≥ 1 for a row-stochastic A.
    assert (lam[..., 0] >= 1 - 1e-12).all()


@pytest.mark.parametrize("measure", MEASURES)
def test_operator_is_positive_semidefinite(measure):
    A = causal_attention(20, heads=4, seed=6, temp=0.3)
    S, D = attention_operator(A, measure, return_difference=True)
    assert float(torch.linalg.eigvalsh(S).min()) > -1e-12
    mu, _ = operator_measures(A, measure)
    f = torch.randn(4, 20, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    quad = (mu * f * (D @ f.unsqueeze(-1)).squeeze(-1)).sum(-1)      # ⟨f, Df⟩_μ
    assert (quad >= -1e-12).all()


def test_pullback_gains_are_at_most_one_and_contraction_is_the_centred_gain():
    A = causal_attention(16, seed=7)
    spec = operator_spectrum(A)
    mu, pi = spec.mu, spec.pi
    g = torch.Generator().manual_seed(2)
    for _ in range(50):
        f = torch.randn(16, generator=g, dtype=torch.float64)
        gain = (pi * (A @ f) ** 2).sum() / (mu * f * f).sum()
        assert gain <= 1 + 1e-12                                       # Proposition 2
        fc = f - (mu * f).sum()                                        # μ-centred
        gain_c = ((pi * (A @ fc) ** 2).sum() / (mu * fc * fc).sum()).sqrt()
        assert gain_c <= operator_metrics(A)["contraction"] + 1e-12    # Corollary 0.1
    phi2 = spec.eigenfunctions[:, 1]
    assert abs(float((mu * phi2).sum())) < 1e-10                       # centred mode
    gain2 = ((pi * (A @ phi2) ** 2).sum() / (mu * phi2 * phi2).sum()).sqrt()
    assert abs(float(gain2 - operator_metrics(A)["contraction"])) < 1e-10
    Phi = spec.eigenfunctions
    assert torch.allclose(Phi.T @ (mu.unsqueeze(-1) * Phi), torch.eye(16, dtype=torch.float64),
                          atol=1e-9)                                   # orthonormal in L²(μ)


def test_participation_ratio_bounds_and_special_cases():
    g = torch.Generator().manual_seed(0)
    for n in (2, 5, 30):
        for _ in range(20):
            lam = torch.rand(n, generator=g, dtype=torch.float64) ** 4
            lam[torch.rand(n, generator=g) < 0.3] = 0
            rank = int((lam > 0).sum())
            pr = float(participation_ratio(lam))
            if rank == 0:
                assert pr == 0.0
                continue
            assert 1 / n - 1e-12 <= pr <= rank / n + 1e-12
            assert abs(pr - float(participation_ratio(7.5 * lam))) < 1e-12  # scale-free
            un = float(participation_ratio(lam, normalized=False))
            assert abs(un - n * pr) < 1e-9 and 1 - 1e-12 <= un <= rank + 1e-12
    assert float(participation_ratio(torch.ones(8))) == pytest.approx(1.0)
    assert float(participation_ratio(torch.tensor([3.0, 0, 0, 0]))) == pytest.approx(0.25)
    # Centred: the largest eigenvalue is left out and the ratio uses n - 1.
    lam = torch.tensor([1.0, 0.5, 0.5, 0.0])
    assert float(participation_ratio(lam, centered=True)) == pytest.approx(2 / 3)
    assert float(participation_ratio(torch.tensor([1.0, 0, 0]), centered=True)) == 0.0
    assert float(participation_ratio(torch.tensor([1.0]), centered=True)) == 0.0
    # Types: NumPy in, NumPy out; any order of the eigenvalues.
    out = participation_ratio(np.array([[0.0, 1.0, 1.0], [2.0, 0.0, 0.0]]))
    assert isinstance(out, np.ndarray) and np.allclose(out, [2 / 3, 1 / 3])
    assert isinstance(participation_ratio([1.0, 1.0]), np.float64)


def test_sink_heavy_head_deflates_the_euclidean_ratio_but_not_the_measure_aware_one():
    # Section 3.1.1 and Figure 2: counting geometry amplifies column mass, so
    # a sink-biased head that still routes along the diagonal scores like a
    # pure sink in PR(AᵀA); the pullback geometry keeps it high.
    n = 40
    euclid, intrinsic = [], []
    for s in (0.0, 0.5, 0.77, 0.9):
        A = sink_biased(n, s)
        euclid.append(float(operator_metrics(A, "counting")["participation_ratio"]))
        intrinsic.append(float(operator_metrics(A)["participation_ratio"]))
    assert euclid[0] == pytest.approx(1.0) and intrinsic[0] == pytest.approx(1.0)
    for e, i in zip(euclid[1:], intrinsic[1:]):
        assert e < 0.1 and i > 0.4 and i > 4 * e
    assert euclid[2] < 2 / n and intrinsic[2] > 0.75          # s = 0.77: 0.029 vs 0.789
    # Pure sink A = 1 e₁ᵀ: AᵀA = n e₁e₁ᵀ, one mode in both geometries.
    sink = sink_biased(n, 1.0)
    lam_c = operator_spectrum(sink, "counting", eigenfunctions=False).eigenvalues
    assert float(lam_c[0]) == pytest.approx(n)
    m = operator_metrics(sink)
    assert float(m["participation_ratio"]) == pytest.approx(1 / n)
    assert float(m["participation_ratio_centered"]) == 0.0 and float(m["contraction"]) == 0.0


def test_zero_mass_keys_get_eigenvalue_zero_and_match_the_supported_operator():
    n = 10
    A = causal_attention(n, seed=8).clone()
    dead = [3, 7]
    A[:, dead] = 0
    A = A / A.sum(-1, keepdim=True)                       # rows 3 and 7 now read earlier keys
    keep = [j for j in range(n) if j not in dead]
    spec = operator_spectrum(A)
    assert torch.isfinite(spec.eigenvalues).all() and torch.isfinite(spec.eigenfunctions).all()
    assert (spec.mu[dead] == 0).all() and (spec.mu[keep] > 0).all()
    # The same as the operator on the keys of positive mass, padded with zeros.
    B = (A[:, keep] / spec.mu[keep].sqrt()) / n ** 0.5
    sub = torch.sort(torch.linalg.eigvalsh(B.T @ B), descending=True).values
    padded = torch.cat([sub, torch.zeros(len(dead), dtype=torch.float64)])
    assert torch.allclose(spec.eigenvalues, padded, atol=1e-12)
    assert abs(float(spec.eigenvalues[0]) - 1) < 1e-12
    # D φ = λ φ for every mode, with Ω_μ⁻¹ read as the pseudo-inverse.
    S, D = attention_operator(A, return_difference=True)
    assert torch.isfinite(D).all() and (D[dead] == 0).all() and (D[:, dead] == 0).all()
    Phi = spec.eigenfunctions
    assert torch.allclose(D @ Phi, Phi * spec.eigenvalues, atol=1e-9)
    # The participation ratio keeps the 1/n normalisation.
    pr = float(operator_metrics(A)["participation_ratio"])
    assert pr == pytest.approx(float(participation_ratio(sub)) * len(keep) / n)


def test_custom_measures():
    A = causal_attention(8, seed=9)
    mu, pi = operator_measures(A)
    assert torch.allclose(attention_operator(A, (mu, pi)), attention_operator(A))
    ones = torch.ones(8, dtype=torch.float64)
    assert torch.allclose(attention_operator(A, (ones, ones)), attention_operator(A, "counting"))
    bad = mu.clone()
    bad[2] = 0                                             # key 2 is read but has no mass
    with pytest.raises(ValueError, match="unbounded"):
        attention_operator(A, (bad, pi))
    B = A.clone()
    B[:, 2] = 0
    B = B / B.sum(-1, keepdim=True)
    attention_operator(B, (bad, pi))                       # unread zero-mass key: fine
    with pytest.raises(ValueError, match="non-negative"):
        attention_operator(A, (-mu, pi))
    with pytest.raises(ValueError, match="measure"):
        attention_operator(A, "uniform")


def test_row_checks_and_value_weighted_rows():
    A = causal_attention(6, seed=10)
    raw = A * torch.arange(1, 7, dtype=torch.float64).unsqueeze(-1)  # rows sum to 1..6
    with pytest.raises(ValueError, match="normalize_rows"):
        operator_metrics(raw)
    got = operator_metrics(raw, normalize_rows=True)
    want = operator_metrics(A)
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-12)
    operator_metrics(raw, "counting")                      # no row condition off the pullback
    with pytest.raises(ValueError, match="non-negative"):
        operator_metrics(-A)
    # A query that reads nothing (a zero row) keeps λ₁ = 1 and D 1 = 1.
    Z = A.clone()
    Z[4] = 0
    lam = operator_spectrum(Z, eigenfunctions=False).eigenvalues
    assert abs(float(lam[0]) - 1) < 1e-12
    S, D = attention_operator(Z, return_difference=True)
    assert torch.allclose(D @ torch.ones(6, dtype=torch.float64), torch.ones(6, dtype=torch.float64))
    # float32 softmax rows pass the default tolerance.
    operator_metrics(A.float())
    with pytest.raises(ValueError, match=r"\[\.\.\., n, n\]"):
        operator_metrics(torch.rand(3, 4))


def test_per_head_batches_and_head_average():
    A = causal_attention(9, heads=5, seed=11)
    batched = operator_metrics(A, return_eigenvalues=True)
    for h in range(5):
        one = operator_metrics(A[h], return_eigenvalues=True)
        for k in one:
            assert torch.allclose(batched[k][h], one[k], atol=1e-12)
    avg = operator_metrics(A, average_heads=True)
    ref = operator_metrics(A.mean(0))
    for k in ref:
        assert torch.allclose(avg[k], ref[k], atol=1e-14) and avg[k].shape == ()
    # The operator of the averaged attention is not the average of the
    # per-head operators.
    assert not torch.allclose(avg["participation_ratio"], batched["participation_ratio"].mean())
    two = operator_metrics(torch.stack([A, A.flip(0)]), average_heads=True)
    assert two["participation_ratio"].shape == (2,)
    S = attention_operator(A.float().numpy(), average_heads=True)
    assert S.dtype == torch.float64 and S.shape == (9, 9)


def test_identity_and_shift_heads():
    n = 12
    m = operator_metrics(torch.eye(n, dtype=torch.float64))
    assert all(float(v) == pytest.approx(1.0) for v in m.values())
    shift = torch.zeros(n, n, dtype=torch.float64)
    shift[0, 0] = 1
    shift[torch.arange(1, n), torch.arange(n - 1)] = 1
    m = operator_metrics(shift)                             # key n-1 has no mass
    assert float(m["participation_ratio"]) == pytest.approx((n - 1) / n)
    assert float(m["contraction"]) == pytest.approx(1.0)
