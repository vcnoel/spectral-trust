"""Regression tests for 0.4.1: per-column metrics, the per-head device
argument, layer workers, and the thread count after concurrent per-head
calls."""
import threading

import numpy as np
import pytest
import torch

import spectral_trust as st
from spectral_trust import per_head as ph


def _causal_attention(heads, seq, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(heads, seq, seq, generator=g) * 2.0
    logits[:, :, 0] += 3.0                                   # a sink column
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
    return torch.softmax(logits.masked_fill(mask, -1e9), dim=-1)


def _sparse_columns(seq, d, seed):
    rng = np.random.default_rng(seed)
    X = np.where(rng.random((seq, d)) < 0.2, rng.exponential(1.0, (seq, d)), 0.0)
    X[:, 0] = 0.0                                            # one empty column
    return X.astype(np.float32)


def _laplacian(attn, cfg):
    gc = st.GraphConstructor(cfg)
    adj = gc.aggregate_heads(gc.symmetrize_attention(attn[None]))
    return gc.construct_laplacian(adj)[0]


@pytest.mark.parametrize("norm", ["sym", "rw", "none"])
def test_per_column_metrics_match_analyze_layer_column_by_column(norm):
    cfg = st.GSPConfig(device="cpu", normalization=norm, verbose=False)
    attn = _causal_attention(6, 48, seed=1)
    L = _laplacian(attn, cfg)
    X = _sparse_columns(48, 12, seed=2)
    got = st.per_column_metrics(X, L, cfg)
    assert got.values.shape == (12, 4) and got.metric_names == st.COLUMN_METRIC_NAMES
    an = st.SpectralAnalyzer(cfg)
    for j in range(12):
        d = an.analyze_layer(torch.from_numpy(X[:, j:j + 1]), L, 0)
        want = [d.energy, d.smoothness_index, d.spectral_entropy, d.hfer]
        np.testing.assert_allclose(got.values[j], want, rtol=1e-5, atol=1e-7)
    assert np.all(got.values[0] == 0.0)                      # empty column
    np.testing.assert_allclose(got.eigenvalues, an.compute_eigendecomposition(L.numpy())[0],
                               atol=1e-6)


def test_per_column_metrics_inputs():
    cfg = st.GSPConfig(device="cpu", verbose=False)
    L = _laplacian(_causal_attention(2, 20, seed=3), cfg)
    x = _sparse_columns(20, 2, seed=4)[:, 1]
    one = st.per_column_metrics(x, L[None], cfg)             # [seq] and [1, S, S]
    two = st.per_column_metrics(torch.from_numpy(x[:, None]), L.numpy(), cfg)
    assert one.values.shape == (1, 4)
    np.testing.assert_array_equal(one.values, two.values)
    hfer = one.metric("hfer")
    assert hfer.shape == (1,) and 0.0 <= hfer[0] <= 1.0
    with pytest.raises(ValueError):
        st.per_column_metrics(np.zeros((19, 3), np.float32), L, cfg)


def test_config_device_is_not_read_by_per_head_functions():
    # GSPConfig(device="cuda") must not send the work to a GPU that is not
    # there; the per-head functions take their own device argument.
    attn = _causal_attention(4, 24, seed=5)
    X = torch.from_numpy(_sparse_columns(24, 5, seed=6))
    cfg = st.GSPConfig(device="cuda", verbose=False)
    a = st.per_head_metrics(attn, config=cfg)
    b = st.per_head_metrics(attn, config=cfg, device="cpu")
    np.testing.assert_array_equal(a.values, b.values)
    c = st.per_head_hybrid_metrics(attn, X, config=cfg)
    d = st.per_head_hybrid_metrics(attn, X, config=cfg, device=torch.device("cpu"))
    np.testing.assert_array_equal(c.values, d.values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_per_head_on_cuda_matches_cpu():
    attn = _causal_attention(8, 64, seed=7)
    X = torch.from_numpy(_sparse_columns(64, 16, seed=8))
    cpu = st.per_head_metrics(attn)
    gpu = st.per_head_metrics(attn, device="cuda")
    np.testing.assert_allclose(gpu.values, cpu.values, rtol=1e-4, atol=1e-5)
    cpu_h = st.per_head_hybrid_metrics(attn, X)
    gpu_h = st.per_head_hybrid_metrics(attn, X, device="cuda")
    np.testing.assert_allclose(gpu_h.values, cpu_h.values, rtol=1e-3, atol=2e-4)


def _fresh_thread_count():
    out = {}
    t = threading.Thread(target=lambda: out.update(n=torch.get_num_threads()))
    t.start()
    t.join()
    return out["n"]


def test_layer_workers_give_the_same_results_and_keep_the_thread_count():
    n0 = torch.get_num_threads()
    attns = [_causal_attention(6, 40, seed=10 + i) for i in range(6)]
    hidden = [torch.from_numpy(_sparse_columns(40, 8, seed=20 + i)) for i in range(7)]
    one = st.per_head_metrics_all_layers(attns)
    four = st.per_head_metrics_all_layers(attns, num_workers=4)
    assert [d.layer for d in four] == list(range(6))
    for a, b in zip(one, four):
        np.testing.assert_array_equal(a.values, b.values)
    one_h = st.per_head_hybrid_metrics_all_layers(attns, hidden)
    four_h = st.per_head_hybrid_metrics_all_layers(attns, hidden, num_workers=4)
    for a, b in zip(one_h, four_h):
        np.testing.assert_array_equal(a.values, b.values)
    assert torch.get_num_threads() == n0
    assert _fresh_thread_count() == n0


@pytest.mark.skipif(torch.get_num_threads() < 2, reason="needs more than one thread")
def test_thread_that_inherits_one_thread_does_not_leave_it_as_the_default():
    # Thread B starts while the main thread is inside a per-head call, so it
    # inherits a count of 1, and it finishes last. Before 0.4.1 its restore
    # set 1 as the default of every thread created afterwards.
    n0 = torch.get_num_threads()
    inside, b_entered, main_left = threading.Event(), threading.Event(), threading.Event()

    def b():
        inside.wait()
        with ph._one_intraop_thread():
            b_entered.set()
            main_left.wait()

    t = threading.Thread(target=b)
    t.start()
    with ph._one_intraop_thread():
        inside.set()
        b_entered.wait()
    main_left.set()
    t.join()
    try:
        assert torch.get_num_threads() == n0
        assert _fresh_thread_count() == n0
    finally:
        torch.set_num_threads(n0)


def test_hybrid_docstring_tolerance_holds():
    # The documented agreement with analyze_layer (float32 rounding).
    cfg = st.GSPConfig(device="cpu", verbose=False)
    gc = st.GraphConstructor(cfg)
    an = st.SpectralAnalyzer(st.GSPConfig(device="cpu", eigen_solver="dense", verbose=False))
    for seed in range(4):
        attn = _causal_attention(8, 96, seed=30 + seed)
        X = torch.from_numpy(_sparse_columns(96, 1, seed=40 + seed) + 0.0)
        X[:, 0] = torch.from_numpy(_sparse_columns(96, 2, seed=50 + seed)[:, 1])
        got = st.per_head_hybrid_metrics(attn, X, cfg).values
        L = gc.construct_laplacian(gc.symmetrize_attention(attn[None])[0])
        for h in range(8):
            d = an.analyze_layer(X, L[h], 0)
            assert abs(got[h, 3] - d.hfer) <= 1e-4
            assert abs(got[h, 2] - d.spectral_entropy) <= 2e-3
            assert abs(got[h, 0] - d.energy) <= 1e-5 * max(1.0, abs(d.energy))
