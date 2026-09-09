"""Per-head spectral diagnostics: correctness and the head-averaging loss."""
import numpy as np
import torch

from spectral_trust import (
    GSPConfig, per_head_metrics, per_head_metrics_all_layers,
    stack_feature_vector, layer_delta_features,
    PER_HEAD_METRIC_NAMES, PER_HEAD_METRIC_FAMILY,
)


def _attn(num_heads=4, seq=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(num_heads, seq, seq, generator=g)
    return a / a.sum(dim=-1, keepdim=True)  # row-stochastic, like softmax


def _reference_lambda2(head_matrix, cfg):
    """Independent reference: symmetrize, normalized Laplacian, exact eigh."""
    w = 0.5 * (head_matrix + head_matrix.T)
    if cfg.remove_self_loops:
        w = w.clone()
        w.fill_diagonal_(0.0)
    deg = w.sum(-1)
    inv = torch.where(deg > 1e-8, deg.clamp(min=1e-8).rsqrt(),
                      torch.zeros_like(deg))
    lap = torch.eye(w.shape[0]) - inv[:, None] * w * inv[None, :]
    return float(torch.linalg.eigvalsh(0.5 * (lap + lap.T))[1])


def test_shape_names_and_family():
    d = per_head_metrics(_attn(), layer=3)
    assert d.values.shape == (4, len(PER_HEAD_METRIC_NAMES))
    assert d.layer == 3
    assert d.family == PER_HEAD_METRIC_FAMILY == "attention"
    assert set(d.to_dict()) == {"layer", "metric_names", "family", "values"}


def test_matches_independent_reference():
    cfg = GSPConfig(normalization="sym")
    attn = _attn()
    d = per_head_metrics(attn, config=cfg)
    got = d.metric("fiedler_value")
    want = [_reference_lambda2(attn[h], cfg) for h in range(attn.shape[0])]
    assert np.allclose(got, want, atol=1e-5), (got, want)


def test_batch_dim_accepted_and_metrics_bounded():
    d = per_head_metrics(_attn().unsqueeze(0))
    lam2 = d.metric("fiedler_value")
    lam_max = d.metric("spectral_radius")
    ent = d.metric("spectral_entropy_norm")
    hfer = d.metric("hfer")
    # symmetric normalized Laplacian spectrum lies in [0, 2]
    assert np.all(lam2 >= -1e-6) and np.all(lam_max <= 2 + 1e-6)
    assert np.all(lam2 <= lam_max + 1e-6)
    assert np.all((ent >= -1e-6) & (ent <= 1 + 1e-6))
    assert np.all((hfer >= -1e-6) & (hfer <= 1 + 1e-6))


def test_token_span_restricts_graph():
    attn = _attn(seq=20)
    full = per_head_metrics(attn)
    span = per_head_metrics(attn, token_span=(10, 20))
    sub = per_head_metrics(attn[:, 10:20, 10:20])
    assert np.allclose(span.values, sub.values, atol=1e-6)
    assert not np.allclose(span.values, full.values, atol=1e-3)


def test_degenerate_sequence_is_zeroed():
    d = per_head_metrics(torch.rand(3, 2, 2))
    assert np.allclose(d.values, 0.0)


def test_head_averaging_destroys_per_head_variance():
    """The motivation for the module: one anomalous head is visible per-head
    and washed out by averaging."""
    attn = _attn(num_heads=8, seq=16)
    attn[3] = torch.eye(16)  # a fully fragmented (identity-routing) head
    per_head = per_head_metrics(attn).metric("fiedler_value")
    averaged = per_head_metrics(
        attn.mean(dim=0, keepdim=True)).metric("fiedler_value")
    # the anomalous head is a clear outlier per-head ...
    spread = per_head.max() - per_head.min()
    assert spread > 0.1
    # ... while the averaged graph reports a single value inside the bulk
    assert averaged.shape == (1,)
    bulk = np.delete(per_head, 3)
    assert abs(averaged[0] - bulk.mean()) < spread


def test_feature_helpers():
    attns = [_attn(seed=s) for s in range(5)]
    diags = per_head_metrics_all_layers(attns)
    assert len(diags) == 5 and [d.layer for d in diags] == list(range(5))

    vec = stack_feature_vector(diags)
    assert vec.shape == (5 * 4 * len(PER_HEAD_METRIC_NAMES),)

    one = stack_feature_vector(diags, metrics=["fiedler_value"])
    assert one.shape == (5 * 4,)
    assert np.allclose(one[:4], diags[0].metric("fiedler_value"))

    deltas = layer_delta_features(diags, metric="spectral_radius")
    assert deltas.shape == (4 * 4,)
    expected = (diags[1].metric("spectral_radius")
                - diags[0].metric("spectral_radius"))
    assert np.allclose(deltas[:4], expected, atol=1e-6)
    assert layer_delta_features(diags[:1]).shape == (0,)
