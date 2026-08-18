"""Uncertainty must respect the unit of independence, not the unit of analysis."""

from __future__ import annotations

import numpy as np
import pytest

from predval import metrics as M
from predval import uncertainty as U


def clustered_sample(n_clusters=25, per_cluster=200, seed=7):
    """Data with a strong per-cluster effect, so row-level intervals are demonstrably wrong.

    Each cluster gets its own risk offset. Rows within a cluster are therefore correlated, which
    is exactly the situation where resampling rows understates uncertainty.
    """
    rng = np.random.default_rng(seed)
    ys, ps, gs = [], [], []
    for c in range(n_clusters):
        shift = rng.normal(0, 1.5)
        z = rng.normal(shift, 1.0, size=per_cluster)
        p = 1.0 / (1.0 + np.exp(-z))
        ys.append(rng.binomial(1, p))
        ps.append(p)
        gs.append(np.full(per_cluster, f"c{c:02d}"))
    return (
        np.concatenate(ys).astype(np.int64),
        np.concatenate(ps),
        np.concatenate(gs),
    )


def _width(y, p, groups, *, cluster: bool, b=400, seed=1337):
    rng = np.random.default_rng(seed)
    clusters = U.cluster_indices(groups)
    out = np.empty(b)
    for i in range(b):
        idx = (
            U.bootstrap_cluster_indices(clusters, rng)
            if cluster
            else rng.integers(0, y.size, size=y.size)
        )
        out[i] = M.auroc(y[idx], p[idx])
    ci = U.percentile_interval(out, 0.95, "x")
    return ci.high - ci.low


def test_cluster_bootstrap_is_wider_than_row_bootstrap() -> None:
    """The central claim of the whole clustering design. If this fails, nothing else matters."""
    y, p, groups = clustered_sample()
    cluster_w = _width(y, p, groups, cluster=True)
    row_w = _width(y, p, groups, cluster=False)
    assert cluster_w > row_w * 3, (
        f"cluster CI width {cluster_w:.4f} should greatly exceed row CI width {row_w:.4f}"
    )


def test_cluster_indices_partition_every_row() -> None:
    _, _, groups = clustered_sample(n_clusters=5, per_cluster=10)
    parts = U.cluster_indices(groups)
    assert len(parts) == 5
    assert sorted(np.concatenate(parts)) == list(range(groups.size))
    for part in parts:
        assert len(set(groups[part])) == 1, "a cluster must contain exactly one group label"


def test_bootstrap_draws_whole_clusters() -> None:
    _, _, groups = clustered_sample(n_clusters=6, per_cluster=10)
    clusters = U.cluster_indices(groups)
    rng = np.random.default_rng(0)
    idx = U.bootstrap_cluster_indices(clusters, rng)
    assert idx.size == groups.size
    counts = {g: int(np.count_nonzero(groups[idx] == g)) for g in np.unique(groups)}
    # every present cluster appears as a whole multiple of its size
    assert all(c % 10 == 0 for c in counts.values())


def test_bootstrap_is_reproducible_from_the_seed() -> None:
    y, p, groups = clustered_sample(n_clusters=8, per_cluster=50)
    assert _width(y, p, groups, cluster=True, seed=99) == _width(
        y, p, groups, cluster=True, seed=99
    )


def test_clustered_mean_interval_covers_the_point() -> None:
    y, p, groups = clustered_sample()
    per_row = (p - y) ** 2
    ci = U.clustered_mean_interval(per_row, groups, 0.95)
    assert ci.low < per_row.mean() < ci.high
    assert ci.method == "cluster_robust_t"


def test_clustered_mean_interval_wider_than_iid(monkeypatch) -> None:
    """Against an iid sample the clustered interval should not be dramatically wider."""
    rng = np.random.default_rng(3)
    vals = rng.normal(size=5000)
    groups = np.arange(5000).astype(str)  # every row its own cluster
    ci = U.clustered_mean_interval(vals, groups, 0.95)
    naive_half = 1.96 * vals.std(ddof=1) / np.sqrt(vals.size)
    assert (ci.high - ci.low) / 2 == pytest.approx(naive_half, rel=0.05)


def test_calibration_interval_returns_value_and_bounds() -> None:
    y, p, groups = clustered_sample()
    for which in ("intercept", "slope"):
        value, ci = U.calibration_interval(y, p, groups, 0.95, which=which)
        assert np.isfinite(value)
        assert ci.low < value < ci.high
        assert ci.method == "cluster_robust_t"


def test_calibration_interval_degrades_to_nan_not_crash() -> None:
    y = np.array([1, 1, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.8])
    groups = np.array(["a", "a", "b", "b"])
    value, ci = U.calibration_interval(y, p, groups, 0.95, which="slope")
    assert np.isnan(value) and ci.is_empty


def test_loso_fragility_finds_the_planted_culprit() -> None:
    """One cluster is made pathological; fragility must name it."""
    y, p, groups = clustered_sample(n_clusters=10, per_cluster=100, seed=11)
    bad = groups == "c03"
    p = p.copy()
    p[bad] = 1.0 - p[bad]  # invert this slide's predictions

    point = M.auroc(y, p)
    delta, culprit = U.loso_fragility(lambda keep: M.auroc(y[keep], p[keep]), groups, point)
    assert culprit == "c03"
    assert delta > 0.01


def test_loso_returns_nan_for_a_single_cluster() -> None:
    groups = np.array(["only"] * 10)
    delta, culprit = U.loso_fragility(lambda keep: 1.0, groups, 1.0)
    assert np.isnan(delta) and culprit is None


def test_few_clusters_note_fires_below_the_threshold() -> None:
    assert U.few_clusters_note(22) is not None
    assert "G=22" in U.few_clusters_note(22)
    assert U.few_clusters_note(100) is None


def test_percentile_interval_ignores_undefined_replicates() -> None:
    samples = np.array([0.1, np.nan, 0.2, np.inf, 0.3])
    ci = U.percentile_interval(samples, 0.95, "m")
    assert np.isfinite(ci.low) and np.isfinite(ci.high)
    assert ci.low >= 0.1 and ci.high <= 0.3
