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
    assert U.few_clusters_note(22, "overall") is not None
    assert "G=22" in U.few_clusters_note(22, "overall")
    assert U.few_clusters_note(100, "overall") is None


def test_few_clusters_note_names_its_stratum() -> None:
    """S6 item 7: the stratum must be in the message, or two flags at different G are
    indistinguishable as to which cell each one describes."""
    overall = U.few_clusters_note(11, "overall")
    subgroup = U.few_clusters_note(22, "scanner_domain=0")
    assert "overall" in overall
    assert "scanner_domain=0" in subgroup
    assert "G=11" in overall and "G=22" in subgroup


def test_percentile_interval_ignores_undefined_replicates() -> None:
    samples = np.array([0.1, np.nan, 0.2, np.inf, 0.3])
    ci = U.percentile_interval(samples, 0.95, "m")
    assert np.isfinite(ci.low) and np.isfinite(ci.high)
    assert ci.low >= 0.1 and ci.high <= 0.3


# --------------------------------------------------------- paired cross-fit gain interval (item 1)


def _recalibrated_pair(seed=7, n_clusters=20, per=200):
    """A published score p0 and a compressed companion p_r on the same rows (correlated)."""
    y, p0, g = clustered_sample(n_clusters=n_clusters, per_cluster=per, seed=seed)
    z = np.log(np.clip(p0, 1e-6, 1 - 1e-6) / (1 - np.clip(p0, 1e-6, 1 - 1e-6)))
    p_r = 1.0 / (1.0 + np.exp(-(0.2 + 0.9 * z)))
    return y, p0, p_r, g


def test_paired_gain_interval_brackets_point_and_is_deterministic() -> None:
    y, p0, p_r, g = _recalibrated_pair()
    point, ci = U.paired_brier_gain_interval(y, p0, p_r, g, 0.95, n_boot=2000, seed=1337)
    assert ci.method == "paired_cluster_bootstrap"
    assert ci.low <= point <= ci.high, "the paired interval must bracket its point estimate"
    # point equals brier(p0) - brier(p_r) exactly (same rows)
    assert point == pytest.approx(M.brier(y, p0) - M.brier(y, p_r))
    p2, ci2 = U.paired_brier_gain_interval(y, p0, p_r, g, 0.95, n_boot=2000, seed=1337)
    assert (point, ci.low, ci.high) == (p2, ci2.low, ci2.high)


def test_paired_gain_is_tighter_than_an_unpaired_difference() -> None:
    """The whole reason to pair: p0 and its recalibration move together within a slide, so the
    paired interval is far tighter than differencing two independent slide resamples."""
    y, p0, p_r, g = _recalibrated_pair()
    _, paired = U.paired_brier_gain_interval(y, p0, p_r, g, 0.95, n_boot=2000, seed=1337)

    clusters = U.cluster_indices(g)
    d0, dr = (p0 - y) ** 2, (p_r - y) ** 2
    rng0 = np.random.default_rng(1337)
    rng1 = np.random.default_rng(24601)  # a DIFFERENT stream for the second term -> unpaired
    unp = np.empty(2000)
    for b in range(2000):
        unp[b] = d0[U.bootstrap_cluster_indices(clusters, rng0)].mean() - dr[
            U.bootstrap_cluster_indices(clusters, rng1)
        ].mean()
    unpaired = U.percentile_interval(unp, 0.95, "x")
    assert (paired.high - paired.low) < 0.5 * (unpaired.high - unpaired.low)


def test_paired_gain_degrades_to_nan_with_one_cluster() -> None:
    y, p0, p_r, _ = _recalibrated_pair(n_clusters=1, per=200)
    g = np.full(y.size, "only")
    point, ci = U.paired_brier_gain_interval(y, p0, p_r, g, 0.95, n_boot=50, seed=1337)
    assert np.isfinite(point) and ci.is_empty


# ------------------------------------------------------------------- Brier skill interval (item 3)


def test_brier_skill_interval_brackets_and_anchors() -> None:
    y, p, groups = clustered_sample()
    bss, ci = U.brier_skill_interval(y, p, groups, 0.95, n_boot=2000, seed=1337)
    assert ci.method == "cluster_bootstrap"
    assert ci.low <= bss <= ci.high
    # a genuinely skilful score sits above the no-skill anchor of 0
    assert bss > 0.0
    pbar = float(np.mean(y))
    expected = 1.0 - M.brier(y, p) / (pbar * (1 - pbar))
    assert bss == pytest.approx(expected)


def test_brier_skill_of_the_prevalence_predictor_is_about_zero() -> None:
    """Always predicting prevalence is the no-skill reference: its BSS must be ~0."""
    y, _, groups = clustered_sample()
    pbar = float(np.mean(y))
    p = np.full(y.size, pbar)
    bss, _ = U.brier_skill_interval(y, p, groups, 0.95, n_boot=200, seed=1337)
    assert bss == pytest.approx(0.0, abs=1e-9)
