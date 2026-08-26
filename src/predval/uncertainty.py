"""Interval estimation that respects the unit of independence.

Three layers, per docs/spec.md section 5:

1. Cluster percentile bootstrap -- the default for every metric.
2. Analytic cluster-robust cross-checks, where the structure allows one.
3. Leave-one-cluster-out fragility, which is deliberately not an interval.

Plus the naive per-row interval, computed only so a report can show how badly it understates
uncertainty when observations are clustered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy import stats

from .metrics import logit

#: Below this many clusters, percentile tails are not resolvable and the report says so.
FEW_CLUSTERS = 40


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    method: str

    @property
    def is_empty(self) -> bool:
        return not (np.isfinite(self.low) and np.isfinite(self.high))


def cluster_indices(groups: np.ndarray) -> list[np.ndarray]:
    """Row indices for each distinct cluster, in stable order."""
    order = np.argsort(groups, kind="mergesort")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    return np.split(order, boundaries[1:])


def bootstrap_cluster_indices(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One cluster-resampled row index: draw G clusters with replacement, concatenate them.

    Resampling whole clusters rather than rows is the entire point -- it propagates the
    within-cluster correlation that a row bootstrap throws away.
    """
    picks = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in picks])


def percentile_interval(samples: np.ndarray, ci_level: float, method: str) -> Interval:
    """Percentile interval over bootstrap replicates, ignoring replicates that were undefined."""
    finite = samples[np.isfinite(samples)]
    if finite.size < 2:
        return Interval(float("nan"), float("nan"), method)
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return Interval(float(low), float(high), method)


def clustered_mean_interval(values: np.ndarray, groups: np.ndarray, ci_level: float) -> Interval:
    """Interval for the mean of a per-row quantity, treating clusters as the sampling unit.

    Used for Brier. The cluster means are the observations, so the degrees of freedom are
    G - 1 rather than n - 1, which is the whole correction.
    """
    clusters = cluster_indices(groups)
    g = len(clusters)
    if g < 2:
        return Interval(float("nan"), float("nan"), "cluster_robust_t")

    sizes = np.array([idx.size for idx in clusters], dtype=float)
    sums = np.array([values[idx].sum() for idx in clusters], dtype=float)
    n = sizes.sum()
    theta = sums.sum() / n

    # Influence-function form, so unequal cluster sizes are handled correctly rather than by
    # averaging cluster means as though every slide contributed equally.
    infl = (sums - sizes * theta) / n
    var = (g / (g - 1.0)) * np.sum(infl**2)
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0.0:
        return Interval(float(theta), float(theta), "cluster_robust_t")
    crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df=g - 1))
    return Interval(float(theta - crit * se), float(theta + crit * se), "cluster_robust_t")


def paired_brier_gain_interval(
    y: np.ndarray,
    p0: np.ndarray,
    p_r: np.ndarray,
    groups: np.ndarray,
    ci_level: float,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, Interval]:
    """Cluster bootstrap interval for the paired Brier gain ``brier(p0) - brier(p_r)``.

    ``p0`` is the as-published probability (rung0) and ``p_r`` a cross-fitted rung's out-of-fold
    probability, both aligned to the same rows. The gain is a *paired* difference: the per-row loss
    difference ``d_i = (p0_i - y_i)^2 - (p_r_i - y_i)^2`` already carries both terms, so resampling
    whole slides and averaging ``d`` over the resample evaluates both briers on the *same* slide
    draw. That pairing is the whole point -- p0 and its recalibration are strongly correlated within
    a slide, and an unpaired interval (two independent slide draws) would inflate the width with
    variance the difference does not actually have.

    This does **not** refit the correction inside each replicate; the cross-fitted mapping is held
    fixed (docs/spec.md section 4.5 keeps the refit-in-replicate interval as backlog). It is the
    interval on the gain of a *given* out-of-fold correction, which is the number the pitch shows.
    """
    d = (p0 - y) ** 2 - (p_r - y) ** 2
    point = float(np.mean(d)) if d.size else float("nan")
    clusters = cluster_indices(groups)
    if len(clusters) < 2 or not np.isfinite(point):
        return point, Interval(float("nan"), float("nan"), "paired_cluster_bootstrap")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = bootstrap_cluster_indices(clusters, rng)
        samples[b] = np.mean(d[idx])
    return point, percentile_interval(samples, ci_level, "paired_cluster_bootstrap")


def brier_skill_interval(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    ci_level: float,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, Interval]:
    """Brier skill score ``1 - brier(p) / (pbar*(1-pbar))`` with a cluster bootstrap interval.

    ``pbar`` is the observed prevalence, so the reference ``pbar*(1-pbar)`` is the Brier of the
    no-skill model that always predicts prevalence. BSS is therefore anchored at **0 = no-skill**
    and **1 = perfect**; it is the fraction of the gap from one to the other that ``p`` closes.

    Both the reference and the loss are recomputed inside each slide resample, so the interval
    carries the sampling variance of prevalence as well as of the loss. Replicates whose resampled
    prevalence is degenerate (all one class -> zero-variance reference) are dropped as undefined
    rather than divided by zero.
    """
    pbar = float(np.mean(y)) if y.size else float("nan")
    ref = pbar * (1.0 - pbar)
    point = 1.0 - float(np.mean((p - y) ** 2)) / ref if ref > 0 else float("nan")
    clusters = cluster_indices(groups)
    if len(clusters) < 2 or not np.isfinite(point):
        return point, Interval(float("nan"), float("nan"), "cluster_bootstrap")
    rng = np.random.default_rng(seed)
    samples = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = bootstrap_cluster_indices(clusters, rng)
        yb, pb = y[idx], p[idx]
        mb = float(np.mean(yb))
        ref_b = mb * (1.0 - mb)
        if ref_b > 0:
            samples[b] = 1.0 - float(np.mean((pb - yb) ** 2)) / ref_b
    return point, percentile_interval(samples, ci_level, "cluster_bootstrap")


def calibration_interval(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    ci_level: float,
    *,
    which: str,
) -> tuple[float, Interval]:
    """Calibration intercept or slope with a cluster-robust sandwich covariance.

    `which` is "intercept" (slope fixed at 1, via an offset) or "slope". Uses statsmodels here
    rather than the fast in-house IRLS because the cluster-robust covariance is precisely what
    this path exists to produce.
    """
    nan = Interval(float("nan"), float("nan"), "cluster_robust_t")
    g = len(np.unique(groups))
    if y.size == 0 or len(np.unique(y)) < 2 or g < 2:
        return float("nan"), nan

    z = logit(p)
    try:
        if which == "intercept":
            exog = np.ones((y.size, 1))
            model = sm.GLM(y, exog, family=sm.families.Binomial(), offset=z)
            idx = 0
        else:
            if np.ptp(z) == 0.0:
                return float("nan"), nan
            exog = sm.add_constant(z, has_constant="add")
            model = sm.GLM(y, exog, family=sm.families.Binomial())
            idx = 1
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
        value = float(fit.params[idx])
        se = float(np.sqrt(fit.cov_params()[idx, idx]))
    except Exception:  # noqa: BLE001 - a non-converging fit is a NaN result, not a crash
        return float("nan"), nan

    if not np.isfinite(value) or not np.isfinite(se):
        return (value if np.isfinite(value) else float("nan")), nan
    crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df=g - 1))
    return value, Interval(value - crit * se, value + crit * se, "cluster_robust_t")


def loso_fragility(estimator, groups: np.ndarray, point: float) -> tuple[float, str | None]:
    """Largest change in a metric from dropping any single cluster, and the culprit.

    Deliberately not an interval. It answers a question a confidence interval cannot: is this
    result carried by one cluster? See docs/spec.md section 5.3.
    """
    if not np.isfinite(point):
        return float("nan"), None
    labels = np.unique(groups)
    if labels.size < 2:
        return float("nan"), None

    worst = 0.0
    culprit: str | None = None
    for label in labels:
        keep = np.flatnonzero(groups != label)
        if keep.size == 0:
            continue
        value = estimator(keep)
        if not np.isfinite(value):
            continue
        delta = abs(value - point)
        if delta > worst:
            worst, culprit = float(delta), str(label)
    return worst, culprit


def few_clusters_note(n_clusters: int, stratum: str) -> str | None:
    """`stratum` names which cell this G belongs to (e.g. "overall", "scanner_domain=0") --
    without it, a report with multiple few-clusters flags at different G values has no way to say
    which stratum each one is about (S6 item 7).
    """
    if n_clusters < FEW_CLUSTERS:
        return (
            f"tail quantiles approximate with few clusters (G={n_clusters}) for {stratum}; "
            f"percentile bootstrap intervals below {FEW_CLUSTERS} clusters are coarse"
        )
    return None
