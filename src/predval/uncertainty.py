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


def few_clusters_note(n_clusters: int) -> str | None:
    if n_clusters < FEW_CLUSTERS:
        return (
            f"tail quantiles approximate with few clusters (G={n_clusters}); "
            f"percentile bootstrap intervals below {FEW_CLUSTERS} clusters are coarse"
        )
    return None
