"""Point estimates for binary outcomes.

Every function here takes plain numpy arrays and returns a float. They are deliberately free of
pandas, of configuration, and of any notion of subsets or strata, because they are called
thousands of times inside the bootstrap and must stay cheap.

Metrics return NaN rather than raising when the analysis set cannot support them -- a stratum
with no events has no AUROC, and that is a fact to report, not an error to crash on. The one
thing they never do is silently substitute a value that looks like a result.
"""

from __future__ import annotations

import numpy as np

#: Clip applied before any logit. See docs/spec.md sections 4.1 and 7.
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds, with the eps-clip that makes exact 0 and 1 survivable."""
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def boundary_count(p: np.ndarray) -> int:
    """Predictions exactly at 0 or 1, where logit is undefined.

    Reported per model as a data-quality signal: it measures how much work the eps-clip is
    doing rather than letting it happen silently.
    """
    return int(np.count_nonzero((p <= 0.0) | (p >= 1.0)))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve, via the rank (Mann-Whitney U) identity, ties averaged.

    Undefined when the set is all events or all non-events; returns NaN.
    """
    n = y.size
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]

    # Average ranks within tied blocks; without this, ties inflate or deflate the estimate
    # depending only on input order. Vectorised rather than looped over tie blocks: this runs
    # once per metric per bootstrap replicate, so a Python-level loop over n dominates the
    # entire evaluation.
    _, inverse, counts = np.unique(p_sorted, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    block_mean_rank = (starts + ends + 1) / 2.0  # mean of 1-based ranks start+1 .. end
    ranks = block_mean_rank[inverse]

    rank_sum_pos = ranks[y[order] == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation, as in average precision)."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y_sorted).sum() / n_pos)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better; it is a proper scoring rule."""
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def prevalence(y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.mean(y))


def _irls_logistic(
    y: np.ndarray,
    x: np.ndarray | None,
    offset: np.ndarray | None,
    *,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> np.ndarray | None:
    """Fit a small logistic regression by iteratively reweighted least squares.

    A hand-rolled fit rather than statsmodels because this runs inside the bootstrap, thousands
    of times; the statsmodels path is used for the analytic cross-checks where its covariance
    machinery is the point. Returns None if the fit does not converge or separates.

    `x` is the design matrix excluding the intercept (or None for intercept-only). `offset` is
    added to the linear predictor and not fitted.
    """
    n = y.size
    design = np.ones((n, 1)) if x is None else np.column_stack([np.ones(n), x])
    off = np.zeros(n) if offset is None else offset
    beta = np.zeros(design.shape[1])

    for _ in range(max_iter):
        eta = design @ beta + off
        # Clip the linear predictor before exp: near-separation can push |eta| past the overflow
        # bound, and mu saturates to 0/1 there anyway. The clip changes nothing in-range and lets
        # the separation guard below fire on a finite weight rather than on a warning.
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0)))
        w = mu * (1.0 - mu)
        # Near-separation drives the weights to zero and the solve to garbage.
        if not np.all(np.isfinite(w)) or w.sum() < 1e-12:
            return None
        z = eta - off + (y - mu) / np.maximum(w, 1e-12)
        wd = design * w[:, None]
        try:
            new = np.linalg.solve(design.T @ wd, wd.T @ z)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(new)):
            return None
        if np.max(np.abs(new - beta)) < tol:
            return new
        beta = new
    # A finite last iterate is not a converged fit, notably under quasi-separation.
    return None


def calibration_intercept(y: np.ndarray, p: np.ndarray) -> float:
    """Calibration-in-the-large: the intercept `a` in sigmoid(logit(p) + a), slope fixed at 1.

    Zero means the model predicts the right average risk. Positive means it under-predicts.
    Fitting with the slope fixed is what makes this a statement about level alone rather than a
    blend of level and spread.
    """
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    beta = _irls_logistic(y, None, logit(p))
    return float("nan") if beta is None else float(beta[0])


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """The slope `b` in sigmoid(a + b * logit(p)).

    One means the spread of predicted risks is right. Below one means over-confidence -- the
    predictions are too extreme in both directions, the classic signature of overfitting.
    """
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    z = logit(p)
    if np.ptp(z) == 0.0:
        return float("nan")
    beta = _irls_logistic(y, z, None)
    return float("nan") if beta is None else float(beta[1])


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """(tp, fp, tn, fn) at a decision threshold, predicting positive when p >= threshold."""
    pred = p >= threshold
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    return tp, fp, tn, fn


def sensitivity(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    tp, _, _, fn = confusion_at(y, p, threshold)
    return float(tp / (tp + fn)) if (tp + fn) else float("nan")


def specificity(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    _, fp, tn, _ = confusion_at(y, p, threshold)
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def ppv(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Positive predictive value. NaN when nothing is flagged positive -- not 0."""
    tp, fp, _, _ = confusion_at(y, p, threshold)
    return float(tp / (tp + fp)) if (tp + fp) else float("nan")


def npv(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    _, _, tn, fn = confusion_at(y, p, threshold)
    return float(tn / (tn + fn)) if (tn + fn) else float("nan")


#: Metrics with no threshold argument, by name.
THRESHOLD_FREE = {
    "auroc": auroc,
    "average_precision": average_precision,
    "brier": brier,
    "calibration_intercept": calibration_intercept,
    "calibration_slope": calibration_slope,
}

#: Metrics evaluated at each declared decision threshold.
THRESHOLDED = {
    "sensitivity": sensitivity,
    "specificity": specificity,
    "ppv": ppv,
    "npv": npv,
}

#: Optimism orientation: does a larger value mean a better model (`score`) or worse (`loss`)?
#: Calibration intercept and slope are target-valued (0 and 1), so "flattered" is undefined for
#: them and they are deliberately absent -- optimism() returns NaN rather than a misleading sign.
METRIC_ORIENTATION = {
    "auroc": "score",
    "average_precision": "score",
    "brier": "loss",
    "sensitivity": "score",
    "specificity": "score",
    "ppv": "score",
    "npv": "score",
}


def optimism(metric: str, apparent: float, crossfit: float) -> float:
    """Recalibration optimism, oriented so **positive means the apparent fit flattered itself**.

    For a score (higher is better) that is `apparent - crossfit`; for a loss (lower is better) it
    is `crossfit - apparent`. Undefined -- returns NaN -- for target-valued metrics not in
    METRIC_ORIENTATION, or when either input is not finite. See docs/spec.md section 4.3.
    """
    orient = METRIC_ORIENTATION.get(metric)
    if orient is None or not (np.isfinite(apparent) and np.isfinite(crossfit)):
        return float("nan")
    return apparent - crossfit if orient == "score" else crossfit - apparent
