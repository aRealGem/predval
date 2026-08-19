"""The recalibration ladder (docs/spec.md section 4).

The ladder never touches the model. It corrects the link-scale mapping of an already-published
score `p` on the logit scale `z = logit(clip(p, eps, 1 - eps))`:

    rung0  p as published                 (no fit)
    rung1  sigmoid(z + a)                  intercept only      -- level
    rung2  sigmoid(a + b*z)                intercept + slope   -- level + spread
    rung3  sigmoid(a + rcs(z) . c)         natural cubic spline (df=4) -- non-monotone shape

Every rung above rung0 is fitted twice. **Apparent** fits the correction on the rows it is then
scored on; its improvement over rung0 is optimistic by construction and answers "how much
miscalibration is present at all". **Cross-fitted** fits on grouped K-fold train folds and scores
the held-out fold, with whole clusters held out together so a correction is never fitted on some
patches of a slide and scored on others -- that leak is the exact error predval exists to catch.
The gap `apparent - crossfit` is the optimism, and it is what a report must quote rather than the
apparent repair alone.

rung0 is not recomputed here: as-published apparent equals the S2 metrics already in the table,
and with no parameters its cross-fitted value is identical. The ladder therefore emits rung1-3
only, and evaluate.py stamps the existing as-published rows as rung0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics as M

#: Rungs the ladder fits. rung0 (as-published) is carried by the S2 metric rows, not refitted.
LADDER_RUNGS = ("rung1", "rung2", "rung3")

# Every metric is recomputed on each recalibrated mapping -- not only calibration. Threshold
# metrics move because recalibration shifts the operating point, and rung3 can reorder scores,
# so AUROC/average precision are no longer invariant. Emitting all of them is what lets the
# report show that recalibration cannot buy discrimination (rung1/rung2 AUROC == rung0) while a
# threshold's sensitivity genuinely changes. The metric set is M.THRESHOLD_FREE + M.THRESHOLDED.

APPARENT = "apparent"
CROSSFIT = "crossfit"

#: Harrell restricted-cubic-spline knot quantiles for df=4 (five knots). Fixed so rung3 is
#: reproducible and knot placement is never tuned to a result.
_RCS_QUANTILES = (0.05, 0.275, 0.5, 0.725, 0.95)


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    """Numerically stable logistic; large |eta| saturates to 0/1 without overflow warnings."""
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _rcs_knots(z: np.ndarray) -> np.ndarray | None:
    """Restricted-cubic-spline knots at fixed quantiles of the training scores.

    Returns None when the score has too few distinct values to place at least three knots -- in
    that case rung3 has no shape to fit and is reported as unavailable rather than forced.
    """
    knots = np.quantile(z, _RCS_QUANTILES)
    knots = np.unique(knots)
    return knots if knots.size >= 3 else None


def _rcs_basis(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell restricted cubic spline basis: columns [z, s_1 .. s_{k-2}], k = len(knots).

    A natural (restricted) cubic spline is linear beyond the boundary knots, which keeps the
    correction from swinging wildly in the sparse tails of the logit scale where confident
    predictions live. The basis excludes the intercept; the logistic fit supplies it.
    """
    k = knots.size
    t1, tk, tkm1 = knots[0], knots[-1], knots[-2]
    denom = tk - tkm1
    scale = (tk - t1) ** 2

    def cube(a: np.ndarray) -> np.ndarray:
        return np.where(a > 0, a, 0.0) ** 3

    cols = [z]
    for j in range(k - 2):
        tj = knots[j]
        term = (
            cube(z - tj)
            - cube(z - tkm1) * (tk - tj) / denom
            + cube(z - tk) * (tkm1 - tj) / denom
        ) / scale
        cols.append(term)
    return np.column_stack(cols)


@dataclass(frozen=True)
class _Calibrator:
    """A fitted rung: maps published `p` to a corrected probability. Purely a link-scale map."""

    predict: object  # Callable[[np.ndarray], np.ndarray] over published p
    #: The fitted logit-scale slope, kept so a rank-inverting fit (slope < 0) can be flagged.
    #: 1.0 for rung1 (slope fixed), the fitted `b` for rung2, None for rung3 (no single slope).
    slope: float | None = None

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self.predict(p)  # type: ignore[operator]


def _fit(rung: str, y: np.ndarray, p: np.ndarray) -> _Calibrator | None:
    """Fit one rung on (y, p). Returns None if the fit degenerates or fails to converge.

    A None here is not an error: near-separation and single-class folds are real, and the honest
    response is an unavailable rung (NaN downstream) rather than a fabricated correction.
    """
    if y.size == 0 or np.unique(y).size < 2:
        return None
    z = M.logit(p)
    if np.ptp(z) == 0.0:
        return None

    if rung == "rung1":
        beta = M._irls_logistic(y, None, z)
        if beta is None:
            return None
        a = float(beta[0])
        return _Calibrator(lambda q: _sigmoid(M.logit(q) + a), slope=1.0)

    if rung == "rung2":
        beta = M._irls_logistic(y, z, None)
        if beta is None:
            return None
        a, b = float(beta[0]), float(beta[1])
        return _Calibrator(lambda q: _sigmoid(a + b * M.logit(q)), slope=b)

    if rung == "rung3":
        knots = _rcs_knots(z)
        if knots is None:
            return None
        beta = M._irls_logistic(y, _rcs_basis(z, knots), None)
        if beta is None:
            return None
        a, coef = float(beta[0]), np.asarray(beta[1:], dtype=float)
        return _Calibrator(lambda q: _sigmoid(a + _rcs_basis(M.logit(q), knots) @ coef))

    raise ValueError(f"unknown rung {rung!r}")


def _is_monotone(cal: _Calibrator, p: np.ndarray, tol: float = 1e-9) -> bool:
    """Is the fitted transform non-decreasing across the observed scores?

    rung1 and rung2 (with slope > 0) are monotone by construction; rung3's spline is not
    guaranteed to be, and a non-monotone recalibration reorders patients -- it can change AUROC
    and is worth flagging rather than applying silently. Checked on the distinct published
    values, which span the same order as their logits.
    """
    u = np.unique(p[np.isfinite(p)])
    if u.size < 2:
        return True
    mapped = cal(u)
    return bool(np.all(np.diff(mapped) >= -tol))


def _grouped_folds(groups: np.ndarray, k: int) -> list[np.ndarray]:
    """Assign whole clusters to k folds; return the eval row indices per fold.

    Deterministic: clusters are sorted and dealt round-robin, which balances cluster *count*
    across folds without any randomness to seed. Whole-cluster assignment is what keeps the
    cross-fit honest -- a slide is entirely train or entirely eval, never split.
    """
    labels = np.unique(groups)
    fold_of = {lab: i % k for i, lab in enumerate(labels)}
    assign = np.array([fold_of[g] for g in groups])
    return [np.flatnonzero(assign == f) for f in range(k)]


def _crossfit_predictions(
    rung: str, y: np.ndarray, p: np.ndarray, groups: np.ndarray, k: int
) -> np.ndarray | None:
    """Out-of-fold corrected probabilities: fit on the other folds, score the held-out one.

    Returns None if any fold cannot be fitted -- a partial cross-fit would score some rows with a
    correction and others without, which is not comparable, so the whole rung is reported
    unavailable instead.
    """
    out = np.full(p.size, np.nan)
    for eval_idx in _grouped_folds(groups, k):
        if eval_idx.size == 0:
            continue
        train = np.ones(p.size, dtype=bool)
        train[eval_idx] = False
        cal = _fit(rung, y[train], p[train])
        if cal is None:
            return None
        out[eval_idx] = cal(p[eval_idx])
    return None if not np.all(np.isfinite(out)) else out


@dataclass(frozen=True)
class RungMetric:
    rung: str
    fit_mode: str
    metric: str
    threshold: float | None
    value: float


@dataclass(frozen=True)
class LadderReport:
    """Quality signals about the fitted ladder, for flags the evaluator labels with the cell."""

    #: Fitted apparent rung2 slope, or None if rung2 was suppressed. Negative == rank-inverting.
    rung2_slope: float | None
    #: Whether the apparent rung3 transform is monotone; None if rung3 was suppressed.
    rung3_monotone: bool | None
    #: Rungs emitted for neither fit mode because one half of the pair was unavailable (§4.3).
    suppressed_rungs: tuple[str, ...]


def _score_all(
    y: np.ndarray, p: np.ndarray, thresholds: tuple[float, ...]
) -> dict[tuple[str, float | None], float]:
    """Every metric -- threshold-free and thresholded -- on one probability vector."""
    out: dict[tuple[str, float | None], float] = {}
    for name, fn in M.THRESHOLD_FREE.items():
        out[(name, None)] = fn(y, p)
    for name, fn in M.THRESHOLDED.items():
        for t in thresholds:
            out[(name, t)] = fn(y, p, t)
    return out


def ladder(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.5,),
    max_folds: int = 5,
) -> tuple[list[RungMetric], LadderReport]:
    """Fit rung1-3 apparent and cross-fitted, scoring every metric on each corrected mapping.

    Half-pair guard (§4.3): a rung is emitted only if BOTH its apparent and cross-fitted mappings
    are available. If either degenerates the whole rung is suppressed -- an apparent number with
    no held-out companion is exactly the flattering figure the ladder exists to discipline, so it
    is withheld rather than shown alone. Suppressed rungs are named in the returned report.
    """
    rows: list[RungMetric] = []
    if y.size == 0 or np.unique(y).size < 2:
        return rows, LadderReport(None, None, ())

    g = np.unique(groups).size
    k = min(max_folds, g)
    rung2_slope: float | None = None
    rung3_monotone: bool | None = None
    suppressed: list[str] = []

    for rung in LADDER_RUNGS:
        cal = _fit(rung, y, p)
        cf_pred = _crossfit_predictions(rung, y, p, groups, k) if g >= 2 else None
        if cal is None or cf_pred is None:
            suppressed.append(rung)
            continue

        if rung == "rung2":
            rung2_slope = cal.slope
        if rung == "rung3":
            rung3_monotone = _is_monotone(cal, p)

        for (metric, t), value in _score_all(y, cal(p), thresholds).items():
            rows.append(RungMetric(rung, APPARENT, metric, t, value))
        for (metric, t), value in _score_all(y, cf_pred, thresholds).items():
            rows.append(RungMetric(rung, CROSSFIT, metric, t, value))

    return rows, LadderReport(rung2_slope, rung3_monotone, tuple(suppressed))
