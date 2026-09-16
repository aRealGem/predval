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

from dataclasses import dataclass, field

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
            cube(z - tj) - cube(z - tkm1) * (tk - tj) / denom + cube(z - tk) * (tkm1 - tj) / denom
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


# Reasons a rung fit is withheld, surfaced verbatim in the recalibration_unavailable flag so a
# reader sees *why* a rung is missing rather than only that it is (§4.3, S4.1 item 3).
R_SINGLE_CLASS = "single-class fold"
R_NO_VARIATION = "no score variation"
R_NON_CONVERGENCE = "non-convergence"
R_RANK_DEFICIENT = "rank-deficient spline design"
R_FEW_CLUSTERS = "G<2 (nothing to hold out)"


def _fit(rung: str, y: np.ndarray, p: np.ndarray) -> tuple[_Calibrator | None, str | None]:
    """Fit one rung on (y, p). Returns (calibrator, None) or (None, reason).

    A None calibrator is not an error: near-separation and single-class folds are real, and the
    honest response is an unavailable rung (with the reason recorded) rather than a fabricated
    correction. The reason distinguishes the four failure modes the report is asked to name.
    """
    if y.size == 0 or np.unique(y).size < 2:
        return None, R_SINGLE_CLASS
    z = M.logit(p)
    if np.ptp(z) == 0.0:
        return None, R_NO_VARIATION

    if rung == "rung1":
        beta = M._irls_logistic(y, None, z)
        if beta is None:
            return None, R_NON_CONVERGENCE
        a = float(beta[0])
        return _Calibrator(lambda q: _sigmoid(M.logit(q) + a), slope=1.0), None

    if rung == "rung2":
        beta = M._irls_logistic(y, z, None)
        if beta is None:
            return None, R_NON_CONVERGENCE
        a, b = float(beta[0]), float(beta[1])
        return _Calibrator(lambda q: _sigmoid(a + b * M.logit(q)), slope=b), None

    if rung == "rung3":
        # Two distinct rung3 failure modes, distinguished (diagnosis 2026-08-19, docs/):
        #   - knot collapse: too few distinct logit values to place df=4 knots -> rank-deficient.
        #   - IRLS separation: knots are fine but the df=4 spline logistic separates / goes
        #     singular on this (sub)set -- e.g. a leave-one-slide-out fold whose training scores
        #     are very flat (near-collinear basis) or saturated near 0/1 (tail separation).
        knots = _rcs_knots(z)
        if knots is None:
            return None, R_RANK_DEFICIENT
        beta = M._irls_logistic(y, _rcs_basis(z, knots), None)
        if beta is None:
            return None, R_NON_CONVERGENCE
        a, coef = float(beta[0]), np.asarray(beta[1:], dtype=float)
        return _Calibrator(lambda q: _sigmoid(a + _rcs_basis(M.logit(q), knots) @ coef)), None

    raise ValueError(f"unknown rung {rung!r}")


def _max_local_decrease(cal: _Calibrator, p: np.ndarray) -> float:
    """The largest downward step of the fitted transform across the distinct observed scores.

    0.0 for a monotone transform. This is the magnitude that decides, against monotone_tol,
    whether a rung3 non-monotonicity is *material* (reorders patients enough to matter) or a
    sub-tolerance wiggle to record but not flag (§4.6).
    """
    u = np.unique(p[np.isfinite(p)])
    if u.size < 2:
        return 0.0
    d = np.diff(cal(u))
    drops = -d[d < 0.0]
    return float(drops.max()) if drops.size else 0.0


def _is_monotone(cal: _Calibrator, p: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether the transform is non-decreasing across the scores (largest decrease <= tol)."""
    return _max_local_decrease(cal, p) <= tol


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
) -> tuple[np.ndarray | None, str | None]:
    """Out-of-fold corrected probabilities: fit on the other folds, score the held-out one.

    Returns (predictions, None), or (None, reason) if any fold cannot be fitted -- a partial
    cross-fit would score some rows with a correction and others without, which is not comparable,
    so the whole rung is reported unavailable with the failing fold's reason.
    """
    out = np.full(p.size, np.nan)
    for eval_idx in _grouped_folds(groups, k):
        if eval_idx.size == 0:
            continue
        train = np.ones(p.size, dtype=bool)
        train[eval_idx] = False
        cal, reason = _fit(rung, y[train], p[train])
        if cal is None:
            return None, reason
        out[eval_idx] = cal(p[eval_idx])
    if not np.all(np.isfinite(out)):
        return None, R_NON_CONVERGENCE
    return out, None


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
    #: Largest downward step of the apparent rung3 transform; None if rung3 was suppressed. Always
    #: recorded (in the artefact); only flagged when it, or delta_auroc_rung3, clears monotone_tol.
    rung3_max_local_decrease: float | None
    #: apparent rung3 AUROC minus rung0 AUROC; None if rung3 suppressed. The discrimination that
    #: the (possibly non-monotone) spline moved -- the outcome-level materiality signal.
    delta_auroc_rung3: float | None
    #: (rung, reason) for each rung emitted in neither fit mode because a half of the pair was
    #: unavailable (§4.3). The reason is one of the R_* strings above.
    suppressed: tuple[tuple[str, str], ...]
    #: Out-of-fold corrected probabilities per emitted rung, aligned to the input rows. The
    #: evaluator uses these to bootstrap the paired cross-fit gain (rung0 - rung_r) over slides
    #: (§4.5, item 1) without refitting -- the mapping is already held-out. Empty when no rung
    #: converged. rung0 is not here: its cross-fit equals its as-published p.
    crossfit_predictions: dict[str, np.ndarray] = field(default_factory=dict)


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
        return rows, LadderReport(None, None, None, ())

    g = np.unique(groups).size
    k = min(max_folds, g)
    auroc0 = M.auroc(y, p)
    rung2_slope: float | None = None
    rung3_mld: float | None = None
    rung3_dauroc: float | None = None
    suppressed: list[tuple[str, str]] = []
    crossfit_preds: dict[str, np.ndarray] = {}

    for rung in LADDER_RUNGS:
        cal, reason_apparent = _fit(rung, y, p)
        if g >= 2:
            cf_pred, reason_crossfit = _crossfit_predictions(rung, y, p, groups, k)
        else:
            cf_pred, reason_crossfit = None, R_FEW_CLUSTERS
        if cal is None or cf_pred is None:
            suppressed.append((rung, reason_apparent or reason_crossfit or R_NON_CONVERGENCE))
            continue

        if rung == "rung2":
            rung2_slope = cal.slope
        crossfit_preds[rung] = cf_pred

        apparent_scores = _score_all(y, cal(p), thresholds)
        if rung == "rung3":
            rung3_mld = _max_local_decrease(cal, p)
            r3_auroc = apparent_scores[("auroc", None)]
            rung3_dauroc = (
                float(r3_auroc - auroc0)
                if np.isfinite(r3_auroc) and np.isfinite(auroc0)
                else float("nan")
            )

        for (metric, t), value in apparent_scores.items():
            rows.append(RungMetric(rung, APPARENT, metric, t, value))
        for (metric, t), value in _score_all(y, cf_pred, thresholds).items():
            rows.append(RungMetric(rung, CROSSFIT, metric, t, value))

    return rows, LadderReport(
        rung2_slope, rung3_mld, rung3_dauroc, tuple(suppressed), crossfit_preds
    )
