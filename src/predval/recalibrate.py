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

#: Metrics recomputed on the recalibrated probabilities. The ladder is a calibration statement,
#: so it is scored with the proper scoring rule and the two calibration diagnostics; the
#: rank-only metrics (AUROC, average precision) are unchanged by the monotone rungs and are not
#: re-emitted. See docs/spec.md section 4.
LADDER_METRICS = ("brier", "calibration_intercept", "calibration_slope")

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
        return _Calibrator(lambda q: _sigmoid(M.logit(q) + a))

    if rung == "rung2":
        beta = M._irls_logistic(y, z, None)
        if beta is None:
            return None
        a, b = float(beta[0]), float(beta[1])
        return _Calibrator(lambda q: _sigmoid(a + b * M.logit(q)))

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
    value: float


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """The ladder metric set on one probability vector."""
    return {
        "brier": M.brier(y, p),
        "calibration_intercept": M.calibration_intercept(y, p),
        "calibration_slope": M.calibration_slope(y, p),
    }


def ladder(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, *, max_folds: int = 5
) -> tuple[list[RungMetric], list[str]]:
    """Fit rung1-3 apparent and cross-fitted, and score each on the ladder metrics.

    Returns (rows, notes). A rung whose fit degenerates contributes NaN-valued rows and a note
    naming what was unavailable, so a missing correction is visible in the artefact rather than
    silently absent.
    """
    rows: list[RungMetric] = []
    notes: list[str] = []
    if y.size == 0 or np.unique(y).size < 2:
        return rows, notes

    g = np.unique(groups).size
    k = min(max_folds, g)

    for rung in LADDER_RUNGS:
        cal = _fit(rung, y, p)
        app = _score(y, cal(p)) if cal is not None else None
        if app is None:
            notes.append(f"{rung} apparent fit unavailable (degenerate or non-converging)")
        for metric in LADDER_METRICS:
            rows.append(RungMetric(rung, APPARENT, metric, app[metric] if app else float("nan")))

        cf_pred = _crossfit_predictions(rung, y, p, groups, k) if g >= 2 else None
        cf = _score(y, cf_pred) if cf_pred is not None else None
        if cf is None and g >= 2:
            notes.append(f"{rung} cross-fit unavailable (a fold degenerated at K={k})")
        for metric in LADDER_METRICS:
            rows.append(RungMetric(rung, CROSSFIT, metric, cf[metric] if cf else float("nan")))

    return rows, notes
