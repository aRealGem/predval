"""Point estimates, checked against sklearn/statsmodels where an independent answer exists."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from predval import metrics as M


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1337)


def _sample(rng, n=500):
    z = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, p)
    return y.astype(np.int64), p


# ------------------------------------------------------------------ agreement with references


def test_auroc_matches_sklearn(rng) -> None:
    y, p = _sample(rng)
    assert M.auroc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_auroc_handles_ties(rng) -> None:
    """Heavy ties are where a naive rank implementation drifts from the reference."""
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.9, 0.9])
    assert M.auroc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_average_precision_matches_sklearn(rng) -> None:
    y, p = _sample(rng)
    assert M.average_precision(y, p) == pytest.approx(average_precision_score(y, p))


def test_brier_matches_sklearn(rng) -> None:
    y, p = _sample(rng)
    assert M.brier(y, p) == pytest.approx(brier_score_loss(y, p))


def test_calibration_slope_matches_statsmodels(rng) -> None:
    y, p = _sample(rng, 2000)
    z = M.logit(p)
    ref = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
    assert M.calibration_slope(y, p) == pytest.approx(float(ref.params[1]), rel=1e-6)


def test_calibration_intercept_matches_statsmodels(rng) -> None:
    y, p = _sample(rng, 2000)
    z = M.logit(p)
    ref = sm.GLM(y, np.ones((y.size, 1)), family=sm.families.Binomial(), offset=z).fit()
    assert M.calibration_intercept(y, p) == pytest.approx(float(ref.params[0]), rel=1e-6)


# ------------------------------------------------------------------------ behaviour under stress


def test_perfectly_calibrated_data_gives_slope_one(rng) -> None:
    """A large sample drawn from its own predicted risks must land near intercept 0, slope 1."""
    y, p = _sample(rng, 40_000)
    assert M.calibration_slope(y, p) == pytest.approx(1.0, abs=0.08)
    assert M.calibration_intercept(y, p) == pytest.approx(0.0, abs=0.05)


def test_overconfident_model_has_slope_below_one(rng) -> None:
    """Predictions pushed toward the extremes must register as over-confidence."""
    y, p = _sample(rng, 20_000)
    sharp = 1.0 / (1.0 + np.exp(-2.5 * M.logit(p)))
    assert M.calibration_slope(y, sharp) < 0.65


def test_auroc_undefined_without_both_classes() -> None:
    assert np.isnan(M.auroc(np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9])))
    assert np.isnan(M.auroc(np.array([0, 0, 0]), np.array([0.2, 0.6, 0.9])))


def test_calibration_undefined_without_both_classes() -> None:
    y, p = np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9])
    assert np.isnan(M.calibration_slope(y, p))
    assert np.isnan(M.calibration_intercept(y, p))


def test_constant_predictions_have_no_slope() -> None:
    y = np.array([0, 1, 0, 1])
    assert np.isnan(M.calibration_slope(y, np.full(4, 0.3)))


# --------------------------------------------------------------------------- the eps-clip path


def test_logit_survives_exact_zero_and_one() -> None:
    z = M.logit(np.array([0.0, 1.0, 0.5]))
    assert np.all(np.isfinite(z))
    assert z[0] < -13 and z[1] > 13
    assert z[2] == pytest.approx(0.0)


def test_calibration_finite_with_boundary_predictions() -> None:
    """Exact 0/1 predictions must not produce NaN calibration -- that is what the clip is for."""
    rng = np.random.default_rng(0)
    y, p = _sample(rng, 1000)
    p = p.copy()
    p[:20] = 0.0
    p[20:40] = 1.0
    assert np.isfinite(M.calibration_slope(y, p))
    assert np.isfinite(M.calibration_intercept(y, p))


def test_boundary_count() -> None:
    assert M.boundary_count(np.array([0.0, 1.0, 0.5, 0.999, 0.0])) == 3


# ------------------------------------------------------------------------- threshold behaviour


def test_threshold_metrics_are_consistent() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    assert M.sensitivity(y, p, 0.5) == pytest.approx(0.5)
    assert M.specificity(y, p, 0.5) == pytest.approx(0.5)
    assert M.ppv(y, p, 0.5) == pytest.approx(0.5)
    assert M.npv(y, p, 0.5) == pytest.approx(0.5)


def test_ppv_is_nan_not_zero_when_nothing_flagged() -> None:
    """Zero would claim every flagged case was wrong; NaN says none were flagged."""
    y = np.array([0, 1])
    assert np.isnan(M.ppv(y, np.array([0.1, 0.2]), 0.9))
