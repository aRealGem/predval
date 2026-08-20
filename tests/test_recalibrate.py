"""The recalibration ladder (docs/spec.md section 4).

The load-bearing claims: the eps-clip survives exact 0/1 from rung1 on; rung2 removes a planted
slope miscalibration on its apparent fit; the spline basis is well-formed and natural (linear in
the tails); and cross-fitting holds whole clusters out so its optimism gap is real rather than an
artefact of scoring a correction on the rows it was fitted on.
"""

from __future__ import annotations

import numpy as np
import pytest

from predval import metrics as M
from predval import recalibrate as R
from predval.evaluate import _material_nonmonotone


def _clustered_scores(rng, n_clusters=12, per=200, slope=1.0, intercept=0.0):
    """Published probabilities whose logit is `intercept + slope * true_logit`.

    The *published* calibration slope this induces is ~1/slope: a generator `slope > 1` stretches
    the logits and reads as over-confident (measured calibration slope < 1), `slope < 1` compresses
    them and reads as under-confident. Returns published p, outcomes y, and cluster labels.
    """
    p_list, y_list, g_list = [], [], []
    for c in range(n_clusters):
        z_true = rng.normal(rng.normal(0, 0.4), 1.2, size=per)
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-z_true)))
        z_pub = intercept + slope * z_true
        p_list.append(1.0 / (1.0 + np.exp(-z_pub)))
        y_list.append(y)
        g_list.append(np.full(per, f"c{c:02d}"))
    return (
        np.concatenate(p_list),
        np.concatenate(y_list).astype(np.int64),
        np.concatenate(g_list),
    )


# --------------------------------------------------------------------------- the spline basis


def test_rcs_basis_has_df_columns_and_is_linear_in_the_tails() -> None:
    """Five knots give four basis columns, and a natural spline is linear beyond the boundaries."""
    z = np.linspace(-6, 6, 400)
    knots = R._rcs_knots(z)
    basis = R._rcs_basis(z, knots)
    assert basis.shape == (400, knots.size - 1)

    # Beyond the last knot the non-linear terms are affine in z: second differences vanish.
    tail = z > knots[-1] + 0.5
    for col in range(1, basis.shape[1]):
        second_diff = np.diff(basis[tail, col], 2)
        assert np.allclose(second_diff, 0.0, atol=1e-6), f"column {col} is not linear in the tail"


def test_rcs_knots_none_when_degenerate() -> None:
    """A near-constant score cannot support a spline; rung3 must decline rather than fabricate."""
    assert R._rcs_knots(np.zeros(50)) is None


# ------------------------------------------------------------------------------- the eps-clip


def test_ladder_survives_exact_zero_and_one() -> None:
    """Real models emit exact 0 and 1; the eps-clip must keep every rung finite through them.

    Scoped to the threshold-free metrics: a thresholded metric can legitimately be NaN when a
    confusion cell is empty, which is unrelated to the clip this test guards.
    """
    rng = np.random.default_rng(0)
    p, y, g = _clustered_scores(rng, slope=0.5)
    p[:50] = 1.0
    p[50:100] = 0.0
    rows, _ = R.ladder(y, p, g)
    assert rows, "ladder produced no rows"
    free = [r for r in rows if r.threshold is None]
    assert all(np.isfinite(r.value) for r in free), "an exact 0/1 leaked a non-finite value"


# ----------------------------------------------------------------------------- rung behaviour


def test_rung2_repairs_a_planted_slope_on_apparent_fit() -> None:
    """Over-confident predictions (slope 0.5) should be pulled back toward slope 1 by rung2."""
    rng = np.random.default_rng(1)
    p, y, g = _clustered_scores(rng, slope=2.0, per=400)  # stretched logits -> over-confident

    before = M.calibration_slope(y, p)
    assert before < 0.75, "fixture is not over-confident enough to be a test"

    rows, _ = R.ladder(y, p, g)
    slope_after = next(
        r.value
        for r in rows
        if r.rung == "rung2" and r.fit_mode == "apparent" and r.metric == "calibration_slope"
    )
    assert slope_after == pytest.approx(1.0, abs=0.1), "rung2 apparent slope should be ~1"


def test_rung1_and_rung2_apparent_improve_brier_over_published() -> None:
    """A miscalibrated score's apparent recalibration cannot worsen the proper scoring rule."""
    rng = np.random.default_rng(2)
    p, y, g = _clustered_scores(rng, slope=0.5, intercept=0.6, per=400)
    published = M.brier(y, p)
    rows, _ = R.ladder(y, p, g)
    by = {(r.rung, r.fit_mode, r.metric): r.value for r in rows}
    assert by[("rung1", "apparent", "brier")] <= published + 1e-9
    assert by[("rung2", "apparent", "brier")] <= published + 1e-9


def test_apparent_beats_or_matches_crossfit_on_average() -> None:
    """Optimism is non-negative: fitting and scoring on the same rows cannot lose to holding out."""
    rng = np.random.default_rng(3)
    p, y, g = _clustered_scores(rng, slope=0.6, intercept=0.4, per=300)
    rows, _ = R.ladder(y, p, g)
    by = {(r.rung, r.fit_mode, r.metric): r.value for r in rows}
    for rung in ("rung1", "rung2", "rung3"):
        app = by[(rung, "apparent", "brier")]
        cf = by[(rung, "crossfit", "brier")]
        assert app <= cf + 1e-6, f"{rung}: apparent brier should not exceed crossfit"


# ------------------------------------------------------------------------------ cross-fitting


def test_crossfit_holds_whole_clusters_out() -> None:
    """Every fold's eval rows belong to clusters absent from that fold's train rows."""
    rng = np.random.default_rng(4)
    _, _, g = _clustered_scores(rng, n_clusters=10)
    folds = R._grouped_folds(g, k=5)
    seen: set[str] = set()
    for eval_idx in folds:
        clusters_here = set(g[eval_idx])
        assert clusters_here.isdisjoint(seen), "a cluster appeared in two folds"
        seen |= clusters_here
    assert seen == set(g), "some cluster was never held out"


def test_fold_count_is_capped_by_cluster_count() -> None:
    """With fewer clusters than the requested K, folds cannot exceed the clusters available."""
    rng = np.random.default_rng(5)
    _, _, g = _clustered_scores(rng, n_clusters=3)
    non_empty = [f for f in R._grouped_folds(g, k=5) if f.size]
    assert len(non_empty) == 3


def test_single_class_cell_yields_no_ladder_rows() -> None:
    """A stratum with one outcome class has no calibration to fit; the ladder declines cleanly."""
    rng = np.random.default_rng(6)
    p, _, g = _clustered_scores(rng)
    y = np.ones_like(p, dtype=np.int64)
    rows, report = R.ladder(y, p, g)
    assert rows == []
    assert report.suppressed == () and report.rung2_slope is None


# ------------------------------------------------------------------------- half-pair guard


def test_no_crossfit_partner_suppresses_the_whole_rung() -> None:
    """With one cluster there is nothing to hold out, so every rung is withheld, not half-shown."""
    rng = np.random.default_rng(7)
    p, y, _ = _clustered_scores(rng, n_clusters=1, per=400, slope=0.5)
    g = np.full(p.size, "only")
    rows, report = R.ladder(y, p, g)
    assert rows == [], "a rung with no cross-fitted partner must emit nothing (no half-pair)"
    assert {rung for rung, _ in report.suppressed} == set(R.LADDER_RUNGS)
    # S4.1 item 3: each withholding carries a reason -- here, nothing to hold out.
    assert {reason for _, reason in report.suppressed} == {R.R_FEW_CLUSTERS}


# ------------------------------------------------------------------ monotonicity and slope sign


def test_rank_inverting_slope_is_captured() -> None:
    """When the published score is inversely related to outcome, rung2 fits a negative slope."""
    rng = np.random.default_rng(8)
    # Invert the published logit relative to the truth: high p now means low risk.
    p, y, g = _clustered_scores(rng, slope=-1.5, per=400)
    _, report = R.ladder(y, p, g)
    assert report.rung2_slope is not None and report.rung2_slope < 0


def test_is_monotone_flags_a_reordering_transform() -> None:
    """The monotonicity check must catch a transform that reorders scores."""
    increasing = R._Calibrator(predict=lambda q: q)
    reordering = R._Calibrator(predict=lambda q: np.abs(q - 0.5))  # U-shaped: not monotone
    grid = np.linspace(0.01, 0.99, 50)
    assert R._is_monotone(increasing, grid) is True
    assert R._is_monotone(reordering, grid) is False


def test_max_local_decrease_measures_the_biggest_downward_step() -> None:
    grid = np.linspace(0.0, 1.0, 11)
    assert R._max_local_decrease(R._Calibrator(predict=lambda q: q), grid) == 0.0
    drop = R._max_local_decrease(R._Calibrator(predict=lambda q: np.abs(q - 0.5)), grid)
    assert drop == pytest.approx(0.1, abs=1e-9)  # 0.5 -> 0.4 is the largest single step down


def test_materiality_gate() -> None:
    """S4.1 item 2: a sub-tolerance wiggle is not flagged; either signal past tol is."""
    tol = 1e-3
    assert _material_nonmonotone(5e-4, 0.0, tol) is False
    assert _material_nonmonotone(2e-3, 0.0, tol) is True          # local decrease past tol
    assert _material_nonmonotone(0.0, -2e-3, tol) is True         # AUROC dropped past tol
    assert _material_nonmonotone(0.0, 2e-3, tol) is True          # AUROC rose past tol
    assert _material_nonmonotone(0.0, float("nan"), tol) is False  # non-finite dAUROC ignored


def test_rung3_rank_deficient_reason_on_narrow_support() -> None:
    """A near-constant score collapses the df=4 knots; rung3 declines with a spline-rank reason."""
    rng = np.random.default_rng(20)
    y = rng.integers(0, 2, size=200).astype(np.int64)
    p = np.full(200, 0.60)
    p[:5] = 0.61  # only two distinct scores -> knot quantiles collapse below df
    cal, reason = R._fit("rung3", y, p)
    assert cal is None and reason == R.R_RANK_DEFICIENT


# ----------------------------------------------------------------------- all metrics per rung


def test_every_rung_carries_discrimination_and_threshold_metrics() -> None:
    """S3.1: the ladder recomputes all metrics, not only calibration -- AUROC and thresholded."""
    rng = np.random.default_rng(9)
    p, y, g = _clustered_scores(rng, slope=0.5, per=300)
    rows, _ = R.ladder(y, p, g, thresholds=(0.5,))
    r2 = {(r.metric, r.threshold) for r in rows if r.rung == "rung2" and r.fit_mode == "apparent"}
    assert ("auroc", None) in r2
    assert ("average_precision", None) in r2
    assert ("sensitivity", 0.5) in r2 and ("specificity", 0.5) in r2


def test_monotone_rungs_leave_auroc_unchanged() -> None:
    """rung1/rung2 are monotone, so they cannot change discrimination -- AUROC equals rung0's."""
    rng = np.random.default_rng(10)
    p, y, g = _clustered_scores(rng, slope=0.5, per=300)
    published_auroc = M.auroc(y, p)
    rows, _ = R.ladder(y, p, g)
    for rung in ("rung1", "rung2"):
        auroc = next(
            r.value
            for r in rows
            if r.rung == rung and r.fit_mode == "apparent" and r.metric == "auroc"
        )
        assert auroc == pytest.approx(published_auroc, abs=1e-9)


# ------------------------------------------------------------------- optimism orientation


def test_optimism_is_oriented_so_positive_means_apparent_flattered() -> None:
    """Positive optimism must mean the apparent fit looked better than held-out, for both senses."""
    # Brier is a loss: apparent lower than crossfit is flattering -> positive.
    assert M.optimism("brier", apparent=0.10, crossfit=0.15) == pytest.approx(0.05)
    # AUROC is a score: apparent higher than crossfit is flattering -> positive.
    assert M.optimism("auroc", apparent=0.90, crossfit=0.85) == pytest.approx(0.05)
    # Target-valued metrics have no flattering direction.
    assert np.isnan(M.optimism("calibration_slope", 1.0, 1.2))
