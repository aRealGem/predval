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
    """Real models emit exact 0 and 1; every rung above rung0 must stay finite through them."""
    rng = np.random.default_rng(0)
    p, y, g = _clustered_scores(rng, slope=0.5)
    p[:50] = 1.0
    p[50:100] = 0.0
    rows, _ = R.ladder(y, p, g)
    assert rows, "ladder produced no rows"
    assert all(np.isfinite(r.value) for r in rows), "an exact 0/1 leaked a non-finite value"


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
    rows, notes = R.ladder(y, p, g)
    assert rows == [] and notes == []
