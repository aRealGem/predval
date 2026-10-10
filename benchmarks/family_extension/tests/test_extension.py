"""Independent contract checks for the prospective family extension."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit

SPEC = importlib.util.spec_from_file_location(
    "extension", Path(__file__).parents[1] / "run_extension.py"
)
ex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ex)


def test_fixed_seed_budget():
    assert ex.SEEDS == tuple(range(20261020, 20261040))


def test_shared_design_and_families():
    pair = ex.paired_fixtures(ex.SEEDS[0])
    a, b = pair.values()
    for column in ("p", "patient", "fold"):
        np.testing.assert_array_equal(a[column], b[column])
    np.testing.assert_allclose(a.q, expit((logit(a.p) + 0.4) / 1.4), atol=1e-14)
    np.testing.assert_allclose(logit(b.q), -2.5 + 5 * b.p, atol=1e-14)
    # Same outcome uniforms imply monotonic coupling when a row's q changes.
    assert np.all(a.y[a.q <= b.q] <= b.y[a.q <= b.q])
    assert np.all(a.y[a.q >= b.q] >= b.y[a.q >= b.q])
    assert a.patient.nunique() == 120
    assert a.groupby("patient").size().between(4, 20).all()


def test_grouped_partitions_match_native():
    d = ex.paired_fixtures(ex.SEEDS[1])["logit_linear"]
    native = ex.recalibrate._grouped_folds(d.patient.to_numpy(), 5)
    for f, rows in enumerate(native):
        np.testing.assert_array_equal(rows, np.flatnonzero(d.fold.to_numpy() == f))
        assert not set(d.patient.iloc[rows]) & set(d.patient[d.fold != f])


def test_expected_loss_identity_and_oracle():
    d = ex.paired_fixtures(ex.SEEDS[2])["probability_linear"]
    p = np.full(len(d), 0.4)
    actual = ex.score(d, p)
    direct = np.mean(d.q * (1 - p) ** 2 + (1 - d.q) * p**2)
    assert actual["expected_brier"] == pytest.approx(direct, abs=1e-14)
    oracle = ex.score(d, d.q.to_numpy())
    assert oracle["risk_mse"] == 0
    assert oracle["expected_recovery"] == 1
    assert ex.score(d, d.p.to_numpy())["expected_recovery"] == 0


@pytest.mark.parametrize("family", ["logit_linear", "probability_linear"])
def test_independent_agreement_and_probabilities(family):
    d = ex.paired_fixtures(ex.SEEDS[3])[family]
    preds, statuses = ex.fit_methods(d)
    assert all(status["available"] for status in statuses.values())
    assert len(preds) == 4
    np.testing.assert_allclose(preds["predval_rung2"], preds["statsmodels_logit"], atol=1e-8)
    assert all(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in preds.values())


def test_unavailability_is_explicit():
    d = ex.frame(np.zeros(20, dtype=int), np.linspace(0.1, 0.9, 20), np.arange(20))
    preds, statuses = ex.fit_methods(d)
    assert set(preds) == {"raw"}
    assert all(not statuses[m]["available"] and statuses[m]["reason"] for m in ex.METHODS[1:])


def test_wdbc_is_not_oracle():
    d = ex.wdbc_control()
    assert len(d) == 285 and d.y.sum() == 106
    assert "q" not in d
    result = ex.score(d, d.p.to_numpy())
    assert "expected_brier" not in result and "risk_mse" not in result
    assert result["brier"] == pytest.approx(0.02947826313148458, abs=1e-12)


def test_mc_precision_and_missing_case_accounting():
    result = ex.mean_precision([1, 2, 3])
    assert result["n"] == 3 and result["mean"] == 2
    assert result["mcse"] == pytest.approx(1 / np.sqrt(3))
    assert ex.mean_precision([])["n"] == 0
    assert ex.mean_precision([1])["mcse"] is None
    rows = [
        {"family": f, "seed": s, "method": m, "available": False}
        for f in ("logit_linear", "probability_linear")
        for s in ex.SEEDS
        for m in ex.METHODS
    ]
    summary = ex.summarize(rows)
    assert all(r["attempted"] == 20 and r["n"] == 0 for r in summary["simulation_summary"])
    assert all(r["attempted"] == 20 and r["n"] == 0 for r in summary["paired_method_contrasts"])
