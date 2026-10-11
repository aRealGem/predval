"""Benchmark contract/failure tests; no optional competitor executes on import."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pycaleva import CalibrationEvaluator

SOURCE = Path(__file__).resolve().parents[1] / "run_benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark", SOURCE)
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def test_fixtures_are_deterministic_and_patient_folds_disjoint():
    first, second = bm.fixtures(), bm.fixtures()
    assert len(first["wdbc_heldout"]) == 285
    assert first["synthetic_clustered"].patient_id.nunique() == 120
    for name, df in first.items():
        assert df.equals(second[name])
        assert df.groupby("patient_id").fold.nunique().eq(1).all()
        for fold in sorted(df.fold.unique()):
            train = df[df.fold != fold]
            test = df[df.fold == fold]
            assert set(train.patient_id).isdisjoint(test.patient_id)
            assert set(train.y) == {0, 1}


def test_crossfit_matches_independent_glm():
    for df in bm.fixtures().values():
        _, report = bm.pr.ladder(df.y.to_numpy(), df.p.to_numpy(), df.patient_id.to_numpy())
        assert np.isfinite(bm.reference_crossfit(df, logit_scale=False)).all()
        assert np.allclose(
            report.crossfit_predictions["rung2"], bm.reference_crossfit(df), atol=1e-8, rtol=0
        )


def test_missing_coverage_and_absent_models_reject(tmp_path):
    results = bm.coverage_cases(bm.fixtures()["synthetic_clustered"], tmp_path)
    assert {f["code"] for f in results["flags"]} == {
        "declared_but_absent",
        "informative_common_selection",
    }
    assert all(e["status"] == "rejected" for e in results["errors"])
    assert len(results["errors"]) == 2


def test_pycaleva_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError):
        CalibrationEvaluator(np.array([0, 1, 0, 1]), np.array([0.1, 1.2, 0.3, 0.9]), True)


def test_invalid_crossfit_is_unavailable_not_fabricated():
    # Every patient is one class, so holding either of the two patients out
    # leaves a single-class training fold. PREDVAL must withhold fitted rungs.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.8])
    _, report = bm.pr.ladder(y, p, np.array(["a", "a", "b", "b"]), max_folds=2)
    assert report.crossfit_predictions == {}
    assert len(report.suppressed) == 3


def test_quasi_separation_must_not_return_finite_converged_slope():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.5, 0.5, 0.8])
    # This quasi-separated likelihood has no finite slope MLE. Exhausting the
    # IRLS iteration budget must mark the slope unavailable.
    assert not np.isfinite(bm.pm.calibration_slope(y, p))
