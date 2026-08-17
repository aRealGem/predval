"""The PCam fixture must load through the real contract and preserve the S0 invariants.

These tests skip when the fixture has not been built, since the parquet files are generated
from a local campaign directory and are deliberately not committed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from predval import coverage_report, load_cohort, load_predictions

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "pcam"
COHORT_YAML = FIXTURE / "cohort.yaml"
PREDICTIONS = FIXTURE / "predictions.parquet"

pytestmark = pytest.mark.skipif(
    not (FIXTURE / "cohort.parquet").exists() or not PREDICTIONS.exists(),
    reason="PCam fixture not built; run examples/pcam/build_fixture.py",
)

N_SUBJECTS = 19_999
N_POSITIVES = 5_796
N_SLIDES = 22
N_MODELS = 15


@pytest.fixture(scope="module")
def cohort():
    return load_cohort(COHORT_YAML)


@pytest.fixture(scope="module")
def predictions():
    return load_predictions(PREDICTIONS)


def test_cohort_shape(cohort) -> None:
    assert cohort.n_subjects == N_SUBJECTS
    assert int((cohort.table["label"] == 1).sum()) == N_POSITIVES
    assert cohort.dropped_subjects == ()


def test_clustering_unit_is_the_slide(cohort) -> None:
    """19,999 subjects but only 22 independent units -- the reason clustering exists."""
    assert cohort.spec.clustering is not None
    assert cohort.table[cohort.spec.clustering.field].nunique() == N_SLIDES


def test_prespecified_subgroup_is_usable(cohort) -> None:
    counts = cohort.table["domain"].value_counts()
    assert len(counts) == 2
    assert counts.min() > 1000, "both scanner strata must be large enough to analyse"


def test_predictions_shape(predictions) -> None:
    assert len(predictions.model_ids) == N_MODELS
    assert predictions.n_rows == N_MODELS * N_SUBJECTS


def test_every_model_scored_the_same_subjects(predictions) -> None:
    sizes = predictions.frame.groupby("model_id")["subject_id"].nunique()
    assert set(sizes) == {N_SUBJECTS}


def test_all_folds_are_holdout(predictions) -> None:
    """The campaign used one fixed GroupShuffleSplit, not k-fold, despite the 'oof_' filenames."""
    assert set(predictions.frame["fold"]) == {"holdout"}


def test_lost_member_is_absent_not_zero_filled(predictions) -> None:
    """p4m_reg's predictions were permanently lost. Absence must survive the round trip."""
    assert "p4m_reg" not in predictions.model_ids
    assert "p4m_reg_vl" in predictions.model_ids, "the recovered member should be present"


def test_exact_zero_and_one_predictions_survive(predictions) -> None:
    """These are why the recalibration ladder must eps-clip from rung1."""
    p = predictions.frame["predicted"]
    assert ((p <= 0.0) | (p >= 1.0)).sum() > 0


def test_full_coverage_for_every_present_model(cohort, predictions) -> None:
    report = coverage_report(cohort, predictions)
    assert len(report) == N_MODELS
    assert (report["coverage_full"] == 1.0).all()
    assert report["meets_minimum"].all()


def test_hashes_are_recorded(cohort, predictions) -> None:
    assert len(cohort.spec_hash) == 64
    assert len(predictions.digest) == 64
