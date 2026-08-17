"""Loaders must round-trip valid inputs and refuse malformed ones with actionable errors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from predval import (
    CohortSpecError,
    CoverageViolation,
    PredictionsError,
    check_coverage,
    coverage_report,
    hash_file,
    load_cohort,
    load_predictions,
)
from tests.conftest import VALID_SPEC, write_cohort, write_predictions

# --------------------------------------------------------------------------- cohort: happy path


def test_load_cohort_round_trip(cohort_path: Path) -> None:
    cohort = load_cohort(cohort_path)
    assert cohort.spec.cohort_id == "toy"
    assert cohort.n_subjects == 10
    assert cohort.spec_hash == hash_file(cohort_path)
    assert cohort.data_hash == hash_file(cohort.data_path)
    assert cohort.dropped_subjects == ()


def test_data_path_is_relative_to_the_yaml(tmp_path: Path) -> None:
    """A cohort directory must relocate as a unit."""
    nested = tmp_path / "deep" / "cohort"
    nested.mkdir(parents=True)
    path = write_cohort(nested)
    cohort = load_cohort(path)
    assert cohort.data_path == (nested / "cohort.parquet").resolve()


# ------------------------------------------------------------------------ cohort: refuse to run


def test_missing_cohort_file(tmp_path: Path) -> None:
    with pytest.raises(CohortSpecError, match="does not exist"):
        load_cohort(tmp_path / "nope.yaml")


def test_cohort_yaml_must_be_a_mapping(tmp_path: Path) -> None:
    p = tmp_path / "cohort.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(CohortSpecError, match="mapping at the top level"):
        load_cohort(p)


def test_missing_declared_column_names_what_is_present(tmp_path: Path) -> None:
    table = pd.DataFrame({"subject_id": ["s0", "s1"], "label": [0, 1], "site": ["a", "b"]})
    path = write_cohort(tmp_path, table=table)  # spec declares subgroup field "arm"
    with pytest.raises(CohortSpecError, match=r"missing columns.*arm") as exc:
        load_cohort(path)
    assert "columns present" in str(exc.value)


def test_duplicate_subject_ids_rejected(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s0", "s0", "s1"],
            "label": [0, 1, 1],
            "site": ["a", "a", "b"],
            "arm": ["x", "y", "x"],
        }
    )
    path = write_cohort(tmp_path, table=table)
    with pytest.raises(CohortSpecError, match="subject_key must be unique"):
        load_cohort(path)


def test_null_subject_id_rejected(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {"subject_id": ["s0", None], "label": [0, 1], "site": ["a", "b"], "arm": ["x", "y"]}
    )
    path = write_cohort(tmp_path, table=table)
    with pytest.raises(CohortSpecError, match="contains nulls"):
        load_cohort(path)


def test_positive_label_absent_from_outcome_column(tmp_path: Path) -> None:
    """If no subject is an event, every metric is degenerate. Say so loudly."""
    table = pd.DataFrame(
        {
            "subject_id": ["s0", "s1"],
            "label": [0, 0],
            "site": ["a", "b"],
            "arm": ["x", "y"],
        }
    )
    path = write_cohort(tmp_path, table=table)
    with pytest.raises(CohortSpecError, match="never occurs in the outcome column"):
        load_cohort(path)


def test_single_level_subgroup_rejected(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s0", "s1"],
            "label": [0, 1],
            "site": ["a", "b"],
            "arm": ["x", "x"],
        }
    )
    path = write_cohort(tmp_path, table=table)
    with pytest.raises(CohortSpecError, match="fewer than two distinct levels"):
        load_cohort(path)


# --------------------------------------------------------------------------------- completeness


def _table_with_missing_outcome() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["s0", "s1", "s2"],
            "label": [0, 1, None],
            "site": ["a", "a", "b"],
            "arm": ["x", "y", "x"],
        }
    )


def test_missing_outcome_dropped_and_reported(tmp_path: Path) -> None:
    path = write_cohort(tmp_path, table=_table_with_missing_outcome())
    cohort = load_cohort(path)
    assert cohort.n_subjects == 2
    assert cohort.dropped_subjects == ("s2",)


def test_missing_outcome_can_fail_instead(tmp_path: Path) -> None:
    spec = {
        **VALID_SPEC,
        "completeness": {"require_outcome": True, "on_violation": "fail"},
    }
    path = write_cohort(tmp_path, spec=spec, table=_table_with_missing_outcome())
    with pytest.raises(CohortSpecError, match="missing outcome"):
        load_cohort(path)


# ---------------------------------------------------------------------- predictions: happy path


def test_load_predictions_round_trip(predictions_path: Path) -> None:
    preds = load_predictions(predictions_path)
    assert preds.n_rows == 20
    assert preds.model_ids == ("m1", "m2")
    assert preds.digest == hash_file(predictions_path)
    assert list(preds.frame.columns) == [
        "subject_id",
        "model_id",
        "fold",
        "horizon",
        "predicted",
    ]


def test_null_fold_normalises_to_holdout(predictions_path: Path) -> None:
    preds = load_predictions(predictions_path)
    assert set(preds.frame["fold"]) == {"holdout"}


def test_exact_zero_and_one_are_legal(tmp_path: Path) -> None:
    """Real models emit them. The ladder eps-clips; the loader must not reject."""
    frame = pd.DataFrame(
        {
            "subject_id": ["s0", "s1"],
            "model_id": ["m1", "m1"],
            "predicted": [0.0, 1.0],
        }
    )
    preds = load_predictions(write_predictions(tmp_path, frame))
    assert sorted(preds.frame["predicted"]) == [0.0, 1.0]


def test_optional_columns_may_be_absent(tmp_path: Path) -> None:
    frame = pd.DataFrame({"subject_id": ["s0"], "model_id": ["m1"], "predicted": [0.5]})
    preds = load_predictions(write_predictions(tmp_path, frame))
    assert preds.frame["fold"].iloc[0] == "holdout"
    assert pd.isna(preds.frame["horizon"].iloc[0])


# ------------------------------------------------------------------- predictions: refuse to run


def test_missing_required_column(tmp_path: Path) -> None:
    frame = pd.DataFrame({"subject_id": ["s0"], "predicted": [0.5]})
    with pytest.raises(PredictionsError, match=r"missing required columns.*model_id"):
        load_predictions(write_predictions(tmp_path, frame))


def test_unknown_column_rejected(tmp_path: Path) -> None:
    """An unexpected column usually means a different contract, not a bonus."""
    frame = pd.DataFrame(
        {"subject_id": ["s0"], "model_id": ["m1"], "predicted": [0.5], "site": ["a"]}
    )
    with pytest.raises(PredictionsError, match="outside the v0 contract"):
        load_predictions(write_predictions(tmp_path, frame))


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_predicted_out_of_range(tmp_path: Path, bad: float) -> None:
    frame = pd.DataFrame({"subject_id": ["s0"], "model_id": ["m1"], "predicted": [bad]})
    with pytest.raises(PredictionsError, match=r"probability in \[0, 1\]"):
        load_predictions(write_predictions(tmp_path, frame))


def test_predicted_nan_rejected_pointing_at_the_absence_rule(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"subject_id": ["s0", "s1"], "model_id": ["m1", "m1"], "predicted": [0.5, float("nan")]}
    )
    with pytest.raises(PredictionsError, match="must have no row at all"):
        load_predictions(write_predictions(tmp_path, frame))


def test_predicted_infinity_rejected(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"subject_id": ["s0", "s1"], "model_id": ["m1", "m1"], "predicted": [0.5, float("inf")]}
    )
    with pytest.raises(PredictionsError, match="infinite|probability"):
        load_predictions(write_predictions(tmp_path, frame))


def test_duplicate_key_rejected(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["s0", "s0"],
            "model_id": ["m1", "m1"],
            "predicted": [0.4, 0.6],
        }
    )
    with pytest.raises(PredictionsError, match="will not silently average"):
        load_predictions(write_predictions(tmp_path, frame))


def test_null_fold_and_literal_holdout_collide(tmp_path: Path) -> None:
    """Normalisation happens before the uniqueness check, so these are the same key."""
    frame = pd.DataFrame(
        {
            "subject_id": ["s0", "s0"],
            "model_id": ["m1", "m1"],
            "fold": [None, "holdout"],
            "predicted": [0.4, 0.6],
        }
    )
    with pytest.raises(PredictionsError, match="duplicate"):
        load_predictions(write_predictions(tmp_path, frame))


def test_null_horizons_compare_equal_in_the_key(tmp_path: Path) -> None:
    """Two null horizons must count as a duplicate, not as two distinct keys."""
    frame = pd.DataFrame(
        {
            "subject_id": ["s0", "s0"],
            "model_id": ["m1", "m1"],
            "horizon": [None, None],
            "predicted": [0.4, 0.6],
        }
    )
    with pytest.raises(PredictionsError, match="duplicate"):
        load_predictions(write_predictions(tmp_path, frame))


def test_distinct_folds_are_not_duplicates(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["s0", "s0"],
            "model_id": ["m1", "m1"],
            "fold": ["oof_0", "oof_1"],
            "predicted": [0.4, 0.6],
        }
    )
    assert load_predictions(write_predictions(tmp_path, frame)).n_rows == 2


# -------------------------------------------------------------------------------------- coverage


def test_coverage_full_and_common(tmp_path: Path) -> None:
    """m2 scores half the cohort, so the intersection costs the other half."""
    cohort = load_cohort(write_cohort(tmp_path))
    frame = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(10)] + [f"s{i}" for i in range(5)],
            "model_id": ["m1"] * 10 + ["m2"] * 5,
            "predicted": [0.5] * 15,
        }
    )
    preds = load_predictions(write_predictions(tmp_path, frame))
    report = coverage_report(cohort, preds).set_index("model_id")

    assert report.loc["m1", "coverage_full"] == 1.0
    assert report.loc["m2", "coverage_full"] == 0.5
    # The common set is the 5 subjects both models scored: half the cohort.
    assert report.loc["m1", "common_fraction"] == 0.5
    assert report.loc["m2", "common_fraction"] == 0.5
    # Comparing on `common` would silently discard 5 of m1's predictions.
    assert report.loc["m1", "n_dropped_if_common"] == 5
    assert report.loc["m2", "n_dropped_if_common"] == 0
    assert not report.loc["m2", "meets_minimum"]


def test_no_meaningless_coverage_of_the_common_set(tmp_path: Path) -> None:
    """Every reported coverage must be a genuine fraction in [0, 1]."""
    cohort = load_cohort(write_cohort(tmp_path))
    frame = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(10)] + [f"s{i}" for i in range(5)],
            "model_id": ["m1"] * 10 + ["m2"] * 5,
            "predicted": [0.5] * 15,
        }
    )
    preds = load_predictions(write_predictions(tmp_path, frame))
    report = coverage_report(cohort, preds)
    for col in ("coverage_full", "common_fraction"):
        assert report[col].between(0.0, 1.0).all(), f"{col} is not a fraction"


def test_check_coverage_raises_below_the_floor(tmp_path: Path) -> None:
    cohort = load_cohort(write_cohort(tmp_path))
    frame = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(5)],
            "model_id": ["m2"] * 5,
            "predicted": [0.5] * 5,
        }
    )
    preds = load_predictions(write_predictions(tmp_path, frame))
    with pytest.raises(CoverageViolation, match="m2"):
        check_coverage(cohort, preds)


def test_absent_model_is_not_zero_filled(tmp_path: Path) -> None:
    """The absence rule: a model with no rows simply is not in the report."""
    cohort = load_cohort(write_cohort(tmp_path))
    frame = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(10)],
            "model_id": ["m1"] * 10,
            "predicted": [0.5] * 10,
        }
    )
    preds = load_predictions(write_predictions(tmp_path, frame))
    report = coverage_report(cohort, preds)
    assert list(report["model_id"]) == ["m1"]
    assert "m2" not in set(report["model_id"])
