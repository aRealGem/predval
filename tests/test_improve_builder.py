"""Synthetic regression tests for the IMPROVE fixture's patient/CV integrity guard.

No source archive, generated fixture, or opt-in golden run is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "examples" / "improve" / "build_fixture.py"
SPEC = importlib.util.spec_from_file_location("improve_build_fixture", PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def rf_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Patient": ["patient_a", "patient_a", "patient_b"],
            "Partition": [0, 0, 1],
            "Mut_peptide": ["AAAAAAAAA", "CCCCCCCC", "DDDDDDDDD"],
            "HLA_allele": ["HLA-A01:01"] * 3,
            "cohort": ["synthetic"] * 3,
            "response": [0, 1, 0],
            "prediction_rf": [0.1, 0.8, 0.2],
        }
    )


def write_rf_tables(src: Path, frame: pd.DataFrame) -> None:
    for sub, name in builder.RF_MODELS.values():
        path = src / "5_fold_CV" / sub / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)


@pytest.mark.parametrize("partitions", [[0, 0, 1], ["01", "1.0", "2"]])
def test_patient_partition_guard_preserves_valid_input(rf_frame, partitions) -> None:
    rf_frame["Partition"] = partitions
    before = rf_frame.copy(deep=True)
    builder.validate_patient_partitions(rf_frame, "test_model")
    pd.testing.assert_frame_equal(rf_frame, before)


@pytest.mark.parametrize("column", ["Patient", "Partition"])
def test_patient_partition_guard_requires_columns(rf_frame, column) -> None:
    with pytest.raises(SystemExit, match="test_model: missing Patient or Partition column"):
        builder.validate_patient_partitions(rf_frame.drop(columns=column), "test_model")


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_patient_partition_guard_rejects_missing_patient(rf_frame, missing) -> None:
    rf_frame.loc[0, "Patient"] = missing
    with pytest.raises(SystemExit, match="test_model: missing patient identifiers"):
        builder.validate_patient_partitions(rf_frame, "test_model")


@pytest.mark.parametrize("invalid", [None, "unknown", 0.5, float("inf")])
def test_patient_partition_guard_rejects_invalid_partition(rf_frame, invalid) -> None:
    rf_frame["Partition"] = rf_frame["Partition"].astype(object)
    rf_frame.loc[0, "Partition"] = invalid
    with pytest.raises(SystemExit, match="test_model: CV partitions must be non-missing integers"):
        builder.validate_patient_partitions(rf_frame, "test_model")


@pytest.mark.parametrize("model_id", list(builder.RF_MODELS))
def test_builder_rejects_split_patient_in_every_model_before_writing(
    tmp_path, rf_frame, model_id
) -> None:
    src = tmp_path / "source"
    out = tmp_path / "output"
    write_rf_tables(src, rf_frame)
    rf_frame.loc[1, "Partition"] = 1
    sub, name = builder.RF_MODELS[model_id]
    rf_frame.to_csv(src / "5_fold_CV" / sub / name, sep="\t", index=False)
    with pytest.raises(
        SystemExit, match=f"{model_id}: a patient appears in multiple CV partitions"
    ):
        builder.main(["--src", str(src), "--out", str(out)])
    assert not out.exists(), "invalid CV metadata must fail before either fixture is written"


def test_builder_preserves_valid_fixture_values(tmp_path, rf_frame, monkeypatch) -> None:
    src = tmp_path / "source"
    out = tmp_path / "output"
    write_rf_tables(src, rf_frame)
    nn = rf_frame.copy()
    nn["subject_id"] = builder._sid(nn)
    nn["target"] = nn["response"]
    nn["predicted"] = nn["prediction_rf"]
    monkeypatch.setattr(builder, "load_nnalign", lambda _: nn)

    assert builder.main(["--src", str(src), "--out", str(out)]) == 0
    cohort = pd.read_parquet(out / "cohort.parquet")
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert cohort["patient"].tolist() == rf_frame["Patient"].tolist()
    assert cohort["cv_partition"].tolist() == rf_frame["Partition"].tolist()
    assert cohort["response"].tolist() == rf_frame["response"].tolist()
    assert cohort["subject_id"].tolist() == [
        "patient_a|AAAAAAAAA|HLA-A01:01",
        "patient_a|CCCCCCCC|HLA-A01:01",
        "patient_b|DDDDDDDDD|HLA-A01:01",
    ]
    assert len(predictions) == len(rf_frame) * (len(builder.RF_MODELS) + 1)
    for _, group in predictions.groupby("model_id"):
        assert group["predicted"].tolist() == rf_frame["prediction_rf"].tolist()
