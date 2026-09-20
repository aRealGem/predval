"""Cross-implementation check: predval's point estimates against predRupdate's.

`validation/predrupdate/predrupdate_estimates.csv` holds the values
`predRupdate::pred_val_probs()` (predRupdate 0.2.1, R 4.5.0) returned for four inputs, and
`validation/predrupdate/synpm_y_p.csv.gz` holds three of those inputs. The reference values are
committed, so the SYNPM half of this test needs no R at test time; see
`validation/predrupdate/README.md` for how to regenerate them.

The tolerances asserted here are the ones registered before the comparison was first run, and
they live in the CSV beside each value rather than in this file, so the reference data and the
bar it is held to travel together.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predval.metrics import auroc, brier, calibration_intercept, calibration_slope

VALIDATION = Path(__file__).resolve().parents[1] / "validation" / "predrupdate"
GUSTO = Path(__file__).resolve().parents[1] / "examples" / "gusto"

METRICS = {
    "calibration_intercept": calibration_intercept,
    "calibration_slope": calibration_slope,
    "auroc": auroc,
    "brier": brier,
}


def _reference() -> dict[tuple[str, str, str], tuple[float, float]]:
    """(input, model, metric) -> (reference value, tolerance)."""
    with (VALIDATION / "predrupdate_estimates.csv").open(newline="") as handle:
        return {
            (row["input"], row["model"], row["metric"]): (
                float(row["value"]),
                float(row["tolerance"]),
            )
            for row in csv.DictReader(handle)
        }


REFERENCE = _reference()
SYNPM_CASES = sorted(key for key in REFERENCE if key[0] == "synpm")
GUSTO_CASES = sorted(key for key in REFERENCE if key[0] == "gusto")


def _synpm() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with gzip.open(VALIDATION / "synpm_y_p.csv.gz", "rt") as handle:
        rows = list(csv.DictReader(handle))
    y = np.array([int(row["y"]) for row in rows])
    p = {f"model{m}": np.array([float(row[f"p_model{m}"]) for row in rows]) for m in (1, 2, 3)}
    return y, p


def _gusto() -> tuple[np.ndarray, np.ndarray]:
    cohort = pd.read_parquet(GUSTO / "cohort.parquet")
    predictions = pd.read_parquet(GUSTO / "predictions.parquet")
    merged = cohort.merge(predictions, on="subject_id").sort_values("subject_id")
    return merged["label"].to_numpy().astype(int), merged["predicted"].to_numpy(dtype=float)


def test_reference_table_is_complete() -> None:
    """Sixteen comparisons: four inputs, four metrics. A dropped row must fail, not vanish."""
    assert len(REFERENCE) == 16
    assert len(SYNPM_CASES) == 12
    assert len(GUSTO_CASES) == 4
    assert {key[2] for key in REFERENCE} == set(METRICS)


def test_synpm_inputs_are_off_the_boundary() -> None:
    """logit(0) and logit(1) are infinite; the comparison is only defined away from them."""
    y, predictions = _synpm()
    assert y.size == 20_000
    assert int(y.sum()) == 2_830
    for p in predictions.values():
        assert p.min() > 0.0
        assert p.max() < 1.0


@pytest.mark.parametrize("case", SYNPM_CASES, ids=lambda case: f"{case[1]}-{case[2]}")
def test_synpm_matches_predrupdate(case: tuple[str, str, str]) -> None:
    _, model, metric = case
    reference, tolerance = REFERENCE[case]
    y, predictions = _synpm()
    value = float(METRICS[metric](y, predictions[model]))
    assert abs(value - reference) <= tolerance, (
        f"{model} {metric}: predval {value!r} vs predRupdate {reference!r}, "
        f"difference {abs(value - reference):.3e} exceeds {tolerance:.0e}"
    )


@pytest.mark.skipif(
    not (GUSTO / "cohort.parquet").exists() or not (GUSTO / "predictions.parquet").exists(),
    reason="GUSTO fixture not built; run examples/gusto/fetch_data.py + make_predictions.py",
)
@pytest.mark.parametrize("case", GUSTO_CASES, ids=lambda case: case[2])
def test_gusto_matches_predrupdate(case: tuple[str, str, str]) -> None:
    _, _, metric = case
    reference, tolerance = REFERENCE[case]
    y, p = _gusto()
    assert p.min() > 0.0
    assert p.max() < 1.0
    value = float(METRICS[metric](y, p))
    assert abs(value - reference) <= tolerance, (
        f"gusto {metric}: predval {value!r} vs predRupdate {reference!r}, "
        f"difference {abs(value - reference):.3e} exceeds {tolerance:.0e}"
    )
