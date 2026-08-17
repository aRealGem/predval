"""Shared fixtures: a minimal but valid cohort + predictions pair, and helpers to break it.

Tests build their inputs on disk rather than in memory, because the loaders' job includes
reading files and hashing them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

VALID_SPEC = {
    "cohort_id": "toy",
    "version": 0,
    "subject_key": "subject_id",
    "data": "cohort.parquet",
    "outcome": {"type": "binary", "field": "label", "positive_label": 1},
    "clustering": {"field": "site"},
    "coverage": {"min_fraction": 0.95, "compare_on": "both"},
    "completeness": {"require_outcome": True, "on_violation": "drop_and_report"},
    "subgroups": [{"name": "site_group", "field": "arm"}],
    "thresholds": [0.5],
}


def write_cohort(
    tmp_path: Path,
    *,
    spec: dict | None = None,
    table: pd.DataFrame | None = None,
) -> Path:
    """Write a cohort.yaml plus its table, returning the yaml path."""
    spec = VALID_SPEC if spec is None else spec
    if table is None:
        table = pd.DataFrame(
            {
                "subject_id": [f"s{i}" for i in range(10)],
                "label": [0, 1] * 5,
                "site": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
                "arm": ["x", "y"] * 5,
            }
        )
    table.to_parquet(tmp_path / "cohort.parquet", index=False)
    path = tmp_path / "cohort.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def write_predictions(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    """Write a predictions.parquet, returning its path."""
    if frame is None:
        frame = pd.DataFrame(
            {
                "subject_id": [f"s{i}" for i in range(10)] * 2,
                "model_id": ["m1"] * 10 + ["m2"] * 10,
                "fold": [None] * 20,
                "horizon": [None] * 20,
                "predicted": [i / 10 for i in range(10)] * 2,
            }
        )
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def cohort_path(tmp_path: Path) -> Path:
    return write_cohort(tmp_path)


@pytest.fixture
def predictions_path(tmp_path: Path) -> Path:
    return write_predictions(tmp_path)
