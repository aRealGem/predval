"""findings.json (item 4): byte-stable, schema-valid, and carrying the verdict + gains.

The machine-readable report is only useful if a consumer can trust it: identical bytes on
regeneration, and a shape that validates against the checked-in JSON Schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from predval import build_findings, dumps_findings, evaluate, load_cohort, load_predictions
from predval.findings import FindingsDocument, findings_schema

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "findings.schema.json"


def _build(tmp_path: Path):
    rng = np.random.default_rng(11)
    rows, preds = [], []
    for c in range(8):
        shift = rng.normal(0, 0.8)
        for i in range(80):
            z = rng.normal(shift, 1.0)
            p = 1.0 / (1.0 + np.exp(-z))
            sid = f"c{c:02d}_{i:03d}"
            rows.append({"subject_id": sid, "label": int(rng.binomial(1, p)),
                         "wsi": f"c{c:02d}", "arm": "x" if c % 2 == 0 else "y"})
            preds.append({"subject_id": sid, "model_id": "good", "predicted": float(p)})
            preds.append({"subject_id": sid, "model_id": "weak",
                          "predicted": float(1.0 / (1.0 + np.exp(-2.0 * z)))})
    pd.DataFrame(rows).to_parquet(tmp_path / "cohort.parquet", index=False)
    pd.DataFrame(preds).to_parquet(tmp_path / "predictions.parquet", index=False)
    spec = {
        "cohort_id": "toy-findings", "version": 0, "subject_key": "subject_id",
        "data": "cohort.parquet",
        "outcome": {"type": "binary", "field": "label", "positive_label": 1},
        "clustering": {"field": "wsi"},
        "coverage": {"min_fraction": 0.5, "compare_on": "both"},
        "completeness": {"require_outcome": True, "on_violation": "drop_and_report"},
        "subgroups": [{"name": "arm_group", "field": "arm"}],
        "thresholds": [0.5], "uncertainty": {"n_boot": 40, "seed": 1337},
        "expected_models": ["good", "weak", "ghost"],
    }
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(spec))
    cohort = load_cohort(tmp_path / "cohort.yaml")
    return evaluate(cohort, load_predictions(tmp_path / "predictions.parquet"))


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("findings"))


def test_findings_regenerate_byte_identical(evaluation) -> None:
    a = dumps_findings(evaluation, git_commit="deadbeef")
    b = dumps_findings(evaluation, git_commit="deadbeef")
    assert a == b


def test_findings_validates_against_the_model(evaluation) -> None:
    doc = build_findings(evaluation, git_commit="deadbeef")
    FindingsDocument.model_validate(doc)  # must not raise
    assert doc["schema_version"]
    assert doc["provenance"]["git_commit"] == "deadbeef"


def test_checked_in_schema_matches_the_model() -> None:
    """The checked-in JSON Schema must be exactly what the model generates -- no drift."""
    assert SCHEMA_PATH.exists(), "schema/findings.schema.json must be checked in"
    on_disk = json.loads(SCHEMA_PATH.read_text())
    assert on_disk == findings_schema()


def test_findings_carry_verdict_and_paired_gains(evaluation) -> None:
    doc = build_findings(evaluation, git_commit="x")
    members = {m["model_id"]: m for m in doc["members"]}
    assert set(members) == {"good", "weak"}
    weak = members["weak"]
    assert weak["verdict"]["best_rung"] in {"rung0", "rung1", "rung2", "rung3"}
    assert weak["verdict"]["gauge_label"]
    # S6.1: the two-axis taxonomy carries independently of best_rung.
    assert isinstance(weak["verdict"]["axis_a_miscalibrated"], bool)
    assert weak["verdict"]["axis_b"] in {"demonstrated", "unproven", "counterproductive"}
    rungs = {g["rung"] for g in weak["recalibration_gains"]}
    assert rungs and rungs <= {"rung1", "rung2", "rung3"}
    for g in weak["recalibration_gains"]:
        assert g["ci_method"] == "paired_cluster_bootstrap"
    # D2 field is present and boolean on every member.
    assert isinstance(weak["analytic_ci_truncated"], bool)


def test_findings_roster_reports_the_absent_model(evaluation) -> None:
    doc = build_findings(evaluation, git_commit="x")
    assert doc["roster"]["absent"] == ["ghost"]
    assert doc["roster"]["declared"] == 3 and doc["roster"]["present"] == 2
