"""The HTML report (docs/spec.md sections 4.4 and 6).

These tests are about integrity, not appearance: the framing block is present and unconditional,
no ladder gain is shown without its pending-interval marker, by-construction identities are
labelled, the roster gap is loud, fragility is never dressed as an interval, and the same
artefacts render byte-identically whether from memory or from disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from predval import (
    evaluate,
    load_cohort,
    load_predictions,
    render,
    render_evaluation,
    render_from_dir,
)
from predval.report import FRAMING, PENDING, build_context

N_CLUSTERS = 8
PER_CLUSTER = 80


def _build(tmp_path: Path, expected_models=None):
    """A small clustered cohort: 2 models, a 2-level subgroup (4 clusters each -> gated)."""
    rng = np.random.default_rng(11)
    rows, preds = [], []
    for c in range(N_CLUSTERS):
        shift = rng.normal(0, 0.8)
        for i in range(PER_CLUSTER):
            z = rng.normal(shift, 1.0)
            p = 1.0 / (1.0 + np.exp(-z))
            sid = f"c{c:02d}_{i:03d}"
            rows.append({"subject_id": sid, "label": int(rng.binomial(1, p)),
                         "wsi": f"c{c:02d}", "arm": "x" if c % 2 == 0 else "y"})
            preds.append({"subject_id": sid, "model_id": "good", "predicted": float(p)})
            # weak: over-confident (stretched logits) so the ladder has something to diagnose
            preds.append({"subject_id": sid, "model_id": "weak",
                          "predicted": float(1.0 / (1.0 + np.exp(-2.0 * z)))})

    pd.DataFrame(rows).to_parquet(tmp_path / "cohort.parquet", index=False)
    pd.DataFrame(preds).to_parquet(tmp_path / "predictions.parquet", index=False)
    spec = {
        "cohort_id": "toy-report", "version": 0, "subject_key": "subject_id",
        "data": "cohort.parquet",
        "outcome": {"type": "binary", "field": "label", "positive_label": 1},
        "clustering": {"field": "wsi"},
        "coverage": {"min_fraction": 0.5, "compare_on": "both"},
        "completeness": {"require_outcome": True, "on_violation": "drop_and_report"},
        "subgroups": [{"name": "arm_group", "field": "arm"}],
        "thresholds": [0.5],
        "uncertainty": {"n_boot": 12, "seed": 1337},
    }
    if expected_models is not None:
        spec["expected_models"] = expected_models
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(spec))
    cohort = load_cohort(tmp_path / "cohort.yaml")
    preds = load_predictions(tmp_path / "predictions.parquet")
    return evaluate(cohort, preds)


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("report")
    return _build(tmp, expected_models=["good", "weak", "ghost"])


@pytest.fixture(scope="module")
def html(evaluation):
    return render_evaluation(evaluation)


# ----------------------------------------------------------------------------- the framing rule


def test_framing_block_is_present_and_unconditional(html) -> None:
    assert FRAMING in html
    assert "Recalibration does not and cannot improve discrimination." in html
    # It appears before the first section heading -- nothing precedes the framing.
    assert html.index(FRAMING) < html.index("1. Coverage and roster")


# ------------------------------------------------------------------------------ naked-delta ban


def test_every_ladder_gain_carries_the_pending_interval_marker(evaluation) -> None:
    ctx = build_context(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest
    )
    assert ctx["calibration_rows"], "expected ladder rows to test"
    for row in ctx["calibration_rows"]:
        assert row["gain"].endswith(PENDING), f"gain shown without pending marker: {row}"
        assert row["optimism"].endswith(PENDING), f"optimism shown without pending marker: {row}"


def test_by_construction_identities_are_labelled(html) -> None:
    assert "by construction" in html
    assert "apparent intercept" in html and "apparent slope" in html


# --------------------------------------------------------------------------------- roster


def test_roster_gap_is_loud(html) -> None:
    assert "declared" in html and "present" in html
    assert "ghost" in html  # declared but absent
    assert "declared but no predictions" in html


# --------------------------------------------------------------------- fragility is not a CI


def test_fragility_is_labelled_not_an_interval(html) -> None:
    assert "not a confidence interval" in html.lower() or "not</strong> an interval" in html


def test_fragility_rows_have_no_interval_fields(evaluation) -> None:
    ctx = build_context(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest
    )
    for row in ctx["fragility_rows"]:
        assert set(row) == {"model", "max_abs_delta", "culprit"}


# ------------------------------------------------------------------ unit-of-analysis exhibit


def test_unit_of_analysis_exhibit_present(html) -> None:
    assert "Unit-of-analysis exhibit" in html
    assert "naive" in html and "ratio" in html.lower()


# --------------------------------------------------------------------- gated subgroup note


def test_gated_subgroup_ladder_is_flagged_in_the_report(html) -> None:
    """The 4-cluster subgroups are below the gate; the report must say the ladder was suppressed."""
    assert "Ladder suppressed on gated subgroups" in html
    assert "recalibration_suppressed" not in html or "suppressed" in html.lower()


# ----------------------------------------------------------------------------- provenance


def test_provenance_carries_hashes_seed_and_versions(html, evaluation) -> None:
    for digest in evaluation.manifest["inputs"].values():
        assert digest in html, "an input hash is missing from the report"
    assert "predval" in html
    assert str(evaluation.manifest["uncertainty"]["seed"]) in html


# ----------------------------------------------------------------------------- determinism


def test_render_is_deterministic(evaluation) -> None:
    a = render_evaluation(evaluation)
    b = render_evaluation(evaluation)
    assert a == b


def test_in_memory_and_from_disk_renders_are_byte_identical(evaluation, tmp_path: Path) -> None:
    """The DoD: a report regenerates from the written artefacts, identically to the live render."""
    evaluation.write(tmp_path / "out")
    from_memory = render_evaluation(evaluation)
    from_disk = render_from_dir(tmp_path / "out")
    assert from_memory == from_disk


def test_render_accepts_bare_dataframes(evaluation) -> None:
    """render() works from the four artefacts directly, not only from an Evaluation object."""
    html = render(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest
    )
    assert FRAMING in html
