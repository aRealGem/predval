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


#: Distinguishes "clustering_name not passed" (keep the old undeclared-name behaviour) from an
#: explicit `clustering_name=None` (also undeclared, but on purpose -- S6 items 5/14).
_UNSET = object()


def _build(tmp_path: Path, expected_models=None, clustering_name=_UNSET, single_model=False):
    """A small clustered cohort: 2 models (or 1, if single_model), a 2-level subgroup (4 clusters
    each -> gated)."""
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
            if not single_model:
                # weak: over-confident (stretched logits) so the ladder has something to diagnose
                preds.append({"subject_id": sid, "model_id": "weak",
                              "predicted": float(1.0 / (1.0 + np.exp(-2.0 * z)))})

    pd.DataFrame(rows).to_parquet(tmp_path / "cohort.parquet", index=False)
    pd.DataFrame(preds).to_parquet(tmp_path / "predictions.parquet", index=False)
    clustering = {"field": "wsi"}
    if clustering_name is not _UNSET and clustering_name is not None:
        clustering["name"] = clustering_name
    spec = {
        "cohort_id": "toy-report", "version": 0, "subject_key": "subject_id",
        "data": "cohort.parquet",
        "outcome": {"type": "binary", "field": "label", "positive_label": 1},
        "clustering": clustering,
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


def test_section3_gains_carry_intervals_not_pending(evaluation, html) -> None:
    """Item 1: every section-3 gain now carries a paired interval; the pending marker is gone.

    The naked-delta ban is satisfied by showing the interval, not by deferring it.
    """
    ctx = build_context(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest,
        evaluation.calibration,
    )
    assert ctx["calibration_rows"], "expected ladder rows to test"
    saw_interval = False
    for row in ctx["calibration_rows"]:
        for rung in ("rung1_gain", "rung2_gain", "rung3_gain"):
            cell = row[rung]
            assert PENDING not in cell, f"gain cell should not defer its interval: {row}"
            if cell != "n/a":
                # a signed point estimate with a bracketed interval, e.g. '+0.0112 [0.005, 0.017]'
                assert "[" in cell and "]" in cell, f"gain cell missing interval: {cell}"
                saw_interval = True
    assert saw_interval, "expected at least one converged rung gain with an interval"
    # The deferred-interval marker no longer appears anywhere in the default report body.
    assert PENDING not in html


def test_displayed_loss_metric_ci_lower_bound_is_nonnegative(evaluation, html) -> None:
    """S4.1 item 4: body Brier is the bootstrap; its lower bound cannot fall below 0."""
    import re

    ctx = build_context(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest
    )
    def lower(cell: str) -> float | None:
        m = re.search(r"\[(-?\d+\.\d+),", cell)
        return float(m.group(1)) if m else None

    # Body Brier -- both the primary (overall) table and the subgroup table, where the original
    # bug lived (an analytic interval crossing 0 in a subgroup).
    for row in ctx["primary_rows"]:
        lo = lower(row["brier"])
        assert lo is None or lo >= 0.0, f"primary Brier CI below 0: {row['brier']}"
    for row in ctx["subgroup_rows"]:
        lo = lower(row["brier"])
        assert lo is None or lo >= 0.0, f"subgroup Brier CI below 0: {row['brier']}"
    # Any analytic interval surfaced in the footnote is truncated at 0.
    for fn in ctx["brier_footnote"]:
        assert fn["interval"].startswith("[0.00")


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


# ----------------------------------------------------------------------- verdict layer (item 3)


def test_verdict_lines_present_with_anchors(html) -> None:
    assert "Verdict" in html
    assert "no-skill" in html and "perfect" in html
    assert "Gauge fault found" in html
    assert "closes" in html and "% of the gap" in html


def test_verdict_line_names_only_binned_gauge_label(evaluation) -> None:
    """The only qualitative token is the gauge-fault rung label; no free adjectives."""
    from predval.report import GAUGE_LABELS, build_context
    ctx = build_context(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest,
        evaluation.calibration,
    )
    assert ctx["verdict_lines"]
    for v in ctx["verdict_lines"]:
        assert v["gauge"] in GAUGE_LABELS.values()
        assert "AUROC" in v["line"] and "Gauge fault found" in v["line"]


# ---------------------------------------------------------------------------- figures (item 2)


def test_report_embeds_the_three_figures(html) -> None:
    # three inline SVGs: calibration small-multiples, dumbbell, brier-by-rung
    assert html.count("<svg") >= 3
    assert 'class="figure"' in html


# ------------------------------------------------------------------------- limitations (item 6b)


def test_limitations_block_is_present(html) -> None:
    assert "Limitations" in html
    assert "recalibration step" in html.lower()
    assert "ensemble" in html


# ------------------------------------------------------------------------------------- S6 items


def test_section3_gains_use_milli_units(html) -> None:
    """Item 4: section-3 gain cells are scaled to x10⁻³ with two decimals, not raw 4dp."""
    assert "x10⁻³" in html
    # the old 4-decimal-place raw format ("+0.0314") should not appear for a gain cell
    import re

    assert re.search(r"[+-]\d+\.\d{2} \[[+-]?\d+\.\d{2}, [+-]?\d+\.\d{2}\] x10⁻³", html)


def test_cluster_noun_is_threaded_from_cohort_yaml(tmp_path: Path) -> None:
    """Item 5: a cohort declaring clustering.name: region reads 'region' in the report, not the
    PCam-flavoured 'slide' hardcoded in earlier versions of the template."""
    ev = _build(tmp_path, clustering_name="region")
    html = render_evaluation(ev)
    assert "region" in html
    assert "slide" not in html.lower()


def test_cluster_noun_defaults_to_cluster_when_undeclared(tmp_path: Path) -> None:
    ev = _build(tmp_path, clustering_name=None)
    html = render_evaluation(ev)
    assert "cluster" in html.lower()
    assert "slide" not in html.lower()


def test_ensemble_paragraph_omitted_for_a_single_model_roster(tmp_path: Path) -> None:
    """Item 5: the ensemble-construction-bias caution only makes sense with >1 model in the
    roster; a single-model cohort (like GUSTO) must not see a dangling reference to it."""
    ev = _build(tmp_path, single_model=True)
    html = render_evaluation(ev)
    assert "Limitations" in html
    assert "ensemble" not in html.lower()
    # the always-shown gain-interval sentence must still be present
    assert "does not carry the correction" in html


def test_fold4_reference_omitted_without_a_rung3_withholding(html) -> None:
    """Item 5: the fold4 diagnosis doc is a PCam-specific writeup; citing it when this run has no
    rung3-could-not-cross-fit withholding would be a dangling, misleading reference."""
    assert "fold4" not in html.lower()
    assert "diagnosis-rung3-scanner_domain0" not in html


def test_exhibit_note_is_conditional_on_the_ratio() -> None:
    """Item 6: never claim the naive interval is narrower when the two intervals actually agree."""
    from predval.report import _exhibit_note

    understates = _exhibit_note([{"_ratio": 3.5}, {"_ratio": 1.2}], "slide")
    assert "understates" in understates
    assert "narrower" not in understates

    agrees = _exhibit_note([{"_ratio": 0.9}, {"_ratio": 0.6}], "region")
    assert "nearly agree" in agrees
    assert "understates" not in agrees
    assert "narrower" not in agrees
    assert "region" in agrees


def test_few_clusters_flags_name_their_stratum(html) -> None:
    """Item 7: two few_clusters flags at different G must be distinguishable by stratum."""
    assert "for overall" in html
    assert "for arm_group=" in html


# ---------------------------------------------------------------------------- appendix (item 5)


def test_appendix_off_by_default_and_bytes_unchanged(evaluation) -> None:
    default = render_evaluation(evaluation)
    with_flag = render_evaluation(evaluation, appendix=True)
    assert "Appendix -- concepts" not in default
    assert "Appendix -- concepts" in with_flag
    # The appendix is purely additive: everything in the default report up to its closing </body>
    # appears verbatim at the front of the appendix render, so turning it on alters no default byte.
    default_body = default[: default.rindex("</body>")].rstrip()
    assert with_flag.startswith(default_body)


def test_appendix_has_placeholder_structure(evaluation) -> None:
    html = render_evaluation(evaluation, appendix=True)
    assert html.count("Placeholder --") == 4
