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


def _build(
    tmp_path: Path,
    expected_models=None,
    clustering_name=_UNSET,
    single_model=False,
    ensemble_members=None,
):
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
    if ensemble_members is not None:
        spec["ensemble_members"] = ensemble_members
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


def test_verdict_line_gauge_is_from_the_composed_vocabulary(evaluation) -> None:
    """S6.1 + S6.2 D3: the gauge clause is composed from a fixed vocabulary -- axis A's triggering
    CI, axis B's outcome and rung, and any harmful rung -- with no free adjectives. Under S6.2 the
    empty string joined that vocabulary: a member with nothing to report says nothing."""
    from predval.report import build_context

    ctx = build_context(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest,
        evaluation.calibration,
    )
    assert ctx["verdict_lines"]
    for v in ctx["verdict_lines"]:
        g = v["gauge"]
        if g == "":
            # Axis A silent and axis B unproven with no harmful rung: correctly no claim at all,
            # and the caller drops the "Gauge fault found" sentence entirely.
            assert "Gauge fault found" not in v["line"]
            continue
        assert g.startswith(
            ("miscalibrated (", "repair demonstrated at", "recalibration harm")
        ), g
        # The retired S6.1 reassurance phrasing must not come back (S6.2 D3).
        for phrase in ("well-calibrated", "calibrated as published", "none --"):
            assert phrase not in g, g


def test_verdict_gauge_every_combination() -> None:
    """S6.2 D3/D5: the exact clause set for every (axis A, axis B, harm) combination.

    Locked directly rather than through a render, because these strings are the report's actual
    claims. The grid is (2 x 3) with harm cutting across it independently -- harm is a property of
    the ladder, not a cell of the taxonomy, which is why it composes onto every row.
    """
    from predval.report import verdict_gauge

    demonstrated = {"best_rung": "rung2", "axis_a_miscalibrated": True, "axis_b": "demonstrated"}
    unproven = {"best_rung": "rung0", "axis_a_miscalibrated": True, "axis_b": "unproven",
                "n_clusters": 22}
    counter = {"best_rung": "rung0", "axis_a_miscalibrated": True, "axis_b": "counterproductive"}
    harm2 = [("rung1", -0.005, -0.002), ("rung2", -0.006, -0.001)]

    assert verdict_gauge(demonstrated, "slope 4.93 [2.72, 7.14]", None) == (
        "miscalibrated (slope 4.93 [2.72, 7.14]); repair demonstrated at rung2 "
        "(level and spread (intercept + slope))"
    )
    assert verdict_gauge(unproven, "slope 4.93 [2.72, 7.14]", None) == (
        "miscalibrated (slope 4.93 [2.72, 7.14]); repair benefit unproven at this cohort's "
        "power (G=22)"
    )
    assert verdict_gauge(unproven, "slope 4.93 [2.72, 7.14]", "few clusters note") == (
        "miscalibrated (slope 4.93 [2.72, 7.14]); repair benefit unproven at this cohort's "
        "power (G=22) -- few clusters note"
    )
    assert verdict_gauge(counter, "slope 4.93 [2.72, 7.14]", None, harm2, 3) == (
        "miscalibrated (slope 4.93 [2.72, 7.14]); recalibration harm "
        "(paired cross-fit Brier gain, x10\u207b\u00b3): rung1 [-5.00, -2.00], rung2 [-6.00, -1.00]"
    )
    # Every rung harmful -> the stronger phrasing, because there is no rung left to fall back to.
    assert verdict_gauge(counter, "slope 4.93 [2.72, 7.14]", None, harm2, 2) == (
        "miscalibrated (slope 4.93 [2.72, 7.14]); recalibration harm at every rung, no "
        "admissible repair (paired cross-fit Brier gain, x10\u207b\u00b3): "
        "rung1 [-5.00, -2.00], rung2 [-6.00, -1.00]"
    )

    quiet_unproven = {**unproven, "axis_a_miscalibrated": False}
    quiet_counter = {**counter, "axis_a_miscalibrated": False}
    quiet_demonstrated = {**demonstrated, "axis_a_miscalibrated": False}

    # S6.2 D3, the asymmetry rule: nothing detected and nothing proven -> SILENCE. This cell used
    # to read "none -- well-calibrated as published", which reported a non-rejection at G=22 as a
    # clean bill of health.
    assert verdict_gauge(quiet_unproven, "", None) == ""
    # Harm is always stated, with or without axis A.
    assert verdict_gauge(quiet_counter, "", None, [("rung3", -0.004, -0.001)], 3) == (
        "recalibration harm (paired cross-fit Brier gain, x10\u207b\u00b3): rung3 [-4.00, -1.00]"
    )
    # The rare cell: axis A silent, but a repair was demonstrated anyway -- stated plainly, and
    # still with no calibration claim attached to it.
    assert verdict_gauge(quiet_demonstrated, "", None) == (
        "repair demonstrated at rung2 (level and spread (intercept + slope))"
    )


# ---------------------------------------------------------------------------- figures (item 2)


def test_report_embeds_the_three_figures(html) -> None:
    # three inline SVGs: calibration small-multiples, dumbbell, brier-by-rung
    assert html.count("<svg") >= 3
    assert 'class="figure"' in html


# ------------------------------------------------------------------------- limitations (item 6b)


def test_limitations_block_is_present(html) -> None:
    """The always-shown gain-interval sentence (item 6b); the ensemble paragraph is conditional
    on a declared ensemble member (S6.1 item 2) and is tested separately -- this fixture doesn't
    declare one, so it is correctly absent here."""
    assert "Limitations" in html
    assert "conditions on the fitted correction" in html


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
    """Item 5: a single-model cohort (like GUSTO) has no declared ensemble_members and must not
    see a dangling reference to ensemble construction."""
    ev = _build(tmp_path, single_model=True)
    html = render_evaluation(ev)
    assert "Limitations" in html
    assert "ensemble" not in html.lower()
    # the always-shown gain-interval sentence must still be present
    assert "does not carry the correction" in html


def test_ensemble_paragraph_omitted_without_a_declared_ensemble_member(tmp_path: Path) -> None:
    """S6.1 item 2: this is the actual behaviour change from S6's `len(models) > 1` rule -- two
    models with NEITHER declared as an ensemble/blend must still omit the caution. predval cannot
    infer "this model_id is a blend" from predictions alone; the old rule fired on the mere
    existence of a second model, which is not evidence of anything."""
    ev = _build(tmp_path)  # 2 models ("good", "weak"), no ensemble_members declared
    html = render_evaluation(ev)
    assert "Limitations" in html
    assert "ensemble" not in html.lower()
    assert "conditions on the fitted correction" in html


def test_ensemble_paragraph_shown_when_a_member_is_declared(tmp_path: Path) -> None:
    """S6.1 item 2: the caution appears when the cohort actually declares an ensemble member --
    a real, checkable fact, not a proxy on roster size."""
    ev = _build(tmp_path, ensemble_members=["weak"])
    html = render_evaluation(ev)
    assert "Limitations" in html
    assert "ensemble" in html.lower()
    assert "model or ensemble construction" in html


def test_fold4_reference_omitted_without_a_rung3_withholding(html) -> None:
    """Item 5: the fold4 diagnosis doc is a PCam-specific writeup; citing it when this run has no
    rung3-could-not-cross-fit withholding would be a dangling, misleading reference."""
    assert "fold4" not in html.lower()
    assert "diagnosis-rung3-scanner_domain0" not in html


def test_exhibit_note_is_conditional_on_the_ratio() -> None:
    """Item 6 + S6.2 D6: the caption is direction-aware, and says which way round the ratio runs.

    "Ratio" alone is reversible; the sentence built on it is not. Both branches therefore define
    it as cluster width over naive width before drawing any conclusion from it.
    """
    from predval.report import _exhibit_note

    too_narrow = _exhibit_note([{"_ratio": 3.5}, {"_ratio": 1.2}], "slide", "patch")
    assert "too narrow" in too_narrow
    assert "cluster-aware interval width divided by the naive per-patch width" in too_narrow
    assert "does not inflate" not in too_narrow
    # S6.2 D1: the cohort's own nouns, both of them, correctly pluralised.
    assert "patches share a slide" in too_narrow

    agrees = _exhibit_note([{"_ratio": 0.9}, {"_ratio": 0.6}], "region")
    assert "does not inflate" in agrees
    assert "cluster-aware interval width divided by the naive per-row width" in agrees
    assert "too narrow" not in agrees
    assert "region" in agrees
    # Never asserts the inverted claim the pre-S6.2 caption made on this branch.
    assert "narrower" not in agrees


def test_cohort_nouns_pluralise(html) -> None:
    """S6.2 D1: a declared unit noun reaches the prose in both numbers. "patchs" would be a
    visible defect in a report handed to a clinician."""
    from predval.report import _plural

    assert _plural("patch") == "patches"
    assert _plural("slide") == "slides"
    assert _plural("row") == "rows"
    assert _plural("study") == "studies"
    # The toy cohort declares neither noun, so the render falls back to both generic defaults.
    assert "clusters" in html and "per-row" in html


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
