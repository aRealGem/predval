"""S6.2 (D3/D4/D5): the verdict line's string layer must follow from the numbers beside it.

These run over BOTH frozen goldens -- PCam (15 members, G=22) and GUSTO (1 member, G=15) -- and
re-derive the expected claim from `metrics.parquet` rather than from the committed `report.html`.
That orientation is the point: the goldens supply the data, the code under test supplies the
prose, and a regression shows up as prose that no longer matches its own numbers. Reading the
committed HTML instead would only re-assert whatever the last golden refresh happened to write.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predval.report import (
    _REPAIR_SHAPE_LABELS,
    build_context,
    few_clusters_note_for_overall,
)

GOLDEN = Path(__file__).resolve().parent / "golden"
COHORTS = ("pcam", "gusto")

#: Phrases that assert calibration rather than report it. Axis A can only ever fail to REJECT
#: "calibrated"; it cannot establish it, so no verdict line may say any of these (S6.2 D3).
REASSURANCE = ("well-calibrated", "calibrated as published", "none --")


def _context(cohort: str) -> dict:
    d = GOLDEN / cohort
    cal = d / "calibration.parquet"
    return build_context(
        pd.read_parquet(d / "metrics.parquet"),
        pd.read_parquet(d / "fragility.parquet"),
        pd.read_parquet(d / "coverage.parquet"),
        json.loads((d / "manifest.json").read_text()),
        pd.read_parquet(cal) if cal.exists() else None,
    )


def _rung0_calibration(metrics: pd.DataFrame, model: str) -> dict[str, tuple[float, float]]:
    """rung0/common/overall analytic slope + intercept intervals -- axis A's own source data."""
    sub = metrics[
        (metrics["model_id"] == model)
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["ci_method"] == "cluster_robust_t")
        & (metrics["metric"].isin(["calibration_slope", "calibration_intercept"]))
    ]
    return {r["metric"]: (float(r["ci_low"]), float(r["ci_high"])) for _, r in sub.iterrows()}


def _harmful_rungs(metrics: pd.DataFrame, model: str) -> list[str]:
    """Every rung whose paired cross-fit gain interval lies entirely below zero, derived here
    independently of `report.harm_lookup` so the test is not a tautology over it."""
    sub = metrics[
        (metrics["model_id"] == model)
        & (metrics["metric"] == "paired_gain_brier")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["ci_method"] == "paired_cluster_bootstrap")
    ]
    return sorted(
        str(r["rung"])
        for _, r in sub.iterrows()
        if np.isfinite(r["ci_high"]) and float(r["ci_high"]) < 0
    )


@pytest.fixture(scope="module", params=COHORTS)
def golden(request):
    d = GOLDEN / request.param
    metrics = pd.read_parquet(d / "metrics.parquet")
    ctx = _context(request.param)
    lines = {v["model"]: v["line"] for v in ctx["verdict_lines"]}
    assert lines, f"{request.param}: no verdict lines rendered"
    return request.param, metrics, ctx, lines


# ------------------------------------------------------------------------------------ D3


def test_verdict_axis_a_consistency(golden) -> None:
    """Axis A drives the calibration clause in BOTH directions.

    When the analytic interval fires, the line must name the term that fired -- slope, intercept,
    or both -- so the claim carries its own evidence. When it does not fire, the line must make no
    calibration claim at all: at G=22 a non-rejection is an absence of evidence, and the previous
    wording ("none -- well-calibrated as published") reported it as evidence of absence.
    """
    cohort, metrics, _ctx, lines = golden
    for model, line in lines.items():
        cal = _rung0_calibration(metrics, model)
        slope_fired = "calibration_slope" in cal and not (
            cal["calibration_slope"][0] <= 1.0 <= cal["calibration_slope"][1]
        )
        intercept_fired = "calibration_intercept" in cal and not (
            cal["calibration_intercept"][0] <= 0.0 <= cal["calibration_intercept"][1]
        )
        where = f"{cohort}/{model}: {line}"

        if slope_fired or intercept_fired:
            assert "miscalibrated (" in line, where
            if slope_fired:
                assert "slope " in line, where
            if intercept_fired:
                assert "intercept " in line, where
        else:
            assert "miscalibrated" not in line, where

        # Never, either way.
        for phrase in REASSURANCE:
            assert phrase not in line, f"reassurance {phrase!r} in {where}"


# ------------------------------------------------------------------------------------ D4


def test_gauge_fault_matches_selected_rung(golden) -> None:
    """The shape-of-fault label comes from the SAME selection that picked the admissible rung.

    A label naming a fault the selected rung does not repair -- "non-monotone shape" against an
    admissible rung2, which is linear and monotone by construction -- describes a different
    finding than the one the ladder actually made.
    """
    cohort, _metrics, ctx, lines = golden
    verdict = ctx["verdict_lines"]
    manifest_verdict = json.loads((GOLDEN / cohort / "manifest.json").read_text())["verdict"]

    for v in verdict:
        model, line = v["model"], v["line"]
        best_rung = manifest_verdict[model]["best_rung"]
        where = f"{cohort}/{model}: {line}"
        if manifest_verdict[model]["axis_b"] != "demonstrated":
            # No repair was demonstrated, so no shape-of-repair label may appear at all.
            assert "repair demonstrated at" not in line, where
            continue
        assert f"repair demonstrated at {best_rung}" in line, where
        assert _REPAIR_SHAPE_LABELS[best_rung] in line, where
        for rung, label in _REPAIR_SHAPE_LABELS.items():
            if rung != best_rung:
                assert label not in line, f"label for {rung} on a {best_rung} repair -- {where}"


def test_p4m_seed7_reads_rung2_wording() -> None:
    """The specific case S6.2 D4 was raised on: p4m_seed7's only admissible rung is rung2, whose
    fault is level-and-spread. rung3 does not exclude zero for it, so no shape claim is earned."""
    lines = {v["model"]: v["line"] for v in _context("pcam")["verdict_lines"]}
    line = lines["p4m_seed7"]
    assert "repair demonstrated at rung2 (level and spread (intercept + slope))" in line, line
    assert "non-monotone shape" not in line, line


# ------------------------------------------------------------------------------------ D5


def test_verdict_states_counterproductive(golden) -> None:
    """Every rung whose gain interval lies entirely below zero is named on that member's line.

    Unconditional on axis A and on axis B alike. A demonstrated repair at rung2 does not make a
    reliably harmful rung1 stop being harmful, and before S6.2 the demonstrated branch swallowed
    it -- which is exactly the case both goldens carry (PCam p4m_seed7, GUSTO's single member).
    """
    cohort, metrics, _ctx, lines = golden
    for model, line in lines.items():
        harmful = _harmful_rungs(metrics, model)
        where = f"{cohort}/{model}: {line}"
        if not harmful:
            assert "recalibration harm" not in line, where
            continue
        assert "recalibration harm" in line, where
        for rung in harmful:
            assert rung in line, f"{rung} harmful but unnamed -- {where}"


def test_all_rungs_harmful_says_no_admissible_repair() -> None:
    """When every rung is harmful there is no repair to fall back to, and the line says so."""
    metrics = pd.read_parquet(GOLDEN / "pcam" / "metrics.parquet")
    lines = {v["model"]: v["line"] for v in _context("pcam")["verdict_lines"]}
    for model in ("e2cnn_s21", "e2cnn_s99"):
        assert _harmful_rungs(metrics, model) == ["rung1", "rung2", "rung3"], model
        assert "harm at every rung, no admissible repair" in lines[model], lines[model]
    # swin is harmful at rung3 only -- the stronger phrasing must NOT appear.
    assert _harmful_rungs(metrics, "swin") == ["rung3"]
    assert "no admissible repair" not in lines["swin"], lines["swin"]


# ------------------------------------------------------------------- S6.3 item 0


#: The flag's own explanatory prose. It belongs in the Flags section, not inside a verdict line.
FLAG_PROSE = ("tail quantiles approximate", "percentile bootstrap intervals below")

#: Verdict lines that carry the citation, per cohort. PCam's seven are the A-fired/B-unproven
#: members; GUSTO's single model has a demonstrated repair, so it never reaches that clause.
EXPECTED_CITATIONS = {"pcam": 7, "gusto": 0}


def test_verdict_cites_flags_by_name(golden) -> None:
    """A verdict line cites the few_clusters flag; it does not reprint it (S6.3 item 0).

    Derived from the manifest's own flag record rather than from a literal, so the expected
    citation is built out of the same stratum and G the flag itself names -- if the flag moves
    stratum or G, the expected string moves with it and the test still means what it says.
    """
    cohort, _metrics, _ctx, lines = golden
    manifest = json.loads((GOLDEN / cohort / "manifest.json").read_text())
    flag = few_clusters_note_for_overall(manifest)
    verdict = manifest["verdict"]

    # 1. The explanatory text appears in no verdict line, in either cohort.
    for model, line in lines.items():
        for prose in FLAG_PROSE:
            assert prose not in line, f"flag prose inlined -- {cohort}/{model}: {line}"

    assert flag is not None, f"{cohort}: expected an overall few_clusters flag at this G"
    m = re.search(r"few clusters \(G=(\d+)\) for ([^;]+);", flag)
    assert m, f"{cohort}: flag text no longer parses -- {flag!r}"
    citation = f"flag: few_clusters ({m.group(2).strip()}, G={m.group(1)})"

    # 2. Exactly the members whose power caveat fires carry the citation, and no others.
    seen = 0
    for model, line in lines.items():
        v = verdict[model]
        where = f"{cohort}/{model}: {line}"
        if v["axis_b"] == "unproven" and v["axis_a_miscalibrated"]:
            assert citation in line, f"power caveat without its citation -- {where}"
            seen += 1
        else:
            assert "few_clusters" not in line, f"citation on a member that earned none -- {where}"
    assert seen == EXPECTED_CITATIONS[cohort], f"{cohort}: {seen} citations, expected differently"


# ------------------------------------------------------------------- S6.4 A9 / A14


def test_limitations_states_predictions_in_boundary(golden) -> None:
    """The predictions-in boundary is stated for EVERY cohort, not just ensemble ones (A9).

    S6.1 item 2 gated the ensemble-construction caution on a declared `ensemble_members` list,
    which was right for the ensemble-specific wording but also deleted the general boundary from
    every cohort declaring none -- i.e. from both fixtures, so in practice the report stopped
    saying it at all. The boundary is a property of predval's contract, so it is unconditional.
    """
    cohort, _metrics, ctx, _lines = golden
    lim = ctx["limitations"]
    manifest = json.loads((GOLDEN / cohort / "manifest.json").read_text())
    noun = manifest["uncertainty"]["clustering_name"]
    where = f"{cohort}: {lim}"

    assert "The optimism correction covers the recalibration step only." in lim, where
    assert "no view of how they were produced" in lim, where
    assert f"tuning used {noun}s inside this cohort" in lim, where
    assert "rung0 is itself optimistically biased" in lim, where
    assert "never tuned on could expose that" in lim, where

    # The refit-in-replicate note is kept, and kept AFTER the boundary.
    assert "conditions on the fitted correction" in lim, where
    assert lim.index("optimism correction covers") < lim.index("conditions on the fitted"), where

    # Neither fixture declares an ensemble member, so ensembles must not be mentioned.
    assert not manifest.get("roster", {}).get("ensemble_members"), f"{cohort}: fixture changed"
    for banned in ("ensemble", "blend"):
        assert banned not in lim.lower(), f"{banned!r} in limitations, undeclared -- {where}"


def test_section3_row_order(golden) -> None:
    """Demonstrated repairs sort to the top, ascending rung then descending ci_low (A14).

    The expected order is re-derived here from metrics.parquet rather than read off the rendered
    table, so this test fails if the ordering rule and the admissible-rung rule ever disagree.
    Ordering by best point estimate alone put tinyvgg_vl first in PCam -- the fixture's largest
    apparent gain, whose interval crosses zero.
    """
    cohort, metrics, ctx, _lines = golden
    rungs = ("rung1", "rung2", "rung3")
    sub = metrics[
        (metrics["metric"] == "paired_gain_brier")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["fit_mode"] == "crossfit")
    ]

    def key(model: str):
        best, demo = float("-inf"), None
        for i, rung in enumerate(rungs):
            r = sub[(sub["model_id"] == model) & (sub["rung"] == rung)]
            if r.empty:
                continue
            v, lo = float(r.iloc[0]["value"]), float(r.iloc[0]["ci_low"])
            if np.isfinite(v):
                best = max(best, v)
            if demo is None and np.isfinite(lo) and lo > 0:
                demo = (i, -lo)
        return (0, *demo) if demo else (1, 0, -best)

    rendered = [r["model"] for r in ctx["calibration_rows"]]
    assert rendered == sorted(rendered, key=key), f"{cohort}: got {rendered}"

    demonstrated = [m for m in rendered if key(m)[0] == 0]
    assert rendered[: len(demonstrated)] == demonstrated, f"{cohort}: demonstrated not on top"
    if cohort == "pcam":
        assert demonstrated == ["p4m_seed7"], demonstrated
        assert rendered[0] == "p4m_seed7", rendered[:3]
