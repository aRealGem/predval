"""Render an evaluation into a standalone HTML report (docs/spec.md sections 4.4 and 6).

The report is a *view* over the artefacts: it computes nothing new, it arranges what
`evaluate` already produced and refuses to let any of it read as more than it is. Three rules are
structural, not cosmetic:

1. The framing block (section 4.4) is unconditional and appears before any number.
2. No ladder gain is shown without "(interval pending section 4.5)" attached -- the naked-delta
   ban. A recalibrated figure has no interval yet, and an improvement printed bare invites a
   confidence the harness has not earned.
3. Identities that hold by construction (rung1 apparent intercept ~ 0, rung2 apparent slope ~ 1)
   are labelled as such, never presented as findings.

Rendering is deterministic: no wall-clock, models in a fixed order, so the same artefacts produce
byte-identical HTML.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape

from . import figures
from .evaluate import Evaluation, coverage_delta

#: The report template lives beside the module (packaged with it), not inline, so it stays out of
#: the linter's way and reads as the HTML it is.
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.j2"


@lru_cache(maxsize=1)
def _template_source() -> str:
    return _TEMPLATE_PATH.read_text()

#: The framing rule, verbatim in substance from docs/spec.md section 4.4, plus the discrimination
#: sentence S4 requires. Rendered unconditionally at the top of every report.
FRAMING = (
    "No rung is a validated model. Rung0 is the model as published. Rungs 1-3 are diagnosis of "
    "rung0, not competitors to it. The cross-fitted figures are the expected performance after "
    "local recalibration on a cohort like this one -- not evidence that the recalibrated model "
    "has been validated, which would require a cohort the correction was never fitted on. "
    "Recalibration does not and cannot improve discrimination."
)

#: Historical marker for a gain shown without an interval. The paired cross-fit gain now carries a
#: real interval (item 1), so section 3 no longer uses this; it survives only for the appendix's
#: description of the deferred refit-in-replicate interval (section 4.5).
PENDING = "(interval pending section 4.5)"

#: D2 footnote for a symmetric analytic loss interval that crossed the 0 boundary and was truncated.
ANALYTIC_TRUNCATED_NOTE = (
    "normal approximation unreliable at this G; percentile bootstrap interval is authoritative"
)

_LADDER_RUNGS = ("rung1", "rung2", "rung3")

#: The recalibration-conditions-on-itself sentence (item 6b) applies to every cohort regardless of
#: roster size; it is always shown.
_LIMITATIONS_GAIN_INTERVAL = (
    "The paired cross-fit gain interval (section 3) conditions on the fitted correction -- it "
    "resamples the {noun}-level loss difference but does not refit the correction inside each "
    "bootstrap replicate, so it does not carry the correction's own fitting variance; that "
    "refit-in-replicate interval remains backlog (spec section 4.5)."
)

#: The ensemble-construction-bias caution is shown only when the cohort actually DECLARES an
#: ensemble/blend model_id (S6.1 item 2) -- not merely "more than one model", which S6's rule
#: fired on regardless of whether any member was really a blend. predval cannot infer "this
#: model_id is a blend" from predictions alone (its contract is predictions-only, model_id is
#: opaque); this is why the declaration is author-asserted in cohort.yaml, not inferred here.
_LIMITATIONS_ENSEMBLE = (
    "The optimism correction in section 3 covers the recalibration step ONLY -- it does not cover "
    "model or ensemble construction. If a roster member's own weights were selected on {plural} "
    "inside this cohort, rung0 is itself optimistically biased, and this harness cannot detect "
    "it: predval evaluates the predictions it is handed and has no view of how they were "
    "produced, including whether any given model_id is a single trained model or a blend of "
    "several (a model_id's name is not evidence either way -- always verify composition against "
    "its source). Only a cohort the ensemble was never tuned on could expose that bias. "
)


def _limitations(clustering_name: str, ensemble_members: list[str]) -> str:
    """Limitations block (item 6b, S6 item 5, S6.1 item 2), verbatim in report and docs/spec.md
    section 9."""
    gain_sentence = _LIMITATIONS_GAIN_INTERVAL.format(noun=clustering_name)
    if not ensemble_members:
        return gain_sentence
    ensemble_sentence = _LIMITATIONS_ENSEMBLE.format(plural=_plural(clustering_name))
    return ensemble_sentence + gain_sentence

#: Appendix concept-explainer (item 5): OFF by default; structure + placeholders this session, the
#: static SVG assets arrive later. Rendered only when --appendix is passed, so default bytes are
#: unaffected.
APPENDIX_SECTIONS = (
    ("Why the unit of analysis is not the unit of independence",
     "Placeholder -- a static SVG explainer of clustered sampling will be inserted here."),
    ("The recalibration ladder, rung by rung",
     "Placeholder -- a static SVG explainer of rungs 0-3 will be inserted here."),
    ("Apparent versus cross-fitted, and what optimism measures",
     "Placeholder -- a static SVG explainer of the cross-fit gap will be inserted here."),
    ("Reading the Brier skill score",
     "Placeholder -- a static SVG explainer of the no-skill and perfect anchors will go here."),
)

#: Shape-of-miscalibration labels for a DEMONSTRATED repair, keyed by the admissible rung -- what
#: the ladder found and fixed. Unchanged in substance from the S6 GAUGE_LABELS table; kept as a
#: small dict since this half of the message genuinely is a fixed bin (S6.1 item 1).
_REPAIR_SHAPE_LABELS = {
    "rung1": "level (calibration-in-the-large)",
    "rung2": "level and spread (intercept + slope)",
    "rung3": "non-monotone shape",
}

#: Kept for any external reader of the old flat rung->label mapping. "rung0" no longer has an
#: entry at all: under the two-axis taxonomy (S6.1 item 1) it meant "axis A found nothing", and
#: S6.2 D3 established that this must be reported as SILENCE, not as the affirmative
#: "well-calibrated as published". Absence of detectable miscalibration at G=22 is not evidence of
#: calibration, and a verdict line that says so reads as a clean bill of health the data cannot
#: support. See `verdict_gauge()`.
GAUGE_LABELS = dict(_REPAIR_SHAPE_LABELS)


# --------------------------------------------------------------------------- formatting helpers


def _plural(noun: str) -> str:
    """English plural of a cohort noun. Enough rule for the job: the nouns here are ordinary
    concrete singulars declared in a cohort spec ("patch", "slide", "region", "row"), and "patchs"
    is a visible defect in a clinical report (S6.2 D1).
    """
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _f(x: float | None, nd: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _ci(lo: float | None, hi: float | None, nd: int = 3) -> str:
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _val_ci(v: float, lo: float, hi: float) -> str:
    return f"{_f(v)} {_ci(lo, hi)}".strip()


def _truncate_loss_lo(lo: float) -> tuple[float, bool]:
    """Clamp a loss-metric interval's lower bound at the 0 boundary. Returns (lo, was_truncated).

    The percentile bootstrap of a non-negative loss cannot cross 0, but the symmetric analytic
    t(G-1) interval can. Where it does, displaying the raw negative bound is nonsense -- Brier is a
    squared error -- so it is truncated at 0 and marked (§5.2, S4.1 item 4).
    """
    if lo is not None and np.isfinite(lo) and lo < 0.0:
        return 0.0, True
    return lo, False


def _one(df: pd.DataFrame, **conds) -> pd.Series | None:
    """First row matching every equality condition, or None. NaN threshold matches via isna."""
    mask = pd.Series(True, index=df.index)
    for col, want in conds.items():
        mask &= df[col].isna() if want is None else (df[col] == want)
    hit = df[mask]
    return None if hit.empty else hit.iloc[0]


def _models_by_auroc(metrics: pd.DataFrame) -> list[str]:
    """Models ordered by rung0 common-subset AUROC, descending -- a stable presentation order."""
    rows = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["ci_method"] == "cluster_bootstrap")
    ]
    ordered = rows.sort_values(["value", "model_id"], ascending=[False, True])
    return list(ordered["model_id"])


# ------------------------------------------------------------------------------ section builders


def _primary_rows(metrics: pd.DataFrame, models: list[str], show_cov_delta: bool) -> list[dict]:
    base = metrics[
        (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
    ]
    cov = coverage_delta(metrics)
    cov_auroc = cov[
        (cov["stratum_kind"] == "overall") & (cov["metric"] == "auroc")
    ].set_index("model_id")["cov_delta"]

    out = []
    for model in models:
        sub = base[base["model_id"] == model]

        def cell(metric: str, method: str, sub: pd.DataFrame = sub) -> str:
            r = _one(sub, metric=metric, ci_method=method, threshold=None)
            return _val_ci(r["value"], r["ci_low"], r["ci_high"]) if r is not None else "n/a"

        row = {
            "model": model,
            "auroc": cell("auroc", "cluster_bootstrap"),
            "average_precision": cell("average_precision", "cluster_bootstrap"),
            # Body Brier is the percentile bootstrap -- a non-negative loss whose interval cannot
            # cross 0 (S4.1 item 4). The analytic t(G-1) cross-check is a footnote below.
            "brier": cell("brier", "cluster_bootstrap"),
            "calibration_slope": cell("calibration_slope", "cluster_robust_t"),
            "calibration_intercept": cell("calibration_intercept", "cluster_robust_t"),
        }
        if show_cov_delta:
            d = cov_auroc.get(model, float("nan"))
            row["cov_delta"] = _f(d, 4) if np.isfinite(d) else "0.0000"
        out.append(row)
    return out


def _brier_analytic_footnote(metrics: pd.DataFrame, models: list[str]) -> list[dict]:
    """Analytic t(G-1) Brier intervals that crossed the 0 loss boundary, truncated with a note.

    The report body shows the bootstrap; this names the models whose symmetric analytic interval
    fell below 0 and was clamped -- the divergence §5.2 asks the report to surface rather than hide.
    """
    base = metrics[
        (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["metric"] == "brier")
        & (metrics["ci_method"] == "cluster_robust_t")
    ]
    out = []
    for model in models:
        r = _one(base, model_id=model)
        if r is None:
            continue
        lo, truncated = _truncate_loss_lo(float(r["ci_low"]))
        if truncated:
            out.append({
                "model": model,
                "interval": _ci(lo, float(r["ci_high"])),
                "raw_low": _f(float(r["ci_low"]), 4),
            })
    return out


def _exhibit_rows(metrics: pd.DataFrame, models: list[str], cluster_noun: str) -> list[dict]:
    """The unit-of-analysis exhibit: AUROC cluster interval vs the naive per-row interval."""
    auroc = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
    ]
    out = []
    for model in models:
        sub = auroc[auroc["model_id"] == model]
        cluster = _one(sub, ci_method="cluster_bootstrap")
        naive = _one(sub, ci_method="naive_row_bootstrap")
        if cluster is None or naive is None:
            continue
        cw = cluster["ci_high"] - cluster["ci_low"]
        nw = naive["ci_high"] - naive["ci_low"]
        if not (np.isfinite(cw) and np.isfinite(nw)) or nw <= 0:
            continue
        ratio = cw / nw
        out.append({
            "model": model,
            "cluster_width": _f(cw, 4),
            "naive_width": _f(nw, 4),
            "ratio": f"{ratio:.1f}x",
            "_ratio": ratio,
        })
    return out


def _exhibit_note(rows: list[dict], cluster_noun: str, unit_noun: str = "row") -> str:
    """The exhibit's framing paragraph (S6 item 6; S6.2 D1/D6): direction-aware, and explicit about
    which way round the ratio is defined.

    The ratio is stated in the prose as cluster-aware width / naive width, because "ratio" alone is
    reversible and the sentence built on it is not: a reader who assumes the other orientation
    reads every claim backwards. On GUSTO the ratio is 0.9x -- the naive interval is *wider* -- so
    the pre-S6 wording ("how much narrower the wrong interval looks") was flatly false there.

    Both branches are kept. A clinical cohort where clustering turns out to be benign is itself
    evidence the harness is not manufacturing findings, so the exhibit is never dropped for being
    undramatic; it just says what it actually found.
    """
    ratios = [r["_ratio"] for r in rows if np.isfinite(r["_ratio"])]
    definition = (
        f"The ratio is the cluster-aware interval width divided by the naive per-{unit_noun} width."
    )
    if not ratios or max(ratios) > 1.0:
        return (
            f"The naive per-{unit_noun} AUROC interval is <em>incorrect</em> when "
            f"{_plural(unit_noun)} share a {cluster_noun}. It is shown only to make the cost of "
            f"ignoring clustering concrete. {definition} Above 1 it is the factor by which the "
            f"naive interval is <em>too narrow</em>."
        )
    return (
        f"The naive per-{unit_noun} AUROC interval is shown alongside the cluster-aware one for "
        f"comparison. {definition} Here it is at or below 1: naive and cluster-aware widths agree "
        f"within rounding on this cohort, so clustering <em>does not inflate</em> uncertainty "
        f"here -- the {_plural(cluster_noun)} are large or homogeneous enough that the check, not "
        f"a dramatic gap, is the point."
    )


def _signed_ci(v: float, lo: float, hi: float) -> str:
    """A signed point estimate with its interval: '+0.0112 [0.005, 0.017]', or 'n/a'."""
    if v is None or not np.isfinite(v):
        return "n/a"
    ci = _ci(lo, hi)
    return f"{v:+.4f} {ci}".strip()


#: Section 3 gains are order 1e-2 to 1e-3; four raw decimals buried the CI bounds in visual noise.
_MILLI = 1000.0


def _signed_ci_milli(v: float, lo: float, hi: float) -> str:
    """A section-3 gain + its interval, scaled to 1e-3 with two decimals each, e.g.
    '+31.43 [-3.78, 67.68] x10⁻³', or 'n/a'.

    Two decimals at 1e-3 scale keeps whether an interval excludes zero legible straight off the
    digits, which four raw decimals ('+0.0314 [-0.004, 0.068]') did not: the sign of the bound is
    now a large, easy-to-compare number rather than a fourth decimal place (S6 item 4).
    """
    if v is None or not np.isfinite(v):
        return "n/a"
    vs = v * _MILLI
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{vs:+.2f} x10⁻³"
    return f"{vs:+.2f} [{lo * _MILLI:.2f}, {hi * _MILLI:.2f}] x10⁻³"


def _gain_lookup(metrics: pd.DataFrame) -> pd.DataFrame:
    """The paired cross-fit gain rows (rung0 - rung_r), common subset, overall stratum (item 1)."""
    return metrics[
        (metrics["metric"] == "paired_gain_brier")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
    ]


def _dumbbell_data(metrics: pd.DataFrame, models: list[str]) -> list[dict]:
    """Figure 2b input: AUROC point with its cluster and naive per-row intervals, per member."""
    auroc = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
    ]
    out = []
    for model in models:
        sub = auroc[auroc["model_id"] == model]
        cluster = _one(sub, ci_method="cluster_bootstrap")
        naive = _one(sub, ci_method="naive_row_bootstrap")
        if cluster is None or naive is None:
            continue
        vals = [cluster["ci_low"], cluster["ci_high"], naive["ci_low"], naive["ci_high"]]
        if not all(np.isfinite(x) for x in vals):
            continue
        out.append({
            "model": model,
            "value": float(cluster["value"]),
            "cluster_low": float(cluster["ci_low"]),
            "cluster_high": float(cluster["ci_high"]),
            "naive_low": float(naive["ci_low"]),
            "naive_high": float(naive["ci_high"]),
        })
    return out


def _calibration_rows(metrics: pd.DataFrame, models: list[str]) -> list[dict]:
    """Ladder diagnosis: rung0 Brier and, per rung, the paired cross-fit gain WITH its interval.

    The "(interval pending §4.5)" marker is gone: every gain now carries a paired cluster-bootstrap
    interval (item 1). A gain is rung0 Brier minus that rung's held-out Brier, positive when the
    out-of-fold recalibration helped; the interval is the paired slide bootstrap. Rows are ordered
    by the best per-member gain descending, so the members a recalibration would most help sit on
    top. rung3 reads 'n/a' where it did not converge.
    """
    overall = metrics[(metrics["subset"] == "common") & (metrics["stratum_kind"] == "overall")]
    gains = _gain_lookup(metrics)

    def brier(model: str, rung: str, mode: str) -> float:
        r = _one(overall, model_id=model, metric="brier", rung=rung, fit_mode=mode, threshold=None)
        return float(r["value"]) if r is not None else float("nan")

    def gain(model: str, rung: str) -> tuple[float, float, float]:
        r = _one(gains, model_id=model, rung=rung, fit_mode="crossfit")
        if r is None:
            return float("nan"), float("nan"), float("nan")
        return float(r["value"]), float(r["ci_low"]), float(r["ci_high"])

    out = []
    for model in models:
        r0 = brier(model, "rung0", "apparent")
        best = float("-inf")
        cells = {}
        for rung in _LADDER_RUNGS:
            v, lo, hi = gain(model, rung)
            cells[f"{rung}_gain"] = _signed_ci_milli(v, lo, hi)
            if np.isfinite(v):
                best = max(best, v)
        out.append({
            "model": model,
            "rung0": _f(r0),
            **cells,
            "_gain_sort": best,
        })
    out.sort(key=lambda r: r["_gain_sort"], reverse=True)
    for r in out:
        del r["_gain_sort"]
    return out


def _brier_by_rung_data(metrics: pd.DataFrame, models: list[str]) -> list[dict]:
    """Figure 2c input: rung0 Brier, cross-fit levels, and paired-gain intervals per member."""
    overall = metrics[(metrics["subset"] == "common") & (metrics["stratum_kind"] == "overall")]
    gains = _gain_lookup(metrics)

    def brier(model: str, rung: str, mode: str) -> float:
        r = _one(overall, model_id=model, metric="brier", rung=rung, fit_mode=mode, threshold=None)
        return float(r["value"]) if r is not None else float("nan")

    out = []
    for model in models:
        r0 = brier(model, "rung0", "apparent")
        if not np.isfinite(r0):
            continue
        levels, gain_ci = {}, {}
        for rung in _LADDER_RUNGS:
            lvl = brier(model, rung, "crossfit")
            if not np.isfinite(lvl):
                continue
            levels[rung] = lvl
            g = _one(gains, model_id=model, rung=rung, fit_mode="crossfit")
            gain_ci[rung] = (
                (float(g["ci_low"]), float(g["ci_high"])) if g is not None else (float("nan"),) * 2
            )
        out.append({"model": model, "rung0": r0, "levels": levels, "gains": gain_ci})
    return out


def calibration_lookup(metrics: pd.DataFrame) -> pd.DataFrame:
    """rung0/common/overall calibration_slope + calibration_intercept rows, cluster_robust_t --
    axis A's source data (S6.1 item 1)."""
    return metrics[
        (metrics["metric"].isin(["calibration_slope", "calibration_intercept"]))
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["ci_method"] == "cluster_robust_t")
    ]


def miscalibration_reason(calibration: pd.DataFrame, model: str) -> str:
    """Which analytic CI(s) triggered axis A, named with their own numbers -- never a free
    adjective, just the values that triggered it (S6.1 item 1)."""
    sub = calibration[calibration["model_id"] == model]
    parts = []
    slope = _one(sub, metric="calibration_slope")
    if slope is not None and np.isfinite(slope["ci_low"]) and np.isfinite(slope["ci_high"]):
        if not (slope["ci_low"] <= 1.0 <= slope["ci_high"]):
            ci = _val_ci(float(slope["value"]), float(slope["ci_low"]), float(slope["ci_high"]))
            parts.append(f"slope {ci}")
    intercept = _one(sub, metric="calibration_intercept")
    if (
        intercept is not None
        and np.isfinite(intercept["ci_low"])
        and np.isfinite(intercept["ci_high"])
    ):
        if not (intercept["ci_low"] <= 0.0 <= intercept["ci_high"]):
            ci = _val_ci(
                float(intercept["value"]), float(intercept["ci_low"]), float(intercept["ci_high"])
            )
            parts.append(f"intercept {ci}")
    return ", ".join(parts) if parts else "calibration"


def harm_lookup(metrics: pd.DataFrame) -> dict[str, tuple[list[tuple[str, float, float]], int]]:
    """Per member: every ladder rung whose paired cross-fit gain interval lies entirely below zero,
    with that interval, plus how many rungs had a finite interval at all (S6.2 D5).

    Read from the same paired_gain_brier/common/overall rows the manifest's verdict was classified
    from, so the report and the manifest cannot disagree. The report needs the intervals themselves
    (the manifest carries only rung names), which is why this is a second read of the same rows
    rather than a manifest field.

    Independent of axis B by construction: a member with a demonstrated repair at rung2 can still
    carry a reliably harmful rung1, and that harm is reported.
    """
    gains = metrics[
        (metrics["metric"] == "paired_gain_brier")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["ci_method"] == "paired_cluster_bootstrap")
    ]
    out: dict[str, tuple[list[tuple[str, float, float]], int]] = {}
    for model, grp in gains.groupby("model_id"):
        harmful: list[tuple[str, float, float]] = []
        n_finite = 0
        for rung in _LADDER_RUNGS:
            row = _one(grp, rung=rung)
            if row is None:
                continue
            lo, hi = float(row["ci_low"]), float(row["ci_high"])
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            n_finite += 1
            if hi < 0:
                harmful.append((rung, lo, hi))
        out[str(model)] = (harmful, n_finite)
    return out


def _rung3_diagnosis_doc(manifest: dict) -> str | None:
    """Path of the rung3 withholding write-up for THIS cohort, or None when nothing was withheld.

    Derived from the stratum named in the flags rather than hardcoded (S6.2 D1). Deterministic by
    construction -- the filesystem is never consulted, since a render whose prose depended on the
    working directory would not be byte-reproducible.
    """
    strata = sorted({
        f["message"].split("(", 1)[1].split(")", 1)[0]
        for f in manifest.get("flags", [])
        if f.get("code") == "recalibration_unavailable"
        and f.get("message", "").startswith("rung3 ")
        and "(" in f.get("message", "")
    })
    if not strata:
        return None
    return f"docs/diagnosis-rung3-{strata[0].replace('=', '')}.md"


def few_clusters_note_for_overall(manifest: dict) -> str | None:
    """The already-computed few_clusters flag text for the overall stratum, quoted verbatim
    (never re-derived) for the A+B-unproven cell's power caveat (S6.1 item 1)."""
    for f in manifest.get("flags", []):
        if f.get("code") == "few_clusters" and "for overall" in f.get("message", ""):
            return f["message"]
    return None


def verdict_gauge(
    v: dict,
    reason: str,
    few_clusters: str | None,
    harm: list[tuple[str, float, float]] | tuple = (),
    n_rungs: int = 0,
) -> str:
    """The gauge-fault clause, from the two-axis verdict (S6.1 item 1; S6.2 D3/D5; spec §4.8).

    Built as a list of independent clauses rather than a lookup over the (axis A x axis B) grid,
    because the clauses are genuinely independent facts and one of them -- recalibration harm --
    cuts across both axes.

    Axis A speaks only when it fired. When it did not, it says NOTHING: "no miscalibration was
    detected at this cohort's power" is not the same claim as "well-calibrated", and printing the
    latter turned a null result into a clean bill of health (S6.2 D3). Silence is the honest
    report, so a member with nothing to say returns "" and the caller drops the sentence entirely.

    Axis B's "unproven" is likewise only worth stating when axis A gave a reason to attempt a
    repair; on a member with no detected fault, "we don't know if a fix would help" is noise.
    "demonstrated" and any harmful rung are always stated -- both are findings in their own right.
    """
    clauses: list[str] = []

    if v["axis_a_miscalibrated"]:
        clauses.append(f"miscalibrated ({reason})")

    if v["axis_b"] == "demonstrated":
        rung = v["best_rung"]
        clauses.append(f"repair demonstrated at {rung} ({_REPAIR_SHAPE_LABELS.get(rung, rung)})")
    elif v["axis_b"] == "unproven" and v["axis_a_miscalibrated"]:
        clause = f"repair benefit unproven at this cohort's power (G={v['n_clusters']})"
        clauses.append(f"{clause} -- {few_clusters}" if few_clusters else clause)

    if harm:
        # Scaled to 1e-3, exactly as section 3 shows the same gains (S6 item 4). At four raw
        # decimals the harmful bound nearest zero prints as "-0.0000", which reads as *not*
        # excluding zero -- the opposite of what the clause is asserting.
        rungs = ", ".join(
            f"{rung} [{lo * _MILLI:.2f}, {hi * _MILLI:.2f}]" for rung, lo, hi in harm
        )
        unit = "paired cross-fit Brier gain, x10\u207b\u00b3"
        if n_rungs and len(harm) >= n_rungs:
            clauses.append(
                f"recalibration harm at every rung, no admissible repair ({unit}): {rungs}"
            )
        else:
            clauses.append(f"recalibration harm ({unit}): {rungs}")

    return "; ".join(clauses)


def _verdict_lines(metrics: pd.DataFrame, manifest: dict, models: list[str]) -> list[dict]:
    """One template-generated plain-language line per member (S6.1 item 1; item 3ii before it).

    Assembled from (AUROC, BSS, the two-axis gauge): AUROC and BSS are inserted as numbers with
    their intervals; the gauge names axis A's exact triggering CI and axis B's exact rung(s) --
    no free adjectives. The line names its references -- the no-skill and perfect anchors of BSS --
    so it stands on its own. See docs/spec.md section 4.8 for the full taxonomy and templates.
    """
    verdict = manifest.get("verdict", {})
    auroc = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["ci_method"] == "cluster_bootstrap")
    ]
    calibration = calibration_lookup(metrics)
    few_clusters = few_clusters_note_for_overall(manifest)
    harm = harm_lookup(metrics)
    out = []
    for model in models:
        v = verdict.get(model)
        a = _one(auroc, model_id=model)
        if v is None or a is None:
            continue
        auroc_str = _val_ci(float(a["value"]), float(a["ci_low"]), float(a["ci_high"]))
        bss_pct = 100.0 * float(v["bss"])
        bss_ci = _ci(100.0 * float(v["bss_ci_low"]), 100.0 * float(v["bss_ci_high"]), nd=1)
        reason = miscalibration_reason(calibration, model) if v["axis_a_miscalibrated"] else ""
        harmful, n_rungs = harm.get(model, ([], 0))
        gauge = verdict_gauge(v, reason, few_clusters, harmful, n_rungs)
        line = (
            f"Ranking: AUROC {auroc_str}. "
            f"Probability quality after best admissible repair: closes {bss_pct:.1f}% "
            f"[{100.0 * float(v['bss_ci_low']):.1f}%, {100.0 * float(v['bss_ci_high']):.1f}%] "
            f"of the gap from no-skill (always predict prevalence) to perfect."
            # Dropped entirely when the gauge is silent (S6.2 D3): an empty "Gauge fault found:"
            # would reintroduce the reassurance the silence exists to withhold.
            + (f" Gauge fault found: {gauge}." if gauge else "")
        )
        out.append({
            "model": model,
            "auroc": auroc_str,
            "bss_pct": f"{bss_pct:.1f}",
            "bss_ci": bss_ci,
            "gauge": gauge,
            "line": line,
        })
    return out


def _subgroup_rows(metrics: pd.DataFrame) -> list[dict]:
    sub = metrics[
        (metrics["stratum_kind"] == "subgroup")
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
    ]
    out = []
    seen = sub[["subgroup_name", "subgroup_level", "model_id"]].drop_duplicates()
    for _, key in seen.sort_values(["subgroup_name", "subgroup_level", "model_id"]).iterrows():
        cell = sub[
            (sub["subgroup_name"] == key["subgroup_name"])
            & (sub["subgroup_level"] == key["subgroup_level"])
            & (sub["model_id"] == key["model_id"])
        ]
        auroc = _one(cell, metric="auroc", ci_method="cluster_bootstrap")
        brier = _one(cell, metric="brier", ci_method="cluster_bootstrap")  # loss: bootstrap body

        def fmt(r: pd.Series | None) -> str:
            return _val_ci(r["value"], r["ci_low"], r["ci_high"]) if r is not None else "n/a"

        out.append({
            "subgroup": f"{key['subgroup_name']}={key['subgroup_level']}",
            "model": key["model_id"],
            "auroc": fmt(auroc),
            "brier": fmt(brier),
        })
    return out


def _fragility_rows(fragility: pd.DataFrame, models: list[str]) -> list[dict]:
    if fragility.empty:
        return []
    frag = fragility[
        (fragility["stratum_kind"] == "overall")
        & (fragility["subset"] == "common")
        & (fragility["metric"] == "auroc")
    ]
    out = []
    for model in models:
        r = _one(frag, model_id=model)
        if r is None or not np.isfinite(r["max_abs_delta"]):
            continue
        out.append({
            "model": model,
            "max_abs_delta": _f(r["max_abs_delta"], 4),
            "culprit": str(r["culprit_cluster"]),
        })
    out.sort(key=lambda d: d["max_abs_delta"], reverse=True)
    return out


def _flags_by_severity(manifest: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {"warning": [], "note": []}
    for f in manifest.get("flags", []):
        grouped.setdefault(f["severity"], []).append(f)
    return grouped


# ------------------------------------------------------------------------------------ rendering


def _curves_by_model(calibration: pd.DataFrame) -> dict[str, list[dict]]:
    """calibration.parquet -> {model: [decile dicts]} for the section-2 figure."""
    if calibration is None or calibration.empty:
        return {}
    out: dict[str, list[dict]] = {}
    for model, grp in calibration.sort_values(["model_id", "bin"]).groupby("model_id", sort=True):
        out[str(model)] = [
            {
                "mean_pred": float(r["mean_pred"]),
                "obs_rate": float(r["obs_rate"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
            }
            for _, r in grp.iterrows()
        ]
    return out


def build_context(
    metrics: pd.DataFrame,
    fragility: pd.DataFrame,
    coverage: pd.DataFrame,
    manifest: dict,
    calibration: pd.DataFrame | None = None,
    *,
    appendix: bool = False,
) -> dict:
    """Assemble everything the template needs. All logic lives here; the template only arranges."""
    models = _models_by_auroc(metrics)
    cluster_noun = manifest.get("uncertainty", {}).get("clustering_name", "cluster")
    # Unit of observation vs unit of independence (S6.2 D1). Defaulted here as well as in the spec
    # so a manifest written by an older predval still renders.
    unit_noun = manifest.get("uncertainty", {}).get("unit_noun", "row")

    cov = coverage_delta(metrics)
    finite = cov["cov_delta"].dropna()
    identical_coverage = len(finite) > 0 and bool(np.allclose(finite, 0.0))

    coverage_rows = [
        {
            "model": r["model_id"],
            "coverage_full": _f(r["coverage_full"], 4),
            "n_scored": int(r["n_scored"]),
            "n_dropped_if_common": int(r["n_dropped_if_common"]),
            "meets_minimum": bool(r["meets_minimum"]),
        }
        for _, r in coverage.sort_values("model_id").iterrows()
    ]

    brier_footnote = _brier_analytic_footnote(metrics, models)
    exhibit_rows = _exhibit_rows(metrics, models, cluster_noun)

    return {
        "framing": FRAMING,
        "cohort_id": manifest.get("cohort_id", ""),
        "cluster_noun": cluster_noun,
        "cluster_nouns": _plural(cluster_noun),
        "unit_noun": unit_noun,
        "unit_nouns": _plural(unit_noun),
        "roster": manifest.get("roster", {}),
        "identical_coverage": identical_coverage,
        "coverage_rows": coverage_rows,
        "primary_rows": _primary_rows(metrics, models, show_cov_delta=not identical_coverage),
        "brier_footnote": brier_footnote,
        "analytic_truncated": bool(brier_footnote),
        "analytic_truncated_note": ANALYTIC_TRUNCATED_NOTE,
        "verdict_lines": _verdict_lines(metrics, manifest, models),
        "calibration_svg": figures.calibration_small_multiples(
            _curves_by_model(calibration), models
        ),
        "exhibit_rows": exhibit_rows,
        "exhibit_note": _exhibit_note(exhibit_rows, cluster_noun, unit_noun),
        "dumbbell_svg": figures.interval_dumbbell(_dumbbell_data(metrics, models)),
        "calibration_rows": _calibration_rows(metrics, models),
        "brier_rung_svg": figures.brier_by_rung(_brier_by_rung_data(metrics, models)),
        "subgroup_rows": _subgroup_rows(metrics),
        "fragility_rows": _fragility_rows(fragility, models),
        "limitations": _limitations(
            cluster_noun, manifest.get("roster", {}).get("ensemble_members", [])
        ),
        # Cross-reference the rung3 diagnosis doc only when THIS run actually has a
        # rung3-could-not-cross-fit withholding for it to be relevant to (S6 item 5), and name the
        # doc after the stratum THIS cohort withheld on rather than PCam's (S6.2 D1) -- the path
        # was hardcoded to scanner_domain0, so any other cohort that hit a withholding would have
        # been pointed at a PCam subgroup that does not exist in it.
        "rung3_diagnosis_doc": _rung3_diagnosis_doc(manifest),
        "provenance": {
            # Sorted, so the report is byte-identical whether it renders from the in-memory
            # manifest (insertion order) or from manifest.json (written with sort_keys).
            "inputs": sorted(manifest.get("inputs", {}).items()),
            "versions": sorted(manifest.get("versions", {}).items()),
            "contract_version": manifest.get("contract_version"),
            "uncertainty": manifest.get("uncertainty", {}),
        },
        "flags": _flags_by_severity(manifest),
        "pending": PENDING,
        "appendix": appendix,
        "appendix_sections": APPENDIX_SECTIONS if appendix else (),
    }


def render(
    metrics: pd.DataFrame,
    fragility: pd.DataFrame,
    coverage: pd.DataFrame,
    manifest: dict,
    calibration: pd.DataFrame | None = None,
    *,
    appendix: bool = False,
) -> str:
    """Render the report HTML from the artefacts."""
    env = Environment(autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(_template_source())
    return template.render(
        **build_context(metrics, fragility, coverage, manifest, calibration, appendix=appendix)
    )


def render_evaluation(evaluation: Evaluation, *, appendix: bool = False) -> str:
    """Render straight from an in-memory Evaluation."""
    return render(
        evaluation.metrics,
        evaluation.fragility,
        evaluation.coverage,
        evaluation.manifest,
        evaluation.calibration,
        appendix=appendix,
    )


def render_from_dir(outdir: str | Path, *, appendix: bool = False) -> str:
    """Render from a directory of written artefacts, so a report regenerates from hashes alone."""
    out = Path(outdir)
    cal_path = out / "calibration.parquet"
    calibration = pd.read_parquet(cal_path) if cal_path.exists() else None
    return render(
        pd.read_parquet(out / "metrics.parquet"),
        pd.read_parquet(out / "fragility.parquet"),
        pd.read_parquet(out / "coverage.parquet"),
        json.loads((out / "manifest.json").read_text()),
        calibration,
        appendix=appendix,
    )


def write_report(evaluation: Evaluation, path: str | Path, *, appendix: bool = False) -> Path:
    p = Path(path)
    p.write_text(render_evaluation(evaluation, appendix=appendix))
    return p




def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Render a predval HTML report from written artefacts.")
    ap.add_argument("outdir", type=Path, help="dir with metrics/fragility/coverage/manifest")
    ap.add_argument("--to", type=Path, default=None, help="output path (default: report.html)")
    ap.add_argument(
        "--appendix",
        action="store_true",
        help="append the concept-explainer appendix (off by default; does not change the "
        "default-report bytes)",
    )
    args = ap.parse_args(argv)
    html = render_from_dir(args.outdir, appendix=args.appendix)
    dest = args.to or (args.outdir / "report.html")
    dest.write_text(html)
    print(f"report -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
