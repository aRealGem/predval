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

#: Limitations block (item 6b), verbatim in report and docs/spec.md section 9. The optimism
#: correction disciplines the recalibration step; it says nothing about how the predictions were
#: built, and it must not be read as if it did.
LIMITATIONS = (
    "The optimism correction in section 3 covers the recalibration step ONLY -- it does not cover "
    "model or ensemble construction. If the ensemble weights (the champion) were selected on "
    "slides inside this cohort, rung0 is itself optimistically biased, and this harness cannot "
    "detect it: predval evaluates the predictions it is handed and has no view of how they were "
    "produced. Only a cohort the ensemble was never tuned on could expose that bias."
)

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

#: The "gauge fault" bin: which rung was the best admissible repair -> a fixed label, no adjectives
#: (item 3ii). rung0 means none was needed. Documented in docs/spec.md section 4.8.
GAUGE_LABELS = {
    "rung0": "none -- well-calibrated as published",
    "rung1": "level (calibration-in-the-large)",
    "rung2": "level and spread (intercept + slope)",
    "rung3": "non-monotone shape",
}


# --------------------------------------------------------------------------- formatting helpers


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


def _exhibit_rows(metrics: pd.DataFrame, models: list[str]) -> list[dict]:
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
        out.append({
            "model": model,
            "cluster_width": _f(cw, 4),
            "naive_width": _f(nw, 4),
            "ratio": f"{cw / nw:.1f}x",
        })
    return out


def _signed_ci(v: float, lo: float, hi: float) -> str:
    """A signed point estimate with its interval: '+0.0112 [0.005, 0.017]', or 'n/a'."""
    if v is None or not np.isfinite(v):
        return "n/a"
    ci = _ci(lo, hi)
    return f"{v:+.4f} {ci}".strip()


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
            cells[f"{rung}_gain"] = _signed_ci(v, lo, hi)
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


def _verdict_lines(metrics: pd.DataFrame, manifest: dict, models: list[str]) -> list[dict]:
    """One template-generated plain-language line per member (item 3ii).

    Assembled only from (AUROC, BSS, best rung): AUROC and BSS are inserted as numbers with their
    intervals, and the single qualitative token is the gauge-fault rung label (GAUGE_LABELS). No
    free adjectives. The line names its references -- the no-skill and perfect anchors of BSS -- so
    it stands on its own. See docs/spec.md section 4.8 for the template and bins.
    """
    verdict = manifest.get("verdict", {})
    auroc = metrics[
        (metrics["metric"] == "auroc")
        & (metrics["rung"] == "rung0")
        & (metrics["subset"] == "common")
        & (metrics["stratum_kind"] == "overall")
        & (metrics["ci_method"] == "cluster_bootstrap")
    ]
    out = []
    for model in models:
        v = verdict.get(model)
        a = _one(auroc, model_id=model)
        if v is None or a is None:
            continue
        auroc_str = _val_ci(float(a["value"]), float(a["ci_low"]), float(a["ci_high"]))
        bss_pct = 100.0 * float(v["bss"])
        bss_ci = _ci(100.0 * float(v["bss_ci_low"]), 100.0 * float(v["bss_ci_high"]), nd=1)
        gauge = GAUGE_LABELS.get(v["best_rung"], v["best_rung"])
        line = (
            f"Ranking: AUROC {auroc_str}. "
            f"Probability quality after best admissible repair: closes {bss_pct:.1f}% "
            f"[{100.0 * float(v['bss_ci_low']):.1f}%, {100.0 * float(v['bss_ci_high']):.1f}%] "
            f"of the gap from no-skill (always predict prevalence) to perfect. "
            f"Gauge fault found: {gauge}."
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

    return {
        "framing": FRAMING,
        "cohort_id": manifest.get("cohort_id", ""),
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
        "exhibit_rows": _exhibit_rows(metrics, models),
        "dumbbell_svg": figures.interval_dumbbell(_dumbbell_data(metrics, models)),
        "calibration_rows": _calibration_rows(metrics, models),
        "brier_rung_svg": figures.brier_by_rung(_brier_by_rung_data(metrics, models)),
        "subgroup_rows": _subgroup_rows(metrics),
        "fragility_rows": _fragility_rows(fragility, models),
        "limitations": LIMITATIONS,
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
