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

from .evaluate import Evaluation, coverage_delta
from .metrics import optimism

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

#: Attached to every recalibrated gain. Ladder rungs carry no interval yet (section 4.5).
PENDING = "(interval pending section 4.5)"

_LADDER_RUNGS = ("rung1", "rung2", "rung3")


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
            "brier": cell("brier", "cluster_robust_t"),
            "calibration_slope": cell("calibration_slope", "cluster_robust_t"),
            "calibration_intercept": cell("calibration_intercept", "cluster_robust_t"),
        }
        if show_cov_delta:
            d = cov_auroc.get(model, float("nan"))
            row["cov_delta"] = _f(d, 4) if np.isfinite(d) else "0.0000"
        out.append(row)
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


def _calibration_rows(metrics: pd.DataFrame, models: list[str]) -> list[dict]:
    """Ladder diagnosis: rung0 Brier vs cross-fitted rungs, with the naked-delta ban applied."""
    overall = metrics[(metrics["subset"] == "common") & (metrics["stratum_kind"] == "overall")]

    def brier(model: str, rung: str, mode: str) -> float:
        r = _one(overall, model_id=model, metric="brier", rung=rung, fit_mode=mode, threshold=None)
        return float(r["value"]) if r is not None else float("nan")

    out = []
    for model in models:
        r0 = brier(model, "rung0", "apparent")
        cf = {rung: brier(model, rung, "crossfit") for rung in _LADDER_RUNGS}
        finite_cf = [v for v in cf.values() if np.isfinite(v)]
        best_cf = min(finite_cf) if finite_cf else float("nan")
        gain = optimism("brier", best_cf, r0)  # oriented: positive == the cross-fit helped
        opt = optimism("brier", brier(model, "rung2", "apparent"), cf["rung2"])
        out.append({
            "model": model,
            "rung0": _f(r0),
            "rung1_cf": _f(cf["rung1"]),
            "rung2_cf": _f(cf["rung2"]),
            "rung3_cf": _f(cf["rung3"]),
            "gain": f"{_f(gain)} {PENDING}",
            "optimism": f"{_f(opt)} {PENDING}",
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
        brier = _one(cell, metric="brier", ci_method="cluster_robust_t")

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


def build_context(
    metrics: pd.DataFrame, fragility: pd.DataFrame, coverage: pd.DataFrame, manifest: dict
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

    return {
        "framing": FRAMING,
        "cohort_id": manifest.get("cohort_id", ""),
        "roster": manifest.get("roster", {}),
        "identical_coverage": identical_coverage,
        "coverage_rows": coverage_rows,
        "primary_rows": _primary_rows(metrics, models, show_cov_delta=not identical_coverage),
        "exhibit_rows": _exhibit_rows(metrics, models),
        "calibration_rows": _calibration_rows(metrics, models),
        "subgroup_rows": _subgroup_rows(metrics),
        "fragility_rows": _fragility_rows(fragility, models),
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
    }


def render(
    metrics: pd.DataFrame, fragility: pd.DataFrame, coverage: pd.DataFrame, manifest: dict
) -> str:
    """Render the report HTML from the four artefacts."""
    env = Environment(autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(_template_source())
    return template.render(**build_context(metrics, fragility, coverage, manifest))


def render_evaluation(evaluation: Evaluation) -> str:
    """Render straight from an in-memory Evaluation."""
    return render(
        evaluation.metrics, evaluation.fragility, evaluation.coverage, evaluation.manifest
    )


def render_from_dir(outdir: str | Path) -> str:
    """Render from a directory of written artefacts, so a report regenerates from hashes alone."""
    out = Path(outdir)
    return render(
        pd.read_parquet(out / "metrics.parquet"),
        pd.read_parquet(out / "fragility.parquet"),
        pd.read_parquet(out / "coverage.parquet"),
        json.loads((out / "manifest.json").read_text()),
    )


def write_report(evaluation: Evaluation, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(render_evaluation(evaluation))
    return p




def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Render a predval HTML report from written artefacts.")
    ap.add_argument("outdir", type=Path, help="dir with metrics/fragility/coverage/manifest")
    ap.add_argument("--to", type=Path, default=None, help="output path (default: report.html)")
    args = ap.parse_args(argv)
    html = render_from_dir(args.outdir)
    dest = args.to or (args.outdir / "report.html")
    dest.write_text(html)
    print(f"report -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
