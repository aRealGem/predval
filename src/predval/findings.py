"""findings.json -- the machine-readable companion to report.html (item 4).

Where the HTML report is for a human, findings.json is for a script: the same numbers, flags, and
verdict, in a shape a downstream tool can diff or gate on. Two properties make that trustworthy:

- **Byte-stable.** Keys are sorted and every float is rounded to a fixed precision before dumping,
  so regenerating the file from the same evaluation yields identical bytes. A findings file that
  drifts run to run cannot be a claim about specific inputs.
- **Schema-checked.** The document is a pydantic model; the JSON Schema at
  ``schema/findings.schema.json`` is generated from that model and checked in. Emitting validates
  against the model, and a test guards the checked-in schema against drift from the model.

The model is validated on the *stable* (rounded, NaN-nulled) document -- the exact bytes written --
so what is validated is what lands on disk.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .evaluate import RUNG0, Evaluation
from .report import (
    calibration_lookup,
    few_clusters_note_for_overall,
    miscalibration_reason,
    verdict_gauge,
)

#: Bumped when the findings shape changes in a way a consumer must notice.
SCHEMA_VERSION = "1.1"

#: Fixed float precision for byte-stability across runs and platforms.
FLOAT_NDIGITS = 6

_LADDER_RUNGS = ("rung1", "rung2", "rung3")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntervalOut(_Model):
    value: float | None
    ci_low: float | None
    ci_high: float | None
    ci_method: str | None


class MemberMetrics(_Model):
    auroc: IntervalOut
    average_precision: IntervalOut
    brier: IntervalOut
    calibration_slope: IntervalOut
    calibration_intercept: IntervalOut


class GainOut(_Model):
    rung: str
    gain: float | None
    ci_low: float | None
    ci_high: float | None
    ci_method: str | None


class VerdictOut(_Model):
    bss: float | None
    bss_ci_low: float | None
    bss_ci_high: float | None
    best_rung: str
    #: Axis A (S6.1 item 1): does rung0's own analytic calibration CI show miscalibration --
    #: slope excludes 1, or intercept excludes 0.
    axis_a_miscalibrated: bool
    #: Axis B: what happened when recalibration was tried -- "demonstrated", "unproven", or
    #: "counterproductive". Independent of axis A; see docs/spec.md section 4.8.
    axis_b: str
    gauge_label: str


class FlagOut(_Model):
    code: str
    severity: str
    message: str


class Member(_Model):
    model_id: str
    metrics: MemberMetrics
    verdict: VerdictOut | None
    recalibration_gains: list[GainOut]
    #: D2: the analytic t(G-1) Brier interval crossed the 0 loss boundary and was truncated.
    analytic_ci_truncated: bool


class Provenance(_Model):
    input_hashes: dict[str, str]
    seed: int
    n_boot: int
    ci_level: float
    git_commit: str
    versions: dict[str, str]
    config: dict[str, Any]


class Roster(_Model):
    declared: int | None
    present: int
    absent: list[str]


class FindingsDocument(_Model):
    schema_version: str
    cohort_id: str
    provenance: Provenance
    roster: Roster
    members: list[Member]
    flags: list[FlagOut]


# ------------------------------------------------------------------------------- extraction


def _stable(o: Any) -> Any:
    """Round floats and null out non-finite values, recursively -- the byte-stability step.

    JSON has no NaN/inf, and an unrounded float can differ in its last digit across runs; both are
    silenced here so the written bytes are reproducible.
    """
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        return round(o, FLOAT_NDIGITS) if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _stable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_stable(v) for v in o]
    return o


def _one(df, **conds):
    mask = None
    for col, want in conds.items():
        m = df[col].isna() if want is None else (df[col] == want)
        mask = m if mask is None else (mask & m)
    hit = df[mask]
    return None if hit.empty else hit.iloc[0]


def _interval(row) -> dict:
    if row is None:
        return {"value": None, "ci_low": None, "ci_high": None, "ci_method": None}
    return {
        "value": float(row["value"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "ci_method": None if row["ci_method"] is None else str(row["ci_method"]),
    }


def _git_commit(cwd: str | Path | None) -> str:
    """The current commit, with a `-dirty` suffix when the tree has uncommitted changes.

    Provenance is environmental by nature; a failure to read git is recorded as 'unknown' rather
    than raised, so findings still generates outside a checkout.
    """
    try:
        root = str(cwd) if cwd is not None else "."
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=root, check=True
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except Exception:  # noqa: BLE001 - provenance read must never crash the report
        return "unknown"


def build_findings(evaluation: Evaluation, *, git_commit: str | None = None) -> dict:
    """Assemble the findings document (a plain dict), byte-stabilised and schema-validated."""
    metrics = evaluation.metrics
    manifest = evaluation.manifest
    verdict = manifest.get("verdict", {})

    common_overall = metrics[
        (metrics["subset"] == "common") & (metrics["stratum_kind"] == "overall")
    ]
    r0 = common_overall[common_overall["rung"] == RUNG0]
    gains = common_overall[
        (common_overall["metric"] == "paired_gain_brier")
        & (common_overall["fit_mode"] == "crossfit")
    ]
    calibration = calibration_lookup(metrics)
    few_clusters = few_clusters_note_for_overall(manifest)

    members = []
    for model in sorted(manifest.get("models_present", [])):
        r0m = r0[r0["model_id"] == model]
        member_metrics = {
            "auroc": _interval(_one(r0m, metric="auroc", ci_method="cluster_bootstrap")),
            "average_precision": _interval(
                _one(r0m, metric="average_precision", ci_method="cluster_bootstrap")
            ),
            "brier": _interval(_one(r0m, metric="brier", ci_method="cluster_bootstrap")),
            "calibration_slope": _interval(
                _one(r0m, metric="calibration_slope", ci_method="cluster_robust_t")
            ),
            "calibration_intercept": _interval(
                _one(r0m, metric="calibration_intercept", ci_method="cluster_robust_t")
            ),
        }
        v = verdict.get(model)
        if v is not None:
            reason = miscalibration_reason(calibration, model) if v["axis_a_miscalibrated"] else ""
            verdict_out = {
                "bss": float(v["bss"]),
                "bss_ci_low": float(v["bss_ci_low"]),
                "bss_ci_high": float(v["bss_ci_high"]),
                "best_rung": str(v["best_rung"]),
                "axis_a_miscalibrated": bool(v["axis_a_miscalibrated"]),
                "axis_b": str(v["axis_b"]),
                "gauge_label": verdict_gauge(v, reason, few_clusters),
            }
        else:
            verdict_out = None
        gm = gains[gains["model_id"] == model]
        recal_gains = []
        for rung in _LADDER_RUNGS:
            g = _one(gm, rung=rung)
            if g is None:
                continue
            recal_gains.append({
                "rung": rung,
                "gain": float(g["value"]),
                "ci_low": float(g["ci_low"]),
                "ci_high": float(g["ci_high"]),
                "ci_method": None if g["ci_method"] is None else str(g["ci_method"]),
            })
        analytic = _one(
            r0m, metric="brier", ci_method="cluster_robust_t"
        )
        truncated = bool(
            analytic is not None
            and math.isfinite(float(analytic["ci_low"]))
            and float(analytic["ci_low"]) < 0.0
        )
        members.append({
            "model_id": model,
            "metrics": member_metrics,
            "verdict": verdict_out,
            "recalibration_gains": recal_gains,
            "analytic_ci_truncated": truncated,
        })

    unc = manifest.get("uncertainty", {})
    cov = manifest.get("coverage", {})
    doc = {
        "schema_version": SCHEMA_VERSION,
        "cohort_id": manifest.get("cohort_id", ""),
        "provenance": {
            "input_hashes": dict(manifest.get("inputs", {})),
            "seed": int(unc.get("seed", 0)),
            "n_boot": int(unc.get("n_boot", 0)),
            "ci_level": float(unc.get("ci_level", 0.0)),
            "git_commit": git_commit if git_commit is not None else _git_commit(None),
            "versions": dict(manifest.get("versions", {})),
            "config": {
                "clustering_field": unc.get("clustering_field"),
                "clustered": bool(unc.get("clustered", False)),
                "coverage_min_fraction": cov.get("min_fraction"),
                "coverage_compare_on": cov.get("compare_on"),
                "expected_models": manifest.get("expected_models"),
            },
        },
        "roster": {
            "declared": manifest.get("roster", {}).get("declared"),
            "present": int(manifest.get("roster", {}).get("present", 0)),
            "absent": list(manifest.get("roster", {}).get("absent", [])),
        },
        "members": members,
        "flags": [
            {"code": f["code"], "severity": f["severity"], "message": f["message"]}
            for f in manifest.get("flags", [])
        ],
    }
    stable = _stable(doc)
    FindingsDocument.model_validate(stable)  # raises on any shape violation
    return stable


def dumps_findings(evaluation: Evaluation, *, git_commit: str | None = None) -> str:
    """The findings document as byte-stable JSON text (sorted keys, fixed float precision)."""
    return json.dumps(build_findings(evaluation, git_commit=git_commit), sort_keys=True, indent=2)


def write_findings(
    evaluation: Evaluation, path: str | Path, *, git_commit: str | None = None
) -> Path:
    p = Path(path)
    p.write_text(dumps_findings(evaluation, git_commit=git_commit))
    return p


def findings_schema() -> dict:
    """The JSON Schema for a findings document, generated from the pydantic model."""
    return FindingsDocument.model_json_schema()
