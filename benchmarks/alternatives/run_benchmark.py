#!/usr/bin/env python3
"""Prespecified, bounded alternatives benchmark. See PROTOCOL.md before running."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from pycaleva import CalibrationEvaluator
from pycaleva.metrics import brier as pycaleva_brier
from scipy.special import expit, logit
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predval import metrics as pm
from predval import recalibrate as pr
from predval import uncertainty as pu
from predval.errors import CohortSpecError, CoverageViolation
from predval.io import (
    check_common_selection,
    check_coverage,
    check_roster,
    coverage_report,
    load_cohort,
    load_predictions,
)

SEED = 20261010
BOOT_SEED = 20261011
N_BOOT = 1000
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(y, p, patient):
    groups = np.asarray(patient).astype(str)
    fold_lookup = {g: i % 5 for i, g in enumerate(np.unique(groups))}
    return pd.DataFrame(
        {
            "subject_id": [f"row_{i:05}" for i in range(len(y))],
            "patient_id": groups,
            "y": np.asarray(y, dtype=int),
            "p": p,
            "fold": [fold_lookup[g] for g in groups],
        }
    )


def fixtures():
    data = load_breast_cancer()
    y = 1 - data.target  # UCI malignant is the positive event.
    train, test = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=SEED)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, solver="lbfgs"))
    model.fit(data.data[train], y[train])
    p = model.predict_proba(data.data[test])[:, 1]
    real = frame(
        y[test],
        expit(-0.4 + 1.4 * logit(np.clip(p, 1e-6, 1 - 1e-6))),
        [f"uci_{i:04}" for i in test],
    )
    rng = np.random.default_rng(SEED)
    sizes = rng.integers(4, 21, 120)
    ids = np.repeat(np.arange(120), sizes)
    latent = rng.normal(0, 1.2, 120)
    x = rng.normal(size=len(ids))
    risk = expit(-0.8 + 0.9 * x + latent[ids])
    # A shared uniform induces positive within-patient outcome dependence while
    # preserving each row's conditional Bernoulli marginal probability.
    shared = rng.uniform(size=120)
    uniforms = np.where(rng.uniform(size=len(ids)) < 0.65, shared[ids], rng.uniform(size=len(ids)))
    sy = (uniforms < risk).astype(int)
    sp = expit(-0.4 + 1.4 * (-0.8 + 0.9 * x + latent[ids]))
    synthetic = frame(sy, sp, [f"patient_{i:03}" for i in ids])
    return {"wdbc_heldout": real, "synthetic_clustered": synthetic}


def metrics(y, p):
    return {"auroc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)}


def reference_calibration(y, p):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    offset = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z)
    joint = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial())
    a = offset.fit(tol=1e-11, maxiter=100)
    b = joint.fit(tol=1e-11, maxiter=100)
    assert a.converged and b.converged
    return {"calibration_intercept": a.params[0], "calibration_slope": b.params[1]}


def reference_crossfit(df, *, logit_scale=True):
    y, p = df.y.to_numpy(), df.p.to_numpy()
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)) if logit_scale else p
    out = np.full(len(df), np.nan)
    for f in sorted(df.fold.unique()):
        train = df.fold.to_numpy() != f
        fit = sm.GLM(
            y[train],
            sm.add_constant(x[train]),
            family=sm.families.Binomial(),
        ).fit(tol=1e-11, maxiter=100)
        assert fit.converged
        out[~train] = fit.predict(sm.add_constant(x[~train]))
    assert np.isfinite(out).all()
    return out


def custom_bootstrap(y, p, corrected, groups):
    """Adapter code, NOT a native PyCalEva uncertainty feature."""
    clusters = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(BOOT_SEED)
    base, gain = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([clusters[k] for k in rng.integers(0, len(clusters), len(clusters))])
        b = pycaleva_brier(y[idx], p[idx])
        base.append(b)
        gain.append(b - pycaleva_brier(y[idx], corrected[idx]))
    return np.quantile(base, [0.025, 0.975]), np.quantile(gain, [0.025, 0.975])


def coverage_cases(df, out):
    case = out / "coverage_inputs"
    case.mkdir()
    df.to_csv(case / "cohort.csv", index=False, float_format="%.17g")
    score = df.groupby("patient_id").p.mean()
    excluded = set(score[score >= score.quantile(0.7)].index)
    keep = ~df.patient_id.isin(excluded)
    a = df[["subject_id", "p"]].rename(columns={"p": "predicted"}).assign(model_id="full")
    b = df.loc[keep, ["subject_id", "p"]].rename(columns={"p": "predicted"})
    b = b.assign(model_id="partial")  # Identical predictions; only availability differs.
    pd.concat([a, b]).to_csv(case / "predictions.csv", index=False, float_format="%.17g")
    spec = {
        "cohort_id": "controlled_coverage",
        "version": 0,
        "subject_key": "subject_id",
        "data": "cohort.csv",
        "outcome": {"type": "binary", "field": "y", "positive_label": 1},
        "clustering": {"field": "patient_id", "name": "patient"},
        "coverage": {"min_fraction": 0.9, "compare_on": "both", "common_warn_frac": 0.2},
        "completeness": {"require_outcome": True, "on_violation": "fail"},
        "expected_models": ["full", "partial", "absent"],
        "on_missing_model": "warn",
    }
    spec_path = case / "cohort.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    cohort, preds = load_cohort(spec_path), load_predictions(case / "predictions.csv")
    report = coverage_report(cohort, preds)
    report.to_csv(out / "coverage.csv", index=False, float_format="%.17g")
    flags = check_roster(cohort, preds) + check_common_selection(cohort, preds)
    errors = []
    try:
        check_coverage(cohort, preds)
    except CoverageViolation as exc:
        errors.append(
            {"check": "minimum_coverage_90_percent", "status": "rejected", "error": str(exc)}
        )
    else:
        raise AssertionError("Missing coverage was not rejected")
    spec["on_missing_model"] = "fail"
    spec_path.write_text(yaml.safe_dump(spec))
    try:
        check_roster(load_cohort(spec_path), preds)
    except CohortSpecError as exc:
        # Avoid run-specific absolute paths in deterministic outputs.
        errors.append(
            {
                "check": "declared_absent_model",
                "status": "rejected",
                "error": exc.message if hasattr(exc, "message") else str(exc).split("\n")[0],
            }
        )
    else:
        raise AssertionError("Missing declared model was not rejected")
    rows = []
    for name, selection in [
        ("full_available", np.ones(len(df), dtype=bool)),
        ("common_intersection", keep.to_numpy()),
    ]:
        for m, value in metrics(df.y.to_numpy()[selection], df.p.to_numpy()[selection]).items():
            rows.append({"subset": name, "metric": m, "value": value, "n": int(selection.sum())})
    pd.DataFrame(rows).to_csv(
        out / "coverage_selection_metrics.csv", index=False, float_format="%.17g"
    )
    return {
        "flags": [f.as_dict() for f in flags],
        "errors": errors,
        "pycaleva_scope": (
            "Accepts aligned y/p arrays; roster and cohort denominators need caller glue."
        ),
    }


def run(out):
    start, cpu = time.monotonic(), time.process_time()
    out.mkdir(parents=True, exist_ok=False)
    point_rows, correction_rows, interval_rows, summary = [], [], [], {}
    for name, df in fixtures().items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.17g")
        y, p, groups = df.y.to_numpy(), df.p.to_numpy(), df.patient_id.to_numpy()
        ce = CalibrationEvaluator(y, p, outsample=True, n_groups=10)
        estimates = {
            "predval": {
                "auroc": pm.auroc(y, p),
                "brier": pm.brier(y, p),
                "calibration_intercept": pm.calibration_intercept(y, p),
                "calibration_slope": pm.calibration_slope(y, p),
            },
            "pycaleva": {"auroc": ce.auroc, "brier": ce.brier},
            "independent_reference": metrics(y, p) | reference_calibration(y, p),
        }
        for tool, values in estimates.items():
            for metric, value in values.items():
                ref = estimates["independent_reference"][metric]
                tolerance = 1e-10 if metric in {"auroc", "brier"} else 1e-8
                difference = abs(value - ref)
                assert difference < tolerance, (name, tool, metric, difference)
                point_rows.append(
                    {
                        "fixture": name,
                        "tool": tool,
                        "metric": metric,
                        "value": value,
                        "abs_diff_reference": difference,
                    }
                )
        _, ladder = pr.ladder(y, p, groups, max_folds=5)
        predval_corrected = ladder.crossfit_predictions["rung2"]
        reference_corrected = reference_crossfit(df)
        max_diff = float(np.max(np.abs(predval_corrected - reference_corrected)))
        assert max_diff < 1e-8, max_diff
        df["predval_rung2"] = predval_corrected
        df["statsmodels_rung2"] = reference_corrected
        raw_reference = reference_crossfit(df, logit_scale=False)
        df["statsmodels_raw_probability"] = raw_reference
        df.to_csv(out / f"{name}_corrected.csv", index=False, float_format="%.17g")
        for tool, corrected in [
            ("predval_rung2", predval_corrected),
            ("statsmodels_custom_grouped", reference_corrected),
            ("statsmodels_raw_probability", raw_reference),
        ]:
            for metric, value in metrics(y, corrected).items():
                correction_rows.append(
                    {"fixture": name, "tool": tool, "metric": metric, "value": value}
                )
        point, ci = pu.paired_brier_gain_interval(
            y, p, predval_corrected, groups, 0.95, n_boot=N_BOOT, seed=BOOT_SEED
        )
        for scheme, g in [("patient", groups), ("row", np.arange(len(df)))]:
            base, gain = custom_bootstrap(y, p, predval_corrected, g)
            for metric, interval in [("brier", base), ("paired_brier_gain", gain)]:
                interval_rows.append(
                    {
                        "fixture": name,
                        "tool": "pycaleva_plus_custom_bootstrap",
                        "scheme": scheme,
                        "metric": metric,
                        "low": interval[0],
                        "high": interval[1],
                        "width": interval[1] - interval[0],
                    }
                )
            if scheme == "patient":
                assert np.allclose(gain, [ci.low, ci.high], atol=1e-12, rtol=0)
        interval_rows.append(
            {
                "fixture": name,
                "tool": "predval",
                "scheme": "patient",
                "metric": "paired_brier_gain",
                "low": ci.low,
                "high": ci.high,
                "width": ci.high - ci.low,
            }
        )
        summary[name] = {
            "rows": len(df),
            "patients": len(np.unique(groups)),
            "events": int(y.sum()),
            "max_corrected_prediction_difference": max_diff,
            "brier_gain": point,
            "brier_gain_ci": [ci.low, ci.high],
            "suppressed_ladder_rungs": ladder.suppressed,
        }
    for name, rows in [
        ("point_metrics", point_rows),
        ("correction_metrics", correction_rows),
        ("intervals", interval_rows),
    ]:
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False, float_format="%.17g")
    summary["coverage"] = coverage_cases(fixtures()["synthetic_clustered"], out)
    summary["scope"] = "Executed: predval and PyCalEva; statsmodels/sklearn independent reference."
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    packages = ["predval", "pycaleva", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels"]
    manifest = {
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "seed": SEED,
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_replicates": N_BOOT,
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "files": {
            str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    usage = {
        "wall_seconds": time.monotonic() - start,
        "cpu_seconds": time.process_time() - cpu,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": (
            "Benchmark body, not a fair tool runtime comparison; excludes imports and installation."
        ),
    }
    (out / "resource_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(json.dumps(usage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
