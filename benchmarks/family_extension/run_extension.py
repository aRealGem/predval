"""Run the prospectively specified, local-only paired-family extension."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from scipy.stats import t
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predval import recalibrate

SEEDS = tuple(range(20261020, 20261040))
HERE = Path(__file__).resolve().parent
METHODS = ("raw", "predval_rung2", "statsmodels_logit", "statsmodels_probability")
OUTCOMES = (
    "brier",
    "auroc",
    "auroc_delta",
    "brier_gain",
    "risk_mse",
    "expected_brier",
    "expected_gain",
    "expected_recovery",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(y, p, groups, q=None):
    groups = np.asarray(groups).astype(str)
    lookup = {g: i % 5 for i, g in enumerate(np.unique(groups))}
    d = pd.DataFrame({"y": y, "p": p, "patient": groups, "fold": [lookup[g] for g in groups]})
    if q is not None:
        d["q"] = q
    return d


def paired_fixtures(seed):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(120), rng.integers(4, 21, 120))
    latent = rng.normal(0, 1.2, 120)
    x = rng.normal(size=len(ids))
    eta = -0.8 + 0.9 * x + latent[ids]
    p = expit(-0.4 + 1.4 * eta)
    shared = rng.uniform(size=120)
    u = np.where(rng.uniform(size=len(ids)) < 0.65, shared[ids], rng.uniform(size=len(ids)))
    groups = [f"patient_{i:03}" for i in ids]
    return {
        name: frame((u < q).astype(int), p, groups, q)
        for name, q in (("logit_linear", expit(eta)), ("probability_linear", expit(-2.5 + 5 * p)))
    }


def wdbc_control():
    data = load_breast_cancer()
    y = 1 - data.target
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.5, stratify=y, random_state=20261010
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, solver="lbfgs"))
    model.fit(data.data[train], y[train])
    return frame(y[test], model.predict_proba(data.data[test])[:, 1], [f"uci_{i:04}" for i in test])


def grouped_glm(d, logit_scale):
    x = logit(np.clip(d.p.to_numpy(), 1e-6, 1 - 1e-6)) if logit_scale else d.p.to_numpy()
    out = np.full(len(d), np.nan)
    for fold in sorted(d.fold.unique()):
        train = d.fold.to_numpy() != fold
        assert not set(d.patient[train]) & set(d.patient[~train])
        if np.unique(d.y[train]).size < 2:
            raise ValueError(f"single-class training fold {fold}")
        fit = sm.GLM(
            d.y.to_numpy()[train],
            sm.add_constant(x[train], has_constant="add"),
            family=sm.families.Binomial(),
        ).fit(tol=1e-11, maxiter=100)
        if not fit.converged:
            raise ValueError(f"non-convergence fold {fold}")
        out[~train] = fit.predict(sm.add_constant(x[~train], has_constant="add"))
    return out


def fit_methods(d):
    predictions = {"raw": d.p.to_numpy()}
    statuses = {"raw": {"available": True, "reason": None, "warnings": []}}
    for method in METHODS[1:]:
        extra = {}
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if method == "predval_rung2":
                    _, ladder = recalibrate.ladder(
                        d.y.to_numpy(), d.p.to_numpy(), d.patient.to_numpy(), max_folds=5
                    )
                    extra["ladder_suppressed"] = ladder.suppressed
                    if "rung2" not in ladder.crossfit_predictions:
                        raise ValueError(f"rung2 unavailable: {ladder.suppressed}")
                    prediction = ladder.crossfit_predictions["rung2"]
                else:
                    prediction = grouped_glm(d, logit_scale=method == "statsmodels_logit")
                if (
                    not np.isfinite(prediction).all()
                    or not ((prediction >= 0) & (prediction <= 1)).all()
                ):
                    raise ValueError("nonfinite or out-of-range prediction")
                predictions[method] = prediction
            statuses[method] = {"available": True, "reason": None, **extra}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            statuses[method] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                **extra,
            }
        statuses[method]["warnings"] = [f"{type(w.message).__name__}: {w.message}" for w in caught]
    if "predval_rung2" in predictions and "statsmodels_logit" in predictions:
        delta = float(
            np.max(np.abs(predictions["predval_rung2"] - predictions["statsmodels_logit"]))
        )
        statuses["predval_rung2"]["max_abs_independent_difference"] = delta
        assert delta < 1e-8, delta
    for method, prediction in predictions.items():
        statuses[method]["prediction_sha256"] = hashlib.sha256(
            np.asarray(prediction, dtype="<f8").tobytes()
        ).hexdigest()
    return predictions, statuses


def score(d, prediction):
    y, raw = d.y.to_numpy(), d.p.to_numpy()
    brier = float(np.mean((y - prediction) ** 2))
    auc = float(roc_auc_score(y, prediction)) if np.unique(y).size == 2 else None
    baseline_auc = float(roc_auc_score(y, raw)) if auc is not None else None
    result = {
        "brier": brier,
        "auroc": auc,
        "auroc_delta": auc - baseline_auc if auc is not None else None,
        "brier_gain": float(np.mean((y - raw) ** 2)) - brier,
    }
    if "q" in d:
        q = d.q.to_numpy()
        mse = float(np.mean((prediction - q) ** 2))
        raw_mse = float(np.mean((raw - q) ** 2))
        oracle = float(np.mean(q * (1 - q)))
        result.update(
            risk_mse=mse,
            oracle_expected_brier=oracle,
            expected_brier=oracle + mse,
            expected_gain=raw_mse - mse,
            expected_recovery=1 - mse / raw_mse if raw_mse > 0 else None,
        )
    return result


def mean_precision(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0, "mean": None, "mcse": None, "mc95_low": None, "mc95_high": None}
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    half = float(t.ppf(0.975, n - 1) * se) if se is not None else None
    return {
        "n": n,
        "mean": mean,
        "mcse": se,
        "mc95_low": mean - half if half is not None else None,
        "mc95_high": mean + half if half is not None else None,
    }


def summarize(rows):
    synthetic = [r for r in rows if r["family"] != "wdbc_undistorted"]
    summaries, contrasts = [], []
    for family in ("logit_linear", "probability_linear"):
        for method in METHODS:
            subset = [r for r in synthetic if r["family"] == family and r["method"] == method]
            for metric in OUTCOMES:
                values = [r[metric] for r in subset if r["available"] and r.get(metric) is not None]
                summaries.append(
                    {
                        "family": family,
                        "method": method,
                        "metric": metric,
                        "attempted": len(subset),
                        **mean_precision(values),
                    }
                )
        for metric in ("brier", "auroc", "expected_brier", "risk_mse"):
            values = []
            for seed in SEEDS:
                pair = {
                    r["method"]: r for r in synthetic if r["family"] == family and r["seed"] == seed
                }
                a, b = pair["predval_rung2"], pair["statsmodels_probability"]
                if (
                    a["available"]
                    and b["available"]
                    and a.get(metric) is not None
                    and b.get(metric) is not None
                ):
                    values.append(a[metric] - b[metric])
            contrasts.append(
                {
                    "family": family,
                    "contrast": "predval_minus_probability_glm",
                    "metric": metric,
                    "attempted": len(SEEDS),
                    **mean_precision(values),
                }
            )
    return {
        "simulation_summary": summaries,
        "paired_method_contrasts": contrasts,
        "limits": (
            "20 seeds; maximum binomial MCSE about 0.112. "
            "Monte Carlo intervals are not patient CIs."
        ),
    }


def run(out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    rows, cases = [], []
    fixtures = [(seed, name, d) for seed in SEEDS for name, d in paired_fixtures(seed).items()]
    fixtures.append((20261010, "wdbc_undistorted", wdbc_control()))
    for seed, family, d in fixtures:
        predictions, statuses = fit_methods(d)
        input_hash = hashlib.sha256(
            d.to_csv(index=False, float_format="%.17g").encode()
        ).hexdigest()
        cases.append(
            {"seed": seed, "family": family, "input_sha256": input_hash, "statuses": statuses}
        )
        for method in METHODS:
            row = {
                "seed": seed,
                "family": family,
                "method": method,
                "rows": len(d),
                "clusters": int(d.patient.nunique()),
                "events": int(d.y.sum()),
                "available": statuses[method]["available"],
                "reason": statuses[method]["reason"],
            }
            if method in predictions:
                row.update(score(d, predictions[method]))
            rows.append(row)
    for name, value in (
        ("metrics.json", rows),
        ("cases.json", cases),
        ("summary.json", summarize(rows)),
    ):
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, float_format="%.17g")
    manifest = {
        "python": platform.python_version(),
        "seeds": SEEDS,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "pandas", "statsmodels", "scikit-learn", "predval")
        },
        "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "runner_sha256": sha(Path(__file__)),
        "predval_source": str(Path(recalibrate.__file__).resolve()),
        "recalibrate_sha256": sha(Path(recalibrate.__file__)),
        "files": {p.name: sha(p) for p in sorted(out.iterdir())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "runtime.json").write_text(
        json.dumps(
            {
                "seconds": time.monotonic() - started,
                "scope": "combined workload, excludes imports/install",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "method_cases": len(rows),
                "unavailable": sum(not r["available"] for r in rows),
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
