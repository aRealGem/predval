"""Evaluation orchestration: cohort + predictions in, metrics/fragility/manifest out.

Structure follows docs/spec.md section 6. Both subsets are always computed, every stratum is
evaluated, and every metric carries an interval whose method is recorded rather than assumed.

As-published metrics are `rung0`/`apparent`. The recalibration ladder (§4, recalibrate.py) adds
`rung1`–`rung3` rows in both `apparent` and `crossfit` fit modes for the calibration-sensitive
metrics; their optimism is `apparent − crossfit`, read back from the paired rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import recalibrate as R
from . import uncertainty as U
from .io import Cohort, Flag, Predictions, check_common_selection, check_roster, coverage_report

#: As-published metrics are fitted on the rows they are evaluated on. S3's ladder adds "crossfit".
APPARENT = "apparent"
#: As-published rung. The recalibration ladder (S3) adds rung1-3; see docs/spec.md section 4.
RUNG0 = "rung0"


@dataclass
class Evaluation:
    """Everything a report needs, and nothing a report has to recompute."""

    metrics: pd.DataFrame
    fragility: pd.DataFrame
    coverage: pd.DataFrame
    manifest: dict
    flags: list[Flag] = field(default_factory=list)

    def write(self, outdir: str | Path) -> dict[str, Path]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": out / "metrics.parquet",
            "fragility": out / "fragility.parquet",
            "coverage": out / "coverage.parquet",
            "manifest": out / "manifest.json",
        }
        self.metrics.to_parquet(paths["metrics"], index=False)
        self.fragility.to_parquet(paths["fragility"], index=False)
        self.coverage.to_parquet(paths["coverage"], index=False)
        paths["manifest"].write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        return paths


def coverage_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    """`metric_full - metric_common` per model and metric.

    The report's primary table is the common subset; this is the column that keeps the effect
    of restriction visible without printing a second full table (docs/spec.md section 2.4).
    """
    keys = ["model_id", "stratum_kind", "subgroup_name", "subgroup_level", "metric", "threshold"]
    # As-published only: coverage restriction is a property of rung0, and mixing recalibrated
    # rungs into the pivot would compare corrections rather than the effect of the common subset.
    primary = metrics[(metrics["ci_method"] != "naive_row_bootstrap") & (metrics["rung"] == RUNG0)]
    wide = primary.pivot_table(
        index=keys, columns="subset", values="value", aggfunc="first", dropna=False
    ).reset_index()
    if "full" not in wide or "common" not in wide:
        return pd.DataFrame(columns=[*keys, "cov_delta"])
    wide["cov_delta"] = wide["full"] - wide["common"]
    return wide[[*keys, "cov_delta"]]


def _strata(cohort: Cohort) -> list[tuple[str, str | None, str | None, np.ndarray]]:
    """(stratum_kind, subgroup_name, subgroup_level, boolean mask over the cohort table)."""
    table = cohort.table
    out: list[tuple[str, str | None, str | None, np.ndarray]] = [
        ("overall", None, None, np.ones(len(table), dtype=bool))
    ]
    for sub in cohort.spec.subgroups:
        for level in sorted(pd.unique(table[sub.field].dropna()), key=str):
            out.append(("subgroup", sub.name, str(level), (table[sub.field] == level).to_numpy()))
    return out


def _subsets(cohort: Cohort, predictions: Predictions) -> dict[str, set[str]]:
    """The `full` and `common` analysis sets, as subject-id sets."""
    subjects = set(cohort.subject_ids.astype(str))
    scored = {
        model: set(grp["subject_id"]) & subjects
        for model, grp in predictions.frame.groupby("model_id", sort=True)
    }
    common = set.intersection(*scored.values()) if scored else set()
    return {"full": subjects, "common": common}


def evaluate(cohort: Cohort, predictions: Predictions) -> Evaluation:
    """Compute every metric, for every model, subset and stratum."""
    spec = cohort.spec
    unc = spec.uncertainty
    ci = unc.ci_level

    flags = check_roster(cohort, predictions) + check_common_selection(cohort, predictions)

    table = cohort.table
    subject_ids = table[spec.subject_key].astype(str).to_numpy()
    y_all = (table[spec.outcome.field] == spec.outcome.positive_label).to_numpy().astype(np.int64)
    if spec.clustering is not None:
        groups_all = table[spec.clustering.field].astype(str).to_numpy()
        clustered = True
    else:
        # No declared clustering: every subject is its own cluster, which makes the cluster
        # bootstrap degenerate to a row bootstrap. The manifest records that this happened.
        groups_all = subject_ids
        clustered = False
        flags.append(
            Flag(
                code="no_clustering_declared",
                severity="note",
                message=(
                    "clustering.field is not declared; subjects are treated as independent "
                    "and intervals assume that"
                ),
            )
        )

    position = {sid: i for i, sid in enumerate(subject_ids)}
    subsets = _subsets(cohort, predictions)
    strata = _strata(cohort)

    # Per-model prediction vectors aligned to cohort row order; NaN where unscored, which is how
    # the absence rule survives into the metric layer rather than becoming a zero.
    preds: dict[str, np.ndarray] = {}
    for model, grp in predictions.frame.groupby("model_id", sort=True):
        vec = np.full(len(table), np.nan)
        idx = np.array([position[s] for s in grp["subject_id"] if s in position], dtype=int)
        vals = grp.loc[[s in position for s in grp["subject_id"]], "predicted"].to_numpy()
        vec[idx] = vals
        preds[model] = vec

    rows: list[dict] = []
    frag_rows: list[dict] = []
    notes: set[tuple[str, str]] = set()

    # A cell is fully determined by (model, analysis-set mask). When every model has identical
    # coverage the `full` and `common` masks coincide, and recomputing 2000 bootstrap replicates
    # to arrive at the same numbers twice is pure waste. Reuse is exact, not an approximation:
    # identical inputs, identical outputs. The subset label is re-stamped on the copy.
    cache: dict[tuple[str, bytes], tuple[list[dict], list[dict], set[tuple[str, str]]]] = {}
    n_reused = 0

    for subset_name, subset_ids in subsets.items():
        in_subset = np.fromiter((s in subset_ids for s in subject_ids), bool, len(subject_ids))
        for stratum_kind, sg_name, sg_level, stratum_mask in strata:
            base_mask = in_subset & stratum_mask
            for model, pvec in preds.items():
                mask = base_mask & np.isfinite(pvec)
                ctx = dict(
                    model_id=model,
                    subset=subset_name,
                    stratum_kind=stratum_kind,
                    subgroup_name=sg_name,
                    subgroup_level=sg_level,
                )
                key = (model, mask.tobytes())
                if key in cache:
                    cached_rows, cached_frag, cached_notes = cache[key]
                    new_rows = [{**r, **ctx} for r in cached_rows]
                    new_frag = [{**r, **ctx} for r in cached_frag]
                    new_notes = cached_notes
                    n_reused += 1
                else:
                    y, p, groups = y_all[mask], pvec[mask], groups_all[mask]
                    new_rows, new_frag, new_notes = _evaluate_one(
                        y, p, groups, spec, unc, ci, ctx, clustered
                    )
                    cache[key] = (new_rows, new_frag, new_notes)
                rows.extend(new_rows)
                frag_rows.extend(new_frag)
                notes |= new_notes

    for code, message in sorted(notes):
        flags.append(Flag(code=code, severity="note", message=message))

    metrics_df = pd.DataFrame(rows)
    manifest = {
        "cohort_id": spec.cohort_id,
        "contract_version": spec.version,
        "inputs": {**cohort.hashes, **predictions.hashes},
        "uncertainty": {
            "n_boot": unc.n_boot,
            "seed": unc.seed,
            "ci_level": ci,
            "show_naive_ci": unc.show_naive_ci,
            "clustered": clustered,
            "clustering_field": spec.clustering.field if spec.clustering else None,
        },
        "coverage": {
            "min_fraction": spec.coverage.min_fraction,
            "compare_on": spec.coverage.compare_on,
            "common_warn_frac": spec.coverage.common_warn_frac,
        },
        "models_present": list(predictions.model_ids),
        "expected_models": list(spec.expected_models) if spec.expected_models else None,
        "n_subjects": cohort.n_subjects,
        "dropped_subjects": list(cohort.dropped_subjects),
        # Recorded rather than left implicit: a reader can confirm that identical subsets were
        # reused rather than independently recomputed.
        "cells_reused_identical": n_reused,
        "flags": [f.as_dict() for f in flags],
    }

    return Evaluation(
        metrics=metrics_df,
        fragility=pd.DataFrame(frag_rows),
        coverage=coverage_report(cohort, predictions),
        manifest=manifest,
        flags=flags,
    )


def _evaluate_one(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    spec,
    unc,
    ci: float,
    ctx: dict,
    clustered: bool,
) -> tuple[list[dict], list[dict], set[tuple[str, str]]]:
    """All metrics for one (model, subset, stratum) cell."""
    rows: list[dict] = []
    frag: list[dict] = []
    notes: set[tuple[str, str]] = set()

    n = int(y.size)
    n_events = int(y.sum())
    clusters = U.cluster_indices(groups) if n else []
    n_clusters = len(clusters)
    sizes = dict(n=n, n_events=n_events, n_clusters=n_clusters)

    def row(
        metric, value, interval=None, threshold=None, *, rung=RUNG0, fit_mode=APPARENT, **extra
    ):
        return {
            **ctx,
            "metric": metric,
            "threshold": threshold,
            "rung": rung,
            "fit_mode": fit_mode,
            "value": float(value) if value is not None else np.nan,
            "ci_low": interval.low if interval else np.nan,
            "ci_high": interval.high if interval else np.nan,
            "ci_level": ci if interval else np.nan,
            "ci_method": interval.method if interval else None,
            **sizes,
            **extra,
        }

    # ---- data quality: reported even when the cell is too small for metrics ----------------
    rows.append(row("boundary_count", M.boundary_count(p)))
    rows.append(row("prevalence", M.prevalence(y)))
    if n == 0:
        return rows, frag, notes

    if note := U.few_clusters_note(n_clusters):
        if clustered:
            notes.add(("few_clusters", note))

    # ---- point estimates -------------------------------------------------------------------
    point: dict[tuple[str, float | None], float] = {}
    for name, fn in M.THRESHOLD_FREE.items():
        point[(name, None)] = fn(y, p)
    for name, fn in M.THRESHOLDED.items():
        for t in spec.thresholds:
            point[(name, t)] = fn(y, p, t)

    # ---- cluster percentile bootstrap: the default for every metric ------------------------
    rng = np.random.default_rng(unc.seed)
    samples = {k: np.empty(unc.n_boot) for k in point}
    if n_clusters >= 2:
        for b in range(unc.n_boot):
            idx = U.bootstrap_cluster_indices(clusters, rng)
            yb, pb = y[idx], p[idx]
            for name, fn in M.THRESHOLD_FREE.items():
                samples[(name, None)][b] = fn(yb, pb)
            for name, fn in M.THRESHOLDED.items():
                for t in spec.thresholds:
                    samples[(name, t)][b] = fn(yb, pb, t)
    else:
        for key in samples:
            samples[key][:] = np.nan

    method = "cluster_bootstrap" if clustered else "row_bootstrap"
    for (name, t), value in point.items():
        interval = U.percentile_interval(samples[(name, t)], ci, method)
        rows.append(row(name, value, interval, threshold=t))

    # ---- analytic cluster-robust cross-checks ----------------------------------------------
    per_row_brier = (p - y) ** 2
    brier_ci = U.clustered_mean_interval(per_row_brier, groups, ci)
    rows.append(row("brier", point[("brier", None)], brier_ci))

    for which, metric_name in (
        ("intercept", "calibration_intercept"),
        ("slope", "calibration_slope"),
    ):
        value, interval = U.calibration_interval(y, p, groups, ci, which=which)
        rows.append(row(metric_name, value, interval))

    # ---- naive per-row interval for AUROC: the unit-of-analysis exhibit ---------------------
    if unc.show_naive_ci and clustered and n_clusters >= 2:
        rng_naive = np.random.default_rng(unc.seed + 1)
        naive = np.empty(unc.n_boot)
        for b in range(unc.n_boot):
            idx = rng_naive.integers(0, n, size=n)
            naive[b] = M.auroc(y[idx], p[idx])
        rows.append(
            row(
                "auroc",
                point[("auroc", None)],
                U.percentile_interval(naive, ci, "naive_row_bootstrap"),
            )
        )

    # ---- recalibration ladder: rung1-3, apparent and cross-fitted --------------------------
    # rung0 is the as-published block above (fit_mode apparent). The ladder recomputes the
    # calibration-sensitive metrics on each corrected mapping; optimism is apparent - crossfit,
    # derived at report time from these paired rows. No interval is attached to a recalibrated
    # rung yet -- bootstrapping a refit correction is deferred (docs/spec.md section 4.5).
    ladder_rows, ladder_notes = R.ladder(y, p, groups, max_folds=5)
    for lr in ladder_rows:
        rows.append(row(lr.metric, lr.value, rung=lr.rung, fit_mode=lr.fit_mode))
    notes |= {("recalibration_unavailable", m) for m in ladder_notes}

    # ---- leave-one-cluster-out fragility ----------------------------------------------------
    if n_clusters >= 2:
        for name, fn in M.THRESHOLD_FREE.items():
            delta, culprit = U.loso_fragility(
                lambda keep, fn=fn: fn(y[keep], p[keep]), groups, point[(name, None)]
            )
            frag.append(
                {
                    **ctx,
                    "metric": name,
                    "threshold": None,
                    "max_abs_delta": delta,
                    "culprit_cluster": culprit,
                    **sizes,
                }
            )

    return rows, frag, notes
