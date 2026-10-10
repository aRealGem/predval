"""Evaluation orchestration: cohort + predictions in, metrics/fragility/manifest out.

Structure follows docs/spec.md section 6. Both subsets are always computed, every stratum is
evaluated, and every metric carries an interval whose method is recorded rather than assumed.

As-published metrics are `rung0`/`apparent`. The recalibration ladder (§4, recalibrate.py) adds
`rung1`–`rung3` rows in both `apparent` and `crossfit` fit modes for every metric, subject to the
half-pair guard and the subgroup gate; their optimism is derived per metric from the paired rows.
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
    #: Per-member rung0 decile calibration points with a cluster-bootstrap band, for the section-2
    #: calibration small-multiples (item 2a). Common subset, overall stratum. A report figure reads
    #: it; nothing statistical is recomputed at render time.
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    flags: list[Flag] = field(default_factory=list)

    def write(self, outdir: str | Path) -> dict[str, Path]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": out / "metrics.parquet",
            "fragility": out / "fragility.parquet",
            "coverage": out / "coverage.parquet",
            "calibration": out / "calibration.parquet",
            "manifest": out / "manifest.json",
        }
        self.metrics.to_parquet(paths["metrics"], index=False)
        self.fragility.to_parquet(paths["fragility"], index=False)
        self.coverage.to_parquet(paths["coverage"], index=False)
        self.calibration.to_parquet(paths["calibration"], index=False)
        paths["manifest"].write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        return paths


def _tool_versions() -> dict[str, str]:
    """predval and its scientific stack, for the report's provenance section.

    Recorded so a report's numbers can be reproduced against the exact library versions that
    produced them -- a calibration slope can move between statsmodels releases.
    """
    import platform

    import scipy
    import sklearn
    import statsmodels

    from . import __version__

    return {
        "predval": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "scikit-learn": sklearn.__version__,
    }


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
    # `dropna=False` keeps a cell whose `common` value is missing -- that absence is the thing
    # this column exists to show. It also fills in the full cross product of the index levels,
    # which invents cells that never existed (stratum_kind="overall" crossed with a
    # subgroup_level, say). Those have neither subset present, so drop exactly those: the result
    # then carries one row per cell the evaluation actually computed.
    wide = wide[wide["full"].notna() | wide["common"].notna()].reset_index(drop=True)
    wide["cov_delta"] = wide["full"] - wide["common"]
    return wide[[*keys, "cov_delta"]]


#: Decile bins for the section-2 calibration small-multiples (item 2a).
N_CALIBRATION_BINS = 10


def _common_overall_cells(
    preds: dict[str, np.ndarray],
    y_all: np.ndarray,
    groups_all: np.ndarray,
    subject_ids: np.ndarray,
    common_ids: set[str],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(model, y, p, groups) on the common subset, overall stratum -- the verdict/figure cells.

    Every model scored every common subject by construction (the common set is the intersection),
    so the mask is the common set itself for each model.
    """
    in_common = np.fromiter((s in common_ids for s in subject_ids), bool, len(subject_ids))
    cells = []
    for model, pvec in preds.items():
        mask = in_common & np.isfinite(pvec)
        cells.append((model, y_all[mask], pvec[mask], groups_all[mask]))
    return cells


#: Ladder order for "lowest rung whose gain interval excludes zero" (S6 item 2; spec section 4.8).
_LADDER_RUNGS_IN_ORDER = ("rung1", "rung2", "rung3")


def _paired_gains_by_model(rows: list[dict]) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Index the common-subset/overall-stratum paired_gain_brier rows already computed in the main
    per-cell loop, keyed by model then rung -> (value, ci_low, ci_high). Reused rather than
    recomputed so the verdict's significance test draws on the exact same bootstrap resample as
    the number section 3 of the report shows -- never a second, independently-seeded one.
    """
    gains: dict[str, dict[str, tuple[float, float, float]]] = {}
    for r in rows:
        if (
            r["metric"] == "paired_gain_brier"
            and r["subset"] == "common"
            and r["stratum_kind"] == "overall"
            and r["fit_mode"] == R.CROSSFIT
        ):
            gains.setdefault(r["model_id"], {})[r["rung"]] = (
                r["value"],
                r["ci_low"],
                r["ci_high"],
            )
    return gains


#: Axis B statuses (S6.1 item 1; spec section 4.8).
REPAIR_DEMONSTRATED = "demonstrated"
REPAIR_UNPROVEN = "unproven"
REPAIR_COUNTERPRODUCTIVE = "counterproductive"


def _classify_repair(
    model_gains: dict[str, tuple[float, float, float]],
) -> tuple[str, str | None, tuple[str, ...]]:
    """Axis B: what happened when recalibration was tried, as three mutually exclusive outcomes.

    Returns (status, admissible_rung, counterproductive_rungs):
    - "demonstrated": the LOWEST rung whose paired cross-fit gain interval excludes zero on the
      improvement side (`ci_low > 0`) -- a real, interval-backed repair. `admissible_rung` names it.
    - "counterproductive": no rung is demonstrated, but at least one rung's interval lies entirely
      BELOW zero (`ci_high < 0`) -- recalibration was tried and reliably made Brier worse.

    `counterproductive_rungs` names every rung whose interval lies entirely below zero and is
    computed INDEPENDENTLY of the status (S6.2 D5). A demonstrated repair at rung2 does not make a
    reliably-harmful rung1 stop being harmful: before S6.2 the demonstrated branch returned an
    empty tuple and that harm went unreported, which understated the count of members carrying a
    harmful rung (4 rather than 5 on the PCam fixture). Which rung is *admissible* is still decided
    by the demonstrated test alone -- this changes what is reported, never what is selected.
    - "unproven": neither of the above -- every rung's interval straddles zero. This is NOT the
      same claim as "counterproductive": one says recalibration measurably helped or hurt, the
      other says the cohort's power was too low to tell either way.
    A rung whose interval sits entirely below zero is never "admissible" under any reading -- an
    admissible repair must be a real improvement, not merely distinguishable from no-op.
    """
    counterproductive = tuple(
        rung
        for rung in _LADDER_RUNGS_IN_ORDER
        if (cell := model_gains.get(rung)) is not None and np.isfinite(cell[2]) and cell[2] < 0
    )

    for rung in _LADDER_RUNGS_IN_ORDER:
        cell = model_gains.get(rung)
        if cell is None:
            continue
        _, ci_low, _ = cell
        if np.isfinite(ci_low) and ci_low > 0:
            return REPAIR_DEMONSTRATED, rung, counterproductive

    if counterproductive:
        return REPAIR_COUNTERPRODUCTIVE, None, counterproductive
    return REPAIR_UNPROVEN, None, ()


def _calibration_by_model(rows: list[dict]) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Index the already-computed rung0/common/overall calibration_slope/intercept rows
    (ci_method=cluster_robust_t), keyed by model then metric name -> (value, ci_low, ci_high).
    Reused the same way `_paired_gains_by_model` reuses gain rows -- axis A draws on numbers
    already computed in the main per-cell loop, never a second fit.
    """
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for r in rows:
        if (
            r["metric"] in ("calibration_slope", "calibration_intercept")
            and r["subset"] == "common"
            and r["stratum_kind"] == "overall"
            and r["rung"] == RUNG0
            and r["ci_method"] == "cluster_robust_t"
        ):
            out.setdefault(r["model_id"], {})[r["metric"]] = (r["value"], r["ci_low"], r["ci_high"])
    return out


def _classify_miscalibration(model_calibration: dict[str, tuple[float, float, float]]) -> bool:
    """Axis A: does rung0's own analytic cluster-robust interval show miscalibration -- slope CI
    excludes 1, OR intercept CI excludes 0. Either alone is sufficient; a model can be
    level-shifted without a spread problem or vice versa.
    """
    slope = model_calibration.get("calibration_slope")
    if slope is not None:
        _, lo, hi = slope
        if np.isfinite(lo) and np.isfinite(hi) and not (lo <= 1.0 <= hi):
            return True
    intercept = model_calibration.get("calibration_intercept")
    if intercept is not None:
        _, lo, hi = intercept
        if np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0.0 <= hi):
            return True
    return False


def _verdict_layer(cells, rows: list[dict], spec, unc, ci: float) -> dict[str, dict]:
    """Per member: a two-axis verdict (S6.1 item 1; spec section 4.8), plus the Brier skill score
    at whichever rung backs it.

    Axis A (miscalibration detected) and axis B (repair outcome: demonstrated / unproven /
    counterproductive) are independent facts -- a member can be miscalibrated with an unproven
    repair, well-calibrated with a demonstrated (if small) gain, or any other combination. BSS
    anchors on the admissible rung when axis B is "demonstrated", else on rung0's own held-out
    performance (unchanged from S6's fallback).
    """
    gains_by_model = _paired_gains_by_model(rows)
    calibration_by_model = _calibration_by_model(rows)
    verdict: dict[str, dict] = {}
    for model, y, p, groups in cells:
        if y.size == 0 or np.unique(y).size < 2:
            continue
        ladder_rows, lrep = R.ladder(y, p, groups, thresholds=tuple(spec.thresholds), max_folds=5)
        cf_brier = {
            lr.rung: lr.value
            for lr in ladder_rows
            if lr.fit_mode == R.CROSSFIT and lr.metric == "brier" and lr.threshold is None
        }
        rung0_brier = float(M.brier(y, p))

        axis_b, admissible_rung, counterproductive_rungs = _classify_repair(
            gains_by_model.get(model, {})
        )
        if (
            admissible_rung is not None
            and admissible_rung in cf_brier
            and np.isfinite(cf_brier[admissible_rung])
        ):
            best_rung = admissible_rung
            best_rung_brier = float(cf_brier[admissible_rung])
        else:
            # Not demonstrated, or the cross-fit Brier for the admissible rung is unavailable
            # (should not happen if its gain interval was computed) -- fall back to rung0 either
            # way, never crash on a missing cell.
            best_rung = RUNG0
            best_rung_brier = rung0_brier

        axis_a = _classify_miscalibration(calibration_by_model.get(model, {}))

        p_best = p if best_rung == RUNG0 else lrep.crossfit_predictions[best_rung]
        bss, bss_ci = U.brier_skill_interval(
            y, p_best, groups, ci, n_boot=unc.n_boot, seed=unc.seed
        )
        verdict[model] = {
            "bss": bss,
            "bss_ci_low": bss_ci.low,
            "bss_ci_high": bss_ci.high,
            "best_rung": best_rung,
            "best_rung_brier": best_rung_brier,
            "rung0_brier": rung0_brier,
            "axis_a_miscalibrated": axis_a,
            "axis_b": axis_b,
            "axis_b_counterproductive_rungs": list(counterproductive_rungs),
            "prevalence": float(np.mean(y)),
            "n": int(y.size),
            "n_clusters": int(np.unique(groups).size),
        }
    return verdict


def _calibration_curves(cells, unc, ci: float) -> pd.DataFrame:
    """Per-member decile calibration points with a cluster-bootstrap band on the observed rate.

    Bins are the deciles of the member's published probabilities on the common subset. Each bin
    carries its mean predicted probability, its observed event rate, and a percentile band over
    slide resamples of that observed rate -- the band the section-2 figure shades (item 2a).
    """
    cols = ["model_id", "bin", "mean_pred", "obs_rate", "ci_low", "ci_high", "n", "n_events"]
    rows: list[dict] = []
    alpha = (1.0 - ci) / 2.0
    for model, y, p, groups in cells:
        if y.size == 0:
            continue
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)))
        if edges.size < 2:
            continue
        nbins = edges.size - 1
        bin_idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, nbins - 1)
        clusters = U.cluster_indices(groups)
        boot = np.full((unc.n_boot, nbins), np.nan)
        if len(clusters) >= 2:
            rng = np.random.default_rng(unc.seed)
            for b in range(unc.n_boot):
                ridx = U.bootstrap_cluster_indices(clusters, rng)
                bi, yy = bin_idx[ridx], y[ridx]
                cnt = np.bincount(bi, minlength=nbins).astype(float)
                pos = np.bincount(bi, weights=yy.astype(float), minlength=nbins)
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot[b] = np.where(cnt > 0, pos / cnt, np.nan)
        for bidx in range(nbins):
            sel = bin_idx == bidx
            n = int(sel.sum())
            if n == 0:
                continue
            col = boot[:, bidx]
            finite = col[np.isfinite(col)]
            lo, hi = (
                (float(x) for x in np.quantile(finite, [alpha, 1.0 - alpha]))
                if finite.size >= 2
                else (float("nan"), float("nan"))
            )
            rows.append(
                {
                    "model_id": model,
                    "bin": bidx,
                    "mean_pred": float(p[sel].mean()),
                    "obs_rate": float(y[sel].mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "n_events": int(y[sel].sum()),
                }
            )
    return pd.DataFrame(rows, columns=cols)


def _material_nonmonotone(max_local_decrease: float, delta_auroc: float | None, tol: float) -> bool:
    """Is a rung3 non-monotonicity material enough to flag (S4.1 item 2; gate revised, D1)?

    The flag is driven by the **outcome-level** signal ALONE: the non-monotonicity is material iff
    it moved AUROC past `tol`, either direction (a reordering that raises or lowers discrimination
    is equally a reordering). `max_local_decrease` -- a dip on the probability-scale transform that
    need not reorder anyone -- is kept in `metrics.parquet` and shown as a secondary note in the
    flag message, but it no longer drives the flag. This quiets epsilon-wiggle members (phikon,
    dAUROC ~2e-5) while still catching genuine reordering (effnet_scratch, dAUROC 0.004).
    """
    del max_local_decrease  # retained in the artefact + message; deliberately not a flag driver
    return bool(delta_auroc is not None and np.isfinite(delta_auroc) and abs(delta_auroc) > tol)


def _stratum_label(ctx: dict) -> str:
    """The stratum half of a cell label: 'overall' or 'subgroup_name=level'. Subset is omitted
    deliberately: cells with identical coverage are reused across subsets, so the stratum is
    stable but the subset is not.
    """
    return (
        "overall"
        if ctx["stratum_kind"] == "overall"
        else f"{ctx['subgroup_name']}={ctx['subgroup_level']}"
    )


def _cell_label(ctx: dict) -> str:
    """A model+stratum label for ladder-quality flags."""
    return f"{ctx['model_id']} ({_stratum_label(ctx)})"


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

    # Verdict layer + calibration curves, both on the common subset / overall stratum (items 2a, 3).
    cells = _common_overall_cells(preds, y_all, groups_all, subject_ids, subsets["common"])
    verdict = _verdict_layer(cells, rows, spec, unc, ci)
    calibration_df = _calibration_curves(cells, unc, ci)

    metrics_df = pd.DataFrame(rows)
    manifest = {
        "cohort_id": spec.cohort_id,
        "contract_version": spec.version,
        "versions": _tool_versions(),
        "inputs": {**cohort.hashes, **predictions.hashes},
        "uncertainty": {
            "n_boot": unc.n_boot,
            "seed": unc.seed,
            "ci_level": ci,
            "show_naive_ci": unc.show_naive_ci,
            "clustered": clustered,
            "clustering_field": spec.clustering.field if spec.clustering else None,
            # Human-readable noun for the clustering unit ("slide", "region", ...), threaded into
            # the report's prose (S6 item 5); "cluster" when no clustering is declared at all, so
            # report.py never needs a null-check of its own.
            "clustering_name": spec.clustering.name if spec.clustering else "cluster",
            # Unit of observation ("patch", "row", ...), the companion to clustering_name's unit
            # of independence (S6.2 D1). Carried here so report.py reads both nouns from one place.
            "unit_noun": spec.unit_noun,
        },
        "coverage": {
            "min_fraction": spec.coverage.min_fraction,
            "compare_on": spec.coverage.compare_on,
            "common_warn_frac": spec.coverage.common_warn_frac,
        },
        "models_present": list(predictions.model_ids),
        "expected_models": list(spec.expected_models) if spec.expected_models else None,
        # Declared-vs-present at a glance, so a report header states both without recounting.
        # For the PCam fixture: 16 declared, 15 present, 1 absent (p4m_reg, predictions lost).
        "roster": {
            "declared": len(spec.expected_models) if spec.expected_models else None,
            "present": len(predictions.model_ids),
            "absent": sorted(set(spec.expected_models) - set(predictions.model_ids))
            if spec.expected_models
            else [],
            # Author-declared model_ids known to be an ensemble/blend (S6.1 item 2); governs the
            # report's ensemble-construction-bias caution. Empty means none are known to be.
            "ensemble_members": list(spec.ensemble_members),
        },
        "n_subjects": cohort.n_subjects,
        "dropped_subjects": list(cohort.dropped_subjects),
        # Recorded rather than left implicit: a reader can confirm that identical subsets were
        # reused rather than independently recomputed.
        "cells_reused_identical": n_reused,
        # Per-member verdict: Brier skill score at the best admissible rung (item 3). Stored in the
        # manifest rather than metrics.parquet so the rung/fit_mode invariants of §6.1 stay clean.
        "verdict": verdict,
        "flags": [f.as_dict() for f in flags],
    }

    return Evaluation(
        metrics=metrics_df,
        fragility=pd.DataFrame(frag_rows),
        coverage=coverage_report(cohort, predictions),
        manifest=manifest,
        calibration=calibration_df,
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

    if note := U.few_clusters_note(n_clusters, _stratum_label(ctx)):
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
    # rung0 is the as-published block above. The ladder recomputes every metric on each corrected
    # mapping (optimism = apparent - crossfit, oriented per metric, derived at report time). No
    # interval is attached to a recalibrated rung yet (docs/spec.md section 4.5). Subgroups below
    # the gate get rung0 only, and a rung with no apparent/crossfit pair is withheld entirely.
    rec = spec.recalibration
    label = _cell_label(ctx)
    if ctx["stratum_kind"] == "subgroup" and (
        n_clusters < rec.min_clusters or min(n_events, n - n_events) < rec.min_events_per_class
    ):
        notes.add(
            (
                "recalibration_suppressed",
                f"ladder suppressed for {label}: {n_clusters} clusters, "
                f"{min(n_events, n - n_events)} events in the smaller class "
                f"(gate: {rec.min_clusters} clusters, {rec.min_events_per_class} events/class)",
            )
        )
    else:
        # Overall is never suppressed; but a below-gate cluster count still weakens the cross-fit,
        # so it runs with a caution rather than silently (S4.1 item 5).
        if ctx["stratum_kind"] == "overall" and n_clusters < rec.min_clusters:
            notes.add(
                (
                    "recalibration_overall_low_power",
                    f"overall ladder run with only {n_clusters} clusters "
                    f"(below gate {rec.min_clusters}); its cross-fit is low-power for {label}",
                )
            )

        ladder_rows, lrep = R.ladder(y, p, groups, thresholds=tuple(spec.thresholds), max_folds=5)
        for lr in ladder_rows:
            rows.append(
                row(lr.metric, lr.value, rung=lr.rung, fit_mode=lr.fit_mode, threshold=lr.threshold)
            )

        # Paired cross-fit Brier gain per converged rung: rung0 (as-published) minus the rung's
        # out-of-fold Brier, with a paired slide bootstrap (item 1). This is the interval the pitch
        # shows in section 3, replacing the deferred "(interval pending §4.5)" marker. Emitted for
        # every rung whose cross-fit mapping is available; ci_method is paired_cluster_bootstrap.
        for lr_rung, cf_pred in lrep.crossfit_predictions.items():
            gain, gain_ci = U.paired_brier_gain_interval(
                y, p, cf_pred, groups, ci, n_boot=unc.n_boot, seed=unc.seed
            )
            rows.append(row("paired_gain_brier", gain, gain_ci, rung=lr_rung, fit_mode=R.CROSSFIT))

        # rung3 non-monotonicity diagnostics: ALWAYS recorded in the artefact (S4.1 item 2), so a
        # sub-tolerance wiggle is visible without being flagged as a finding.
        if lrep.rung3_max_local_decrease is not None:
            mld_val = lrep.rung3_max_local_decrease
            rows.append(row("rung3_max_local_decrease", mld_val, rung="rung3"))
            rows.append(row("delta_auroc_rung3", lrep.delta_auroc_rung3, rung="rung3"))
            # Flag ONLY when the non-monotonicity is material at the outcome level: |dAUROC| past
            # monotone_tol (D1). |dAUROC| is direction-agnostic on purpose -- a reordering that
            # raises or lowers discrimination is equally a reordering. The transform's largest local
            # decrease rides along as a secondary note but never drives the flag.
            mld = lrep.rung3_max_local_decrease
            dauroc = lrep.delta_auroc_rung3
            if _material_nonmonotone(mld, dauroc, rec.monotone_tol):
                notes.add(
                    (
                        "recalibration_non_monotone",
                        f"rung3 materially non-monotone for {label}: delta_auroc={dauroc:+.4f} "
                        f"exceeds tol {rec.monotone_tol} (secondary note: largest local decrease "
                        f"on the probability-scale transform = {mld:.4f}, not a flag driver)",
                    )
                )

        if lrep.rung2_slope is not None and lrep.rung2_slope < 0:
            notes.add(
                (
                    "recalibration_rank_inverting",
                    f"rung2 slope {lrep.rung2_slope:.3f} < 0 (rank-inverting) for {label}",
                )
            )
        for supp_rung, reason in lrep.suppressed:
            notes.add(
                (
                    "recalibration_unavailable",
                    f"{supp_rung} withheld for {label}: {reason}",
                )
            )

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
