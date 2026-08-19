"""Orchestration: both subsets, every stratum, roster flags, and the manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from predval import (
    CohortSpecError,
    coverage_delta,
    evaluate,
    load_cohort,
    load_predictions,
)

N_CLUSTERS = 8
PER_CLUSTER = 60
MODELS = ("good", "weak")


def build(tmp_path: Path, *, spec_extra: dict | None = None, drop_for_weak: int = 0) -> tuple:
    """A small clustered cohort with two models and a two-level subgroup."""
    rng = np.random.default_rng(5)
    rows, preds = [], []
    for c in range(N_CLUSTERS):
        shift = rng.normal(0, 1.0)
        for i in range(PER_CLUSTER):
            z = rng.normal(shift, 1.0)
            p = 1.0 / (1.0 + np.exp(-z))
            sid = f"c{c:02d}_{i:03d}"
            rows.append(
                {
                    "subject_id": sid,
                    "label": int(rng.binomial(1, p)),
                    "wsi": f"c{c:02d}",
                    "arm": "x" if c % 2 == 0 else "y",
                }
            )
            preds.append({"subject_id": sid, "model_id": "good", "predicted": float(p)})
            preds.append(
                {"subject_id": sid, "model_id": "weak", "predicted": float(0.5 + 0.05 * (p - 0.5))}
            )

    cohort_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(preds)
    if drop_for_weak:
        victims = set(cohort_df["subject_id"].head(drop_for_weak))
        pred_df = pred_df[~((pred_df["model_id"] == "weak") & pred_df["subject_id"].isin(victims))]

    cohort_df.to_parquet(tmp_path / "cohort.parquet", index=False)
    pred_df.to_parquet(tmp_path / "predictions.parquet", index=False)

    spec = {
        "cohort_id": "toy",
        "version": 0,
        "subject_key": "subject_id",
        "data": "cohort.parquet",
        "outcome": {"type": "binary", "field": "label", "positive_label": 1},
        "clustering": {"field": "wsi"},
        "coverage": {"min_fraction": 0.5, "compare_on": "both"},
        "completeness": {"require_outcome": True, "on_violation": "drop_and_report"},
        "subgroups": [{"name": "arm_group", "field": "arm"}],
        "thresholds": [0.5],
        "uncertainty": {"n_boot": 60, "seed": 1337},
        **(spec_extra or {}),
    }
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(spec))
    return load_cohort(tmp_path / "cohort.yaml"), load_predictions(tmp_path / "predictions.parquet")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    cohort, preds = build(tmp)
    return evaluate(cohort, preds)


# ------------------------------------------------------------------------------ shape and cover


def test_both_subsets_always_present(result) -> None:
    """metrics.parquet is complete regardless of compare_on; the report is what is opinionated."""
    assert set(result.metrics["subset"]) == {"full", "common"}


def test_every_stratum_evaluated(result) -> None:
    sub = result.metrics[result.metrics["stratum_kind"] == "subgroup"]
    assert set(sub["subgroup_name"]) == {"arm_group"}
    assert set(sub["subgroup_level"]) == {"x", "y"}
    assert (result.metrics["stratum_kind"] == "overall").any()


def test_all_models_present(result) -> None:
    assert set(result.metrics["model_id"]) == set(MODELS)


def test_fit_modes_are_apparent_and_crossfit(result) -> None:
    """S3 adds the cross-fitted rungs; as-published rung0 stays apparent-only."""
    assert set(result.metrics["fit_mode"]) == {"apparent", "crossfit"}
    rung0 = result.metrics[result.metrics["rung"] == "rung0"]
    assert set(rung0["fit_mode"]) == {"apparent"}
    assert set(result.metrics["rung"]) == {"rung0", "rung1", "rung2", "rung3"}


def test_expected_metrics_reported(result) -> None:
    names = set(result.metrics["metric"])
    for expected in (
        "auroc",
        "average_precision",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "boundary_count",
        "prevalence",
    ):
        assert expected in names, f"{expected} missing"


# ------------------------------------------------------------------------------- interval layers


def test_bootstrap_and_analytic_both_present_for_brier(result) -> None:
    """The report shows the bootstrap and footnotes whether the analytic agrees."""
    brier = result.metrics[
        (result.metrics["metric"] == "brier")
        & (result.metrics["rung"] == "rung0")
        & (result.metrics["subset"] == "full")
        & (result.metrics["stratum_kind"] == "overall")
        & (result.metrics["model_id"] == "good")
    ]
    assert set(brier["ci_method"]) == {"cluster_bootstrap", "cluster_robust_t"}


def test_the_two_brier_methods_broadly_agree(result) -> None:
    brier = result.metrics[
        (result.metrics["metric"] == "brier")
        & (result.metrics["rung"] == "rung0")
        & (result.metrics["subset"] == "full")
        & (result.metrics["stratum_kind"] == "overall")
    ]
    for _, grp in brier.groupby("model_id"):
        boot = grp[grp["ci_method"] == "cluster_bootstrap"].iloc[0]
        anal = grp[grp["ci_method"] == "cluster_robust_t"].iloc[0]
        assert boot["value"] == pytest.approx(anal["value"])
        # overlapping intervals; they are different estimators, not the same one twice
        assert boot["ci_low"] < anal["ci_high"] and anal["ci_low"] < boot["ci_high"]


def test_naive_auroc_exhibit_is_present_and_narrower(result) -> None:
    """The unit-of-analysis exhibit: the naive interval must visibly understate uncertainty."""
    auroc = result.metrics[
        (result.metrics["metric"] == "auroc")
        & (result.metrics["subset"] == "full")
        & (result.metrics["stratum_kind"] == "overall")
        & (result.metrics["model_id"] == "good")
    ]
    methods = dict(zip(auroc["ci_method"], auroc["ci_high"] - auroc["ci_low"], strict=False))
    assert "naive_row_bootstrap" in methods and "cluster_bootstrap" in methods
    assert methods["naive_row_bootstrap"] < methods["cluster_bootstrap"]


def test_fragility_names_a_culprit_cluster(result) -> None:
    frag = result.fragility
    assert not frag.empty
    assert set(frag["metric"]) >= {"auroc", "brier"}
    assert frag["culprit_cluster"].notna().any()
    # fragility is a magnitude, never an interval
    assert "ci_low" not in frag.columns


def test_few_clusters_note_fires(result) -> None:
    codes = {f.code for f in result.flags}
    assert "few_clusters" in codes, "G=8 is well under 40 and must be flagged"


# ---------------------------------------------------------------------------------- the roster


def test_roster_not_declared_is_itself_a_flag(result) -> None:
    codes = {f.code for f in result.flags}
    assert "roster_not_declared" in codes


def test_declared_but_absent_flags(tmp_path: Path) -> None:
    cohort, preds = build(tmp_path, spec_extra={"expected_models": ["good", "weak", "ghost"]})
    res = evaluate(cohort, preds)
    flag = next(f for f in res.flags if f.code == "declared_but_absent")
    assert "ghost" in flag.message


def test_present_but_undeclared_flags_a_typo(tmp_path: Path) -> None:
    """The typo direction: 'waek' is declared, so real 'weak' reads as unexpected."""
    cohort, preds = build(tmp_path, spec_extra={"expected_models": ["good", "waek"]})
    res = evaluate(cohort, preds)
    codes = {f.code for f in res.flags}
    assert codes >= {"declared_but_absent", "present_but_undeclared"}
    undeclared = next(f for f in res.flags if f.code == "present_but_undeclared")
    assert "weak" in undeclared.message


def test_on_missing_model_fail_refuses_to_run(tmp_path: Path) -> None:
    cohort, preds = build(
        tmp_path,
        spec_extra={"expected_models": ["good", "weak", "ghost"], "on_missing_model": "fail"},
    )
    with pytest.raises(CohortSpecError, match="ghost"):
        evaluate(cohort, preds)


# ------------------------------------------------------------------- informative common subset


def test_large_intersection_loss_is_flagged(tmp_path: Path) -> None:
    cohort, preds = build(tmp_path, drop_for_weak=int(N_CLUSTERS * PER_CLUSTER * 0.3))
    res = evaluate(cohort, preds)
    flag = next(f for f in res.flags if f.code == "informative_common_selection")
    assert "may be informative" in flag.message


def test_small_intersection_loss_is_not_flagged(result) -> None:
    assert not any(f.code == "informative_common_selection" for f in result.flags)


def test_coverage_delta_is_zero_when_coverage_is_identical(result) -> None:
    delta = coverage_delta(result.metrics)
    finite = delta["cov_delta"].dropna()
    assert len(finite) > 0
    assert np.allclose(finite, 0.0), "full == common here, so every delta must be zero"


# -------------------------------------------------------------------------------- the manifest


def test_subgroup_ladder_is_gated_below_threshold(result) -> None:
    """The toy's subgroups have 4 clusters each -- below the gate -- so they get rung0 only."""
    sub = result.metrics[result.metrics["stratum_kind"] == "subgroup"]
    assert set(sub["rung"]) == {"rung0"}
    overall = result.metrics[result.metrics["stratum_kind"] == "overall"]
    assert {"rung1", "rung2", "rung3"} <= set(overall["rung"]), "overall is never gated"
    assert "recalibration_suppressed" in {f.code for f in result.flags}


def test_manifest_roster_counts_present_and_absent(result) -> None:
    r = result.manifest["roster"]
    assert r["present"] == len(MODELS)
    assert r["declared"] is None and r["absent"] == []  # toy declares no roster


def test_manifest_records_provenance_and_config(result) -> None:
    m = result.manifest
    assert set(m["inputs"]) == {"cohort_spec", "cohort_data", "predictions"}
    assert all(len(v) == 64 for v in m["inputs"].values())
    assert m["uncertainty"]["seed"] == 1337
    assert m["uncertainty"]["clustered"] is True
    assert m["uncertainty"]["clustering_field"] == "wsi"
    assert m["flags"]


def test_write_emits_all_artefacts(result, tmp_path: Path) -> None:
    paths = result.write(tmp_path / "out")
    assert set(paths) == {"metrics", "fragility", "coverage", "manifest"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    assert len(pd.read_parquet(paths["metrics"])) == len(result.metrics)


# ------------------------------------------------------------------------------ absence handling


def test_unscored_subjects_are_excluded_not_zero_filled(tmp_path: Path) -> None:
    """The absence rule must survive into the metric layer."""
    dropped = 120
    cohort, preds = build(tmp_path, drop_for_weak=dropped)
    res = evaluate(cohort, preds)
    weak_full = res.metrics[
        (res.metrics["model_id"] == "weak")
        & (res.metrics["subset"] == "full")
        & (res.metrics["stratum_kind"] == "overall")
        & (res.metrics["metric"] == "auroc")
    ].iloc[0]
    assert weak_full["n"] == N_CLUSTERS * PER_CLUSTER - dropped


def test_no_clustering_declared_is_flagged(tmp_path: Path) -> None:
    cohort, preds = build(tmp_path)
    spec = yaml.safe_load((tmp_path / "cohort.yaml").read_text())
    del spec["clustering"]
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(spec))
    res = evaluate(load_cohort(tmp_path / "cohort.yaml"), preds)
    assert any(f.code == "no_clustering_declared" for f in res.flags)
    assert res.manifest["uncertainty"]["clustered"] is False
