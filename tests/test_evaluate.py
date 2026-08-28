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


def build(
    tmp_path: Path,
    *,
    spec_extra: dict | None = None,
    drop_for_weak: int = 0,
    n_clusters: int = N_CLUSTERS,
) -> tuple:
    """A small clustered cohort with two models and a two-level subgroup."""
    rng = np.random.default_rng(5)
    rows, preds = [], []
    for c in range(n_clusters):
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


def test_overall_ladder_runs_with_a_low_cluster_caution(tmp_path: Path) -> None:
    """S4.1 item 5: overall is never suppressed, but G < min_clusters raises a caution flag."""
    cohort, preds = build(tmp_path, n_clusters=3)  # G=3 overall, below the default gate of 5
    res = evaluate(cohort, preds)
    codes = {f.code for f in res.flags}
    assert "recalibration_overall_low_power" in codes
    overall = res.metrics[res.metrics["stratum_kind"] == "overall"]
    assert {"rung1", "rung2"} <= set(overall["rung"]), "overall must still run the ladder"


def test_rung3_nonmonotonicity_diagnostics_are_always_recorded(result) -> None:
    """S4.1 item 2: the two rung3 diagnostics land in the artefact regardless of the flag."""
    overall = result.metrics[result.metrics["stratum_kind"] == "overall"]
    names = set(overall["metric"])
    assert "rung3_max_local_decrease" in names
    assert "delta_auroc_rung3" in names


def test_paired_gain_rows_present_with_intervals(result) -> None:
    """Item 1: paired cross-fit Brier gain rows (overall/common) carry a paired interval."""
    pg = result.metrics[
        (result.metrics["metric"] == "paired_gain_brier")
        & (result.metrics["subset"] == "common")
        & (result.metrics["stratum_kind"] == "overall")
    ]
    assert not pg.empty
    assert set(pg["rung"]) <= {"rung1", "rung2", "rung3"}
    assert set(pg["fit_mode"]) == {"crossfit"}
    assert (pg["ci_method"] == "paired_cluster_bootstrap").all()
    # the point estimate lies within its own interval
    for _, r in pg.iterrows():
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            assert r["ci_low"] <= r["value"] <= r["ci_high"]


def test_verdict_layer_in_manifest(result) -> None:
    """Item 3: per-member BSS + best rung on the common/overall stratum."""
    verdict = result.manifest["verdict"]
    assert set(verdict) == set(MODELS)
    for v in verdict.values():
        assert v["best_rung"] in {"rung0", "rung1", "rung2", "rung3"}
        assert v["bss_ci_low"] <= v["bss"] <= v["bss_ci_high"]
        # S6.1 item 1: the two-axis taxonomy is independent of best_rung.
        assert isinstance(v["axis_a_miscalibrated"], bool)
        assert v["axis_b"] in {"demonstrated", "unproven", "counterproductive"}
        assert isinstance(v["axis_b_counterproductive_rungs"], list)


def test_best_rung_requires_a_significant_gain(result) -> None:
    """S6 item 2: best rung is the lowest whose paired gain interval excludes zero, not the
    lowest cross-fitted Brier point estimate.

    "good" is well-calibrated by construction; the ladder should find nothing admissible and fall
    back to rung0. "weak" is deliberately, substantially miscalibrated; the ladder should find a
    real, interval-backed gain. Locks the directionality choice (ci_low > 0, not merely "excludes
    zero either way") and that the chosen rung is genuinely the lowest admissible one, not skipped
    past.
    """
    verdict = result.manifest["verdict"]
    assert verdict["good"]["best_rung"] == "rung0"

    weak_best = verdict["weak"]["best_rung"]
    assert weak_best in {"rung1", "rung2", "rung3"}

    gains = result.metrics[
        (result.metrics["metric"] == "paired_gain_brier")
        & (result.metrics["subset"] == "common")
        & (result.metrics["stratum_kind"] == "overall")
        & (result.metrics["fit_mode"] == "crossfit")
        & (result.metrics["model_id"] == "weak")
    ].set_index("rung")

    # the chosen rung's own gain interval excludes zero on the improvement side
    assert gains.loc[weak_best, "ci_low"] > 0
    # no lower rung was skipped past
    for rung in ("rung1", "rung2", "rung3"):
        if rung == weak_best:
            break
        if rung in gains.index:
            assert not (gains.loc[rung, "ci_low"] > 0)


# ---------------------------------------------------- S6.1 item 1: two-axis verdict taxonomy


def test_classify_repair_all_three_outcomes() -> None:
    """Pure unit test on the axis-B classifier: demonstrated (lowest admissible rung, not
    skipped past), counterproductive (some rung's interval lies entirely below zero), unproven
    (every rung straddles zero)."""
    from predval.evaluate import _classify_repair

    # rung1 crosses zero, rung2 excludes zero on the positive side -> demonstrated at rung2.
    demonstrated = {
        "rung1": (-0.001, -0.004, 0.002),
        "rung2": (0.002, 0.0002, 0.004),
        "rung3": (0.002, -0.001, 0.005),
    }
    assert _classify_repair(demonstrated) == ("demonstrated", "rung2", ())

    # every rung's interval lies entirely below zero -> counterproductive at all three.
    counterproductive = {
        "rung1": (-0.003, -0.005, -0.002),
        "rung2": (-0.004, -0.006, -0.002),
        "rung3": (-0.003, -0.005, -0.001),
    }
    assert _classify_repair(counterproductive) == (
        "counterproductive",
        None,
        ("rung1", "rung2", "rung3"),
    )

    # every rung crosses zero -> unproven.
    unproven = {
        "rung1": (-0.001, -0.040, 0.045),
        "rung2": (0.031, -0.004, 0.068),
        "rung3": (0.030, -0.007, 0.067),
    }
    assert _classify_repair(unproven) == ("unproven", None, ())

    # empty input (no gain rows at all) -> unproven, never crashes.
    assert _classify_repair({}) == ("unproven", None, ())


def test_classify_miscalibration_either_axis_triggers() -> None:
    """Pure unit test on the axis-A classifier: slope excluding 1 OR intercept excluding 0 is
    sufficient on its own; both crossing their null means well-calibrated."""
    from predval.evaluate import _classify_miscalibration

    slope_only = {
        "calibration_slope": (1.44, 1.24, 1.65),
        "calibration_intercept": (-0.04, -0.59, 0.52),
    }
    assert _classify_miscalibration(slope_only) is True

    intercept_only = {
        "calibration_slope": (1.0, 0.9, 1.1),
        "calibration_intercept": (0.5, 0.1, 0.9),
    }
    assert _classify_miscalibration(intercept_only) is True

    neither = {"calibration_slope": (1.0, 0.9, 1.1), "calibration_intercept": (0.0, -0.1, 0.1)}
    assert _classify_miscalibration(neither) is False

    assert _classify_miscalibration({}) is False


PCAM_GOLDEN_METRICS = (
    Path(__file__).resolve().parent / "golden" / "pcam" / "metrics.parquet"
)


@pytest.mark.skipif(
    not PCAM_GOLDEN_METRICS.exists(), reason="tests/golden/pcam/metrics.parquet not committed"
)
def test_pcam_four_named_members_hit_all_four_taxonomy_cells() -> None:
    """S6.1 item 1's explicit ask: the four PCam members named in the session brief must land in
    the four distinct cells, verified against the real, frozen golden data (not a re-run of the
    B=2000 pipeline -- the classification functions are pure and take the golden's own numbers).
    """
    from predval.evaluate import _classify_miscalibration, _classify_repair

    metrics = pd.read_parquet(PCAM_GOLDEN_METRICS)

    def gains_for(model: str) -> dict:
        sub = metrics[
            (metrics["metric"] == "paired_gain_brier")
            & (metrics["model_id"] == model)
            & (metrics["subset"] == "common")
            & (metrics["stratum_kind"] == "overall")
            & (metrics["fit_mode"] == "crossfit")
        ]
        return {r["rung"]: (r["value"], r["ci_low"], r["ci_high"]) for _, r in sub.iterrows()}

    def calibration_for(model: str) -> dict:
        sub = metrics[
            (metrics["metric"].isin(["calibration_slope", "calibration_intercept"]))
            & (metrics["model_id"] == model)
            & (metrics["subset"] == "common")
            & (metrics["stratum_kind"] == "overall")
            & (metrics["rung"] == "rung0")
            & (metrics["ci_method"] == "cluster_robust_t")
        ]
        return {r["metric"]: (r["value"], r["ci_low"], r["ci_high"]) for _, r in sub.iterrows()}

    # A+B+: miscalibrated, repair demonstrated at rung2.
    assert _classify_miscalibration(calibration_for("p4m_seed7")) is True
    assert _classify_repair(gains_for("p4m_seed7"))[:2] == ("demonstrated", "rung2")

    # A+B-unproven: miscalibrated, no rung's gain interval excludes zero.
    assert _classify_miscalibration(calibration_for("tinyvgg_vl")) is True
    assert _classify_repair(gains_for("tinyvgg_vl")) == ("unproven", None, ())

    # A-B-counterproductive: well-calibrated, rung3 alone reliably worse.
    assert _classify_miscalibration(calibration_for("swin")) is False
    assert _classify_repair(gains_for("swin")) == ("counterproductive", None, ("rung3",))

    # A-B-counterproductive: well-calibrated, all three rungs reliably worse.
    assert _classify_miscalibration(calibration_for("e2cnn_s21")) is False
    assert _classify_repair(gains_for("e2cnn_s21")) == (
        "counterproductive",
        None,
        ("rung1", "rung2", "rung3"),
    )


def test_calibration_artefact_shape(result) -> None:
    """Item 2a: per-member decile points with a cluster band."""
    cal = result.calibration
    assert set(cal.columns) == {
        "model_id", "bin", "mean_pred", "obs_rate", "ci_low", "ci_high", "n", "n_events"
    }
    assert set(cal["model_id"]) == set(MODELS)
    assert (cal["ci_low"] <= cal["obs_rate"] + 1e-9).all() or cal["ci_low"].isna().any()


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
    assert set(paths) == {"metrics", "fragility", "coverage", "calibration", "manifest"}
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
