#!/usr/bin/env python
"""Evaluate the IMPROVE neoepitope fixture end to end and write the artefacts.

    uv run python examples/improve/run_evaluation.py [--out DIR] [--n-boot N]

Writes metrics.parquet, fragility.parquet, coverage.parquet, manifest.json, report.html and
findings.json, then prints the summary: prevalence anchoring, discrimination, the
unit-of-analysis exhibit over patients, calibration slope and intercept, the recalibration
ladder (as-published versus cross-fitted) and fragility to dropping any single patient.
At the declared B=2000 over 7 models this takes a few minutes; pass --n-boot for a smoke run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predval import evaluate, load_cohort, load_predictions, write_findings, write_report
from predval.metrics import optimism

HERE = Path(__file__).resolve().parent
W = 26  # model ids here are long; widen every column that prints one


def _overall(metrics: pd.DataFrame, metric: str, ci_method: str | None = None) -> pd.DataFrame:
    sel = (
        (metrics["stratum_kind"] == "overall")
        & (metrics["subset"] == "common")
        & (metrics["metric"] == metric)
    )
    if ci_method is not None:
        sel &= metrics["ci_method"] == ci_method
    return metrics[sel]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--n-boot", type=int, default=None, help="override uncertainty.n_boot")
    args = ap.parse_args(argv)

    cohort = load_cohort(HERE / "cohort.yaml")
    predictions = load_predictions(HERE / "predictions.parquet")

    if args.n_boot is not None:
        unc = cohort.spec.uncertainty.model_copy(update={"n_boot": args.n_boot})
        spec = cohort.spec.model_copy(update={"uncertainty": unc})
        cohort = type(cohort)(**{**cohort.__dict__, "spec": spec})

    result = evaluate(cohort, predictions)
    paths = result.write(args.out)
    report_path = write_report(result, args.out / "report.html")
    findings_path = write_findings(result, args.out / "findings.json")

    print(f"cohort:      {cohort.n_subjects:,} peptides")
    print(f"models:      {len(predictions.model_ids)}")
    print(f"metrics:     {len(result.metrics):,} rows -> {paths['metrics']}")
    print(f"report:      {report_path}")
    print(f"findings:    {findings_path}")
    print()

    print("FLAGS")
    for flag in result.flags:
        print(f"  [{flag.severity:7}] {flag.code}: {flag.message}")
    print()

    # ---- prevalence anchoring: the first thing to look at under 2.7% positives -------------
    frame = predictions.frame
    prev = float(cohort.table[cohort.spec.outcome.field].mean())
    print(f"Prevalence anchoring -- observed immunogenic rate {prev:.4%}")
    print(f"  {'model':{W}} {'mean pred':>10} {'ratio':>7} {'min':>8} {'max':>8}")
    for model, grp in frame.groupby("model_id"):
        mean_p = float(grp["predicted"].mean())
        print(
            f"  {model:{W}} {mean_p:10.4f} {mean_p / prev:6.1f}x "
            f"{grp['predicted'].min():8.4f} {grp['predicted'].max():8.4f}"
        )
    print()

    auroc = _overall(result.metrics, "auroc", "cluster_bootstrap").sort_values(
        "value", ascending=False
    )
    print("AUROC by model (common subset, patient-clustered bootstrap 95% CI)")
    for _, r in auroc.iterrows():
        print(f"  {r['model_id']:{W}} {r['value']:.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print()

    ap_rows = _overall(result.metrics, "average_precision", "cluster_bootstrap").sort_values(
        "value", ascending=False
    )
    if len(ap_rows):
        print(f"Average precision (no-skill baseline = prevalence = {prev:.4f})")
        for _, r in ap_rows.iterrows():
            print(
                f"  {r['model_id']:{W}} {r['value']:.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            )
        print()

    naive = _overall(result.metrics, "auroc")
    print("Unit-of-analysis exhibit -- AUROC interval width, patient vs naive per-peptide")
    for model, grp in naive.groupby("model_id"):
        widths = {r["ci_method"]: r["ci_high"] - r["ci_low"] for _, r in grp.iterrows()}
        if "naive_row_bootstrap" in widths and widths["naive_row_bootstrap"] > 0:
            ratio = widths["cluster_bootstrap"] / widths["naive_row_bootstrap"]
            print(
                f"  {model:{W}} patient {widths['cluster_bootstrap']:.4f}  "
                f"naive {widths['naive_row_bootstrap']:.4f}  ratio {ratio:5.1f}x"
            )
    print()

    for metric, blurb in (
        ("calibration_slope", "1.0 = correct spread; <1 = over-spread/over-confident"),
        ("calibration_intercept", "0.0 = correctly anchored; <0 = systematically too high"),
    ):
        rows = _overall(result.metrics, metric, "cluster_robust_t").sort_values("value")
        if not len(rows):
            continue
        print(f"{metric.replace('_', ' ').capitalize()} ({blurb})")
        for _, r in rows.iterrows():
            interval = f"[{r['ci_low']:8.3f}, {r['ci_high']:8.3f}]"
            print(f"  {r['model_id']:{W}} {r['value']:8.3f}  {interval}")
        print()

    lad = _overall(result.metrics, "brier").pivot_table(
        index="model_id", columns=["rung", "fit_mode"], values="value"
    )

    def cell(model: str, rung: str, mode: str) -> float:
        col = (rung, mode)
        return float(lad.loc[model, col]) if col in lad.columns else float("nan")

    def cf_gain(model: str) -> float:
        r0 = cell(model, "rung0", "apparent")
        cf = min(cell(model, r, "crossfit") for r in ("rung1", "rung2", "rung3"))
        return r0 - cf

    order = sorted(lad.index, key=cf_gain, reverse=True)
    print("Recalibration ladder -- Brier, as-published vs cross-fitted by held-out patients")
    print(f"  reference: a constant predictor at the prevalence scores {prev * (1 - prev):.4f}")
    print("  optim_r2 > 0 == the apparent rung2 fit flattered itself versus held-out patients")
    print(
        f"  {'model':{W}} {'rung0':>8} {'r1_cf':>8} {'r2_cf':>8} {'r3_cf':>8} "
        f"{'gain_cf':>8} {'optim_r2':>9}"
    )
    for model in order:
        r0 = cell(model, "rung0", "apparent")
        r1_cf = cell(model, "rung1", "crossfit")
        r2_cf = cell(model, "rung2", "crossfit")
        r3_cf = cell(model, "rung3", "crossfit")
        gain = r0 - min(r1_cf, r2_cf, r3_cf)
        optim = optimism("brier", cell(model, "rung2", "apparent"), r2_cf)
        print(
            f"  {model:{W}} {r0:8.4f} {r1_cf:8.4f} {r2_cf:8.4f} {r3_cf:8.4f} "
            f"{gain:8.4f} {optim:+9.4f}"
        )
    print()

    gains = _overall(result.metrics, "paired_gain_brier")
    if len(gains):
        print("Paired Brier gain, cross-fitted (CI excluding 0 == a demonstrated repair)")
        for model, grp in gains.groupby("model_id"):
            parts = []
            for _, r in grp.sort_values("rung").iterrows():
                mark = " "
                if pd.notna(r["ci_low"]) and pd.notna(r["ci_high"]):
                    if r["ci_low"] > 0:
                        mark = "+"
                    elif r["ci_high"] < 0:
                        mark = "-"
                parts.append(f"{r['rung']}{mark}{r['value']:+.5f}")
            print(f"  {model:{W}} " + "  ".join(parts))
        print("  + = repair demonstrated   - = recalibration reliably HARMS   blank = unproven")
        print()

    frag = result.fragility[
        (result.fragility["stratum_kind"] == "overall")
        & (result.fragility["subset"] == "common")
        & (result.fragility["metric"] == "auroc")
    ].sort_values("max_abs_delta", ascending=False)
    print("Fragility -- largest AUROC change from dropping any single patient")
    for _, r in frag.head(7).iterrows():
        print(f"  {r['model_id']:{W}} {r['max_abs_delta']:.4f}  (patient {r['culprit_cluster']})")

    boundary = _overall(result.metrics, "boundary_count")
    boundary = boundary[boundary["value"] > 0]
    if len(boundary):
        print()
        print("Data quality -- predictions exactly at 0 or 1 (eps-clip load)")
        for _, r in boundary.iterrows():
            print(f"  {r['model_id']:{W}} {int(r['value']):5d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
