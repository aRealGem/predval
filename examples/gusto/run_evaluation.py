#!/usr/bin/env python
"""Evaluate the GUSTO-I example end to end and write the artefacts.

    uv run --group examples python examples/gusto/run_evaluation.py [--out DIR] [--n-boot N]

Writes metrics.parquet, fragility.parquet, calibration.parquet, coverage.parquet, manifest.json,
report.html, and findings.json, then prints a short summary. At the default B=2000 over a single
model and ~38.6k subjects this is fast (seconds, not minutes -- unlike PCam's 15-model B=2000 run).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from predval import evaluate, load_cohort, load_predictions, write_findings, write_report

HERE = Path(__file__).resolve().parent


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

    print(f"cohort:      {cohort.n_subjects:,} subjects")
    print(f"models:      {len(predictions.model_ids)}")
    print(f"metrics:     {len(result.metrics):,} rows -> {paths['metrics']}")
    print(f"fragility:   {len(result.fragility):,} rows -> {paths['fragility']}")
    print(f"report:      {report_path}")
    print(f"findings:    {findings_path}")
    print()

    print("FLAGS")
    for flag in result.flags:
        print(f"  [{flag.severity:7}] {flag.code}: {flag.message}")
    print()

    m = result.metrics
    overall_common = m[(m["stratum_kind"] == "overall") & (m["subset"] == "common")]

    auroc = overall_common[
        (overall_common["metric"] == "auroc") & (overall_common["ci_method"] == "cluster_bootstrap")
    ]
    print("AUROC (cluster bootstrap 95% CI)")
    for _, r in auroc.iterrows():
        print(f"  {r['model_id']:28} {r['value']:.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print()

    cal = overall_common[
        (overall_common["metric"] == "calibration_slope")
        & (overall_common["ci_method"] == "cluster_robust_t")
    ]
    print("Calibration slope (1.0 = correct spread; <1 = over-confident)")
    for _, r in cal.iterrows():
        print(f"  {r['model_id']:28} {r['value']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    intercept = overall_common[
        (overall_common["metric"] == "calibration_intercept")
        & (overall_common["ci_method"] == "cluster_robust_t")
    ]
    print("Calibration intercept (0 = correct level)")
    for _, r in intercept.iterrows():
        print(f"  {r['model_id']:28} {r['value']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print()

    ladder = overall_common[overall_common["metric"] == "brier"].pivot_table(
        index="model_id", columns=["rung", "fit_mode"], values="value"
    )
    print("Recalibration ladder -- Brier by rung (as-published vs cross-fitted)")
    print(ladder)
    print()

    verdict = result.manifest.get("verdict", {})
    print("Verdict")
    for model, v in verdict.items():
        print(
            f"  {model:28} BSS={v['bss']:.4f} [{v['bss_ci_low']:.4f}, {v['bss_ci_high']:.4f}]"
            f"  best_rung={v['best_rung']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
