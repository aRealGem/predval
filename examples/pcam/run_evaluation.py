#!/usr/bin/env python
"""Evaluate the PCam fixture end to end and write the S2 artefacts.

    uv run python examples/pcam/run_evaluation.py [--out DIR] [--n-boot N]

Writes metrics.parquet, fragility.parquet, coverage.parquet and manifest.json, then prints a
short summary. At the default B=2000 over 15 models this takes several minutes; pass --n-boot
for a quick smoke run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predval import evaluate, load_cohort, load_predictions

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

    print(f"cohort:      {cohort.n_subjects:,} subjects")
    print(f"models:      {len(predictions.model_ids)}")
    print(f"metrics:     {len(result.metrics):,} rows -> {paths['metrics']}")
    print(f"fragility:   {len(result.fragility):,} rows -> {paths['fragility']}")
    print()

    print("FLAGS")
    for flag in result.flags:
        print(f"  [{flag.severity:7}] {flag.code}: {flag.message}")
    print()

    overall = result.metrics[
        (result.metrics["stratum_kind"] == "overall")
        & (result.metrics["subset"] == "common")
        & (result.metrics["metric"] == "auroc")
        & (result.metrics["ci_method"] == "cluster_bootstrap")
    ].sort_values("value", ascending=False)

    print("AUROC by model (common subset, cluster bootstrap 95% CI)")
    for _, r in overall.iterrows():
        print(f"  {r['model_id']:16} {r['value']:.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print()

    naive = result.metrics[
        (result.metrics["stratum_kind"] == "overall")
        & (result.metrics["subset"] == "common")
        & (result.metrics["metric"] == "auroc")
    ]
    print("Unit-of-analysis exhibit -- AUROC interval width, cluster vs naive per-row")
    for model, grp in naive.groupby("model_id"):
        widths = {r["ci_method"]: r["ci_high"] - r["ci_low"] for _, r in grp.iterrows()}
        if "naive_row_bootstrap" in widths and widths["naive_row_bootstrap"] > 0:
            ratio = widths["cluster_bootstrap"] / widths["naive_row_bootstrap"]
            print(
                f"  {model:16} cluster {widths['cluster_bootstrap']:.4f}  "
                f"naive {widths['naive_row_bootstrap']:.4f}  ratio {ratio:5.1f}x"
            )
    print()

    cal = result.metrics[
        (result.metrics["stratum_kind"] == "overall")
        & (result.metrics["subset"] == "common")
        & (result.metrics["metric"] == "calibration_slope")
        & (result.metrics["ci_method"] == "cluster_robust_t")
    ].sort_values("value")
    print("Calibration slope (1.0 = correct spread; <1 = over-confident)")
    for _, r in cal.iterrows():
        print(f"  {r['model_id']:16} {r['value']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print()

    frag = result.fragility[
        (result.fragility["stratum_kind"] == "overall")
        & (result.fragility["subset"] == "common")
        & (result.fragility["metric"] == "auroc")
    ].sort_values("max_abs_delta", ascending=False)
    print("Fragility -- largest AUROC change from dropping any single slide")
    for _, r in frag.head(5).iterrows():
        print(f"  {r['model_id']:16} {r['max_abs_delta']:.4f}  (slide {r['culprit_cluster']})")

    boundary = result.metrics[
        (result.metrics["metric"] == "boundary_count")
        & (result.metrics["subset"] == "common")
        & (result.metrics["stratum_kind"] == "overall")
        & (result.metrics["value"] > 0)
    ].sort_values("value", ascending=False)
    if len(boundary):
        print()
        print("Data quality -- predictions exactly at 0 or 1 (eps-clip load)")
        for _, r in boundary.iterrows():
            print(f"  {r['model_id']:16} {int(r['value']):5d}")
        print(f"  {'TOTAL':16} {int(boundary['value'].sum()):5d}")

    with pd.option_context("display.width", 120):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
