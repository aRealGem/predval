#!/usr/bin/env python
"""Budgeted-yield sidecar for the IMPROVE fixture: what a slot budget actually buys.

    uv run python examples/improve/budget_yield.py [--out DIR] [--n-boot N]

Calibration is a property of probabilities; antigen selection is a property of a *budget*. A
personalised vaccine encodes on the order of 20 neoepitopes and a TCR-T product carries 3-5, so
the operative question is not "is peptide X immunogenic" but "of the k slots I am going to fill
for this patient, how many carry a real target, and how many did the score promise?"

For each model and each budget k this script takes every patient's own top-k candidates by
score and reports three numbers per patient, averaged over patients with a patient-level
bootstrap interval:

  observed   -- immunogenic peptides actually in the top-k (the truth)
  promised   -- the sum of the model's own published probabilities over that same top-k, which
                is what those probabilities assert the yield will be
  recalibrated -- the same sum after a cross-fitted intercept correction (predval's rung1),
                fitted on other patients and applied here

`promised` minus `observed` is the slot budget's miscalibration cost, in units of antigens. It
is not a ranking error: the same top-k set is used for all three columns.

This is a sidecar, not part of predval's contract: predval reports calibration, and the mapping
from calibration to a decision is cohort-specific. It writes budget_yield.parquet and prints a
summary. No network calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predval import load_cohort, load_predictions

# predval's own cross-fitting, reused rather than reimplemented so the correction applied here is
# byte-for-byte the one the main report's rung1 column is scored on. Private because it is not a
# contract surface; this example is allowed to reach for it, library code is not.
from predval.recalibrate import _crossfit_predictions

HERE = Path(__file__).resolve().parent
BUDGETS = (3, 5, 10, 20)
W = 26


def _per_patient_topk(
    frame: pd.DataFrame, k: int, score: str, extra: tuple[str, ...]
) -> pd.DataFrame:
    """Each patient's own top-k rows by `score`, ties broken deterministically by subject_id."""
    ordered = frame.sort_values([score, "subject_id"], ascending=[False, True])
    top = ordered.groupby("patient", sort=True).head(k)
    cols = ["patient", "response", score, *extra]
    return top[cols]


def _cluster_bootstrap(
    per_patient: pd.DataFrame, columns: list[str], n_boot: int, seed: int, ci: float
) -> dict[str, tuple[float, float]]:
    """Resample whole patients with replacement; return a percentile interval per column."""
    rng = np.random.default_rng(seed)
    values = per_patient[columns].to_numpy(dtype=float)
    n = len(per_patient)
    draws = np.empty((n_boot, len(columns)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = values[idx].mean(axis=0)
    lo = (1.0 - ci) / 2.0 * 100.0
    return {
        col: (float(np.percentile(draws[:, j], lo)), float(np.percentile(draws[:, j], 100.0 - lo)))
        for j, col in enumerate(columns)
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args(argv)

    cohort = load_cohort(HERE / "cohort.yaml")
    predictions = load_predictions(HERE / "predictions.parquet")
    spec = cohort.spec
    n_boot = args.n_boot if args.n_boot is not None else spec.uncertainty.n_boot
    seed = spec.uncertainty.seed
    ci = spec.uncertainty.ci_level

    table = cohort.table[[spec.subject_key, "patient", spec.outcome.field]].rename(
        columns={spec.subject_key: "subject_id", spec.outcome.field: "response"}
    )
    prevalence = float(table["response"].mean())

    rows = []
    for model_id, grp in predictions.frame.groupby("model_id"):
        merged = table.merge(
            grp[["subject_id", "predicted"]], on="subject_id", how="inner", validate="1:1"
        )
        y = merged["response"].to_numpy(dtype=float)
        p = merged["predicted"].to_numpy(dtype=float)
        groups = merged["patient"].to_numpy()
        k_folds = min(5, np.unique(groups).size)
        cf, reason = _crossfit_predictions("rung1", y, p, groups, k_folds)
        if cf is None:
            raise SystemExit(f"{model_id}: rung1 cross-fit unavailable ({reason})")
        merged["recalibrated"] = cf

        for k in BUDGETS:
            top = _per_patient_topk(merged, k, "predicted", ("recalibrated",))
            per_patient = top.groupby("patient", sort=True).agg(
                observed=("response", "sum"),
                promised=("predicted", "sum"),
                recalibrated=("recalibrated", "sum"),
                slots=("response", "size"),
            )
            cols = ["observed", "promised", "recalibrated"]
            intervals = _cluster_bootstrap(per_patient, cols, n_boot, seed, ci)
            means = {c: float(per_patient[c].mean()) for c in cols}
            rows.append(
                {
                    "model_id": model_id,
                    "budget_k": k,
                    "n_patients": int(len(per_patient)),
                    "slots_median": float(per_patient["slots"].median()),
                    **means,
                    **{f"{c}_ci_low": intervals[c][0] for c in cols},
                    **{f"{c}_ci_high": intervals[c][1] for c in cols},
                    "patients_with_zero_hits": int((per_patient["observed"] == 0).sum()),
                    "random_baseline": k * prevalence,
                    "n_boot": n_boot,
                    "seed": seed,
                    "ci_level": ci,
                }
            )

    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "budget_yield.parquet"
    out.to_parquet(path, index=False)

    print(f"budget yield -> {path}")
    print(
        f"cohort prevalence {prevalence:.4%}; {out['n_patients'].iloc[0]} patients; "
        f"B={n_boot} patient-level resamples, seed {seed}\n"
    )
    for k in BUDGETS:
        sub = out[out["budget_k"] == k].sort_values("observed", ascending=False)
        print(f"budget k={k} slots per patient  (random pick would yield {k * prevalence:.3f})")
        print(
            f"  {'model':{W}} {'observed':>24} {'promised':>10} {'recalib':>9} "
            f"{'over by':>8} {'0-hit pts':>10}"
        )
        for _, r in sub.iterrows():
            print(
                f"  {r['model_id']:{W}} {r['observed']:8.3f} "
                f"[{r['observed_ci_low']:.3f}, {r['observed_ci_high']:.3f}]"
                f" {r['promised']:10.3f} {r['recalibrated']:9.3f}"
                f" {r['promised'] / max(r['observed'], 1e-9):7.1f}x"
                f" {r['patients_with_zero_hits']:6d}/{r['n_patients']}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
