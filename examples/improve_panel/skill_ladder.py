#!/usr/bin/env python
"""Brier skill at EVERY rung, for the panel: is negative skill a model fact or a rung choice?

    uv run python examples/improve_panel/skill_ladder.py [--out DIR] [--n-boot N]

predval's report quotes one Brier skill score per model, at the *admissible* rung -- the lowest
rung whose cross-fitted paired gain interval excludes zero. That rule is parsimony-first by
design: do not buy a more complex correction than you can prove you need. It also means the
headline number depends on a selection rule, and on this roster four of five models come out with
NEGATIVE skill, i.e. worse than quoting the cohort base rate to every patient. A reviewer's first
objection is the right one: maybe a higher rung would have rescued them and the rule hid it.

So this sidecar drops the selection rule and scores every rung the ladder could fit:

  rung0  the published probabilities, uncorrected
  rung1  cross-fitted intercept shift        (level)
  rung2  cross-fitted intercept + slope      (level and spread)
  rung3  cross-fitted restricted cubic spline

Each is scored with predval's own `brier_skill_interval`, anchored at 0 = always predict the
observed prevalence and 1 = perfect, with the same patient-clustered bootstrap, B and seed the
cohort spec declares. A rung the ladder cannot fit for a model is reported unavailable with its
reason, never imputed.

If a model's interval stays below zero at every rung, "negative skill" is a property of the
scores, not of the admissible-rung rule. That is the claim this script exists to let a reader
check, and it is the claim predval's own report is not structured to make.

Sidecar, not part of predval's contract: the library reports the admissible rung on purpose.
Writes skill_ladder.parquet and prints a summary. No network calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predval import load_cohort, load_predictions

# predval's own cross-fitting, reused rather than reimplemented so the corrections scored here are
# byte-for-byte those the main report's rung1-3 columns are scored on. Private because it is not a
# contract surface; this example is allowed to reach for it, library code is not.
from predval.recalibrate import _crossfit_predictions
from predval.uncertainty import brier_skill_interval

HERE = Path(__file__).resolve().parent
RUNGS = ("rung1", "rung2", "rung3")
MAX_FOLDS = 5
W = 22


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--n-boot", type=int, default=None, help="override uncertainty.n_boot")
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

    rows = []
    for model_id, grp in predictions.frame.groupby("model_id"):
        # inner join: a peptide this model did not score contributes no row, and the model's own
        # prevalence reference is recomputed on the rows it does cover (see the coverage note in
        # cohort.yaml -- the full and common subsets are different questions).
        merged = table.merge(
            grp[["subject_id", "predicted"]], on="subject_id", how="inner", validate="1:1"
        )
        y = merged["response"].to_numpy(dtype=float)
        p = merged["predicted"].to_numpy(dtype=float)
        groups = merged["patient"].to_numpy()
        k_folds = min(MAX_FOLDS, np.unique(groups).size)

        corrected: dict[str, np.ndarray | None] = {"rung0": p}
        reasons: dict[str, str | None] = {"rung0": None}
        for rung in RUNGS:
            cf, reason = _crossfit_predictions(rung, y, p, groups, k_folds)
            corrected[rung] = cf
            reasons[rung] = reason

        for rung, q in corrected.items():
            if q is None:
                rows.append(
                    {
                        "model_id": model_id,
                        "rung": rung,
                        "n": int(len(merged)),
                        "available": False,
                        "unavailable_reason": reasons[rung],
                        "bss": float("nan"),
                        "bss_ci_low": float("nan"),
                        "bss_ci_high": float("nan"),
                        "brier": float("nan"),
                        "prevalence": float(np.mean(y)),
                        "n_boot": n_boot,
                        "seed": seed,
                        "ci_level": ci,
                    }
                )
                continue
            bss, interval = brier_skill_interval(y, q, groups, ci, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "model_id": model_id,
                    "rung": rung,
                    "n": int(len(merged)),
                    "available": True,
                    "unavailable_reason": None,
                    "bss": float(bss),
                    "bss_ci_low": float(interval.low),
                    "bss_ci_high": float(interval.high),
                    "brier": float(np.mean((q - y) ** 2)),
                    "prevalence": float(np.mean(y)),
                    "n_boot": n_boot,
                    "seed": seed,
                    "ci_level": ci,
                }
            )

    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "skill_ladder.parquet"
    out.to_parquet(path, index=False)

    print(f"skill ladder -> {path}")
    print(f"B={n_boot} patient-clustered resamples, seed {seed}, {ci:.0%} intervals")
    print("BSS: 0 = always predict this model's own observed prevalence, 1 = perfect.")
    print("A rung whose whole interval is below 0 is reliably WORSE than quoting the base rate.\n")

    for model_id, grp in out.groupby("model_id"):
        prev = float(grp["prevalence"].iloc[0])
        n = int(grp["n"].iloc[0])
        print(
            f"{model_id}  (n={n:,}, prevalence {prev:.4%}, reference Brier {prev * (1 - prev):.4f})"
        )
        for rung in ("rung0", *RUNGS):
            r = grp[grp["rung"] == rung].iloc[0]
            if not bool(r["available"]):
                print(f"  {rung:{8}} unavailable: {r['unavailable_reason']}")
                continue
            mark = " "
            if np.isfinite(r["bss_ci_high"]) and r["bss_ci_high"] < 0:
                mark = "-"  # reliably worse than the base rate
            elif np.isfinite(r["bss_ci_low"]) and r["bss_ci_low"] > 0:
                mark = "+"  # reliably better
            interval = f"[{100 * r['bss_ci_low']:+8.2f}%, {100 * r['bss_ci_high']:+8.2f}%]"
            print(
                f"  {rung:{8}} brier {r['brier']:.4f}  BSS {100 * r['bss']:+8.2f}% "
                f"{interval} {mark}"
            )
        ok = grp[grp["available"]]
        if len(ok):
            # Three outcomes, not two. "No rung is reliably positive" is the finding; it is NOT
            # the same claim as "every rung is reliably negative", and collapsing the two would
            # overstate the result in exactly the direction this fixture is arguing.
            if (ok["bss_ci_low"] > 0).any():
                verdict = "reliably POSITIVE skill at some rung"
            elif (ok["bss_ci_high"] < 0).all():
                verdict = "reliably NEGATIVE skill at every fitted rung"
            else:
                verdict = "no rung reaches reliably positive skill (best rung straddles zero)"
            print(f"  -> {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
