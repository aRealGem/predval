#!/usr/bin/env python3
"""Prepared adapter only. No isitfair execution was authorized in this benchmark.

After installing an approved pinned isitfair build:
python isitfair_adapter.py path/to/python-results output-dir
Native L2-regularized Platt is a DIFFERENT correction from PREDVAL's unpenalized rung2.
The scores API internally computes native intervals; this adapter discards them.
Those intervals assume one row per patient and are invalid for repeated-row data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(source: Path, out: Path):
    from isitfair import FairnessAudit
    from sklearn.metrics import brier_score_loss, roc_auc_score

    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in ("wdbc_heldout", "synthetic_clustered"):
        df = pd.read_csv(source / f"{name}.csv")
        # A single overall group requests the common correction; no claim of a
        # clinically meaningful subgroup or fairness comparison is made here.
        audit = FairnessAudit(
            df.y.to_numpy(),
            df.p.to_numpy(),
            {"cohort": np.repeat("all", len(df))},
            threshold=0.5,
            n_bootstrap=20,
            random_state=20261010,
        )
        corrected = audit.recalibration_ladder_scores(
            by="cohort", method="platt", folds=df.fold.to_numpy()
        )["common"]
        for method, p in [("published", df.p.to_numpy()), ("native_platt", corrected)]:
            rows.append(
                {
                    "fixture": name,
                    "method": method,
                    "brier": brier_score_loss(df.y, p),
                    "auroc": roc_auc_score(df.y, p),
                }
            )
        df["isitfair_native_platt"] = corrected
        df.to_csv(out / f"{name}_isitfair.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "isitfair_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.source, args.out)
