#!/usr/bin/env python3
"""Verify clean repeats and emit compact, regenerable R summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(py: Path, py_repeat: Path, r: Path, r_repeat: Path):
    assert (py / "manifest.json").read_bytes() == (py_repeat / "manifest.json").read_bytes()
    files = json.loads((py / "manifest.json").read_text())["files"]
    for relative, expected in files.items():
        assert sha(py / relative) == sha(py_repeat / relative) == expected
    shared_r = ["r_metrics.csv", "r_agreement.csv", "sessionInfo.txt"]
    summary = []
    for name in ("wdbc_heldout", "synthetic_clustered"):
        shared_r.extend([f"{name}_r_corrected.csv", f"{name}_rsample_gains.csv"])
        gains = pd.read_csv(r / f"{name}_rsample_gains.csv").gain.to_numpy()
        assert len(gains) == 1000 and np.isfinite(gains).all()
        lo, hi = np.quantile(gains, [0.025, 0.975])
        summary.append(
            {
                "fixture": name,
                "replicates": len(gains),
                "low": lo,
                "high": hi,
                "method": "rsample native group_bootstraps; fixed R logit-GLM predictions",
            }
        )
    for name in shared_r:
        assert (r / name).read_bytes() == (r_repeat / name).read_bytes(), name
    pd.DataFrame(summary).to_csv(r / "r_bootstrap_summary.csv", index=False, float_format="%.17g")
    result = {
        "python_stable_files_verified": len(files),
        "python_manifest_sha256": sha(py / "manifest.json"),
        "r_repeated_stable_files_identical": len(shared_r),
        "r_files_sha256": {name: sha(r / name) for name in shared_r},
        "r_timing_first": (r / "resource_usage.txt").read_text().strip(),
        "r_timing_repeat": (r_repeat / "resource_usage.txt").read_text().strip(),
        "scope": "Same final inputs and fixed dependencies. Runtime excluded from hash equality.",
    }
    (r / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python", type=Path)
    parser.add_argument("python_repeat", type=Path)
    parser.add_argument("r", type=Path)
    parser.add_argument("r_repeat", type=Path)
    args = parser.parse_args()
    verify(args.python, args.python_repeat, args.r, args.r_repeat)
