"""Verify repeat hashes and complete accounting for the fixed extension design."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(a, b):
    manifests = []
    for root in (a, b):
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
        rows = json.loads((root / "metrics.json").read_text())
        cases = json.loads((root / "cases.json").read_text())
        assert len(rows) == 164 and len(cases) == 41
        assert len({(r["seed"], r["family"], r["method"]) for r in rows}) == 164
        for family in ("logit_linear", "probability_linear"):
            assert {r["seed"] for r in rows if r["family"] == family} == set(
                range(20261020, 20261040)
            )
        assert sum(r["family"] == "wdbc_undistorted" for r in rows) == 4
        assert all(r["available"] or r["reason"] for r in rows)
        summaries = json.loads((root / "summary.json").read_text())["simulation_summary"]
        for s in summaries:
            selected = [
                r for r in rows if r["family"] == s["family"] and r["method"] == s["method"]
            ]
            assert s["attempted"] == len(selected) == 20
            assert s["n"] == sum(
                r["available"] and r.get(s["metric"]) is not None for r in selected
            )
        manifests.append(manifest)
    assert manifests[0] == manifests[1]
    print(
        json.dumps(
            {
                "stable_files_checked_per_run": len(manifests[0]["files"]),
                "repeat_manifest_equal": True,
                "cases": 41,
                "method_cases": 164,
                "all_fixed_seeds_accounted_for": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    args = p.parse_args()
    verify(args.run_a, args.run_b)
