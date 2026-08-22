#!/usr/bin/env python
"""One-command reproduction of the PCam example, checked against the frozen golden baseline.

    uv run python examples/pcam/reproduce.py [--campaign PATH] [--out DIR] [--golden DIR]

This is a local, single-machine reproduction, not a from-anywhere clean-clone one, and that
limit is deliberate: `examples/pcam/cohort.parquet` and `predictions.parquet` are derived from
real histopathology predictions and are intentionally not committed to this repository (see
`.gitignore`). What "clean checkout" means here is: given either the fixture parquet files or a
local `~/histopath-cancer-detection` campaign checkout to build them from, one command produces
the full report and findings with no other manual steps.

Steps:
  1. Use the fixture parquet files if already present; otherwise build them from the campaign
     checkout (read-only over that directory). If neither is available, fail with a clear message
     rather than guessing.
  2. Run the full evaluation (B=2000, seed 1337, the same defaults `run_evaluation.py` uses).
  3. Compare every written artefact against `tests/golden/pcam/` and report PASS/FAIL per file.

Exit code is non-zero if the fixture cannot be built/found, the evaluation fails, or any artefact
does not match the golden baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from predval.hashing import hash_file  # noqa: E402

DEFAULT_CAMPAIGN = Path.home() / "histopath-cancer-detection"
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "golden" / "pcam"

#: Compared byte-for-byte. findings.json is handled separately because it legitimately embeds
#: predval's own git commit, which drifts every time this repo is committed to -- see below.
BYTE_IDENTICAL_ARTIFACTS = (
    "metrics.parquet",
    "fragility.parquet",
    "calibration.parquet",
    "coverage.parquet",
    "manifest.json",
    "report.html",
)

_GIT_COMMIT_PLACEHOLDER = "<normalized-for-golden-comparison>"


def _normalize_findings(path: Path) -> str:
    """findings.json with provenance.git_commit blanked out.

    git_commit records *predval's own* commit at generation time (see findings.py); it is
    expected to differ on every commit to this repo and is not a signal about the PCam evaluation
    itself, so the golden comparison excludes it rather than failing on every commit forever.
    """
    doc = json.loads(path.read_text())
    doc["provenance"]["git_commit"] = _GIT_COMMIT_PLACEHOLDER
    return json.dumps(doc, sort_keys=True, indent=2)


def ensure_fixture(campaign: Path) -> None:
    cohort_pq = HERE / "cohort.parquet"
    predictions_pq = HERE / "predictions.parquet"
    if cohort_pq.exists() and predictions_pq.exists():
        print(f"fixture already present: {cohort_pq.name}, {predictions_pq.name}")
        return

    if not campaign.exists():
        print(
            "PCam fixture is not built, and no campaign checkout was found to build it from.\n"
            f"  looked for: {campaign}\n"
            "This example evaluates real histopathology predictions that are not committed to "
            "this repository. Either point --campaign at a local "
            "~/histopath-cancer-detection checkout, or copy in the two fixture files "
            "(cohort.parquet, predictions.parquet) directly.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"building fixture from campaign checkout: {campaign}")
    subprocess.run(
        [sys.executable, str(HERE / "build_fixture.py"), "--campaign", str(campaign)],
        check=True,
    )


def run_evaluation(out: Path) -> None:
    print(f"running full evaluation (B=2000) -> {out}")
    subprocess.run(
        [sys.executable, str(HERE / "run_evaluation.py"), "--out", str(out)],
        check=True,
    )


def compare_to_golden(out: Path, golden: Path) -> bool:
    if not golden.exists():
        print(f"no golden baseline at {golden}; nothing to compare against", file=sys.stderr)
        return False

    print()
    print(f"comparing {out} against golden baseline {golden}")
    all_match = True
    for name in BYTE_IDENTICAL_ARTIFACTS:
        produced, expected = out / name, golden / name
        if not expected.exists():
            print(f"  [SKIP ] {name} (no golden file)")
            continue
        match = produced.exists() and hash_file(produced) == hash_file(expected)
        all_match &= match
        print(f"  [{'PASS' if match else 'FAIL'}] {name}")

    produced_findings, expected_findings = out / "findings.json", golden / "findings.json"
    if expected_findings.exists():
        match = produced_findings.exists() and (
            _normalize_findings(produced_findings) == _normalize_findings(expected_findings)
        )
        all_match &= match
        print(f"  [{'PASS' if match else 'FAIL'}] findings.json (git_commit excluded)")

    return all_match


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    args = ap.parse_args(argv)

    ensure_fixture(args.campaign)
    run_evaluation(args.out)
    matched = compare_to_golden(args.out, args.golden)

    print()
    if matched:
        print("reproduction matches the frozen golden baseline.")
        return 0
    print(
        "reproduction DID NOT match the golden baseline -- see FAIL lines above.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
