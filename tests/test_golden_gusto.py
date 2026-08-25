"""Golden-file regression test for the GUSTO-I example (second worked example).

`tests/golden/gusto/` freezes the real, showable output of `examples/gusto/run_evaluation.py` at
its default B=2000. This test regenerates that same run and checks every artefact against the
frozen baseline, so a change that silently shifts the example's headline numbers -- rather than
being a deliberate, reviewed update to the golden files -- fails loudly. It is the GUSTO
counterpart to `tests/test_golden_pcam.py`, closing the "no golden-file baseline for GUSTO yet"
backlog item.

Like the PCam golden it is opt-in (`PREDVAL_RUN_GOLDEN=1`), for consistency and because it needs
the git-ignored GUSTO fixture. Unlike PCam it is cheap (a single logistic over ~38.6k subjects runs
in seconds, not minutes), so running it is not a meaningful cost when you do opt in.

To refresh the golden files deliberately after an intentional change to the fixture or the
metrics/report/findings code, regenerate and copy:

    uv run --group examples python examples/gusto/run_evaluation.py
    cp examples/gusto/out/{metrics,fragility,calibration,coverage}.parquet tests/golden/gusto/
    cp examples/gusto/out/{manifest,findings}.json tests/golden/gusto/
    cp examples/gusto/out/report.html tests/golden/gusto/

and note *why* the numbers moved in the commit that updates them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from predval.hashing import hash_file

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "gusto"
GOLDEN = Path(__file__).resolve().parent / "golden" / "gusto"

pytestmark = [
    pytest.mark.skipif(
        not (EXAMPLE / "cohort.parquet").exists() or not (EXAMPLE / "predictions.parquet").exists(),
        reason="GUSTO fixture not built; run examples/gusto/fetch_data.py + make_predictions.py",
    ),
    pytest.mark.skipif(
        os.environ.get("PREDVAL_RUN_GOLDEN") != "1",
        reason="B=2000 golden regression is opt-in; set PREDVAL_RUN_GOLDEN=1 to run it",
    ),
]

#: findings.json is compared separately -- it embeds predval's own git commit, which legitimately
#: differs on every commit to this repo and is not a signal about the GUSTO evaluation itself.
BYTE_IDENTICAL_ARTIFACTS = (
    "metrics.parquet",
    "fragility.parquet",
    "calibration.parquet",
    "coverage.parquet",
    "manifest.json",
    "report.html",
)


def _normalize_findings(path: Path) -> dict:
    doc = json.loads(path.read_text())
    doc["provenance"]["git_commit"] = None
    return doc


def test_gusto_full_run_matches_golden_baseline(tmp_path: Path) -> None:
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(EXAMPLE / "run_evaluation.py"), "--out", str(out)],
        check=True,
        capture_output=True,
    )

    mismatches = []
    for name in BYTE_IDENTICAL_ARTIFACTS:
        golden_file = GOLDEN / name
        assert golden_file.exists(), f"missing golden file: {golden_file}"
        if hash_file(out / name) != hash_file(golden_file):
            mismatches.append(name)

    if _normalize_findings(out / "findings.json") != _normalize_findings(GOLDEN / "findings.json"):
        mismatches.append("findings.json")

    assert not mismatches, (
        f"artefacts drifted from tests/golden/gusto/: {mismatches}. "
        "If this is an intentional change, refresh the golden files (see this module's "
        "docstring) and explain why the numbers moved in the commit."
    )
