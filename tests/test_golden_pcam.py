"""Golden-file regression test for the PCam example (S5 item 2).

`tests/golden/pcam/` freezes the real, showable output of `examples/pcam/run_evaluation.py` at
its default B=2000. This test regenerates that same run and checks every artefact against the
frozen baseline, so a change that silently shifts the pitch's headline numbers -- rather than
being a deliberate, reviewed update to the golden files -- fails loudly.

It is opt-in (`PREDVAL_RUN_GOLDEN=1`) because a real B=2000 run over 15 models takes several
minutes; the fast default suite should not pay that cost on every invocation. Determinism at
smaller B is already covered by the byte-identical tests in test_report.py/test_findings.py/
test_figures.py -- this test instead guards the actual numbers a reader sees, not just that two
runs of arbitrary B agree with each other.

To refresh the golden files deliberately after an intentional change to the fixture or the
metrics/report/findings code, regenerate and copy:

    uv run python examples/pcam/run_evaluation.py
    cp examples/pcam/out/{metrics,fragility,calibration,coverage}.parquet tests/golden/pcam/
    cp examples/pcam/out/{manifest,findings}.json tests/golden/pcam/
    cp examples/pcam/out/report.html tests/golden/pcam/

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

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "pcam"
GOLDEN = Path(__file__).resolve().parent / "golden" / "pcam"

pytestmark = [
    pytest.mark.skipif(
        not (EXAMPLE / "cohort.parquet").exists() or not (EXAMPLE / "predictions.parquet").exists(),
        reason="PCam fixture not built; run examples/pcam/build_fixture.py",
    ),
    pytest.mark.skipif(
        os.environ.get("PREDVAL_RUN_GOLDEN") != "1",
        reason="full B=2000 golden regression is opt-in; set PREDVAL_RUN_GOLDEN=1 to run it",
    ),
]

#: findings.json is compared separately -- it embeds predval's own git commit, which legitimately
#: differs on every commit to this repo and is not a signal about the PCam evaluation itself.
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


def test_pcam_full_run_matches_golden_baseline(tmp_path: Path) -> None:
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
        f"artefacts drifted from tests/golden/pcam/: {mismatches}. "
        "If this is an intentional change, refresh the golden files (see this module's "
        "docstring) and explain why the numbers moved in the commit."
    )
