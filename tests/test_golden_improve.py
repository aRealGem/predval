"""Golden-file regression test for the IMPROVE neoepitope example (third worked example).

`tests/golden/improve/` freezes the real, showable output of `examples/improve/run_evaluation.py`
at its default B=2000. This test regenerates that same run and checks every artefact against the
frozen baseline (continuous metrics use a fixed absolute 1e-10 precision budget; schema,
metadata, classifications, interval null sides and report output remain exact), so a change
that silently shifts the example's headline numbers -- rather than
being a deliberate, reviewed update to the golden files -- fails loudly. It is the IMPROVE
counterpart to `tests/test_golden_gusto.py` and `tests/test_golden_pcam.py`.

Opt-in (`PREDVAL_RUN_GOLDEN=1`) like the other two, and it needs the git-ignored fixture built by
`examples/improve/build_fixture.py` from the public IMPROVE_paper release (see
`examples/improve/README.md` and `tests/golden/improve/GOLDEN.md`).

To refresh the golden files deliberately after an intentional change, regenerate and copy:

    uv run python examples/improve/run_evaluation.py
    cp examples/improve/out/{metrics,fragility,calibration,coverage}.parquet tests/golden/improve/
    cp examples/improve/out/{manifest,findings}.json tests/golden/improve/
    cp examples/improve/out/report.html tests/golden/improve/

and note *why* the numbers moved in the commit that updates them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._improve_golden_comparison import assert_improve_outputs_match

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "improve"
GOLDEN = Path(__file__).resolve().parent / "golden" / "improve"

pytestmark = [
    pytest.mark.skipif(
        not (EXAMPLE / "cohort.parquet").exists() or not (EXAMPLE / "predictions.parquet").exists(),
        reason="IMPROVE fixture not built; run examples/improve/build_fixture.py",
    ),
    pytest.mark.skipif(
        os.environ.get("PREDVAL_RUN_GOLDEN") != "1",
        reason="B=2000 golden regression is opt-in; set PREDVAL_RUN_GOLDEN=1 to run it",
    ),
]


def test_improve_full_run_matches_golden_baseline(tmp_path: Path) -> None:
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(EXAMPLE / "run_evaluation.py"), "--out", str(out)],
        check=True,
        capture_output=True,
    )

    assert_improve_outputs_match(out, GOLDEN)
