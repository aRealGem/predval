"""The ingestion door: predval must fail loudly, legibly, and actionably on hostile input, and
must never emit a report from a corrupted file.

Parametrized over `make_corruptions.CASE_SPECS`. Each case asserts exactly one contract:

* hard_fail  -- the loader raises the right error, the message names the file, and running the full
                pipeline writes NO artefact (proven in-process by pointing the runner at a fresh out
                dir and confirming it stays empty; the raise happens at load, before `evaluate`).
* warn_or_drop -- a documented flag or drop with a count, asserted directly.
* loads_ok   -- valid (if degenerate) input the loader accepts; cluster-count adequacy is a
                downstream `few_clusters` concern, not an ingestion rejection.

A separate representative subset also spawns the runner as a subprocess to prove the *exit code* is
non-zero on failure -- a uniform CLI property (unhandled contract error -> `SystemExit`), so it is
checked on a spread of failure modes rather than re-paying interpreter startup for every case.

See docs/ADVERSARIAL.md for the prose catalogue of every case.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from predval.errors import CohortSpecError, PredictionsError
from predval.io import check_roster, load_cohort, load_predictions
from predval.metrics import boundary_count

from . import make_corruptions as mc
from ._runner import run_pipeline

_ERRORS = {"PredictionsError": PredictionsError, "CohortSpecError": CohortSpecError}

#: Any of these appearing in the out dir after a failure is a breach of the door.
ARTEFACTS = (
    "report.html",
    "findings.json",
    "metrics.parquet",
    "fragility.parquet",
    "coverage.parquet",
    "calibration.parquet",
    "manifest.json",
)


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> dict[str, mc.CaseFiles]:
    """Generate every adversarial case once per test session."""
    dest = tmp_path_factory.mktemp("adversarial_cases")
    return mc.build_all(dest)


def _load_target(spec: mc.CaseSpec, files: mc.CaseFiles):
    if spec.target == "predictions":
        return load_predictions(files.predictions)
    return load_cohort(files.cohort_yaml)


@pytest.mark.parametrize("spec", mc.CASE_SPECS, ids=lambda s: s.id)
def test_case(spec: mc.CaseSpec, built: dict[str, mc.CaseFiles], tmp_path) -> None:
    files = built[spec.id]

    if spec.assert_kind == "raises":
        err = _ERRORS[spec.error]
        # (i) the loader raises the right error with a message that matches the case.
        with pytest.raises(err, match=spec.match) as excinfo:
            _load_target(spec, files)
        # message names the offending file, so a human knows what to open.
        assert files.predictions.name in str(excinfo.value) or "cohort" in str(excinfo.value)

        # (ii) running the full pipeline writes NO artefact -- the raise precedes `evaluate`.
        out = tmp_path / "out"
        with pytest.raises((PredictionsError, CohortSpecError)):
            run_pipeline(files.case_dir, out)
        leaked = [a for a in ARTEFACTS if (out / a).exists()]
        assert leaked == [], f"{spec.id}: artefacts leaked on a failure path: {leaked}"

    elif spec.assert_kind == "drops":
        cohort = load_cohort(files.cohort_yaml)
        assert cohort.dropped_subjects, f"{spec.id}: expected drop_and_report to drop >=1 subject"

    elif spec.assert_kind == "roster_undeclared":
        preds = load_predictions(files.predictions)
        cohort = load_cohort(files.cohort_yaml)
        flags = check_roster(cohort, preds)
        codes = {f.code for f in flags}
        assert "present_but_undeclared" in codes, f"{spec.id}: flags were {codes}"

    elif spec.assert_kind == "roster_absent":
        preds = load_predictions(files.predictions)
        cohort = load_cohort(files.cohort_yaml)
        flags = check_roster(cohort, preds)
        codes = {f.code for f in flags}
        assert "declared_but_absent" in codes, f"{spec.id}: flags were {codes}"

    elif spec.assert_kind == "boundary":
        # Locks the eps-clip design: exact 0/1 are legal at the loader and counted downstream.
        preds = load_predictions(files.predictions)
        values = preds.frame["predicted"].to_numpy(dtype=float)
        assert set(np.unique(values)).issubset({0.0, 1.0})
        assert boundary_count(values) == len(values) > 0

    elif spec.assert_kind == "loads_pred":
        preds = load_predictions(files.predictions)
        assert preds.n_rows > 0

    elif spec.assert_kind == "loads_cohort":
        cohort = load_cohort(files.cohort_yaml)
        assert cohort.n_subjects > 0

    else:  # pragma: no cover - guards against an unhandled assert_kind
        raise AssertionError(f"unknown assert_kind {spec.assert_kind!r} for case {spec.id}")


# A spread of failure modes: predictions-load, cohort-load, bad-encoding, no-rows, unknown-column.
_EXIT_CODE_REPRESENTATIVES = [
    "na_prediction",
    "outcome_third_value",
    "na_cluster",
    "enc_utf16",
    "header_only",
    "extra_trailing_cols",
]


@pytest.mark.parametrize("case_id", _EXIT_CODE_REPRESENTATIVES)
def test_runner_exits_nonzero_and_writes_nothing(
    case_id: str, built: dict[str, mc.CaseFiles], tmp_path
) -> None:
    """The CLI runner exits non-zero on a corrupted input and leaves the out dir artefact-free."""
    files = built[case_id]
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "tests.adversarial._runner", str(files.case_dir), str(out)],
        cwd=mc.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, f"{case_id}: expected non-zero exit\nstdout={proc.stdout}"
    if out.exists():
        leaked = [a for a in ARTEFACTS if (out / a).exists()]
        assert leaked == [], f"{case_id}: artefacts leaked on a failure path: {leaked}"
