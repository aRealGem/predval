"""Thin end-to-end runner for the adversarial suite: load -> evaluate -> write artefacts.

The whole point of this module is the *failure path*: on any ingestion/validation error it raises
(and, run as a script, exits non-zero) BEFORE `evaluate` is ever reached, so no report, findings,
or metrics file can be written from a corrupted input. On a clean input it runs the full pipeline
at a small bootstrap count and writes the usual artefacts.

    python -m tests.adversarial._runner <case_dir> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from predval import evaluate, load_cohort, load_predictions, write_findings, write_report


def run_pipeline(case_dir: str | Path, out_dir: str | Path, *, n_boot: int = 50) -> Path:
    """Load, evaluate, and write artefacts. Raises on any contract violation before writing."""
    case_dir = Path(case_dir)
    out_dir = Path(out_dir)

    cohort = load_cohort(case_dir / "cohort.yaml")
    predictions = load_predictions(case_dir / "predictions.csv")

    # Keep the clean-input path fast; the corrupted paths never get here.
    unc = cohort.spec.uncertainty.model_copy(update={"n_boot": n_boot})
    spec = cohort.spec.model_copy(update={"uncertainty": unc})
    cohort = type(cohort)(**{**cohort.__dict__, "spec": spec})

    result = evaluate(cohort, predictions)
    result.write(out_dir)
    write_report(result, out_dir / "report.html")
    write_findings(result, out_dir / "findings.json")
    return out_dir


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m tests.adversarial._runner <case_dir> <out_dir>", file=sys.stderr)
        return 2
    run_pipeline(argv[0], argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
