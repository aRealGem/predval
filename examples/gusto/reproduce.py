#!/usr/bin/env python
"""One-command reproduction of the GUSTO-I example.

    uv run --group examples python examples/gusto/reproduce.py [--cache DIR] [--out DIR]

Mirrors examples/pcam/reproduce.py's shape, with one real difference stated up front: **there is
no golden baseline for GUSTO yet** (explicit backlog per the session that added this example --
see docs/DELTA-gusto.md). This script fetches-or-caches, builds predictions, runs the evaluation,
and prints the headline numbers -- it does not check them against a frozen expectation. Anyone
wanting a determinism check should run this script twice and hash-compare the two `out/`
directories, the same way this example's own session verified byte-identity.

`examples/gusto/cohort.parquet` and `predictions.parquet` are not committed for the same reason
PCam's aren't: they are generated derivatives (this script's own output), not sensitive data --
GUSTO-I is a public trial dataset. See .gitignore and docs/DELTA-gusto.md.

Steps:
  1. Fetch (or reuse a cached) GUSTO-I raw table; sha256 is printed either way.
  2. Build predictions.parquet + cohort.parquet via the West-fit / non-West-scored logistic
     (make_predictions.py), or fail with a clear message if fetching fails.
  3. Run the full evaluation (B=2000, seed 1337) and print the headline metrics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "raw_cache"


def build_predictions(cache: Path, out: Path) -> None:
    print(f"building predictions -> {out}")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "make_predictions.py"),
            "--cache",
            str(cache),
            "--out",
            str(out),
        ],
        check=True,
    )


def run_evaluation(out: Path) -> None:
    print(f"running full evaluation (B=2000) -> {out / 'out'}")
    subprocess.run(
        [sys.executable, str(HERE / "run_evaluation.py"), "--out", str(out / "out")],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(argv)

    try:
        build_predictions(args.cache, args.out)
    except subprocess.CalledProcessError as exc:
        print(
            f"fixture build failed ({exc}); see docs/DELTA-gusto.md for the documented fallbacks "
            "(this script does not chase beyond the primary source and the two named mirrors).",
            file=sys.stderr,
        )
        return 2

    run_evaluation(args.out)

    print()
    print("done. No golden baseline exists for GUSTO yet -- see this script's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
