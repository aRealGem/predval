#!/usr/bin/env python
"""Fetch the GUSTO-I public cohort and log its raw shape.

    uv run --group examples python examples/gusto/fetch_data.py [--cache DIR]

Downloads the pinned source, caches it (gitignored -- an external fetched input, not a generated
derivative, same reasoning as PCam's campaign checkout), records its sha256, and logs the variable
list verbatim. This script makes NO decision about how to build predictions or a cohort table --
that split is make_predictions.py's job, so predval's core never has to know this data started as
an .rda file at all.

If the primary URL fails, this script does NOT scrape or guess a replacement -- it prints the two
documented fallbacks (the hbiostat.org/data index, and the Kaggle mirror) and stops. Chasing beyond
those is out of scope for this session.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import pyreadr

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "raw_cache"

PRIMARY_URL = "https://hbiostat.org/data/repo/gusto.rda"
FALLBACK_INDEX = "https://hbiostat.org/data"
FALLBACK_KAGGLE = "https://www.kaggle.com/c/gusto/data"


class FetchError(RuntimeError):
    """The pinned source could not be retrieved; no attempt was made to guess a replacement."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fetch_raw(cache_dir: Path) -> pd.DataFrame:
    """Return the GUSTO-I table, downloading and caching it if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rda_path = cache_dir / "gusto.rda"

    if not rda_path.exists():
        print(f"fetching {PRIMARY_URL}")
        try:
            urllib.request.urlretrieve(PRIMARY_URL, rda_path)
        except Exception as exc:
            if rda_path.exists():
                rda_path.unlink()
            raise FetchError(
                f"primary source failed: {exc}\n"
                f"documented fallbacks (not chased automatically):\n"
                f"  index: {FALLBACK_INDEX}\n"
                f"  kaggle mirror: {FALLBACK_KAGGLE}"
            ) from exc

    digest = _sha256(rda_path)
    print(f"cached at {rda_path}")
    print(f"sha256: {digest}")

    result = pyreadr.read_r(str(rda_path))
    if len(result) != 1:
        raise FetchError(f"expected exactly one object in {rda_path}, found {list(result.keys())}")
    (df,) = result.values()
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args(argv)

    df = fetch_raw(args.cache)

    print()
    print(f"rows: {len(df):,}")
    print(f"columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col:20} dtype={df[col].dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
