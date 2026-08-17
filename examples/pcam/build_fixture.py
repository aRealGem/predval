#!/usr/bin/env python
"""Build the PCam example fixture from the histopath campaign's holdout predictions.

READ-ONLY over the campaign directory. This script opens files under CAMPAIGN for reading and
writes only into its own directory. It never modifies, moves, or creates anything in the
campaign repo.

What it produces
----------------
cohort.parquet       19,999 rows: subject_id, label, wsi, domain
predictions.parquet  15 models x 19,999 rows: subject_id, model_id, fold, horizon, predicted

Why this fixture is worth having
--------------------------------
It is not synthetic. The 19,999 subjects are image patches drawn from only 22 whole slides, so
the clustering problem is real. Two of the fifteen models are badly miscalibrated in ways a
recalibration ladder should detect. And one member of the published 11-model ensemble --
`p4m_reg` -- had its predictions permanently lost, giving us a genuine coverage gap rather than
one manufactured by deleting rows.

Usage
-----
    uv run python examples/pcam/build_fixture.py [--campaign PATH]
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN = Path.home() / "histopath-cancer-detection"

#: Members whose holdout file sits in autoloop/members/.
MEMBERS_DIR_MODELS = (
    "champion",
    "e2cnn",
    "e2cnn_s21",
    "e2cnn_s99",
    "effnet_scratch",
    "macenko",
    "p4m_dense",
    "p4m_s13",
    "p4m_seed7",
    "p4m_svm",
    "phikon",
    "swin",
    "tinyvgg",
    "tinyvgg_vl",
)

#: `state.json` points at members/oof_p4m_reg_vl.csv, which does not exist; the real file is in
#: inbox/. Verified identical id set and labels, and AUROC 0.985116 matching the ledger.
INBOX_MODELS = ("p4m_reg_vl",)

#: Published champion member whose holdout predictions are permanently lost -- the `.keras`
#: weights lived in ephemeral Colab storage, so re-inference is impossible. It gets no rows.
#: This is the fixture's built-in test of the absence rule (docs/spec.md section 1.1).
PERMANENTLY_LOST = ("p4m_reg",)

EXPECTED_SUBJECTS = 19_999
EXPECTED_POSITIVES = 5_796
EXPECTED_SLIDES = 22
EXPECTED_MODELS = len(MEMBERS_DIR_MODELS) + len(INBOX_MODELS)


class FixtureError(RuntimeError):
    """The campaign data does not look the way Session 0 verified it looked."""


def _read_oof(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FixtureError(f"expected holdout file is missing: {path}")
    df = pd.read_csv(path, dtype={"id": str})
    expected_cols = {"id", "label", "pred"}
    if set(df.columns) != expected_cols:
        raise FixtureError(
            f"{path} has columns {sorted(df.columns)}, expected {sorted(expected_cols)}"
        )
    return df


def build(campaign: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = campaign / "autoloop" / "members"
    inbox = campaign / "autoloop" / "inbox"
    wsi_map_path = campaign / "data" / "wsi" / "patch_id_wsi_full.csv.gz"
    domain_path = campaign / "data" / "wsi" / "wsi_domain_k2.csv"

    sources: dict[str, Path] = {m: members / f"oof_{m}.csv" for m in MEMBERS_DIR_MODELS}
    sources.update({m: inbox / f"oof_{m}.csv" for m in INBOX_MODELS})

    frames: dict[str, pd.DataFrame] = {}
    for model, path in sources.items():
        frames[model] = _read_oof(path)

    # --- invariant: every model scored exactly the same subjects, with the same labels -------
    reference = frames["champion"].sort_values("id").reset_index(drop=True)
    ref_ids = set(reference["id"])
    if len(reference) != EXPECTED_SUBJECTS:
        raise FixtureError(f"expected {EXPECTED_SUBJECTS} holdout rows, got {len(reference)}")
    if len(ref_ids) != EXPECTED_SUBJECTS:
        raise FixtureError("holdout ids are not unique")

    for model, df in frames.items():
        if set(df["id"]) != ref_ids:
            raise FixtureError(f"model {model!r} scores a different subject set than 'champion'")
        aligned = df.sort_values("id").reset_index(drop=True)
        if not aligned["label"].equals(reference["label"]):
            raise FixtureError(f"model {model!r} disagrees with 'champion' about the labels")
        frames[model] = aligned

    n_pos = int(reference["label"].sum())
    if n_pos != EXPECTED_POSITIVES:
        raise FixtureError(f"expected {EXPECTED_POSITIVES} positives, got {n_pos}")

    # --- cohort table: outcome + clustering unit + prespecified subgroup ---------------------
    with gzip.open(wsi_map_path, "rt") as fh:
        wsi_map = pd.read_csv(fh, dtype=str)
    domain = pd.read_csv(domain_path, dtype={"wsi": str, "domain": str})

    cohort = reference.rename(columns={"id": "subject_id"})[["subject_id", "label"]]
    cohort = cohort.merge(wsi_map.rename(columns={"id": "subject_id"}), on="subject_id", how="left")
    if cohort["wsi"].isna().any():
        raise FixtureError(f"{int(cohort['wsi'].isna().sum())} subjects have no slide mapping")
    cohort = cohort.merge(domain, on="wsi", how="left")
    if cohort["domain"].isna().any():
        raise FixtureError(f"{int(cohort['domain'].isna().sum())} subjects have no domain label")

    n_slides = cohort["wsi"].nunique()
    if n_slides != EXPECTED_SLIDES:
        raise FixtureError(f"expected {EXPECTED_SLIDES} holdout slides, got {n_slides}")

    # --- predictions: long format, one row per (subject, model) ------------------------------
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "subject_id": df["id"].astype(str),
                    "model_id": model,
                    "fold": pd.NA,  # single fixed holdout; null == holdout
                    "horizon": pd.NA,  # binary outcome
                    "predicted": df["pred"].astype(float),
                }
            )
            for model, df in sorted(frames.items())
        ],
        ignore_index=True,
    )
    predictions["fold"] = predictions["fold"].astype("string")
    predictions["horizon"] = predictions["horizon"].astype("Float64").astype("float64")

    if predictions["model_id"].nunique() != EXPECTED_MODELS:
        raise FixtureError(
            f"expected {EXPECTED_MODELS} models, got {predictions['model_id'].nunique()}"
        )
    if len(predictions) != EXPECTED_MODELS * EXPECTED_SUBJECTS:
        raise FixtureError(f"expected {EXPECTED_MODELS * EXPECTED_SUBJECTS} rows")
    for lost in PERMANENTLY_LOST:
        if lost in set(predictions["model_id"]):
            raise FixtureError(f"{lost!r} has no surviving predictions and must not appear")

    lo, hi = predictions["predicted"].min(), predictions["predicted"].max()
    if lo < 0.0 or hi > 1.0:
        raise FixtureError(f"predictions outside [0, 1]: min={lo}, max={hi}")

    return cohort, predictions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--campaign",
        type=Path,
        default=DEFAULT_CAMPAIGN,
        help=f"campaign root, read-only (default: {DEFAULT_CAMPAIGN})",
    )
    args = ap.parse_args(argv)

    if not args.campaign.exists():
        print(
            f"campaign directory not found: {args.campaign}\n"
            "This fixture is built from a local histopath campaign; pass --campaign to point "
            "at it.",
            file=sys.stderr,
        )
        return 2

    cohort, predictions = build(args.campaign)

    cohort_out = HERE / "cohort.parquet"
    pred_out = HERE / "predictions.parquet"
    cohort.to_parquet(cohort_out, index=False)
    predictions.to_parquet(pred_out, index=False)

    n_exact = int(((predictions["predicted"] <= 0.0) | (predictions["predicted"] >= 1.0)).sum())
    print(f"wrote {cohort_out}")
    print(
        f"  {len(cohort):,} subjects | {int(cohort['label'].sum()):,} positives "
        f"| {cohort['wsi'].nunique()} slides | {cohort['domain'].nunique()} domains"
    )
    print(f"wrote {pred_out}")
    print(f"  {len(predictions):,} rows | {predictions['model_id'].nunique()} models")
    print(f"  {n_exact} predictions sit exactly at 0.0 or 1.0 (the ladder's eps-clip matters)")
    print(f"  absent by design: {', '.join(PERMANENTLY_LOST)} (predictions permanently lost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
