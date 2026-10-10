#!/usr/bin/env python
"""Build the IMPROVE third-party predictor panel fixture.

Same 70-patient cohort as examples/improve/, scored by independently published predictors from
other groups instead of by IMPROVE's own random forest. The question this fixture asks is
whether the miscalibration found in examples/improve/ is a property of that one model's
training pipeline or a property of how the field ships probabilities.

    uv run python examples/improve_panel/build_fixture.py --src <IMPROVE_paper checkout>

Writes cohort.parquet and predictions.parquet next to cohort.yaml. No network calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

#: model_id -> (column in the released panel table, divisor to put it on the [0, 1] probability
#: scale). Only predictors whose published output is nominally an immunogenicity PROBABILITY are
#: admitted; see cohort.yaml for the exclusions and the reasoning behind them.
PANEL = {
    "deepimmuno": ("DeepImmuno", 1.0),
    "deephlapan_immuno": ("DeepHLApan immunogenic score", 1.0),
    "deepnetbim_immuno": ("DeepNetBim immunogenicity probability", 1.0),
    "ittca_rf": ("iTTCA-RF probability", 100.0),
}

#: The in-house reference point, carried over from examples/improve/ so the panel is read
#: against a model whose calibration this repo has already characterised.
IMPROVE_REFERENCE = "improve_tme_incl"


def _join_key(frame: pd.DataFrame) -> pd.Series:
    """The only key that joins the panel table to the cross-validation tables 1:1.

    `Mut_peptide` + `HLA_allele` is NOT unique: 167 peptide-HLA pairs recur across patients, and
    joining on them silently produces a many-to-many expansion (17,884 rows from 17,520) with
    apparent label disagreements that are really the cross-product of those duplicates. Adding
    `Expression` disambiguates, because expression is measured per patient. The panel table
    rounds it to three decimals while the CV tables carry full precision, so both sides are
    formatted to three decimals before comparison; at that precision the join is exactly 1:1
    over all 17,520 rows with zero label disagreements, which `main` asserts rather than trusts.
    """
    return (
        frame["Mut_peptide"].astype(str)
        + "|"
        + frame["HLA_allele"].astype(str)
        + "|"
        + frame["Expression"].map(lambda v: f"{float(v):.3f}")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Two explicit file paths rather than one archive root: the two tables come from two
    # different zips in the release, and where each lands depends on how they were extracted.
    ap.add_argument(
        "--panel-tsv",
        type=Path,
        required=True,
        help="data.zip -> data/benchmark_comparison/neoepitopes_predictions.tsv",
    )
    ap.add_argument(
        "--cv-txt",
        type=Path,
        required=True,
        help="results.zip -> 5_fold_CV/TME_included/pred_df_TME_included.txt",
    )
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(argv)

    panel = pd.read_csv(args.panel_tsv, sep="\t")
    cv = pd.read_csv(args.cv_txt, sep=r"\s+")

    panel["join_key"] = _join_key(panel)
    cv["join_key"] = _join_key(cv)
    if not panel["join_key"].is_unique or not cv["join_key"].is_unique:
        raise SystemExit("join key is not unique on both sides; refusing to guess the pairing")

    merged = panel.merge(
        cv[["join_key", "Patient", "cohort", "Partition", "response", "prediction_rf"]],
        on="join_key",
        how="inner",
        validate="1:1",
        suffixes=("_panel", ""),
    )
    if len(merged) != len(cv):
        raise SystemExit(f"join dropped rows: {len(merged)} of {len(cv)}")
    if (merged["response_panel"] != merged["response"]).any():
        raise SystemExit("labels disagree between the panel table and the CV table")

    merged["subject_id"] = (
        merged["Patient"].astype(str)
        + "|"
        + merged["Mut_peptide"].astype(str)
        + "|"
        + merged["HLA_allele"].astype(str)
        + "|"
        + merged["Expression"].map(lambda v: f"{float(v):.3f}")
    )
    if not merged["subject_id"].is_unique:
        raise SystemExit("subject_id is not unique")

    pep_len = merged["Mut_peptide"].str.len()
    cohort = pd.DataFrame(
        {
            "subject_id": merged["subject_id"],
            "patient": merged["Patient"].astype(str),
            "tumour_cohort": merged["cohort"].astype(str),
            "peptide_length": pep_len.map(lambda n: "9mer" if n == 9 else "non_9mer"),
            "cv_partition": merged["Partition"].astype(int),
            "mut_peptide": merged["Mut_peptide"].astype(str),
            "hla_allele": merged["HLA_allele"].astype(str),
            "response": merged["response"].astype(int),
        }
    ).reset_index(drop=True)

    rows = [
        pd.DataFrame(
            {
                "subject_id": merged["subject_id"],
                "model_id": IMPROVE_REFERENCE,
                "predicted": merged["prediction_rf"].astype(float),
            }
        )
    ]
    for model_id, (column, divisor) in PANEL.items():
        score = pd.to_numeric(merged[column], errors="coerce") / divisor
        keep = score.notna()
        out_of_range = keep & ((score < 0.0) | (score > 1.0))
        if out_of_range.any():
            raise SystemExit(
                f"{model_id}: {int(out_of_range.sum())} scores outside [0, 1] after scaling "
                f"by {divisor}; the published scale is not what this mapping assumes"
            )
        # A model that did not score a peptide gets NO ROW. predval measures coverage against
        # the declared floor and refuses to read "no opinion" as "confidently negative".
        rows.append(
            pd.DataFrame(
                {
                    "subject_id": merged.loc[keep, "subject_id"],
                    "model_id": model_id,
                    "predicted": score[keep].astype(float),
                }
            )
        )

    predictions = pd.concat(rows, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(args.out / "cohort.parquet", index=False)
    predictions.to_parquet(args.out / "predictions.parquet", index=False)

    prevalence = float(cohort["response"].mean())
    print(f"cohort:      {len(cohort):,} peptides, {cohort['patient'].nunique()} patients")
    print(f"outcome:     {int(cohort['response'].sum())} immunogenic ({prevalence:.3%} prevalence)")
    print(f"predictions: {len(predictions):,} rows, {predictions['model_id'].nunique()} models")
    for model_id, grp in predictions.groupby("model_id"):
        mean_p = float(grp["predicted"].mean())
        print(
            f"  {model_id:22} n={len(grp):6,}  coverage={len(grp) / len(cohort):.4f}  "
            f"mean={mean_p:.4f}  ratio={mean_p / prevalence:5.1f}x  "
            f"range=[{grp['predicted'].min():.4f}, {grp['predicted'].max():.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
