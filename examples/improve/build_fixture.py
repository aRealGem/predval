#!/usr/bin/env python
"""Build the IMPROVE neoantigen-immunogenicity fixture for predval.

Source: github.com/SRHgroup/IMPROVE_paper (Borch et al. 2024, Front Immunol), results.zip,
the 5-fold cross-validation out-of-fold prediction tables plus the NNAlign comparator.

    uv run python examples/improve/build_fixture.py --src <extracted results dir>

Writes cohort.parquet and predictions.parquet next to cohort.yaml. No network calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# model_id -> (subdirectory, filename) under the extracted `results copy/5_fold_CV/` tree.
RF_MODELS = {
    "improve_simple": ("Simple", "pred_df_Simple.txt"),
    "improve_simple_wo_prime": ("Simple", "pred_df_wo_primeSimple.txt"),
    "improve_tme_excl": ("TME_excluded", "pred_df_TME_excluded.txt"),
    "improve_tme_excl_wo_prime": ("TME_excluded", "pred_df_wo_primeTME_excluded.txt"),
    "improve_tme_incl": ("TME_included", "pred_df_TME_included.txt"),
    "improve_tme_incl_wo_prime": ("TME_included", "pred_df_wo_primeTME_included.txt"),
}


def _sid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["Patient"].astype(str)
        + "|"
        + frame["Mut_peptide"].astype(str)
        + "|"
        + frame["HLA_allele"].astype(str)
    )


def load_rf(src: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for model_id, (sub, name) in RF_MODELS.items():
        path = src / "5_fold_CV" / sub / name
        frame = pd.read_csv(path, sep=r"\s+")
        frame["subject_id"] = _sid(frame)
        out[model_id] = frame
    return out


def load_nnalign(src: Path) -> pd.DataFrame:
    """Parse the NNAlign comparator table.

    The released file's header line crams its first five field names into one tab-delimited cell
    ("Peptide Target Split HLA donor") while the data rows are space-delimited there, so pandas
    reads peptide/target/split/HLA as a four-level index and shifts every named column one to
    the left: `donor` lands in the first named column and `Measure`/`Prediction` are the target
    and the score. Read the identifiers off the index levels rather than trusting the header,
    and drop the four trailing malformed rows (they carry column *names* as values).
    """
    path = src / "NNalign" / "all.test_pred_cmb_ny"
    raw = pd.read_csv(path, sep="\t")
    donor_col = raw.columns[0]
    frame = pd.DataFrame(
        {
            "Mut_peptide": pd.Index(raw.index.get_level_values(0)).astype(str).str.strip(),
            "target": pd.to_numeric(
                pd.Index(raw.index.get_level_values(1)).astype(str), errors="coerce"
            ),
            "HLA_allele": pd.Index(raw.index.get_level_values(3)).astype(str).str.strip(),
            "Patient": raw[donor_col].astype(str).str.strip().to_numpy(),
            "predicted": pd.to_numeric(raw["Prediction"], errors="coerce").to_numpy(),
        }
    )
    frame = frame.dropna(subset=["target", "predicted"]).reset_index(drop=True)
    frame["subject_id"] = _sid(frame)
    return frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="extracted IMPROVE_paper results.zip directory (the 'results copy' folder)",
    )
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(argv)

    rf = load_rf(args.src)
    spine = rf["improve_simple"]

    # One row per (patient, peptide, HLA). The released tables already satisfy this; assert it
    # rather than trusting it, because a silent duplicate would be averaged away downstream.
    if spine["subject_id"].duplicated().any():
        raise SystemExit("spine has duplicate subject_id values")

    pep_len = spine["Mut_peptide"].str.len()
    cohort = pd.DataFrame(
        {
            "subject_id": spine["subject_id"],
            "patient": spine["Patient"].astype(str),
            "tumour_cohort": spine["cohort"].astype(str),
            # Predictors are trained overwhelmingly on 9-mers; off-length behaviour is the
            # prespecified question, so the split is 9 versus everything else.
            "peptide_length": pep_len.map(lambda n: "9mer" if n == 9 else "non_9mer"),
            "cv_partition": spine["Partition"].astype(int),
            "mut_peptide": spine["Mut_peptide"].astype(str),
            "hla_allele": spine["HLA_allele"].astype(str),
            "response": spine["response"].astype(int),
        }
    ).reset_index(drop=True)

    rows = []
    for model_id, frame in rf.items():
        # Outcome must agree across tables; they are six models over one labelled row set.
        merged = frame[["subject_id", "response", "prediction_rf"]].merge(
            cohort[["subject_id", "response"]], on="subject_id", suffixes=("_m", "_c")
        )
        if len(merged) != len(cohort) or (merged["response_m"] != merged["response_c"]).any():
            raise SystemExit(f"{model_id}: row set or labels disagree with the spine")
        rows.append(
            pd.DataFrame(
                {
                    "subject_id": frame["subject_id"],
                    "model_id": model_id,
                    "predicted": frame["prediction_rf"].astype(float),
                }
            )
        )

    nn = load_nnalign(args.src)
    nn = nn[nn["subject_id"].isin(set(cohort["subject_id"]))].copy()
    nn = nn.drop_duplicates(subset="subject_id", keep=False)
    labels = cohort.set_index("subject_id")["response"]
    nn_label = (nn["target"] > 0.5).astype(int).to_numpy()
    if (labels.loc[nn["subject_id"]].to_numpy() != nn_label).any():
        raise SystemExit("nnalign: labels disagree with the spine")
    rows.append(
        pd.DataFrame(
            {
                "subject_id": nn["subject_id"],
                "model_id": "nnalign",
                "predicted": nn["predicted"].clip(0.0, 1.0).astype(float),
            }
        )
    )

    predictions = pd.concat(rows, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(args.out / "cohort.parquet", index=False)
    predictions.to_parquet(args.out / "predictions.parquet", index=False)

    print(f"cohort:      {len(cohort):,} peptides, {cohort['patient'].nunique()} patients")
    print(
        f"outcome:     {int(cohort['response'].sum())} immunogenic "
        f"({cohort['response'].mean():.3%} prevalence)"
    )
    print(f"predictions: {len(predictions):,} rows, {predictions['model_id'].nunique()} models")
    for model_id, grp in predictions.groupby("model_id"):
        print(
            f"  {model_id:28} n={len(grp):6,}  "
            f"coverage={len(grp) / len(cohort):.4f}  "
            f"mean={grp['predicted'].mean():.4f}  "
            f"range=[{grp['predicted'].min():.4f}, {grp['predicted'].max():.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
