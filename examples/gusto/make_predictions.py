#!/usr/bin/env python
"""Turn the raw GUSTO-I table into predval's two contract files, and nothing else.

    uv run --group examples python examples/gusto/make_predictions.py [--out DIR]

This is the ONLY place in the GUSTO example that touches raw GUSTO column names, fits anything,
or applies a published coefficient. predval's core never sees any of that -- it only ever opens
the two parquet files this script writes, exactly like PCam's build_fixture.py boundary.

Path decision (logged, not asserted): GUSTO-I's raw table carries `regl`, an int-coded 16-level
region variable. `regl == 1` has exactly 2188 rows. That number is independently corroborated in
the external-validation methodology literature as the size of the well-known GUSTO-I "West"
teaching subset (Harrell/Steyerberg), and the session brief independently cited the same n=2188
before this script ever ran. No literal "1 = West" codebook label ships with the .rda (pyreadr
carries no attrs on this object, confirmed empirically), so this mapping rests on convergent
evidence -- row-count match across independent sources plus internal consistency -- not a quoted
codebook entry. That evidentiary basis is stated here plainly rather than overclaimed.

Predictor semantics were verified empirically against OTHER raw columns in this same dataset
before being trusted, because a third-party mirror's auto-generated description of this dataset
disagreed with the standard literature names (claiming `hrt` = "hormone replacement therapy" and
`hig` = "high blood pressure" -- implausible for a 1990s all-comers AMI trial's core risk model,
and contradicted by direct crosstab against `pulse`/`sysbp`/`ant`/`pmi` in this script's own
inspection pass, see docs/DELTA-gusto.md). GUSTO-I already ships five of the eight standard
predictors pre-derived exactly (sho=shock, hig=high-risk [anterior or prior MI], dia=diabetes,
hyp=hypotension, hrt=tachycardia, ttr=time-to-relief>1h, all confirmed >99.9% consistent with
their defining raw columns); only age>65 and female need deriving here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_data import DEFAULT_CACHE, fetch_raw  # noqa: E402

HERE = Path(__file__).resolve().parent

#: The eight predictors, in the order Steyerberg's teaching model reports them.
PREDICTORS = ["age65", "female", "dia", "hyp", "hrt", "hig", "sho", "ttr"]

WEST_REGION_CODE = 1
WEST_N_EXPECTED = 2188


class PathDecisionError(RuntimeError):
    """The raw table doesn't look like what this script was written to handle."""


def detect_path(raw: pd.DataFrame) -> Literal["A", "B"]:
    """Path A if a region/site column resolves to an unambiguous West subset; else Path B."""
    candidates = [c for c in raw.columns if c.lower() in ("regl", "region", "site", "regn")]
    print(f"region-column candidates checked: {candidates}")
    if not candidates:
        print("no region-like column found -> Path B")
        return "B"

    col = candidates[0]
    counts = raw[col].value_counts()
    print(f"'{col}' value counts:\n{counts.sort_index()}")

    if WEST_REGION_CODE not in counts.index:
        print(f"'{col}'=={WEST_REGION_CODE} not present -> Path B")
        return "B"

    n_west = int(counts.loc[WEST_REGION_CODE])
    print(
        f"'{col}'=={WEST_REGION_CODE} has n={n_west}; expected West subset n={WEST_N_EXPECTED} "
        f"(independently corroborated in the literature and the session brief, not a codebook "
        f"label -- see this module's docstring)"
    )
    if n_west != WEST_N_EXPECTED:
        print("row count does not match the expected West subset -> Path B")
        return "B"

    print(f"-> Path A, region column '{col}', West = {col}=={WEST_REGION_CODE}")
    return "A"


def _derive_predictors(raw: pd.DataFrame) -> pd.DataFrame:
    """The only two predictors GUSTO-I doesn't already ship pre-derived."""
    out = raw[["sho", "hig", "dia", "hyp", "hrt", "ttr"]].copy()
    out["age65"] = (raw["age"] > 65).astype(int)
    out["female"] = (raw["sex"] == "female").astype(int)
    return out[PREDICTORS]


def build_path_a(raw: pd.DataFrame, region_col: str = "regl") -> tuple[pd.DataFrame, pd.DataFrame]:
    x_all = _derive_predictors(raw)
    y_all = raw["day30"].astype(int)
    west_mask = raw[region_col] == WEST_REGION_CODE

    x_west = sm.add_constant(x_all[west_mask])
    model = sm.Logit(y_all[west_mask], x_west).fit(disp=0)
    print(model.summary())

    non_west = raw[~west_mask].reset_index(drop=True)
    x_non_west = sm.add_constant(x_all[~west_mask].reset_index(drop=True), has_constant="add")
    predicted = model.predict(x_non_west)

    subject_id = pd.Series([f"gusto_{i:06d}" for i in non_west.index], name="subject_id")
    cohort = pd.DataFrame(
        {
            "subject_id": subject_id,
            "label": non_west["day30"].astype(int).to_numpy(),
            region_col: non_west[region_col].astype(int).to_numpy(),
        }
    )
    predictions = pd.DataFrame(
        {
            "subject_id": subject_id,
            "model_id": "gusto_west_refit_logistic",
            "fold": pd.NA,
            "horizon": 30.0,
            "predicted": predicted.clip(0.0, 1.0).to_numpy(),
        }
    )
    return cohort, predictions


def build_path_b(
    raw: pd.DataFrame, region_col_if_present: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Not implemented: this session's detect_path() returned 'A', so Path B never ran.

    Deliberately left with no placeholder coefficients -- a stub with invented-looking numbers
    risks being mistaken for real transcribed values later. Implementing this requires sourcing
    genuine published coefficients from a citable edition + page (e.g. Steyerberg, "Clinical
    Prediction Models") before writing anything here, not fabricating plausible-looking ones.
    """
    raise NotImplementedError(
        "Path B was not exercised this session -- detect_path() returned 'A'. Implementing it "
        "requires transcribing real, cited published coefficients first; do not fabricate."
    )


def write_outputs(cohort: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> None:
    assert predictions["predicted"].between(0.0, 1.0).all(), "predicted outside [0,1]"
    assert predictions["subject_id"].is_unique, "duplicate subject_id in predictions"
    assert cohort["subject_id"].is_unique, "duplicate subject_id in cohort"
    assert not predictions["predicted"].isna().any(), "NaN predicted"

    out_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out_dir / "cohort.parquet", index=False)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    print(f"wrote {out_dir / 'cohort.parquet'}  ({len(cohort):,} rows)")
    print(f"wrote {out_dir / 'predictions.parquet'}  ({len(predictions):,} rows)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(argv)

    raw = fetch_raw(args.cache)
    path = detect_path(raw)

    if path == "A":
        cohort, predictions = build_path_a(raw)
    else:
        cohort, predictions = build_path_b(raw, None)

    write_outputs(cohort, predictions, args.out)

    print()
    print(f"path taken: {path}")
    print(f"cohort n: {len(cohort):,}, events: {int(cohort['label'].sum()):,}")
    quick_auc = None
    try:
        from sklearn.metrics import roc_auc_score

        quick_auc = roc_auc_score(cohort["label"], predictions["predicted"])
    except Exception:
        pass
    if quick_auc is not None:
        print(
            f"quick (non-authoritative) AUROC: {quick_auc:.4f} -- "
            "predval's own evaluate() is the real number"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
