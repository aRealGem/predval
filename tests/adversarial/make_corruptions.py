"""Deterministic corruption generator for predval's ingestion door.

Reads the clean GUSTO-I inputs (`examples/gusto/predictions.parquet` + `cohort.parquet`) if they
are present, otherwise synthesises a GUSTO-shaped base with identical schema and semantics
(single logistic model, region clustering, binary 30-day-mortality outcome). Either way it takes a
small deterministic slice so the adversarial suite stays fast and runs in a clean checkout where
the git-ignored GUSTO derivatives are absent.

For every case in `CASE_SPECS` it writes a self-contained case directory containing a `cohort.yaml`,
its cohort data file, and a predictions file -- exactly ONE of which is corrupted, the rest clean --
so each case can be exercised either at the loader level or end to end through the runner.

pandas + stdlib only, no new dependencies, no randomness that varies between runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GUSTO_DIR = REPO_ROOT / "examples" / "gusto"

MODEL_ID = "gusto_west_refit_logistic"
#: A deterministic slice size for the base; large enough for both outcome classes and >=2 regions.
BASE_N = 400


# --------------------------------------------------------------------------- base construction


def _synthetic_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A GUSTO-shaped base built without the real files (deterministic, no RNG)."""
    n = BASE_N
    idx = np.arange(n)
    subject_id = [f"gusto_{i:06d}" for i in idx]
    # ~15% prevalence, both classes; 15 regions cycled; all deterministic.
    label = ((idx % 7) == 0).astype(int)
    regl = (idx % 15) + 1
    predicted = 0.02 + 0.9 * ((idx * 2654435761) % 1000) / 1000.0  # spread across (0,1), no 0/1
    cohort = pd.DataFrame({"subject_id": subject_id, "label": label, "regl": regl})
    preds = pd.DataFrame({"subject_id": subject_id, "model_id": MODEL_ID, "predicted": predicted})
    return preds, cohort


def load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The clean (predictions, cohort) base, sliced deterministically.

    Uses the real GUSTO inputs when available; the slice preserves both outcome classes and
    multiple regions so the base itself always loads cleanly.
    """
    pred_path = GUSTO_DIR / "predictions.parquet"
    cohort_path = GUSTO_DIR / "cohort.parquet"
    if not (pred_path.exists() and cohort_path.exists()):
        return _synthetic_base()

    cohort = pd.read_parquet(cohort_path).sort_values("subject_id").reset_index(drop=True)
    step = max(1, len(cohort) // BASE_N)
    cohort = cohort.iloc[::step].reset_index(drop=True)
    keep = set(cohort["subject_id"])
    preds = pd.read_parquet(pred_path)
    preds = preds[preds["subject_id"].isin(keep)].reset_index(drop=True)
    # Reduce to the columns the loader requires; fold/horizon are optional and normalised anyway.
    preds = preds[["subject_id", "model_id", "predicted"]].astype(
        {"subject_id": str, "model_id": str, "predicted": float}
    )
    cohort = cohort[["subject_id", "label", "regl"]].copy()
    cohort["subject_id"] = cohort["subject_id"].astype(str)
    return preds, cohort


# --------------------------------------------------------------------------- yaml + write helpers


def _base_spec(
    data_name: str = "cohort.parquet", *, expected_extra: list[str] | None = None
) -> dict:
    """A GUSTO-shaped cohort.yaml as a plain dict."""
    expected = [MODEL_ID] + (expected_extra or [])
    return {
        "cohort_id": "gusto-adversarial",
        "version": 0,
        "subject_key": "subject_id",
        "data": data_name,
        "outcome": {"type": "binary", "field": "label", "positive_label": 1},
        "clustering": {"field": "regl"},
        "coverage": {"min_fraction": 0.95, "compare_on": "both"},
        "completeness": {"require_outcome": True, "on_violation": "drop_and_report"},
        "thresholds": [0.5],
        "expected_models": expected,
        "on_missing_model": "warn",
    }


def _write_yaml(case_dir: Path, spec: dict) -> None:
    (case_dir / "cohort.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))


def _write_clean_cohort(case_dir: Path, cohort: pd.DataFrame, spec: dict | None = None) -> None:
    cohort.to_parquet(case_dir / "cohort.parquet", index=False)
    _write_yaml(case_dir, spec or _base_spec())


def _write_clean_predictions_csv(case_dir: Path, preds: pd.DataFrame) -> None:
    preds.to_csv(case_dir / "predictions.csv", index=False)


def _pred_csv_text(preds: pd.DataFrame) -> str:
    return preds.to_csv(index=False)


# --------------------------------------------------------------------------- case metadata


@dataclass(frozen=True)
class CaseSpec:
    """Static metadata for one adversarial case (importable at collection time)."""

    id: str
    taxonomy: str  # the a-l bucket from the brief
    target: str  # "predictions" | "cohort"
    category: str  # "hard_fail" | "warn_or_drop" | "loads_ok"
    assert_kind: str  # dispatch key for the test
    error: str | None = None  # exception class name for hard_fail
    match: str | None = None  # substring/regex expected in the message
    note: str = ""


@dataclass(frozen=True)
class CaseFiles:
    """Where a generated case's files live."""

    case_dir: Path
    predictions: Path  # file the loader/runner should read as predictions
    cohort_yaml: Path


CASE_SPECS: list[CaseSpec] = [
    # a. renamed columns
    CaseSpec("renamed_pred_col", "a", "predictions", "hard_fail", "raises",
             "PredictionsError", r"missing required columns"),
    CaseSpec("renamed_outcome_col", "a", "cohort", "hard_fail", "raises",
             "CohortSpecError", r"missing columns declared"),
    # b. NA in prediction / outcome / cluster
    CaseSpec("na_prediction", "b", "predictions", "hard_fail", "raises",
             "PredictionsError", r"nulls or NaN"),
    CaseSpec("na_outcome", "b", "cohort", "warn_or_drop", "drops"),
    CaseSpec("na_cluster", "b", "cohort", "hard_fail", "raises",
             "CohortSpecError", r"clustering.+null|null.+clustering|regl"),
    # c. duplicate subject ids
    CaseSpec("dup_exact", "c", "predictions", "hard_fail", "raises",
             "PredictionsError", r"duplicate"),
    CaseSpec("dup_conflicting", "c", "predictions", "hard_fail", "raises",
             "PredictionsError", r"duplicate"),
    # d. out-of-range predictions
    CaseSpec("pred_negative", "d", "predictions", "hard_fail", "raises",
             "PredictionsError", r"probability in \[0, 1\]"),
    CaseSpec("pred_gt1", "d", "predictions", "hard_fail", "raises",
             "PredictionsError", r"probability in \[0, 1\]"),
    CaseSpec("pred_bulk_zero_one", "d", "predictions", "loads_ok", "boundary"),
    # e. logit-scale
    CaseSpec("pred_logit_scale", "e", "predictions", "hard_fail", "raises",
             "PredictionsError", r"logit"),
    # f. string floats
    CaseSpec("str_comma_decimal", "f", "predictions", "hard_fail", "raises",
             "PredictionsError", r"not numeric"),
    CaseSpec("str_padded", "f", "predictions", "loads_ok", "loads_pred"),
    CaseSpec("str_NULL", "f", "predictions", "hard_fail", "raises",
             "PredictionsError", r"nulls or NaN"),
    CaseSpec("str_NaN", "f", "predictions", "hard_fail", "raises",
             "PredictionsError", r"nulls or NaN"),
    CaseSpec("str_empty_value", "f", "predictions", "hard_fail", "raises",
             "PredictionsError", r"nulls or NaN"),
    # g. encoding damage
    CaseSpec("enc_utf8_bom", "g", "predictions", "loads_ok", "loads_pred"),
    CaseSpec("enc_crlf", "g", "predictions", "loads_ok", "loads_pred"),
    CaseSpec("enc_utf16", "g", "predictions", "hard_fail", "raises",
             "PredictionsError", r"could not be read"),
    # h. outcome pathologies
    CaseSpec("outcome_yes_no", "h", "cohort", "hard_fail", "raises",
             "CohortSpecError", r"never occurs"),
    CaseSpec("outcome_true_false", "h", "cohort", "loads_ok", "loads_cohort"),
    CaseSpec("outcome_float_1_0", "h", "cohort", "loads_ok", "loads_cohort"),
    CaseSpec("outcome_third_value", "h", "cohort", "hard_fail", "raises",
             "CohortSpecError", r"exactly two|distinct"),
    CaseSpec("outcome_all_zero", "h", "cohort", "hard_fail", "raises",
             "CohortSpecError", r"never occurs"),
    CaseSpec("outcome_all_one", "h", "cohort", "hard_fail", "raises",
             "CohortSpecError", r"one distinct|both.+class|non-event"),
    CaseSpec("outcome_one_event", "h", "cohort", "loads_ok", "loads_cohort"),
    # i. cluster pathologies (valid input; adequacy is a downstream few_clusters concern)
    CaseSpec("cluster_g_one", "i", "cohort", "loads_ok", "loads_cohort"),
    CaseSpec("cluster_g_two", "i", "cohort", "loads_ok", "loads_cohort"),
    CaseSpec("cluster_float_ids", "i", "cohort", "loads_ok", "loads_cohort"),
    CaseSpec("cluster_zero_event", "i", "cohort", "loads_ok", "loads_cohort"),
    CaseSpec("cluster_g_equals_n", "i", "cohort", "loads_ok", "loads_cohort"),
    CaseSpec("cluster_constant", "i", "cohort", "loads_ok", "loads_cohort"),
    # j. roster mismatch (loud flags, not load failures)
    CaseSpec("roster_undeclared", "j", "predictions", "warn_or_drop", "roster_undeclared"),
    CaseSpec("roster_declared_absent", "j", "cohort", "warn_or_drop", "roster_absent"),
    # k. Excel damage
    CaseSpec("excel_scientific", "k", "predictions", "loads_ok", "loads_pred"),
    CaseSpec("excel_mangled_id", "k", "predictions", "loads_ok", "loads_pred"),
    CaseSpec("excel_thousands_sep", "k", "predictions", "hard_fail", "raises",
             "PredictionsError", r"not numeric"),
    # l. structural
    CaseSpec("empty_file", "l", "predictions", "hard_fail", "raises",
             "PredictionsError", r"could not be read|no rows"),
    CaseSpec("header_only", "l", "predictions", "hard_fail", "raises",
             "PredictionsError", r"no rows"),
    CaseSpec("truncated_row", "l", "predictions", "hard_fail", "raises",
             "PredictionsError", r"nulls or NaN|not numeric"),
    CaseSpec("extra_trailing_cols", "l", "predictions", "hard_fail", "raises",
             "PredictionsError", r"outside the v0 contract"),
]


# --------------------------------------------------------------------------- corruption builders
#
# Each builder receives (case_dir, base_pred, base_cohort) and writes the case's files, returning
# the predictions filename to hand to the loader/runner. It writes a clean counterpart for whichever
# input it does not corrupt, so every case dir is a runnable cohort+predictions pair.


def _b_renamed_pred_col(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    text = _pred_csv_text(p).replace("predicted", "pred_prob", 1)
    (d / "predictions.csv").write_text(text)
    return "predictions.csv"


def _b_renamed_outcome_col(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_predictions_csv(d, p)
    c2 = c.rename(columns={"label": "outcome_30d"})
    c2.to_parquet(d / "cohort.parquet", index=False)
    _write_yaml(d, _base_spec())  # yaml still declares `label`, which is now absent
    return "predictions.csv"


def _b_na_prediction(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    p2 = p.copy()
    p2.loc[p2.index[1], "predicted"] = np.nan
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _b_na_outcome(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_predictions_csv(d, p)
    c2 = c.copy().astype({"label": "float"})
    c2.loc[c2.index[0], "label"] = np.nan  # drop_and_report should drop exactly this subject
    c2.to_parquet(d / "cohort.parquet", index=False)
    _write_yaml(d, _base_spec())
    return "predictions.csv"


def _b_na_cluster(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_predictions_csv(d, p)
    c2 = c.copy().astype({"regl": "float"})
    c2.loc[c2.index[2], "regl"] = np.nan
    c2.to_parquet(d / "cohort.parquet", index=False)
    _write_yaml(d, _base_spec())
    return "predictions.csv"


def _b_dup_exact(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    p2 = pd.concat([p, p.iloc[[0]]], ignore_index=True)  # byte-identical duplicate row
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _b_dup_conflicting(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    row = p.iloc[[0]].copy()
    row["predicted"] = float(1.0 - float(p.iloc[0]["predicted"]))  # same key, different value
    p2 = pd.concat([p, row], ignore_index=True)
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _b_pred_negative(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    p2 = p.copy()
    p2.loc[p2.index[1], "predicted"] = -0.25
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _b_pred_gt1(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    p2 = p.copy()
    p2.loc[p2.index[1], "predicted"] = 1.4
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _b_pred_bulk_zero_one(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    p2 = p.copy()
    # Half exactly 0, half exactly 1 -- legal, eps-clipped downstream, counted by boundary_count.
    p2["predicted"] = np.where(np.arange(len(p2)) % 2 == 0, 0.0, 1.0)
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _b_pred_logit_scale(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    p2 = p.copy()
    # Map probabilities to log-odds so the column now ranges roughly [-6, 6].
    q = np.clip(p2["predicted"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    p2["predicted"] = np.log(q / (1 - q))
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _corrupt_first_pred_value(d: Path, p: pd.DataFrame, c: pd.DataFrame, token: str) -> str:
    """Write predictions CSV with the first prediction value replaced by a raw string token."""
    _write_clean_cohort(d, c)
    lines = _pred_csv_text(p).splitlines()
    header = lines[0]
    first = lines[1].split(",")
    first[-1] = token
    lines[1] = ",".join(first)
    (d / "predictions.csv").write_text("\n".join([header] + lines[1:]) + "\n")
    return "predictions.csv"


def _b_str_comma_decimal(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _corrupt_first_pred_value(d, p, c, '"0,73"')


def _b_str_padded(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _corrupt_first_pred_value(d, p, c, '" 0.73 "')


def _b_str_NULL(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _corrupt_first_pred_value(d, p, c, "NULL")


def _b_str_NaN(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _corrupt_first_pred_value(d, p, c, "NaN")


def _b_str_empty_value(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _corrupt_first_pred_value(d, p, c, "")


def _b_enc_utf8_bom(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    (d / "predictions.csv").write_bytes(b"\xef\xbb\xbf" + _pred_csv_text(p).encode("utf-8"))
    return "predictions.csv"


def _b_enc_crlf(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    (d / "predictions.csv").write_bytes(_pred_csv_text(p).replace("\n", "\r\n").encode("utf-8"))
    return "predictions.csv"


def _b_enc_utf16(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    (d / "predictions.csv").write_bytes(_pred_csv_text(p).encode("utf-16"))
    return "predictions.csv"


def _write_cohort_labels(d: Path, p: pd.DataFrame, c: pd.DataFrame, labels: list) -> str:
    """Clean predictions + a cohort whose first len(labels) rows carry the given label values."""
    _write_clean_predictions_csv(d, p)
    c2 = c.copy()
    c2 = c2.iloc[: len(labels)].copy()
    c2["label"] = labels
    # keep predictions covering the same subjects
    p2 = p[p["subject_id"].isin(set(c2["subject_id"]))]
    p2.to_csv(d / "predictions.csv", index=False)
    c2.to_parquet(d / "cohort.parquet", index=False)
    _write_yaml(d, _base_spec())
    return "predictions.csv"


def _b_outcome_yes_no(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_labels(d, p, c, ["yes", "no", "yes", "no", "yes", "no"])


def _b_outcome_true_false(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_labels(d, p, c, [True, False, True, False, True, False])


def _b_outcome_float_1_0(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_labels(d, p, c, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def _b_outcome_third_value(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_labels(d, p, c, [1, 0, -9, 1, 0, -9])


def _b_outcome_all_zero(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_labels(d, p, c, [0, 0, 0, 0, 0, 0])


def _b_outcome_all_one(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_labels(d, p, c, [1, 1, 1, 1, 1, 1])


def _b_outcome_one_event(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_labels(d, p, c, [1, 0, 0, 0, 0, 0])


def _write_cohort_clusters(
    d: Path, p: pd.DataFrame, c: pd.DataFrame, regl: list, labels=None
) -> str:
    _write_clean_predictions_csv(d, p)
    n = len(regl)
    c2 = c.iloc[:n].copy()
    c2["regl"] = regl
    if labels is not None:
        c2["label"] = labels
    else:
        # guarantee both classes across the slice
        c2["label"] = [1 if i % 2 == 0 else 0 for i in range(n)]
    p2 = p[p["subject_id"].isin(set(c2["subject_id"]))]
    p2.to_csv(d / "predictions.csv", index=False)
    c2.to_parquet(d / "cohort.parquet", index=False)
    _write_yaml(d, _base_spec())
    return "predictions.csv"


def _b_cluster_g_one(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_clusters(d, p, c, [7, 7, 7, 7, 7, 7])


def _b_cluster_g_two(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_clusters(d, p, c, [1, 1, 1, 2, 2, 2])


def _b_cluster_float_ids(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_clusters(d, p, c, [1.5, 1.5, 2.5, 2.5, 3.5, 3.5])


def _b_cluster_zero_event(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    # region 2 has only non-events; region 1 carries the events.
    return _write_cohort_clusters(d, p, c, [1, 1, 1, 2, 2, 2], labels=[1, 1, 0, 0, 0, 0])


def _b_cluster_g_equals_n(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_clusters(d, p, c, [10, 11, 12, 13, 14, 15])


def _b_cluster_constant(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _write_cohort_clusters(d, p, c, [3, 3, 3, 3, 3, 3])


def _b_roster_undeclared(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)  # expected_models = [MODEL_ID]
    p2 = p.copy()
    # Relabel a slice of rows to a model_id not on the roster (a likely typo).
    mask = np.arange(len(p2)) % 5 == 0
    p2.loc[mask, "model_id"] = "gusto_west_refit_logisitc"  # transposed typo
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


def _b_roster_declared_absent(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_predictions_csv(d, p)  # only MODEL_ID present
    c.to_parquet(d / "cohort.parquet", index=False)
    _write_yaml(d, _base_spec(expected_extra=["gusto_east_refit_logistic"]))  # ghost model declared
    return "predictions.csv"


def _b_excel_scientific(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _corrupt_first_pred_value(d, p, c, "6.2E-1")


def _b_excel_mangled_id(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    # Excel turning a subject id into a date-like token; still a valid string id, must load.
    _write_clean_cohort(d, c)
    lines = _pred_csv_text(p).splitlines()
    first = lines[1].split(",")
    first[0] = "MAR-4"
    lines[1] = ",".join(first)
    # mirror the id into the cohort so coverage still holds
    c2 = c.copy()
    c2.loc[c2.index[0], "subject_id"] = "MAR-4"
    c2.to_parquet(d / "cohort.parquet", index=False)
    _write_yaml(d, _base_spec())
    (d / "predictions.csv").write_text("\n".join(lines) + "\n")
    return "predictions.csv"


def _b_excel_thousands_sep(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    return _corrupt_first_pred_value(d, p, c, '"1,234"')


def _b_empty_file(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    (d / "predictions.csv").write_text("")
    return "predictions.csv"


def _b_header_only(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    (d / "predictions.csv").write_text("subject_id,model_id,predicted\n")
    return "predictions.csv"


def _b_truncated_row(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    lines = _pred_csv_text(p).splitlines()
    lines[-1] = ",".join(lines[-1].split(",")[:-1])  # drop the final (predicted) field
    (d / "predictions.csv").write_text("\n".join(lines) + "\n")
    return "predictions.csv"


def _b_extra_trailing_cols(d: Path, p: pd.DataFrame, c: pd.DataFrame) -> str:
    _write_clean_cohort(d, c)
    p2 = p.copy()
    p2["junk"] = "x"
    p2.to_csv(d / "predictions.csv", index=False)
    return "predictions.csv"


_BUILDERS = {
    "renamed_pred_col": _b_renamed_pred_col,
    "renamed_outcome_col": _b_renamed_outcome_col,
    "na_prediction": _b_na_prediction,
    "na_outcome": _b_na_outcome,
    "na_cluster": _b_na_cluster,
    "dup_exact": _b_dup_exact,
    "dup_conflicting": _b_dup_conflicting,
    "pred_negative": _b_pred_negative,
    "pred_gt1": _b_pred_gt1,
    "pred_bulk_zero_one": _b_pred_bulk_zero_one,
    "pred_logit_scale": _b_pred_logit_scale,
    "str_comma_decimal": _b_str_comma_decimal,
    "str_padded": _b_str_padded,
    "str_NULL": _b_str_NULL,
    "str_NaN": _b_str_NaN,
    "str_empty_value": _b_str_empty_value,
    "enc_utf8_bom": _b_enc_utf8_bom,
    "enc_crlf": _b_enc_crlf,
    "enc_utf16": _b_enc_utf16,
    "outcome_yes_no": _b_outcome_yes_no,
    "outcome_true_false": _b_outcome_true_false,
    "outcome_float_1_0": _b_outcome_float_1_0,
    "outcome_third_value": _b_outcome_third_value,
    "outcome_all_zero": _b_outcome_all_zero,
    "outcome_all_one": _b_outcome_all_one,
    "outcome_one_event": _b_outcome_one_event,
    "cluster_g_one": _b_cluster_g_one,
    "cluster_g_two": _b_cluster_g_two,
    "cluster_float_ids": _b_cluster_float_ids,
    "cluster_zero_event": _b_cluster_zero_event,
    "cluster_g_equals_n": _b_cluster_g_equals_n,
    "cluster_constant": _b_cluster_constant,
    "roster_undeclared": _b_roster_undeclared,
    "roster_declared_absent": _b_roster_declared_absent,
    "excel_scientific": _b_excel_scientific,
    "excel_mangled_id": _b_excel_mangled_id,
    "excel_thousands_sep": _b_excel_thousands_sep,
    "empty_file": _b_empty_file,
    "header_only": _b_header_only,
    "truncated_row": _b_truncated_row,
    "extra_trailing_cols": _b_extra_trailing_cols,
}


def build_all(dest: Path) -> dict[str, CaseFiles]:
    """Generate every case under `dest/<case_id>/`, returning a map of case id -> CaseFiles."""
    dest = Path(dest)
    base_pred, base_cohort = load_base()
    out: dict[str, CaseFiles] = {}
    for spec in CASE_SPECS:
        case_dir = dest / spec.id
        case_dir.mkdir(parents=True, exist_ok=True)
        pred_name = _BUILDERS[spec.id](case_dir, base_pred, base_cohort)
        out[spec.id] = CaseFiles(
            case_dir=case_dir,
            predictions=case_dir / pred_name,
            cohort_yaml=case_dir / "cohort.yaml",
        )
    return out


def _self_check() -> None:
    """Sanity generation into a temp dir when run as a script."""
    import tempfile

    dest = Path(tempfile.mkdtemp(prefix="predval_corruptions_"))
    files = build_all(dest)
    missing = [
        k for k, v in files.items() if not v.predictions.exists() or not v.cohort_yaml.exists()
    ]
    print(f"generated {len(files)} cases into {dest}")
    print("missing:", missing or "none")


if __name__ == "__main__":
    _self_check()
