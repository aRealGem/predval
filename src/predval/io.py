"""Loaders for the two predval inputs, plus coverage accounting.

Every loader refuses to run on a contract violation rather than coercing, dropping, or guessing.
The one exception is completeness handling, which drops only when the cohort explicitly asked
for `on_violation: drop_and_report`, and then reports exactly what it dropped.

See docs/spec.md sections 1, 2 and 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError

from .errors import CohortSpecError, CoverageViolation, PredictionsError
from .hashing import hash_file
from .schema import (
    HOLDOUT_FOLD,
    OPTIONAL_PREDICTION_COLUMNS,
    REQUIRED_PREDICTION_COLUMNS,
    CohortSpec,
)

_MAX_EXAMPLES = 5


def _examples(values: object) -> list:
    """A short, printable sample of offending values for an error message."""
    return list(pd.Series(values).head(_MAX_EXAMPLES))


@dataclass(frozen=True)
class Cohort:
    """A validated cohort: its spec, its subject-level table, and provenance digests."""

    spec: CohortSpec
    table: pd.DataFrame
    spec_path: Path
    data_path: Path
    spec_hash: str
    data_hash: str
    #: subject_ids dropped for a missing outcome under `on_violation: drop_and_report`.
    dropped_subjects: tuple[str, ...] = ()

    @property
    def subject_ids(self) -> pd.Index:
        return pd.Index(self.table[self.spec.subject_key])

    @property
    def n_subjects(self) -> int:
        return len(self.table)

    @property
    def hashes(self) -> dict[str, str]:
        return {"cohort_spec": self.spec_hash, "cohort_data": self.data_hash}


@dataclass(frozen=True)
class Predictions:
    """A validated predictions table, normalised and provenance-hashed."""

    frame: pd.DataFrame
    path: Path
    digest: str
    model_ids: tuple[str, ...] = field(default=())

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def hashes(self) -> dict[str, str]:
        return {"predictions": self.digest}


def _read_table(path: Path, *, what: str, error: type) -> pd.DataFrame:
    """Read a parquet or csv table, or fail with an actionable message."""
    if not path.exists():
        raise error(f"{what} file does not exist", path=path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".gz"}:
            return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise error(f"{what} file could not be read: {exc}", path=path) from exc
    raise error(
        f"{what} file must be .parquet or .csv (got {suffix!r})",
        path=path,
    )


def load_cohort(path: str | Path) -> Cohort:
    """Load and validate a cohort.yaml and the subject-level table it references.

    Validates the spec structurally, then cross-checks it against the data: every declared
    column must exist, subject ids must be unique and non-null, and `positive_label` must
    actually occur in the outcome column.
    """
    spec_path = Path(path)
    if not spec_path.exists():
        raise CohortSpecError("cohort spec does not exist", path=spec_path)

    try:
        raw = yaml.safe_load(spec_path.read_text())
    except yaml.YAMLError as exc:
        raise CohortSpecError(f"cohort.yaml is not valid YAML: {exc}", path=spec_path) from exc
    if not isinstance(raw, dict):
        raise CohortSpecError(
            f"cohort.yaml must contain a mapping at the top level (got {type(raw).__name__})",
            path=spec_path,
        )

    try:
        spec = CohortSpec.model_validate(raw)
    except ValidationError as exc:
        raise CohortSpecError(f"cohort.yaml failed validation:\n{exc}", path=spec_path) from exc

    # Relative to the yaml, so a cohort directory relocates as a unit.
    data_path = (spec_path.parent / spec.data).resolve()
    table = _read_table(data_path, what="cohort data", error=CohortSpecError)

    missing = [c for c in spec.required_cohort_columns if c not in table.columns]
    if missing:
        raise CohortSpecError(
            f"cohort table is missing columns declared in cohort.yaml: {missing}\n"
            f"  columns present: {sorted(table.columns)}",
            path=data_path,
        )

    key = spec.subject_key
    if table[key].isna().any():
        n = int(table[key].isna().sum())
        raise CohortSpecError(
            "subject_key column contains nulls", path=data_path, column=key, n_offending=n
        )
    if table[key].duplicated().any():
        dupes = table.loc[table[key].duplicated(), key]
        raise CohortSpecError(
            "subject_key must be unique; the cohort table is one row per subject",
            path=data_path,
            column=key,
            n_offending=int(len(dupes)),
            examples=_examples(dupes),
        )

    table, dropped = _apply_completeness(table, spec, data_path)

    outcome_col = table[spec.outcome.field]
    if not (outcome_col == spec.outcome.positive_label).any():
        raise CohortSpecError(
            f"outcome.positive_label {spec.outcome.positive_label!r} never occurs in the "
            f"outcome column; every subject would be a non-event\n"
            f"  values present: {sorted(pd.unique(outcome_col.dropna()))[:10]}",
            path=data_path,
            column=spec.outcome.field,
        )

    for sub in spec.subgroups:
        if table[sub.field].nunique(dropna=True) < 2:
            raise CohortSpecError(
                f"subgroup {sub.name!r} has fewer than two distinct levels, so it cannot "
                f"stratify anything",
                path=data_path,
                column=sub.field,
                examples=_examples(pd.unique(table[sub.field])),
            )

    return Cohort(
        spec=spec,
        table=table.reset_index(drop=True),
        spec_path=spec_path,
        data_path=data_path,
        spec_hash=hash_file(spec_path),
        data_hash=hash_file(data_path),
        dropped_subjects=dropped,
    )


def _apply_completeness(
    table: pd.DataFrame, spec: CohortSpec, data_path: Path
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Handle subjects with a missing outcome, per the cohort's completeness policy."""
    if not spec.completeness.require_outcome:
        return table, ()

    missing_mask = table[spec.outcome.field].isna()
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        return table, ()

    dropped = tuple(str(s) for s in table.loc[missing_mask, spec.subject_key])
    if spec.completeness.on_violation == "fail":
        raise CohortSpecError(
            f"{n_missing} subjects have a missing outcome and completeness.on_violation is 'fail'",
            path=data_path,
            column=spec.outcome.field,
            n_offending=n_missing,
            examples=list(dropped[:_MAX_EXAMPLES]),
        )
    return table.loc[~missing_mask], dropped


def load_predictions(path: str | Path) -> Predictions:
    """Load and validate a predictions.parquet.

    Normalises null `fold` to ``"holdout"`` *before* the uniqueness check, so that null and
    "holdout" are genuinely interchangeable rather than merely documented as such.
    """
    pred_path = Path(path)
    frame = _read_table(pred_path, what="predictions", error=PredictionsError)

    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise PredictionsError(
            f"predictions is missing required columns: {missing}\n"
            f"  columns present: {sorted(frame.columns)}",
            path=pred_path,
        )

    known = set(REQUIRED_PREDICTION_COLUMNS) | set(OPTIONAL_PREDICTION_COLUMNS)
    unknown = sorted(set(frame.columns) - known)
    if unknown:
        raise PredictionsError(
            f"predictions has columns outside the v0 contract: {unknown}\n"
            f"  allowed: {sorted(known)}",
            path=pred_path,
        )

    frame = frame.copy()

    for col in ("subject_id", "model_id"):
        if frame[col].isna().any():
            raise PredictionsError(
                f"{col} contains nulls",
                path=pred_path,
                column=col,
                n_offending=int(frame[col].isna().sum()),
            )
        frame[col] = frame[col].astype(str)

    frame["fold"] = (
        frame["fold"].astype("object").where(frame["fold"].notna(), HOLDOUT_FOLD).astype(str)
        if "fold" in frame.columns
        else HOLDOUT_FOLD
    )
    if "horizon" not in frame.columns:
        frame["horizon"] = np.nan

    predicted = pd.to_numeric(frame["predicted"], errors="coerce")
    non_numeric = predicted.isna() & frame["predicted"].notna()
    if non_numeric.any():
        raise PredictionsError(
            "predicted contains values that are not numeric",
            path=pred_path,
            column="predicted",
            n_offending=int(non_numeric.sum()),
            examples=_examples(frame.loc[non_numeric, "predicted"]),
        )
    if predicted.isna().any():
        raise PredictionsError(
            "predicted contains nulls or NaN; a subject a model did not score must have no "
            "row at all (docs/spec.md section 1.1)",
            path=pred_path,
            column="predicted",
            n_offending=int(predicted.isna().sum()),
        )

    values = predicted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        n = int((~np.isfinite(values)).sum())
        raise PredictionsError(
            "predicted contains infinite values",
            path=pred_path,
            column="predicted",
            n_offending=n,
        )
    out_of_range = (values < 0.0) | (values > 1.0)
    if out_of_range.any():
        raise PredictionsError(
            "predicted must be a probability in [0, 1]",
            path=pred_path,
            column="predicted",
            n_offending=int(out_of_range.sum()),
            examples=_examples(values[out_of_range]),
        )
    frame["predicted"] = values

    _check_unique(frame, pred_path)

    frame = frame[["subject_id", "model_id", "fold", "horizon", "predicted"]]
    return Predictions(
        frame=frame.reset_index(drop=True),
        path=pred_path,
        digest=hash_file(pred_path),
        model_ids=tuple(sorted(frame["model_id"].unique())),
    )


def _check_unique(frame: pd.DataFrame, pred_path: Path) -> None:
    """Enforce uniqueness of (subject_id, model_id, fold, horizon).

    `horizon` is stringified first so that NaN compares equal to NaN regardless of how the
    pandas version in use treats nulls in `duplicated`.
    """
    key = pd.DataFrame(
        {
            "subject_id": frame["subject_id"],
            "model_id": frame["model_id"],
            "fold": frame["fold"],
            "horizon": frame["horizon"].map(
                lambda v: (
                    "\x00null"
                    if v is None or (isinstance(v, float) and math.isnan(v))
                    else repr(float(v))
                )
            ),
        }
    )
    dup = key.duplicated(keep=False)
    if dup.any():
        offenders = key.loc[dup].drop_duplicates()
        raise PredictionsError(
            "duplicate (subject_id, model_id, fold, horizon) rows; predval will not silently "
            "average them",
            path=pred_path,
            n_offending=int(dup.sum()),
            examples=offenders.head(_MAX_EXAMPLES).to_dict("records"),
        )


def coverage_report(cohort: Cohort, predictions: Predictions) -> pd.DataFrame:
    """Per-model coverage against the cohort.

    Columns:

    ``coverage_full``
        The model's coverage of the whole cohort -- the honest headline number.
    ``common_fraction``
        How much of the cohort survives intersecting across all models. Identical for every
        row, because it describes the analysis set rather than any one model.
    ``n_dropped_if_common``
        Subjects this model scored that would be discarded by comparing on the common set.

    There is deliberately no "coverage of the common set": that quantity is 1.0 for every
    model by construction, since the common set is the intersection. The number that actually
    carries information is how much of the cohort the intersection costs -- see docs/spec.md
    section 2.4.
    """
    subjects = set(cohort.subject_ids.astype(str))
    n_full = len(subjects)

    scored: dict[str, set[str]] = {
        model: set(grp["subject_id"]) & subjects
        for model, grp in predictions.frame.groupby("model_id", sort=True)
    }
    common = set.intersection(*scored.values()) if scored else set()
    n_common = len(common)

    rows = []
    for model in sorted(scored):
        n_scored = len(scored[model])
        coverage_full = n_scored / n_full if n_full else math.nan
        rows.append(
            {
                "model_id": model,
                "n_scored": n_scored,
                "n_full": n_full,
                "coverage_full": coverage_full,
                "n_common": n_common,
                "common_fraction": n_common / n_full if n_full else math.nan,
                "n_dropped_if_common": n_scored - n_common,
                "meets_minimum": (coverage_full if n_full else 0.0)
                >= cohort.spec.coverage.min_fraction,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Flag:
    """A condition a reader must be told about, carried into the report and the manifest."""

    code: str
    severity: str  # "warning" | "note"
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


def check_roster(cohort: Cohort, predictions: Predictions) -> list[Flag]:
    """Compare the declared model roster against what the predictions actually contain.

    Checked in both directions. Declared-but-absent is the loud case the roster exists for;
    present-but-undeclared usually means a mistyped model_id, which is worse than it looks --
    it produces plausible metrics for a model that corresponds to nothing while the model it
    was meant to be silently reads as absent. See docs/spec.md section 2.6.
    """
    spec = cohort.spec
    present = set(predictions.model_ids)

    if spec.expected_models is None:
        return [
            Flag(
                code="roster_not_declared",
                severity="note",
                message=(
                    "model roster not declared; predval reports on whichever models appear "
                    "in the predictions file and cannot tell you what is missing"
                ),
            )
        ]

    declared = set(spec.expected_models)
    flags: list[Flag] = []

    absent = sorted(declared - present)
    if absent:
        message = f"declared in expected_models but absent from predictions: {absent}"
        if spec.on_missing_model == "fail":
            raise CohortSpecError(message, path=cohort.spec_path, column="expected_models")
        flags.append(Flag(code="declared_but_absent", severity="warning", message=message))

    undeclared = sorted(present - declared)
    if undeclared:
        flags.append(
            Flag(
                code="present_but_undeclared",
                severity="warning",
                message=(
                    f"present in predictions but not declared in expected_models: "
                    f"{undeclared}; check for a mistyped model_id"
                ),
            )
        )
    return flags


def check_common_selection(cohort: Cohort, predictions: Predictions) -> list[Flag]:
    """Caution when restricting to the common subset discards a large share of the cohort.

    The subjects every model happened to score are not a random sample. If models decline to
    score the hard cases, the common subset is the easy cases and every model looks better on
    it. See docs/spec.md section 2.4.
    """
    report = coverage_report(cohort, predictions)
    if report.empty:
        return []
    excluded = 1.0 - float(report["common_fraction"].iloc[0])
    threshold = cohort.spec.coverage.common_warn_frac
    if excluded > threshold:
        return [
            Flag(
                code="informative_common_selection",
                severity="warning",
                message=(
                    f"the common subset excludes {excluded:.1%} of cohort rows "
                    f"(> {threshold:.0%}); selection into the common subset may be informative"
                ),
            )
        ]
    return []


def check_coverage(cohort: Cohort, predictions: Predictions) -> pd.DataFrame:
    """Return the coverage report, raising if any model falls below the declared floor."""
    report = coverage_report(cohort, predictions)
    failing = report.loc[~report["meets_minimum"]]
    if len(failing):
        worst = failing.sort_values("coverage_full").iloc[0]
        raise CoverageViolation(
            f"{len(failing)} model(s) fall below coverage.min_fraction; worst shown",
            model_id=str(worst["model_id"]),
            observed=float(worst["coverage_full"]),
            required=float(cohort.spec.coverage.min_fraction),
        )
    return report
