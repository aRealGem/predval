"""pydantic v2 models for the cohort.yaml contract.

These models validate the *specification*. Checks that need the data itself -- that
``outcome.field`` exists in the cohort table, that coverage clears the floor -- live in io.py,
because they cannot be answered from the yaml alone.

See docs/spec.md section 2.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: The only contract version v0 accepts. Bumping this is a breaking change.
CONTRACT_VERSION = 0

#: Column names required in predictions.parquet.
REQUIRED_PREDICTION_COLUMNS = ("subject_id", "model_id", "predicted")

#: Optional columns predval understands in predictions.parquet.
OPTIONAL_PREDICTION_COLUMNS = ("fold", "horizon")

#: Value that a null `fold` normalises to. See docs/spec.md section 1.3.
HOLDOUT_FOLD = "holdout"

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    """Base config: unknown keys are errors.

    A typo'd key in cohort.yaml must not be silently ignored -- that is how a prespecified
    subgroup quietly stops being analysed while the file still looks correct.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class OutcomeSpec(StrictModel):
    """What is being predicted."""

    type: Literal["binary"]
    field: str = Field(min_length=1)
    positive_label: Any

    @model_validator(mode="after")
    def _positive_label_present(self) -> OutcomeSpec:
        if self.positive_label is None:
            raise ValueError(
                "outcome.positive_label must be set; it names the value counted as an event"
            )
        return self


class ClusteringSpec(StrictModel):
    """The unit within which subjects are correlated.

    Optional. When absent, S2 treats subjects as independent and says so in the report. See
    docs/spec.md section 2.1 for why the unit of analysis and the unit of independence differ.
    """

    field: str = Field(min_length=1)


class CoverageSpec(StrictModel):
    """How much of the cohort a model must have scored."""

    min_fraction: Probability
    compare_on: Literal["full", "common", "both"] = "both"


class CompletenessSpec(StrictModel):
    """What to do about subjects with a missing outcome."""

    require_outcome: bool = True
    on_violation: Literal["drop_and_report", "fail"] = "drop_and_report"


class SubgroupSpec(StrictModel):
    """One prespecified subgroup analysis."""

    name: str = Field(min_length=1)
    field: str = Field(min_length=1)


class CohortSpec(StrictModel):
    """The parsed, validated contents of a cohort.yaml.

    Validation here is structural only. Cross-checks against the cohort table happen in
    io.load_cohort, which has the data in hand.
    """

    cohort_id: str = Field(min_length=1)
    version: int
    subject_key: str = Field(min_length=1)
    data: str = Field(min_length=1)
    outcome: OutcomeSpec
    coverage: CoverageSpec
    completeness: CompletenessSpec
    clustering: ClusteringSpec | None = None
    subgroups: tuple[SubgroupSpec, ...] = ()
    thresholds: tuple[Probability, ...] = (0.5,)

    @field_validator("version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if v != CONTRACT_VERSION:
            raise ValueError(
                f"unsupported contract version {v!r}; this predval understands "
                f"version {CONTRACT_VERSION}"
            )
        return v

    @field_validator("thresholds")
    @classmethod
    def _thresholds_sane(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v:
            raise ValueError("thresholds must list at least one decision threshold")
        if len(set(v)) != len(v):
            raise ValueError(f"thresholds contains duplicates: {sorted(v)}")
        return v

    @model_validator(mode="after")
    def _subgroup_names_unique(self) -> CohortSpec:
        names = [s.name for s in self.subgroups]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"subgroup names must be unique; repeated: {dupes}")
        return self

    @model_validator(mode="after")
    def _roles_do_not_collide(self) -> CohortSpec:
        """Each column may play at most one role.

        Using the outcome as a subgroup would stratify by the thing being predicted, producing
        degenerate strata; using it as the clustering unit is equally meaningless. Catching it
        here is cheaper than explaining an undefined AUROC later.
        """
        roles: dict[str, str] = {self.outcome.field: "outcome.field"}
        if self.clustering is not None:
            roles.setdefault(self.clustering.field, "clustering.field")
            if roles[self.clustering.field] != "clustering.field":
                raise ValueError(
                    f"column {self.clustering.field!r} is used as both "
                    f"{roles[self.clustering.field]} and clustering.field"
                )
        for sub in self.subgroups:
            if sub.field in roles:
                raise ValueError(
                    f"column {sub.field!r} is used as both {roles[sub.field]} and "
                    f"subgroup {sub.name!r}"
                )
            roles[sub.field] = f"subgroup {sub.name!r}"
        return self

    @property
    def subgroup_fields(self) -> tuple[str, ...]:
        return tuple(s.field for s in self.subgroups)

    @property
    def required_cohort_columns(self) -> tuple[str, ...]:
        """Every column load_cohort must find in the cohort table."""
        cols = [self.subject_key, self.outcome.field]
        if self.clustering is not None:
            cols.append(self.clustering.field)
        cols.extend(self.subgroup_fields)
        return tuple(cols)
