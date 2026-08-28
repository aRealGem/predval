"""The cohort spec must accept what the contract allows and reject what it does not."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from predval import CohortSpec
from tests.conftest import VALID_SPEC


def spec_with(**overrides) -> dict:
    return {**VALID_SPEC, **overrides}


def test_valid_spec_parses() -> None:
    spec = CohortSpec.model_validate(VALID_SPEC)
    assert spec.cohort_id == "toy"
    assert spec.clustering is not None
    assert spec.clustering.field == "site"
    assert spec.subgroup_fields == ("arm",)


def test_round_trip_is_stable() -> None:
    """Parsing, dumping and re-parsing must reach the same object."""
    spec = CohortSpec.model_validate(VALID_SPEC)
    again = CohortSpec.model_validate(spec.model_dump())
    assert spec == again


def test_clustering_is_optional() -> None:
    spec = spec_with()
    del spec["clustering"]
    assert CohortSpec.model_validate(spec).clustering is None


def test_ensemble_members_defaults_empty() -> None:
    """S6.1 item 2: undeclared means 'none known to be an ensemble', not 'unknown'."""
    spec = CohortSpec.model_validate(VALID_SPEC)
    assert spec.ensemble_members == ()


def test_ensemble_members_can_be_declared() -> None:
    spec = CohortSpec.model_validate(spec_with(ensemble_members=["blend_model"]))
    assert spec.ensemble_members == ("blend_model",)


def test_unknown_key_is_rejected() -> None:
    """A typo must not silently disable a prespecified analysis."""
    with pytest.raises(ValidationError, match="subgrops|extra"):
        CohortSpec.model_validate(spec_with(subgrops=[]))


def test_wrong_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unsupported contract version"):
        CohortSpec.model_validate(spec_with(version=1))


def test_non_binary_outcome_rejected_in_v0() -> None:
    with pytest.raises(ValidationError):
        CohortSpec.model_validate(
            spec_with(outcome={"type": "survival", "field": "t", "positive_label": 1})
        )


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_threshold_must_be_a_probability(bad: float) -> None:
    with pytest.raises(ValidationError):
        CohortSpec.model_validate(spec_with(thresholds=[bad]))


def test_empty_thresholds_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one decision threshold"):
        CohortSpec.model_validate(spec_with(thresholds=[]))


def test_duplicate_thresholds_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        CohortSpec.model_validate(spec_with(thresholds=[0.5, 0.5]))


def test_coverage_min_fraction_must_be_a_probability() -> None:
    with pytest.raises(ValidationError):
        CohortSpec.model_validate(spec_with(coverage={"min_fraction": 1.5, "compare_on": "both"}))


def test_duplicate_subgroup_names_rejected() -> None:
    with pytest.raises(ValidationError, match="subgroup names must be unique"):
        CohortSpec.model_validate(
            spec_with(subgroups=[{"name": "g", "field": "arm"}, {"name": "g", "field": "site"}])
        )


def test_outcome_cannot_also_be_a_subgroup() -> None:
    """Stratifying by the thing being predicted produces degenerate strata."""
    with pytest.raises(ValidationError, match="outcome.field"):
        CohortSpec.model_validate(spec_with(subgroups=[{"name": "g", "field": "label"}]))


def test_outcome_cannot_also_be_the_clustering_unit() -> None:
    with pytest.raises(ValidationError, match="outcome.field"):
        CohortSpec.model_validate(spec_with(clustering={"field": "label"}))


def test_required_cohort_columns_covers_every_declared_field() -> None:
    spec = CohortSpec.model_validate(VALID_SPEC)
    assert set(spec.required_cohort_columns) == {"subject_id", "label", "site", "arm"}
