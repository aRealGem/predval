"""predval -- a predictions-in validation harness for clinical prediction models.

predval evaluates predictions that already exist. It never loads model weights, never trains or
tunes, and never touches the network. See docs/spec.md for the frozen v0 contracts.
"""

from __future__ import annotations

from .errors import (
    CohortSpecError,
    CoverageViolation,
    PredictionsError,
    PredvalError,
    SchemaViolation,
)
from .hashing import hash_bytes, hash_file, hash_inputs
from .io import (
    Cohort,
    Predictions,
    check_coverage,
    coverage_report,
    load_cohort,
    load_predictions,
)
from .schema import (
    CONTRACT_VERSION,
    HOLDOUT_FOLD,
    ClusteringSpec,
    CohortSpec,
    CompletenessSpec,
    CoverageSpec,
    OutcomeSpec,
    SubgroupSpec,
)

__version__ = "0.0.0"

__all__ = [
    "CONTRACT_VERSION",
    "HOLDOUT_FOLD",
    "ClusteringSpec",
    "Cohort",
    "CohortSpec",
    "CohortSpecError",
    "CompletenessSpec",
    "CoverageSpec",
    "CoverageViolation",
    "OutcomeSpec",
    "PredictionsError",
    "Predictions",
    "PredvalError",
    "SchemaViolation",
    "SubgroupSpec",
    "check_coverage",
    "coverage_report",
    "hash_bytes",
    "hash_file",
    "hash_inputs",
    "load_cohort",
    "load_predictions",
    "__version__",
]
