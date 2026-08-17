"""Error types for predval.

Every error carries enough context to act on without re-reading the input by hand: which file,
which column, and where practical an offending value and a count of how many rows are affected.
A validation harness that fails vaguely is a harness people learn to route around.
"""

from __future__ import annotations

from pathlib import Path


class PredvalError(Exception):
    """Base class for every error predval raises deliberately."""


class SchemaViolation(PredvalError):
    """An input does not satisfy the v0 contract.

    Raised instead of coercing, dropping, or guessing. See docs/spec.md section 5.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | Path | None = None,
        column: str | None = None,
        n_offending: int | None = None,
        examples: object = None,
    ) -> None:
        self.path = str(path) if path is not None else None
        self.column = column
        self.n_offending = n_offending
        self.examples = examples
        super().__init__(self._render(message))

    def _render(self, message: str) -> str:
        parts = [message]
        if self.path is not None:
            parts.append(f"  file:   {self.path}")
        if self.column is not None:
            parts.append(f"  column: {self.column}")
        if self.n_offending is not None:
            parts.append(f"  rows affected: {self.n_offending}")
        if self.examples is not None:
            parts.append(f"  examples: {self.examples}")
        return "\n".join(parts)


class CohortSpecError(SchemaViolation):
    """cohort.yaml is malformed, or references something that does not exist."""


class PredictionsError(SchemaViolation):
    """predictions.parquet violates the prediction contract."""


class CoverageViolation(PredvalError):
    """A model's coverage falls below the cohort's declared minimum.

    Distinct from SchemaViolation: the inputs are well-formed, but there are not enough
    predictions to support the analysis the cohort asked for.
    """

    def __init__(self, message: str, *, model_id: str, observed: float, required: float) -> None:
        self.model_id = model_id
        self.observed = observed
        self.required = required
        super().__init__(
            f"{message}\n"
            f"  model:    {model_id}\n"
            f"  coverage: {observed:.6f}\n"
            f"  required: {required:.6f}"
        )
