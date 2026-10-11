"""IMPROVE-only precision contract; never a general JSON/Parquet float tolerance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal

# A fixed ten-decimal continuous-output budget, not fitted to an observed drift.
# This is 5,000 times below half a six-decimal findings unit. No relative tolerance.
ATOL = 1e-10
VERDICT_FLOATS = frozenset(
    {"best_rung_brier", "bss", "bss_ci_low", "bss_ci_high", "prevalence", "rung0_brier"}
)


def _same_side(actual, expected, boundary: float) -> None:
    np.testing.assert_array_equal(np.sign(actual - boundary), np.sign(expected - boundary))


def _continuous(actual, expected) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(actual), mask(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], atol=ATOL, rtol=0)


def _valid_interval(low, high) -> None:
    finite = np.isfinite(low) & np.isfinite(high)
    assert np.all(np.asarray(low)[finite] <= np.asarray(high)[finite]), "inverted CI"


def assert_frame_matches(actual: pd.DataFrame, expected: pd.DataFrame, *, kind: str) -> None:
    """Keep row positions, schema and every non-whitelisted cell exact."""
    assert kind in {"metrics", "fragility"}
    assert actual.shape == expected.shape
    assert_index_equal(actual.columns, expected.columns, exact=True)
    assert_index_equal(actual.index, expected.index, exact=True)
    assert actual.dtypes.equals(expected.dtypes), "dtype drift"
    approximate = {"value", "ci_low", "ci_high"} if kind == "metrics" else {"max_abs_delta"}
    exact = [column for column in expected.columns if column not in approximate]
    assert_frame_equal(actual[exact], expected[exact], check_exact=True)
    for column in approximate:
        a, e = actual[column].to_numpy(), expected[column].to_numpy()
        if kind == "metrics":
            count = expected["metric"].eq("boundary_count").to_numpy()
            np.testing.assert_array_equal(a[count], e[count])
            if column in {"ci_low", "ci_high"}:
                _same_side(a, e, 0.0)
                slope = expected["metric"].eq("calibration_slope").to_numpy()
                _same_side(a[slope], e[slope], 1.0)
        _continuous(a, e)
    if kind == "metrics":
        for frame in (actual, expected):
            _valid_interval(frame["ci_low"].to_numpy(), frame["ci_high"].to_numpy())


def assert_manifest_matches(actual, expected, path: tuple = (), *, tolerant: bool = True) -> None:
    """Exact recursive types/structure, with six named verdict numeric leaves."""
    assert type(actual) is type(expected), f"type drift at {path}"
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"key drift at {path}"
        for key in expected:
            assert_manifest_matches(actual[key], expected[key], (*path, key), tolerant=tolerant)
        if len(path) == 2 and path[0] == "verdict":
            for doc in (actual, expected):
                low, high = doc.get("bss_ci_low"), doc.get("bss_ci_high")
                if isinstance(low, float) and isinstance(high, float):
                    _valid_interval(np.array([low]), np.array([high]))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"length drift at {path}"
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_manifest_matches(a, e, (*path, index), tolerant=tolerant)
    elif (
        tolerant
        and isinstance(expected, float)
        and len(path) == 3
        and path[0] == "verdict"
        and path[2] in VERDICT_FLOATS
    ):
        _continuous(actual, expected)
        if path[2] in {"bss_ci_low", "bss_ci_high"}:
            _same_side(np.array(actual), np.array(expected), 0.0)
    else:
        # Exact equality including matching nonfinite masks; bool/int types checked above.
        if isinstance(expected, float) and np.isnan(expected):
            assert np.isnan(actual), f"NaN drift at {path}"
        else:
            assert actual == expected, f"value drift at {path}"


def assert_improve_outputs_match(actual: Path, expected: Path) -> None:
    for kind in ("metrics", "fragility"):
        assert_frame_matches(
            pd.read_parquet(actual / f"{kind}.parquet"),
            pd.read_parquet(expected / f"{kind}.parquet"),
            kind=kind,
        )
    assert_manifest_matches(
        json.loads((actual / "manifest.json").read_text()),
        json.loads((expected / "manifest.json").read_text()),
    )
    for name in ("calibration.parquet", "coverage.parquet", "report.html"):
        assert (actual / name).read_bytes() == (expected / name).read_bytes(), name
    findings = []
    for root in (actual, expected):
        doc = json.loads((root / "findings.json").read_text())
        doc["provenance"]["git_commit"] = None
        findings.append(doc)
    assert_manifest_matches(*findings, tolerant=False)
