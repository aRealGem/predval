"""Unskipped mutation tests for the IMPROVE-only golden precision budget."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from tests._improve_golden_comparison import (
    ATOL,
    VERDICT_FLOATS,
    assert_frame_matches,
    assert_improve_outputs_match,
    assert_manifest_matches,
)


def metrics():
    return pd.DataFrame(
        {
            "model": ["a", "a", "b"],
            "metric": ["brier", "calibration_slope", "boundary_count"],
            "threshold": [0.5] * 3,
            "ci_level": [0.95] * 3,
            "n": [10] * 3,
            "value": [0.2, 1.2, 2.0],
            "ci_low": [0.1, 1.1, np.nan],
            "ci_high": [0.3, 1.3, np.nan],
        }
    )


def fragility():
    return pd.DataFrame({"model": ["a", "b"], "n": [10, 20], "max_abs_delta": [0.1, 0.2]})


def manifest():
    return {
        "config": {"threshold": 0.5, "B": 2000},
        "versions": {"python": "3.13.16"},
        "flags": ["nonconvergence", "other"],
        "verdict": {
            "a": {
                **dict.fromkeys(VERDICT_FLOATS, 0.2),
                "bss_ci_low": 0.1,
                "bss_ci_high": 0.3,
                "classification": "calibration",
                "best_rung": 2,
            }
        },
    }


@pytest.mark.parametrize("column", ["value", "ci_low", "ci_high"])
@pytest.mark.parametrize("delta,accepted", [(ATOL / 2, True), (ATOL * 2, False)])
def test_metric_budget(column, delta, accepted):
    expected = metrics()
    actual = expected.copy(deep=True)
    actual.loc[0, column] += delta
    if accepted:
        assert_frame_matches(actual, expected, kind="metrics")
    else:
        with pytest.raises(AssertionError):
            assert_frame_matches(actual, expected, kind="metrics")


@pytest.mark.parametrize("delta,accepted", [(ATOL / 2, True), (ATOL * 2, False)])
def test_fragility_budget(delta, accepted):
    expected = fragility()
    actual = expected.copy(deep=True)
    actual.loc[0, "max_abs_delta"] += delta
    if accepted:
        assert_frame_matches(actual, expected, kind="fragility")
    else:
        with pytest.raises(AssertionError):
            assert_frame_matches(actual, expected, kind="fragility")


@pytest.mark.parametrize("field", sorted(VERDICT_FLOATS))
@pytest.mark.parametrize("delta,accepted", [(ATOL / 2, True), (ATOL * 2, False)])
def test_verdict_budget(field, delta, accepted):
    expected = manifest()
    actual = copy.deepcopy(expected)
    actual["verdict"]["a"][field] += delta
    if accepted:
        assert_manifest_matches(actual, expected)
    else:
        with pytest.raises(AssertionError):
            assert_manifest_matches(actual, expected)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.iloc[:-1],
        lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True),
        lambda d: d.iloc[::-1].reset_index(drop=True),
        lambda d: d[d.columns[::-1]],
        lambda d: d.set_axis([1, 2, 3]),
        lambda d: d.assign(threshold=d.threshold + 1e-12),
        lambda d: d.assign(ci_level=d.ci_level + 1e-12),
        lambda d: d.assign(n=d.n + 1),
        lambda d: d.assign(n=d.n.astype(float)),
        lambda d: d.assign(model=["changed", "a", "b"]),
        lambda d: d.assign(value=[0.2, 1.2, 2.0 + ATOL / 2]),
        lambda d: d.assign(ci_low=[0.1, 1.1, 0.0]),
    ],
)
def test_exact_frame_guards(mutation):
    expected = metrics()
    with pytest.raises(AssertionError):
        assert_frame_matches(mutation(expected.copy(deep=True)), expected, kind="metrics")


@pytest.mark.parametrize(
    "before,after", [(np.nan, 0.0), (np.inf, -np.inf), (np.inf, 1.0), (-np.inf, 1.0), (1.0, np.inf)]
)
def test_nonfinite_masks(before, after):
    expected = metrics()
    expected.loc[0, "value"] = before
    assert_frame_matches(expected.copy(), expected, kind="metrics")
    actual = expected.copy(deep=True)
    actual.loc[0, "value"] = after
    with pytest.raises(AssertionError):
        assert_frame_matches(actual, expected, kind="metrics")
    e = manifest()
    e["verdict"]["a"]["bss"] = before
    assert_manifest_matches(copy.deepcopy(e), e)
    a = copy.deepcopy(e)
    a["verdict"]["a"]["bss"] = after
    with pytest.raises(AssertionError):
        assert_manifest_matches(a, e)


@pytest.mark.parametrize("column", ["ci_low", "ci_high"])
@pytest.mark.parametrize("row,boundary", [(0, 0.0), (1, 0.0), (1, 1.0)])
@pytest.mark.parametrize("cross", [False, True])
def test_interval_boundary_crossing(column, row, boundary, cross):
    expected = metrics()
    expected.loc[row, "ci_low"] = boundary - ATOL / 4
    expected.loc[row, "ci_high"] = boundary + ATOL / 4
    actual = expected.copy(deep=True)
    actual.loc[row, column] = (
        boundary + (ATOL / 8 if column == "ci_low" else -ATOL / 8) if cross else boundary
    )
    with pytest.raises(AssertionError):
        assert_frame_matches(actual, expected, kind="metrics")


def test_inverted_interval():
    expected = metrics()
    expected.loc[0, ["ci_low", "ci_high"]] = [0.2, 0.2 + ATOL / 4]
    actual = expected.copy(deep=True)
    actual.loc[0, "ci_low"] += ATOL / 2
    with pytest.raises(AssertionError, match="inverted CI"):
        assert_frame_matches(actual, expected, kind="metrics")
    e = manifest()
    e["verdict"]["a"].update(bss_ci_low=0.2, bss_ci_high=0.2 + ATOL / 4)
    a = copy.deepcopy(e)
    a["verdict"]["a"]["bss_ci_low"] += ATOL / 2
    with pytest.raises(AssertionError, match="inverted CI"):
        assert_manifest_matches(a, e)


@pytest.mark.parametrize("field", ["bss_ci_low", "bss_ci_high"])
@pytest.mark.parametrize("cross", [False, True])
def test_verdict_null_crossing(field, cross):
    e = manifest()
    e["verdict"]["a"].update(bss_ci_low=-ATOL / 4, bss_ci_high=ATOL / 4)
    a = copy.deepcopy(e)
    a["verdict"]["a"][field] = (ATOL / 8 if field == "bss_ci_low" else -ATOL / 8) if cross else 0.0
    with pytest.raises(AssertionError):
        assert_manifest_matches(a, e)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop("flags"),
        lambda d: d["flags"].pop(),
        lambda d: d["flags"].reverse(),
        lambda d: d["flags"].__setitem__(0, "changed"),
        lambda d: d["config"].update(threshold=0.5 + 1e-12),
        lambda d: d["config"].update(B=2000.0),
        lambda d: d["config"].update(B=True),
        lambda d: d["versions"].update(python="changed"),
        lambda d: d["verdict"]["a"].update(classification="changed"),
        lambda d: d["verdict"]["a"].update(best_rung=3),
        lambda d: d["verdict"]["a"].update(bss=None),
        lambda d: d["verdict"].update(b=copy.deepcopy(d["verdict"]["a"])),
    ],
)
def test_exact_manifest_guards(mutation):
    expected = manifest()
    actual = copy.deepcopy(expected)
    mutation(actual)
    with pytest.raises(AssertionError):
        assert_manifest_matches(actual, expected)


def test_complete_gate(tmp_path):
    actual, expected = tmp_path / "actual", tmp_path / "expected"
    for root in (actual, expected):
        root.mkdir()
        metrics().to_parquet(root / "metrics.parquet")
        fragility().to_parquet(root / "fragility.parquet")
        (root / "manifest.json").write_text(json.dumps(manifest()))
        for name in ("calibration.parquet", "coverage.parquet", "report.html"):
            (root / name).write_bytes(b"exact")
        (root / "findings.json").write_text(
            json.dumps({"provenance": {"git_commit": root.name}, "value": 0.1})
        )
    assert_improve_outputs_match(actual, expected)
    for name in ("calibration.parquet", "coverage.parquet", "report.html", "findings.json"):
        original = (actual / name).read_bytes()
        if name == "findings.json":
            doc = json.loads(original)
            doc["value"] += 1e-12
            (actual / name).write_text(json.dumps(doc))
        else:
            (actual / name).write_bytes(b"changed")
        with pytest.raises(AssertionError):
            assert_improve_outputs_match(actual, expected)
        (actual / name).write_bytes(original)
