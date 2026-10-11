"""Execute the README's public-API quickstart from an otherwise empty directory."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest


def test_readme_quickstart(tmp_path: Path, monkeypatch) -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    section = readme.split("## Quickstart with your own predictions\n", 1)[1].split("\n## ", 1)[0]
    csv_blocks = re.findall(r"```csv\n(.*?)```", section, re.DOTALL)
    yaml_blocks = re.findall(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert len(csv_blocks) == 2 and len(yaml_blocks) == 1
    cohort_dir = tmp_path / "my-cohort"
    cohort_dir.mkdir()
    for name, content in zip(
        ("cohort.csv", "predictions.csv", "cohort.yaml"),
        (*csv_blocks, *yaml_blocks),
        strict=True,
    ):
        (cohort_dir / name).write_text(content)
    code = section.split("uv run python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "README.md quickstart", "exec"), {})
    output = cohort_dir / "results"
    assert (output / "report.html").stat().st_size > 1000
    assert "configured fixed-prediction resampling" in (output / "report.html").read_text()
    assert json.loads((output / "findings.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["n_subjects"] == 12
    assert manifest["models_present"] == manifest["expected_models"] == ["my_model"]
    assert manifest["uncertainty"]["clustered"] is True
    assert manifest["uncertainty"]["clustering_field"] == "patient_id"
    assert manifest["uncertainty"]["n_boot"] == 20
    assert "few_clusters" in {flag["code"] for flag in manifest["flags"]}
    metrics = pd.read_parquet(output / "metrics.parquet")
    overall = metrics.loc[metrics["stratum_kind"] == "overall"]
    assert not overall.empty
    assert (overall["n"] == 12).all()
    assert (overall["n_events"] == 6).all()
    assert (overall["n_clusters"] == 6).all()
    baseline = overall.loc[
        (overall["model_id"] == "my_model")
        & (overall["subset"] == "full")
        & (overall["rung"] == "rung0")
        & (overall["fit_mode"] == "apparent")
    ]
    # Independent hand calculations from the CSVs: squared errors sum to 2.255;
    # 29 of the 6 * 6 event/non-event probability pairs are correctly ordered, with no ties.
    for metric, expected in {"brier": 2.255 / 12, "auroc": 29 / 36}.items():
        values = baseline.loc[baseline["metric"] == metric, "value"].tolist()
        assert values
        assert values == pytest.approx([expected] * len(values))
