"""Report figures (item 2): the bytes must be reproducible and self-contained.

A figure is part of the report's bytes, so the same inputs must render the same SVG every time,
and the SVG must not depend on a system-installed font or leak a wall-clock date.
"""

from __future__ import annotations

from predval import figures as F

CURVES = {
    "m1": [
        {
            "mean_pred": i / 10 + 0.02,
            "obs_rate": (i / 10) ** 1.2,
            "ci_low": max(0.0, (i / 10) ** 1.2 - 0.05),
            "ci_high": min(1.0, (i / 10) ** 1.2 + 0.05),
        }
        for i in range(10)
    ],
    "m2": [
        {"mean_pred": i / 10 + 0.02, "obs_rate": i / 10, "ci_low": i / 10, "ci_high": i / 10}
        for i in range(10)
    ],
}
DUMBBELL = [
    {
        "model": "m1",
        "value": 0.80,
        "cluster_low": 0.72,
        "cluster_high": 0.88,
        "naive_low": 0.785,
        "naive_high": 0.815,
    },
    {
        "model": "m2",
        "value": 0.76,
        "cluster_low": 0.70,
        "cluster_high": 0.82,
        "naive_low": 0.748,
        "naive_high": 0.772,
    },
]
BRIER = [
    {
        "model": "m1",
        "rung0": 0.20,
        "levels": {"rung1": 0.19, "rung2": 0.17, "rung3": 0.175},
        "gains": {"rung1": (0.005, 0.015), "rung2": (0.02, 0.04), "rung3": (0.01, 0.04)},
    },
]


def test_all_three_figures_are_byte_identical_across_runs() -> None:
    a = F.calibration_small_multiples(CURVES, ["m1", "m2"])
    b = F.calibration_small_multiples(CURVES, ["m1", "m2"])
    assert a == b and a

    c = F.interval_dumbbell(DUMBBELL)
    d = F.interval_dumbbell(DUMBBELL)
    assert c == d and c

    e = F.brier_by_rung(BRIER)
    f = F.brier_by_rung(BRIER)
    assert e == f and e


def test_figures_are_inline_svg_with_no_date_leak() -> None:
    svg = F.calibration_small_multiples(CURVES, ["m1", "m2"])
    assert svg.lstrip().startswith("<svg")
    assert "<?xml" not in svg and "<!DOCTYPE" not in svg
    assert "dc:date" not in svg.lower()


def test_empty_inputs_yield_empty_string() -> None:
    assert F.calibration_small_multiples({}, []) == ""
    assert F.interval_dumbbell([]) == ""
    assert F.brier_by_rung([]) == ""


def test_svg_embeds_glyph_paths_not_a_font_dependency() -> None:
    """svg.fonttype='path' means text is vector outlines: no reference to a named system font."""
    svg = F.interval_dumbbell(DUMBBELL)
    assert "<path" in svg
    # DejaVu Sans is matplotlib's bundled default; with fonttype=path we do not emit a font-family
    # dependency on the (never-installed) IBM Plex.
    assert "IBM Plex" not in svg
