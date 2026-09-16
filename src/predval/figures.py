"""Deterministic inline-SVG figures for the report (item 2).

Every figure is rendered from artefacts predval already produced -- calibration.parquet and
metrics.parquet -- and embedded directly in the HTML. Nothing statistical is computed here; this
module only draws.

**Determinism is a contract, not a nicety.** A report that is byte-identical across runs is what
lets the provenance section mean something, and a figure is part of those bytes. Three settings
make matplotlib's SVG reproducible, applied once at import before any figure exists:

- ``SOURCE_DATE_EPOCH=0`` and suppressing the ``Date`` metadata on save, so no wall-clock leaks in;
- ``svg.hashsalt`` fixed, so clip-path and gradient element ids are stable rather than random;
- ``svg.fonttype='path'`` with matplotlib's **bundled** DejaVu Sans, so glyphs are embedded as
  vector outlines. The report never depends on a system-installed font (IBM Plex or otherwise), and
  the same matplotlib version draws the same paths every time.
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

# Applied before the first figure is created. font.family is matplotlib's bundled default; we never
# reach for a system font. svg.fonttype='path' embeds the glyph outlines so the SVG stands alone.
matplotlib.rcParams.update(
    {
        "svg.hashsalt": "predval",
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.linewidth": 0.6,
        "figure.dpi": 100,
    }
)

#: Report palette (item 2). Deep blue leads; teal/ochre/rose are the accents.
BLUE = "#123B5E"
TEAL = "#1F7A78"
OCHRE = "#B4762A"
ROSE = "#C56B7A"


def _svg(fig) -> str:
    """Serialise a figure to an inline SVG fragment (no XML prolog / DOCTYPE), and close it.

    The ``Date`` metadata is explicitly dropped -- without it matplotlib stamps the current time
    into the SVG and the report stops being byte-stable.
    """
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = buf.getvalue()
    start = svg.index("<svg")
    return svg[start:]


def _grid(n: int, ncols: int = 3) -> tuple[int, int]:
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def calibration_small_multiples(curves: dict[str, list[dict]], models: list[str]) -> str:
    """Per-member rung0 calibration: decile points, a monotone smooth curve, a cluster-bootstrap
    band, and the 45-degree "perfect" reference (item 2a).

    ``curves[model]`` is a list of decile dicts with ``mean_pred``, ``obs_rate``, ``ci_low``,
    ``ci_high`` (from calibration.parquet). Members with no curve are skipped.
    """
    present = [m for m in models if curves.get(m)]
    if not present:
        return ""
    nrows, ncols = _grid(len(present))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, model in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        pts = sorted(curves[model], key=lambda d: d["mean_pred"])
        x = [d["mean_pred"] for d in pts]
        y = [d["obs_rate"] for d in pts]
        lo = [d["ci_low"] for d in pts]
        hi = [d["ci_high"] for d in pts]
        ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#999999", zorder=1)
        ax.fill_between(x, lo, hi, color=TEAL, alpha=0.18, zorder=2, linewidth=0)
        # A monotone smooth curve needs >= 2 strictly increasing x; deciles can tie in a saturated
        # score, so fall back to the connecting line when PCHIP cannot be built.
        ux, uy = _dedupe_increasing(x, y)
        if len(ux) >= 2:
            gx = _linspace(min(ux), max(ux), 60)
            ax.plot(gx, PchipInterpolator(ux, uy)(gx), lw=1.1, color=BLUE, zorder=3)
        ax.scatter(x, y, s=9, color=BLUE, zorder=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(model, fontsize=8)
        ax.tick_params(length=2)
    # One "perfect" label, on the first visible axis, so the reference line is named once.
    axes[0][0].annotate(
        "perfect",
        xy=(0.62, 0.62),
        xytext=(0.66, 0.42),
        fontsize=6,
        color="#777777",
        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#999999"},
    )
    fig.supxlabel("predicted probability", fontsize=8)
    fig.supylabel("observed frequency", fontsize=8)
    return _svg(fig)


def interval_dumbbell(rows: list[dict]) -> str:
    """The unit-of-analysis exhibit as a picture (item 2b): per member, the AUROC cluster interval
    against the naive per-row interval, so the width gap is legible at a glance.

    ``rows`` carry ``model``, ``value``, ``cluster_low/high``, ``naive_low/high``.
    """
    if not rows:
        return ""
    rows = list(rows)
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(rows) + 1.0))
    ys = list(range(len(rows)))
    for y, r in zip(ys, rows, strict=True):
        ax.plot(
            [r["naive_low"], r["naive_high"]],
            [y + 0.16, y + 0.16],
            color=OCHRE,
            lw=3.2,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [r["cluster_low"], r["cluster_high"]],
            [y - 0.16, y - 0.16],
            color=BLUE,
            lw=3.2,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter([r["value"]], [y], s=14, color="#333333", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("AUROC (95% interval)")
    ax.tick_params(length=2)
    # Legend by proxy lines, so the two bars are named without a per-row label.
    ax.plot([], [], color=BLUE, lw=3.2, label="cluster (correct)")
    ax.plot([], [], color=OCHRE, lw=3.2, label="naive per-row (understates)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    return _svg(fig)


def brier_by_rung(rows: list[dict]) -> str:
    """Cross-fitted Brier by rung per member, with the paired-gain intervals from item 1 (item 2c).

    ``rows`` carry ``model``, ``rung0`` (as-published Brier), ``levels`` ({rung: crossfit Brier}),
    and ``gains`` ({rung: (low, high)}) -- the paired cross-fit gain interval. The error bar on a
    rung's level is that gain interval reflected around the fixed rung0 anchor, so the bar shows the
    uncertainty of the *improvement*, which is what item 1 measured.
    """
    present = [r for r in rows if r.get("levels")]
    if not present:
        return ""
    nrows, ncols = _grid(len(present))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.1), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    rung_x = {"rung0": 0, "rung1": 1, "rung2": 2, "rung3": 3}
    for i, r in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        r0 = r["rung0"]
        ax.axhline(r0, ls=":", lw=0.7, color="#999999", zorder=1)
        xs, ys, elo, ehi = [0], [r0], [0.0], [0.0]
        for rung, level in sorted(r["levels"].items(), key=lambda kv: rung_x[kv[0]]):
            xs.append(rung_x[rung])
            ys.append(level)
            g = r["gains"].get(rung)
            if g is not None and g[0] == g[0] and g[1] == g[1]:  # finite
                # level = r0 - gain; CI(level) = [r0 - gain_high, r0 - gain_low]
                elo.append(max(0.0, level - (r0 - g[1])))
                ehi.append((r0 - g[0]) - level)
            else:
                elo.append(0.0)
                ehi.append(0.0)
        order = sorted(range(len(xs)), key=lambda j: xs[j])
        xs = [xs[j] for j in order]
        ys = [ys[j] for j in order]
        elo = [elo[j] for j in order]
        ehi = [ehi[j] for j in order]
        ax.errorbar(
            xs,
            ys,
            yerr=[elo, ehi],
            fmt="o-",
            ms=4,
            lw=1.0,
            color=BLUE,
            ecolor=TEAL,
            elinewidth=1.4,
            capsize=2,
            zorder=3,
        )
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0", "1", "2", "3"], fontsize=7)
        ax.set_title(r["model"], fontsize=8)
        ax.tick_params(length=2)
    fig.supxlabel("rung (0 = as published)", fontsize=8)
    fig.supylabel("cross-fitted Brier", fontsize=8)
    return _svg(fig)


# ------------------------------------------------------------------------------ small numerics


def _dedupe_increasing(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """Keep points with strictly increasing x (PchipInterpolator requires that)."""
    ux: list[float] = []
    uy: list[float] = []
    for xi, yi in zip(x, y, strict=True):
        if not ux or xi > ux[-1]:
            ux.append(xi)
            uy.append(yi)
    return ux, uy


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2 or b <= a:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]
