# predval

A predictions-in validation harness for clinical prediction models.

predval answers one question: **do a model's published probabilities hold up on your cohort?**
It takes predictions that already exist and evaluates them — discrimination, calibration,
prespecified subgroup behaviour, and a recalibration ladder that reports how much of any
miscalibration is repairable without touching the model, and discloses the ladder's own
optimism rather than reporting only its best face.

## The pitch, narrowed

Most tools in this space either train something (a new calibration layer, a monitoring model)
or assume you have live inference access to watch a model over time. predval assumes neither. It
is for the moment you have a `predictions.parquet` — a batch of probabilities a model already
produced on a cohort — and a validation question that has to be answered from that file alone.
Three things follow from that scope, and they are what predval spends its effort on:

- **Absence is a fact, not a gap to fill.** A model that did not score a subject gets no row.
  predval measures coverage per model against a declared floor and refuses to silently treat
  "no opinion" as "confidently negative" — the direction every naive join gets wrong, because it
  flatters the model.
- **The unit of analysis and the unit of independence are usually not the same file column.**
  When predictions come from grouped data — patients with repeat visits, patches from the same
  slide, images from the same device — per-row confidence intervals are not slightly optimistic,
  they are wrong by roughly the square root of the cluster size. predval takes a declared
  clustering unit and resamples whole clusters everywhere it computes uncertainty, and renders
  the naive-vs-cluster comparison once so the gap is a number, not a caveat in a footnote.
- **A recalibration that improves the numbers is a finding about the ladder's own fit, not a
  promotion of the model, until it survives being scored on clusters it never saw.** Every rung of
  predval's ladder — intercept, intercept+slope, spline — is fit twice: once on the rows it will
  be judged on (apparent, optimistic by construction) and once cross-fitted with whole clusters
  held out. The report shows both, states in an unconditional framing block that no rung is a
  validated model, and — this is the part most recalibration tooling skips — refuses to
  manufacture a gain where none exists: on the PCam fixture below, nine of the fifteen members
  carry a detected calibration fault, ten have the repair's benefit left honestly unproven at
  this cohort's small cluster count, four are *reliably harmed* by at least one rung, and exactly
  one member gets a real, interval-backed repair. The remaining six show no detected fault, which
  the report states as a null result and never as a clean bill of health: this harness's
  calibration test can fail to reject "calibrated", but it can never establish it.

## What predval deliberately does not do

- **It never loads model weights.** No `torch`, no `tensorflow`, no checkpoints. Inputs are a
  `cohort.yaml` and a `predictions.parquet`, and nothing else.
- **It never trains or tunes a model.** The recalibration ladder fits a link-scale correction to
  published probabilities; refitting on covariates is explicitly out of contract.
- **It never touches the network.** Library code makes zero network calls.
- **It has no server and no telemetry.**

These are permanent design constraints, not a roadmap. A validation harness that could quietly
retrain the thing it is validating would not be a validation harness.

## Prior art, and where predval sits relative to it

None of this is invented from nothing. predval borrows from, and narrows against, several
existing traditions:

- **`sklearn.calibration` (`CalibratedClassifierCV`, `calibration_curve`)** is the closest
  sibling in mechanism — fitting a correction (sigmoid or isotonic) to already-scored
  predictions. It does not, however, distinguish apparent from cross-fitted correction quality by
  default, has no notion of a declared clustering unit for uncertainty, and has no concept of
  prediction *absence* as distinct from a missing row being an error. predval's rung ladder is
  narrower in what it fits (link-scale only, never isotonic — see §4.2 of the spec) but wider in
  what it reports about the fit.
- **Harrell's `rms::val.prob`** (R) is the closest tradition in *intent*: externally validating a
  single published prediction model's calibration-in-the-large and calibration slope on a new
  cohort. predval generalizes this to many models at once, adds a cluster-aware uncertainty layer
  `val.prob` does not have, and extends the single calibration check into a ladder that also
  reports which correction was needed and whether it holds up out-of-fold.
- **TRIPOD-AI** and similar reporting guidelines specify *what a validation study should state*.
  predval does not replace that judgment — it is closer to an executable subset of it: the
  coverage table, the calibration diagnosis, and the subgroup gating are the kind of checks a
  TRIPOD-AI-conformant report has to make somewhere, produced as artefacts rather than prose a
  reader has to trust.
- **`calzone`** ([DIDSR/calzone](https://github.com/DIDSR/calzone), an FDA/CDRH regulatory-science
  tool, JOSS 2025) measures calibration: metrics, reliability diagrams, bootstrap confidence
  intervals, subgroup breakdowns, and prevalence adjustment for enriched-versus-population data.
  It is a measurement library, not a validation harness — no declared clustering unit or
  cluster-aware resampling, no recalibration ladder, no apparent-versus-cross-fitted optimism
  disclosure, no report artefact. Where its metrics apply to a cohort predval also measures,
  the two are complementary rather than competing; predval does not attempt to duplicate its
  breadth of calibration metrics.
- **`pmcalibration` and `pminternal`** (Stephen Rhodes,
  [stephenrho.github.io](https://stephenrho.github.io/pminternal/)) are the closest R analogues.
  `pmcalibration` fits smoothed calibration curves (spline/GAM/loess) and reports Eavg/E50/E90/Emax
  for binary and time-to-event outcomes. `pminternal` goes further and produces bias-corrected
  performance measures — but by *refitting the model itself* inside each bootstrap or
  cross-validation resample (bootstrap optimism, `.632`, CV optimism), which requires a model
  development function, not just its output. That is the structural difference from predval:
  predval's contract is predictions-only and has no model to refit, by design (see "What predval
  deliberately does not do," above) — nor, as far as this harness is aware, does either package
  resample at a declared clustering unit rather than at the row.
- **`predRupdate`** ([CRAN](https://cran.r-project.org/package=predRupdate)) is the nearest
  conceptual relative of the recalibration ladder: given an existing model's coefficients, it
  offers graded updating methods from an intercept-only correction up to a full recalibration
  refit, for both logistic and Cox models. predval's rungs cover the same *idea* — increasingly
  flexible corrections to a published score — but stay link-scale-only (never a covariate refit)
  and add the apparent-versus-cross-fitted split with a paired-gain interval that
  `predRupdate`, as far as this harness is aware, does not.
- **Decision curve analysis** (e.g. the `dcurves` family) answers a question predval does not yet
  ask — net benefit at a chosen treatment threshold. It is deliberately deferred (see the
  `pyproject.toml` dependency notes and the spec's backlog): a decision-curve number is only
  meaningful once a specific clinical action is on the table, and the fixture below does not have
  one.
- **Production ML monitoring tools** (drift dashboards, live-inference observability platforms)
  solve an adjacent but different problem: watching a *deployed* model's inputs and outputs
  change over time. predval has no notion of time-ordering or a live pipeline; it validates one
  static batch of predictions against one cohort, once, and says so.

predval's outcome contract is binary only (`docs/spec.md` §2.1); survival and competing-risks
outcomes are out of scope for v0, not a deferred feature of this version. For that territory,
[`SurvivalEVAL`](https://github.com/shi-ang/SurvivalEVAL) is the adjacent tool — a comparably
scoped evaluation package for individual survival distributions.

## Example: a real fixture, not a synthetic one

`examples/pcam/` runs predval end to end against a real 15-member PatchCamelyon histopathology
model roster: 19,999 image patches drawn from only 22 whole slides across two scanner/stain
domains. Every member is a single trained model — the roster contains no ensembles or blends
(see the last bullet below).
It is worth more than a clean synthetic dataset for exactly the reasons predval exists:

- Patches within a slide are heavily correlated, so the unit of analysis and the unit of
  independence genuinely differ — this is where the naive-vs-cluster exhibit is not academic.
- Two of the fifteen members are badly miscalibrated (one spans AUROC-implied confidence from
  0.216 to 0.865 at AUROC 0.764); the recalibration ladder finds a large apparent Brier gain for
  both at rung2 (+0.031 and +0.015) -- but with only G=22 slides to resample, the paired
  cross-fit interval on that gain **crosses zero** for both, so the ladder correctly declines to
  call either a real repair (S6: an earlier write-up of this fixture claimed both gains excluded
  zero -- a transcription error, not a code bug; corrected in `docs/spec.md` section 4.8 and the
  card log). That refusal to overclaim from a large point estimate at small G is itself the
  finding predval exists to make legible.
- One declared member's predictions were permanently lost — `p4m_reg`, an ingredient of the
  campaign's 11-model blend recipe, whose weights lived in ephemeral storage, so re-inference is
  impossible. It is declared in `expected_models` and ships no rows, so the fixture carries a
  genuine coverage gap rather than one manufactured by deleting rows.
- 732 predictions sit at exactly 0.0 or 1.0, where `logit` is undefined, making the eps-clip in
  the recalibration ladder load-bearing from the first rung, not a theoretical edge case.
- One member, `mobilenetv3`, is a single trained baseline model (its own solo out-of-fold AUROC
  0.916). It was named `champion` through S6, a name the campaign separately, confusingly, also
  uses for an unrelated 11-member blend recipe — S6.1 renamed it to its true architecture and
  attempted to export that real blend as its own fixture member too; the export is blocked
  (one of the blend's 11 ingredients has no surviving predictions anywhere, the same permanent
  loss recorded above), so no member in this fixture is actually an ensemble (see
  `examples/pcam/cohort.yaml` for the full account).

### Reproducing the example

```bash
uv sync
uv run python examples/pcam/reproduce.py
```

This is a **local, single-machine reproduction**, not a from-anywhere clean-clone one, and that
limit is deliberate rather than an oversight. `examples/pcam/cohort.parquet` and
`predictions.parquet` are **not** excluded because the data is sensitive — PatchCamelyon derives
from Camelyon16, released [CC0](https://github.com/basveeling/pcam#license), so it is public
domain. They are excluded because they are *generated derivatives*: `build_fixture.py` produces
them deterministically from a local campaign checkout, and a repository does not need to carry
regenerable multi-megabyte binary artefacts alongside the script that makes them (see
`.gitignore`). `reproduce.py` will:

1. Use the fixture parquet files if they are already present, or build them from a local
   `~/histopath-cancer-detection` campaign checkout if one is found (pass `--campaign PATH`
   otherwise) — this is read-only over that directory.
2. Run the full evaluation (`B=2000`, seed `1337`) and write `metrics.parquet`, `fragility.parquet`,
   `calibration.parquet`, `coverage.parquet`, `manifest.json`, `findings.json`, and `report.html`
   to `examples/pcam/out/`.
3. Compare every artefact byte-for-byte against the frozen baseline in `tests/golden/pcam/` and
   report PASS/FAIL per file.

If neither the fixture files nor the campaign checkout is available, the script says so and
exits non-zero rather than guessing — the same refusal-over-guessing rule the harness applies to
its own inputs (see `docs/spec.md` §8).

## Second example: GUSTO-I, a different cohort shape

`examples/gusto/` runs predval against GUSTO-I, a public acute-MI thrombolytic trial (40,830
patients). It is structurally the opposite of PCam in almost every way that matters: tabular, not
imaging; a single prespecified logistic, not a 15-model roster; geographic external validation
across trial regions, not a scanner/stain subgroup — the same shape as validating a model against
a cohort it was never fit on.

- **Path taken: A** (a region column exists and resolves to an unambiguous subset — see
  `docs/DELTA-gusto.md` for the exact evidentiary basis, which is convergent row-count evidence,
  not a literal codebook quote). The standard 8-predictor 30-day-mortality logistic (age, sex,
  diabetes, hypotension, tachycardia, high-risk MI, shock, time-to-relief) is fit on the trial's
  West region only (n=2,188) and scored on every other region (n=38,642) — five of the eight
  predictors ship pre-derived in the raw data and were empirically verified against other raw
  columns before being trusted, not assumed from column-name guesses (a third-party mirror's
  auto-generated description of this dataset turned out to be wrong about two of them).
- Source: `https://hbiostat.org/data/repo/gusto.rda`, fetched 2026-08-23, sha256
  `e12bc58730894fa26f31b5b4ea963a878e855d7f2d53e47991cf8bb80b84d8e1`.
- Model: `gusto_west_refit_logistic`, one model, one roster entry.
- Clustering by trial region — **exactly `G = 15` clusters** among the scored cohort (16 region
  codes exist in the full trial; the West development region is removed, leaving 15 non-West
  regions) — rather than by an imaging artefact, the same mechanism as PCam's slide clustering
  applied to a completely different kind of correlation. Every reported interval, AUROC included, is
  a cluster-level percentile bootstrap interval (whole regions resampled, `B = 2000`; see
  `docs/spec.md` §5.1).
- Unlike PCam, this cohort declares **no subgroups** and produces **zero** boundary predictions
  (a fitted logistic never lands exactly on 0 or 1) — a useful contrast to PCam's 732.

### Reproducing the example

```bash
uv sync --group examples
uv run python examples/gusto/reproduce.py
```

Same non-sensitivity framing as PCam: `examples/gusto/cohort.parquet` and `predictions.parquet`
are generated derivatives of a public dataset, excluded from git for size and provenance, not
because GUSTO-I is restricted. GUSTO now has a **golden-file baseline** at `tests/golden/gusto/`,
matching PCam's — the full B=2000 run is frozen and regression-checked by
`tests/test_golden_gusto.py` (opt-in via `PREDVAL_RUN_GOLDEN=1`), so a silent shift in the example's
headline numbers fails loudly.

**Honest read of the verdict:** AUROC 0.791, comfortably inside the literature's expected
0.72–0.82 range on the first attempt. Calibration slope 0.835 shows real, if modest, displacement
scoring off the development region — the recalibration ladder finds a small but statistically
real gain at rungs 2–3 (both exclude zero), while rung 1 alone is actually slightly negative. This
is a much milder miscalibration story than PCam's two badly-miscalibrated members, which is an
honest property of this cohort rather than something to chase. The real finding of this example
isn't the numbers — it's that predval's ingestion, coverage, roster, clustering, and ladder code
all ran end to end on a genuinely different cohort shape with **zero** changes to `src/predval`.
That's evidence the contract generalizes past the one fixture it was built against; it is not, on
its own, evidence predval is ready for a blind, operator-run external validation, since the
region split here was chosen by inspecting the data rather than prespecified by an outside party.

## Install

```bash
uv sync
uv run pytest
```

## Contracts

See [`docs/spec.md`](docs/spec.md) for the frozen v0 contracts. In short:

`predictions.parquet` is long-format — one row per (subject, model, fold, horizon):

| subject_id | model_id | fold | horizon | predicted |
|---|---|---|---|---|
| `a24ce148…` | `swin` | *null* | *null* | 0.9134 |

The central rule: **a model that did not score a subject has no row.** Absence is never
imputed as zero. predval reports coverage instead of silently filling gaps.

`cohort.yaml` declares the outcome, the clustering unit, the prespecified subgroups, and the
coverage and completeness policies that decide whether a run is allowed to proceed at all.

## Status

S0–S6 done: schema, IO, and hashing; discrimination/calibration/threshold metrics with
cluster-aware uncertainty; the recalibration ladder (rungs 0–3, apparent and cross-fitted, with
paired-gain intervals and a two-axis verdict layer); the standalone HTML report and
machine-readable `findings.json`; an adversarial ingestion suite; and frozen golden baselines for
both worked examples. 246 tests pass and `ruff` is clean; a further 2 tests — the full B=2000
golden regressions for PCam and GUSTO — are opt-in via `PREDVAL_RUN_GOLDEN=1` and skip by
default, for 248 in total. See [`docs/spec.md`](docs/spec.md) for what is implemented versus
still backlog (wild cluster bootstrap, decision curves, per-rung refit-in-replicate intervals).

## License

predval itself is Apache-2.0. The PCam example fixture derives from
[PatchCamelyon](https://github.com/basveeling/pcam), which follows Camelyon16 and is released
[CC0](https://github.com/basveeling/pcam#license) — public domain, not a restricted or sensitive
dataset. It is not committed to this repository for the reasons described above
(regenerable derivative, not a licensing or sensitivity concern).
