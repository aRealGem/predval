# DELTA: GUSTO-I second worked example

**Date:** 2026-08-23. **Status:** Path A taken, complete. No portability fixes required.

## Source and provenance

`https://hbiostat.org/data/repo/gusto.rda`, fetched 2026-08-23. sha256
`e12bc58730894fa26f31b5b4ea963a878e855d7f2d53e47991cf8bb80b84d8e1`. 40,830 rows, 29 columns, as
fetched (`examples/gusto/fetch_data.py` logs the full column list on every run). GUSTO-I is a
public thrombolytic-strategy trial in acute MI; PatchCamelyon derives from Camelyon16 (CC0), and
GUSTO-I's public teaching release carries the same non-sensitivity profile -- this file is not
committed for size/provenance reasons (see `.gitignore`), not because the data is restricted.

## Path decision

`make_predictions.detect_path()` checked for a region/site-like column and found `regl`, a
16-level `int32` code. `regl == 1` has exactly 2,188 rows.

That number is not read off a codebook label -- `pyreadr` carries no variable-label metadata for
this object (`df.attrs` is empty, confirmed empirically), so there is no literal "1 = West" text
shipped with the file. The evidentiary basis for treating `regl == 1` as the West subset is
**convergent, not a citation**:

1. The session brief independently stated "n=2188 West subset" before this script ever ran.
2. A web search corroborated n=2188 as a real, specific, independently published figure in the
   external-validation methodology literature for exactly this kind of GUSTO-I teaching exercise
   ("a GUSTO-I dataset with n=2188 patients has been used to model the binary endpoint of 30-day
   survival using clinical covariates").
3. `regl`'s 16 levels are far finer than a simple US/non-US split (a different, separately
   published GUSTO-I partition uses non-US n=17,796 / US n=23,034 -- summing to the same 40,830
   total, so both partitions are real and consistent with each other, just at different
   granularities); a single code among 16 resolving to exactly the brief's cited number is a
   specific enough coincidence to trust.

This is stated plainly rather than dressed up as a codebook quote, per the session's own
portability-fix discipline: don't assert more than the evidence supports.

## Predictor verification (the actual finding of this section)

A third-party mirror's auto-generated description of this dataset (fetched via a web tool, not
this repo's own code) claimed `hrt` = "hormone replacement therapy" and `hig` = "high blood
pressure." Both are implausible for a 1990s all-comers AMI trial's core risk model and were
**empirically refuted** by cross-tabulating against other raw columns in the same fetched table,
before any of it was trusted:

| flag | hypothesis | check | result |
|---|---|---|---|
| `sho` | shock (Killip III/IV) | crosstab vs `Killip` | exact: `sho==1` iff Killip in {III, IV} |
| `sho` | -- | mean `day30` by `sho` | 41.9% (sho=1) vs 6.2% (sho=0) -- shock-scale mortality |
| `hrt` | tachycardia, pulse>80 | `(pulse>80) == hrt` | 99.98% match |
| `hyp` | hypotension, sysbp<100 | `(sysbp<100) == hyp` | 99.99% match |
| `hig` | high risk (anterior or prior MI) | crosstab vs `ant`, `pmi` | `hig==0` iff `ant==0` and `pmi==no`, exactly |
| `dia` | diabetes | prevalence | 14.7%, consistent with AMI-trial diabetes prevalence |
| `ttr` | time-to-relief>1h | unique values | already `{0, 1}` -- pre-derived, not continuous |

Five of the eight standard predictors (`sho`, `hig`, `dia`, `hyp`, `hrt`) and `ttr` ship
pre-derived in the raw table, matching Steyerberg's/Harrell's standard GUSTO-I teaching model
exactly. Only `age65` (`age > 65`) and `female` (`sex == "female"`) needed deriving.

### Codebook re-check (2026-08-24)

A follow-up session tried to upgrade the wording above from empirical-inference to
documentary-with-citation by locating an official variable codebook via `https://hbiostat.org/data`.
Outcome: **no authoritative codebook was found, and the inferential wording stands** — deliberately,
not for lack of trying.

- The hbiostat index page and the per-dataset URLs return no machine-readable variable dictionary
  for this object; as already noted, `gusto.rda` itself carries no label metadata.
- The one accessible "data dictionary" — the `predtools` R package's `gusto` help page
  (`search.r-project.org/CRAN/refmans/predtools/html/gusto.html`, 40,830 rows × 29 variables,
  matching this file) — is **the very source this section refuted**: it defines `hrt` as "hormone
  replacement therapies" and `hig` as "high blood pressure," both of which the cross-tabs above
  disprove for this dataset (`hrt` matches `pulse > 80` at 99.98%; `hig == 0` iff `ant == 0 and
  pmi == no`). It does not document the region variable at all.
- An independent web-search snippet did corroborate `hrt = tachycardia (Yes=1/No=0)`, agreeing with
  the cross-tab and against the R-package label — but from an unverifiable aggregate, so it is
  treated as weak corroboration, not a citation.

Net: the empirical cross-tabulation is a **stronger** basis than the only "official" text on offer,
which is wrong. The West = `regl == 1` mapping likewise remains convergent-evidence, not codebook-
confirmed (no source consulted documents `regl`'s levels). Nothing here is upgraded to a codebook
quote, in keeping with the don't-assert-more-than-the-evidence rule.

## Cohort construction

| raw GUSTO column | predval field | notes |
|---|---|---|
| (row index) | `subject_id` | synthesized `gusto_NNNNNN` -- no ID column exists in the raw table |
| `day30` | `label` / `outcome.field` | 30-day mortality, 1 = death |
| `regl` | `clustering.field` | region code; **exactly 15 clusters** in the scored cohort (16 region codes exist in the full trial, the West development region `regl == 1` is removed, leaving `G = 15` distinct non-West regions) |
| `age`, `sex`, `dia`, `hyp`, `hrt`, `hig`, `sho`, `ttr` | (sidecar-only) | inputs to the Path A logistic fit; predval never sees these columns or the fitted model |

Model fit on `regl == 1` (West, n=2,188) via `statsmodels.Logit`, no regularization, all eight
predictors significant or near-significant in the expected direction (shock has the largest
coefficient, 2.39, consistent with cardiogenic shock being the strongest single AMI mortality
predictor in the clinical literature). Scored on all `regl != 1` patients (non-West, n=38,642,
2,716 events, prevalence 7.03%).

## Config that sufficed (no code change)

Every decision below required zero `src/predval` changes -- each is already-declared-optional
contract surface, confirmed against `schema.py` before writing this cohort:

- **No new `cohort.yaml` keys.** `clustering.field: regl` uses the existing optional
  `ClusteringSpec`.
- **Roster of one.** `expected_models: [gusto_west_refit_logistic]` -- `check_roster` has no
  special case for a single-model roster; declared/present/absent reconciled cleanly (1/1/0).
- **No subgroups declared.** The `subgroups` key is entirely absent from `cohort.yaml`; nothing
  in `io.py`/`schema.py` requires it.
- **`horizon = 30.0` throughout.** The first real (non-null) use of this column in either
  example; no special handling was needed -- it flows through as an ordinary float column.
- **Single-model coverage.** Because the sidecar scores every non-West subject by construction,
  `coverage_full == coverage_common == 1.0` for the one model; the `informative-selection`
  caution never fires and the `cov_delta` column collapses, exactly as `docs/spec.md` §2.4
  describes for identical-coverage cohorts.

## Portability fixes made

**None.** Ingestion, coverage checking, roster reconciliation, cluster-aware uncertainty (G=15),
the recalibration ladder, fragility, and report/findings rendering all ran end to end against
this genuinely different cohort shape (tabular, single model, real geographic clustering, a real
time horizon) without a single `src/predval` change. Per the portability-fix rule
(`docs/spec.md`-compliant input rejected/mishandled -> real bug; anything else -> config), there
was nothing to escalate. This is not a surprising result -- `io.py`/`schema.py` already declare
clustering, subgroups, and the model roster as optional, and nothing in the actual validation
logic assumes PCam's specific scale (15 models, 22 slides) rather than the general contract.

## Evaluation results

Common-subset, overall stratum. **Every reported 95% interval below, AUROC included, is a
cluster-level percentile bootstrap interval**: whole `regl` regions (the `G = 15` clusters) are
resampled with replacement, `B = 2000`, and the interval is the percentile interval at the 95% level
(`docs/spec.md` §5.1) — not a normal-theory or row-level interval.

| metric | value | 95% CI |
|---|---|---|
| AUROC | 0.7908 | [0.7829, 0.7980] |
| Average precision | 0.2656 | [0.2512, 0.2854] |
| Calibration slope | 0.835 | [0.801, 0.870] |
| Calibration intercept | 0.092 | [0.011, 0.173] |
| Brier (rung0, as-published) | 0.05889 | -- |
| Brier skill score (BSS, best admissible rung) | 0.1051 | [0.0947, 0.1155] |

Best admissible rung: **rung2** (S6: corrected from an earlier **rung3** read under a prior
selection rule that picked the rung with the lowest cross-fitted Brier *point estimate*; the
current rule -- docs/spec.md §4.8 -- picks the **lowest** rung whose paired cross-fit gain interval
excludes zero on the improvement side, since rung2 already clears that bar there is no reason to
prefer rung3's slightly larger, but not significantly different, point estimate). Paired cross-fit
Brier gains (rung0 vs. held-out rung, slide-analogue cluster bootstrap): rung1 **-0.000093**
[-0.000177, -0.000010] (a negative gain -- intercept-only correction is not merely useless here, its
interval excludes zero on the negative side); rung2 **+0.000410** [0.000293, 0.000527]; rung3
**+0.000539** [0.000350, 0.000735]. Both rung2 and rung3 gains exclude zero on the positive side,
so both are legitimately admissible -- rung2 is reported because it is the lower (simpler) of the
two, not because rung3 is wrong. `boundary_count = 0` (a fitted logistic never produces
exact 0/1 outputs, unlike PCam's neural-network members). Fragility: largest AUROC change from
dropping any single region is 0.0021 (region 7) -- low, no single site dominates the result. One
flag fired: `few_clusters` (G=15, below the 40-cluster threshold `docs/spec.md` §5.1 names for
tail-quantile reliability) -- working as documented, not a defect.

## Literature sanity check

Expected AUROC range (session brief, from published GUSTO-I 8-predictor model literature):
0.72-0.82. Observed: **0.7908**, comfortably inside range on the first attempt -- no recode-refit
iteration was needed (the session's timebox cap was 2 iterations; 0 were used). Calibration slope
0.835 shows real, non-trivial displacement (a model that discriminates well on its own development
cohort loses calibration, not discrimination, when scored on a geographically different cohort --
exactly the failure mode external validation exists to catch, and exactly what predval's ladder
is built to diagnose rather than paper over).

## Verdict: does this clear the bar for a second worked example

**Yes, for what it actually demonstrates: ingestion generality.** predval's contract and code
handled a cohort that differs from PCam in nearly every structural dimension that matters --
tabular versus imaging, a single prespecified logistic versus a 15-model ensemble, 15 loosely-held
geographic clusters versus 22 tightly-defined imaging slides, a real time horizon versus an
undefined one, zero boundary predictions versus 732 -- without a single line of `src/predval`
changing. That is real evidence the predictions-only contract generalizes past the one cohort it
was developed against, which is precisely what a "one external adopter" success test would
need to be true before it is worth attempting.

**No, for what it does not demonstrate.** This is still two author-selected cohorts, not an
independent operator running predval on data they collected. The "external validation" here is
retrospective and synthetic in the sense that the West/non-West split was chosen by inspecting
the data, not prespecified before looking -- a genuinely blind external validation would need
the cohort holder, rather than the harness's author, deciding the split. Path B (transcribed
published coefficients, zero refitting) was never exercised, so the "given someone else's model
with no ability to refit" case -- arguably the harder and more realistic one -- remains
untested. And a single, mild, real miscalibration finding (calibration slope 0.835, small but
real Brier gains) is a much less dramatic demonstration of the recalibration ladder's value than
PCam's two badly-miscalibrated members; that is an honest property of this cohort, not something
to be tuned away, but it does mean GUSTO-I alone is a weaker "look what the ladder catches" pitch
than PCam.

**Net read.** This exercise is evidence that ingestion generalizes across cohort shapes, but it
is **not** sufficient evidence that predval is ready for a blind, operator-run external
validation. That gap is structural -- an author-selected split is not external operation -- and a
third cohort of the same kind would not close it either. Closing it requires a cohort holder who
is not the harness's author choosing the split and running the tool.

## Options not pursued

- Path B (transcribed Steyerberg coefficients, no refit) -- Path A succeeded cleanly, so Path B's
  placeholder in `make_predictions.py` was deliberately left un-implemented rather than filled
  with invented-looking coefficients. A real Path B exercise (a genuinely un-refittable published
  model) is closer to the genuinely un-refittable shape and remains worth exercising.
- A golden-file freeze for GUSTO, matching PCam's `tests/golden/pcam/` -- **done** (2026-08-25):
  the B=2000 run is frozen at `tests/golden/gusto/` and regression-checked by
  `tests/test_golden_gusto.py` (opt-in via `PREDVAL_RUN_GOLDEN=1`), reproducibility verified at
  freeze time.
- A third cohort with a genuinely competing-risks or survival outcome, to stress binary-only's
  actual boundary (see README's `SurvivalEVAL` scope note) -- not attempted.
