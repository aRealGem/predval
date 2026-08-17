# predval v0 specification

Status: **frozen for v0**. Changes to anything in this document are breaking changes and require
a `version` bump in `cohort.yaml`.

## 0. Scope

predval consumes predictions that already exist and reports whether they hold up on a cohort. It
does not produce predictions. The following are permanent non-goals, not deferred features:

- No model loading, of any format, ever.
- No training, no tuning, no covariate refitting.
- No network calls from library code.
- No server, no telemetry.

---

## 1. `predictions.parquet`

Long format. One row per (subject, model, fold, horizon).

| column | type | required | meaning |
|---|---|---|---|
| `subject_id` | string | yes | pseudonymous subject identifier; joins to the cohort table |
| `model_id` | string | yes | identifier of the model that produced the prediction |
| `fold` | string | no | `holdout`, or `oof_<k>`. Null means `holdout`. |
| `horizon` | float | no | prediction horizon; null for binary outcomes |
| `predicted` | float | yes | predicted probability, in `[0, 1]` |

### 1.1 The absence rule

> **A model that did not score a subject has NO row. Absence is never zero.**

This is the single most important rule in the contract. A missing prediction and a prediction of
0.0 are different claims about the world: one is "no opinion", the other is "confidently
negative". Imputing the first as the second silently destroys both calibration and
discrimination metrics, and it does so in the direction that flatters the model.

predval therefore never fills gaps. It measures them (§2.4) and refuses to run when coverage
falls below the declared floor.

### 1.2 Uniqueness

`(subject_id, model_id, fold, horizon)` must be unique. Duplicates are a schema violation, not a
silently-averaged convenience. Nulls in `fold` and `horizon` participate in the key as nulls,
so two rows that differ only by `fold=null` versus `fold="holdout"` are a violation — see §1.3.

### 1.3 Null `fold` normalisation

`fold` null is normalised to `holdout` on load, **before** the uniqueness check. This makes
`null` and `"holdout"` genuinely interchangeable rather than merely documented as such.

### 1.4 `predicted` bounds

`predicted` must lie in the closed interval `[0, 1]`. NaN and infinity are violations. Exactly
`0.0` and exactly `1.0` are **legal** — real models emit them — which is why every logit-scale
operation in the recalibration ladder must eps-clip first (§4).

---

## 2. `cohort.yaml`

```yaml
cohort_id: pcam-holdout-wsi-grouped
version: 0
subject_key: subject_id
data: cohort.parquet

outcome:
  type: binary
  field: label
  positive_label: 1

clustering:
  field: wsi

coverage:
  min_fraction: 0.95
  compare_on: both        # full | common | both

completeness:
  require_outcome: true
  on_violation: drop_and_report

subgroups:
  - {name: scanner_domain, field: domain}

thresholds: [0.5]
```

### 2.1 Two additions to the brief's v0 contract

Both are recorded here rather than made silently.

**`data`** — a path to a subject-level table (parquet or csv, one row per subject) carrying the
outcome and subgroup columns. The contract references `outcome.field` and `subgroups[].field` by
name, so those columns must live somewhere resolvable. The path is interpreted relative to the
`cohort.yaml` file, so a cohort directory is relocatable as a unit.

This preserves the "predictions in, no model loading" constraint exactly: the cohort table holds
outcomes and prespecified subgroup labels. It is not a feature matrix and predval never fits on
it.

**`clustering`** — optional, nullable. Names a column in the cohort table identifying the
grouping unit within which subjects are correlated.

This exists because the unit of analysis and the unit of independence are frequently not the
same. In the PCam fixture, 19,999 image patches derive from just 22 whole slides; patches from
one slide are heavily correlated. Confidence intervals computed as though there were 19,999
independent observations are not slightly optimistic, they are wrong by roughly the square root
of the cluster size. S1 validates and plumbs this field. S2 consumes it for cluster-robust or
cluster-bootstrap intervals over the grouping unit.

If `clustering` is omitted, S2 will treat subjects as independent and say so in the report.

### 2.2 Field reference

| key | required | meaning |
|---|---|---|
| `cohort_id` | yes | stable identifier for this cohort |
| `version` | yes | contract version; must be `0` |
| `subject_key` | yes | name of the subject id column in the cohort table |
| `data` | yes | path to the cohort table, relative to the yaml |
| `outcome.type` | yes | `binary` only in v0 |
| `outcome.field` | yes | outcome column in the cohort table |
| `outcome.positive_label` | yes | the value counted as a positive event |
| `clustering.field` | no | correlation grouping column |
| `coverage.min_fraction` | yes | float in `[0, 1]` |
| `coverage.compare_on` | yes | `full`, `common`, or `both` |
| `completeness.require_outcome` | yes | bool |
| `completeness.on_violation` | yes | `drop_and_report` or `fail` |
| `subgroups[].name` | no | label used in reports |
| `subgroups[].field` | no | column in the cohort table |
| `thresholds` | yes | decision thresholds, each in `[0, 1]` |

### 2.3 Subgroups are prespecified only

The `subgroups` list is the complete set of subgroup analyses predval will perform. There is no
API for discovering subgroups from the data, and this is deliberate: subgroup analysis chosen
after seeing outcomes is how a null result becomes a press release. Declaring subgroups in a
file that is hashed (§3) makes prespecification checkable after the fact.

`outcome.field`, `clustering.field`, and every `subgroups[].field` must be distinct from each
other and must exist in the cohort table.

### 2.4 Coverage

Coverage is per-model: the fraction of cohort subjects for which that model has a prediction.

- `full` — denominator is every subject in the cohort.
- `common` — denominator is the subjects scored by *every* model, i.e. the intersection.
- `both` — report both. This is the recommended setting, because the gap between the two
  numbers is exactly the information a single number hides.

A model whose coverage falls below `min_fraction` is reported as a coverage violation. Metrics
computed on `full` and on `common` are not comparable across models with different coverage;
`both` is what makes that non-comparability visible rather than latent.

### 2.5 Completeness

`require_outcome: true` means subjects with a missing outcome cannot contribute.
`on_violation: drop_and_report` drops them and records the count and the ids dropped;
`fail` refuses to run. Dropping is never silent.

---

## 3. Hashing and provenance

Every input file is hashed with **blake2b, 32-byte digest**, read in 1 MiB chunks. Hashing is a
pure function of file bytes: no timestamps, no paths, no environment. The same file hashed twice
in the same process, or in two different processes, yields the same digest.

The purpose is claim-checking. A validation report asserts something about specific inputs; the
digests are what let a reader confirm months later that the inputs were the ones named.

---

## 4. Recalibration ladder

**Specified in S1, implemented in S3.** Documented now so the rungs are fixed before any results
exist to be tempted by.

Let `p` be the published probability and `z = logit(clip(p, 1e-6, 1 - 1e-6))`.

| rung | model | fitted parameters | question it answers |
|---|---|---|---|
| **rung0** | `p` as published | none | does it work as shipped? |
| **rung1** | `σ(z + a)` | intercept `a` | is it just miscalibrated in prevalence? |
| **rung2** | `σ(b·z + a)` | intercept `a`, slope `b` | is it also over- or under-confident? |
| **rung3** | `σ(f(z))`, `f` = logistic on a natural cubic spline of `z`, `df=4` | spline coefficients | is the miscalibration non-monotone in the score? |

### 4.1 The eps-clip is load-bearing from rung1

`logit(p)` is undefined at `p = 0` and `p = 1`, and real models emit both. In the PCam fixture,
one member emits exactly `1.0` for 535 of 19,999 subjects. Clipping to `[1e-6, 1 - 1e-6]` is
therefore required before *any* rung above rung0, not merely for the spline in rung3.

### 4.2 Covariate refit is out of contract

The ladder corrects the link-scale mapping of an existing score. It never adds covariates,
never reweights subjects, and never touches the model. A "recalibration" that consumes patient
features is a new model, and validating a model against the cohort it was just fitted on is the
error this entire harness exists to prevent.

### 4.3 Ladder results are diagnostic, not promotional

A rung that improves calibration reports how much miscalibration was repairable in principle. It
is not a claim that the recalibrated model is validated — that would require a held-out cohort
the correction was not fitted on. Reports must present rungs 1–3 as diagnosis of rung0.

---

## 5. Failure behaviour

predval refuses to run on schema violation rather than degrading. Every error names the file,
the column, and where practical an offending value and row count. A harness that guesses is
worse than no harness, because its output looks the same either way.
