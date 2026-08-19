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

If `clustering` is omitted, S2 treats subjects as independent and says so in the report.

**Clustering is not a subgroup, even when the column is the same.** `clustering.field` names the
unit of *independence*: it changes how uncertainty is computed and never appears as a reporting
stratum. `subgroups[].field` names a unit of *reporting*: it splits results into strata and never
changes how uncertainty is computed within one. Declaring the same column as both is rejected
(§2.3), because the two roles answer different questions and conflating them silently produces
intervals that look stratified but are not.

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
| `coverage.common_warn_frac` | no | caution threshold for intersection loss (default `0.20`) |
| `completeness.require_outcome` | yes | bool |
| `completeness.on_violation` | yes | `drop_and_report` or `fail` |
| `subgroups[].name` | no | label used in reports |
| `subgroups[].field` | no | column in the cohort table |
| `thresholds` | yes | decision thresholds, each in `[0, 1]` |
| `expected_models` | no | declared model roster (§2.6) |
| `on_missing_model` | no | `warn` or `fail` (default `warn`) |
| `uncertainty.n_boot` | no | bootstrap replicates (default `2000`) |
| `uncertainty.seed` | no | bootstrap seed, recorded in the manifest (default `1337`) |
| `uncertainty.ci_level` | no | interval level (default `0.95`) |
| `uncertainty.show_naive_ci` | no | render the unit-of-analysis exhibit (default `true`) |

### 2.3 Subgroups are prespecified only

The `subgroups` list is the complete set of subgroup analyses predval will perform. There is no
API for discovering subgroups from the data, and this is deliberate: subgroup analysis chosen
after seeing outcomes is how a null result becomes a press release. Declaring subgroups in a
file that is hashed (§3) makes prespecification checkable after the fact.

`outcome.field`, `clustering.field`, and every `subgroups[].field` must be distinct from each
other and must exist in the cohort table.

### 2.4 Coverage

Coverage is per-model: the fraction of cohort subjects for which that model has a prediction.

- `full` — the analysis set is every subject in the cohort.
- `common` — the analysis set is the subjects scored by *every* model, i.e. the intersection.
- `both` — the recommended setting.

A model whose coverage falls below `min_fraction` is a coverage violation.

**`compare_on` is a hierarchy, not a duplication.** Reporting every number twice doubles what a
reader must hold in their head and invites them to quote whichever is more flattering. So:

- **`metrics.parquet` always contains both subsets**, `subset ∈ {full, common}`, regardless of
  `compare_on`. The machine-readable artefact is complete; the report is what is opinionated.
- **The report's primary comparison table is the `common` subset**, because it is the only
  subset on which models are comparable to each other. Each model carries a
  `cov_delta = metric_full - metric_common` column, so the effect of restriction is visible
  without a second full table.
- **Full-cohort per-model detail goes to an appendix.** It answers "how does this model do on
  everyone?", which is a per-model question, not a comparison.
- **When coverage is identical across all models**, `full` and `common` are the same set. The
  report renders once, notes `full == common`, and suppresses the delta column rather than
  printing a column of zeros.

**Informative-selection caution.** When the common subset excludes more than
`coverage.common_warn_frac` (default 0.20) of otherwise usable rows, the report raises:
*"selection into the common subset may be informative."* The subjects every model happened to
score are not a random sample of the cohort — if models decline to score the hard cases, the
common subset is the easy cases, and every model looks better on it.

### 2.5 Completeness

`require_outcome: true` means subjects with a missing outcome cannot contribute.
`on_violation: drop_and_report` drops them and records the count and the ids dropped;
`fail` refuses to run. Dropping is never silent.

### 2.6 Model roster

`expected_models` is an optional list of `model_id`s the cohort expects to find.

Absence is the failure mode this whole contract is built around (§1.1), and a model missing
*entirely* is the same error one level up: predval would otherwise report on whatever happens
to be in the file and say nothing about what is not. The roster makes absence loud.

It is checked in **both directions**:

| condition | meaning | behaviour |
|---|---|---|
| declared but absent | an expected model produced no rows at all | flag; `fail` if `on_missing_model: fail` |
| present but undeclared | a `model_id` nobody expected — usually a typo | flag (always a warning) |
| key omitted | no roster declared | report line: *"model roster not declared"* |

The present-but-undeclared direction matters more than it looks: a mistyped `model_id` produces
a model with plausible-looking metrics that corresponds to nothing, while the model it was meant
to be silently reads as absent.

Flags are written to the report header and to `manifest.flags`.

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

### 4.3 Apparent and cross-fitted, with the optimism reported

Every rung is computed twice, and `metrics.parquet` carries `fit_mode ∈ {apparent, crossfit}`.

**Apparent** fits the correction on the same rows it is evaluated on. Its improvement over rung0
is optimistic by construction, and it is retained because it answers a real question: how much
miscalibration is present *at all*.

**Cross-fitted** uses grouped K-fold with `K = min(5, G)`, where `G` is the number of clusters.
**Folds respect `clustering.field`** — a whole cluster is held out together, never split. Fitting
a correction on some patches of a slide and evaluating it on other patches of the same slide
would leak, and would reproduce in miniature exactly the error predval exists to detect.

**Optimism** is reported explicitly as `apparent - crossfit` per rung. It is the number that
says how much of the apparent repair was the correction memorising this particular cohort.

### 4.4 Framing rule

Report templates must state, verbatim in substance:

> No rung is a validated model. Rung0 is the model as published. Rungs 1–3 are diagnosis of
> rung0. The cross-fitted figures are the expected performance *after local recalibration* on a
> cohort like this one — not evidence that the recalibrated model has been validated, which
> would require a cohort the correction was never fitted on.

A recalibration that improves the numbers is a finding about the cohort as much as about the
model, and the report must not let it read as a promotion.

### 4.5 Rung intervals are deferred

`rung0` carries the full interval layers of §5. The recalibrated rungs (`rung1`–`rung3`) are
emitted as **point estimates with null `ci_*`** in this version. A correct interval for a
recalibrated metric must refit the correction inside every bootstrap replicate — and, for the
cross-fitted rungs, refit it inside every replicate *and* every fold — so that the interval
carries the variance of the correction itself, not just of the metric. That is a real cost and a
design surface of its own, so it is backlog rather than a silent omission. The load-bearing S3
number is the **optimism gap** `apparent − crossfit`, which a point estimate already delivers.

---

## 5. Uncertainty

Intervals are **layered**: one default method everywhere, analytic cross-checks where the
structure allows, and a separate fragility view that is deliberately not an interval.

### 5.1 Default: cluster percentile bootstrap

For every metric, resample **whole `clustering.field` groups with replacement**, `B` =
`uncertainty.n_boot` (default 2000), seeded from `uncertainty.seed` and recorded in the
manifest. The interval is the percentile interval at `uncertainty.ci_level`.

Resampling clusters rather than rows is the entire point: it propagates the within-cluster
correlation that makes row-level intervals wrong.

When the number of clusters `G < 40`, the report notes: *"tail quantiles approximate with few
clusters."* The PCam fixture has `G = 22`, so this note fires there — a bootstrap over 22 units
cannot resolve its own tails finely, and saying so is better than implying a precision the data
does not carry.

If `clustering` is not declared, the bootstrap resamples rows and the report says so.

### 5.2 Analytic cross-checks

Written to `metrics.parquet` alongside the bootstrap rows, distinguished by `ci_method`:

| metric | analytic method | `ci_method` |
|---|---|---|
| Brier | clustered mean, `t(G-1)` | `cluster_robust_t` |
| calibration intercept | logistic fit, cluster-robust covariance, `t(G-1)` | `cluster_robust_t` |
| calibration slope | logistic fit, cluster-robust covariance, `t(G-1)` | `cluster_robust_t` |

The report body shows the bootstrap interval and **footnotes whether the analytic interval
agrees or diverges**. Two methods that disagree are information, not a problem to hide: it
usually means the cluster count is too small for one of them to be trusted.

### 5.3 Fragility, not an interval

Leave-one-cluster-out appears **only** in a *Fragility* section, and is never rendered as a
confidence interval. For each metric predval reports the maximum `|Δ|` across dropping each
cluster in turn, together with the id of the culprit cluster.

This answers a question a confidence interval cannot: *is this result carried by one slide?* A
result whose AUROC moves by 0.04 when one of 22 slides is removed is fragile in a way that a
tight bootstrap interval will not reveal.

### 5.4 The unit-of-analysis exhibit

For AUROC, the report renders the **naive per-row interval beside the cluster interval, once**
(`uncertainty.show_naive_ci`, default true). This is didactic on purpose. The naive interval is
wrong, and showing how much narrower it is makes the cost of ignoring clustering concrete rather
than theoretical. It is labelled as incorrect and appears exactly once, not per metric.

### 5.5 Backlog

Wild cluster bootstrap is noted for a future version. It is the better tool when `G` is very
small, but it is not in the MVP.

---

## 6. Output artefacts

### 6.1 `metrics.parquet`

One row per (model, subset, stratum, metric, threshold, rung, fit_mode, ci_method).

| column | meaning |
|---|---|
| `model_id` | model |
| `subset` | `full` or `common` — always both present (§2.4) |
| `stratum_kind` | `overall` or `subgroup` |
| `subgroup_name`, `subgroup_level` | null for `overall` |
| `metric` | metric name |
| `threshold` | null except for threshold-dependent metrics |
| `rung` | `rung0` (as published) or `rung1`/`rung2`/`rung3` (§4). **S3 addition** |
| `fit_mode` | `apparent` or `crossfit`; `apparent` for every `rung0` row |
| `value` | point estimate |
| `ci_low`, `ci_high`, `ci_level` | interval, null where none applies |
| `ci_method` | `cluster_bootstrap`, `cluster_robust_t`, `naive_row_bootstrap`, or null |
| `n`, `n_events`, `n_clusters` | analysis-set sizes |

**`rung` is an S3 addition to the frozen §6.1 columns** — §4.3 requires rung rows in this table,
and the original column list omitted the discriminator. Every as-published metric (including the
rank and threshold metrics and the data-quality rows) is `rung0`/`apparent`; the ladder adds
`rung1`–`rung3` rows for the calibration-sensitive metrics (`brier`, `calibration_intercept`,
`calibration_slope`), each in both fit modes. Optimism is `apparent − crossfit` per rung, derived
from these paired rows rather than stored as its own row.

### 6.2 `fragility.parquet`

One row per (model, subset, stratum, metric): `max_abs_delta` and `culprit_cluster` (§5.3).

### 6.3 `manifest.json`

Input digests (§3), the resolved configuration, the bootstrap seed and `B`, and `flags` — the
roster findings (§2.6), the informative-selection caution (§2.4), and the few-clusters note
(§5.1). The manifest is what makes a report's claims checkable later.

---

## 7. Data quality

Reported per model, before any metric:

- **`boundary_count`** — the number of predictions exactly equal to 0 or 1. These are the values
  at which `logit` is undefined, so this count is the direct measure of how much the eps-clip
  (§4.1) is doing. In the PCam fixture the total is **732**, concentrated in a few members.
- **`n`, `n_events`, `prevalence`, `n_clusters`** for the analysis set.

The eps value is `1e-6`, clipping to `[1e-6, 1 - 1e-6]`. It is chosen to be far smaller than any
plausible probability resolution while keeping `logit` finite: at the clip, `logit` is roughly
`∓13.8`, which is extreme enough to preserve the ordering of confident predictions but finite
enough not to dominate a fitted slope. A model emitting many boundary values is being told
something about itself, which is why the count is reported rather than silently absorbed.

---

## 8. Failure behaviour

predval refuses to run on schema violation rather than degrading. Every error names the file,
the column, and where practical an offending value and row count. A harness that guesses is
worse than no harness, because its output looks the same either way.
