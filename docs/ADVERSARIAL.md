# Adversarial ingestion: what happens when you feed predval garbage

**Date:** 2026-08-24. Companion to `tests/adversarial/` (generator `make_corruptions.py`, suite
`test_ingestion.py`). This is the ingestion *door*: predval must fail loudly, legibly, and
actionably on hostile input, and must **never** emit a report, findings, or metrics file from a
corrupted input.

Every case below was generated from the clean GUSTO-I inputs (a deterministic small slice, so the
suite runs fast and in a clean checkout where the git-ignored GUSTO derivatives are absent), fed to
the loaders, and — for failure cases — run end to end through the pipeline runner to confirm no
artefact is written and the process exits non-zero. "Observed-before" is the real behavior recorded
by probing `main` at commit `774599d` before any fix in this session.

Design rule held throughout: fixes live **only** in the ingest/validation layer (`src/predval/io.py`
loaders + error-message text). Zero changes to statistics, ladder, bootstrap, or report content. The
GUSTO and PCam golden artefacts are byte-identical after this work (none of the new checks fire on a
valid binary-outcome, non-null-cluster, non-empty cohort).

## The five ingest-layer fixes made this session

1. **Header-only / empty-of-rows predictions now reject.** A file with a valid header and zero data
   rows used to load "successfully" as an empty predictions table, which would then flow into an
   empty, degenerate report. It now fails at load.
2. **Out-of-range predictions get a logit hint.** The `[0, 1]` range error now says, in words, that
   values outside the range may be log-odds and should be passed through a sigmoid first.
3. **Null cluster ids now reject.** A null in `clustering.field` used to flow silently into the
   cluster bootstrap. It now fails at load with a count.
4. **Non-binary / single-class outcomes now reject.** An outcome column with a third value (e.g. a
   `-9` missing-value sentinel) used to be silently folded into the non-event class; an all-positive
   outcome used to load with no negative class at all. Both now fail at load, naming the distinct
   values found.

Two things the probe showed were **already handled** and needed no change: a UTF-8 BOM (pandas
strips it, columns parse correctly) and a padded value like `" 0.73 "` (parsed as `0.73`).

## a. Renamed columns

- **`renamed_pred_col`** (`predicted` → `pred_prob`). Expected: hard fail. Observed-before: hard
  fail, unchanged — locked. Message: *"predictions is missing required columns: ['predicted'] —
  columns present: [...]"*.
- **`renamed_outcome_col`** (`label` → `outcome_30d`, yaml still declares `label`). Expected: hard
  fail. Observed-before: hard fail — locked. Message: *"cohort table is missing columns declared in
  cohort.yaml: ['label']"*.

## b. Missing values (NA) in prediction / outcome / cluster

- **`na_prediction`**. Expected: hard fail (the absence rule §1.1 — a subject a model did not score
  must have no row, not a null). Observed-before: hard fail — locked. Message: *"predicted contains
  nulls or NaN; a subject a model did not score must have no row at all (docs/spec.md section 1.1)"*.
- **`na_outcome`**. Expected: documented drop, not failure — `completeness.on_violation:
  drop_and_report`. Observed-before: drops the subject and records it in `dropped_subjects` —
  locked. This is the one place a loader is allowed to drop rather than reject, and it reports
  exactly what it dropped.
- **`na_cluster`**. Expected: hard fail. Observed-before: **loaded silently** with a NaN cluster —
  **fixed**. Message: *"clustering column contains nulls; every subject must belong to a cluster,
  because a null cluster cannot be resampled by the cluster bootstrap"*.

## c. Duplicate subject ids

- **`dup_exact`** (byte-identical duplicate row) and **`dup_conflicting`** (same
  `(subject_id, model_id, fold, horizon)` key, different `predicted`). Expected: both hard fail —
  predval will not silently average two claims about the same subject. Observed-before: both hard
  fail via the uniqueness check — locked. Message: *"duplicate (subject_id, model_id, fold, horizon)
  rows; predval will not silently average them"*, with the offending key shown.

## d. Out-of-range predictions

- **`pred_negative`** and **`pred_gt1`**. Expected: hard fail. Observed-before: hard fail — locked.
  Message: *"predicted must be a probability in [0, 1]; ..."* with offending examples.
- **`pred_bulk_zero_one`** (half exactly `0.0`, half exactly `1.0`). Expected: **legal** — real
  models emit exact 0 and 1; the recalibration ladder eps-clips (`EPS = 1e-6`) and `boundary_count`
  reports how many there were. Observed-before: loads, values preserved — **locked with a test** (no
  warn is emitted at ingest by design; the count surfaces downstream as `boundary_count`). Not
  changed.

## e. Logit-scale predictions

- **`pred_logit_scale`** (probabilities mapped to log-odds, range ≈ `[-6, 6]`). Expected: range
  rejection with a hint that these look like logits. Observed-before: hard fail on range, but the
  message did not mention logits — **message improved**. Message: *"predicted must be a probability
  in [0, 1]; values fall outside [0, 1] — if these are log-odds/logits, apply a sigmoid before
  writing predictions"*.

## f. String floats

- **`str_comma_decimal`** (`"0,73"`). Expected: hard fail. Observed-before: hard fail (non-numeric,
  the comma-decimal is not parsed as `0.73`) — locked. Message: *"predicted contains values that are
  not numeric"*, examples `['0,73']`.
- **`str_padded`** (`" 0.73 "`). Expected: parse to `0.73`. Observed-before: loads as `0.73` (pandas
  strips surrounding whitespace) — locked, no change.
- **`str_NULL`**, **`str_NaN`**, **`str_empty_value`**. Expected: treated as missing → hard fail.
  Observed-before: pandas reads `NULL`/`NaN`/empty as NaN, so all three hit the null rule and hard
  fail — locked. Message: the §1.1 nulls-or-NaN error.

## g. Encoding damage

- **`enc_utf8_bom`**. Expected: load. Observed-before: **loads** — pandas already strips a UTF-8 BOM
  and parses the first column name correctly, so no fix was needed (contrary to the plan's guess) —
  locked.
- **`enc_crlf`**. Expected: load. Observed-before: loads (universal newlines) — locked.
- **`enc_utf16`**. Expected: hard fail. Observed-before: hard fail — a UTF-16 file is not decodable
  as UTF-8 and the read error names the file — locked. Message: *"predictions file could not be
  read: 'utf-8' codec can't decode byte 0xff ..."*.

## h. Outcome pathologies

- **`outcome_yes_no`** (`positive_label: 1`, values `yes`/`no`). Expected: reject. Observed-before:
  reject — the integer `positive_label` never occurs among the strings — locked. Message:
  *"outcome.positive_label 1 never occurs in the outcome column; ... values present: ['no', 'yes']"*.
- **`outcome_true_false`** and **`outcome_float_1_0`**. Decision: **accept with coercion** — pandas
  compares `True == 1` and `1.0 == 1`, both classes are present, exactly two distinct values.
  Observed-before: loads — locked and documented (this is the deliberate coercion, not silence).
- **`outcome_third_value`** (`{1, 0, -9}`). Expected: reject. Observed-before: **loaded silently**,
  folding `-9` into the non-event class — **fixed**. Message: *"outcome column has 3 distinct values
  ([-9, 0, 1]); a binary outcome must have exactly two — is one of these a missing-value sentinel
  (e.g. -9)?"*.
- **`outcome_all_zero`**. Expected: reject. Observed-before: reject via the `positive_label`-never-
  occurs check — locked.
- **`outcome_all_one`**. Expected: reject (no negative class). Observed-before: **loaded silently**
  with no non-event — **fixed**. Message: *"outcome column has only one distinct value ([1]); a
  binary outcome needs both an event and a non-event class present to compute anything"*.
- **`outcome_one_event`** (one positive, rest negative). Expected: load (valid, if extreme).
  Observed-before: loads — locked. Adequacy of a single event is a downstream reliability concern,
  not an ingestion error.

## i. Cluster pathologies (valid input; adequacy is a downstream `few_clusters` concern)

All six load at ingest. The number and shape of clusters is not an ingestion contract — the spec
(§5.1) already handles thin clustering downstream with the `few_clusters` note when `G < 40`.
Rejecting these at load would confuse "malformed input" with "input that will produce a
wide/uncertain interval," which is exactly the distinction predval's flag system exists to keep.

- **`cluster_g_one`** / **`cluster_constant`** (`G = 1`, all rows one region): load. Locked.
- **`cluster_g_two`** (`G = 2`): load. Locked.
- **`cluster_float_ids`** (float region codes): load — cluster ids are labels, not numbers. Locked.
- **`cluster_zero_event`** (a region with only non-events): load. Locked.
- **`cluster_g_equals_n`** (every row its own cluster): load. Locked.

The one cluster corruption that *does* reject is a null cluster id (`na_cluster`, bucket b) — a null
is not a thin cluster, it is a subject with no cluster at all.

## j. Roster mismatch (loud flags, not load failures)

- **`roster_undeclared`** (a `model_id` present in the file but not in `expected_models`, a likely
  transposition typo). Expected: load, then a loud warning. Observed-before: loads;
  `check_roster` returns a `present_but_undeclared` warning — locked. This is the loud path the
  roster exists for: a mistyped id produces plausible metrics for a model that is really absent.
- **`roster_declared_absent`** (a model declared in `expected_models` but not present, under
  `on_missing_model: warn`). Expected: load, then a loud warning (would be a hard fail under
  `on_missing_model: fail`). Observed-before: `check_roster` returns a `declared_but_absent`
  warning — locked, and confirmed still loud under the GUSTO-shaped config.

## k. Excel damage

- **`excel_scientific`** (`"6.2E-1"`). Expected: parse to `0.62`. Observed-before: loads as `0.62` —
  locked (scientific notation is a valid float).
- **`excel_mangled_id`** (a subject id turned into a date-like `MAR-4`). Expected: load — an id is a
  string, any string is fine as long as coverage still holds. Observed-before: loads — locked.
- **`excel_thousands_sep`** (`"1,234"` in `predicted`). Expected: hard fail. Observed-before: hard
  fail (non-numeric, same path as the comma-decimal) — locked.

## l. Structural damage

- **`empty_file`** (zero bytes). Expected: hard fail. Observed-before: hard fail — the CSV reader has
  no columns to parse and the error names the file — locked. Message: *"predictions file could not
  be read: No columns to parse from file"*.
- **`header_only`** (header row, no data). Expected: hard fail. Observed-before: **loaded silently**
  as an empty predictions table — **fixed**. Message: *"predictions file has no rows; a header with
  no data (or an all-empty file) is not a valid predictions table — there is nothing to evaluate"*.
- **`truncated_row`** (final row missing its `predicted` field). Expected: hard fail. Observed-
  before: hard fail — the missing field reads as NaN and hits the null rule — locked.
- **`extra_trailing_cols`** (an extra `junk` column). Expected: hard fail. Observed-before: hard fail
  — locked. Message: *"predictions has columns outside the v0 contract: ['junk']"*.

## The no-artefact guarantee

For every hard-fail case the suite points the full pipeline runner at the corrupted case directory
and asserts the output directory contains none of `report.html`, `findings.json`, `metrics.parquet`,
`fragility.parquet`, `coverage.parquet`, `calibration.parquet`, or `manifest.json`. This holds
structurally: artefacts are only ever written by `Evaluation.write` / `write_report` /
`write_findings`, all of which run *after* both loaders return — so a load that raises can never
leave a partial or misleading artefact behind. The non-zero **exit code** is verified by running the
runner as a subprocess over a spread of failure modes (predictions-load, cohort-load, unreadable
encoding, no-rows, unknown-column); it is the same unhandled-error-to-`SystemExit` path for every
case.
