# IMPROVE neoepitope immunogenicity, third-party predictor panel — predval example cohort

The companion to [`examples/improve/`](../improve/). Same 17,520 candidate neoepitopes, same 70
patients, same experimentally determined T-cell recognition labels — scored by immunogenicity
predictors published by **other groups** instead of by IMPROVE's own random forest.

## The question this fixture asks

`examples/improve/` found that IMPROVE's published probabilities sit roughly 15x above the base
rate, with the whole fault in the intercept, repairable by a single global shift. On its own that
is a statement about one training pipeline: train on a class-balanced set, deploy at natural
prevalence, and this is what you get.

The obvious objection is that it is one model. So this fixture holds everything else fixed — the
patients, the peptides, the assay, the clustering, the two prespecified subgroups — and swaps in
four independently developed and independently published predictors from four other groups. If
the fault is a pipeline artefact, it should look different in each. If it is what the field does,
it should look the same in all of them.

**This is explicitly not independent replication.** The patients, labels and assay are shared
with `examples/improve/`, so nothing here tests whether the finding survives a new patient
sample. It tests breadth across *predictors*, not breadth across *cohorts*. Replication needs a
second cohort that publishes per-peptide scores with a patient-level clustering unit at natural
prevalence, which is an acquisition problem, not a code problem.

## Provenance

Source: the predictor comparison table released with Borch et al. 2024, *IMPROVE: a feature model
to predict neoepitope immunogenicity through broad-scale validation of T cell recognition*,
Frontiers in Immunology — [github.com/SRHgroup/IMPROVE_paper](https://github.com/SRHgroup/IMPROVE_paper),
`data.zip` → `data/benchmark_comparison/neoepitopes_predictions.tsv`, joined to that same
release's patient-partitioned cross-validation output.

The join is the part worth checking, and `build_fixture.py` documents it in full. In short:
`Mut_peptide` + `HLA_allele` is **not** a unique key — 167 peptide-HLA pairs recur across
patients, and joining on them produces a silent many-to-many expansion (17,884 rows from 17,520)
with apparent label disagreements that are really the cross-product of those duplicates. Adding
`Expression`, which is measured per patient, makes it exact: 17,520 of 17,520 rows, both sides
unique, `validate="1:1"`, zero label disagreements. The builder asserts all of that rather than
trusting it, and asserts the `[0, 1]` bound after rescaling.

## Roster

Admitted, because every member's published output is nominally an immunogenicity **probability**:

- `deepimmuno` — DeepImmuno.
- `deephlapan_immuno` — DeepHLApan's *immunogenic* score, not its binding score.
- `deepnetbim_immuno` — DeepNetBim's immunogenicity probability. 9-mers only, so it covers 10,561
  of 17,520 peptides; see the coverage note below.
- `ittca_rf` — iTTCA-RF, published on a 0–100 scale and divided by 100 in the builder.
- `improve_tme_incl` — IMPROVE's best-discriminating variant, carried over from
  `examples/improve/` as the in-house reference point. It is **not** a competitor on this table:
  it was cross-validated on this very cohort, so it reads with a home advantage the other four
  do not have.

Excluded, and the exclusion list is part of the prespecification because scoring these as
immunogenicity probabilities would be a strawman rather than a finding:

- **MHCflurry** (affinity, processing, presentation) and **HLAthena** (MSiC / MSiE / MSiCE)
  predict antigen *presentation*. A presentation probability of 0.5 is not a claim that the
  peptide is immunogenic, so holding it to the immunogenicity base rate would manufacture a
  miscalibration the model never asserted.
- **MixMHCpred**, **RankEL**, **RankBA** — binding predictors, and the ranks are percentiles.
- **PRIME** — binding-weighted, not published on a probability scale.
- **IEDB immunogenicity** — runs from −0.914 to 0.639 on this cohort. It is a score, not a
  probability, and predval would reject it at the `[0, 1]` bound anyway.

The distinction being enforced: a calibration critique applies only to a model that claims to
emit a probability of the event the labels record. Anything else is a ranking tool, and
`examples/improve/`'s own result is that the ranking is the part that works.

## Coverage, and why the floor was left where it is

`coverage.min_fraction` stays at 0.95 knowing that DeepNetBim breaches it. Scoring only 9-mers is
a real property of that model, and the right response is a coverage flag plus an honest
common-subset comparison — not a relaxed floor, and not imputation. A model that did not score a
peptide gets no row for it.

The excluded 40% is far above the informative-selection threshold, so the report also cautions
that selection into the common subset matters here: the intersection is 9-mers only, which is the
easiest stratum for every binding-derived feature. **Read each model's full-subset column for its
own behaviour, and the common-subset column only for model-to-model comparison.**

Building this fixture is also what exposed a real defect in predval itself: the per-model
coverage-delta column in the report renders only when coverage differs across the roster, and
every prior fixture (GUSTO, PCam, `examples/improve/`) had uniform coverage, so that branch had
never executed. DeepNetBim is the first partial-coverage model in the repo's history and it
crashed the renderer. Fixed, with regression tests covering non-uniform coverage, in the same
change as this directory.

## Unit of analysis and prespecified subgroups

Identical to `examples/improve/`, carried over unchanged so the two fixtures are readable side by
side, and neither subgroup was chosen after seeing a result on this roster.

The cluster is the **patient**. Peptides from one patient share that patient's HLA genotype,
mutational process, tumour transcriptome and microenvironment — true of these predictors' inputs
exactly as it was of IMPROVE's. Subgroups are `tumour_cohort` (the release's three accrual
groups) and `peptide_length` (9-mer versus everything else).

## The skill-ladder sidecar

`skill_ladder.py` exists to answer the first objection a reviewer should raise.

predval's report quotes one Brier skill score per model, at the **admissible** rung — the lowest
rung whose cross-fitted paired gain interval excludes zero. That rule is parsimony-first by
design: do not buy a more complex correction than you can prove you need. But it means the
headline number depends on a selection rule, and on this roster most models come out at or below
zero skill. Maybe a higher rung would have rescued them and the rule hid it.

So the sidecar drops the rule and scores **every** rung the ladder can fit — rung0 as published,
then cross-fitted rung1, rung2 and rung3 — each with predval's own `brier_skill_interval` and the
same patient-clustered bootstrap, B and seed. A rung the ladder cannot fit for a model is
reported unavailable with its reason, never imputed. If no rung reaches reliably positive skill,
that is a property of the scores rather than of the selection rule, and that is the claim this
script lets a reader check. It reports three outcomes, not two: "reliably positive at some rung",
"reliably negative at every fitted rung", and "no rung reaches reliably positive skill" — the
last is not the same claim as the middle one, and collapsing them would overstate the result in
exactly the direction this fixture argues.

Like `budget_yield.py` in `examples/improve/`, it is a sidecar and deliberately not part of
predval's contract. It reuses predval's private `_crossfit_predictions` rather than
reimplementing it, so the corrections scored here are byte-for-byte the ones the report's own
rung columns are scored on.

## Reproducing

```sh
# 1. fetch the release and extract both archives
git clone https://github.com/SRHgroup/IMPROVE_paper.git
unzip -q IMPROVE_paper/data.zip    -x '__MACOSX/*' -d improve_data
unzip -q IMPROVE_paper/results.zip -x '__MACOSX/*' -d improve_results

# 2. build the fixture (writes cohort.parquet and predictions.parquet here).
#    Two explicit paths, because the two tables come from two different archives.
uv run python examples/improve_panel/build_fixture.py \
    --panel-tsv improve_data/data/benchmark_comparison/neoepitopes_predictions.tsv \
    --cv-txt 'improve_results/results copy/5_fold_CV/TME_included/pred_df_TME_included.txt'

# 3. the main evaluation -- B=2000 patient-clustered bootstrap, several minutes
uv run python examples/improve_panel/run_evaluation.py

# 4. the skill-at-every-rung sidecar
uv run python examples/improve_panel/skill_ladder.py
```

`run_evaluation.py` writes `out/report.html`, `out/findings.json` and the parquet artefacts;
`skill_ladder.py` writes `out/skill_ladder.parquet`.

## Limits worth stating plainly

- **Not a replication.** Shared patients, labels and assay with `examples/improve/`. Breadth
  across predictors only.
- **`improve_tme_incl` is not a competitor.** It was cross-validated on this cohort; the other
  four were trained elsewhere and are being applied to it cold. That asymmetry is the normal
  condition for anyone selecting antigens with an off-the-shelf tool, which is why it is on the
  table at all — but it is not a like-for-like ranking of model quality, and reading it as one
  would be wrong.
- **Discrimination near chance for the third-party models is itself a finding about this cohort,
  not only about the tools.** A cohort on which nothing discriminates cannot establish that a
  tool is uninformative in general.
- Three tumour types, none colorectal. Nothing here licenses a claim about MSS CRC.
- The labels are a screen: "not immunogenic" means "not detected under the assays applied to that
  patient". A calibration result is a statement about agreement with these labels.
- A calibration test can fail to reject "calibrated". It can never establish it.
