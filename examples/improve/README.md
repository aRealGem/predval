# IMPROVE neoepitope immunogenicity — predval example cohort

This directory applies predval to a problem outside the clinical-risk-score setting it was
built in: deciding which candidate neoepitopes from a patient's tumour are worth putting into a
personalised vaccine or a TCR-T product.

## Why this cohort

Every published neoepitope immunogenicity predictor reports **ranking** metrics — AUROC, AUPRC,
top-N recall, fraction-ranked. A literature check (2026-10-04) found no published work reporting
**calibration** for peptide-level immunogenicity predictors: whether a peptide scored 0.8
validates 80% of the time. The nearest work is adjacent rather than overlapping and is cited
below.

Calibration is not a cosmetic concern here, because selection is budget-constrained. A
personalised vaccine encodes on the order of 20 neoepitopes; a TCR-T product carries 3-5. Those
slots are the scarce resource, and a probability that over-promises by a large factor misprices
every one of them.

## Provenance

Source: Borch et al. 2024, *IMPROVE: a feature model to predict neoepitope immunogenicity
through broad-scale validation of T cell recognition*, Frontiers in Immunology. Data release:
[github.com/SRHgroup/IMPROVE_paper](https://github.com/SRHgroup/IMPROVE_paper), `results.zip`.

- 17,520 candidate neoepitopes, 70 patients, three accrual cohorts (bladder 6,237 peptides / 24
  patients; melanoma 5,921 / 26; a mixed "Basket" group 5,362 / 20).
- 467 peptides are experimentally immunogenic — **2.666% prevalence**.
- Predictions are the paper's own **out-of-fold** scores from its 5-fold cross-validation. That
  CV was partitioned **by patient**. `build_fixture.py` checks every released random-forest
  table for non-missing patient IDs and integer partitions, and rejects any patient appearing
  in more than one partition before writing either fixture. This verifies the tables' grouping
  metadata; it does not independently audit upstream training or the NNAlign comparator's
  split assignments. Interpreting the released scores as out-of-fold predictions still relies
  on the source authors' training procedure.
- Seven models: the IMPROVE random forest in its three released feature configurations
  (`Simple`; `TME_excluded`, adding cellular prevalence and a prioritisation score;
  `TME_included`, further adding cytolytic activity, HLA expression and MCP-counter estimates),
  each in full and in the `wo_prime` ablation that drops the PRIME feature, plus the paper's
  `nnalign` comparator.

Nothing in this directory re-trains, re-tunes or re-ranks anything. predval reads probabilities
and labels and nothing else.

## Unit of analysis

The cluster is the **patient**, not the peptide. Peptides from one patient share that patient's
HLA genotype, mutational process, tumour transcriptome and microenvironment — the dominant
sources of variance in every feature these models consume. Mean cluster size is 250 peptides,
an order of magnitude larger than the PCam fixture's slides, so the cost of getting this wrong
is correspondingly larger. Every interval in the report resamples whole patients.

## Prespecified subgroups

Both were declared before any metric was computed on this cohort.

- `tumour_cohort` — the release's own three accrual groups. A model that holds in melanoma, the
  indication these predictors were overwhelmingly trained and tuned on, and fails elsewhere is
  precisely the failure mode that matters for carrying a verdict to another disease. This is the
  closest available proxy for that transfer question, and its result bounds how far anything
  here may be carried. MSS colorectal cancer is not in this cohort at all.
- `peptide_length` — 9-mers versus everything else. The binding predictors that supply the
  dominant features are trained overwhelmingly on 9-mers and are known to degrade off that
  length, so the off-length stratum is where a calibration fault would be expected *a priori*.

## Reproducing

```sh
# 1. fetch the release and extract results.zip
git clone https://github.com/SRHgroup/IMPROVE_paper.git
unzip -q IMPROVE_paper/results.zip -x '__MACOSX/*' -d improve_results

# 2. build the fixture (writes cohort.parquet and predictions.parquet here)
uv run python examples/improve/build_fixture.py --src 'improve_results/results copy'

# 3. the main evaluation -- B=2000 patient-clustered bootstrap, several minutes
uv run python examples/improve/run_evaluation.py

# 4. the budgeted-yield sidecar
uv run python examples/improve/budget_yield.py
```

`run_evaluation.py` writes `out/report.html`, `out/findings.json` and the parquet artefacts.
`budget_yield.py` writes `out/budget_yield.parquet`.

## The budgeted-yield sidecar

`budget_yield.py` is deliberately *not* part of predval's contract. predval reports calibration;
the mapping from calibration to a decision is cohort-specific, and this is that mapping for this
cohort. For each model and each budget k ∈ {3, 5, 10, 20} it takes every patient's own top-k
candidates by score and reports, per patient and averaged with a patient-level bootstrap:

- `observed` — immunogenic peptides actually in that top-k;
- `promised` — the sum of the model's published probabilities over the *same* top-k set, which is
  what those probabilities assert the yield will be;
- `recalibrated` — the same sum after predval's cross-fitted rung1 (intercept-only) correction,
  fitted on other patients.

Because all three columns score the identical top-k set, `promised − observed` isolates the
miscalibration cost in units of antigens. It is not a ranking error.

## Prior art this sits next to

- **ImmUQBench** (Qayyum et al. 2026, *Oxford Open Immunology* iqag003, PMID 41859687) reports
  ECE, Brier and NLL for uncertainty-quantification methods on **whole-protein** antigen
  immunogenicity. Complementary scope: no peptide-HLA level, no patient clustering, no budget.
- **arXiv 2604.13254** — calibrated conformal abstention for TCR-pMHC **binding**.
- **CaliPPer** (arXiv 2606.07258) — distance-aware Platt recalibration, largely for binding and
  presentation rather than immunogenicity.
- **TESLA** (Wells et al. 2020, *Cell*) and the shortcut-bias analysis in Zhang et al. 2026
  (*Cell Genomics*, PMC13347938) report ranking metrics only.

## Limits worth stating plainly

- Three tumour types, none of them colorectal. Nothing here licenses a claim about MSS CRC.
- 70 patients is a small number of independent units for interval estimation, which is exactly
  why the patient-clustered intervals are reported and the naive per-peptide ones are shown
  beside them labelled incorrect.
- The experimental labels are themselves a screen: "not immunogenic" means "not detected under
  the assays applied to that patient", which is not the same as "not immunogenic". A calibration
  result is a statement about agreement with these labels.
- A calibration test can fail to reject "calibrated". It can never establish it.
