# Alternatives benchmark protocol (registered before benchmark execution)

Date: 2026-10-10. PREDVAL baseline: `99ca3d72f1d4f8340df9a59de3616277e9ebdb7a`.
This is an engineering/method-comparison benchmark, not clinical validation or a claim
that a fitted model is suitable for patient care. No runtime, accuracy, or novelty winner
will be inferred from unavailable software.

## Fixed scope and inputs

1. Independent binary rows: Wisconsin Diagnostic Breast Cancer, bundled with scikit-learn,
   UCI dataset 17 (CC BY 4.0). Use its original measurements to create strictly held-out
   predictions from an explicitly specified logistic regression pipeline. This is a real,
   redistributable methods fixture, not an external clinical validation study.
2. Synthetic clustered patients: seed 20261010; 120 patients, unequal 4–20 repeated rows,
   patient random effect and patient-correlated binary labels. Published predictions are
   deliberately distorted. This controlled fixture exercises clustering, correction, and
   missing-coverage failure paths, and is explicitly synthetic.
3. Missing-model-coverage variants: remove predictions from patients in the upper 30% of
   mean predicted risk, and separately remove an entire declared model. The cohort and
   model roster stay fixed. Compare full available rows and the common intersection.

## Matched estimands and configuration

- Positive outcome is coded 1 throughout. Brier is mean squared error over rows, not a
  patient-weighted mean. AUROC uses tied-score midranks. Point-metric tolerance: 1e-10.
- Calibration slope is the slope from unpenalized logistic `y ~ 1 + logit(p)`.
  Calibration-in-the-large is a separate intercept with logit(p) as an offset (slope 1).
  Do not equate an intercept from a free-slope fit with calibration-in-the-large.
- Matched correction: unpenalized logistic intercept+slope on clipped logit probability
  (epsilon 1e-6), five disjoint patient folds. Use PREDVAL's fixed sorted-patient round-robin
  assignment as an exported common split, and give competitors those exact folds when
  their public API supports it. Compare pooled held-out Brier and AUROC, not unweighted
  fold-mean metrics. Label non-equivalent fits separately rather than declaring disagreement.
- Intervals: 1,000 paired whole-patient percentile bootstrap replicates; seed 20261011;
  95% nominal intervals. Conditional on the fixed held-out predictions; no claim of full
  correction-training uncertainty. Use the exact same resample indices for custom adapters.
  Compare row-bootstrap intervals separately as different assumptions, not matched inference.
- Independent-row fixture: stratified 50% held-out split, seed 20261010; StandardScaler plus
  LogisticRegression(C=1, max_iter=2000, solver='lbfgs'); no hyperparameter tuning. The positive
  class is malignant. A fixed `sigmoid(-0.4 + 1.4*logit(p))` distortion supplies a correction case.

## Competitors and fair interpretation

- PyCalEva: installed registry release, public CalibrationEvaluator for available metrics.
  It is primarily a calibration evaluator, so missing end-to-end cohort machinery is not
  a numerical defect. Document required adapter work rather than handicap it.
- probably + rsample: public `cal_validate_logistic` accepts an rset; grouped rsets and group
  bootstraps are supported. Export matched folds and provide a runnable R adapter. If R is
  absent or cannot be installed safely, mark execution unavailable, never failed metrics.
- isitfair: use official registry release if available, otherwise inspect official source
  without running unrecognized code. Respect explicit folds on its cross-fitted ladder and
  its one-row-per-patient convention. Do not call a row bootstrap a grouped-patient bootstrap.
- PREDVAL is not privileged as the numerical oracle: independently calculate metrics with
  scikit-learn/statsmodels and a small shared resampling reference.

## Deliverables and stopping rule

Commit protocol before results; record exact software versions, input hashes, outputs,
commands, convergence/failure information, supported/configurable/unavailable capabilities,
and adapter source line counts (not invented developer-time savings). Run each executable
path twice and compare stable output hashes. Deliver CSV/JSON summaries, reproducible scripts,
and a bounded findings report. Stop after both fixtures, coverage variants, and clean repeat
have completed, or document specific dependency/access blockers. No paid compute, private
clinical data, data-use agreements, outreach, push, PR, merge, or deployment.

## Source-review amendment (before R or isitfair execution)

Official source inspection found that probably's `smooth=FALSE` fits `y ~ p`, while
PREDVAL rung2 fits `y ~ logit(p)`. These are different correction families. Report probably's
native result separately and cross-check it against a raw-probability statsmodels GLM;
retain a separate R `glm(y ~ logit(p))` adapter as the exactly matched control. Do not feed
logits into a probability API merely to force equality. Isitfair's native Platt fit uses
L2-regularized sklearn logistic regression and epsilon 1e-10, so it too is a separate native
configuration, not an exact unpenalized rung2 equivalent. No executed isitfair result will
be claimed without an authorized executable package.

An isolated conda-forge R setup may be attempted for at most 15 minutes and less than 1 GB
of planned package downloads. No system-level R installation or security changes.
