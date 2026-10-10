# Prospective local family extension, 2026-10-10

This extension follows inspection of the original two fixtures and their results. It is
not a retroactive preregistration of those results. This protocol is committed locally
before executing the new extension. Local Git chronology is an audit trail, not independent
proof of preregistration or a trusted external timestamp. Baseline: integrated `150998f`.
Historical benchmarks/alternatives inputs, protocol and results remain untouched.

## Fixed design

Use 20 seeds, the consecutive integers 20261020 through 20261039 inclusive. For each seed:
120 independent patient clusters, uniform integer sizes 4–20 inclusive; patient effect
Normal(0,1.2), independent row covariate Normal(0,1), eta=-0.8+0.9*x+patient_effect.
Published p=expit(-0.4+1.4*eta). Share cluster sizes, covariates, p and uniform outcome
randomness between the two generating families:

- logit_linear: q=expit(eta), so logit(q)=(logit(p)+0.4)/1.4;
- probability_linear: q=expit(-2.5+5*p), so logit(q) is linear in p.

Outcome uniforms select a patient-shared uniform with probability 0.65, otherwise an
independent row uniform. y=1[uniform<q]. This preserves q's Bernoulli marginal and introduces
within-patient dependence. Families differ in true risks and difficulty; do not pool them
into a universal method ranking or infer cross-family superiority. These deliberately
matched families are controlled diagnostics, not broad realistic miscalibration coverage.

Add one undistorted WDBC control using the original seed 20261010, stratified half split,
StandardScaler/LogisticRegression(C=1,max_iter=2000,solver=lbfgs) fitted only on training
rows. Use its unmodified held-out probabilities. They are estimates, NOT known true risks.
No oracle loss or risk recovery is reported for WDBC. Retain UCI CC BY 4.0 attribution.

## Methods and matched folds

For each fixture: raw probabilities; native PREDVAL ladder rung2; independent statsmodels
unpenalized GLM y~1+logit(clip(p,1e-6,1-1e-6)); statsmodels y~1+p. Use five folds from sorted
patient identifiers round-robin; all methods use identical folds. Assert no patient overlap.
GLM tolerance 1e-11, maxiter 100. Record warnings and non-convergence, exceptions or
nonfinite predictions as unavailable; never silently discard cases. Logit GLM is a numerical
control, not another independent method. Raw-probability GLM represents probably's family,
NOT execution of the probably R package. PREDVAL rung3 suppression is recorded separately
and does not make rung2 unavailable. Do not change any production estimator.

## Outcomes, uncertainty and reporting

Report per seed and family: rows, clusters, events, availability/reasons; realized Brier,
pooled AUROC and delta from raw, realized Brier gain. Synthetic only: mean[(prediction-q)^2],
oracle expected Brier mean[q(1-q)], fresh-outcome expected Brier as their sum, expected gain,
and recovery 1-MSE(prediction,q)/MSE(raw,q). This expectation holds predictions/covariates
fixed and draws fresh outcomes; the realized oracle Brier is not an irreducible lower bound.

Within each family and method report mean and Monte Carlo SE (sample SD/sqrt(n)) across
available independent seed runs, plus n available out of 20. Also report paired between-family
method contrasts only within each family on jointly available seeds. Use approximate
Student-t 95% Monte Carlo intervals for the mean. They describe simulation precision, not
patient-data confidence intervals, clinical validation, or universal superiority. At n=20,
a Bernoulli rate has maximum MCSE about 0.112, so rare failures/coverage cannot be estimated
precisely; report this limit. No confidence-interval coverage study or significance gate.
No result-dependent seed selection, sample enlargement or tuning. Preserve failed cases.

## Verification and scope

Use existing locked Python dependencies. Check expected-loss identity, family specification,
shared inputs/randomness, grouped partitions, finite bounded predictions, availability
accounting, and PREDVAL-versus-independent logit agreement where both converge. Run twice;
verify stable output hashes. Record interpreter/package versions, source/protocol hashes,
commands, warnings, test results and runtime separately. Stop after 40 synthetic cases and
one WDBC control, their fixed verification checks, and a local patch/review bundle.

CalibrationCurves is a verified but UNEXECUTED additional cluster-calibration comparator:
https://bavodc.github.io/websiteCalibrationCurves/reference/valProbCluster.html
https://cran.r-project.org/web/packages/CalibrationCurves/CalibrationCurves.pdf
https://bavodc.github.io/websiteCalibrationCurves/articles/CalibrationCurves.html
Its MIXC/CGC/MAC2 curves and uncertainty target different quantities from paired Brier gain.
The linked website describes a MIXC default; CRAN 3.1.0 describes combined MAC2+MIXC.
Any future execution must pin version and explicit approach. R exists but CalibrationCurves,
lme4, metafor and rms were absent during inspection. Installation/execution is out of scope.
Singleton WDBC clusters are not a fair within-cluster calibration-curve benchmark.

Keep rare-event/separation/low-cluster-count interval stress separate. This bounded extension
is neither a literature survey nor a substitute for that stress study. No production code,
public writes, package installation, PR update, merge, or deployment.
