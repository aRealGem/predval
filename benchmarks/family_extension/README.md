# Paired calibration-family extension

A bounded extension of the historical alternatives benchmark, not clinical validation or a
universal ranking. The local prospective protocol was committed as `4a28915` before this
extension's execution, after inspecting the original benchmark. The local commit sequence
is an audit trail, not independently timestamped preregistration. The protocol phrase
“paired between-family method contrasts only within each family” is a wording error: the
implemented, reported contrasts are between methods within each family. The frozen protocol
is retained unchanged. Historical results under
`benchmarks/alternatives` are unchanged. No production code or public branch was changed.

## Main finding

Performance reverses when the known generating calibration family changes. On 20 fixed
seeds, PREDVAL rung2 improves the logit-linear fixture but worsens the probability-linear
fixture relative to raw predictions. The probability-scale GLM has the smaller Brier in
its matched family. This supports testing assumptions, not declaring a universal winner.

| Generating family | Method | Mean Brier | Mean known-risk MSE | Mean pooled AUROC delta |
|---|---|---:|---:|---:|
| Logit-linear | Raw | 0.168878 | 0.007849 | 0 |
| Logit-linear | PREDVAL rung2 | 0.162805 | 0.001155 | -0.004083 |
| Logit-linear | Probability GLM | 0.165599 | 0.004141 | -0.007555 |
| Probability-linear | Raw | 0.138728 | 0.001522 | 0 |
| Probability-linear | PREDVAL rung2 | 0.141261 | 0.004010 | -0.005336 |
| Probability-linear | Probability GLM | 0.138268 | 0.001115 | -0.008699 |

The independent logit GLM agrees with PREDVAL on every case where both are available.
The probability GLM represents the family used by probably's smooth=FALSE calibrator;
**probably was not executed in this extension**. This is a Python family comparison, not a
new native R package benchmark.

Within each family, the paired mean Brier difference (PREDVAL minus probability GLM) is:

- Logit-linear: -0.002795; approximate 95% Monte Carlo interval [-0.003681, -0.001909].
- Probability-linear: +0.002993; interval [0.001829, 0.004156].

These intervals quantify precision of this simulation mean across 20 independent seeds,
not patient-level uncertainty or broad method superiority. The families share covariates,
published scores, patients, folds and outcome uniform draws, but differ in true risks,
difficulty and raw miscalibration. Do not compare their absolute performance or recovery
percentages as if those were matched difficulty settings. No seed or parameter was selected
after looking at this extension's outputs.

Mean per-seed recovery of expected reducible error is 85.34% (MCSE 3.00 percentage points)
for rung2 and 47.20% (3.70 points) for probability GLM in the logit family. In the probability
family it is -160.99% (31.59 points) and 27.82% (15.91 points), respectively. Negative recovery
means calibration worsens known-risk squared error; the raw-error denominator is much smaller
in the second family. These are means of per-seed ratios, not ratios of pooled means.

All 164 method-cases (41 fixtures times four methods) were available; no fitting warnings
were captured. Suppression of other native ladder rungs is retained in cases.json. Zero
failures in 20 seeds is weak evidence about rare failures: the maximum binomial MCSE at
this simulation size is about 0.112. This is not an interval-coverage study.

## Undistorted real-data control

| Method | Brier | AUROC | AUROC delta |
|---|---:|---:|---:|
| Raw WDBC model | 0.029478 | 0.988089 | 0 |
| PREDVAL rung2 | 0.029271 | 0.984769 | -0.003320 |
| Probability GLM | 0.027865 | 0.979024 | -0.009065 |

WDBC model probabilities are fitted estimates, not known true risks. No expected-oracle
metric is calculated for WDBC. Brier improvement does not imply discrimination improvement.
Fold-specific increasing calibrations can change ordering between folds, so pooled AUROC
need not be preserved even when each fold's AUROC is unchanged.

The public Wisconsin Diagnostic Breast Cancer dataset is credited to Wolberg, Mangasarian,
Street and Street (1993), UCI, https://doi.org/10.24432/C5DW2B, CC BY 4.0. We use the
scikit-learn bundled data, a held-out subset and fitted unmodified logistic probabilities.
No row-level WDBC data are bundled; the runner regenerates them. No endorsement is implied.

## Expected risk versus realized oracle loss

For synthetic q, the reported fresh-outcome expected Brier is
mean[q*(1-q)] + mean[(prediction-q)^2], holding fitted predictions and covariates fixed.
It averages over new outcome randomness, including patient shared uniforms. It does not
condition on the originally realized shared uniforms. Cluster dependence changes variance,
not this expectation identity. The realized loss of q on one noisy outcome vector is not an
irreducible floor. The original synthetic fixture's roughly 98% realized raw-to-oracle gain
ratio should therefore not be called recovery of expected reducible risk; that quantity is
roughly 81.31% for that historical fixture. The new simulation uses different fixed seeds.

## Comparator scope and remaining work

CalibrationCurves supplies native cluster-aware calibration curves via MIXC, CGC and MAC2:
[official API](https://bavodc.github.io/websiteCalibrationCurves/reference/valProbCluster.html),
[official vignette](https://bavodc.github.io/websiteCalibrationCurves/articles/CalibrationCurves.html),
[current CRAN manual](https://cran.r-project.org/web/packages/CalibrationCurves/CalibrationCurves.pdf).
It is **unexecuted here**. The inspected website uses MIXC as default; CRAN 3.1.0 documents a
combined MAC2/MIXC default. Any adapter should pin version and explicit approach. R 4.5.3 is
available, but CalibrationCurves, lme4, metafor and rms were absent. No dependency installation
was attempted. A future bounded run should start with official bundled clustered data, then
an explicitly configured MIXC small-cluster diagnostic. Singleton WDBC is unsuitable for
within-cluster curve fitting. Curve confidence/prediction bands are not interchangeable with
PREDVAL's paired Brier-gain intervals.

A nonlinear/spline-shaped fixture would test additional flexibility; this two-family
extension does not establish it. Keep rare-event, separation and low-cluster-count stress
separate, with unavailable fits and interval coverage explicitly measured. The existing
coverage/roster contract demonstrates integration rather than unique statistical methods;
no programmer-time savings or clinical utility were measured.

## Reproduce and verify

Use the existing Python lock from benchmarks/alternatives/requirements-lock.txt and install
this worktree's package as in that benchmark. In a compatible existing environment:

```sh
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/predval-family-mpl XDG_CACHE_HOME=/tmp/predval-cache
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python benchmarks/family_extension/run_extension.py --out /tmp/family-a
python benchmarks/family_extension/run_extension.py --out /tmp/family-b
python benchmarks/family_extension/verify_results.py /tmp/family-a /tmp/family-b
python -m pytest tests benchmarks/alternatives/tests benchmarks/family_extension/tests -ra
```

Output directories must be new. The runner records per-case scores, warning/suppression
states, input and prediction hashes, summary MCSEs, exact versions and source/protocol
hashes. Run-body timing is separate and excludes imports/install; it is not a package-speed
comparison. The manifest records the absolute import path as provenance, so transferring the
worktree changes that field even if the numerical output hashes match. Verification records
and retained results accompany this report. See verification/status.json for exact checks.

Verified: 291 tests passed, 20 optional fixture-dependent tests skipped; Ruff lint and
formatting passed. Both complete runs reproduced four stable result files and the manifest.
Independent reconstruction of all 40 synthetic cases and GLM refits reproduced losses within
4.45e-15 and all aggregate MCSE/paired-contrast calculations. Unexpected exceptions outside
the selected fitting exception types would abort rather than become an unavailable entry;
none occurred in this fixed design.
