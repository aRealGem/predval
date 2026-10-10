# Bounded comparison: PREDVAL, PyCalEva, probably + rsample, and isitfair

This is a reproducible **methods/engineering comparison**, not clinical validation.
PREDVAL's added value here is explicit cohort/cluster contracts and connected diagnostics.
Its point metrics are not novel, and correct alternatives can reproduce its results.

The protocol was committed before execution (`dae6e1a`), with a source-review amendment
(`756e5a1`) distinguishing correction families before any R or isitfair execution.
Both commits descend from patient-partition guard `99ca3d7`; no core PREDVAL code changed.

## What is and is not executed

- **PREDVAL 0.1.0, PyCalEva 0.8.2:** executable numerical checks on both fixtures.
- **Independent sklearn/statsmodels control:** point metrics, unpenalized grouped correction,
  and a separate custom patient bootstrap. PREDVAL is not its own numerical oracle.
- **probably 1.2.0 + rsample 1.3.2 (R 4.5.3):** native adapter executed on both fixtures,
  including grouped cross-validation and 1,000 grouped bootstrap replicates. `cal_validate_logistic` accepts an rset, including
  grouped rsample splits. It is not fair to describe this stack as lacking grouped validation.
- **isitfair:** official source reviewed at `70ffb8d21c3e24eddfee412220b8a7a701768937`.
  No registry package was found; no source code was executed. An unexecuted public-API adapter
  is included. Its existing integrated reports and cross-fitted recalibration ladder make it
  a close competitor; this benchmark does not establish superiority over it.

## Fixtures and estimands

- `wdbc_heldout`: 285 held-out observations (106 malignant), from the 569-row Wisconsin
  Diagnostic Breast Cancer dataset bundled with scikit-learn. Train/test separation is
  fixed, stratified, and 50:50. Scaling and model fitting use only the training half.
  Logistic predictions are deliberately distorted to exercise calibration. This is a
  real-data test fixture, not an externally validated clinical model or deployment study.
- `synthetic_clustered`: 1,332 repeated rows in 120 synthetic patients, 560 events, unequal
  cluster sizes and correlated outcomes. Its generator and all seeds are in the protocol.
  It contains no real patient records and supplies no evidence of clinical effectiveness.
- Row-weighted Brier and AUROC are matched. Calibration intercept fixes slope at one;
  calibration slope is from a separate joint intercept+slope logistic fit.
- Correction uses the same exported, disjoint, five-fold patient assignments. Pooled
  held-out metrics are used, not averages of fold AUROCs. PREDVAL rung2 and the independent
  GLM use unpenalized logit-scale regression, epsilon 1e-6.
- Gain intervals use 1,000 paired patient bootstrap replicates, conditional on fixed
  held-out predictions. They do not include correction-training uncertainty. One fixture's
  interval is not an empirical confidence-interval coverage study.

## Python findings

PREDVAL and PyCalEva agree exactly on the four checked AUROC/Brier point estimates.
Against independent statsmodels, PREDVAL's calibration intercept/slope differ by less than
2e-15. Grouped rung2 probabilities agree within 4e-14 on these fixtures.

| Fixture | Published Brier | Grouped corrected Brier | Paired gain | 95% patient-bootstrap gain CI |
|---|---:|---:|---:|---:|
| WDBC held out | 0.031263 | 0.029256 | 0.002007 | [-0.005336, 0.009427] |
| Synthetic clustered | 0.181410 | 0.170541 | 0.010869 | [-0.001163, 0.022167] |

Both gains are uncertain in these fixtures. The synthetic row-bootstrap interval is
[0.005338, 0.016573], which excludes zero; respecting patients gives an interval 2.08 times
as wide that crosses zero. For WDBC, every cluster is a singleton; small row/patient interval differences reflect
ordering and Monte Carlo draws, not clustering. For the synthetic fixture this is a concrete
consequence of changing the independence
assumption, not evidence that PyCalEva is numerically wrong. A PyCalEva-plus-custom-patient-
bootstrap adapter reproduces PREDVAL's paired interval to numerical precision.

PREDVAL also withholds the WDBC spline rung after a non-convergence signal. This is reported,
not hidden to make the tool look better; this benchmark does not independently diagnose that
spline fit or establish that every PREDVAL unavailable-reason label is correct.

### Coverage faults

The synthetic partial model has **identical scores** to the full model where present; only
availability differs. Removing the top 30% of patients by mean predicted risk leaves
919/1,332 rows (68.99%). Thus a separate-model comparison on available rows changes the
cohort while appearing to compare models:

| Available cohort | n | AUROC | Brier |
|---|---:|---:|---:|
| Full model's available rows | 1,332 | 0.821898 | 0.181410 |
| Shared/partial rows | 919 | 0.783196 | 0.177929 |

PREDVAL rejects this at a declared 90% coverage floor, warns that the intersection excludes
31.0% of rows, and detects a completely absent declared model. The same checks are easy to
implement around array-based tools, but the caller must supply the full cohort and roster;
aligned y/p arrays alone cannot reveal missing rows or a missing model. These checks exercise
public I/O/contract functions, not a complete rendered-report/UI workflow.

## Fair capability comparison

| Capability | PREDVAL | PyCalEva | probably + rsample | isitfair |
|---|---|---|---|---|
| Binary point AUROC/Brier | Native, executed | Native, executed | Native yardstick, executed | Native, source reviewed |
| Patient-grouped correction folds | Native | Caller supplies correction | Configurable grouped rset or explicit splits | Configurable explicit fold vector |
| Correct repeated-patient bootstrap | Native | Custom loop | Native group_bootstraps plus metric mapping | Custom outer cluster loop needed; built-in patient resampling assumes one row per patient |
| Coverage denominator and declared roster | Native contract, executed | Caller glue | Caller glue | Caller glue in reviewed API |
| Integrated report | Native | Calibration PDF | Requires composition | Native integrated HTML/PDF |
| Correction-family equality to rung2 | Reference definition | No native correction in checked evaluator | Native smooth=FALSE uses y~p, a different family | Platt uses L2 regularization and epsilon 1e-10, different configuration |

“Caller glue” is not “impossible.” No developer-time savings were measured. Code counts
are implementation-specific and not evidence of programmer effort or package quality.
PyCalEva's `metrics()` documentation calls Brier scaled, but the checked 0.8.2 implementation
and `.brier` property return ordinary mean squared error; direct numerical agreement confirms
which quantity was compared.

## Reproduce Python without R

From the repository root, using Python 3.12.14 (recorded lock):

```sh
uv venv .venv-benchmark --python 3.12
uv pip install --python .venv-benchmark/bin/python -r benchmarks/alternatives/requirements-lock.txt
uv pip install --python .venv-benchmark/bin/python --no-deps -e .
MPLCONFIGDIR=/tmp/predval-mpl XDG_CACHE_HOME=/tmp/predval-cache \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-benchmark/bin/python benchmarks/alternatives/run_benchmark.py --out /tmp/predval-run-a
MPLCONFIGDIR=/tmp/predval-mpl XDG_CACHE_HOME=/tmp/predval-cache \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-benchmark/bin/pytest benchmarks/alternatives/tests
```

Use a fresh output directory; the script refuses to overwrite one. Repeat in a second fresh
venv from the same lock and compare `manifest.json`: all stable data and result hashes must
match. `resource_usage.json` is intentionally excluded from that manifest because timing/RSS
are machine- and load-dependent. Its time covers the benchmark body after imports and excludes
installation; it is **not a between-tool speed benchmark**.

Optional R command, once the registry packages listed in the adapter are installed:

```sh
Rscript benchmarks/alternatives/probably_adapter.R /tmp/predval-run-a /tmp/predval-r-run-a
```

This gives native probably's raw-probability logistic correction the exact same folds and
checks it against R's raw-probability GLM. A separate R logit GLM checks the truly matched
PREDVAL rung2. Native grouped bootstrap uses R/rsample RNG draws, not identical NumPy draws,
so its Monte Carlo interval need not match the Python interval bit for bit.

## Sources, redistribution, and versions

The WDBC-derived fixture is redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993),
*Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository,
[doi:10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
Changes: selected a held-out subset, fitted logistic probabilities using separate training
rows, deliberately distorted probabilities, and replaced identifiers with row-derived labels.
No endorsement by the data creators is implied.

Primary sources checked 2026-10-10:

- [UCI dataset and license](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [scikit-learn bundled data](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [PyCalEva official repository](https://github.com/MartinWeigl/pycaleva)
- [probably resampling validation](https://probably.tidymodels.org/reference/cal_validate_logistic.html)
- [probably logistic estimation](https://probably.tidymodels.org/reference/cal_estimate_logistic.html)
- [rsample grouped CV](https://rsample.tidymodels.org/reference/group_vfold_cv.html)
- [rsample grouped bootstrap](https://rsample.tidymodels.org/reference/group_bootstraps.html)
- [isitfair pinned official source](https://github.com/HugoGuillen/isitfair/tree/70ffb8d21c3e24eddfee412220b8a7a701768937)

Python numerical dependencies: NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.9.0,
statsmodels 0.14.6. Every transitive Python dependency is pinned in requirements-lock.txt.
The checked probably GitHub source is commit `974f0154cc7d44ae41d8072161996b23a19d3d2a`;
installed R package versions, when executed, take precedence over that source-only snapshot.

## Known limitation found during independent review

PREDVAL's IRLS implementation can return its last iterate after hitting the iteration limit
without convergence. On `y=[0,0,1,1]`, `p=[0.2,0.5,0.5,0.8]`, the quasi-separated likelihood
has no finite slope MLE, but the baseline returns a finite slope near 22.288 and does not
mark rung2 unavailable. A strict expected-failure regression test records this defect.
The matched rung2 estimates on the two main fixtures still agree with independently
converged GLMs; that agreement does not validate the general failure guard. No core fix is
included in this comparison branch. Do not market the tool as universally detecting
non-convergence or statistically validated on the strength of these checks.

## Reproducibility and measured resource use

Two isolated Python environments were created from the complete pinned package list.
After removing the shared PREDVAL transform from the independent GLM control, all stable
input/result file hashes matched across the two fresh-environment runs. Their manifest
SHA-256 is `a2a541ec30fbffe83103e04d2c85a5e83114a671e9818488e896a2f34eaf510d`.
Measured benchmark-body wall times were 1.96 and 2.38 seconds, with peak process RSS about
285 MiB. These exclude imports/installations, include the combined benchmark, and cannot
support a between-package speed claim. Exact records: `results/reproducibility.json`.

`results/glue_inventory.json` counts physical lines for the custom correction, cluster
bootstrap, coverage wrapper, and language-specific adapters. Counts include comments and
validation/export scaffolding; they are not a minimum implementation size or measured time
saved. This deliberately avoids an invented developer-effort advantage.

Row-level generated fixture/prediction files are not committed or bundled. Regenerate them
with the script; their hashes remain in the manifest for verification. Compact metric tables,
flags, seeds, software locks, and verification records are retained.

## Executed R comparison

R 4.5.3 with probably 1.2.0, rsample 1.3.2, and yardstick 1.4.0 ran successfully in an
isolated conda-forge environment. Resolver-planned downloads were 439,535,932 bytes across
227 packages, within the prespecified setup budget. No system-wide installation was used.
`conda-linux-64.lock` records exact official package URLs and checksums; use micromamba's
`create --file` with a fresh prefix to reproduce. Full package versions are in
`results/r/sessionInfo.txt`.

| Fixture | probably native y~p Brier | PREDVAL y~logit(p) Brier | probably AUROC | PREDVAL AUROC |
|---|---:|---:|---:|---:|
| WDBC held out | 0.027718 | 0.029256 | 0.968747 | 0.984031 |
| Synthetic clustered | 0.172383 | 0.170541 | 0.818836 | 0.818551 |

These are native different correction families on identical held-out folds, not a contest
with one universally better method. The explicit raw-probability statsmodels control checks
probably's family; the separate R logit GLM checks PREDVAL's family. Prediction-difference
checks and native grouped-bootstrap summaries are recorded under `results/r/`.

R `group_vfold_cv` passed the no-patient-overlap check. `group_bootstraps` supplied 1,000
native cluster resamples for each fixture. R's resamples differ from NumPy's; their intervals
are independently generated Monte Carlo results on fixed corrected predictions.

## Verification status

- Existing PREDVAL suite: 269 passed, 20 skipped because optional example/golden fixtures
  were unavailable. This does not claim those skipped integration cases passed.
- New benchmark contracts: 5 passed, 1 strict expected failure for quasi-separation.
- Ruff and `git diff --check`: passed.
- Two fresh Python environments: identical stable input/result manifests.
- isitfair: source-reviewed only; its adapter has not been executed or validated.

The independent raw-probability statsmodels checks agree with native probably predictions
within 1.04e-8 (WDBC) and 9.71e-11 (synthetic). R's matched logit-GLM checks agree with PREDVAL
within 6.73e-12 and 5.78e-15. Default IRLS stopping tolerances differ; the benchmark assertions
are 1e-6 and 1e-8 respectively, specified before the final R run.

For same-final-input repeats, the compact verification/summary command is:

```sh
python benchmarks/alternatives/verify_results.py PY_RUN_A PY_RUN_B R_RUN_A R_RUN_B
```

It verifies every Python input/result hash, checks R metrics, predictions, bootstrap draws
and session information byte for byte, and writes a compact native-R bootstrap summary.

Final same-input R repeats matched all seven stable files byte for byte (metrics, independent
agreement, both corrected vectors, both 1,000-draw bootstrap sequences, session information).
Native rsample 95% gain intervals were WDBC [-0.005282, 0.009331] and synthetic
[-0.001058, 0.022890], both crossing zero, consistent with the Python conclusion of uncertainty.
The R benchmark bodies took 142.26 and 187.48 seconds after imports; these include native
rsample bootstrap-object construction and are not workload-matched to the Python body.
R timing/session text has trailing whitespace stripped for repository hygiene; numerical
outputs are untouched. The pre-normalization repeats were also byte-identical.
