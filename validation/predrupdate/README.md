# Cross-implementation check against predRupdate

This directory holds the inputs and reference values for the comparison described in
the "Cross-implementation check" section of the top-level README, and the test that
asserts it.

## What was compared

predval's point estimates for calibration intercept, calibration slope, AUROC and Brier
score, against `predRupdate::pred_val_probs()` given identical outcome and
predicted-probability vectors.

## Reference implementation

- Package: **predRupdate**, version **0.2.1**
- Licence: MIT (see `THIRD_PARTY_LICENSE` in this directory)
- Citation: Martin G, Jenkins D, Sperrin M (2025). *predRupdate: Prediction Model
  Validation and Updating*. R package version 0.2.1.
  doi:10.32614/CRAN.package.predRupdate
- Source used: `predRupdate_0.2.1.tar.gz` from CRAN,
  sha256 `80da821af02a52d4915aedafb47cff7efbfab9219b90a4c0ca0b9e49d061c4fb`,
  installed from that tarball rather than from the CRAN index, so the version and hash
  recorded here are the ones that produced the reference values.

## Versions that produced the reference values

| Component | Version |
|---|---|
| R | 4.5.0 (2025-04-11), aarch64-unknown-linux-gnu |
| predRupdate | 0.2.1 |
| pROC | 1.18.5 |
| Python | 3.13.5 |
| numpy | 2.4.6 |
| predval | 0.1.0 |

## Files

- `synpm_y_p.csv.gz` — outcomes and predicted risks for the three existing logistic
  models in `predRupdate::SYNPM`, evaluated on `SYNPM$ValidationData` via
  `pred_input_info()` + `pred_predict()`. Columns `y, p_model1, p_model2, p_model3`;
  20,000 rows; 2,830 events; every probability written with 17 significant digits, which
  round-trips a double exactly. Derived from predRupdate's data under its MIT licence.
- `predrupdate_estimates.csv` — the 16 reference values (4 inputs x 4 metrics) with the
  tolerance asserted for each, at 17 significant digits.
- `THIRD_PARTY_LICENSE` — predRupdate's MIT licence in full.
- `make_reference.R` — regenerates `synpm_y_p.csv.gz` and the SYNPM half of
  `predrupdate_estimates.csv`.
- `gusto_reference.R` — produces the GUSTO half from an exported `y,p` CSV.

## Reproducing

The committed data file is what the test reads, so no R is needed to run the test:

    uv run pytest tests/test_predrupdate_agreement.py

To regenerate the reference values you need R with predRupdate 0.2.1 installed from the
tarball named above. **Every command below is run from the repository root.** Both R
scripts resolve their own inputs and outputs relative to their own location, so running
them from this directory instead also works.

    Rscript validation/predrupdate/make_reference.R

That reproduces the *content* of `synpm_y_p.csv.gz`, not its gzip blob: the decompressed
bytes match the committed file byte for byte, the compressed bytes need not.

For the GUSTO half, build the GUSTO fixture first (see `examples/gusto/`), then:

    uv run python -c "from pathlib import Path; import pandas as pd; \
      c = pd.read_parquet('examples/gusto/cohort.parquet'); \
      p = pd.read_parquet('examples/gusto/predictions.parquet'); \
      d = c.merge(p, on='subject_id').sort_values('subject_id'); \
      out = Path('validation/predrupdate/gusto_y_p.csv'); \
      out.write_text('y,p\n' + ''.join(f'{int(a)},{b:.17g}\n' \
        for a, b in zip(d['label'], d['predicted'])))"
    Rscript validation/predrupdate/gusto_reference.R

The three files those commands write — `gusto_y_p.csv`, `synpm_estimates.csv` and
`gusto_estimates.csv` — are gitignored; `predrupdate_estimates.csv` is the merge of the
two `*_estimates.csv` halves.

## Note on the residual differences

The two implementations fit the calibration intercept and slope with separate iteratively
reweighted least squares routines that stop on different criteria. predval stops when the
largest absolute change in any coefficient falls below 1e-9, with a cap of 50 iterations.
R's `glm` stops when the relative change in deviance falls below 1e-8, with a cap of 25
iterations. Both tolerances are tighter than the 1e-5 asserted here. AUROC and Brier
involve no iterative fit.

The GUSTO intercept difference of 2.1e-8 is against R's default glm.control(epsilon = 1e-8), under which the fit stops after 5 iterations. With epsilon = 1e-10 or smaller the fit takes 6 iterations and the difference from predval is 4.2e-15. gusto_reference.R prints both.
