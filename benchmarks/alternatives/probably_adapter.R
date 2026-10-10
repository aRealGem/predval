#!/usr/bin/env Rscript
# Native probably calibration and matched R GLM controls, with shared patient folds.
# Rscript benchmarks/alternatives/probably_adapter.R path/to/python-results output-dir
# Requires probably, rsample, yardstick, dplyr, readr, tidyr and purrr.
suppressPackageStartupMessages({
  library(probably); library(rsample); library(yardstick)
  library(dplyr); library(readr); library(tidyr); library(purrr)
})
started <- proc.time()
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
dir.create(args[[2]], recursive = TRUE, showWarnings = FALSE)
results <- list()
agreements <- list()
for (fixture in c("wdbc_heldout", "synthetic_clustered")) {
  d <- read_csv(file.path(args[[1]], paste0(fixture, ".csv")), show_col_types = FALSE)
  d <- d |> mutate(truth = factor(if_else(y == 1, "event", "nonevent"),
                                  levels = c("event", "nonevent")),
                   .pred_event = p, .pred_nonevent = 1 - p, .row = row_number())
  # Exported assignments are identical to PREDVAL, not a newly randomized split.
  splits <- lapply(sort(unique(d$fold)), function(f) {
    train <- which(d$fold != f); test <- which(d$fold == f)
    stopifnot(length(intersect(d$patient_id[train], d$patient_id[test])) == 0)
    make_splits(list(analysis = train, assessment = test), d)
  })
  folds <- manual_rset(splits, paste0("Fold", seq_along(splits)))
  validated <- cal_validate_logistic(folds, truth, c(.pred_event, .pred_nonevent),
                                    smooth = FALSE, save_pred = TRUE,
                                    metrics = metric_set(brier_class, roc_auc))
  # Pool held-out observations; do not average AUROC across unequal folds.
  native <- bind_rows(validated$.predictions_cal) |> arrange(.row)
  stopifnot(identical(native$.row, seq_len(nrow(d))))
  d$probably_raw_probability <- native$.pred_event
  d$r_glm_logit <- NA_real_
  d$r_glm_raw_probability <- NA_real_
  for (f in sort(unique(d$fold))) {
    train <- d$fold != f
    d$z <- qlogis(pmin(pmax(d$p, 1e-6), 1 - 1e-6))
    logit_fit <- glm(y ~ z, data = d[train, ], family = binomial(),
                     control = glm.control(epsilon = 1e-11, maxit = 100))
    raw_fit <- glm(y ~ p, data = d[train, ], family = binomial(),
                   control = glm.control(epsilon = 1e-11, maxit = 100))
    stopifnot(logit_fit$converged, raw_fit$converged)
    d$r_glm_logit[!train] <- predict(logit_fit, d[!train, ], type = "response")
    d$r_glm_raw_probability[!train] <- predict(raw_fit, d[!train, ], type = "response")
  }
  stopifnot(max(abs(d$probably_raw_probability - d$r_glm_raw_probability)) < 1e-6)
  py <- read_csv(file.path(args[[1]], paste0(fixture, "_corrected.csv")),
                 show_col_types = FALSE)
  stopifnot(max(abs(py$predval_rung2 - d$r_glm_logit)) < 1e-8)
  stopifnot(max(abs(py$statsmodels_raw_probability - d$probably_raw_probability)) < 1e-6)
  agreements[[length(agreements) + 1]] <- tibble(
    fixture = fixture,
    max_abs_predval_vs_r_logit = max(abs(py$predval_rung2 - d$r_glm_logit)),
    max_abs_statsmodels_vs_probably_raw = max(abs(py$statsmodels_raw_probability -
                                                d$probably_raw_probability))
  )
  for (column in c("p", "probably_raw_probability", "r_glm_logit", "r_glm_raw_probability")) {
    p <- d[[column]]
    results[[length(results) + 1]] <- tibble(
      fixture = fixture, tool = column, brier = mean((d$y - p)^2),
      auroc = roc_auc_vec(d$truth, p, event_level = "first")
    )
  }
  write_csv(d |> select(subject_id, patient_id, fold, y, p,
                        probably_raw_probability, r_glm_logit, r_glm_raw_probability),
            file.path(args[[2]], paste0(fixture, "_r_corrected.csv")))
  # Demonstrate rsample native grouped splitting and grouped bootstrap support.
  # These random folds are a capability check, not the matched comparison partition.
  set.seed(20261010)
  grouped <- group_vfold_cv(d, patient_id, v = 5)
  stopifnot(all(map_lgl(grouped$splits, function(s) {
    length(intersect(analysis(s)$patient_id, assessment(s)$patient_id)) == 0
  })))
  set.seed(20261011)
  boot <- group_bootstraps(d, patient_id, times = 1000)
  # Conditional-on-predictions paired gain; whole patients remain intact.
  gains <- map_dbl(boot$splits, function(s) {
    b <- analysis(s)
    mean((b$y - b$p)^2 - (b$y - b$r_glm_logit)^2)
  })
  write_csv(tibble(fixture = fixture, replicate = seq_along(gains), gain = gains),
            file.path(args[[2]], paste0(fixture, "_rsample_gains.csv")))
}
write_csv(bind_rows(results), file.path(args[[2]], "r_metrics.csv"))
writeLines(trimws(capture.output(sessionInfo()), which = "right"),
           file.path(args[[2]], "sessionInfo.txt"))
write_csv(bind_rows(agreements), file.path(args[[2]], "r_agreement.csv"))
writeLines(trimws(capture.output(proc.time() - started), which = "right"),
           file.path(args[[2]], "resource_usage.txt"))
