# Regenerates synpm_y_p.csv.gz and the SYNPM rows of predrupdate_estimates.csv.
#
# Requires predRupdate 0.2.1 installed from the tarball named in README.md.
# Run from the repository root:  Rscript validation/predrupdate/make_reference.R
# Inputs and outputs are resolved relative to this script, so running it from this
# directory works as well.

# dirname of the --file= entry that Rscript passes to the interpreter.
script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) != 1L) {
    stop("cannot locate this script; run it with Rscript, not source()")
  }
  dirname(normalizePath(sub("^--file=", "", file_arg)))
}
HERE <- script_dir()

suppressPackageStartupMessages(library(predRupdate))
stopifnot(as.character(packageVersion("predRupdate")) == "0.2.1")
data(SYNPM)

info <- pred_input_info(model_type = "logistic",
                        model_info = SYNPM$Existing_logistic_models)
pr <- pred_predict(x = info, new_data = SYNPM$ValidationData, binary_outcome = "Y")
stopifnot(length(pr) == 3L)

y <- pr[[1]]$Outcomes
for (m in 2:3) stopifnot(identical(y, pr[[m]]$Outcomes))
stopifnot(all(y %in% c(0, 1)))

P <- sapply(1:3, function(m) pr[[m]]$PredictedRisk)
# The comparison is only defined off the boundary: logit(0) and logit(1) are infinite.
stopifnot(all(P > 0), all(P < 1))

# 17 significant digits round-trips a double exactly.
g17 <- function(v) sprintf("%.17g", v)
out <- data.frame(y = as.integer(y), p_model1 = g17(P[, 1]),
                  p_model2 = g17(P[, 2]), p_model3 = g17(P[, 3]))
con <- gzfile(file.path(HERE, "synpm_y_p.csv.gz"), "wt")
write.csv(out, con, row.names = FALSE, quote = FALSE)
close(con)

METRIC <- c(calibration_intercept = "CalInt", calibration_slope = "CalSlope",
            auroc = "AUC", brier = "BrierScore")
TOL <- c(calibration_intercept = "1e-5", calibration_slope = "1e-5",
         auroc = "1e-8", brier = "1e-10")
rows <- list()
for (m in 1:3) {
  v <- pred_val_probs(binary_outcome = y, Prob = P[, m], cal_plot = FALSE)
  for (metric in names(METRIC)) {
    rows[[length(rows) + 1]] <- data.frame(
      input = "synpm", model = paste0("model", m), metric = metric,
      value = sprintf("%.17g", as.numeric(v[[METRIC[[metric]]]])),
      tolerance = TOL[[metric]])
  }
}
write.csv(do.call(rbind, rows), file.path(HERE, "synpm_estimates.csv"),
          row.names = FALSE, quote = FALSE)
cat("wrote synpm_y_p.csv.gz and synpm_estimates.csv\n")
cat("merge synpm_estimates.csv with the gusto rows to rebuild predrupdate_estimates.csv\n")
