# Produces the GUSTO rows of predrupdate_estimates.csv from an exported y,p CSV.
#
# Requires predRupdate 0.2.1 and a gusto_y_p.csv written by the command in README.md.
# Run from this directory:  Rscript gusto_reference.R

suppressPackageStartupMessages(library(predRupdate))
stopifnot(as.character(packageVersion("predRupdate")) == "0.2.1")

d <- read.csv("gusto_y_p.csv", colClasses = c("integer", "numeric"))
y <- d$y
p <- d$p
stopifnot(all(y %in% c(0, 1)))
# The comparison is only defined off the boundary: logit(0) and logit(1) are infinite.
stopifnot(all(p > 0), all(p < 1))

v <- pred_val_probs(binary_outcome = y, Prob = p, cal_plot = FALSE)
METRIC <- c(calibration_intercept = "CalInt", calibration_slope = "CalSlope",
            auroc = "AUC", brier = "BrierScore")
TOL <- c(calibration_intercept = "1e-5", calibration_slope = "1e-5",
         auroc = "1e-8", brier = "1e-10")
rows <- lapply(names(METRIC), function(metric) {
  data.frame(input = "gusto", model = "gusto_west_refit_logistic", metric = metric,
             value = sprintf("%.17g", as.numeric(v[[METRIC[[metric]]]])),
             tolerance = TOL[[metric]])
})
write.csv(do.call(rbind, rows), "gusto_estimates.csv", row.names = FALSE, quote = FALSE)
cat("wrote gusto_estimates.csv\n")

# --- glm convergence sweep ---------------------------------------------------
# The calibration intercept is the one estimate whose difference from predval is visible above
# the floating-point floor. R's default stopping rule is responsible; see README.md.
cat("\nglm calibration-intercept sweep (y ~ 1, offset = qlogis(p)):\n")
for (eps in c(1e-8, 1e-10)) {
  fit <- glm(y ~ 1, offset = qlogis(p), family = binomial,
             control = glm.control(epsilon = eps))
  cat(sprintf("  epsilon %-6s intercept %.17g  iterations %d\n",
              format(eps), as.numeric(coef(fit)[1]), fit$iter))
}
