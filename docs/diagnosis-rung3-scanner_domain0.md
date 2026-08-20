# Diagnosis: rung3 cross-fit failures in `scanner_domain=0` (S4.1 item 3)

**Date:** 2026-08-19. **Status:** diagnosed, fitting behaviour intentionally unchanged (per the
S4.1 brief). This documents *why* four members withhold rung3 in one subgroup, so the flag reads as
a known limitation rather than a mystery.

## Symptom

On the PCam fixture, the `scanner_domain=0` subgroup withholds **rung3** for exactly four members:
`e2cnn`, `macenko`, `p4m_dense`, `tinyvgg_vl`. The half-pair guard (spec §4.3) does this: the
*apparent* rung3 fit succeeds for all four, but a *cross-fitted* fold fails, so the whole rung is
withheld rather than shown half-populated. rung0/rung1/rung2 are unaffected.

## What it is NOT

the review's suspected cause was "narrow logit support versus the df=4 knots" collapsing the knots. That
is **not** what happens. Checked per fold: `_rcs_knots` returns five distinct knots in every fold
for every one of the four. The knots never collapse.

## Root cause

The failure is **IRLS separation on one specific cross-fit fold**, not knot rank-deficiency.

- All four fail on the **same fold (fold4)** and only that fold. The grouped folds are
  model-independent (deterministic round-robin over the 11 slides), so fold4 holds out the same two
  slides for every member. Its leave-two-slides-out *training* set is what breaks the df=4 spline
  logistic — the IRLS solve separates / goes singular and `_irls_logistic` returns None.
- Two score geometries reach the same wall:
  - **Narrow support** — `macenko` logit range `[-1.29, 1.75]`, `tinyvgg_vl` `[-1.00, 0.29]`. The
    five knots sit close together, the restricted-cubic-spline basis columns are near-collinear
    (cond(XᵀX) ~1e6), and on fold4 the weighted design is singular.
  - **Wide, saturated support** — `e2cnn` `[-13.8, 13.1]`, `p4m_dense` `[-6.3, 13.8]`, with
    predictions near 0/1. The spline has freedom in the tails to drive fitted probabilities to the
    boundary, so fold4's training set is quasi-separable (cond(XᵀX) ~1e7).

In both regimes a **df=4 spline is simply too flexible to cross-fit** on an 11-slide,
leave-two-out subgroup for these members. The lower-parameter rungs (rung1: 1 param, rung2: 2) are
robust and fit every fold.

## Consequence and correctness

This is the harness **working as designed**. The half-pair guard refuses to show an apparent rung3
Brier with no honest held-out companion, and names the reason. As a result of this diagnosis the
reason label was corrected: rung3 IRLS separation is now reported as **`non-convergence`**, and
**`rank-deficient spline design`** is reserved for genuine knot collapse — two of the review's four
named reasons, now used precisely.

## Options (NOT implemented — backlog, needs sign-off)

If robust rung3 cross-fit on small subgroups becomes worth it: (a) ridge-penalised spline IRLS,
(b) Firth-corrected logistic for the separation cases, or (c) reduce spline df on small strata.
All change fitting behaviour and are deferred per the S4.1 brief.
