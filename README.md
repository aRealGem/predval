# predval

A predictions-in validation harness for clinical prediction models.

predval answers one question: **do a model's published probabilities hold up on your cohort?**
It takes predictions that already exist and evaluates them — discrimination, calibration,
prespecified subgroup behaviour, and a recalibration ladder that reports how much of any
miscalibration is repairable without touching the model.

## What predval deliberately does not do

- **It never loads model weights.** No `torch`, no `tensorflow`, no checkpoints. Inputs are a
  `cohort.yaml` and a `predictions.parquet`, and nothing else.
- **It never trains or tunes a model.** The recalibration ladder fits a link-scale correction to
  published probabilities; refitting on covariates is explicitly out of contract.
- **It never touches the network.** Library code makes zero network calls.
- **It has no server and no telemetry.**

These are permanent design constraints, not a roadmap. A validation harness that could quietly
retrain the thing it is validating would not be a validation harness.

## Status

Session 1: schema, IO, and hashing. Metrics (S2) and the recalibration ladder (S3) are specified
in [`docs/spec.md`](docs/spec.md) but not yet implemented.

## Install

```bash
uv sync
uv run pytest
```

## Contracts

See [`docs/spec.md`](docs/spec.md) for the frozen v0 contracts. In short:

`predictions.parquet` is long-format — one row per (subject, model, fold, horizon):

| subject_id | model_id | fold | horizon | predicted |
|---|---|---|---|---|
| `a24ce148…` | `swin` | *null* | *null* | 0.9134 |

The central rule: **a model that did not score a subject has no row.** Absence is never
imputed as zero. predval reports coverage instead of silently filling gaps.

`cohort.yaml` declares the outcome, the prespecified subgroups, and the coverage and
completeness policies that decide whether a run is allowed to proceed at all.

## Example

`examples/pcam/` builds a real fixture from a 15-model PatchCamelyon ensemble: 19,999
whole-slide-grouped holdout patches drawn from 22 slides, with a scanner/stain subgroup and one
ensemble member whose predictions were permanently lost — a genuine coverage gap to validate
against rather than a synthetic one.

## License

Apache-2.0.
