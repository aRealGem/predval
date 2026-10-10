# Reviewed-change integration: 2026-10-10

## Result

The reviewed patient-partition guard, alternatives comparator, and IRLS iteration-limit
fix are combined on local branch `integrate/reviewed-predval-20261010`.
The combined suite passes **282 tests**, with **20 optional fixture-dependent skips**,
**zero failures**, and **zero expected failures**. The comparator contracts also pass
separately: **6/6**, including the formerly expected-failure quasi-separation case.
Ruff lint, Ruff formatting (63 files), and `git diff --check` pass.

Both Python comparator reruns reproduce **all 13 stable input/result hashes exactly**
from the original benchmark. Thus the two main fixtures' point estimates, corrections,
intervals, coverage flags, and suppression summaries have not changed. This does not imply
that every possible input is unchanged: exhausted IRLS fits now become unavailable rather
than being reported as converged finite fits.

## Source and evidence provenance

- Patient-partition guard: `99ca3d72f1d4f8340df9a59de3616277e9ebdb7a`.
- Original comparator: `868276ef960f0bc7f2bc75cc79926ff7b3c2e1f1`; its original
  preregistration commits `dae6e1a` and `756e5a1` remain in the branch history.
- Reviewed convergence fix: `62c8af368824d75e1dd915e78c131f46e333a340`, cherry-picked
  without conflicts as `617ddac480849ac48de41c1c1b1d050ddc427492`.
- Integration-specific source edit: remove the strict xfail marker from
  `tests/test_benchmark.py` and update its explanatory comment. The core implementation,
  benchmark runner, R adapter, and frozen benchmark protocol are unchanged beyond the
  reviewed guard. README wording now distinguishes historical and integrated checks.
- Package metadata still says PREDVAL 0.1.0. The source commits and file hashes in
  [`verification/integrated/status.json`](verification/integrated/status.json), rather
  than the package version alone, identify the code tested here.

Original compact results and verification logs remain byte-for-byte unchanged. Their old
269-pass/5-pass/1-xfail counts describe the original comparator, not this integrated source.
New integration logs and the hash-comparison record live separately under
`verification/integrated/`. The independent statistical-stress study is **not integrated**;
its frozen baseline and worktree were not changed by this task.

## What was rerun

The full combined Python suite ran against this worktree. The comparator ran twice with
`PYTHONPATH` explicitly selecting this worktree's `src`; source imports were checked in
both interpreters. These used two existing isolated Python 3.12.14 environments originally
created from the benchmark's complete requirements lock, not two new installations during
integration. Each run verified 13 stable file hashes, for 26 new file-hash checks. Both
manifests equal the historical manifest, SHA-256:

`a2a541ec30fbffe83103e04d2c85a5e83114a671e9818488e896a2f34eaf510d`

The R adapter consumes the exported fixtures and corrected predictions, all byte-identical
here, and its source is unchanged. All seven retained R stable files were checked against
the original recorded hashes. **R computation was not repeated for integration.** The
original repeated R computation is reused evidence, not a newly executed R result.

The benchmark-body Python wall times were 3.86 and 5.47 seconds. As in the original report,
these exclude imports/installation, reflect concurrent machine load, and do not compare
individual tools' speed. Exact resource records are in the integration JSON.

## Reproduce the integration checks

Use the locked Python setup in the main benchmark README. From the repository root:

```sh
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/predval-mpl XDG_CACHE_HOME=/tmp/predval-cache
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv-benchmark/bin/python -m pytest tests benchmarks/alternatives/tests -ra
.venv-benchmark/bin/ruff check .
.venv-benchmark/bin/ruff format --check .
git diff --check
.venv-benchmark/bin/python benchmarks/alternatives/run_benchmark.py --out /tmp/predval-integrated-a
.venv-benchmark/bin/python benchmarks/alternatives/run_benchmark.py --out /tmp/predval-integrated-b
cmp benchmarks/alternatives/results/python/manifest.json /tmp/predval-integrated-a/manifest.json
cmp /tmp/predval-integrated-a/manifest.json /tmp/predval-integrated-b/manifest.json
```

The output directories must not already exist. To check the contents as well as the manifest:

```sh
.venv-benchmark/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
baseline = Path('benchmarks/alternatives/results/python/manifest.json')
expected = json.loads(baseline.read_text())['files']
for directory in ('/tmp/predval-integrated-a', '/tmp/predval-integrated-b'):
    root = Path(directory)
    assert (root / 'manifest.json').read_bytes() == baseline.read_bytes()
    for relative, digest in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
print(f'Passed: {2 * len(expected)} file-hash checks and both baseline-manifest comparisons')
PY
```

## Remaining limits

- Twenty tests did not run because optional PCam, GUSTO, and IMPROVE example/golden fixtures
  were unavailable. No full golden-output revalidation is claimed.
- The guard retains the same estimator, tolerance, and iteration limit. A valid finite-MLE
  fit can be unavailable if it needs more iterations. This change does not prove every
  unavailable-reason label is correct or detect every possible separation condition.
- The two comparator gain intervals still cross zero. They condition on fixed held-out
  predictions and do not measure repeated-sampling coverage or correction-training uncertainty.
- The original isitfair adapter remains unexecuted; source review is not an execution result.
- These are bounded engineering/method checks, not clinical validation or a superiority claim.
- All work is local. No push, pull request, merge, or deployment was performed.
