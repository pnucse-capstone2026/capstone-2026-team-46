# E14 implementation amendment — 2026-07-24 UTC

Status: prospective technical amendment, written before matched-real training,
`--prepare-only`, checkpoint inference, or any E14 outcome was run.

This amendment closes implementation-audit findings without changing the
registered bases, factors, cells, arms, seeds, endpoints, contrasts,
aggregation order, multiplicity families, or scientific-verdict rules in
`PREREG.md`.

## 1. Atomic training-artifact publication

The generic trainer registered in PREREG Section 3.1 used an existence check
followed by direct `torch.save()`/text writes. That left a time-of-check to
time-of-use interval in which a racing file or dangling symlink could be
overwritten.

The trainer now stages each checkpoint and log in the target directory, flushes
and `fsync`s it, and publishes it with an exclusive hard link. A target that
appears at any point causes a technical stop and is never replaced. The exact
registered child argv and every training, sampling, optimization,
standardization, validation, and checkpoint-selection rule remain unchanged.

The Section 3.1 trainer hash is therefore superseded as follows:

```text
path:
  journal/scripts/train_generator_extension_cnn.py
registered SHA-256:
  b4a95cdb476805dae4d58cd6a81082f5f92f5858864d1cd821b175b1d93e9247
amended SHA-256:
  a5d88bcb579da77e234e991a9b050c040df21d2f75e971c0c952c6226749db61
```

The wrapper also preserves separate child stdout/stderr transcripts and emits
a versioned, atomic failure record after an execute-mode preflight if the
child or any post-training gate fails. A trainer `skip existing` message is a
technical failure rather than evidence of a completed seed.

## 2. Legacy optimizer-step reporting

PREREG Section 8.2 requires realized total and selected optimizer steps. Newer
logs record these values directly. Some frozen legacy logs predate those
fields, so their values are derived without outcome-dependent choice:

```text
steps_per_epoch = ceil(train_windows / 512)
total_optimizer_steps = steps_per_epoch * epochs_run
selected_optimizer_step = steps_per_epoch * selected_epoch
```

The preparation record labels whether each value was recorded or derived and
retains the source log hash. Matched-step logs must use their recorded
`optimizer_step` history and are not eligible for this legacy derivation.

## 3. Exact-zero legacy direction

The three PREREG Section 8.2 budget classes did not define the boundary case in
which a resolved matched-budget primary effect has nonzero sign but the
corresponding legacy point estimate is exactly zero. Zero is neither the same
nor the opposite strict sign. This case is fixed prospectively as:

```text
budget-direction-unresolved-zero
```

It cannot support `budget-stable`, cannot be called `budget-discordant`, and
cannot upgrade or otherwise alter the primary matched-budget verdict.

## 4. Failure-state records

A failed preparation invocation publishes a versioned
`T-STOP-MANIPULATION` record. A failed scoring invocation publishes a
versioned `T-INCOMPLETE` record. These records contain the exception,
traceback, output-presence audit, and any retained isolated staging directory;
they never contain or license a scientific verdict. Canonical success outputs
remain atomic and no-clobber.

The analyzer likewise retains an isolated staged bundle if final publication
fails after computation, while rolling back any canonical links created by
that failed publication.

## 5. Audit-enforcement clarifications

The following changes implement rules already present in PREREG and do not
amend the analysis:

- cell `000` is checked against an independent conditional Rule-D0 byte oracle;
- the analyzer revalidates the Stage-A implementation hashes and every
  Stage-B output path, byte count, and SHA-256 before issuing `T-PASS`;
- only the three registered seven-test families retain test p-values;
  supporting analyses retain nominal confidence intervals only;
- block sensitivity reports the three block values plus descriptive mean,
  sample SD, minimum, and maximum.

This amendment must itself be hashed by the matched-real preflight and
`prepare_v1.json`.
