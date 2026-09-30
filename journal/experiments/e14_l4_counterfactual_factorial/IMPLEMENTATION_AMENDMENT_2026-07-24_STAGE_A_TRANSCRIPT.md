# E14 Stage-A transcript-readiness amendment — 2026-07-24 UTC

Status: prospective technical amendment written after matched-real training but
before `--prepare-only`, checkpoint inference, factorial scoring, analysis, or
inspection of any E14 factorial outcome.

This amendment corrects one provenance-readiness omission. It does not change
the registered bases, factors, cells, arms, seeds, endpoints, contrasts,
aggregation order, multiplicity families, scientific-verdict rules, training
protocol, or checkpoint-selection rule.

## 1. Discovery sequence and technical stop

The production matched-real wrapper completed successfully:

```text
start_utc: 2026-07-24T16:57:31.904002Z
end_utc:   2026-07-24T17:02:06.893259Z
source commit:
  ef7afd2da361718c3a6fe4ad75cd82068430c61c
```

The following sequence then occurred:

1. all five new checkpoint/log bundles and the wrapper records passed an
   independent artifact audit;
2. a read-only Stage-A readiness call was attempted;
3. preparation stopped before constructing or publishing any Stage-A output
   because `source_provenance_for_prepare()` reproduced:

```text
canonical E14 preparation found undeclared untracked files:
["journal/experiments/e14_l4_counterfactual_factorial/matched_real_training_child_output_v1.log"]
```

At discovery, `prepare_v1.json`, all three Stage-A tables, all Stage-B score
tables, the Stage-B `run_v1.json`, the effects table, and the final artifact
manifest were absent. No checkpoint had been loaded by the evaluator, no model
inference had occurred, and no E14 factorial result had been computed. The only
observed model-dependent values were the validation histories necessarily
produced by the already registered matched-real training protocol.

## 2. Cause and frozen production bundle

The first implementation amendment required the wrapper to preserve a separate
child stdout/stderr transcript. The wrapper did so, and the run record
cross-linked it. The evaluator's Stage-A untracked-artifact allowlist admitted
the preflight, run record, checkpoints, and logs, but accidentally omitted that
required transcript path.

The successfully audited production records are frozen byte-for-byte:

| Artifact | SHA-256 |
|---|---|
| `matched_real_training_preflight_v1.json` | `3b895fe3c84e5188968fedf0ef0450f2ace03c98e3fe1e3ea11aa3a8f98e1e14` |
| `matched_real_training_run_v1.json` | `53865949ecf4383fa6f9a21c224b76c289baeb9c88daa7fbf80eddfacfe0de28` |
| `matched_real_training_child_output_v1.log` | `e362b71ca4cdff7755e27b16ca614ca2f1e574c12c8fc95777aeb095a631e0e5` |

The transcript is 4,977 bytes, contains 4,853 stdout bytes and zero stderr
bytes, preserves the streams separately, and contains no `skip existing`
line. No matched-real failure record exists.

The five seed bundles are also frozen:

| Seed | Checkpoint SHA-256 | Log SHA-256 | Total steps | Validation checks | Selected step |
|---:|---|---|---:|---:|---:|
| 7 | `2eccea8cfc2d734602b8d70f0cd728b7b9c311e262dc51bdec4e7c9957f9fc32` | `91a8c9b5338daf5f56274c595b1951677344d758c450d3e7e2dc69f00462fe8c` | 6156 | 12 | 6156 |
| 42 | `96df37d40ea9afd7694d6e15f6c38de4c18283e7e2ca2fa8b60207fcb9d4e32d` | `918ded24729fccdb65e8925e0f982e33e2d47a34d3623e1097d236c20ff115e5` | 6156 | 12 | 6156 |
| 123 | `b32a663c646ccfd50538af024ec2107cf167462d814487287359301ff2ee0ebb` | `6b2caa3e895f4e362bf006ce552c8f55db452293e62acffc23334d4f7691c8b2` | 6156 | 12 | 3591 |
| 2026 | `1020d15f99c0de5335633ce9de97f53485faf281fc3e5cafa1182a993e6d7a94` | `5e32902651194ae8fcd729fb969a2f22680e0f8be4ec9f834280009b091581ed` | 6156 | 12 | 5130 |
| 3407 | `d0005ebfbef6562450a70e0c8e4d9ef5769e792f66d6725340fa551a9a673a88` | `ac75d7e8f81fba7e1b0a792f654387afe77561f24c74eb590a1c22e111024b37` | 6156 | 12 | 3591 |

Every checkpoint passed a `weights_only=True` load, the exact registered
23-key architecture and shapes, and the all-tensors-finite gate.

## 3. Prospective correction

This amendment must be committed alone as the first descendant of training
commit `ef7afd2da361718c3a6fe4ad75cd82068430c61c`. A second commit may then change
only:

```text
journal/scripts/evaluate_l4_counterfactual_factorial.py
journal/scripts/analyze_l4_counterfactual_factorial.py
journal/tests/test_l4_counterfactual_factorial.py
```

The correction must:

- add the canonical transcript path to the Stage-A allowlist;
- verify the actual transcript path, byte count, SHA-256, format, separate
  stream flag, stdout/stderr byte counts, and empty skip-line list against the
  unchanged training run record;
- include the transcript in `prepare_v1.json` and revalidate it before scoring
  and final analysis;
- hash this amendment in Stage A and the final artifact manifest;
- record and verify the exact ancestry and path sets for the two-commit
  transition from the training implementation to the amended preparation
  implementation.

The generic trainer, matched-real wrapper, preregistration, first amendment,
existing production artifacts, scientific estimands, and verdict logic must
not change.

## 4. No rerun and remaining order

The existing v1 training bundle must not be deleted, renamed, overwritten, or
rerun. The blocker is independent of learned weights and is fixed entirely by
the Stage-A provenance gate. Repeating the same seeds after their validation
histories were observed would add provenance and selection concerns without
repairing the omission.

After this amendment and the limited implementation correction are committed
and independently tested:

1. re-run the read-only Stage-A readiness and full 30-bundle audit;
2. execute `--prepare-only` exactly once;
3. verify and commit the training/preparation records and manifests without
   changing implementation code;
4. require a fully clean worktree before any Stage-B scoring;
5. score and analyze only under the original complete-outcome rules.
