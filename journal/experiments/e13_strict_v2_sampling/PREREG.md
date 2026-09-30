# E13 prospective corrective record — strict no-replacement sampling v2

Recorded on **2026-07-23**, before generating any strict-v2 pool, training any
strict-v2 detector, or inspecting any strict-v2 result.

## Purpose and evidential status

E13 corrects the training-array construction for nominal synthetic
augmentation `+100%` arms. The legacy pools contain 65,000 windows per attack
class, whereas a Car-Hacking `+100%` arm requests 65,538/65,537/65,537/65,537
windows. The legacy sampler therefore used replacement. E13 extends the
already specified Rule and protocol-valid WGAN configurations and draws
without replacement from separately named pools.
Because changing the requested pool size re-instantiates Rule draws, WGAN
samples, and final permutations, the legacy-to-v2 contrast is a joint
pool-reinstantiation plus no-replacement corrective sensitivity. It does not
isolate a causal replacement-policy effect.

This is a corrective sensitivity rerun, not a new hypothesis, generator
family, detector family, dataset, or realism claim. Existing checkpoints and
results remain immutable and are labelled as nominal pool-reuse sensitivity.
The clean-dose interpretation remains blocked until the complete E13 matrix
and its audits exist.

## Frozen construction and provenance

- Source split: Car-Hacking train only. External datasets remain
  evaluation-only.
- Frozen source archives:
  `train_windows.npz` SHA-256
  `44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4`;
  `val_windows.npz` SHA-256
  `b4920e7f60fbf38b38e12976129f120ebff160a21a4c77093f3746f3c1aa218e`.
  Both match the frozen WISA processed-dataset hash record and are rechecked
  when a strict-v2 training bundle is consumed.
- Frozen local evaluation arrays:
  `test_windows.npz`
  `4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7`,
  `variant_test_windows.npz`
  `91bd4d5dca2bd73907fad11bb94ea6dec075f60bb3b24e3c517617bf2d50b2ba`,
  `variant_sensitivity_windows.npz`
  `4266fb3bbb3e248ef1ab53aaca19cb4534a3815a18c9134dc3db446cdfcf5674`,
  `target_id_shift_stress_windows.npz`
  `9f685e629b4c8acaf3312d95dd1d94145c502b2c07fedd8da9a938526ed92bc1`,
  `out_of_generator_stress_windows.npz`
  `c91e5a026f9f0b771e9551a471efeb1faca66bda21e7035bc5b3620927dfa346`,
  and `otids_cross_windows.npz`
  `b521f3546f633c6220e2696234abb6d7c37a03d64f19e71ec37bf36a100ed596`.
  The first five match the frozen WISA processed-dataset hash record; all six
  are hard-gated before a strict-labelled ladder evaluation.
- Frozen streamed-external inputs:
  `can-train-and-test.zip`
  `a9c607b38bd28f1768021ad01c29ffbfe4e82bb0ae5815ac3ce7ad74751ae061`
  and `road.zip`
  `0e4fe6ed7f99b5cdabf6b772c91a2025c614a071ba8579c532904d0ef7aea5f6`.
  ROAD inference consumes 41 processed parquet captures; their sorted
  canonical per-file manifest SHA-256 is
  `1d434f4043ec7fdfddba636f2c479eff35ef246a331586a72f20236a271a5c64`.
  The capture-role profile SHA-256 is
  `0d6202ae75e7e8e883ef5fb40fd205c205803b25307932c2b2bd9cf0eef7df83`.
- Detector-training pipeline seeds: **7, 42, 123, 2026, 3407**.
- Sampling policy:
  `strict_without_replacement_v2`, class-balanced, deterministic remainder
  assignment to the class-order prefix, and seed rule
  `pipeline_seed + int(ratio * 1000)`.
- Required capacity: at least **65,538 windows per attack class**.
- Rule construction seed: **314159**.
- WGAN generator-training seeds: **314159, 271828, 161803**.
- WGAN pool-sampling seed: **314159** for all three generator configurations.
- Frozen WGAN generator checkpoint SHA-256 values:
  seed 314159
  `2a83647aafc9ae62df8d390ac5c896a4c6307e825e083b4c33bf09b5bef04517`;
  seed 271828
  `bc1003d286bab53f2b9ad9a534391f626a94c8e55e90084e03e01a23714886c8`;
  seed 161803
  `a13c88185dc0c6a3123c8540f8dff13f002770c599c634dd1a47911318beea01`.
- Frozen WGAN standardizer SHA-256 for all three configurations:
  `7ffaff380c0e4af079ca68f94bd9c1ec35e597be99266e3631d7284d8a05e389`.
- Official generation and training require a clean tracked/source state and
  record its source commit. Previously generated Git-visible no-clobber
  artifacts under the declared result/model/synthetic roots may remain
  between sequential processes, and their paths are recorded. Gitignored
  pool/model inputs are not claimed to be enumerated by Git; their exact
  bytes are bound by generation/training SHA-256 records. Uncommitted source,
  test, manuscript, or experiment-definition files remain a hard failure.
  Every pool records its identifier, exact generation
  configuration, source/input/output hashes, per-class counts, protocol and
  condition checks, and exact row-content uniqueness.

The four immutable strict-v2 pool identities are:

1. `rule_based_windows_unique_pool_v2.npz`
2. `gan_protocol_valid_windows_unique_pool_v2.npz`
3. `gan_gseed271828_protocol_valid_windows_unique_pool_v2.npz`
4. `gan_gseed161803_protocol_valid_windows_unique_pool_v2.npz`

Any missing metadata, hash mismatch, wrong generator seed for the declared
path, class shortage, duplicate exact row, or protocol/condition failure stops
the run. The training consumer independently rechecks pool schema, shape,
frame protocol, condition metadata, exact-content uniqueness, and the embedded
content digest, then requires the current NPZ SHA-256 to equal the designated
generation log's output SHA-256. It must not fall back to a legacy pool or
replacement sampling.

## Fixed detector arms

The full rerun matrix is:

- Rule `+100%`: CNN, RF, BiLSTM, and transformer, five pipeline seeds each.
- Protocol-valid WGAN primary configuration `+100%`: CNN and RF, five seeds
  each.
- Protocol-valid WGAN generator configurations 271828 and 161803 `+100%`:
  CNN and RF, five seeds each.
- Matched-optimizer-step WGAN primary `+100%`: CNN, five seeds.

Canonical original-schedule artifacts use the `sampling_v2` model/result tag.
The matched-step arm uses `matchedsteps_sampling_v2`. Smoke runs use a
different tag containing `smoke` and are never merged into the five-seed
matrix. Any reduced setting, dataset, or rung scope also requires an
additional output scope token beyond the model tag.

The retained untagged `real_only`, `rule_0p30`, and `over_1p00` controls are
not retrained because their source arrays are unaffected. `over_1p00` remains
the intentionally duplicated real-attack equal-budget control and must not be
described as strict no-replacement synthetic sampling.

## Fixed exclusions

- Conv-AE has no `+100%` arm in the current study; it is not retrained.
- AR-LM has no `+100%` arm in the current study; its primary `+30%` evidence is
  not rerun.
- All other `+30%` arms are within existing per-class pool capacity and are
  not rerun.
- No new external dataset, generator family, detector architecture, or model
  selection rule is added.
- A BiLSTM matched-update arm remains optional and cannot be used to replace
  the fixed original-schedule E13 result.

## Fixed evaluation and aggregation

Each strict-v2 training log is a complete bundle sidecar: it records the
sampling audit and hashes of every checkpoint and standardizer that belongs to
that run. Partial, mismatched, or tampered bundles are rejected instead of
being resumed or skipped.

Evaluation reads only the requested versioned checkpoint matrix and refuses
missing seeds or legacy fallback. Per-seed rows are saved before aggregation.
The five pipeline seeds are the inferential units; summaries report mean,
sample SD, and paired untruncated 95% Student-t intervals. Prespecified
supporting multiplicity adjustments retain their existing comparison family.
Before inference, every strict-labelled evaluator requires the clean committed
source state, requires it to equal the strict training-bundle commit, records
the evaluator entrypoint and provenance-helper SHA-256, and verifies the exact
frozen inputs for its requested rung/dataset scope.
The run log records those input hashes and the ROAD per-file manifest where
applicable. BiLSTM and transformer inference loads the exact persisted
training standardizer; every requested saved mean/std array must be
byte-exact equal, including dtype. Mixed retained-control/strict evaluations
hash every actual tagged and untagged checkpoint and standardizer before
inference. The E13 BiLSTM ladder excludes the legacy Conv-AE and its
synthetic-calibrated threshold. A code, commit, input, model, manifest, or
scaler mismatch stops the run before canonical output publication.

For the E8 evaluation-realization diagnostic, the E13 rerun uses journal
models only: tagged `rule_1p00` strict-v2 checkpoints plus the retained
untagged journal `real_only` and `rule_0p30` controls. It keeps the original
three blocks, rungs, seeds, metrics, and aggregation unchanged. This record is
the required prospective journal-run record. The E8 extension remains
diagnostic and cannot be promoted to a primary causal claim.

## Reporting rule and stopping

Binary attack detection and exact attack-type recall remain separate. External
results remain FPR/normal-recall first. Legacy-v2 differences are reported as
obtained, including null, sign-changing, or adverse changes. No arm, seed,
setting, rung, or metric is removed or changed after outcomes are inspected.
Legacy-v2 differences are labelled as the joint corrective sensitivity above,
not as an isolated effect of replacement sampling.

The sampling correction alone does not license a monotone dose claim. A clean
`+100%` statement requires the full prespecified matrix, complete bundle and
pool audits, per-seed tables, paired summaries, synchronized figures and
claim ledger, and a fresh reproducibility snapshot from a clean tag.
