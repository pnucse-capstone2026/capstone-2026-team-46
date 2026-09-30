# B-v1 evaluation implementation contract

This implements, without changing, the already frozen B protocol. The main
training grid is running. No B main checkpoint may be loaded for inference
until the entire 1,000-fit grid and its paired training receipts are complete.
The evaluator is separately hash-frozen before new B test outcomes.

## Preparation without new model outcomes

Use exactly the frozen historical 6,000 bases and 12,000 Gear/RPM latents.
Verify source roles first; the sealed 24,000-window tail is not selected.
For each block, family and k={2,4,8,16,32}, reproduce the existing E20 intact
and broken transforms. Independently check untouched channels, off-injection
values, per-window target multisets, intact predicates and reported broken
predicate counts. For RPM also check each window's unavoidable residual.
Small-k residual/unchanged pairs are reported, not discarded or retuned.
All 12,000 k=32 intact transforms must match E14 cell 000/canonical exactly.

Cache standardized float32 N×128×11 arrays for 60 attack cells and 3 normal
blocks (126,000 total windows) with the original real-train standardizer.
Serialize, reload and hash each cache. CPU float32 inference, 8 intra-op
threads, 1 inter-op thread and batch 4,096 retain E20's scoring convention.
No GPU inference or additional training is part of this preparation.

One fixed historical shared real-only checkpoint (pipeline 7) is the score
continuity control. Reproduce all its 60 E20 integer exact/binary counts using
the cached inputs; these are already-observed historical outcomes, not B
results. This verifies model loading, standardization, batching, argmax and
cell labeling together. A failure blocks evaluator freezing, not the running
training grid. Unit tests additionally use constructed counts and synthetic
windows, never partial B outcomes.

## Completion gate and scoring

Require the main launcher completion receipt, all 100 preparation receipts,
all 500 paired training receipts, exactly 1,000 declared model/fit identities,
the 6,156-update/12-validation budget, registered earliest-best selection,
paired initialization/sampling/RNG progression and matching standardizers.
Verify checkpoint file hashes before loading. No partial-grid override.

Score every declared checkpoint against the same 63 cache cells. Preserve
all uint8 class predictions and integer counts in new per-fit outputs. No
checkpoint, pool, old E20 result, training script or protocol is modified.
Failures are preserved; the evaluator refuses overwrite and has no silent
retry/resume or outcome-dependent filtering policy.

## Aggregation and interpretation

Require exact grid coverage and unique keys before aggregation. Recompute
recalls from integer counts, with n=2,000 for each cell. Compute family/block
equal-weight recalls per construction×pipeline, then average the five fixed
pipelines within construction. The inference unit is the construction, n=100.

Report U and L separately at all five k values, per-family and equal-family
macro, exact and binary recall; retain all four arm/condition absolute recalls.
Only exact-macro U(8) and L(8) get co-primary 97.5% t intervals. All other
intervals are pointwise descriptive 95%, including normal FPR and its paired
Rule-minus-placebo difference. Provide per-pipeline sensitivity with the same
construction units. No equivalence margin, best-k selection or conversion of
nonsignificance into absence of utility.

The freeze receipt binds this file, implementation, tests, training freeze,
cached input hashes and historical continuity result. It is a local
prospective implementation freeze, not a public preregistration. Main scoring
is an explicit later action; nothing here auto-starts it while training runs.
