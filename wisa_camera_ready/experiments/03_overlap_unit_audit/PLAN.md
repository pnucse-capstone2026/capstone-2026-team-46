# Overlap-Aware Evaluation-Unit Audit v1

Status: frozen before any overlap-thinned model prediction was generated or inspected  
Frozen UTC date: 2026-07-29  
Accepted-paper source commit: `57880aa350ddf0f916b7e1bd2e940529d0fc6ea6`  
Camera-ready state immediately before this plan:
`2e0df4893ced5258ecbb7306a1ca8ba75f690204`

## Purpose and evidential status

This post-review evaluation-only audit addresses the concern that adjacent
128-frame windows use stride 32 and therefore overlap by 75%. It asks whether the
two primary generated-evaluation contrasts from the matched-update analysis retain
their direction after restricting evaluation to base windows that share no raw
frames.

No model is retrained or selected. The audit reuses the 15 source-validation-selected
checkpoints from matched-update v1. It does not create capture-, vehicle-, generator-
construction-, or population-level replication. Its strongest licensed statement is
that a result does or does not persist on one deterministic, raw-frame-disjoint
subset of the existing generated evaluations.

The aggregate matched-update results and reviewer comments were known when this plan
was written. During a feasibility check before this plan, the train/validation/test
processed Parquet files were observed to have disjoint contiguous
`(source_file, row_in_source)` ranges. That integrity outcome is therefore post-hoc
and must not be described as preregistered. No checkpoint prediction on the
overlap-thinned fixed or sensitivity subsets had been generated or inspected before
this plan.

## Frozen data and model lineage

| Role | Path | SHA-256 |
|---|---|---|
| source test windows and base provenance | `datasets/windows/test_windows.npz` | `4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7` |
| fixed generated evaluation | `datasets/windows/variant_test_windows.npz` | `91bd4d5dca2bd73907fad11bb94ea6dec075f60bb3b24e3c517617bf2d50b2ba` |
| sensitivity generated evaluation | `datasets/windows/variant_sensitivity_windows.npz` | `4266fb3bbb3e248ef1ab53aaca19cb4534a3815a18c9134dc3db446cdfcf5674` |
| fixed-variant generator | `wisa/scripts/generate_variant_test.py` | `fde7346a8f32a84c03c0c0340a9c892764ba5e2ef737d6afbab59b64194595bc` |
| sensitivity generator | `wisa/scripts/generate_variant_sensitivity.py` | `6368717317c679f79f41ca05626fdbcc427db61ceeca419f60f662b67a916069` |
| matched-update training manifest | `wisa_camera_ready/experiments/02_matched_update/training_manifest_v1.json` | `c03419848737d29714381c84f84e8587756ec4b2685a26e9cb2be46b0a9fb228` |
| matched-update final manifest | `wisa_camera_ready/experiments/02_matched_update/manifest_v1.json` | `cc582638a9b086cff6f0cb0a7374ce72396205fd915ac168c554a939e0767671` |
| matched-update implementation | `wisa_camera_ready/scripts/run_matched_update_v1.py` | `afacf325e8fca206d5ae15ac928fcc98a1fd8d70d77433962fbbac25deee6f71` |
| real-train-only standardizer | `wisa_camera_ready/models/matched_update_v1/standardizer_source_real_train_v1.npz` | `14f5ee3f90429c04c08da93d00ac2cfc2f13ac7727347838f14af68cba731810` |

The evaluator must verify every path and hash above, every checkpoint hash recorded
in the matched-update training manifest, and the selected update recorded for each
arm and pipeline seed.

Frozen arms:

1. `real_only`;
2. `rule_0p30`;
3. `real_oversampling_0p30`.

Paired pipeline seeds are `7`, `42`, `123`, `2026`, and `3407`. These remain
pipeline repeats conditional on one split and one Rule construction.

## Deterministic base-origin reconstruction

The generated NPZ files do not retain the selected source-test indices. The source
generators select all base indices before applying any payload/timing mutations and
then apply a final RNG permutation after all mutations. The evaluator reconstructs
both the selected indices and final permutation without using labels or predictions.
It must replay the frozen generator on the frozen source-test bases and require every
saved generated tensor, label, and metadata array to be bit-exact.

Plan amendment before model prediction: the first implementation preflight correctly
failed because the initial plan omitted the final generator permutation. No
overlap-thinned checkpoint prediction was run or inspected. The rules below replace
the initial row-order assumption and were frozen before model inference.

### Fixed generated evaluation

1. Load `test_windows.npz` and set
   `normal_idx = flatnonzero(y_binary == 0)`.
2. Initialize `numpy.random.default_rng(42)`.
3. Compute
   `selected = rng.choice(normal_idx, size=64000, replace=False)`.
4. Replay the four frozen injection loops on the selected base windows to advance
   the same RNG state and reconstruct the unshuffled generated arrays.
5. Reconstruct `order = rng.permutation(64000)`.
6. Require the replayed `x`, binary/exact labels, variant type, and injection counts
   after `order` to be bit-identical to the saved NPZ arrays.
7. Map saved generated row `i` to source-test row `selected[order[i]]`.

### Sensitivity generated evaluation

1. Use the same source-test `normal_idx`.
2. Initialize `numpy.random.default_rng(20260611)`.
3. Compute
   `selected = rng.choice(normal_idx, size=30000, replace=False)`.
4. Replay the twelve frozen scenario injection loops on the selected base windows to
   advance the same RNG state and reconstruct every unshuffled array.
5. Reconstruct `order = rng.permutation(30000)`.
6. Require the replayed `x`, binary/exact labels, family, severity, scenario, burst,
   step, amplitude, and injection-count arrays after `order` to be bit-identical to
   the saved NPZ arrays.
7. Map saved generated row `i` to source-test row `selected[order[i]]`.

The evaluator must record the SHA-256 of each reconstructed contiguous native-endian
`int64` selected-index array, final permutation, and ordered base-index array. Failure
of any reconstruction or validation gate stops the audit without producing result
tables.

## Frozen non-overlap rule

For each mapped generated row, attach the source-test `source_file`, `segment_id`,
`start_index`, and `end_index`.

A generated row is retained if and only if:

```text
start_index mod 128 == 0
and end_index - start_index + 1 == 128
```

Because source windows have stride 32, selected indices are unique, and retained
128-frame intervals start on 128-frame boundaries, retained intervals cannot share
raw frames within a `(source_file, segment_id)` group. Different source/segment
groups are disjoint by construction.

The evaluator must independently verify:

- selected source-test indices are unique within each evaluation set;
- retained `(source_file, segment_id, start_index, end_index)` rows are unique;
- pairwise retained intervals do not overlap within any source/segment group;
- every evaluation label and scenario retains at least one row;
- retained counts and fractions are reported overall, by binary/exact label, by
  sensitivity scenario, and by source file.

The rule may not be changed after outcomes are inspected. No alternate offset may be
selected based on performance. Full-set matched-update v1 results remain the primary
camera-ready results; this audit is a reviewer-requested sensitivity.

## Frozen evaluation and metrics

Every checkpoint uses the unchanged real-train-only standardizer and unchanged
source policy: attack if and only if the five-class argmax is not class 0. No
threshold, checkpoint, arm, seed, subset offset, or scenario is selected using this
audit.

Primary outcomes:

1. fixed-variant binary attack recall on the non-overlap subset;
2. sensitivity binary attack recall on the non-overlap subset;
3. paired `rule_0p30 - real_only` differences for both outcomes;
4. paired `rule_0p30 - real_oversampling_0p30` differences for both outcomes.

Secondary outcomes:

- binary FPR, normal recall, binary Macro-F1, and exact attack recall;
- fixed-evaluation detection and exact recall by construction family;
- sensitivity detection and exact recall by family, coupled severity, and all
  twelve family-by-severity scenarios;
- the same metrics descriptively by source file;
- attenuation relative to the corresponding full-set matched-update v1 paired
  contrast.

For every arm and metric, report all five paired pipeline-seed values, arithmetic
mean, and sample standard deviation (`ddof=1`). Report no window-level significance
test or confidence interval. Source-file and non-overlap-window summaries are
descriptive and must not be relabeled capture-level.

## Interpretation branches fixed before prediction

- If both primary Rule contrasts remain positive in every paired seed, report that
  the controlled generated-evaluation advantage survives this deterministic
  raw-frame-disjoint sensitivity. Do not claim capture- or population-level
  replication.
- If a primary mean contrast remains positive but one or more paired seeds are
  non-positive, report a positive but seed-mixed overlap sensitivity.
- If either primary mean contrast is non-positive, report that the corresponding
  controlled gain is overlap-sensitive and narrow C1 accordingly.
- If fixed and sensitivity results disagree, report them separately rather than
  averaging them into one conclusion.
- Any unfavorable construction-family or coupled-severity counterexample remains
  visible.

## Frozen outputs and overwrite policy

The implementation must refuse to overwrite a completed v1 result set and write
only:

- `wisa_camera_ready/scripts/evaluate_overlap_unit_v1.py`;
- `wisa_camera_ready/results/tables/overlap_unit_origin_summary_v1.csv`;
- `wisa_camera_ready/results/tables/overlap_unit_by_seed_v1.csv`;
- `wisa_camera_ready/results/tables/overlap_unit_summary_v1.csv`;
- `wisa_camera_ready/results/tables/overlap_unit_by_source_v1.csv`;
- `wisa_camera_ready/results/tables/overlap_unit_sensitivity_by_scenario_v1.csv`;
- `wisa_camera_ready/results/logs/overlap_unit_v1.log`;
- `wisa_camera_ready/experiments/03_overlap_unit_audit/manifest_v1.json`;
- `wisa_camera_ready/experiments/03_overlap_unit_audit/RESULTS_v1.md`.

The manifest must contain plan and implementation commits/hashes, environment and
device information, all input/checkpoint hashes, reconstruction hashes, retained
counts, validation gates, runtime, warnings, and output hashes.

## Claims explicitly not licensed

This audit does not establish:

- capture-level, vehicle-level, or population uncertainty;
- independence of model-training observations;
- generator-construction replication;
- real-attack realism or semantic validity;
- cross-dataset robustness;
- a causal effect of window overlap.
