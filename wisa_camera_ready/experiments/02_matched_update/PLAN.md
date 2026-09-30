# Matched Optimizer-Update Sensitivity v1

Status: frozen before any new matched-update model was trained or evaluated  
Frozen UTC date: 2026-07-28  
Accepted-paper source commit: `57880aa350ddf0f916b7e1bd2e940529d0fc6ea6`  
Camera-ready state immediately before this plan:
`53f9be29cc800b58c2c02464e5496e54aa1040f1`

## Purpose and evidential status

This is a post-review training-policy sensitivity requested after acceptance. The
accepted comparison used different maximum-epoch and early-stopping policies across
the real-only and Rule-augmented fits. This experiment asks whether the controlled
generated-variant gain attributed to Rule +30% persists when optimizer-update count,
validation opportunities, and checkpoint-selection rules are identical.

The experiment is not a new generator-construction replication, a capture-level
uncertainty analysis, or evidence for cross-dataset transfer. Its five repeated fits
are paired pipeline seeds: sampling, initialization, dropout, and minibatch order vary
within one fixed source split and one fixed Rule pool.

No matched-update outcome had been generated or inspected when this contract was
written. Existing accepted-paper results and the reviewer comments were known.

## Frozen data roles and inputs

Only Car-Hacking training data enter fitting, validation, checkpoint selection, class
weighting, or standardization. The Rule generator pool was fitted previously from the
Car-Hacking train split. Fixed-variant and sensitivity windows remain evaluation-only
and are not opened by the training phase. OTIDS, can-train-and-test, ROAD, target-ID
stress, and out-of-rule stress are outside this experiment.

| Role | Path | Windows | SHA-256 |
|---|---|---:|---|
| real train | `datasets/windows/train_windows.npz` | 262,149 | `44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4` |
| source validation | `datasets/windows/val_windows.npz` | 84,890 | `b4920e7f60fbf38b38e12976129f120ebff160a21a4c77093f3746f3c1aa218e` |
| source test, evaluation-only | `datasets/windows/test_windows.npz` | 201,606 | `4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7` |
| fixed generated variant, evaluation-only | `datasets/windows/variant_test_windows.npz` | 64,000 | `91bd4d5dca2bd73907fad11bb94ea6dec075f60bb3b24e3c517617bf2d50b2ba` |
| full sensitivity construction, evaluation-only | `datasets/windows/variant_sensitivity_windows.npz` | 30,000 | `4266fb3bbb3e248ef1ab53aaca19cb4534a3815a18c9134dc3db446cdfcf5674` |
| fixed legacy Rule pool | `datasets/synthetic/rule_based_windows.npz` | 260,000 | `4078ab00b8531f760b296b5fd293e3df914bef996d317c958ead4762137bafd5` |

All six files use 128-frame windows, stride 32, and the same 11 features. Train,
validation, and test class counts are:

| Split | Normal | DoS | Fuzzy | Gear | RPM |
|---|---:|---:|---:|---:|---:|
| train | 138,976 | 20,651 | 29,321 | 35,531 | 37,670 |
| validation | 46,224 | 7,007 | 5,617 | 12,134 | 13,908 |
| test | 161,666 | 7,196 | 6,297 | 12,427 | 14,020 |

The Rule pool contains 65,000 windows per construction family and records generator
seed `314159`. The fixed variant contains 32,000 normal and 8,000 windows per attack
family and records seed `42`. The sensitivity set contains 12,000 normal windows and
1,500 attack windows for each of twelve family-by-coupled-severity scenarios and
records seed `20260611`.

Frozen construction dependencies:

| Path | SHA-256 |
|---|---|
| `wisa/scripts/train_real_only_baselines.py` | `5f4d423a07c573d69210034e752b2a847eb4b87486907cd8ea81aee06d9aef31` |
| `wisa/scripts/train_rule_synthetic_ratio_sweep.py` | `846069660c8daf303587c11f05619c57c45d377153dbf8621aa076ed661e19dd` |
| `wisa/scripts/train_oversampling_baseline.py` | `3f8f25448ab5d0b0646652a9bc86a7e0bb989a1a40d1dfea540d10d75c468d38` |
| `wisa/scripts/run_seed_extension_stress.py` | `0c1fc8b40c4cb10d1fbabd1e2d0a47135e1a1479aa869795bdb84e35d117f3c2` |
| `wisa/experiments/03_synthetic/rule_based_config.yaml` | `edfa574a1b9a325697563e627e3906976648f1806eee5047cc6a4dd8f8e49b15` |
| `wisa/experiments/01_split/variant_generation_config.yaml` | `f4912d94083d5e7e5be7ad68c3175e6281ea1896594afec6497da689556b8add` |
| `wisa/experiments/01_split/variant_sensitivity_config.yaml` | `f8d020c96afa8b3715823cf080a92b5276dda06d7eac943ddafb1001c42e1f57` |

The standardizer is fitted once on real train only. Recomputing it with the frozen
implementation is exactly array-equal to
`wisa/models/baseline/cnn_standardizer.npz` (file SHA-256
`7ffaff380c0e4af079ca68f94bd9c1ec35e597be99266e3631d7284d8a05e389`).
The float32 mean- and standard-deviation-array byte hashes are respectively
`1e53f10b4dfb1de8522513ca20c7d0c2b5a19a8de8f76067e4273a93467da8a2`
and `092f42e9a1019fe709233a91c6e876e976f8b3b8616807b42eb6e6da7b7dbb7c`.

## Frozen arms

The augmentation count is Python `round(262149 * 0.30) = 78,645`.

1. `real_only`: the 262,149 real training windows without a construction-stage
   permutation.
2. `rule_0p30`: append 78,645 Rule-pool windows, balanced over the four attack
   classes using the accepted-paper sampler, then apply
   `numpy.random.default_rng(pipeline_seed).permutation` to all 340,794 windows.
3. `real_oversampling_0p30`: append 78,645 sampled real attack windows, balanced over
   the four attack classes using the accepted-paper oversampling policy, then apply
   the same form of construction-stage permutation to all 340,794 windows.

Both added sets contain 19,662 DoS and 19,661 windows for each of Fuzzy, Gear, and
RPM. Sampling is without replacement within each added set because every eligible
class pool is large enough. “Real oversampling” still duplicates selected source
windows in the final training multiset because each selected window also remains in
the original real set.

The accepted seed derivations are retained:

- Rule sampling seed: `pipeline_seed + 300`;
- real-attack sampling seed: `pipeline_seed + 401`;
- augmented construction-stage permutation seed: `pipeline_seed`.

The following selected-index hashes use contiguous native-endian `int64` bytes.
Permutation hashes are also frozen so construction ordering can be audited.

| Arm | Pipeline seed | Sampling seed | Selected-index SHA-256 | Full permutation SHA-256 |
|---|---:|---:|---|---|
| Rule +30% | 7 | 307 | `4ff00984e1dd97a48447b2abac6c347eb8967b20e3c90782db081f9681c11581` | `be7de8a2ee2090736ec3d10ae4bf5077b394edc9344ec46d0eb23b98d0483890` |
| real oversampling +30% | 7 | 408 | `38874baf2d0d22afa2101e5850b28b6f7b7b7d010cdb30abe27b8a6d56403f26` | `be7de8a2ee2090736ec3d10ae4bf5077b394edc9344ec46d0eb23b98d0483890` |
| Rule +30% | 42 | 342 | `bdfcbdf15dcd5001122de17dd721508c5347353a022becd20df94c0e59a0f917` | `fac21c3b58e98a5a3d69d115a72f530b2ce3893e6521bfadd83f9c17fbe18df6` |
| real oversampling +30% | 42 | 443 | `ae9551ed910a8d31ec06daa82b3e53b6dc20924c787735fb6537b30bfeb15aca` | `fac21c3b58e98a5a3d69d115a72f530b2ce3893e6521bfadd83f9c17fbe18df6` |
| Rule +30% | 123 | 423 | `22bd0a549e33ad17662a6b5ac84065a8b229768aace1c5721ece4a2408255ba5` | `53db276d4ce609b438b3e9d26010d8c0c1da3ae6ccf9ef760738badec8ff9dc1` |
| real oversampling +30% | 123 | 524 | `c4a849ac1d069d22f534c9dc33d4bafe7330b37e1f916c8e94b2ec461e4090a7` | `53db276d4ce609b438b3e9d26010d8c0c1da3ae6ccf9ef760738badec8ff9dc1` |
| Rule +30% | 2026 | 2326 | `59a6f31755fb18e7f19c4ba9e819b050f4d5db617854e0f2c2e78f4ae31a3319` | `88f4a265bc1983516751fdb346d790fa6e0cda0bf89bebaa22d805ad6b249ab3` |
| real oversampling +30% | 2026 | 2427 | `0ef614d94468a8a91c9ebe04e2764dcc5940562d0308f7a7bb8844a36b6a85d3` | `88f4a265bc1983516751fdb346d790fa6e0cda0bf89bebaa22d805ad6b249ab3` |
| Rule +30% | 3407 | 3707 | `3ed08fb8b14d3796264fba3cdac76ca7563fd0d951ad295209d0798fd875e3bf` | `6b173f6013ab3bfd8ae74bc9d9a08431912f67200973d9155c3bddf9e2297667` |
| real oversampling +30% | 3407 | 3808 | `ed5f6263a7f011666b40413a8d167fc2f595a9fab8e22b6c4e9e45a3ed52c7b2` | `6b173f6013ab3bfd8ae74bc9d9a08431912f67200973d9155c3bddf9e2297667` |

## Frozen paired pipeline seeds

Every arm is run for `7, 42, 123, 2026, 3407`. For a given pipeline seed:

- Python, NumPy, PyTorch CPU, and PyTorch CUDA model/dropout seeds equal the
  pipeline seed;
- the deterministic cyclic minibatch sampler seed is
  `pipeline_seed + 1,000,003`;
- the data-loader worker seed is `pipeline_seed + 2,000,003`;
- cuDNN benchmarking is disabled, cuDNN deterministic mode is enabled, and PyTorch
  deterministic algorithms are required.

These pairings are a variance-reduction and descriptive comparison device, not five
independent captures, vehicles, Rule constructions, or population draws.

## Frozen architecture, optimizer, and preprocessing

- unchanged five-class 1D-CNN: Conv1d(11,64,k=5), batch normalization, ReLU,
  max-pooling; Conv1d(64,128,k=5), batch normalization, ReLU, max-pooling;
  Conv1d(128,128,k=3), batch normalization, ReLU, adaptive average pooling;
  flatten, dropout 0.2, and a 128-to-5 linear head;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `512`, `drop_last=False`;
- weighted cross entropy with the accepted arm-specific
  `compute_class_weight(class_weight="balanced")` policy;
- real-train-only float32 standardization for CAN ID, DLC, eight payload bytes, and
  `delta_t`; no target-fitted preprocessing.

The seed-invariant class weights, ordered Normal/DoS/Fuzzy/Gear/RPM, are:

| Arm | Class weights |
|---|---|
| real-only | `0.3772579438, 2.5388504189, 1.7881313734, 1.4756072162, 1.3918184231` |
| Rule +30% | `0.4904357587, 1.6907399598, 1.3915070842, 1.2349398464, 1.1888646631` |
| real oversampling +30% | `0.4904357587, 1.6907399598, 1.3915070842, 1.2349398464, 1.1888646631` |

Arm-specific weights are retained rather than forcing common weights because the
accepted training policy recomputed balanced weights from each arm's training
multiset. Consequently, this sensitivity matches optimizer-update opportunity while
preserving the full accepted arm-specific data-and-weighting intervention.

## Frozen update and checkpoint-selection policy

- Maximum optimizer updates: exactly `7,992` for every fit.
- No early stopping; training continues through the full budget even after a
  checkpoint ceases to improve.
- Source validation is evaluated after updates
  `666, 1332, 1998, 2664, 3330, 3996, 4662, 5328, 5994, 6660, 7326, 7992`.
- The selected checkpoint is the validation point with the highest five-class source
  validation Macro-F1. Exact ties select the earlier update by using a strict
  improvement rule.
- The cyclic sampler draws a fresh deterministic `torch.randperm` at each training
  multiset exhaustion and retains the final short batch (`drop_last=False`).
- Rule +30% and real-oversampling +30% have 666 update batches per cycle and
  therefore complete exactly 12 cycles.
- Real-only has 513 update batches per cycle and therefore completes 15 full cycles
  plus 297 update batches of cycle 16.

The budget is the Rule +30% accepted configuration's maximum
`12 × ceil(340794 / 512) = 7,992`, and does not exceed the real-only accepted
configuration's maximum of `20 × ceil(262149 / 512) = 10,260`.

The implementation must separate `train` and `evaluate` phases. Training may load
only real train, source validation, and the Rule training pool. Evaluation must
refuse to start unless all 15 fits have a complete 7,992-update history, a selected
checkpoint, and matching provenance. No source-test, fixed-variant, sensitivity,
target-ID, out-of-rule, or external outcome may influence checkpoint selection.

## Frozen evaluation policy

The selected checkpoints use the unchanged source policy: a window is an attack iff
the five-class argmax is not class 0. No evaluation threshold is fitted.

Required evaluation sets:

1. Car-Hacking source test;
2. fixed generated variant;
3. full generated sensitivity construction.

Primary outcomes:

- source-test binary Macro-F1 and FPR;
- fixed-variant binary attack recall;
- sensitivity binary attack recall.

Secondary outcomes:

- binary normal recall, attack recall, confusion counts, and binary Macro-F1 where
  both classes are present;
- exact DoS/Fuzzy/Gear/RPM construction-family recall;
- sensitivity detection and exact-family recall for all twelve family-by-severity
  scenarios;
- detection and exact-family recall pooled separately over the coupled
  low/medium/high severity settings.

“Exact” means five-class argmax equals the recorded construction-family label.
Sensitivity levels are coupled parameter settings; they are not interpreted as an
isolated one-variable intensity intervention.

For every arm and metric, report all five seeds plus arithmetic mean and sample
standard deviation (`ddof=1`). Report paired differences for
`rule_0p30 - real_only` and
`rule_0p30 - real_oversampling_0p30` seed by seed, then their mean and sample
standard deviation. Any test over the five pipeline seeds is supporting evidence
only; no capture- or population-level interval is licensed.

## Frozen outputs and overwrite policy

The implementation must write only below `wisa_camera_ready/`:

- `scripts/run_matched_update_v1.py`;
- `models/matched_update_v1/cnn1d_<arm>_seed<seed>_v1.pt`;
- `models/matched_update_v1/standardizer_source_real_train_v1.npz`;
- `experiments/02_matched_update/histories/<arm>_seed<seed>_v1.csv`;
- `experiments/02_matched_update/runs/<arm>_seed<seed>_v1.json`;
- `experiments/02_matched_update/training_manifest_v1.json`;
- `results/tables/matched_update_checkpoint_selection_v1.csv`;
- `results/tables/matched_update_by_seed_v1.csv`;
- `results/tables/matched_update_summary_v1.csv`;
- `results/tables/matched_update_sensitivity_by_scenario_v1.csv`;
- `results/logs/matched_update_v1.log`;
- `experiments/02_matched_update/manifest_v1.json`;
- `experiments/02_matched_update/RESULTS_v1.md`.

The training phase may resume by skipping a fit only after verifying its input,
contract, history, checkpoint, and run-metadata hashes. It must not overwrite a
completed fit. The evaluation phase must refuse to overwrite a completed v1 result
set. Partial fits remain clearly labeled and no partial-seed aggregate may be used
in the paper.

The final manifest must record plan and implementation commits/hashes, package and
device versions, UTC start/end times, runtimes, all input hashes, sample-index and
permutation hashes, class counts and weights, updates run, selected update and
validation value, checkpoint/history hashes, evaluation counts, validation gates,
warnings, and output hashes.

## QA gates

Before training:

1. verify all frozen input and dependency hashes;
2. verify feature order, window size, stride, class counts, and construction seeds;
3. verify the recomputed standardizer is exactly array-equal to the frozen one;
4. verify every seed's sampler and construction-permutation hashes above;
5. unit-test cyclic batch counts, update grid, tie selection, binary metrics, exact
   family recall, and sensitivity grouping;
6. verify the script cannot load evaluation paths in `train` mode.

Before publishing aggregates:

1. require all 15 histories to end at update 7,992 and contain exactly 12 validation
   rows;
2. require every selected update to belong to the frozen grid;
3. require finite model outputs and one-to-one label alignment;
4. require expected evaluation class and scenario counts;
5. recompute each selected checkpoint's source-validation Macro-F1 and require
   agreement with its run metadata;
6. verify every by-seed, summary, paired-difference, and scenario row against the
   individual predictions or confusion counts;
7. hash every output, update the camera-ready artifact manifest, run the workspace
   audit, `git diff --check`, and confirm `git status --short -- wisa` is empty.

## Interpretation branches fixed before outcomes

- If the Rule gain survives relative to both real-only and matched real
  oversampling, state that it is not explained by optimizer-update count under this
  matched budget. Keep the conclusion construction-local.
- If the Rule gain attenuates or becomes seed-dependent, state that its magnitude is
  training-schedule-sensitive and use the matched-update estimate in the
  camera-ready paper.
- If the Rule gain disappears or reverses, withdraw unconditional wording that the
  controlled gain is real and report that the accepted estimate depends on its
  original training policy.
- If Rule beats real-only but not matched real oversampling, do not attribute the
  difference specifically to synthetic content; report that matched added data
  volume/class rebalancing accounts for the observed controlled comparison.
- If source-test behavior degrades materially while generated-variant recall rises,
  report the trade-off rather than selecting only the favorable metric.
- If all 15 fits and the complete QA cannot finish, publish no partial matched-update
  result and leave optimizer-update matching explicitly unresolved.
