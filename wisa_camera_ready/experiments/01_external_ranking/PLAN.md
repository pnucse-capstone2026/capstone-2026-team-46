# External Ranking Diagnostic v1 — Frozen Analysis Plan

## Status and purpose

- Frozen before model inference: **2026-07-28T15:34:49Z**
- Accepted-paper source commit at freeze: `57880aa350ddf0f916b7e1bd2e940529d0fc6ea6`
- Submission: WISA 2026 submission 16
- Reviewer concern: distinguish source-policy operating-point collapse from
  threshold-independent ranking behavior.
- Design status: **post-review diagnostic requested after acceptance**, not a
  preregistered primary analysis.

This analysis uses external target labels only to measure ranking and attainable
target-dataset operating points after training is complete. External data remain
evaluation-only and do not enter training, validation, checkpoint selection, scaler
fitting, setting selection, or paper-wide threshold selection.

## Frozen model family

All existing five-class 1D-CNN checkpoints are evaluated. No checkpoint is retrained
or selected using an external result.

| Setting | Frozen checkpoints | Frozen scaler |
|---|---|---|
| `real_only` | `wisa/models/baseline/cnn1d_real_only_seed{7,42,123}.pt` | `wisa/models/baseline/cnn_standardizer.npz` |
| `real_oversampling_0p50` | `wisa/models/oversampling/cnn1d_real_oversampling_ratio0p50_seed{7,42,123}.pt` | matching per-seed `standardizer_real_oversampling_ratio0p50_seed*.npz` |
| `rule_0p10` | `wisa/models/ratio_sweep/cnn1d_rule_ratio0p10_seed{7,42,123}.pt` | matching per-seed `standardizer_rule_ratio0p10_seed*.npz` |
| `rule_0p30` | `wisa/models/ratio_sweep/cnn1d_rule_ratio0p30_seed{7,42,123}.pt` | matching per-seed `standardizer_rule_ratio0p30_seed*.npz` |
| `rule_0p50` | `wisa/models/ratio_sweep/cnn1d_rule_ratio0p50_seed{7,42,123}.pt` | matching per-seed `standardizer_rule_ratio0p50_seed*.npz` |
| `rule_1p00` | `wisa/models/ratio_sweep/cnn1d_rule_ratio1p00_seed{7,42,123}.pt` | matching per-seed `standardizer_rule_ratio1p00_seed*.npz` |

- Paired pipeline seeds: `7`, `42`, `123`
- Primary prespecified contrast: `rule_0p30 - real_only`, paired by pipeline seed
- Secondary settings: every other frozen setting above; none may be omitted based on
  its result.
- Binary attack score: `1 - p(normal)` from the unchanged five-class softmax.
- Source-policy decision: attack iff the five-class argmax is not class 0.

The 34 unique checkpoint/scaler files had canonical inventory SHA-256
`9bfae8a7c58a639cc2f42e414061c6fea9058b447e67c1566a085c41c54f0e17`.
The canonical inventory is the UTF-8 concatenation of
`relative_path<TAB>byte_size<TAB>file_sha256<LF>`, sorted by relative path. The
execution manifest must retain every individual entry and reproduce this aggregate.

## Frozen external datasets and label audit

Window size is 128 frames and stride is 32 frames. A streamed raw-data window is
binary attack-positive if it contains at least one attack-labelled frame.

| Dataset | Frozen input | Groups | Audited labels and limitations |
|---|---|---:|---|
| OTIDS | `datasets/windows/otids_cross_windows.npz` | 4 source files | 144,156 windows: 74,040 normal and 70,116 attack. `Attack_free_dataset.txt` is the only normal group; the three attack scenarios are attack-only and all 70,116 attack windows carry weak labels. Scenario-local ROC-AUC/AP and fixed-FPR estimates are therefore undefined. |
| can-train-and-test | `datasets/can-train-and-test.zip` | 176 test CSVs | All four sets and all four test subset axes are included; each CSV is preprocessed separately and windows never cross files. Set, subset, known/unknown vehicle, known/unknown attack, and source stem remain attached to every group. |
| ROAD | `datasets/processed/road_frames/*.parquet` | 41 captures | 12 ambient and 29 attack captures from the accepted-paper processed lineage are included. Each capture is windowed separately. Attack-window labels are frame-attributable through the processed `attack` column. Exclusions remain those recorded in the frozen ROAD profile. |

Frozen dataset lineage:

| Input | Bytes | SHA-256 |
|---|---:|---|
| `datasets/windows/otids_cross_windows.npz` | 43,289,129 | `b521f3546f633c6220e2696234abb6d7c37a03d64f19e71ec37bf36a100ed596` |
| `datasets/can-train-and-test.zip` | 1,507,455,719 | `a9c607b38bd28f1768021ad01c29ffbfe4e82bb0ae5815ac3ce7ad74751ae061` |
| ROAD 41-file canonical tree | 315,092,201 | `a2c0471df17466e7cbc0b9a5b997a6e49cf6a59c413259c39719313ed1510156` |

The ROAD tree digest uses the same sorted
`relative_path<TAB>byte_size<TAB>file_sha256<LF>` rule. Loader dependencies frozen
at planning time are:

- `wisa/scripts/train_real_only_baselines.py`:
  `5f4d423a07c573d69210034e752b2a847eb4b87486907cd8ea81aee06d9aef31`
- `wisa/scripts/run_cantt_tier1.py`:
  `6f7fd11fe17c727889c5d0ec56fce199e1656a3fabf22a8f061fd217de9a114e`
- `wisa/results/tables/road_dataset_profile.csv`:
  `0d6202ae75e7e8e883ef5fb40fd205c205803b25307932c2b2bd9cf0eef7df83`

## Frozen metrics

### Source-policy deployment diagnostic

For each dataset, setting, and seed, report normal windows first, then:

1. source-policy false-positive rate and normal recall;
2. source-policy binary attack recall;
3. the corresponding confusion counts.

This is kept separate from target-label-dependent ranking. Attack recall near one is
not treated as useful detection when FPR is near one.

### Target-label-dependent ranking diagnostic

For each dataset, setting, and seed, report:

1. ROC-AUC;
2. average precision, explicitly labelled as the implementation of PR-AUC;
3. attack prevalence;
4. TPR at empirical FPR ceilings `0.001` and `0.01`;
5. the selected threshold and achieved FPR for both ceilings.

For a target FPR `alpha`, let `K = floor(alpha * N_normal)`. Predictions use
`score >= threshold`. Normal scores are grouped by exact score ties in descending
order. Include every complete high-score tie group that keeps the false-positive
count at or below `K`, then place the threshold immediately above the next excluded
normal score with `nextafter(excluded_score, +infinity)`. If even the highest score
tie exceeds `K`, this reduces to `nextafter(max_normal_score, +infinity)` and zero
false positives. This is the most permissive threshold that excludes the indivisible
boundary tie. The next excluded score, normal and attack tie mass, unused FP
capacity, and whether including that tie would exceed the ceiling are reported. No
interpolation or fractional tie breaking is used.

All score arrays must be finite and aligned one-to-one with binary labels. Both
classes must be present for pooled ranking metrics; otherwise the metric is missing
with an explicit reason.

## Group sensitivity and uncertainty

- Window-pooled metrics are the primary descriptive deployment-volume view.
- Raw group rows retain dataset-native identifiers:
  - OTIDS: `source_file`/scenario, without calling it a capture;
  - can-train-and-test: test CSV within set/subset;
  - ROAD: capture.
- Group-local ROC-AUC, AP, and low-FPR values are reported only for groups containing
  both normal and attack windows.
- Group-macro means and 95% percentile cluster-bootstrap intervals use 2,000
  resamples and seed `20260728`, separately for each dataset/setting/pipeline seed.
- can-train bootstrap resampling is stratified by `set_id × subset_id`; ROAD
  resampling is stratified by `role × attack family × masquerade status`.
- A macro interval is emitted only when at least two eligible mixed-class groups
  exist. Missing-class groups are excluded from that metric and counted.
- OTIDS receives no group-macro ranking interval because its four scenario groups are
  class-separated and only one normal group exists. This is reported as a design
  limitation, not silently pooled into a false scenario-level uncertainty claim.
- Pipeline-seed mean and standard deviation are descriptive; they are not described
  as capture-, vehicle-, construction-, or population-level uncertainty.

## Frozen outputs

The evaluator must refuse to overwrite a completed v1 result set and publish:

- `scripts/evaluate_external_ranking_v1.py`
- `results/tables/external_ranking_by_seed_v1.csv`
- `results/tables/external_ranking_summary_v1.csv`
- `results/tables/external_low_fpr_by_group_v1.csv`
- `results/logs/external_ranking_v1.log`
- `experiments/01_external_ranking/manifest_v1.json`

The manifest must include the plan commit, evaluator commit/hash, device and package
versions, all individual input hashes, start/end UTC times, group/window/class counts,
bootstrap configuration, validation gates, output hashes, and any warnings.

## Interpretation branches fixed before outcomes

- If Rule improves ranking while source-policy FPR remains unusable, report
  operating-policy failure and residual target-label-dependent ranking benefit
  separately; remove any universal “gain vanishes” claim.
- If the primary Rule contrast does not improve ranking, state “no observed
  threshold-independent improvement in the prespecified contrast”; do not claim
  equivalence.
- If datasets or metrics disagree, report the mixed dataset-by-dataset pattern; do
  not average it into a universal transfer claim.
- If score ties prevent a requested FPR, report only attainable conservative points.
