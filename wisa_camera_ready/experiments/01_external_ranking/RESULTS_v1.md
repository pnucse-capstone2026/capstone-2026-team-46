# External Ranking Diagnostic v1 — Results and QA

## Outcome

The prespecified primary contrast, `rule_0p30 - real_only`, showed **no observed
threshold-independent ranking improvement** on OTIDS, ROAD, or
can-train-and-test. The source-policy operating point remained unusable on every
dataset. At target-dataset empirical FPR ceilings of 0.1% and 1%, the attainable
TPR was zero for all 54 dataset × setting × pipeline-seed combinations because the
highest normal-score tie alone exceeded each false-positive budget.

This is a post-review diagnostic. External labels were not used for training,
validation, checkpoint/scaler selection, setting selection, or source-policy
calibration.

## Pooled primary results

Values are mean ± sample standard deviation over paired pipeline seeds 7, 42, and
123. Average precision is the implementation used for PR-AUC. FPR is deliberately
reported before attack recall.

| Dataset | Setting | Source-policy FPR | Source-policy attack recall | ROC-AUC | AP / PR-AUC | TPR @ FPR 0.1% | TPR @ FPR 1% |
|---|---|---:|---:|---:|---:|---:|---:|
| OTIDS | Real only | 0.998087 ± 0.002133 | 0.990815 ± 0.008429 | 0.549482 ± 0.069362 | 0.532737 ± 0.065244 | 0 | 0 |
| OTIDS | Rule +30% | 0.999973 ± 0.000036 | 0.999933 ± 0.000092 | 0.501117 ± 0.003514 | 0.486965 ± 0.001786 | 0 | 0 |
| ROAD | Real only | 0.999616 ± 0.000605 | 1.000000 ± 0.000000 | 0.516688 ± 0.028174 | 0.057679 ± 0.003274 | 0 | 0 |
| ROAD | Rule +30% | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 | 0.055744 ± 0.000000 | 0 | 0 |
| can-train-and-test | Real only | 0.999945 ± 0.000047 | 1.000000 ± 0.000000 | 0.500018 ± 0.000312 | 0.069918 ± 0.000041 | 0 | 0 |
| can-train-and-test | Rule +30% | 0.999997 ± 0.000004 | 0.999850 ± 0.000259 | 0.499863 ± 0.000350 | 0.069898 ± 0.000046 | 0 | 0 |

The paired mean differences for `Rule +30% - Real only` were:

| Dataset | Δ source FPR | Δ attack recall | Δ ROC-AUC | Δ AP / PR-AUC |
|---|---:|---:|---:|---:|
| OTIDS | +0.001886 | +0.009118 | -0.048365 | -0.045772 |
| ROAD | +0.000384 | 0.000000 | -0.016688 | -0.001935 |
| can-train-and-test | +0.000051 | -0.000150 | -0.000155 | -0.000020 |

The OTIDS ROC/AP difference is seed-variable and is not treated as a
capture-level or population-level estimate. ROAD's Rule +30% score was exactly
1.0 for every window in all three seeds, so its ROC-AUC is 0.5 and its AP equals
the attack prevalence. can-train-and-test was likewise effectively chance-ranked.

## Low-FPR tie finding

For every pooled row, the largest external normal score was 1.0 and the number of
normal windows tied at 1.0 exceeded both allowed false-positive counts. The
prespecified conservative rule therefore placed the threshold at
`nextafter(1.0, +infinity)`, yielding FPR 0 and TPR 0. No interpolation or
fractional tie breaking was used.

Examples for the primary arms:

- OTIDS Real-only seed 7: 22,166 normal and 43,428 attack windows tied at 1.0;
  the 0.1% FPR budget allowed only 74 normal false positives.
- ROAD Rule +30%, every seed: all 808,854 normal and 47,751 attack windows tied
  at 1.0; the 0.1% budget allowed 808 false positives.
- can-train-and-test Rule +30% seed 42: 4,176,444 normal and all 313,985 attack
  windows tied at 1.0; the 0.1% budget allowed 4,176 false positives.

Thus the requested low-FPR point is not merely poor after interpolation: the
finite-precision model scores do not expose any nonzero-TPR operating point below
either requested ceiling.

## Group sensitivity

- OTIDS has one normal-only source file and three attack-only weak-label source
  files. Scenario-local ROC/AP and a scenario bootstrap are not identifiable, so
  none was reported.
- ROAD has 29 mixed attack captures and 12 normal-only ambient captures. Its
  mixed-capture group-macro analysis agreed with the pooled finding: Rule +30%
  remained at ROC-AUC 0.5 for every seed; low-FPR TPR remained zero.
- can-train-and-test has 175 mixed test files and one normal-only test file.
  Group-macro ROC-AUC remained near 0.5, the paired Rule contrast changed sign
  across seeds, and low-FPR TPR remained zero.
- ROAD bootstrap resampling was stratified by role × family × masquerade status.
  can-train resampling was stratified by set × subset. Both used 2,000 replicates,
  base seed 20260728, zero rejected draws, and explicit missing-class exclusion.

## Secondary setting worth retaining

All six settings were reported. Real-attack oversampling +50% produced the only
clear secondary pooled ranking signal: on OTIDS, ROC-AUC was
0.676400 ± 0.028051 and AP was 0.671556 ± 0.051047. Its source-policy FPR was
still unusable at 0.859117 ± 0.146635, and its TPR at both requested low-FPR
ceilings was still zero. This is not Rule augmentation and is not the
prespecified primary contrast.

## Verification

- Analysis-plan commit: `18b2176140a0458509eaf10c9e0bb1e129b15017`
- Evaluator commit: `db383345e15bf04b10536215d23deefb167a07f7`
- Runtime: NVIDIA GB10, 737.175 seconds
- Evaluated windows:
  - OTIDS: 144,156 = 74,040 normal + 70,116 attack
  - ROAD: 856,605 = 808,854 normal + 47,751 attack
  - can-train-and-test: 4,490,896 = 4,176,911 normal + 313,985 attack
- Complete output coverage:
  - 54 pooled dataset × setting × seed rows
  - 3,978 raw group × setting × seed rows
  - 54 group-macro rows
  - 21 setting-summary and primary-contrast rows
- New optimized inference matched the frozen WISA evaluator exactly on a
  17-window equivalence check.
- All 216 source-policy confusion counts matched the corresponding frozen WISA
  tables exactly across 3 datasets × 6 settings × 3 seeds × 4 cells.
- Input hashes, output hashes, finite scores, binary labels, alignment,
  ROC/PR monotonicity, fixed-FPR ceilings, and low-FPR monotonicity passed.
- `wisa/` remained unmodified.

## Camera-ready claim licensed by this diagnostic

> Under the prespecified source-trained operating policy, all three external
> datasets remain unusable because normal traffic is almost universally alarmed.
> In the prespecified Rule +30% contrast, no external dataset shows an observed
> threshold-independent ROC-AUC or average-precision improvement, and score ties
> leave zero attack recall at empirical FPR ceilings of 0.1% and 1%.

This wording is dataset- and contrast-bounded. It does not establish equivalence,
identify a universal causal mechanism, or convert the external diagnostic into
model selection.
