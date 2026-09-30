# Matched Optimizer-Update Sensitivity v1 — Results and QA

## Outcome

The controlled Rule +30% gain **survived optimizer-update matching**. All 15 fits
ran for exactly 7,992 optimizer updates, used the same 12-point source-validation
grid, and selected checkpoints by the same five-class validation Macro-F1 rule.
Relative to both real-only and equal-size real-attack oversampling +30%, Rule +30%
improved fixed-variant and full-sensitivity attack recall in every one of the five
paired pipeline seeds.

This licenses a narrow conclusion: the controlled generator-adjacent gain is not
explained by optimizer-update count under this matched budget. It does not license
a realism, unseen-generator, external-transfer, capture-population, or
vehicle-population claim.

## Primary results

Values are mean ± sample standard deviation over paired pipeline seeds
`7, 42, 123, 2026, 3407`.

| Arm | Source-test binary Macro-F1 | Source-test FPR | Fixed-variant binary attack recall | Sensitivity binary attack recall |
|---|---:|---:|---:|---:|
| Real only | 0.999172 ± 0.000207 | 0.000169 ± 0.000010 | 0.497450 ± 0.015704 | 0.551000 ± 0.031373 |
| Rule +30% | 0.999449 ± 0.000071 | 0.000126 ± 0.000071 | 0.999931 ± 0.000064 | 0.720033 ± 0.013211 |
| Real-attack oversampling +30% | 0.999414 ± 0.000040 | 0.000145 ± 0.000030 | 0.487406 ± 0.016524 | 0.528178 ± 0.033527 |

The prespecified paired differences were:

| Contrast | Δ source-test binary Macro-F1 | Δ source-test FPR | Δ fixed-variant attack recall | Δ sensitivity attack recall |
|---|---:|---:|---:|---:|
| Rule − real only | +0.000277 ± 0.000184 | −0.000043 ± 0.000076 | +0.502481 ± 0.015693 | +0.169033 ± 0.040607 |
| Rule − real oversampling | +0.000034 ± 0.000079 | −0.000019 ± 0.000049 | +0.512525 ± 0.016551 | +0.191856 ± 0.026665 |

Both generated-evaluation recall contrasts were positive in all five paired seeds.
Rule versus real-only source-test Macro-F1 was also positive in all five seeds.
Rule and matched oversampling were effectively tied on saturated source-test
performance: the Macro-F1 difference was negative for seed 7 and positive for the
other four seeds, while the small FPR differences changed sign across seeds. The
claim therefore rests on the generated evaluation sets, not on declaring a
meaningful source-test win.

For context, the frozen five-seed accepted-policy sensitivity means were 0.725800
for Rule +30% and 0.548133 for real-only, a gap of 0.177667. The matched values
were 0.720033 and 0.551000, a gap of 0.169033. The gap attenuated by 0.008633
absolute but retained the same direction in every paired seed. This is a small
schedule sensitivity, not disappearance of the controlled gain.

## Binary detection versus exact construction-family retention

Exact-family results followed the same overall direction:

| Arm | Fixed-variant exact recall | Sensitivity exact recall |
|---|---:|---:|
| Real only | 0.491819 ± 0.009737 | 0.445144 ± 0.031493 |
| Rule +30% | 0.999888 ± 0.000017 | 0.644022 ± 0.075074 |
| Real-attack oversampling +30% | 0.486900 ± 0.016451 | 0.420922 ± 0.040595 |

The paired Rule gains in sensitivity exact recall were
`+0.198878 ± 0.066537` versus real-only and
`+0.223100 ± 0.065784` versus matched oversampling, positive in every paired
seed. These aggregate exact-recall gains must not be generalized to every family:
Gear exact recall was 0.534622 ± 0.155984 for Rule versus
0.557111 ± 0.161456 for real-only, whereas Fuzzy and RPM supplied much clearer
Rule gains. Binary detection and exact construction-family retention therefore
remain separate claims.

## Coupled-severity sensitivity

The low, medium, and high labels vary multiple generator parameters together.
They are reported as coupled severity settings, not as a one-variable causal
intensity intervention.

| Coupled severity | Real-only detection recall | Rule +30% detection recall | Oversampling detection recall | Rule − real | Rule − oversampling |
|---|---:|---:|---:|---:|---:|
| Low | 0.092500 ± 0.065482 | 0.208233 ± 0.043630 | 0.064033 ± 0.024536 | +0.115733 ± 0.100897 | +0.144200 ± 0.055571 |
| Medium | 0.560500 ± 0.041811 | 0.951867 ± 0.009513 | 0.521700 ± 0.096087 | +0.391367 ± 0.038298 | +0.430167 ± 0.098974 |
| High | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.998800 ± 0.002414 | 0.000000 ± 0.000000 | +0.001200 ± 0.002414 |

The aggregate low- and medium-severity Rule contrasts were positive in every
paired seed. This does not mean every scenario improved: the DoS-low
Rule-minus-real difference changed sign across seeds
(`+0.416000, −0.212667, −0.222000, −0.049333, +0.526000`), and RPM-low remained
near zero for all arms. High-severity binary detection was saturated, so it
cannot support a gain claim. High-severity exact recall was higher on average for
Rule but not paired-consistent against either control.

The defensible sensitivity result is thus an overall and coupled-medium gain,
with a smaller aggregate low-severity gain and scenario-specific failures still
present.

## Training and checkpoint audit

- Plan commit: `819686733ea87bf5a6cbdfb5f1c8402ddf46cf35`
- Implementation commit: `cab014f30b89a22fbaaaabb42b11ff0e558178f9`
- Device: NVIDIA GB10; PyTorch `2.12.0+cu130`
- Training time: 1,084.217 seconds
- Evaluation time: 43.074 seconds
- Fits: 3 arms × 5 seeds = 15
- Updates: 7,992 per fit; 119,880 total
- Source-validation checks: 12 per fit; 180 total
- Selected update range:
  - real-only: 4,662–7,992
  - Rule +30%: 4,662–7,992
  - real oversampling +30%: 5,994–7,992
- All selected checkpoints reproduced their recorded source-validation
  five-class Macro-F1 exactly during the post-training evaluation phase.
- All input, construction-sample, construction-permutation, checkpoint, history,
  table, and log hashes passed.
- No source-test, fixed-variant, sensitivity, or external outcome entered
  training or checkpoint selection.
- No external dataset was loaded by this experiment.
- `wisa/` remained unmodified.

One training-dynamics anomaly is retained rather than hidden: real oversampling
seed 3407 fell from a selected validation Macro-F1 of 0.999229 at update 7,326 to
0.741266 at update 7,992. The fit still completed the full budget, and the
prespecified best-on-grid rule selected update 7,326. The selected checkpoint
then reproduced 0.999229 exactly. This result is therefore about source-validation
selected checkpoints on the common grid, not about the final-update checkpoint.

## Camera-ready claim licensed by this sensitivity

> With all three arms trained for 7,992 optimizer updates and selected on the
> same source-validation grid, Rule +30% retained its fixed-variant and
> sensitivity gains over both real-only and equal-size real-attack oversampling
> in all five paired pipeline seeds. The matched sensitivity-recall gain was
> 0.169 over real-only and 0.192 over oversampling. Thus optimizer-update count
> does not explain the controlled generator-adjacent gain under this budget,
> although low-severity and exact-family failures remain.

The five repetitions quantify paired pipeline variation conditional on one
split and one fixed Rule construction. They are not capture-level, vehicle-level,
generator-construction, or population replication.
