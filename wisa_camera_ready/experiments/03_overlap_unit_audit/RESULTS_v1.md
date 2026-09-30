# Overlap-Aware Evaluation-Unit Audit v1 — Results

## Evidential status

This is a post-review evaluation-only sensitivity over the existing 15 matched-update checkpoints. No model was retrained, selected, or calibrated with these outcomes. The retained base windows are raw-frame-disjoint within each generated evaluation, but they are not independent captures or vehicles.

## Retained evaluation support

| Evaluation | Full windows | Non-overlap windows | Retained fraction |
|---|---:|---:|---:|
| fixed_variant | 64000 | 15976 | 0.2496 |
| sensitivity | 30000 | 7465 | 0.2488 |

## Primary outcomes

| Evaluation | Contrast | Full mean | Non-overlap mean ± sample std | Non-overlap minus full | Positive in all five seeds |
|---|---|---:|---:|---:|:---:|
| fixed_variant | `rule_0p30_minus_real_only` | 0.5025 | 0.5002 ± 0.0165 | -0.0023 | yes |
| fixed_variant | `rule_0p30_minus_real_oversampling_0p30` | 0.5125 | 0.5115 ± 0.0169 | -0.0010 | yes |
| sensitivity | `rule_0p30_minus_real_only` | 0.1690 | 0.1715 ± 0.0399 | 0.0024 | yes |
| sensitivity | `rule_0p30_minus_real_oversampling_0p30` | 0.1919 | 0.1951 ± 0.0257 | 0.0033 | yes |

All four primary paired contrasts remain positive in every pipeline seed. The controlled fixed/sensitivity advantage survives this deterministic raw-frame-disjoint evaluation sensitivity.

Arm-level non-overlap binary attack recall:

| Evaluation | Real only | Rule +30% | Real oversampling +30% |
|---|---:|---:|---:|
| fixed_variant | 0.4998 | 0.9999 | 0.4884 |
| sensitivity | 0.5538 | 0.7252 | 0.5301 |

## Secondary boundary

Negative Rule-minus-real exact-family/scenario mean contrasts remain visible: Gear (-0.0283), Gear high (-0.1899).

The five repeats remain paired pipeline seeds conditional on one split and one Rule construction. This audit does not estimate capture-, vehicle-, or population-level uncertainty and does not establish a causal effect of overlap.
