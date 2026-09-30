# Generator Rule Specification

This artifact records the rule-based generator and generated evaluation stress-test parameters used by the Shift Ladder manuscript. The rules are controlled experimental perturbations, not claims of realistic or independent real-world attack traces.

## Shared Representation

- Window size: 128 CAN frames.
- Stride: 32 frames for canonical windowed datasets.
- Frame features: `can_id`, `dlc`, `data0`-`data7`, `delta_t`.
- Labels: class 0 is normal; attack classes are DoS=1, Fuzzy=2, Gear=3, RPM=4.
- Training synthetic source: Car-Hacking train normal windows only.
- Evaluation variant source: held-out Car-Hacking test normal windows only.
- OTIDS usage: external binary cross-dataset evaluation only; never used for generator fitting, training, validation, model selection, or deployment-performance claims.

## Training Synthetic Pool

Source script: `scripts/generate_rule_based_synthetic.py`

- Seed: 314159.
- Output: `datasets/synthetic/rule_based_windows.npz`.
- Total: 260,000 synthetic attack windows.
- Per family: 65,000 windows for DoS, Fuzzy, Gear, and RPM.
- Augmentation ratios: +10%, +30%, +50%, +100% relative to the number of real Car-Hacking training windows. For a ratio `r`, the training script samples `round(len(train_windows) * r)` synthetic windows, balanced across the four attack classes when possible.

| Attack | Position rule | CAN ID / DLC rule | Payload rule | Timing rule |
|---|---|---|---|---|
| DoS | Random burst length 32-90, random start, step in {1,2} | ID `0x000`, DLC 8 | each payload byte is 0 with probability 0.92 or 255 with probability 0.08 | `delta_t ~ U(0.00003, 0.002)` |
| Fuzzy | Random burst length 24-88, random start, step in {1,2,3,4} | ID sampled uniformly from 0..2047; DLC in {2,5,8} with probabilities {0.08,0.12,0.80} | bytes sampled uniformly from 0..255; bytes beyond DLC set to 0 | `delta_t ~ U(0.00005, 0.02)` |
| Gear | Random burst length 32-90, random start, step in {1,2,3} | ID `0x43F`, DLC 8 | byte0 = clip(level*40 + U{0..15}), level in {0..5}; byte1 = clip(255 - byte0 + U{-8..8}) | normal-like, clamped to at least `1e-5` |
| RPM | Random burst length 32-90, random start, step in {1,2,3} | ID `0x316`, DLC 8 | rpm sampled from 0..7999; byte0=(rpm//32)%256; byte1=(rpm//4)%256; byte2 += U{-50..50} with clipping | normal-like, clamped to at least `1e-5` |

Injection density is the number of selected attack positions divided by 128. The generated statistics file records injection-count mean/min/max per attack family: `results/tables/rule_based_synthetic_statistics.csv`.

## Fixed Variant Control

Source script: `scripts/generate_variant_test.py`

- Seed: 42.
- Output: `datasets/windows/variant_test_windows.npz`.
- Source: held-out Car-Hacking test normal windows only.
- Normal windows: 32,000.
- Per attack family: 8,000.

| Attack | Position rule | CAN ID / DLC rule | Payload rule | Timing rule |
|---|---|---|---|---|
| DoS | Random burst length 48-96, step in {1,2} | ID `0x000`, DLC 8 | all payload bytes set to 0 | original `delta_t * 0.25`, clamped to `1e-5` |
| Fuzzy | Random burst length 40-96, step in {1,2,3} | ID uniformly 0..2047, DLC 8 | all 8 bytes sampled uniformly from 0..255 | `delta_t ~ U(0.00005, 0.015)` |
| Gear | Random burst length 48-96, step 2 | ID `0x43F`, DLC 8 | byte2 ramp 0..255; byte3 inverse ramp; byte4 += U{-20..20} | timing preserved |
| RPM | Random burst length 48-96, step 2 | ID `0x316`, DLC 8 | byte2 sinusoid; byte3 ramp 0..255; byte5 += U{-30..30} | timing preserved |

Interpretation: same-family generated positive control; not independent real-world attack evidence.

## Sensitivity Stress Test

Source script: `scripts/generate_variant_sensitivity.py`

- Seed: 20260611.
- Output: `datasets/windows/variant_sensitivity_windows.npz`.
- Source: held-out Car-Hacking test normal windows only.
- Normal windows: 12,000.
- Per attack/severity scenario: 1,500.

| Attack | Severity | Burst length | Step | Amplitude | Approx. injected frames |
|---|---:|---:|---:|---:|---:|
| DoS | low | 24 | 3 | 0.25 | 8 |
| DoS | medium | 48 | 2 | 0.50 | 24 |
| DoS | high | 96 | 1 | 1.00 | 96 |
| Fuzzy | low | 24 | 4 | 0.25 | 6 |
| Fuzzy | medium | 48 | 3 | 0.50 | 16 |
| Fuzzy | high | 96 | 1 | 1.00 | 96 |
| Gear | low | 24 | 4 | 0.25 | 6 |
| Gear | medium | 48 | 2 | 0.50 | 24 |
| Gear | high | 96 | 1 | 1.00 | 96 |
| RPM | low | 24 | 4 | 0.25 | 6 |
| RPM | medium | 48 | 2 | 0.50 | 24 |
| RPM | high | 96 | 1 | 1.00 | 96 |

Payload and timing details:

- DoS: ID `0x000`, DLC 8, zero payload; `delta_t = max(original_delta_t * (0.4 - 0.25*amplitude), 1e-5)`.
- Fuzzy: ID uniformly 0..2047, DLC 8; payload is `(1-amplitude)*base + amplitude*random_payload`; `delta_t ~ U(0.00005, 0.005 + 0.01*amplitude)`.
- Gear: ID `0x43F`, DLC 8; byte2 ramp, byte3 inverse ramp, byte4 perturbation scaled by amplitude.
- RPM: ID `0x316`, DLC 8; byte2 sinusoid, byte3 ramp, byte5 perturbation scaled by amplitude.

## Target-ID Shift Stress Test

Source script: `scripts/generate_target_id_shift_stress.py`

- Seed: 20260612.
- Output: `datasets/windows/target_id_shift_stress_windows.npz`.
- Source: held-out Car-Hacking test normal windows only.
- Normal windows: 8,000.
- Per scenario: 4,000.
- Position rule: burst length 64, step 2, about 32 injected frames.

| Attack | Canonical ID | Shifted ID | Payload grammar |
|---|---|---|---|
| Gear | `0x43F` | `0x440` | byte2 ramp, byte3 inverse ramp, byte4 += U{-8..8} |
| RPM | `0x316` | `0x329` | byte2 sinusoid, byte3 ramp, byte5 += U{-12..12} |

Interpretation: weakens pure canonical-ID memorization concern, but remains generated controlled evidence because payload grammar is preserved.

## Out-of-Generator Payload-Position Stress Test

Source script: `scripts/generate_out_of_generator_stress.py`

- Seed: 20260613.
- Output: `datasets/windows/out_of_generator_stress_windows.npz`.
- Source: held-out Car-Hacking test normal windows only.
- Normal windows: 8,000.
- Per scenario: 4,000.
- Position rule: burst length 96, step 3, about 32 injected frames.

| Attack | Canonical ID | Shifted ID | Payload-position shift |
|---|---|---|---|
| Gear | `0x43F` | `0x440` | bytes 6/7 use ramp and noisy inverse ramp |
| RPM | `0x316` | `0x329` | bytes 4/6/7 use sawtooth, sinusoid, and noisy ramp |

Interpretation: bounds exact attack-type robustness under payload-position changes outside the disclosed training/fixed/sensitivity rules.
