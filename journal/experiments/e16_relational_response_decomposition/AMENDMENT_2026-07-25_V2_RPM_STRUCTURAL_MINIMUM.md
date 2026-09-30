# E16 amendment 2026-07-25 v2 — the RPM zero-residual gate is unachievable; gate on the structural minimum

Amendment time: 2026-07-25T18:41:00Z

Amends: `journal/experiments/e16_relational_response_decomposition/PREREG.md`
section 5.2 (RPM placebo construction) and section 15.2 gate G5 (RPM criterion).
Supersedes output version `v1` with `v2`.

Registration commit amended: `7371443`. Prior amendment: `v1`
(`AMENDMENT_2026-07-25_V1_HETEROGENEITY_SCREEN.md`).

Status: **PRE-OUTCOME technical stop.** Stage S2 halted on a registered gate
before publishing anything. No E16 pool, checkpoint, training log, score, table,
figure, or verdict existed at any point, and none exists now:

```text
journal/datasets/synthetic/e16_*            0 files
journal/models/generator_extension/*e16*    0 files
journal/results/tables/e16_*                0 files
journal/results/logs/e16_*                  0 files
```

Publication in this stage is atomic and happens only after every construction
passes, so the halt left no canonical artifact. The `v1` staging temporaries were
removed. No outcome informed this change.

## 1. What happened

Stage S2 generated construction `415637` (all gates passed) and halted on
construction `748203`:

```text
E16PoolError: E16 pool stage halted:
  construction 748203 failed gate(s): ['G5_rpm_coupling_residual_matches']
```

## 2. Why the registered gate cannot be met

PREREG section 5.2 destroys the RPM relation by rejection resampling until no
injected frame satisfies `((data1 - 8*data0) mod 256) in [0, 8)`, and gate G5
required that residual to be **exactly zero**.

The generator writes `data0 = (rpm // 32) % 256` and `data1 = (rpm // 4) % 256`,
so `data1 = 8*data0 + r` with `r = (rpm // 4) % 8`. Giving frame `i` the `data1`
of frame `j` leaves

```text
(data1_j - 8*data0_i) mod 256 = (8*(data0_j - data0_i) + r_j) mod 256
```

which stays below 8 **exactly when** `data0_j == data0_i (mod 32)`. The coupling
therefore survives precisely the same-class assignments, where the class is
`data0 mod 32`.

This makes the zero-residual requirement a combinatorial constraint. If one class
holds `c` of a window's `n` injected frames and `2c > n`, the pigeonhole
principle forces at least `2c - n` frames to receive a same-class value: **no
permutation whatsoever attains zero**, and no rejection cap can help.

Construction `748203` contains exactly one such window:

| construction | impossible windows | largest class / frames | forced residual | RPM frames | fraction |
|---|---|---|---|---|---|
| 314159 (E10a/E15 anchor) | 0 | — | 0 | 2,434,076 | 0 |
| 271828 (E15) | 0 | — | 0 | 2,434,790 | 0 |
| **748203 (E16)** | **1** (window 58187) | **6 / 11** | **1** | 2,438,345 | 4.10e-07 |
| 415637 (E16) | 0 | — | 0 | 2,441,113 | 0 |
| 560657 (E16) | 0 | — | 0 | 2,436,122 | 0 |
| 459336 (E16) | 0 | — | 0 | 2,434,005 | 0 |

The zero-residual gate was inherited from E10a, where it was calibrated on a
single construction seed that happened to contain no such window. It was never a
property of the construction law; it was a property of two lucky seeds. Drawing
twenty fresh constructions exposed this, which is what a prospective design is
for.

## 3. The correction

**Construction (PREREG section 5.2).** The RPM placebo permutation is replaced by
the same deterministic off-class rotation already used for Gear, grouping by
`data0 mod 32` instead of `data0 // 40`: sort frames by class with the placebo
stream randomising within class, then rotate by the largest class size (or by
`n - largest` when that exceeds half the window). This is deterministic given the
registered placebo seed and attains the structural minimum for every window.
Rejection resampling and the 1,000-attempt cap are removed.

**Gate (PREREG section 15.2, G5).** The RPM criterion becomes, jointly:

* `G5_rpm_residual_attains_structural_minimum` — the realised residual equals
  `sum over windows of max(0, 2*largest_class - n)` **exactly**; and
* `G5_rpm_residual_fraction` — that residual is below `0.01` of RPM injected
  frames, the same tolerance already registered for Gear.

`G5_rpm_structurally_constrained_windows` is reported for auditability.

The Gear criterion, all other gates, and every other registered quantity are
unchanged.

## 4. Why this is stricter, not weaker

The replaced gate demanded a value that is unattainable in general. The new gate
demands **optimality**: the permutation must destroy every coupling that any
permutation could destroy. A merely "good enough" permutation now fails, where
under a plain fraction threshold it would have passed. The additional fraction
bound keeps the absolute size of the unavoidable residue bounded by the tolerance
already registered for the Gear arm.

Verified on the construction that halted the stage:

```text
748203: RPM residual 1 = structural minimum 1, suboptimal windows 0,
        residual fraction 4.10e-07, structurally constrained windows 1
        Gear residual fraction 2.63e-05, off-bin fraction 0.999961
        all abort gates passed
```

The unavoidable residue is one injected frame in 2,438,345, or 0.00004% of the
RPM arm. Every other frame's coupling is destroyed.

## 5. Output version

All E16 canonical outputs move from `v1` to `v2`. No `v1` artifact exists to
preserve, because the stage halted before publication; the `v1` identity is
retired and never reused.

## 6. Unchanged by this amendment

The primary estimand and aggregation order; `delta = 0.133855625`; `n = 20` and
the frozen construction seed list; the placebo seed derivation; the decision
order as amended in `v1`; the mandatory `P = 1` secondary; every secondary
family; the Gear construction and its gate; gates G1--G4, G6--G13; and the claim
boundaries of PREREG sections 13, 16, and 17.
