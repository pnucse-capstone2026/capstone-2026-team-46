# E16 amendment 2026-07-25 v1 — heterogeneity screen must detect disagreement, not magnitude

Amendment time: 2026-07-25T17:58:00Z

Amends: `journal/experiments/e16_relational_response_decomposition/PREREG.md`
section 10, Step 1 only.

Registration commit amended: `7371443`

Status: **PRE-OUTCOME.** This amendment is committed before stage S2. At
amendment time no E16 pool, checkpoint, training log, score, table, figure, or
verdict exists. The defect was found by running the registered decision rule on
synthetic `theta` vectors during stage S1 implementation testing, which is the
purpose of that stage. No E16 outcome informed this change.

Evidence of the pre-outcome state at amendment time:

```text
journal/datasets/synthetic/e16_*            0 files
journal/models/generator_extension/*e16*    0 files
journal/results/tables/e16_*                0 files
journal/results/logs/e16_*                  0 files
```

## 1. The defect

PREREG section 10, Step 1 reads:

> **Step 1 — heterogeneity screen.** Count the construction-level `theta_g`
> values falling outside `[-delta, +delta]`. If **five or more of twenty** fall
> outside, the registered headline is **Branch C** (realization-contingent
> relation), regardless of the aggregate.

Because this step is evaluated first and counts magnitude only, it makes
Branch B unreachable. A genuine material relational increment — the exact result
Branch B exists to name — places **every** construction on the same side of the
band. Under the rule as written, a uniform true increment of `+0.30` (34% of the
total response) yields twenty constructions outside the band and is therefore
reported as "realization-contingent", which is the opposite of what the data
would show.

Verified on synthetic vectors before any real outcome existed:

| synthetic truth | constructions above / below | registered verdict | correct verdict |
|---|---|---|---|
| `theta = +0.30`, SD 0.04 | 20 / 0 | C | **B** |
| `theta = -0.25`, SD 0.04 | 0 / 20 | C | **B** |
| `theta = +0.18`, SD 0.04 | 16 / 0 | C | **B** |

The screen conflated *large* with *inconsistent*. Twenty constructions agreeing
that the increment is large is agreement, not heterogeneity.

## 2. The correction

Step 1 is replaced by the following. Steps 2, 3, and 4 are unchanged, as is
their order.

> **Step 1 — heterogeneity screen.** Let `above` be the number of
> construction-level `theta_g` greater than `+delta` and `below` the number less
> than `-delta`. The registered headline is **Branch C** if either criterion
> holds:
>
> * **C1 — directional disagreement:** `above >= 3` **and** `below >= 3`. The
>   constructions materially disagree about the direction of the relational
>   increment.
> * **C2 — aggregate/unit conflict:** `above + below >= 5` **and** the Step 2
>   equivalence condition holds. The aggregate declares the increment immaterial
>   while at least a quarter of the individual realizations do not.
>
> If neither holds, proceed to Step 2. The aggregate verdict from Steps 2--3 is
> computed and reported as descriptive whenever Branch C is triggered.

`HETEROGENEITY_MIN_EACH_SIDE = 3` and `HETEROGENEITY_MIN_OUTSIDE = 5` are frozen
by this amendment.

## 3. Why this is not a weakening

The correction makes Branch C **harder** to trigger for uniform effects and
leaves it **exactly as easy** to trigger for the case it was designed for:
constructions that disagree. It does not change `delta`, the confidence levels,
the decision order, or the conditions for Branches A, B, or D. Under the
planning assumption of PREREG section 9.2 the expected branch is unchanged.

The corrected rule was verified against ten synthetic scenarios spanning all
four branches, all of which now return the intended verdict:

| scenario | above / below | verdict |
|---|---|---|
| `theta = +0.063`, SD 0.04 (planning value) | 0 / 0 | A |
| `theta = 0`, SD 0.03 | 0 / 0 | A |
| `theta = +0.30`, SD 0.04 | 20 / 0 | B |
| `theta = -0.25`, SD 0.04 | 0 / 20 | B |
| `theta = +0.18`, SD 0.04 | 16 / 0 | B |
| `+0.5` x5, `-0.5` x5, `+0.02` x10 | 5 / 5 | C |
| `+0.2` x4, `-0.2` x4, `0` x12 | 4 / 4 | C |
| `theta = +0.10`, SD 0.35 | 9 / 5 | C |
| `theta = +0.134`, SD 0.09 | 7 / 0 | D |
| `theta = +0.16`, SD 0.13 | 10 / 0 | D |

## 4. Unchanged by this amendment

Everything else in the registration stands: the primary estimand and its
aggregation order; `delta = 0.133855625` and its multiplier and denominator;
`n = 20` and the frozen construction seed list; the placebo seed derivation; the
90% TOST and 95% material-difference confidence levels; the Student-`t` /
bootstrap disagreement downgrade; the relational-share rule; the mandatory
`P = 1` secondary in both absolute and ratio form; every secondary family; all
thirteen technical gates; and the claim boundaries of sections 13, 16, and 17.

## 5. Implementation note (not a design change)

Separately, stage S1 testing found and fixed an implementation bug in the
factorial contrast normalization: the divisor must be `2 ** (3 - k)` for an
effect naming `k` factors, not a constant `4`. A constant divisor understated
the two-way effects by a factor of two and the three-way effect by a factor of
four. This was caught by requiring the analyzer to reproduce the twenty-one
published E14 `Rule-minus-real`, `Placebo-minus-real`, and `Rule-minus-placebo`
effect values, which it now does exactly. This changed no registered quantity —
the primary estimand is a main effect and was already correct — but it is
recorded here because the secondary factorial families depend on it.
