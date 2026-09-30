# E18 amendment v1 — the float32 reduction artifact in `std(data1)`, and a tightened G17

Amendment date: 2026-08-07

Amends: `journal/experiments/e18_relation_visible_rf/PREREG.md` sections 5.1
(analytic expectation), 12.0 and 15.5 (gate G17).

Registration commit amended: `f2f9930`

Status: **PROSPECTIVE. Committed before stage S2b is rerun and before any G17
number is published.** At the time this amendment is written, no
`e18_manipulation_visibility_v1.csv`, no
`manipulation_visibility_v1.json`, no E18 checkpoint, no E18 score and no E18
verdict exists. Verified:

```text
journal/results/tables/e18_manipulation_visibility_*   0 files
journal/experiments/e18_relation_visible_rf/manipulation_visibility_*  0 files
journal/models/generator_extension/*e18*               0 files
journal/results/logs/e18_*                             0 files
```

The only published E18 artifacts are the stage-S2 pool inheritance record and
audit, which passed and are untouched by this amendment.

## 1. What happened

Stage S2b halted on the first pool pair with the registered technical stop of
PREREG section 5.1:

```text
T-STOP-E18-TRAIN: 415637: 9 of 83 features differ, above the analytic
maximum of 8; PREREG section 5.1 makes this an implementation error
```

The registration predicted that exactly 8 of the 83 features are capable of
differing between the Rule and placebo arms — `delta(data1)` and the seven
correlations `r(data1, data_k)` — and registered a realized count above 8 as an
implementation error.

## 2. Diagnosis: it is not an implementation error

The nine differing features on realization `415637` are:

| index | feature | windows differing | max abs difference |
|---|---|---|---|
| 47 | `delta(data1)` | 3,252 | 252 |
| 55 | `corr(data0, data1)` | 129,997 | 0.714 |
| 62--67 | `corr(data1, data2..data7)` | 129,993--130,000 | 0.315--0.391 |
| **14** | **`std(data1)`** | **47,814** | **3.81e-05** |

The eight predicted features are exactly the eight predicted features. The ninth
is `std(data1)`, and its largest observed difference over 260,000 windows is
`3.8e-05` on a statistic of order 70 — a **relative** difference of about
`5e-07`, which is the scale of `float32` epsilon.

The cause is floating-point non-associativity in the **inherited, frozen**
`rf_features` implementation
(`journal/scripts/train_generator_extension_rf.py:47`). `std` is a `float32`
reduction over the 128-frame axis; its summation order therefore depends on the
order of the values, and the placebo twin permutes exactly that order. In exact
arithmetic `std` is invariant under a permutation, and PREREG section 1.1's
invariance argument is unaffected: `std(data1)` carries no information about the
manipulation beyond `float32` rounding.

`mean(data1)` did **not** differ on this realization, and the reason is
arithmetic rather than luck: the raw channels are integers in `[0, 255]`, so a
128-term sum is at most 32,640 and is exactly representable in `float32`, and
division by 128 is exact. `std` involves squared deviations, which are not
exactly representable, so its reduction is order-sensitive. `min` and `max` are
order-independent selections and `delta` is a two-element difference; neither
can produce an artifact.

Raw channel confirmation on the same pair: the only raw channel that ever
differs between the arms is index 3, i.e. `data1`, exactly as the E16 generator
gates G1b/G1c record.

The E18 implementation is therefore correct, and the registered technical stop
fired on a false premise: the analytic prediction counted the features whose
**exact-arithmetic** value can change, and compared it against a count of
features whose **float32** value can change. Those are different sets.

## 3. Amendments

This amendment **only tightens**. No floor is lowered, no threshold is relaxed,
no metric is replaced, and nothing that could favour one outcome over another is
changed.

### 3.1 The analytic maximum becomes 10, defined structurally

PREREG section 5.1's analytic expectation is replaced by:

- **Substantive set (8 features), unchanged:** `delta(data1)` and
  `r(data1, data_k)` for `k in {0, 2, 3, 4, 5, 6, 7}`. These are the features
  whose exact-arithmetic value can change under the manipulation.
- **Reduction-artifact set (2 features), added:** `mean(data1)` and
  `std(data1)`. These are provably invariant in exact arithmetic; any observed
  difference is a `float32` reduction artifact of the frozen `rf_features` and
  carries no information about the permutation.
- **Every other one of the 83 features remains provably invariant.** The
  technical stop is retained and is now: a differing feature index outside the
  union of those two sets is an implementation error and halts the stage.

The set membership is defined **structurally**, from the channel identity of
`data1` and the frozen correlation pair list, not by a magnitude threshold.
No number in this amendment was chosen by looking at a visibility outcome.

### 3.2 The G17 table records both metrics, per pool pair

In addition to every quantity already registered in PREREG section 15.5, the
published table records, per pool pair and per feature map:

- `substantive_features_ever_differing` and their indices;
- `artifact_features_ever_differing` and their indices;
- `substantive_visible_windows` and `substantive_visible_fraction`, counting a
  window as visible only if at least one **substantive** feature differs;
- `artifact_max_abs_difference`, the largest absolute between-arm difference
  observed on any artifact feature, so a reader can verify its magnitude
  directly rather than take this amendment's word for it.

The already-registered mechanical quantities — exact `float32` equality over all
83 features, which is what the estimator actually consumes — are retained
unchanged and are still published.

### 3.3 The G17 floor must now be met by BOTH metrics

PREREG section 15.5 registered the floor on the mechanical metric:

```text
POOLED FLOOR   pooled visibility over the 20 primary pairs   >= 0.50
PER-PAIR FLOOR min visibility over the 20 primary pairs      >= 0.25
```

Both thresholds are unchanged. This amendment requires the same two thresholds
to be cleared **also** by the substantive metric of section 3.2. Passing G17
now requires four conditions instead of two.

**Why this direction and not the other.** The mechanical metric can be inflated
by the `std(data1)` artifact: a window in which the permutation moved no
first/last value and changed no correlation could still register as "visible"
through a `1e-05` rounding difference. Judging E18's precondition on a metric
that a rounding artifact can inflate would be exactly the kind of
self-serving measurement this experiment exists to avoid. Requiring the
substantive metric to clear the same floor removes that possibility. An
amendment that made passing easier would be illegitimate here; one that makes it
strictly harder cannot bias the result toward a finding.

### 3.4 The E17 comparison figures are reported on both metrics

PREREG section 1.1 quotes "1 of 55 features, 2.54% of altered windows" for the
E17 marginal map. Those are **substantive-metric** figures. The same
`std(data1)` artifact applies to the 55-dimensional map, so its mechanical
figure is larger while consisting of `1e-05`-scale rounding noise. Both are
published side by side in the G17 table so the contrast with E18 is stated on
like-for-like terms and the 2.54% figure is not quietly compared against a
mechanically inflated E18 number.

This does not weaken PREREG section 1.1's reclassification of E17. Whether the
E17 detector saw 2.54% of the manipulation substantively or additionally saw a
`1e-05` rounding shadow of it, 54 of its 55 features remained provably invariant
in exact arithmetic and its representation could not express the destroyed
relation.

## 4. Output version identity

PREREG section 15.4 requires a new output version identity when a stage is
rerun after a technical stop, so that a failed stage's artifacts are preserved
as failure provenance rather than overwritten.

Stage S2b **published no artifact**: the stop fired inside the measurement loop,
before the publication step, and the no-clobber preflight had already confirmed
both target paths absent. There is nothing to preserve and nothing to overwrite.
The `_v1` output identity is therefore retained for S2b and for every later E18
stage, and this amendment is the failure provenance of the aborted attempt. The
retention is recorded here explicitly rather than assumed.

Stage S2b is not resumed in place; it is rerun from the beginning under the
amended gate.

## 5. What this amendment does not change

- the Section 5.1 feature map — 55 marginals plus 28 payload-byte correlations,
  83 features, constant-channel convention `0.0`, frozen pair order
- the G17 pooled floor `0.50` and per-pair floor `0.25`
- the registered mechanical measurement definition (exact `float32` equality on
  the values the estimator consumes)
- the endpoint, aggregation order, inferential unit, margin `delta`,
  `delta_strict`, decision order, heterogeneity screen, bootstrap count or seed
- the fit grid, the seeds, the pools, the draws, or the evaluation material
- Branch E and its consequences
