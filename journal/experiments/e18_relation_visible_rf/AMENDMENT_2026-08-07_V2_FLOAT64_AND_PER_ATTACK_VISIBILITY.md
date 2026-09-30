# E18 amendment v2 — measure G17 in float64, and split visibility by attack

Amendment date: 2026-08-07

Amends: `journal/experiments/e18_relation_visible_rf/PREREG.md` sections 12.0,
12.8 and 15.5, as already amended by
`AMENDMENT_2026-08-07_V1_FLOAT32_REDUCTION_ARTIFACT.md`.

Registration commit amended: `f2f9930`
Prior amendment: `73aa169`

Status: **PROSPECTIVE with respect to every quantity it introduces.** No
float64 visibility measurement and no per-attack visibility measurement exists
for either feature map at the time this amendment is written. The stage-S2b
`_v1` measurement has been published and passed; it is **preserved verbatim**
and is not recomputed, revised or superseded. This amendment adds a second,
independently versioned measurement (`_v2`) alongside it.

The E18 fits proceeded under the passed `_v1` gate. If the `_v2` measurement
introduced here failed either registered floor, the consequence would be Branch
E and the fits would be reported as uninterpretable — that consequence is
registered here, before the `_v2` numbers exist.

## 1. Why

Two points were raised independently while stage S3/S4 was running, both of
which improve the measurement.

**Point one — precision.** The `_v1` mechanical metric compares `float32`
feature values. As amendment v1 established, that metric can be inflated by
summation-order rounding in the frozen `rf_features` reduction: `std(data1)`
drifts by up to `4.58e-05` under a permutation that leaves it mathematically
invariant. Amendment v1 answered this **structurally**, by excluding
`mean(data1)` and `std(data1)` from a substantive metric. A second and more
direct answer is to measure in **`float64`**, where the artifact does not arise
at all.

It does not arise for a reason worth recording, because it makes the `float64`
measurement exact rather than merely more precise. The raw channels are
integers in `[0, 255]`, so a 128-frame sum is at most `32,640`; the mean is that
sum divided by 128, exact in binary. Each deviation is therefore an exact
multiple of `1/128`, each squared deviation an exact multiple of `1/16384` with
numerator at most `255^2 * 128^2 ≈ 1.07e9`, and the 128-term sum has numerator
at most about `1.4e11`, comfortably below `2^53`. Every partial sum is exactly
representable in `float64`, so the reduction is **exactly** order-independent
and `std` is **exactly** invariant under the permutation. The `float64` metric
is thus not an approximation of the substantive metric; it is a numerically
exact realization of it, derived by a different route.

Publishing both routes and requiring both to agree is stronger than either
alone.

**Point two — per-attack visibility.** The `_v1` measurement pools Gear and RPM
windows. PREREG section 12.8 makes the Gear/RPM split of `theta` mandatory and
motivates it by the Gear relation being the cleanest manipulation. That
motivation is about **destruction completeness**, not about **visibility**, and
the two must not be conflated. An independent measurement on the 55-dimensional
map found Gear `2.54%` and RPM `2.55%` — the same visibility — which means E17's
Gear-versus-RPM asymmetry was *not* a visibility effect. Nothing in the `_v1`
table can rule the analogous confusion out for E18, because it never splits
visibility by attack.

## 2. Amendments

This amendment, like amendment v1, **only adds measurement and only tightens
gates**. No floor is lowered, no metric is replaced, no registered quantity is
recomputed, and nothing that could favour one outcome over another is changed.

### 2.1 A `float64` metric, added and gated

The G17 measurement is additionally computed with both feature maps evaluated in
`float64`:

- the 28 correlations are already computed in `float64` internally and are
  simply not cast down;
- the 55 marginals are computed by a `float64` mirror of the frozen
  `rf_features` — the same five statistics over the same axis, without the
  terminal `float32` cast.

The `float64` mirror is a **measurement instrument only**. The fitted feature
map is unchanged: every E18 checkpoint was, and remains, fit on the registered
`float32` map of PREREG section 5.1. The mirror is asserted to reproduce the
frozen function exactly when cast back to `float32`, so it cannot silently
diverge from it.

`visible_float64` counts a raw-differing window as visible if its `float64`
feature vector differs. Both registered floors — pooled `>= 0.50`, per-pair
`>= 0.25` — must be cleared by this metric as well. Passing G17 now requires
six conditions: two floors on each of three metrics (mechanical `float32`,
substantive `float32`, `float64`).

The record must state explicitly which precision each published figure was
measured at.

### 2.2 Per-attack visibility, added and mandatory to report

The same quantities are additionally computed separately for the **Gear**
(`y_attack_type == 3`) and **RPM** (`y_attack_type == 4`) windows of each pool
pair, for both feature maps and all three metrics, and published per pool pair
and pooled.

**Registered interpretive constraint, fixed here before the numbers exist.** No
Gear-versus-RPM difference in `theta`, in the `P = 1` cell, or in any other E18
outcome may be attributed to differential visibility unless the per-attack
visibility figures published under this amendment actually differ materially
between the two attacks. If Gear and RPM visibility are equal, or near-equal,
any observed Gear/RPM asymmetry must be reported as dispersion or as an effect
of the manipulation's own completeness — never as the detector seeing one attack
better than the other. This constraint binds regardless of which direction the
E18 split runs.

The per-attack figures are **reported, not gated**: the registered floor stays
on the pooled figure, because the inferential unit is the generator realization
and the endpoint is the equally-weighted Gear/RPM macro. Adding a per-attack
floor after the fact would be an unregistered tightening of the pass condition
in a place the registration did not contemplate.

### 2.3 Output identity

The `_v1` visibility table and record are **preserved unchanged**. The `_v2`
measurement is published to new paths:

```text
journal/results/tables/e18_manipulation_visibility_v2.csv
journal/experiments/e18_relation_visible_rf/manipulation_visibility_v2.json
```

No other E18 output version identity changes; the fits, scores and analysis
remain `_v1`, because nothing about them is affected. The registered analysis
stage is required to load **both** gate records and to refuse to run unless
**both** passed.

## 3. What this amendment does not change

- the Section 5.1 feature map actually fitted: 55 `float32` marginals plus 28
  `float32` correlations, 83 features, constant-channel convention `0.0`
- the G17 pooled floor `0.50` and per-pair floor `0.25`
- the `_v1` measurement, its table, or its published numbers
- the endpoint, aggregation order, inferential unit, margin `delta`,
  `delta_strict`, decision order, heterogeneity screen, bootstrap count or seed
- the fit grid, the seeds, the pools, the draws, or the evaluation material
- Branch E and its consequences
