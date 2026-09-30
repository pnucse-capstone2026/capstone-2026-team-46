# E17 preregistration — Random-Forest replication of the relational-versus-marginal decomposition

Registration time: 2026-08-07T00:37:13Z

Registration source commit: `42ec54c`

Status: **PRE-RESULT. E17 implementation, Random-Forest fitting, scoring, and
analysis are prohibited until this document is committed and the freeze gates in
Section 15 pass.**

At registration time no E17 Random-Forest checkpoint, training log, score,
outcome table, figure, or verdict existed, and no Random-Forest fit on any E16
Rule or placebo pool had ever been performed. Verified at the registration
commit:

```text
journal/datasets/synthetic/e17_*                        0 files
journal/models/generator_extension/*e17*                0 files
journal/models/generator_extension/rf_*cseed*           0 files
journal/results/tables/e17_*                            0 files
journal/results/logs/e17_*                              0 files
```

The E16 CNN outcomes are already known and are quoted throughout this document
as **planning inputs only**. This registration is therefore prospective with
respect to every Random-Forest quantity it names. Any later design change
requires a dated amendment committed before the affected stage; it must not
silently edit this record.

**Freeze intent.** This document is frozen on commit. Sections 3, 5.2, 7, 8, 9,
10, 12, 13, and 15 are the registered decision surface. The margin of Section 9,
the aggregation order of Section 8, the decision order of Section 10, the
inferential unit of Section 8.3, the bootstrap resample count and seed of
Section 10, and the mandatory secondary reporting of Section 12 may not be
altered after any Random-Forest outcome exists.

### Vocabulary

This registration says **generator realization** for what the E16 code and CSV
schemas call a `construction_seed`. The legacy column name `construction_seed`
is retained verbatim inside CSV and NPZ artifacts for schema continuity with
E16; it is never used in reader-visible prose. The five repeated model runs are
**paired pipeline seeds**. `C_P(Placebo - Real)` is the **marginal-exposure
component** and `C_P(Rule - Placebo)` is the **relational increment**.

## 1. Question and evidential scope

E16 established, on a prospective registration over twenty generator
realizations, that destroying the implemented Gear/RPM inter-byte relation while
holding every registered marginal fixed leaves the registered
payload-destination contrast practically unchanged in a CNN detector
(`theta = -0.005915208`, 90% TOST `[-0.021375, +0.009544]` inside
`delta = 0.133855625`, Branch A; the placebo arm reproduces 99.3% of the Rule
arm's payload-destination response). That result is conditional on one detector
family, and on that family's training dynamics: a fixed 6,156-update matched
budget, minibatch ordering, validation cadence, and best-validation selection.

E17 asks:

> Does the E16 practical-equivalence verdict replicate in a second detector
> family that shares no training dynamics with the CNN — no gradient descent, no
> optimizer-update budget, no early-stopping trajectory, no minibatch order, and
> no learned representation?

A Random Forest over fixed per-channel window summary statistics is the
strongest available contrast to the CNN inside this repository's frozen
evaluation protocol: it is a non-parametric, non-gradient, non-representational
learner whose only inputs are 55 hand-specified aggregates of the same training
rows.

### 1.1 Decomposition identity

For generator realization `g` and pipeline seed `p`, let `C_P{·}` denote the
registered `P` factorial contrast on the primary endpoint (Section 8), and let

```text
Delta^Rule_{g,p}     = C_P{ Y_Rule,g,p    - Y_real,p }
Delta^Placebo_{g,p}  = C_P{ Y_Placebo,g,p - Y_real,p }
```

where `Y_real,p` is the shared E17 Random-Forest real-only reference at pipeline
seed `p`. Because `C_P{·}` is linear and the real reference is shared within
pipeline seed `p`, the following identity holds exactly and is not an
approximation:

```text
C_P(Rule - Real)  =  C_P(Placebo - Real)  +  C_P(Rule - Placebo)
      total       =   marginal-exposure   +      relational
                          component            increment
```

E17 estimates all three terms on the same twenty generator realizations that E16
used. The identity is verified numerically to floating-point closure as a gate
(Section 15.3, G11); it is not assumed.

### 1.2 What E17 cannot establish

All conclusions are conditional on the frozen Car-Hacking train/validation/test
split, the disclosed 65,000-window-per-class Rule construction law, the twenty
registered generator realizations, the five paired pipeline seeds, the E14
evaluation constructions and three descriptive blocks, the frozen E8 evaluation
manifest, and the single Random-Forest configuration of Section 5.1.

Specifically, E17 does **not**:

- make `theta` a **detector-population** result. Two detector families are two
  points, not a sample from a population of detectors. The inferential unit
  remains the generator realization; the detector family is a fixed, registered
  factor with exactly two observed levels and is never treated as random.
- establish synthetic-data **realism**, physical payload causation, or
  **replacement** of real attack data.
- establish **cross-dataset or cross-domain robustness**. E17 contains no
  external-dataset evaluation and adds no new evaluation construction.
- establish universal generator behavior, or behavior of grammars outside the
  implemented Rule law.
- **re-open, replace, adjudicate, or supersede the E16 CNN registered primary
  result.** E16 remains the registered primary of the manuscript regardless of
  the E17 outcome. E17 is a registered second-family replication reported
  alongside it.
- speak to whether the CNN's layer-local behavior has any Random-Forest
  analogue. The Random Forest has no layers; no layer-local claim may be
  transported across families in either direction.

A Branch A outcome in E17 is **not** evidence that relational structure is
irrelevant to CAN intrusion detection in general.

### 1.3 Why a second detector family is the right test, and not a foregone conclusion

The E16 result has one salient alternative reading: that the CNN's practical
equivalence is an artifact of its *training schedule* — a fixed matched update
budget may simply not give a gradient learner enough room to exploit relational
structure that a differently-trained model would exploit. That reading is
answerable only by a learner with no update budget at all.

The Random Forest also *inverts* the representational argument. If the
implemented Gear/RPM inter-byte relation matters, it should be **easier**, not
harder, for a tree ensemble to exploit: the relation is an algebraic constraint
between `data0` and `data1`, and axis-aligned splits on per-channel summary
statistics of those two channels are a natural way to encode it. A Random Forest
that also finds the relation immaterial is therefore a stronger negative than
the CNN's, and a Random Forest that finds a material relational increment where
the CNN did not is a genuine, publishable disagreement that constrains the E16
headline. Both outcomes are informative and must be reported as positive
findings in the registered form of Section 13.

### 1.4 Relation to the E16 registered primary

E17 is subordinate in the reporting hierarchy and independent in its inference.
It shares the E16 pools, the E16 evaluation material, the E16 endpoint, the E16
aggregation order, and the E16 margin. It does not share the E16 detector, the
E16 checkpoints, or the E16 real-only references. No E17 gate, outcome, or
verdict may modify an E16 artifact, and no E16 conclusion is restated as a
function of an E17 outcome.

## 2. Mandatory execution order

Stages must run in this order. A later stage must not begin until the earlier
stage's gates have passed and its run record has been committed.

1. **S0 — freeze.** Section 15.1 gate G0; commit this document.
2. **S1 — implementation.** Scripts and tests (Section 14.1) committed with no
   E17 outcome, checkpoint, or score present.
3. **S2 — pool inheritance verification.** Every E16 pool consumed by E17 is
   re-hashed and matched against the frozen `e16_pool_audit_v2.csv` record.
   Gates G1i--G9i (Section 15.2). No fitting occurs in this stage.
4. **S3 — shared Random-Forest real-only references.** Five fits, one per
   pipeline seed. Gates G10a, G14, G15.
5. **S4 — paired Random-Forest fits.** 240 fits. Gates G10a, G10c-i, G10c-ii,
   G10d, G14, G15.
6. **S5 — real-only reference continuity.** Gate G12, before any E17 Rule or
   placebo checkpoint is scored.
7. **S6 — scoring.** Exact and binary scored separately.
8. **S7 — registered analysis and verdict.** Gates G11, G13, G16.

No stage may be resumed in place after an abort. Retry requires a dated
amendment and a new output version identity (Section 15.4).

## 3. Frozen seed axes and roles

### 3.1 Generator-realization seeds

E17 reuses the E16 registered draw **verbatim**. The seeds are not redrawn, and
the meta-seed draw of E16 PREREG section 3.1 is not re-executed. They are read
from `journal/scripts/generate_e16_rule_placebo_pairs.py` as
`CONSTRUCTION_SEEDS`:

```text
415637  748203  560657  459336  457191  999191  715822  298581  922089  864178
583909  467232  550427  485791  820632  837943  222537  685012  192753  739263
```

These twenty are the **primary inferential units** (`n = 20`, `df = 19`). No
seed may be added, removed, reordered, or substituted after registration. If a
realization fails a gate, Section 15.4 applies; the seed is not silently
replaced.

Reusing E16's exact twenty units is a registered design choice, not a
convenience. It makes the CNN and Random-Forest `theta_g` vectors **paired at
the level of the inferential unit**, which is what licenses the descriptive
family comparison of Section 12.7.

### 3.2 Bridge realizations — included, sensitivity only

The four E15 bridge realizations are read from the same module as
`BRIDGE_SEEDS`:

```text
271828  161803  141421  173205
```

They **are** included in E17, and they are **a bridge sensitivity only, never
primary**. They are excluded from `n`, from `df`, from the primary interval,
from the heterogeneity screen, from the margin-scale sensitivity, and from every
Section 12 family except Section 12.9. They are reported in a separate table and
always labelled as a bridge sensitivity attached to the E15 lineage.

Unlike E16 — which fit only the placebo arm for the bridge realizations, because
their CNN Rule outcomes were already observed — E17 fits **both arms** for the
bridge realizations, since no Random-Forest outcome exists for either. This does
not promote them: they are still excluded from the primary, so that the E17
inferential unit set is **identical** to E16's and the two families are compared
on the same twenty units.

The bridge Rule arm consumes the frozen E15 v2 pools
`journal/datasets/synthetic/rule_cseed<g>_windows_v2.npz`; the bridge placebo arm
consumes `journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz`.

### 3.3 Placebo pairing

E17 derives no placebo seed and generates no pool. Each realization's placebo
pool is the E16 published aligned twin, consumed by path and verified by
SHA-256 (Section 4.1). The E16 placebo seed derivation, the Gear off-bin
rotation, and the RPM structural-minimum rotation (E16 amendment
`AMENDMENT_2026-07-25_V2_RPM_STRUCTURAL_MINIMUM.md`) are inherited unchanged.

### 3.4 Detector-pipeline seeds

Frozen and identical to E14/E15/E16:

```text
7, 42, 123, 2026, 3407
```

The paired pipeline seeds are a fixed repetition axis averaged inside each
generator realization. They are **not** an inferential unit. The 200 primary
realization × arm × pipeline cells are never counted as 200 independent
replicates.

### 3.5 Evaluation blocks

The three frame-disjoint blocks and 6,000 held-out normal bases of the frozen
E8 manifest are reused without redrawing:

```text
journal/results/tables/evaluation_realization_blocks_e13_sampling_v2.csv
```

Blocks are averaged inside each generator realization and are never an
inferential unit.

## 4. Frozen inputs and reused lineage

### 4.1 Pools — consumed by hash, never regenerated

E17 generates **no** synthetic pool. It consumes, read-only:

```text
journal/datasets/synthetic/e16_rule_cseed<g>_windows_v2.npz        (20 primary)
journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz     (20 primary)
journal/datasets/synthetic/rule_cseed<g>_windows_v2.npz            ( 4 bridge)
journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz     ( 4 bridge)
```

Every consumed pool's `x_digest` must equal the value frozen in
`journal/results/tables/e16_pool_audit_v2.csv` for that realization and arm.
This is blocking gate G1i (Section 15.2). No pool file is opened for writing at
any point in E17.

### 4.2 Data and evaluation inputs

Reused unchanged, with SHA-256 recorded at S0 and re-verified before each
subsequent stage: the Car-Hacking train/test window NPZs, the E8 evaluation
realization manifest, the E14 `2x2x2` factorial transformation code path, and
the four ID strata (Gear canonical `0x43F`, Gear shifted `0x440`, RPM canonical
`0x316`, RPM shifted `0x329`).

### 4.3 Shared Random-Forest real-only references

E17 fits **five new** shared real-only Random Forests, one per pipeline seed:

```text
journal/models/generator_extension/rf_real_only_e17_v1_seed{7,42,123,2026,3407}.joblib
```

They are new because the pre-existing `rf_real_only_seed<p>.joblib` artifacts
were fit under the legacy sampling policy and carry no strict-v2 training log;
reusing them would silently mix sampling policies with the strict-v2 Rule and
placebo arms. Those legacy artifacts are **not** read, modified, or overwritten
by E17. The `e17_v1` model tag keeps the new checkpoints on disjoint paths
(gate G10a).

Every `Delta^Rule_{g,p}` and `Delta^Placebo_{g,p}` at the same `p` shares one
real reference. This correlation structure is preserved by the pairing and is
disclosed as a limitation in Section 16.

### 4.4 Frozen evaluation material

The E14 eight factorial cells, three frame-disjoint blocks, four strata, and the
`S0 = start + 2*arange(32)` / `S1 = start + 3*arange(32)` mask schedules are
frozen evaluation material. E17 introduces no new evaluation construction, no
new generator family, no new pool, and no external dataset. The only new axis is
the detector family.

## 5. Detector family and the matching contract

### 5.1 Random-Forest contract

Frozen verbatim from `journal/scripts/train_generator_extension_rf.py`. No
hyperparameter is tuned, searched, or selected on any E17 outcome.

**Features** — `rf_features` at `train_generator_extension_rf.py:47`, computed on
**raw, un-standardized** windows. Per channel, over the 128-frame axis:
`mean`, `std`, `min`, `max`, and `delta = x[:, -1, :] - x[:, 0, :]`, concatenated
to a **55-dimensional** `float32` vector per window (11 channels x 5 statistics).

**Estimator** —

```python
RandomForestClassifier(
    n_estimators=160,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=seed,
)
```

`seed` is the pipeline seed. Row order before fitting is
`numpy.random.default_rng(seed).permutation(len(y))`, identical for both arms of
a realization because the two pools are array-aligned and equal in length.

**No standardizer.** Unlike the CNN arms, the Random-Forest arms fit on raw
window statistics; no standardizer is derived, fitted, or applied. This is the
existing frozen convention, not a new choice.

**Setting names.** `rule_0p30` and `placebo_0p30` are already supported settings
in `VALID_SETTINGS`; `placebo_0p30` was added for E10 and is reused unchanged.
Synthetic consumption is the `+30%` no-replacement strict-v2 draw.

**Harness note (not a design change).** `train_generator_extension_rf.py` cannot
be invoked directly for E17: it has no per-realization pool override, and its
strict-v2 path admits only the declared `STRICT_V2_POOL_CONFIGS` pools. E17
therefore adds a runner (Section 14.1) that parameterizes the pool path and
model identity and adds the gates below. The existing script's feature function,
estimator configuration, permutation convention, and no-clobber preflight are
reused unmodified; the E17 runner must import them rather than re-implement
them, and gate G14 asserts this.

### 5.2 The identical-training-row contract (blocking)

**Registered requirement.** For every generator realization `g`, arm `a`, and
pipeline seed `p`, the Random-Forest fit must consume **exactly the same
synthetic row indices** as the corresponding E16 CNN fit. Operationally, the
E17 training manifest's `sampling_index_sha256` for `(g, a, p)` must equal the
value recorded for `(g, a, p)` in
`journal/results/tables/e16_training_manifest_v2.csv`.

This is enforced as two blocking gates:

- **G10c-i — internal pairing.** Within each realization and pipeline seed, the
  Rule and placebo arms drew identical synthetic row indices. Applies to all 24
  realizations, 120 pairs.
- **G10c-ii — cross-family inheritance.** The E17 `sampling_index_sha256`
  equals the frozen E16 CNN `sampling_index_sha256` for the same
  `(realization, arm, pipeline seed)`. Applies to the 200 primary fits and the
  20 bridge placebo fits. The 20 bridge Rule fits have **no** E16 CNN
  counterpart — E16 fit only the placebo arm for bridge realizations — and are
  therefore exempt from G10c-ii and covered by G10c-i alone. This exemption is
  recorded explicitly in the training run record; it is never silently waived.

Additionally, per fit, `requested = drawn = unique` and `repeated = 0` are
verified, with the per-class request below 65,000 (gate G9i).

**Why this is a methodological advantage worth registering, not merely a
bookkeeping constraint.** The E16 CNN design had to *absorb* a training-schedule
confound. Because a gradient learner's outcome depends on how long it trains, in
what minibatch order, and where its best-validation checkpoint falls, E16 was
obliged to equalize a 6,156-update budget, a 513-step validation cadence, a
minibatch RNG convention, and a best-validation selection rule across the two
arms, and to defend that budget as the right one. Every one of those is a
matched *nuisance* parameter: matched, but still present, and still a channel
through which an arm difference could in principle arise or be masked.

A Random Forest has none of them. There is no optimizer-update budget, no
early-stopping trajectory, no minibatch order, no learning-rate schedule, and no
validation-based checkpoint selection. Given the same training rows, the same
row permutation, and the same `random_state`, the fitted ensemble is a
deterministic function of the data alone. The CNN's matched-budget contract is
therefore **replaced** in E17 by the identical-training-row contract, which is
strictly stronger: instead of equalizing a training schedule, E17 removes the
training schedule from the design. Any Rule-minus-placebo difference observed in
E17 cannot be attributed to training dynamics, because there are none to
attribute it to.

This is registered as a stated advantage of the replication and must be reported
as such in Section 13 under every branch.

### 5.3 Naming and atomic publication

```text
journal/models/generator_extension/
  rf_rule_0p30_cseed<g>_e17_v1_seed<p>.joblib
  rf_placebo_0p30_cseed<g>_e17_v1_seed<p>.joblib
  rf_real_only_e17_v1_seed<p>.joblib
```

All outputs are written atomically with a no-clobber preflight (gate G10a). No
E10, E14, E15, E16, legacy RF, or WISA artifact is overwritten, and `wisa/` and
`wisa_camera_ready/` are not modified.

## 6. Fit grid and counts

| role | realizations | arms | pipeline seeds | fits |
|---|---|---|---|---|
| primary | 20 | rule, placebo | 5 | **200** |
| bridge sensitivity | 4 | rule, placebo | 5 | **40** |
| shared real-only reference | — | real_only | 5 | **5** |
| | | | **total** | **245** |

The 200 primary fits are 100 paired Rule/placebo comparisons. They are never
counted as 200 independent replicates.

## 7. Endpoints

**Primary endpoint.** Exact attack-type recall, canonical-ID strata, Gear and
RPM computed separately then equally weighted (exact-macro). Identical to E16.

**Mandatory companion.** Binary attack recall on the same fits.

Exact and binary are scored and reported separately and are never merged into a
single detection number.

## 8. Contrasts and aggregation order

### 8.1 Cell-level augmentation deltas

For generator realization `g`, pipeline seed `p`, block `b`, attack `a`, and
factorial cell `(P, S, D)`:

```text
A^Rule_{g,p,b,a}(P,S,D)    = Recall_rule(g,p,b,a,P,S,D)    - Recall_real(p,b,a,P,S,D)
A^Placebo_{g,p,b,a}(P,S,D) = Recall_placebo(g,p,b,a,P,S,D) - Recall_real(p,b,a,P,S,D)
```

where `Recall_real` is the E17 Random-Forest real-only reference at pipeline
seed `p`.

### 8.2 Registered aggregation order

Fixed; must not be permuted:

1. **attack** — Gear and RPM equally weighted
2. **block** — the three blocks equally weighted
3. **cell** — `C_P{A} = mean over (S,D) of [ A(P=1,S,D) - A(P=0,S,D) ]`
4. **pipeline** — the five paired pipeline seeds equally weighted
5. **generator realization** — the twenty realizations are the inferential unit

giving, for each realization `g`:

```text
Delta^Rule_g     = mean_p C_P{A^Rule_{g,p}}       (total response)
Delta^Placebo_g  = mean_p C_P{A^Placebo_{g,p}}    (marginal-exposure component)
theta_g          = Delta^Rule_g - Delta^Placebo_g (relational increment)
```

The factorial contrast normalization divisor is `2 ** (3 - k)` for an effect
naming `k` factors, inherited from E16 amendment v1 section 5.

### 8.3 Inferential unit

**`theta_g` is the primary estimand.** The primary statistic is the mean of the
twenty `theta_g` values with an untruncated Student-`t` interval on `df = 19`.

The inferential unit is the **generator realization**: `n = 20`, `df = 19`.
Every new numeric result in E17 is reported as a mean over the twenty generator
realizations, with the sample standard deviation, and states `n = 20, df = 19`
explicitly. Blocks, cells, attacks, pipeline seeds, windows, and the 200 primary
fits are never counted as inferential units.

## 9. Equivalence margin, planning values, and power

### 9.1 The margin is inherited, not re-derived

```text
delta = 0.133855625
```

This is the frozen E16 CNN registered margin, taken **verbatim**, and used as a
**fixed absolute margin on the same exact-macro recall-contrast scale**. It is
not recomputed, not rescaled to Random-Forest recall levels, and not re-derived
from any Random-Forest quantity. Its provenance is recorded here for the reader
but is not re-executed:

```text
delta = 0.15 x 0.8923708333333333
```

with the multiplier `0.15` and the denominator the published E15 v2
new-four-realization Rule-minus-real `P` mean on the primary endpoint
(`journal/results/tables/e15_l4_factorial_hierarchical_summary_v2.csv`,
`row_type = summary_new4`).

**Registered reason for inheriting rather than re-deriving.** Re-deriving the
margin from Random-Forest outcomes would be circular: the margin is the
threshold that decides the Random-Forest verdict, and a threshold computed from
the very outcomes it adjudicates can be made to return either verdict by the
choice of denominator. It would also destroy comparability — a replication whose
decision threshold is rescaled to its own effect size is not testing whether the
first result replicates, it is testing a different hypothesis. E17 asks whether
the CNN verdict replicates, so it must be judged against the CNN's registered
threshold on the CNN's registered scale.

The consequence is registered in advance and accepted: because Random-Forest
recall levels on this endpoint may be lower than the CNN's, an inherited
absolute margin may be **relatively more permissive** for the Random Forest than
it was for the CNN. That is precisely why the margin-scale sensitivity of
Section 12.2 is registered now as mandatory, and why its verdict must be
reported alongside the primary in every branch.

### 9.2 Planning values (CNN, already observed; quoted as planning inputs only)

| quantity | CNN value (E16, `n = 20`) |
|---|---|
| `theta` mean | `-0.005915208` |
| `theta` SD | `0.039983478` |
| `theta` range | `[-0.063625, +0.085629]` |
| `Delta^Rule` mean (SD) | `-0.881133333` (`0.040395291`) |
| `Delta^Placebo` mean (SD) | `-0.875218125` (`0.041077081`) |
| placebo share of total response | `99.3%` |
| realizations outside `[-delta, +delta]` | `0 / 20` |
| decomposition closure residual (max) | `0.000e+00` |

Planning value for the Random-Forest `theta`: `0.000`, with a planning SD range
of `0.04`--`0.12`. The Random-Forest between-realization dispersion is unknown
and cannot be known before the experiment; the upper end of the range is set at
three times the observed CNN SD to be pessimistic about a coarser feature space.

### 9.3 Decision geometry and power

Under the Section 10 rule the verdict is a deterministic function of the sample
mean and SD:

```text
Branch A requires   |mean| + 0.386646 * s  <  0.133855625
Branch B requires   |mean| - 0.468014 * s  >  0.133855625
```

with `0.386646 = t(0.95, 19) / sqrt(20)` and `0.468014 = t(0.975, 19) / sqrt(20)`,
`t(0.95,19) = 1.729133`, `t(0.975,19) = 2.093024`.

Largest `|mean|` still admitting Branch A, and smallest `|mean|` admitting
Branch B, as a function of the realized SD:

| SD of `theta_g` | max `|mean|` for A | min `|mean|` for B |
|---|---|---|
| 0.020 | 0.126123 | 0.143216 |
| 0.040 | 0.118390 | 0.152576 |
| 0.060 | 0.110657 | 0.161936 |
| 0.080 | 0.102924 | 0.171297 |
| 0.120 | 0.087458 | 0.190017 |

Registered-rule probabilities, Monte Carlo over 200,000 draws of `n = 20`
Gaussian `theta_g` per configuration (planning computation performed at
registration time on synthetic vectors; no Random-Forest outcome was involved):

| true `theta` | SD 0.040 | SD 0.060 | SD 0.080 | SD 0.120 |
|---|---|---|---|---|
| `0.00` — P(A) | 100% | 100% | 100% | 99.9% |
| `+0.05` — P(A) | 100% | 100% | 99.8% | 91.4% |
| `+0.18` — P(B) | 99.8% | 90.4% | 68.6% | 37.3% |
| `+0.20` — P(B) | 100% | 99.7% | 93.9% | 64.7% |
| `+0.25` — P(B) | 100% | 100% | 100% | 98.4% |
| `+0.30` — P(B) | 100% | 100% | 100% | 100% |

The design is not biased toward equivalence: a uniform relational increment at
22% of the E15 total response (`+0.20`) is detected as a material difference with
probability 94% even at three times the CNN's observed dispersion. At the
pessimistic `SD = 0.120` a `+0.18` increment falls to 37% power for Branch B and
would most often return Branch D, which is the honest outcome and is reported as
such.

`n = 20` is registered and is not a free parameter: it is fixed by the
requirement that E17 use E16's exact inferential units (Section 3.1).

## 10. Primary decision rule

Evaluated in this order. The order is registered and must not be changed after
seeing outcomes. It mirrors E16 as amended by
`AMENDMENT_2026-07-25_V1_HETEROGENEITY_SCREEN.md`.

**Step 1 — heterogeneity screen.** Let `above` be the number of
realization-level `theta_g` greater than `+delta` and `below` the number less
than `-delta`. The registered headline is **Branch C** if either criterion
holds:

- **C1 — directional disagreement:** `above >= 3` **and** `below >= 3`. The
  realizations materially disagree about the direction of the relational
  increment.
- **C2 — aggregate/unit conflict:** `above + below >= 5` **and** the Step 2
  equivalence condition holds. The aggregate declares the increment immaterial
  while at least a quarter of the individual realizations do not.

`HETEROGENEITY_MIN_EACH_SIDE = 3` and `HETEROGENEITY_MIN_OUTSIDE = 5` are frozen.
If neither criterion holds, proceed to Step 2. When Branch C triggers, the
aggregate verdict from Steps 2--3 is still computed and reported as descriptive.

**Step 2 — practical equivalence.** If the 90% Student-`t` interval for the mean
of `theta_g` (two one-sided tests at `alpha = 0.05`, `df = 19`) lies entirely
within `[-delta, +delta]`, the verdict is **Branch A**.

**Step 3 — material difference.** Otherwise, if the 95% Student-`t` interval
lies entirely outside `[-delta, +delta]` on one side, the verdict is
**Branch B**.

**Step 4 — inconclusive.** Otherwise the verdict is **Branch D**. A Branch D
result is reported as such and is never promoted to Branch A or Branch B.

The primary is a single test on a single endpoint and therefore carries no
multiplicity correction. Effect size, interval, realization sign pattern, and
relational share are reported before any p-value; no p-value is presented as the
primary scientific claim.

**Robustness, reported alongside the primary and never substituted for it:** a
realization-level exact sign test on `theta_g`, and a percentile bootstrap
interval over the twenty generator realizations with

```text
bootstrap resamples = 10000
bootstrap seed      = 20260807      # numpy.random.default_rng, PCG64
```

If the bootstrap and Student-`t` verdicts disagree — that is, if the branch
returned by Steps 1--4 using the bootstrap 90%/95% percentile intervals differs
from the branch returned using the Student-`t` intervals — **the disagreement is
reported and the verdict is downgraded to Branch D.**

## 11. Relational share

Reported descriptively for interpretation, not as a test statistic:

```text
share_g = theta_g / |Delta^Rule_g|
```

Summarized by median and interquartile range across the twenty generator
realizations, with the denominator distribution reported. Any realization with
`|Delta^Rule_g| < 0.10` is flagged and excluded from the share summary only,
never from the primary.

## 12. Mandatory secondary reporting, registered now

Each family is separately Holm-corrected where indicated. None may replace the
primary or be promoted to the headline after seeing outcomes. Sections 12.1,
12.2, and 12.3 are **mandatory** and are registered now because they materially
affect how the headline may be phrased.

### 12.1 (a) The absolute out-of-grammar `P = 1` cell — mandatory

At `P = 1` the evaluation payload is placed **outside** the Rule grammar bytes.
For the CNN this is where the entire Rule-minus-placebo difference lives.

**Registered reporting.** For each of the twenty generator realizations, at
`P = 1`, canonical strata, exact recall:

- the **absolute** recall of all three Random-Forest arms — Rule, placebo, and
  the shared real-only reference — reported per attack (Gear, RPM) and
  exact-macro;
- the Rule-minus-real and placebo-minus-real gains;
- the **paired Rule-minus-placebo difference**, with its mean, sample SD, and a
  Student-`t` interval over the twenty realizations (`n = 20`, `df = 19`).

This absolute form is the form used in any claim sentence.

**Descriptive companion (ratio).** The Rule-gain-to-placebo-gain ratio at
`P = 1`, computed per realization and summarized by **median and interquartile
range** — not the mean — with the denominator distribution reported alongside.
Any realization whose placebo denominator is `< 0.01` is flagged, reported as
unstable, and excluded from the ratio summary only.

**Registered joint statement.** Neither scale may be reported alone. Whichever
branch the primary returns, the manuscript reports both together.

CNN planning values at `P = 1`, exact-macro, `n = 20` (already observed):
real `0.018387`, Rule gain mean `0.116764`, placebo gain mean `0.122685`
(SD `0.041077`).

### 12.2 (b) Margin-scale sensitivity — mandatory, dual-margin reporting

Because `delta` is inherited on an absolute scale (Section 9.1), the primary
verdict is reported **jointly** with a second, stricter margin scaled to the
Random Forest's own response.

**Definition.** Let

```text
G_placebo = mean over the 20 generator realizations of
            [ Recall_placebo,g(P=1, canonical, exact-macro)
              - Recall_real(P=1, canonical, exact-macro) ]

delta_strict = 0.15 * G_placebo
```

`G_placebo` is the placebo arm's own out-of-grammar gain over the real-only
reference, computed from E17 Random-Forest outcomes under the Section 8
aggregation order. The multiplier `0.15` is frozen here and is the same
multiplier used to construct the inherited `delta`.

**Registered reporting.** Report, in one table:

- the 90% Student-`t` interval for the mean of `theta_g` — the same interval as
  the primary, recomputed against nothing, merely re-compared;
- `delta` and its verdict (the primary verdict);
- `delta_strict`, its numeric value, and the verdict obtained by running the
  full Section 10 decision rule — heterogeneity screen included — with
  `delta_strict` substituted for `delta`;
- the ratio `delta_strict / delta`.

**Status.** `delta_strict` is **secondary only** and can never override, replace,
or annotate away the primary. It has a random denominator computed from E17
outcomes, which is exactly why it is disqualified from the primary and exactly
why it is informative as a sensitivity: it answers "is the equivalence verdict an
artifact of judging a possibly smaller Random-Forest response against a margin
sized on the CNN's response?"

**Registered handling of disagreement.** If the two margins return different
branches, **that disagreement is itself the reportable finding.** It is stated in
the results section and carried into the limitations. It is not a reason to
suppress either margin, to relabel the primary, to declare the sensitivity
invalid, or to declare the primary invalid. The registered sentence form is:

> Against the inherited CNN margin the Random-Forest replication returns
> Branch `<X>`; against a margin rescaled to the Random Forest's own
> out-of-grammar placebo gain it returns Branch `<Y>`. The verdict is therefore
> margin-scale dependent, and the equivalence claim is reported as conditional
> on the registered absolute margin.

with the observed branches substituted.

**Degenerate-denominator handling, registered now.** If `G_placebo <= 0.01` —
including the case `G_placebo <= 0` — then `delta_strict` is reported as
**undefined**, the observed `G_placebo` is published, and the reason is stated.
No substitute denominator, floor value, or alternative multiplier is invented
after the fact, and the primary verdict stands unannotated by a sensitivity that
could not be computed.

CNN planning scale: `0.15 x 0.122685 = 0.018403`, i.e. roughly `0.137 x delta`.
The Random-Forest value is unknown at registration time.

### 12.3 (c) The `P = 0` in-grammar cell — mandatory

**Registered reporting.** For each of the twenty generator realizations, at
`P = 0` (payload inside the Rule-grammar bytes), canonical strata, exact recall:
the absolute recall of the Rule arm, the placebo arm, and the real-only
reference, per attack and exact-macro, with means, sample SDs, and the paired
Rule-minus-placebo difference with a Student-`t` interval (`n = 20`, `df = 19`).

**Purpose, registered in advance.** For the CNN this cell is saturated in both
augmented arms and therefore cancels from `C_P`:

| CNN, `P = 0`, exact-macro, canonical | value |
|---|---|
| real-only | `0.001796` |
| Rule arm (mean over 20, SD) | `0.999694` (`0.000017`) |
| placebo arm (mean over 20, SD) | `0.999699` (`0.000022`) |
| Rule − placebo | `-0.000005` |

Publishing the Random-Forest `P = 0` values lets the reader verify directly
whether the same cancellation occurs, or whether the Random Forest is
unsaturated at `P = 0` — in which case `C_P` is **not** dominated by the `P = 1`
cell and the contrast must be read differently. The registered report must state
explicitly which of the two holds, using the criterion: the cell is declared
**saturated** if both augmented arms exceed `0.99` exact-macro recall and their
absolute difference is below `0.01`; otherwise it is declared **unsaturated**
and the `P = 0` contribution to `C_P` is reported separately from the `P = 1`
contribution.

### 12.4 Exact Rule-minus-placebo remaining effects

`S, D, PS, PD, SD, PSD`. Separate 6-test Holm family.

### 12.5 Binary companion

`P, S, D, PS, PD, SD, PSD` on binary attack recall, Rule-minus-placebo. Separate
mandatory 7-test Holm family.

### 12.6 Component means

`Delta^Placebo_g` (marginal-exposure component) and `Delta^Rule_g` (total
response), each as a mean with sample SD and Student-`t` interval over the twenty
generator realizations. `Delta^Rule_g` is reported as a Random-Forest
replication of the response-surface result, not as a new claim.

### 12.7 CNN/Random-Forest family comparison — descriptive only

Because E17 reuses E16's exact twenty units, the two `theta_g` vectors are
paired. Report: the twenty paired `(theta_g^CNN, theta_g^RF)` values, their
Pearson and Spearman correlation, the paired difference
`theta_g^RF - theta_g^CNN` with mean and sample SD, and a scatter plot.

**Registered status: descriptive only, no test, no verdict.** Two detector
families are two points, not a sample. No p-value, confidence interval, or
equivalence test on the family difference may be reported, and no statement of
the form "the effect does not differ across detector families" is permitted.
The permitted language is agreement or disagreement of the two registered
verdicts, plus the descriptive correlation.

### 12.8 Attack and stratum heterogeneity

Attack-specific `theta` values for Gear and RPM, and shifted-ID effect
modification. Descriptive.

### 12.9 Bridge sensitivity

The four bridge realizations, both arms, reported separately and always labelled
as a bridge sensitivity attached to the E15 lineage. Never pooled with the
primary twenty, never used to change `n` or `df`.

### 12.10 Finite-grid dispersion

Realization, pipeline, and interaction shares, descriptive only, explicitly
labelled as finite-grid dispersion and not random-effects variance components.

## 13. Registered branches and permitted claims

Every branch's reported form must include: the Section 12.2 dual-margin verdict,
the Section 12.1 `P = 1` absolute and ratio scales, the Section 12.3 `P = 0`
saturation determination, and the statement that the Random-Forest arms carry no
optimizer-update budget, early-stopping trajectory, or minibatch order, so the
comparison is made under an identical-training-row contract rather than a matched
training schedule (Section 5.2).

### Branch A — replicated marginal sufficiency

Condition: Step 2 of Section 10, heterogeneity screen not triggered.

Permitted headline:

> The payload-destination response survives destruction of the implemented Rule
> relations in a second detector family that shares no training dynamics with
> the CNN; matched marginal exposure accounts for the response in a Random
> Forest as it does in the CNN, across the same twenty prospective generator
> realizations.

Forbidden extensions: relational structure is irrelevant in general; the result
holds for detectors in general or for any untested family; synthetic realism is
unnecessary; the CNN result is thereby confirmed as a population fact; the E16
primary is strengthened into a claim it did not register.

### Branch B — detector-family-dependent relational increment

Condition: Step 3 of Section 10, heterogeneity screen not triggered.

Permitted headline:

> A Random Forest recovers a relational increment beyond matched marginals that
> the CNN did not, under identical pools, identical training rows, and the same
> registered margin. The relational increment is therefore detector-family
> dependent in the implemented Rule grammar.

This is reported as a constraint on the scope of the E16 headline, not as a
refutation of the E16 registered primary, which stands as registered for the CNN.

Forbidden extensions: physical mechanism, realism, real attack semantics,
cross-domain transfer, or any claim that the CNN result was wrong rather than
family-specific.

### Branch C — realization-contingent relation

Condition: Step 1 of Section 10.

Permitted headline:

> In the Random-Forest family, relational utility is realization-contingent even
> under matched marginals and identical training rows.

Reported as a methodological result requiring generator realization to enter the
inference axis, not as a failed replication.

### Branch D — inconclusive

Condition: Step 4 of Section 10, or a Student-`t` / bootstrap disagreement.

The planning assumptions of Section 9 and the observed dispersion are published,
the E16 CNN registered primary is retained unchanged and is **not** restated as
weakened, and no Random-Forest equivalence or difference claim is made. Title
and abstract are not rewritten. A Branch D is reported as a completed
replication attempt with an inconclusive outcome, which is a result.

## 14. Required outputs and completeness

### 14.1 Implementation

```text
journal/scripts/run_e17_rf_paired_training.py
journal/scripts/evaluate_e17_rf_relational_response.py
journal/scripts/analyze_e17_rf_relational_response.py
journal/scripts/make_e17_rf_relational_response_figure.py
journal/tests/test_e17_rf_relational_replication.py
```

The runner imports `rf_features`, the estimator construction, the permutation
convention, and the no-clobber preflight from
`journal/scripts/train_generator_extension_rf.py` rather than re-implementing
them (gate G14). The evaluator reuses the E16 evaluation-window construction,
factorial cell definitions, and contrast normalization from
`journal/scripts/evaluate_e16_relational_response.py`. Scoring runs on CPU.

Tests run on toy bases and must not compute any E17 outcome or write any
canonical checkpoint or score.

### 14.2 Canonical outputs

```text
journal/results/tables/
  e17_pool_inheritance_audit_v1.csv
  e17_sampling_audit_v1.csv
  e17_training_manifest_v1.csv
  e17_by_cell_v1.csv
  e17_by_scenario_v1.csv
  e17_augmentation_delta_v1.csv
  e17_decomposition_by_realization_v1.csv
  e17_primary_theta_summary_v1.csv
  e17_p1_cell_comparison_v1.csv
  e17_p0_cell_comparison_v1.csv
  e17_margin_sensitivity_v1.csv
  e17_secondary_effects_v1.csv
  e17_cnn_rf_family_comparison_v1.csv
  e17_crossed_dispersion_v1.csv
  e17_bridge_sensitivity_v1.csv
  e17_real_only_reference_continuity_v1.csv
journal/results/logs/
  e17_rf_relational_replication_v1.log
  e17_rf_relational_replication_artifact_manifest_v1.json
journal/experiments/e17_rf_relational_replication/
  {pool_inheritance,training_preflight,training_run,prepare,run}_v1.json
```

Completeness requirements: 24 realization pools verified by hash; 245 fits
complete; expected row counts verified; every output SHA-256 recorded in the
artifact manifest with its source commit; a single machine-readable `verdict`
field carrying one of `A`, `B`, `C`, `D`; a second machine-readable
`verdict_strict_margin` field carrying `A`, `B`, `C`, `D`, or `undefined`; and a
boolean `margin_disagreement` field. Every new table and figure is registered in
`journal/results/paper_artifacts_manifest.md`.

## 15. Technical gates and stopping rules

### 15.1 Freeze gate (S0)

- **G0** — `git status --short --branch` clean for the E17 target paths;
  `git status --short -- wisa` empty; all E17 target paths absent; all frozen
  input SHA-256 recorded.

### 15.2 Inherited pool gates (S2)

E17 generates no pool. The E16 pool gates **G1--G9** — aligned-pair identity,
seed integrity, marginal identity, non-target channel identity, Gear off-bin
relation destruction, RPM structural-minimum relation destruction, DoS/Fuzzy bit
identity, E15 lineage reproduction, provenance continuity, and draw integrity —
already passed at E16 stage S2 and are **inherited, not re-run**. They are
re-verified by hash:

- **G1i pool hash inheritance (blocking)** — for every consumed pool, the
  recomputed `x_digest` equals the value frozen in
  `journal/results/tables/e16_pool_audit_v2.csv` for that realization and arm.
  Any mismatch halts S2 for all realizations.
- **G2i seed integrity** — the realization seed list read from
  `generate_e16_rule_placebo_pairs.py` equals Section 3.1 exactly; the bridge
  list equals Section 3.2 exactly; no seed is added, removed, or reordered.
- **G3i--G8i manipulation-check inheritance** — the frozen
  `journal/results/tables/e16_manipulation_checks_v2.csv` is read and every abort
  gate recorded there is confirmed `passed`. The Gear off-bin criterion and the
  RPM structural-minimum criterion (E16 amendment v2) are inherited in this way.
  The manipulation checks are **not** recomputed, because the pools are
  bit-identical by G1i and recomputation could only produce the same values at
  the cost of implying a new measurement.
- **G9i draw integrity** — per fit, per class,
  `requested = drawn = unique`, `repeated = 0`, per-class request below 65,000.

### 15.3 Training, continuity, and analysis gates

- **G10a no-clobber** — every E17 target checkpoint, table, log, and record path
  is absent at preflight and written atomically. No legacy `rf_*` checkpoint, no
  E10/E14/E15/E16 artifact, and nothing under `wisa/` or `wisa_camera_ready/` is
  opened for writing.
- **G10c-i paired draw identity (blocking)** — within each realization and
  pipeline seed, Rule and placebo arms share one `sampling_index_sha256`. 120
  pairs checked.
- **G10c-ii cross-family draw inheritance (blocking)** — the E17
  `sampling_index_sha256` equals the frozen E16 CNN value in
  `e16_training_manifest_v2.csv` for the same realization, arm, and pipeline
  seed. 220 fits checked; the 20 bridge Rule fits are recorded as exempt with
  their reason (Section 5.2).
- **G10d fit completeness** — 245 target checkpoints absent at preflight and
  present at completion; per-fit pool path, pool SHA-256, sampling index
  SHA-256, feature dimension, estimator hyperparameters, permutation seed,
  training row count, elapsed time, and device recorded.
- **G11 decomposition closure** — the Section 1.1 identity holds to
  floating-point closure (`|residual| <= 1e-12`) for every generator realization
  and pipeline seed.
- **G12 real-only reference continuity** — the five shared E17 real-only
  Random Forests are scored **first**, before any Rule or placebo checkpoint is
  scored; their per-cell integer counts are frozen into
  `e17_real_only_reference_continuity_v1.csv` and re-verified byte-identically
  after the full grid has been scored. Any drift halts S6. Scoring is performed
  on CPU. This gate is blocking and mirrors E16's G12 role: it establishes that
  the shared reference is stable across the scoring session before any
  augmented arm is compared against it.
- **G13 output integrity** — atomic no-clobber publication, expected row counts,
  20 x 2 x 5 primary completeness, 4 x 2 x 5 bridge completeness, 5 reference
  fits, all SHA-256 recorded, `paper_artifacts_manifest.md` updated.
- **G14 Random-Forest contract fidelity** — the realized feature dimension is
  exactly 55; features are computed on raw un-standardized windows; the
  estimator's `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`,
  and `random_state` equal the Section 5.1 values for every fit; no standardizer
  object is constructed; the runner's feature function is the imported
  `train_generator_extension_rf.rf_features` object, asserted by identity.
- **G15 fit determinism** — one Rule fit and one placebo fit are repeated in a
  fresh child interpreter and must produce byte-identical `joblib` payloads and
  identical predictions on the evaluation set. This replaces the CNN's
  deterministic-runtime-flag record, which has no Random-Forest analogue.
- **G16 dual-margin reporting completeness** — the analysis stage must emit both
  `verdict` and `verdict_strict_margin` (or `undefined` with a published
  `G_placebo`), the `margin_disagreement` boolean, the `P = 1` absolute table,
  the `P = 1` ratio summary, and the `P = 0` saturation determination. Missing
  any one of these is a gate failure, not an omission: it is the mechanism that
  makes Section 12.2's anti-suppression rule enforceable.

### 15.4 Failure, amendment, and exclusion policy

A **technical stop** (any gate failure) halts the stage. The cause and the fix
are committed as a dated prospective amendment with a new output version
identity before rerunning. Failed artifacts are preserved as failure provenance
and are not deleted. No outcome from a gate-failed stage is interpreted. No stage
is resumed in place.

A **valid scientific result** — including Branch D, sign heterogeneity, a null
relational increment, a margin-scale disagreement, or disagreement with the E16
CNN verdict — is never hidden, rerun, reclassified as a technical stop, or
resolved by rerunning with different settings.

If a generator realization cannot complete for a technical reason after
amendment, it is reported as missing with its cause, the primary is recomputed on
the completed realizations with the reduced `df` disclosed, and the reduction is
reported in the abstract-level limitations. Realizations are never dropped for
their outcome values.

## 16. Disclosed limitations

- All twenty `Delta` pairs at a given pipeline seed share one Random-Forest
  real-only reference, so realization-level estimates are conditional on those
  five shared references.
- Twenty generator realizations sample realizations of one disclosed Rule
  construction law; they do not sample a population of generators.
- Two detector families are two points; the family axis is fixed, not sampled,
  and E17 licenses no detector-population claim (Section 12.7).
- The inherited absolute margin may be relatively more permissive at
  Random-Forest recall levels than it was at CNN recall levels. This is a known,
  registered consequence of refusing a circular re-derivation, and is the reason
  Section 12.2 is mandatory.
- The Random-Forest feature map is a fixed 55-dimensional per-channel summary.
  A relational statistic not expressible through per-channel mean, standard
  deviation, minimum, maximum, or first-to-last difference is invisible to this
  detector by construction. A Random-Forest null is therefore partly a statement
  about this feature map.
- The evaluation split is source-local; E17 inherits this limitation from E8.
- The `S1` mask span of 96 exceeds the training maximum of 90; this known
  boundary condition is inherited from E14 and is not re-litigated here.
- Exact-macro (equal) weighting of Gear and RPM is a registered choice, not a
  property of the data; attack-specific values are reported as Section 12.8.
- The placebo destroys the *implemented* Gear/RPM relation only. A relational
  effect carried by some unimplemented statistic would not be detected.

## 17. What this registration forbids

- changing `delta`, re-deriving it from any Random-Forest outcome, rescaling it,
  or swapping it for `delta_strict` as the primary margin
- changing the sample size, the seed list, the bridge/primary partition, the
  aggregation order, the decision order, the bootstrap resample count or seed, or
  any family membership after seeing an outcome
- promoting the bridge realizations into the primary, or counting them in `n` or
  `df`
- counting the 200 primary fits, the 100 primary pairs, the three blocks, the
  five pipeline seeds, or raw windows as independent inferential units
- treating the detector family as a random effect, testing the CNN-versus-RF
  difference, or claiming that `theta` is invariant across detectors
- promoting Branch D to Branch A or Branch B
- suppressing either margin in Section 12.2, or reporting a margin-scale
  disagreement as anything other than a finding
- reporting the `P = 1` ratio without the absolute scale, or the absolute scale
  without the ratio
- omitting the `P = 0` cell values or the saturation determination
- interpreting practical equivalence as evidence that relational structure is
  generally irrelevant, or as evidence about any untested detector family
- re-opening, revising, re-scoring, or restating the E16 CNN registered primary
  as a function of an E17 outcome
- tuning any Random-Forest hyperparameter, feature, or sampling choice on an E17
  outcome
- adding external datasets, generator families, or further detector families to
  E17
- overwriting E10, E14, E15, E16, legacy RF, or WISA artifacts, or modifying
  `wisa/` or `wisa_camera_ready/`
