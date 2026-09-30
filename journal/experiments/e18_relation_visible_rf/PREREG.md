# E18 preregistration — relational decomposition under a feature map that can represent cross-byte joint structure

Registration time: 2026-08-07T09:12:44Z

Registration source commit: `93dbdc4`

Status: **PRE-RESULT. E18 implementation, gate-G17 measurement, Random-Forest
fitting, scoring, and analysis are prohibited until this document is committed
and the freeze gates in Section 15 pass.**

At registration time no E18 feature-visibility measurement, Random-Forest
checkpoint, training log, score, outcome table, figure, or verdict existed, and
no Random Forest had ever been fit on the 83-dimensional feature map of Section
5.1 on any pool, real or synthetic. Verified at the registration commit:

```text
journal/experiments/e18_relation_visible_rf/*_v1.json      0 files
journal/datasets/synthetic/e18_*                           0 files
journal/models/generator_extension/*e18*                   0 files
journal/results/tables/e18_*                               0 files
journal/results/logs/e18_*                                 0 files
journal/scripts/*e18*                                      0 files
```

The E16 CNN outcomes and the E17 Random-Forest outcomes are already known and
are quoted throughout this document as **planning inputs only**. This
registration is therefore prospective with respect to every quantity E18 names,
including the gate-G17 visibility measurement of Section 15.5, which has never
been computed for the 83-dimensional map on any pool pair. Any later design
change requires a dated amendment committed before the affected stage; it must
not silently edit this record.

**Freeze intent.** This document is frozen on commit. Sections 3, 5.1, 5.2, 7,
8, 9, 10, 12, 13, 15.5, and 15 are the registered decision surface. The feature
map of Section 5.1, the G17 floor of Section 15.5, the margin of Section 9, the
aggregation order of Section 8, the decision order of Section 10, the
inferential unit of Section 8.3, the bootstrap resample count and seed of
Section 10, and the mandatory secondary reporting of Section 12 may not be
altered after any E18 outcome — including any G17 number — exists.

### Vocabulary

This registration says **generator realization** for what the E16/E17 code and
CSV schemas call a `construction_seed`. The legacy column name
`construction_seed` is retained verbatim inside CSV and NPZ artifacts for schema
continuity; it is never used in reader-visible prose. The five repeated model
runs are **paired pipeline seeds**. `C_P(Placebo - Real)` is the
**marginal-exposure component** and `C_P(Rule - Placebo)` is the **relational
increment**. The 55-dimensional per-channel summary map of E17 is the
**marginal feature map**; the 83-dimensional map registered here is the
**joint-structure feature map**.

## 1. Question, motivating fact, and the reclassification of E17

### 1.1 The motivating fact: E17's detector could not perceive the manipulation

E17 fit a Random Forest on `rf_features`
(`journal/scripts/train_generator_extension_rf.py:47`): per raw channel, over
the 128-frame axis, `mean`, `std`, `min`, `max`, and
`delta = x[:, -1, :] - x[:, 0, :]`, for each of 11 raw channels — 55 features.

The E16/E17 placebo twin destroys the implemented Gear/RPM inter-byte relation
by a **constrained within-window permutation of `data1` on injected spoof frames
only, preserving that window's `data1` multiset** and leaving every other
channel bit-identical
(`journal/scripts/generate_e16_rule_placebo_pairs.py`, gates G1b, G1c, G3).

These two facts are incompatible in a specific and provable way. `mean`, `std`,
`min`, and `max` of a channel are **invariant under any permutation of that
channel's values within the window**, and the other ten channels are untouched.
Of the 55 features, therefore, at most one — `delta(data1)` — can ever respond
to the manipulation, and it responds only when the permutation happens to move
the value at frame 0 or frame 127.

This was measured on the frozen pool pair
`e16_rule_cseed192753_windows_v2.npz` / `e16_placebo_cseed192753_windows_v2.npz`
(260,000 windows, of which 130,000 differ in raw `x`):

| measured quantity | value |
|---|---|
| features of 55 that ever differ | **1** (`delta(data1)`) |
| features of 55 bit-identical across all 260,000 windows | **54** |
| raw-differing windows producing any feature-vector difference | **3,303 / 130,000 = 2.54%** |

The E17 detector could perceive **2.54%** of the manipulation it was asked to
respond to. Corroborating this from the outcome side: E17's `P = 1` per-attack
Rule-minus-placebo difference was `+0.000087` for **Gear** — the attack whose
inverse relation `|255 - data0 - data1| <= 8` is the cleanest and most
completely destroyed manipulation in the design — against `+0.008878` for RPM.

### 1.2 Reclassification of E17

E17 is therefore reclassified, prospectively and in this document, as a
**negative control**:

> E17 confirms that the frozen E16/E17 pipeline — pools, draws, evaluation
> construction, aggregation order, decision rule — manufactures **no spurious
> relational increment** when the detector is provably almost blind to the
> manipulation. Its Branch A verdict is a statement about the *pipeline* and
> about *that feature map*. It is **not** evidence that the implemented Gear/RPM
> relations are irrelevant to a detector that can represent them.

E17's registered outputs, verdict, and text are **preserved verbatim**. Nothing
in E17 is rerun, revised, rescored, deleted, or annotated away. The
reclassification is an addition to the interpretation, made here, before any E18
outcome exists.

### 1.3 The E18 question

> Does a detector whose representation **can** express within-window cross-byte
> joint structure respond to the destruction of the implemented Gear/RPM
> inter-byte relation, on the same pools, the same draws, the same endpoint, the
> same aggregation order, the same inferential units, and the same margin that
> returned Branch A in E16 and E17?

E18 changes **exactly one thing**: the detector's feature map. Every other
element of the design is inherited verbatim (Section 4).

### 1.4 Decomposition identity

For generator realization `g` and pipeline seed `p`, let `C_P{·}` denote the
registered `P` factorial contrast on the primary endpoint (Section 8), and let

```text
Delta^Rule_{g,p}     = C_P{ Y_Rule,g,p    - Y_real,p }
Delta^Placebo_{g,p}  = C_P{ Y_Placebo,g,p - Y_real,p }
```

where `Y_real,p` is the shared E18 real-only reference at pipeline seed `p`.
Because `C_P{·}` is linear and the real reference is shared within pipeline seed
`p`, the following identity holds exactly and is not an approximation:

```text
C_P(Rule - Real)  =  C_P(Placebo - Real)  +  C_P(Rule - Placebo)
      total       =   marginal-exposure   +      relational
                          component            increment
```

E18 estimates all three terms on the same twenty generator realizations that E16
and E17 used. The identity is verified numerically to floating-point closure as
a gate (Section 15.3, G11); it is not assumed.

### 1.5 What E18 cannot establish

All conclusions are conditional on the frozen Car-Hacking train/validation/test
split, the disclosed 65,000-window-per-class Rule construction law, the twenty
registered generator realizations, the five paired pipeline seeds, the E14
evaluation constructions and three descriptive blocks, the frozen E8 evaluation
manifest, and the single Random-Forest configuration and feature map of Section
5.1.

Specifically, E18 does **not**:

- make a **positive** result a statement about deployed detectors. A material
  relational increment in E18 would show that the implemented relation is
  **representable-and-used by this 83-dimensional feature map under this
  estimator**. It would *not* show that any particular deployed CAN intrusion
  detector uses it, that a detector must use it, or that detectors in the wild
  have access to a comparable representation. The permitted claim names the
  feature map.
- make a **negative** result a statement that the relation is irrelevant. A null
  under a map that G17 shows *can* see the manipulation is stronger than E17's
  null, but it remains conditional on this estimator, these pools, and this
  endpoint.
- make `theta` a **detector-population** result. Three feature maps are three
  points, not a sample. The inferential unit remains the generator realization;
  the detector representation is a fixed, registered factor and is never treated
  as random.
- establish synthetic-data **realism**, physical payload causation, or
  **replacement** of real attack data.
- establish **cross-dataset or cross-domain robustness**. E18 contains no
  external-dataset evaluation and adds no new evaluation construction.
- establish universal generator behavior, or behavior of grammars outside the
  implemented Rule law.
- **re-open, replace, adjudicate, or supersede the E16 CNN registered primary
  result.** E16 remains the registered primary of the manuscript regardless of
  the E18 outcome. E18 is a registered representational-sensitivity experiment
  reported alongside it, and E17 as its negative control.
- establish anything about the **CNN's** representation. The CNN is a convolution
  over the raw window and is not subject to the E17 invariance argument; whether
  it does or does not use the relation is an E16 question that E18 does not
  re-open.

### 1.6 Why this is a real test and not a rigged one

The obvious failure mode of "make the detector able to see the manipulation" is
to hand it a **bespoke Gear/RPM relation detector** — for example a feature
`mean_t |255 - data0_t - data1_t|`, or the count of frames violating the Rule
inequality. Such a feature would trivially separate Rule from placebo, and the
resulting "finding" would be a tautology: the experiment would have measured its
own construction.

E18 refuses that. The registered addition is the **28 within-window Pearson
correlations between every unordered pair of the eight payload bytes**
`data0..data7`. This set is:

- **generic** — it is the standard second-order description of a multivariate
  time window; it mentions no threshold, no inequality, no modulus, no bin
  width, no attack, and no channel pair privileged over another;
- **symmetric** — all 28 pairs enter, not only `(data0, data1)`; 21 of the 28
  cannot respond to the manipulation at all, and are carried anyway;
- **not tuned** — it is fixed here, before any E18 outcome, and is not selected
  from a set of candidate maps by its result.

The question E18 therefore asks is *"does a detector able to represent cross-byte
joint structure respond to the manipulation?"*, not *"does a bespoke Gear
detector detect Gear?"*.

The test can genuinely fail in both directions. It returns Branch A if the
relational structure, though now visible in the representation, carries no
usable discriminative signal beyond the matched marginals — a substantive and
publishable strengthening of E16. It returns Branch B if it does — a substantive
constraint on the scope of the E16 headline. Both outcomes are informative and
must be reported as positive findings in the registered form of Section 13.

### 1.7 Relation to the E16 registered primary and to E17

E18 is subordinate in the reporting hierarchy and independent in its inference.
It shares the E16 pools, the E16 evaluation material, the E16 endpoint, the E16
aggregation order, the E16 margin, and the E16/E17 draws. It does not share the
E16 detector, the E16 or E17 checkpoints, or the E16 or E17 real-only
references. No E18 gate, outcome, or verdict may modify an E16 or E17 artifact,
and no E16 or E17 conclusion is restated as a function of an E18 outcome beyond
the E17 reclassification registered in Section 1.2, which is made here and not
after the fact.

## 2. Mandatory execution order

Stages must run in this order. A later stage must not begin until the earlier
stage's gates have passed and its run record has been committed.

1. **S0 — freeze.** Section 15.1 gate G0; commit this document as a single-file
   commit, before any E18 runner, evaluator, or analyzer exists.
2. **S1 — implementation.** Scripts (Section 14.1) committed with no E18
   outcome, G17 number, checkpoint, or score present.
3. **S2 — pool inheritance verification.** Every E16 pool consumed by E18 is
   re-hashed and matched against the frozen `e16_pool_audit_v2.csv` record (and,
   for the four bridge Rule pools, `e15_rule_construction_pool_audit_v2.csv`).
   Gates G1i--G9i (Section 15.2). No fitting occurs in this stage.
4. **S2b — manipulation-visibility gate G17.** Section 15.5. Computed on the 24
   pool pairs, **before any fit**. Its table is published **regardless of
   outcome**. If the registered floor fails, E18 stops here and is reported as
   inconclusive-by-construction; stages S3--S7 are not run and no inferential
   claim is made.
5. **S3 — shared real-only references.** Five fits, one per pipeline seed. Gates
   G10a, G14, G15.
6. **S4 — paired fits.** 240 fits. Gates G10a, G10c-i, G10c-ii, G10d, G14, G15.
7. **S5 — real-only reference continuity.** Gate G12, before any E18 Rule or
   placebo checkpoint is scored.
8. **S6 — scoring.** Exact and binary scored separately.
9. **S7 — registered analysis and verdict.** Gates G11, G13, G16.

No stage may be resumed in place after an abort. Retry requires a dated
amendment and a new output version identity (Section 15.4).

## 3. Frozen seed axes and roles

### 3.1 Generator-realization seeds

E18 reuses the E16/E17 registered draw **verbatim**. The seeds are not redrawn,
and no meta-seed draw is re-executed. They are read from
`journal/scripts/generate_e16_rule_placebo_pairs.py` as `CONSTRUCTION_SEEDS`:

```text
415637  748203  560657  459336  457191  999191  715822  298581  922089  864178
583909  467232  550427  485791  820632  837943  222537  685012  192753  739263
```

These twenty are the **primary inferential units** (`n = 20`, `df = 19`). No
seed may be added, removed, reordered, or substituted after registration. If a
realization fails a gate, Section 15.4 applies; the seed is not silently
replaced.

Reusing the exact twenty units is a registered design choice, not a convenience.
It makes the CNN, the E17 marginal-map Random Forest, and the E18
joint-structure Random Forest **paired at the level of the inferential unit**,
which is what licenses the descriptive comparisons of Section 12.7.

### 3.2 Bridge realizations — included, sensitivity only

The four E15 bridge realizations are read from the same module as
`BRIDGE_SEEDS`:

```text
271828  161803  141421  173205
```

They **are** included in E18, and they are **a bridge sensitivity only, never
primary**. They are excluded from `n`, from `df`, from the primary interval,
from the heterogeneity screen, from the margin-scale sensitivity, and from every
Section 12 family except Section 12.9. They are reported in a separate table and
always labelled as a bridge sensitivity attached to the E15 lineage. Both arms
are fit, exactly as in E17, and this does not promote them.

The bridge Rule arm consumes the frozen E15 v2 pools
`journal/datasets/synthetic/rule_cseed<g>_windows_v2.npz`; the bridge placebo arm
consumes `journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz`.

### 3.3 Placebo pairing

E18 derives no placebo seed and generates no pool. Each realization's placebo
pool is the E16 published aligned twin, consumed by path and verified by
SHA-256 (Section 4.1). The E16 placebo seed derivation, the Gear off-bin
rotation, and the RPM structural-minimum rotation (E16 amendment
`AMENDMENT_2026-07-25_V2_RPM_STRUCTURAL_MINIMUM.md`) are inherited unchanged.

### 3.4 Detector-pipeline seeds

Frozen and identical to E14/E15/E16/E17:

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

E18 generates **no** synthetic pool. It consumes, read-only:

```text
journal/datasets/synthetic/e16_rule_cseed<g>_windows_v2.npz        (20 primary)
journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz     (20 primary)
journal/datasets/synthetic/rule_cseed<g>_windows_v2.npz            ( 4 bridge)
journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz     ( 4 bridge)
```

Every consumed pool's `x_digest` must equal the value frozen in
`journal/results/tables/e16_pool_audit_v2.csv` for that realization and arm; the
four bridge Rule pools, which E16 never published, are matched against the
frozen E15 v2 audit's whole-file digest, exactly as E17 did under its amendment
v1. This is blocking gate G1i (Section 15.2). No pool file is opened for writing
at any point in E18.

### 4.2 Data and evaluation inputs

Reused unchanged, with SHA-256 recorded at S0 and re-verified before each
subsequent stage: the Car-Hacking train/test window NPZs, the E8 evaluation
realization manifest, the E14 `2x2x2` factorial transformation code path, and
the four ID strata (Gear canonical `0x43F`, Gear shifted `0x440`, RPM canonical
`0x316`, RPM shifted `0x329`).

### 4.3 Paired synthetic row draws — inherited, not redrawn

E18 reuses the **same paired synthetic row draws** as E16 and E17. Operationally
this is the identical-training-row contract of Section 5.2: the E18
`sampling_index_sha256` for every `(realization, arm, pipeline seed)` must equal
the frozen E16 value. E18 does not re-derive, re-seed, or re-order the draw.

### 4.4 Shared real-only references

E18 fits **five new** shared real-only Random Forests, one per pipeline seed:

```text
journal/models/generator_extension/rf_real_only_e18_v1_seed{7,42,123,2026,3407}.joblib
```

They are new because the reference must be fit on the **same 83-dimensional
feature map** as the augmented arms; the E17 `e17_v1` references were fit on 55
features and are dimensionally incompatible. The E17 and legacy references are
**not** read, modified, or overwritten by E18. The `e18_v1` model tag keeps the
new checkpoints on disjoint paths (gate G10a).

Every `Delta^Rule_{g,p}` and `Delta^Placebo_{g,p}` at the same `p` shares one
real reference. This correlation structure is preserved by the pairing and is
disclosed as a limitation in Section 16.

### 4.5 Frozen evaluation material

The E14 eight factorial cells, three frame-disjoint blocks, four strata, and the
`S0 = start + 2*arange(32)` / `S1 = start + 3*arange(32)` mask schedules are
frozen evaluation material. E18 introduces no new evaluation construction, no
new generator family, no new pool, and no external dataset. The only new axis is
the feature map.

## 5. Detector contract

### 5.1 The joint-structure feature map — fixed now, not tuned

**Block 1 — the inherited marginal map (55).** `rf_features` at
`journal/scripts/train_generator_extension_rf.py:47`, imported and not
re-implemented, computed on **raw, un-standardized** windows. Per channel, over
the 128-frame axis: `mean`, `std`, `min`, `max`, and
`delta = x[:, -1, :] - x[:, 0, :]`, concatenated to 55 `float32` values per
window (11 channels × 5 statistics). Channel order is
`lib_common.FEATURE_NAMES = [can_id, dlc, data0..data7, delta_t]`.

**Block 2 — within-window payload-byte correlations (28).** For every unordered
pair `(i, j)`, `i < j`, of the eight payload channels `data0..data7` (raw
channel indices 2..9), the within-window Pearson correlation over the 128
frames:

```text
r_ij = cov(x[:, i], x[:, j]) / ( sd(x[:, i]) * sd(x[:, j]) )
```

with the population (biased, `ddof = 0`) covariance and standard deviation over
the 128 frames, computed on **raw, un-standardized** windows in `float64` and
cast to `float32` for the feature vector.

**Registered constant-channel convention.** If either channel of a pair has
`sd = 0` within the window — the correlation is then mathematically undefined —
the feature is defined as **`0.0`**. This convention is registered here, before
any measurement, because it is not derivable and because a different convention
(`NaN`, imputation, a sentinel) would change the estimator's behavior. `0.0` is
chosen as the value that encodes "no linear association was observable", which
is the honest reading of a degenerate window, and because it keeps the feature
matrix finite and `NaN`-free without imputation.

**Pair enumeration order**, frozen:
`(0,1), (0,2), ..., (0,7), (1,2), ..., (1,7), (2,3), ..., (6,7)` — `itertools`
lexicographic order over `data` indices, 28 pairs.

**Total dimension: `55 + 28 = 83`**, `float32`, concatenated marginal block
first.

**Registered anti-tuning statement.** This feature set is fixed by this
document. It was **not** selected against any E18 outcome, was not chosen from a
screened set of candidate maps, and no alternative map was evaluated on any E18
quantity before registration. It contains no Gear-specific and no RPM-specific
term: 21 of the 28 correlation features involve neither `data1` nor its partner
channels in a privileged way and cannot respond to the registered manipulation
at all. Adding, removing, reweighting, standardizing, selecting, or reordering
features after registration is prohibited by Section 17.

**Analytic expectation, registered as a prediction.** Because the placebo
manipulation touches only `data1`, and only by a within-window permutation on
injected frames that preserves the window's `data1` multiset, exactly **8** of
the 83 features are analytically capable of differing between the arms:
`delta(data1)`, and the 7 correlations `r(data1, data_k)` for
`k in {0, 2, 3, 4, 5, 6, 7}`. The remaining 75 are provably invariant. Gate G17
measures the realized count and compares it against this prediction; a realized
count above 8 would indicate an implementation error and is a technical stop.

### 5.2 Estimator — frozen verbatim from E17

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
a realization because the two pools are array-aligned and equal in length. No
hyperparameter is tuned, searched, or selected on any E18 outcome. `n_jobs=-1`
is a resource knob, not a model choice; fitting is CPU-only.

**No standardizer.** As in E17, the arms fit on raw window statistics; no
standardizer is derived, fitted, or applied. Gate G14 enforces this by
monkey-patching `lib_common.fit_standardizer` to raise inside every child fit.

**Setting names.** `rule_0p30` and `placebo_0p30`, unchanged. Synthetic
consumption is the `+30%` no-replacement strict-v2 draw.

### 5.3 The identical-training-row contract (blocking)

**Registered requirement.** For every generator realization `g`, arm `a`, and
pipeline seed `p`, the E18 fit must consume **exactly the same synthetic row
indices** as the corresponding E16 CNN fit and E17 Random-Forest fit.
Operationally, the E18 training manifest's `sampling_index_sha256` for
`(g, a, p)` must equal the value recorded for `(g, a, p)` in
`journal/results/tables/e16_training_manifest_v2.csv`.

Enforced as two blocking gates:

- **G10c-i — internal pairing.** Within each realization and pipeline seed, the
  Rule and placebo arms drew identical synthetic row indices. Applies to all 24
  realizations, 120 pairs.
- **G10c-ii — cross-family inheritance.** The E18 `sampling_index_sha256` equals
  the frozen E16 CNN value for the same `(realization, arm, pipeline seed)`.
  Applies to the 200 primary fits and the 20 bridge placebo fits. The 20 bridge
  Rule fits have **no** E16 CNN counterpart — E16 fit only the placebo arm for
  bridge realizations — and are therefore exempt from G10c-ii and covered by
  G10c-i alone. This exemption is recorded explicitly in the training run
  record; it is never silently waived.

Additionally, per fit, `requested = drawn = unique` and `repeated = 0` are
verified, with the per-class request below 65,000 (gate G9i).

Because a Random Forest has no optimizer-update budget, no early-stopping
trajectory, no minibatch order, no learning-rate schedule, and no
validation-based checkpoint selection, the fitted ensemble is a deterministic
function of the training rows, the row permutation, and `random_state`. Any
Rule-minus-placebo difference observed in E18 therefore cannot be attributed to
training dynamics, because there are none to attribute it to. Combined with the
identical-training-row contract, the **only** difference between the two arms of
a realization is the `data1` permutation itself, read through the Section 5.1
feature map.

### 5.4 Naming and atomic publication

```text
journal/models/generator_extension/
  rf_rule_0p30_cseed<g>_e18_v1_seed<p>.joblib
  rf_placebo_0p30_cseed<g>_e18_v1_seed<p>.joblib
  rf_real_only_e18_v1_seed<p>.joblib
```

All outputs are written atomically with a no-clobber preflight (gate G10a). No
E10, E14, E15, E16, E17, legacy RF, or WISA artifact is overwritten, and `wisa/`
and `wisa_camera_ready/` are not modified.

## 6. Fit grid and counts

| role | realizations | arms | pipeline seeds | fits |
|---|---|---|---|---|
| primary | 20 | rule, placebo | 5 | **200** |
| bridge sensitivity | 4 | rule, placebo | 5 | **40** |
| shared real-only reference | — | real_only | 5 | **5** |
| | | | **total** | **245** |

The 200 primary fits are 100 paired Rule/placebo comparisons. They are never
counted as 200 independent replicates.

## 7. Endpoints — inherited from E17 verbatim

**Primary endpoint.** Exact attack-type recall, canonical-ID strata, Gear and
RPM computed separately then equally weighted (exact-macro). Identical to E16
and E17.

**Mandatory companion.** Binary attack recall on the same fits.

Exact and binary are scored and reported separately and are never merged into a
single detection number.

## 8. Contrasts and aggregation order — inherited from E17 verbatim

### 8.1 Cell-level augmentation deltas

For generator realization `g`, pipeline seed `p`, block `b`, attack `a`, and
factorial cell `(P, S, D)`:

```text
A^Rule_{g,p,b,a}(P,S,D)    = Recall_rule(g,p,b,a,P,S,D)    - Recall_real(p,b,a,P,S,D)
A^Placebo_{g,p,b,a}(P,S,D) = Recall_placebo(g,p,b,a,P,S,D) - Recall_real(p,b,a,P,S,D)
```

where `Recall_real` is the E18 real-only reference at pipeline seed `p`.

### 8.2 Registered aggregation order

Fixed; must not be permuted; **identical to E17 section 8.2**:

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
Every new numeric result in E18 is reported as a mean over the twenty generator
realizations, with the sample standard deviation, and states `n = 20, df = 19`
explicitly. Blocks, cells, attacks, pipeline seeds, windows, and the 200 primary
fits are never counted as inferential units.

## 9. Equivalence margin — inherited from E17 verbatim

### 9.1 The margin is inherited, not re-derived

```text
delta = 0.133855625
```

This is the frozen E16 CNN registered margin, taken **verbatim** exactly as E17
took it, and used as a **fixed absolute margin on the same exact-macro
recall-contrast scale**. It is not recomputed, not rescaled to E18 recall
levels, and not re-derived from any E18 quantity. Its provenance is recorded
here for the reader but is not re-executed:

```text
delta = 0.15 x 0.8923708333333333
```

with the multiplier `0.15` and the denominator the published E15 v2
new-four-realization Rule-minus-real `P` mean on the primary endpoint
(`journal/results/tables/e15_l4_factorial_hierarchical_summary_v2.csv`,
`row_type = summary_new4`).

**Registered reason for inheriting rather than re-deriving.** Re-deriving the
margin from E18 outcomes would be circular, and it would destroy comparability:
E18 asks whether the E16/E17 verdict survives a change of representation, so it
must be judged against the same registered threshold on the same registered
scale. The known consequence — that an inherited absolute margin may be
relatively permissive at Random-Forest recall levels — is registered in advance
and accepted, and is exactly why the dual-margin reporting of Section 12.2 is
mandatory.

### 9.2 Planning values (already observed; quoted as planning inputs only)

| quantity | E16 CNN (`n = 20`) | E17 marginal-map RF (`n = 20`) |
|---|---|---|
| `theta` mean | `-0.005915208` | `+0.004490208` |
| `theta` SD | `0.039983478` | `0.026334111` |
| `theta` 90% TOST | `[-0.021375, +0.009544]` | `[-0.005692, +0.014672]` |
| `Delta^Rule` mean (SD) | `-0.881133333` (`0.040395291`) | `-0.917481458` (`0.026593880`) |
| `Delta^Placebo` mean (SD) | `-0.875218125` (`0.041077081`) | `-0.921971667` (`0.021010937`) |
| realizations outside `[-delta, +delta]` | `0 / 20` | `0 / 20` |
| verdict at `delta` | `A` | `A` |
| verdict at `delta_strict` | — | `C` |
| `P = 1` placebo gain (`G_placebo`) | `0.122685` | `0.077070` |
| `P = 1` Rule − placebo, Gear | — | `+0.000087` |
| `P = 1` Rule − placebo, RPM | — | `+0.008878` |
| `C_P` per-attack `theta`, Gear | — | `+0.000066` |
| `C_P` per-attack `theta`, RPM | — | `+0.008915` |

Planning value for the E18 `theta`: no point prediction is registered, because
the whole purpose of the experiment is that the answer is not derivable from
E17. A planning SD range of `0.02`--`0.12` is assumed, spanning the observed E17
dispersion at the low end and three times the CNN's at the high end.

### 9.3 Decision geometry and power

Under the Section 10 rule the verdict is a deterministic function of the sample
mean and SD:

```text
Branch A requires   |mean| + 0.386646 * s  <  0.133855625
Branch B requires   |mean| - 0.468014 * s  >  0.133855625
```

with `0.386646 = t(0.95, 19) / sqrt(20)` and `0.468014 = t(0.975, 19) / sqrt(20)`,
`t(0.95,19) = 1.729133`, `t(0.975,19) = 2.093024`.

| SD of `theta_g` | max `|mean|` for A | min `|mean|` for B |
|---|---|---|
| 0.020 | 0.126123 | 0.143216 |
| 0.040 | 0.118390 | 0.152576 |
| 0.060 | 0.110657 | 0.161936 |
| 0.080 | 0.102924 | 0.171297 |
| 0.120 | 0.087458 | 0.190017 |

These are the E17 geometry values, unchanged, because the margin, `n`, and the
decision rule are unchanged. The registered-rule probabilities of E17 section
9.3 carry over verbatim and are not recomputed.

`n = 20` is registered and is not a free parameter: it is fixed by the
requirement that E18 use E16's and E17's exact inferential units (Section 3.1).

## 10. Primary decision rule — inherited from E17 verbatim

Evaluated in this order. The order is registered and must not be changed after
seeing outcomes.

**Step 1 — heterogeneity screen.** Let `above` be the number of
realization-level `theta_g` greater than `+delta` and `below` the number less
than `-delta`. The registered headline is **Branch C** if either criterion
holds:

- **C1 — directional disagreement:** `above >= 3` **and** `below >= 3`.
- **C2 — aggregate/unit conflict:** `above + below >= 5` **and** the Step 2
  equivalence condition holds.

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

inherited from E17 verbatim, including the seed. If the bootstrap and Student-`t`
verdicts disagree, **the disagreement is reported and the verdict is downgraded
to Branch D.**

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
primary or be promoted to the headline after seeing outcomes. Sections 12.0,
12.1, 12.2, 12.3, and 12.8 are **mandatory**.

### 12.0 (z) The manipulation-visibility table — mandatory

The gate-G17 table of Section 15.5 is published in full, per pool pair, with the
per-pair count of differing features, the per-pair count of raw-differing
windows, the per-pair visible fraction, and the pooled figures, **whatever the
outcome**. It is reported before the primary verdict, because it is the
precondition that makes the primary interpretable. The E17 comparison figures
(1 of 55 features, 2.54% of altered windows) are printed in the same table for
contrast.

### 12.1 (a) The absolute out-of-grammar `P = 1` cell — mandatory

At `P = 1` the evaluation payload is placed **outside** the Rule grammar bytes.

**Registered reporting.** For each of the twenty generator realizations, at
`P = 1`, canonical strata, exact recall:

- the **absolute** recall of all three arms — Rule, placebo, and the shared
  real-only reference — reported per attack (Gear, RPM) and exact-macro;
- the Rule-minus-real and placebo-minus-real gains;
- the **paired Rule-minus-placebo difference**, with its mean, sample SD, and a
  Student-`t` interval over the twenty realizations (`n = 20`, `df = 19`).

This absolute form is the form used in any claim sentence.

**Descriptive companion (ratio).** The Rule-gain-to-placebo-gain ratio at
`P = 1`, per realization, summarized by **median and interquartile range** — not
the mean — with the denominator distribution reported alongside. Any realization
whose placebo denominator is `< 0.01` is flagged, reported as unstable, and
excluded from the ratio summary only.

**Registered joint statement.** Neither scale may be reported alone.

### 12.2 (b) Margin-scale sensitivity — mandatory, dual-margin reporting

Because `delta` is inherited on an absolute scale (Section 9.1), the primary
verdict is reported **jointly** with a second, stricter margin scaled to E18's
own response.

**Definition.** Let

```text
G_placebo = mean over the 20 generator realizations of
            [ Recall_placebo,g(P=1, canonical, exact-macro)
              - Recall_real(P=1, canonical, exact-macro) ]

delta_strict = 0.15 * G_placebo
```

`G_placebo` is the placebo arm's own out-of-grammar gain over the real-only
reference, computed from E18 outcomes under the Section 8 aggregation order. The
multiplier `0.15` is frozen here and is the same multiplier used to construct
the inherited `delta`.

**Registered reporting.** Report, in one table: the 90% Student-`t` interval for
the mean of `theta_g`; `delta` and its verdict (the primary verdict);
`delta_strict`, its numeric value, and the verdict obtained by running the full
Section 10 decision rule — heterogeneity screen included — with `delta_strict`
substituted for `delta`; and the ratio `delta_strict / delta`.

**Status.** `delta_strict` is **secondary only** and can never override, replace,
or annotate away the primary.

**Registered handling of disagreement.** If the two margins return different
branches, **that disagreement is itself the reportable finding.** It is stated
in the results section and carried into the limitations. It is not a reason to
suppress either margin, to relabel the primary, to declare the sensitivity
invalid, or to declare the primary invalid. **Neither verdict may be
suppressed or annotated away.** The registered sentence form is:

> Against the inherited CNN margin the joint-structure replication returns
> Branch `<X>`; against a margin rescaled to its own out-of-grammar placebo gain
> it returns Branch `<Y>`. The verdict is therefore margin-scale dependent, and
> the equivalence claim is reported as conditional on the registered absolute
> margin.

with the observed branches substituted. E17 returned `A` against `delta` and `C`
against `delta_strict`; that precedent is a planning input and does not
constrain the E18 result in either direction.

**Degenerate-denominator handling, registered now.** If `G_placebo <= 0.01` —
including the case `G_placebo <= 0` — then `delta_strict` is reported as
**undefined**, the observed `G_placebo` is published, and the reason is stated.
No substitute denominator, floor value, or alternative multiplier is invented
after the fact, and the primary verdict stands unannotated by a sensitivity that
could not be computed.

### 12.3 (c) The `P = 0` in-grammar cell — mandatory

**Registered reporting.** For each of the twenty generator realizations, at
`P = 0` (payload inside the Rule-grammar bytes), canonical strata, exact recall:
the absolute recall of the Rule arm, the placebo arm, and the real-only
reference, per attack and exact-macro, with means, sample SDs, and the paired
Rule-minus-placebo difference with a Student-`t` interval (`n = 20`, `df = 19`).

**Saturation criterion, frozen:** the cell is declared **saturated** if both
augmented arms exceed `0.99` exact-macro recall and their absolute difference is
below `0.01`; otherwise it is declared **unsaturated** and the `P = 0`
contribution to `C_P` is reported separately from the `P = 1` contribution. The
registered report must state explicitly which of the two holds. E16 and E17 were
both saturated at `P = 0`; whether E18 is remains open.

### 12.4 Exact Rule-minus-placebo remaining effects

`S, D, PS, PD, SD, PSD`. Separate 6-test Holm family.

### 12.5 Binary companion

`P, S, D, PS, PD, SD, PSD` on binary attack recall, Rule-minus-placebo. Separate
mandatory 7-test Holm family.

### 12.6 Component means

`Delta^Placebo_g` (marginal-exposure component) and `Delta^Rule_g` (total
response), each as a mean with sample SD and Student-`t` interval over the twenty
generator realizations.

### 12.7 Three-way descriptive comparison against E16 and E17 — mandatory

Because E18 reuses E16's and E17's exact twenty units, the three `theta_g`
vectors are paired. Report: the twenty paired
`(theta_g^CNN, theta_g^RF55, theta_g^RF83)` values, the Pearson and Spearman
correlation of E18 with each of E16 and E17, the paired differences
`theta_g^RF83 - theta_g^CNN` and `theta_g^RF83 - theta_g^RF55` with mean and
sample SD, and a scatter plot.

**Registered status: descriptive only, no test, no verdict.** Three detector
configurations are three points, not a sample. No p-value, confidence interval,
or equivalence test on the between-configuration difference may be reported, and
no statement of the form "the effect does not differ across detectors" is
permitted. The permitted language is agreement or disagreement of the registered
verdicts, plus the descriptive correlation, plus the Section 12.0 visibility
figures that distinguish E17 from E18 mechanically.

### 12.8 Per-attack Gear and RPM splits — mandatory

Attack-specific `theta` for Gear and RPM under the full Section 8 aggregation
order, and the `P = 1` per-attack Rule-minus-placebo absolute differences, each
with mean, sample SD, and Student-`t` interval over the twenty realizations.

**Registered reason this family is mandatory rather than descriptive.** The Gear
inverse relation `|255 - data0 - data1| <= 8` is the cleanest and most
completely destroyed manipulation in the design: the placebo's off-bin rotation
violates it deterministically on every injected frame, whereas the RPM
manipulation is only guaranteed to attain a *structural minimum* of residual
matches and leaves some windows partially intact (E16 amendment v2). If the
relational increment is real and representable, **Gear is where it should be
largest**. Under the E17 marginal map the Gear increment was `+0.000087` at
`P = 1` — the smallest number in the experiment — which is precisely the
signature of a detector that cannot see the manipulation. The Gear/RPM split is
therefore the sharpest available diagnostic of whether E18's representation
changed anything, and it must be reported whatever it shows.

Shifted-ID effect modification is reported alongside, descriptively.

### 12.9 Bridge sensitivity

The four bridge realizations, both arms, reported separately and always labelled
as a bridge sensitivity attached to the E15 lineage. Never pooled with the
primary twenty, never used to change `n` or `df`.

### 12.10 Finite-grid dispersion

Realization, pipeline, and interaction shares, descriptive only, explicitly
labelled as finite-grid dispersion and not random-effects variance components.

## 13. Registered branches and permitted claims

Every branch's reported form must include: the Section 12.0 visibility table,
the Section 12.2 dual-margin verdict, the Section 12.1 `P = 1` absolute and
ratio scales, the Section 12.3 `P = 0` saturation determination, the Section
12.8 Gear/RPM split, and the statement that the arms carry no optimizer-update
budget, early-stopping trajectory, or minibatch order, so the comparison is made
under an identical-training-row contract (Section 5.3).

### Branch A — marginal sufficiency survives a representation that can see the relation

Condition: Step 2 of Section 10, heterogeneity screen not triggered, G17 floor
passed.

Permitted headline:

> When the detector's representation is extended so that it demonstrably
> perceives the relational manipulation — `<V>`% of altered windows produce a
> feature-vector change, against 2.5% for the marginal map — destroying the
> implemented Gear/RPM relations still leaves the registered payload-destination
> contrast practically unchanged. Matched marginal exposure accounts for the
> response even under a representation that can express cross-byte joint
> structure.

Forbidden extensions: relational structure is irrelevant in general; the result
holds for detectors in general or for any untested representation; synthetic
realism is unnecessary; the CNN result is thereby confirmed as a population
fact.

### Branch B — the relation is representable and used

Condition: Step 3 of Section 10, heterogeneity screen not triggered, G17 floor
passed.

Permitted headline:

> A detector whose representation can express within-window cross-byte joint
> structure recovers a relational increment beyond matched marginals, under
> identical pools, identical training rows, and the same registered margin. The
> implemented Gear/RPM relation is therefore representable-and-used by this
> feature map, and the practical-equivalence verdict of E16 and E17 is
> conditional on representations that cannot express it.

This is reported as a constraint on the scope of the E16 headline, not as a
refutation of the E16 registered primary, which stands as registered for the
CNN. It must be stated in the same breath that this shows the relation is used
**by this feature map**, not by any particular deployed detector.

Forbidden extensions: physical mechanism, realism, real attack semantics,
cross-domain transfer, deployed-detector behavior, or any claim that the E16 or
E17 results were wrong rather than representation-conditional.

### Branch C — realization-contingent relation

Condition: Step 1 of Section 10.

Permitted headline:

> Under a representation that can express cross-byte joint structure, relational
> utility is realization-contingent even under matched marginals and identical
> training rows.

Reported as a methodological result requiring generator realization to enter the
inference axis, not as a failed replication.

### Branch D — inconclusive

Condition: Step 4 of Section 10, or a Student-`t` / bootstrap disagreement.

The planning assumptions of Section 9 and the observed dispersion are published,
the E16 CNN registered primary is retained unchanged and is **not** restated as
weakened, and no equivalence or difference claim is made. A Branch D is reported
as a completed experiment with an inconclusive outcome, which is a result.

### Branch E — inconclusive-by-construction

Condition: the Section 15.5 G17 floor fails.

The G17 table is published, the observed visibility is reported next to E17's
2.54%, **no fits are run**, and no inferential claim of any kind is made. The
registered statement is that the chosen feature map does not perceive materially
more of the manipulation than E17's did, so E18 cannot answer its question and
does not pretend to. This is not a technical stop and is not retried with a
different feature map under the E18 label; a different map is a different
experiment with its own prospective registration.

## 14. Required outputs and completeness

### 14.1 Implementation

```text
journal/scripts/run_e18_relation_visible_rf_training.py
journal/scripts/evaluate_e18_relation_visible_rf.py
journal/scripts/analyze_e18_relation_visible_rf.py
```

The runner imports `rf_features`, the estimator construction, the permutation
convention, and the no-clobber preflight from
`journal/scripts/train_generator_extension_rf.py` rather than re-implementing
them (gate G14), and imports the E17 pool-inheritance machinery, gate helpers,
grid construction, and provenance-record shape from
`journal/scripts/run_e17_rf_paired_training.py` rather than re-implementing
them. The evaluator reuses the E14 evaluation-window construction and factorial
cell definitions; the analyzer reuses the E16/E17 contrast normalization,
decision rule, and dual-margin machinery. All stages run on CPU.

### 14.2 Canonical outputs

```text
journal/results/tables/
  e18_pool_inheritance_audit_v1.csv
  e18_manipulation_visibility_v1.csv          <- gate G17, published always
  e18_sampling_audit_v1.csv
  e18_training_manifest_v1.csv
  e18_by_cell_v1.csv
  e18_by_scenario_v1.csv
  e18_normal_by_arm_v1.csv
  e18_augmentation_delta_v1.csv
  e18_decomposition_by_realization_v1.csv
  e18_primary_theta_summary_v1.csv
  e18_p1_cell_comparison_v1.csv
  e18_p0_cell_comparison_v1.csv
  e18_margin_sensitivity_v1.csv
  e18_secondary_effects_v1.csv
  e18_family_comparison_v1.csv
  e18_crossed_dispersion_v1.csv
  e18_bridge_sensitivity_v1.csv
  e18_real_only_reference_continuity_v1.csv
journal/results/logs/
  e18_relation_visible_rf_v1.log
  e18_relation_visible_rf_artifact_manifest_v1.json
journal/experiments/e18_relation_visible_rf/
  {pool_inheritance,manipulation_visibility,training_preflight,training_run,
   prepare,run}_v1.json
```

Completeness requirements: 24 realization pools verified by hash; the G17 table
published with one row per pool pair plus pooled rows; 245 fits complete;
expected row counts verified; every output SHA-256 recorded in the artifact
manifest with its source commit; a single machine-readable `verdict` field
carrying one of `A`, `B`, `C`, `D`, `E`; a second machine-readable
`verdict_strict_margin` field carrying `A`, `B`, `C`, `D`, or `undefined`; and a
boolean `margin_disagreement` field. Every new table and figure is registered in
`journal/results/paper_artifacts_manifest.md`.

## 15. Technical gates and stopping rules

### 15.1 Freeze gate (S0)

- **G0** — `git status --short --branch` clean for the E18 target paths;
  `git status --short -- wisa` empty; all E18 target paths absent; this document
  committed as a single-file commit before any E18 script exists; all frozen
  input SHA-256 recorded. The commit ordering — registration strictly before
  implementation — is a registered requirement and both commit hashes are
  recorded in the run records.

### 15.2 Inherited pool gates (S2)

E18 generates no pool. The E16 pool gates **G1--G9** already passed at E16 stage
S2 and are **inherited, not re-run**. They are re-verified by hash exactly as
E17 did:

- **G1i pool hash inheritance (blocking)** — for every consumed pool, the
  recomputed `x_digest` equals the value frozen in
  `journal/results/tables/e16_pool_audit_v2.csv`; the four bridge Rule pools are
  matched against the frozen `e15_rule_construction_pool_audit_v2.csv`
  whole-file digest. 48 slots, no waivers. Any mismatch halts S2 for all
  realizations.
- **G2i seed integrity** — the realization seed list read from
  `generate_e16_rule_placebo_pairs.py` equals Section 3.1 exactly; the bridge
  list equals Section 3.2 exactly.
- **G3i--G8i manipulation-check inheritance** — the frozen
  `journal/results/tables/e16_manipulation_checks_v2.csv` is read and every abort
  gate recorded there is confirmed `passed`. Not recomputed.
- **G9i draw integrity** — per fit, per class,
  `requested = drawn = unique`, `repeated = 0`, per-class request below 65,000.

### 15.3 Training, continuity, and analysis gates

- **G10a no-clobber** — every E18 target checkpoint, table, log, and record path
  is absent at preflight and written atomically. No legacy `rf_*` checkpoint, no
  E10/E14/E15/E16/E17 artifact, and nothing under `wisa/` or `wisa_camera_ready/`
  is opened for writing.
- **G10c-i paired draw identity (blocking)** — 120 pairs checked.
- **G10c-ii cross-family draw inheritance (blocking)** — the E18
  `sampling_index_sha256` equals the frozen E16 CNN value in
  `e16_training_manifest_v2.csv`. 220 fits checked; the 20 bridge Rule fits are
  recorded as exempt with their reason.
- **G10d fit completeness** — 245 target checkpoints absent at preflight and
  present at completion; per-fit provenance recorded.
- **G11 decomposition closure** — the Section 1.4 identity holds to
  floating-point closure (`|residual| <= 1e-12`) for every generator realization.
- **G12 real-only reference continuity** — the five shared E18 real-only forests
  are scored **first**, before any Rule or placebo checkpoint; their per-cell
  integer counts are frozen and re-verified byte-identically after the full grid
  has been scored. Any drift halts S6. Blocking.
- **G13 output integrity** — atomic no-clobber publication, expected row counts,
  20 × 2 × 5 primary completeness, 4 × 2 × 5 bridge completeness, 5 reference
  fits, all SHA-256 recorded, `paper_artifacts_manifest.md` updated.
- **G14 feature-map and estimator contract fidelity** — the realized feature
  dimension is exactly **83**; the first 55 columns are bit-identical to the
  imported `train_generator_extension_rf.rf_features` output on the same input,
  asserted numerically on a probe and by object identity of the imported
  function; the 28 correlation columns are in the frozen pair order of Section
  5.1; constant-channel pairs evaluate to exactly `0.0`; features are computed on
  raw un-standardized windows; no standardizer object is constructed; the
  estimator's `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`,
  and `random_state` equal the Section 5.2 values for every fit and equal the
  keywords parsed out of the frozen harness source.
- **G15 fit determinism** — one Rule fit and one placebo fit are repeated in a
  fresh child interpreter and must produce byte-identical `joblib` payloads and
  identical predictions on the evaluation set.
- **G16 dual-margin reporting completeness** — the analysis stage must emit both
  `verdict` and `verdict_strict_margin` (or `undefined` with a published
  `G_placebo`), the `margin_disagreement` boolean, the `P = 1` absolute table,
  the `P = 1` ratio summary, the `P = 0` saturation determination, and the
  Section 12.8 Gear/RPM split. Missing any one is a gate failure, not an
  omission.

### 15.4 Failure, amendment, and exclusion policy

A **technical stop** (any gate failure other than G17's floor) halts the stage.
The cause and the fix are committed as a dated prospective amendment with a new
output version identity before rerunning. Failed artifacts are preserved as
failure provenance and are not deleted. No outcome from a gate-failed stage is
interpreted. No stage is resumed in place.

A **G17 floor failure is not a technical stop.** It is a registered scientific
outcome — Branch E — and is reported, not repaired.

A **valid scientific result** — including Branch D, Branch E, sign
heterogeneity, a null relational increment, a material relational increment, a
margin-scale disagreement, or disagreement with E16 or E17 — is never hidden,
rerun, reclassified as a technical stop, or resolved by rerunning with different
settings.

If a generator realization cannot complete for a technical reason after
amendment, it is reported as missing with its cause, the primary is recomputed on
the completed realizations with the reduced `df` disclosed, and the reduction is
reported in the abstract-level limitations. Realizations are never dropped for
their outcome values.

### 15.5 G17 — the manipulation-visibility gate (S2b), registered before measurement

**Purpose.** E17's null is uninterpretable as evidence about the relation
because its feature map was provably almost blind to the manipulation
(Section 1.1). E18 must not repeat that. G17 establishes, **before any fit and
therefore before any outcome can influence it**, whether the Section 5.1 feature
map perceives materially more of the manipulation than E17's did.

**Measurement, fixed now.** For each of the 24 pool pairs
`(rule_g, placebo_g)`, with `X_rule` and `X_placebo` the aligned raw window
arrays and `F_rule = e18_features(X_rule)`, `F_placebo = e18_features(X_placebo)`
the 83-dimensional `float32` feature matrices:

```text
W_g          = #{ windows w : X_rule[w] != X_placebo[w] }        (raw-differing)
V_g          = #{ windows w in that set : F_rule[w] != F_placebo[w] }
visibility_g = V_g / W_g
K_g          = #{ features k : F_rule[:, k] != F_placebo[:, k] for some window }
```

Comparison is exact equality on the published `float32` feature values — the
values the estimator actually consumes. `K_g` is compared against the analytic
prediction of exactly 8 (Section 5.1). The same quantities are recomputed for
the E17 55-dimensional map on the same pairs and reported side by side, so the
2.54% reference figure is reproduced inside E18's own audit rather than quoted.

**The registered floor, chosen and justified now, before measuring.**

```text
POOLED FLOOR   pooled visibility over the 20 primary pairs   >= 0.50
PER-PAIR FLOOR min visibility over the 20 primary pairs      >= 0.25
```

Both must hold. Justification, registered in advance:

1. **It must be a large multiple of E17's 2.54%, not a marginal improvement.**
   0.50 is roughly a **20-fold** increase and 0.25 roughly a **10-fold**
   increase. A map that perceived, say, 5% of the manipulation would be a
   different kind of blind detector, not a seeing one, and its null would be as
   uninterpretable as E17's.
2. **A majority is the natural qualitative threshold.** "The detector perceives
   most of the manipulation it is asked to respond to" is the weakest statement
   under which a null becomes evidence about the relation rather than about the
   representation. 0.50 is the unique numerical expression of that statement and
   is therefore not a tuned parameter.
3. **The per-pair floor prevents a pooled pass carried by a few pools.** The
   inferential unit is the generator realization; a pooled figure that averaged
   over one visible realization and nineteen blind ones would make the primary
   uninterpretable for nineteen of its twenty units. 0.25 — still ten times E17
   — is the registered per-unit minimum.
4. **It is set where an analytic argument says it should comfortably pass, so
   that a failure is informative.** The manipulation moves `data1` values across
   injected frames while `data0` and the other payload bytes vary within the
   window, so `r(data1, data_k)` should change for essentially every window in
   which `data1` actually moved. A floor of 0.50 should therefore be cleared by
   a wide margin. Setting it at a level the design predicts will pass is
   deliberate: if it nonetheless fails, that failure is a real and surprising
   fact about the manipulation, not an artifact of a punitive threshold.
5. **It is not derivable from any E18 outcome**, and no E18 outcome exists at
   registration time. It cannot be moved afterwards (Section 17).

**Consequence of failure, registered now.** If either floor fails, E18 is
reported as **inconclusive-by-construction** (Branch E, Section 13). The G17
table is published in full. No fit is run, no `theta` is estimated, and no
inferential claim of any kind is made. The failure is reported as a result.

**Consequence of passing.** S3 proceeds. The realized visibility figures are
carried into every branch's reported form (Section 12.0) and into the Branch A
and Branch B headline sentences, because the strength of both readings depends
on them.

**G17 is published regardless of outcome.** Its table is written before the
floor is evaluated, so that a failing measurement cannot be discarded.

## 16. Disclosed limitations

- All twenty `Delta` pairs at a given pipeline seed share one real-only
  reference, so realization-level estimates are conditional on those five shared
  references.
- Twenty generator realizations sample realizations of one disclosed Rule
  construction law; they do not sample a population of generators.
- Three detector configurations are three points; the representation axis is
  fixed, not sampled, and E18 licenses no detector-population claim.
- The inherited absolute margin may be relatively more permissive at
  Random-Forest recall levels than at CNN recall levels. This is a known,
  registered consequence of refusing a circular re-derivation, and is the reason
  Section 12.2 is mandatory.
- **The E18 feature map is still a fixed, finite summary.** Pearson correlation
  captures *linear* within-window co-variation between payload byte pairs. The
  implemented Gear relation `|255 - data0 - data1| <= 8` is linear and is
  therefore well matched to it; the RPM relation `data1 ≈ 8*data0 + r (mod 256)`
  is linear only modulo 256 and is only partially expressible. A relational
  statistic expressible through neither the marginal block nor a pairwise linear
  association remains invisible by construction, and an E18 null is partly a
  statement about this map. This is the same class of limitation as E17's, made
  one level weaker, not removed.
- A positive E18 result shows the relation is representable-and-used by **this**
  feature map under **this** estimator. It says nothing about whether any
  deployed CAN intrusion detector has, or should have, a comparable
  representation.
- The evaluation split is source-local; E18 inherits this limitation from E8.
- The `S1` mask span of 96 exceeds the training maximum of 90; this known
  boundary condition is inherited from E14 and is not re-litigated here.
- Exact-macro (equal) weighting of Gear and RPM is a registered choice, not a
  property of the data; attack-specific values are reported as Section 12.8.
- The placebo destroys the *implemented* Gear/RPM relation only. A relational
  effect carried by some unimplemented statistic would not be detected.
- The correlation block adds 28 features to 55, a 51% increase in dimension, at
  fixed `n_estimators` and fixed `min_samples_leaf`. Random-Forest split
  selection samples `sqrt(p)` features per node, so the marginal features are
  individually sampled slightly less often in E18 than in E17. This is an
  unavoidable consequence of changing the representation at a frozen estimator
  and is disclosed rather than corrected, because correcting it would mean
  tuning a hyperparameter across the very contrast being tested.

## 17. What this registration forbids

- changing the Section 5.1 feature map — adding, removing, reweighting,
  standardizing, selecting, or reordering features — after registration, or
  substituting a Gear- or RPM-specific relational indicator for the generic
  correlation block
- changing the Section 15.5 G17 floor, its two thresholds, its measurement
  definition, or its per-pair requirement after any visibility number exists
- suppressing the G17 table, or reporting a Branch E as anything other than a
  result
- changing `delta`, re-deriving it from any E18 outcome, rescaling it, or
  swapping it for `delta_strict` as the primary margin
- changing the sample size, the seed list, the bridge/primary partition, the
  aggregation order, the decision order, the bootstrap resample count or seed, or
  any family membership after seeing an outcome
- redrawing, re-seeding, or re-ordering the synthetic row draws inherited under
  Section 4.3
- promoting the bridge realizations into the primary, or counting them in `n` or
  `df`
- counting the 200 primary fits, the 100 primary pairs, the three blocks, the
  five pipeline seeds, or raw windows as independent inferential units
- treating the feature map as a random effect, testing the E17-versus-E18
  difference, or claiming that `theta` is invariant across representations
- promoting Branch D or Branch E to Branch A or Branch B
- suppressing either margin in Section 12.2, or reporting a margin-scale
  disagreement as anything other than a finding
- reporting the `P = 1` ratio without the absolute scale, or the absolute scale
  without the ratio
- omitting the `P = 0` cell values, the saturation determination, or the Section
  12.8 Gear/RPM split
- interpreting a Branch B as evidence about any deployed detector, or a Branch A
  as evidence that relational structure is generally irrelevant
- re-opening, revising, re-scoring, or restating the E16 CNN registered primary,
  or the E17 registered outputs, as a function of an E18 outcome; the E17
  reclassification of Section 1.2 is made here and prospectively, and does not
  alter a single E17 artifact
- tuning any hyperparameter, feature, or sampling choice on an E18 outcome
- adding external datasets, generator families, or further detector families to
  E18
- overwriting E10, E14, E15, E16, E17, legacy RF, or WISA artifacts, or modifying
  `wisa/` or `wisa_camera_ready/`
