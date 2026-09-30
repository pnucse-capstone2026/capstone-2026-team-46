# E16 preregistration — relational-versus-marginal decomposition of the payload-destination response

Registration time: 2026-07-25T16:37:23Z

Registration source commit: `8bd3b29`

Status: **PRE-RESULT. E16 pool generation, training, scoring, and analysis are
prohibited until this document is committed and the freeze gates in Section 15
pass.**

At registration time no E16 pool, checkpoint, training log, outcome table,
figure, or verdict existed. E14 outcomes for Rule construction seed `314159`
(including its single marginal-matched placebo arm) and E15 outcomes for
construction seeds `{314159, 271828, 161803, 141421, 173205}` are already
known and are quoted in this document as planning inputs. This registration is
therefore prospective for twenty previously undrawn Rule/placebo construction
pairs. Any later design change requires a dated amendment committed before the
affected stage; it must not silently edit this record.

## 1. Question and evidential scope

E14 localized the matched-budget Rule-minus-real augmentation response to the
payload-destination factor `P` within one Rule pool. E15 showed that the `P`
direction repeats across four further Rule-pool realizations and five detector
pipelines. Neither experiment identified **why** the response is localized to
`P`, because neither crossed a marginal-matched placebo with the construction
axis. E16 asks:

> Is the repeated payload-destination response specific to Rule-defined
> relational structure, or is it reproduced by a marginal-matched augmentation
> that destroys that structure while preserving every registered marginal?

### 1.1 Decomposition identity

For construction `g` and pipeline `p`, let `C_P{·}` denote the registered `P`
factorial contrast on the primary endpoint (Section 8), and let

```text
Delta^Rule_{g,p}     = C_P{ Y_Rule,g,p    - Y_real,p }
Delta^Placebo_{g,p}  = C_P{ Y_Placebo,g,p - Y_real,p }
```

Because `C_P{·}` is linear and the real reference is shared within pipeline `p`,
the following identity holds exactly and is not an approximation:

```text
C_P(Rule - Real)  =  C_P(Placebo - Real)  +  C_P(Rule - Placebo)
      total       =   marginal-exposure   +  relational-structure
                          component            increment
```

E16 estimates all three terms on the same twenty construction pairs. The
identity is verified numerically to floating-point closure as a gate
(Section 15, G11); it is not assumed.

### 1.2 What E16 cannot establish

All conclusions are conditional on the frozen Car-Hacking train/validation/test
split, the disclosed 65,000-window-per-class Rule construction law, the twenty
drawn construction seeds, the five selected paired detector-pipeline seeds, the
E14 evaluation constructions and three descriptive blocks, the fixed
matched-training-budget CNN protocol, and the shared E14 matched-real
references.

E16 does not establish synthetic-data realism, physical payload causation,
replacement of real attack data, universal generator behavior, cross-dataset
robustness, or cellwise stability of individual construction × pipeline cells.
A practical-equivalence outcome is **not** evidence that relational structure
is irrelevant to CAN intrusion detection in general; it is evidence that, for
the implemented Rule grammar and the registered marginals, destroying the
implemented relation does not materially change the registered
payload-destination contrast. E16 contains no external-dataset evaluation.

### 1.3 Why this is a positive identification, not a defensive null

The registered equivalence branch is not "we failed to find an effect". It is a
counterfactual identification result: the headline gain and its
payload-destination fragility, which superficially read as evidence that the
detector learned the Rule relation, are tested against an augmentation that is
marginally identical and relationally empty. Either outcome is informative and
must be reported as a positive finding in the registered form of Section 13.

## 2. Mandatory execution order

Stages must run in this order. A later stage must not begin until the earlier
stage's gates have passed and its run record has been committed.

1. **S0 — freeze.** Section 15 G0 gates; commit this document.
2. **S1 — implementation.** Scripts and tests (Section 14.1) committed with no
   E16 outcome, pool, checkpoint, or score present.
3. **S2 — paired pool generation.** Twenty fresh Rule/placebo pairs plus four
   bridge placebo pools. Gates G1--G9.
4. **S3 — paired training.** 220 fits. Gates G10.
5. **S4 — shared-real runtime continuity.** Gate G12, before any E16 Rule or
   placebo checkpoint is scored.
6. **S5 — scoring.** Exact and binary scored separately.
7. **S6 — registered analysis and verdict.** Gates G11, G13.

No stage may be resumed in place after an abort. Retry requires a dated
amendment and a new output version identity (Section 15.4).

## 3. Frozen seed axes and roles

### 3.1 Construction seeds

Twenty fresh construction seeds are drawn by the following deterministic
algorithm, fixed before the draw and reproducible by any reader:

```python
META_SEED = 20260725                 # ISO date constant, declared before drawing
LO, HI    = 100_000, 999_999         # sampling frame
EXCLUDED  = (314159, 271828, 161803, 141421, 173205)   # E15 lineage
N_FRESH   = 20

rng, seen, seeds = numpy.random.default_rng(META_SEED), set(EXCLUDED), []
while len(seeds) < N_FRESH:
    c = int(rng.integers(LO, HI + 1))
    if c not in seen:
        seen.add(c)
        seeds.append(c)
```

Environment for the draw: NumPy `2.4.6`, `PCG64`. The resulting ordered seed
list is frozen here:

```text
415637  748203  560657  459336  457191  999191  715822  298581  922089  864178
583909  467232  550427  485791  820632  837943  222537  685012  192753  739263
```

These twenty are the **primary inferential units** (`n = 20`, `df = 19`). No
seed may be added, removed, reordered, or substituted after registration. If a
construction fails a gate, Section 15.4 applies; the seed is not silently
replaced by a fresh draw.

The five E15 seeds are excluded from the primary because their Rule-arm
outcomes are already observed. They are used only as the bridge sensitivity of
Section 12.9.

### 3.2 Placebo seed derivation

Each construction's placebo permutation stream is derived deterministically and
independently of the construction stream:

```python
placebo_seed(g) = int.from_bytes(
    hashlib.sha256(f"e16-placebo|{g}".encode()).digest()[-4:], "big")
```

Frozen values for the twenty primary constructions:

| construction | placebo | construction | placebo |
|---|---|---|---|
| 415637 | 1590757002 | 583909 | 1135262528 |
| 748203 | 3788055808 | 467232 | 3528887398 |
| 560657 | 959103541 | 550427 | 1799510100 |
| 459336 | 1976626086 | 485791 | 2515679318 |
| 457191 | 1860529192 | 820632 | 3506727299 |
| 999191 | 1062549808 | 837943 | 3309880480 |
| 715822 | 1793600208 | 222537 | 60393616 |
| 298581 | 779204027 | 685012 | 1322814775 |
| 922089 | 3873785419 | 192753 | 3614171623 |
| 864178 | 3337536839 | 739263 | 413835320 |

All twenty construction seeds and all twenty placebo seeds are unique, and no
construction seed collides with the E15 lineage. This is re-verified as gate G2.

### 3.3 Detector-pipeline seeds

Frozen and identical to E14/E15:

```text
7, 42, 123, 2026, 3407
```

Pipeline seeds are a fixed repetition axis averaged inside each construction.
They are **not** an inferential unit in the primary analysis. The 100 primary
construction × arm × pipeline cells are never counted as 100 independent
replicates.

### 3.4 Evaluation blocks

The three frame-disjoint blocks and 6,000 held-out normal bases of the frozen
E8 manifest are reused without redrawing:

```text
journal/results/tables/evaluation_realization_blocks_e13_sampling_v2.csv
```

Blocks are averaged inside each construction and are never an inferential unit.

## 4. Frozen inputs and reused lineage

### 4.1 Data and evaluation inputs

Reused unchanged, with SHA-256 recorded at S0 and re-verified before each
subsequent stage: the Car-Hacking train/test window NPZs, the E8 evaluation
realization manifest, the E14 `2×2×2` factorial transformation code path, the
four ID strata (Gear canonical `0x43F`, Gear shifted `0x440`, RPM canonical
`0x316`, RPM shifted `0x329`), and the standardizer derivation fitted on the
real train split only.

### 4.2 Shared matched-real checkpoints

The five E14 matched-budget real-only checkpoints are reused as the shared
reference for every construction and both arms:

```text
cnn_real_only_matchedsteps_e14_v1_seed{7,42,123,2026,3407}.pt
```

No new real-only training occurs in E16. The resulting correlation structure —
every `Delta^Rule_{g,p}` and `Delta^Placebo_{g,p}` at the same `p` shares one
real reference — is preserved by the pairing and is disclosed as a limitation
in Section 16.

### 4.3 Frozen evaluation material

The E14 eight factorial cells, three frame-disjoint blocks, four strata, and
the `S0 = start + 2*arange(32)` / `S1 = start + 3*arange(32)` mask schedules are
frozen evaluation material. E16 introduces no new evaluation construction, no
new detector architecture, no new generator family, and no external dataset.

## 5. Paired pool construction

### 5.1 Rule arm

Each Rule pool is generated by the E15 construction law with the construction
seed as the only varying input: 65,000 windows per class, four classes
(`DoS`, `Fuzzy`, `Gear`, `RPM`), identical generator configuration, identical
class allocation, identical serialization convention (NumPy `2.4.6`, exact
`PCG64`, little-endian).

### 5.2 Placebo arm

Within each construction, the placebo pool is an **aligned twin** of that
construction's Rule pool. The following are preserved array-for-array:

- source normal base-window selection
- attack class and label
- injection positions and frame count
- CAN ID and DLC
- `delta_t`
- `data0` and `data2`--`data7`
- the per-window multiset of injected-frame `data1` values
- row order, final pool order, added sample count, per-class quota

The only quantity destroyed is the implemented Gear/RPM inter-byte relation, by
constrained within-window permutation of `data1` over injected frames only:

- **Gear** — off-bin rotation over the six `data0 // 40` level bins, so
  `|255 - data0 - data1| <= 8` is violated for every off-bin assignment.
- **RPM** — rejection resampling (cap 1000 attempts) until no injected frame
  satisfies `((data1 - 8*data0) mod 256) in [0, 8)`.

`DoS` and `Fuzzy` windows are copied bit-identically between the two arms.

This logic is ported from `journal/scripts/generate_rule_placebo_twin.py`, which
must **not** be reused directly: its construction seed `314159`, placebo seed
`1618033`, and output paths are hardcoded. The E16 wrapper parameterizes seed
and output identity and adds the gates below.

### 5.3 Lineage reproduction gate

For each of the four E15 bridge constructions, the E16 generator must reproduce
the frozen E15 v2 Rule pool array-for-array (`x`, labels, injection counts,
sampling indices, positions). This validates the E16 wrapper against an
independently frozen artifact at zero additional cost and is gate G7. Failure
halts S2 for all constructions, not only the bridge ones.

### 5.4 Naming and atomic publication

```text
journal/datasets/synthetic/
  e16_rule_cseed<g>_windows_v1.npz
  e16_placebo_cseed<g>_windows_v1.npz
journal/models/generator_extension/
  cnn_rule_0p30_cseed<g>_matchedsteps_e16_v1_seed<p>.pt
  cnn_placebo_0p30_cseed<g>_matchedsteps_e16_v1_seed<p>.pt
```

All outputs are written atomically with a no-clobber preflight. No E14, E15, or
WISA artifact is overwritten, and `wisa/` is not modified.

## 6. Training

Within each construction, the Rule and placebo arms are paired on:

- the same detector initialization seed
- the same synthetic row indices and per-class quota
- the same minibatch RNG convention
- the same maximum optimizer-update budget (6,156 updates), 513-step
  validation cadence, and best-validation selection rule

The standardizer is fitted on the real train split only. Deterministic
execution mode is enabled and the actual runtime flags are recorded. The
trainer receives an explicit `--synthetic-pool` override; the `lib_common`
global default path is not modified.

Synthetic consumption uses the E14 `+30%` consumer-seed rule with a
no-replacement draw. `requested = drawn = unique` and `repeated = 0` are
verified per fit (gate G9). Validation metrics are recorded but are never used
to select an E16 factorial outcome.

Total fits: `20 constructions × 2 arms × 5 pipelines = 200` primary, plus
`4 bridge constructions × 1 placebo arm × 5 pipelines = 20` sensitivity, for
**220 new fits**.

## 7. Endpoints

**Primary endpoint.** Exact attack-type recall, canonical-ID strata, Gear and
RPM computed separately then equally weighted (equal-macro).

**Mandatory companion.** Binary attack recall on the same construction.

Exact and binary are scored and reported separately and are never merged into a
single detection number.

## 8. Contrasts and aggregation order

For construction `g`, pipeline `p`, block `b`, attack `a`, and factorial cell
`(P, S, D)`:

```text
A^Rule_{g,p,b,a}(P,S,D)    = Recall_rule(g,p,b,a,P,S,D)    - Recall_real(p,b,a,P,S,D)
A^Placebo_{g,p,b,a}(P,S,D) = Recall_placebo(g,p,b,a,P,S,D) - Recall_real(p,b,a,P,S,D)
```

The registered aggregation order is fixed and must not be permuted:

1. **attack** — Gear and RPM equally weighted
2. **block** — the three blocks equally weighted
3. **cell** — `C_P{A} = mean over (S,D) of [ A(P=1,S,D) - A(P=0,S,D) ]`
4. **pipeline** — the five pipeline seeds equally weighted
5. **construction** — the twenty constructions are the inferential unit

giving, for each construction `g`:

```text
Delta^Rule_g     = mean_p C_P{A^Rule_{g,p}}
Delta^Placebo_g  = mean_p C_P{A^Placebo_{g,p}}
theta_g          = Delta^Rule_g - Delta^Placebo_g
```

**`theta_g` is the primary estimand.** The primary statistic is the mean of the
twenty `theta_g` values with an untruncated Student-`t` interval on `df = 19`.

## 9. Sample size, SESOI, and power

### 9.1 SESOI

The equivalence margin is frozen as an absolute constant:

```text
delta = 0.133855625
      = 0.15 × 0.8923708333333333
```

The multiplier is `0.15`. The denominator is the already-published E15 v2
new-four-construction Rule-minus-real `P` mean on the primary endpoint
(`journal/results/tables/e15_l4_factorial_hierarchical_summary_v2.csv`,
`row_type = summary_new4`). Freezing `delta` as a constant computed from
pre-existing published data avoids a random denominator in the decision rule.

**Justification.** The margin is derived from the claim the equivalence branch
would license, not from any observed `theta`. The Branch A headline asserts
that matched marginal exposure accounts for **most** of the payload-destination
response; "most" requires a relational share below 50%. A margin at 15% of the
total response is substantially stricter than the claim requires, and is
therefore defensible independently of E14's observed value.

The absolute margin `0.05` is explicitly rejected. Under the planning value of
Section 9.2 it lies below the planning `theta`, which makes both equivalence and
material difference undecidable and leaves the inconclusive branch at 70--92%
regardless of sample size. Choosing a margin that cannot decide is a design
defect, not a conservative choice.

### 9.2 Planning values

Planning value for `theta`: `+0.063392`, the E14 observed Rule-minus-placebo
exact equal-macro canonical `P` effect at construction `314159`.

The between-construction standard deviation of `theta_g` is the quantity E16
exists to measure and cannot be known in advance. It is bracketed indirectly
from the E14 pipeline-level `theta` SD (`0.114450`) and the E15
construction-level Rule-minus-real `P` SD (`0.031959`), giving a planning range
of `0.02`--`0.08` with a pessimistic bound of `0.12` should construction
heterogeneity match pipeline heterogeneity.

### 9.3 Power

Simulated probability of the equivalence branch at `delta = 0.133855625` and
planning `theta = +0.063392`:

| SD of `theta_g` | n=10 | n=16 | **n=20** |
|---|---|---|---|
| 0.035 | 100% | 100% | **100%** |
| 0.050 | 99% | 100% | **100%** |
| 0.080 | 82% | 96% | **98%** |
| 0.120 (pessimistic) | 53% | 73% | **81%** |

The design is not biased toward equivalence. At `n = 20` and `SD = 0.05`, a
true relational increment at 20% of the total response is detected as a material
difference with probability 97%, and at 28% of the total with probability 100%.

`n = 20` is registered. The additional cost over `n = 10` is approximately two
GPU hours and five GiB.

## 10. Primary decision rule

Evaluated in this order. The order is registered and must not be changed after
seeing outcomes.

**Step 1 — heterogeneity screen.** Count the construction-level `theta_g` values
falling outside `[-delta, +delta]`. If **five or more of twenty** fall outside,
the registered headline is **Branch C** (realization-contingent relation),
regardless of the aggregate. The aggregate verdict from Steps 2--3 is still
computed and reported as descriptive.

**Step 2 — practical equivalence.** If the 90% Student-`t` interval for the mean
of `theta_g` (two one-sided tests at `alpha = 0.05`, `df = 19`) lies entirely
within `[-delta, +delta]`, the verdict is **Branch A**.

**Step 3 — material difference.** Otherwise, if the 95% Student-`t` interval
lies entirely outside `[-delta, +delta]` on one side, the verdict is
**Branch B**.

**Step 4 — inconclusive.** Otherwise the verdict is **Branch D**. A Branch D
result is reported as such and is never promoted to Branch A or Branch B.

The primary is a single test on a single endpoint and therefore carries no
multiplicity correction. Effect size, interval, construction sign pattern, and
relational share are reported before any p-value; no p-value is presented as
the primary scientific claim.

**Robustness, reported alongside the primary and never substituted for it:** a
construction-level exact sign test on `theta_g`, and a percentile bootstrap
interval over constructions (`10,000` resamples, seed `20260725`). If the
bootstrap and Student-`t` verdicts disagree, the disagreement is reported and
the verdict is downgraded to Branch D.

## 11. Relational share

Reported descriptively for interpretation, not as a test statistic:

```text
share_g = theta_g / |Delta^Rule_g|
```

Summarized by median and interquartile range across the twenty constructions,
with the denominator distribution reported. Any construction with
`|Delta^Rule_g| < 0.10` is flagged and excluded from the share summary only,
never from the primary.

## 12. Secondary families

Each family is separately Holm-corrected where indicated. None may replace the
primary or be promoted to the headline after seeing outcomes.

1. **Exact Rule-minus-placebo remaining effects** — `S, D, PS, PD, SD, PSD`.
   Separate 6-test Holm family.
2. **Binary companion** — `P, S, D, PS, PD, SD, PSD` on binary attack recall,
   Rule-minus-placebo. Separate mandatory 7-test Holm family.
3. **Marginal-exposure component** — `Delta^Placebo_g`, mean and interval over
   the twenty constructions.
4. **Total response replication** — `Delta^Rule_g`, mean and interval over the
   twenty constructions. This replicates E15's `n = 4` construction-marginal
   `P` finding at `n = 20` and is reported as a strengthened replication of the
   existing result, not as a new claim.
5. **`P = 1` cell comparison (mandatory).** See Section 12.1 below.
6. **Gear/RPM heterogeneity** — attack-specific `theta` values, descriptive.
7. **Shifted-ID effect modification** — descriptive sensitivity.
8. **Pipeline-first and finite-grid dispersion** — construction, pipeline, and
   interaction shares, descriptive only, explicitly labelled as finite-grid
   dispersion and not random-effects variance components.
9. **E15 bridge sensitivity** — the four E15 constructions with post-hoc placebo
   twins. Reported separately and always labelled as a post-hoc control
   attached to already-observed Rule outcomes.

### 12.1 The `P = 1` cell comparison

This secondary is mandatory and is registered now because it materially affects
how the headline may be phrased.

At `P = 0` (payload in Rule-grammar bytes) E14 observed Rule and placebo
augmentation gains that agree to four decimals (`+0.9979` both). The entire
Rule-minus-placebo difference lives at `P = 1` (payload outside the grammar),
where E14 observed:

| arm | Gear | RPM |
|---|---|---|
| Rule gain | +0.1915 | +0.0444 |
| Placebo gain | +0.1019 | +0.0073 |
| absolute difference | +0.0896 | +0.0372 |
| ratio | 1.88× | 6.12× |

On the absolute scale the relational increment is 7% of the total response; on
the relative scale at `P = 1` the Rule arm is roughly twice the placebo arm
overall and six times on RPM. Both statements are true, and reporting only the
absolute scale would be misleading to a reader who inspects the `P = 1` cell.

Registered reporting for E16:

- **Primary form of this secondary (absolute).** For each construction, the
  absolute recall of all three arms at `P = 1`, per attack and equal-macro, plus
  the Rule-minus-placebo difference with a Student-`t` interval over the twenty
  constructions. This is the form used in any claim sentence.
- **Descriptive form (ratio).** The Rule-gain-to-placebo-gain ratio at `P = 1`,
  computed per construction and summarized by **median and interquartile
  range** — not the mean — with the denominator distribution reported
  alongside. Any construction whose placebo denominator is `< 0.01` is flagged
  and the ratio for that construction is reported as unstable and excluded from
  the ratio summary only.

**Registered joint statement.** Whichever branch the primary returns, the
manuscript must report both scales together in the results section, in the
registered form:

> Destroying the implemented Rule relation reproduces most of the
> payload-destination response. The residual difference is concentrated in the
> out-of-grammar cell, where absolute recall remains low in both arms but the
> Rule arm recovers a multiple of the placebo arm.

with the observed numbers substituted. Neither scale may be reported alone.

## 13. Registered branches and permitted claims

### Branch A — marginal sufficiency

Condition: Step 2 of Section 10, heterogeneity screen not triggered.

Permitted headline:

> The dominant payload-destination response is reproduced after destruction of
> the implemented Rule relations; matched marginal exposure accounts for most of
> the response across twenty prospective constructions.

Forbidden extensions: relational structure is irrelevant in general; synthetic
realism is unnecessary; the detector learns nothing relational; any statement
about generators or grammars outside the implemented Rule law.

### Branch B — relational increment

Condition: Step 3 of Section 10, heterogeneity screen not triggered.

Permitted headline:

> Rule-defined relational structure contributes a construction-repeated
> increment beyond matched marginals.

Forbidden extensions: physical mechanism, realism, real attack semantics,
cross-domain transfer.

### Branch C — realization-contingent relation

Condition: Step 1 of Section 10.

Permitted headline:

> Relational utility is realization-contingent even under matched marginals and
> matched training budgets.

This is reported as a methodological result requiring generator realization to
enter the inference axis, not as a failed replication.

### Branch D — inconclusive

Condition: Step 4 of Section 10, or a Student-`t` / bootstrap disagreement.

The power assumptions of Section 9 and the observed dispersion are published,
the E14/E15 response-surface claims are retained unchanged, and no equivalence
or difference claim is made. Title and abstract are not rewritten to assert a
decomposition that was not established.

## 14. Required outputs and completeness

### 14.1 Implementation

```text
journal/scripts/generate_e16_rule_placebo_pairs.py
journal/scripts/run_e16_paired_training.py
journal/scripts/evaluate_e16_relational_response.py
journal/scripts/analyze_e16_relational_response.py
journal/scripts/make_e16_relational_response_figure.py
journal/tests/test_e16_relational_response.py
```

Tests run on toy bases and must not compute any E16 outcome or write any
canonical checkpoint, pool, or score.

### 14.2 Canonical outputs

```text
journal/results/tables/
  e16_pool_audit_v1.csv
  e16_manipulation_checks_v1.csv
  e16_sampling_audit_v1.csv
  e16_training_manifest_v1.csv
  e16_by_cell_v1.csv
  e16_by_scenario_v1.csv
  e16_augmentation_delta_v1.csv
  e16_decomposition_by_construction_v1.csv
  e16_primary_theta_summary_v1.csv
  e16_p1_cell_comparison_v1.csv
  e16_secondary_effects_v1.csv
  e16_crossed_dispersion_v1.csv
  e16_bridge_sensitivity_v1.csv
  e16_real_runtime_continuity_v1.csv
journal/results/logs/
  e16_relational_response_v1.log
  e16_relational_response_artifact_manifest_v1.json
journal/experiments/e16_relational_response_decomposition/
  {pool_generation_preflight,pool_generation_run,training_preflight,
   training_run,prepare,run}_v1.json
```

Completeness requirements: 20 construction pairs × 2 arms present; 220 fits
complete; expected row counts verified; every output SHA-256 recorded in the
artifact manifest with its source commit; a single machine-readable verdict
field carrying one of `A`, `B`, `C`, `D`.

## 15. Technical gates and stopping rules

### 15.1 Freeze gate (S0)

- **G0** — `git status --short --branch` and `git status --short -- wisa` clean
  for `wisa/`; workspace audit passes; all E16 target paths absent; all frozen
  input SHA-256 recorded.

### 15.2 Pool gates (S2)

- **G1 aligned-pair identity** — within each construction, Rule and placebo
  arrays are equal for every field except `data1` on spoof-injected frames.
- **G2 seed integrity** — twenty unique construction seeds, twenty unique
  derived placebo seeds, no intersection with the E15 lineage, list matching
  Section 3.1 exactly.
- **G3 marginal identity** — per-window sorted `data1` multiset on injected
  spoof frames identical between arms.
- **G4 non-target channel identity** — `can_id`, `dlc`, `data0`, `data2`--
  `data7`, `delta_t`, positions, labels, injection counts, row order identical.
- **G5 relation destruction** — Gear residual fraction with
  `|255 - data0 - data1| <= 8` below `0.01`; RPM coupling residual matches
  exactly `0` after the rejection cap.
- **G6 DoS/Fuzzy bit identity** — non-spoof windows bit-identical between arms.
- **G7 lineage reproduction** — the four E15 bridge constructions reproduce the
  frozen E15 v2 Rule pools array-for-array (Section 5.3).
- **G8 provenance continuity** — source commit, worktree state, generator and
  helper hashes recorded and matching the S1 record.
- **G9 draw integrity** — per class, `requested = drawn = unique`,
  `repeated = 0`, and the per-class request below 65,000.

### 15.3 Training, continuity, and analysis gates

- **G10 training completeness** — 220 target checkpoints absent at preflight and
  present at completion; per-fit update budget, validation cadence, exact pool
  override, no-fallback, and deterministic runtime flags recorded.
- **G11 decomposition closure** — the Section 1.1 identity holds to
  floating-point closure (`|residual| <= 1e-12`) for every construction and
  pipeline.
- **G12 shared-real runtime continuity** — the five shared E14 matched-real
  checkpoints are re-scored under the E16 evaluator and every integer count
  matches the E14 record exactly across all 495 continuity rows. **This gate
  runs before any E16 Rule or placebo checkpoint is scored.** Scoring is
  performed on CPU to preserve integer-count equality with the E14/E15 records;
  the scoring device is not changed.
- **G13 output integrity** — atomic no-clobber publication, expected row counts,
  20 × 2 × 5 completeness, all SHA-256 recorded.

### 15.4 Failure, amendment, and exclusion policy

A **technical stop** (any gate failure) halts the stage. The cause and the fix
are committed as a dated prospective amendment with a new output version
identity before rerunning. Failed artifacts are preserved as failure provenance
and are not deleted. No outcome from a gate-failed stage is interpreted.

A **valid scientific result** — including Branch D, sign heterogeneity, or a
null relational increment — is never hidden, rerun, or reclassified as a
technical stop.

If a construction cannot complete for a technical reason after amendment, it is
reported as missing with its cause, the primary is recomputed on the completed
constructions with the reduced `df` disclosed, and the reduction is reported in
the abstract-level limitations. Constructions are never dropped for their
outcome values.

## 16. Disclosed limitations

- All twenty `Delta` pairs at a given pipeline seed share one real-only
  reference checkpoint, so construction-level estimates are conditional on those
  five shared references.
- Twenty construction seeds sample realizations of one disclosed Rule
  construction law; they do not sample a population of generators.
- The evaluation split is source-local; E16 inherits this limitation from E8.
- The `S1` mask span of 96 exceeds the training maximum of 90; this known
  boundary condition is inherited from E14 and is not re-litigated here.
- Equal-macro weighting of Gear and RPM is a registered choice, not a property
  of the data; attack-specific values are reported as Section 12.6.
- The placebo destroys the *implemented* Gear/RPM relation only. It does not
  destroy every conceivable relational structure, and a relational effect
  carried by some unimplemented statistic would not be detected.

## 17. What this registration forbids

- changing `delta`, the multiplier, the denominator, the sample size, the seed
  list, the aggregation order, the decision order, or any family membership
  after seeing an outcome
- reusing a single placebo across multiple constructions
- counting the 100 primary cells, the 20 × 5 crossed cells, the three blocks, or
  raw windows as independent inferential units
- promoting Branch D to Branch A or Branch B
- reporting the `P = 1` ratio without the absolute scale, or the absolute scale
  without the ratio
- interpreting practical equivalence as evidence that relational structure is
  generally irrelevant
- adding external datasets, detector families, or generator families to E16
- overwriting E14, E15, or WISA artifacts, or modifying `wisa/`
- submitting to any venue before the WISA notification date 2026-07-29
