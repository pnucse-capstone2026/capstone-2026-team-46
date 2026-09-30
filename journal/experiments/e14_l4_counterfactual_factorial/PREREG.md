# E14 preregistration — matched L4 construction counterfactual factorial

Registration time: 2026-07-24T10:20:10Z

Status: **PRE-RESULT; scoring prohibited until the freeze gates in Section 12 pass**

At registration time, no E14 model inference, E14 outcome table, or E14 figure
had been produced. The five E14 matched-budget `real_only` checkpoints specified
below also did not exist. This document must be committed and its SHA-256
recorded before those checkpoints are trained. Any later design change must be
disclosed in a dated amendment; it must not silently edit this registration.

## 1. Question and evidential scope

The existing L4 evaluation changes payload destination, frame-mask layout, and
payload dynamics together. E14 asks:

> In the frozen Car-Hacking test backgrounds and selected CNN pipeline
> realizations, which implemented L4-side construction factors or interactions
> are associated with the Rule-augmentation response in exact Gear/RPM typing?

E14 is an evaluation-construction factorial, not an intervention on a physical
vehicle or attacker. `P`, `S`, and `D` below are implemented field
transformations. Even a resolved effect licenses only an
augmentation-specific response to that construction contrast.

All conclusions are conditional on:

- the frozen Car-Hacking split and selected evaluation bases;
- the five paired pipeline seeds;
- the Rule pool built under construction seed `314159` and its marginal-matched
  placebo twin built with permutation seed `1618033`;
- the disclosed CNN/training and checkpoint-selection protocol.

E14 cannot establish a physical attack mechanism, realism, replacement of real
attacks, cross-domain transfer, or construction-seed stability. It does not
reopen E10's final `SX-cal` margin-scale verdict.

## 2. Fixed execution order

The following order is mandatory.

1. Commit this preregistration after an independent code/statistics audit.
2. Implement toy-data tests, the matched-real training wrapper, the evaluator,
   and the analyzer without scoring any E14 outcome.
3. Commit the tested implementation and record its clean source commit.
4. Train only the five preregistered matched-budget `real_only` baselines.
5. Run `--prepare-only`. This stage may read source windows and construct
   manifests/checks, but must not load a checkpoint or perform model inference.
6. Freeze and verify all input, checkpoint, standardizer, transformation, test,
   and source-code hashes in `prepare_v1.json`.
7. Commit the training/preparation records and manifests without changing the
   frozen implementation.
8. Score all preregistered arms, cells, strata, blocks, and seeds exactly once.
9. Analyze the complete tables and publish results regardless of direction.
10. Create a figure and edit the manuscript only after the complete scientific
   verdict exists.

No canonical scoring is allowed from a dirty source tree or before the Stage-A
manipulation gates pass.

## 3. Frozen source inputs

### 3.1 Data, manifest, and reference code

| Input | SHA-256 |
|---|---|
| `journal/results/tables/evaluation_realization_blocks_e13_sampling_v2.csv` | `9e18b3844e20f259938c732a10bc0251767f0d2eaae9c731ba8da7ef087f5efb` |
| `journal/experiments/e8_evaluation_realization/run_e13_sampling_v2.json` | `fae2ca82bb26ab24a5699fce9f331951be23746093fef9e28d0b240fd2c3b2db` |
| `journal/datasets/windows/test_windows.npz` | `4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7` |
| `journal/datasets/windows/train_windows.npz` | `44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4` |
| `journal/datasets/synthetic/rule_based_windows.npz` | `4078ab00b8531f760b296b5fd293e3df914bef996d317c958ead4762137bafd5` |
| `journal/datasets/synthetic/rule_placebo_windows.npz` | `2b336ed46bb8f3248d10831c80fdfa6db25c3e6eb44f4fd680f78546a63ec4fa` |
| `journal/results/logs/generate_rule_placebo_twin.log` | `5ca8287c5f7cef322808f73759fa1ad69e5b6d1c80182ab4b795809d0bfaddc6` |
| `journal/scripts/evaluate_evaluation_realization.py` | `eadf93df237946b7f670a74e316f0b80fc1d1920d4cc7072c7434c699a2ff2bb` |
| `journal/scripts/generate_rule_based_synthetic_v2.py` | `34ff0374af1856fd842da8226580d5ff0d47cbd16b36fb4273de3e06bcac2570` |
| `journal/scripts/e10a_crossover.py` | `a1156ce1228b1975906fcfe8aad1c888e307944e7e88b6242aa32e5c3713432f` |
| `journal/scripts/train_generator_extension_cnn.py` | `b4a95cdb476805dae4d58cd6a81082f5f92f5858864d1cd821b175b1d93e9247` |
| `journal/scripts/lib_common.py` | `5848ff4b0063d4394cd1d6627faf49e072cb363d7796cc64fce3acc48b05639a` |

The standardizer is not a separately stored model. It is recomputed using
`lib_common.fit_standardizer()` from the complete real
`train_windows.npz["x"]` only. The derived arrays must be contiguous
little-endian `float32`, shape `(1,1,11)`, C order:

| Array | SHA-256 of raw C-order bytes |
|---|---|
| mean | `1e53f10b4dfb1de8522513ca20c7d0c2b5a19a8de8f76067e4273a93467da8a2` |
| standard deviation | `092f42e9a1019fe709233a91c6e876e976f8b3b8616807b42eb6e6da7b7dbb7c` |
| concatenated mean then standard deviation | `9019b1a91c983859e8dd998717eb5c7b612ed03dc2a65ec63402fd54ab42ef4e` |

The E14 evaluator/analyzer/test hashes are unknown at registration and must be
frozen in `prepare_v1.json` before scoring.

### 3.2 Existing checkpoint hashes

The five seeds are fixed as `7, 42, 123, 2026, 3407`. Existing checkpoints are
read-only and must match the following hashes.

| Seed | `real_std` | `rule_ms` | `placebo_ms` | `rule_std` | `placebo_std` |
|---:|---|---|---|---|---|
| 7 | `a16f97511c74629db2e76075eab94d82f0d1a3f169737a09369967261c36dac3` | `27c37e443ea6880761163bc2ea10ed5a86ab3ab1a6a6969770a284e2fd0d1182` | `60bcf0a0d36a9b7725d64239a7e9cabe639ebc13248b62a436f05b7fa770869f` | `77e90846fb1ab3282ef8b48a3000c2f13dbb5d37dbb3ad68208ea892f5bd6947` | `ac7fe4a64dcb9205425db8865b1bcd8d6f4faade46e42496b729a879bdbb2919` |
| 42 | `a6a06376f0348fcbbb058047eaf8e64b488c5695054265e8f5ef28b6a4388d0e` | `36616d6c1450337e51a172ab97a63e702c4147ab0da1411b0ced9e359a326364` | `c0baf3e93be225dc1d08bbb1efbcd82be242f63d6b4719b619f269169eb6ea3f` | `3ad60e856c38cd2c94ea6fcbdb5156b3ab82222648e48ce84aebdc100775511d` | `640970486f0581fd24b4a4ff9f2b5c73ae8893a665129357b0f62db35bce7d02` |
| 123 | `b395d863bb67c382b66f002cdc17bfe96e30e4dadebe90005266f113ac139149` | `d68719b70933d1b91d9cec34ddc8300e55c6710df622627c9702db9b6e2932e0` | `6eae99dfd6ac4899e2f5a6c56207389fec653962b303a96aadc6bceab50337df` | `2f2bf9b2f9bfa75ec903656864eaee2eb42b454c966dbe6872c1533d0d2f6193` | `938fca913ce50778e8f7df0f31c90671aac0437ce007034b923ec9a99870074e` |
| 2026 | `5bc1f150bea0e2f76e407ea7f4df80423a54db871255e23dd268fc690d1ff792` | `3623d62c3bc7d4f56771a000f02c08eec36ea91f2dccecd8588599b2490118f9` | `7bf9a3ce69e98915c7c989e26cb3a0decd6fba12901ab348a45e09ea029ea8ec` | `fed3aa9ff056f2f82dfe24ba67b37c25b4d8fb4f3a2ac2eda093d7e0af0c023f` | `5673e6137a82e39294f05b073c1fb2e3f474382d6b741f29581fb4ee1b1079cb` |
| 3407 | `d57c5cde43323fff38bb594bb7716b965bc171a3c598b4bddfb94d1f4c2307d9` | `2248b276a2ac4c380faf890ed8fc6ff2cc8e27bdd90d3d41b60dcf3beee4a5cf` | `cfe3a9def49cb35af4c18e8c5a03ab807c4694f8a48021dd7e6a2b5ff368a7a8` | `e23120af83183f5d967d4761ab92f5f31b98833fed19289569d77122d95efabb` | `3b36a329f78d1098defa6ceda13ab58952364a505b5375405281392760fa41bf` |

Checkpoint templates are:

```text
real_std:    journal/models/generator_extension/cnn_real_only_seed{seed}.pt
rule_ms:     journal/models/generator_extension/cnn_rule_0p30_matchedsteps_seed{seed}.pt
placebo_ms:  journal/models/generator_extension/cnn_placebo_0p30_matchedsteps_seed{seed}.pt
rule_std:    journal/models/generator_extension/cnn_rule_0p30_seed{seed}.pt
placebo_std: journal/models/generator_extension/cnn_placebo_0p30_seed{seed}.pt
```

All checkpoints must load with `weights_only=True`, have the expected 23
state-dict keys, and match the frozen `CNN1D` tensor shapes.

## 4. New matched-budget real-only baselines

The existing `real_std` checkpoints are early-stopped legacy checkpoints and are
not matched to the 6,156-update Rule/placebo runs. Therefore they are not used
as the primary real reference.

Before any E14 scoring, train exactly five new baselines:

```text
journal/models/generator_extension/
  cnn_real_only_matchedsteps_e14_v1_seed{7,42,123,2026,3407}.pt
```

The versioned E14 training wrapper specified below must first verify that all
five checkpoint paths and all five log paths are absent and atomically publish:

```text
journal/experiments/e14_l4_counterfactual_factorial/
  matched_real_training_preflight_v1.json
```

That record captures the pre-training UTC time, source commit/worktree status,
exact child command, executable path, trainer/data hashes, Python/NumPy/
PyTorch/sklearn versions, CPU/thread settings, and the CUDA/driver/cuDNN/device
state used by the trainer. It must be written before the child process starts
and its own SHA-256 must be linked from the post-training run record.

Only after that executable preflight passes may the wrapper launch this exact
frozen child command:

```bash
source .venv/bin/activate
python journal/scripts/train_generator_extension_cnn.py \
  --settings real_only \
  --seeds 7,42,123,2026,3407 \
  --budget-mode matched_steps \
  --max-steps 6156 \
  --model-tag matchedsteps_e14_v1
```

The fixed protocol is AdamW `lr=1e-3`, weight decay `1e-4`, batch size `512`,
balanced cross-entropy, real-train-only standardization, and validation
multiclass macro-F1 checks every 513 updates. Each run executes 6,156 updates
and 12 validation checks; the saved state is the highest validation macro-F1
state, so the selected state can come from an earlier check. “Matched budget”
therefore means the same maximum update budget, validation cadence, and
selection rule—not necessarily the same selected checkpoint step.

The new logs must report `budget_mode=matched_steps`, `optimizer_steps=6156`,
`validation_checkpoints=12`, the exact seed/tag, all 12 history rows, and a
finite selected validation score. Training must use `sampling_policy=legacy`
because the existing `rule_ms` and `placebo_ms` inputs use that lineage.

The output paths are no-clobber. An unexpected pre-existing checkpoint or log is
a technical stop, not permission to reuse or overwrite it. This explicit
preflight is required because the underlying generic trainer would otherwise
print `skip existing` for an existing checkpoint. On success the wrapper
atomically publishes:

```text
journal/experiments/e14_l4_counterfactual_factorial/
  matched_real_training_run_v1.json
```

The post-training record contains the child exit status, start/end times, the
five new checkpoint/log hashes, selected steps, and the preflight-record hash.
It also confirms that the post-run environment/device identity matches the
preflight. Both records and the frozen trainer hash must be linked from
`prepare_v1.json`.

These five fits are the only new detector training permitted in E14. Their
validation metrics may be inspected for the fixed checkpoint-selection rule;
no E14 factorial outcome may be computed during training.

## 5. Frozen bases and units

Read the frozen E8/E13 manifest without resampling. For each of `block_01`,
`block_02`, and `block_03`, select exactly the rows with
`block_position < 2000`, preserving manifest order.

Required properties:

- exactly 2,000 bases per block and 6,000 total;
- 6,000 unique `test_window_index` values;
- 6,000 unique `(source_file,start_index,end_index)` triples;
- `y_binary=0` and `y_attack_type=0` for every base;
- metadata and window bytes match `test_windows.npz`;
- represented raw frames are mutually disjoint across the three blocks.

Every base is reused in all four attack/ID strata and all eight factorial cells.
This is a new paired design: the selected first 2,000 bases are not claimed to
be the same scenario-specific subset that E8's later permutation assigned to
its original L4 rows.

The three blocks are repeated evaluation realizations used for descriptive
block sensitivity. They are not the primary inferential sample. The 6,000
windows and their transformed copies are not independent inferential units.

## 6. Fixed 2×2×2 construction

### 6.1 Factors and cells

| Factor | Level 0: Rule-side anchor | Level 1: L4-side construction |
|---|---|---|
| `P`, payload-role destination | Gear `data0,data1`; RPM `data0,data1,data2` | Gear `data6,data7`; RPM `data4,data6,data7` |
| `S`, frame-mask layout | `start + 2r`, `r=0…31` | `start + 3r`, `r=0…31` |
| `D`, payload dynamics | Rule Gear/RPM role values below | L4 Gear/RPM role values below |

The cells are `000,001,010,011,100,101,110,111`, ordered as `P,S,D`.

- `000` is a conditional, fixed-length, count-matched Rule-support anchor. It is
  not a random draw from the full Rule generator distribution.
- `111` is a field-level reproduction of the current L4 transformation law on
  the new paired bases. It is not a reproduction of E8's original base
  assignment.
- `S0` is conditioned to the common start support `0…32`; the full Rule
  `span64/step2` start support would extend to 64.
- Both `S` levels alter exactly 32 frames. Step 3 is itself Rule-supported; the
  main disclosed support exit is the L4 span argument 96 exceeding the Rule
  training maximum 90. `S` is therefore called a frame-mask layout/span
  contrast, not a generic timing effect or a pure span effect.

For every cell, injected frames receive the target CAN ID, DLC 8, and
`delta_t=max(original_delta_t,1e-5)`. Jitter below is payload jitter only; it is
never applied to the timing feature.

### 6.2 Attack/ID strata and labels

| Attack | Canonical ID | Shifted ID | Exact label |
|---|---:|---:|---:|
| Gear | `0x43F` | `0x440` | 3 |
| RPM | `0x316` | `0x329` | 4 |

All transformed rows have binary label 1. The shifted IDs occur in normal
Car-Hacking traffic and must not be described as unseen IDs. They are
attack-target-ID substitutions. Canonical and shifted pairs share every base,
start, mask, and payload latent and differ only in injected CAN ID.

### 6.3 Deterministic latent construction

The master construction seed is `2026071400`. For each base/attack, generate
order-independent child seeds exactly as:

```python
ss = np.random.SeedSequence([
    2026071400,
    block_number,       # 1, 2, or 3
    test_window_index,
    attack_label,       # Gear=3, RPM=4
    stream_code,        # L4=1, Rule=2
])
child_seed = int(ss.generate_state(1, dtype=np.uint64)[0])
rng = np.random.default_rng(child_seed)
```

Record both child seeds and all derived arrays in the latent manifest. Changing
loop order or adding a stratum must not change an existing row.

The L4 stream (`stream_code=1`) first draws:

```python
start = int(rng.integers(0, 33))
```

and then draws the D1 jitter in the same order as the frozen `inject_l4()`
oracle. The two masks are:

```text
S0 = start + 2 * arange(32)
S1 = start + 3 * arange(32)
```

The Rule stream (`stream_code=2`) supplies D0 role values independently of loop
order. Rank-ordered role vectors are shared across `P`, `S`, and ID levels.

Gear D0:

```text
level ~ choice({0,1,2,3,4,5}, 32)
role0 = clip(level*40 + integers(0,16), 0, 255)
role1 = clip(255-role0 + integers(-8,9), 0, 255)
```

Gear D1:

```text
role0 = float32(linspace(0,255,32))
role1 = clip(255-role0 + integers(-10,11), 0, 255)
```

RPM D0:

```text
rpm ~ integers(0,8000,32)
role0 = (rpm//32) mod 256
role1 = (rpm//4) mod 256
role2 = clip(original base data2 at S0 positions
             (start+2*arange(32)) + integers(-50,51), 0, 255)
```

The RPM D0 `role2` value is constructed once from the Rule-side source role and
the S0 mask, then transplanted unchanged between `P` destinations and `S`
levels. This preserves the shared rank-ordered latent contract while making
`000` agree with the Rule formula.

RPM D1:

```text
role0 = (37*arange(32)) mod 256
role1 = float32(127.5 + 127.5*sin(linspace(0,3*pi,32)))
role2 = clip(float32(linspace(0,255,32)) + integers(-12,13), 0, 255)
```

All transformed windows are `float32`. Payload fields outside the active role
destinations retain the base values.

## 7. Outcomes

For arm `a`, pipeline seed `s`, block `b`, attack `k`, ID stratum `i`, cell
`c=(P,S,D)`, and the 2,000 bases in that block, define exact recall:

\[
Y^E_{a,s,b,k,i}(c)
=\frac{1}{2000}\sum_x
I\{\arg\max_j z_{a,s,j}(T_{k,i,c}(x))=k\}.
\]

Define binary attack recall:

\[
Y^B_{a,s,b,k,i}(c)
=\frac{1}{2000}\sum_x
I\{\arg\max_j z_{a,s,j}(T_{k,i,c}(x))\ne0\}.
\]

The **primary endpoint** is canonical-ID exact recall, computed separately for
Gear and RPM and then equal-weight macro-averaged. A pooled-window micro recall
must not replace this attack-equal macro, even though both attacks currently
have the same row count.

Binary recall is a mandatory companion endpoint. A binary-only result cannot
support exact typing localization.

The untouched 2,000 normal bases per block must also be scored for every
arm/seed. Report normal recall and FPR (`1-normal recall`) before transformed
attack recall. These source-test controls are not external-domain FPR.

### 7.1 Margin sensitivity

Raw logits are not commensurate across independently trained checkpoints.
Cross-arm raw-margin differences are prohibited.

For each checkpoint and target attack, define on the same clean bases:

\[
m^E_k(x)=z_k(x)-\max_{j\ne k}z_j(x),\qquad
m^B(x)=\max_{j\ge1}z_j(x)-z_0(x).
\]

Let `MAD` be `median(abs(m-median(m)))` without a normal-consistency scale
factor. For each transformed row, compute the checkpoint-local shift:

\[
\Delta\widetilde m_{a,s,k,i,c}(x)=
\frac{m_{a,s,k}(T_{k,i,c}(x))-m_{a,s,k}(x)}
{\operatorname{MAD}_{a,s,k,\mathrm{clean}}}.
\]

If the relevant clean MAD is at most `1e-6`, mark
`no clean-margin dynamic range` and omit that checkpoint/metric from this
supporting standardized-margin analysis; preserve all recall outcomes.
If even one of the five seeds for a margin effect fails this gate, do not
compute a reduced-\(n\) seed-level \(t\)-CI. Report only the available values
and `n_available` descriptively.

Margins are estimation-focused supporting sensitivities only. They measure a
checkpoint-local transformation response, not calibrated probability,
absolute confidence, or E10 margin-scale equivalence.

## 8. Arms and fixed estimands

### 8.1 Primary matched-training-budget panel

```text
real_ms     = new cnn_real_only_matchedsteps_e14_v1
rule_ms     = existing cnn_rule_0p30_matchedsteps
placebo_ms  = existing cnn_placebo_0p30_matchedsteps
```

All three use the same 6,156-update maximum budget, 513-update validation
cadence, and best-validation selection rule. `rule_ms` and `placebo_ms`
additionally have the same added count and class quota.

For each cell, compute paired same-seed/same-window contrasts:

\[
A^R(c)=Y_{\mathrm{rule\_ms}}(c)-Y_{\mathrm{real\_ms}}(c),
\]

\[
A^P(c)=Y_{\mathrm{placebo\_ms}}(c)-Y_{\mathrm{real\_ms}}(c),
\]

\[
B(c)=Y_{\mathrm{rule\_ms}}(c)-Y_{\mathrm{placebo\_ms}}(c)
=A^R(c)-A^P(c).
\]

`A^R` is a total equal-budget Rule-augmentation pipeline contrast. It includes
the changed training sample mix and associated class weighting/minibatch
sequence; it is not a pure causal effect of one grammar relation. `B` is the
cleaner equal-added-count/equal-budget relational training contrast, but still
only compares the implemented Rule and placebo constructions.

### 8.2 Legacy standard-harness sensitivity

```text
real_std, rule_std, placebo_std
```

Use the existing early-stopped checkpoints and report their realized total and
selected optimizer steps. This panel bridges E14 to the current paper but does
not open a new confirmatory family.

Its three same-seed/same-window contrasts are:

\[
A^R_{\mathrm{std}}(c)=
Y_{\mathrm{rule\_std}}(c)-Y_{\mathrm{real\_std}}(c),
\]

\[
A^P_{\mathrm{std}}(c)=
Y_{\mathrm{placebo\_std}}(c)-Y_{\mathrm{real\_std}}(c),
\]

\[
B_{\mathrm{std}}(c)=
Y_{\mathrm{rule\_std}}(c)-Y_{\mathrm{placebo\_std}}(c).
\]

For every primary resolved effect, classify its same contrast in this panel:

- `budget-stable`: same sign and nominal 95% CI excludes zero in that direction;
- `directionally-concordant-unresolved`: same sign but CI includes zero;
- `budget-discordant`: opposite sign, or an interval excluding zero in the
  opposite direction.

The legacy panel cannot upgrade an unresolved primary result. Discordance
narrows the claim to the matched-budget protocol.

### 8.3 Factorial contrast operators

For a cell function `f(P,S,D)`, the main-effect operator for `P` is:

\[
C_P[f]=\frac14\sum_{S,D}\{f(1,S,D)-f(0,S,D)\},
\]

with analogous `C_S` and `C_D`.

The `P×S` interaction is:

\[
C_{PS}[f]=\frac12\sum_D[
f(1,1,D)-f(0,1,D)-f(1,0,D)+f(0,0,D)],
\]

with analogous `C_PD` and `C_SD`.

The three-way interaction is:

\[
\begin{aligned}
C_{PSD}[f]
={}&[f(1,1,1)-f(0,1,1)-f(1,0,1)+f(0,0,1)]\\
 &-[f(1,1,0)-f(0,1,0)-f(1,0,0)+f(0,0,0)].
\end{aligned}
\]

`f(1,1,1)-f(0,0,0)` is a separate supporting corner contrast. It is not a
factorial main effect or interaction.

## 9. Aggregation and inferential units

Within each of the five paired pipeline seeds, use this fixed order:

1. compute arm and factorial contrasts within each block and attack;
2. equal-weight average Gear and RPM contrasts;
3. equal-weight average the three blocks;
4. use the resulting five seed-level effects as inferential units.

The primary endpoint uses canonical IDs only. Shifted IDs, attack-specific
effects, and block-first summaries remain separate sensitivities.

For each effect report all five seed values, mean, sample SD, and an
untruncated two-sided 95% Student-\(t\) interval with 4 degrees of freedom.
Block sensitivity instead averages the five seeds within each block and
reports the three block values, mean, SD, minimum, and maximum descriptively.

Do not use the three blocks as `n=3`, seed×block combinations as `n=15`, or
windows/transforms as the inferential sample size. The five seeds are paired
pipeline realizations that jointly vary initialization, sampling, and minibatch
order. Inference is conditional on this selected seed set; it is not universal
population inference over all possible seeds.

If all five values for an effect are identical, retain the point estimate, mark
its variance and CI as `degenerate variance`/`NA`, and make it ineligible for a
`resolved-and-stable` finding. For Holm bookkeeping only, insert conservative
raw \(p=1\); label that value `bookkeeping p`, not an inferential test result.

## 10. Multiplicity

Because an interaction-only result may change the scientific verdict, the
primary exact family contains all seven factorial effects:

\[
\mathcal F_E^R=
\{C_P[A^R],C_S[A^R],C_D[A^R],
C_{PS}[A^R],C_{PD}[A^R],C_{SD}[A^R],C_{PSD}[A^R]\}.
\]

For each effect, use a two-sided one-sample \(t\)-test of \(H_0:\mu=0\) on the
five seed values and jointly adjust the seven raw \(p\)-values using Holm's
method. Report mean, SD, **nominal untruncated** 95% \(t\)-CI, raw \(p\), and
Holm-adjusted \(p\). Do not call the CI Holm-corrected.

The matched Rule-minus-placebo exact effects
`{C_j[B]: j=P,S,D,PS,PD,SD,PSD}` form a separate, preregistered secondary
seven-test Holm family. This family cannot rescue or replace the primary
Rule-minus-real family.

The seven canonical binary `C_j[A^R]` effects form a separate mandatory
companion Holm family. Exact and binary outcomes must not be combined.

The following are estimation-focused supporting analyses with nominal CIs only:

- placebo-minus-real exact and binary effects;
- matched Rule-minus-placebo binary effects;
- `111-000` corner contrasts;
- shifted-ID and canonical-minus-shifted effects;
- Gear- and RPM-specific effects and attack heterogeneity;
- block-first summaries;
- standardized-margin effects;
- every legacy standard-harness contrast.

They must not be promoted post hoc to confirmatory findings. No SESOI or
equivalence band is registered, so `no effect`, `equivalent`, and `material
effect` verdicts are prohibited.

## 11. Scientific verdicts

An effect is `resolved-and-stable` only if it is nondegenerate, its
Holm-adjusted \(p<0.05\), and at least four of five seed effects have the same
strict sign as its mean. If the mean is positive, at least four values must be
strictly greater than zero; if the mean is negative, at least four must be
strictly less than zero. An exact zero counts toward neither direction.

The primary exact family alone determines the E14 localization verdict:

- `F-MAIN`: at least one `P`, `S`, or `D` main effect is
  `resolved-and-stable`;
- `F-INTERACTION`: no main effect passes that rule, but at least one two- or
  three-way interaction does;
- `F-MIXED`: a Holm test is significant but its direction is not repeated in at
  least four of five seeds, and no effect is `resolved-and-stable`;
- `F-UNRESOLVED`: no nondegenerate effect is Holm-significant.

If one or more effects have degenerate variance, append
`+DEGENERATE(effect names)` to the applicable verdict. Degeneracy is a
disclosed modifier, not a separate verdict that can overlap `F-MIXED` or
`F-UNRESOLVED`.

If a main effect and an interaction are both resolved, the main effect is only
the marginal effect averaged over the other two factors; do not claim that the
response is concentrated in one factor. `F-INTERACTION` licenses only a
compound construction response, not an individual-factor mechanism.
`F-UNRESOLVED` means localization was inconclusive, not that factors have no
effect.

If Gear and RPM point effects have opposite signs, append
`attack-heterogeneous` and do not describe the macro effect as shared by both
attacks.

A resolved Rule-minus-placebo secondary result permits only:

> The matched Rule and placebo training arms exhibited different responses to
> the implemented construction contrast in this frozen pool realization.

It does not establish realism, a physical mechanism, or E10's unresolved
margin-scale structure claim. Every Rule E14 conclusion must state
`conditional on Rule construction seed 314159`; Rule-minus-placebo conclusions
must additionally disclose placebo permutation seed `1618033`.
Construction-seed replication requires E15. If a Rule-minus-placebo result is
elevated beyond supporting evidence, E15 must also build a paired placebo twin
and placebo fits for each Rule construction realization; otherwise that result
remains a single-construction sensitivity.

## 12. Technical gates

### 12.1 Freeze and provenance

Before training the new baselines:

- this preregistration is committed;
- the wrapper, evaluator, analyzer, and tests are committed and the tracked
  source paths are clean;
- `wisa/` is unchanged;
- the frozen trainer/data/reference hashes match Section 3;
- the executable wrapper proves that the five target checkpoints and five
  target logs do not exist and captures the actual pre-training environment.

Before preparation:

- all 30 checkpoint/log bundles exist and validate;
- both matched-real training records exist, cross-link, and validate;
- the 25 existing checkpoint hashes match Section 3;
- all five new runs report the fixed training protocol;
- every checkpoint architecture and tensor is finite and compatible;
- train/test/manifest/run-record and derived standardizer hashes match;
- source commit, worktree status, Python/NumPy/PyTorch/sklearn versions, device,
  CUDA/cuDNN information when applicable, and thread settings are recorded.

### 12.2 Stage A — manipulation-only preparation

`--prepare-only` must not load PyTorch checkpoints or expose predictions. It
must publish, atomically and without overwrite:

```text
journal/results/tables/e14_l4_factorial_base_manifest_v1.csv
journal/results/tables/e14_l4_factorial_latent_manifest_v1.csv
journal/results/tables/e14_l4_factorial_manipulation_checks_v1.csv
journal/experiments/e14_l4_counterfactual_factorial/prepare_v1.json
```

Required manipulation checks:

- exactly 6,000 base rows and 12,000 base×attack latent rows;
- `0 <= start <= 32` for every latent row;
- `S0 == start+2*arange(32)` and `S1 == start+3*arange(32)`;
- exactly eight cells and 32 injected frames per transform;
- exactly 192,000 unique scenario keys, each mapped to exactly one generated
  window: `6000 bases × 2 attacks × 2 IDs × 8 cells`; byte-level uniqueness
  among generated windows is not required;
- all cells and strata preserve base ordering;
- identical D has identical rank-ordered role values across P/S/ID;
- with P/D fixed, S changes only the frame mask and consequent injected-frame
  fields;
- with S/D fixed, P changes only the union of the two role-destination sets;
- with S/P fixed, D changes only role payload values;
- canonical/shifted pairs differ only in injected CAN ID;
- target ID, DLC 8, timing floor, label, feature range, finiteness, and
  DLC-conditioned padding checks pass;
- Gear/RPM D0 and D1 role ordering/formulas pass separately;
- payload jitter never changes `delta_t`;
- on toy bases with identical child seeds, cell `111` is byte-exact to the
  frozen `inject_l4()` oracle;
- cell `000` passes the conditional Rule formula/protocol checks but is not
  labeled a full Rule-generator draw;
- repeated regeneration produces the same latent, scenario, and transformation
  digest.

The transformation-instance key is:

```text
(block_id, block_position, test_window_index,
 attack_label, id_stratum, P, S, D)
```

Sort lexicographically by that tuple, with integer fields sorted numerically.
For each row, serialize the key as compact ASCII JSON of the ordered eight-item
list (`separators=(",",":")`, no whitespace), prefix it with its unsigned
little-endian 64-bit byte length, then append:

```text
ASCII dtype token "<f4"
little-endian uint32 shape values 128 and 11
contiguous little-endian float32 C-order window bytes
```

Update one SHA-256 stream with those bytes for all 192,000 instances. Record
the digest and this serialization version in `prepare_v1.json`. Stage B must
regenerate all transforms and reproduce the digest before publishing scores.

Any failed manipulation gate yields `T-STOP-MANIPULATION`; checkpoint loading
and scoring are prohibited until a disclosed amendment and new version pass.

### 12.3 Stage B — complete scoring

The canonical run uses the six arms (`real_ms`, `rule_ms`, `placebo_ms`,
`real_std`, `rule_std`, `placebo_std`), five seeds, three blocks, two attacks,
two ID strata, and eight cells.

Expected completeness:

- 30 checkpoint evaluations;
- 192,000 unique scenario keys, each mapped to one transformed window;
- 5,760,000 transformed forward evaluations;
- 180,000 clean-base forward evaluations;
- 90 `arm × seed × block` normal-control rows;
- 2,880
  `arm × seed × block × attack × ID × cell` scenario rows;
- 1,440 primary-panel scenario rows, of which 720 are canonical-ID rows;
- 1,440 attack-macro `arm × seed × block × ID × cell` rows;
- 2,880 same-seed scenario contrast rows for the three matched and three legacy
  pairwise arm contrasts;
- 15,552 fully keyed effect rows as specified below.

The uniqueness key for scenario rows is:

```text
(arm, seed, block, attack, id_stratum, P, S, D)
```

The six effect-table contrasts are `rule_ms-real_ms`,
`placebo_ms-real_ms`, `rule_ms-placebo_ms`, `rule_std-real_std`,
`placebo_std-real_std`, and `rule_std-placebo_std`. The effect table is the
complete Cartesian product:

```text
6 contrasts
× 4 endpoints
    {exact_recall, binary_recall,
     exact_standardized_margin_shift, binary_standardized_margin_shift}
× 3 ID scopes {canonical, shifted, canonical_minus_shifted}
× 3 attack scopes {Gear, RPM, equal_macro}
× 8 effects {P, S, D, PS, PD, SD, PSD, corner_111_minus_000}
× 9 row units {five seed rows, three block rows, one summary row}
= 15,552 rows
```

Its uniqueness key is:

```text
(contrast, endpoint, id_scope, attack_scope, effect, row_type, unit_id)
```

Seed rows average the three blocks; block rows average the five seeds and are
descriptive; the summary row reports inference across the five seed rows.
Margin-gate failures do not delete rows: the affected value is `NA` with a
status field. If any seed is unavailable, the margin summary reports available
values and `n_available` without a \(t\)-CI, as fixed in Section 7.1.

Required outputs are:

```text
journal/results/tables/e14_l4_factorial_normal_by_seed_v1.csv
journal/results/tables/e14_l4_factorial_by_cell_v1.csv
journal/results/tables/e14_l4_factorial_by_scenario_v1.csv
journal/results/tables/e14_l4_factorial_augmentation_delta_v1.csv
journal/results/tables/e14_l4_factorial_effects_v1.csv
journal/results/logs/e14_l4_factorial_v1.log
journal/experiments/e14_l4_counterfactual_factorial/matched_real_training_preflight_v1.json
journal/experiments/e14_l4_counterfactual_factorial/matched_real_training_run_v1.json
journal/experiments/e14_l4_counterfactual_factorial/run_v1.json
journal/results/logs/e14_l4_factorial_artifact_manifest_v1.json
```

All outputs are atomic and no-clobber. Full publication occurs only after every
cell, arm, seed, block, finite-value, row-count, pairing, and digest gate
passes. A partial run remains preserved in an isolated temporary/versioned
location and cannot support a scientific verdict.

Do not store one monolithic 192,000-window NPZ. Regenerate cell batches from
the frozen base/latent manifests. Do not create the final figure or update the
paper artifact manifest until all tables and run records validate.

Technical status is:

- `T-PASS`: every provenance, manipulation, checkpoint, standardizer,
  completeness, digest, and no-clobber gate passes;
- `T-STOP-MANIPULATION`: a pre-scoring construction gate fails, so inference is
  forbidden;
- `T-INCOMPLETE`: scoring began but a checkpoint/cell/nonfinite/output gate
  failed; preserve partial artifacts and issue no scientific verdict.

Floor, ceiling, mixed-sign, null, or interaction-only outcomes are valid
scientific results, not technical failures.

## 13. Amendments, failures, and exclusions

- There are no outcome-based seed, block, attack, ID, cell, or arm exclusions.
- A missing/nonfinite row is a technical failure; it is not silently dropped.
- Technical changes before any score exists require a dated amendment and new
  code hash. If a versioned output already exists, use a new output suffix.
- After any E14 score is observed, endpoints, families, aggregation,
  multiplicity, and verdict rules cannot change for the canonical analysis.
- A failed run may be resumed only when existing checkpoint/log pairs validate
  byte-for-byte and the resume is recorded. Nothing is overwritten.
- An unresolved or adverse result is reported as registered. It is not rerun
  with new seeds, bases, thresholds, factors, or endpoint definitions.
- No external dataset, threshold tuning, model selection, new detector family,
  or E15 construction seed may enter E14.
