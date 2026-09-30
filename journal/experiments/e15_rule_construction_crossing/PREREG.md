# E15 preregistration — Rule construction seed × detector pipeline seed crossing

Registration time: 2026-07-24T19:50:23Z

Status: **PRE-RESULT for the four new Rule constructions; E15 pool generation,
training, and scoring are prohibited until this document is committed and the
freeze gates in Section 13 pass**

At registration time, no E15 pool, checkpoint, training log, outcome table, or
figure existed. E14 outcomes for Rule construction seed `314159` are already
known. This registration is therefore prospective for four new construction
realizations and their crossed E15 results, not an independent registration of
five previously unseen constructions. Any later design change requires a dated
amendment committed before the affected stage; it must not silently edit this
record.

## 1. Question and evidential scope

E14 localized the matched-budget Rule-minus-real augmentation response within
one Rule pool, construction seed `314159`. E15 asks:

> After fully crossing four additional prespecified Rule-pool constructions
> with the five fixed detector-pipeline seeds, do E14's canonical exact-recall
> factorial contrasts show a directionally repeated marginal association with
> the implemented `P`, `S`, and `D` construction factors or their interactions?

E15 varies the pseudorandom pool realization while holding the same disclosed
Rule construction law fixed. It does not compare generator families or
establish a population law over all possible generators, seeds, vehicles,
attacks, or detector pipelines. The pools are called **seed-distinct Rule-pool
realizations**, not independent datasets, independent generators, or
independent attack mechanisms.

All conclusions are conditional on:

- the frozen Car-Hacking train/validation/test split;
- the disclosed 65,000-window-per-class Rule construction law;
- four new construction seeds plus one already observed historical anchor;
- the five selected paired detector-pipeline seeds;
- the E14 evaluation constructions and three descriptive blocks;
- the fixed matched-training-budget CNN protocol and shared E14 real-only
  references.

E15 cannot establish realism, physical payload causation, replacement of real
attacks, universal grammar behavior, external-domain robustness, or a
construction-independent generator effect. It contains no external dataset
evaluation.

### 1.1 Known E14 anchor

E14's only resolved primary exact effect was the payload-role-destination main
effect:

```text
C_P[Rule-real] = -0.879958
SD across five paired pipeline seeds = 0.191046
nominal 95% t CI = [-1.117173, -0.642744]
Holm-adjusted p = 0.003509
strict signs = 0 positive / 5 negative
```

The companion binary value was `-0.746033`, SD `0.187221`, nominal CI
`[-0.978499,-0.513568]`, Holm-adjusted `p=0.006139`, with five negative
pipeline-seed effects. These known values fix the E14 reference direction but
do not enter E15's four-new-construction primary test.

E15 will construct and train a fresh, E15-tagged `314159` anchor for uniform
lineage and implementation-continuity checks. It is labelled
`fresh_known_construction_anchor`; its outcomes remain historical-anchor
evidence because the corresponding E14 pool, pipeline seeds, and scientific
endpoint have already been observed.

### 1.2 Excluded Rule-minus-placebo claim

E15 does **not** register a Rule-minus-placebo family. A valid construction-
replication contrast would require a marginal-matched placebo twin and matched
placebo fits for every Rule construction. Reusing the single E14 placebo
permutation seed/pool against four new Rule pools would break row-level and
marginal pairing and confound Rule construction with placebo mismatch.

The E14 Rule-minus-placebo result therefore remains an inconclusive,
single-construction sensitivity. No fixed-placebo E15 comparison may be called
paired, marginal-matched, confirmatory, or evidence of a Rule-specific
relation.

## 2. Mandatory execution order

1. Commit this preregistration after independent statistics, provenance, and
   claim-boundary review.
2. Implement the generator wrapper, training wrapper, evaluator, analyzer, and
   toy-data tests without generating an E15 pool or loading a model.
3. Commit the tested implementation and record its clean source commit.
4. Run a pool-generation preflight that proves every versioned pool,
   statistics, and generation-log target is absent.
5. Generate the five E15 pools, validate them without model inference, and
   publish the pool-generation record.
6. Pass the `314159` anchor-array equivalence gate and all five pool,
   provenance, uniqueness, and no-replacement capacity gates.
7. Run a training preflight that proves all 25 checkpoint/log pairs (50 file
   targets) are absent, validates the frozen E14 real references, and records
   the execution environment.
8. Train all 25 E15 Rule fits under the fixed protocol. Training may inspect
   the preregistered validation score used for checkpoint selection but no E14
   factorial or L2 outcome.
9. Run `--prepare-only`. It may regenerate and hash transformations and audit
   checkpoints, but it must not perform model inference.
10. Commit the generation, training, and preparation records without changing
    the frozen implementation.
11. Before using any E15 Rule checkpoint for inference, rescore the five frozen
    real checkpoints on the complete L4 panel and require exact integer-count
    continuity with E14.
12. After that gate passes, score the complete L4 Rule factorial and supporting
    L2 panel once.
13. Analyze and publish all complete results regardless of direction.
14. Create a figure, update the paper-artifact manifest, and revise the
    manuscript only after a complete machine-readable scientific verdict
    exists.

No canonical pool construction, training, or scoring is allowed from a dirty
tracked/source tree. Once a canonical stage preflight is published, that
version is one-shot: a failed or interrupted generation/training stage is not
resumed under the same output version. Section 13.6 governs preservation and a
prospective versioned amendment; nothing is overwritten.

## 3. Frozen seed axes and roles

### 3.1 Construction seeds

Historical anchor:

```text
314159
```

New primary construction realizations:

```text
271828, 161803, 141421, 173205
```

The four new seeds, in the order above, are the primary construction-level
inferential units. The anchor is displayed separately and in an explicitly
labelled five-realization descriptive summary. It is never inserted into the
primary `n=4` test or described as a fifth independent replication.

### 3.2 Detector-pipeline seeds

```text
7, 42, 123, 2026, 3407
```

A pipeline seed jointly controls:

- the synthetic-subset draw through sampling seed `pipeline_seed + 300`;
- Python, NumPy, and PyTorch fit-state initialization;
- CNN initialization, dropout, DataLoader shuffle, and minibatch order.

Each `(construction_seed,pipeline_seed)` fit runs in a separate process and
reinitializes the complete pipeline seed. The construction seed controls only
normal-background selection, injector draws, and final pool permutation during
pool construction. Code must not add, hash, nest, or otherwise combine the two
seed axes into one RNG seed.

The full training grid is `5 constructions × 5 pipelines = 25` fresh Rule
fits. It is not `n=25` generator replication.

### 3.3 Evaluation blocks

The three frozen blocks are `block_01`, `block_02`, and `block_03`. They are
mutually raw-frame-disjoint evaluation realizations used for pairing and
descriptive block sensitivity. They are averaged before construction-level
inference and are never counted as `n=3`, `n=12`, `n=15`, `n=60`, or `n=75`
inferential replicates.

## 4. Frozen inputs and historical evidence

### 4.1 Data, construction, and evaluation inputs

| Input | SHA-256 |
|---|---|
| `journal/datasets/windows/train_windows.npz` | `44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4` |
| `journal/datasets/windows/val_windows.npz` | `b4920e7f60fbf38b38e12976129f120ebff160a21a4c77093f3746f3c1aa218e` |
| `journal/datasets/windows/test_windows.npz` | `4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7` |
| historical `journal/datasets/synthetic/rule_based_windows.npz` | `4078ab00b8531f760b296b5fd293e3df914bef996d317c958ead4762137bafd5` |
| frozen `wisa/scripts/generate_rule_based_synthetic.py` reference | `df7fa574efa2b2bd94daa846970fbcd7beb7c6b6ecf13ecd2c6a5b2855cc5f69` |
| `journal/scripts/generate_rule_based_synthetic_v2.py` | `34ff0374af1856fd842da8226580d5ff0d47cbd16b36fb4273de3e06bcac2570` |
| `journal/scripts/generator_protocol.py` | `f23c52ad9f08673c365a0db158d8c6e2db8213b5d30c2584ac777e066a4f85c0` |
| `journal/scripts/train_generator_extension_cnn.py` | `a5d88bcb579da77e234e991a9b050c040df21d2f75e971c0c952c6226749db61` |
| `journal/scripts/lib_common.py` | `5848ff4b0063d4394cd1d6627faf49e072cb363d7796cc64fce3acc48b05639a` |
| `journal/scripts/evaluate_evaluation_realization.py` | `eadf93df237946b7f670a74e316f0b80fc1d1920d4cc7072c7434c699a2ff2bb` |
| E8/E13 block manifest | `9e18b3844e20f259938c732a10bc0251767f0d2eaae9c731ba8da7ef087f5efb` |
| E8/E13 prospective run record | `fae2ca82bb26ab24a5699fce9f331951be23746093fef9e28d0b240fd2c3b2db` |

The new E15 implementation may call audited construction functions, but it
must publish an E15-specific schema and identifier. It must not weaken or
replace the canonical strict-v2 configuration, its `314159` seed enforcement,
or its 65,538-per-class identities.

### 4.2 Frozen E14 factorial lineage

| Input | SHA-256 |
|---|---|
| E14 `PREREG.md` | `f0465fea952b9f7d5fd1b20a7864a96192099b9d3346f6c81e4274ce0fe0424a` |
| E14 `prepare_v1.json` | `5945790bee2e0bf11739901e0fdd9237d499353d9d8ead6b34de06fe0311a16e` |
| E14 matched-real preflight | `3b895fe3c84e5188968fedf0ef0450f2ace03c98e3fe1e3ea11aa3a8f98e1e14` |
| E14 matched-real training run | `53865949ecf4383fa6f9a21c224b76c289baeb9c88daa7fbf80eddfacfe0de28` |
| E14 matched-real child transcript | `e362b71ca4cdff7755e27b16ca614ca2f1e574c12c8fc95777aeb095a631e0e5` |
| E14 score run | `3f72bfcbefec16705c25208f91e50f4e3834110353fa5ca0458401add6eb6938` |
| E14 artifact manifest | `41243eca977c5acf4b115b9f30d7a9c905d08867b23c3d5751d7e8a02246897b` |
| E14 base manifest | `0852d9d4281c309fc1f5926935ebe97b00aff312396c0163e32ab4a93f2d418d` |
| E14 latent manifest | `07e300706a4dc91f3e08e621967e0bb45d875b0ef97a1ec271145beaf1a25281` |
| E14 manipulation checks | `462e228f64303a7d01703c5c206cabca23c956a44ca45db9878e1104bae61ddc` |
| E14 normal controls | `84f6ef7f6722a7fda7d069bb38aa05744144afef75d6537bba4581a42e257849` |
| E14 scenario table | `833ee70d7dd74e75d402783ec0956abbe065a083b62ec44b09c3163361392843` |
| E14 effects table | `63ea3c4bb09167033c1b2c8d7760c2329fe701de6608b274d26a6cb357e4c21f` |

The E14 transformation identity is:

```text
base rows                         6,000
latent rows                      12,000
scenario instances             192,000
scenario-key SHA-256  be55a8259ea045cd9a8107de72d248d220c3d06e6b83f70f2202fabb8c0e1be5
transform SHA-256     52b0ad4fe9365d6f0793d6d14e847bfe3e07f974aa737d861e75b4ea9228d56c
serialization         e14-transform-instance-le-f4-v1
```

E15 Stage A must reproduce these identities before any L4 checkpoint is loaded
for inference.

### 4.3 Shared E14 matched-real checkpoints

Only the following five E14 matched-real checkpoints are reused.

| Pipeline seed | Checkpoint SHA-256 | Training-log SHA-256 |
|---:|---|---|
| 7 | `2eccea8cfc2d734602b8d70f0cd728b7b9c311e262dc51bdec4e7c9957f9fc32` | `91a8c9b5338daf5f56274c595b1951677344d758c450d3e7e2dc69f00462fe8c` |
| 42 | `96df37d40ea9afd7694d6e15f6c38de4c18283e7e2ca2fa8b60207fcb9d4e32d` | `918ded24729fccdb65e8925e0f982e33e2d47a34d3623e1097d236c20ff115e5` |
| 123 | `b32a663c646ccfd50538af024ec2107cf167462d814487287359301ff2ee0ebb` | `6b2caa3e895f4e362bf006ce552c8f55db452293e62acffc23334d4f7691c8b2` |
| 2026 | `1020d15f99c0de5335633ce9de97f53485faf281fc3e5cafa1182a993e6d7a94` | `5e32902651194ae8fcd729fb969a2f22680e0f8be4ec9f834280009b091581ed` |
| 3407 | `d0005ebfbef6562450a70e0c8e4d9ef5769e792f66d6725340fa551a9a673a88` | `ac75d7e8f81fba7e1b0a792f654387afe77561f24c74eb590a1c22e111024b37` |

Every checkpoint must load with `weights_only=True`, contain the expected 23
state-dict keys and shapes, and have finite tensors. E14's full environment
identity,
`7994e84f061901278df487cc02d27c7ae30ea75e09060542f0e2f96c9b32b85a`,
includes process IDs and therefore must not be used as a cross-process equality
gate.

Instead, remove the top-level `process` and `identity_sha256` fields from the
E14 environment object and hash its canonical JSON
(`sort_keys=True,separators=(",",":"),ensure_ascii=True`, UTF-8). The frozen
stable projection is 5,059 bytes with SHA-256:

```text
0fa13120b151b1c8bcb5a0c68ea7a0846f3b4c19a8bcc7d895393e030d9c0cf9
```

The E15 pool-generation and training preflights each record their own full
environment identity and must match this stable projection or stop before the
affected stage for a prospective technical amendment. In particular, pool
generation requires NumPy `2.4.6`, `sys.byteorder == "little"`, and
`np.random.default_rng(...).bit_generator` with class exactly
`np.random.PCG64`.

The real-train-only standardizer is recomputed, never refit on a synthetic
pool:

| Array | SHA-256 of contiguous little-endian `float32` bytes |
|---|---|
| mean | `1e53f10b4dfb1de8522513ca20c7d0c2b5a19a8de8f76067e4273a93467da8a2` |
| standard deviation | `092f42e9a1019fe709233a91c6e876e976f8b3b8616807b42eb6e6da7b7dbb7c` |
| mean then standard deviation | `9019b1a91c983859e8dd998717eb5c7b612ed03dc2a65ec63402fd54ab42ef4e` |

## 5. E15 Rule-pool construction

### 5.1 Fixed construction law

For each construction seed, build exactly 65,000 windows for each attack class
in the fixed order `DoS, Fuzzy, Gear, RPM`, for 260,000 windows per pool and
1,300,000 windows across the five pools.

The construction uses:

- only normal windows from the frozen Car-Hacking train split;
- NumPy `2.4.6` and `np.random.default_rng(construction_seed)` backed exactly
  by `np.random.PCG64`;
- a little-endian host (`sys.byteorder == "little"`);
- the frozen Rule injector formulas and RNG-consumption order;
- source-normal selection with replacement, as in the historical Rule
  generator;
- one final RNG permutation after all four classes are constructed;
- little-endian `float32` (`<f4`) windows of shape `(N,128,11)`;
- one-byte signed `int8` (`|i1`) binary and exact attack labels;
- little-endian `int16` (`<i2`) injection counts;
- window size 128, stride 32, and the frozen feature order.

Source-normal reuse during **pool construction** is allowed and reported. It
must not be confused with the prohibited replacement of synthetic indices by
the **training consumer**.

Every pool must have:

- exact class counts `{1:65000,2:65000,3:65000,4:65000}`;
- finite and protocol-valid frames;
- labels and condition metadata aligned with every row;
- zero exact duplicate generated-window content within the pool;
- the declared construction seed in filename, metadata, configuration, and
  generation log;
- the train-source hash, generator/source-code hashes, clean source commit,
  class/content audit, and ordered-content digest embedded or cross-linked.

All five ordered-content digests and file hashes must be distinct. Cross-pool
row overlap is reported descriptively and is not itself a failure unless two
complete pools are identical.

### 5.2 E15-specific identity

The E15 schema and identifiers are:

```text
schema:      e15_rule_construction_pool_v1
identifier:  e15_rule_cseed<construction>_v1
policy:      e15_strict_without_replacement_0p30_v1
```

Canonical pool paths are:

```text
journal/datasets/synthetic/
  rule_cseed314159_windows_v1.npz
  rule_cseed271828_windows_v1.npz
  rule_cseed161803_windows_v1.npz
  rule_cseed141421_windows_v1.npz
  rule_cseed173205_windows_v1.npz
```

The generator publishes one statistics CSV and one generation JSON per pool,
plus:

```text
journal/results/tables/
  e15_rule_cseed<construction>_statistics_v1.csv
journal/results/logs/
  e15_generate_rule_cseed<construction>_v1.json
journal/experiments/e15_rule_construction_crossing/
  pool_generation_preflight_v1.json
  pool_generation_run_v1.json
journal/results/tables/
  e15_rule_construction_pool_audit_v1.csv
```

The 65,538-per-class E13 strict-v2 pool must not be truncated, substituted,
aliased, or silently selected. No global legacy/default pool path is changed.

### 5.3 Historical-anchor equivalence

The new E15-schema `314159` pool must reproduce the historical 65,000/class
scientific arrays:

```text
x
y_binary
y_attack_type
injection_count
synthetic_type values
feature_names values
window_size
stride
```

`x`, labels, and `injection_count` require `np.array_equal`; string arrays
require value equality because the historical archive uses object dtype while
the new schema may use fixed-width Unicode. The historical ordered-content
digest is:

```text
9c05045481fc235ea58d85709295996a99c6842c39053332507c6d7ebfbbbc43
```

It is SHA-256 over, in stored row order, one unsigned attack-label byte
followed by the raw 32-byte SHA-256 digest of
`np.ascontiguousarray(row,dtype="<f4").tobytes(order="C")` for each `(128,11)`
row. There is no header or delimiter; native-endian serialization is
forbidden.

Whole-NPZ byte equality is neither expected nor accepted as the scientific
gate because E15 adds versioned metadata. Anchor-array mismatch is
`T-STOP-POOL`; it is not permission to reuse the legacy file or continue with
a changed construction law.

### 5.4 Atomic publication

Pool, statistics, and log bundles are no-clobber. The publisher must:

- use `lexists()` so files, symlinks, and dangling symlinks all collide;
- stage every bundle member on its target filesystem;
- publish with exclusive links and directory `fsync`;
- roll back only links created by the current failed publication;
- refuse `--allow-overwrite`;
- preserve an incomplete stage in an isolated temporary/versioned location.

The existing v2 generator's permissive overwrite option is not exposed by the
E15 wrapper.

## 6. Synthetic consumption and detector training

### 6.1 Fixed no-replacement draw

The real train split has 262,149 windows. At ratio `0.30`:

```text
synthetic total requested = round(262149 × 0.30) = 78,645
class 1 requested = 19,662
classes 2,3,4 requested = 19,661 each
```

For pipeline seed `p`, use sampling seed `p+300`, class order
`[1,2,3,4]`, `np.random.default_rng(p+300)`, per-class
`rng.choice(..., replace=False)`, concatenation in class order, and one final
`rng.shuffle`.

For every `(construction,pipeline)` bundle:

```text
requested = drawn = unique
repeated = 0
total unique indices = 78,645
```

The anchor index-vector hashes below are SHA-256 over contiguous little-endian
signed `int64` bytes with no header:

| Pipeline seed | Sampling seed | Anchor index SHA-256 |
|---:|---:|---|
| 7 | 307 | `4ff00984e1dd97a48447b2abac6c347eb8967b20e3c90782db081f9681c11581` |
| 42 | 342 | `bdfcbdf15dcd5001122de17dd721508c5347353a022becd20df94c0e59a0f917` |
| 123 | 423 | `22bd0a549e33ad17662a6b5ac84065a8b229768aace1c5721ece4a2408255ba5` |
| 2026 | 2326 | `59a6f31755fb18e7f19c4ba9e819b050f4d5db617854e0f2c2e78f4ae31a3319` |
| 3407 | 3707 | `3ed08fb8b14d3796264fba3cdac76ca7563fd0d951ad295209d0798fd875e3bf` |

Each new construction records its own index hash. An explicit pool path,
identifier, construction seed, schema, configuration, generation-log hash,
pool file hash, and content hash are mandatory. A missing override is an
immediate failure; no legacy or strict-v2 fallback is allowed. The E15
consumer independently revalidates metadata, frame protocol, condition
metadata, class counts, exact-content uniqueness, source hashes, and
generation-log binding instead of relying on the generic strict-v2 override
path.

### 6.2 Fresh 25-fit grid

No historical E14 Rule checkpoint is reused. Its training logs lack the
complete pool hash, sampling audit, clean source commit, and execution-
environment attestation needed to mix it with new fits without confounding
construction seed and training lineage.

Train exactly:

```text
journal/models/generator_extension/
  cnn_rule_0p30_cseed<construction>_matchedsteps_e15_v1_seed<pipeline>.pt
```

for every construction and pipeline seed in Section 3. Corresponding logs use:

```text
journal/results/logs/
  train_cnn_rule_0p30_cseed<construction>_matchedsteps_e15_v1_seed<pipeline>.log
```

The fixed protocol is:

- CNN1D with the E14 23-key architecture;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `512`;
- balanced multiclass cross-entropy;
- standardization from the complete real train split only;
- exactly 6,156 optimizer updates;
- validation every 513 updates, exactly 12 checks;
- checkpoint selection at the first maximum validation multiclass macro-F1;
- no threshold tuning and no E14/L2 outcome-based model selection.

The selected checkpoint may come from an earlier validation check. “Matched
training budget” means the same maximum updates, validation cadence, and
selection rule, not the same selected step.

The wrapper first atomically publishes:

```text
journal/experiments/e15_rule_construction_crossing/
  training_preflight_v1.json
```

It proves all 50 checkpoint/log targets absent; validates all pools, indices,
real inputs, trainer, standardizer, source commit, environment identity, and
device; and records every exact child command. The fixed execution order is
construction order from Section 3.1, then pipeline order from Section 3.2.
Order must not affect results because every fit is a separate process with a
fresh pipeline seed.

After successful completion it atomically publishes:

```text
journal/experiments/e15_rule_construction_crossing/
  training_run_v1.json
journal/results/tables/
  e15_rule_construction_sampling_audit_v1.csv
  e15_rule_construction_training_manifest_v1.csv
```

The run record binds the preflight, environment before/after, all pool/index/
checkpoint/log hashes, selected steps, validation histories, and source
commit. Generic `skip existing` output is never accepted as completion
evidence.

## 7. Evaluation constructions and outcomes

### 7.1 Primary E14 L4 factorial

E15 reuses the frozen E14 base/latent manifests and deterministic
transformations without changing a base, attack, ID, factor, or latent. The
factors remain:

```text
P = payload-role destination
S = frame-mask layout/span
D = payload dynamics
cells = 000,001,010,011,100,101,110,111
```

The primary evaluation has:

- 6,000 base windows in three 2,000-window blocks;
- Gear and RPM exact labels;
- canonical and shifted-ID strata;
- all eight factorial cells;
- the same 192,000 unique transformation instances as E14.

For construction `g`, pipeline `p`, block `b`, attack `a`, ID stratum `i`,
cell `x`, and its 2,000 transformed windows, define:

\[
Y^E_{g,p,b,a,i}(x)
=\frac1{2000}\sum_r I\{\widehat y_{g,p}(r)=a\},
\]

\[
Y^B_{g,p,b,a,i}(x)
=\frac1{2000}\sum_r I\{\widehat y_{g,p}(r)\ne0\}.
\]

The **primary endpoint** is canonical-ID exact recall, equal-macro averaged
over Gear and RPM after within-attack effects are computed. Binary recall is a
mandatory companion endpoint and never substitutes for exact typing.

Each untouched 2,000-window base block is also scored for every new Rule
checkpoint. Report normal recall and FPR before transformed attack recall.
These are source-test controls, not external-domain FPR.

Before any E15 Rule checkpoint is loaded for inference, the current
evaluator/runtime must rescore all five frozen `real_ms` checkpoints on the
complete L4 scenario panel and untouched normal bases. The resulting 480
scenario rows and 15 normal rows must match the E14 tables frozen in Section
4.2 at the integer correct-count level: exact-correct and binary-correct counts
for scenario rows, and normal-correct counts for control rows. No tolerance is
allowed. Recall and FPR are recomputed from those counts and denominators.

Only after this runtime-continuity gate passes are E14's validated `real_ms`
scenario and normal values reused once and joined to every construction at the
same pipeline seed, block, attack, ID, and cell. The continuity rescore is a
technical equality check; it is not copied into new inferential observations
or counted as five additional real replicates. Any nonzero count difference is
`T-STOP-RUNTIME`, prohibits E15 Rule inference, and requires a prospective
technical amendment.

Preflight and Stage A may hash a Rule checkpoint and load it with
`weights_only=True` solely for structural and finiteness validation. Such
validation performs no forward pass and is not model inference.

Raw cross-checkpoint logits and margins are not comparable. E15 registers no
new margin family and does not alter E10 or E14 margin conclusions.

### 7.2 Supporting L2 continuity panel

The frozen E8/E13 block manifest and `L2_sensitivity` construction are reused
without resampling:

- three 12,000-window blocks with evaluation seeds
  `20260711,20260712,20260713`;
- 6,000 normal rows per block;
- four attacks × three coupled settings (`low`, `medium`, `high`) × 500 rows.

Stage A must regenerate all 36,000 L2 rows and freeze a keyed byte digest before
model inference. The five shared `real_ms` checkpoints and 25 new Rule
checkpoints are scored. Normal FPR/recall is reported first.

The prespecified continuity estimands are Rule-minus-real exact and binary
recall for the `low` and `medium` settings, separately and equal-macro averaged
over the four attacks. The `high` rows are retained and reported
descriptively; they are not silently discarded.

L2 jointly changes burst length, step, amplitude, payload position, and other
attack-specific fields. It is a coupled-setting bridge to the retained ladder,
not an intensity-only intervention. L2 cannot determine the L4 factorial
verdict.

## 8. Contrasts and factorial effects

For L4, join the shared real reference at the same `p,b,a,i,x`:

\[
A^E_{g,p,b,a,i}(x)
=Y^E_{\mathrm{rule},g,p,b,a,i}(x)
 -Y^E_{\mathrm{real},p,b,a,i}(x),
\]

with analogous \(A^B\).

This is a total matched-budget augmentation-pipeline contrast. It includes the
changed training sample mix, class weighting, and minibatch trajectory; it is
not a pure causal effect of one grammar relation.

For any cell function \(f(P,S,D)\), use the E14 operators:

\[
C_P[f]=\frac14\sum_{S,D}\{f(1,S,D)-f(0,S,D)\},
\]

with analogous `S` and `D`;

\[
C_{PS}[f]=\frac12\sum_D[
f(1,1,D)-f(0,1,D)-f(1,0,D)+f(0,0,D)],
\]

with analogous `PD` and `SD`; and

\[
C_{PSD}[f]
=[f_{111}-f_{011}-f_{101}+f_{001}]
 -[f_{110}-f_{010}-f_{100}+f_{000}].
\]

The seven factorial effects are:

```text
P, S, D, PS, PD, SD, PSD
```

`111-000` is a separate supporting corner contrast, not a factorial main
effect or interaction.

ID scopes are:

```text
canonical
shifted
canonical_minus_shifted
```

Attack scopes are Gear, RPM, and their equal macro. Shifted-ID and
canonical-minus-shifted results are supporting effect-modification summaries.

## 9. Aggregation and inferential hierarchy

For endpoint, ID scope, attack scope, and effect `j`, first compute the
factorial effect within each `(g,p,b,attack)` paired Rule-minus-real table.
Let \(w_{g,p,b,j}\) denote the attack-aggregated effect before block averaging.
The primary canonical equal-macro order is:

1. subtract real from Rule within the same pipeline, block, attack, ID, and
   cell;
2. apply the factorial operator within that block and attack;
3. equal-weight average Gear and RPM;
4. equal-weight average the three blocks;
5. average the five pipeline seeds to obtain one construction-level estimate.

Let the block- and attack-averaged crossed value be:

\[
z_{g,p,j}=\frac13\sum_b w_{g,p,b,j}.
\]

Define:

\[
\theta_{g,j}=\frac15\sum_p z_{g,p,j}.
\]

The primary test uses the four \(\theta_{g,j}\) values for:

```text
271828, 161803, 141421, 173205
```

Report those four values, mean, sample SD, minimum, maximum, strict-sign
pattern, and an untruncated two-sided 95% Student-\(t\) interval with 3 degrees
of freedom.

Report the fresh `314159` \(\theta\) as
`fresh_known_construction_anchor`, then report a clearly labelled
anchor-inclusive five-value mean, SD, range, sign pattern, and nominal interval
descriptively. The anchor-inclusive interval receives no confirmatory p-value
or verdict.

For each pipeline seed, define the new-construction-first value:

\[
\phi_{p,j}=\frac14\sum_{g\in G_{\mathrm{new}}} z_{g,p,j}.
\]

Report all five pipeline values, mean, SD, range, sign pattern, and nominal
interval descriptively. Pipeline-first summaries do not create another
confirmatory family.

For each block, report the new-construction-first descriptive value:

\[
\psi_{b,j}=\frac1{20}
\sum_{g\in G_{\mathrm{new}}}\sum_p w_{g,p,b,j}.
\]

The hierarchical table has exactly these 15 row identities for every
endpoint/ID/attack/effect combination:

```text
5 × construction               (one per g; anchor role explicitly labelled)
5 × pipeline_new4              (one per p)
3 × block_new4                 (one per b)
1 × summary_new4
1 × summary_anchor_inclusive5
```

### 9.1 Shared-reference dependence

Each real-only value is stored once and paired with every construction at the
same pipeline seed, block, attack, ID, and cell. Copying it across
constructions does not create observations.

Construction-level intervals condition on the fixed real-only checkpoints and
selected pipeline-seed set. They quantify dispersion across four new Rule-pool
realizations around that shared reference; they do not jointly estimate
construction, pipeline, dataset, and model uncertainty.

### 9.2 Descriptive crossed dispersion

For a finite `G×P` grid after attack/block averaging, define:

\[
\bar z,\quad
\bar z_{g\cdot},\quad
\bar z_{\cdot p},\quad
r_{g,p}=z_{g,p}-\bar z_{g\cdot}-\bar z_{\cdot p}+\bar z.
\]

With `P=5`, use `G=4` for `new4` and `G=5` for
`anchor_inclusive5`. Compute exactly:

\[
\begin{aligned}
SS_C &= P\sum_g(\bar z_{g\cdot}-\bar z)^2,\\
SS_P &= G\sum_p(\bar z_{\cdot p}-\bar z)^2,\\
SS_I &= \sum_{g,p}r_{g,p}^2,\\
SS_T &= \sum_{g,p}(z_{g,p}-\bar z)^2,\\
RMS_I &= \sqrt{SS_I/(GP)}.
\end{aligned}
\]

For `SS_T>0`, each finite-grid share is `SS_component/SS_T`. If
`SS_T=0`, all three shares are `NA` with status `degenerate total
dispersion`. Report:

- sample SD of the `G` construction marginal means;
- sample SD of the five pipeline marginal means;
- `RMS_I`;
- finite-grid centered sums of squares
  `SS_construction`, `SS_pipeline`, and `SS_interaction`;
- their shares of finite-grid total centered sum of squares.

These are **descriptive finite-grid dispersion shares**, not random-effects
variance components, population variance explained, or mixed-model tests.
Blocks are already averaged and do not supply independent residual replication.

Compute the same finite-grid quantities once more on the labelled
anchor-inclusive `5×5` grid. Those values are historical-anchor descriptive
sensitivities only. They do not enter a p-value, a verdict, or the
construction-resolved rule.

The 20 new crossed cells, 25 anchor-inclusive cells, blocks, attacks, and
windows are never used as the primary inferential sample size.

### 9.3 Supporting L2 hierarchy

For each L2 endpoint, setting, and attack scope, first compute Rule-minus-real
within the same pipeline seed, block, attack, and scenario. Then:

1. equal-weight average the three blocks within each construction/pipeline;
2. for the four-attack macro, equal-weight average the four attack-specific
   deltas after step 1;
3. average the five pipelines to obtain each construction-level value;
4. report the four new construction values, their mean, sample SD, range,
   strict-sign pattern, and nominal untruncated 95% \(t\)-interval;
5. report the `314159` anchor and anchor-inclusive five-value summary
   separately;
6. report pipeline-first values averaged over the four new constructions.

No L2 p-value, multiplicity decision, stability label, or scientific verdict
is computed. A nominal interval containing zero is `inconclusive`. All four
attack-specific values accompany the macro so the coupled-setting response is
not described as common to attacks when directions differ.

The L2 summary has exactly 360 rows:

```text
2 endpoints × 3 settings × 5 attack scopes
× (5 construction + 5 pipeline_new4
   + 1 summary_new4 + 1 summary_anchor_inclusive5)
```

Its unique key is:

```text
(endpoint, setting, attack_scope, row_type, unit_id)
```

`unit_id` is the construction seed, pipeline seed, or the literal summary
scope as appropriate. `Continuity` means the separately reported low- and
medium-setting augmentation deltas. No medium-minus-low contrast,
between-setting slope, trend test, or monotonicity estimand is registered.

## 10. Multiplicity and primary tests

The primary canonical exact Rule-minus-real family is:

\[
\mathcal F_E=
\{C_P,C_S,C_D,C_{PS},C_{PD},C_{SD},C_{PSD}\}.
\]

For each effect, use a two-sided one-sample Student-\(t\) test of
\(H_0:\mu=0\) on the four new construction-level estimates and jointly adjust
the seven raw p-values with Holm's method. Report mean, SD, untruncated nominal
95% \(t\)-CI, raw p, and Holm-adjusted p. Do not call the nominal CI
Holm-corrected. With only four selected new realizations, these tests are
supporting evidence conditional on the frozen seed grid, not population
inference.

The seven canonical binary effects form a separate mandatory companion Holm
family. Binary results cannot rescue, replace, or change the exact primary
verdict.

The following are estimation-focused supporting analyses with no confirmatory
Holm family:

- anchor-only and anchor-inclusive summaries;
- fresh-versus-E14 anchor outcome continuity;
- pipeline-first and block-first summaries;
- `111-000`;
- shifted-ID and canonical-minus-shifted effects;
- Gear- and RPM-specific effects and attack heterogeneity;
- L2 low/medium continuity and all L2 high-setting values;
- descriptive crossed dispersion;
- normal-control FPR/recall.

If all four new construction values for an effect are identical, retain the
point estimate, mark variance/CI/test `degenerate variance`/`NA`, insert
conservative raw `p=1` only for Holm bookkeeping, and make the effect
ineligible for a resolved finding. The bookkeeping value is not an inferential
test result.

No smallest effect size of interest or equivalence band is registered.
`Equivalent`, `no effect`, `negligible`, and `material effect` are prohibited.
An interval containing zero is `inconclusive`.

## 11. Scientific verdicts

### 11.1 Effect-level rules

A primary exact effect is `construction-resolved` only if:

1. it is nondegenerate;
2. its seven-test-family Holm-adjusted `p<0.05`; and
3. at least three of four new construction values have the same strict sign
   as their mean.

An exact zero counts toward neither sign.

A primary effect is `construction-sign-heterogeneous` exactly when:

\[
\min_{g\in G_{\mathrm{new}}}\theta_{g,j}<0<
\max_{g\in G_{\mathrm{new}}}\theta_{g,j}.
\]

This flag is reported regardless of multiplicity status. It never upgrades an
effect, and it exposes the one opposite-direction construction that can remain
when a resolved effect satisfies the `3/4` rule.

A canonical binary effect is `binary-construction-resolved` under the same
three rules, using its separate seven-test binary Holm family.

A construction-resolved effect is
`pipeline-margin-directionally-concordant` only if at least four of five
\(\phi_{p,j}\) values have the same strict sign as the new-construction mean.
This second label is descriptive; failure does not erase the construction
result but narrows it to an effect obtained after averaging the selected
pipelines. Agreement of the construction and pipeline marginal summaries does
not establish sign stability in all 20 construction×pipeline cells.

### 11.2 Exact localization verdict

The primary exact family alone determines one base verdict:

- `C-MAIN-AND-INTERACTION`: at least one of `P,S,D` and at least one of
  `PS,PD,SD,PSD` are construction-resolved;
- `C-MAIN`: at least one main effect and no interaction are
  construction-resolved;
- `C-INTERACTION`: no main effect, but at least one interaction is
  construction-resolved;
- `C-MIXED`: one or more effects are nondegenerate and Holm-significant, but
  none passes the `3/4` strict-sign rule;
- `C-UNRESOLVED`: no nondegenerate effect is Holm-significant.

Append `+DEGENERATE(effect names)` when applicable.
Append `+CONSTRUCTION-SIGN-HETEROGENEOUS(effect names)` for every effect
meeting the strict rule in Section 11.1.

For each resolved effect failing pipeline-margin directional concordance,
append `+PIPELINE-MARGIN-HETEROGENEOUS(effect names)`. If the
new-construction Gear and RPM grand means for a resolved macro effect have
opposite strict signs, append `+ATTACK-HETEROGENEOUS(effect names)`.

Calling a resolved macro direction “shared by both attacks” requires more:
Gear and RPM must have the macro direction as their grand-mean sign, and at
least three of four attack-specific construction-marginal values must have
that sign for each attack. Otherwise retain equal-macro wording even when the
two attack grand means have the same sign.

If a main effect and an interaction both resolve, a main effect remains a
marginal average over the other factors. Do not claim that the response is
concentrated in one factor or that the factor alone caused it.

### 11.3 E14 payload-position-direction status

Separately classify the E14 `P` direction:

- `P-DIRECTION-REPEATED`: new-construction `P` is construction-resolved and
  negative;
- `P-BOTH-MARGINS-CONCORDANT`: the preceding rule holds and `P` is also
  pipeline-margin-directionally-concordant;
- `P-REVERSED`: new-construction `P` is construction-resolved and positive;
- `P-MIXED`: `P` is Holm-significant but fails the `3/4` sign rule;
- `P-INCONCLUSIVE`: all other cases.

`P-BOTH-MARGINS-CONCORDANT` is the strongest applicable P label and is reported
instead of, not in addition to, `P-DIRECTION-REPEATED`.

If all four new `P` values are negative but the Holm test is unresolved, report
“directionally concordant but construction-level evidence remained
inconclusive,” not replication. A positive resolved result is a directional
reversal relative to E14, not a new mechanism.

### 11.4 Exact/binary endpoint relation

For each exact construction-resolved effect:

- binary-construction-resolved with the same sign:
  `endpoint-directionally-concordant`;
- binary unresolved: `exact-only resolved under the registered families`;
- binary-construction-resolved with the opposite sign:
  `endpoint-discordant`.

A binary-only resolved effect is labelled
`binary-only directional repetition` and cannot support an exact attack-typing
claim.

### 11.5 Outcome-to-claim boundary

| Outcome | Permitted conclusion | Prohibited expansion |
|---|---|---|
| resolved main, both margins concordant | direction repeats in at least 3/4 construction-marginal estimates, with a separate same-direction pattern in at least 4/5 pipeline-marginal estimates | cellwise 4×5 stability, physical cause, universal grammar mechanism, factor-only causation |
| resolved main, pipeline margin heterogeneous | marginal association after averaging the selected pipelines; pipeline realization limits the claim | cross-axis or cellwise stability |
| interaction only | the construction-marginal response is compound across the tested factor combination; any pipeline-marginal pattern is reported separately | isolated single-factor mechanism, cellwise stability |
| strict positive and negative construction marginals | Rule-pool realization limits the E14 localization, regardless of the other three values | stable Rule-generator effect |
| unresolved | E15 did not establish construction-level directional resolution | null, no effect, equivalence |
| binary only | construction-marginal direction repeated for attack-vs-normal discrimination only; pipeline-marginal direction is reported separately | exact typing localization, cellwise stability |

Every E15 conclusion must disclose one historical anchor plus four new selected
constructions, the fixed generator law, five selected pipeline seeds, shared
real references, and the rough `n=4` construction variance estimate.

## 12. Required outputs and completeness

### 12.1 Implementation

```text
journal/scripts/generate_rule_construction_sensitivity.py
journal/scripts/train_rule_construction_seed_crossing.py
journal/scripts/evaluate_rule_construction_seed_crossing.py
journal/scripts/analyze_rule_construction_seed_crossing.py
journal/tests/test_rule_construction_seed_crossing.py
```

Implementation hashes are unknown at registration and must be frozen in the
generation/training/prepare records before the relevant stage.

### 12.2 L4 scoring and analysis outputs

```text
journal/results/tables/e15_l4_factorial_normal_by_crossed_seed_v1.csv
journal/results/tables/e15_l4_factorial_by_scenario_v1.csv
journal/results/tables/e15_l4_factorial_by_cell_v1.csv
journal/results/tables/e15_l4_factorial_delta_v1.csv
journal/results/tables/e15_l4_factorial_effects_by_crossed_cell_v1.csv
journal/results/tables/e15_l4_factorial_hierarchical_summary_v1.csv
journal/results/tables/e15_l4_factorial_crossed_dispersion_v1.csv
journal/results/tables/e15_real_runtime_continuity_v1.csv
journal/results/tables/e15_anchor_outcome_continuity_v1.csv
```

Required L4 completeness:

- 5/5 shared-real runtime-continuity evaluations completed before Rule
  inference;
- 960,000 transformed and 30,000 clean-base real continuity forward
  evaluations;
- 495 all-zero real runtime-continuity rows: 480 scenario plus 15
  normal-control rows;
- 25/25 new Rule checkpoint evaluations;
- 4,800,000 transformed Rule forward evaluations;
- 150,000 clean-base Rule forward evaluations;
- 75 Rule normal-control rows;
- 2,400 Rule scenario rows;
- 1,200 Rule equal-attack-macro cell rows;
- 2,400 paired Rule-minus-shared-real scenario rows;
- 10,800 crossed effect rows:
  `5 constructions × 5 pipelines × 3 blocks × 2 endpoints × 3 ID scopes ×
  3 attack scopes × 8 effects`;
- 2,160 hierarchical rows:
  `2 endpoints × 3 ID scopes × 3 attack scopes × 8 effects ×
  (5 construction + 5 pipeline + 3 block + 2 summary rows)`;
- 288 crossed-dispersion rows:
  144 endpoint/ID/attack/effect combinations × two explicitly labelled
  scopes (`new4`, `anchor_inclusive5`).
- 495 anchor-continuity rows: 480 scenario rows plus 15 normal-control rows.

`8 effects` means the seven factorial effects plus `corner_111_minus_000`.
Primary inference filters the registered canonical/equal-macro/seven-effect
subset; completeness still requires all rows.

The real runtime-continuity table compares the fresh continuity rescore with
the frozen E14 `real_ms` values. It uses the same key as the anchor-continuity
table below, has 480 scenario and 15 normal-control rows, and must contain zero
integer-count differences throughout. It is published before any Rule
checkpoint inference; failure publishes only the technical failure record, not
partial E15 outcomes.

The anchor-continuity table compares the fresh E15 `314159` Rule outcome with
the frozen E14 `rule_ms` outcome at the same pipeline, block, attack, ID, and
cell. It stores exact- and binary-correct-count/recall differences in 480
scenario rows and normal-correct-count/recall/FPR differences in 15 control
rows. Its unique key is:

```text
(row_scope, pipeline_seed, block_id, attack, id_stratum, cell)
```

For normal rows, `attack`, `id_stratum`, and `cell` use the literal
`not_applicable`.
All-zero differences yield `anchor-outcome-identical`; any difference yields
`anchor-outcome-drift`. Checkpoint-byte inequality alone is recorded but is
not outcome drift. Neither status changes the new-four-construction primary
test; drift must be disclosed as implementation/runtime continuity evidence
and prevents calling the fresh anchor an exact outcome reproduction.

### 12.3 L2 outputs

```text
journal/results/tables/e15_l2_continuity_by_scenario_v1.csv
journal/results/tables/e15_l2_continuity_delta_v1.csv
journal/results/tables/e15_l2_continuity_summary_v1.csv
```

Required L2 completeness:

- 30 checkpoint evaluations: 25 Rule plus five shared real;
- 1,080,000 forward evaluations:
  `30 checkpoints × 3 blocks × 12,000 rows`;
- 1,170 arm/scenario rows:
  `30 checkpoints × 3 blocks × 13 scenarios`;
- 900 Rule-minus-real attack-scenario rows:
  `5 constructions × 5 pipelines × 3 blocks × 12 attack scenarios`;
- exactly 360 summary rows with the key registered in Section 9.3;
- all low, medium, high, normal, exact, and binary values retained.

### 12.4 Run records and manifest

```text
journal/experiments/e15_rule_construction_crossing/prepare_v1.json
journal/experiments/e15_rule_construction_crossing/run_v1.json
journal/results/logs/e15_rule_construction_crossing_v1.log
journal/results/logs/e15_rule_construction_crossing_artifact_manifest_v1.json
```

The final machine manifest binds every pool, generation log, sampling vector,
checkpoint, training log, E14 reference, transformation digest, table,
implementation source, source commit, environment, verdict, row count, and
unique key.

Every output is atomic and no-clobber. No figure or paper-manifest update is
created until every registered table and verdict validates.

## 13. Technical gates and stopping rules

### 13.1 Freeze/source gate

Before pool generation:

- this preregistration is committed;
- the tested implementation is committed;
- tracked experiment/source/test paths are clean;
- `wisa/` is unchanged;
- every frozen input hash in Section 4 matches;
- the pool-generation environment matches the 5,059-byte stable projection in
  Section 4.3, with NumPy `2.4.6`, exact `PCG64`, and little-endian byte order;
- all declared E15 output targets are absent.

Any freeze/source-gate failure is `T-STOP-POOL`; the pool preflight is not
published and generation is prohibited.

### 13.2 Pool gate

Before training:

- all five pool/statistics/log bundles exist and cross-validate;
- all five have the exact schema, identifier, seed, source, size, class,
  protocol, condition, finiteness, and uniqueness properties in Section 5;
- all five content and file hashes are distinct;
- the `314159` shared scientific arrays and anchor index hashes match;
- all 25 sampling requests pass capacity and no-replacement audits;
- no legacy/default/strict-v2 fallback occurred.

Failure is `T-STOP-POOL`; no training is allowed.

### 13.3 Training gate

Before the first child fit:

- all 25 checkpoint/log pairs are absent;
- preflight/environment/source/device records are complete;
- the current environment matches the frozen E14 stable projection while its
  process-specific full identity is recorded separately;
- pool and per-fit sampling hashes are frozen;
- every command contains explicit construction and pipeline identities.

After training:

- all 25 bundles exist;
- every log reports 6,156 steps, 12 checks, a finite selected score, the exact
  pool/sampling/config hashes, and the two distinct seed fields;
- every checkpoint validates with `weights_only=True`, expected shapes, and
  finite tensors;
- no target was skipped, reused, overwritten, or silently resumed.

Failure before a child starts is `T-STOP-TRAINING`. After the first child
starts, any missing, invalid, incomplete, corrupt, or mismatched
checkpoint/log bundle or training record is `T-INCOMPLETE`, including a grid
with 25 paths present but any failed bundle validation.

### 13.4 Stage A transformation gate

`--prepare-only` must not load a checkpoint for inference or expose
predictions. Hashing and `weights_only=True` structural/finiteness validation
are allowed. It must:

- reproduce the E14 base, latent, key, and transform digests;
- reproduce all eight L4 cells and registered strata;
- regenerate the complete E8 L2 block/scenario construction;
- publish an L2 keyed transformation digest and manipulation checks;
- validate all frozen references, implementation hashes, checkpoint/log
  bundles, standardizer hashes, and source state;
- prove every canonical score/table target absent.

Failure is `T-STOP-MANIPULATION`; scoring is prohibited.

### 13.5 Stage B scoring/analysis gate

Stage B must:

- first re-regenerate L4 and L2 transforms and match Stage A digests;
- then score only the five shared real checkpoints and atomically publish 495
  runtime-continuity rows with zero integer-count differences from E14;
- stop as `T-STOP-RUNTIME` before loading any Rule checkpoint for inference if
  that continuity check fails;
- complete every registered checkpoint, scenario, cell, endpoint, effect, and
  hierarchy row;
- reject duplicate keys, nonfinite required values, wrong pairing, or a copied
  real value mislabelled as a new construction observation;
- atomically publish the scientific output bundle only after all scientific
  completeness checks pass;
- produce exactly one scientific verdict from the frozen rules.

A Stage B transformation-digest mismatch is `T-STOP-MANIPULATION`, and a real
runtime-continuity mismatch is `T-STOP-RUNTIME`. Every other scoring or
analysis completeness/validity failure is `T-INCOMPLETE`, including a missing
or invalid fit/row, duplicate key, nonfinite required value, wrong pair, wrong
row count, or failure to produce exactly one verdict. Reduced-`n` analysis is
forbidden. Floor, ceiling, mixed-sign, reversed, adverse, binary-only, or null
outcomes are valid scientific results, not technical failures.

### 13.6 Resume, amendments, and exclusions

- There are no outcome-based pool, seed, block, attack, ID, cell, scenario,
  endpoint, or model exclusions.
- No construction or pipeline seed is substituted, added, or removed after an
  outcome is inspected.
- Publication of a pool-generation or training preflight starts a one-shot
  canonical stage. Any interruption or failure after that point makes the
  version incomplete; the same output identity is never resumed or retried.
- An incomplete or corrupted bundle is preserved for diagnosis and is never
  accepted, overwritten, or silently regenerated under the same identity.
- A retry requires a dated prospective amendment, a new output version, and a
  new clean preflight that proves every target of that new version absent.
  Complete old-version bundles may be read only for diagnosis, never imported
  into the new canonical grid.
- Technical changes before scoring require a dated amendment and new source/
  output version. After any E15 outcome is observed, estimands, families,
  aggregation, multiplicity, and verdict rules do not change.
- Valid inconclusive, heterogeneous, reversed, or adverse results are
  published. They are not rerun with new seeds, thresholds, endpoints,
  factors, or model-selection rules.
- No external dataset, new detector family, learned generator, paired placebo,
  or manuscript headline may enter the canonical E15 run.

Canonical technical status is one of:

```text
T-PASS
T-STOP-POOL
T-STOP-TRAINING
T-STOP-MANIPULATION
T-STOP-RUNTIME
T-INCOMPLETE
```

Gates are evaluated in the mandatory order in Section 2 and execution stops at
the first failure. The mapping for that first failure is the sole canonical
technical status; `T-PASS` is assigned only after every gate and registered
output validates.

Only `T-PASS` permits a scientific verdict or manuscript integration.
