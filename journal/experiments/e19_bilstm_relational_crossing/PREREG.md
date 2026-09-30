# E19 preregistration — BiLSTM sequence-family repetition of the relational-versus-marginal decomposition

- Written: 2026-08-07 UTC
- Repository HEAD at writing: `3dd5d9a` (`main`)
- Status at freeze: **no E19 pool, fit, score, table, figure, or outcome of any
  kind exists.** This document is committed before any implementation code for
  E19 is committed and before any E19 fitting begins.
- Registration lineage: E16 (`e16_relational_response_decomposition/PREREG.md`,
  commit `7371443`), E17 (`e17_rf_relational_replication/PREREG.md`), and E18
  (`e18_relation_visible_rf/PREREG.md`) are frozen predecessors. E19 changes
  exactly one axis relative to E16: the detector architecture.

## 1. Question and evidential scope

The manuscript names a sequence-family crossing as "the necessary next test"
(Threats, single-architecture limitation). E19 runs that test. The question is
the registered E16 question with only the detector family changed:

> Across the same twenty prospectively drawn generator realizations and the
> same five detector-training pipeline seeds, does the implemented Gear/RPM
> relational structure add a material increment to the payload-destination
> response of a bidirectional LSTM classifier, beyond the marginal-matched
> placebo twin?

### 1.1 What E19 can establish

A fourth registered estimate of the relational increment theta, on a detector
family that consumes the raw window as an ordered sequence and mixes channels
through a shared recurrent state. Unlike the E16 CNN (no cross-channel weight
sharing) and the E17/E18 forests (fixed feature maps), the BiLSTM couples all
eleven channels at every timestep through its input projection, so within-frame
cross-byte structure has a representational route to the decision at every
byte position, including the relocated `P = 1` destinations.

### 1.2 What E19 cannot establish

- It is one architecture with one pooling convention (mean over time); it does
  not make the detector axis a population.
- Endpoint-level utility, not representational capacity: a null does not show
  the relation is invisible to the network.
- The E14 cell-structure limits are inherited: `P = 0` has little headroom,
  and no training signal distinguishes the arms on the relocated byte pair.
  The missing-cell limitation named in Threats survives this registration and
  is probed separately by the E20 evaluation-only design.
- Nothing about realism, replacement of real attacks, or external transfer.

## 2. Mandatory execution order

1. Commit this PREREG. 2. Commit the E19 implementation scripts. 3. Read-only
preflight (registration-committed gate, pool inheritance audit, no-clobber).
4. Training stage. 5. Evaluation stage. 6. Registered analysis. No stage is
resumed in place; a failed stage halts with a `T-STOP`/`T-INCOMPLETE` record
and any continuation is a dated prospective amendment committed before
outcomes, or a dated deviation disclosure if outcomes exist. Published
checkpoints and tables are never overwritten.

## 3. Frozen axes

### 3.1 Generator-realization seeds (transcribed literals; gate compares)

Primary twenty (identical to E16 section 3.1):

```
415637, 748203, 560657, 459336, 457191,
999191, 715822, 298581, 922089, 864178,
583909, 467232, 550427, 485791, 820632,
837943, 222537, 685012, 192753, 739263
```

Bridge four (sensitivity only, never in the primary `n = 20`): `271828,
161803, 141421, 173205`. As in E17, no frozen same-family Rule checkpoint
exists, so E19 fits both bridge arms; they remain a sensitivity.

### 3.2 Pools — consumed by hash, never regenerated

E19 consumes the frozen E16 pools
(`journal/datasets/synthetic/e16_{rule,placebo}_cseed*_windows_v2.npz`) and
inherits their manipulation-gate record
(`journal/results/tables/e16_manipulation_checks_v2.csv`). The preflight
records every consumed pool's SHA-256. No pool is generated or modified.

### 3.3 Detector-pipeline seeds and arms

Pipeline seeds `7, 42, 123, 2026, 3407`; arms `rule`, `placebo`, plus five
`real_only` shared references trained fresh under the identical budget (no
frozen matched-budget BiLSTM reference exists).

## 4. Detector family and the matching contract

### 4.1 Frozen architecture

`LSTMClassifier` exactly as defined in
`journal/scripts/train_family_extension_lstm.py` at the implementation commit:
2-layer bidirectional LSTM, input 11, hidden 128, dropout 0.2, mean pooling
over the 128 timesteps, head `Linear(256, 128) -> ReLU -> Dropout(0.2) ->
Linear(128, 5)`. The class is imported, not redefined.

### 4.2 Training protocol — the E16 budget verbatim

Class-weighted cross entropy, AdamW `1e-3`/`1e-4`, batch 512, maximum
optimizer updates 6,156, validation every 513 updates (12 checkpoints),
best-validation-macro-F1 selection, standardizer fit on real training windows
only. Within a realization and pipeline seed, the Rule and placebo fits share
the detector initialization seed, the synthetic row indices, the minibatch RNG
convention, and the update budget. All constants are imported from
`train_rule_construction_seed_crossing.py`; none is retuned for the BiLSTM.

### 4.3 Identical-training-row contract (blocking)

For every `(realization, arm, pipeline seed)` with a frozen E16 counterpart,
the E19 `sampling_index_sha256` must equal the frozen value in
`journal/results/tables/e16_sampling_audit_v2.csv`. The 40 bridge fits have no
E16 CNN counterpart for the Rule arm and are recorded as exempt, mirroring
E17. Gate failure halts training.

## 5. Fit grid and counts

| Component | Count |
|---|---|
| 20 realizations x 2 arms x 5 pipelines | 200 |
| 4 bridge realizations x 2 arms x 5 pipelines | 40 |
| Shared real-only references | 5 |
| **Total** | **245** |

Checkpoint identity: `bilstm_{arm}_0p30_cseed{g}_matchedsteps_e19_v1_seed{p}.pt`
(`bilstm_real_only_matchedsteps_e19_v1_seed{p}.pt` for references) under
`journal/models/generator_extension/`, with one JSON log per fit under
`journal/results/logs/`. Atomic publication; no-clobber preflight over the
complete target list.

## 6. Evaluation

Frozen E14 material: the same three blocks x 2,000 held-out bases, the same
latents, the same eight factorial cells, both ID strata, exact and binary
endpoints scored separately, untouched-base normal outcomes reported first.

**Scoring device: CUDA, declared here.** E16/E17/E18 scored on CPU because a
frozen same-family CPU integer record existed and integer equality was the
continuity gate. No frozen BiLSTM record exists, so no integer-equality gate
binds the device; the completeness gates below apply instead. Determinism mode
`torch.use_deterministic_algorithms(True, warn_only=True)` with the runtime
flags recorded. Completeness gates: exactly 245 scored identities plus the
shared-real identities' own scores; the registered scenario-row count; every
fit at the frozen budget.

## 7. Endpoints, aggregation, and decision rule

All identical to E16 sections 7--10, restated compactly:

- Primary endpoint: canonical-ID Gear/RPM exact-macro attack-type recall,
  factorial `P` contrast `C_P`, averaging attack, then block, then cell, then
  the five pipeline seeds within each generator realization.
- Primary inferential unit: the twenty realization-level
  `theta_g = C_P(Rule) - C_P(Placebo)`; `n = 20`, `df = 19`; single
  registered test, no multiplicity correction.
- Equivalence margin: inherited `delta = 0.133855625`. Practical equivalence:
  90% TOST interval wholly inside `[-delta, +delta]`. Material difference:
  95% interval wholly outside on one side. Heterogeneity screen first,
  exactly as registered for E16: C1 at least three realizations materially
  outside on each side; C2 five or more outside despite aggregate
  equivalence; a Student-t/bootstrap disagreement downgrades to inconclusive.
  Bootstrap: 10,000 percentile resamples, seed 20260807.
- Decomposition closure `C_P(Rule-Real) = C_P(Placebo-Real) + theta` verified
  to floating-point closure as a release gate.

### 7.1 Dual-margin reporting (mandatory, registered now)

As in E17/E18, the inherited absolute margin is permissive for a family whose
out-of-grammar response may be smaller than the CNN's. E19 therefore reports,
alongside the registered verdict, the stricter margin
`delta_strict = 0.15 x |placebo-minus-real out-of-grammar (P = 1) exact-macro
gain of this family|`, computed from the E19 data by this fixed rule, with the
same screen-first order E18 registered. Both verdicts are reported; neither
overrides the other. The margin-free bound — the 90% interval as a share of
the family's own out-of-grammar exposure gain — is mandatory secondary
reporting.

### 7.2 Registered secondary reporting

(a) `P = 1` out-of-grammar cell: absolute recalls for real/placebo/Rule and
the Rule-to-placebo gain ratio, ratio rows with placebo-gain denominator
below 0.01 flagged and excluded from the ratio summary; (b) `P = 0`
in-grammar cell absolute recalls; (c) six remaining exact Rule-minus-placebo
effects in one Holm family; (d) seven binary companion effects in a separate
Holm family; (e) relational share with denominator floor 0.10; (f) bridge
sensitivity reported apart from the primary; (g) per-realization decomposition
table and sign counts.

## 8. Planning values (quoted, not re-derived)

Observed realization-level SDs of theta: 0.0400 (E16 CNN), 0.0263 (E17),
0.0261 (E18). At `n = 20` these give 90% TOST half-widths of roughly
0.010--0.016, far inside the inherited margin; against the stricter margin the
sibling registrations were realization-contingent, and the same outcome is
plausible here. That expectation is recorded so that a realization-contingent
stricter verdict is not presented as surprising.

## 9. Failure and rerun policy

A child fit that crashes before publishing is rerun once from its own clean
preflight (the fit publishes atomically, so no partial bundle can exist); a
second failure halts the stage. Analysis-stage code errors are fixed and the
analysis rerun; the decision rule itself is frozen by this document. Any
event outside this policy is a dated deviation disclosure.
