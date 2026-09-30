# B-v1: fresh-construction, source-conditional U/L replication

## Scope and observed design inputs

The user approved the A/B/C program on 2026-09-06 and prioritized competitiveness
over a tight deadline. This experiment is new; it does not increase E16's
registered sample or promote exploratory E20 into a confirmatory result.

The historical E20 k=8 means were U=-0.0105908 and L=+0.0267767. The k=8 focus
is outcome-informed and disclosed. Across its 20 construction units, the
historical U standard deviation was 0.0413506 (pipeline means within unit).
The new seeds and outcomes have not been generated when this protocol is frozen.

## Fixed grid and precision rationale

- 100 fresh construction seeds × 5 fixed pipeline levels (7, 42, 123, 2026,
  3407) × Rule/placebo = 1,000 five-class CNN fits.
- One additional, disjoint construction seed with pipeline 7 and both arms is
  a full-budget technical/timing pilot, excluded from all confirmation estimates.
- Seeds are deterministically derived from SHA256 of
  `vehcom-attribution-B-v1|index|attempt`: map the first four big-endian bytes
  to 100000 + value mod 900000; reject collisions with E16 construction/bridge
  seeds, 314159 and previously selected seeds. Index 0 is pilot; 1–100 are main.
- With the historical U SD, 100 units project a 97.5% interval half-width of
  approximately 0.00941. This targets roughly one percentage point of precision,
  not a guaranteed achieved width and not power based on an assumed true effect.
  Twenty new units would project about 0.0225, too broad for this purpose.
- No significance-based stopping, adaptive sample increase, replacement of
  unfavorable seeds, hyperparameter search, or equivalence claim. A technical
  failure stops the affected stage, is preserved, and requires an explicit
  correction/version. Budget changes require a new record before main outcomes.

## Training and matching

Reuse the current pure E16 generator/control functions, with 65,000 windows
per generated attack family and all 13 reported control checks. Any abort gate
failure blocks training for that pair. The Rule mirror must match the canonical
generator on numeric/label arrays. The RPM control changes the designated
coupling instance, not every residual relation.

Use the unchanged real train/validation arrays, real-train standardizer,
262,149 real windows and 78,645 synthetic windows (30% augmentation). Draw
without replacement with pipeline seed + 300 and the existing class-prefix
allocation. Rule/placebo share draw indices, initial model state and stochastic
pipeline settings. Same existing 11-channel, 128-frame, five-class CNN, balanced
cross-entropy, AdamW lr=0.001/weight_decay=0.0001, batch 512, 6,156 updates,
validation every 513 updates with batch 1,024, best validation macro-F1 and
earliest tie. No early stopping. Five pipeline levels are a fixed crossed axis,
not five independent datasets. Validation is source-local and historically used.

Save both full pools, selected checkpoints and per-fit training receipts in new
versioned paths. Unicode metadata replaces object string arrays only for safe
serialization; numeric generator content is unchanged. No prior file is replaced.

## Evaluation data roles and estimands

Use exactly the historical E14/E20 6,000 normal base windows, their frozen
latents, three equally weighted blocks and the same P0 missing-cell transforms.
These 6,000 are the `checkpoint_selection_attack_historical` subset in the
CFOT role manifest. None of its 24,000 `source_locked_new_method_test` windows
may be selected or scored. New generator/checkpoint randomness does not make
the bases or chosen k outcome-blind. Call this **fresh-construction conditional
replication**, never untouched-test or cross-source confirmation.

For each model, score both construction families (Gear label 3, RPM label 4),
both relation conditions, and every k in {2,4,8,16,32}. Each family×block cell
is equally weighted; within a cell recall is its correct-label count / 2,000.
For each construction unit, first average the five paired pipeline results.

Co-primary estimands at k=8:

    U = R_rule,intact - R_placebo,intact
    L = (R_rule,intact - R_rule,broken)
        - (R_placebo,intact - R_placebo,broken)

Use two-sided 97.5% Student-t intervals over the 100 construction means for
each co-primary (Bonferroni family coverage at least 95% under the interval
assumptions). Report means, SDs, both intervals, unit distributions and fixed
pipeline sensitivity. These quantify construction variability conditional on
the shared source, evaluation bases, latent grid and fixed pipelines.

All other k values, per-family results, binary recall, absolute recalls and
normal-base FPR are secondary/descriptive with pointwise 95% intervals labeled
as such. Normal FPR is on the unmodified 6,000 bases, not an unseen deployment
population. No selection of the best new k or operational-benefit claim from
exact recall alone. Real-only references may be contextualized from their
existing checkpoints but are not new fits or new independent replication units.

Interpret separately: L's lower bound > 0 supports positive local reliance;
U's lower bound > 0 supports positive conditional local utility. If U spans
zero, utility direction remains unresolved, not demonstrated absent. If L is
positive while U's upper bound ≤ 0, that supports the local separation in this
fixed construction setting. No post-hoc SESOI or nonsignificance-to-equivalence
conversion is allowed.

## Execution gates and resource estimate

First freeze this protocol, settings, source code and input hashes; audit the
evaluation roles without scoring. Run the separate full-size pilot pair to
measure wall time/storage and check paired training. The historical E16 run
(220 fits, 15,790.394 s) projects about 19.94 h for 1,000 fits, excluding new
pool generation and scoring. This is historical, not a current measurement.

Main training requires a completed pilot technical receipt. Main evaluation
requires a separately hashed evaluator, transform-continuity/matching checks,
and an exact checkpoint grid check before new test outcomes. Pilot validation
scores must not be used to tune the settings. All stages preserve failures and
refuse overwrite. The local freeze is not a public preregistration.

This experiment does not resolve second-source/native-attack generalization,
physical semantics or deployment FPR. C remains necessary regardless of B's
outcome; additional precision on one source is not its replacement.
