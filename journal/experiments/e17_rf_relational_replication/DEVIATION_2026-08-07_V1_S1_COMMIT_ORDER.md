# E17 deviation 2026-08-07 v1 — the S1 implementation commit trails the start of S4 fitting

Deviation time: 2026-08-07T01:20:00Z

Concerns: `journal/experiments/e17_rf_relational_replication/PREREG.md` section 2
(mandatory execution order), stages S1 and S3/S4.

Registration commit: `d18e027` (freeze). Registration source commit as written
in the PREREG header: `42ec54c`. Prior amendment:
`AMENDMENT_2026-08-07_V1_BRIDGE_POOL_AUDIT_SOURCE.md` (`42ae5c5`).

**This is a deviation, not an amendment.** It changes no registered design
element. It alters no seed list, no margin, no aggregation order, no decision
order, no inferential unit, no gate, no output identity, and no claim boundary.
Nothing in the PREREG is amended, superseded, relaxed, or reinterpreted by this
document. It exists solely to record, in the experiment directory and in the
supplement, that the registered *execution order* was not followed exactly, so
that a reader does not have to discover it from commit timestamps.

## 1. What section 2 requires

PREREG section 2 fixes the stage order and states:

> A later stage must not begin until the earlier stage's gates have passed and
> its run record has been committed.

with

> 2. **S1 — implementation.** Scripts and tests (Section 14.1) committed with no
>    E17 outcome, checkpoint, or score present.

So the five section 14.1 implementation files were required to be committed
before stage S3 (reference fits) and S4 (paired fits) began.

## 2. What actually happened

| event | UTC |
|---|---|
| S2 pool-inheritance verification completed | 2026-08-07T01:02:51Z |
| S3/S4 fitting process launched | 2026-08-07T01:04:41Z |
| S1 implementation committed as `516b990` | 2026-08-07T01:14:20Z |

The implementation commit trails the start of fitting by 9 minutes 39 seconds.
The five files were authored, imported, and gate-tested before the launch — the
pre-outcome test module passes every negative control, and the read-only
preflight, gate G14, and stage S2 had all been executed against these exact
bytes — but they were committed after fitting had begun rather than before.

`516b990` is a five-file commit and contains only the section 14.1
implementation:

```text
journal/scripts/run_e17_rf_paired_training.py
journal/scripts/evaluate_e17_rf_relational_response.py
journal/scripts/analyze_e17_rf_relational_response.py
journal/scripts/make_e17_rf_relational_response_figure.py
journal/tests/test_e17_rf_relational_replication.py
```

## 3. Exact scope of the deviation

Only **one** of the five files had been used at commit time: the training
runner. Its bytes were frozen at process load — the running interpreter read
`run_e17_rf_paired_training.py` at 01:04:41Z and every fit in the grid is
produced by that loaded image, so the committed bytes are the bytes that
produced the fits. The file was not edited after launch.

The other four files had not executed anything that produces an outcome. In
particular, **the evaluator and the analyzer — the only code an observed
outcome could plausibly shape — were committed with zero E17 scores in
existence.** No score, contrast, decomposition, interval, margin, or verdict had
been computed by any process at any point before `516b990`.

Stage S2 also ran before `516b990`. S2 performs no fitting; it recomputes
SHA-256 digests of 48 read-only input pools and compares them against frozen
E16 and E15 audit tables. Its single output,
`journal/results/tables/e17_pool_inheritance_audit_v1.csv`, is a hash record of
inputs, not an outcome, and it is unchanged by this deviation.

## 4. What was verifiably absent at commit time

Enumerated the way the PREREG header enumerates its pre-outcome file counts,
measured at `516b990` (2026-08-07T01:14:20Z):

```text
journal/datasets/synthetic/e17_*                        0 files
journal/results/figures/e17_*                           0 files
journal/results/logs/e17_*                              0 files
journal/results/tables/e17_by_scenario_v1.csv           absent
journal/results/tables/e17_by_cell_v1.csv               absent
journal/results/tables/e17_decomposition_by_realization_v1.csv   absent
journal/results/tables/e17_primary_theta_summary_v1.csv          absent
journal/results/tables/e17_p1_cell_comparison_v1.csv             absent
journal/results/tables/e17_p0_cell_comparison_v1.csv             absent
journal/results/tables/e17_margin_sensitivity_v1.csv             absent
journal/results/tables/e17_secondary_effects_v1.csv              absent
journal/results/tables/e17_cnn_rf_family_comparison_v1.csv       absent
journal/results/tables/e17_bridge_sensitivity_v1.csv             absent
journal/results/tables/e17_real_only_reference_continuity_v1.csv absent
journal/results/tables/e17_pool_inheritance_audit_v1.csv         present (S2 input hashes)
journal/models/generator_extension/*e17_v1*.joblib      35 of 245 fits
journal/results/logs/train_rf_*e17_v1*.log              35 of 245 fits
```

35 of the 245 registered fits had completed. Those 35 are Random-Forest
checkpoints and their per-fit provenance records. **No checkpoint had been
scored.** A fitted checkpoint on disk is not an outcome in the registered sense:
the E17 estimand is a contrast over scored recalls, and the scoring stage S6 had
not started, could not have started (it refuses to run without the S4 run
record), and produced its first byte hours later.

## 5. Why this was not corrected by restarting

PREREG section 15.4 states:

> No stage is resumed in place after an abort.

and gate G10a forbids overwriting any published checkpoint. Aborting S4 and
restarting to obtain a cleaner commit order would therefore have required either
resuming in place, which 15.4 forbids, or deleting 35 published checkpoints and
their training logs, which G10a's no-clobber discipline and the repository's
never-delete-failure-provenance rule both forbid. A restart would also have
required a dated amendment and a new output version identity, converting a
timestamp discrepancy that changed no result into a version churn that discards
correct work.

Restarting would have made the *record* tidier and the *experiment* no more
sound. Disclosure is the correct remedy here, so this document is the remedy.

## 6. Assessed exposure

The risk that stage-order rules of this kind exist to control is that
implementation choices get made, or revised, after the analyst has seen an
outcome. That risk is bounded here to nil, and the bound is checkable rather
than asserted:

- no E17 score existed before `516b990`, so no analysis or evaluation choice
  could have been informed by one;
- the analyzer's decision constants — `delta = 0.133855625`, the multiplier
  `0.15`, `df = 19`, the heterogeneity thresholds `3` and `5`, the bootstrap
  count `10000` and seed `20260807`, the saturation criterion `0.99` / `0.01`,
  and the degenerate-denominator floor `0.01` — are all transcribed from the
  frozen registration and are checked against it by the committed test module;
- the training runner's bytes were fixed at process load, before any fit;
- gate G10c-ii independently binds the fits to the frozen E16 CNN draw record,
  so the training rows were not free to change regardless of commit order.

No number in E17 is different from what it would have been had the commit
landed ten minutes earlier.

## 7. Disclosure

Following the E16 precedent, which disclosed its training-crash retry both in
the experiment directory and in the supplement, this deviation is recorded here
**and must be named in the E17 supplement section**. It is not to be left in the
experiment directory alone.

## 8. Unchanged by this deviation

Everything registered. The primary estimand `theta_g` and the aggregation order
of PREREG section 8; the inferential unit `n = 20`, `df = 19`; the inherited
margin `delta = 0.133855625` and the mandatory `delta_strict` sensitivity; the
decision order of section 10; the frozen generator-realization, bridge, and
pipeline seed lists; the Random-Forest contract of section 5.1; the
identical-training-row contract of section 5.2; the 245-fit grid; gates
G0--G16; the required outputs of section 14.2; and the claim boundaries of
sections 13, 16, and 17.
