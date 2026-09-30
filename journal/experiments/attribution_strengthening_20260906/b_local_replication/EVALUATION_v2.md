# B-v2 evaluation: recovered-grid completion adapter — 2026-09-07

Complete B scoring and analysis follow the finished 1,000-fit grid.
This revision changes the technical completion gate, not the
scientific contract in `EVALUATION_v1.md`. No new B checkpoint has been scored
when this revision is written. The original evaluator and freeze stay intact.

## Completion and lineage

Accept only `confirmation_resume_v1/run.json`: complete, 600 original jobs,
1,000 fits, 24 inherited plus 976 new fits, 15 inherited plus 585 new jobs.
Verify the recovery audit/code/scientific freeze, launch receipt, archived
start-only attempt, original preserved files, ordered full-grid manifest,
both append-only completion logs, and current artifacts against all 600
manifest entries. Reject a recovery failure or fabricated original completion.
Retain v1's complete model identity, checkpoint hash, paired RNG/sampling,
standardizer, full training budget, and earliest-best validation checks.
No checkpoint is loaded until the whole gate passes.

## Unchanged evaluation and analysis

Reuse the already frozen v1 caches and historical continuity control; do not
regenerate inputs or rescore the historical checkpoint. Verify the original
freeze on every entry to scoring or aggregation. The 63 cells, 126,000 windows,
CPU float32 inference (8 intra-op threads, 1 inter-op thread, batch 4,096),
all five k values, exact/binary counts and all native source-model outputs
are unchanged. CPU inference is retained; no additional training is needed.

The calculation functions are regression-checked against v1. Analyze all
1,000 models, with the five fixed pipelines averaged within each of 100
construction realizations. Exact-macro U(8) and L(8) retain their co-primary
97.5% t intervals; other intervals are descriptive pointwise 95%. Preserve
all arm/condition recalls, family curves, pipeline sensitivity, normal FPR
and its paired difference. Do not inspect partial B outcomes to adjust the
design, select a condition, or stop early. Runtime progress may be inspected.

## Outputs and execution

Freeze this revision, its tests, the original evaluator freeze, and recovery
receipts/logs before inference in `evaluation_v2/freeze.json`. New files are
written exclusively under `evaluation_v2/`, with exclusive-create behavior.
`preflight.json` records the full gate before model loading; `scored/` retains
all per-model predictions and counts; `analysis/` retains the complete
fixed-contract analysis. The `run` action performs scoring and then summary
only after complete scoring. Failures are retained; no silent retry or resume.

Run as a separate background service with `Restart=no`, Nice=10 and
control-group cleanup, without stopping the training service or other
workloads. Service output records completed model identities, timestamps and
elapsed time, not interim effects.

Completion means B evaluation/analysis, not completion of C or proof of new
vehicle/source generalization. Reporting follows the complete result,
regardless of its direction. Old NO_GO branches and sealed evaluation
data remain unopened.
