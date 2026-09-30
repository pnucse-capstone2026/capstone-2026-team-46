# B complete-grid evaluation handoff — 2026-09-07

Training completed at 06:03:30 UTC / 15:03:30 KST: 1,000 fits, 24 inherited
and 976 new, with exit status 0. The completion receipt is
`confirmation_resume_v1/run.json`; its full manifest hash was rechecked.

The user explicitly approved connecting the evaluator to this recovery
lineage, running complete scoring/analysis, and screening C during evaluation.
The v1 evaluator, training code/config/freeze, caches, old models/results,
manuscript and frozen WISA directories were not modified.

## Verified new evaluator

- Contract: `EVALUATION_v2.md`; runner:
  `journal/scripts/evaluate_attribution_local_replication_v2.py`.
- 66 relevant tests passed: v1 scientific tests, v2 recovery/continuity tests,
  and existing technical recovery tests. Eleven scientific function bodies
  match v1 exactly, including inference/counts and aggregation.
- Freeze: `evaluation_v2/freeze.json`, SHA-256
  `6e660fd667dbe100b008b66ca28af5a69cb2148fe0b2d00343ee475f17296bc9`.
  It was written before any new B inference.
- Full runtime preflight passed at 14:53:39 UTC / 23:53:39 KST, verifying all
  600 jobs and 1,000 models. Receipt: `evaluation_v2/preflight.json`.
- First completed model output at 14:53:43 UTC; first paired outputs at
  14:53:46 UTC. These are runtime observations, not complete scientific results.

## Background execution and status

User service: `vehcom-b-eval-20260907.service`, launched 23:53:08 KST,
PID 1933393 at launch (dated; recheck live). CPU-only, eight intra-op threads,
Nice=10, `Restart=no`, `KillMode=control-group`, `OOMPolicy=stop`.
The program's `run` action scores the complete grid, then automatically
summarizes all saved predictions/counts. It never summarizes a partial grid.
No arbitrary GPU job or service was stopped or reconfigured.

Read-only status commands:

```bash
systemctl --user show vehcom-b-eval-20260907.service --property=ActiveState,SubState,MainPID,Result,ExecMainStatus
journalctl --user -u vehcom-b-eval-20260907.service -n 5 --no-pager
```

With `RemainAfterExit=yes`, `active/exited` and MainPID 0 mean the process
has ended; inspect Result and completion/failure receipts. `active/running`
alone is insufficient evidence of ongoing scoring: check new model outputs
and journal timestamps. Do not rerun `freeze`, `run`, or `score` in the same
output directory. No automatic retry/resume is enabled; preserve any failure.

Complete scoring: `evaluation_v2/scored/run.json`.
Complete analysis: `evaluation_v2/analysis/run.json` (co-primary U/L),
`summary.csv`, `pipeline_sensitivity.csv`, and `fpr_summary.csv`.
Overall B workflow: `evaluation_v2/run.json`.
Failure: `evaluation_v2/failure.json` and, if scoring had begun,
`evaluation_v2/scored/failure.json`.

Do not inspect partial effect estimates to change the protocol. Once all
scoring and analysis finish, review co-primary intervals and all descriptive
tables, then incorporate the complete evidence in the canonical manuscript.
This service does not edit the manuscript or complete C. B remains inference
over 100 construction units on historical source-conditional evaluation bases,
not a new-source or untouched-test confirmation.
