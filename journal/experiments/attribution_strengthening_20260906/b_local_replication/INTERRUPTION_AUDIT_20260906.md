# B interruption audit — 2026-09-06 11:31 UTC snapshot

This is an observation of an interrupted run, not a model failure verdict
or a replacement for the original launch receipts.

- The original launcher PID 1611804 and last child PID 1728724 are absent.
  No matching B launcher/training process is running; tool session 53578
  also no longer exists.
- The last complete event is `g002_fit-pair_p42` at 11:13:23.784589 UTC.
  Twelve paired completion receipts and 24 checkpoint files are present;
  three construction-pair preparation receipts are present.
- `g002_fit-pair_p123` started at 11:13:23.785466 UTC. Its directory contains
  only `started.json`, and its child log is empty. There is no checkpoint
  or completion receipt for that pair.
- The launch directory has neither final `run.json` nor `failure.json`.
  Thus the normal Python failure handler did not leave its expected receipt.
  A process/session interruption is plausible, but the cause is unconfirmed.
- The inspected kernel-journal interval 11:10–11:20 UTC contained no matching
  OOM/killed-process/NVIDIA fault record. This does not establish absence of
  a resource issue or identify who/what stopped the processes.
- Other GPU work was active. It was not stopped or modified; no B relaunch,
  output movement, deletion, overwrite, or checkpoint inference was done.

The evaluator preparation, historical replay and freeze are complete.
Its full-grid gate was checked with checkpoint loading forbidden: it rejects
this partial grid before any new checkpoint loads. There is no `scored/`.

The v1 launcher intentionally has no resume policy and cannot be rerun on
existing outputs. A continuation needs an explicit new technical recovery
record/launcher: preserve all 24 completed fits and the interrupted pair,
verify completed receipts/hashes, reuse the identical frozen seeds/config,
finish every missing pair without outcome-based exclusion, and consolidate
full-grid provenance before scoring. Do not change frozen training code or
manufacture the original launch completion receipt. Confirm restart/resource
coordination with the user before launching another long GPU workload.
