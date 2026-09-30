# B technical continuation v1 — 2026-09-06

The interrupted B grid was restarted while preserving the
24 completed models. This is technical recovery, not a new scientific
registration. The original seeds, pools, training code/configuration, budget,
validation selector, evaluation roles and estimands remain unchanged.

## Fixed recovery boundary

Verify the original 15 completed jobs (three pool pairs and twelve fitting
pairs), all pool/checkpoint hashes, fit identities, full update/validation
budgets, earliest-best selection and arm pairing. Compare receipt hashes
with the original append-only launch log. Hash-inventory all existing main
and launch files before any mutation. Do not load checkpoints for inference.

The interrupted `confirmation_v1/g002/pipeline_123/` may contain only its
original `started.json`. Preserve that entire directory by a checked rename
to `confirmation_resume_v1/interrupted_attempts/g002_pipeline_123/`, with a
source/destination/hash receipt. If any checkpoint, partial model, failure
record or other file appears, stop instead of silently replacing it.
All 24 completed models and the original launch records remain in place.

Run exactly the remaining original 585 jobs: 97 pool preparations and 488
paired fitting jobs (976 fits), starting with construction 235483 / pipeline
123. Invoke the unchanged original training CLI. Its no-overwrite behavior
still applies; missing canonical paths under `confirmation_v1/` receive new
outputs. Never re-run a completed fit or change seeds based on outcomes.

The new launcher, this recovery note, its tests, the original launcher and
scientific freeze are hash-bound by `recovery_preflight_v1.json` before launch.
This receipt also inventories the completed artifacts and exact pending grid.
Verify preserved files after archiving and again at completion. A new failure
stops the service and retains its partial outputs; no automatic retries.

## Process lifetime and resources

Run as a background service with `Restart=no`, `KillMode=control-group`,
`OOMPolicy=stop` and CPU Nice=10. The foreground manager runs one original
child at a time. Per-job logs/progress/receipts are persisted in
`confirmation_resume_v1/`. No other GPU workload is stopped or reconfigured.
Observed before launch: 53 GiB host memory available and 20.17 GiB
CUDA-reported free memory; the GPU is shared with another workload.
These are dated observations, not reservations or peak-usage guarantees.
GPU sharing can increase duration; no pilot-derived completion guarantee.
The service does not recover automatically after a reboot.

## Completion and evaluation

Only after all original 600 jobs / 1,000 fits are verified, write a composite
`confirmation_resume_v1/full_grid_manifest.json` and `run.json` documenting
24 inherited + 976 new fits. Do not fabricate a completion receipt under
`confirmation_launch_v1/`; its interrupted history stays intact.

Training completion does not trigger scoring. The frozen evaluator v1 expects
the old launcher completion file and intentionally remains blocked. Before
scoring, a new separately frozen evaluator revision must accept and validate
the composite recovery lineage while retaining the existing caches, endpoints,
all-cell coverage and complete-grid checks. Do not patch frozen v1 in place.
