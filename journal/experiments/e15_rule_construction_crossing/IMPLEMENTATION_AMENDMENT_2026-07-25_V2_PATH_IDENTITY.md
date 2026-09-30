# E15 implementation amendment: v2 path identity

Date: 2026-07-25 UTC
Status: prospective, written before any v2 pool generation, training,
evaluation, or analysis
Scope: technical path-identity correction only

This amendment applies Section 13.6 of `PREREG.md`. It does not replace or
modify that preregistration.

## Observed v1 failure

The canonical v1 pool-generation stage ran from
`2026-07-25T07:55:39.150636Z` to `2026-07-25T08:01:16.023653Z` and completed
with `T-PASS`. It generated five distinct 260,000-window pools, performed no
model inference, and produced no E15 scientific outcome.

The canonical v1 training preflight was published at
`2026-07-25T08:03:53.048933Z`. The first child process for construction seed
`314159` and pipeline seed `7` then stopped before entering `run_child_fit`.
The explicit child path had been resolved through the repository symlink:

```text
/home/hyunjin/workspace/wisa_can_ids/datasets/synthetic/rule_cseed314159_windows_v1.npz
```

The frozen grid retained the equivalent unresolved logical path:

```text
/home/hyunjin/workspace/wisa_can_ids/journal/datasets/synthetic/rule_cseed314159_windows_v1.npz
```

Comparing those path strings caused the child gate to fail. No model fit,
validation forward pass, checkpoint, training log, evaluation, analysis, or
scientific outcome was produced. The sole canonical v1 training status is
`T-INCOMPLETE`; no `training_run_v1.json` exists or will be synthesized.

The structured diagnostic record is
`training_failure_v1.json`, SHA-256
`b5e4cde58d0e6ba207bf5ba91bf39a150c0485be66a27171810a83e3daadb828`.

## Frozen v1 evidence

The complete v1 generation bundle and published training preflight are
preserved only for diagnosis:

| Evidence | SHA-256 |
|---|---|
| `pool_generation_preflight_v1.json` | `229dd633e71de87fdec40e4b3c1223cf0b477a0b15e2f7c55b0115c3c0b2e76e` |
| `pool_generation_run_v1.json` | `8c9e4d4bf732def4c3275b0bf5a81daf64f070f9b859ac408b3b38749616941b` |
| `training_preflight_v1.json` | `2bb94cf69ad5e00962f98f3201dc082d038f0943b847bbeed1f74baeb980b312` |

The v1 pool-file hashes recorded by the completed generation run are:

| Construction seed | Pool-file SHA-256 | Ordered-content SHA-256 |
|---:|---|---|
| 314159 | `e22903b66676a24a325db585be6cc7fac8cdc36567ffe57973b6f7aefdaa8fe9` | `9c05045481fc235ea58d85709295996a99c6842c39053332507c6d7ebfbbbc43` |
| 271828 | `522e8090777464353ec7f6ac5b009765ad62b09e5e70405ad9ed69344cf6b7f0` | `8ffb54dc909930ebc6f2150258c7415fb7e4e050bf3937fea02e645b28007ffc` |
| 161803 | `57f96165a193613f06b57466f10b13a0dd6db15c30b547edbc7e718de4cb2b0a` | `49b970d605b11bebe460e1d53ca926875610b785feaa39c3a4a23c4d3103367d` |
| 141421 | `1053917c4ee2822a64cb26131c25f0c315255bf5574caf38ddfae5a5bf770e0d` | `87fe8f00fa7e4f995608fd065d42d675bb652a6093d874b4d548226968925b91` |
| 173205 | `87a400c83522dcac65a1ca1d8b8a920cab8a1360a6fe322b88e344c79e14050a` | `983c19fc159ce94f42038e80d9a231b82631afc372e9273802b37a73542a3775` |

These v1 files are never resumed, retried, overwritten, renamed into v2,
copied into v2, symlinked as v2, or imported into the v2 canonical grid.

## Prospective technical correction

The only failure correction is to compare the resolved explicit child path
against `expected.pool_path.resolve()`. The checkpoint and log path checks are
unchanged. A regression test exercises the actual
`journal/datasets -> ../datasets` symlink.

All E15-owned canonical output identities are advanced to v2, from pool
generation through the final analysis manifest:

- five regenerated pool archives, statistics files, and generation logs;
- generation preflight, generation run, and pool audit;
- 25 newly fitted checkpoints and 25 training logs;
- training preflight, run, sampling audit, and training manifest;
- evaluation prepare/run records and every raw evaluation table/log; and
- every analysis table and the final artifact manifest.

The v2 preflight must prove every v2 target absent before execution. The v2
run starts at pool generation and regenerates all five pools; it does not use
the complete v1 generation bundle as an upstream input.

## Unchanged scientific contract

No schema semantics, serialization format, data split, construction seed,
pipeline seed, sampling vector, training budget, validation rule, model
selection rule, block, scenario, endpoint, estimand, aggregation order,
multiplicity adjustment, verdict rule, or reporting rule changes. Existing
schema and policy identifiers ending in `.v1` therefore remain unchanged.
Frozen E14 input filenames also remain at their registered v1 identities.

No E15 outcome was observed before this amendment. This correction neither
selects nor excludes a pool, seed, model, result, or scientific finding.
