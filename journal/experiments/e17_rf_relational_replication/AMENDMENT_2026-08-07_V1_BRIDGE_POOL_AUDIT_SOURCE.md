# E17 amendment 2026-08-07 v1 — the bridge Rule pools have no row in the E16 pool audit; name their frozen source table

Amendment time: 2026-08-07T01:52:00Z

Amends: `journal/experiments/e17_rf_relational_replication/PREREG.md` section 4.1
(pools consumed by hash) and section 15.2 gate G1i (pool hash inheritance).
Output version `v1` is retained; nothing is superseded.

Registration commit amended: `d18e027` (freeze commit). Registration source
commit as written in the PREREG header: `42ec54c`. Prior amendment: none.

Status: **PRE-OUTCOME.** Written and committed before stage S2 runs. No E17
Random-Forest fit, checkpoint, training log, score, table, figure, or verdict
existed at any point while this amendment was written, and none exists now:

```text
journal/datasets/synthetic/e17_*                        0 files
journal/models/generator_extension/*e17*                0 files
journal/models/generator_extension/rf_*cseed*           0 files
journal/results/tables/e17_*                            0 files
journal/results/logs/e17_*                              0 files
```

No Random-Forest fit on any E16 Rule or placebo pool has been performed. No
outcome informed this change, and no outcome could have: this amendment concerns
only which frozen table a read-only hash is compared against.

## 1. What the registration says

PREREG section 4.1 enumerates four classes of pool consumed read-only by E17:

```text
journal/datasets/synthetic/e16_rule_cseed<g>_windows_v2.npz        (20 primary)
journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz     (20 primary)
journal/datasets/synthetic/rule_cseed<g>_windows_v2.npz            ( 4 bridge)
journal/datasets/synthetic/e16_placebo_cseed<g>_windows_v2.npz     ( 4 bridge)
```

and then states, as blocking gate G1i:

> Every consumed pool's `x_digest` must equal the value frozen in
> `journal/results/tables/e16_pool_audit_v2.csv` for that realization and arm.

## 2. The factual gap

`e16_pool_audit_v2.csv` has **44 rows**, not 48:

| arm | role | rows |
|---|---|---|
| rule | primary | 20 |
| placebo | primary | 20 |
| placebo | bridge_sensitivity | 4 |
| **rule** | **bridge_sensitivity** | **0** |

The four bridge Rule pools are absent because **E16 never fit a bridge Rule
arm**. E16 PREREG section 12.9 fit only the placebo arm for the bridge
realizations, since their CNN Rule outcomes were already observed in E15 and the
frozen E15 v2 Rule checkpoints could simply be reused. E16 consequently
published no bridge Rule pool and audited none: the pools
`rule_cseed{271828,161803,141421,173205}_windows_v2.npz` are E15 v2 artifacts,
not E16 artifacts, and they live under the E15 name (no `e16_` prefix), which is
itself the tell.

E17 is in a different position and this is not incidental. **No frozen E15
Random-Forest checkpoint exists for any arm**, so E17 cannot reuse a bridge Rule
model the way E16 reused a bridge Rule CNN. PREREG section 3.2 registers this
explicitly: E17 fits **both** arms for the bridge realizations, 40 bridge fits
= 4 realizations x 2 arms x 5 pipeline seeds. Consuming those four Rule pools is
therefore required by the registered grid, not optional.

G1i as literally written can cover 44 of the 48 consumed pool slots. The
remaining 4 have no row in the named table.

## 3. The correction

G1i remains blocking over **all 48** consumed pool slots. The single change is
that the gate names, per pool, which frozen table supplies its reference value:

* **44 slots — unchanged, exactly as registered.** The 20 Rule primary, 20
  placebo primary, and 4 placebo bridge pools are verified by recomputing
  `x_digest` and requiring equality with `e16_pool_audit_v2.csv`
  (SHA-256 `98d02bd6f706c8a74694b104bca0cb26f4ff21c423471a0356fce7f83ca1ddc4`).
* **4 slots — bridge Rule.** Verified against
  `journal/results/tables/e15_rule_construction_pool_audit_v2.csv`
  (SHA-256 `e98e3fe74629e77d4a115a4716e77fef1a311ab837fb7c6d3c965b785b327c1b`),
  whose `pool_file_sha256` column freezes the identity of exactly these four
  files. The recomputed `x_digest` is published alongside for continuity with
  the other 44 rows, but the **blocking** comparison for these four is the
  frozen file digest.

Every row of `e17_pool_inheritance_audit_v1.csv` carries a `reference_table`
column and a `reference_field` column, so the split is visible in the published
artifact and is never silently waived.

### Cross-link tying the two tables to the same arrays

`journal/results/tables/e16_manipulation_checks_v2.csv`
(SHA-256 `5000d22b5414444c17af77556c9368c2f9ce5895227b0f4d4162215100b4e298`)
records `G7_e15_lineage_reproduction` as `passed = True` for all four bridge
realizations `271828`, `161803`, `141421`, `173205`. That gate is E16 asserting
that it regenerated those four Rule pools **array-for-array** from its own code
path and reproduced the frozen E15 v2 files exactly. The two frozen tables
therefore describe the same arrays, attested by E16 itself, and the substitute
reference is not a parallel or looser lineage. `e17_pool_inheritance_audit_v1.csv`
cites this cross-link in a `lineage_crosslink` column on the four bridge Rule
rows.

## 4. Why this is stricter, not weaker

* **Coverage rises from 44/48 to 48/48.** Under the literal reading, four
  consumed pools have no reference value and no comparison can be made at all.
  Under this amendment every consumed pool is hash-verified before any fit.
* **The substitute reference is a whole-file digest, which is a strictly
  stronger identity claim than an `x_digest`.** `x_digest` covers the `x` array;
  `pool_file_sha256` covers the entire NPZ, including labels, metadata,
  provenance scalars, and the embedded ordered-content digest. A pool that
  passed `x_digest` but had a mutated label vector would pass G1i on the 44 and
  fail on the 4.
* **No threshold is relaxed and no failure is tolerated.** Any mismatch on any of
  the 48 halts S2 for all realizations, exactly as registered.
* **Nothing is substituted for a value that exists.** The 44 pools with a
  registered reference are compared against that registered reference and
  against nothing else.

Verified at amendment time, before any fit:

```text
271828  file digest matches frozen E15 v2 audit   windows 260000
161803  file digest matches frozen E15 v2 audit   windows 260000
141421  file digest matches frozen E15 v2 audit   windows 260000
173205  file digest matches frozen E15 v2 audit   windows 260000
415637  rule x_digest matches frozen E16 v2 audit (spot check of the 44)
e16_manipulation_checks_v2.csv: 244 abort gates, all passed
```

## 5. Scope of what this touches

This amendment touches the **bridge sensitivity only**, and only its Rule arm.
The bridge realizations are excluded from `n`, from `df`, from the primary
interval, from the heterogeneity screen, from the margin-scale sensitivity, and
from every PREREG section 12 family except 12.9. The primary inferential set
remains the twenty generator realizations of PREREG section 3.1, whose pools are
verified against `e16_pool_audit_v2.csv` with no change whatsoever.

Had this amendment not been written, the honest alternative would have been to
report four consumed pools as unverified. It is not a route to a different
number, because a pool-identity hash has no bearing on any estimate: the pools
are read-only in E17 and their contents are fixed regardless of which table names
their digest.

## 6. Unchanged by this amendment

The primary estimand `theta_g` and the aggregation order of PREREG section 8;
the inferential unit `n = 20`, `df = 19`; the inherited margin
`delta = 0.133855625` and the prohibition on re-deriving it; `delta_strict` and
the mandatory dual-margin reporting of section 12.2; the decision order of
section 10 including the heterogeneity thresholds, the bootstrap resample count
`10000` and seed `20260807`; the frozen generator-realization and bridge seed
lists; the placebo pairing; the Random-Forest contract of section 5.1; the
identical-training-row contract of section 5.2 and gates G10c-i and G10c-ii; the
fit grid of 245; gates G0, G2i, G3i--G9i, G10a, G10d, G11--G16; the required
outputs of section 14.2; and the claim boundaries of sections 13, 16, and 17.
