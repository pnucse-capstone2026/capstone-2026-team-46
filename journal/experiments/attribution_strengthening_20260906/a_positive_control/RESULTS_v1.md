# A-v1 results: known-useful-relation diagnostic

Completed 2026-09-06. The separately generated pilot (8 fits) was excluded from
the fixed confirmation grid: 20 realizations × 5 pipelines × 2 arms × 2 model
families = 400 fits. No realization or pipeline was removed.

| Family | Rule intact recall | Placebo intact recall | Paired U | 95% t interval for mean U |
|---|---:|---:|---:|---:|
| CNN | 0.849932 | 0.490938 | +0.358994 | [0.350702, 0.367286] |
| RF83 | 0.850098 | 0.493135 | +0.356963 | [0.350104, 0.363822] |

Each row averages pipelines within each realization before treating the 20
realizations as units. Both families had positive U in all 20 realizations and
met the prospectively fixed recovery criterion (mean U ≥ 0.175 and lower
95% bound > 0). Requiring both families to pass is a joint recovery requirement,
not a search for a successful model. These are synthetic construction-label
macro recalls; balanced class counts make them equal to accuracy here.

The population oracle accuracy is 0.85; the realized test mean is 0.850098.
Opposite-label placebo pairs are bit-identical. Consequently every evaluated
deterministic model has broken-condition recall exactly 0.5, and **L = U by
construction**. This experiment does not independently demonstrate separation
of reliance from utility. It demonstrates recoverability of a known useful
relation in this intentionally controlled diagnostic.

The data-generating process, train/validation/test random streams, models,
budget, seed grid and analysis were locally hash-frozen before the pilot and
confirmation. This is not an externally timestamped/public preregistration.
The 400 fits took 228.84 seconds on the local GB10; this small synthetic training
budget must not be used to predict the much larger real-plus-synthetic B cost.

## Checks and evidence

- All matching, label-pair identity, feature-visibility and pipeline-pairing
  checks passed. The frozen implementation and protocol hashes remain intact.
- [Verification receipt](confirmation_v1/verification_v1.json): checked all 400
  checkpoint hashes and 800 saved prediction metrics; independently recomputed
  the summary intervals; replayed 16 fixed first/last-realization and
  first/last-pipeline serialized models across both arms/families, with exact
  agreement for both test conditions. This is not independent retraining.
- [Summary](confirmation_v1/summary.json),
  [realization table](confirmation_v1/realization_summary.csv),
  [run receipt](confirmation_v1/run.json), [protocol](PROTOCOL_v1.md).

## Claim boundary

Supports: the audit can recover a designed-in useful relation with both the
existing CNN feature extractor and relation-visible RF feature map.

Does not support: useful physical CAN attack semantics, real-plus-synthetic
augmentation benefit, second-source generalization, removal of all relational
structure, or resolution of the original local U/L uncertainty. A uses a
two-class head and entirely synthetic training, not the exact five-class E16
training regime.
