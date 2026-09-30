# A-v1: known-useful-relation positive control

Status at writing: no pilot or confirmation outcome from this experiment.
This is a prospective local protocol, not a public preregistration. Its
implementation/configuration hashes are recorded before any fit by `freeze`.
It is motivated by already observed E16–E20 limitations.

## Question and scope

Can the current raw CNN and relation-visible forest recover a training-control
benefit when joint information is known to be useful and marginals carry no
class information? This validates sensitivity on a disclosed artificial task.
It does not validate realistic attack synthesis, real-attack-conditioned
augmentation, cross-position invariance, or second-source portability.

Labels 0/1 are two diagnostic construction classes, NOT Normal/Attack.
There is no real-data training scaffold in this isolated positive control.
Its classification errors must not be called operational CAN false alarms.

## Data law and matching

Each independent base produces a pair with opposite class labels. Each window
has 128 frames and the established 11-channel field layout. Other channels
are nuisance draws identical within the class pair. At fixed data0/data1
positions, data0 contains 32 copies of each value in {31,95,160,224}, shuffled.

In the intact arm, data1 equals data0 for relation class 0 or 255-data0 for
relation class 1. A Bernoulli(0.15) base-level label flip swaps the two mappings,
introducing irreducible population error and avoiding a unit-recall target.
Because the value set is complement-symmetric, both class-wise and per-window
byte multisets are identical. The relation oracle has population accuracy
0.85; its realized accuracy is reported, not substituted for a model result.

The placebo independently shuffles the same balanced data1 multiset and uses
the identical shuffled sequence for both class labels of a base. All other
channels, labels, frame positions and counts are unchanged. Therefore every
placebo pair is bit-identical with opposite labels: a deterministic classifier
has exactly 0.5 balanced accuracy on broken evaluation pairs. This is a true
label-uninformative control in this DGP; it does not assert that the E16 RPM
control removes all joint structure.

Train, validation and test use independent SHA-256-derived RNG streams.
Per realization: 1,024 training bases (2,048 windows), 256 validation bases,
512 test bases. No Car-Hacking or sealed confirmation arrays are consumed.
Inputs are regenerated from seeds, and their hashes are recorded.

## Models and paired fitting

- CNN: import the existing `CNN1D` feature extractor; binary head for the two
  diagnostic classes. Training-only per-channel standardization, identical
  for both arms up to roundoff because byte multisets are preserved. AdamW,
  lr 0.001, weight decay 0.0001, batch 512, exactly 128 optimizer steps.
  Select the best intact-validation macro-F1 every 16 steps, earliest tie.
  Both arms use the same initial state and minibatch index sequence.
- RF83: import the unchanged E18 55 marginal + 28 correlation feature map.
  160 trees, unlimited depth, min_samples_leaf=2, balanced_subsample, four
  CPU workers. Same training-row permutation and forest seed between arms.
- Pipeline seeds: 7, 42, 123, 2026, 3407, fixed rather than population-sampled.

The pilot uses two independent development realizations and pipeline 7 (8
fits). Confirmation uses twenty new realization streams and all five pipelines
(400 fits). Pilot outcomes are excluded from confirmation. Confirmation is
not changed based on whether pilot results are favorable. Technical failure
stops the run; a redesign, if needed, has a new protocol and output identity.

## Endpoints, inference and decision

All models are evaluated on intact and broken pairs with argmax decisions.
Balanced class recall equals accuracy because paired labels are balanced.
U = Rule-intact recall minus Placebo-intact recall.
L = (Rule-intact - Rule-broken) - (Placebo-intact - Placebo-broken).
Since both broken recalls are exactly 0.5 by construction, L=U here. These
are not independent confirmations of two scientific hypotheses.

Primary positive-control evidence is U at trained positions, separately for
the CNN and RF83. Average the five pipelines inside each of twenty realization
units; report means, sample SD, signs and two-sided 95% Student-t intervals.
The joint statement 'both tested families recover benefit' requires both U
intervals above zero (an intersection claim), and each mean U to recover at
least half the known population oracle advantage: U >= (0.85-0.5)/2 = 0.175.
This threshold is an artificial-task recovery criterion, not a CAN SESOI.
Any family failure is retained and blocks that joint statement. No stopping
for favorable intermediate significance and no exclusion of poor seeds.

Technical checks require exact non-target matching and multiset preservation,
opposite-label control identity, disjoint RNG keys, target-property visibility,
finite features/losses and exact broken-test balanced recall 0.5. These are
checked separately from the scientific recovery criterion.

## Outputs and preservation

New paths only: `pilot_v1/` and `confirmation_v1/` beneath this protocol.
Each contains model files, per-fit histories/metrics/predictions, data/gate
hashes, per-realization and aggregate CSV/JSON, and an explicit run status.
Existing outputs cause refusal; partial failed runs remain as evidence.
No old model, result, registration, manuscript or external dataset is changed.
