#!/usr/bin/env python3
"""Pre-outcome tests for the E16 relational-response decomposition.

These tests must not compute any E16 scientific outcome and must not write any
canonical pool, checkpoint, score, or table.  They exercise the registered seed
axis, the placebo construction, and the manipulation gates on toy-scale data,
plus one full-scale lineage reproduction against the frozen E15 v2 pool.

Run:  python journal/tests/test_e16_relational_response.py
      python journal/tests/test_e16_relational_response.py --full
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "journal" / "scripts"))

import lib_common as lc  # noqa: E402
import generate_e16_rule_placebo_pairs as e16  # noqa: E402
import analyze_e16_relational_response as e16an  # noqa: E402
import run_e16_paired_training as e16train  # noqa: E402
from generate_rule_based_synthetic_v2 import (  # noqa: E402
    generate_pool,
    validate_train_source,
)

TOY_PER_ATTACK = 200
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def load_source() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(lc.WINDOWS / "train_windows.npz", allow_pickle=True)
    return validate_train_source(data)


# ---------------------------------------------------------------------------


def test_seed_axis() -> None:
    print("\n[1] registered seed axis (PREREG 3.1, 3.2)")
    drawn = e16.draw_construction_seeds()
    check("declared draw reproduces the frozen construction seed list",
          drawn == e16.CONSTRUCTION_SEEDS)
    check("twenty construction seeds", len(e16.CONSTRUCTION_SEEDS) == 20)
    check("construction seeds unique",
          len(set(e16.CONSTRUCTION_SEEDS)) == 20)
    check("no intersection with the E15 lineage",
          not set(e16.CONSTRUCTION_SEEDS) & set(e16.E15_LINEAGE))
    check("all seeds inside the declared frame",
          all(e16.SEED_FRAME[0] <= s <= e16.SEED_FRAME[1]
              for s in e16.CONSTRUCTION_SEEDS))

    derived = [e16.placebo_seed(g) for g in e16.CONSTRUCTION_SEEDS]
    check("placebo seeds unique", len(set(derived)) == 20)
    check("placebo seeds disjoint from construction seeds",
          not set(derived) & set(e16.CONSTRUCTION_SEEDS))
    expected = int.from_bytes(
        hashlib.sha256(b"e16-placebo|415637").digest()[-4:], "big")
    check("placebo derivation matches the registered formula",
          e16.placebo_seed(415637) == expected, f"{expected}")
    check("seed-axis gate passes", e16.assert_seed_axis()["passed"])


def test_mirror_equivalence(train_x, normal_idx) -> None:
    print("\n[2] position-recording mirror reproduces generate_pool (gate G1a)")
    for seed in (e16.CONSTRUCTION_SEEDS[0], e16.CONSTRUCTION_SEEDS[7]):
        pair = e16.generate_pair(
            train_x, normal_idx,
            per_attack=TOY_PER_ATTACK, construction_seed=seed)
        canonical = generate_pool(
            train_x, normal_idx,
            per_attack=TOY_PER_ATTACK, construction_seed=seed)
        mine = e16.arrays_digest(pair.rule)
        theirs = e16.arrays_digest(canonical["arrays"])
        bad = [k for k in e16.ARRAY_KEYS if mine[k] != theirs[k]]
        check(f"construction {seed} mirrors generate_pool array-for-array",
              not bad, f"mismatched={bad}" if bad else "9/9 arrays")


def test_placebo_construction(train_x, normal_idx) -> None:
    print("\n[3] placebo preserves marginals and destroys the relation")
    seed = e16.CONSTRUCTION_SEEDS[0]
    pair = e16.generate_pair(
        train_x, normal_idx,
        per_attack=TOY_PER_ATTACK, construction_seed=seed)
    x_rule, x_plac = pair.rule["x"], pair.placebo_x
    y = pair.rule["y_attack_type"]
    spoof = np.isin(y, e16.SPOOF_LABELS)

    check("arms have identical shape", x_rule.shape == x_plac.shape)
    check("DoS/Fuzzy windows bit-identical",
          np.array_equal(x_rule[~spoof], x_plac[~spoof]))
    other = [c for c in range(x_rule.shape[2]) if c != e16.DATA1]
    check("every non-data1 channel identical on spoof windows",
          np.array_equal(x_rule[spoof][:, :, other], x_plac[spoof][:, :, other]))
    check("per-window data1 multiset preserved",
          np.array_equal(np.sort(x_rule[spoof][:, :, e16.DATA1], axis=1),
                         np.sort(x_plac[spoof][:, :, e16.DATA1], axis=1)))
    check("placebo actually differs from Rule",
          not np.array_equal(x_rule[spoof], x_plac[spoof]))

    # relation residual, measured directly on the published arrays
    gear_rows = np.flatnonzero(y == e16.GEAR_LABEL)
    residual_hits = kept = 0
    for row in gear_rows:
        count = int(pair.position_count[row])
        pos = pair.positions[row, :count].astype(np.int64)
        d0 = x_plac[row][pos, e16.DATA0]
        d1 = x_plac[row][pos, e16.DATA1]
        residual_hits += int((np.abs(255.0 - d0 - d1) <= 8).sum())
        kept += count
    fraction = residual_hits / max(kept, 1)
    check("Gear inverse relation destroyed below the 1% threshold",
          fraction < e16.GEAR_RESIDUAL_MAX_FRACTION, f"residual={fraction:.6f}")

    rpm_rows = np.flatnonzero(y == e16.RPM_LABEL)
    coupled = 0
    for row in rpm_rows:
        count = int(pair.position_count[row])
        pos = pair.positions[row, :count].astype(np.int64)
        d0 = x_plac[row][pos, e16.DATA0].astype(np.int64)
        d1 = x_plac[row][pos, e16.DATA1].astype(np.int64)
        coupled += int((((d1 - 8 * d0) % 256) < 8).sum())
    check("RPM modular coupling fully destroyed", coupled == 0,
          f"residual matches={coupled}")

    # the Rule arm must still carry the relation, else the contrast is empty
    rule_hits = rule_kept = 0
    for row in gear_rows:
        count = int(pair.position_count[row])
        pos = pair.positions[row, :count].astype(np.int64)
        d0 = x_rule[row][pos, e16.DATA0]
        d1 = x_rule[row][pos, e16.DATA1]
        rule_hits += int((np.abs(255.0 - d0 - d1) <= 8).sum())
        rule_kept += count
    rule_fraction = rule_hits / max(rule_kept, 1)
    check("Rule arm retains the Gear relation", rule_fraction > 0.95,
          f"retained={rule_fraction:.6f}")


def test_gates(train_x, normal_idx) -> None:
    print("\n[4] manipulation gates (PREREG 15.2)")
    seed = e16.CONSTRUCTION_SEEDS[3]
    pair = e16.generate_pair(
        train_x, normal_idx,
        per_attack=TOY_PER_ATTACK, construction_seed=seed)
    canonical = generate_pool(
        train_x, normal_idx,
        per_attack=TOY_PER_ATTACK, construction_seed=seed)
    checks = e16.evaluate_pair_gates(pair, e16.arrays_digest(canonical["arrays"]))
    names = {c["check"] for c in checks}
    for required in ("G1a_mirror_reproduces_generate_pool",
                     "G1b_non_data1_channels_identical",
                     "G1c_data1_untouched_off_injected_frames",
                     "G3_data1_multiset_identical",
                     "G4_metadata_internally_consistent",
                     "G4_per_class_quota",
                     "G5_gear_residual_within_tolerance_fraction",
                     "G5_rpm_residual_attains_structural_minimum",
                     "G5_rpm_residual_fraction",
                     "G6_dos_fuzzy_bit_identical"):
        check(f"gate present: {required}", required in names)
    check("no abort gate fails on a valid pair",
          not e16.failed_gates(checks), str(e16.failed_gates(checks)))

    # a corrupted placebo must be caught
    tampered = e16.PairResult(
        construction_seed=pair.construction_seed,
        placebo_seed=pair.placebo_seed,
        rule=pair.rule,
        placebo_x=pair.rule["x"].copy(),   # relation NOT destroyed
        positions=pair.positions,
        position_count=pair.position_count,
        stats={"gear": {"offbin": 0, "frames": 1,
                        "residual_within_tolerance": 1},
               "rpm": {"residual_matches": 7, "structural_minimum": 0,
                       "frames": 1, "structurally_constrained_windows": 0,
                       "suboptimal_windows": 1}},
    )
    bad = e16.failed_gates(
        e16.evaluate_pair_gates(tampered, e16.arrays_digest(canonical["arrays"])))
    check("gates reject a placebo whose relation was not destroyed",
          "G5_gear_residual_within_tolerance_fraction" in bad
          and "G5_rpm_residual_attains_structural_minimum" in bad,
          f"caught={bad}")


def test_output_identity() -> None:
    print("\n[5] output identity and no-clobber surface (PREREG 5.4)")
    paths = e16.target_paths()
    expected = 20 + 20 + 4 + 3
    check(f"target surface is {expected} artifacts", len(paths) == expected,
          f"got {len(paths)}")
    check("all target paths are unique", len(set(paths)) == len(paths))
    # Before stage S2 every target must be absent; after S2 every target must
    # be present.  A mixture means a partially published stage, which the
    # atomic publication is supposed to make impossible.
    present = [p for p in paths if p.exists()]
    check("pool stage is either wholly unpublished or wholly published",
          len(present) in (0, len(paths)),
          f"{len(present)}/{len(paths)} present")
    if present:
        check("every published pool is non-empty",
              all(p.stat().st_size > 0 for p in present))
    check("rule and placebo paths differ per construction",
          e16.rule_pool_path(415637) != e16.placebo_pool_path(415637))
    check("no target collides with an E14/E15 artifact",
          not [p for p in paths
               if p.name.startswith(("rule_cseed", "rule_based", "rule_placebo"))])
    for path in paths:
        if path.suffix == ".npz":
            check(f"versioned name: {path.name}",
                  e16.OUTPUT_VERSION in path.name and path.name.startswith("e16_"))
            break


def test_full_scale_lineage(train_x, normal_idx) -> None:
    print("\n[6] full-scale E15 lineage reproduction (gate G7)")
    seed = e16.BRIDGE_SEEDS[0]
    pair = e16.generate_pair(
        train_x, normal_idx,
        per_attack=e16.PER_ATTACK, construction_seed=seed)
    result = e16._bridge_lineage_check(pair)
    check(f"construction {seed} reproduces the frozen E15 v2 pool",
          result["passed"], result["value"] or "9/9 arrays match")
    gear = pair.stats["gear"]
    rpm = pair.stats["rpm"]
    check("Gear residual below threshold at full scale",
          gear["residual_within_tolerance"] / gear["frames"]
          < e16.GEAR_RESIDUAL_MAX_FRACTION,
          f"{gear['residual_within_tolerance'] / gear['frames']:.8f}")
    check("RPM coupling fully destroyed at full scale",
          rpm["residual_matches"] == 0)


def test_training_grid() -> None:
    print("\n[7] training grid (PREREG 6)")
    jobs = e16train.build_training_grid()
    check("220 registered fits", len(jobs) == 220, f"got {len(jobs)}")
    primary = [j for j in jobs if j.role == "primary"]
    bridge = [j for j in jobs if j.role == "bridge_sensitivity"]
    check("200 primary fits (20 x 2 arms x 5 pipelines)", len(primary) == 200)
    check("20 bridge placebo fits (4 x 5)", len(bridge) == 20)
    check("bridge fits are placebo-only",
          all(j.arm == "placebo" for j in bridge))
    check("job keys unique", len({j.key for j in jobs}) == len(jobs))
    check("checkpoint paths unique",
          len({j.checkpoint_path for j in jobs}) == len(jobs))
    check("no checkpoint name collides with E14/E15",
          not [j for j in jobs
               if "_e14_" in j.checkpoint_path.name
               or "_e15_" in j.checkpoint_path.name])
    check("every job maps to an E16 pool path",
          all(j.pool_path.name.startswith("e16_") for j in jobs))
    # Training publishes one fit at a time, so a partial grid is a legitimate
    # mid-run state.  The invariant that must hold at every instant is that a
    # published checkpoint always has its matching log: the pair is published
    # as one atomic bundle, so a checkpoint without a log means a torn publish.
    orphans = [j.key for j in jobs
               if j.checkpoint_path.exists() != j.log_path.exists()]
    check("every published fit has both its checkpoint and its log",
          not orphans, f"orphaned={orphans[:5]}")
    done = [j for j in jobs if j.checkpoint_path.exists()]
    check("published fits follow the registered grid order",
          [j.key for j in done] == [j.key for j in jobs[:len(done)]],
          f"{len(done)}/{len(jobs)} published")


def test_evaluation_identities() -> None:
    print("\n[8] evaluation identities (PREREG 7, 15.3)")
    import evaluate_e16_relational_response as e16ev
    identities = e16ev.build_identities()
    check("245 identities (5 real + 200 primary + 20 + 20 bridge)",
          len(identities) == 245, f"got {len(identities)}")
    shared = [i for i in identities if i["role"] == "shared_reference"]
    check("five shared matched-real references", len(shared) == 5)
    check("shared references are the E14 checkpoints",
          all("_e14_v1_" in i["checkpoint"].name for i in shared))

    bridge_rule = [i for i in identities
                   if i["role"] == "bridge_sensitivity" and i["arm"] == "rule"]
    check("bridge sensitivity has a Rule arm", len(bridge_rule) == 20,
          f"got {len(bridge_rule)}")
    check("bridge Rule arm reuses the frozen E15 checkpoints",
          all("_e15_v2_" in i["checkpoint"].name for i in bridge_rule))
    check("all bridge Rule checkpoints already exist",
          all(i["checkpoint"].exists() for i in bridge_rule))
    check("bridge constructions have both arms",
          {i["construction_seed"] for i in bridge_rule}
          == set(e16train.BRIDGE_SEEDS))
    check("model keys unique",
          len({i["model_key"] for i in identities}) == len(identities))
    check("scoring device stays on CPU", e16ev.SCORING_DEVICE == "cpu")
    check("continuity grid is the registered 495 rows",
          e16ev.CONTINUITY_ROWS == 495)

    e14_scenario = pd.read_csv(e16ev.E14_SCENARIO_TABLE)
    e14_normal = pd.read_csv(e16ev.E14_NORMAL_TABLE)
    real_scenario = e14_scenario[e14_scenario["arm"].astype(str) == "real_ms"]
    real_normal = e14_normal[e14_normal["arm"].astype(str) == "real_ms"]
    check("E14 record supplies 480 shared-real scenario rows",
          len(real_scenario) == 480, f"got {len(real_scenario)}")
    check("E14 record supplies 15 shared-real normal rows",
          len(real_normal) == 15, f"got {len(real_normal)}")


def test_analyzer_reproduces_e14() -> None:
    print("\n[9] analyzer reproduces the published E14 effects")
    e14 = pd.read_csv(
        REPO / "journal" / "results" / "tables"
        / "e14_l4_factorial_by_scenario_v1.csv")
    mapping = {"real_ms": "real_ms", "rule_ms": "rule", "placebo_ms": "placebo"}
    frame = e14[e14["arm"].isin(mapping)].copy()
    frame["arm"] = frame["arm"].map(mapping)
    frame["construction_seed"] = np.where(
        frame["arm"] == "real_ms", 0, 314159)
    frame["pipeline_seed"] = frame["seed"]
    frame["role"] = "primary"

    deltas = e16an.build_augmentation_deltas(frame)
    _, per_construction = e16an.build_effects_by_construction(deltas)

    published = {
        "rule": {"P": -0.879958, "S": 0.019858, "D": -0.059175,
                 "PS": 0.037800, "PD": -0.117367, "SD": -0.000700,
                 "PSD": -0.003500},
        "placebo": {"P": -0.943350, "S": 0.030542, "D": -0.039267,
                    "PS": 0.059167, "PD": -0.077550, "SD": -0.020667,
                    "PSD": -0.043433},
    }
    worst = 0.0
    for arm, effects in published.items():
        for effect, expected in effects.items():
            got = float(per_construction[
                (per_construction["arm"] == arm)
                & (per_construction["effect"] == effect)]["value"].iloc[0])
            worst = max(worst, abs(got - expected))
    check("all 14 published E14 arm effects reproduced", worst < 1e-5,
          f"max |difference| = {worst:.2e}")

    decomposition = e16an.build_decomposition(per_construction, effect="P")
    theta = float(decomposition["theta_rule_minus_placebo"].iloc[0])
    check("theta reproduces the published +0.063392",
          abs(theta - 0.063392) < 1e-5, f"got {theta:+.6f}")
    gate = e16an.assert_decomposition_closure(decomposition)
    check("decomposition identity closes (gate G11)", gate["passed"],
          f"max residual {gate['max_abs_residual']:.2e}")

    # the interaction divisor must be 2 ** (3 - k), not a constant
    cells = {(p, s, d): float(p * 4 + s * 2 + d)
             for p in (0, 1) for s in (0, 1) for d in (0, 1)}
    check("main-effect divisor is 4",
          abs(e16an.factor_contrast(cells, "P") - 4.0) < 1e-12)
    check("two-way divisor is 2",
          abs(e16an.factor_contrast(cells, "PS") - 0.0) < 1e-12)


def test_decision_rule() -> None:
    print("\n[10] registered decision rule (PREREG 10 as amended 2026-07-25)")
    rng = np.random.default_rng(1)
    scenarios = [
        ("planning value theta=+0.063", rng.normal(0.063392, 0.04, 20), "A"),
        ("no relational contribution", rng.normal(0.0, 0.03, 20), "A"),
        ("uniform large +0.30", rng.normal(0.30, 0.04, 20), "B"),
        ("uniform large -0.25", rng.normal(-0.25, 0.04, 20), "B"),
        ("material +0.18 (20% of total)", rng.normal(0.18, 0.04, 20), "B"),
        ("directional split +/-0.5",
         np.array([0.5] * 5 + [-0.5] * 5 + [0.02] * 10), "C"),
        ("aggregate equivalent, units outside",
         np.array([0.2] * 4 + [-0.2] * 4 + [0.0] * 12), "C"),
        ("wide dispersion with real split", rng.normal(0.10, 0.35, 20), "C"),
        ("straddling delta", rng.normal(0.134, 0.09, 20), "D"),
        ("one-sided wide spread", rng.normal(0.16, 0.13, 20), "D"),
    ]
    for name, theta, expected in scenarios:
        result = e16an.primary_decision(theta)
        check(f"{name} -> Branch {expected}",
              result["verdict"] == expected,
              f"got {result['verdict']}")

    check("a uniformly large effect is NOT called heterogeneous",
          not e16an.primary_decision(
              rng.normal(0.30, 0.04, 20))["heterogeneity_triggered"])
    check("delta is the registered constant",
          e16an.DELTA == 0.133855625, str(e16an.DELTA))
    check("delta equals multiplier x denominator",
          abs(e16an.SESOI_MULTIPLIER * e16an.SESOI_DENOMINATOR
              - e16an.DELTA) < 1e-15)
    check("equivalence uses the 90% TOST interval",
          e16an.EQUIVALENCE_CONFIDENCE == 0.90)
    check("bootstrap is seeded and reproducible",
          e16an.primary_decision(theta)["bootstrap_ci90_low"]
          == e16an.primary_decision(theta)["bootstrap_ci90_low"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="also run the full-scale 65,000-per-class lineage test")
    args = parser.parse_args()

    print("E16 pre-outcome tests — no outcome is computed, nothing is published")
    test_seed_axis()
    test_output_identity()
    train_x, normal_idx = load_source()
    test_mirror_equivalence(train_x, normal_idx)
    test_placebo_construction(train_x, normal_idx)
    test_gates(train_x, normal_idx)
    if args.full:
        test_full_scale_lineage(train_x, normal_idx)
    else:
        print("\n[6] full-scale lineage test skipped (pass --full to run)")
    test_training_grid()
    test_evaluation_identities()
    test_analyzer_reproduces_e14()
    test_decision_rule()

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("all E16 pre-outcome tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
