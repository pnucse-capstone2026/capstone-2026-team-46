#!/usr/bin/env python3
"""Pre-outcome tests for the E17 Random-Forest relational replication.

These tests must not compute any E17 scientific outcome and must not write any
canonical checkpoint, score, or table.  They exercise the registered grid, the
Random-Forest contract, the pool-inheritance and pairing gates, the aggregation
order, the decomposition identity, the dual-margin decision rule, and the
saturation determination on toy-scale data.

Every gate assertion below is written as a *negative control* wherever the gate
is meant to be blocking: a gate that cannot fail is not a gate, so each one is
fed an input that must make it raise.

Run:  python journal/tests/test_e17_rf_relational_replication.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "journal" / "scripts"))

import analyze_e17_rf_relational_response as e17an  # noqa: E402
import run_e17_rf_paired_training as e17train  # noqa: E402
import train_generator_extension_rf as rfmod  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def raises(name: str, error: type[BaseException], call) -> None:
    try:
        call()
    except error as exc:
        check(name, True, str(exc)[:70])
        return
    except BaseException as exc:  # noqa: BLE001 - surface the wrong error type
        check(name, False, f"raised {type(exc).__name__}, expected "
                           f"{error.__name__}")
        return
    check(name, False, "did not raise")


# ---------------------------------------------------------------------------


def test_seed_axis_and_grid() -> None:
    print("\n[1] registered seed axis and fit grid (PREREG 3.1, 3.2, 6)")
    check("twenty generator realizations",
          len(e17train.CONSTRUCTION_SEEDS) == 20)
    check("four bridge realizations", len(e17train.BRIDGE_SEEDS) == 4)
    check("realization list equals the PREREG 3.1 literal, in order",
          tuple(e17train.CONSTRUCTION_SEEDS)
          == e17train.REGISTERED_CONSTRUCTION_SEEDS)
    check("bridge list equals the PREREG 3.2 literal, in order",
          tuple(e17train.BRIDGE_SEEDS) == e17train.REGISTERED_BRIDGE_SEEDS)
    check("pipeline seeds frozen",
          e17train.PIPELINE_SEEDS == (7, 42, 123, 2026, 3407))
    check("seed-axis gate passes", e17train.verify_seed_axis()["passed"])

    grid = e17train.build_training_grid()
    check("245 registered fits", len(grid) == 245, f"{len(grid)}")
    roles = {r: sum(1 for j in grid if j.role == r)
             for r in {j.role for j in grid}}
    check("200 primary fits", roles.get("primary") == 200)
    check("40 bridge fits", roles.get("bridge_sensitivity") == 40)
    check("5 shared reference fits", roles.get("shared_reference") == 5)
    check("bridge fits both arms",
          {j.arm for j in grid if j.role == "bridge_sensitivity"}
          == {"rule", "placebo"})
    check("references are scored first in registered order",
          all(j.role == "shared_reference" for j in grid[:5]))
    check("48 consumed pool slots", len(e17train.consumed_pool_slots()) == 48)

    names = {j.checkpoint_path.name for j in grid}
    check("245 distinct checkpoint names", len(names) == 245)
    check("checkpoint naming matches PREREG 5.3",
          "rf_rule_0p30_cseed415637_e17_v1_seed7.joblib" in names
          and "rf_real_only_e17_v1_seed7.joblib" in names)
    check("no legacy rf_real_only_seed<p> path is targeted",
          not any(n == f"rf_real_only_seed{p}.joblib"
                  for n in names for p in e17train.PIPELINE_SEEDS))

    bridge_rule = e17train.E17Job(271828, "rule", 7, "bridge_sensitivity")
    check("bridge Rule consumes the frozen E15 v2 pool",
          bridge_rule.pool_path.name == "rule_cseed271828_windows_v2.npz")
    primary_rule = e17train.E17Job(415637, "rule", 7, "primary")
    check("primary Rule consumes the E16 pool",
          primary_rule.pool_path.name == "e16_rule_cseed415637_windows_v2.npz")


def test_rf_contract() -> None:
    print("\n[2] Random-Forest contract fidelity (PREREG 5.1, gate G14)")
    gate = e17train.verify_rf_contract()
    check("G14 passes", gate["passed"])
    check("feature function is the imported object, not a copy",
          e17train.rf_features is rfmod.rf_features)
    check("55-dimensional features", gate["feature_dimension"] == 55)
    check("no standardizer", gate["standardizer_constructed"] is False)
    check("estimator matches the frozen harness source",
          gate["estimator_matches_frozen_harness_source"])
    for key, value in (("n_estimators", 160), ("max_depth", None),
                       ("min_samples_leaf", 2),
                       ("class_weight", "balanced_subsample")):
        check(f"{key} = {value!r}", gate["upstream_keywords"][key] == value)
    check("random_state is the pipeline seed",
          gate["upstream_keywords"]["random_state"] == "seed")

    rng = np.random.default_rng(0)
    toy = rng.normal(size=(7, 128, 11)).astype(np.float32)
    features = e17train.rf_features(toy)
    check("toy features are (7, 55) float32",
          features.shape == (7, 55) and features.dtype == np.float32)
    expected_mean = toy.mean(axis=1)
    check("first eleven columns are the per-channel means",
          np.allclose(features[:, :11], expected_mean, atol=1e-6))
    expected_delta = toy[:, -1, :] - toy[:, 0, :]
    check("last eleven columns are the first-to-last differences",
          np.allclose(features[:, 44:], expected_delta, atol=1e-6))
    check("row-wise: featurizing a subset equals subsetting the features",
          np.array_equal(e17train.rf_features(toy[[5, 1, 3]]),
                         features[[5, 1, 3]]))

    raises("the standardizer sentinel fires when called",
           e17train.E17TrainingError,
           lambda: e17train._forbidden_standardizer(np.zeros((2, 3))))


def test_pairing_gates() -> None:
    print("\n[3] pairing and inheritance gates (PREREG 5.2, G10c-i, G10c-ii)")
    paired = [
        {"construction_seed": g, "arm": arm, "pipeline_seed": p,
         "role": "primary", "sampling_index_sha256": f"{g}:{p}"}
        for g in e17train.CONSTRUCTION_SEEDS for arm in ("rule", "placebo")
        for p in e17train.PIPELINE_SEEDS
    ] + [
        {"construction_seed": g, "arm": arm, "pipeline_seed": p,
         "role": "bridge_sensitivity", "sampling_index_sha256": f"{g}:{p}"}
        for g in e17train.BRIDGE_SEEDS for arm in ("rule", "placebo")
        for p in e17train.PIPELINE_SEEDS
    ]
    gate = e17train.verify_internal_pairing(paired)
    check("G10c-i passes on 120 aligned pairs", gate["pairs_checked"] == 120)

    broken = [dict(row) for row in paired]
    broken[1]["sampling_index_sha256"] = "divergent"
    raises("G10c-i fires when one arm's draw diverges",
           e17train.E17TrainingError,
           lambda: e17train.verify_internal_pairing(broken))

    dropped = [row for row in paired if not (
        row["construction_seed"] == e17train.CONSTRUCTION_SEEDS[0]
        and row["arm"] == "placebo" and row["pipeline_seed"] == 7)]
    raises("G10c-i fires when a pair is incomplete",
           e17train.E17TrainingError,
           lambda: e17train.verify_internal_pairing(dropped))

    frozen = pd.read_csv(e17train.E16_TRAINING_MANIFEST)
    inherited = [
        {"construction_seed": int(r.construction_seed), "arm": r.arm,
         "pipeline_seed": int(r.pipeline_seed), "role": r.role,
         "sampling_index_sha256": r.sampling_index_sha256}
        for r in frozen.itertuples()
    ] + [
        {"construction_seed": g, "arm": "rule", "pipeline_seed": p,
         "role": "bridge_sensitivity", "sampling_index_sha256": "bridge"}
        for g in e17train.BRIDGE_SEEDS for p in e17train.PIPELINE_SEEDS
    ]
    gate = e17train.verify_cross_family_inheritance(inherited)
    check("G10c-ii checks 220 fits", gate["fits_checked"] == 220)
    check("G10c-ii records 20 bridge Rule exemptions",
          len(gate["exempt_fits"]) == 20)
    check("the exemption states its reason",
          "no E16 CNN counterpart" in gate["exemption_reason"])

    violated = [dict(row) for row in inherited]
    violated[0]["sampling_index_sha256"] = "not-the-e16-draw"
    raises("G10c-ii fires when a draw differs from the frozen E16 CNN draw",
           e17train.E17TrainingError,
           lambda: e17train.verify_cross_family_inheritance(violated))

    smuggled = [dict(row) for row in inherited]
    smuggled.append({"construction_seed": 999999, "arm": "rule",
                     "pipeline_seed": 7, "role": "primary",
                     "sampling_index_sha256": "x"})
    raises("G10c-ii refuses an unregistered fit with no frozen counterpart",
           e17train.E17TrainingError,
           lambda: e17train.verify_cross_family_inheritance(smuggled))


def _toy_scenario(theta_shift: float = 0.0, *, p0_rule: float = 0.999,
                  p0_placebo: float = 0.999) -> pd.DataFrame:
    """A toy scored table with a known, exactly reconstructible theta.

    Recall is set analytically per cell so the registered aggregation order has
    a closed-form answer: only the ``P = 1`` canonical cells carry the Rule and
    placebo signal, so ``theta`` must come out as ``theta_shift`` plus the
    ``P = 0`` imbalance.
    """
    rows: list[dict[str, object]] = []
    arms = [("real_rf", 0, "shared_reference")]
    arms += [(arm, g, "primary") for g in (11, 22, 33) for arm in
             ("rule", "placebo")]
    for arm, seed, role in arms:
        for pipeline in (7, 42):
            for block in ("block_01", "block_02"):
                for label, attack in ((3, "Gear"), (4, "RPM")):
                    for stratum in ("canonical", "shifted"):
                        for p in (0, 1):
                            for s in (0, 1):
                                for d in (0, 1):
                                    if arm == "real_rf":
                                        value = 0.02 if p == 1 else 0.001
                                    elif p == 1:
                                        value = 0.30 + (
                                            theta_shift if arm == "rule"
                                            else 0.0)
                                    else:
                                        value = (p0_rule if arm == "rule"
                                                 else p0_placebo)
                                    rows.append({
                                        "arm": arm, "construction_seed": seed,
                                        "pipeline_seed": pipeline,
                                        "role": role, "block_id": block,
                                        "attack": attack,
                                        "attack_label": label,
                                        "id_stratum": stratum,
                                        "P": p, "S": s, "D": d,
                                        "cell": f"{p}{s}{d}", "n": 100,
                                        "exact_recall": value,
                                        "binary_recall": min(1.0, value + 0.01),
                                    })
    return pd.DataFrame(rows)


def test_aggregation_and_closure() -> None:
    print("\n[4] aggregation order, decomposition identity (PREREG 8, G11)")
    scenario = _toy_scenario(theta_shift=0.04)
    deltas = e17an.build_augmentation_deltas(scenario)
    check("deltas exclude the real arm",
          "real_rf" not in set(deltas["arm"]))
    _pipeline, per_realization = e17an.build_effects_by_realization(deltas)
    decomposition = e17an.build_decomposition(per_realization, effect="P")
    check("one row per realization", len(decomposition) == 3)
    gate = e17an.assert_decomposition_closure(decomposition)
    check("G11 closes to floating point", gate["passed"],
          f"max |residual| = {gate['max_abs_residual']:.3e}")
    theta = decomposition["theta_rule_minus_placebo"].to_numpy(float)
    check("theta recovers the injected shift exactly",
          np.allclose(theta, 0.04, atol=1e-12), f"{theta[0]!r}")

    broken = decomposition.copy()
    broken.loc[0, "closure_residual"] = 1e-6
    raises("G11 fires on a broken identity", e17an.E17AnalysisError,
           lambda: e17an.assert_decomposition_closure(broken))

    halves = decomposition.copy()
    halves.loc[0, "theta_p0_component"] += 0.5
    raises("G11 fires when the P=1/P=0 halves do not reconstruct theta",
           e17an.E17AnalysisError,
           lambda: e17an.assert_decomposition_closure(halves))

    unbalanced = e17an.build_decomposition(
        e17an.build_effects_by_realization(
            e17an.build_augmentation_deltas(
                _toy_scenario(theta_shift=0.04, p0_rule=0.90,
                              p0_placebo=0.80)))[1], effect="P")
    theta_unbalanced = unbalanced["theta_rule_minus_placebo"].to_numpy(float)
    check("an unsaturated P=0 cell moves theta by the in-grammar imbalance",
          np.allclose(theta_unbalanced, 0.04 - 0.10, atol=1e-12),
          f"{theta_unbalanced[0]!r}")


def test_decision_rule() -> None:
    print("\n[5] registered decision order and dual margins (PREREG 10, 12.2)")
    delta = e17an.DELTA
    check("delta is inherited verbatim", delta == 0.133855625)
    check("delta equals its recorded provenance",
          abs(delta - e17an.SESOI_MULTIPLIER * e17an.SESOI_DENOMINATOR) < 1e-15)
    check("bootstrap seed is the E17 registered value",
          e17an.BOOTSTRAP_SEED == 20260807)
    check("bootstrap resamples frozen", e17an.BOOTSTRAP_RESAMPLES == 10_000)

    rng = np.random.default_rng(7)
    tight = rng.normal(0.0, 0.01, 20)
    branch_a = e17an.primary_decision(tight, delta=delta)
    check("a tight null returns Branch A", branch_a["verdict"] == "A",
          branch_a["verdict_reason"][:60])

    large = np.full(20, 0.30) + rng.normal(0.0, 0.01, 20)
    branch_b = e17an.primary_decision(large, delta=delta)
    check("a uniform large effect returns Branch B",
          branch_b["verdict"] == "B")
    check("Branch B is not misread as heterogeneity",
          not branch_b["heterogeneity_triggered"])

    split = np.concatenate([np.full(10, 0.30), np.full(10, -0.30)])
    branch_c = e17an.primary_decision(split, delta=delta)
    check("directional disagreement returns Branch C",
          branch_c["verdict"] == "C" and branch_c["heterogeneity_c1_directional"])

    wide = rng.normal(0.0, 0.30, 20)
    branch_d = e17an.primary_decision(wide, delta=delta)
    check("a dispersed null is not promoted above Branch D",
          branch_d["verdict"] in ("C", "D"), branch_d["verdict"])

    check("the same theta can flip verdict under a stricter margin",
          e17an.primary_decision(np.full(20, 0.05) + rng.normal(0, 0.005, 20),
                                 delta=delta)["verdict"]
          != e17an.primary_decision(np.full(20, 0.05) + rng.normal(0, 0.005, 20),
                                    delta=0.01)["verdict"])

    check("the decision is reproducible for a fixed theta vector",
          e17an.primary_decision(tight, delta=delta)
          == e17an.primary_decision(tight, delta=delta))


def test_margin_and_saturation() -> None:
    print("\n[6] margin sensitivity and P=0 saturation (PREREG 12.2, 12.3)")
    scenario = _toy_scenario(theta_shift=0.01)
    deltas = e17an.build_augmentation_deltas(scenario)
    _pipeline, per_realization = e17an.build_effects_by_realization(deltas)
    decomposition = e17an.build_decomposition(per_realization, effect="P")
    theta = decomposition["theta_rule_minus_placebo"].to_numpy(float)
    _p1_table, p1_summary = e17an.build_cell_comparison(scenario, p_level=1)
    primary = e17an.primary_decision(theta, delta=e17an.DELTA)
    table, record = e17an.build_margin_sensitivity(theta, p1_summary, primary)
    check("both margins are reported in one table", len(table) == 2)
    check("G_placebo is the placebo P=1 gain over the reference",
          abs(record["g_placebo"] - (0.30 - 0.02)) < 1e-12,
          f"{record['g_placebo']!r}")
    check("delta_strict = 0.15 x G_placebo",
          abs(record["delta_strict"] - 0.15 * record["g_placebo"]) < 1e-15)
    check("the ratio delta_strict/delta is published",
          record["delta_strict_over_delta"] is not None)
    check("margin_disagreement is a boolean",
          isinstance(record["margin_disagreement"], bool))

    degenerate = {"equal_macro": dict(p1_summary["equal_macro"],
                                      placebo_gain_mean=0.004)}
    _table, degraded = e17an.build_margin_sensitivity(
        theta, degenerate, primary)
    check("a degenerate denominator yields an undefined strict margin",
          degraded["verdict_strict_margin"] == "undefined"
          and degraded["delta_strict"] is None)
    check("the observed G_placebo is still published when undefined",
          degraded["g_placebo"] == 0.004)
    check("no substitute denominator is invented",
          "invents no substitute" in
          _table.loc[1, "verdict_reason"])

    _p0_table, p0_summary = e17an.build_cell_comparison(scenario, p_level=0)
    saturation = e17an.determine_p0_saturation(p0_summary, decomposition)
    check("a matched 0.999/0.999 in-grammar cell is declared saturated",
          saturation["saturated"] and saturation["determination"] == "saturated")
    check("a saturated cell cancels from theta",
          saturation["in_grammar_cell_cancels_from_theta"]
          and abs(saturation["theta_p0_contribution_mean"]) < 1e-12)

    unsat_scenario = _toy_scenario(theta_shift=0.01, p0_rule=0.90,
                                   p0_placebo=0.80)
    unsat_decomposition = e17an.build_decomposition(
        e17an.build_effects_by_realization(
            e17an.build_augmentation_deltas(unsat_scenario))[1], effect="P")
    _t, unsat_summary = e17an.build_cell_comparison(unsat_scenario, p_level=0)
    unsat = e17an.determine_p0_saturation(unsat_summary, unsat_decomposition)
    check("an imbalanced in-grammar cell is declared unsaturated",
          not unsat["saturated"] and unsat["determination"] == "unsaturated")
    check("the unsaturated P=0 contribution is reported separately",
          abs(unsat["theta_p0_contribution_mean"] - 0.10) < 1e-12)


def test_g16_reporting_completeness() -> None:
    print("\n[7] dual-margin reporting completeness (gate G16)")
    complete = {
        "verdict": "A", "verdict_strict_margin": "B",
        "margin_disagreement": True, "g_placebo": 0.2,
        "p1_cell_summary": {"equal_macro": {
            "real_recall": 0.02, "rule_recall_mean": 0.3,
            "placebo_recall_mean": 0.3, "absolute_difference_mean": 0.0,
            "absolute_difference_sd": 0.01,
            "absolute_difference_ci95_low": -0.01,
            "absolute_difference_ci95_high": 0.01,
            "ratio_median": 1.0, "ratio_realizations_used": 20}},
        "p0_saturation": {"determination": "saturated"},
    }
    gate = e17an.verify_dual_margin_reporting(complete)
    check("G16 passes on a complete record", gate["passed"])

    for field in ("verdict", "verdict_strict_margin", "margin_disagreement"):
        broken = dict(complete)
        broken.pop(field)
        raises(f"G16 fires when {field} is missing", e17an.E17AnalysisError,
               lambda b=broken: e17an.verify_dual_margin_reporting(b))

    no_p1 = dict(complete)
    no_p1["p1_cell_summary"] = {"equal_macro": {"rule_recall_mean": 0.3}}
    raises("G16 fires when the P=1 absolute scale is missing",
           e17an.E17AnalysisError,
           lambda: e17an.verify_dual_margin_reporting(no_p1))

    no_p0 = dict(complete)
    no_p0["p0_saturation"] = {}
    raises("G16 fires when the P=0 saturation determination is missing",
           e17an.E17AnalysisError,
           lambda: e17an.verify_dual_margin_reporting(no_p0))

    undefined_without_g = dict(complete)
    undefined_without_g["verdict_strict_margin"] = "undefined"
    undefined_without_g["g_placebo"] = None
    raises("G16 fires when an undefined strict margin publishes no G_placebo",
           e17an.E17AnalysisError,
           lambda: e17an.verify_dual_margin_reporting(undefined_without_g))


def test_no_canonical_writes() -> None:
    print("\n[8] the test module writes no canonical E17 artifact")
    targets = list(e17train.record_paths().values())
    targets += [p for k, p in e17an.output_paths().items()]
    written_by_tests = [str(p) for p in targets
                        if p.exists() and p.stat().st_mtime > _START]
    check("no canonical E17 output was created by these tests",
          not written_by_tests, str(written_by_tests[:3]))


_START = 0.0


def main() -> int:
    global _START
    import time

    _START = time.time()
    print("E17 Random-Forest relational replication — pre-outcome tests")
    test_seed_axis_and_grid()
    test_rf_contract()
    test_pairing_gates()
    test_aggregation_and_closure()
    test_decision_rule()
    test_margin_and_saturation()
    test_g16_reporting_completeness()
    test_no_canonical_writes()
    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'ALL PASS'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
