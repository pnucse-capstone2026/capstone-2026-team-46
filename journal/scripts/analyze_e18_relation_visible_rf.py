#!/usr/bin/env python3
"""Registered analysis of the E18 relation-visible Random-Forest experiment.

Registered by ``journal/experiments/e18_relation_visible_rf/PREREG.md``
sections 8--13 and gates G11, G13, G16.

Everything inferential is inherited from E17 verbatim -- endpoint, aggregation
order, inferential unit, margin, decision order, bootstrap count and seed -- and
the corresponding functions are **imported** from
``journal/scripts/analyze_e17_rf_relational_response.py`` rather than
re-implemented, so the two experiments cannot drift apart.  The only thing E18
changes is the detector's feature map, which is upstream of this module.

Primary estimand
----------------
For generator realization ``g`` and pipeline seed ``p``, with the shared E18
real-only arm as reference::

    A^arm(P,S,D) = recall_arm(P,S,D) - recall_real(P,S,D)
    C_P{A}       = mean over (S,D) of [ A(1,S,D) - A(0,S,D) ]
    theta_g      = C_P{A^Rule}_g - C_P{A^Placebo}_g          (n = 20, df = 19)

Decision order (registered, not reorderable)
--------------------------------------------
1. heterogeneity screen (C1 directional, C2 aggregate/unit conflict) -> C
2. practical equivalence: 90% t interval inside [-delta, +delta]     -> A
3. material difference:   95% t interval entirely outside            -> B
4. otherwise                                                         -> D

A Student-t / bootstrap disagreement downgrades the verdict to D.  Branch E --
inconclusive-by-construction -- is decided upstream by gate G17 and never
reaches this module, which refuses to run unless G17 passed.

Dual margins (PREREG section 12.2, mandatory)
---------------------------------------------
The full decision rule is run twice: once against the inherited CNN margin
``delta = 0.133855625`` -- the primary -- and once against
``delta_strict = 0.15 * G_placebo``.  Disagreement is a reportable finding and
is never a reason to suppress either verdict.  Gate G16 refuses to publish
unless both verdicts, the disagreement flag, the ``P = 1`` absolute and ratio
scales, the ``P = 0`` saturation determination, and the mandatory Gear/RPM split
are all present.

This module computes no outcome on import and writes nothing until ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_e17_rf_relational_response as e17ana  # noqa: E402
import run_e18_relation_visible_rf_training as e18train  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _environment_record,
    _stage_bytes,
    _stage_json,
)

REPO = e18train.REPO
EXPERIMENT_DIR = e18train.EXPERIMENT_DIR
TABLE_DIR = e18train.TABLE_DIR
LOG_DIR = e18train.LOG_DIR

OUTPUT_VERSION = e18train.OUTPUT_VERSION
ANALYSIS_SCHEMA = "e18.registered_analysis.v1"
MANIFEST_SCHEMA = "e18.artifact_manifest.v1"

publish_and_cleanup = e18train.publish_and_cleanup
_sha256_file = e18train._sha256_file

# --- frozen decision constants, inherited from E17 verbatim -----------------

SESOI_MULTIPLIER = e17ana.SESOI_MULTIPLIER
SESOI_DENOMINATOR = e17ana.SESOI_DENOMINATOR
DELTA = e17ana.DELTA
EQUIVALENCE_CONFIDENCE = e17ana.EQUIVALENCE_CONFIDENCE
MATERIAL_CONFIDENCE = e17ana.MATERIAL_CONFIDENCE
HETEROGENEITY_MIN_OUTSIDE = e17ana.HETEROGENEITY_MIN_OUTSIDE
HETEROGENEITY_MIN_EACH_SIDE = e17ana.HETEROGENEITY_MIN_EACH_SIDE
BOOTSTRAP_RESAMPLES = e17ana.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = e17ana.BOOTSTRAP_SEED
STRICT_MARGIN_MIN_DENOMINATOR = e17ana.STRICT_MARGIN_MIN_DENOMINATOR

PRIMARY_ENDPOINT = e17ana.PRIMARY_ENDPOINT
PRIMARY_STRATUM = e17ana.PRIMARY_STRATUM
COMPANION_ENDPOINT = e17ana.COMPANION_ENDPOINT
REAL_ARM = e17ana.REAL_ARM
RULE_ARM = e17ana.RULE_ARM
PLACEBO_ARM = e17ana.PLACEBO_ARM

# --- inherited computation, imported and never re-implemented ---------------

_t_interval = e17ana._t_interval
primary_decision = e17ana.primary_decision
relational_share = e17ana.relational_share
build_augmentation_deltas = e17ana.build_augmentation_deltas
build_effects_by_realization = e17ana.build_effects_by_realization
build_decomposition = e17ana.build_decomposition
assert_decomposition_closure = e17ana.assert_decomposition_closure
build_cell_comparison = e17ana.build_cell_comparison
determine_p0_saturation = e17ana.determine_p0_saturation
build_secondary_family = e17ana.build_secondary_family
build_arm_summary = e17ana.build_arm_summary
build_attack_heterogeneity = e17ana.build_attack_heterogeneity
build_crossed_dispersion = e17ana.build_crossed_dispersion
verify_output_integrity = e17ana.verify_output_integrity

E16_DECOMPOSITION = TABLE_DIR / "e16_decomposition_by_construction_v2.csv"
E17_DECOMPOSITION = TABLE_DIR / "e17_decomposition_by_realization_v1.csv"


class E18AnalysisError(RuntimeError):
    """Raised when the registered analysis cannot be computed as specified."""


def _stop(message: str) -> E18AnalysisError:
    return E18AnalysisError(f"T-STOP-E18-ANALYSIS: {message}")


def utc_now() -> str:
    return e18train.utc_now()


def output_paths() -> dict[str, Path]:
    return {
        "augmentation_delta": TABLE_DIR
        / f"e18_augmentation_delta_{OUTPUT_VERSION}.csv",
        "decomposition": TABLE_DIR
        / f"e18_decomposition_by_realization_{OUTPUT_VERSION}.csv",
        "primary": TABLE_DIR / f"e18_primary_theta_summary_{OUTPUT_VERSION}.csv",
        "p1_cell": TABLE_DIR / f"e18_p1_cell_comparison_{OUTPUT_VERSION}.csv",
        "p0_cell": TABLE_DIR / f"e18_p0_cell_comparison_{OUTPUT_VERSION}.csv",
        "margin": TABLE_DIR / f"e18_margin_sensitivity_{OUTPUT_VERSION}.csv",
        "secondary": TABLE_DIR / f"e18_secondary_effects_{OUTPUT_VERSION}.csv",
        "family": TABLE_DIR / f"e18_family_comparison_{OUTPUT_VERSION}.csv",
        "dispersion": TABLE_DIR / f"e18_crossed_dispersion_{OUTPUT_VERSION}.csv",
        "bridge": TABLE_DIR / f"e18_bridge_sensitivity_{OUTPUT_VERSION}.csv",
        "verdict": LOG_DIR / f"e18_relation_visible_rf_{OUTPUT_VERSION}.log",
        "manifest": LOG_DIR
        / f"e18_relation_visible_rf_artifact_manifest_{OUTPUT_VERSION}.json",
    }


# ---------------------------------------------------------------------------
# Margin-scale sensitivity (PREREG section 12.2, mandatory)
# ---------------------------------------------------------------------------


def build_margin_sensitivity(theta: np.ndarray, p1_summary: Mapping[str, Any],
                             primary: Mapping[str, Any]
                             ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dual-margin reporting; disagreement is a finding, never a suppression."""
    g_placebo = float(p1_summary["equal_macro"]["placebo_gain_mean"])
    degenerate = g_placebo <= STRICT_MARGIN_MIN_DENOMINATOR
    if degenerate:
        strict = None
        strict_verdict = "undefined"
        strict_reason = (
            f"G_placebo = {g_placebo:.9f} <= {STRICT_MARGIN_MIN_DENOMINATOR}; "
            "PREREG section 12.2 registers delta_strict as undefined in this "
            "case, publishes the observed G_placebo, and invents no substitute "
            "denominator, floor, or alternative multiplier")
        delta_strict = None
        ratio = None
    else:
        delta_strict = SESOI_MULTIPLIER * g_placebo
        strict = primary_decision(theta, delta=delta_strict,
                                  margin_label="delta_strict")
        strict_verdict = strict["verdict"]
        strict_reason = strict["verdict_reason"]
        ratio = delta_strict / DELTA

    disagreement = bool(
        strict_verdict != "undefined"
        and strict_verdict != primary["verdict"])

    rows = [{
        "margin_label": "registered_inherited_delta",
        "delta": DELTA,
        "delta_provenance": (
            f"{SESOI_MULTIPLIER} x {SESOI_DENOMINATOR} (E15 v2 new-four "
            "Rule-minus-real P mean); inherited verbatim from E16 via E17, not "
            "re-derived from any E18 quantity"),
        "theta_mean": primary["theta_mean"],
        "theta_sd": primary["theta_sd"],
        "ci90_low": primary["ci90_low"],
        "ci90_high": primary["ci90_high"],
        "ci95_low": primary["ci95_low"],
        "ci95_high": primary["ci95_high"],
        "realizations_outside_band": primary["realizations_outside_band"],
        "heterogeneity_triggered": primary["heterogeneity_triggered"],
        "verdict": primary["verdict"],
        "verdict_reason": primary["verdict_reason"],
        "delta_strict_over_delta": "",
        "status": "primary",
    }, {
        "margin_label": "delta_strict",
        "delta": "" if delta_strict is None else delta_strict,
        "delta_provenance": (
            f"{SESOI_MULTIPLIER} x G_placebo, G_placebo = {g_placebo!r}, the "
            "placebo arm's own out-of-grammar P=1 exact-macro gain over the "
            "E18 real-only reference"),
        "theta_mean": primary["theta_mean"],
        "theta_sd": primary["theta_sd"],
        "ci90_low": primary["ci90_low"],
        "ci90_high": primary["ci90_high"],
        "ci95_low": primary["ci95_low"],
        "ci95_high": primary["ci95_high"],
        "realizations_outside_band": (
            "" if strict is None else strict["realizations_outside_band"]),
        "heterogeneity_triggered": (
            "" if strict is None else strict["heterogeneity_triggered"]),
        "verdict": strict_verdict,
        "verdict_reason": strict_reason,
        "delta_strict_over_delta": "" if ratio is None else ratio,
        "status": "secondary_sensitivity_never_overrides_the_primary",
    }]

    record = {
        "g_placebo": g_placebo,
        "g_placebo_degenerate": degenerate,
        "delta": DELTA,
        "delta_strict": delta_strict,
        "delta_strict_over_delta": ratio,
        "verdict": primary["verdict"],
        "verdict_strict_margin": strict_verdict,
        "margin_disagreement": disagreement,
        "strict_decision": strict,
        "registered_sentence": (
            f"Against the inherited CNN margin the joint-structure replication "
            f"returns Branch {primary['verdict']}; against a margin rescaled to "
            f"its own out-of-grammar placebo gain it returns Branch "
            f"{strict_verdict}. The verdict is therefore margin-scale "
            f"dependent, and the equivalence claim is reported as conditional "
            f"on the registered absolute margin."
        ) if disagreement else (
            f"Both the inherited CNN margin and a margin rescaled to the "
            f"detector's own out-of-grammar placebo gain return Branch "
            f"{primary['verdict']}."
        ) if strict_verdict != "undefined" else (
            f"The inherited CNN margin returns Branch {primary['verdict']}. "
            f"delta_strict is undefined because G_placebo = {g_placebo:.9f} "
            f"<= {STRICT_MARGIN_MIN_DENOMINATOR}; the primary verdict stands "
            f"unannotated by a sensitivity that could not be computed."),
    }
    return pd.DataFrame(rows), record


# ---------------------------------------------------------------------------
# Mandatory Gear/RPM split (PREREG section 12.8)
# ---------------------------------------------------------------------------


def build_gear_rpm_split(heterogeneity: pd.DataFrame,
                         p1_summary: Mapping[str, Any]) -> dict[str, Any]:
    """The registered per-attack split, mandatory under PREREG section 12.8."""
    split: dict[str, Any] = {
        "status": "mandatory (PREREG section 12.8)",
        "reason": (
            "the Gear inverse relation |255 - data0 - data1| <= 8 is the "
            "cleanest and most completely destroyed manipulation in the "
            "design, so if the relational increment is real and representable "
            "Gear is where it should be largest; under the E17 marginal map "
            "the Gear P=1 increment was +0.000087, the smallest number in that "
            "experiment"),
    }
    for attack, effect in (("Gear", "Gear_canonical"), ("RPM", "RPM_canonical")):
        row = heterogeneity[heterogeneity["effect"] == effect]
        if len(row) != 1:
            raise _stop(f"missing per-attack theta for {attack}")
        record = row.iloc[0]
        cell = p1_summary[attack]
        split[attack] = {
            "theta_mean": float(record["mean"]),
            "theta_sd": float(record["sd"]),
            "theta_ci95_low": float(record["ci95_low"]),
            "theta_ci95_high": float(record["ci95_high"]),
            "theta_strict_positive": int(record["strict_positive"]),
            "theta_strict_negative": int(record["strict_negative"]),
            "p1_real_recall": float(cell["real_recall"]),
            "p1_rule_recall_mean": float(cell["rule_recall_mean"]),
            "p1_placebo_recall_mean": float(cell["placebo_recall_mean"]),
            "p1_absolute_difference_mean": float(
                cell["absolute_difference_mean"]),
            "p1_absolute_difference_sd": float(cell["absolute_difference_sd"]),
            "p1_absolute_difference_ci95_low": float(
                cell["absolute_difference_ci95_low"]),
            "p1_absolute_difference_ci95_high": float(
                cell["absolute_difference_ci95_high"]),
            "n": 20,
            "degrees_of_freedom": 19,
        }
    return split


# ---------------------------------------------------------------------------
# Three-way descriptive comparison (PREREG section 12.7, mandatory)
# ---------------------------------------------------------------------------


def build_family_comparison(decomposition: pd.DataFrame
                            ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Paired CNN / marginal-map RF / joint-structure RF theta, descriptive only."""
    for path in (E16_DECOMPOSITION, E17_DECOMPOSITION):
        if not path.exists():
            raise _stop(f"missing frozen decomposition table: {path}")
    cnn = pd.read_csv(E16_DECOMPOSITION)
    cnn = cnn[(cnn["role"] == "primary") & (cnn["effect"] == "P")]
    rf55 = pd.read_csv(E17_DECOMPOSITION)
    rf55 = rf55[(rf55["role"] == "primary") & (rf55["effect"] == "P")]
    rf83 = decomposition[decomposition["role"] == "primary"]

    columns = ["construction_seed", "theta_rule_minus_placebo",
               "delta_rule_minus_real", "delta_placebo_minus_real"]
    merged = cnn[columns].merge(
        rf55[columns], on="construction_seed", suffixes=("_cnn", "_rf55"),
        validate="1:1")
    merged = merged.merge(
        rf83[columns].rename(columns={
            c: f"{c}_rf83" for c in columns if c != "construction_seed"}),
        on="construction_seed", validate="1:1")
    if len(merged) != len(rf83):
        raise _stop(
            f"family comparison pairs {len(merged)} of {len(rf83)} "
            "realizations; the three configurations must share the same twenty "
            "units")
    merged = merged.sort_values("construction_seed").reset_index(drop=True)

    cnn_theta = merged["theta_rule_minus_placebo_cnn"].to_numpy(dtype=float)
    rf55_theta = merged["theta_rule_minus_placebo_rf55"].to_numpy(dtype=float)
    rf83_theta = merged["theta_rule_minus_placebo_rf83"].to_numpy(dtype=float)
    merged["theta_rf83_minus_cnn"] = rf83_theta - cnn_theta
    merged["theta_rf83_minus_rf55"] = rf83_theta - rf55_theta

    summary: dict[str, Any] = {
        "n": int(len(merged)),
        "cnn_theta_mean": float(cnn_theta.mean()),
        "cnn_theta_sd": float(cnn_theta.std(ddof=1)),
        "rf55_theta_mean": float(rf55_theta.mean()),
        "rf55_theta_sd": float(rf55_theta.std(ddof=1)),
        "rf83_theta_mean": float(rf83_theta.mean()),
        "rf83_theta_sd": float(rf83_theta.std(ddof=1)),
        "status": "descriptive_only",
        "prohibition": (
            "three detector configurations are three points, not a sample; no "
            "p-value, confidence interval, or equivalence test on the "
            "between-configuration difference is reported, and no "
            "invariance-across-representations claim is permitted"),
    }
    for label, reference in (("cnn", cnn_theta), ("rf55", rf55_theta)):
        difference = rf83_theta - reference
        summary[f"paired_difference_vs_{label}_mean"] = float(difference.mean())
        summary[f"paired_difference_vs_{label}_sd"] = float(
            difference.std(ddof=1))
        summary[f"pearson_r_vs_{label}"] = float(
            stats.pearsonr(reference, rf83_theta).statistic)
        summary[f"spearman_rho_vs_{label}"] = float(
            stats.spearmanr(reference, rf83_theta).statistic)
    return merged, summary


# ---------------------------------------------------------------------------
# Gate G16 — dual-margin and mandatory-secondary reporting completeness
# ---------------------------------------------------------------------------


def verify_reporting_completeness(record: Mapping[str, Any]) -> dict[str, Any]:
    """Gate G16 — E17's anti-suppression rule plus the E18 Gear/RPM split."""
    inherited = e17ana.verify_dual_margin_reporting(record)
    missing: list[str] = []
    split = record.get("gear_rpm_split") or {}
    for attack in ("Gear", "RPM"):
        entry = split.get(attack) or {}
        for field in ("theta_mean", "theta_sd", "p1_absolute_difference_mean",
                      "p1_rule_recall_mean", "p1_placebo_recall_mean"):
            if entry.get(field) is None:
                missing.append(f"gear_rpm_split.{attack}.{field}")
    visibility = record.get("manipulation_visibility") or {}
    if not visibility.get("passed"):
        missing.append("manipulation_visibility_gate")
    if visibility.get("pooled_visible_fraction") is None:
        missing.append("manipulation_visibility.pooled_visible_fraction")
    if missing:
        raise _stop(
            f"G16 reporting completeness FAILED; missing {missing}")
    return {
        **inherited,
        "gate": "G16_reporting_completeness",
        "gear_rpm_split_reported": True,
        "manipulation_visibility_reported": True,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze(scenario: pd.DataFrame) -> dict[str, Any]:
    """Full registered analysis over an already-scored scenario table."""
    deltas = build_augmentation_deltas(scenario)
    per_pipeline, per_realization = build_effects_by_realization(deltas)
    decomposition = build_decomposition(per_realization, effect="P")
    closure = assert_decomposition_closure(decomposition)
    integrity = verify_output_integrity(scenario, decomposition)

    primary_rows = decomposition[decomposition["role"] == "primary"]
    theta = primary_rows["theta_rule_minus_placebo"].to_numpy(dtype=float)
    if len(theta) != len(e18train.CONSTRUCTION_SEEDS):
        raise _stop(
            f"primary has {len(theta)} realizations, registered "
            f"{len(e18train.CONSTRUCTION_SEEDS)}")
    decision = primary_decision(theta, delta=DELTA, margin_label="registered")
    share = relational_share(primary_rows)

    p1_table, p1_summary = build_cell_comparison(scenario, p_level=1)
    p0_table, p0_summary = build_cell_comparison(scenario, p_level=0)
    saturation = determine_p0_saturation(p0_summary, decomposition)
    margin_table, margin_record = build_margin_sensitivity(
        theta, p1_summary, decision)

    exact_family = build_secondary_family(
        per_realization, effects=e17ana.SECONDARY_EXACT_FAMILY,
        label="exact_rule_minus_placebo_secondary", endpoint=PRIMARY_ENDPOINT)
    binary_deltas = build_augmentation_deltas(
        scenario, endpoint=COMPANION_ENDPOINT)
    _, binary_per_realization = build_effects_by_realization(binary_deltas)
    binary_family = build_secondary_family(
        binary_per_realization, effects=e17ana.EFFECTS,
        label="binary_rule_minus_placebo_companion",
        endpoint=COMPANION_ENDPOINT)
    heterogeneity = build_attack_heterogeneity(scenario)
    gear_rpm_split = build_gear_rpm_split(heterogeneity, p1_summary)

    dispersion = build_crossed_dispersion(per_pipeline)
    family_table, family_summary = build_family_comparison(decomposition)

    bridge = decomposition[decomposition["role"] == "bridge_sensitivity"].copy()
    bridge["status"] = ("bridge sensitivity attached to the E15 lineage; never "
                        "pooled with the primary twenty, never changes n or df")

    return {
        "augmentation_delta": deltas,
        "per_pipeline": per_pipeline,
        "per_realization": per_realization,
        "decomposition": decomposition,
        "closure_gate": closure,
        "integrity_gate": integrity,
        "decision": decision,
        "relational_share": share,
        "p1_table": p1_table,
        "p1_summary": p1_summary,
        "p0_table": p0_table,
        "p0_summary": p0_summary,
        "p0_saturation": saturation,
        "margin_table": margin_table,
        "margin_record": margin_record,
        "secondary_exact": exact_family,
        "secondary_binary": binary_family,
        "heterogeneity": heterogeneity,
        "gear_rpm_split": gear_rpm_split,
        "dispersion": dispersion,
        "family_table": family_table,
        "family_summary": family_summary,
        "bridge": bridge,
        "marginal_component": build_arm_summary(
            per_realization, arm=PLACEBO_ARM),
        "total_response": build_arm_summary(per_realization, arm=RULE_ARM),
    }


def _load_scenario() -> pd.DataFrame:
    path = TABLE_DIR / f"e18_by_scenario_{OUTPUT_VERSION}.csv"
    if not path.exists():
        raise _stop(f"missing scored table: {path}")
    return pd.read_csv(path)


def _load_visibility_gate() -> dict[str, Any]:
    """Both published G17 measurements must exist and must both have passed."""
    records = e18train.record_paths()
    gates: dict[str, Any] = {}
    for key, label in (("visibility_record", "v1"),
                       ("visibility_record_v2", "v2")):
        path = records[key]
        if not path.exists():
            raise _stop(f"missing gate G17 {label} record: {path}")
        gate = json.loads(path.read_text())["visibility_gate"]
        if not gate["passed"]:
            raise _stop(
                f"gate G17 {label} did not pass; E18 is "
                "inconclusive-by-construction (Branch E) and no inferential "
                "analysis may be run")
        gates[label] = gate
    return {
        "passed": True,
        "pooled_visible_fraction": gates["v1"]["pooled_visible_fraction"],
        "pooled_float64_visible_fraction": gates["v2"]["metrics"]["float64"][
            "pooled_visible_fraction"],
        "per_attack_visible_fraction": gates["v2"][
            "per_attack_visible_fraction"],
        "per_attack_differential": gates["v2"]["per_attack_differential"],
        "v1": gates["v1"],
        "v2": gates["v2"],
    }


def execute() -> dict[str, Any]:
    started = time.monotonic()
    visibility = _load_visibility_gate()
    scenario = _load_scenario()
    result = analyze(scenario)
    paths = output_paths()
    present = [str(p.relative_to(REPO)) for p in paths.values() if p.exists()]
    if present:
        raise _stop(f"refusing to overwrite existing outputs: {present}")

    decision = result["decision"]
    margin = result["margin_record"]
    verdict_record: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "stage": "e18_registered_analysis",
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "registration": e18train.assert_registration_committed(),
        "environment": _environment_record(REPO),
        "feature_map": {
            "dimension": e18train.FEATURE_DIMENSION,
            "marginal_block": e18train.MARGINAL_DIMENSION,
            "correlation_block": e18train.CORRELATION_DIMENSION,
            "description": (
                "rf_features (55) plus the 28 within-window Pearson "
                "correlations over unordered pairs of data0..data7; constant "
                "channels give 0.0; generic cross-byte joint structure, not a "
                "Gear- or RPM-specific indicator"),
        },
        "manipulation_visibility": visibility,
        "inferential_unit": {
            "unit": "generator realization", "n": decision["n"],
            "degrees_of_freedom": decision["degrees_of_freedom"],
            "never_units": ["block", "cell", "attack", "pipeline seed",
                            "window", "fit"],
        },
        "matching_contract": (
            "identical training rows inherited from the frozen E16 draws; the "
            "arms carry no optimizer-update budget, early-stopping trajectory, "
            "or minibatch order, so the only difference between the two arms of "
            "a realization is the data1 permutation itself (PREREG 5.3)"),
        "closure_gate": result["closure_gate"],
        "output_integrity_gate": result["integrity_gate"],
        "primary": decision,
        "relational_share": result["relational_share"],
        "p1_cell_summary": result["p1_summary"],
        "p0_cell_summary": result["p0_summary"],
        "p0_saturation": result["p0_saturation"],
        "gear_rpm_split": result["gear_rpm_split"],
        "marginal_exposure_component": result["marginal_component"],
        "total_augmentation_response": result["total_response"],
        "family_comparison": result["family_summary"],
        "verdict": decision["verdict"],
        "verdict_strict_margin": margin["verdict_strict_margin"],
        "margin_disagreement": margin["margin_disagreement"],
        "g_placebo": margin["g_placebo"],
        "delta": margin["delta"],
        "delta_strict": margin["delta_strict"],
        "delta_strict_over_delta": margin["delta_strict_over_delta"],
        "strict_margin_decision": margin["strict_decision"],
        "registered_margin_sentence": margin["registered_sentence"],
        "scientific_verdict": decision["verdict"],
        "status": "analysis_complete",
    }
    verdict_record["reporting_completeness_gate"] = (
        verify_reporting_completeness(verdict_record))

    frames = {
        "augmentation_delta": result["augmentation_delta"],
        "decomposition": result["decomposition"],
        "p1_cell": result["p1_table"],
        "p0_cell": result["p0_table"],
        "margin": result["margin_table"],
        "family": result["family_table"],
        "dispersion": result["dispersion"],
        "bridge": result["bridge"],
        "secondary": pd.concat(
            [result["secondary_exact"], result["secondary_binary"],
             result["heterogeneity"]], ignore_index=True),
        "primary": pd.DataFrame([decision]),
    }
    staged = [
        (_stage_bytes(paths[key], frame.to_csv(index=False).encode()),
         paths[key])
        for key, frame in frames.items()
    ]
    staged.append((_stage_json(paths["verdict"], verdict_record),
                   paths["verdict"]))
    publish_and_cleanup(staged)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "e18_relation_visible_rf",
        "created_utc": utc_now(),
        "registration": verdict_record["registration"],
        "manipulation_visibility": visibility,
        "verdict": verdict_record["verdict"],
        "verdict_strict_margin": verdict_record["verdict_strict_margin"],
        "margin_disagreement": verdict_record["margin_disagreement"],
        "artifacts": {},
    }
    for path in _canonical_artifacts():
        if not path.exists():
            raise _stop(f"registered artifact missing at manifest time: {path}")
        manifest["artifacts"][str(path.relative_to(REPO))] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    publish_and_cleanup([(_stage_json(paths["manifest"], manifest),
                          paths["manifest"])])
    verdict_record["artifact_manifest"] = str(
        paths["manifest"].relative_to(REPO))
    return verdict_record


def _canonical_artifacts() -> list[Path]:
    """Every registered E18 output (PREREG section 14.2), for the manifest."""
    from evaluate_e18_relation_visible_rf import output_paths as eval_paths

    paths = list(e18train.record_paths().values())
    paths += list(eval_paths().values())
    paths += [p for key, p in output_paths().items() if key != "manifest"]
    return sorted(set(paths))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="run the registered analysis and publish outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({
            "stage": "e18_registered_analysis",
            "mode": "read-only",
            "feature_dimension": e18train.FEATURE_DIMENSION,
            "delta": DELTA,
            "sesoi_multiplier": SESOI_MULTIPLIER,
            "sesoi_denominator": SESOI_DENOMINATOR,
            "strict_margin_multiplier": SESOI_MULTIPLIER,
            "equivalence_confidence": EQUIVALENCE_CONFIDENCE,
            "heterogeneity_min_outside": HETEROGENEITY_MIN_OUTSIDE,
            "heterogeneity_min_each_side": HETEROGENEITY_MIN_EACH_SIDE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inherited_from": "journal/scripts/analyze_e17_rf_relational_response.py",
            "outputs": {k: str(v.relative_to(REPO))
                        for k, v in output_paths().items()},
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
