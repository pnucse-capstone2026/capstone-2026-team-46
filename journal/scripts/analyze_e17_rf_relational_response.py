#!/usr/bin/env python3
"""Registered analysis of the E17 Random-Forest relational replication.

Registered by ``journal/experiments/e17_rf_relational_replication/PREREG.md``
sections 8--13 and gates G11, G13, G16.

Primary estimand
----------------
For generator realization ``g`` and pipeline seed ``p``, with the shared E17
Random-Forest real-only arm as reference::

    A^arm(P,S,D) = recall_arm(P,S,D) - recall_real(P,S,D)
    C_P{A}       = mean over (S,D) of [ A(1,S,D) - A(0,S,D) ]

The registered aggregation order is attack -> block -> cell -> pipeline, after
which the generator realizations are the inferential unit::

    theta_g = C_P{A^Rule}_g - C_P{A^Placebo}_g          (n = 20, df = 19)

and the decomposition identity ``C_P(Rule-Real) = C_P(Placebo-Real) + theta``
is verified to floating-point closure as gate G11, not assumed.

Decision order (registered, not reorderable)
--------------------------------------------
1. heterogeneity screen (C1 directional, C2 aggregate/unit conflict) -> C
2. practical equivalence: 90% t interval inside [-delta, +delta]     -> A
3. material difference:   95% t interval entirely outside            -> B
4. otherwise                                                         -> D

A Student-t / bootstrap disagreement downgrades the verdict to D.

Dual margins (PREREG section 12.2, mandatory)
---------------------------------------------
The full decision rule is run twice: once against the inherited CNN margin
``delta = 0.133855625`` -- the primary -- and once against
``delta_strict = 0.15 * G_placebo``, rescaled to the Random Forest's own
out-of-grammar placebo gain.  Disagreement between the two is a reportable
finding and is never a reason to suppress either.  Gate G16 refuses to publish
unless both verdicts, the disagreement flag, the ``P = 1`` absolute and ratio
scales, and the ``P = 0`` saturation determination are all present.

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

import run_e17_rf_paired_training as e17train  # noqa: E402
from analyze_e16_relational_response import (  # noqa: E402
    EFFECTS,
    SECONDARY_EXACT_FAMILY,
    factor_contrast,
    holm_adjust,
)
from generate_rule_construction_sensitivity import (  # noqa: E402
    _environment_record,
    _stage_bytes,
    _stage_json,
)

REPO = e17train.REPO
EXPERIMENT_DIR = e17train.EXPERIMENT_DIR
TABLE_DIR = e17train.TABLE_DIR
LOG_DIR = e17train.LOG_DIR

OUTPUT_VERSION = e17train.OUTPUT_VERSION
ANALYSIS_SCHEMA = "e17.registered_analysis.v1"
MANIFEST_SCHEMA = "e17.artifact_manifest.v1"

publish_and_cleanup = e17train.publish_and_cleanup
_sha256_file = e17train._sha256_file

# --- frozen decision constants (PREREG sections 9.1, 10, 12.2) --------------

SESOI_MULTIPLIER = 0.15
SESOI_DENOMINATOR = 0.8923708333333333   # E15 v2 new4 Rule-minus-real P mean
DELTA = 0.133855625                      # inherited verbatim from E16
EQUIVALENCE_CONFIDENCE = 0.90            # TOST: two one-sided 5% tests
MATERIAL_CONFIDENCE = 0.95
HETEROGENEITY_MIN_OUTSIDE = 5            # of 20, conflict criterion C2
HETEROGENEITY_MIN_EACH_SIDE = 3          # of 20, directional criterion C1
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260807
RELATIONAL_SHARE_MIN_DENOMINATOR = 0.10
RATIO_MIN_DENOMINATOR = 0.01
#: PREREG section 12.2 degenerate-denominator handling.
STRICT_MARGIN_MIN_DENOMINATOR = 0.01
#: PREREG section 12.3 saturation criterion.
P0_SATURATION_RECALL = 0.99
P0_SATURATION_DIFFERENCE = 0.01

PRIMARY_ENDPOINT = "exact_recall"
PRIMARY_STRATUM = "canonical"
COMPANION_ENDPOINT = "binary_recall"

REAL_ARM = "real_rf"
RULE_ARM = "rule"
PLACEBO_ARM = "placebo"

E16_DECOMPOSITION = TABLE_DIR / "e16_decomposition_by_construction_v2.csv"


class E17AnalysisError(RuntimeError):
    """Raised when the registered analysis cannot be computed as specified."""


def _stop(message: str) -> E17AnalysisError:
    return E17AnalysisError(f"T-STOP-E17-ANALYSIS: {message}")


def utc_now() -> str:
    return e17train.utc_now()


def output_paths() -> dict[str, Path]:
    return {
        "augmentation_delta": TABLE_DIR
        / f"e17_augmentation_delta_{OUTPUT_VERSION}.csv",
        "decomposition": TABLE_DIR
        / f"e17_decomposition_by_realization_{OUTPUT_VERSION}.csv",
        "primary": TABLE_DIR / f"e17_primary_theta_summary_{OUTPUT_VERSION}.csv",
        "p1_cell": TABLE_DIR / f"e17_p1_cell_comparison_{OUTPUT_VERSION}.csv",
        "p0_cell": TABLE_DIR / f"e17_p0_cell_comparison_{OUTPUT_VERSION}.csv",
        "margin": TABLE_DIR / f"e17_margin_sensitivity_{OUTPUT_VERSION}.csv",
        "secondary": TABLE_DIR / f"e17_secondary_effects_{OUTPUT_VERSION}.csv",
        "family": TABLE_DIR
        / f"e17_cnn_rf_family_comparison_{OUTPUT_VERSION}.csv",
        "dispersion": TABLE_DIR / f"e17_crossed_dispersion_{OUTPUT_VERSION}.csv",
        "bridge": TABLE_DIR / f"e17_bridge_sensitivity_{OUTPUT_VERSION}.csv",
        "verdict": LOG_DIR
        / f"e17_rf_relational_replication_{OUTPUT_VERSION}.log",
        "manifest": LOG_DIR
        / f"e17_rf_relational_replication_artifact_manifest_"
          f"{OUTPUT_VERSION}.json",
    }


# ---------------------------------------------------------------------------
# Factorial contrasts and the registered aggregation order
# ---------------------------------------------------------------------------


def _cell_map(frame: pd.DataFrame, column: str
              ) -> dict[tuple[int, int, int], float]:
    values = {
        (int(row["P"]), int(row["S"]), int(row["D"])): float(row[column])
        for row in frame.to_dict(orient="records")
    }
    if len(values) != 8:
        raise _stop(f"expected 8 factorial cells, got {len(values)}")
    return values


def build_augmentation_deltas(
        scenario: pd.DataFrame,
        *,
        endpoint: str = PRIMARY_ENDPOINT,
        id_stratum: str = PRIMARY_STRATUM,
        attack_scope: str = "equal_macro",
) -> pd.DataFrame:
    """Registered aggregation steps 1--2: attack -> block -> per-cell delta."""
    frame = scenario[scenario["id_stratum"].astype(str) == id_stratum].copy()
    if attack_scope != "equal_macro":
        frame = frame[frame["attack"].astype(str) == attack_scope]
    if frame.empty:
        raise _stop(f"no rows for stratum={id_stratum} scope={attack_scope}")

    keys = ["arm", "construction_seed", "pipeline_seed", "role",
            "block_id", "P", "S", "D"]
    by_block = frame.groupby(keys, as_index=False)[endpoint].mean()
    cell_keys = ["arm", "construction_seed", "pipeline_seed", "role",
                 "P", "S", "D"]
    by_cell = by_block.groupby(cell_keys, as_index=False)[endpoint].mean()

    real = by_cell[by_cell["arm"].astype(str) == REAL_ARM]
    if real.empty:
        raise _stop("shared real-only arm missing from the scored table")
    real = real[["pipeline_seed", "P", "S", "D", endpoint]].rename(
        columns={endpoint: "real_value"})

    arms = by_cell[by_cell["arm"].astype(str) != REAL_ARM].merge(
        real, on=["pipeline_seed", "P", "S", "D"], how="left", validate="m:1")
    if arms["real_value"].isna().any():
        raise _stop("an arm cell has no matching shared real-only reference")
    arms["augmentation_delta"] = arms[endpoint] - arms["real_value"]
    arms["endpoint"] = endpoint
    arms["id_stratum"] = id_stratum
    arms["attack_scope"] = attack_scope
    return arms


def build_effects_by_realization(
        deltas: pd.DataFrame, *, effects: Sequence[str] = EFFECTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Registered steps 3--4: cell contrast, then pipeline equal weighting."""
    rows: list[dict[str, Any]] = []
    group_keys = ["arm", "construction_seed", "pipeline_seed", "role"]
    for key, group in deltas.groupby(group_keys):
        arm, construction_seed, pipeline_seed, role = key
        cells = _cell_map(group, "augmentation_delta")
        # The P = 1 and P = 0 halves of C_P, kept separately so the section 12.3
        # saturation determination can say whether the in-grammar cell cancels.
        p1_component = float(np.mean(
            [v for (p, _s, _d), v in cells.items() if p == 1]))
        p0_component = float(np.mean(
            [v for (p, _s, _d), v in cells.items() if p == 0]))
        for effect in effects:
            rows.append({
                "arm": arm,
                "construction_seed": int(construction_seed),
                "pipeline_seed": int(pipeline_seed),
                "role": role,
                "effect": effect,
                "value": factor_contrast(cells, effect),
                "p1_component": p1_component if effect == "P" else np.nan,
                "p0_component": p0_component if effect == "P" else np.nan,
            })
    per_pipeline = pd.DataFrame(rows)
    per_realization = per_pipeline.groupby(
        ["arm", "construction_seed", "role", "effect"], as_index=False).agg(
            value=("value", "mean"),
            p1_component=("p1_component", "mean"),
            p0_component=("p0_component", "mean"))
    return per_pipeline, per_realization


def build_decomposition(per_realization: pd.DataFrame, *, effect: str = "P"
                        ) -> pd.DataFrame:
    """The registered three-term decomposition, one row per realization."""
    frame = per_realization[per_realization["effect"] == effect]
    rule = frame[frame["arm"] == RULE_ARM].set_index("construction_seed")
    placebo = frame[frame["arm"] == PLACEBO_ARM].set_index("construction_seed")
    shared = sorted(set(rule.index) & set(placebo.index))
    rows: list[dict[str, Any]] = []
    for construction_seed in shared:
        delta_rule = float(rule.loc[construction_seed, "value"])
        delta_placebo = float(placebo.loc[construction_seed, "value"])
        theta = delta_rule - delta_placebo
        rows.append({
            "construction_seed": construction_seed,
            "role": str(rule.loc[construction_seed, "role"]),
            "effect": effect,
            "delta_rule_minus_real": delta_rule,
            "delta_placebo_minus_real": delta_placebo,
            "theta_rule_minus_placebo": theta,
            "closure_residual": delta_rule - (delta_placebo + theta),
            "theta_p1_component": float(
                rule.loc[construction_seed, "p1_component"]
                - placebo.loc[construction_seed, "p1_component"]),
            "theta_p0_component": float(
                rule.loc[construction_seed, "p0_component"]
                - placebo.loc[construction_seed, "p0_component"]),
        })
    return pd.DataFrame(rows)


def assert_decomposition_closure(decomposition: pd.DataFrame,
                                 tolerance: float = 1e-12) -> dict[str, Any]:
    """Gate G11 — the registered identity holds to floating-point closure."""
    residual = decomposition["closure_residual"].abs().max()
    if not np.isfinite(residual) or residual > tolerance:
        raise _stop(
            f"decomposition closure violated: max |residual| = {residual}")
    # theta = (P=1 half) - (P=0 half) must reproduce theta itself.
    halves = (decomposition["theta_p1_component"]
              - decomposition["theta_p0_component"])
    half_residual = float(
        (halves - decomposition["theta_rule_minus_placebo"]).abs().max())
    if half_residual > tolerance:
        raise _stop(
            f"P=1/P=0 decomposition of theta does not close: {half_residual}")
    return {"gate": "G11_decomposition_closure", "passed": True,
            "max_abs_residual": float(residual),
            "max_abs_cell_half_residual": half_residual,
            "tolerance": tolerance,
            "realizations": int(len(decomposition))}


# ---------------------------------------------------------------------------
# Primary decision (PREREG section 10), run once per margin
# ---------------------------------------------------------------------------


def _t_interval(values: np.ndarray, confidence: float
                ) -> tuple[float, float, float, float]:
    n = len(values)
    if n < 2:
        raise _stop("cannot form an interval from fewer than two units")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * sd / np.sqrt(n))
    return mean, sd, mean - half, mean + half


def primary_decision(theta: np.ndarray, *, delta: float,
                     margin_label: str = "registered") -> dict[str, Any]:
    """Apply the registered decision order and return the machine verdict."""
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    mean, sd, lo90, hi90 = _t_interval(theta, EQUIVALENCE_CONFIDENCE)
    _, _, lo95, hi95 = _t_interval(theta, MATERIAL_CONFIDENCE)

    above = int(np.count_nonzero(theta > delta))
    below = int(np.count_nonzero(theta < -delta))
    outside = above + below

    equivalent = bool(lo90 >= -delta and hi90 <= delta)
    material = bool(lo95 > delta or hi95 < -delta)

    c1 = bool(above >= HETEROGENEITY_MIN_EACH_SIDE
              and below >= HETEROGENEITY_MIN_EACH_SIDE)
    c2 = bool(outside >= HETEROGENEITY_MIN_OUTSIDE and equivalent)
    heterogeneous = c1 or c2

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    boot_means = theta[draws].mean(axis=1)
    boot_lo90, boot_hi90 = np.percentile(boot_means, [5.0, 95.0])
    boot_lo95, boot_hi95 = np.percentile(boot_means, [2.5, 97.5])
    boot_equivalent = bool(boot_lo90 >= -delta and boot_hi90 <= delta)
    boot_material = bool(boot_lo95 > delta or boot_hi95 < -delta)

    positive = int(np.count_nonzero(theta > 0))
    negative = int(np.count_nonzero(theta < 0))
    sign_p = float(stats.binomtest(
        max(positive, negative), n, 0.5, alternative="two-sided").pvalue)

    disagreement = (equivalent != boot_equivalent) or (material != boot_material)

    if heterogeneous:
        verdict = "C"
        reason = (
            f"directional disagreement: {above} realizations above +{delta} "
            f"and {below} below -{delta} (threshold "
            f"{HETEROGENEITY_MIN_EACH_SIDE} each side)"
        ) if c1 else (
            f"aggregate/unit conflict: the aggregate interval is inside "
            f"[-{delta}, +{delta}] while {outside} of {n} realizations are "
            f"outside it (threshold {HETEROGENEITY_MIN_OUTSIDE})")
    elif disagreement:
        verdict, reason = "D", (
            "Student-t and bootstrap verdicts disagree; the registered rule "
            "downgrades to inconclusive")
    elif equivalent:
        verdict, reason = "A", (
            f"the {int(EQUIVALENCE_CONFIDENCE * 100)}% interval "
            f"[{lo90:.6f}, {hi90:.6f}] lies inside [-{delta}, +{delta}]")
    elif material:
        verdict, reason = "B", (
            f"the {int(MATERIAL_CONFIDENCE * 100)}% interval "
            f"[{lo95:.6f}, {hi95:.6f}] lies entirely outside "
            f"[-{delta}, +{delta}]")
    else:
        verdict, reason = "D", (
            "neither practical equivalence nor a material difference was "
            "established")

    return {
        "margin_label": margin_label,
        "verdict": verdict,
        "verdict_reason": reason,
        "n": n,
        "degrees_of_freedom": n - 1,
        "theta_mean": mean,
        "theta_sd": sd,
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "delta": delta,
        "ci90_low": lo90, "ci90_high": hi90,
        "ci95_low": lo95, "ci95_high": hi95,
        "practical_equivalence": equivalent,
        "material_difference": material,
        "realizations_above_band": above,
        "realizations_below_band": below,
        "realizations_outside_band": outside,
        "heterogeneity_c1_directional": c1,
        "heterogeneity_c2_conflict": c2,
        "heterogeneity_triggered": heterogeneous,
        "bootstrap_ci90_low": float(boot_lo90),
        "bootstrap_ci90_high": float(boot_hi90),
        "bootstrap_ci95_low": float(boot_lo95),
        "bootstrap_ci95_high": float(boot_hi95),
        "bootstrap_equivalence": boot_equivalent,
        "bootstrap_material": boot_material,
        "bootstrap_disagreement": disagreement,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sign_positive": positive,
        "sign_negative": negative,
        "sign_test_p": sign_p,
    }


def relational_share(decomposition: pd.DataFrame) -> dict[str, Any]:
    """PREREG section 11 — descriptive share with the registered exclusion."""
    frame = decomposition.copy()
    frame["denominator"] = frame["delta_rule_minus_real"].abs()
    usable = frame[frame["denominator"] >= RELATIONAL_SHARE_MIN_DENOMINATOR]
    excluded = sorted(
        int(s) for s in
        frame.loc[frame["denominator"] < RELATIONAL_SHARE_MIN_DENOMINATOR,
                  "construction_seed"])
    if usable.empty:
        return {"median": None, "iqr_low": None, "iqr_high": None,
                "excluded_realizations": excluded, "n_used": 0,
                "denominator_median": None}
    share = usable["theta_rule_minus_placebo"] / usable["denominator"]
    return {
        "median": float(share.median()),
        "iqr_low": float(share.quantile(0.25)),
        "iqr_high": float(share.quantile(0.75)),
        "minimum": float(share.min()),
        "maximum": float(share.max()),
        "excluded_realizations": excluded,
        "n_used": int(len(share)),
        "denominator_median": float(usable["denominator"].median()),
        "denominator_min": float(usable["denominator"].min()),
        "denominator_max": float(usable["denominator"].max()),
    }


# ---------------------------------------------------------------------------
# Mandatory P = 1 and P = 0 cell secondaries (PREREG sections 12.1, 12.3)
# ---------------------------------------------------------------------------


def build_cell_comparison(
        scenario: pd.DataFrame,
        *,
        p_level: int,
        endpoint: str = PRIMARY_ENDPOINT,
        id_stratum: str = PRIMARY_STRATUM,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Absolute arm recall in one ``P`` cell, plus the registered ratio form."""
    frame = scenario[
        (scenario["id_stratum"].astype(str) == id_stratum)
        & (scenario["P"].astype(int) == p_level)].copy()
    if frame.empty:
        raise _stop(f"no P={p_level} rows available")

    rows: list[dict[str, Any]] = []
    for attack_scope in ("equal_macro", "Gear", "RPM"):
        scoped = frame if attack_scope == "equal_macro" else frame[
            frame["attack"].astype(str) == attack_scope]
        by_attack = scoped.groupby(
            ["arm", "construction_seed", "pipeline_seed", "role", "attack"],
            as_index=False)[endpoint].mean()
        by_pipeline = by_attack.groupby(
            ["arm", "construction_seed", "pipeline_seed", "role"],
            as_index=False)[endpoint].mean()
        by_realization = by_pipeline.groupby(
            ["arm", "construction_seed", "role"],
            as_index=False)[endpoint].mean()

        real = by_pipeline[by_pipeline["arm"] == REAL_ARM]
        real_mean = float(real[endpoint].mean()) if not real.empty else np.nan
        rule = by_realization[
            (by_realization["arm"] == RULE_ARM)
            & (by_realization["role"] == "primary")]
        placebo = by_realization[
            (by_realization["arm"] == PLACEBO_ARM)
            & (by_realization["role"] == "primary")]
        merged = rule.merge(
            placebo, on="construction_seed", suffixes=("_rule", "_placebo"))
        for record in merged.to_dict(orient="records"):
            rule_recall = float(record[f"{endpoint}_rule"])
            placebo_recall = float(record[f"{endpoint}_placebo"])
            rule_gain = rule_recall - real_mean
            placebo_gain = placebo_recall - real_mean
            unstable = abs(placebo_gain) < RATIO_MIN_DENOMINATOR
            rows.append({
                "P": p_level,
                "attack_scope": attack_scope,
                "construction_seed": int(record["construction_seed"]),
                "real_recall": real_mean,
                "rule_recall": rule_recall,
                "placebo_recall": placebo_recall,
                "rule_gain": rule_gain,
                "placebo_gain": placebo_gain,
                "absolute_difference": rule_recall - placebo_recall,
                "gain_ratio": (np.nan if unstable
                               else rule_gain / placebo_gain),
                "ratio_unstable": unstable,
                "ratio_denominator": placebo_gain,
            })
    table = pd.DataFrame(rows)

    summary: dict[str, Any] = {}
    for attack_scope in ("equal_macro", "Gear", "RPM"):
        scoped = table[table["attack_scope"] == attack_scope]
        differences = scoped["absolute_difference"].to_numpy(dtype=float)
        mean, sd, lo, hi = _t_interval(differences, MATERIAL_CONFIDENCE)
        stable = scoped[~scoped["ratio_unstable"]]["gain_ratio"].dropna()
        summary[attack_scope] = {
            "n": int(len(scoped)),
            "degrees_of_freedom": int(len(scoped) - 1),
            "real_recall": float(scoped["real_recall"].iloc[0]),
            "rule_recall_mean": float(scoped["rule_recall"].mean()),
            "rule_recall_sd": float(scoped["rule_recall"].std(ddof=1)),
            "placebo_recall_mean": float(scoped["placebo_recall"].mean()),
            "placebo_recall_sd": float(scoped["placebo_recall"].std(ddof=1)),
            "rule_gain_mean": float(scoped["rule_gain"].mean()),
            "placebo_gain_mean": float(scoped["placebo_gain"].mean()),
            "placebo_gain_sd": float(scoped["placebo_gain"].std(ddof=1)),
            "absolute_difference_mean": mean,
            "absolute_difference_sd": sd,
            "absolute_difference_ci95_low": lo,
            "absolute_difference_ci95_high": hi,
            "ratio_median": float(stable.median()) if len(stable) else None,
            "ratio_iqr_low": (float(stable.quantile(0.25)) if len(stable)
                              else None),
            "ratio_iqr_high": (float(stable.quantile(0.75)) if len(stable)
                               else None),
            "ratio_realizations_used": int(len(stable)),
            "ratio_realizations_unstable": int(scoped["ratio_unstable"].sum()),
            "ratio_denominator_median": float(
                scoped["ratio_denominator"].median()),
            "ratio_denominator_min": float(scoped["ratio_denominator"].min()),
            "ratio_denominator_max": float(scoped["ratio_denominator"].max()),
        }
    return table, summary


def determine_p0_saturation(p0_summary: Mapping[str, Any],
                            decomposition: pd.DataFrame) -> dict[str, Any]:
    """PREREG section 12.3 — the registered saturation determination.

    Saturated iff both augmented arms exceed 0.99 exact-macro recall and their
    absolute difference is below 0.01.  When unsaturated, the ``P = 0``
    contribution to ``C_P`` is reported separately from the ``P = 1``
    contribution, because ``C_P`` is then not dominated by the ``P = 1`` cell.
    """
    macro = p0_summary["equal_macro"]
    rule_recall = float(macro["rule_recall_mean"])
    placebo_recall = float(macro["placebo_recall_mean"])
    difference = abs(rule_recall - placebo_recall)
    saturated = bool(
        rule_recall > P0_SATURATION_RECALL
        and placebo_recall > P0_SATURATION_RECALL
        and difference < P0_SATURATION_DIFFERENCE)
    primary = decomposition[decomposition["role"] == "primary"]
    p1_part = primary["theta_p1_component"].to_numpy(dtype=float)
    p0_part = primary["theta_p0_component"].to_numpy(dtype=float)
    return {
        "criterion": (
            f"saturated iff both augmented arms exceed {P0_SATURATION_RECALL} "
            f"exact-macro recall and |Rule - placebo| < "
            f"{P0_SATURATION_DIFFERENCE}"),
        "real_recall": float(macro["real_recall"]),
        "rule_recall_mean": rule_recall,
        "rule_recall_sd": float(macro["rule_recall_sd"]),
        "placebo_recall_mean": placebo_recall,
        "placebo_recall_sd": float(macro["placebo_recall_sd"]),
        "rule_minus_placebo": rule_recall - placebo_recall,
        "saturated": saturated,
        "determination": "saturated" if saturated else "unsaturated",
        "in_grammar_cell_cancels_from_theta": saturated,
        "theta_p1_contribution_mean": float(p1_part.mean()),
        "theta_p1_contribution_sd": float(p1_part.std(ddof=1)),
        "theta_p0_contribution_mean": float(p0_part.mean()),
        "theta_p0_contribution_sd": float(p0_part.std(ddof=1)),
        "note": (
            "theta = (P=1 contribution) - (P=0 contribution); a saturated P=0 "
            "cell makes the second term vanish and C_P is dominated by the "
            "out-of-grammar cell, as it is for the CNN"),
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
            "Rule-minus-real P mean); inherited verbatim from E16, not "
            "re-derived from any Random-Forest quantity"),
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
            "E17 real-only reference"),
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
            f"Against the inherited CNN margin the Random-Forest replication "
            f"returns Branch {primary['verdict']}; against a margin rescaled "
            f"to the Random Forest's own out-of-grammar placebo gain it "
            f"returns Branch {strict_verdict}. The verdict is therefore "
            f"margin-scale dependent, and the equivalence claim is reported as "
            f"conditional on the registered absolute margin."
        ) if disagreement else (
            f"Both the inherited CNN margin and a margin rescaled to the "
            f"Random Forest's own out-of-grammar placebo gain return Branch "
            f"{primary['verdict']}."
        ) if strict_verdict != "undefined" else (
            f"The inherited CNN margin returns Branch {primary['verdict']}. "
            f"delta_strict is undefined because G_placebo = {g_placebo:.9f} "
            f"<= {STRICT_MARGIN_MIN_DENOMINATOR}; the primary verdict stands "
            f"unannotated by a sensitivity that could not be computed."),
    }
    return pd.DataFrame(rows), record


# ---------------------------------------------------------------------------
# Secondary families
# ---------------------------------------------------------------------------


def build_secondary_family(per_realization: pd.DataFrame, *,
                           effects: Sequence[str], label: str,
                           endpoint: str, holm: bool = True) -> pd.DataFrame:
    """Rule-minus-placebo effects for one registered Holm family."""
    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for effect in effects:
        frame = per_realization[
            (per_realization["effect"] == effect)
            & (per_realization["role"] == "primary")]
        rule = frame[frame["arm"] == RULE_ARM].set_index("construction_seed")
        placebo = frame[frame["arm"] == PLACEBO_ARM].set_index(
            "construction_seed")
        shared = sorted(set(rule.index) & set(placebo.index))
        values = np.array(
            [float(rule.loc[c, "value"]) - float(placebo.loc[c, "value"])
             for c in shared], dtype=float)
        mean, sd, lo, hi = _t_interval(values, MATERIAL_CONFIDENCE)
        result = stats.ttest_1samp(values, 0.0)
        raw_p.append(float(result.pvalue))
        rows.append({
            "family": label, "endpoint": endpoint, "effect": effect,
            "n": len(values), "mean": mean, "sd": sd,
            "ci95_low": lo, "ci95_high": hi,
            "raw_p": float(result.pvalue),
            "strict_positive": int(np.count_nonzero(values > 0)),
            "strict_negative": int(np.count_nonzero(values < 0)),
        })
    adjusted = holm_adjust(raw_p) if holm else [np.nan] * len(raw_p)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["resolved"] = bool(value < 0.05) if holm else ""
    return pd.DataFrame(rows)


def build_arm_summary(per_realization: pd.DataFrame, *, arm: str,
                      effect: str = "P") -> dict[str, Any]:
    """PREREG section 12.6 — component means over the twenty realizations."""
    frame = per_realization[
        (per_realization["effect"] == effect)
        & (per_realization["arm"] == arm)
        & (per_realization["role"] == "primary")]
    values = frame["value"].to_numpy(dtype=float)
    mean, sd, lo, hi = _t_interval(values, MATERIAL_CONFIDENCE)
    return {"arm": arm, "effect": effect, "n": len(values),
            "degrees_of_freedom": len(values) - 1, "mean": mean, "sd": sd,
            "ci95_low": lo, "ci95_high": hi,
            "strict_negative": int(np.count_nonzero(values < 0)),
            "strict_positive": int(np.count_nonzero(values > 0))}


def build_attack_heterogeneity(scenario: pd.DataFrame) -> pd.DataFrame:
    """PREREG section 12.8 — attack-specific theta and shifted-ID modification."""
    rows: list[dict[str, Any]] = []
    for scope_label, kwargs in (
            ("Gear_canonical", {"attack_scope": "Gear"}),
            ("RPM_canonical", {"attack_scope": "RPM"}),
            ("equal_macro_shifted", {"id_stratum": "shifted"}),
    ):
        deltas = build_augmentation_deltas(scenario, **kwargs)
        _, per_realization = build_effects_by_realization(deltas, effects=("P",))
        decomposition = build_decomposition(per_realization, effect="P")
        primary = decomposition[decomposition["role"] == "primary"]
        theta = primary["theta_rule_minus_placebo"].to_numpy(dtype=float)
        mean, sd, lo, hi = _t_interval(theta, MATERIAL_CONFIDENCE)
        rows.append({
            "family": "attack_and_stratum_heterogeneity_descriptive",
            "endpoint": PRIMARY_ENDPOINT,
            "effect": scope_label,
            "n": len(theta), "mean": mean, "sd": sd,
            "ci95_low": lo, "ci95_high": hi,
            "raw_p": np.nan,
            "strict_positive": int(np.count_nonzero(theta > 0)),
            "strict_negative": int(np.count_nonzero(theta < 0)),
            "holm_adjusted_p": np.nan,
            "resolved": "",
        })
    return pd.DataFrame(rows)


def build_crossed_dispersion(per_pipeline: pd.DataFrame, *, effect: str = "P"
                             ) -> pd.DataFrame:
    """PREREG section 12.10 — descriptive finite-grid dispersion of theta."""
    frame = per_pipeline[
        (per_pipeline["effect"] == effect)
        & (per_pipeline["role"] == "primary")]
    rule = frame[frame["arm"] == RULE_ARM]
    placebo = frame[frame["arm"] == PLACEBO_ARM]
    merged = rule.merge(
        placebo, on=["construction_seed", "pipeline_seed"],
        suffixes=("_rule", "_placebo"))
    merged["theta"] = merged["value_rule"] - merged["value_placebo"]
    grid = merged.pivot_table(
        index="construction_seed", columns="pipeline_seed", values="theta")
    values = grid.to_numpy(dtype=float)
    grand = float(values.mean())
    realization_margin = values.mean(axis=1)
    pipeline_margin = values.mean(axis=0)
    interaction = (values - realization_margin[:, None]
                   - pipeline_margin[None, :] + grand)
    ss_c = float(values.shape[1] * ((realization_margin - grand) ** 2).sum())
    ss_p = float(values.shape[0] * ((pipeline_margin - grand) ** 2).sum())
    ss_i = float((interaction ** 2).sum())
    ss_total = ss_c + ss_p + ss_i
    return pd.DataFrame([{
        "effect": effect,
        "grid_cells": int(values.size),
        "realizations": int(values.shape[0]),
        "pipelines": int(values.shape[1]),
        "grand_mean": grand,
        "realization_marginal_sample_sd": float(
            realization_margin.std(ddof=1)),
        "pipeline_marginal_sample_sd": float(pipeline_margin.std(ddof=1)),
        "rms_interaction": float(np.sqrt((interaction ** 2).mean())),
        "ss_realization": ss_c, "ss_pipeline": ss_p, "ss_interaction": ss_i,
        "ss_total": ss_total,
        "share_realization": ss_c / ss_total if ss_total else np.nan,
        "share_pipeline": ss_p / ss_total if ss_total else np.nan,
        "share_interaction": ss_i / ss_total if ss_total else np.nan,
        "interpretation": "descriptive finite-grid dispersion; not a "
                          "random-effects variance component",
    }])


def build_family_comparison(decomposition: pd.DataFrame
                            ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """PREREG section 12.7 — paired CNN/RF theta, descriptive only, no test."""
    if not E16_DECOMPOSITION.exists():
        raise _stop(f"missing frozen E16 decomposition: {E16_DECOMPOSITION}")
    cnn = pd.read_csv(E16_DECOMPOSITION)
    cnn = cnn[(cnn["role"] == "primary") & (cnn["effect"] == "P")]
    rf = decomposition[decomposition["role"] == "primary"]
    merged = cnn[["construction_seed", "theta_rule_minus_placebo",
                  "delta_rule_minus_real", "delta_placebo_minus_real"]].merge(
        rf[["construction_seed", "theta_rule_minus_placebo",
            "delta_rule_minus_real", "delta_placebo_minus_real"]],
        on="construction_seed", suffixes=("_cnn", "_rf"), validate="1:1")
    if len(merged) != len(rf):
        raise _stop(
            f"family comparison pairs {len(merged)} of {len(rf)} realizations; "
            "the two families must share the same twenty units")
    merged = merged.sort_values("construction_seed").reset_index(drop=True)
    cnn_theta = merged["theta_rule_minus_placebo_cnn"].to_numpy(dtype=float)
    rf_theta = merged["theta_rule_minus_placebo_rf"].to_numpy(dtype=float)
    difference = rf_theta - cnn_theta
    merged["theta_rf_minus_cnn"] = difference
    pearson = stats.pearsonr(cnn_theta, rf_theta)
    spearman = stats.spearmanr(cnn_theta, rf_theta)
    summary = {
        "n": int(len(merged)),
        "cnn_theta_mean": float(cnn_theta.mean()),
        "cnn_theta_sd": float(cnn_theta.std(ddof=1)),
        "rf_theta_mean": float(rf_theta.mean()),
        "rf_theta_sd": float(rf_theta.std(ddof=1)),
        "paired_difference_mean": float(difference.mean()),
        "paired_difference_sd": float(difference.std(ddof=1)),
        "pearson_r": float(pearson.statistic),
        "spearman_rho": float(spearman.statistic),
        "status": "descriptive_only",
        "prohibition": (
            "two detector families are two points, not a sample; no p-value, "
            "confidence interval, or equivalence test on the family difference "
            "is reported, and no invariance-across-families claim is permitted"),
    }
    return merged, summary


# ---------------------------------------------------------------------------
# Gate G16 — dual-margin reporting completeness
# ---------------------------------------------------------------------------


def verify_dual_margin_reporting(record: Mapping[str, Any]) -> dict[str, Any]:
    """Gate G16 — the anti-suppression rule, made enforceable."""
    missing: list[str] = []
    verdict = record.get("verdict")
    strict = record.get("verdict_strict_margin")
    if verdict not in ("A", "B", "C", "D"):
        missing.append("verdict")
    if strict not in ("A", "B", "C", "D", "undefined"):
        missing.append("verdict_strict_margin")
    if strict == "undefined" and record.get("g_placebo") is None:
        missing.append("g_placebo_published_with_undefined_strict_margin")
    if not isinstance(record.get("margin_disagreement"), bool):
        missing.append("margin_disagreement")
    p1 = record.get("p1_cell_summary") or {}
    macro = p1.get("equal_macro") or {}
    for field in ("rule_recall_mean", "placebo_recall_mean", "real_recall",
                  "absolute_difference_mean", "absolute_difference_sd",
                  "absolute_difference_ci95_low",
                  "absolute_difference_ci95_high"):
        if macro.get(field) is None:
            missing.append(f"p1_absolute.{field}")
    if macro.get("ratio_median") is None and macro.get(
            "ratio_realizations_used", 0) != 0:
        missing.append("p1_ratio_summary")
    if "ratio_realizations_used" not in macro:
        missing.append("p1_ratio_summary")
    saturation = record.get("p0_saturation") or {}
    if saturation.get("determination") not in ("saturated", "unsaturated"):
        missing.append("p0_saturation_determination")
    if missing:
        raise _stop(
            f"G16 dual-margin reporting completeness FAILED; missing {missing}")
    return {
        "gate": "G16_dual_margin_reporting_completeness",
        "passed": True,
        "verdict": verdict,
        "verdict_strict_margin": strict,
        "margin_disagreement": record["margin_disagreement"],
        "p1_absolute_reported": True,
        "p1_ratio_reported": True,
        "p0_saturation_reported": True,
    }


def verify_output_integrity(scenario: pd.DataFrame,
                            decomposition: pd.DataFrame) -> dict[str, Any]:
    """Gate G13 — registered completeness of the scored grid."""
    counts = (scenario.groupby(["arm", "role"])["construction_seed"]
              .nunique().to_dict())
    cells = len(e17train.PIPELINE_SEEDS) * 3 * 2 * 2 * 8
    per_identity = scenario.groupby(
        ["arm", "construction_seed", "pipeline_seed"]).size()
    if not (per_identity == 3 * 2 * 2 * 8).all():
        raise _stop("an identity does not have its 96 registered scenario rows")
    primary = decomposition[decomposition["role"] == "primary"]
    bridge = decomposition[decomposition["role"] == "bridge_sensitivity"]
    if len(primary) != len(e17train.CONSTRUCTION_SEEDS):
        raise _stop(
            f"primary has {len(primary)} realizations, registered "
            f"{len(e17train.CONSTRUCTION_SEEDS)}")
    if len(bridge) != len(e17train.BRIDGE_SEEDS):
        raise _stop(
            f"bridge has {len(bridge)} realizations, registered "
            f"{len(e17train.BRIDGE_SEEDS)}")
    reference = scenario[scenario["arm"] == REAL_ARM]
    reference_fits = reference["pipeline_seed"].nunique()
    if reference_fits != len(e17train.PIPELINE_SEEDS):
        raise _stop(f"{reference_fits} shared reference fits, registered 5")
    return {
        "gate": "G13_output_integrity",
        "passed": True,
        "primary_completeness": f"{len(primary)} x 2 arms x "
                                f"{len(e17train.PIPELINE_SEEDS)} pipelines",
        "bridge_completeness": f"{len(bridge)} x 2 arms x "
                               f"{len(e17train.PIPELINE_SEEDS)} pipelines",
        "reference_fits": reference_fits,
        "scenario_rows_per_identity": 3 * 2 * 2 * 8,
        "scenario_rows_per_arm_realization": cells,
        "unique_realizations_by_arm_role": {
            f"{k[0]}:{k[1]}": int(v) for k, v in counts.items()},
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
    if len(theta) != len(e17train.CONSTRUCTION_SEEDS):
        raise _stop(
            f"primary has {len(theta)} realizations, registered "
            f"{len(e17train.CONSTRUCTION_SEEDS)}")
    decision = primary_decision(theta, delta=DELTA, margin_label="registered")
    share = relational_share(primary_rows)

    p1_table, p1_summary = build_cell_comparison(scenario, p_level=1)
    p0_table, p0_summary = build_cell_comparison(scenario, p_level=0)
    saturation = determine_p0_saturation(p0_summary, decomposition)
    margin_table, margin_record = build_margin_sensitivity(
        theta, p1_summary, decision)

    exact_family = build_secondary_family(
        per_realization, effects=SECONDARY_EXACT_FAMILY,
        label="exact_rule_minus_placebo_secondary", endpoint=PRIMARY_ENDPOINT)
    binary_deltas = build_augmentation_deltas(
        scenario, endpoint=COMPANION_ENDPOINT)
    _, binary_per_realization = build_effects_by_realization(binary_deltas)
    binary_family = build_secondary_family(
        binary_per_realization, effects=EFFECTS,
        label="binary_rule_minus_placebo_companion",
        endpoint=COMPANION_ENDPOINT)
    heterogeneity = build_attack_heterogeneity(scenario)

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
        "dispersion": dispersion,
        "family_table": family_table,
        "family_summary": family_summary,
        "bridge": bridge,
        "marginal_component": build_arm_summary(
            per_realization, arm=PLACEBO_ARM),
        "total_response": build_arm_summary(per_realization, arm=RULE_ARM),
    }


def _load_scenario() -> pd.DataFrame:
    path = TABLE_DIR / f"e17_by_scenario_{OUTPUT_VERSION}.csv"
    if not path.exists():
        raise _stop(f"missing scored table: {path}")
    return pd.read_csv(path)


def execute() -> dict[str, Any]:
    started = time.monotonic()
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
        "stage": "e17_registered_analysis",
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "registration": e17train.assert_registration_committed(),
        "environment": _environment_record(REPO),
        "inferential_unit": {
            "unit": "generator realization", "n": decision["n"],
            "degrees_of_freedom": decision["degrees_of_freedom"],
            "never_units": ["block", "cell", "attack", "pipeline seed",
                            "window", "fit"],
        },
        "matching_contract": (
            "identical training rows; the Random-Forest arms carry no "
            "optimizer-update budget, early-stopping trajectory, or minibatch "
            "order, so the comparison is made under an identical-training-row "
            "contract rather than a matched training schedule (PREREG 5.2)"),
        "closure_gate": result["closure_gate"],
        "output_integrity_gate": result["integrity_gate"],
        "primary": decision,
        "relational_share": result["relational_share"],
        "p1_cell_summary": result["p1_summary"],
        "p0_cell_summary": result["p0_summary"],
        "p0_saturation": result["p0_saturation"],
        "marginal_exposure_component": result["marginal_component"],
        "total_augmentation_response": result["total_response"],
        "cnn_rf_family_comparison": result["family_summary"],
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
    verdict_record["dual_margin_gate"] = verify_dual_margin_reporting(
        verdict_record)

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
        "stage": "e17_rf_relational_replication",
        "created_utc": utc_now(),
        "registration": verdict_record["registration"],
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
    """Every registered E17 output (PREREG section 14.2), for the manifest."""
    from evaluate_e17_rf_relational_response import output_paths as eval_paths

    paths = [p for p in e17train.record_paths().values()]
    paths += [p for p in eval_paths().values()]
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
            "stage": "e17_registered_analysis",
            "mode": "read-only",
            "delta": DELTA,
            "sesoi_multiplier": SESOI_MULTIPLIER,
            "sesoi_denominator": SESOI_DENOMINATOR,
            "strict_margin_multiplier": SESOI_MULTIPLIER,
            "equivalence_confidence": EQUIVALENCE_CONFIDENCE,
            "heterogeneity_min_outside": HETEROGENEITY_MIN_OUTSIDE,
            "heterogeneity_min_each_side": HETEROGENEITY_MIN_EACH_SIDE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "outputs": {k: str(v.relative_to(REPO))
                        for k, v in output_paths().items()},
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
