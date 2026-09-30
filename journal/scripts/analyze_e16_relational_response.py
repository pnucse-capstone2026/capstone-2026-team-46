#!/usr/bin/env python3
"""Registered analysis of the E16 relational-versus-marginal decomposition.

Registered by ``journal/experiments/e16_relational_response_decomposition/PREREG.md``
sections 8--13.

Primary estimand
----------------
For construction ``g`` and pipeline ``p``, with the shared matched-real arm as
reference::

    A^arm(P,S,D) = recall_arm(P,S,D) - recall_real(P,S,D)
    C_P{A}       = mean over (S,D) of [ A(1,S,D) - A(0,S,D) ]

The registered aggregation order is attack -> block -> cell -> pipeline, after
which the constructions are the inferential unit::

    theta_g = C_P{A^Rule}_g - C_P{A^Placebo}_g          (n = 20, df = 19)

Decision order (registered, not reorderable)
--------------------------------------------
1. heterogeneity screen: >= 5 of 20 ``theta_g`` outside [-delta, +delta] -> C
2. practical equivalence: 90% t interval inside [-delta, +delta]       -> A
3. material difference:   95% t interval entirely outside              -> B
4. otherwise                                                           -> D

A Student-t / bootstrap disagreement downgrades the verdict to D.

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

import generate_e16_rule_placebo_pairs as e16gen  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _environment_record,
    _source_record,
    _stage_bytes,
    _stage_json,
    atomic_publish_bundle,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    REPO / "journal" / "experiments" / "e16_relational_response_decomposition"
)
TABLE_DIR = REPO / "journal" / "results" / "tables"
LOG_DIR = REPO / "journal" / "results" / "logs"

OUTPUT_VERSION = e16gen.OUTPUT_VERSION
ANALYSIS_SCHEMA = "e16.registered_analysis.v1"

# --- frozen decision constants (PREREG sections 9.1, 10) --------------------

SESOI_MULTIPLIER = 0.15
SESOI_DENOMINATOR = 0.8923708333333333   # E15 v2 new4 Rule-minus-real P mean
DELTA = 0.133855625                      # = SESOI_MULTIPLIER * SESOI_DENOMINATOR
EQUIVALENCE_CONFIDENCE = 0.90            # TOST: two one-sided 5% tests
MATERIAL_CONFIDENCE = 0.95
HETEROGENEITY_MIN_OUTSIDE = 5            # of 20, conflict criterion C2
HETEROGENEITY_MIN_EACH_SIDE = 3          # of 20, directional criterion C1
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260725
RELATIONAL_SHARE_MIN_DENOMINATOR = 0.10
RATIO_MIN_DENOMINATOR = 0.01

PRIMARY_ENDPOINT = "exact_recall"
PRIMARY_STRATUM = "canonical"
COMPANION_ENDPOINT = "binary_recall"
EFFECTS = ("P", "S", "D", "PS", "PD", "SD", "PSD")
SECONDARY_EXACT_FAMILY = ("S", "D", "PS", "PD", "SD", "PSD")

REAL_ARM = "real_ms"
RULE_ARM = "rule"
PLACEBO_ARM = "placebo"


class E16AnalysisError(RuntimeError):
    """Raised when the registered analysis cannot be computed as specified."""


def _stop(message: str) -> E16AnalysisError:
    return E16AnalysisError(f"T-STOP-E16-ANALYSIS: {message}")


def utc_now() -> str:
    return e16gen.utc_now()



def publish_and_cleanup(staged: Sequence[tuple[Path, Path]]) -> None:
    """Publish atomically, then remove the staged hard-link sources.

    ``atomic_publish_bundle`` links staged files into place and deliberately
    leaves them behind so a failed bundle can be diagnosed.  On success the
    caller owns that cleanup; skipping it leaves dot-prefixed temporaries that
    a later stage's clean-source gate will reject.
    """
    atomic_publish_bundle(staged)
    for source, _ in staged:
        try:
            Path(source).unlink()
        except FileNotFoundError:
            pass

def output_paths() -> dict[str, Path]:
    return {
        "decomposition": TABLE_DIR
        / f"e16_decomposition_by_construction_{OUTPUT_VERSION}.csv",
        "primary": TABLE_DIR / f"e16_primary_theta_summary_{OUTPUT_VERSION}.csv",
        "p1_cell": TABLE_DIR / f"e16_p1_cell_comparison_{OUTPUT_VERSION}.csv",
        "secondary": TABLE_DIR / f"e16_secondary_effects_{OUTPUT_VERSION}.csv",
        "dispersion": TABLE_DIR / f"e16_crossed_dispersion_{OUTPUT_VERSION}.csv",
        "bridge": TABLE_DIR / f"e16_bridge_sensitivity_{OUTPUT_VERSION}.csv",
        "verdict": LOG_DIR / f"e16_relational_response_{OUTPUT_VERSION}.log",
        "manifest": LOG_DIR
        / f"e16_relational_response_artifact_manifest_{OUTPUT_VERSION}.json",
    }


# ---------------------------------------------------------------------------
# Factorial contrasts
# ---------------------------------------------------------------------------


def factor_contrast(cell_values: Mapping[tuple[int, int, int], float],
                    effect: str) -> float:
    """Saturated 2x2x2 contrast for one registered effect label.

    Each named factor contributes a ``(+1 at level 1, -1 at level 0)`` sign.  The
    signed sum is averaged over the factors the effect does **not** name, which
    is the E14/E15 convention::

        E_P   = mean_{S,D}[ A(1,S,D) - A(0,S,D) ]                divisor 4
        E_PS  = mean_D[ (A(1,1,D)-A(1,0,D)) - (A(0,1,D)-A(0,0,D)) ]  divisor 2
        E_PSD = the full three-fold difference                    divisor 1

    so the divisor is ``2 ** (3 - k)`` for an effect naming ``k`` factors.  A
    divisor of ``4`` for every effect would understate the two-way effects by a
    factor of two and the three-way effect by a factor of four; this is checked
    against the published E14 values in the pre-outcome tests.
    """
    if effect not in EFFECTS:
        raise _stop(f"unknown effect label: {effect}")
    total = 0.0
    for (p, s, d), value in cell_values.items():
        sign = 1.0
        for factor, level in (("P", p), ("S", s), ("D", d)):
            if factor in effect:
                sign *= 1.0 if level == 1 else -1.0
        total += sign * value
    return total / float(2 ** (3 - len(effect)))


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
    """Registered aggregation: attack -> block -> per-cell augmentation delta.

    Returns one row per (arm, construction_seed, pipeline_seed, P, S, D).
    """
    frame = scenario[scenario["id_stratum"].astype(str) == id_stratum].copy()
    if attack_scope != "equal_macro":
        frame = frame[frame["attack"].astype(str) == attack_scope]
    if frame.empty:
        raise _stop(f"no rows for stratum={id_stratum} scope={attack_scope}")

    keys = ["arm", "construction_seed", "pipeline_seed", "role",
            "block_id", "P", "S", "D"]
    # step 1: attack equally weighted
    by_block = frame.groupby(keys, as_index=False)[endpoint].mean()
    # step 2: blocks equally weighted
    cell_keys = ["arm", "construction_seed", "pipeline_seed", "role",
                 "P", "S", "D"]
    by_cell = by_block.groupby(cell_keys, as_index=False)[endpoint].mean()

    real = by_cell[by_cell["arm"].astype(str) == REAL_ARM]
    if real.empty:
        raise _stop("shared real arm missing from the scored table")
    real = real[["pipeline_seed", "P", "S", "D", endpoint]].rename(
        columns={endpoint: "real_value"})

    arms = by_cell[by_cell["arm"].astype(str) != REAL_ARM].merge(
        real, on=["pipeline_seed", "P", "S", "D"], how="left", validate="m:1")
    if arms["real_value"].isna().any():
        raise _stop("an arm cell has no matching shared-real reference")
    arms["augmentation_delta"] = arms[endpoint] - arms["real_value"]
    return arms


def build_effects_by_construction(
        deltas: pd.DataFrame,
        *,
        effects: Sequence[str] = EFFECTS,
) -> pd.DataFrame:
    """Registered steps 3--4: cell contrast, then pipeline equal weighting."""
    rows: list[dict[str, Any]] = []
    group_keys = ["arm", "construction_seed", "pipeline_seed", "role"]
    for key, group in deltas.groupby(group_keys):
        arm, construction_seed, pipeline_seed, role = key
        cells = _cell_map(group, "augmentation_delta")
        for effect in effects:
            rows.append({
                "arm": arm,
                "construction_seed": int(construction_seed),
                "pipeline_seed": int(pipeline_seed),
                "role": role,
                "effect": effect,
                "value": factor_contrast(cells, effect),
            })
    per_pipeline = pd.DataFrame(rows)
    per_construction = per_pipeline.groupby(
        ["arm", "construction_seed", "role", "effect"],
        as_index=False)["value"].mean()
    return per_pipeline, per_construction


def build_decomposition(per_construction: pd.DataFrame, *, effect: str = "P"
                        ) -> pd.DataFrame:
    """The registered three-term decomposition, one row per construction."""
    frame = per_construction[per_construction["effect"] == effect]
    rule = frame[frame["arm"] == RULE_ARM].set_index("construction_seed")
    placebo = frame[frame["arm"] == PLACEBO_ARM].set_index("construction_seed")
    shared = sorted(set(rule.index) & set(placebo.index))
    rows: list[dict[str, Any]] = []
    for construction_seed in shared:
        delta_rule = float(rule.loc[construction_seed, "value"])
        delta_placebo = float(placebo.loc[construction_seed, "value"])
        rows.append({
            "construction_seed": construction_seed,
            "role": str(rule.loc[construction_seed, "role"]),
            "effect": effect,
            "delta_rule_minus_real": delta_rule,
            "delta_placebo_minus_real": delta_placebo,
            "theta_rule_minus_placebo": delta_rule - delta_placebo,
            "closure_residual": delta_rule
            - (delta_placebo + (delta_rule - delta_placebo)),
        })
    return pd.DataFrame(rows)


def assert_decomposition_closure(decomposition: pd.DataFrame,
                                 tolerance: float = 1e-12) -> dict[str, Any]:
    """Gate G11 — the registered identity holds to floating-point closure."""
    residual = decomposition["closure_residual"].abs().max()
    if not np.isfinite(residual) or residual > tolerance:
        raise _stop(
            f"decomposition closure violated: max |residual| = {residual}")
    return {"gate": "G11_decomposition_closure", "passed": True,
            "max_abs_residual": float(residual), "tolerance": tolerance}


# ---------------------------------------------------------------------------
# Primary decision (PREREG section 10)
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


def primary_decision(theta: np.ndarray, *, delta: float = DELTA
                     ) -> dict[str, Any]:
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

    # Heterogeneity must detect construction *disagreement*, not a uniformly
    # large effect.  A real Branch B effect puts every construction on the same
    # side of the band, which is agreement, not heterogeneity.
    #   C1 directional disagreement: constructions materially on both sides
    #   C2 aggregate/unit conflict: the mean says immaterial while a quarter of
    #      the constructions do not
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
            f"directional disagreement: {above} constructions above +{delta} "
            f"and {below} below -{delta} (threshold "
            f"{HETEROGENEITY_MIN_EACH_SIDE} each side)"
        ) if c1 else (
            f"aggregate/unit conflict: the aggregate interval is inside "
            f"[-{delta}, +{delta}] while {outside} of {n} constructions are "
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
        "verdict": verdict,
        "verdict_reason": reason,
        "n": n,
        "degrees_of_freedom": n - 1,
        "theta_mean": mean,
        "theta_sd": sd,
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "delta": delta,
        "sesoi_multiplier": SESOI_MULTIPLIER,
        "sesoi_denominator": SESOI_DENOMINATOR,
        "ci90_low": lo90, "ci90_high": hi90,
        "ci95_low": lo95, "ci95_high": hi95,
        "practical_equivalence": equivalent,
        "material_difference": material,
        "constructions_outside_band": outside,
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
                "excluded_constructions": excluded, "n_used": 0}
    share = usable["theta_rule_minus_placebo"] / usable["denominator"]
    return {
        "median": float(share.median()),
        "iqr_low": float(share.quantile(0.25)),
        "iqr_high": float(share.quantile(0.75)),
        "minimum": float(share.min()),
        "maximum": float(share.max()),
        "excluded_constructions": excluded,
        "n_used": int(len(share)),
        "denominator_median": float(usable["denominator"].median()),
    }


# ---------------------------------------------------------------------------
# Mandatory P = 1 secondary (PREREG section 12.1)
# ---------------------------------------------------------------------------


def build_p1_cell_comparison(
        scenario: pd.DataFrame,
        *,
        endpoint: str = PRIMARY_ENDPOINT,
        id_stratum: str = PRIMARY_STRATUM,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Absolute arm recall at P=1 plus the registered ratio form."""
    frame = scenario[
        (scenario["id_stratum"].astype(str) == id_stratum)
        & (scenario["P"].astype(int) == 1)].copy()
    if frame.empty:
        raise _stop("no P=1 rows available")

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
        by_construction = by_pipeline.groupby(
            ["arm", "construction_seed", "role"],
            as_index=False)[endpoint].mean()

        real = by_pipeline[by_pipeline["arm"] == REAL_ARM]
        real_mean = float(real[endpoint].mean()) if not real.empty else np.nan
        # both arms restricted to the primary constructions; the bridge is
        # reported separately and never enters this secondary
        rule = by_construction[
            (by_construction["arm"] == RULE_ARM)
            & (by_construction["role"] == "primary")]
        placebo = by_construction[
            (by_construction["arm"] == PLACEBO_ARM)
            & (by_construction["role"] == "primary")]
        merged = rule.merge(
            placebo, on="construction_seed", suffixes=("_rule", "_placebo"))
        for record in merged.to_dict(orient="records"):
            rule_recall = float(record[f"{endpoint}_rule"])
            placebo_recall = float(record[f"{endpoint}_placebo"])
            rule_gain = rule_recall - real_mean
            placebo_gain = placebo_recall - real_mean
            unstable = abs(placebo_gain) < RATIO_MIN_DENOMINATOR
            rows.append({
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
            "absolute_difference_mean": mean,
            "absolute_difference_sd": sd,
            "absolute_difference_ci95_low": lo,
            "absolute_difference_ci95_high": hi,
            "rule_recall_mean": float(scoped["rule_recall"].mean()),
            "placebo_recall_mean": float(scoped["placebo_recall"].mean()),
            "ratio_median": float(stable.median()) if len(stable) else None,
            "ratio_iqr_low": float(stable.quantile(0.25)) if len(stable) else None,
            "ratio_iqr_high": float(stable.quantile(0.75)) if len(stable) else None,
            "ratio_constructions_used": int(len(stable)),
            "ratio_constructions_unstable": int(scoped["ratio_unstable"].sum()),
        }
    return table, summary


# ---------------------------------------------------------------------------
# Secondary families
# ---------------------------------------------------------------------------


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * float(pvalues[index])
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def build_secondary_family(per_construction: pd.DataFrame, *,
                           effects: Sequence[str], label: str,
                           endpoint: str) -> pd.DataFrame:
    """Rule-minus-placebo effects for one registered Holm family."""
    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for effect in effects:
        frame = per_construction[
            (per_construction["effect"] == effect)
            & (per_construction["role"] == "primary")]
        rule = frame[frame["arm"] == RULE_ARM].set_index("construction_seed")
        placebo = frame[frame["arm"] == PLACEBO_ARM].set_index("construction_seed")
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
    for row, adjusted in zip(rows, holm_adjust(raw_p), strict=True):
        row["holm_adjusted_p"] = adjusted
        row["resolved"] = bool(adjusted < 0.05)
    return pd.DataFrame(rows)


def build_arm_summary(per_construction: pd.DataFrame, *, arm: str,
                      effect: str = "P") -> dict[str, Any]:
    frame = per_construction[
        (per_construction["effect"] == effect)
        & (per_construction["arm"] == arm)
        & (per_construction["role"] == "primary")]
    values = frame["value"].to_numpy(dtype=float)
    mean, sd, lo, hi = _t_interval(values, MATERIAL_CONFIDENCE)
    return {"arm": arm, "effect": effect, "n": len(values), "mean": mean,
            "sd": sd, "ci95_low": lo, "ci95_high": hi,
            "strict_negative": int(np.count_nonzero(values < 0)),
            "strict_positive": int(np.count_nonzero(values > 0))}


def build_crossed_dispersion(per_pipeline: pd.DataFrame, *, effect: str = "P"
                             ) -> pd.DataFrame:
    """Descriptive finite-grid dispersion of theta over the construction grid."""
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
    construction_margin = values.mean(axis=1)
    pipeline_margin = values.mean(axis=0)
    interaction = (values - construction_margin[:, None]
                   - pipeline_margin[None, :] + grand)
    ss_c = float(values.shape[1] * ((construction_margin - grand) ** 2).sum())
    ss_p = float(values.shape[0] * ((pipeline_margin - grand) ** 2).sum())
    ss_i = float((interaction ** 2).sum())
    ss_total = ss_c + ss_p + ss_i
    return pd.DataFrame([{
        "effect": effect,
        "grid_cells": int(values.size),
        "constructions": int(values.shape[0]),
        "pipelines": int(values.shape[1]),
        "grand_mean": grand,
        "construction_marginal_sample_sd": float(
            construction_margin.std(ddof=1)),
        "pipeline_marginal_sample_sd": float(pipeline_margin.std(ddof=1)),
        "rms_interaction": float(np.sqrt((interaction ** 2).mean())),
        "ss_construction": ss_c, "ss_pipeline": ss_p, "ss_interaction": ss_i,
        "ss_total": ss_total,
        "share_construction": ss_c / ss_total if ss_total else np.nan,
        "share_pipeline": ss_p / ss_total if ss_total else np.nan,
        "share_interaction": ss_i / ss_total if ss_total else np.nan,
        "interpretation": "descriptive finite-grid dispersion; not a "
                          "random-effects variance component",
    }])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze(scenario: pd.DataFrame) -> dict[str, Any]:
    """Full registered analysis over an already-scored scenario table."""
    deltas = build_augmentation_deltas(scenario)
    per_pipeline, per_construction = build_effects_by_construction(deltas)
    decomposition = build_decomposition(per_construction, effect="P")
    primary_rows = decomposition[decomposition["role"] == "primary"]
    closure = assert_decomposition_closure(decomposition)

    theta = primary_rows["theta_rule_minus_placebo"].to_numpy(dtype=float)
    if len(theta) != len(e16gen.CONSTRUCTION_SEEDS):
        raise _stop(
            f"primary has {len(theta)} constructions, registered "
            f"{len(e16gen.CONSTRUCTION_SEEDS)}")
    decision = primary_decision(theta)
    share = relational_share(primary_rows)
    p1_table, p1_summary = build_p1_cell_comparison(scenario)

    exact_family = build_secondary_family(
        per_construction, effects=SECONDARY_EXACT_FAMILY,
        label="exact_rule_minus_placebo_secondary",
        endpoint=PRIMARY_ENDPOINT)

    binary_deltas = build_augmentation_deltas(
        scenario, endpoint=COMPANION_ENDPOINT)
    _, binary_per_construction = build_effects_by_construction(binary_deltas)
    binary_family = build_secondary_family(
        binary_per_construction, effects=EFFECTS,
        label="binary_rule_minus_placebo_companion",
        endpoint=COMPANION_ENDPOINT)

    dispersion = build_crossed_dispersion(per_pipeline)

    return {
        "decomposition": decomposition,
        "per_pipeline": per_pipeline,
        "per_construction": per_construction,
        "decision": decision,
        "closure_gate": closure,
        "relational_share": share,
        "p1_table": p1_table,
        "p1_summary": p1_summary,
        "secondary_exact": exact_family,
        "secondary_binary": binary_family,
        "dispersion": dispersion,
        "marginal_component": build_arm_summary(
            per_construction, arm=PLACEBO_ARM),
        "total_response": build_arm_summary(per_construction, arm=RULE_ARM),
    }


def _load_scenario() -> pd.DataFrame:
    path = TABLE_DIR / f"e16_by_scenario_{OUTPUT_VERSION}.csv"
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
    verdict_record = {
        "schema_version": ANALYSIS_SCHEMA,
        "stage": "e16_registered_analysis",
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "registration": e16gen.assert_registration_committed(REPO),
        "environment": _environment_record(REPO),
        "source": _source_record(REPO),
        "closure_gate": result["closure_gate"],
        "primary": decision,
        "relational_share": result["relational_share"],
        "p1_cell_summary": result["p1_summary"],
        "marginal_exposure_component": result["marginal_component"],
        "total_augmentation_response": result["total_response"],
        "scientific_verdict": decision["verdict"],
        "status": "analysis_complete",
    }

    staged = []
    for key, frame_key in (
            ("decomposition", "decomposition"),
            ("p1_cell", "p1_table"),
            ("dispersion", "dispersion"),
    ):
        frame = result[frame_key]
        staged.append((_stage_bytes(paths[key],
                                    frame.to_csv(index=False).encode()),
                       paths[key]))
    secondary = pd.concat(
        [result["secondary_exact"], result["secondary_binary"]],
        ignore_index=True)
    staged.append((_stage_bytes(paths["secondary"],
                                secondary.to_csv(index=False).encode()),
                   paths["secondary"]))
    primary_frame = pd.DataFrame([decision])
    staged.append((_stage_bytes(paths["primary"],
                                primary_frame.to_csv(index=False).encode()),
                   paths["primary"]))
    bridge = result["decomposition"][
        result["decomposition"]["role"] != "primary"]
    staged.append((_stage_bytes(paths["bridge"],
                                bridge.to_csv(index=False).encode()),
                   paths["bridge"]))
    staged.append((_stage_json(paths["verdict"], verdict_record),
                   paths["verdict"]))
    publish_and_cleanup(staged)
    return verdict_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="run the registered analysis and publish outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({
            "stage": "e16_registered_analysis",
            "mode": "read-only",
            "delta": DELTA,
            "sesoi_multiplier": SESOI_MULTIPLIER,
            "sesoi_denominator": SESOI_DENOMINATOR,
            "equivalence_confidence": EQUIVALENCE_CONFIDENCE,
            "heterogeneity_min_outside": HETEROGENEITY_MIN_OUTSIDE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "outputs": {k: str(v.relative_to(REPO))
                        for k, v in output_paths().items()},
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
