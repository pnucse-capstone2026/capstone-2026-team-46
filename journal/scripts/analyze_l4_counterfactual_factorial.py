#!/usr/bin/env python3
"""Analyze the preregistered E14 matched L4 counterfactual factorial.

This module is intentionally model-free.  It consumes the complete tables
published by the E14 evaluator, validates their Cartesian products and paired
contrasts, computes the fixed factorial estimands, and atomically publishes
only the effects table and the final artifact manifest.

The implementation follows
``journal/experiments/e14_l4_counterfactual_factorial/PREREG.md`` at commit
``686f4aa``.  In particular, detector seeds are the inferential units; blocks
are averaged within seed and are otherwise descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "journal"
TABLES = JOURNAL / "results" / "tables"
LOGS = JOURNAL / "results" / "logs"
EXPERIMENT = JOURNAL / "experiments" / "e14_l4_counterfactual_factorial"

ARMS = (
    "real_ms",
    "rule_ms",
    "placebo_ms",
    "real_std",
    "rule_std",
    "placebo_std",
)
SEEDS = (7, 42, 123, 2026, 3407)
BLOCKS = ("block_01", "block_02", "block_03")
ATTACKS = ("Gear", "RPM")
ID_LEVELS = ("canonical", "shifted")
BITS = (0, 1)

CONTRASTS = (
    "rule_ms-real_ms",
    "placebo_ms-real_ms",
    "rule_ms-placebo_ms",
    "rule_std-real_std",
    "placebo_std-real_std",
    "rule_std-placebo_std",
)
CONTRAST_ARMS = {
    "rule_ms-real_ms": ("rule_ms", "real_ms"),
    "placebo_ms-real_ms": ("placebo_ms", "real_ms"),
    "rule_ms-placebo_ms": ("rule_ms", "placebo_ms"),
    "rule_std-real_std": ("rule_std", "real_std"),
    "placebo_std-real_std": ("placebo_std", "real_std"),
    "rule_std-placebo_std": ("rule_std", "placebo_std"),
}

ENDPOINTS = (
    "exact_recall",
    "binary_recall",
    "exact_standardized_margin_shift",
    "binary_standardized_margin_shift",
)
MARGIN_ENDPOINTS = frozenset({
    "exact_standardized_margin_shift",
    "binary_standardized_margin_shift",
})
ID_SCOPES = ("canonical", "shifted", "canonical_minus_shifted")
ATTACK_SCOPES = ("Gear", "RPM", "equal_macro")
EFFECTS = (
    "P",
    "S",
    "D",
    "PS",
    "PD",
    "SD",
    "PSD",
    "corner_111_minus_000",
)
FACTORIAL_EFFECTS = EFFECTS[:7]

NORMAL_KEY = ("arm", "seed", "block_id")
SCENARIO_KEY = (
    "arm", "seed", "block_id", "attack", "id_stratum", "P", "S", "D",
)
CELL_KEY = ("arm", "seed", "block_id", "id_stratum", "P", "S", "D")
DELTA_KEY = (
    "contrast", "seed", "block_id", "attack", "id_stratum", "P", "S", "D",
)
EFFECT_KEY = (
    "contrast", "endpoint", "id_scope", "attack_scope", "effect",
    "row_type", "unit_id",
)

EXPECTED_NORMAL_ROWS = 90
EXPECTED_SCENARIO_ROWS = 2_880
EXPECTED_PRIMARY_SCENARIO_ROWS = 1_440
EXPECTED_CELL_ROWS = 1_440
EXPECTED_DELTA_ROWS = 2_880
EXPECTED_EFFECT_ROWS = 15_552

PRIMARY_HOLM_FAMILIES = {
    "primary_exact_rule_minus_real": (
        "rule_ms-real_ms", "exact_recall",
    ),
    "secondary_exact_rule_minus_placebo": (
        "rule_ms-placebo_ms", "exact_recall",
    ),
    "companion_binary_rule_minus_real": (
        "rule_ms-real_ms", "binary_recall",
    ),
}

DEFAULT_NORMAL = TABLES / "e14_l4_factorial_normal_by_seed_v1.csv"
DEFAULT_SCENARIO = TABLES / "e14_l4_factorial_by_scenario_v1.csv"
DEFAULT_CELL = TABLES / "e14_l4_factorial_by_cell_v1.csv"
DEFAULT_DELTA = TABLES / "e14_l4_factorial_augmentation_delta_v1.csv"
DEFAULT_EFFECTS = TABLES / "e14_l4_factorial_effects_v1.csv"
DEFAULT_MANIFEST = LOGS / "e14_l4_factorial_artifact_manifest_v1.json"

PREREG = EXPERIMENT / "PREREG.md"
AMENDMENT = EXPERIMENT / "IMPLEMENTATION_AMENDMENT_2026-07-24.md"
TRANSCRIPT_AMENDMENT = (
    EXPERIMENT
    / "IMPLEMENTATION_AMENDMENT_2026-07-24_STAGE_A_TRANSCRIPT.md"
)
POST_TRAINING_AMENDMENT = TRANSCRIPT_AMENDMENT
TRAINING_WRAPPER = JOURNAL / "scripts" / "run_e14_matched_real_training.py"
GENERIC_TRAINER = JOURNAL / "scripts" / "train_generator_extension_cnn.py"
EVALUATOR = JOURNAL / "scripts" / "evaluate_l4_counterfactual_factorial.py"
ANALYZER = Path(__file__).resolve()
E14_TESTS = JOURNAL / "tests" / "test_l4_counterfactual_factorial.py"
PREPARE_RECORD = EXPERIMENT / "prepare_v1.json"
TRAINING_PREFLIGHT = EXPERIMENT / "matched_real_training_preflight_v1.json"
TRAINING_RUN = EXPERIMENT / "matched_real_training_run_v1.json"
MATCHED_REAL_TRANSCRIPT = (
    EXPERIMENT / "matched_real_training_child_output_v1.log"
)
BASE_MANIFEST = TABLES / "e14_l4_factorial_base_manifest_v1.csv"
LATENT_MANIFEST = TABLES / "e14_l4_factorial_latent_manifest_v1.csv"
MANIPULATION_CHECKS = TABLES / "e14_l4_factorial_manipulation_checks_v1.csv"
SCORE_LOG = LOGS / "e14_l4_factorial_v1.log"
SCORE_RUN = EXPERIMENT / "run_v1.json"

PREREG_SHA256 = (
    "f0465fea952b9f7d5fd1b20a7864a96192099b9d3346f6c81e4274ce0fe0424a"
)
AMENDMENT_SHA256 = (
    "f6ecf1c56a08ef9d6fc02f3a2f9f9277b6c4e7cf4c972dd4314bbcc4bb431b62"
)
TRANSCRIPT_AMENDMENT_SHA256 = (
    "bae44420c2d8745363b8baf8cb0d0d1652b6bea9fc980b9b32196be41b445b54"
)
POST_TRAINING_AMENDMENT_SHA256 = TRANSCRIPT_AMENDMENT_SHA256
PREREG_COMMIT = "686f4aaae8d35e1b807480c56abd02ead7a13c27"
TRAINING_SOURCE_COMMIT = "ef7afd2da361718c3a6fe4ad75cd82068430c61c"
TRANSCRIPT_AMENDMENT_COMMIT = "ee21d150277fe83b0bcbc188829920de164b2b01"
AMENDMENT_SOURCE_COMMIT = TRANSCRIPT_AMENDMENT_COMMIT
TRAINING_PREFLIGHT_SHA256 = (
    "3b895fe3c84e5188968fedf0ef0450f2ace03c98e3fe1e3ea11aa3a8f98e1e14"
)
TRAINING_RUN_SHA256 = (
    "53865949ecf4383fa6f9a21c224b76c289baeb9c88daa7fbf80eddfacfe0de28"
)
MATCHED_REAL_TRANSCRIPT_SHA256 = (
    "e362b71ca4cdff7755e27b16ca614ca2f1e574c12c8fc95777aeb095a631e0e5"
)
MATCHED_REAL_TRANSCRIPT_FORMAT = (
    "e14.matched_real_training_child_output.v1"
)
MATCHED_REAL_RECORD_HASHES = {
    TRAINING_PREFLIGHT: TRAINING_PREFLIGHT_SHA256,
    TRAINING_RUN: TRAINING_RUN_SHA256,
    MATCHED_REAL_TRANSCRIPT: MATCHED_REAL_TRANSCRIPT_SHA256,
}
MATCHED_REAL_RECORD_BYTES = {
    TRAINING_PREFLIGHT: 21_223,
    TRAINING_RUN: 37_000,
    MATCHED_REAL_TRANSCRIPT: 4_977,
}
MATCHED_REAL_TRAINING_RECORD_KEYS = frozenset(
    {"preflight", "run", "child_transcript"}
)
TRANSFORM_SERIALIZATION_VERSION = "e14-transform-instance-le-f4-v1"
IMPLEMENTATION_SOURCES = {
    "training_wrapper": TRAINING_WRAPPER,
    "evaluator": EVALUATOR,
    "analyzer": ANALYZER,
    "tests": E14_TESTS,
    "amendment": AMENDMENT,
    "transcript_amendment": TRANSCRIPT_AMENDMENT,
}
HISTORICAL_TRAINING_SOURCE_PATHS = frozenset({
    _path.as_posix()
    for _path in (
        PREREG.relative_to(ROOT),
        TRAINING_WRAPPER.relative_to(ROOT),
        EVALUATOR.relative_to(ROOT),
        ANALYZER.relative_to(ROOT),
        E14_TESTS.relative_to(ROOT),
        AMENDMENT.relative_to(ROOT),
    )
})
TRAINING_TO_AMENDMENT_CHANGES = (
    {
        "status": "A",
        "path": TRANSCRIPT_AMENDMENT.relative_to(ROOT).as_posix(),
    },
)
TRAINING_TO_AMENDMENT_PATHS = tuple(
    record["path"] for record in TRAINING_TO_AMENDMENT_CHANGES
)
AMENDMENT_TO_PREPARE_CHANGES = tuple(
    {
        "status": "M",
        "path": path.relative_to(ROOT).as_posix(),
    }
    for path in (ANALYZER, EVALUATOR, E14_TESTS)
)
AMENDMENT_TO_PREPARE_PATHS = tuple(
    record["path"] for record in AMENDMENT_TO_PREPARE_CHANGES
)
PREPARE_OUTPUTS = {
    "base_manifest": BASE_MANIFEST,
    "latent_manifest": LATENT_MANIFEST,
    "manipulation_checks": MANIPULATION_CHECKS,
}
RECORDED_SCORE_OUTPUTS = {
    "normal_by_seed": DEFAULT_NORMAL,
    "by_cell": DEFAULT_CELL,
    "by_scenario": DEFAULT_SCENARIO,
    "augmentation_delta": DEFAULT_DELTA,
    "log": SCORE_LOG,
}
CANONICAL_SCORE_OUTPUTS = (*RECORDED_SCORE_OUTPUTS.values(), SCORE_RUN)
CANONICAL_MANIFEST_INPUTS = (
    PREREG,
    AMENDMENT,
    TRANSCRIPT_AMENDMENT,
    TRAINING_WRAPPER,
    EVALUATOR,
    ANALYZER,
    E14_TESTS,
    BASE_MANIFEST,
    LATENT_MANIFEST,
    MANIPULATION_CHECKS,
    PREPARE_RECORD,
    TRAINING_PREFLIGHT,
    TRAINING_RUN,
    MATCHED_REAL_TRANSCRIPT,
    DEFAULT_NORMAL,
    DEFAULT_CELL,
    DEFAULT_SCENARIO,
    DEFAULT_DELTA,
    SCORE_LOG,
    SCORE_RUN,
)


def _empty(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_int_column(frame: pd.DataFrame, column: str) -> None:
    numeric = pd.to_numeric(frame[column], errors="raise")
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{column} contains nonfinite values")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{column} must contain integers")
    frame[column] = numeric.astype(np.int64)


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], label: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _require_finite(
    frame: pd.DataFrame, columns: Iterable[str], label: str,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} contains missing/nonfinite values")


def _require_unique(
    frame: pd.DataFrame, key: Sequence[str], expected: int, label: str,
) -> None:
    if len(frame) != expected:
        raise ValueError(
            f"{label} row count must be {expected}, observed {len(frame)}")
    duplicated = frame.duplicated(list(key), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(key)].head(8).to_dict("records")
        raise ValueError(f"{label} has duplicate keys: {examples}")
    if frame[list(key)].isna().any().any():
        raise ValueError(f"{label} key contains missing values")


def _expected_index(
    key: Sequence[str], levels: Mapping[str, Sequence[object]],
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [levels[column] for column in key], names=list(key))


def _require_cartesian(
    frame: pd.DataFrame,
    key: Sequence[str],
    levels: Mapping[str, Sequence[object]],
    label: str,
) -> None:
    observed = pd.MultiIndex.from_frame(frame[list(key)])
    expected = _expected_index(key, levels)
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"{label} is not the fixed Cartesian product; "
            f"missing={list(missing[:5])}, extra={list(extra[:5])}")


def _status_column(frame: pd.DataFrame, endpoint: str) -> str | None:
    candidates = [f"{endpoint}_status"]
    if endpoint.startswith("exact_standardized_margin"):
        candidates.append("exact_standardized_margin_status")
    if endpoint.startswith("binary_standardized_margin"):
        candidates.append("binary_standardized_margin_status")
    for old, new in (
        ("_standardized_margin_shift", "_margin_status"),
        ("_shift", "_status"),
    ):
        candidate = endpoint.replace(old, new)
        if candidate != endpoint:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _canonicalize_endpoint_columns(
    frame: pd.DataFrame, *, table_kind: str,
) -> pd.DataFrame:
    """Map the evaluator's explicit derived-column names to analysis names."""
    result = frame.copy()
    if table_kind == "cell":
        aliases = {
            f"{endpoint}_equal_macro": endpoint for endpoint in ENDPOINTS
        }
    elif table_kind == "delta":
        aliases = {f"{endpoint}_delta": endpoint for endpoint in ENDPOINTS}
    else:
        aliases = {}
    for source, destination in aliases.items():
        if destination in result.columns and source in result.columns:
            left = pd.to_numeric(result[destination], errors="coerce")
            right = pd.to_numeric(result[source], errors="coerce")
            equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            if not equal.all():
                raise ValueError(
                    f"{table_kind} contains conflicting {source} and "
                    f"{destination} columns")
        elif destination not in result.columns and source in result.columns:
            result = result.rename(columns={source: destination})
    return result


def endpoint_available(
    row: pd.Series, endpoint: str, status_column: str | None = None,
) -> bool:
    value = pd.to_numeric(pd.Series([row.get(endpoint)]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return False
    if status_column is None:
        return True
    status = str(row.get(status_column, "")).strip().casefold()
    return status in {
        "", "available", "ok", "pass", "valid", "complete", "t-pass",
    }


def _normalize_tables(
    normal: pd.DataFrame,
    scenario: pd.DataFrame,
    cell: pd.DataFrame,
    delta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [normal.copy(), scenario.copy(), cell.copy(), delta.copy()]
    normal, scenario, cell, delta = frames
    cell = _canonicalize_endpoint_columns(cell, table_kind="cell")
    delta = _canonicalize_endpoint_columns(delta, table_kind="delta")
    for frame, key in (
        (normal, NORMAL_KEY),
        (scenario, SCENARIO_KEY),
        (cell, CELL_KEY),
        (delta, DELTA_KEY),
    ):
        _require_columns(frame, key, "input")
        for column in ("seed", "P", "S", "D"):
            if column in key:
                _as_int_column(frame, column)
        for column in key:
            if column not in {"seed", "P", "S", "D"}:
                frame[column] = frame[column].astype(str)
    return normal, scenario, cell, delta


def validate_complete_tables(
    normal: pd.DataFrame,
    scenario: pd.DataFrame,
    cell: pd.DataFrame,
    delta: pd.DataFrame,
    *,
    atol: float = 1e-10,
) -> dict[str, int]:
    """Validate every preregistered evaluator-table Cartesian product."""
    normal, scenario, cell, delta = _normalize_tables(
        normal, scenario, cell, delta)

    _require_unique(normal, NORMAL_KEY, EXPECTED_NORMAL_ROWS, "normal")
    _require_cartesian(normal, NORMAL_KEY, {
        "arm": ARMS, "seed": SEEDS, "block_id": BLOCKS,
    }, "normal")
    _require_columns(normal, ("normal_recall", "fpr"), "normal")
    _require_finite(normal, ("normal_recall", "fpr"), "normal")
    if "n" not in normal:
        raise ValueError("normal is missing n")
    _as_int_column(normal, "n")
    if not normal["n"].eq(2_000).all():
        raise ValueError("every normal-control row must contain n=2000")
    if not np.allclose(
        normal["normal_recall"].astype(float)
        + normal["fpr"].astype(float),
        1.0,
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError("normal_recall + fpr must equal one")
    for column in ("normal_recall", "fpr"):
        values = normal[column].to_numpy(dtype=float)
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"normal.{column} must lie in [0,1]")

    _require_unique(
        scenario, SCENARIO_KEY, EXPECTED_SCENARIO_ROWS, "scenario")
    _require_cartesian(scenario, SCENARIO_KEY, {
        "arm": ARMS,
        "seed": SEEDS,
        "block_id": BLOCKS,
        "attack": ATTACKS,
        "id_stratum": ID_LEVELS,
        "P": BITS,
        "S": BITS,
        "D": BITS,
    }, "scenario")
    _require_columns(scenario, ("n", *ENDPOINTS), "scenario")
    _as_int_column(scenario, "n")
    if not scenario["n"].eq(2_000).all():
        raise ValueError("every scenario row must contain n=2000 bases")
    _require_finite(scenario, ENDPOINTS[:2], "scenario")
    for endpoint in ENDPOINTS[:2]:
        values = scenario[endpoint].to_numpy(dtype=float)
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"scenario.{endpoint} must lie in [0,1]")
    for endpoint in MARGIN_ENDPOINTS:
        status_column = _status_column(scenario, endpoint)
        if status_column is None:
            raise ValueError(f"scenario is missing the status for {endpoint}")
        statuses = set(scenario[status_column].astype(str))
        allowed = {"ok", "no clean-margin dynamic range"}
        if not statuses.issubset(allowed):
            raise ValueError(
                f"scenario.{status_column} has unsupported values: "
                f"{sorted(statuses - allowed)}")
        ok = scenario[status_column].eq("ok")
        numeric = pd.to_numeric(scenario[endpoint], errors="coerce")
        if not np.isfinite(numeric[ok]).all() or numeric[~ok].notna().any():
            raise ValueError(
                f"scenario.{endpoint} availability disagrees with status")
    primary = scenario[scenario["arm"].isin(ARMS[:3])]
    if len(primary) != EXPECTED_PRIMARY_SCENARIO_ROWS:
        raise ValueError(
            "primary matched-budget scenario rows must total "
            f"{EXPECTED_PRIMARY_SCENARIO_ROWS}")
    if len(primary[primary["id_stratum"].eq("canonical")]) != 720:
        raise ValueError("primary canonical-ID scenario rows must total 720")

    _require_unique(cell, CELL_KEY, EXPECTED_CELL_ROWS, "cell")
    _require_cartesian(cell, CELL_KEY, {
        "arm": ARMS,
        "seed": SEEDS,
        "block_id": BLOCKS,
        "id_stratum": ID_LEVELS,
        "P": BITS,
        "S": BITS,
        "D": BITS,
    }, "cell")
    _require_columns(cell, ENDPOINTS, "cell")
    _require_finite(cell, ENDPOINTS[:2], "cell")
    for endpoint in ENDPOINTS[:2]:
        values = cell[endpoint].to_numpy(dtype=float)
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"cell.{endpoint} must lie in [0,1]")

    _require_unique(delta, DELTA_KEY, EXPECTED_DELTA_ROWS, "delta")
    _require_cartesian(delta, DELTA_KEY, {
        "contrast": CONTRASTS,
        "seed": SEEDS,
        "block_id": BLOCKS,
        "attack": ATTACKS,
        "id_stratum": ID_LEVELS,
        "P": BITS,
        "S": BITS,
        "D": BITS,
    }, "delta")
    _require_columns(delta, ENDPOINTS, "delta")
    _require_columns(delta, ("n",), "delta")
    _as_int_column(delta, "n")
    if not delta["n"].eq(2_000).all():
        raise ValueError("every augmentation-delta row must contain n=2000")
    _require_finite(delta, ENDPOINTS[:2], "delta")

    # Verify the evaluator's attack-equal macro table for the two recall
    # endpoints.  Margin rows can legitimately be unavailable by seed.
    macro = (
        scenario.groupby(list(CELL_KEY), sort=False)[list(ENDPOINTS[:2])]
        .mean()
        .reset_index()
    )
    checked_cell = cell.merge(
        macro,
        on=list(CELL_KEY),
        how="left",
        validate="one_to_one",
        suffixes=("", "_recomputed"),
    )
    for endpoint in ENDPOINTS[:2]:
        if not np.allclose(
            checked_cell[endpoint].astype(float),
            checked_cell[f"{endpoint}_recomputed"].astype(float),
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(f"cell {endpoint} is not the attack-equal macro")

    for endpoint in MARGIN_ENDPOINTS:
        scenario_status = _status_column(scenario, endpoint)
        cell_status = _status_column(cell, endpoint)
        if scenario_status is None or cell_status is None:
            raise ValueError(f"missing margin status for cell {endpoint}")
        grouped = scenario.groupby(list(CELL_KEY), sort=False)
        expected = grouped[endpoint].mean().rename(
            f"{endpoint}_recomputed").reset_index()
        expected_status = grouped[scenario_status].agg(
            lambda values: (
                "ok" if set(values.astype(str)) == {"ok"}
                else "no clean-margin dynamic range"
            )
        ).rename(f"{cell_status}_recomputed").reset_index()
        checked = (
            cell.merge(expected, on=list(CELL_KEY), validate="one_to_one")
            .merge(expected_status, on=list(CELL_KEY), validate="one_to_one")
        )
        status_matches = (
            checked[cell_status].astype(str)
            == checked[f"{cell_status}_recomputed"].astype(str)
        )
        if not status_matches.all():
            raise ValueError(f"cell {endpoint} status is not attack-equal")
        ok = checked[cell_status].eq("ok")
        if not np.allclose(
            checked.loc[ok, endpoint].astype(float),
            checked.loc[ok, f"{endpoint}_recomputed"].astype(float),
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(f"cell {endpoint} is not the attack-equal macro")
        if checked.loc[~ok, endpoint].notna().any():
            raise ValueError(f"cell unavailable {endpoint} must remain NA")

    # Verify all six same-seed scenario contrasts for recall endpoints.
    indexed = scenario.set_index(list(SCENARIO_KEY))
    for contrast, (arm_a, arm_b) in CONTRAST_ARMS.items():
        observed = delta[delta["contrast"].eq(contrast)].copy()
        base_key = list(DELTA_KEY[1:])
        a_key = pd.MultiIndex.from_frame(
            observed.assign(arm=arm_a)[
                ["arm", *base_key]
            ].rename(columns={"block_id": "block_id"})
        )
        b_key = pd.MultiIndex.from_frame(
            observed.assign(arm=arm_b)[["arm", *base_key]]
        )
        upper_rows = indexed.loc[a_key]
        lower_rows = indexed.loc[b_key]
        for endpoint in ENDPOINTS[:2]:
            expected_values = (
                upper_rows[endpoint].to_numpy(dtype=float)
                - lower_rows[endpoint].to_numpy(dtype=float)
            )
            if not np.allclose(
                observed[endpoint].to_numpy(dtype=float),
                expected_values,
                rtol=0.0,
                atol=atol,
            ):
                raise ValueError(
                    f"delta {contrast}/{endpoint} does not equal paired arms")
        for endpoint in MARGIN_ENDPOINTS:
            scenario_status = _status_column(scenario, endpoint)
            delta_status = _status_column(delta, endpoint)
            if scenario_status is None or delta_status is None:
                raise ValueError(f"missing delta status for {endpoint}")
            expected_ok = (
                upper_rows[scenario_status].astype(str).eq("ok").to_numpy()
                & lower_rows[scenario_status].astype(str).eq("ok").to_numpy()
            )
            expected_status = np.where(
                expected_ok, "ok", "no clean-margin dynamic range")
            observed_status = observed[delta_status].astype(str).to_numpy()
            if not np.array_equal(expected_status, observed_status):
                raise ValueError(
                    f"delta {contrast}/{endpoint} status is not paired")
            observed_values = pd.to_numeric(
                observed[endpoint], errors="coerce").to_numpy(dtype=float)
            expected_values = (
                upper_rows[endpoint].to_numpy(dtype=float)
                - lower_rows[endpoint].to_numpy(dtype=float)
            )
            if not np.allclose(
                observed_values[expected_ok],
                expected_values[expected_ok],
                rtol=0.0,
                atol=atol,
            ):
                raise ValueError(
                    f"delta {contrast}/{endpoint} does not equal paired arms")
            if np.isfinite(observed_values[~expected_ok]).any():
                raise ValueError(
                    f"delta unavailable {contrast}/{endpoint} must remain NA")

    return {
        "normal_rows": len(normal),
        "scenario_rows": len(scenario),
        "primary_scenario_rows": len(primary),
        "primary_canonical_scenario_rows": len(
            primary[primary["id_stratum"].eq("canonical")]),
        "cell_rows": len(cell),
        "delta_rows": len(delta),
        "checkpoint_evaluations": len(ARMS) * len(SEEDS),
        "transformed_forward_evaluations": int(scenario["n"].sum()),
        "clean_forward_evaluations": int(normal["n"].sum()),
    }


def factorial_coefficients(effect: str) -> dict[tuple[int, int, int], float]:
    """Return the exact preregistered linear operator for one effect."""
    if effect not in EFFECTS:
        raise ValueError(f"unknown factorial effect: {effect}")
    cells = list(itertools.product(BITS, repeat=3))
    if effect == "P":
        return {c: (1.0 if c[0] else -1.0) / 4.0 for c in cells}
    if effect == "S":
        return {c: (1.0 if c[1] else -1.0) / 4.0 for c in cells}
    if effect == "D":
        return {c: (1.0 if c[2] else -1.0) / 4.0 for c in cells}
    if effect == "PS":
        return {
            c: ((1.0 if c[0] else -1.0)
                * (1.0 if c[1] else -1.0)) / 2.0
            for c in cells
        }
    if effect == "PD":
        return {
            c: ((1.0 if c[0] else -1.0)
                * (1.0 if c[2] else -1.0)) / 2.0
            for c in cells
        }
    if effect == "SD":
        return {
            c: ((1.0 if c[1] else -1.0)
                * (1.0 if c[2] else -1.0)) / 2.0
            for c in cells
        }
    if effect == "PSD":
        return {
            c: ((1.0 if c[0] else -1.0)
                * (1.0 if c[1] else -1.0)
                * (1.0 if c[2] else -1.0))
            for c in cells
        }
    return {
        c: 1.0 if c == (1, 1, 1) else (-1.0 if c == (0, 0, 0) else 0.0)
        for c in cells
    }


def factorial_effect(
    values: Mapping[tuple[int, int, int], float],
    effect: str,
) -> float:
    """Apply a fixed E14 factorial operator to all eight cells."""
    expected = set(itertools.product(BITS, repeat=3))
    if set(values) != expected:
        raise ValueError(
            f"factorial values must contain exactly eight cells; "
            f"missing={sorted(expected - set(values))}, "
            f"extra={sorted(set(values) - expected)}")
    array = np.asarray([values[cell] for cell in sorted(expected)], dtype=float)
    if not np.isfinite(array).all():
        return float("nan")
    coefficients = factorial_coefficients(effect)
    return float(sum(coefficients[cell] * float(values[cell])
                     for cell in expected))


def t_summary(values: Sequence[float]) -> dict[str, object]:
    """Summarize five seed effects under the fixed degeneracy rule."""
    array = np.asarray(values, dtype=float)
    available = array[np.isfinite(array)]
    result: dict[str, object] = {
        "n_available": int(len(available)),
        "mean": float(np.mean(available)) if len(available) else float("nan"),
        "sd": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "test_p": float("nan"),
        "holm_input_p": float("nan"),
        "p_status": "not_tested",
        "degenerate": False,
        "strict_positive": int((available > 0).sum()),
        "strict_negative": int((available < 0).sum()),
    }
    if len(available) != len(array):
        result["status"] = "margin_n_available_only"
        return result
    if len(array) != 5:
        raise ValueError(f"seed summary requires five values, got {len(array)}")
    if np.all(array == array[0]):
        result.update({
            "status": "degenerate_variance",
            "degenerate": True,
            "holm_input_p": 1.0,
            "p_status": "bookkeeping_p",
        })
        return result
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=4))
    half = critical * sd / math.sqrt(5.0)
    statistic = mean / (sd / math.sqrt(5.0))
    result.update({
        "status": "complete",
        "sd": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "test_p": float(2.0 * student_t.sf(abs(statistic), df=4)),
        "holm_input_p": float(2.0 * student_t.sf(abs(statistic), df=4)),
        "p_status": "inferential_p",
    })
    return result


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Holm step-down adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Holm inputs must be a finite one-dimensional array")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Holm inputs must lie in [0,1]")
    order = np.argsort(p, kind="mergesort")
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p) - rank) * p[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _effect_from_subset(
    subset: pd.DataFrame,
    endpoint: str,
    id_scope: str,
    attack_scope: str,
    effect: str,
) -> tuple[float, str]:
    attacks = ATTACKS if attack_scope == "equal_macro" else (attack_scope,)
    ids = (
        ID_LEVELS if id_scope == "canonical_minus_shifted"
        else (id_scope,)
    )
    status_column = _status_column(subset, endpoint)
    attack_values = []
    for attack in attacks:
        id_values: dict[str, float] = {}
        for identity in ids:
            selected = subset[
                subset["attack"].eq(attack)
                & subset["id_stratum"].eq(identity)
            ]
            if len(selected) != 8:
                raise ValueError(
                    f"effect input must have eight cells for "
                    f"{attack}/{identity}; got {len(selected)}")
            values = {}
            available = True
            for row in selected.itertuples(index=False):
                series = pd.Series(row._asdict())
                cell = (int(series["P"]), int(series["S"]), int(series["D"]))
                values[cell] = float(series[endpoint])
                available &= endpoint_available(
                    series, endpoint, status_column)
            if not available:
                return float("nan"), "unavailable_margin_input"
            id_values[identity] = factorial_effect(values, effect)
        value = (
            id_values["canonical"] - id_values["shifted"]
            if id_scope == "canonical_minus_shifted"
            else id_values[id_scope]
        )
        attack_values.append(value)
    return float(np.mean(attack_values)), "available"


def build_effect_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed 15,552-row E14 effect table from complete deltas."""
    delta = _canonicalize_endpoint_columns(delta, table_kind="delta")
    _require_unique(delta, DELTA_KEY, EXPECTED_DELTA_ROWS, "delta")
    _require_columns(delta, (*DELTA_KEY, *ENDPOINTS), "delta")
    for column in ("seed", "P", "S", "D"):
        _as_int_column(delta, column)
    for column in ("contrast", "block_id", "attack", "id_stratum"):
        delta[column] = delta[column].astype(str)

    _require_cartesian(delta, DELTA_KEY, {
        "contrast": CONTRASTS,
        "seed": SEEDS,
        "block_id": BLOCKS,
        "attack": ATTACKS,
        "id_stratum": ID_LEVELS,
        "P": BITS,
        "S": BITS,
        "D": BITS,
    }, "delta")

    # Precompute the eight-cell operators once per atomic
    # contrast/endpoint/seed/block/attack/ID combination.  Besides being much
    # faster than repeatedly filtering a DataFrame, this makes the exact
    # pairing order explicit.
    delta_index = delta.set_index(list(DELTA_KEY))
    if not delta_index.index.is_unique:
        raise ValueError("delta key is not unique after indexing")
    status_columns = {
        endpoint: _status_column(delta, endpoint) for endpoint in ENDPOINTS
    }
    atomic: dict[tuple[object, ...], float] = {}
    for (
        contrast, endpoint, seed, block, attack, identity, effect,
    ) in itertools.product(
        CONTRASTS, ENDPOINTS, SEEDS, BLOCKS, ATTACKS, ID_LEVELS, EFFECTS,
    ):
        values: dict[tuple[int, int, int], float] = {}
        available = True
        status_column = status_columns[endpoint]
        for p, s, d in itertools.product(BITS, repeat=3):
            row = delta_index.loc[
                (contrast, seed, block, attack, identity, p, s, d)
            ]
            value = float(row[endpoint])
            values[(p, s, d)] = value
            available &= endpoint_available(row, endpoint, status_column)
        atomic[
            (contrast, endpoint, seed, block, attack, identity, effect)
        ] = factorial_effect(values, effect) if available else float("nan")

    def scoped_value(
        contrast: str,
        endpoint: str,
        seed: int,
        block: str,
        id_scope: str,
        attack_scope: str,
        effect: str,
    ) -> float:
        selected_attacks = (
            ATTACKS if attack_scope == "equal_macro" else (attack_scope,)
        )
        attack_values = []
        for attack in selected_attacks:
            if id_scope == "canonical_minus_shifted":
                value = (
                    atomic[(
                        contrast, endpoint, seed, block, attack,
                        "canonical", effect,
                    )]
                    - atomic[(
                        contrast, endpoint, seed, block, attack,
                        "shifted", effect,
                    )]
                )
            else:
                value = atomic[(
                    contrast, endpoint, seed, block, attack, id_scope, effect,
                )]
            attack_values.append(value)
        array = np.asarray(attack_values, dtype=float)
        return float(array.mean()) if np.isfinite(array).all() else float("nan")

    rows: list[dict[str, object]] = []
    for contrast, endpoint, id_scope, attack_scope, effect in itertools.product(
        CONTRASTS, ENDPOINTS, ID_SCOPES, ATTACK_SCOPES, EFFECTS,
    ):
        seed_values: dict[int, float] = {}
        for seed in SEEDS:
            block_values = np.asarray([
                scoped_value(
                    contrast, endpoint, seed, block, id_scope, attack_scope,
                    effect,
                )
                for block in BLOCKS
            ], dtype=float)
            available = np.isfinite(block_values)
            value = (
                float(np.mean(block_values))
                if available.all() else float("nan")
            )
            status = (
                "available" if available.all()
                else "unavailable_margin_input"
            )
            seed_values[seed] = value
            rows.append(_effect_row(
                contrast, endpoint, id_scope, attack_scope, effect,
                "seed", str(seed), value, status,
                int(np.isfinite(value)),
            ))

        block_descriptive_values: list[float] = []
        for block in BLOCKS:
            available_values = np.asarray([
                scoped_value(
                    contrast, endpoint, seed, block, id_scope, attack_scope,
                    effect,
                )
                for seed in SEEDS
            ], dtype=float)
            available_values = available_values[np.isfinite(available_values)]
            value = (
                float(available_values.mean())
                if len(available_values) else float("nan")
            )
            status = (
                "descriptive_complete" if len(available_values) == 5
                else "descriptive_n_available"
            )
            block_descriptive_values.append(value)
            rows.append(_effect_row(
                contrast, endpoint, id_scope, attack_scope, effect,
                "block", block, value, status, len(available_values),
            ))

        summary = t_summary([seed_values[seed] for seed in SEEDS])
        row = _effect_row(
            contrast, endpoint, id_scope, attack_scope, effect,
            "summary", "summary", float(summary["mean"]),
            str(summary["status"]), int(summary["n_available"]),
        )
        for column in (
            "mean", "sd", "ci_low", "ci_high", "test_p", "holm_input_p",
            "p_status", "degenerate", "strict_positive", "strict_negative",
        ):
            row[column] = summary[column]
        available_blocks = np.asarray(
            block_descriptive_values, dtype=float)
        available_blocks = available_blocks[np.isfinite(available_blocks)]
        block_mean = (
            float(available_blocks.mean())
            if len(available_blocks) else float("nan")
        )
        row.update({
            "block_n_available": int(len(available_blocks)),
            "block_mean": block_mean,
            "block_sd": (
                float(available_blocks.std(ddof=1))
                if len(available_blocks) >= 2 else float("nan")
            ),
            "block_min": (
                min(float(available_blocks.min()), block_mean)
                if len(available_blocks) else float("nan")
            ),
            "block_max": (
                max(float(available_blocks.max()), block_mean)
                if len(available_blocks) else float("nan")
            ),
        })
        rows.append(row)

    effects = pd.DataFrame(rows)
    _require_unique(effects, EFFECT_KEY, EXPECTED_EFFECT_ROWS, "effects")
    return finalize_effect_inference(effects)


def _effect_row(
    contrast: str,
    endpoint: str,
    id_scope: str,
    attack_scope: str,
    effect: str,
    row_type: str,
    unit_id: str,
    value: float,
    status: str,
    n_available: int,
) -> dict[str, object]:
    return {
        "contrast": contrast,
        "endpoint": endpoint,
        "id_scope": id_scope,
        "attack_scope": attack_scope,
        "effect": effect,
        "row_type": row_type,
        "unit_id": unit_id,
        "value": value,
        "status": status,
        "n_available": n_available,
        "mean": float("nan"),
        "sd": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "block_n_available": float("nan"),
        "block_mean": float("nan"),
        "block_sd": float("nan"),
        "block_min": float("nan"),
        "block_max": float("nan"),
        "raw_p": float("nan"),
        "holm_input_p": float("nan"),
        "holm_adjusted_p": float("nan"),
        "p_status": "not_applicable",
        "degenerate": False,
        "strict_positive": 0,
        "strict_negative": 0,
        "resolved_and_stable": False,
        "holm_family": "",
        "legacy_budget_classification": "",
        "localization_verdict": "",
        "attack_heterogeneous_effects": "",
    }


def _validate_effect_units(effects: pd.DataFrame) -> None:
    expected_units = {
        "seed": {str(seed) for seed in SEEDS},
        "block": set(BLOCKS),
        "summary": {"summary"},
    }
    for row_type, units in expected_units.items():
        selected = effects[effects["row_type"].eq(row_type)]
        expected_count = (
            len(CONTRASTS) * len(ENDPOINTS) * len(ID_SCOPES)
            * len(ATTACK_SCOPES) * len(EFFECTS) * len(units)
        )
        if len(selected) != expected_count:
            raise ValueError(
                f"effects {row_type} rows must total {expected_count}, "
                f"observed {len(selected)}")
        if set(selected["unit_id"].astype(str)) != units:
            raise ValueError(
                f"effects {row_type} unit IDs are not fixed: "
                f"{sorted(set(selected['unit_id'].astype(str)))}")


def finalize_effect_inference(effects: pd.DataFrame) -> pd.DataFrame:
    """Apply the three fixed Holm families, verdict, and budget bridge."""
    effects = effects.copy()
    _validate_effect_units(effects)
    summary_mask = effects["row_type"].eq("summary")
    registered_mask = pd.Series(False, index=effects.index)
    for contrast, endpoint in PRIMARY_HOLM_FAMILIES.values():
        registered_mask |= (
            summary_mask
            & effects["contrast"].eq(contrast)
            & effects["endpoint"].eq(endpoint)
            & effects["id_scope"].eq("canonical")
            & effects["attack_scope"].eq("equal_macro")
            & effects["effect"].isin(FACTORIAL_EFFECTS)
        )
    supporting_mask = summary_mask & ~registered_mask
    effects.loc[
        supporting_mask,
        ["test_p", "holm_input_p", "raw_p", "holm_adjusted_p"],
    ] = np.nan
    effects.loc[supporting_mask, "p_status"] = "not_tested_supporting"

    for family, (contrast, endpoint) in PRIMARY_HOLM_FAMILIES.items():
        selected_mask = (
            summary_mask
            & effects["contrast"].eq(contrast)
            & effects["endpoint"].eq(endpoint)
            & effects["id_scope"].eq("canonical")
            & effects["attack_scope"].eq("equal_macro")
            & effects["effect"].isin(FACTORIAL_EFFECTS)
        )
        selected = effects.loc[selected_mask].copy()
        if len(selected) != 7:
            raise ValueError(f"Holm family {family} must contain seven rows")
        selected = selected.set_index("effect").loc[list(FACTORIAL_EFFECTS)]
        p_inputs = selected["holm_input_p"].astype(float).to_numpy()
        if not np.isfinite(p_inputs).all():
            bad = selected.loc[
                ~np.isfinite(p_inputs),
                ["mean", "sd", "status", "holm_input_p"],
            ].reset_index().to_dict("records")
            raise ValueError(
                f"Holm family {family} has unavailable p inputs: {bad}")
        adjusted = holm_adjust(p_inputs)
        for effect, adjusted_p in zip(FACTORIAL_EFFECTS, adjusted):
            index = effects.index[
                selected_mask & effects["effect"].eq(effect)
            ]
            # ``test_p`` remains NA for a degenerate effect.  ``raw_p`` is
            # the seven-family input and is therefore the explicitly labelled
            # conservative bookkeeping value 1 in that case.
            effects.loc[index, "raw_p"] = effects.loc[
                index, "holm_input_p"
            ]
            effects.loc[index, "holm_adjusted_p"] = adjusted_p
            effects.loc[index, "holm_family"] = family
            stable = _resolved_row(effects.loc[index[0]])
            effects.loc[index, "resolved_and_stable"] = stable

    p_columns = ("test_p", "holm_input_p", "raw_p", "holm_adjusted_p")
    if effects.loc[supporting_mask, list(p_columns)].notna().any().any():
        raise AssertionError("supporting summary rows retained p-values")
    registered = effects.loc[registered_mask]
    if (
        len(registered) != 3 * len(FACTORIAL_EFFECTS)
        or registered["holm_family"].astype(str).eq("").any()
        or registered["holm_input_p"].isna().any()
        or registered["raw_p"].isna().any()
        or registered["holm_adjusted_p"].isna().any()
    ):
        raise AssertionError("registered Holm-family p-value contract failed")

    verdict = localization_verdict(effects)
    heterogeneous = attack_heterogeneous_effects(effects)
    primary_mask = (
        summary_mask
        & effects["contrast"].eq("rule_ms-real_ms")
        & effects["endpoint"].eq("exact_recall")
        & effects["id_scope"].eq("canonical")
        & effects["attack_scope"].eq("equal_macro")
        & effects["effect"].isin(FACTORIAL_EFFECTS)
    )
    effects.loc[primary_mask, "localization_verdict"] = verdict
    effects.loc[
        primary_mask, "attack_heterogeneous_effects"
    ] = ",".join(heterogeneous)
    _apply_legacy_budget_classification(effects)

    _require_unique(effects, EFFECT_KEY, EXPECTED_EFFECT_ROWS, "effects")
    if len(effects) != EXPECTED_EFFECT_ROWS:
        raise AssertionError("effect table row count changed during inference")
    return effects.sort_values(list(EFFECT_KEY), kind="mergesort").reset_index(
        drop=True)


def _resolved_row(row: pd.Series) -> bool:
    if bool(row["degenerate"]):
        return False
    adjusted = float(row["holm_adjusted_p"])
    mean = float(row["mean"])
    if not np.isfinite(adjusted) or adjusted >= 0.05 or mean == 0:
        return False
    if mean > 0:
        return int(row["strict_positive"]) >= 4
    return int(row["strict_negative"]) >= 4


def localization_verdict(effects: pd.DataFrame) -> str:
    """Return the fixed primary exact-family E14 scientific verdict."""
    selected = effects[
        effects["row_type"].eq("summary")
        & effects["contrast"].eq("rule_ms-real_ms")
        & effects["endpoint"].eq("exact_recall")
        & effects["id_scope"].eq("canonical")
        & effects["attack_scope"].eq("equal_macro")
        & effects["effect"].isin(FACTORIAL_EFFECTS)
    ].set_index("effect")
    if set(selected.index) != set(FACTORIAL_EFFECTS):
        raise ValueError("primary exact verdict requires all seven effects")
    resolved = {
        effect for effect, row in selected.iterrows() if _resolved_row(row)
    }
    significant = {
        effect
        for effect, row in selected.iterrows()
        if (not bool(row["degenerate"])
            and np.isfinite(float(row["holm_adjusted_p"]))
            and float(row["holm_adjusted_p"]) < 0.05)
    }
    if resolved & {"P", "S", "D"}:
        verdict = "F-MAIN"
    elif resolved:
        verdict = "F-INTERACTION"
    elif significant:
        verdict = "F-MIXED"
    else:
        verdict = "F-UNRESOLVED"

    degenerate = [
        effect for effect in FACTORIAL_EFFECTS
        if bool(selected.loc[effect, "degenerate"])
    ]
    if degenerate:
        verdict += "+DEGENERATE(" + ",".join(degenerate) + ")"

    if attack_heterogeneous_effects(effects):
        verdict += "+attack-heterogeneous"
    return verdict


def attack_heterogeneous_effects(effects: pd.DataFrame) -> list[str]:
    """List primary exact effects whose Gear/RPM point means oppose."""
    heterogeneous = []
    for effect in FACTORIAL_EFFECTS:
        gear = _summary_mean(
            effects, "rule_ms-real_ms", "exact_recall", "canonical",
            "Gear", effect)
        rpm = _summary_mean(
            effects, "rule_ms-real_ms", "exact_recall", "canonical",
            "RPM", effect)
        if np.isfinite(gear) and np.isfinite(rpm) and gear * rpm < 0:
            heterogeneous.append(effect)
    return heterogeneous


def _summary_mean(
    effects: pd.DataFrame,
    contrast: str,
    endpoint: str,
    id_scope: str,
    attack_scope: str,
    effect: str,
) -> float:
    selected = effects[
        effects["row_type"].eq("summary")
        & effects["contrast"].eq(contrast)
        & effects["endpoint"].eq(endpoint)
        & effects["id_scope"].eq(id_scope)
        & effects["attack_scope"].eq(attack_scope)
        & effects["effect"].eq(effect)
    ]
    if len(selected) != 1:
        raise ValueError("summary effect lookup is not unique")
    return float(selected.iloc[0]["mean"])


def legacy_budget_classification(
    primary_mean: float,
    legacy_mean: float,
    legacy_ci_low: float,
    legacy_ci_high: float,
) -> str:
    """Classify a resolved primary effect in the legacy harness."""
    if legacy_mean == 0:
        return "budget-direction-unresolved-zero"
    if primary_mean == 0 or primary_mean * legacy_mean < 0:
        return "budget-discordant"
    same_positive = primary_mean > 0 and legacy_mean > 0
    same_negative = primary_mean < 0 and legacy_mean < 0
    excludes_same = (
        (same_positive and np.isfinite(legacy_ci_low) and legacy_ci_low > 0)
        or (same_negative and np.isfinite(legacy_ci_high)
            and legacy_ci_high < 0)
    )
    excludes_opposite = (
        (primary_mean > 0 and np.isfinite(legacy_ci_high)
         and legacy_ci_high < 0)
        or (primary_mean < 0 and np.isfinite(legacy_ci_low)
            and legacy_ci_low > 0)
    )
    if excludes_opposite:
        return "budget-discordant"
    if excludes_same:
        return "budget-stable"
    return "directionally-concordant-unresolved"


def _apply_legacy_budget_classification(effects: pd.DataFrame) -> None:
    primary_mask = (
        effects["row_type"].eq("summary")
        & effects["contrast"].eq("rule_ms-real_ms")
        & effects["endpoint"].eq("exact_recall")
        & effects["id_scope"].eq("canonical")
        & effects["attack_scope"].eq("equal_macro")
        & effects["effect"].isin(FACTORIAL_EFFECTS)
        & effects["resolved_and_stable"].eq(True)
    )
    for index, primary in effects.loc[primary_mask].iterrows():
        effect = str(primary["effect"])
        legacy_mask = (
            effects["row_type"].eq("summary")
            & effects["contrast"].eq("rule_std-real_std")
            & effects["endpoint"].eq("exact_recall")
            & effects["id_scope"].eq("canonical")
            & effects["attack_scope"].eq("equal_macro")
            & effects["effect"].eq(effect)
        )
        if legacy_mask.sum() != 1:
            raise ValueError(f"missing legacy effect corresponding to {effect}")
        legacy = effects.loc[legacy_mask].iloc[0]
        classification = legacy_budget_classification(
            float(primary["mean"]),
            float(legacy["mean"]),
            float(legacy["ci_low"]),
            float(legacy["ci_high"]),
        )
        effects.loc[index, "legacy_budget_classification"] = classification
        effects.loc[legacy_mask, "legacy_budget_classification"] = classification


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(ROOT.absolute()).as_posix()
    except ValueError as exc:
        raise ValueError(f"E14 canonical path escapes repository: {path}") from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid E14 JSON record: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"E14 JSON root must be an object: {path}")
    return payload


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _git_text(arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Git provenance command failed: git {' '.join(arguments)}"
        ) from exc


def _current_head() -> str:
    head = _git_text(["rev-parse", "HEAD"]).strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError(f"could not resolve a full Git commit: {head!r}")
    return head


def _require_ancestor(ancestor: str, descendant: str) -> None:
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not validate E14 source ancestry") from exc
    if completed.returncode != 0:
        raise ValueError(
            f"E14 implementation commit {ancestor} is not an ancestor of "
            f"scoring commit {descendant}")


def _git_commit_count(ancestor: str, descendant: str) -> int:
    value = _git_text(
        ["rev-list", "--count", f"{ancestor}..{descendant}"]
    ).strip()
    try:
        count = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"could not count E14 commits {ancestor}..{descendant}"
        ) from exc
    if count < 0 or str(count) != value:
        raise RuntimeError(
            f"invalid E14 commit count {ancestor}..{descendant}: {value!r}")
    return count


def _git_name_status(
    ancestor: str, descendant: str,
) -> tuple[dict[str, str], ...]:
    """Return an exact no-rename Git path transition using NUL framing."""
    try:
        payload = subprocess.run(
            [
                "git", "diff", "--name-status", "--no-renames", "-z",
                ancestor, descendant,
            ],
            cwd=ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"could not inspect E14 transition {ancestor}..{descendant}"
        ) from exc
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise RuntimeError("Git emitted a non-NUL-terminated E14 diff")
    fields = payload[:-1].split(b"\0")
    if len(fields) % 2:
        raise RuntimeError("Git emitted a malformed E14 name-status diff")
    changes: list[dict[str, str]] = []
    for offset in range(0, len(fields), 2):
        try:
            status = fields[offset].decode("ascii")
            path = fields[offset + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Git emitted a nonportable E14 transition path"
            ) from exc
        if status not in {"A", "M", "D"}:
            raise ValueError(
                f"E14 transition contains unsupported status {status!r}")
        if not path or "\n" in path or "\r" in path:
            raise ValueError(f"E14 transition contains invalid path {path!r}")
        changes.append({"status": status, "path": path})
    return tuple(changes)


def _validate_implementation_transition(
    record: object,
    prepare_commit: str,
    *,
    current_head: str | None = None,
) -> dict[str, object]:
    """Validate the only allowed post-training implementation transition."""
    if current_head is None:
        current_head = _current_head()
    transition = _require_mapping(record, "prepare implementation_transition")
    if set(transition) != {
        "schema_version",
        "training_source_commit",
        "amendment_source_commit",
        "prepare_source_commit",
        "training_to_amendment",
        "amendment_to_prepare",
        "ancestry",
    }:
        raise ValueError("prepare implementation-transition field set mismatch")
    if (
        transition.get("schema_version") != "e14.implementation_transition.v1"
        or transition.get("training_source_commit") != TRAINING_SOURCE_COMMIT
        or transition.get("amendment_source_commit")
        != TRANSCRIPT_AMENDMENT_COMMIT
        or transition.get("prepare_source_commit") != prepare_commit
    ):
        raise ValueError("prepare implementation-transition commit mismatch")

    _require_ancestor(TRAINING_SOURCE_COMMIT, TRANSCRIPT_AMENDMENT_COMMIT)
    _require_ancestor(TRANSCRIPT_AMENDMENT_COMMIT, prepare_commit)
    _require_ancestor(prepare_commit, current_head)
    first_count = _git_commit_count(
        TRAINING_SOURCE_COMMIT, TRANSCRIPT_AMENDMENT_COMMIT)
    second_count = _git_commit_count(
        TRANSCRIPT_AMENDMENT_COMMIT, prepare_commit)
    if first_count != 1 or second_count != 1:
        raise ValueError(
            "E14 implementation transition must contain one commit per leg")

    expected_legs = (
        (
            "training_to_amendment",
            TRAINING_SOURCE_COMMIT,
            TRANSCRIPT_AMENDMENT_COMMIT,
            TRAINING_TO_AMENDMENT_CHANGES,
            first_count,
        ),
        (
            "amendment_to_prepare",
            TRANSCRIPT_AMENDMENT_COMMIT,
            prepare_commit,
            AMENDMENT_TO_PREPARE_CHANGES,
            second_count,
        ),
    )
    observed_legs: dict[str, object] = {}
    for label, from_commit, to_commit, expected_changes, count in expected_legs:
        actual_changes = _git_name_status(from_commit, to_commit)
        if actual_changes != expected_changes:
            raise ValueError(
                f"E14 implementation transition path set mismatch: {label}")
        leg = _require_mapping(
            transition.get(label), f"prepare transition {label}")
        if set(leg) != {
            "from_commit",
            "to_commit",
            "commit_count",
            "exactly_one_commit",
            "changes",
            "expected_changes_match",
        }:
            raise ValueError(
                f"prepare implementation-transition leg fields mismatch: "
                f"{label}")
        if (
            leg.get("from_commit") != from_commit
            or leg.get("to_commit") != to_commit
            or leg.get("commit_count") != count
            or leg.get("exactly_one_commit") is not True
            or leg.get("changes") != list(expected_changes)
            or leg.get("expected_changes_match") is not True
        ):
            raise ValueError(
                f"prepare implementation-transition leg mismatch: {label}")
        observed_legs[label] = {
            "from_commit": from_commit,
            "to_commit": to_commit,
            "commit_count": count,
            "exactly_one_commit": True,
            "changes": list(actual_changes),
            "expected_changes_match": True,
        }

    ancestry = _require_mapping(
        transition.get("ancestry"), "prepare transition ancestry")
    if dict(ancestry) != {
        "training_to_amendment": True,
        "amendment_to_prepare": True,
    }:
        raise ValueError("prepare implementation-transition ancestry mismatch")
    return {
        "schema_version": transition["schema_version"],
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "amendment_source_commit": TRANSCRIPT_AMENDMENT_COMMIT,
        "prepare_source_commit": prepare_commit,
        **observed_legs,
        "ancestry": {
            "training_to_amendment": True,
            "amendment_to_prepare": True,
            "prepare_to_current_head": True,
        },
    }


def _head_source_record(path: Path) -> dict[str, object]:
    """Return a current-file record after byte equality with its HEAD blob."""
    path = Path(os.path.abspath(path))
    relative = _portable_path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"missing regular E14 implementation source: {relative}")
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
        )
        head_bytes = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"E14 implementation source is unavailable from HEAD: {relative}"
        ) from exc
    if tracked.returncode != 0:
        raise RuntimeError(
            f"E14 implementation source is not tracked: {relative}")
    working_bytes = path.read_bytes()
    if working_bytes != head_bytes:
        raise RuntimeError(
            f"E14 implementation source differs from HEAD: {relative}")
    return {
        "path": relative,
        "bytes": len(working_bytes),
        "sha256": hashlib.sha256(working_bytes).hexdigest(),
        "tracked": True,
        "head_blob_matches": True,
    }


def _commit_source_record(commit: str, path: Path) -> dict[str, object]:
    relative = _portable_path(path)
    try:
        content = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"E14 source is unavailable at {commit}: {relative}") from exc
    return {
        "path": relative,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _validate_analysis_worktree() -> dict[str, object]:
    """Require clean tracked state and only the six canonical score outputs."""
    head = _current_head()
    tracked = sorted(set(filter(None, (
        _git_text(["diff", "--name-only"])
        + _git_text(["diff", "--cached", "--name-only"])
    ).splitlines())))
    if tracked:
        raise RuntimeError(
            "canonical E14 analysis requires clean tracked state: "
            + json.dumps(tracked, sort_keys=True))
    allowed = sorted(_portable_path(path) for path in CANONICAL_SCORE_OUTPUTS)
    status_records = sorted(filter(
        None,
        _git_text([
            "status", "--porcelain=v1", "-z", "--untracked-files=all",
        ]).split("\0"),
    ))
    expected_status_records = sorted(f"?? {path}" for path in allowed)
    if status_records != expected_status_records:
        raise RuntimeError(
            "canonical E14 analysis requires exactly the six untracked "
            "score outputs and no other status: "
            + json.dumps({
                "expected": expected_status_records,
                "observed": status_records,
            }, sort_keys=True))
    if _git_text(["status", "--short", "--", "wisa"]).strip():
        raise RuntimeError("frozen wisa/ archive is not clean")
    return {
        "source_commit": head,
        "tracked_state_clean": True,
        "wisa_clean": True,
        "allowed_untracked_score_outputs": allowed,
        "observed_status_records": status_records,
        "undeclared_untracked_files": [],
    }


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _require_record_matches(
    record: object,
    path: Path,
    label: str,
    *,
    require_bytes: bool = True,
) -> dict[str, object]:
    declared = _require_mapping(record, label)
    observed = _artifact_record(path)
    fields = ("path", "sha256", "bytes") if require_bytes else ("path", "sha256")
    for field in fields:
        if declared.get(field) != observed[field]:
            raise ValueError(f"{label} {field} does not match {path}")
    return observed


def _parse_transcript_decimal(line: bytes, field: bytes) -> int:
    prefix = field + b"="
    if not line.startswith(prefix):
        raise ValueError(
            f"matched-real transcript is missing {field.decode('ascii')}")
    value = line[len(prefix):]
    if (
        not value
        or any(character < ord("0") or character > ord("9")
               for character in value)
        or (len(value) > 1 and value.startswith(b"0"))
    ):
        raise ValueError(
            f"matched-real transcript has invalid {field.decode('ascii')}")
    return int(value)


def _parse_matched_real_transcript(payload: bytes) -> dict[str, object]:
    """Parse and fail closed on the wrapper's byte-exact transcript format."""
    if not isinstance(payload, bytes):
        raise TypeError("matched-real transcript payload must be bytes")
    lines: list[bytes] = []
    remainder = payload
    for _ in range(4):
        line, separator, remainder = remainder.partition(b"\n")
        if not separator:
            raise ValueError("matched-real transcript header is truncated")
        lines.append(line)
    if lines[0] != b"E14 matched-real frozen trainer transcript v1":
        raise ValueError("matched-real transcript magic header mismatch")
    returncode = _parse_transcript_decimal(lines[1], b"returncode")
    stdout_bytes = _parse_transcript_decimal(lines[2], b"stdout_bytes")
    stderr_bytes = _parse_transcript_decimal(lines[3], b"stderr_bytes")
    if returncode != 0:
        raise ValueError("matched-real transcript child returncode is nonzero")

    stdout_marker = b"\n--- stdout ---\n"
    stderr_marker = b"\n--- stderr ---\n"
    if not remainder.startswith(stdout_marker):
        raise ValueError("matched-real transcript stdout delimiter mismatch")
    remainder = remainder[len(stdout_marker):]
    if len(remainder) < stdout_bytes:
        raise ValueError("matched-real transcript stdout payload is truncated")
    stdout = remainder[:stdout_bytes]
    remainder = remainder[stdout_bytes:]
    if not remainder.startswith(stderr_marker):
        raise ValueError("matched-real transcript stderr delimiter mismatch")
    stderr = remainder[len(stderr_marker):]
    if len(stderr) != stderr_bytes:
        raise ValueError("matched-real transcript stderr byte count mismatch")

    skip_lines = [
        line
        for line in stdout.decode("utf-8", errors="replace").splitlines()
        if line.startswith("skip existing ")
    ]
    if skip_lines:
        raise ValueError(
            "matched-real transcript reports existing-target skips: "
            + json.dumps(skip_lines, sort_keys=True))
    return {
        "format": MATCHED_REAL_TRANSCRIPT_FORMAT,
        "returncode": returncode,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "existing_target_skip_lines": [],
    }


parse_matched_real_transcript = _parse_matched_real_transcript


def _validate_matched_real_transcript(
    training_run: Mapping[str, Any],
) -> dict[str, object]:
    observed = _artifact_record(MATCHED_REAL_TRANSCRIPT)
    if (
        observed["bytes"] != 4_977
        or observed["sha256"] != MATCHED_REAL_TRANSCRIPT_SHA256
    ):
        raise ValueError("frozen matched-real transcript hash/size mismatch")
    parsed = _parse_matched_real_transcript(
        MATCHED_REAL_TRANSCRIPT.read_bytes())
    registered = _require_mapping(
        training_run.get("child_transcript"),
        "matched-real run child_transcript",
    )
    expected = {
        **observed,
        "format": MATCHED_REAL_TRANSCRIPT_FORMAT,
        "stdout_bytes": parsed["stdout_bytes"],
        "stderr_bytes": parsed["stderr_bytes"],
        "streams_preserved_separately": True,
        "existing_target_skip_lines": [],
    }
    for field, value in expected.items():
        if registered.get(field) != value:
            raise ValueError(
                f"matched-real transcript run link mismatch: {field}")
    return {
        **expected,
        "returncode": parsed["returncode"],
        "stdout_sha256": parsed["stdout_sha256"],
        "stderr_sha256": parsed["stderr_sha256"],
    }


validate_matched_real_transcript = _validate_matched_real_transcript


def _validate_training_record_chain(
    prepare: Mapping[str, Any],
    implementation_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    preflight_artifact = _artifact_record(TRAINING_PREFLIGHT)
    training_run_artifact = _artifact_record(TRAINING_RUN)
    if (
        preflight_artifact["bytes"] != 21_223
        or preflight_artifact["sha256"] != TRAINING_PREFLIGHT_SHA256
        or training_run_artifact["bytes"] != 37_000
        or training_run_artifact["sha256"] != TRAINING_RUN_SHA256
    ):
        raise ValueError("frozen matched-real training-record hash/size mismatch")
    preflight = _read_json_object(TRAINING_PREFLIGHT)
    training_run = _read_json_object(TRAINING_RUN)
    if (
        preflight.get("schema_version")
        != "e14.matched_real_training_preflight.v1"
        or preflight.get("record_type") != "matched_real_training_preflight"
        or preflight.get("status") != "T-PASS-PREFLIGHT"
        or preflight.get("mode") != "execute"
        or preflight.get("pre_result") is not True
        or preflight.get("factorial_outcomes_computed") is not False
    ):
        raise ValueError("matched-real preflight schema/status mismatch")
    if (
        training_run.get("schema_version")
        != "e14.matched_real_training_run.v1"
        or training_run.get("record_type") != "matched_real_training_run"
        or training_run.get("status") != "T-PASS-TRAINING"
        or training_run.get("child_exit_status") != 0
        or training_run.get("environment_matches_preflight") is not True
        or training_run.get("factorial_outcomes_computed") is not False
    ):
        raise ValueError("matched-real training run schema/status mismatch")

    preflight_source = _require_mapping(
        preflight.get("source_provenance"),
        "matched-real preflight source_provenance",
    )
    implementation_commit = str(preflight_source.get("head_commit", ""))
    if (
        implementation_commit != TRAINING_SOURCE_COMMIT
        or training_run.get("source_commit") != implementation_commit
    ):
        raise ValueError("matched-real implementation commit chain mismatch")
    _require_record_matches(
        training_run.get("preflight"),
        TRAINING_PREFLIGHT,
        "matched-real training preflight link",
    )
    if _require_mapping(
        training_run.get("preflight"), "matched-real preflight link"
    ).get("schema_version") != preflight["schema_version"]:
        raise ValueError("matched-real preflight schema link mismatch")
    if (
        preflight_source.get("worktree_clean") is not True
        or preflight_source.get("wisa_clean") is not True
        or preflight_source.get("preregistration_commit") != PREREG_COMMIT
        or preflight_source.get("preregistration_commit_is_ancestor") is not True
    ):
        raise ValueError("matched-real preflight source gates did not pass")
    _require_ancestor(PREREG_COMMIT, implementation_commit)

    source_files = _require_mapping(
        preflight_source.get("source_files"),
        "matched-real preflight source_files",
    )
    if set(source_files) != HISTORICAL_TRAINING_SOURCE_PATHS:
        raise ValueError("matched-real preflight source-file set mismatch")
    for relative, registered_value in source_files.items():
        registered_source = _require_mapping(
            registered_value, f"matched-real source {relative}")
        path = ROOT / relative
        committed = _commit_source_record(implementation_commit, path)
        if any(
            registered_source.get(field) != committed[field]
            for field in ("path", "bytes", "sha256")
        ):
            raise ValueError(
                f"matched-real source differs from implementation commit: "
                f"{relative}")
        if (
            registered_source.get("committed_blob_sha256")
            != committed["sha256"]
            or registered_source.get("matches_head") is not True
        ):
            raise ValueError(
                f"matched-real source HEAD attestation failed: {relative}")
        if relative == _portable_path(PREREG):
            if committed["sha256"] != PREREG_SHA256:
                raise ValueError("matched-real preregistration hash mismatch")

    # These three inputs were outside the allowed correction path set and
    # therefore must remain byte-identical to the training implementation.
    immutable_sources = {
        "training_wrapper": TRAINING_WRAPPER,
        "amendment": AMENDMENT,
    }
    prereg_now = _artifact_record(PREREG)
    prereg_then = _commit_source_record(implementation_commit, PREREG)
    if prereg_now != prereg_then:
        raise ValueError("preregistration changed since matched-real training")
    for role, path in immutable_sources.items():
        committed = _commit_source_record(implementation_commit, path)
        current = implementation_sources[role]
        if any(
            committed[field] != current[field]
            for field in ("path", "bytes", "sha256")
        ):
            raise ValueError(
                f"immutable implementation source changed since training: "
                f"{role}")
    generic_trainer_now = _artifact_record(GENERIC_TRAINER)
    generic_trainer_then = _commit_source_record(
        implementation_commit, GENERIC_TRAINER)
    if generic_trainer_now != generic_trainer_then:
        raise ValueError(
            "generic trainer changed since matched-real training")

    validation = _require_mapping(
        training_run.get("validation"), "matched-real run validation")
    required_true = (
        "all_logs_valid",
        "all_checkpoints_weights_only_load",
        "all_checkpoint_architectures_compatible",
        "all_checkpoint_tensors_finite",
        "all_artifact_hashes_recorded",
        "preflight_cross_linked",
        "no_e14_factorial_inference",
        "child_output_persisted",
    )
    if (
        validation.get("validated_seed_count") != 5
        or validation.get("seeds") != list(SEEDS)
        or any(validation.get(field) is not True for field in required_true)
    ):
        raise ValueError("matched-real run validation gates did not all pass")
    artifacts = training_run.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 5:
        raise ValueError("matched-real run must contain five seed artifacts")
    observed_seeds = []
    for artifact_value in artifacts:
        artifact = _require_mapping(
            artifact_value, "matched-real seed artifact")
        observed_seeds.append(int(artifact.get("seed", -1)))
        for kind in ("checkpoint", "log"):
            record = _require_mapping(
                artifact.get(kind), f"matched-real artifact {kind}")
            _require_sha256(
                record.get("sha256"), f"matched-real artifact {kind} sha256")
            if (
                not isinstance(record.get("path"), str)
                or not isinstance(record.get("bytes"), int)
                or int(record["bytes"]) <= 0
            ):
                raise ValueError(
                    f"matched-real artifact {kind} metadata is invalid")
    if sorted(observed_seeds) != sorted(SEEDS) or len(set(observed_seeds)) != 5:
        raise ValueError("matched-real artifact seed set mismatch")

    transcript = _validate_matched_real_transcript(training_run)
    if validation.get("child_reported_existing_target_skips") is not False:
        raise ValueError("matched-real run transcript skip gate did not pass")

    registered = _require_mapping(
        prepare.get("matched_real_training_records"),
        "prepare matched_real_training_records",
    )
    if set(registered) != {"preflight", "run", "child_transcript"}:
        raise ValueError("prepare matched-real record set mismatch")
    if dict(_require_mapping(
        registered["preflight"], "prepare matched-real preflight"
    )) != preflight_artifact:
        raise ValueError("prepare matched-real preflight record mismatch")
    if dict(_require_mapping(
        registered["run"], "prepare matched-real run"
    )) != training_run_artifact:
        raise ValueError("prepare matched-real run record mismatch")
    registered_transcript = _require_mapping(
        registered["child_transcript"],
        "prepare matched-real child_transcript",
    )
    if dict(registered_transcript) != transcript:
        raise ValueError("prepare matched-real transcript record mismatch")
    return {
        "implementation_commit": implementation_commit,
        "preflight": preflight_artifact,
        "run": training_run_artifact,
        "child_transcript": transcript,
        "preflight_status": preflight["status"],
        "run_status": training_run["status"],
        "cross_link_valid": True,
    }


def _validate_prepare_record(
    head: str,
    implementation_sources: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, Any], dict[str, object]]:
    prepare = _read_json_object(PREPARE_RECORD)
    if (
        prepare.get("schema_version") != "e14.l4_factorial_prepare.v1"
        or prepare.get("status") != "T-PASS-MANIPULATION"
        or prepare.get("checkpoint_deserialization_performed") is not False
        or prepare.get("model_inference_performed") is not False
    ):
        raise ValueError("E14 prepare record schema/status mismatch")
    prospective = _require_mapping(
        prepare.get("prospective_record"), "prepare prospective_record")
    prereg = _artifact_record(PREREG)
    amendment = _artifact_record(AMENDMENT)
    transcript_amendment = _artifact_record(TRANSCRIPT_AMENDMENT)
    if (
        prereg["sha256"] != PREREG_SHA256
        or prospective.get("path") != prereg["path"]
        or prospective.get("sha256") != prereg["sha256"]
    ):
        raise ValueError("prepare preregistration link/hash mismatch")
    if amendment["sha256"] != AMENDMENT_SHA256:
        raise ValueError("implementation-amendment hash mismatch")
    if transcript_amendment["sha256"] != TRANSCRIPT_AMENDMENT_SHA256:
        raise ValueError("transcript-amendment hash mismatch")

    registered_sources = _require_mapping(
        prepare.get("implementation_sources"),
        "prepare implementation_sources",
    )
    if set(registered_sources) != set(IMPLEMENTATION_SOURCES):
        raise ValueError("prepare implementation-source role set mismatch")
    for role, observed in implementation_sources.items():
        registered = _require_mapping(
            registered_sources[role], f"prepare implementation source {role}")
        for field in ("path", "bytes", "sha256"):
            if registered.get(field) != observed[field]:
                raise ValueError(
                    f"prepare implementation source changed: {role}/{field}")
        if (
            registered.get("tracked") is not True
            or registered.get("head_blob_matches") is not True
        ):
            raise ValueError(
                f"prepare implementation source attestation failed: {role}")
    technical_amendments = _require_mapping(
        prepare.get("technical_amendments"), "prepare technical_amendments")
    if set(technical_amendments) != {"implementation", "stage_a_transcript"}:
        raise ValueError("prepare technical-amendment alias set mismatch")
    for alias, role in (
        ("implementation", "amendment"),
        ("stage_a_transcript", "transcript_amendment"),
    ):
        technical_record = _require_mapping(
            technical_amendments[alias],
            f"prepare technical amendment {alias}",
        )
        source_record = implementation_sources[role]
        if dict(technical_record) != dict(source_record):
            raise ValueError(
                f"prepare technical amendment alias mismatch: {alias}")
    evaluator_alias = _require_mapping(
        prepare.get("evaluator"), "prepare evaluator alias")
    if any(
        evaluator_alias.get(field)
        != implementation_sources["evaluator"][field]
        for field in ("path", "bytes", "sha256")
    ):
        raise ValueError("prepare evaluator alias disagrees with source map")

    output_records = _require_mapping(
        prepare.get("outputs"), "prepare outputs")
    if set(output_records) != set(PREPARE_OUTPUTS):
        raise ValueError("prepare output set mismatch")
    observed_outputs = {}
    expected_rows = {
        "base_manifest": 6_000,
        "latent_manifest": 12_000,
        "manipulation_checks": 96,
    }
    for name, path in PREPARE_OUTPUTS.items():
        observed_outputs[name] = _require_record_matches(
            output_records[name], path, f"prepare output {name}")
        if _require_mapping(
            output_records[name], f"prepare output {name}"
        ).get("rows") != expected_rows[name]:
            raise ValueError(f"prepare output row count mismatch: {name}")

    gates = _require_mapping(
        prepare.get("manipulation_gates"), "prepare manipulation_gates")
    if (
        gates.get("all_manipulation_gates_pass") is not True
        or gates.get("repeated_regeneration_match") is not True
        or gates.get("repeated_latent_regeneration_match") is not True
        or gates.get("base_rows") != 6_000
        or gates.get("latent_rows") != 12_000
        or gates.get("scenario_instances") != 192_000
        or gates.get("unique_scenario_keys") != 192_000
        or gates.get("groups") != 96
        or gates.get("instances_per_group") != 2_000
        or gates.get("transform_serialization_version")
        != TRANSFORM_SERIALIZATION_VERSION
    ):
        raise ValueError("prepare manipulation gates did not all pass")
    transformation_sha = _require_sha256(
        gates.get("transformation_sha256"),
        "prepare transformation_sha256",
    )
    scenario_key_sha = _require_sha256(
        gates.get("scenario_key_sha256"),
        "prepare scenario_key_sha256",
    )
    latent_sha = _require_sha256(
        gates.get("latent_semantic_sha256"),
        "prepare latent_semantic_sha256",
    )
    latent_record = _require_mapping(
        output_records["latent_manifest"], "prepare latent_manifest output")
    if latent_record.get("semantic_sha256") != latent_sha:
        raise ValueError("prepare latent semantic digest link mismatch")
    training = _validate_training_record_chain(
        prepare, implementation_sources)
    prepare_source = _require_mapping(
        prepare.get("source_provenance"), "prepare source_provenance")
    prepare_commit = str(prepare_source.get("source_commit", ""))
    if (
        len(prepare_commit) != 40
        or any(character not in "0123456789abcdef"
               for character in prepare_commit)
        or prepare_commit in {
            training["implementation_commit"],
            TRANSCRIPT_AMENDMENT_COMMIT,
        }
    ):
        raise ValueError("prepare implementation commit is invalid")
    if prepare_source.get("source_tracked_state_clean") is not True:
        raise ValueError("prepare source tracked-state gate did not pass")
    transition = _validate_implementation_transition(
        prepare_source.get("implementation_transition"),
        prepare_commit,
        current_head=head,
    )
    return prepare, {
        "record": _artifact_record(PREPARE_RECORD),
        "schema_version": prepare["schema_version"],
        "status": prepare["status"],
        "implementation_commit": prepare_commit,
        "implementation_commit_is_ancestor_of_scoring_commit": True,
        "implementation_sources_match_current_head": True,
        "prospective_record": prereg,
        "implementation_amendment": amendment,
        "transcript_amendment": transcript_amendment,
        "technical_amendments": {
            alias: dict(_require_mapping(record, alias))
            for alias, record in technical_amendments.items()
        },
        "implementation_transition": transition,
        "outputs": observed_outputs,
        "construction": {
            "transformation_sha256": transformation_sha,
            "scenario_key_sha256": scenario_key_sha,
            "latent_semantic_sha256": latent_sha,
            "scenario_instances": 192_000,
        },
        "training_records": training,
    }


def _canonical_count_projection(counts: Mapping[str, int]) -> dict[str, int]:
    return {
        "checkpoint_evaluations": int(counts["checkpoint_evaluations"]),
        "normal_rows": int(counts["normal_rows"]),
        "scenario_rows": int(counts["scenario_rows"]),
        "primary_scenario_rows": int(counts["primary_scenario_rows"]),
        "canonical_primary_scenario_rows": int(
            counts["primary_canonical_scenario_rows"]),
        "by_cell_rows": int(counts["cell_rows"]),
        "augmentation_delta_rows": int(counts["delta_rows"]),
        "transformed_forward_evaluations": int(
            counts["transformed_forward_evaluations"]),
        "clean_forward_evaluations": int(
            counts["clean_forward_evaluations"]),
    }


def _validate_score_run(
    prepare: Mapping[str, Any],
    prepare_evidence: Mapping[str, object],
    head: str,
    counts: Mapping[str, int],
    implementation_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    run = _read_json_object(SCORE_RUN)
    score_log = _read_json_object(SCORE_LOG)
    if (
        run.get("schema_version") != "e14.l4_factorial_run.v1"
        or run.get("status") != "T-PASS"
        or run.get("effects_table_published") is not False
        or run.get("artifact_manifest_published") is not False
    ):
        raise ValueError("E14 score run schema/status mismatch")
    source = _require_mapping(
        run.get("source_provenance"), "score source_provenance")
    if (
        source.get("source_commit") != head
        or source.get("source_worktree_clean") is not True
        or source.get("wisa_clean") is not True
    ):
        raise ValueError("score source-provenance commit/status mismatch")

    prospective = _require_mapping(
        run.get("prospective_record"), "score prospective_record")
    prereg = _artifact_record(PREREG)
    if (
        prospective.get("path") != prereg["path"]
        or prospective.get("sha256") != PREREG_SHA256
    ):
        raise ValueError("score preregistration link mismatch")
    _require_record_matches(
        run.get("prepare_record"),
        PREPARE_RECORD,
        "score prepare-record link",
        require_bytes=False,
    )
    evaluator_link = _require_mapping(
        run.get("evaluator"), "score evaluator link")
    evaluator = implementation_sources["evaluator"]
    if any(
        evaluator_link.get(field) != evaluator[field]
        for field in ("path", "sha256")
    ):
        raise ValueError("score evaluator link does not match current HEAD")

    prepare_construction = _require_mapping(
        prepare_evidence.get("construction"), "validated prepare construction")
    construction = _require_mapping(
        run.get("construction"), "score construction")
    if (
        construction.get("matches_prepare") is not True
        or construction.get("transformation_instances") != 192_000
        or construction.get("transformation_sha256")
        != prepare_construction["transformation_sha256"]
        or construction.get("scenario_key_sha256")
        != prepare_construction["scenario_key_sha256"]
        or construction.get("serialization_version")
        != TRANSFORM_SERIALIZATION_VERSION
    ):
        raise ValueError("score/prepare construction digest link mismatch")
    prepare_standardizer = _require_mapping(
        _require_mapping(
            prepare.get("standardizer"), "prepare standardizer"
        ).get("hashes"),
        "prepare standardizer hashes",
    )
    if run.get("standardizer_hashes") != dict(prepare_standardizer):
        raise ValueError("score/prepare standardizer hash link mismatch")

    prepared_bundles_value = prepare.get("checkpoint_bundles")
    score_checkpoints_value = run.get("checkpoint_validation")
    if (
        not isinstance(prepared_bundles_value, list)
        or len(prepared_bundles_value) != 30
        or not isinstance(score_checkpoints_value, list)
        or len(score_checkpoints_value) != 30
    ):
        raise ValueError("score/prepare checkpoint matrix must contain 30 rows")
    prepared_checkpoints = {}
    for bundle_value in prepared_bundles_value:
        bundle = _require_mapping(
            bundle_value, "prepare checkpoint bundle")
        key = (str(bundle.get("arm")), int(bundle.get("seed", -1)))
        checkpoint = _require_mapping(
            bundle.get("checkpoint"), "prepare checkpoint record")
        prepared_checkpoints[key] = {
            "path": checkpoint.get("path"),
            "sha256": checkpoint.get("sha256"),
        }
    expected_checkpoint_keys = {
        (arm, seed) for arm in ARMS for seed in SEEDS
    }
    if set(prepared_checkpoints) != expected_checkpoint_keys:
        raise ValueError("prepare checkpoint key set mismatch")
    observed_checkpoint_keys = set()
    for validation_value in score_checkpoints_value:
        validation = _require_mapping(
            validation_value, "score checkpoint validation")
        key = (
            str(validation.get("arm")),
            int(validation.get("seed", -1)),
        )
        if key in observed_checkpoint_keys or key not in expected_checkpoint_keys:
            raise ValueError("score checkpoint validation key mismatch")
        observed_checkpoint_keys.add(key)
        expected_checkpoint = prepared_checkpoints[key]
        if (
            validation.get("path") != expected_checkpoint["path"]
            or validation.get("sha256") != expected_checkpoint["sha256"]
            or validation.get("weights_only_load") is not True
            or validation.get("state_dict_keys") != 23
            or validation.get("architecture_compatible") is not True
            or validation.get("all_tensors_finite") is not True
        ):
            raise ValueError(
                f"score checkpoint attestation mismatch: {key}")
    if observed_checkpoint_keys != expected_checkpoint_keys:
        raise ValueError("score checkpoint validation matrix is incomplete")

    expected_counts = _canonical_count_projection(counts)
    completeness = _require_mapping(
        run.get("completeness"), "score completeness")
    if dict(completeness) != expected_counts:
        raise ValueError("score run completeness disagrees with validated tables")
    if (
        score_log.get("schema_version")
        != "e14.l4_factorial_score_log.v1"
        or score_log.get("status") != "T-PASS"
        or score_log.get("device") != "cpu"
        or score_log.get("torch_threads") != 8
        or score_log.get("torch_interop_threads") != 1
        or score_log.get("batch_size") != 4_096
        or score_log.get("transformation_sha256")
        != prepare_construction["transformation_sha256"]
        or score_log.get("counts") != expected_counts
        or score_log.get(
            "normal_control_reported_before_attack_results") is not True
        or score_log.get("exact_and_binary_endpoints_separate") is not True
        or score_log.get(
            "effects_and_scientific_verdict_deferred_to_registered_analyzer"
        ) is not True
    ):
        raise ValueError("score log schema/status/count/digest mismatch")

    output_records = _require_mapping(run.get("outputs"), "score outputs")
    if set(output_records) != set(RECORDED_SCORE_OUTPUTS):
        raise ValueError("score output set mismatch")
    observed_outputs = {}
    for name, path in RECORDED_SCORE_OUTPUTS.items():
        observed_outputs[name] = _require_record_matches(
            output_records[name], path, f"score output {name}")
    observed_outputs["run_record"] = _artifact_record(SCORE_RUN)
    return {
        "record": observed_outputs["run_record"],
        "schema_version": run["schema_version"],
        "status": run["status"],
        "source_commit": head,
        "prepare_record_sha256": _artifact_record(PREPARE_RECORD)["sha256"],
        "evaluator_sha256": evaluator["sha256"],
        "construction": dict(construction),
        "standardizer_hashes_match_prepare": True,
        "checkpoint_attestations_cross_linked": 30,
        "completeness": expected_counts,
        "outputs": observed_outputs,
        "all_output_paths_bytes_sha256_valid": True,
        "score_log_status": score_log["status"],
    }


def validate_production_provenance(
    counts: Mapping[str, int],
) -> dict[str, object]:
    """Validate the complete canonical Stage-A-to-analysis provenance chain."""
    worktree = _validate_analysis_worktree()
    head = str(worktree["source_commit"])
    implementation_sources = {
        role: _head_source_record(path)
        for role, path in IMPLEMENTATION_SOURCES.items()
    }
    prepare, prepare_evidence = _validate_prepare_record(
        head, implementation_sources)
    score_evidence = _validate_score_run(
        prepare,
        prepare_evidence,
        head,
        counts,
        implementation_sources,
    )
    return {
        "status": "T-PASS-PROVENANCE",
        "analysis_source_state": worktree,
        "implementation_sources": implementation_sources,
        "prepare_validation": prepare_evidence,
        "score_validation": score_evidence,
        "canonical_manifest_inputs": [
            _artifact_record(path) for path in CANONICAL_MANIFEST_INPUTS
        ],
    }


def _artifact_record(path: Path) -> dict[str, object]:
    path = Path(os.path.abspath(path))
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required E14 artifact is missing/empty: {path}")
    return {
        "path": _portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def publish_analysis_bundle(
    effects: pd.DataFrame,
    effects_path: Path,
    manifest_path: Path,
    artifact_paths: Sequence[Path],
    counts: Mapping[str, int],
    provenance_validation: Mapping[str, object],
) -> dict[str, object]:
    """Publish effects and final manifest atomically and without overwrite."""
    effects_path = effects_path.resolve()
    manifest_path = manifest_path.resolve()
    if effects_path == manifest_path:
        raise ValueError("effects and manifest paths must be distinct")
    for path in (effects_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite E14 output: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {path.parent}")

    unique_artifacts = []
    seen = set()
    for path in artifact_paths:
        resolved = Path(path).resolve()
        if resolved in {effects_path, manifest_path} or resolved in seen:
            continue
        seen.add(resolved)
        unique_artifacts.append(resolved)
    canonical_inputs = [path.resolve() for path in CANONICAL_MANIFEST_INPUTS]
    if unique_artifacts != canonical_inputs:
        raise ValueError(
            "final manifest inputs must be the ordered canonical E14 inputs")
    input_records = [_artifact_record(path) for path in unique_artifacts]
    if (
        provenance_validation.get("status") != "T-PASS-PROVENANCE"
        or provenance_validation.get("canonical_manifest_inputs")
        != input_records
    ):
        raise ValueError(
            "final manifest input records disagree with provenance validation")
    verdict_rows = effects[
        effects["row_type"].eq("summary")
        & effects["localization_verdict"].astype(str).ne("")
    ]
    verdicts = sorted(set(verdict_rows["localization_verdict"].astype(str)))
    if len(verdicts) != 1:
        raise ValueError(f"effects must contain one localization verdict: {verdicts}")

    common_parent = Path(os.path.commonpath([
        str(effects_path.parent), str(manifest_path.parent),
    ]))
    temporary = Path(tempfile.mkdtemp(
        prefix=".e14_l4_analysis.", dir=common_parent,
    ))
    published = False
    try:
        staged_effects = temporary / effects_path.name
        effects.to_csv(staged_effects, index=False)
        effects_record = _artifact_record(staged_effects)
        effects_record["path"] = effects_path.relative_to(ROOT).as_posix()
        payload = {
            "schema_version": "e14_l4_factorial_artifact_manifest_v1",
            "analysis": "E14 matched L4 counterfactual factorial",
            "technical_status": "T-PASS",
            "source_state": provenance_validation["analysis_source_state"],
            "provenance_validation": dict(provenance_validation),
            "counts": {**dict(counts), "effect_rows": len(effects)},
            "effect_key": list(EFFECT_KEY),
            "holm_families": {
                name: {
                    "contrast": contrast,
                    "endpoint": endpoint,
                    "id_scope": "canonical",
                    "attack_scope": "equal_macro",
                    "effects": list(FACTORIAL_EFFECTS),
                }
                for name, (contrast, endpoint)
                in PRIMARY_HOLM_FAMILIES.items()
            },
            "localization_verdict": verdicts[0],
            "inputs": input_records,
            "effects": effects_record,
        }
        staged_manifest = temporary / manifest_path.name
        staged_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        created: list[tuple[Path, Path]] = []
        try:
            os.link(staged_effects, effects_path)
            created.append((staged_effects, effects_path))
            os.link(staged_manifest, manifest_path)
            created.append((staged_manifest, manifest_path))
        except BaseException as exc:
            for staged_path, target in reversed(created):
                try:
                    target_status = target.stat(follow_symlinks=False)
                    staged_status = staged_path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if (
                    target_status.st_dev == staged_status.st_dev
                    and target_status.st_ino == staged_status.st_ino
                ):
                    target.unlink()
            raise RuntimeError(
                "E14 analysis publication failed; staged bundle retained at "
                f"{temporary}"
            ) from exc
        published = True
    finally:
        if published:
            shutil.rmtree(temporary)
    return payload


def _toy_tables(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return coherent evaluator-schema tables without reading repository data."""
    normal_rows = []
    for arm, seed, block in itertools.product(ARMS, SEEDS, BLOCKS):
        normal_recall = 0.99 - 0.0001 * SEEDS.index(seed)
        normal_rows.append({
            "arm": arm,
            "seed": seed,
            "block_id": block,
            "n": 2_000,
            "normal_recall": normal_recall,
            "fpr": 1.0 - normal_recall,
        })
    normal = pd.DataFrame(normal_rows)

    arm_scale = {
        "real_ms": 0.00,
        "rule_ms": 1.00,
        "placebo_ms": 0.35,
        "real_std": 0.05,
        "rule_std": 0.90,
        "placebo_std": 0.30,
    }
    seed_scale = dict(zip(SEEDS, (0.96, 0.98, 1.00, 1.02, 1.04)))
    scenario_rows = []
    for arm, seed, block, attack, identity, p, s, d in itertools.product(
        ARMS, SEEDS, BLOCKS, ATTACKS, ID_LEVELS, BITS, BITS, BITS,
    ):
        attack_scale = 1.0 if attack == "Gear" else 0.8
        id_scale = 1.0 if identity == "canonical" else 0.7
        block_scale = 1.0 + 0.01 * BLOCKS.index(block)
        polynomial = (
            0.030 * p - 0.015 * s + 0.010 * d
            + 0.006 * p * s - 0.004 * p * d
            + 0.003 * s * d + 0.002 * p * s * d
        )
        response = (
            arm_scale[arm] * attack_scale * id_scale * block_scale
            * seed_scale[seed] * polynomial
        )
        scenario_rows.append({
            "arm": arm,
            "seed": seed,
            "block_id": block,
            "attack": attack,
            "attack_label": 3 if attack == "Gear" else 4,
            "id_stratum": identity,
            "P": p,
            "S": s,
            "D": d,
            "cell": f"{p}{s}{d}",
            "n": 2_000,
            "exact_recall": 0.45 + response,
            "binary_recall": 0.70 + response / 2.0,
            "exact_margin_shift_mean": response * 2.5,
            "binary_margin_shift_mean": response * 1.5,
            "exact_standardized_margin_shift": response * 3.0,
            "binary_standardized_margin_shift": response * 2.0,
            "exact_standardized_margin_status": "ok",
            "binary_standardized_margin_status": "ok",
        })
    scenario = pd.DataFrame(scenario_rows)

    group_key = list(CELL_KEY)
    cell_rows = []
    for key, group in scenario.groupby(group_key, sort=True):
        row = dict(zip(group_key, key, strict=True))
        row.update({
            "cell": f"{row['P']}{row['S']}{row['D']}",
            "attacks": 2,
            "n_per_attack": 2_000,
            "macro_weighting": "equal Gear/RPM",
            "exact_recall_equal_macro": float(group["exact_recall"].mean()),
            "binary_recall_equal_macro": float(group["binary_recall"].mean()),
            "exact_standardized_margin_shift_equal_macro": float(
                group["exact_standardized_margin_shift"].mean()),
            "binary_standardized_margin_shift_equal_macro": float(
                group["binary_standardized_margin_shift"].mean()),
            "exact_standardized_margin_status": "ok",
            "binary_standardized_margin_status": "ok",
        })
        cell_rows.append(row)
    cell = pd.DataFrame(cell_rows)

    scenario_index = scenario.set_index(list(SCENARIO_KEY))
    delta_rows = []
    for contrast, (numerator, denominator) in CONTRAST_ARMS.items():
        for seed, block, attack, identity, p, s, d in itertools.product(
            SEEDS, BLOCKS, ATTACKS, ID_LEVELS, BITS, BITS, BITS,
        ):
            upper = scenario_index.loc[
                (numerator, seed, block, attack, identity, p, s, d)]
            lower = scenario_index.loc[
                (denominator, seed, block, attack, identity, p, s, d)]
            row = {
                "contrast": contrast,
                "numerator_arm": numerator,
                "denominator_arm": denominator,
                "seed": seed,
                "block_id": block,
                "attack": attack,
                "attack_label": 3 if attack == "Gear" else 4,
                "id_stratum": identity,
                "P": p,
                "S": s,
                "D": d,
                "cell": f"{p}{s}{d}",
                "n": 2_000,
            }
            for endpoint in ENDPOINTS:
                row[f"{endpoint}_delta"] = (
                    float(upper[endpoint]) - float(lower[endpoint]))
            row["exact_standardized_margin_status"] = "ok"
            row["binary_standardized_margin_status"] = "ok"
            delta_rows.append(row)
    delta = pd.DataFrame(delta_rows)
    return normal, scenario, cell, delta


def run_self_test() -> dict[str, object]:
    coefficients = {
        effect: factorial_coefficients(effect) for effect in EFFECTS
    }
    for effect, values in coefficients.items():
        if effect in {"P", "S", "D", "PS", "PD", "SD", "PSD"}:
            if not math.isclose(sum(values.values()), 0.0, abs_tol=1e-15):
                raise AssertionError(f"{effect} coefficients do not sum to zero")
    cube = {
        (p, s, d): 3 * p + 2 * s - d + 5 * p * s + 7 * p * s * d
        for p, s, d in itertools.product(BITS, repeat=3)
    }
    expected = {"P": 7.25, "S": 6.25, "D": 0.75, "PS": 8.5, "PSD": 7.0}
    for effect, target in expected.items():
        observed = factorial_effect(cube, effect)
        if not math.isclose(observed, target, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(
                f"operator {effect}: expected {target}, observed {observed}")
    adjusted = holm_adjust([0.01, 0.04, 0.03, 1.0])
    if not np.allclose(adjusted, [0.04, 0.09, 0.09, 1.0]):
        raise AssertionError(f"unexpected Holm result: {adjusted}")
    degenerate = t_summary([0.2] * 5)
    if not (
        degenerate["degenerate"]
        and degenerate["holm_input_p"] == 1.0
        and degenerate["p_status"] == "bookkeeping_p"
        and math.isnan(float(degenerate["test_p"]))
        and math.isnan(float(degenerate["ci_low"]))
    ):
        raise AssertionError("degenerate bookkeeping rule failed")
    if legacy_budget_classification(0.2, 0.0, -0.1, 0.1) != (
        "budget-direction-unresolved-zero"
    ):
        raise AssertionError("legacy zero-direction classification failed")

    normal, scenario, cell, delta = _toy_tables()
    counts = validate_complete_tables(normal, scenario, cell, delta)
    effects = build_effect_table(delta)
    if len(delta) != EXPECTED_DELTA_ROWS:
        raise AssertionError("toy delta count mismatch")
    if len(effects) != EXPECTED_EFFECT_ROWS:
        raise AssertionError("toy effect count mismatch")
    if effects.duplicated(list(EFFECT_KEY)).any():
        raise AssertionError("toy effect key is not unique")
    summaries = effects[effects["row_type"].eq("summary")]
    registered = summaries[summaries["holm_family"].astype(str).ne("")]
    supporting = summaries[summaries["holm_family"].astype(str).eq("")]
    if (
        len(registered) != 3 * len(FACTORIAL_EFFECTS)
        or registered[[
            "test_p", "holm_input_p", "raw_p", "holm_adjusted_p",
        ]].isna().any().any()
        or supporting[[
            "test_p", "holm_input_p", "raw_p", "holm_adjusted_p",
        ]].notna().any().any()
        or not supporting["p_status"].eq(
            "not_tested_supporting").all()
    ):
        raise AssertionError("registered-only p-value contract failed")
    if (
        not summaries["block_n_available"].eq(3).all()
        or summaries[[
            "block_mean", "block_sd", "block_min", "block_max",
        ]].isna().any().any()
        or effects.loc[
            ~effects["row_type"].eq("summary"),
            [
                "block_n_available", "block_mean", "block_sd",
                "block_min", "block_max",
            ],
        ].notna().any().any()
    ):
        raise AssertionError("block-descriptive summary contract failed")
    canonical_args = build_parser().parse_args([])
    _require_canonical_cli_paths(canonical_args)
    return {
        "status": "pass",
        "delta_rows": len(delta),
        "effect_rows": len(effects),
        "effect_key_unique": True,
        "registered_holm_rows": len(registered),
        "supporting_p_values_cleared": True,
        "summary_block_descriptives_complete": True,
        "validated_counts": counts,
        "verdict": sorted(set(
            effects.loc[
                effects["localization_verdict"].astype(str).ne(""),
                "localization_verdict",
            ].astype(str)
        )),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--by-scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--by-cell", type=Path, default=DEFAULT_CELL)
    parser.add_argument("--augmentation-delta", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--effects-output", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument(
        "--artifact-manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="additional required non-model artifact to hash in the final manifest",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic in-memory validation without reading evaluator tables",
    )
    return parser


def _require_canonical_cli_paths(args: argparse.Namespace) -> None:
    expected = {
        "normal": DEFAULT_NORMAL,
        "by_scenario": DEFAULT_SCENARIO,
        "by_cell": DEFAULT_CELL,
        "augmentation_delta": DEFAULT_DELTA,
        "effects_output": DEFAULT_EFFECTS,
        "artifact_manifest_output": DEFAULT_MANIFEST,
    }
    mismatches = {}
    for field, canonical in expected.items():
        supplied = Path(os.path.abspath(getattr(args, field)))
        canonical_absolute = Path(os.path.abspath(canonical))
        if supplied != canonical_absolute:
            mismatches[field] = {
                "expected": str(canonical_absolute),
                "observed": str(supplied),
            }
    if args.artifact:
        mismatches["artifact"] = {
            "expected": "no additional inputs",
            "observed": [str(path) for path in args.artifact],
        }
    if mismatches:
        raise ValueError(
            "production E14 analysis requires canonical input/output paths: "
            + json.dumps(mismatches, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return

    _require_canonical_cli_paths(args)
    for output in (DEFAULT_EFFECTS, DEFAULT_MANIFEST):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite E14 output: {output}")
    normal = pd.read_csv(args.normal)
    scenario = pd.read_csv(args.by_scenario)
    cell = pd.read_csv(args.by_cell)
    delta = pd.read_csv(args.augmentation_delta)
    counts = validate_complete_tables(normal, scenario, cell, delta)
    provenance = validate_production_provenance(counts)
    effects = build_effect_table(delta)
    payload = publish_analysis_bundle(
        effects,
        args.effects_output,
        args.artifact_manifest_output,
        CANONICAL_MANIFEST_INPUTS,
        counts,
        provenance,
    )
    print(json.dumps({
        "effects": str(args.effects_output),
        "artifact_manifest": str(args.artifact_manifest_output),
        "technical_status": payload["technical_status"],
        "effect_rows": payload["counts"]["effect_rows"],
        "localization_verdict": payload["localization_verdict"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
