#!/usr/bin/env python3
"""Analyze the preregistered E15 Rule construction-seed crossing.

This module is deliberately model-free.  It consumes only the complete paired
Rule-minus-shared-real L4 and L2 delta tables emitted by
``evaluate_rule_construction_seed_crossing.py``.  The construction, pipeline,
and block axes remain explicit until the aggregation step registered in
``journal/experiments/e15_rule_construction_crossing/PREREG.md``.

The canonical command is guarded by a required ``--analyze`` flag.  Importing
the module, running ``--help``, or running ``--self-test`` cannot publish a
canonical E15 artifact.
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
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "journal"
TABLES = JOURNAL / "results" / "tables"
LOGS = JOURNAL / "results" / "logs"
EXPERIMENT = JOURNAL / "experiments" / "e15_rule_construction_crossing"
OUTPUT_VERSION = "v2"

PREREG = EXPERIMENT / "PREREG.md"
V1_TRAINING_FAILURE = EXPERIMENT / "training_failure_v1.json"
PREPARE_RECORD = EXPERIMENT / f"prepare_{OUTPUT_VERSION}.json"
RUN_RECORD = EXPERIMENT / f"run_{OUTPUT_VERSION}.json"
TRAINING_RUN = EXPERIMENT / f"training_run_{OUTPUT_VERSION}.json"
POOL_GENERATION_RUN = (
    EXPERIMENT / f"pool_generation_run_{OUTPUT_VERSION}.json"
)

L4_NORMAL = TABLES / (
    f"e15_l4_factorial_normal_by_crossed_seed_{OUTPUT_VERSION}.csv"
)
L4_SCENARIO = TABLES / (
    f"e15_l4_factorial_by_scenario_{OUTPUT_VERSION}.csv"
)
L4_CELL = TABLES / (
    f"e15_l4_factorial_by_cell_{OUTPUT_VERSION}.csv"
)
L4_DELTA = TABLES / f"e15_l4_factorial_delta_{OUTPUT_VERSION}.csv"
RUNTIME_CONTINUITY = TABLES / (
    f"e15_real_runtime_continuity_{OUTPUT_VERSION}.csv"
)
ANCHOR_CONTINUITY = TABLES / (
    f"e15_anchor_outcome_continuity_{OUTPUT_VERSION}.csv"
)
L2_SCENARIO = TABLES / (
    f"e15_l2_continuity_by_scenario_{OUTPUT_VERSION}.csv"
)
L2_DELTA = TABLES / (
    f"e15_l2_continuity_delta_{OUTPUT_VERSION}.csv"
)
SCORE_LOG = LOGS / (
    f"e15_rule_construction_crossing_{OUTPUT_VERSION}.log"
)

L4_EFFECTS_OUTPUT = (
    TABLES / (
        "e15_l4_factorial_effects_by_crossed_cell_"
        f"{OUTPUT_VERSION}.csv"
    )
)
L4_HIERARCHY_OUTPUT = (
    TABLES / (
        "e15_l4_factorial_hierarchical_summary_"
        f"{OUTPUT_VERSION}.csv"
    )
)
L4_DISPERSION_OUTPUT = (
    TABLES / (
        "e15_l4_factorial_crossed_dispersion_"
        f"{OUTPUT_VERSION}.csv"
    )
)
L2_SUMMARY_OUTPUT = TABLES / (
    f"e15_l2_continuity_summary_{OUTPUT_VERSION}.csv"
)
ARTIFACT_MANIFEST_OUTPUT = (
    LOGS / (
        "e15_rule_construction_crossing_artifact_manifest_"
        f"{OUTPUT_VERSION}.json"
    )
)

CONSTRUCTION_SEEDS = (314159, 271828, 161803, 141421, 173205)
NEW_CONSTRUCTION_SEEDS = (271828, 161803, 141421, 173205)
ANCHOR_CONSTRUCTION_SEED = 314159
PIPELINE_SEEDS = (7, 42, 123, 2026, 3407)
BLOCKS = ("block_01", "block_02", "block_03")
ATTACKS = ("Gear", "RPM")
L2_ATTACKS = ("DoS", "Fuzzy", "Gear", "RPM")
SETTINGS = ("low", "medium", "high")
ENDPOINTS = ("exact_recall", "binary_recall")
ID_SCOPES = ("canonical", "shifted", "canonical_minus_shifted")
ATTACK_SCOPES = ("Gear", "RPM", "equal_macro")
L2_ATTACK_SCOPES = (*L2_ATTACKS, "equal_macro")
BITS = (0, 1)
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
MAIN_EFFECTS = frozenset({"P", "S", "D"})
INTERACTION_EFFECTS = frozenset({"PS", "PD", "SD", "PSD"})

EXPECTED_L4_DELTA_ROWS = 2_400
EXPECTED_L2_DELTA_ROWS = 900
EXPECTED_CROSSED_EFFECT_ROWS = 10_800
EXPECTED_HIERARCHICAL_ROWS = 2_160
EXPECTED_CROSSED_DISPERSION_ROWS = 288
EXPECTED_L2_SUMMARY_ROWS = 360

L4_DELTA_KEY = (
    "construction_seed",
    "pipeline_seed",
    "block_id",
    "attack",
    "id_stratum",
    "P",
    "S",
    "D",
)
CROSSED_EFFECT_KEY = (
    "construction_seed",
    "pipeline_seed",
    "block_id",
    "endpoint",
    "id_scope",
    "attack_scope",
    "effect",
)
HIERARCHY_KEY = (
    "endpoint",
    "id_scope",
    "attack_scope",
    "effect",
    "row_type",
    "unit_id",
)
DISPERSION_KEY = (
    "endpoint",
    "id_scope",
    "attack_scope",
    "effect",
    "scope",
)
L2_DELTA_KEY = (
    "construction_seed",
    "pipeline_seed",
    "block_id",
    "attack",
    "setting",
)
L2_SUMMARY_KEY = (
    "endpoint",
    "setting",
    "attack_scope",
    "row_type",
    "unit_id",
)

PREREG_SHA256 = (
    "5fedd45f2b81608e6358281e97843a0600f1d5676aacd4b1e909c64e08bc1905"
)
ANALYSIS_SCHEMA = "e15.rule_construction_analysis.v1"
MANIFEST_SCHEMA = "e15.rule_construction_artifact_manifest.v1"

RAW_EVALUATOR_OUTPUTS = (
    L4_NORMAL,
    L4_SCENARIO,
    L4_CELL,
    L4_DELTA,
    RUNTIME_CONTINUITY,
    ANCHOR_CONTINUITY,
    L2_SCENARIO,
    L2_DELTA,
    SCORE_LOG,
    RUN_RECORD,
)
ANALYSIS_OUTPUTS = (
    L4_EFFECTS_OUTPUT,
    L4_HIERARCHY_OUTPUT,
    L4_DISPERSION_OUTPUT,
    L2_SUMMARY_OUTPUT,
    ARTIFACT_MANIFEST_OUTPUT,
)
IMPLEMENTATION_PATHS = (
    "journal/scripts/generate_rule_construction_sensitivity.py",
    "journal/scripts/train_rule_construction_seed_crossing.py",
    "journal/scripts/evaluate_rule_construction_seed_crossing.py",
    "journal/scripts/analyze_rule_construction_seed_crossing.py",
    "journal/tests/test_rule_construction_seed_crossing.py",
    (
        "journal/experiments/e15_rule_construction_crossing/"
        "IMPLEMENTATION_AMENDMENT_2026-07-25_V2_PATH_IDENTITY.md"
    ),
    "journal/experiments/e15_rule_construction_crossing/PREREG.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, *, repo: Path = ROOT) -> str:
    return str(Path(os.path.abspath(path)).relative_to(Path(repo).absolute()))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _require_unique(
    frame: pd.DataFrame,
    key: Sequence[str],
    expected: int,
    label: str,
) -> None:
    _require_columns(frame, key, label)
    if len(frame) != int(expected):
        raise ValueError(
            f"{label} row count must be {expected}, observed {len(frame)}"
        )
    duplicate = frame.duplicated(list(key), keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, list(key)].head(5).to_dict("records")
        raise ValueError(f"{label} has duplicate unique keys: {sample}")


def _as_integer_column(frame: pd.DataFrame, column: str, label: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label}.{column} must contain finite integers")
    frame[column] = values.astype(np.int64)


def _require_finite_bounded(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    low: float,
    high: float,
    label: str,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if (
            not np.isfinite(values).all()
            or (values < low).any()
            or (values > high).any()
        ):
            raise ValueError(
                f"{label}.{column} must be finite in [{low},{high}]"
            )
        frame[column] = values


def _require_cartesian(
    frame: pd.DataFrame,
    key: Sequence[str],
    levels: Sequence[Sequence[Any]],
    label: str,
) -> None:
    expected = pd.MultiIndex.from_product(levels, names=list(key))
    observed = pd.MultiIndex.from_frame(frame[list(key)])
    if len(observed) != len(expected) or set(observed) != set(expected):
        missing = list(set(expected) - set(observed))[:5]
        extra = list(set(observed) - set(expected))[:5]
        raise ValueError(
            f"{label} Cartesian grid mismatch; missing={missing}, extra={extra}"
        )


def factorial_coefficients(
    effect: str,
) -> dict[tuple[int, int, int], float]:
    """Return the exact frozen E14 operator coefficients."""
    if effect not in EFFECTS:
        raise ValueError(f"unknown factorial effect: {effect}")
    cells = tuple(itertools.product(BITS, repeat=3))
    if effect == "P":
        return {cell: (1.0 if cell[0] else -1.0) / 4.0 for cell in cells}
    if effect == "S":
        return {cell: (1.0 if cell[1] else -1.0) / 4.0 for cell in cells}
    if effect == "D":
        return {cell: (1.0 if cell[2] else -1.0) / 4.0 for cell in cells}
    if effect == "PS":
        return {
            cell: (
                (1.0 if cell[0] else -1.0)
                * (1.0 if cell[1] else -1.0)
                / 2.0
            )
            for cell in cells
        }
    if effect == "PD":
        return {
            cell: (
                (1.0 if cell[0] else -1.0)
                * (1.0 if cell[2] else -1.0)
                / 2.0
            )
            for cell in cells
        }
    if effect == "SD":
        return {
            cell: (
                (1.0 if cell[1] else -1.0)
                * (1.0 if cell[2] else -1.0)
                / 2.0
            )
            for cell in cells
        }
    if effect == "PSD":
        return {
            cell: (
                (1.0 if cell[0] else -1.0)
                * (1.0 if cell[1] else -1.0)
                * (1.0 if cell[2] else -1.0)
            )
            for cell in cells
        }
    return {
        cell: (
            1.0
            if cell == (1, 1, 1)
            else (-1.0 if cell == (0, 0, 0) else 0.0)
        )
        for cell in cells
    }


def factorial_effect(
    values: Mapping[tuple[int, int, int], float],
    effect: str,
) -> float:
    """Apply one frozen factorial operator to exactly eight finite cells."""
    expected = set(itertools.product(BITS, repeat=3))
    if set(values) != expected:
        raise ValueError(
            "factorial values must contain exactly eight cells; "
            f"missing={sorted(expected - set(values))}, "
            f"extra={sorted(set(values) - expected)}"
        )
    array = np.asarray([values[cell] for cell in sorted(expected)], dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("factorial values must all be finite")
    coefficients = factorial_coefficients(effect)
    return float(
        sum(coefficients[cell] * float(values[cell]) for cell in expected)
    )


def t_summary_n4(values: Sequence[float]) -> dict[str, object]:
    """Registered construction-level summary with n=4 and df=3."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != 4:
        raise ValueError(
            f"construction summary requires exactly four values (n=4), "
            f"got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("n=4 construction summary values must be finite")
    mean = float(array.mean())
    result: dict[str, object] = {
        "n_available": 4,
        "mean": mean,
        "sd": float("nan"),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "test_p": float("nan"),
        "holm_input_p": float("nan"),
        "p_status": "not_tested",
        "status": "complete",
        "degenerate": False,
        "strict_positive": int((array > 0).sum()),
        "strict_negative": int((array < 0).sum()),
        "strict_zero": int((array == 0).sum()),
        "degrees_of_freedom": 3,
    }
    if np.all(array == array[0]):
        result.update(
            {
                "status": "degenerate_variance",
                "degenerate": True,
                "holm_input_p": 1.0,
                "p_status": "bookkeeping_p",
            }
        )
        return result
    sd = float(array.std(ddof=1))
    if not math.isfinite(sd) or sd <= 0.0:
        raise ValueError("nonidentical n=4 values produced invalid sample SD")
    critical = float(student_t.ppf(0.975, df=3))
    half_width = critical * sd / math.sqrt(4.0)
    statistic = mean / (sd / math.sqrt(4.0))
    test_p = float(2.0 * student_t.sf(abs(statistic), df=3))
    result.update(
        {
            "sd": sd,
            "ci_low": mean - half_width,
            "ci_high": mean + half_width,
            "test_p": test_p,
            "holm_input_p": test_p,
            "p_status": "inferential_p",
        }
    )
    return result


def _descriptive_summary(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("descriptive summary requires at least two finite values")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=len(array) - 1))
    half_width = critical * sd / math.sqrt(float(len(array)))
    return {
        "n_available": int(len(array)),
        "mean": mean,
        "sd": sd,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "strict_positive": int((array > 0).sum()),
        "strict_negative": int((array < 0).sum()),
        "strict_zero": int((array == 0).sum()),
        "degrees_of_freedom": int(len(array) - 1),
        "status": (
            "degenerate_descriptive_variance"
            if np.all(array == array[0])
            else "descriptive_complete"
        ),
        "degenerate": bool(np.all(array == array[0])),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Holm step-down adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Holm inputs must be a finite one-dimensional array")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Holm inputs must lie in [0,1]")
    order = np.argsort(p, kind="mergesort")
    sorted_adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        running = max(running, (len(p) - rank) * float(p[original_index]))
        sorted_adjusted[rank] = min(running, 1.0)
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = sorted_adjusted
    return adjusted


def validate_l4_delta(delta: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized, complete 2,400-row L4 delta table."""
    frame = delta.copy()
    _require_unique(
        frame,
        L4_DELTA_KEY,
        EXPECTED_L4_DELTA_ROWS,
        "E15 L4 delta",
    )
    _require_columns(
        frame, (*L4_DELTA_KEY, "n", *ENDPOINTS), "E15 L4 delta"
    )
    for column in ("construction_seed", "pipeline_seed", "P", "S", "D"):
        _as_integer_column(frame, column, "E15 L4 delta")
    for column in ("block_id", "attack", "id_stratum"):
        frame[column] = frame[column].astype(str)
    _require_finite_bounded(
        frame,
        ENDPOINTS,
        low=-1.0,
        high=1.0,
        label="E15 L4 delta",
    )
    _as_integer_column(frame, "n", "E15 L4 delta")
    if not frame["n"].eq(2000).all():
        raise ValueError("E15 L4 delta denominators must all equal 2000")
    # The registered endpoints are differences of integer correct counts over
    # the common denominator 2,000.  Recovering those numerators avoids
    # baseline-dependent floating cancellation from turning an exact zero
    # factorial contrast into a tiny, spuriously nondegenerate value.
    for endpoint in ENDPOINTS:
        scaled = frame[endpoint].to_numpy(float) * frame["n"].to_numpy(float)
        numerator = np.rint(scaled)
        if not np.allclose(scaled, numerator, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"E15 L4 delta {endpoint} is not integer-count realizable"
            )
        frame[f"_{endpoint}_count_delta"] = numerator.astype(np.int64)
    _require_cartesian(
        frame,
        L4_DELTA_KEY,
        (
            CONSTRUCTION_SEEDS,
            PIPELINE_SEEDS,
            BLOCKS,
            ATTACKS,
            ("canonical", "shifted"),
            BITS,
            BITS,
            BITS,
        ),
        "E15 L4 delta",
    )
    return frame.sort_values(list(L4_DELTA_KEY), kind="mergesort").reset_index(
        drop=True
    )


def _scoped_atomic_effect(
    indexed: pd.DataFrame,
    *,
    construction_seed: int,
    pipeline_seed: int,
    block_id: str,
    endpoint: str,
    id_scope: str,
    attack_scope: str,
    effect: str,
) -> float:
    attacks = ATTACKS if attack_scope == "equal_macro" else (attack_scope,)
    attack_values: list[Fraction] = []
    for attack in attacks:
        identity_values: dict[str, Fraction] = {}
        identities = (
            ("canonical", "shifted")
            if id_scope == "canonical_minus_shifted"
            else (id_scope,)
        )
        for identity in identities:
            cells: dict[tuple[int, int, int], Fraction] = {}
            for p, s, d in itertools.product(BITS, repeat=3):
                row = indexed.loc[
                    (
                        construction_seed,
                        pipeline_seed,
                        block_id,
                        attack,
                        identity,
                        p,
                        s,
                        d,
                    )
                ]
                if isinstance(row, pd.DataFrame):
                    raise ValueError("duplicate L4 delta key reached effect builder")
                cells[(p, s, d)] = Fraction(
                    int(row[f"_{endpoint}_count_delta"]),
                    int(row["n"]),
                )
            coefficients = factorial_coefficients(effect)
            identity_values[identity] = sum(
                (
                    Fraction(str(coefficients[cell])) * cells[cell]
                    for cell in cells
                ),
                Fraction(0, 1),
            )
        attack_values.append(
            identity_values["canonical"] - identity_values["shifted"]
            if id_scope == "canonical_minus_shifted"
            else identity_values[id_scope]
        )
    return float(sum(attack_values, Fraction(0, 1)) / len(attack_values))


def build_crossed_effects(delta: pd.DataFrame) -> pd.DataFrame:
    """Build all 10,800 block-resolved crossed factorial effects."""
    frame = validate_l4_delta(delta)
    indexed = frame.set_index(list(L4_DELTA_KEY))
    if not indexed.index.is_unique:
        raise ValueError("E15 L4 delta key is not unique after indexing")
    rows: list[dict[str, object]] = []
    for (
        construction,
        pipeline,
        block,
        endpoint,
        id_scope,
        attack_scope,
        effect,
    ) in itertools.product(
        CONSTRUCTION_SEEDS,
        PIPELINE_SEEDS,
        BLOCKS,
        ENDPOINTS,
        ID_SCOPES,
        ATTACK_SCOPES,
        EFFECTS,
    ):
        rows.append(
            {
                "construction_seed": construction,
                "construction_role": (
                    "fresh_known_construction_anchor"
                    if construction == ANCHOR_CONSTRUCTION_SEED
                    else "new_construction_primary"
                ),
                "pipeline_seed": pipeline,
                "block_id": block,
                "endpoint": endpoint,
                "id_scope": id_scope,
                "attack_scope": attack_scope,
                "effect": effect,
                "value": _scoped_atomic_effect(
                    indexed,
                    construction_seed=construction,
                    pipeline_seed=pipeline,
                    block_id=block,
                    endpoint=endpoint,
                    id_scope=id_scope,
                    attack_scope=attack_scope,
                    effect=effect,
                ),
                "input_attack_count": (
                    2 if attack_scope == "equal_macro" else 1
                ),
                "input_block_count": 1,
                "status": "complete",
            }
        )
    effects = pd.DataFrame(rows)
    _require_unique(
        effects,
        CROSSED_EFFECT_KEY,
        EXPECTED_CROSSED_EFFECT_ROWS,
        "E15 crossed effects",
    )
    if not np.isfinite(effects["value"].to_numpy(float)).all():
        raise ValueError("E15 crossed effects contain nonfinite values")
    return effects.sort_values(
        list(CROSSED_EFFECT_KEY), kind="mergesort"
    ).reset_index(drop=True)


def _blank_hierarchy_row(
    *,
    endpoint: str,
    id_scope: str,
    attack_scope: str,
    effect: str,
    row_type: str,
    unit_id: str,
    value: float,
    unit_role: str,
    status: str,
) -> dict[str, object]:
    return {
        "endpoint": endpoint,
        "id_scope": id_scope,
        "attack_scope": attack_scope,
        "effect": effect,
        "row_type": row_type,
        "unit_id": str(unit_id),
        "unit_role": unit_role,
        "value": float(value),
        "status": status,
        "n_available": 1,
        "mean": float("nan"),
        "sd": float("nan"),
        "minimum": float("nan"),
        "maximum": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "degrees_of_freedom": float("nan"),
        "test_p": float("nan"),
        "holm_input_p": float("nan"),
        "raw_p": float("nan"),
        "holm_adjusted_p": float("nan"),
        "p_status": "not_applicable",
        "holm_family": "",
        "degenerate": False,
        "unit_strict_sign": _strict_sign(value),
        "pipeline_margin_summary_repeated": False,
        "strict_positive": int(value > 0),
        "strict_negative": int(value < 0),
        "strict_zero": int(value == 0),
        "construction_resolved": False,
        "binary_construction_resolved": False,
        "construction_sign_heterogeneous": False,
        "pipeline_margin_directionally_concordant": False,
        "pipeline_margin_heterogeneous": False,
        "attack_heterogeneous": False,
        "shared_attack_direction_eligible": False,
        "endpoint_relation": "",
        "p_direction_status": "",
        "scientific_verdict": "",
    }


def _update_summary_row(
    row: dict[str, object],
    summary: Mapping[str, object],
    *,
    allow_test: bool,
) -> None:
    for key in (
        "n_available",
        "mean",
        "sd",
        "minimum",
        "maximum",
        "ci_low",
        "ci_high",
        "degrees_of_freedom",
        "status",
        "degenerate",
        "strict_positive",
        "strict_negative",
        "strict_zero",
    ):
        row[key] = summary[key]
    row["value"] = summary["mean"]
    if allow_test:
        for key in ("test_p", "holm_input_p", "p_status"):
            row[key] = summary[key]
    else:
        row["test_p"] = float("nan")
        row["holm_input_p"] = float("nan")
        row["raw_p"] = float("nan")
        row["holm_adjusted_p"] = float("nan")
        row["p_status"] = "not_tested_descriptive"


def build_hierarchical_summary(crossed_effects: pd.DataFrame) -> pd.DataFrame:
    """Build the registered 15-row hierarchy for all 144 effect scopes."""
    effects = crossed_effects.copy()
    _require_unique(
        effects,
        CROSSED_EFFECT_KEY,
        EXPECTED_CROSSED_EFFECT_ROWS,
        "E15 crossed effects",
    )
    _require_columns(effects, (*CROSSED_EFFECT_KEY, "value"), "crossed effects")
    if not np.isfinite(pd.to_numeric(effects["value"], errors="coerce")).all():
        raise ValueError("crossed effects values must be finite")
    _require_cartesian(
        effects,
        CROSSED_EFFECT_KEY,
        (
            CONSTRUCTION_SEEDS,
            PIPELINE_SEEDS,
            BLOCKS,
            ENDPOINTS,
            ID_SCOPES,
            ATTACK_SCOPES,
            EFFECTS,
        ),
        "E15 crossed effects",
    )
    grouped = effects.groupby(
        ["endpoint", "id_scope", "attack_scope", "effect"],
        sort=False,
        observed=True,
    )
    rows: list[dict[str, object]] = []
    for (endpoint, id_scope, attack_scope, effect), subset in grouped:
        if len(subset) != 75:
            raise ValueError(
                "each hierarchy combination requires 75 crossed rows, "
                f"got {len(subset)} for "
                f"{endpoint}/{id_scope}/{attack_scope}/{effect}"
            )
        construction_values: dict[int, float] = {}
        for construction in CONSTRUCTION_SEEDS:
            selected = subset[
                subset["construction_seed"].astype(int).eq(construction)
            ]
            if len(selected) != 15:
                raise ValueError("construction hierarchy input is incomplete")
            value = float(selected["value"].mean())
            construction_values[construction] = value
            rows.append(
                _blank_hierarchy_row(
                    endpoint=str(endpoint),
                    id_scope=str(id_scope),
                    attack_scope=str(attack_scope),
                    effect=str(effect),
                    row_type="construction",
                    unit_id=str(construction),
                    value=value,
                    unit_role=(
                        "fresh_known_construction_anchor"
                        if construction == ANCHOR_CONSTRUCTION_SEED
                        else "new_construction_primary"
                    ),
                    status="complete",
                )
            )
        pipeline_rows: list[dict[str, object]] = []
        pipeline_values: list[float] = []
        for pipeline in PIPELINE_SEEDS:
            selected = subset[
                subset["construction_seed"].astype(int).isin(
                    NEW_CONSTRUCTION_SEEDS
                )
                & subset["pipeline_seed"].astype(int).eq(pipeline)
            ]
            if len(selected) != 12:
                raise ValueError("pipeline_new4 hierarchy input is incomplete")
            pipeline_value = float(selected["value"].mean())
            pipeline_values.append(pipeline_value)
            pipeline_rows.append(
                _blank_hierarchy_row(
                    endpoint=str(endpoint),
                    id_scope=str(id_scope),
                    attack_scope=str(attack_scope),
                    effect=str(effect),
                    row_type="pipeline_new4",
                    unit_id=str(pipeline),
                    value=pipeline_value,
                    unit_role="pipeline_margin_descriptive_new4",
                    status="descriptive_complete",
                )
            )
        pipeline_summary = _descriptive_summary(pipeline_values)
        for pipeline_row in pipeline_rows:
            for key in (
                "n_available",
                "mean",
                "sd",
                "minimum",
                "maximum",
                "ci_low",
                "ci_high",
                "degrees_of_freedom",
                "degenerate",
                "strict_positive",
                "strict_negative",
                "strict_zero",
            ):
                pipeline_row[key] = pipeline_summary[key]
            pipeline_row["p_status"] = "not_tested_descriptive"
            pipeline_row["pipeline_margin_summary_repeated"] = True
            rows.append(pipeline_row)
        for block in BLOCKS:
            selected = subset[
                subset["construction_seed"].astype(int).isin(
                    NEW_CONSTRUCTION_SEEDS
                )
                & subset["block_id"].astype(str).eq(block)
            ]
            if len(selected) != 20:
                raise ValueError("block_new4 hierarchy input is incomplete")
            rows.append(
                _blank_hierarchy_row(
                    endpoint=str(endpoint),
                    id_scope=str(id_scope),
                    attack_scope=str(attack_scope),
                    effect=str(effect),
                    row_type="block_new4",
                    unit_id=block,
                    value=float(selected["value"].mean()),
                    unit_role="block_descriptive_new4",
                    status="descriptive_complete",
                )
            )
        new4_summary = t_summary_n4(
            [construction_values[seed] for seed in NEW_CONSTRUCTION_SEEDS]
        )
        new4_row = _blank_hierarchy_row(
            endpoint=str(endpoint),
            id_scope=str(id_scope),
            attack_scope=str(attack_scope),
            effect=str(effect),
            row_type="summary_new4",
            unit_id="new4",
            value=float(new4_summary["mean"]),
            unit_role="primary_new_constructions",
            status=str(new4_summary["status"]),
        )
        _update_summary_row(new4_row, new4_summary, allow_test=True)
        rows.append(new4_row)

        inclusive_summary = _descriptive_summary(
            [construction_values[seed] for seed in CONSTRUCTION_SEEDS]
        )
        inclusive_row = _blank_hierarchy_row(
            endpoint=str(endpoint),
            id_scope=str(id_scope),
            attack_scope=str(attack_scope),
            effect=str(effect),
            row_type="summary_anchor_inclusive5",
            unit_id="anchor_inclusive5",
            value=float(inclusive_summary["mean"]),
            unit_role="historical_anchor_descriptive",
            status=str(inclusive_summary["status"]),
        )
        _update_summary_row(
            inclusive_row,
            inclusive_summary,
            allow_test=False,
        )
        rows.append(inclusive_row)

    hierarchy = pd.DataFrame(rows)
    _require_unique(
        hierarchy,
        HIERARCHY_KEY,
        EXPECTED_HIERARCHICAL_ROWS,
        "E15 hierarchical summary",
    )
    expected_row_types = {
        "construction": 5,
        "pipeline_new4": 5,
        "block_new4": 3,
        "summary_new4": 1,
        "summary_anchor_inclusive5": 1,
    }
    base_combinations = (
        len(ENDPOINTS)
        * len(ID_SCOPES)
        * len(ATTACK_SCOPES)
        * len(EFFECTS)
    )
    for row_type, per_combination in expected_row_types.items():
        observed = int(hierarchy["row_type"].eq(row_type).sum())
        expected = base_combinations * per_combination
        if observed != expected:
            raise AssertionError(
                f"{row_type} hierarchy rows must be {expected}, got {observed}"
            )
    return apply_primary_inference(hierarchy)


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(
        lambda value: bool(value)
        if not isinstance(value, str)
        else value.strip().lower() in {"true", "1", "yes"}
    )


def apply_primary_inference(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact and binary seven-effect Holm families separately."""
    result = hierarchy.copy()
    defaults: dict[str, object] = {
        "raw_p": float("nan"),
        "holm_adjusted_p": float("nan"),
        "holm_family": "",
        "construction_resolved": False,
        "binary_construction_resolved": False,
        "construction_sign_heterogeneous": False,
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
    _require_columns(
        result,
        (
            "endpoint",
            "effect",
            "row_type",
            "id_scope",
            "attack_scope",
            "holm_input_p",
            "mean",
            "degenerate",
            "strict_positive",
            "strict_negative",
        ),
        "E15 hierarchy inference input",
    )
    primary_base = (
        result["row_type"].astype(str).eq("summary_new4")
        & result["id_scope"].astype(str).eq("canonical")
        & result["attack_scope"].astype(str).eq("equal_macro")
        & result["effect"].astype(str).isin(FACTORIAL_EFFECTS)
    )
    supporting_summary = (
        result["row_type"].astype(str).eq("summary_new4") & ~primary_base
    )
    for column in (
        "test_p",
        "holm_input_p",
        "raw_p",
        "holm_adjusted_p",
    ):
        if column in result:
            result.loc[supporting_summary, column] = np.nan
    if "p_status" in result:
        result.loc[
            supporting_summary, "p_status"
        ] = "not_tested_supporting"
    result.loc[supporting_summary, "holm_family"] = ""
    for endpoint, family, resolved_column in (
        (
            "exact_recall",
            "primary_exact_construction_new4",
            "construction_resolved",
        ),
        (
            "binary_recall",
            "companion_binary_construction_new4",
            "binary_construction_resolved",
        ),
    ):
        mask = primary_base & result["endpoint"].astype(str).eq(endpoint)
        selected = result.loc[mask].copy()
        if len(selected) != 7 or set(selected["effect"]) != set(
            FACTORIAL_EFFECTS
        ):
            raise ValueError(
                f"Holm family {family} requires exactly seven effects"
            )
        selected = selected.set_index("effect").loc[list(FACTORIAL_EFFECTS)]
        p_values = pd.to_numeric(
            selected["holm_input_p"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(p_values).all():
            raise ValueError(f"Holm family {family} has nonfinite inputs")
        adjusted = holm_adjust(p_values)
        for effect, adjusted_p in zip(
            FACTORIAL_EFFECTS, adjusted, strict=True
        ):
            row_index = result.index[
                mask & result["effect"].astype(str).eq(effect)
            ]
            if len(row_index) != 1:
                raise AssertionError("primary effect row identity is not unique")
            index = row_index[0]
            raw_p = float(result.at[index, "holm_input_p"])
            result.at[index, "raw_p"] = raw_p
            result.at[index, "holm_adjusted_p"] = float(adjusted_p)
            result.at[index, "holm_family"] = family
            mean = float(result.at[index, "mean"])
            degenerate = bool(
                _truthy_series(pd.Series([result.at[index, "degenerate"]])).iat[
                    0
                ]
            )
            same_sign = (
                int(result.at[index, "strict_positive"]) >= 3
                if mean > 0.0
                else (
                    int(result.at[index, "strict_negative"]) >= 3
                    if mean < 0.0
                    else False
                )
            )
            result.at[index, resolved_column] = bool(
                not degenerate and adjusted_p < 0.05 and same_sign
            )
            if (
                "minimum" in result
                and "maximum" in result
                and pd.notna(result.at[index, "minimum"])
                and pd.notna(result.at[index, "maximum"])
            ):
                result.at[index, "construction_sign_heterogeneous"] = bool(
                    float(result.at[index, "minimum"]) < 0.0
                    < float(result.at[index, "maximum"])
                )
    return result


def finite_grid_dispersion(
    frame: pd.DataFrame,
    *,
    scope: str,
) -> dict[str, object]:
    """Compute the preregistered descriptive finite-grid decomposition."""
    if scope not in {"new4", "anchor_inclusive5"}:
        raise ValueError(f"unknown dispersion scope: {scope}")
    constructions = (
        NEW_CONSTRUCTION_SEEDS
        if scope == "new4"
        else CONSTRUCTION_SEEDS
    )
    required = ("construction_seed", "pipeline_seed", "value")
    _require_columns(frame, required, f"{scope} finite grid")
    selected = frame[list(required)].copy()
    for column in ("construction_seed", "pipeline_seed"):
        _as_integer_column(selected, column, f"{scope} finite grid")
    values = pd.to_numeric(selected["value"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{scope} finite grid values must be finite")
    selected["value"] = values
    expected_rows = len(constructions) * len(PIPELINE_SEEDS)
    _require_unique(
        selected,
        ("construction_seed", "pipeline_seed"),
        expected_rows,
        f"{scope} finite grid",
    )
    _require_cartesian(
        selected,
        ("construction_seed", "pipeline_seed"),
        (constructions, PIPELINE_SEEDS),
        f"{scope} finite grid",
    )
    pivot = selected.pivot(
        index="construction_seed",
        columns="pipeline_seed",
        values="value",
    ).loc[list(constructions), list(PIPELINE_SEEDS)]
    array = pivot.to_numpy(float)
    grand = float(array.mean())
    construction_margins = array.mean(axis=1)
    pipeline_margins = array.mean(axis=0)
    residual = (
        array
        - construction_margins[:, None]
        - pipeline_margins[None, :]
        + grand
    )
    g_count, p_count = array.shape
    ss_construction = float(
        p_count * np.square(construction_margins - grand).sum()
    )
    ss_pipeline = float(
        g_count * np.square(pipeline_margins - grand).sum()
    )
    ss_interaction = float(np.square(residual).sum())
    ss_total = float(np.square(array - grand).sum())
    closure_error = float(
        abs(
            ss_total
            - ss_construction
            - ss_pipeline
            - ss_interaction
        )
    )
    tolerance = 1e-12 * max(1.0, abs(ss_total))
    if closure_error > tolerance:
        raise AssertionError(
            "finite-grid sums of squares do not close: "
            f"error={closure_error}, tolerance={tolerance}"
        )
    degenerate = ss_total == 0.0
    return {
        "scope": scope,
        "G": g_count,
        "P": p_count,
        "grid_cells": g_count * p_count,
        "grand_mean": grand,
        "construction_marginal_sample_sd": float(
            construction_margins.std(ddof=1)
        ),
        "pipeline_marginal_sample_sd": float(
            pipeline_margins.std(ddof=1)
        ),
        "rms_interaction": float(
            math.sqrt(ss_interaction / float(g_count * p_count))
        ),
        "ss_construction": ss_construction,
        "ss_pipeline": ss_pipeline,
        "ss_interaction": ss_interaction,
        "ss_total": ss_total,
        "share_construction": (
            float("nan") if degenerate else ss_construction / ss_total
        ),
        "share_pipeline": (
            float("nan") if degenerate else ss_pipeline / ss_total
        ),
        "share_interaction": (
            float("nan") if degenerate else ss_interaction / ss_total
        ),
        "closure_error": closure_error,
        "status": (
            "degenerate_total_dispersion" if degenerate else "complete"
        ),
        "interpretation": (
            "descriptive finite-grid dispersion; not a random-effects "
            "variance component or population variance explained"
        ),
    }


def build_crossed_dispersion(
    crossed_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Build two finite-grid summaries for every one of 144 effect scopes."""
    effects = crossed_effects.copy()
    _require_unique(
        effects,
        CROSSED_EFFECT_KEY,
        EXPECTED_CROSSED_EFFECT_ROWS,
        "E15 crossed effects",
    )
    averaged = (
        effects.groupby(
            [
                "construction_seed",
                "pipeline_seed",
                "endpoint",
                "id_scope",
                "attack_scope",
                "effect",
            ],
            as_index=False,
            sort=False,
            observed=True,
        )["value"]
        .mean()
    )
    if len(averaged) != 3_600:
        raise AssertionError("block averaging must produce 3,600 crossed cells")
    rows: list[dict[str, object]] = []
    for (
        endpoint,
        id_scope,
        attack_scope,
        effect,
    ), subset in averaged.groupby(
        ["endpoint", "id_scope", "attack_scope", "effect"],
        sort=False,
        observed=True,
    ):
        if len(subset) != 25:
            raise ValueError("dispersion combination requires a complete 5x5 grid")
        for scope in ("new4", "anchor_inclusive5"):
            selected = (
                subset[
                    subset["construction_seed"].astype(int).isin(
                        NEW_CONSTRUCTION_SEEDS
                    )
                ]
                if scope == "new4"
                else subset
            )
            rows.append(
                {
                    "endpoint": str(endpoint),
                    "id_scope": str(id_scope),
                    "attack_scope": str(attack_scope),
                    "effect": str(effect),
                    **finite_grid_dispersion(selected, scope=scope),
                }
            )
    dispersion = pd.DataFrame(rows)
    _require_unique(
        dispersion,
        DISPERSION_KEY,
        EXPECTED_CROSSED_DISPERSION_ROWS,
        "E15 crossed dispersion",
    )
    return dispersion.sort_values(
        list(DISPERSION_KEY), kind="mergesort"
    ).reset_index(drop=True)


def validate_l2_delta(delta: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized, complete 900-row L2 paired delta table."""
    frame = delta.copy()
    _require_unique(
        frame, L2_DELTA_KEY, EXPECTED_L2_DELTA_ROWS, "E15 L2 delta"
    )
    _require_columns(frame, (*L2_DELTA_KEY, *ENDPOINTS), "E15 L2 delta")
    for column in ("construction_seed", "pipeline_seed"):
        _as_integer_column(frame, column, "E15 L2 delta")
    for column in ("block_id", "attack", "setting"):
        frame[column] = frame[column].astype(str)
    _require_finite_bounded(
        frame,
        ENDPOINTS,
        low=-1.0,
        high=1.0,
        label="E15 L2 delta",
    )
    _require_cartesian(
        frame,
        L2_DELTA_KEY,
        (
            CONSTRUCTION_SEEDS,
            PIPELINE_SEEDS,
            BLOCKS,
            L2_ATTACKS,
            SETTINGS,
        ),
        "E15 L2 delta",
    )
    return frame.sort_values(list(L2_DELTA_KEY), kind="mergesort").reset_index(
        drop=True
    )


def _l2_scoped_crossed(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    setting: str,
    attack_scope: str,
) -> pd.DataFrame:
    attacks = (
        L2_ATTACKS if attack_scope == "equal_macro" else (attack_scope,)
    )
    selected = frame[
        frame["setting"].eq(setting) & frame["attack"].isin(attacks)
    ]
    expected = 5 * 5 * 3 * len(attacks)
    if len(selected) != expected:
        raise ValueError("L2 scoped crossed input is incomplete")
    crossed = (
        selected.groupby(
            ["construction_seed", "pipeline_seed"],
            as_index=False,
            sort=False,
            observed=True,
        )[endpoint]
        .mean()
        .rename(columns={endpoint: "value"})
    )
    _require_unique(
        crossed,
        ("construction_seed", "pipeline_seed"),
        25,
        "L2 scoped crossed values",
    )
    return crossed


def build_l2_summary(delta: pd.DataFrame) -> pd.DataFrame:
    """Build the registered 360-row, estimation-only L2 hierarchy."""
    frame = validate_l2_delta(delta)
    rows: list[dict[str, object]] = []
    for endpoint, setting, attack_scope in itertools.product(
        ENDPOINTS, SETTINGS, L2_ATTACK_SCOPES
    ):
        crossed = _l2_scoped_crossed(
            frame,
            endpoint=endpoint,
            setting=setting,
            attack_scope=attack_scope,
        )
        construction_values: dict[int, float] = {}
        for construction in CONSTRUCTION_SEEDS:
            value = float(
                crossed[
                    crossed["construction_seed"].astype(int).eq(construction)
                ]["value"].mean()
            )
            construction_values[construction] = value
            rows.append(
                {
                    "endpoint": endpoint,
                    "setting": setting,
                    "attack_scope": attack_scope,
                    "row_type": "construction",
                    "unit_id": str(construction),
                    "unit_role": (
                        "fresh_known_construction_anchor"
                        if construction == ANCHOR_CONSTRUCTION_SEED
                        else "new_construction_primary"
                    ),
                    "value": value,
                    "status": "complete",
                    "n_available": 1,
                    "mean": float("nan"),
                    "sd": float("nan"),
                    "minimum": float("nan"),
                    "maximum": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "strict_positive": int(value > 0),
                    "strict_negative": int(value < 0),
                    "strict_zero": int(value == 0),
                    "interval_interpretation": "",
                    "p_value": float("nan"),
                    "scientific_verdict": "",
                }
            )
        for pipeline in PIPELINE_SEEDS:
            selected = crossed[
                crossed["construction_seed"].astype(int).isin(
                    NEW_CONSTRUCTION_SEEDS
                )
                & crossed["pipeline_seed"].astype(int).eq(pipeline)
            ]
            value = float(selected["value"].mean())
            rows.append(
                {
                    "endpoint": endpoint,
                    "setting": setting,
                    "attack_scope": attack_scope,
                    "row_type": "pipeline_new4",
                    "unit_id": str(pipeline),
                    "unit_role": "pipeline_margin_descriptive_new4",
                    "value": value,
                    "status": "descriptive_complete",
                    "n_available": 4,
                    "mean": float("nan"),
                    "sd": float("nan"),
                    "minimum": float("nan"),
                    "maximum": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "strict_positive": int(value > 0),
                    "strict_negative": int(value < 0),
                    "strict_zero": int(value == 0),
                    "interval_interpretation": "",
                    "p_value": float("nan"),
                    "scientific_verdict": "",
                }
            )
        new4 = _descriptive_summary(
            [construction_values[seed] for seed in NEW_CONSTRUCTION_SEEDS]
        )
        inclusive = _descriptive_summary(
            [construction_values[seed] for seed in CONSTRUCTION_SEEDS]
        )
        for row_type, unit_id, unit_role, summary in (
            (
                "summary_new4",
                "new4",
                "new_construction_estimation_only",
                new4,
            ),
            (
                "summary_anchor_inclusive5",
                "anchor_inclusive5",
                "historical_anchor_descriptive",
                inclusive,
            ),
        ):
            ci_low = float(summary["ci_low"])
            ci_high = float(summary["ci_high"])
            interval_interpretation = (
                "inconclusive"
                if (
                    math.isfinite(ci_low)
                    and math.isfinite(ci_high)
                    and ci_low <= 0.0 <= ci_high
                )
                else "directional_nominal_interval"
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "setting": setting,
                    "attack_scope": attack_scope,
                    "row_type": row_type,
                    "unit_id": unit_id,
                    "unit_role": unit_role,
                    "value": float(summary["mean"]),
                    "status": str(summary["status"]),
                    "n_available": int(summary["n_available"]),
                    "mean": float(summary["mean"]),
                    "sd": float(summary["sd"]),
                    "minimum": float(summary["minimum"]),
                    "maximum": float(summary["maximum"]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "strict_positive": int(summary["strict_positive"]),
                    "strict_negative": int(summary["strict_negative"]),
                    "strict_zero": int(summary["strict_zero"]),
                    "interval_interpretation": interval_interpretation,
                    "p_value": float("nan"),
                    "scientific_verdict": "",
                }
            )
    summary = pd.DataFrame(rows)
    _require_unique(
        summary,
        L2_SUMMARY_KEY,
        EXPECTED_L2_SUMMARY_ROWS,
        "E15 L2 summary",
    )
    if summary["p_value"].notna().any():
        raise AssertionError("L2 summary must not contain p-values")
    if summary["scientific_verdict"].astype(str).ne("").any():
        raise AssertionError("L2 summary must not determine a verdict")
    return summary.sort_values(
        list(L2_SUMMARY_KEY), kind="mergesort"
    ).reset_index(drop=True)


def _primary_summary_rows(hierarchy: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    selected = hierarchy[
        hierarchy["row_type"].astype(str).eq("summary_new4")
        & hierarchy["endpoint"].astype(str).eq(endpoint)
        & hierarchy["id_scope"].astype(str).eq("canonical")
        & hierarchy["attack_scope"].astype(str).eq("equal_macro")
        & hierarchy["effect"].astype(str).isin(FACTORIAL_EFFECTS)
    ].copy()
    if len(selected) != 7 or set(selected["effect"]) != set(FACTORIAL_EFFECTS):
        raise ValueError(
            f"{endpoint} primary summary must contain all seven effects"
        )
    return selected.set_index("effect").loc[list(FACTORIAL_EFFECTS)]


def _strict_sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def _effect_names(values: Iterable[str]) -> list[str]:
    selected = set(values)
    return [effect for effect in FACTORIAL_EFFECTS if effect in selected]


def build_scientific_verdict(
    hierarchy: pd.DataFrame,
    *,
    construction_heterogeneous_effects: Iterable[str] | None = None,
    pipeline_heterogeneous_effects: Iterable[str] | None = None,
    attack_heterogeneous_effects: Iterable[str] | None = None,
) -> dict[str, object]:
    """Return exactly one verdict under the frozen E15 exact-family rules.

    Explicit heterogeneity lists are accepted for toy-data and audit callers.
    Canonical analysis leaves them as ``None`` and uses the already annotated
    primary summary columns.
    """
    exact = _primary_summary_rows(hierarchy, "exact_recall")
    _require_columns(
        exact,
        (
            "mean",
            "degenerate",
            "holm_adjusted_p",
            "construction_resolved",
        ),
        "E15 exact verdict rows",
    )
    resolved = {
        effect
        for effect, row in exact.iterrows()
        if bool(row["construction_resolved"])
    }
    significant = {
        effect
        for effect, row in exact.iterrows()
        if (
            not bool(row["degenerate"])
            and np.isfinite(float(row["holm_adjusted_p"]))
            and float(row["holm_adjusted_p"]) < 0.05
        )
    }
    degenerate = {
        effect for effect, row in exact.iterrows() if bool(row["degenerate"])
    }
    resolved_mains = resolved & MAIN_EFFECTS
    resolved_interactions = resolved & INTERACTION_EFFECTS
    if resolved_mains and resolved_interactions:
        base = "C-MAIN-AND-INTERACTION"
    elif resolved_mains:
        base = "C-MAIN"
    elif resolved_interactions:
        base = "C-INTERACTION"
    elif significant:
        base = "C-MIXED"
    else:
        base = "C-UNRESOLVED"

    def inferred_effects(
        explicit: Iterable[str] | None,
        column: str,
    ) -> list[str]:
        if explicit is not None:
            values = set(explicit)
        elif column in exact:
            values = {
                effect
                for effect, value in exact[column].items()
                if bool(value)
            }
        else:
            values = set()
        unknown = values - set(FACTORIAL_EFFECTS)
        if unknown:
            raise ValueError(
                f"unknown effects in {column}: {sorted(unknown)}"
            )
        return _effect_names(values)

    construction_heterogeneous = inferred_effects(
        construction_heterogeneous_effects,
        "construction_sign_heterogeneous",
    )
    pipeline_heterogeneous = inferred_effects(
        pipeline_heterogeneous_effects,
        "pipeline_margin_heterogeneous",
    )
    attack_heterogeneous = inferred_effects(
        attack_heterogeneous_effects,
        "attack_heterogeneous",
    )

    verdict = base
    if degenerate:
        verdict += "+DEGENERATE(" + ",".join(_effect_names(degenerate)) + ")"
    if construction_heterogeneous:
        verdict += (
            "+CONSTRUCTION-SIGN-HETEROGENEOUS("
            + ",".join(construction_heterogeneous)
            + ")"
        )
    if pipeline_heterogeneous:
        verdict += (
            "+PIPELINE-MARGIN-HETEROGENEOUS("
            + ",".join(pipeline_heterogeneous)
            + ")"
        )
    if attack_heterogeneous:
        verdict += (
            "+ATTACK-HETEROGENEOUS("
            + ",".join(attack_heterogeneous)
            + ")"
        )
    p_row = exact.loc["P"]
    p_sign = _strict_sign(float(p_row["mean"]))
    p_resolved = bool(p_row["construction_resolved"])
    p_pipeline_concordant = bool(
        p_row.get("pipeline_margin_directionally_concordant", False)
    )
    if p_resolved and p_sign < 0 and p_pipeline_concordant:
        p_status = "P-BOTH-MARGINS-CONCORDANT"
    elif p_resolved and p_sign < 0:
        p_status = "P-DIRECTION-REPEATED"
    elif p_resolved and p_sign > 0:
        p_status = "P-REVERSED"
    elif (
        not bool(p_row["degenerate"])
        and np.isfinite(float(p_row["holm_adjusted_p"]))
        and float(p_row["holm_adjusted_p"]) < 0.05
    ):
        p_status = "P-MIXED"
    else:
        p_status = "P-INCONCLUSIVE"
    return {
        "schema_version": "e15.scientific_verdict.v1",
        "verdict_count": 1,
        "base_verdict": base,
        "scientific_verdict": verdict,
        "p_direction_status": p_status,
        "resolved_exact_effects": _effect_names(resolved),
        "nondegenerate_holm_significant_exact_effects": _effect_names(
            significant
        ),
        "degenerate_exact_effects": _effect_names(degenerate),
        "construction_sign_heterogeneous_effects":
            construction_heterogeneous,
        "pipeline_margin_heterogeneous_effects": pipeline_heterogeneous,
        "attack_heterogeneous_effects": attack_heterogeneous,
        "inferential_units": {
            "new_constructions": 4,
            "historical_anchor_excluded": True,
            "pipeline_seeds_fixed": 5,
            "blocks_averaged": 3,
            "degrees_of_freedom": 3,
        },
        "claim_boundary": (
            "conditional construction-marginal association under one fixed "
            "Rule construction law, four selected new construction seeds, "
            "five selected detector-pipeline seeds, and shared real references"
        ),
    }


def annotate_scientific_findings(
    hierarchy: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach registered heterogeneity and endpoint-relation annotations."""
    result = apply_primary_inference(hierarchy)
    for column, default in (
        ("pipeline_margin_directionally_concordant", False),
        ("pipeline_margin_heterogeneous", False),
        ("attack_heterogeneous", False),
        ("shared_attack_direction_eligible", False),
        ("endpoint_relation", ""),
        ("p_direction_status", ""),
        ("scientific_verdict", ""),
    ):
        if column not in result:
            result[column] = default

    for endpoint in ENDPOINTS:
        summary = _primary_summary_rows(result, endpoint)
        resolved_column = (
            "construction_resolved"
            if endpoint == "exact_recall"
            else "binary_construction_resolved"
        )
        for effect, row in summary.iterrows():
            mask = (
                result["endpoint"].astype(str).eq(endpoint)
                & result["id_scope"].astype(str).eq("canonical")
                & result["attack_scope"].astype(str).eq("equal_macro")
                & result["effect"].astype(str).eq(effect)
            )
            pipeline_rows = result[
                mask & result["row_type"].astype(str).eq("pipeline_new4")
            ]
            if len(pipeline_rows) != 5:
                raise ValueError(
                    f"{endpoint}/{effect} requires five pipeline margins"
                )
            mean_sign = _strict_sign(float(row["mean"]))
            pipeline_signs = np.sign(
                pd.to_numeric(
                    pipeline_rows["value"], errors="coerce"
                ).to_numpy(float)
            )
            concordant_count = int((pipeline_signs == mean_sign).sum())
            concordant = bool(mean_sign != 0 and concordant_count >= 4)
            summary_index = result.index[
                mask & result["row_type"].astype(str).eq("summary_new4")
            ]
            if len(summary_index) != 1:
                raise AssertionError("primary summary identity is not unique")
            index = summary_index[0]
            result.at[
                index, "pipeline_margin_directionally_concordant"
            ] = concordant
            resolved = bool(row[resolved_column])
            result.at[index, "pipeline_margin_heterogeneous"] = bool(
                resolved and not concordant
            )

            attack_summaries: dict[str, pd.Series] = {}
            for attack in ATTACKS:
                selected = result[
                    result["row_type"].astype(str).eq("summary_new4")
                    & result["endpoint"].astype(str).eq(endpoint)
                    & result["id_scope"].astype(str).eq("canonical")
                    & result["attack_scope"].astype(str).eq(attack)
                    & result["effect"].astype(str).eq(effect)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"{endpoint}/{effect}/{attack} summary is not unique"
                    )
                attack_summaries[attack] = selected.iloc[0]
            attack_signs = {
                attack: _strict_sign(float(attack_row["mean"]))
                for attack, attack_row in attack_summaries.items()
            }
            result.at[index, "attack_heterogeneous"] = bool(
                resolved
                and attack_signs["Gear"] != 0
                and attack_signs["RPM"] != 0
                and attack_signs["Gear"] != attack_signs["RPM"]
            )
            shared = bool(
                resolved
                and mean_sign != 0
                and all(sign == mean_sign for sign in attack_signs.values())
                and all(
                    (
                        int(attack_row["strict_positive"]) >= 3
                        if mean_sign > 0
                        else int(attack_row["strict_negative"]) >= 3
                    )
                    for attack_row in attack_summaries.values()
                )
            )
            result.at[index, "shared_attack_direction_eligible"] = shared

    exact = _primary_summary_rows(result, "exact_recall")
    binary = _primary_summary_rows(result, "binary_recall")
    for effect in FACTORIAL_EFFECTS:
        exact_row = exact.loc[effect]
        binary_row = binary.loc[effect]
        exact_resolved = bool(exact_row["construction_resolved"])
        binary_resolved = bool(binary_row["binary_construction_resolved"])
        exact_sign = _strict_sign(float(exact_row["mean"]))
        binary_sign = _strict_sign(float(binary_row["mean"]))
        if exact_resolved:
            if binary_resolved and binary_sign == exact_sign:
                relation = "endpoint-directionally-concordant"
            elif binary_resolved and binary_sign == -exact_sign:
                relation = "endpoint-discordant"
            else:
                relation = "exact-only resolved under the registered families"
        elif binary_resolved:
            relation = "binary-only directional repetition"
        else:
            relation = "neither-endpoint-resolved"
        for endpoint in ENDPOINTS:
            mask = (
                result["row_type"].astype(str).eq("summary_new4")
                & result["endpoint"].astype(str).eq(endpoint)
                & result["id_scope"].astype(str).eq("canonical")
                & result["attack_scope"].astype(str).eq("equal_macro")
                & result["effect"].astype(str).eq(effect)
            )
            result.loc[mask, "endpoint_relation"] = relation

    verdict = build_scientific_verdict(result)
    primary_p = (
        result["row_type"].astype(str).eq("summary_new4")
        & result["endpoint"].astype(str).eq("exact_recall")
        & result["id_scope"].astype(str).eq("canonical")
        & result["attack_scope"].astype(str).eq("equal_macro")
        & result["effect"].astype(str).eq("P")
    )
    if int(primary_p.sum()) != 1:
        raise AssertionError("exactly one designated verdict row is required")
    result.loc[primary_p, "scientific_verdict"] = verdict[
        "scientific_verdict"
    ]
    result.loc[primary_p, "p_direction_status"] = verdict[
        "p_direction_status"
    ]
    if int(result["scientific_verdict"].astype(str).ne("").sum()) != 1:
        raise AssertionError("hierarchy must contain exactly one verdict value")
    return result, verdict


def allowed_evaluator_status_paths(*, repo: Path = ROOT) -> set[str]:
    """Exact untracked-path allowlist for completed evaluator outputs."""
    Path(repo).resolve()
    return {
        path.relative_to(ROOT).as_posix()
        for path in RAW_EVALUATOR_OUTPUTS
    }


def validate_analysis_source_status(
    status_records: Sequence[str],
    *,
    allowed_untracked: Iterable[str] = (),
) -> dict[str, object]:
    """Reject every change except exact untracked upstream output paths."""
    allowed = {Path(path).as_posix() for path in allowed_untracked}
    observed_allowed: list[str] = []
    rejected: list[str] = []
    for record in status_records:
        if not record:
            continue
        if len(record) < 4:
            rejected.append(record)
            continue
        code = record[:2]
        path = record[3:]
        if code == "??" and Path(path).as_posix() in allowed:
            observed_allowed.append(record)
        else:
            rejected.append(record)
    if rejected:
        raise RuntimeError(
            "T-INCOMPLETE: analysis source/status contains changes outside "
            f"the exact evaluator-output allowlist: {rejected[:10]}"
        )
    return {
        "tracked_index_head_clean": True,
        "observed_allowed_untracked_records": sorted(observed_allowed),
        "rejected_status_records": [],
    }


def _git_output(
    args: Sequence[str],
    *,
    repo: Path = ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=Path(repo),
        text=True,
        capture_output=True,
        check=check,
    )


def validate_analysis_source_snapshot(*, repo: Path = ROOT) -> dict[str, object]:
    """Bind analysis to one clean committed E15 implementation snapshot."""
    resolved_repo = Path(repo).resolve()
    head = _git_output(("rev-parse", "HEAD"), repo=resolved_repo).stdout.strip()
    if sha256_file(resolved_repo / PREREG.relative_to(ROOT)) != PREREG_SHA256:
        raise RuntimeError("T-INCOMPLETE: E15 preregistration hash mismatch")
    implementation: dict[str, object] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = resolved_repo / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"T-INCOMPLETE: missing/nonregular implementation {relative}"
            )
        tracked = _git_output(
            ("ls-files", "--error-unmatch", "--", relative),
            repo=resolved_repo,
            check=False,
        )
        if tracked.returncode != 0:
            raise RuntimeError(
                f"T-INCOMPLETE: implementation is not tracked: {relative}"
            )
        committed = _git_output(
            ("show", f"HEAD:{relative}"),
            repo=resolved_repo,
        ).stdout.encode("utf-8")
        committed_hash = hashlib.sha256(committed).hexdigest()
        worktree_hash = sha256_file(path)
        if committed_hash != worktree_hash:
            raise RuntimeError(
                f"T-INCOMPLETE: implementation differs from HEAD: {relative}"
            )
        implementation[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": worktree_hash,
            "matches_head": True,
        }
    status_raw = _git_output(
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        repo=resolved_repo,
    ).stdout
    status_records = [record for record in status_raw.split("\0") if record]
    allowed = allowed_evaluator_status_paths(repo=resolved_repo)
    status = validate_analysis_source_status(
        status_records, allowed_untracked=allowed
    )
    wisa = _git_output(
        ("status", "--porcelain=v1", "--", "wisa"),
        repo=resolved_repo,
    ).stdout
    if wisa.strip():
        raise RuntimeError("T-INCOMPLETE: frozen wisa/ is not clean")
    return {
        "source_commit": head,
        "implementation": implementation,
        "source_status": status,
        "wisa_clean": True,
        "allowed_untracked_evaluator_outputs": sorted(
            allowed_evaluator_status_paths(repo=resolved_repo)
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _validate_path_binding(
    binding: Mapping[str, Any],
    *,
    path: Path,
    label: str,
    expected_rows: int | None = None,
) -> dict[str, object]:
    if binding.get("path") != portable_path(path):
        raise ValueError(f"{label} path binding mismatch")
    observed_hash = sha256_file(path)
    if binding.get("sha256") != observed_hash:
        raise ValueError(f"{label} SHA-256 binding mismatch")
    if expected_rows is not None and int(binding.get("rows", -1)) != expected_rows:
        raise ValueError(
            f"{label} row binding must be {expected_rows}, "
            f"got {binding.get('rows')!r}"
        )
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": observed_hash,
        **({"rows": expected_rows} if expected_rows is not None else {}),
    }


def validate_evaluator_run(
    run: Mapping[str, Any],
    *,
    source_commit: str,
) -> dict[str, object]:
    """Validate the complete evaluator handoff and its raw-output bindings."""
    expected_scalars = {
        "schema_version": "e15.rule_construction_crossing_run.v1",
        "status": "T-PASS",
        "stage_status": "evaluation_complete",
        "scientific_verdict_status": "pending_registered_analyzer",
    }
    for field, expected in expected_scalars.items():
        if run.get(field) != expected:
            raise ValueError(
                f"evaluator run {field} mismatch: "
                f"expected={expected!r}, observed={run.get(field)!r}"
            )
    if run.get("source_commit") != source_commit:
        raise ValueError("evaluator run source commit differs from analysis HEAD")
    environment = run.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("evaluator run lacks evaluation environment binding")
    expected_environment = {
        "schema_version": "e15.evaluation_environment.v1",
        "stage": "score",
        "device": "cpu",
        "torch_threads": 8,
        "torch_interop_threads": 1,
        "batch_size": 4_096,
        "weights_only_checkpoint_validation": True,
        "model_forward_passes_performed": True,
    }
    for field, expected in expected_environment.items():
        if environment.get(field) != expected:
            raise ValueError(
                f"evaluator environment {field} mismatch: "
                f"expected={expected!r}, observed={environment.get(field)!r}"
            )
    for field in (
        "python",
        "python_executable",
        "platform",
        "numpy",
        "pandas",
        "torch",
    ):
        if not isinstance(environment.get(field), str) or not environment[field]:
            raise ValueError(f"evaluator environment lacks {field}")
    if run.get("environment_snapshot_unchanged") is not True:
        raise ValueError("evaluator environment changed during scoring")
    if run.get("rule_inference_started_after_runtime_continuity") is not True:
        raise ValueError(
            "evaluator run does not prove Rule inference followed continuity"
        )
    runtime_gate = run.get("runtime_continuity_gate")
    if not isinstance(runtime_gate, Mapping) or runtime_gate.get("status") not in {
        "PASS",
        "T-PASS",
    }:
        raise ValueError("evaluator runtime-continuity gate did not pass")
    if int(runtime_gate.get("rows", -1)) != 495:
        raise ValueError("runtime-continuity gate must bind 495 rows")
    prepare = run.get("prepare")
    if not isinstance(prepare, Mapping):
        raise ValueError("evaluator run lacks prepare-record binding")
    prepare_record = _validate_path_binding(
        prepare, path=PREPARE_RECORD, label="E15 prepare record"
    )
    outputs = run.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("evaluator run lacks outputs mapping")
    expected_outputs: dict[str, tuple[Path, int | None]] = {
        "l4_normal": (L4_NORMAL, 75),
        "l4_scenario": (L4_SCENARIO, 2_400),
        "l4_cell": (L4_CELL, 1_200),
        "l4_delta": (L4_DELTA, 2_400),
        "runtime_continuity": (RUNTIME_CONTINUITY, 495),
        "anchor_continuity": (ANCHOR_CONTINUITY, 495),
        "l2_scenario": (L2_SCENARIO, 1_170),
        "l2_delta": (L2_DELTA, 900),
        "log": (SCORE_LOG, None),
    }
    validated: dict[str, object] = {}
    for name, (path, rows) in expected_outputs.items():
        binding = outputs.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"evaluator run lacks {name} output binding")
        validated[name] = _validate_path_binding(
            binding, path=path, label=f"evaluator output {name}",
            expected_rows=rows,
        )
    unexpected_scientific = set(outputs) & {
        "l4_effects",
        "l4_hierarchy",
        "l4_dispersion",
        "l2_summary",
        "manifest",
    }
    if unexpected_scientific:
        raise ValueError(
            "evaluator improperly claimed analyzer-owned outputs: "
            f"{sorted(unexpected_scientific)}"
        )
    return {
        "run_record": {
            "path": portable_path(RUN_RECORD),
            "bytes": RUN_RECORD.stat().st_size,
            "sha256": sha256_file(RUN_RECORD),
        },
        "prepare_record": prepare_record,
        "outputs": validated,
        "runtime_continuity_gate": dict(runtime_gate),
        "environment": dict(environment),
        "source_commit": source_commit,
        "handoff_complete": True,
    }


def validate_runtime_continuity_table(frame: pd.DataFrame) -> dict[str, object]:
    """Require all 495 shared-real continuity differences to be integer zero."""
    key = (
        "row_scope",
        "pipeline_seed",
        "block_id",
        "attack",
        "id_stratum",
        "cell",
    )
    _require_unique(
        frame,
        key,
        495,
        "E15 real runtime continuity",
    )
    scopes = frame["row_scope"].astype(str)
    if int(scopes.eq("scenario").sum()) != 480 or int(
        scopes.eq("normal").sum()
    ) != 15:
        raise ValueError(
            "runtime continuity must contain 480 scenario and 15 normal rows"
        )
    difference_columns = (
        "exact_correct_difference",
        "binary_correct_difference",
        "normal_correct_difference",
    )
    _require_columns(frame, difference_columns, "runtime continuity")
    observed_values = 0
    for column in difference_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        available = numeric.dropna().to_numpy(float)
        observed_values += len(available)
        if (
            not np.isfinite(available).all()
            or not np.equal(available, np.floor(available)).all()
            or np.any(available != 0.0)
        ):
            raise RuntimeError(
                f"T-STOP-RUNTIME: nonzero/noninteger {column}"
            )
    if observed_values != 480 * 2 + 15:
        raise ValueError(
            "runtime continuity integer-difference availability mismatch"
        )
    return {
        "rows": 495,
        "scenario_rows": 480,
        "normal_rows": 15,
        "integer_difference_values": observed_values,
        "all_differences_zero": True,
        "status": "PASS",
    }


def validate_anchor_continuity_table(frame: pd.DataFrame) -> dict[str, object]:
    """Validate 495 fresh-anchor comparison rows and classify outcome drift."""
    key = (
        "row_scope",
        "pipeline_seed",
        "block_id",
        "attack",
        "id_stratum",
        "cell",
    )
    _require_unique(frame, key, 495, "E15 anchor outcome continuity")
    scopes = frame["row_scope"].astype(str)
    if int(scopes.eq("scenario").sum()) != 480 or int(
        scopes.eq("normal").sum()
    ) != 15:
        raise ValueError(
            "anchor continuity must contain 480 scenario and 15 normal rows"
        )
    numeric_difference_columns = [
        column
        for column in frame.columns
        if column.endswith("_difference")
    ]
    if not numeric_difference_columns:
        raise ValueError("anchor continuity has no difference columns")
    drift = False
    compared_values = 0
    for column in numeric_difference_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        values = numeric.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"anchor continuity {column} is nonfinite")
        compared_values += len(values)
        drift |= bool(np.any(values != 0.0))
    return {
        "rows": 495,
        "scenario_rows": 480,
        "normal_rows": 15,
        "compared_difference_values": compared_values,
        "status": (
            "anchor-outcome-drift"
            if drift
            else "anchor-outcome-identical"
        ),
        "changes_primary_new4_test": False,
    }


def _artifact_record(
    path: Path,
    *,
    role: str,
    rows: int | None = None,
    unique_key: Sequence[str] | None = None,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"T-INCOMPLETE: missing/nonregular artifact {path}")
    record: dict[str, object] = {
        "path": portable_path(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    if unique_key is not None:
        record["unique_key"] = list(unique_key)
    return record


def _lineage_artifact_paths() -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = [
        (PREREG, "e15_preregistration"),
        (V1_TRAINING_FAILURE, "e15_v1_training_failure_evidence"),
        (
            JOURNAL / "datasets" / "windows" / "train_windows.npz",
            "frozen_car_hacking_train_windows",
        ),
        (
            JOURNAL / "datasets" / "windows" / "val_windows.npz",
            "frozen_car_hacking_validation_windows",
        ),
        (
            JOURNAL / "datasets" / "windows" / "test_windows.npz",
            "frozen_car_hacking_test_windows",
        ),
        (
            JOURNAL / "datasets" / "synthetic" / "rule_based_windows.npz",
            "historical_rule_anchor_arrays",
        ),
        (
            ROOT / "wisa" / "scripts" / "generate_rule_based_synthetic.py",
            "frozen_wisa_rule_generator_reference",
        ),
        (
            JOURNAL / "scripts" / "generate_rule_based_synthetic_v2.py",
            "frozen_rule_generator_v2",
        ),
        (
            JOURNAL / "scripts" / "generator_protocol.py",
            "frozen_generator_protocol",
        ),
        (
            JOURNAL / "scripts" / "train_generator_extension_cnn.py",
            "frozen_generic_cnn_trainer",
        ),
        (
            JOURNAL / "scripts" / "lib_common.py",
            "frozen_training_common_library",
        ),
        (
            JOURNAL / "scripts" / "evaluate_evaluation_realization.py",
            "frozen_l2_evaluator",
        ),
        (
            TABLES / "evaluation_realization_blocks_e13_sampling_v2.csv",
            "frozen_l2_block_manifest",
        ),
        (
            JOURNAL / "experiments" / "e8_evaluation_realization"
            / "run_e13_sampling_v2.json",
            "frozen_l2_prospective_run",
        ),
        (
            EXPERIMENT / (
                f"pool_generation_preflight_{OUTPUT_VERSION}.json"
            ),
            "pool_generation_preflight",
        ),
        (POOL_GENERATION_RUN, "pool_generation_run"),
        (
            TABLES / (
                f"e15_rule_construction_pool_audit_{OUTPUT_VERSION}.csv"
            ),
            "pool_audit",
        ),
        (
            EXPERIMENT / f"training_preflight_{OUTPUT_VERSION}.json",
            "training_preflight",
        ),
        (TRAINING_RUN, "training_run"),
        (
            TABLES / (
                f"e15_rule_construction_sampling_audit_{OUTPUT_VERSION}.csv"
            ),
            "sampling_audit",
        ),
        (
            TABLES / (
                "e15_rule_construction_training_manifest_"
                f"{OUTPUT_VERSION}.csv"
            ),
            "training_manifest",
        ),
        (
            JOURNAL / "experiments" / "e14_l4_counterfactual_factorial"
            / "PREREG.md",
            "frozen_e14_preregistration",
        ),
        (
            JOURNAL / "experiments" / "e14_l4_counterfactual_factorial"
            / "prepare_v1.json",
            "frozen_e14_prepare",
        ),
        (
            JOURNAL / "experiments" / "e14_l4_counterfactual_factorial"
            / "matched_real_training_preflight_v1.json",
            "frozen_e14_real_training_preflight",
        ),
        (
            JOURNAL / "experiments" / "e14_l4_counterfactual_factorial"
            / "matched_real_training_run_v1.json",
            "frozen_e14_real_training_run",
        ),
        (
            JOURNAL / "experiments" / "e14_l4_counterfactual_factorial"
            / "matched_real_training_child_output_v1.log",
            "frozen_e14_real_training_transcript",
        ),
        (
            LOGS / "e14_l4_factorial_artifact_manifest_v1.json",
            "frozen_e14_artifact_manifest",
        ),
        (
            TABLES / "e14_l4_factorial_base_manifest_v1.csv",
            "frozen_e14_base_manifest",
        ),
        (
            TABLES / "e14_l4_factorial_latent_manifest_v1.csv",
            "frozen_e14_latent_manifest",
        ),
        (
            TABLES / "e14_l4_factorial_manipulation_checks_v1.csv",
            "frozen_e14_manipulation_checks",
        ),
        (
            TABLES / "e14_l4_factorial_normal_by_seed_v1.csv",
            "frozen_e14_normal_reference",
        ),
        (
            TABLES / "e14_l4_factorial_by_scenario_v1.csv",
            "frozen_e14_scenario_reference",
        ),
    ]
    for pipeline in PIPELINE_SEEDS:
        paths.extend(
            [
                (
                    JOURNAL
                    / "models"
                    / "generator_extension"
                    / (
                        "cnn_real_only_matchedsteps_e14_v1_"
                        f"seed{pipeline}.pt"
                    ),
                    "shared_e14_real_checkpoint",
                ),
                (
                    LOGS
                    / (
                        "train_cnn_real_only_matchedsteps_e14_v1_"
                        f"seed{pipeline}.log"
                    ),
                    "shared_e14_real_training_log",
                ),
            ]
        )
    already_bound = {path for path, _ in paths}
    for relative in IMPLEMENTATION_PATHS:
        implementation_path = ROOT / relative
        if implementation_path not in already_bound:
            paths.append((implementation_path, "e15_implementation_source"))
            already_bound.add(implementation_path)
    for construction in CONSTRUCTION_SEEDS:
        paths.extend(
            [
                (
                    JOURNAL
                    / "datasets"
                    / "synthetic"
                    / (
                        f"rule_cseed{construction}_windows_"
                        f"{OUTPUT_VERSION}.npz"
                    ),
                    "e15_rule_pool",
                ),
                (
                    TABLES / (
                        f"e15_rule_cseed{construction}_statistics_"
                        f"{OUTPUT_VERSION}.csv"
                    ),
                    "e15_pool_statistics",
                ),
                (
                    LOGS / (
                        f"e15_generate_rule_cseed{construction}_"
                        f"{OUTPUT_VERSION}.json"
                    ),
                    "e15_pool_generation_log",
                ),
            ]
        )
        for pipeline in PIPELINE_SEEDS:
            paths.extend(
                [
                    (
                        JOURNAL
                        / "models"
                        / "generator_extension"
                        / (
                            "cnn_rule_0p30_"
                            f"cseed{construction}_matchedsteps_e15_"
                            f"{OUTPUT_VERSION}_"
                            f"seed{pipeline}.pt"
                        ),
                        "e15_rule_checkpoint",
                    ),
                    (
                        LOGS
                        / (
                            "train_cnn_rule_0p30_"
                            f"cseed{construction}_matchedsteps_e15_"
                            f"{OUTPUT_VERSION}_"
                            f"seed{pipeline}.log"
                        ),
                        "e15_rule_training_log",
                    ),
                ]
            )
    return paths


def _sampling_vector_bindings(
    training_run: Mapping[str, Any],
) -> list[dict[str, object]]:
    bundles = training_run.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 25:
        raise ValueError("training run must contain 25 bundles")
    bindings: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            raise ValueError("training bundle is not an object")
        construction = int(bundle.get("construction_seed", -1))
        pipeline = int(bundle.get("pipeline_seed", -1))
        key = (construction, pipeline)
        if (
            key in seen
            or construction not in CONSTRUCTION_SEEDS
            or pipeline not in PIPELINE_SEEDS
        ):
            raise ValueError(f"invalid/duplicate training bundle seed key {key}")
        seen.add(key)
        digest = _require_sha256(
            bundle.get("sampling_index_sha256"),
            f"sampling vector {key}",
        )
        bindings.append(
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "sampling_seed": pipeline + 300,
                "sampling_index_sha256": digest,
                "requested": 78_645,
                "drawn": 78_645,
                "unique": 78_645,
                "repeated": 0,
            }
        )
    expected = {
        (construction, pipeline)
        for construction in CONSTRUCTION_SEEDS
        for pipeline in PIPELINE_SEEDS
    }
    if seen != expected:
        raise ValueError("training sampling-vector crossed grid is incomplete")
    return bindings


def publish_analysis_bundle_no_clobber(
    items: Sequence[tuple[Path, bytes]],
) -> None:
    """Atomically publish four tables and a final-manifest completion marker."""
    artifacts = [(Path(path), bytes(payload)) for path, payload in items]
    if len(artifacts) != 5:
        raise ValueError("E15 analysis publication requires exactly five items")
    targets = [path for path, _ in artifacts]
    if targets[-1] != ARTIFACT_MANIFEST_OUTPUT:
        raise ValueError("final artifact manifest must be the completion marker")
    if len(set(targets)) != len(targets):
        raise ValueError("analysis publication contains duplicate targets")
    for target in targets:
        absolute = Path(os.path.abspath(target))
        if not absolute.is_relative_to(JOURNAL.absolute()):
            raise ValueError(f"analysis output escapes journal/: {target}")
        if _lexists(target):
            raise FileExistsError(f"refusing to overwrite E15 output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".e15_analysis_staged_", dir=EXPERIMENT))
    staged: list[tuple[Path, Path]] = []
    linked: list[tuple[Path, Path]] = []
    success = False
    try:
        for index, (target, payload) in enumerate(artifacts):
            staged_path = stage / f"{index:02d}_{target.name}"
            with staged_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((staged_path, target))
        for staged_path, target in staged:
            try:
                os.link(staged_path, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"E15 analysis target appeared during publication: {target}"
                ) from exc
            linked.append((staged_path, target))
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        success = True
    except BaseException:
        for staged_path, target in reversed(linked):
            try:
                if (
                    _lexists(target)
                    and not target.is_symlink()
                    and os.path.samefile(staged_path, target)
                ):
                    target.unlink()
                    directory_fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        if success:
            shutil.rmtree(stage)


def _output_payload_record(
    path: Path,
    payload: bytes,
    *,
    rows: int,
    unique_key: Sequence[str],
) -> dict[str, object]:
    return {
        "path": portable_path(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": int(rows),
        "unique_key": list(unique_key),
    }


def build_final_manifest(
    *,
    source: Mapping[str, Any],
    evaluator_handoff: Mapping[str, Any],
    verdict: Mapping[str, Any],
    runtime_continuity: Mapping[str, Any],
    anchor_continuity: Mapping[str, Any],
    output_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Build the sole T-PASS + scientific-verdict completion marker."""
    if verdict.get("verdict_count") != 1 or not verdict.get(
        "scientific_verdict"
    ):
        raise ValueError("final manifest requires exactly one scientific verdict")
    evaluation_environment = evaluator_handoff.get("environment")
    if not isinstance(evaluation_environment, Mapping):
        raise ValueError(
            "final manifest requires a validated evaluation environment"
        )
    expected_output_names = {
        "crossed_effects",
        "hierarchical_summary",
        "crossed_dispersion",
        "l2_summary",
    }
    if set(output_records) != expected_output_names:
        raise ValueError(
            "final output record identities mismatch: "
            f"{sorted(output_records)}"
        )
    lineage = [
        _artifact_record(path, role=role)
        for path, role in _lineage_artifact_paths()
    ]
    training_run = _read_json_object(TRAINING_RUN)
    pool_generation_run = _read_json_object(POOL_GENERATION_RUN)
    sampling_vectors = _sampling_vector_bindings(training_run)
    pool_bindings = pool_generation_run.get("pools")
    if not isinstance(pool_bindings, list) or len(pool_bindings) != 5:
        raise ValueError("pool-generation run must bind exactly five pools")
    pool_seeds: set[int] = set()
    for binding in pool_bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("pool-generation binding is not an object")
        seed = int(binding.get("construction_seed", -1))
        if seed in pool_seeds or seed not in CONSTRUCTION_SEEDS:
            raise ValueError(f"invalid/duplicate pool binding seed {seed}")
        pool_seeds.add(seed)
        _require_sha256(
            binding.get("ordered_content_sha256"),
            f"pool {seed} ordered-content digest",
        )
    if pool_seeds != set(CONSTRUCTION_SEEDS):
        raise ValueError("pool-generation seed axis is incomplete")
    prepare = _read_json_object(PREPARE_RECORD)
    evaluator_run = _read_json_object(RUN_RECORD)
    transformations = {
        "stage_a_l4": prepare.get("l4_transformation"),
        "stage_a_l2": prepare.get("l2_transformation"),
        "stage_b": evaluator_run.get("transformations"),
    }
    if not isinstance(transformations["stage_a_l4"], Mapping) or not isinstance(
        transformations["stage_a_l2"], Mapping
    ):
        raise ValueError("prepare record lacks transformation identities")
    if (
        not isinstance(transformations["stage_b"], Mapping)
        or transformations["stage_b"].get("match_prepare") is not True
    ):
        raise ValueError("evaluator run does not prove Stage-B transform replay")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_schema_version": ANALYSIS_SCHEMA,
        "record_type": "e15_final_artifact_manifest",
        "created_utc": utc_now(),
        "technical_status": "T-PASS",
        "stage_status": "analysis_complete",
        "completion_marker": True,
        "source_commit": source["source_commit"],
        "scientific_verdict_status": "complete",
        "scientific_verdict_count": 1,
        "scientific_verdict": dict(verdict),
        "runtime_continuity": dict(runtime_continuity),
        "fresh_anchor_outcome_continuity": dict(anchor_continuity),
        "registered_counts": {
            "l4_delta_input_rows": EXPECTED_L4_DELTA_ROWS,
            "l2_delta_input_rows": EXPECTED_L2_DELTA_ROWS,
            "crossed_effect_rows": EXPECTED_CROSSED_EFFECT_ROWS,
            "hierarchical_rows": EXPECTED_HIERARCHICAL_ROWS,
            "crossed_dispersion_rows": EXPECTED_CROSSED_DISPERSION_ROWS,
            "l2_summary_rows": EXPECTED_L2_SUMMARY_ROWS,
            "primary_construction_inferential_n": 4,
            "primary_construction_degrees_of_freedom": 3,
            "exact_holm_family_size": 7,
            "binary_holm_family_size": 7,
        },
        "registered_unique_keys": {
            "crossed_effects": list(CROSSED_EFFECT_KEY),
            "hierarchical_summary": list(HIERARCHY_KEY),
            "crossed_dispersion": list(DISPERSION_KEY),
            "l2_summary": list(L2_SUMMARY_KEY),
        },
        "output_tables": {
            name: dict(record) for name, record in output_records.items()
        },
        "evaluator_handoff": dict(evaluator_handoff),
        "lineage_artifacts": lineage,
        "lineage_artifact_count": len(lineage),
        "pool_generation_bindings": pool_bindings,
        "training_bundle_bindings": training_run.get("bundles"),
        "sampling_vectors": sampling_vectors,
        "sampling_vector_count": len(sampling_vectors),
        "transformation_identities": transformations,
        "environments": {
            "pool_generation": pool_generation_run.get("environment"),
            "training_before": training_run.get("environment_before"),
            "training_after": training_run.get("environment_after"),
            "evaluation": dict(evaluation_environment),
        },
        "implementation": source["implementation"],
        "source_status": source["source_status"],
        "wisa_clean": source["wisa_clean"],
        "analysis_contract": {
            "shared_real_reference_counted_once": True,
            "blocks_averaged_before_construction_inference": True,
            "anchor_excluded_from_primary_n4": True,
            "exact_and_binary_holm_families_separate": True,
            "l2_p_values_or_verdicts": False,
            "dispersion_is_descriptive_finite_grid": True,
            "reduced_n_analysis": False,
            "external_dataset_evaluation": False,
            "figure_or_manuscript_update_performed": False,
        },
        "next_permitted_step": (
            "create a figure, update the paper-artifact manifest, and revise "
            "the manuscript only from this complete machine-readable verdict"
        ),
    }


def run_analysis() -> dict[str, object]:
    """Validate, analyze, and no-clobber publish the canonical E15 bundle."""
    for path in ANALYSIS_OUTPUTS:
        if _lexists(path):
            raise FileExistsError(f"refusing to overwrite E15 output: {path}")
    source = validate_analysis_source_snapshot()
    if not RUN_RECORD.is_file() or RUN_RECORD.is_symlink():
        raise RuntimeError("T-INCOMPLETE: missing/nonregular evaluator run record")
    evaluator_run = _read_json_object(RUN_RECORD)
    handoff = validate_evaluator_run(
        evaluator_run, source_commit=str(source["source_commit"])
    )

    l4_delta = pd.read_csv(L4_DELTA)
    l2_delta = pd.read_csv(L2_DELTA)
    runtime_frame = pd.read_csv(RUNTIME_CONTINUITY)
    anchor_frame = pd.read_csv(ANCHOR_CONTINUITY)
    runtime = validate_runtime_continuity_table(runtime_frame)
    anchor = validate_anchor_continuity_table(anchor_frame)
    crossed = build_crossed_effects(l4_delta)
    hierarchy = build_hierarchical_summary(crossed)
    hierarchy, verdict = annotate_scientific_findings(hierarchy)
    dispersion = build_crossed_dispersion(crossed)
    l2_summary = build_l2_summary(l2_delta)

    _require_unique(
        crossed,
        CROSSED_EFFECT_KEY,
        EXPECTED_CROSSED_EFFECT_ROWS,
        "final E15 crossed effects",
    )
    _require_unique(
        hierarchy,
        HIERARCHY_KEY,
        EXPECTED_HIERARCHICAL_ROWS,
        "final E15 hierarchy",
    )
    _require_unique(
        dispersion,
        DISPERSION_KEY,
        EXPECTED_CROSSED_DISPERSION_ROWS,
        "final E15 dispersion",
    )
    _require_unique(
        l2_summary,
        L2_SUMMARY_KEY,
        EXPECTED_L2_SUMMARY_ROWS,
        "final E15 L2 summary",
    )
    if int(hierarchy["scientific_verdict"].astype(str).ne("").sum()) != 1:
        raise AssertionError("analysis did not produce exactly one verdict row")

    final_source = validate_analysis_source_snapshot()
    if (
        final_source["source_commit"] != source["source_commit"]
        or final_source["implementation"] != source["implementation"]
    ):
        raise RuntimeError(
            "T-INCOMPLETE: source snapshot changed during analysis"
        )
    final_handoff = validate_evaluator_run(
        _read_json_object(RUN_RECORD),
        source_commit=str(final_source["source_commit"]),
    )
    if final_handoff != handoff:
        raise RuntimeError(
            "T-INCOMPLETE: evaluator handoff changed during analysis"
        )

    payloads = {
        "crossed_effects": (
            L4_EFFECTS_OUTPUT,
            _csv_bytes(crossed),
            EXPECTED_CROSSED_EFFECT_ROWS,
            CROSSED_EFFECT_KEY,
        ),
        "hierarchical_summary": (
            L4_HIERARCHY_OUTPUT,
            _csv_bytes(hierarchy),
            EXPECTED_HIERARCHICAL_ROWS,
            HIERARCHY_KEY,
        ),
        "crossed_dispersion": (
            L4_DISPERSION_OUTPUT,
            _csv_bytes(dispersion),
            EXPECTED_CROSSED_DISPERSION_ROWS,
            DISPERSION_KEY,
        ),
        "l2_summary": (
            L2_SUMMARY_OUTPUT,
            _csv_bytes(l2_summary),
            EXPECTED_L2_SUMMARY_ROWS,
            L2_SUMMARY_KEY,
        ),
    }
    output_records = {
        name: _output_payload_record(
            path,
            payload,
            rows=rows,
            unique_key=key,
        )
        for name, (path, payload, rows, key) in payloads.items()
    }
    manifest = build_final_manifest(
        source=final_source,
        evaluator_handoff=final_handoff,
        verdict=verdict,
        runtime_continuity=runtime,
        anchor_continuity=anchor,
        output_records=output_records,
    )
    manifest_payload = _json_bytes(manifest)
    publication = [
        (payloads[name][0], payloads[name][1])
        for name in (
            "crossed_effects",
            "hierarchical_summary",
            "crossed_dispersion",
            "l2_summary",
        )
    ]
    publication.append((ARTIFACT_MANIFEST_OUTPUT, manifest_payload))
    publish_analysis_bundle_no_clobber(publication)
    return manifest


def run_self_test() -> dict[str, object]:
    cube = {
        (p, s, d): (
            2 * p
            + 3 * s
            + 4 * d
            + 5 * p * s
            + 6 * p * d
            + 7 * s * d
            + 8 * p * s * d
        )
        for p, s, d in itertools.product(BITS, repeat=3)
    }
    expected = {
        "P": 9.5,
        "S": 11.0,
        "D": 12.5,
        "PS": 9.0,
        "PD": 10.0,
        "SD": 11.0,
        "PSD": 8.0,
        "corner_111_minus_000": 35.0,
    }
    observed = {
        effect: factorial_effect(cube, effect) for effect in EFFECTS
    }
    if observed != expected:
        raise AssertionError(
            f"factorial operator self-test failed: {observed}"
        )
    n4 = t_summary_n4([0.0, 0.1, 0.2, 0.3])
    degenerate = t_summary_n4([0.25] * 4)
    if (
        n4["degrees_of_freedom"] != 3
        or n4["n_available"] != 4
        or degenerate["holm_input_p"] != 1.0
        or not degenerate["degenerate"]
    ):
        raise AssertionError("n=4/degeneracy self-test failed")
    grid = pd.DataFrame(
        [
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "value": 10.0 * construction_index + pipeline_index,
            }
            for construction_index, construction in enumerate(
                NEW_CONSTRUCTION_SEEDS
            )
            for pipeline_index, pipeline in enumerate(PIPELINE_SEEDS)
        ]
    )
    dispersion = finite_grid_dispersion(grid, scope="new4")
    if not math.isclose(
        float(dispersion["ss_total"]), 2540.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError("finite-grid dispersion self-test failed")
    rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for effect in FACTORIAL_EFFECTS:
            rows.append(
                {
                    "endpoint": endpoint,
                    "effect": effect,
                    "row_type": "summary_new4",
                    "id_scope": "canonical",
                    "attack_scope": "equal_macro",
                    "holm_input_p": 1.0,
                    "mean": -0.1 if effect == "P" else 0.1,
                    "degenerate": False,
                    "strict_positive": 0 if effect == "P" else 4,
                    "strict_negative": 4 if effect == "P" else 0,
                }
            )
    inferred = apply_primary_inference(pd.DataFrame(rows))
    verdict = build_scientific_verdict(inferred)
    if (
        verdict["verdict_count"] != 1
        or verdict["scientific_verdict"] != "C-UNRESOLVED"
    ):
        raise AssertionError("scientific-verdict self-test failed")
    return {
        "status": "PASS",
        "factorial_effects": observed,
        "n4_df": n4["degrees_of_freedom"],
        "degenerate_bookkeeping_p": degenerate["holm_input_p"],
        "finite_grid_ss_total": dispersion["ss_total"],
        "scientific_verdict": verdict["scientific_verdict"],
        "canonical_outputs_created": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the preregistered E15 Rule construction-seed crossing."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "validate the complete evaluator handoff and publish the four "
            "registered analysis tables plus final manifest"
        ),
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="run model-free toy checks without creating canonical outputs",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    manifest = run_analysis()
    print(
        json.dumps(
            {
                "technical_status": manifest["technical_status"],
                "scientific_verdict": manifest["scientific_verdict"][
                    "scientific_verdict"
                ],
                "manifest": portable_path(ARTIFACT_MANIFEST_OUTPUT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
