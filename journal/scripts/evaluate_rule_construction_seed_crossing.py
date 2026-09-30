#!/usr/bin/env python3
"""Evaluate the preregistered E15 construction × pipeline seed crossing.

The frozen contract is
``journal/experiments/e15_rule_construction_crossing/PREREG.md``.  Stage A
(``--prepare-only``) is deliberately prediction-free.  PyTorch is imported
only by explicit checkpoint-structure or Stage-B scoring functions; importing
this module never imports torch or the E8 evaluator (which imports torch at
module scope).

The public validation and construction helpers are pure NumPy/pandas
functions.  They are intentionally usable with toy frames so the preparation,
continuity, Cartesian-product, and no-clobber gates can be tested without
opening a canonical model or publishing an E15 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import evaluate_l4_counterfactual_factorial as e14


ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "journal"
TABLES = JOURNAL / "results" / "tables"
LOGS = JOURNAL / "results" / "logs"
MODELS = JOURNAL / "models" / "generator_extension"
WINDOWS = JOURNAL / "datasets" / "windows"
SYNTHETIC = JOURNAL / "datasets" / "synthetic"
EXPERIMENT = JOURNAL / "experiments" / "e15_rule_construction_crossing"
OUTPUT_VERSION = "v2"

PREREG = EXPERIMENT / "PREREG.md"
AMENDMENT = (
    EXPERIMENT
    / "IMPLEMENTATION_AMENDMENT_2026-07-25_V2_PATH_IDENTITY.md"
)
V1_TRAINING_FAILURE = EXPERIMENT / "training_failure_v1.json"
E14_EXPERIMENT = JOURNAL / "experiments" / "e14_l4_counterfactual_factorial"
E14_PREPARE = E14_EXPERIMENT / "prepare_v1.json"
E14_SCENARIO = TABLES / "e14_l4_factorial_by_scenario_v1.csv"
E14_NORMAL = TABLES / "e14_l4_factorial_normal_by_seed_v1.csv"
BLOCK_MANIFEST = TABLES / "evaluation_realization_blocks_e13_sampling_v2.csv"
TRAIN_WINDOWS = WINDOWS / "train_windows.npz"
TEST_WINDOWS = WINDOWS / "test_windows.npz"
POOL_GENERATION_PREFLIGHT = (
    EXPERIMENT / f"pool_generation_preflight_{OUTPUT_VERSION}.json"
)
POOL_GENERATION_RUN = (
    EXPERIMENT / f"pool_generation_run_{OUTPUT_VERSION}.json"
)
TRAINING_PREFLIGHT = (
    EXPERIMENT / f"training_preflight_{OUTPUT_VERSION}.json"
)
TRAINING_RUN = EXPERIMENT / f"training_run_{OUTPUT_VERSION}.json"
PREPARE_RECORD = EXPERIMENT / f"prepare_{OUTPUT_VERSION}.json"
RUN_RECORD = EXPERIMENT / f"run_{OUTPUT_VERSION}.json"

CONSTRUCTION_SEEDS = (314159, 271828, 161803, 141421, 173205)
NEW_CONSTRUCTION_SEEDS = (271828, 161803, 141421, 173205)
ANCHOR_CONSTRUCTION_SEED = 314159
PIPELINE_SEEDS = (7, 42, 123, 2026, 3407)
BLOCKS = ("block_01", "block_02", "block_03")
EVAL_SEEDS = (20260711, 20260712, 20260713)
ATTACKS = ("Gear", "RPM")
L2_ATTACKS = ("DoS", "Fuzzy", "Gear", "RPM")
ID_STRATA = ("canonical", "shifted")
FACTOR_CELLS = tuple(
    (p, s, d) for p in (0, 1) for s in (0, 1) for d in (0, 1)
)
CELLS = tuple(f"{p}{s}{d}" for p, s, d in FACTOR_CELLS)
ENDPOINTS = ("exact_recall", "binary_recall")
EFFECTS = (
    "P", "S", "D", "PS", "PD", "SD", "PSD",
    "corner_111_minus_000",
)

EXPECTED_L4_NORMAL_ROWS = 75
EXPECTED_L4_SCENARIO_ROWS = 2_400
EXPECTED_L4_CELL_ROWS = 1_200
EXPECTED_L4_DELTA_ROWS = 2_400
EXPECTED_RUNTIME_CONTINUITY_ROWS = 495
EXPECTED_L2_SCENARIO_ROWS = 1_170
EXPECTED_L2_DELTA_ROWS = 900
EXPECTED_L4_TRANSFORM_INSTANCES = 192_000
EXPECTED_L2_TRANSFORM_INSTANCES = 36_000

L4_TRANSFORM_SHA256 = (
    "52b0ad4fe9365d6f0793d6d14e847bfe3e07f974aa737d861e75b4ea9228d56c"
)
L4_SCENARIO_KEY_SHA256 = (
    "be55a8259ea045cd9a8107de72d248d220c3d06e6b83f70f2202fabb8c0e1be5"
)
L4_SERIALIZATION = "e14-transform-instance-le-f4-v1"
L2_SERIALIZATION = "e15-l2-keyed-instance-le-f4-v1"

CONTINUITY_KEY = (
    "row_scope",
    "pipeline_seed",
    "block_id",
    "attack",
    "id_stratum",
    "cell",
)
EXPECTED_L4_ROWS = {
    "normal": 75,
    "scenario": 2_400,
    "cell": 1_200,
    "delta": 2_400,
    "crossed_effects": 10_800,
    "hierarchical": 2_160,
    "crossed_dispersion": 288,
    "runtime_continuity": 495,
    "anchor_continuity": 495,
}
EXPECTED_L2_ROWS = {
    "by_scenario": 1_170,
    "delta": 900,
    "summary": 360,
}

FROZEN_HASHES = {
    PREREG:
        "5fedd45f2b81608e6358281e97843a0600f1d5676aacd4b1e909c64e08bc1905",
    AMENDMENT:
        "e747c5c8877a5dfd4e4d6b5b23c54b0ec23f122cd40f145011fa33cb622afd12",
    V1_TRAINING_FAILURE:
        "b5e4cde58d0e6ba207bf5ba91bf39a150c0485be66a27171810a83e3daadb828",
    TRAIN_WINDOWS:
        "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4",
    TEST_WINDOWS:
        "4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7",
    BLOCK_MANIFEST:
        "9e18b3844e20f259938c732a10bc0251767f0d2eaae9c731ba8da7ef087f5efb",
    E14_SCENARIO:
        "833ee70d7dd74e75d402783ec0956abbe065a083b62ec44b09c3163361392843",
    E14_NORMAL:
        "84f6ef7f6722a7fda7d069bb38aa05744144afef75d6537bba4581a42e257849",
    E14_PREPARE:
        "5945790bee2e0bf11739901e0fdd9237d499353d9d8ead6b34de06fe0311a16e",
}

OUTPUT_PATHS = {
    "l4_normal": TABLES / (
        f"e15_l4_factorial_normal_by_crossed_seed_{OUTPUT_VERSION}.csv"
    ),
    "l4_scenario": TABLES / (
        f"e15_l4_factorial_by_scenario_{OUTPUT_VERSION}.csv"
    ),
    "l4_cell": TABLES / (
        f"e15_l4_factorial_by_cell_{OUTPUT_VERSION}.csv"
    ),
    "l4_delta": TABLES / (
        f"e15_l4_factorial_delta_{OUTPUT_VERSION}.csv"
    ),
    "l4_effects":
        TABLES / (
            "e15_l4_factorial_effects_by_crossed_cell_"
            f"{OUTPUT_VERSION}.csv"
        ),
    "l4_hierarchy":
        TABLES / (
            "e15_l4_factorial_hierarchical_summary_"
            f"{OUTPUT_VERSION}.csv"
        ),
    "l4_dispersion":
        TABLES / (
            "e15_l4_factorial_crossed_dispersion_"
            f"{OUTPUT_VERSION}.csv"
        ),
    "runtime_continuity":
        TABLES / f"e15_real_runtime_continuity_{OUTPUT_VERSION}.csv",
    "anchor_continuity":
        TABLES / f"e15_anchor_outcome_continuity_{OUTPUT_VERSION}.csv",
    "l2_scenario": TABLES / (
        f"e15_l2_continuity_by_scenario_{OUTPUT_VERSION}.csv"
    ),
    "l2_delta": TABLES / (
        f"e15_l2_continuity_delta_{OUTPUT_VERSION}.csv"
    ),
    "l2_summary": TABLES / (
        f"e15_l2_continuity_summary_{OUTPUT_VERSION}.csv"
    ),
    "log": LOGS / f"e15_rule_construction_crossing_{OUTPUT_VERSION}.log",
    "manifest":
        LOGS / (
            "e15_rule_construction_crossing_artifact_manifest_"
            f"{OUTPUT_VERSION}.json"
        ),
    "run_record": RUN_RECORD,
}

L2_SCENARIOS = (
    {"attack_type": "DoS", "setting": "low", "burst_len": 24,
     "step": 3, "amplitude": 0.25},
    {"attack_type": "DoS", "setting": "medium", "burst_len": 48,
     "step": 2, "amplitude": 0.50},
    {"attack_type": "DoS", "setting": "high", "burst_len": 96,
     "step": 1, "amplitude": 1.00},
    {"attack_type": "Fuzzy", "setting": "low", "burst_len": 24,
     "step": 4, "amplitude": 0.25},
    {"attack_type": "Fuzzy", "setting": "medium", "burst_len": 48,
     "step": 3, "amplitude": 0.50},
    {"attack_type": "Fuzzy", "setting": "high", "burst_len": 96,
     "step": 1, "amplitude": 1.00},
    {"attack_type": "Gear", "setting": "low", "burst_len": 24,
     "step": 4, "amplitude": 0.25},
    {"attack_type": "Gear", "setting": "medium", "burst_len": 48,
     "step": 2, "amplitude": 0.50},
    {"attack_type": "Gear", "setting": "high", "burst_len": 96,
     "step": 1, "amplitude": 1.00},
    {"attack_type": "RPM", "setting": "low", "burst_len": 24,
     "step": 4, "amplitude": 0.25},
    {"attack_type": "RPM", "setting": "medium", "burst_len": 48,
     "step": 2, "amplitude": 0.50},
    {"attack_type": "RPM", "setting": "high", "burst_len": 96,
     "step": 1, "amplitude": 1.00},
)
ATTACK_LABEL = {"DoS": 1, "Fuzzy": 2, "Gear": 3, "RPM": 4}


@dataclass
class InferenceLedger:
    """Runtime ordering evidence; only prediction functions may mutate it."""

    inference_calls: int = 0
    prediction_rows: int = 0
    rule_checkpoint_loads: int = 0
    runtime_gate_passed: bool = False

    def record_prediction(self, rows: int) -> None:
        if rows < 0:
            raise ValueError("prediction rows cannot be negative")
        self.inference_calls += 1
        self.prediction_rows += int(rows)

    def record_rule_load(self) -> None:
        if not self.runtime_gate_passed:
            raise RuntimeError(
                "T-STOP-RUNTIME: Rule inference is prohibited before the "
                "shared-real continuity gate"
            )
        self.rule_checkpoint_loads += 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def evaluation_environment_record(
    *,
    stage: str,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Capture the concrete E15 evaluation runtime without import side effects.

    ``torch`` remains a lazy dependency: importing this evaluator does not
    import it, while both explicit Stage-A weights-only validation and Stage-B
    scoring bind the actual installed/runtime values they use.
    """
    if stage not in {"prepare_only", "score"}:
        raise ValueError(f"unknown E15 evaluation environment stage: {stage}")
    if torch_module is None:
        import torch as torch_module

    return {
        "schema_version": "e15.evaluation_environment.v1",
        "stage": stage,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": str(torch_module.__version__),
        "device": "cpu",
        "torch_threads": int(torch_module.get_num_threads()),
        "torch_interop_threads": int(
            torch_module.get_num_interop_threads()
        ),
        "batch_size": int(e14.SCORING_BATCH_SIZE),
        "weights_only_checkpoint_validation": True,
        "model_forward_passes_performed": stage == "score",
    }


def validate_evaluation_environment_record(
    record: Mapping[str, Any],
    *,
    expected_stage: str,
) -> dict[str, Any]:
    """Validate a complete, stage-specific E15 environment binding."""
    expected = {
        "schema_version": "e15.evaluation_environment.v1",
        "stage": expected_stage,
        "device": "cpu",
        "batch_size": int(e14.SCORING_BATCH_SIZE),
        "weights_only_checkpoint_validation": True,
        "model_forward_passes_performed": expected_stage == "score",
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise RuntimeError(
                "T-STOP-MANIPULATION: evaluation environment "
                f"{field} mismatch"
            )
    for field in (
        "python",
        "python_executable",
        "platform",
        "numpy",
        "pandas",
        "torch",
    ):
        if not isinstance(record.get(field), str) or not record[field]:
            raise RuntimeError(
                "T-STOP-MANIPULATION: evaluation environment "
                f"lacks {field}"
            )
    for field in ("torch_threads", "torch_interop_threads"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                "T-STOP-MANIPULATION: evaluation environment "
                f"has invalid {field}"
            )
    return dict(record)


def assert_score_environment_unchanged(
    environment: Mapping[str, Any],
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Recollect and require the identical Stage-B runtime binding."""
    final_environment = evaluation_environment_record(
        stage="score", torch_module=torch_module
    )
    if final_environment != dict(environment):
        raise RuntimeError(
            "T-INCOMPLETE: evaluation environment changed during Stage B"
        )
    return {
        "environment_snapshot_unchanged": True,
        "environment_sha256": hashlib.sha256(
            _json_bytes(final_environment)
        ).hexdigest(),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    return str(Path(os.path.abspath(path)).relative_to(ROOT.absolute()))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def publish_bundle_no_clobber(
    items: Sequence[tuple[Path, bytes]],
) -> None:
    """Publish a same-filesystem bundle with exclusive hard links.

    The final item is the completion marker.  A synchronous failure removes
    only links created by this call; the isolated stage is retained on failure
    and removed after success.
    """
    artifacts = [(Path(path), bytes(payload)) for path, payload in items]
    targets = [path for path, _ in artifacts]
    if not artifacts:
        raise ValueError("empty publication bundle")
    if len(targets) != len(set(targets)):
        raise ValueError("bundle contains duplicate target paths")
    for target in targets:
        absolute = Path(os.path.abspath(target))
        if not absolute.is_relative_to(JOURNAL.absolute()):
            raise ValueError(f"E15 output escapes journal/: {target}")
        if _lexists(target):
            raise FileExistsError(f"refusing to overwrite E15 output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".e15_staged_", dir=EXPERIMENT))
    staged: list[tuple[Path, Path]] = []
    linked: list[tuple[Path, Path]] = []
    success = False
    try:
        for ordinal, (target, payload) in enumerate(artifacts):
            staged_path = stage / f"{ordinal:02d}-{target.name}"
            with staged_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((staged_path, target))
        for _, target in staged:
            if _lexists(target):
                raise FileExistsError(
                    f"E15 output appeared during publication: {target}"
                )
        for staged_path, target in staged:
            os.link(staged_path, target)
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
                target_stat = target.stat(follow_symlinks=False)
                stage_stat = staged_path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (
                target_stat.st_dev == stage_stat.st_dev
                and target_stat.st_ino == stage_stat.st_ino
            ):
                target.unlink()
        raise
    finally:
        if success:
            shutil.rmtree(stage)


def assert_prepare_only_inference_guard(
    inference_calls: int = 0,
    prediction_rows: int = 0,
) -> dict[str, Any]:
    """Fail Stage A if any prediction or forward-evaluation evidence exists."""
    calls = int(inference_calls)
    rows = int(prediction_rows)
    if calls != 0 or rows != 0:
        raise RuntimeError(
            "T-STOP-MANIPULATION: --prepare-only observed model inference "
            f"(calls={calls}, prediction_rows={rows})"
        )
    return {
        "inference_calls": 0,
        "prediction_rows": 0,
        "prepare_only_prediction_free": True,
    }


def require_unique(
    frame: pd.DataFrame,
    key: Sequence[str],
    expected: int | None,
    label: str,
) -> int:
    """Require columns, optional exact row count, and a duplicate-free key."""
    missing = [column for column in key if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing key columns: {missing}")
    if expected is not None and len(frame) != int(expected):
        raise ValueError(
            f"{label} row count mismatch: expected={expected}, got={len(frame)}"
        )
    duplicate = frame.duplicated(list(key), keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, list(key)].head(5).to_dict("records")
        raise ValueError(f"{label} duplicate key rows: {sample}")
    return int(len(frame))


def validate_transform_identity(
    instances: int,
    transform_sha256: str,
    scenario_key_sha256: str,
) -> dict[str, Any]:
    """Validate the complete frozen E14 L4 transformation identity."""
    observed = (
        int(instances), str(transform_sha256), str(scenario_key_sha256)
    )
    expected = (
        EXPECTED_L4_TRANSFORM_INSTANCES,
        L4_TRANSFORM_SHA256,
        L4_SCENARIO_KEY_SHA256,
    )
    if observed != expected:
        raise RuntimeError(
            "T-STOP-MANIPULATION: L4 transformation identity mismatch: "
            f"expected={expected}, observed={observed}"
        )
    return {
        "instances": expected[0],
        "transformation_sha256": expected[1],
        "scenario_key_sha256": expected[2],
        "serialization": L4_SERIALIZATION,
        "status": "T-PASS",
    }


def _pipeline_column(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    if "pipeline_seed" not in result:
        if "seed" not in result:
            raise ValueError(f"{label} lacks pipeline_seed/seed")
        result = result.rename(columns={"seed": "pipeline_seed"})
    result["pipeline_seed"] = pd.to_numeric(
        result["pipeline_seed"], errors="raise"
    ).astype(int)
    return result


def _select_arm(frame: pd.DataFrame, arm: str, label: str) -> pd.DataFrame:
    if "arm" not in frame:
        return frame.copy()
    selected = frame[frame["arm"].astype(str).eq(arm)].copy()
    if selected.empty:
        raise ValueError(f"{label} has no arm={arm!r} rows")
    return selected


def _scenario_count_frame(
    frame: pd.DataFrame, *, arm: str, label: str
) -> pd.DataFrame:
    result = _pipeline_column(_select_arm(frame, arm, label), label)
    required = (
        "pipeline_seed", "block_id", "attack", "id_stratum",
        "P", "S", "D", "n", "exact_correct", "binary_correct",
    )
    missing = [column for column in required if column not in result]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    result = result.copy()
    result["cell"] = (
        result["P"].astype(int).astype(str)
        + result["S"].astype(int).astype(str)
        + result["D"].astype(int).astype(str)
    )
    key = (
        "pipeline_seed", "block_id", "attack", "id_stratum", "cell"
    )
    require_unique(result, key, None, label)
    return result


def _normal_count_frame(
    frame: pd.DataFrame, *, arm: str, label: str
) -> pd.DataFrame:
    result = _pipeline_column(_select_arm(frame, arm, label), label)
    required = ("pipeline_seed", "block_id", "n", "normal_correct")
    missing = [column for column in required if column not in result]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    require_unique(result, ("pipeline_seed", "block_id"), None, label)
    return result


def build_real_runtime_continuity(
    fresh_scenario: pd.DataFrame,
    frozen_scenario: pd.DataFrame,
    fresh_normal: pd.DataFrame,
    frozen_normal: pd.DataFrame,
    *,
    fresh_arm: str = "real_ms",
    frozen_arm: str = "real_ms",
) -> pd.DataFrame:
    """Build the integer-count continuity table used before Rule inference."""
    fresh_s = _scenario_count_frame(
        fresh_scenario, arm=fresh_arm, label="fresh scenario"
    )
    frozen_s = _scenario_count_frame(
        frozen_scenario, arm=frozen_arm, label="frozen scenario"
    )
    scenario_key = [
        "pipeline_seed", "block_id", "attack", "id_stratum", "cell"
    ]
    joined = fresh_s.merge(
        frozen_s,
        on=scenario_key,
        how="outer",
        suffixes=("_fresh", "_frozen"),
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("T-STOP-RUNTIME: unpaired scenario continuity rows")
    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        record = row._asdict()
        n_fresh = int(record["n_fresh"])
        n_frozen = int(record["n_frozen"])
        if n_fresh != n_frozen or n_fresh <= 0:
            raise RuntimeError("T-STOP-RUNTIME: scenario denominator drift")
        exact_diff = (
            int(record["exact_correct_fresh"])
            - int(record["exact_correct_frozen"])
        )
        binary_diff = (
            int(record["binary_correct_fresh"])
            - int(record["binary_correct_frozen"])
        )
        rows.append({
            "row_scope": "scenario",
            **{column: record[column] for column in scenario_key},
            "n": n_fresh,
            "fresh_exact_correct": int(record["exact_correct_fresh"]),
            "frozen_exact_correct": int(record["exact_correct_frozen"]),
            "exact_correct_difference": exact_diff,
            "fresh_binary_correct": int(record["binary_correct_fresh"]),
            "frozen_binary_correct": int(record["binary_correct_frozen"]),
            "binary_correct_difference": binary_diff,
            "fresh_exact_recall":
                int(record["exact_correct_fresh"]) / n_fresh,
            "frozen_exact_recall":
                int(record["exact_correct_frozen"]) / n_frozen,
            "exact_recall_difference": exact_diff / n_fresh,
            "fresh_binary_recall":
                int(record["binary_correct_fresh"]) / n_fresh,
            "frozen_binary_recall":
                int(record["binary_correct_frozen"]) / n_frozen,
            "binary_recall_difference": binary_diff / n_fresh,
            "fresh_normal_correct": pd.NA,
            "frozen_normal_correct": pd.NA,
            "normal_correct_difference": pd.NA,
            "fresh_normal_recall": np.nan,
            "frozen_normal_recall": np.nan,
            "normal_recall_difference": np.nan,
            "fpr_difference": np.nan,
            "status": (
                "runtime-identical"
                if exact_diff == 0 and binary_diff == 0
                else "runtime-drift"
            ),
        })

    fresh_n = _normal_count_frame(
        fresh_normal, arm=fresh_arm, label="fresh normal"
    )
    frozen_n = _normal_count_frame(
        frozen_normal, arm=frozen_arm, label="frozen normal"
    )
    normal_key = ["pipeline_seed", "block_id"]
    normal = fresh_n.merge(
        frozen_n,
        on=normal_key,
        how="outer",
        suffixes=("_fresh", "_frozen"),
        validate="one_to_one",
        indicator=True,
    )
    if not normal["_merge"].eq("both").all():
        raise RuntimeError("T-STOP-RUNTIME: unpaired normal continuity rows")
    for row in normal.itertuples(index=False):
        record = row._asdict()
        n_fresh = int(record["n_fresh"])
        n_frozen = int(record["n_frozen"])
        if n_fresh != n_frozen or n_fresh <= 0:
            raise RuntimeError("T-STOP-RUNTIME: normal denominator drift")
        difference = (
            int(record["normal_correct_fresh"])
            - int(record["normal_correct_frozen"])
        )
        fresh_recall = int(record["normal_correct_fresh"]) / n_fresh
        frozen_recall = int(record["normal_correct_frozen"]) / n_frozen
        rows.append({
            "row_scope": "normal",
            "pipeline_seed": record["pipeline_seed"],
            "block_id": record["block_id"],
            "attack": "not_applicable",
            "id_stratum": "not_applicable",
            "cell": "not_applicable",
            "n": n_fresh,
            "fresh_exact_correct": pd.NA,
            "frozen_exact_correct": pd.NA,
            "exact_correct_difference": pd.NA,
            "fresh_binary_correct": pd.NA,
            "frozen_binary_correct": pd.NA,
            "binary_correct_difference": pd.NA,
            "fresh_exact_recall": np.nan,
            "frozen_exact_recall": np.nan,
            "exact_recall_difference": np.nan,
            "fresh_binary_recall": np.nan,
            "frozen_binary_recall": np.nan,
            "binary_recall_difference": np.nan,
            "fresh_normal_correct": int(record["normal_correct_fresh"]),
            "frozen_normal_correct": int(record["normal_correct_frozen"]),
            "normal_correct_difference": difference,
            "fresh_normal_recall": fresh_recall,
            "frozen_normal_recall": frozen_recall,
            "normal_recall_difference": fresh_recall - frozen_recall,
            "fpr_difference": (
                (1.0 - fresh_recall) - (1.0 - frozen_recall)
            ),
            "status": (
                "runtime-identical" if difference == 0 else "runtime-drift"
            ),
        })
    result = pd.DataFrame(rows)
    key = (
        "row_scope", "pipeline_seed", "block_id",
        "attack", "id_stratum", "cell",
    )
    require_unique(result, key, None, "runtime continuity")
    order = pd.Categorical(result["row_scope"], ["scenario", "normal"])
    return (
        result.assign(_scope_order=order)
        .sort_values(
            ["_scope_order", "pipeline_seed", "block_id", "attack",
             "id_stratum", "cell"],
            kind="mergesort",
        )
        .drop(columns="_scope_order")
        .reset_index(drop=True)
    )


def validate_registered_runtime_continuity_grid(
    table: pd.DataFrame,
) -> dict[str, Any]:
    """Require the exact preregistered 480-scenario + 15-normal key grid."""
    required_columns = (*CONTINUITY_KEY, "n")
    missing_columns = [
        column for column in required_columns if column not in table
    ]
    if missing_columns:
        raise RuntimeError(
            "T-STOP-RUNTIME: registered continuity grid lacks columns: "
            f"{missing_columns}"
        )

    scenario = table[
        table["row_scope"].astype(str).eq("scenario")
    ].copy()
    normal = table[
        table["row_scope"].astype(str).eq("normal")
    ].copy()
    if len(scenario) != 480 or len(normal) != 15:
        raise RuntimeError(
            "T-STOP-RUNTIME: registered continuity composition must be "
            "480 scenario + 15 normal rows, got "
            f"{len(scenario)} + {len(normal)}"
        )

    def integer_values(frame: pd.DataFrame, column: str) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce")
        array = values.to_numpy(dtype=float)
        if (
            not np.isfinite(array).all()
            or not np.equal(array, np.floor(array)).all()
        ):
            raise RuntimeError(
                "T-STOP-RUNTIME: registered continuity "
                f"{column} values are not finite integers"
            )
        return values.astype(int)

    scenario_pipeline = integer_values(scenario, "pipeline_seed")
    normal_pipeline = integer_values(normal, "pipeline_seed")
    scenario_n = integer_values(scenario, "n")
    normal_n = integer_values(normal, "n")
    if not scenario_n.eq(2_000).all() or not normal_n.eq(2_000).all():
        raise RuntimeError(
            "T-STOP-RUNTIME: registered continuity denominators must all "
            "equal 2,000"
        )

    observed_scenario = {
        (
            int(pipeline),
            str(block),
            str(attack),
            str(id_stratum),
            str(cell),
        )
        for pipeline, block, attack, id_stratum, cell in zip(
            scenario_pipeline,
            scenario["block_id"],
            scenario["attack"],
            scenario["id_stratum"],
            scenario["cell"],
            strict=True,
        )
    }
    expected_scenario = {
        (pipeline, block, attack, id_stratum, cell)
        for pipeline in PIPELINE_SEEDS
        for block in BLOCKS
        for attack in ATTACKS
        for id_stratum in ID_STRATA
        for cell in CELLS
    }
    if observed_scenario != expected_scenario:
        missing = sorted(expected_scenario - observed_scenario)[:5]
        extra = sorted(observed_scenario - expected_scenario)[:5]
        raise RuntimeError(
            "T-STOP-RUNTIME: registered scenario continuity grid mismatch; "
            f"missing={missing}, extra={extra}"
        )

    observed_normal = {
        (
            int(pipeline),
            str(block),
            str(attack),
            str(id_stratum),
            str(cell),
        )
        for pipeline, block, attack, id_stratum, cell in zip(
            normal_pipeline,
            normal["block_id"],
            normal["attack"],
            normal["id_stratum"],
            normal["cell"],
            strict=True,
        )
    }
    expected_normal = {
        (
            pipeline,
            block,
            "not_applicable",
            "not_applicable",
            "not_applicable",
        )
        for pipeline in PIPELINE_SEEDS
        for block in BLOCKS
    }
    if observed_normal != expected_normal:
        missing = sorted(expected_normal - observed_normal)[:5]
        extra = sorted(observed_normal - expected_normal)[:5]
        raise RuntimeError(
            "T-STOP-RUNTIME: registered normal continuity grid mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return {
        "registered_grid_complete": True,
        "scenario_rows": 480,
        "normal_rows": 15,
        "scenario_denominator": 2_000,
        "normal_denominator": 2_000,
    }


def require_real_runtime_continuity(
    table: pd.DataFrame,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Require exact integer equality; production passes ``expected_rows=495``."""
    key = (
        "row_scope", "pipeline_seed", "block_id",
        "attack", "id_stratum", "cell",
    )
    require_unique(table, key, expected_rows, "real runtime continuity")
    required = {
        "scenario": ("exact_correct_difference", "binary_correct_difference"),
        "normal": ("normal_correct_difference",),
    }
    unknown = set(table["row_scope"].astype(str)) - set(required)
    if unknown:
        raise RuntimeError(
            f"T-STOP-RUNTIME: unknown continuity scopes: {sorted(unknown)}"
        )
    registered_grid: dict[str, Any] = {}
    if expected_rows == EXPECTED_RUNTIME_CONTINUITY_ROWS:
        registered_grid = validate_registered_runtime_continuity_grid(table)
    mismatches = []
    for scope, columns in required.items():
        subset = table[table["row_scope"].astype(str).eq(scope)]
        for column in columns:
            if column not in subset:
                raise RuntimeError(
                    f"T-STOP-RUNTIME: missing continuity column {column}"
                )
            values = pd.to_numeric(subset[column], errors="coerce")
            array = values.to_numpy(dtype=float)
            if (
                not np.isfinite(array).all()
                or not np.equal(array, np.floor(array)).all()
                or not values.eq(0).all()
            ):
                mismatches.append(f"{scope}/{column}")
    if mismatches:
        raise RuntimeError(
            "T-STOP-RUNTIME: nonzero or missing integer-count differences: "
            + ", ".join(mismatches)
        )
    if "status" in table and not table["status"].astype(str).eq(
        "runtime-identical"
    ).all():
        raise RuntimeError("T-STOP-RUNTIME: continuity status reports drift")
    return {
        "status": "T-PASS",
        "rows": int(len(table)),
        "scenario_rows": int(table["row_scope"].eq("scenario").sum()),
        "normal_rows": int(table["row_scope"].eq("normal").sum()),
        "all_integer_count_differences_zero": True,
        **registered_grid,
    }


def build_anchor_outcome_continuity(
    fresh_scenario: pd.DataFrame,
    frozen_scenario: pd.DataFrame,
    fresh_normal: pd.DataFrame,
    frozen_normal: pd.DataFrame,
) -> pd.DataFrame:
    """Build the same-key fresh-E15 versus historical-E14 anchor audit."""
    table = build_real_runtime_continuity(
        fresh_scenario,
        frozen_scenario,
        fresh_normal,
        frozen_normal,
        fresh_arm="rule",
        frozen_arm="rule_ms",
    )
    table["status"] = np.where(
        table["status"].eq("runtime-identical"),
        "anchor-outcome-identical",
        "anchor-outcome-drift",
    )
    fresh_hashes = (
        _pipeline_column(fresh_normal, "fresh anchor normal")
        .groupby("pipeline_seed")["checkpoint_sha256"]
        .first()
        .astype(str)
        .to_dict()
    )
    frozen_hashes = (
        _pipeline_column(frozen_normal, "frozen anchor normal")
        .groupby("pipeline_seed")["checkpoint_sha256"]
        .first()
        .astype(str)
        .to_dict()
    )
    table["fresh_checkpoint_sha256"] = table["pipeline_seed"].map(
        fresh_hashes
    )
    table["frozen_checkpoint_sha256"] = table["pipeline_seed"].map(
        frozen_hashes
    )
    table["checkpoint_bytes_identical"] = (
        table["fresh_checkpoint_sha256"]
        == table["frozen_checkpoint_sha256"]
    )
    table["checkpoint_byte_identity_changes_outcome_status"] = False
    return table


def _finite_probability(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame:
        raise ValueError(f"{label} missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError(f"{label} has invalid probability column {column}")


def _exact_grid(
    frame: pd.DataFrame,
    columns: Sequence[str],
    values: Sequence[Iterable[Any]],
    label: str,
) -> None:
    expected = pd.MultiIndex.from_product(values, names=list(columns))
    observed = pd.MultiIndex.from_frame(frame[list(columns)])
    if len(observed) != len(expected) or set(observed) != set(expected):
        missing = list(set(expected) - set(observed))[:5]
        extra = list(set(observed) - set(expected))[:5]
        raise ValueError(
            f"{label} Cartesian grid mismatch; missing={missing}, extra={extra}"
        )


def validate_l4_scoring_tables(
    normal: pd.DataFrame,
    scenario: pd.DataFrame,
    cell: pd.DataFrame,
    delta: pd.DataFrame,
) -> dict[str, int]:
    """Validate all registered E15 L4 score-table counts and identities."""
    normal_key = ("construction_seed", "pipeline_seed", "block_id")
    scenario_key = (
        "construction_seed", "pipeline_seed", "block_id", "attack",
        "id_stratum", "P", "S", "D",
    )
    cell_key = (
        "construction_seed", "pipeline_seed", "block_id",
        "id_stratum", "P", "S", "D",
    )
    require_unique(
        normal, normal_key, EXPECTED_L4_NORMAL_ROWS, "E15 L4 normal"
    )
    require_unique(
        scenario, scenario_key, EXPECTED_L4_SCENARIO_ROWS, "E15 L4 scenario"
    )
    require_unique(cell, cell_key, EXPECTED_L4_CELL_ROWS, "E15 L4 cell")
    require_unique(
        delta, scenario_key, EXPECTED_L4_DELTA_ROWS, "E15 L4 delta"
    )
    _exact_grid(
        normal, normal_key,
        (CONSTRUCTION_SEEDS, PIPELINE_SEEDS, BLOCKS), "E15 L4 normal",
    )
    _exact_grid(
        scenario, scenario_key,
        (
            CONSTRUCTION_SEEDS, PIPELINE_SEEDS, BLOCKS, ATTACKS,
            ID_STRATA, (0, 1), (0, 1), (0, 1),
        ),
        "E15 L4 scenario",
    )
    _exact_grid(
        cell, cell_key,
        (
            CONSTRUCTION_SEEDS, PIPELINE_SEEDS, BLOCKS,
            ID_STRATA, (0, 1), (0, 1), (0, 1),
        ),
        "E15 L4 cell",
    )
    _exact_grid(
        delta, scenario_key,
        (
            CONSTRUCTION_SEEDS, PIPELINE_SEEDS, BLOCKS, ATTACKS,
            ID_STRATA, (0, 1), (0, 1), (0, 1),
        ),
        "E15 L4 delta",
    )
    for frame, label in (
        (normal, "normal"), (scenario, "scenario"), (cell, "cell"),
    ):
        for endpoint in (
            ("normal_recall", "fpr") if label == "normal" else ENDPOINTS
        ):
            _finite_probability(frame, endpoint, f"E15 L4 {label}")
    if not pd.to_numeric(normal["n"], errors="coerce").eq(2_000).all():
        raise ValueError("E15 L4 normal denominators must all equal 2000")
    if not pd.to_numeric(scenario["n"], errors="coerce").eq(2000).all():
        raise ValueError("E15 L4 scenario denominators must all equal 2000")
    if not pd.to_numeric(delta["n"], errors="coerce").eq(2000).all():
        raise ValueError("E15 L4 delta denominators must all equal 2000")
    for endpoint in ENDPOINTS:
        values = pd.to_numeric(delta[endpoint], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (np.abs(values) > 1.0).any():
            raise ValueError(f"E15 L4 delta has invalid {endpoint}")
    return {
        "checkpoint_evaluations": 25,
        "normal_rows": len(normal),
        "scenario_rows": len(scenario),
        "cell_rows": len(cell),
        "delta_rows": len(delta),
        "transformed_rule_forward_evaluations": int(scenario["n"].sum()),
        "clean_rule_forward_evaluations": int(normal["n"].sum()),
    }


def validate_l2_scoring_tables(
    by_scenario: pd.DataFrame,
    delta: pd.DataFrame,
) -> dict[str, int]:
    """Validate the complete raw and paired E15 L2 panels."""
    raw_key = (
        "arm", "construction_seed", "pipeline_seed",
        "block_id", "scenario_id",
    )
    delta_key = (
        "construction_seed", "pipeline_seed", "block_id",
        "attack", "setting",
    )
    require_unique(
        by_scenario, raw_key, EXPECTED_L2_SCENARIO_ROWS, "E15 L2 scenario"
    )
    require_unique(delta, delta_key, EXPECTED_L2_DELTA_ROWS, "E15 L2 delta")
    scenarios = {"Normal_normal"} | {
        f"{attack}_{setting}"
        for attack in L2_ATTACKS
        for setting in ("low", "medium", "high")
    }
    observed = set(by_scenario["scenario_id"].astype(str))
    if observed != scenarios:
        raise ValueError(
            f"E15 L2 scenario identities mismatch: {sorted(observed)}"
        )
    rule = by_scenario[by_scenario["arm"].astype(str).eq("rule")]
    real = by_scenario[by_scenario["arm"].astype(str).eq("shared_real")]
    if len(rule) != 975 or len(real) != 195:
        raise ValueError("E15 L2 arm row counts must be Rule=975, real=195")
    rule_index = pd.MultiIndex.from_product(
        (
            CONSTRUCTION_SEEDS,
            PIPELINE_SEEDS,
            BLOCKS,
            tuple(sorted(scenarios)),
        ),
        names=[
            "construction_seed",
            "pipeline_seed",
            "block_id",
            "scenario_id",
        ],
    )
    observed_rule = pd.MultiIndex.from_frame(
        rule[
            [
                "construction_seed",
                "pipeline_seed",
                "block_id",
                "scenario_id",
            ]
        ]
    )
    if set(observed_rule) != set(rule_index):
        raise ValueError("E15 L2 Rule Cartesian grid mismatch")
    real_index = pd.MultiIndex.from_product(
        (PIPELINE_SEEDS, BLOCKS, tuple(sorted(scenarios))),
        names=["pipeline_seed", "block_id", "scenario_id"],
    )
    observed_real = pd.MultiIndex.from_frame(
        real[["pipeline_seed", "block_id", "scenario_id"]]
    )
    if (
        set(observed_real) != set(real_index)
        or set(real["construction_seed"].astype(str)) != {"shared_reference"}
    ):
        raise ValueError("E15 L2 shared-real Cartesian grid mismatch")
    expected_delta = pd.MultiIndex.from_product(
        (
            CONSTRUCTION_SEEDS, PIPELINE_SEEDS, BLOCKS,
            L2_ATTACKS, ("low", "medium", "high"),
        ),
        names=list(delta_key),
    )
    observed_delta = pd.MultiIndex.from_frame(delta[list(delta_key)])
    if set(observed_delta) != set(expected_delta):
        raise ValueError("E15 L2 delta Cartesian grid mismatch")
    normal_rows = by_scenario[
        by_scenario["scenario_id"].astype(str).eq("Normal_normal")
    ]
    attack_rows = by_scenario[
        by_scenario["scenario_id"].astype(str).ne("Normal_normal")
    ]
    for endpoint in ("normal_recall", "fpr"):
        _finite_probability(normal_rows, endpoint, "E15 L2 normal")
    if not pd.to_numeric(normal_rows["n"], errors="coerce").eq(6_000).all():
        raise ValueError("E15 L2 normal denominators must equal 6000")
    if not pd.to_numeric(attack_rows["n"], errors="coerce").eq(500).all():
        raise ValueError("E15 L2 attack denominators must equal 500")
    for endpoint in ENDPOINTS:
        _finite_probability(attack_rows, endpoint, "E15 L2 attack")
        values = pd.to_numeric(delta[endpoint], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (np.abs(values) > 1.0).any():
            raise ValueError(f"E15 L2 delta has invalid {endpoint}")
    return {
        "checkpoint_evaluations": 30,
        "scenario_rows": len(by_scenario),
        "delta_rows": len(delta),
        "forward_evaluations": 1_080_000,
    }


def choose_positions(
    rng: np.random.Generator, burst_len: int, step: int
) -> np.ndarray:
    """E8 L2 position law, copied locally to keep Stage A torch-free."""
    start = int(rng.integers(0, 128 - int(burst_len) + 1))
    return np.arange(start, start + int(burst_len), int(step))


def inject_l2(
    rng: np.random.Generator, window: np.ndarray, scenario: Mapping[str, Any]
) -> int:
    """Apply the frozen E8 L2 coupled-setting injector in place."""
    attack = str(scenario["attack_type"])
    amplitude = float(scenario["amplitude"])
    positions = choose_positions(
        rng, int(scenario["burst_len"]), int(scenario["step"])
    )
    if attack == "DoS":
        window[positions, 0] = 0x000
        window[positions, 1] = 8
        window[positions, 2:10] = 0
        window[positions, 10] = np.maximum(
            window[positions, 10] * (0.4 - 0.25 * amplitude), 1e-5
        )
    elif attack == "Fuzzy":
        window[positions, 0] = rng.integers(0, 2048, size=len(positions))
        window[positions, 1] = 8
        base = window[positions, 2:10].copy()
        random_payload = rng.integers(
            0, 256, size=(len(positions), 8)
        )
        window[positions, 2:10] = np.clip(
            (1.0 - amplitude) * base + amplitude * random_payload, 0, 255
        )
        window[positions, 10] = rng.uniform(
            0.00005, 0.005 + 0.01 * amplitude, size=len(positions)
        )
    elif attack in {"Gear", "RPM"}:
        window[positions, 0] = 0x43F if attack == "Gear" else 0x316
        window[positions, 1] = 8
        scale = 255.0 * amplitude
        ramp = np.linspace(0, scale, len(positions), dtype=np.float32)
        base = window[positions, 2:10].copy()
        if attack == "Gear":
            base[:, 2] = np.clip(
                base[:, 2] * (1.0 - amplitude) + ramp, 0, 255
            )
            base[:, 3] = np.clip(
                base[:, 3] * (1.0 - amplitude) + (255 - ramp), 0, 255
            )
            base[:, 4] = np.clip(
                base[:, 4]
                + rng.integers(-10, 11, size=len(positions)) * amplitude,
                0, 255,
            )
        else:
            wave = (
                127.5
                + 127.5
                * np.sin(np.linspace(0, 2 * np.pi, len(positions)))
            ).astype(np.float32) * amplitude
            base[:, 2] = np.clip(
                base[:, 2] * (1.0 - amplitude) + wave, 0, 255
            )
            base[:, 3] = np.clip(
                base[:, 3] * (1.0 - amplitude) + ramp, 0, 255
            )
            base[:, 5] = np.clip(
                base[:, 5]
                + rng.integers(-15, 16, size=len(positions)) * amplitude,
                0, 255,
            )
        window[positions, 2:10] = base
        window[positions, 10] = np.maximum(window[positions, 10], 1e-5)
    else:
        raise ValueError(f"unknown L2 attack type: {attack}")
    return int(len(positions))


def generate_l2_block(
    x_test: np.ndarray,
    base_indices: np.ndarray,
    eval_seed: int,
) -> dict[str, np.ndarray]:
    """Regenerate one exact 12,000-row E8 L2 block without model inference."""
    indices = np.asarray(base_indices, dtype=np.int64)
    if len(indices) != 12_000:
        raise ValueError(f"L2 block needs 12,000 bases, got {len(indices)}")
    rng = np.random.default_rng(int(eval_seed))
    assigned = rng.permutation(indices)
    windows = np.asarray(x_test)[assigned].astype(np.float32, copy=True)
    labels = np.zeros(12_000, dtype=np.int8)
    scenario_ids = np.full(12_000, "Normal_normal", dtype=object)
    attacks = np.full(12_000, "Normal", dtype=object)
    settings = np.full(12_000, "normal", dtype=object)
    injection_count = np.zeros(12_000, dtype=np.int16)
    cursor = 6_000
    for scenario in L2_SCENARIOS:
        selection = slice(cursor, cursor + 500)
        attack = str(scenario["attack_type"])
        setting = str(scenario["setting"])
        labels[selection] = ATTACK_LABEL[attack]
        scenario_ids[selection] = f"{attack}_{setting}"
        attacks[selection] = attack
        settings[selection] = setting
        for index in range(selection.start, selection.stop):
            injection_count[index] = inject_l2(
                rng, windows[index], scenario
            )
        cursor = selection.stop
    if cursor != 12_000:
        raise AssertionError("incomplete frozen L2 construction")
    if windows.shape != (12_000, 128, 11):
        raise AssertionError("L2 window shape drift")
    if windows.dtype != np.float32 or not np.isfinite(windows).all():
        raise ValueError("L2 windows must be finite float32")
    return {
        "x": np.ascontiguousarray(windows),
        "y_attack_type": labels,
        "scenario_id": scenario_ids,
        "attack_type": attacks,
        "setting": settings,
        "injection_count": injection_count,
        "base_index": assigned,
    }


def l2_keyed_digest(
    blocks: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[str, pd.DataFrame]:
    """Hash keyed L2 rows and return per-block/scenario manipulation checks."""
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    expected_blocks = dict(zip(BLOCKS, EVAL_SEEDS, strict=True))
    if set(blocks) != set(expected_blocks):
        raise ValueError("L2 digest requires the three registered blocks")
    total = 0
    for block_id in BLOCKS:
        block = blocks[block_id]
        size = len(block["x"])
        if size != 12_000:
            raise ValueError(f"{block_id} L2 row count mismatch")
        for index in range(size):
            key = [
                block_id,
                int(expected_blocks[block_id]),
                index,
                int(block["base_index"][index]),
                str(block["scenario_id"][index]),
                str(block["attack_type"][index]),
                str(block["setting"][index]),
                int(block["y_attack_type"][index]),
                int(block["injection_count"][index]),
            ]
            encoded = json.dumps(
                key, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
            window = np.ascontiguousarray(block["x"][index], dtype="<f4")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
            digest.update(b"<f4")
            digest.update(struct.pack("<II", 128, 11))
            digest.update(memoryview(window).cast("B"))
            total += 1
        audit = pd.DataFrame({
            "scenario_id": block["scenario_id"],
            "attack": block["attack_type"],
            "setting": block["setting"],
            "injection_count": block["injection_count"],
        })
        for scenario_id, group in audit.groupby("scenario_id", sort=True):
            rows.append({
                "block_id": block_id,
                "eval_seed": expected_blocks[block_id],
                "scenario_id": scenario_id,
                "attack": str(group["attack"].iloc[0]),
                "setting": str(group["setting"].iloc[0]),
                "n": len(group),
                "injection_count_min":
                    int(group["injection_count"].min()),
                "injection_count_max":
                    int(group["injection_count"].max()),
                "injection_count_mean":
                    float(group["injection_count"].mean()),
                "status": "PASS",
            })
    if total != EXPECTED_L2_TRANSFORM_INSTANCES:
        raise AssertionError("L2 keyed digest row count mismatch")
    checks = pd.DataFrame(rows)
    require_unique(
        checks, ("block_id", "scenario_id"), 39, "L2 manipulation checks"
    )
    return digest.hexdigest(), checks


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _record_schema(record: Mapping[str, Any]) -> str:
    return str(record.get("schema_version", record.get("schema", "")))


def _path_hash_binding(
    binding: Mapping[str, Any], expected_path: Path, label: str
) -> None:
    if binding.get("path") != portable_path(expected_path):
        raise ValueError(f"{label} path binding mismatch")
    if binding.get("sha256") != sha256_file(expected_path):
        raise ValueError(f"{label} hash binding mismatch")


def validate_pool_training_provenance() -> dict[str, Any]:
    """Independently validate generation/training records and all checkpoints."""
    for path in (
        POOL_GENERATION_PREFLIGHT, POOL_GENERATION_RUN,
        TRAINING_PREFLIGHT, TRAINING_RUN,
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"T-STOP-MANIPULATION: missing record {path}")
    pool_preflight = _read_json(POOL_GENERATION_PREFLIGHT)
    pool_run = _read_json(POOL_GENERATION_RUN)
    training_preflight = _read_json(TRAINING_PREFLIGHT)
    training_run = _read_json(TRAINING_RUN)
    expected_schemas = (
        (pool_preflight, "e15.pool_generation_preflight.v1"),
        (pool_run, "e15.pool_generation_run.v1"),
        (training_preflight, "e15.training_preflight.v1"),
        (training_run, "e15.training_run.v1"),
    )
    for record, schema in expected_schemas:
        if _record_schema(record) != schema:
            raise ValueError(
                f"E15 record schema mismatch: expected={schema}, "
                f"got={_record_schema(record)}"
            )
    if pool_run.get("status") != "T-PASS":
        raise ValueError("pool-generation run did not finish T-PASS")
    if training_run.get("status") != "T-PASS":
        raise ValueError("training run did not finish T-PASS")
    for run, preflight, path, label in (
        (
            pool_run, pool_preflight, POOL_GENERATION_PREFLIGHT,
            "pool preflight",
        ),
        (
            training_run, training_preflight, TRAINING_PREFLIGHT,
            "training preflight",
        ),
    ):
        binding = run.get("preflight")
        if not isinstance(binding, Mapping):
            raise ValueError(f"{label} binding missing")
        _path_hash_binding(binding, path, label)

    bundles = training_run.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 25:
        raise ValueError("training run must bind exactly 25 bundles")
    run_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    seen: set[tuple[int, int]] = set()
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            raise ValueError("training bundle rows must be objects")
        construction = int(bundle.get("construction_seed", -1))
        pipeline = int(bundle.get("pipeline_seed", -1))
        key = (construction, pipeline)
        if (
            key in seen
            or construction not in CONSTRUCTION_SEEDS
            or pipeline not in PIPELINE_SEEDS
        ):
            raise ValueError(f"invalid/duplicate training bundle key: {key}")
        seen.add(key)
        expected_checkpoint = rule_checkpoint_path(construction, pipeline)
        expected_log = rule_training_log_path(construction, pipeline)
        for field, path in (
            ("checkpoint", expected_checkpoint), ("training_log", expected_log)
        ):
            binding = bundle.get(field)
            if not isinstance(binding, Mapping):
                raise ValueError(f"bundle {key} lacks {field} binding")
            _path_hash_binding(binding, path, f"{key}/{field}")
        if (
            bundle.get("explicit_synthetic_pool_override") is not True
            or bundle.get("fallback_used") is not False
        ):
            raise ValueError(
                f"explicit no-fallback evidence is missing for {key}"
            )
        run_by_key[key] = bundle
    expected_grid = {
        (construction, pipeline)
        for construction in CONSTRUCTION_SEEDS
        for pipeline in PIPELINE_SEEDS
    }
    if seen != expected_grid:
        raise ValueError("training bundle crossed grid is incomplete")

    # Stage A repeats the weights-only and log validation instead of trusting
    # path presence or the upstream training-run declaration.
    import generate_rule_construction_sensitivity as generator
    import train_rule_construction_seed_crossing as training

    pool_run_rows = pool_run.get("pools")
    if not isinstance(pool_run_rows, list) or len(pool_run_rows) != 5:
        raise ValueError("pool-generation run lacks five pool bindings")
    pool_run_by_seed = {
        int(row["construction_seed"]): row
        for row in pool_run_rows
        if isinstance(row, Mapping)
    }
    if set(pool_run_by_seed) != set(CONSTRUCTION_SEEDS):
        raise ValueError("pool-generation run construction grid mismatch")
    validated_pools: list[dict[str, Any]] = []
    recomputed_sampling: dict[tuple[int, int], str] = {}
    for construction in CONSTRUCTION_SEEDS:
        target = generator.pool_target_map(ROOT)[construction]
        audit = generator.validate_pool_archive(
            target.pool_path, construction
        )
        run_row = pool_run_by_seed[construction]
        if (
            run_row.get("pool_file_sha256") != audit["pool_file_sha256"]
            or run_row.get("ordered_content_sha256")
            != audit["ordered_content_sha256"]
            or run_row.get("pool_identifier") != audit["pool_identifier"]
        ):
            raise ValueError(
                f"current pool differs from generation run: {construction}"
            )
        generation_log = LOGS / (
            f"e15_generate_rule_cseed{construction}_{OUTPUT_VERSION}.json"
        )
        if (
            run_row.get("generation_log_sha256")
            != sha256_file(generation_log)
        ):
            raise ValueError(
                f"generation log hash drift for construction {construction}"
            )
        with np.load(target.pool_path, allow_pickle=False) as archive:
            labels = np.asarray(archive["y_attack_type"])
            for pipeline in PIPELINE_SEEDS:
                _, sampling = training.sample_indices_without_replacement(
                    labels,
                    training.SYNTHETIC_TOTAL,
                    pipeline + training.SAMPLING_SEED_OFFSET,
                )
                recomputed_sampling[(construction, pipeline)] = str(
                    sampling["index_sha256"]
                )
        validated_pools.append(
            {
                "construction_seed": construction,
                "pool": {
                    "path": portable_path(target.pool_path),
                    "bytes": target.pool_path.stat().st_size,
                    "sha256": audit["pool_file_sha256"],
                    "ordered_content_sha256":
                        audit["ordered_content_sha256"],
                    "pool_identifier": audit["pool_identifier"],
                    "configuration_sha256":
                        audit["configuration_sha256"],
                },
                "generation_log": {
                    "path": portable_path(generation_log),
                    "bytes": generation_log.stat().st_size,
                    "sha256": sha256_file(generation_log),
                },
            }
        )

    validated_rule_bundles: list[dict[str, Any]] = []
    for job in training.build_training_grid(ROOT):
        key = (job.construction_seed, job.pipeline_seed)
        audit = training.validate_training_bundle(job)
        recorded = run_by_key[key]
        for field in ("checkpoint", "training_log"):
            current = audit[field]
            prior = recorded[field]
            if (
                current["sha256"] != prior["sha256"]
                or current["bytes"] != prior["bytes"]
                or current["path"] != prior["path"]
            ):
                raise ValueError(
                    f"training run/current {field} binding mismatch for {key}"
                )
        if (
            audit.get("sampling_index_sha256")
            != recomputed_sampling[key]
            or recorded.get("sampling_index_sha256")
            != recomputed_sampling[key]
        ):
            raise ValueError(
                f"recomputed sampling vector mismatch for {key}"
            )
        if (
            audit.get("explicit_synthetic_pool_override") is not True
            or audit.get("fallback_used") is not False
        ):
            raise ValueError(f"revalidated bundle lacks no-fallback proof: {key}")
        validated_rule_bundles.append(audit)

    e14_prepare = _read_json(E14_PREPARE)
    e14_bundles = e14_prepare.get("checkpoint_bundles")
    if not isinstance(e14_bundles, list):
        raise ValueError("E14 prepare checkpoint bundle is missing")
    shared_records = {
        int(record["seed"]): record
        for record in e14_bundles
        if isinstance(record, Mapping) and record.get("arm") == "real_ms"
    }
    if set(shared_records) != set(PIPELINE_SEEDS):
        raise ValueError("E14 prepare lacks five shared real checkpoints")

    from run_e14_matched_real_training import validate_checkpoint

    validated_shared_real: list[dict[str, Any]] = []
    for pipeline in PIPELINE_SEEDS:
        record = shared_records[pipeline]
        checkpoint = record.get("checkpoint")
        training_log = record.get("training_log")
        if not isinstance(checkpoint, Mapping) or not isinstance(
            training_log, Mapping
        ):
            raise ValueError(
                f"E14 shared-real artifact binding is missing for {pipeline}"
            )
        checkpoint_path = shared_real_checkpoint_path(pipeline)
        log_path = ROOT / str(training_log.get("path", ""))
        _path_hash_binding(
            checkpoint,
            checkpoint_path,
            f"shared real {pipeline}/checkpoint",
        )
        _path_hash_binding(
            training_log,
            log_path,
            f"shared real {pipeline}/training log",
        )
        validated_shared_real.append(
            {
                "pipeline_seed": pipeline,
                "checkpoint": {
                    "path": portable_path(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "sha256": sha256_file(checkpoint_path),
                    "weights_only_validation": validate_checkpoint(
                        checkpoint_path
                    ),
                },
                "training_log": {
                    "path": portable_path(log_path),
                    "bytes": log_path.stat().st_size,
                    "sha256": sha256_file(log_path),
                },
            }
        )
    return {
        "pool_generation_preflight": sha256_file(POOL_GENERATION_PREFLIGHT),
        "pool_generation_run": sha256_file(POOL_GENERATION_RUN),
        "training_preflight": sha256_file(TRAINING_PREFLIGHT),
        "training_run": sha256_file(TRAINING_RUN),
        "training_bundles": 25,
        "rule_checkpoint_weights_only_validated": len(
            validated_rule_bundles
        ),
        "shared_real_checkpoint_weights_only_validated": len(
            validated_shared_real
        ),
        "validated_rule_bundles": validated_rule_bundles,
        "validated_shared_real": validated_shared_real,
        "validated_pools": validated_pools,
        "recomputed_sampling_vectors": len(recomputed_sampling),
        "crossed_grid_complete": True,
        "no_fallback": True,
        "model_forward_passes": 0,
    }


def rule_checkpoint_path(construction_seed: int, pipeline_seed: int) -> Path:
    if (
        int(construction_seed) not in CONSTRUCTION_SEEDS
        or int(pipeline_seed) not in PIPELINE_SEEDS
    ):
        raise ValueError("undeclared E15 construction/pipeline seed")
    return MODELS / (
        "cnn_rule_0p30_"
        f"cseed{int(construction_seed)}_matchedsteps_e15_{OUTPUT_VERSION}_"
        f"seed{int(pipeline_seed)}.pt"
    )


def rule_training_log_path(
    construction_seed: int, pipeline_seed: int
) -> Path:
    if (
        int(construction_seed) not in CONSTRUCTION_SEEDS
        or int(pipeline_seed) not in PIPELINE_SEEDS
    ):
        raise ValueError("undeclared E15 construction/pipeline seed")
    return LOGS / (
        "train_cnn_rule_0p30_"
        f"cseed{int(construction_seed)}_matchedsteps_e15_{OUTPUT_VERSION}_"
        f"seed{int(pipeline_seed)}.log"
    )


def shared_real_checkpoint_path(pipeline_seed: int) -> Path:
    if int(pipeline_seed) not in PIPELINE_SEEDS:
        raise ValueError("undeclared E15 pipeline seed")
    return MODELS / (
        f"cnn_real_only_matchedsteps_e14_v1_seed{int(pipeline_seed)}.pt"
    )


def _validate_frozen_hashes() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_HASHES.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"T-STOP-MANIPULATION: missing frozen input {path}"
            )
        current = sha256_file(path)
        if current != expected:
            raise RuntimeError(
                "T-STOP-MANIPULATION: frozen input hash mismatch "
                f"{path}: expected={expected}, observed={current}"
            )
        observed[portable_path(path)] = current
    return observed


def _load_e14_construction() -> tuple[
    Mapping[str, Any], pd.DataFrame, np.ndarray, list[Any]
]:
    prepare = _read_json(E14_PREPARE)
    base_manifest, bases, latents = e14.load_prepared_inputs(prepare)
    return prepare, base_manifest, bases, latents


def regenerate_l4_identity() -> dict[str, Any]:
    """Regenerate all 192,000 transforms with the audited E14 pure helpers."""
    _, base_manifest, bases, latents = _load_e14_construction()
    transform_sha, key_sha, instances, _ = e14._transform_pass(
        base_manifest, bases, latents, validate_relations=True
    )
    return validate_transform_identity(instances, transform_sha, key_sha)


def regenerate_l2_blocks() -> tuple[
    dict[str, Any],
    pd.DataFrame,
    dict[str, Mapping[str, np.ndarray]],
]:
    """Regenerate all three frozen L2 blocks and return their keyed identity."""
    manifest = pd.read_csv(BLOCK_MANIFEST)
    require_unique(
        manifest, ("block_id", "block_position"), 36_000,
        "E8/E13 block manifest",
    )
    with np.load(TEST_WINDOWS, allow_pickle=False) as archive:
        x_test = np.asarray(archive["x"])
        y_binary = np.asarray(archive["y_binary"])
        y_attack = np.asarray(archive["y_attack_type"])
        blocks: dict[str, Mapping[str, np.ndarray]] = {}
        for block_id, eval_seed in zip(BLOCKS, EVAL_SEEDS, strict=True):
            subset = (
                manifest[manifest["block_id"].astype(str).eq(block_id)]
                .sort_values("block_position", kind="mergesort")
            )
            indices = subset["test_window_index"].to_numpy(np.int64)
            if (
                len(indices) != 12_000
                or np.any(y_binary[indices] != 0)
                or np.any(y_attack[indices] != 0)
            ):
                raise ValueError(f"{block_id} is not 12,000 normal bases")
            blocks[block_id] = generate_l2_block(x_test, indices, eval_seed)
    digest, checks = l2_keyed_digest(blocks)
    identity = {
        "instances": EXPECTED_L2_TRANSFORM_INSTANCES,
        "keyed_transformation_sha256": digest,
        "serialization": L2_SERIALIZATION,
        "manipulation_rows": len(checks),
        "all_manipulation_gates_pass": True,
    }
    return identity, checks, blocks


def regenerate_l2_identity() -> tuple[dict[str, Any], pd.DataFrame]:
    """Regenerate the frozen L2 panel without retaining its windows."""
    identity, checks, _ = regenerate_l2_blocks()
    return identity, checks


def assert_prepare_snapshot_unchanged(
    *,
    source: Mapping[str, Any],
    frozen_inputs: Mapping[str, str],
    upstream_lineage: Mapping[str, Any],
    environment: Mapping[str, Any],
    standardizer_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Revalidate every Stage-A input class immediately before publication."""
    final_source = source_provenance_for_prepare()
    if final_source != dict(source):
        raise RuntimeError(
            "T-STOP-MANIPULATION: source snapshot changed during Stage A"
        )
    final_frozen = _validate_frozen_hashes()
    if final_frozen != dict(frozen_inputs):
        raise RuntimeError(
            "T-STOP-MANIPULATION: frozen inputs changed during Stage A"
        )
    final_lineage = validate_pool_training_provenance()
    if final_lineage != dict(upstream_lineage):
        raise RuntimeError(
            "T-STOP-MANIPULATION: upstream lineage changed during Stage A"
        )
    final_environment = evaluation_environment_record(stage="prepare_only")
    if final_environment != dict(environment):
        raise RuntimeError(
            "T-STOP-MANIPULATION: evaluation environment changed during "
            "Stage A"
        )
    _, _, final_standardizer_hashes = e14.fit_registered_standardizer()
    if final_standardizer_hashes != dict(standardizer_hashes):
        raise RuntimeError(
            "T-STOP-MANIPULATION: standardizer changed during Stage A"
        )
    return {
        "snapshot_unchanged": True,
        "source_unchanged": True,
        "frozen_inputs_unchanged": True,
        "upstream_lineage_unchanged": True,
        "environment_unchanged": True,
        "standardizer_unchanged": True,
        "frozen_inputs_sha256": hashlib.sha256(
            _json_bytes(final_frozen)
        ).hexdigest(),
        "upstream_lineage_sha256": hashlib.sha256(
            _json_bytes(final_lineage)
        ).hexdigest(),
        "environment_sha256": hashlib.sha256(
            _json_bytes(final_environment)
        ).hexdigest(),
    }


def run_prepare_only() -> dict[str, Any]:
    """Run Stage A and atomically publish only the prediction-free record."""
    if _lexists(PREPARE_RECORD):
        raise FileExistsError(f"refusing to overwrite {PREPARE_RECORD}")
    for name, path in OUTPUT_PATHS.items():
        if _lexists(path):
            raise FileExistsError(
                f"prepare requires absent canonical target {name}: {path}"
            )
    ledger = InferenceLedger()
    source = source_provenance_for_prepare()
    frozen = _validate_frozen_hashes()
    lineage = validate_pool_training_provenance()
    environment = evaluation_environment_record(stage="prepare_only")
    mean, std, standardizer_hashes = e14.fit_registered_standardizer()
    del mean, std
    e14_prepare = _read_json(E14_PREPARE)
    expected_standardizer = e14_prepare.get("standardizer", {}).get("hashes")
    if standardizer_hashes != expected_standardizer:
        raise RuntimeError(
            "T-STOP-MANIPULATION: real-train standardizer identity drift"
        )
    l4 = regenerate_l4_identity()
    l2, checks = regenerate_l2_identity()
    guard = assert_prepare_only_inference_guard(
        ledger.inference_calls, ledger.prediction_rows
    )
    final_snapshot = assert_prepare_snapshot_unchanged(
        source=source,
        frozen_inputs=frozen,
        upstream_lineage=lineage,
        environment=environment,
        standardizer_hashes=standardizer_hashes,
    )
    record = {
        "schema_version": "e15.evaluation_prepare.v1",
        "status": "T-PASS",
        "created_utc": utc_now(),
        "analysis_role": (
            "prediction-free transformation/provenance/checkpoint bundle "
            "validation; no model forward pass and no outcomes"
        ),
        "source_commit": source["source_commit"],
        "source_provenance": source,
        "frozen_inputs": frozen,
        "upstream_lineage": lineage,
        "environment": environment,
        "standardizer": {
            "source": portable_path(TRAIN_WINDOWS),
            "derivation": "complete real train split only",
            "hashes": standardizer_hashes,
        },
        "l4_transformation": l4,
        "l2_transformation": l2,
        "l2_manipulation_checks": checks.to_dict("records"),
        "prepare_only_guard": guard,
        "checkpoint_deserialization_performed": True,
        "checkpoint_deserialization_mode": (
            "torch.load(weights_only=True), CPU structure/finiteness only"
        ),
        "canonical_score_targets_absent": True,
        "source_snapshot_before_publication": final_snapshot,
        "next_required_gate": (
            "rescore all five shared-real checkpoints and require 495 exact "
            "integer-count continuity rows before loading any Rule checkpoint "
            "for inference"
        ),
    }
    publish_bundle_no_clobber([(PREPARE_RECORD, _json_bytes(record))])
    return record


def _git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    )


IMPLEMENTATION_PATHS = (
    Path("journal/scripts/generate_rule_construction_sensitivity.py"),
    Path("journal/scripts/train_rule_construction_seed_crossing.py"),
    Path("journal/scripts/evaluate_rule_construction_seed_crossing.py"),
    Path("journal/scripts/analyze_rule_construction_seed_crossing.py"),
    Path("journal/tests/test_rule_construction_seed_crossing.py"),
    Path(
        "journal/experiments/e15_rule_construction_crossing/"
        "IMPLEMENTATION_AMENDMENT_2026-07-25_V2_PATH_IDENTITY.md"
    ),
)


def _implementation_source_records() -> dict[str, dict[str, Any]]:
    """Bind every E15 implementation file to the current HEAD blob."""
    records: dict[str, dict[str, Any]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"T-STOP-MANIPULATION: invalid implementation file {relative}"
            )
        try:
            blob = _git_output(
                ["rev-parse", f"HEAD:{relative.as_posix()}"]
            ).strip()
            committed = subprocess.check_output(
                ["git", "cat-file", "blob", blob],
                cwd=ROOT,
            )
        except Exception as exc:
            raise RuntimeError(
                "T-STOP-MANIPULATION: implementation is not committed at "
                f"HEAD: {relative}"
            ) from exc
        worktree_hash = sha256_file(path)
        committed_hash = hashlib.sha256(committed).hexdigest()
        if worktree_hash != committed_hash:
            raise RuntimeError(
                "T-STOP-MANIPULATION: implementation differs from HEAD: "
                f"{relative}"
            )
        records[relative.as_posix()] = {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": worktree_hash,
            "git_blob_oid": blob,
            "matches_head": True,
        }
    return records


def _declared_prepare_upstream_paths() -> set[str]:
    """Exact generation/training outputs allowed untracked before Stage A."""
    paths: set[Path] = {
        POOL_GENERATION_PREFLIGHT,
        POOL_GENERATION_RUN,
        TABLES / (
            f"e15_rule_construction_pool_audit_{OUTPUT_VERSION}.csv"
        ),
        TRAINING_PREFLIGHT,
        TRAINING_RUN,
        TABLES / (
            f"e15_rule_construction_sampling_audit_{OUTPUT_VERSION}.csv"
        ),
        TABLES / (
            f"e15_rule_construction_training_manifest_{OUTPUT_VERSION}.csv"
        ),
    }
    for construction in CONSTRUCTION_SEEDS:
        paths.update(
            {
                SYNTHETIC / (
                    f"rule_cseed{construction}_windows_{OUTPUT_VERSION}.npz"
                ),
                TABLES / (
                    f"e15_rule_cseed{construction}_statistics_"
                    f"{OUTPUT_VERSION}.csv"
                ),
                LOGS / (
                    f"e15_generate_rule_cseed{construction}_"
                    f"{OUTPUT_VERSION}.json"
                ),
            }
        )
        for pipeline in PIPELINE_SEEDS:
            paths.update(
                {
                    rule_checkpoint_path(construction, pipeline),
                    rule_training_log_path(construction, pipeline),
                }
            )
    return {portable_path(path) for path in paths}


def _porcelain_records() -> list[str]:
    raw = _git_output(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    return [record for record in raw.split("\0") if record]


def source_provenance_for_prepare() -> dict[str, Any]:
    """Require clean committed source and only declared upstream artifacts."""
    head = _git_output(["rev-parse", "HEAD"]).strip()
    allowed = _declared_prepare_upstream_paths()
    records = _porcelain_records()
    observed_allowed: list[str] = []
    rejected: list[str] = []
    for record in records:
        if len(record) < 4:
            rejected.append(record)
            continue
        status = record[:2]
        relative = Path(record[3:]).as_posix()
        if status == "??" and relative in allowed:
            observed_allowed.append(record)
        else:
            rejected.append(record)
    if rejected:
        raise RuntimeError(
            "T-STOP-MANIPULATION: Stage A found dirty source or undeclared "
            f"untracked files: {rejected}"
        )
    if _git_output(["status", "--short", "--", "wisa"]).strip():
        raise RuntimeError("T-STOP-MANIPULATION: frozen wisa/ is not clean")

    upstream_heads: dict[str, str] = {}
    for label, path in (
        ("pool_generation", POOL_GENERATION_PREFLIGHT),
        ("training", TRAINING_PREFLIGHT),
    ):
        payload = _read_json(path)
        source = payload.get("source_provenance")
        if not isinstance(source, Mapping):
            raise RuntimeError(
                f"T-STOP-MANIPULATION: {label} source record is missing"
            )
        source_head = str(source.get("head_commit", ""))
        if source_head != head:
            raise RuntimeError(
                "T-STOP-MANIPULATION: Stage A HEAD differs from the "
                f"{label} preflight source commit"
            )
        upstream_heads[label] = source_head
    return {
        "source_commit": head,
        "tracked_index_source_head_clean": True,
        "wisa_clean": True,
        "allowed_untracked_upstream_paths": sorted(allowed),
        "observed_allowed_untracked_records": sorted(observed_allowed),
        "upstream_source_commits": upstream_heads,
        "implementation": _implementation_source_records(),
    }


def source_provenance_for_score(
    prepare: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a fully clean post-prepare commit with unchanged implementation."""
    records = _porcelain_records()
    if records:
        raise RuntimeError(
            "T-STOP-MANIPULATION: Stage B requires a fully clean worktree "
            f"after prepare commit: {records}"
        )
    if _git_output(["status", "--short", "--", "wisa"]).strip():
        raise RuntimeError("T-STOP-MANIPULATION: frozen wisa/ is not clean")
    current = _implementation_source_records()
    prepared_source = prepare.get("source_provenance")
    if not isinstance(prepared_source, Mapping):
        raise RuntimeError(
            "T-STOP-MANIPULATION: prepare source provenance is missing"
        )
    prepared_implementation = prepared_source.get("implementation")
    if not isinstance(prepared_implementation, Mapping):
        raise RuntimeError(
            "T-STOP-MANIPULATION: prepare implementation binding is missing"
        )
    for relative, record in current.items():
        prior = prepared_implementation.get(relative)
        if not isinstance(prior, Mapping) or prior.get("sha256") != record[
            "sha256"
        ]:
            raise RuntimeError(
                "T-STOP-MANIPULATION: implementation changed after Stage A: "
                f"{relative}"
            )
    return {
        "source_commit": _git_output(["rev-parse", "HEAD"]).strip(),
        "source_worktree_clean": True,
        "wisa_clean": True,
        "implementation_matches_prepare": True,
        "implementation": current,
    }


def _validate_prepare_record() -> dict[str, Any]:
    """Validate the committed Stage-A record and its current bound artifacts."""
    if not PREPARE_RECORD.is_file() or PREPARE_RECORD.is_symlink():
        raise RuntimeError(
            f"T-STOP-MANIPULATION: missing prepare record {PREPARE_RECORD}"
        )
    prepare = _read_json(PREPARE_RECORD)
    if (
        prepare.get("schema_version") != "e15.evaluation_prepare.v1"
        or prepare.get("status") != "T-PASS"
        or prepare.get("canonical_score_targets_absent") is not True
    ):
        raise RuntimeError(
            "T-STOP-MANIPULATION: invalid E15 prepare completion record"
        )
    guard = prepare.get("prepare_only_guard")
    if (
        not isinstance(guard, Mapping)
        or guard.get("prepare_only_prediction_free") is not True
        or guard.get("inference_calls") != 0
        or guard.get("prediction_rows") != 0
    ):
        raise RuntimeError(
            "T-STOP-MANIPULATION: prepare prediction-free guard is invalid"
        )
    environment = prepare.get("environment")
    if not isinstance(environment, Mapping):
        raise RuntimeError(
            "T-STOP-MANIPULATION: prepare environment binding is missing"
        )
    validate_evaluation_environment_record(
        environment, expected_stage="prepare_only"
    )
    final_snapshot = prepare.get("source_snapshot_before_publication")
    if not isinstance(final_snapshot, Mapping) or any(
        final_snapshot.get(field) is not True
        for field in (
            "snapshot_unchanged",
            "source_unchanged",
            "frozen_inputs_unchanged",
            "upstream_lineage_unchanged",
            "environment_unchanged",
            "standardizer_unchanged",
        )
    ):
        raise RuntimeError(
            "T-STOP-MANIPULATION: prepare final snapshot gate is invalid"
        )
    frozen = _validate_frozen_hashes()
    if prepare.get("frozen_inputs") != frozen:
        raise RuntimeError(
            "T-STOP-MANIPULATION: frozen inputs differ from Stage A"
        )
    for path_key, expected_hash in (
        ("pool_generation_preflight", POOL_GENERATION_PREFLIGHT),
        ("pool_generation_run", POOL_GENERATION_RUN),
        ("training_preflight", TRAINING_PREFLIGHT),
        ("training_run", TRAINING_RUN),
    ):
        lineage = prepare.get("upstream_lineage")
        if (
            not isinstance(lineage, Mapping)
            or lineage.get(path_key) != sha256_file(expected_hash)
        ):
            raise RuntimeError(
                "T-STOP-MANIPULATION: upstream lineage changed after "
                f"Stage A: {path_key}"
            )
    lineage = prepare.get("upstream_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError(
            "T-STOP-MANIPULATION: prepare lineage object is missing"
        )
    snapshot_hashes = {
        "frozen_inputs_sha256": hashlib.sha256(
            _json_bytes(frozen)
        ).hexdigest(),
        "upstream_lineage_sha256": hashlib.sha256(
            _json_bytes(lineage)
        ).hexdigest(),
        "environment_sha256": hashlib.sha256(
            _json_bytes(environment)
        ).hexdigest(),
    }
    if any(
        final_snapshot.get(field) != digest
        for field, digest in snapshot_hashes.items()
    ):
        raise RuntimeError(
            "T-STOP-MANIPULATION: prepare final snapshot digest mismatch"
        )
    bound_artifacts: list[Mapping[str, Any]] = []
    for row_key, artifact_keys in (
        ("validated_pools", ("pool", "generation_log")),
        (
            "validated_rule_bundles",
            ("checkpoint", "training_log"),
        ),
        (
            "validated_shared_real",
            ("checkpoint", "training_log"),
        ),
    ):
        rows = lineage.get(row_key)
        if not isinstance(rows, list):
            raise RuntimeError(
                f"T-STOP-MANIPULATION: prepare lacks {row_key}"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"T-STOP-MANIPULATION: invalid {row_key} row"
                )
            for artifact_key in artifact_keys:
                artifact = row.get(artifact_key)
                if not isinstance(artifact, Mapping):
                    raise RuntimeError(
                        "T-STOP-MANIPULATION: missing artifact binding "
                        f"{row_key}/{artifact_key}"
                    )
                bound_artifacts.append(artifact)
    for artifact in bound_artifacts:
        path = ROOT / str(artifact.get("path", ""))
        if (
            not path.is_file()
            or path.is_symlink()
            or artifact.get("bytes") != path.stat().st_size
            or artifact.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(
                "T-STOP-MANIPULATION: Stage-A artifact binding drift: "
                f"{artifact.get('path')}"
            )
    return prepare


def _assert_score_source_unchanged(
    prepare: Mapping[str, Any],
    *,
    allowed_untracked_outputs: Iterable[Path] = (),
) -> dict[str, Any]:
    """Recheck Stage-B source while allowing only outputs already published."""
    allowed = {
        portable_path(path) for path in allowed_untracked_outputs
    }
    records = _porcelain_records()
    rejected: list[str] = []
    observed: list[str] = []
    for record in records:
        if len(record) >= 4 and record[:2] == "??":
            relative = Path(record[3:]).as_posix()
            if relative in allowed:
                observed.append(record)
                continue
        rejected.append(record)
    if rejected:
        raise RuntimeError(
            "T-INCOMPLETE: source changed during Stage B: "
            f"{rejected}"
        )
    current = _implementation_source_records()
    source = prepare.get("source_provenance")
    prior = source.get("implementation") if isinstance(source, Mapping) else None
    if not isinstance(prior, Mapping):
        raise RuntimeError(
            "T-INCOMPLETE: prepare implementation snapshot is missing"
        )
    if any(
        not isinstance(prior.get(relative), Mapping)
        or prior[relative].get("sha256") != record["sha256"]
        for relative, record in current.items()
    ):
        raise RuntimeError("T-INCOMPLETE: implementation drift during Stage B")
    return {
        "implementation_matches_prepare": True,
        "allowed_untracked_outputs": sorted(allowed),
        "observed_allowed_untracked_records": sorted(observed),
        "rejected_status_records": [],
    }


def _configure_torch_for_scoring() -> Any:
    """Lazily configure the frozen E14 CPU inference runtime."""
    import torch

    torch.set_num_threads(e14.TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if (
        torch.get_num_threads() != e14.TORCH_THREADS
        or torch.get_num_interop_threads() != 1
    ):
        raise RuntimeError(
            "T-STOP-RUNTIME: could not establish frozen torch thread counts"
        )
    return torch


def _load_checkpoint_for_inference(
    torch: Any,
    path: Path,
    expected_sha256: str,
) -> Any:
    """Load one frozen CNN state dict after exact hash/shape checks."""
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_sha256
    ):
        raise RuntimeError(f"T-INCOMPLETE: checkpoint binding drift: {path}")
    model = e14._make_cnn(torch)
    expected = model.state_dict()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise RuntimeError(f"T-INCOMPLETE: checkpoint key mismatch: {path}")
    for key, expected_tensor in expected.items():
        tensor = state[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != tuple(expected_tensor.shape)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise RuntimeError(
                f"T-INCOMPLETE: invalid checkpoint tensor {key}: {path}"
            )
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(
            f"T-INCOMPLETE: checkpoint changed while loading: {path}"
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _prepare_checkpoint_bindings(
    prepare: Mapping[str, Any],
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[tuple[int, int], Mapping[str, Any]],
]:
    lineage = prepare.get("upstream_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("T-INCOMPLETE: prepare upstream lineage is missing")
    shared_rows = lineage.get("validated_shared_real")
    rule_rows = lineage.get("validated_rule_bundles")
    if not isinstance(shared_rows, list) or not isinstance(rule_rows, list):
        raise RuntimeError("T-INCOMPLETE: prepare checkpoint bindings missing")
    shared = {
        int(row["pipeline_seed"]): row
        for row in shared_rows
        if isinstance(row, Mapping)
    }
    rule = {
        (int(row["construction_seed"]), int(row["pipeline_seed"])): row
        for row in rule_rows
        if isinstance(row, Mapping)
    }
    if set(shared) != set(PIPELINE_SEEDS) or set(rule) != {
        (construction, pipeline)
        for construction in CONSTRUCTION_SEEDS
        for pipeline in PIPELINE_SEEDS
    }:
        raise RuntimeError("T-INCOMPLETE: prepare checkpoint grid incomplete")
    return shared, rule


def load_shared_real_models(
    torch: Any,
    prepare: Mapping[str, Any],
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Load only the five shared real references before the runtime gate."""
    shared, _ = _prepare_checkpoint_bindings(prepare)
    models: dict[int, Any] = {}
    records: list[dict[str, Any]] = []
    for pipeline in PIPELINE_SEEDS:
        binding = shared[pipeline]["checkpoint"]
        path = shared_real_checkpoint_path(pipeline)
        model = _load_checkpoint_for_inference(
            torch, path, str(binding["sha256"])
        )
        models[pipeline] = model
        records.append(
            {
                "arm": "shared_real",
                "pipeline_seed": pipeline,
                "path": portable_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "loaded_before_runtime_gate": True,
            }
        )
    return models, records


def load_rule_checkpoint_for_inference(
    torch: Any,
    *,
    construction_seed: int,
    pipeline_seed: int,
    binding: Mapping[str, Any],
    ledger: InferenceLedger,
) -> Any:
    """Enforce the runtime gate immediately before a Rule checkpoint load."""
    ledger.record_rule_load()
    return _load_checkpoint_for_inference(
        torch,
        rule_checkpoint_path(construction_seed, pipeline_seed),
        str(binding["sha256"]),
    )


def load_rule_models(
    torch: Any,
    prepare: Mapping[str, Any],
    ledger: InferenceLedger,
) -> tuple[dict[tuple[int, int], Any], list[dict[str, Any]]]:
    """Load all 25 Rule checkpoints only after continuity passed."""
    _, rule = _prepare_checkpoint_bindings(prepare)
    models: dict[tuple[int, int], Any] = {}
    records: list[dict[str, Any]] = []
    for construction in CONSTRUCTION_SEEDS:
        for pipeline in PIPELINE_SEEDS:
            key = (construction, pipeline)
            binding = rule[key]["checkpoint"]
            models[key] = load_rule_checkpoint_for_inference(
                torch,
                construction_seed=construction,
                pipeline_seed=pipeline,
                binding=binding,
                ledger=ledger,
            )
            path = rule_checkpoint_path(construction, pipeline)
            records.append(
                {
                    "arm": "rule",
                    "construction_seed": construction,
                    "pipeline_seed": pipeline,
                    "path": portable_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "loaded_after_runtime_gate": True,
                }
            )
    return models, records


def _predict_classes(
    torch: Any,
    model: Any,
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ledger: InferenceLedger,
) -> np.ndarray:
    standardized = e14._standardize(values, mean, std)
    logits = e14.predict_standardized_logits(
        torch,
        model,
        standardized,
        batch_size=e14.SCORING_BATCH_SIZE,
    )
    ledger.record_prediction(len(values))
    return logits.argmax(axis=1).astype(np.int8, copy=False)


def _score_l4_models(
    torch: Any,
    *,
    models: Mapping[Any, Any],
    identities: Sequence[Mapping[str, Any]],
    base_manifest: pd.DataFrame,
    bases: np.ndarray,
    latents: Sequence[Any],
    mean: np.ndarray,
    std: np.ndarray,
    ledger: InferenceLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score untouched bases and all 192,000 L4 transformations."""
    block_orders = {
        block: base_manifest.index[
            base_manifest["block_id"].astype(str).eq(block)
        ].to_numpy(np.int64)
        for block in BLOCKS
    }
    if any(len(order) != 2_000 for order in block_orders.values()):
        raise RuntimeError("T-INCOMPLETE: L4 block identity is incomplete")
    latent_lookup = {
        (
            latent.block_id,
            latent.block_position,
            latent.test_window_index,
            latent.attack_label,
        ): latent
        for latent in latents
    }

    normal_rows: list[dict[str, Any]] = []
    for identity in identities:
        predictions = _predict_classes(
            torch,
            models[identity["model_key"]],
            bases,
            mean,
            std,
            ledger,
        )
        for block, order in block_orders.items():
            correct = int(np.count_nonzero(predictions[order] == 0))
            row = {
                "arm": identity["arm"],
                "construction_seed": identity["construction_seed"],
                "pipeline_seed": identity["pipeline_seed"],
                "block_id": block,
                "control_scope": "Car-Hacking untouched normal base",
                "n": len(order),
                "normal_correct": correct,
                "normal_recall": correct / len(order),
                "fpr": 1.0 - correct / len(order),
                "checkpoint_sha256": identity["checkpoint_sha256"],
            }
            normal_rows.append(row)

    scenario_rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        order = block_orders[block]
        block_manifest = base_manifest.loc[order]
        block_bases = bases[order]
        for attack_label in sorted(e14.ATTACK_SPECS):
            attack = str(e14.ATTACK_SPECS[attack_label]["attack"])
            block_latents = [
                latent_lookup[
                    (
                        block,
                        int(row.block_position),
                        int(row.test_window_index),
                        attack_label,
                    )
                ]
                for row in block_manifest.itertuples(index=False)
            ]
            for id_stratum in ID_STRATA:
                for p, s, d in FACTOR_CELLS:
                    transformed = np.stack(
                        [
                            e14.apply_factorial_transform(
                                base,
                                latent,
                                attack_label=attack_label,
                                id_stratum=id_stratum,
                                p=p,
                                s=s,
                                d=d,
                            )
                            for base, latent in zip(
                                block_bases, block_latents, strict=True
                            )
                        ]
                    ).astype(np.float32, copy=False)
                    for identity in identities:
                        predictions = _predict_classes(
                            torch,
                            models[identity["model_key"]],
                            transformed,
                            mean,
                            std,
                            ledger,
                        )
                        exact_correct = int(
                            np.count_nonzero(predictions == attack_label)
                        )
                        binary_correct = int(
                            np.count_nonzero(predictions != 0)
                        )
                        scenario_rows.append(
                            {
                                "arm": identity["arm"],
                                "construction_seed":
                                    identity["construction_seed"],
                                "pipeline_seed":
                                    identity["pipeline_seed"],
                                "block_id": block,
                                "attack": attack,
                                "attack_label": attack_label,
                                "id_stratum": id_stratum,
                                "P": p,
                                "S": s,
                                "D": d,
                                "cell": f"{p}{s}{d}",
                                "n": len(transformed),
                                "exact_correct": exact_correct,
                                "binary_correct": binary_correct,
                                "exact_recall":
                                    exact_correct / len(transformed),
                                "binary_recall":
                                    binary_correct / len(transformed),
                                "checkpoint_sha256":
                                    identity["checkpoint_sha256"],
                            }
                        )
    return pd.DataFrame(normal_rows), pd.DataFrame(scenario_rows)


def _shared_real_l4_identities(
    models: Mapping[int, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model_key": pipeline,
            "arm": "real_ms",
            "construction_seed": "shared_reference",
            "pipeline_seed": pipeline,
            "checkpoint_sha256": sha256_file(
                shared_real_checkpoint_path(pipeline)
            ),
        }
        for pipeline in PIPELINE_SEEDS
        if pipeline in models
    ]


def _rule_l4_identities(
    models: Mapping[tuple[int, int], Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model_key": (construction, pipeline),
            "arm": "rule",
            "construction_seed": construction,
            "pipeline_seed": pipeline,
            "checkpoint_sha256": sha256_file(
                rule_checkpoint_path(construction, pipeline)
            ),
        }
        for construction in CONSTRUCTION_SEEDS
        for pipeline in PIPELINE_SEEDS
        if (construction, pipeline) in models
    ]


def _build_l4_cells(scenario: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "construction_seed",
        "pipeline_seed",
        "block_id",
        "id_stratum",
        "P",
        "S",
        "D",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in scenario.groupby(keys, sort=True):
        if len(group) != 2 or set(group["attack"].astype(str)) != set(ATTACKS):
            raise RuntimeError(
                "T-INCOMPLETE: L4 equal-attack macro pairing failed"
            )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "cell": f"{int(key[-3])}{int(key[-2])}{int(key[-1])}",
                "attacks": 2,
                "n": 2_000,
                "n_per_attack": 2_000,
                "macro_weighting": "equal Gear/RPM",
                "exact_recall": float(group["exact_recall"].mean()),
                "binary_recall": float(group["binary_recall"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_l4_delta(
    rule_scenario: pd.DataFrame,
    frozen_real_scenario: pd.DataFrame,
) -> pd.DataFrame:
    real = _scenario_count_frame(
        frozen_real_scenario,
        arm="real_ms",
        label="frozen shared-real scenario",
    )
    keys = [
        "pipeline_seed",
        "block_id",
        "attack",
        "id_stratum",
        "P",
        "S",
        "D",
    ]
    real_columns = [
        *keys,
        "n",
        "exact_correct",
        "binary_correct",
        "exact_recall",
        "binary_recall",
    ]
    joined = rule_scenario.merge(
        real[real_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("_rule", "_real"),
    )
    if len(joined) != EXPECTED_L4_SCENARIO_ROWS or joined[
        "n_real"
    ].isna().any():
        raise RuntimeError("T-INCOMPLETE: shared-real L4 join failed")
    if not joined["n_rule"].eq(joined["n_real"]).all():
        raise RuntimeError("T-INCOMPLETE: L4 pairing denominator mismatch")
    result = joined[
        [
            "construction_seed",
            *keys,
            "cell",
            "n_rule",
            "exact_correct_rule",
            "binary_correct_rule",
            "exact_recall_rule",
            "binary_recall_rule",
            "exact_correct_real",
            "binary_correct_real",
            "exact_recall_real",
            "binary_recall_real",
        ]
    ].rename(columns={"n_rule": "n"})
    result.insert(0, "contrast", "rule_minus_shared_real")
    result["exact_recall"] = (
        result["exact_recall_rule"] - result["exact_recall_real"]
    )
    result["binary_recall"] = (
        result["binary_recall_rule"] - result["binary_recall_real"]
    )
    result["shared_real_reference_reused_once"] = True
    return result


def _score_l2_models(
    torch: Any,
    *,
    models: Mapping[Any, Any],
    identities: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
    ledger: InferenceLedger,
) -> pd.DataFrame:
    """Score the complete three-block, thirteen-scenario L2 panel."""
    rows: list[dict[str, Any]] = []
    for identity in identities:
        model = models[identity["model_key"]]
        for block in BLOCKS:
            values = np.asarray(blocks[block]["x"], dtype=np.float32)
            predictions = _predict_classes(
                torch, model, values, mean, std, ledger
            )
            scenario_ids = np.asarray(blocks[block]["scenario_id"]).astype(str)
            attacks = np.asarray(blocks[block]["attack_type"]).astype(str)
            settings = np.asarray(blocks[block]["setting"]).astype(str)
            labels = np.asarray(
                blocks[block]["y_attack_type"], dtype=np.int8
            )
            for scenario_id in (
                "Normal_normal",
                *(
                    f"{attack}_{setting}"
                    for attack in L2_ATTACKS
                    for setting in ("low", "medium", "high")
                ),
            ):
                selected = np.flatnonzero(scenario_ids == scenario_id)
                expected_n = 6_000 if scenario_id == "Normal_normal" else 500
                if len(selected) != expected_n:
                    raise RuntimeError(
                        "T-INCOMPLETE: L2 scenario cardinality mismatch: "
                        f"{block}/{scenario_id}/{len(selected)}"
                    )
                attack = str(attacks[selected[0]])
                setting = str(settings[selected[0]])
                if scenario_id == "Normal_normal":
                    normal_correct = int(
                        np.count_nonzero(predictions[selected] == 0)
                    )
                    row = {
                        "normal_correct": normal_correct,
                        "normal_recall": normal_correct / expected_n,
                        "fpr": 1.0 - normal_correct / expected_n,
                        "exact_correct": pd.NA,
                        "binary_correct": pd.NA,
                        "exact_recall": np.nan,
                        "binary_recall": np.nan,
                    }
                else:
                    label_values = np.unique(labels[selected])
                    if len(label_values) != 1 or int(label_values[0]) not in (
                        1, 2, 3, 4
                    ):
                        raise RuntimeError(
                            "T-INCOMPLETE: invalid L2 attack labels"
                        )
                    label = int(label_values[0])
                    exact_correct = int(
                        np.count_nonzero(predictions[selected] == label)
                    )
                    binary_correct = int(
                        np.count_nonzero(predictions[selected] != 0)
                    )
                    row = {
                        "normal_correct": pd.NA,
                        "normal_recall": np.nan,
                        "fpr": np.nan,
                        "exact_correct": exact_correct,
                        "binary_correct": binary_correct,
                        "exact_recall": exact_correct / expected_n,
                        "binary_recall": binary_correct / expected_n,
                    }
                rows.append(
                    {
                        "arm": identity["l2_arm"],
                        "construction_seed":
                            identity["construction_seed"],
                        "pipeline_seed": identity["pipeline_seed"],
                        "block_id": block,
                        "scenario_id": scenario_id,
                        "attack": attack,
                        "setting": setting,
                        "n": expected_n,
                        **row,
                        "checkpoint_sha256":
                            identity["checkpoint_sha256"],
                    }
                )
    return pd.DataFrame(rows)


def _shared_real_l2_identities(
    models: Mapping[int, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model_key": pipeline,
            "l2_arm": "shared_real",
            "construction_seed": "shared_reference",
            "pipeline_seed": pipeline,
            "checkpoint_sha256": sha256_file(
                shared_real_checkpoint_path(pipeline)
            ),
        }
        for pipeline in PIPELINE_SEEDS
        if pipeline in models
    ]


def _rule_l2_identities(
    models: Mapping[tuple[int, int], Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model_key": (construction, pipeline),
            "l2_arm": "rule",
            "construction_seed": construction,
            "pipeline_seed": pipeline,
            "checkpoint_sha256": sha256_file(
                rule_checkpoint_path(construction, pipeline)
            ),
        }
        for construction in CONSTRUCTION_SEEDS
        for pipeline in PIPELINE_SEEDS
        if (construction, pipeline) in models
    ]


def _build_l2_delta(by_scenario: pd.DataFrame) -> pd.DataFrame:
    attack_rows = by_scenario[
        by_scenario["scenario_id"].astype(str).ne("Normal_normal")
    ].copy()
    rule = attack_rows[attack_rows["arm"].astype(str).eq("rule")].copy()
    real = attack_rows[
        attack_rows["arm"].astype(str).eq("shared_real")
    ].copy()
    real_keys = ["pipeline_seed", "block_id", "attack", "setting"]
    real = real[
        [
            *real_keys,
            "n",
            "exact_correct",
            "binary_correct",
            "exact_recall",
            "binary_recall",
        ]
    ]
    joined = rule.merge(
        real,
        on=real_keys,
        how="left",
        validate="many_to_one",
        suffixes=("_rule", "_real"),
    )
    if len(joined) != EXPECTED_L2_DELTA_ROWS or joined["n_real"].isna().any():
        raise RuntimeError("T-INCOMPLETE: shared-real L2 join failed")
    if not joined["n_rule"].eq(joined["n_real"]).all():
        raise RuntimeError("T-INCOMPLETE: L2 denominator mismatch")
    result = joined[
        [
            "construction_seed",
            *real_keys,
            "scenario_id",
            "n_rule",
            "exact_correct_rule",
            "binary_correct_rule",
            "exact_recall_rule",
            "binary_recall_rule",
            "exact_correct_real",
            "binary_correct_real",
            "exact_recall_real",
            "binary_recall_real",
        ]
    ].rename(columns={"n_rule": "n"})
    result.insert(0, "contrast", "rule_minus_shared_real")
    result["exact_recall"] = (
        result["exact_recall_rule"] - result["exact_recall_real"]
    )
    result["binary_recall"] = (
        result["binary_recall_rule"] - result["binary_recall_real"]
    )
    result["shared_real_reference_reused_once"] = True
    return result


def run_rule_after_runtime_continuity(
    continuity_table: pd.DataFrame,
    *,
    ledger: InferenceLedger,
    rule_operation: Any,
    expected_rows: int = EXPECTED_RUNTIME_CONTINUITY_ROWS,
    continuity_publish: Any | None = None,
) -> Any:
    """Publish/pass runtime continuity before invoking any Rule operation."""
    gate = require_real_runtime_continuity(
        continuity_table, expected_rows=expected_rows
    )
    if continuity_publish is not None:
        continuity_publish()
    ledger.runtime_gate_passed = True
    result = rule_operation()
    return result, gate


def run_score() -> None:
    """Execute Stage B.

    The scoring implementation is intentionally factored below so the runtime
    gate is an executable control-flow boundary, not a post-hoc assertion.
    """
    _run_score_production()


def _payload_record(
    path: Path,
    payload: bytes,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": portable_path(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _run_score_production() -> None:
    """Run the complete runtime-first E15 L4/L2 scoring stage."""
    for name, path in OUTPUT_PATHS.items():
        if _lexists(path):
            raise FileExistsError(
                f"refusing to overwrite E15 Stage-B target {name}: {path}"
            )
    started = utc_now()
    prepare = _validate_prepare_record()
    prepare_hash = sha256_file(PREPARE_RECORD)
    source = source_provenance_for_score(prepare)

    l4_identity = regenerate_l4_identity()
    l2_identity, l2_checks, l2_blocks = regenerate_l2_blocks()
    if l4_identity != prepare.get("l4_transformation"):
        raise RuntimeError(
            "T-STOP-MANIPULATION: Stage-B L4 identity differs from Stage A"
        )
    if l2_identity != prepare.get("l2_transformation"):
        raise RuntimeError(
            "T-STOP-MANIPULATION: Stage-B L2 identity differs from Stage A"
        )
    if l2_checks.to_dict("records") != prepare.get(
        "l2_manipulation_checks"
    ):
        raise RuntimeError(
            "T-STOP-MANIPULATION: Stage-B L2 manipulation rows drifted"
        )

    mean, std, standardizer_hashes = e14.fit_registered_standardizer()
    prepared_standardizer = prepare.get("standardizer")
    if (
        not isinstance(prepared_standardizer, Mapping)
        or prepared_standardizer.get("hashes") != standardizer_hashes
    ):
        raise RuntimeError(
            "T-STOP-MANIPULATION: Stage-B standardizer differs from Stage A"
        )
    _, base_manifest, bases, latents = _load_e14_construction()
    torch = _configure_torch_for_scoring()
    evaluation_environment = evaluation_environment_record(
        stage="score", torch_module=torch
    )
    validate_evaluation_environment_record(
        evaluation_environment, expected_stage="score"
    )
    ledger = InferenceLedger()

    # The only models loaded for inference before continuity are the five
    # shared-real references.
    shared_models, shared_checkpoint_records = load_shared_real_models(
        torch, prepare
    )
    fresh_real_normal, fresh_real_scenario = _score_l4_models(
        torch,
        models=shared_models,
        identities=_shared_real_l4_identities(shared_models),
        base_manifest=base_manifest,
        bases=bases,
        latents=latents,
        mean=mean,
        std=std,
        ledger=ledger,
    )
    frozen_scenario = pd.read_csv(E14_SCENARIO)
    frozen_normal = pd.read_csv(E14_NORMAL)
    continuity = build_real_runtime_continuity(
        fresh_real_scenario,
        frozen_scenario,
        fresh_real_normal,
        frozen_normal,
    )
    continuity_bytes = _csv_bytes(continuity)

    def score_rule_phase() -> dict[str, Any]:
        # Shared-real L2 values are stored once per pipeline.  Rule checkpoint
        # loading remains the first Rule operation and is ledger-gated.
        shared_l2 = _score_l2_models(
            torch,
            models=shared_models,
            identities=_shared_real_l2_identities(shared_models),
            blocks=l2_blocks,
            mean=mean,
            std=std,
            ledger=ledger,
        )
        rule_models, rule_checkpoint_records = load_rule_models(
            torch, prepare, ledger
        )
        rule_normal, rule_scenario = _score_l4_models(
            torch,
            models=rule_models,
            identities=_rule_l4_identities(rule_models),
            base_manifest=base_manifest,
            bases=bases,
            latents=latents,
            mean=mean,
            std=std,
            ledger=ledger,
        )
        rule_cell = _build_l4_cells(rule_scenario)
        l4_delta = _build_l4_delta(rule_scenario, frozen_scenario)
        l4_counts = validate_l4_scoring_tables(
            rule_normal, rule_scenario, rule_cell, l4_delta
        )

        anchor_normal = rule_normal[
            pd.to_numeric(
                rule_normal["construction_seed"], errors="coerce"
            ).eq(ANCHOR_CONSTRUCTION_SEED)
        ].copy()
        anchor_scenario = rule_scenario[
            pd.to_numeric(
                rule_scenario["construction_seed"], errors="coerce"
            ).eq(ANCHOR_CONSTRUCTION_SEED)
        ].copy()
        anchor = build_anchor_outcome_continuity(
            anchor_scenario,
            frozen_scenario,
            anchor_normal,
            frozen_normal,
        )
        require_unique(
            anchor,
            CONTINUITY_KEY,
            EXPECTED_L4_ROWS["anchor_continuity"],
            "anchor outcome continuity",
        )

        rule_l2 = _score_l2_models(
            torch,
            models=rule_models,
            identities=_rule_l2_identities(rule_models),
            blocks=l2_blocks,
            mean=mean,
            std=std,
            ledger=ledger,
        )
        l2_scenario = pd.concat(
            [shared_l2, rule_l2], ignore_index=True
        )
        l2_delta = _build_l2_delta(l2_scenario)
        l2_counts = validate_l2_scoring_tables(
            l2_scenario, l2_delta
        )
        return {
            "l4_normal": rule_normal,
            "l4_scenario": rule_scenario,
            "l4_cell": rule_cell,
            "l4_delta": l4_delta,
            "anchor_continuity": anchor,
            "l2_scenario": l2_scenario,
            "l2_delta": l2_delta,
            "l4_counts": l4_counts,
            "l2_counts": l2_counts,
            "rule_checkpoint_records": rule_checkpoint_records,
        }

    scored, runtime_gate = run_rule_after_runtime_continuity(
        continuity,
        ledger=ledger,
        rule_operation=score_rule_phase,
        expected_rows=EXPECTED_RUNTIME_CONTINUITY_ROWS,
        continuity_publish=lambda: publish_bundle_no_clobber(
            [
                (
                    OUTPUT_PATHS["runtime_continuity"],
                    continuity_bytes,
                )
            ]
        ),
    )
    if (
        ledger.rule_checkpoint_loads != 25
        or ledger.runtime_gate_passed is not True
    ):
        raise RuntimeError(
            "T-INCOMPLETE: Rule checkpoint runtime-order ledger mismatch"
        )

    environment_snapshot = assert_score_environment_unchanged(
        evaluation_environment, torch_module=torch
    )
    final_prepare = _validate_prepare_record()
    if (
        sha256_file(PREPARE_RECORD) != prepare_hash
        or final_prepare.get("source_commit") != prepare.get("source_commit")
    ):
        raise RuntimeError("T-INCOMPLETE: prepare record changed during Stage B")
    final_source = _assert_score_source_unchanged(
        prepare,
        allowed_untracked_outputs=(
            OUTPUT_PATHS["runtime_continuity"],
        ),
    )
    table_frames = {
        "l4_normal": scored["l4_normal"],
        "l4_scenario": scored["l4_scenario"],
        "l4_cell": scored["l4_cell"],
        "l4_delta": scored["l4_delta"],
        "anchor_continuity": scored["anchor_continuity"],
        "l2_scenario": scored["l2_scenario"],
        "l2_delta": scored["l2_delta"],
    }
    table_payloads = {
        name: _csv_bytes(frame)
        for name, frame in table_frames.items()
    }
    runtime_record = {
        "path": portable_path(OUTPUT_PATHS["runtime_continuity"]),
        "bytes": OUTPUT_PATHS["runtime_continuity"].stat().st_size,
        "sha256": sha256_file(OUTPUT_PATHS["runtime_continuity"]),
        "rows": len(continuity),
    }
    completed = utc_now()
    log = {
        "schema_version": "e15.rule_construction_crossing_log.v1",
        "status": "T-PASS",
        "stage_status": "evaluation_complete",
        "started_utc": started,
        "completed_utc": completed,
        "normal_control_reported_before_attack_results": True,
        "runtime_continuity_gate": {
            "status": "PASS",
            "technical_status": "T-PASS",
            **runtime_gate,
        },
        "rule_inference_started_after_runtime_continuity": True,
        "rule_checkpoint_loads": ledger.rule_checkpoint_loads,
        "inference_calls": ledger.inference_calls,
        "prediction_rows": ledger.prediction_rows,
        "environment": evaluation_environment,
        **environment_snapshot,
        "l4_counts": scored["l4_counts"],
        "l2_counts": scored["l2_counts"],
        "scientific_verdict_status": "pending_registered_analyzer",
    }
    log_bytes = _json_bytes(log)
    output_records = {
        name: _payload_record(
            OUTPUT_PATHS[name],
            payload,
            rows=len(table_frames[name]),
        )
        for name, payload in table_payloads.items()
    }
    output_records["runtime_continuity"] = runtime_record
    output_records["log"] = _payload_record(
        OUTPUT_PATHS["log"], log_bytes
    )
    run = {
        "schema_version": "e15.rule_construction_crossing_run.v1",
        "status": "T-PASS",
        "technical_status": "T-PASS",
        "stage_status": "evaluation_complete",
        "started_utc": started,
        "completed_utc": completed,
        "source_commit": source["source_commit"],
        "source_provenance": source,
        "source_snapshot_before_publication": final_source,
        "environment": evaluation_environment,
        **environment_snapshot,
        "prepare": {
            "path": portable_path(PREPARE_RECORD),
            "bytes": PREPARE_RECORD.stat().st_size,
            "sha256": sha256_file(PREPARE_RECORD),
        },
        "transformations": {
            "l4": l4_identity,
            "l2": l2_identity,
            "match_prepare": True,
        },
        "standardizer_hashes": standardizer_hashes,
        "checkpoint_validation": {
            "shared_real": shared_checkpoint_records,
            "rule": scored["rule_checkpoint_records"],
        },
        "runtime_continuity_gate": {
            "status": "PASS",
            "technical_status": "T-PASS",
            **runtime_gate,
            "artifact": runtime_record,
        },
        "rule_inference_started_after_runtime_continuity": True,
        "shared_real_reference_stored_once_per_pipeline": True,
        "normal_control_reported_before_attack_results": True,
        "exact_and_binary_endpoints_separate": True,
        "completeness": {
            "l4": scored["l4_counts"],
            "l2": scored["l2_counts"],
            "runtime_continuity_rows": len(continuity),
            "anchor_continuity_rows": len(
                scored["anchor_continuity"]
            ),
        },
        "inference_ledger": {
            "inference_calls": ledger.inference_calls,
            "prediction_rows": ledger.prediction_rows,
            "rule_checkpoint_loads": ledger.rule_checkpoint_loads,
            "runtime_gate_passed": ledger.runtime_gate_passed,
        },
        "outputs": output_records,
        "scientific_verdict_status": "pending_registered_analyzer",
        "next_required_stage": (
            "registered analyzer publishes crossed effects, hierarchy, "
            "dispersion, L2 summary, exactly one verdict, and final manifest"
        ),
    }
    run_bytes = _json_bytes(run)
    publish_bundle_no_clobber(
        [
            *[
                (OUTPUT_PATHS[name], table_payloads[name])
                for name in (
                    "l4_normal",
                    "l4_scenario",
                    "l4_cell",
                    "l4_delta",
                    "anchor_continuity",
                    "l2_scenario",
                    "l2_delta",
                )
            ],
            (OUTPUT_PATHS["log"], log_bytes),
            (OUTPUT_PATHS["run_record"], run_bytes),
        ]
    )
    print(
        json.dumps(
            {
                "status": "T-PASS",
                "run_record": portable_path(OUTPUT_PATHS["run_record"]),
                "runtime_continuity_rows": len(continuity),
                "rule_checkpoint_loads": ledger.rule_checkpoint_loads,
                "scientific_verdict_status":
                    "pending_registered_analyzer",
            },
            sort_keys=True,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or score the preregistered E15 Rule construction-seed "
            "crossing."
        )
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument(
        "--prepare-only", action="store_true",
        help="regenerate/audit transformations without model inference",
    )
    stage.add_argument(
        "--score", action="store_true",
        help="run shared-real continuity first, then the complete E15 panel",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_only:
        result = run_prepare_only()
        print(json.dumps({
            "status": result["status"],
            "prepare_record": portable_path(PREPARE_RECORD),
        }, sort_keys=True))
    else:
        run_score()


if __name__ == "__main__":
    main()
