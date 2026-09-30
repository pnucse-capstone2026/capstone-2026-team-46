#!/usr/bin/env python3
"""Train the preregistered E15 construction-seed × pipeline-seed grid.

The default mode performs a read-only preflight.  ``--execute`` first
publishes that preflight and then launches exactly 25 fresh child processes in
the frozen construction-major order.  ``--child-fit`` is an internal,
explicit-only interface used by those commands; it never discovers a default
pool and never combines the construction and pipeline RNG axes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import lib_common as lc
from generate_rule_construction_sensitivity import (
    CONSTRUCTION_SEEDS,
    EXPECTED_ENVIRONMENT_STABLE_SHA256,
    IMPLEMENTATION_PATHS,
    OUTPUT_VERSION,
    PIPELINE_SEEDS,
    POOL_AUDIT_SCHEMA,
    POOL_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA,
    PREREG_COMMIT,
    REPO,
    RUN_SCHEMA,
    SAMPLING_POLICY,
    TRAIN_SHA256,
    _environment_record,
    atomic_publish_bundle,
    generation_record_paths,
    generation_output_status_paths,
    path_lexists,
    pool_identifier,
    pool_target_map,
    validate_hash_snapshot,
    validate_source_status,
    validate_pool_archive,
)
from run_e14_matched_real_training import (
    EXPECTED_STANDARDIZER_SHA256,
    EXPECTED_STATE_SHAPES,
    canonical_json_sha256,
    repo_relative,
    sha256_file,
    validate_checkpoint,
    verify_derived_standardizer,
)
from train_generator_extension_cnn import CNN1D, predict


E15_DIR = (
    REPO / "journal" / "experiments" / "e15_rule_construction_crossing"
)
MODEL_DIR = REPO / "journal" / "models" / "generator_extension"
LOG_DIR = REPO / "journal" / "results" / "logs"
TABLE_DIR = REPO / "journal" / "results" / "tables"

REAL_TRAIN_WINDOWS = 262_149
SYNTHETIC_RATIO = 0.30
SYNTHETIC_TOTAL = 78_645
PER_CLASS_REQUESTED = {1: 19_662, 2: 19_661, 3: 19_661, 4: 19_661}
SAMPLING_SEED_OFFSET = 300
MAX_STEPS = 6_156
VALIDATION_CADENCE = 513
VALIDATION_CHECKPOINTS = 12
BATCH_SIZE = 512
VALIDATION_BATCH_SIZE = 1_024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_TAG = f"matchedsteps_e15_{OUTPUT_VERSION}"
TRAINING_PREFLIGHT_SCHEMA = "e15.training_preflight.v1"
TRAINING_RUN_SCHEMA = "e15.training_run.v1"
TRAINING_LOG_SCHEMA = "e15.rule_construction_training_log.v1"
SAMPLING_AUDIT_SCHEMA = "e15.rule_construction_sampling_audit.v1"
TRAINING_MANIFEST_SCHEMA = "e15.rule_construction_training_manifest.v1"

VAL_SHA256 = (
    "b4920e7f60fbf38b38e12976129f120ebff160a21a4c77093f3746f3c1aa218e"
)
TEST_SHA256 = (
    "4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7"
)
ANCHOR_INDEX_SHA256 = {
    7: "4ff00984e1dd97a48447b2abac6c347eb8967b20e3c90782db081f9681c11581",
    42: "bdfcbdf15dcd5001122de17dd721508c5347353a022becd20df94c0e59a0f917",
    123: "22bd0a549e33ad17662a6b5ac84065a8b229768aace1c5721ece4a2408255ba5",
    2026: "59a6f31755fb18e7f19c4ba9e819b050f4d5db617854e0f2c2e78f4ae31a3319",
    3407: "3ed08fb8b14d3796264fba3cdac76ca7563fd0d951ad295209d0798fd875e3bf",
}


class E15TrainingError(RuntimeError):
    """A preregistered E15 training gate failed."""


def _training_stop(message: str) -> E15TrainingError:
    return E15TrainingError(f"T-STOP-TRAINING: {message}")


def _incomplete(message: str) -> E15TrainingError:
    return E15TrainingError(f"T-INCOMPLETE: {message}")


@dataclass(frozen=True)
class TrainingJob:
    """One explicit cell in the frozen 5 × 5 training grid."""

    construction_seed: int
    pipeline_seed: int
    pool_path: Path
    pool_identifier: str
    command: tuple[str, ...]
    checkpoint_path: Path
    log_path: Path

    @property
    def sampling_seed(self) -> int:
        """The frozen consumer RNG seed, independent of construction seed."""
        return self.pipeline_seed + SAMPLING_SEED_OFFSET


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def portable_repo_path(path: Path, repo: Path = REPO) -> str:
    """Use the canonical repo-relative schema, with a test-fixture fallback."""
    path = Path(path).absolute()
    repo = Path(repo).absolute()
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def checkpoint_path(
    construction_seed: int,
    pipeline_seed: int,
    repo: Path = REPO,
) -> Path:
    return (
        Path(repo)
        / "journal"
        / "models"
        / "generator_extension"
        / (
            "cnn_rule_0p30_"
            f"cseed{int(construction_seed)}_"
            f"{MODEL_TAG}_seed{int(pipeline_seed)}.pt"
        )
    )


def training_log_path(
    construction_seed: int,
    pipeline_seed: int,
    repo: Path = REPO,
) -> Path:
    return (
        Path(repo)
        / "journal"
        / "results"
        / "logs"
        / (
            "train_cnn_rule_0p30_"
            f"cseed{int(construction_seed)}_"
            f"{MODEL_TAG}_seed{int(pipeline_seed)}.log"
        )
    )


def training_record_paths(repo: Path = REPO) -> dict[str, Path]:
    repo = Path(repo)
    e15_dir = (
        repo / "journal" / "experiments" / "e15_rule_construction_crossing"
    )
    return {
        "preflight": (
            e15_dir / f"training_preflight_{OUTPUT_VERSION}.json"
        ),
        "run": e15_dir / f"training_run_{OUTPUT_VERSION}.json",
        "sampling_audit": (
            repo
            / "journal"
            / "results"
            / "tables"
            / (
                "e15_rule_construction_sampling_audit_"
                f"{OUTPUT_VERSION}.csv"
            )
        ),
        "training_manifest": (
            repo
            / "journal"
            / "results"
            / "tables"
            / (
                "e15_rule_construction_training_manifest_"
                f"{OUTPUT_VERSION}.csv"
            )
        ),
    }


def training_output_status_paths(
    repo: Path = REPO,
    *,
    completed_jobs: Sequence[TrainingJob] = (),
) -> set[str]:
    """Exact upstream/preflight/completed paths allowed during training."""
    repo = Path(repo).resolve()
    allowed = set(generation_output_status_paths(repo))
    allowed.add(
        repo_relative(training_record_paths(repo)["preflight"], repo)
    )
    for job in completed_jobs:
        allowed.add(repo_relative(job.checkpoint_path, repo))
        allowed.add(repo_relative(job.log_path, repo))
    return allowed


def _generation_log_path(construction_seed: int, repo: Path) -> Path:
    return (
        repo
        / "journal"
        / "results"
        / "logs"
        / (
            f"e15_generate_rule_cseed{construction_seed}_"
            f"{OUTPUT_VERSION}.json"
        )
    )


def _child_command(
    *,
    construction_seed: int,
    pipeline_seed: int,
    pool_path: Path,
    pool_id: str,
    checkpoint: Path,
    log: Path,
    repo: Path,
) -> tuple[str, ...]:
    wrapper = (
        repo
        / "journal"
        / "scripts"
        / "train_rule_construction_seed_crossing.py"
    )
    generation_log = _generation_log_path(construction_seed, repo)
    return (
        sys.executable,
        os.fspath(wrapper),
        "--child-fit",
        "--construction-seed",
        str(construction_seed),
        "--pipeline-seed",
        str(pipeline_seed),
        "--synthetic-pool",
        os.fspath(pool_path),
        "--pool-identifier",
        pool_id,
        "--pool-schema",
        POOL_SCHEMA_VERSION,
        "--generation-log",
        os.fspath(generation_log),
        "--sampling-policy",
        SAMPLING_POLICY,
        "--checkpoint-output",
        os.fspath(checkpoint),
        "--log-output",
        os.fspath(log),
    )


def build_training_grid(repo: Path = REPO) -> tuple[TrainingJob, ...]:
    """Return all 25 jobs in construction-major, pipeline-minor order."""
    repo = Path(repo).resolve()
    pool_targets = pool_target_map(repo)
    jobs: list[TrainingJob] = []
    for construction_seed in CONSTRUCTION_SEEDS:
        pool = pool_targets[construction_seed]
        for pipeline_seed in PIPELINE_SEEDS:
            checkpoint = checkpoint_path(
                construction_seed, pipeline_seed, repo
            )
            log = training_log_path(construction_seed, pipeline_seed, repo)
            command = _child_command(
                construction_seed=construction_seed,
                pipeline_seed=pipeline_seed,
                pool_path=pool.pool_path,
                pool_id=pool.pool_identifier,
                checkpoint=checkpoint,
                log=log,
                repo=repo,
            )
            jobs.append(
                TrainingJob(
                    construction_seed=construction_seed,
                    pipeline_seed=pipeline_seed,
                    pool_path=pool.pool_path,
                    pool_identifier=pool.pool_identifier,
                    command=command,
                    checkpoint_path=checkpoint,
                    log_path=log,
                )
            )
    result = tuple(jobs)
    if len(result) != 25:
        raise AssertionError("E15 training grid must contain exactly 25 jobs")
    identities = {
        (job.construction_seed, job.pipeline_seed) for job in result
    }
    if len(identities) != 25:
        raise AssertionError("E15 training grid contains duplicate seed cells")
    return result


def training_target_paths(
    grid: Sequence[TrainingJob],
) -> tuple[Path, ...]:
    """Return the exact 50 checkpoint/log targets in grid order."""
    paths = tuple(
        path
        for job in grid
        for path in (job.checkpoint_path, job.log_path)
    )
    if len(paths) != 2 * len(grid) or len(set(paths)) != len(paths):
        raise AssertionError("training target paths must be distinct pairs")
    return paths


def assert_training_targets_absent(
    paths: Sequence[Path],
) -> dict[str, Any]:
    collisions = [Path(path) for path in paths if path_lexists(Path(path))]
    if collisions:
        raise FileExistsError(
            "T-STOP-TRAINING: checkpoint/log target collision(s): "
            + ", ".join(str(path) for path in collisions)
        )
    return {
        "expected_count": len(paths),
        "verified_absent_count": len(paths),
        "all_absent": True,
        "paths": [str(path) for path in paths],
    }


def sampling_indices_sha256(indices: np.ndarray) -> str:
    """Hash one index vector as contiguous little-endian signed int64 bytes."""
    vector = np.ascontiguousarray(np.asarray(indices).reshape(-1), dtype="<i8")
    return hashlib.sha256(vector.tobytes(order="C")).hexdigest()


def _class_requests(
    total_count: int,
    classes: Sequence[int],
) -> dict[int, int]:
    total_count = int(total_count)
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be a non-empty unique sequence")
    quotient, remainder = divmod(total_count, len(classes))
    return {
        int(label): quotient + int(position < remainder)
        for position, label in enumerate(classes)
    }


def sample_indices_without_replacement(
    y: np.ndarray,
    total_count: int,
    sampling_seed: int,
    classes: Sequence[int] = (1, 2, 3, 4),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen class-prefix, choice-without-replacement draw."""
    labels = np.asarray(y).reshape(-1)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError(
            "T-STOP-TRAINING: synthetic label vector must be non-empty"
        )
    classes = tuple(int(label) for label in classes)
    requests = _class_requests(total_count, classes)
    rng = np.random.default_rng(int(sampling_seed))
    selected_parts: list[np.ndarray] = []
    per_class: dict[str, Any] = {}
    for label in classes:
        available = np.flatnonzero(labels == label)
        requested = requests[label]
        if len(available) < requested:
            raise ValueError(
                "T-STOP-TRAINING: "
                f"class {label} capacity {len(available)} < {requested}"
            )
        drawn = np.asarray(
            rng.choice(available, size=requested, replace=False),
            dtype="<i8",
        )
        unique = int(np.unique(drawn).size)
        if unique != requested:
            raise ValueError(
                "T-STOP-TRAINING: "
                f"class {label} no-replacement invariant failed"
            )
        selected_parts.append(drawn)
        per_class[str(label)] = {
            "available": int(len(available)),
            "requested": requested,
            "drawn": int(len(drawn)),
            "unique": unique,
            "repeated": int(len(drawn) - unique),
        }
    indices = np.ascontiguousarray(
        np.concatenate(selected_parts), dtype="<i8"
    )
    rng.shuffle(indices)
    unique_total = int(np.unique(indices).size)
    if len(indices) != int(total_count) or unique_total != int(total_count):
        raise ValueError(
            "T-STOP-TRAINING: global no-replacement invariant failed"
        )
    audit = {
        "schema_version": SAMPLING_AUDIT_SCHEMA,
        "sampling_policy": SAMPLING_POLICY,
        "sampling_seed": int(sampling_seed),
        "rng": "numpy.random.default_rng/PCG64",
        "class_order": list(classes),
        "selection": "per-class choice(replace=False), concatenate, one shuffle",
        "requested_total": int(total_count),
        "drawn_total": int(len(indices)),
        "unique_total": unique_total,
        "repeated_total": int(len(indices) - unique_total),
        "total": {
            "requested": int(total_count),
            "drawn": int(len(indices)),
            "unique": unique_total,
            "repeated": int(len(indices) - unique_total),
        },
        "per_class": per_class,
        "index_dtype": "<i8",
        "index_sha256": sampling_indices_sha256(indices),
    }
    return indices, audit


def _require_exact(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    if key not in payload:
        raise _incomplete(f"training log is missing {key}")
    if payload[key] != expected:
        raise _incomplete(
            f"training log {key} mismatch: {payload[key]!r} != {expected!r}"
        )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise _incomplete(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _incomplete(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise _incomplete(f"{field} must be finite")
    return number


def validate_training_log(
    payload: Mapping[str, Any],
    job: TrainingJob,
    *,
    repo: Path = REPO,
) -> dict[str, Any]:
    """Validate one E15 child log independently of checkpoint loading."""
    _require_exact(payload, "schema_version", TRAINING_LOG_SCHEMA)
    _require_exact(payload, "status", "T-PASS")
    _require_exact(
        payload, "construction_seed", int(job.construction_seed)
    )
    _require_exact(payload, "pipeline_seed", int(job.pipeline_seed))
    _require_exact(payload, "seed_axes_combined", False)
    _require_exact(payload, "pool_identifier", job.pool_identifier)
    _require_exact(payload, "pool_schema_version", POOL_SCHEMA_VERSION)
    _require_exact(payload, "sampling_policy", SAMPLING_POLICY)
    _require_exact(payload, "explicit_synthetic_pool_override", True)
    _require_exact(payload, "fallback_used", False)
    _require_exact(payload, "sampling_seed", job.pipeline_seed + 300)
    _require_exact(payload, "optimizer_steps", MAX_STEPS)
    _require_exact(payload, "validation_cadence", VALIDATION_CADENCE)
    _require_exact(payload, "validation_checkpoints", VALIDATION_CHECKPOINTS)
    _require_exact(payload, "real_train_windows", REAL_TRAIN_WINDOWS)
    _require_exact(payload, "synthetic_windows", SYNTHETIC_TOTAL)
    _require_exact(
        payload, "train_windows", REAL_TRAIN_WINDOWS + SYNTHETIC_TOTAL
    )
    _require_exact(
        payload,
        "checkpoint_selection",
        "first maximum validation multiclass macro-F1",
    )
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != VALIDATION_CHECKPOINTS:
        raise _incomplete("training log must contain exactly 12 checks")
    expected_steps = [
        VALIDATION_CADENCE * index
        for index in range(1, VALIDATION_CHECKPOINTS + 1)
    ]
    observed_steps: list[int] = []
    scores: list[float] = []
    for position, item in enumerate(history, start=1):
        if not isinstance(item, Mapping):
            raise _incomplete("training history rows must be objects")
        if item.get("validation_checkpoint") != position:
            raise _incomplete("validation checkpoint ordinal mismatch")
        step = item.get("optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise _incomplete("history optimizer_step must be an integer")
        observed_steps.append(step)
        scores.append(
            _finite_number(
                item.get("val_macro_f1"),
                f"history[{position}].val_macro_f1",
            )
        )
        _finite_number(
            item.get("train_loss"),
            f"history[{position}].train_loss",
        )
        if scores[-1] < 0 or scores[-1] > 1:
            raise _incomplete("validation macro-F1 must be within [0,1]")
        if float(item["train_loss"]) < 0:
            raise _incomplete("training loss must be nonnegative")
    if observed_steps != expected_steps:
        raise _incomplete(
            f"validation steps mismatch: {observed_steps}"
        )
    best_score = max(scores)
    first_max = scores.index(best_score) + 1
    if payload.get("selected_validation_checkpoint") != first_max:
        raise _incomplete("selected checkpoint is not the first maximum")
    if payload.get("selected_optimizer_step") != (
        first_max * VALIDATION_CADENCE
    ):
        raise _incomplete("selected optimizer step mismatch")
    declared_best = _finite_number(
        payload.get("best_val_macro_f1"), "best_val_macro_f1"
    )
    if declared_best != best_score:
        raise _incomplete("best validation score/history mismatch")
    elapsed = _finite_number(payload.get("elapsed_seconds"), "elapsed_seconds")
    if elapsed < 0:
        raise _incomplete("elapsed_seconds must be nonnegative")

    sampling = payload.get("sampling_audit")
    if not isinstance(sampling, Mapping):
        raise _incomplete("training log sampling_audit must be an object")
    for key, expected in (
        ("sampling_policy", SAMPLING_POLICY),
        ("sampling_seed", job.pipeline_seed + 300),
        ("requested_total", SYNTHETIC_TOTAL),
        ("drawn_total", SYNTHETIC_TOTAL),
        ("unique_total", SYNTHETIC_TOTAL),
        ("repeated_total", 0),
    ):
        if sampling.get(key) != expected:
            raise _incomplete(f"sampling audit {key} mismatch")
    index_hash = sampling.get("index_sha256")
    if not isinstance(index_hash, str) or len(index_hash) != 64:
        raise _incomplete("sampling index hash is missing or malformed")
    if (
        job.construction_seed == CONSTRUCTION_SEEDS[0]
        and index_hash != ANCHOR_INDEX_SHA256[job.pipeline_seed]
    ):
        raise _incomplete("anchor sampling index hash mismatch")
    for label, requested in PER_CLASS_REQUESTED.items():
        item = sampling.get("per_class", {}).get(str(label))
        if not isinstance(item, Mapping):
            raise _incomplete(f"sampling class {label} audit is missing")
        for field, expected in (
            ("requested", requested),
            ("drawn", requested),
            ("unique", requested),
            ("repeated", 0),
        ):
            if item.get(field) != expected:
                raise _incomplete(
                    f"sampling class {label} {field} mismatch"
                )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise _incomplete("training log artifacts must be an object")
    checkpoint = artifacts.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise _incomplete("checkpoint artifact record is missing")
    checkpoint_hash = checkpoint.get("sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise _incomplete("checkpoint artifact hash is malformed")
    expected_checkpoint_path = portable_repo_path(
        job.checkpoint_path, repo
    )
    if checkpoint.get("path") != expected_checkpoint_path:
        raise _incomplete("checkpoint artifact path mismatch")
    pool = payload.get("pool")
    if not isinstance(pool, Mapping):
        raise _incomplete("pool binding is missing")
    for field in (
        "pool_file_sha256",
        "ordered_content_sha256",
        "generation_log_sha256",
        "configuration_sha256",
    ):
        value = pool.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise _incomplete(f"pool binding {field} is malformed")
    if pool.get("path") != portable_repo_path(job.pool_path, repo):
        raise _incomplete("pool path binding mismatch")
    if pool.get("pool_identifier") != job.pool_identifier:
        raise _incomplete("pool identifier binding mismatch")
    if pool.get("pool_schema_version") != POOL_SCHEMA_VERSION:
        raise _incomplete("pool schema binding mismatch")
    if pool.get("generation_log_path") != portable_repo_path(
        _generation_log_path(job.construction_seed, repo), repo
    ):
        raise _incomplete("generation-log path binding mismatch")
    if pool.get("source_train_sha256") != TRAIN_SHA256:
        raise _incomplete("pool source-train hash binding mismatch")
    standardizer = payload.get("standardizer_sha256")
    if standardizer != EXPECTED_STANDARDIZER_SHA256:
        raise _incomplete("real-only standardizer hash mismatch")
    preflight = payload.get("training_preflight")
    if not isinstance(preflight, Mapping):
        raise _incomplete("training preflight binding is missing")
    preflight_hash = preflight.get("sha256")
    if not isinstance(preflight_hash, str) or len(preflight_hash) != 64:
        raise _incomplete("training preflight hash is malformed")
    if (
        not isinstance(preflight.get("source_snapshot"), Mapping)
        or preflight["source_snapshot"].get("snapshot_unchanged") is not True
    ):
        raise _incomplete(
            "training preflight live source snapshot is missing"
        )
    final_source_snapshot = payload.get(
        "source_snapshot_before_publication"
    )
    if (
        not isinstance(final_source_snapshot, Mapping)
        or final_source_snapshot.get("snapshot_unchanged") is not True
    ):
        raise _incomplete(
            "training log final source snapshot is missing"
        )
    environment = payload.get("environment")
    if not isinstance(environment, Mapping) or (
        environment.get("stable_projection_sha256")
        != EXPECTED_ENVIRONMENT_STABLE_SHA256
    ):
        raise _incomplete("training environment binding mismatch")
    _require_exact(
        payload, "scientific_outcome_inference_performed", False
    )
    _require_exact(payload, "validation_forward_passes_performed", True)

    return {
        "construction_seed": job.construction_seed,
        "pipeline_seed": job.pipeline_seed,
        "selected_validation_checkpoint": first_max,
        "selected_optimizer_step": first_max * VALIDATION_CADENCE,
        "best_val_macro_f1": best_score,
        "sampling_index_sha256": index_hash,
        "checkpoint_sha256": checkpoint_hash,
        "pool_file_sha256": pool["pool_file_sha256"],
        "generation_log_sha256": pool["generation_log_sha256"],
        "configuration_sha256": pool["configuration_sha256"],
        "preflight_sha256": preflight_hash,
        "history": history,
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _incomplete(f"could not read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise _incomplete(f"JSON root must be an object: {path}")
    return payload


def validate_training_bundle(
    job: TrainingJob,
    *,
    repo: Path = REPO,
) -> dict[str, Any]:
    """Validate a checkpoint/log pair with weights-only structural loading."""
    if (
        not job.checkpoint_path.is_file()
        or job.checkpoint_path.is_symlink()
        or not job.log_path.is_file()
        or job.log_path.is_symlink()
    ):
        raise _incomplete(
            "checkpoint/log bundle is missing or not regular files: "
            f"{job.checkpoint_path}, {job.log_path}"
        )
    payload = _load_json(job.log_path)
    log_audit = validate_training_log(payload, job, repo=repo)
    checkpoint_hash = sha256_file(job.checkpoint_path)
    if checkpoint_hash != log_audit["checkpoint_sha256"]:
        raise _incomplete("checkpoint hash does not match training log")
    try:
        checkpoint_audit = validate_checkpoint(job.checkpoint_path)
    except Exception as exc:
        raise _incomplete(
            f"checkpoint weights-only validation failed: {job.checkpoint_path}"
        ) from exc
    return {
        **log_audit,
        "pool": {
            "path": portable_repo_path(job.pool_path, repo),
            "pool_identifier": job.pool_identifier,
            "pool_schema_version": POOL_SCHEMA_VERSION,
            "pool_file_sha256": log_audit["pool_file_sha256"],
            "generation_log_path": portable_repo_path(
                _generation_log_path(job.construction_seed, repo), repo
            ),
            "generation_log_sha256":
                log_audit["generation_log_sha256"],
            "configuration_sha256":
                log_audit["configuration_sha256"],
        },
        "checkpoint": {
            "path": portable_repo_path(job.checkpoint_path, repo),
            "bytes": job.checkpoint_path.stat().st_size,
            "sha256": checkpoint_hash,
            "structure": checkpoint_audit,
        },
        "training_log": {
            "path": portable_repo_path(job.log_path, repo),
            "bytes": job.log_path.stat().st_size,
            "sha256": sha256_file(job.log_path),
        },
        "explicit_synthetic_pool_override": True,
        "fallback_used": False,
        "scientific_outcome_inference_performed": False,
        "validation_forward_passes_performed": True,
    }


def _validate_generation_log(
    path: Path,
    *,
    job: TrainingJob,
    pool_audit: Mapping[str, Any],
    repo: Path = REPO,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise _training_stop(f"generation log is missing: {path}")
    payload = _load_json(path)
    for key, expected in (
        ("schema_version", "e15.rule_construction_generation_log.v1"),
        ("status", "T-PASS"),
        ("construction_seed", job.construction_seed),
        ("pipeline_seed", None),
        ("seed_axes_combined", False),
        ("pool_identifier", job.pool_identifier),
        ("pool_schema_version", POOL_SCHEMA_VERSION),
        ("sampling_policy", SAMPLING_POLICY),
        ("explicit_train_source_override", True),
        ("legacy_or_strict_v2_fallback_used", False),
        ("source_train_sha256", TRAIN_SHA256),
    ):
        if payload.get(key) != expected:
            raise _training_stop(
                f"generation log {key} mismatch for construction "
                f"{job.construction_seed}"
            )
    pool = payload.get("pool")
    if not isinstance(pool, Mapping):
        raise _training_stop("generation log pool binding is missing")
    if pool.get("sha256") != pool_audit["pool_file_sha256"]:
        raise _training_stop("generation log/pool file hash mismatch")
    target = pool_target_map(repo)[job.construction_seed]
    if pool.get("path") != repo_relative(target.pool_path, repo):
        raise _training_stop("generation log/pool path mismatch")
    if pool.get("bytes") != target.pool_path.stat().st_size:
        raise _training_stop("generation log/pool byte count mismatch")
    if (
        pool.get("ordered_content_sha256")
        != pool_audit["ordered_content_sha256"]
    ):
        raise _training_stop("generation log/pool content hash mismatch")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise _training_stop("generation log configuration is missing")
    config_hash = canonical_json_sha256(configuration)
    if payload.get("configuration_sha256") != config_hash:
        raise _training_stop("generation log configuration hash mismatch")
    if config_hash != pool_audit["configuration_sha256"]:
        raise _training_stop("generation log/pool configuration mismatch")
    statistics = payload.get("statistics")
    if not isinstance(statistics, Mapping):
        raise _training_stop("generation log statistics binding is missing")
    if (
        not target.statistics_path.is_file()
        or target.statistics_path.is_symlink()
    ):
        raise _training_stop("pool statistics CSV is missing")
    statistics_hash = sha256_file(target.statistics_path)
    if (
        statistics.get("path")
        != repo_relative(target.statistics_path, repo)
        or statistics.get("bytes") != target.statistics_path.stat().st_size
        or statistics.get("sha256") != statistics_hash
    ):
        raise _training_stop("generation log/statistics binding mismatch")
    generation_preflight = generation_record_paths(repo)["preflight"]
    preflight_binding = payload.get("preflight")
    if (
        not generation_preflight.is_file()
        or not isinstance(preflight_binding, Mapping)
        or preflight_binding.get("path")
        != repo_relative(generation_preflight, repo)
        or preflight_binding.get("sha256")
        != sha256_file(generation_preflight)
    ):
        raise _training_stop("generation log/preflight binding mismatch")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "configuration_sha256": config_hash,
        "statistics_sha256": statistics_hash,
        "payload": payload,
    }


def _validate_pool_generation_stage(
    repo: Path,
    pools: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require and cross-bind the complete upstream generation stage."""
    paths = generation_record_paths(repo)
    for name in ("preflight", "run", "pool_audit"):
        path = paths[name]
        if not path.is_file() or path.is_symlink():
            raise _training_stop(
                f"upstream pool-generation {name} is missing: {path}"
            )
    try:
        preflight = _load_json(paths["preflight"])
        run = _load_json(paths["run"])
    except E15TrainingError as exc:
        raise _training_stop(
            "upstream pool-generation JSON record is invalid"
        ) from exc
    for key, expected in (
        ("schema_version", PREFLIGHT_SCHEMA),
        ("status", "validated_preflight"),
        ("technical_status", "T-PASS"),
        ("mode", "execute"),
        ("pre_result", True),
        ("model_inference_performed", False),
    ):
        if preflight.get(key) != expected:
            raise _training_stop(
                f"pool-generation preflight {key} mismatch"
            )
    construction_axis = preflight.get("construction_axis")
    if not isinstance(construction_axis, Mapping) or (
        construction_axis.get("ordered_values")
        != list(CONSTRUCTION_SEEDS)
    ):
        raise _training_stop(
            "pool-generation preflight construction axis mismatch"
        )
    preflight_targets = preflight.get("targets")
    if not isinstance(preflight_targets, Mapping) or set(
        preflight_targets
    ) != {str(seed) for seed in CONSTRUCTION_SEEDS}:
        raise _training_stop(
            "pool-generation preflight must declare all five pool bundles"
        )
    target_absence = preflight.get("target_absence")
    if (
        not isinstance(target_absence, Mapping)
        or target_absence.get("all_absent") is not True
        or target_absence.get("expected_count") != 18
        or target_absence.get("verified_absent_count") != 18
        or target_absence.get("verified_absent_count")
        != target_absence.get("expected_count")
        or not isinstance(target_absence.get("paths"), list)
        or len(target_absence["paths"]) != 18
        or len(set(target_absence["paths"])) != 18
        or set(target_absence["paths"])
        != generation_output_status_paths(repo)
    ):
        raise _training_stop(
            "pool-generation preflight target-absence proof is incomplete"
        )
    preflight_hash = sha256_file(paths["preflight"])

    for key, expected in (
        ("schema_version", RUN_SCHEMA),
        ("status", "T-PASS"),
        ("model_inference_performed", False),
        ("construction_axis", list(CONSTRUCTION_SEEDS)),
        ("pipeline_axis_used", False),
        ("seed_axes_combined", False),
        ("one_shot_complete", True),
    ):
        if run.get(key) != expected:
            raise _training_stop(f"pool-generation run {key} mismatch")
    run_preflight = run.get("preflight")
    if (
        not isinstance(run_preflight, Mapping)
        or run_preflight.get("sha256") != preflight_hash
        or run_preflight.get("path")
        != repo_relative(paths["preflight"], repo)
    ):
        raise _training_stop(
            "pool-generation run/preflight hash binding mismatch"
        )
    run_pools = run.get("pools")
    if not isinstance(run_pools, list) or len(run_pools) != 5:
        raise _training_stop(
            "pool-generation run must contain exactly five pools"
        )
    run_by_seed: dict[int, Mapping[str, Any]] = {}
    for item in run_pools:
        if not isinstance(item, Mapping):
            raise _training_stop(
                "pool-generation run pool rows must be objects"
            )
        seed = item.get("construction_seed")
        if isinstance(seed, bool) or seed not in CONSTRUCTION_SEEDS:
            raise _training_stop(
                "pool-generation run contains an invalid construction seed"
            )
        if int(seed) in run_by_seed:
            raise _training_stop(
                "pool-generation run contains duplicate construction seed"
            )
        run_by_seed[int(seed)] = item
    if tuple(run_by_seed) != CONSTRUCTION_SEEDS:
        raise _training_stop(
            "pool-generation run pool order/coverage mismatch"
        )
    for seed in CONSTRUCTION_SEEDS:
        current = pools[seed]
        item = run_by_seed[seed]
        target = pool_target_map(repo)[seed]
        expected = {
            "pool_identifier": target.pool_identifier,
            "pool_path": repo_relative(target.pool_path, repo),
            "pool_file_sha256": current["pool_file_sha256"],
            "ordered_content_sha256": current[
                "ordered_content_sha256"
            ],
            "statistics_path": repo_relative(
                target.statistics_path, repo
            ),
            "statistics_sha256": current["generation_log"][
                "statistics_sha256"
            ],
            "generation_log_path": repo_relative(
                target.generation_log_path, repo
            ),
            "generation_log_sha256": current["generation_log"]["sha256"],
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise _training_stop(
                    f"pool-generation run {seed}/{key} binding mismatch"
                )

    overlap = run.get("cross_pool_row_overlap")
    if (
        not isinstance(overlap, Mapping)
        or overlap.get("schema_version")
        != "e15.cross_pool_row_overlap.v1"
        or overlap.get("construction_order") != list(CONSTRUCTION_SEEDS)
        or overlap.get("pool_count") != 5
        or overlap.get("pair_count") != 10
        or overlap.get("descriptive_only") is not True
    ):
        raise _training_stop(
            "pool-generation run cross-pool overlap audit is incomplete"
        )
    overlap_pairs = overlap.get("pairs")
    if not isinstance(overlap_pairs, list) or len(overlap_pairs) != 10:
        raise _training_stop(
            "pool-generation run must contain ten overlap pairs"
        )
    expected_pairs = {
        (left, right)
        for position, left in enumerate(CONSTRUCTION_SEEDS)
        for right in CONSTRUCTION_SEEDS[position + 1:]
    }
    observed_pairs: set[tuple[int, int]] = set()
    for pair in overlap_pairs:
        if not isinstance(pair, Mapping):
            raise _training_stop("cross-pool overlap row is not an object")
        identity = (
            pair.get("left_construction_seed"),
            pair.get("right_construction_seed"),
        )
        if identity not in expected_pairs or identity in observed_pairs:
            raise _training_stop(
                "cross-pool overlap pair identity mismatch"
            )
        observed_pairs.add(identity)
        if pair.get("identical_complete_digest_sets") is not False:
            raise _training_stop(
                "cross-pool audit reports identical complete pools"
            )
        overlap_count = pair.get("exact_row_content_overlap")
        if (
            isinstance(overlap_count, bool)
            or not isinstance(overlap_count, int)
            or overlap_count < 0
        ):
            raise _training_stop(
                "cross-pool overlap count is invalid"
            )
    if observed_pairs != expected_pairs:
        raise _training_stop("cross-pool overlap pair coverage mismatch")

    try:
        with paths["pool_audit"].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            audit_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise _training_stop(
            "could not read upstream pool audit CSV"
        ) from exc
    pool_rows = [
        row for row in audit_rows if row.get("record_scope") == "pool"
    ]
    pair_rows = [
        row
        for row in audit_rows
        if row.get("record_scope") == "cross_pool_pair"
    ]
    if len(audit_rows) != 15 or len(pool_rows) != 5 or len(pair_rows) != 10:
        raise _training_stop(
            "pool audit CSV must contain five pool and ten pair rows"
        )
    if {int(row["construction_seed"]) for row in pool_rows} != set(
        CONSTRUCTION_SEEDS
    ):
        raise _training_stop("pool audit CSV construction coverage mismatch")
    csv_pairs = {
        (
            int(row["left_construction_seed"]),
            int(row["right_construction_seed"]),
        )
        for row in pair_rows
    }
    if csv_pairs != expected_pairs:
        raise _training_stop("pool audit CSV pair coverage mismatch")
    run_pool_audit = run.get("pool_audit")
    pool_audit_hash = sha256_file(paths["pool_audit"])
    if (
        not isinstance(run_pool_audit, Mapping)
        or run_pool_audit.get("sha256") != pool_audit_hash
        or run_pool_audit.get("path")
        != repo_relative(paths["pool_audit"], repo)
    ):
        raise _training_stop("pool-generation run/audit CSV binding mismatch")
    return {
        "preflight": {
            "path": repo_relative(paths["preflight"], repo),
            "bytes": paths["preflight"].stat().st_size,
            "sha256": preflight_hash,
            "schema_version": PREFLIGHT_SCHEMA,
        },
        "run": {
            "path": repo_relative(paths["run"], repo),
            "bytes": paths["run"].stat().st_size,
            "sha256": sha256_file(paths["run"]),
            "schema_version": RUN_SCHEMA,
            "status": "T-PASS",
        },
        "pool_audit": {
            "path": repo_relative(paths["pool_audit"], repo),
            "bytes": paths["pool_audit"].stat().st_size,
            "sha256": pool_audit_hash,
            "schema_version": POOL_AUDIT_SCHEMA,
            "pool_rows": 5,
            "pair_rows": 10,
        },
        "pool_count": 5,
        "cross_pool_pair_count": 10,
        "complete": True,
    }


def _verify_runtime_inputs(repo: Path) -> dict[str, Any]:
    expected = {
        "journal/datasets/windows/train_windows.npz": TRAIN_SHA256,
        "journal/datasets/windows/val_windows.npz": VAL_SHA256,
        "journal/datasets/windows/test_windows.npz": TEST_SHA256,
    }
    records: dict[str, Any] = {}
    failures: list[str] = []
    for relative, expected_hash in expected.items():
        path = repo / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        observed = sha256_file(path)
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": observed,
            "expected_sha256": expected_hash,
            "matches": observed == expected_hash,
        }
        if observed != expected_hash:
            failures.append(f"{relative}: {observed}")
    if failures:
        raise _training_stop(
            "frozen real input verification failed: " + "; ".join(failures)
        )
    return records


def _git_output(
    repo: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _training_stop(
            f"Git source gate failed: git {shlex.join(arguments)}"
        ) from exc


def _allowed_generation_status_paths(repo: Path) -> set[str]:
    """Return the exact upstream outputs allowed untracked before training."""
    pool_targets = pool_target_map(repo)
    generation_records = generation_record_paths(repo)
    paths = [
        path
        for target in pool_targets.values()
        for path in (
            target.pool_path,
            target.statistics_path,
            target.generation_log_path,
        )
    ] + list(generation_records.values())
    return {repo_relative(path, repo) for path in paths}


def _training_source_record(repo: Path) -> dict[str, Any]:
    """Require HEAD-clean source while allowing only fresh generation outputs.

    PREREG Section 2 commits generation, training, and preparation records
    together after training.  Therefore a training preflight necessarily sees
    the just-created generation records as untracked when they are not ignored.
    Only those exact paths, and only with ``??`` status, are allowed.
    """
    head = _git_output(repo, ("rev-parse", "HEAD")).stdout.strip()
    ancestor = _git_output(
        repo,
        ("merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"),
        check=False,
    )
    if ancestor.returncode != 0:
        raise _training_stop(
            "frozen E15 preregistration commit is not an ancestor"
        )
    raw = _git_output(
        repo,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    ).stdout
    status_records = [item for item in raw.split("\0") if item]
    allowed_paths = _allowed_generation_status_paths(repo)
    try:
        status_audit = validate_source_status(
            status_records,
            allowed_untracked=allowed_paths,
            stage="T-STOP-TRAINING",
        )
    except RuntimeError as exc:
        raise _training_stop(
            "training requires HEAD-clean tracked/index/source state; only "
            "declared untracked generation outputs are allowed: "
            + str(exc)
        ) from exc
    wisa_records = [
        record
        for record in status_records
        if len(record) >= 4 and record[3:].startswith("wisa/")
    ]
    if wisa_records:
        raise _training_stop(
            "frozen wisa/ tree is not clean: "
            + json.dumps(wisa_records, ensure_ascii=True)
        )

    implementation: dict[str, Any] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise _training_stop(
                f"required implementation is missing: {relative}"
            )
        tracked = _git_output(
            repo,
            ("ls-files", "--error-unmatch", "--", relative),
            check=False,
        )
        if tracked.returncode != 0:
            raise _training_stop(
                f"required implementation is not tracked: {relative}"
            )
        blob_result = _git_output(
            repo, ("rev-parse", f"HEAD:{relative}"), check=False
        )
        if blob_result.returncode != 0:
            raise _training_stop(
                f"required implementation is absent from HEAD: {relative}"
            )
        blob = blob_result.stdout.strip()
        try:
            committed = subprocess.run(
                ["git", "cat-file", "blob", blob],
                cwd=repo,
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise _training_stop(
                f"could not read committed implementation: {relative}"
            ) from exc
        worktree_sha = sha256_file(path)
        committed_sha = hashlib.sha256(committed).hexdigest()
        if worktree_sha != committed_sha:
            raise _training_stop(
                f"implementation differs from HEAD: {relative}"
            )
        implementation[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": worktree_sha,
            "git_blob_oid": blob,
            "matches_head": True,
        }
    return {
        "head_commit": head,
        "preregistration_commit": PREREG_COMMIT,
        "preregistration_commit_is_ancestor": True,
        "tracked_index_source_head_clean": True,
        "wisa_clean": True,
        "status_records": status_records,
        "source_status": status_audit,
        "allowed_untracked_generation_output_paths":
            sorted(allowed_paths),
        "observed_allowed_untracked_generation_records":
            status_audit["observed_allowed_untracked_records"],
        "rejected_status_records": [],
        "implementation": implementation,
    }


def assert_training_snapshot_unchanged(
    preflight: Mapping[str, Any],
    *,
    repo: Path = REPO,
    allowed_untracked: Sequence[str] | set[str] = (),
    stage: str = "T-INCOMPLETE",
) -> dict[str, Any]:
    """Revalidate a published training preflight against the live workspace."""
    repo = Path(repo).resolve()
    source = preflight.get("source_provenance")
    frozen_inputs = preflight.get("frozen_inputs")
    pools = preflight.get("pools")
    pool_stage = preflight.get("pool_generation_stage")
    environment = preflight.get("environment")
    if not all(
        isinstance(value, Mapping)
        for value in (
            source,
            frozen_inputs,
            pools,
            pool_stage,
            environment,
        )
    ):
        raise RuntimeError(f"{stage}: training preflight snapshot is incomplete")
    observed_head = _git_output(repo, ("rev-parse", "HEAD")).stdout.strip()
    expected_head = source.get("head_commit")
    if observed_head != expected_head:
        raise RuntimeError(
            f"{stage}: source HEAD drifted: "
            f"{expected_head!r} -> {observed_head!r}"
        )

    expected_hashes: dict[str, str] = {}
    implementation = source.get("implementation")
    if not isinstance(implementation, Mapping):
        raise RuntimeError(
            f"{stage}: implementation snapshot is missing"
        )
    for relative, record in implementation.items():
        if not isinstance(record, Mapping) or not isinstance(
            record.get("sha256"), str
        ):
            raise RuntimeError(
                f"{stage}: invalid implementation snapshot for {relative}"
            )
        expected_hashes[str(relative)] = record["sha256"]
    for relative, record in frozen_inputs.items():
        if not isinstance(record, Mapping) or not isinstance(
            record.get("sha256"), str
        ):
            raise RuntimeError(
                f"{stage}: invalid frozen-input snapshot for {relative}"
            )
        expected_hashes[str(relative)] = record["sha256"]
    for name in ("preflight", "run", "pool_audit"):
        record = pool_stage.get(name)
        if not isinstance(record, Mapping) or not isinstance(
            record.get("path"), str
        ) or not isinstance(record.get("sha256"), str):
            raise RuntimeError(
                f"{stage}: invalid upstream {name} snapshot"
            )
        expected_hashes[record["path"]] = record["sha256"]
    for seed in CONSTRUCTION_SEEDS:
        record = pools.get(str(seed))
        if not isinstance(record, Mapping):
            raise RuntimeError(
                f"{stage}: missing pool snapshot for construction {seed}"
            )
        pool_path_raw = record.get("path")
        generation_log = record.get("generation_log")
        if (
            not isinstance(pool_path_raw, str)
            or not isinstance(record.get("pool_file_sha256"), str)
            or not isinstance(generation_log, Mapping)
            or not isinstance(generation_log.get("path"), str)
            or not isinstance(generation_log.get("sha256"), str)
            or not isinstance(
                generation_log.get("statistics_sha256"), str
            )
        ):
            raise RuntimeError(
                f"{stage}: invalid pool/log snapshot for construction {seed}"
            )
        pool_path = Path(pool_path_raw)
        pool_relative = (
            repo_relative(pool_path, repo)
            if pool_path.is_absolute()
            else pool_path.as_posix()
        )
        log_path = Path(generation_log["path"])
        log_relative = (
            repo_relative(log_path, repo)
            if log_path.is_absolute()
            else log_path.as_posix()
        )
        expected_hashes[pool_relative] = record["pool_file_sha256"]
        expected_hashes[log_relative] = generation_log["sha256"]
        stats_relative = repo_relative(
            pool_target_map(repo)[seed].statistics_path, repo
        )
        expected_hashes[stats_relative] = generation_log[
            "statistics_sha256"
        ]
    observed_hashes = {
        relative: (
            sha256_file(repo / relative)
            if (repo / relative).is_file()
            else None
        )
        for relative in expected_hashes
    }
    hash_audit = validate_hash_snapshot(
        expected_hashes, observed_hashes, stage=stage
    )

    raw = _git_output(
        repo,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    ).stdout
    status_records = [item for item in raw.split("\0") if item]
    status_audit = validate_source_status(
        status_records,
        allowed_untracked=allowed_untracked,
        stage=stage,
    )
    live_environment = _environment_record(repo)
    expected_stable = environment.get("stable_projection_sha256")
    if live_environment["stable_projection_sha256"] != expected_stable:
        raise RuntimeError(
            f"{stage}: training environment stable projection drifted"
        )
    return {
        "stage": stage,
        "head_commit": observed_head,
        "hash_snapshot": hash_audit,
        "source_status": status_audit,
        "environment_stable_projection_sha256": expected_stable,
        "snapshot_unchanged": True,
    }


def _target_absence(paths: Sequence[Path], repo: Path) -> dict[str, Any]:
    audit = assert_training_targets_absent(paths)
    return {
        **audit,
        "paths": [repo_relative(Path(path), repo) for path in paths],
    }


def build_training_preflight(
    repo: Path = REPO,
    mode: str = "preflight",
    *,
    invocation_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the full read-only training gate, including all 25 draws."""
    if mode == "dry-run":
        mode = "preflight"
    if mode not in {"preflight", "execute"}:
        raise ValueError("mode must be 'preflight' or 'execute'")
    repo = Path(repo).resolve()
    grid = build_training_grid(repo)
    records = training_record_paths(repo)
    artifact_absence = _target_absence(training_target_paths(grid), repo)
    record_absence = _target_absence(tuple(records.values()), repo)
    source = _training_source_record(repo)
    environment = _environment_record(repo)
    runtime_inputs = _verify_runtime_inputs(repo)
    standardizer = verify_derived_standardizer(repo=repo)

    pools: dict[int, dict[str, Any]] = {}
    sampling_rows: list[dict[str, Any]] = []
    for construction_seed in CONSTRUCTION_SEEDS:
        construction_jobs = [
            job for job in grid if job.construction_seed == construction_seed
        ]
        representative = construction_jobs[0]
        pool_audit = validate_pool_archive(
            representative.pool_path, construction_seed
        )
        generation_log = _validate_generation_log(
            _generation_log_path(construction_seed, repo),
            job=representative,
            pool_audit=pool_audit,
            repo=repo,
        )
        try:
            with np.load(
                representative.pool_path, allow_pickle=False
            ) as archive:
                labels = np.asarray(archive["y_attack_type"])
                for job in construction_jobs:
                    _, sampling = sample_indices_without_replacement(
                        labels,
                        SYNTHETIC_TOTAL,
                        job.pipeline_seed + SAMPLING_SEED_OFFSET,
                    )
                    if (
                        construction_seed == CONSTRUCTION_SEEDS[0]
                        and sampling["index_sha256"]
                        != ANCHOR_INDEX_SHA256[job.pipeline_seed]
                    ):
                        raise _training_stop(
                            "anchor sampling index hash mismatch for pipeline "
                            f"{job.pipeline_seed}"
                        )
                    sampling_rows.append(
                        {
                            "construction_seed": construction_seed,
                            "pipeline_seed": job.pipeline_seed,
                            "sampling_seed": (
                                job.pipeline_seed + SAMPLING_SEED_OFFSET
                            ),
                            "pool_identifier": job.pool_identifier,
                            "pool_file_sha256": pool_audit[
                                "pool_file_sha256"
                            ],
                            "ordered_content_sha256": pool_audit[
                                "ordered_content_sha256"
                            ],
                            "generation_log_sha256": generation_log["sha256"],
                            "configuration_sha256": generation_log[
                                "configuration_sha256"
                            ],
                            "sampling_policy": SAMPLING_POLICY,
                            "requested": SYNTHETIC_TOTAL,
                            "drawn": sampling["drawn_total"],
                            "unique": sampling["unique_total"],
                            "repeated": sampling["repeated_total"],
                            "index_sha256": sampling["index_sha256"],
                            "audit": sampling,
                        }
                    )
        except E15TrainingError:
            raise
        except (OSError, KeyError, ValueError) as exc:
            raise _training_stop(
                f"could not derive frozen samples for {construction_seed}"
            ) from exc
        pools[construction_seed] = {
            **pool_audit,
            "generation_log": {
                key: generation_log[key]
                for key in (
                    "path",
                    "bytes",
                    "sha256",
                    "configuration_sha256",
                    "statistics_sha256",
                )
            },
        }
    if len(sampling_rows) != 25:
        raise AssertionError("training preflight must freeze 25 sample vectors")
    if len({item["pool_file_sha256"] for item in pools.values()}) != 5:
        raise _training_stop("five E15 pool file hashes must be distinct")
    if len({item["ordered_content_sha256"] for item in pools.values()}) != 5:
        raise _training_stop("five E15 pool content hashes must be distinct")
    pool_generation_stage = _validate_pool_generation_stage(repo, pools)

    sampling_by_cell = {
        (row["construction_seed"], row["pipeline_seed"]): row
        for row in sampling_rows
    }
    invocation = list(invocation_argv if invocation_argv is not None else sys.argv)
    return {
        "schema_version": TRAINING_PREFLIGHT_SCHEMA,
        "record_type": "e15_rule_construction_training_preflight",
        "status": "validated_preflight",
        "technical_status": "T-PASS",
        "mode": mode,
        "created_utc": utc_now(),
        "pre_result": True,
        "model_inference_performed": False,
        "source_provenance": source,
        "environment": environment,
        "frozen_inputs": runtime_inputs,
        "derived_standardizer": standardizer,
        "construction_axis": {
            "ordered_values": list(CONSTRUCTION_SEEDS),
            "anchor": CONSTRUCTION_SEEDS[0],
            "new_primary": list(CONSTRUCTION_SEEDS[1:]),
        },
        "pipeline_axis": {
            "ordered_values": list(PIPELINE_SEEDS),
            "sampling_seed_rule": "pipeline_seed + 300",
            "controls_fit_state": True,
        },
        "seed_axes_combined": False,
        "fixed_protocol": {
            "architecture": "CNN1D 23-key E14 architecture",
            "state_dict_shapes": {
                key: list(shape) for key, shape in EXPECTED_STATE_SHAPES.items()
            },
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "loss": "balanced multiclass cross-entropy",
            "real_train_windows": REAL_TRAIN_WINDOWS,
            "synthetic_ratio": SYNTHETIC_RATIO,
            "synthetic_windows": SYNTHETIC_TOTAL,
            "per_class_requested": {
                str(key): value for key, value in PER_CLASS_REQUESTED.items()
            },
            "max_optimizer_steps": MAX_STEPS,
            "validation_cadence": VALIDATION_CADENCE,
            "validation_checkpoints": VALIDATION_CHECKPOINTS,
            "checkpoint_selection":
                "first maximum validation multiclass macro-F1",
            "standardization": "complete real train only",
        },
        "pools": {str(key): value for key, value in pools.items()},
        "pool_generation_stage": pool_generation_stage,
        "sampling": sampling_rows,
        "grid": [
            {
                "construction_seed": job.construction_seed,
                "pipeline_seed": job.pipeline_seed,
                "pool_path": repo_relative(job.pool_path, repo),
                "pool_identifier": job.pool_identifier,
                "explicit_synthetic_pool_override": True,
                "fallback_used": False,
                "sampling": sampling_by_cell[
                    (job.construction_seed, job.pipeline_seed)
                ],
                "checkpoint_path": repo_relative(
                    job.checkpoint_path, repo
                ),
                "log_path": repo_relative(job.log_path, repo),
                "command": {
                    "argv": list(job.command),
                    "display": shlex.join(job.command),
                    "cwd": str(repo),
                },
            }
            for job in grid
        ],
        "target_absence": artifact_absence,
        "record_target_absence": record_absence,
        "invocation": {
            "argv": invocation,
            "display": shlex.join(invocation),
            "cwd": str(Path.cwd()),
        },
        "one_shot_policy": (
            "publication of this execute-mode preflight starts one-shot "
            f"{OUTPUT_VERSION}; "
            "failure or interruption is preserved and never resumed"
        ),
    }


def _stage_bytes(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.e15-stage-",
        suffix=".tmp",
    )
    path = Path(name)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _stage_json(target: Path, payload: Mapping[str, Any]) -> Path:
    return _stage_bytes(
        target,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _load_real_train_val(
    repo: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_path = repo / "journal" / "datasets" / "windows" / "train_windows.npz"
    val_path = repo / "journal" / "datasets" / "windows" / "val_windows.npz"
    try:
        with np.load(train_path, allow_pickle=True) as train:
            train_x = np.asarray(train["x"], dtype=np.float32)
            train_y = np.asarray(train["y_attack_type"], dtype=np.int64)
        with np.load(val_path, allow_pickle=True) as val:
            val_x = np.asarray(val["x"], dtype=np.float32)
            val_y = np.asarray(val["y_attack_type"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as exc:
        raise _training_stop("could not load frozen train/validation arrays") from exc
    if train_x.shape != (REAL_TRAIN_WINDOWS, 128, 11):
        raise _training_stop(f"real train shape mismatch: {train_x.shape}")
    if train_y.shape != (REAL_TRAIN_WINDOWS,):
        raise _training_stop(f"real train label shape mismatch: {train_y.shape}")
    if val_x.ndim != 3 or val_x.shape[1:] != (128, 11):
        raise _training_stop(f"validation shape mismatch: {val_x.shape}")
    if val_y.shape != (len(val_x),):
        raise _training_stop("validation label row mismatch")
    return train_x, train_y, val_x, val_y


def _child_pool_binding(
    job: TrainingJob,
    generation_log_path: Path,
    *,
    repo: Path = REPO,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    pool_audit = validate_pool_archive(job.pool_path, job.construction_seed)
    generation = _validate_generation_log(
        generation_log_path,
        job=job,
        pool_audit=pool_audit,
        repo=repo,
    )
    return (
        {
            "path": portable_repo_path(job.pool_path, repo),
            "pool_identifier": job.pool_identifier,
            "pool_schema_version": POOL_SCHEMA_VERSION,
            "pool_file_sha256": pool_audit["pool_file_sha256"],
            "ordered_content_sha256": pool_audit[
                "ordered_content_sha256"
            ],
            "generation_log_path": portable_repo_path(
                generation_log_path, repo
            ),
            "generation_log_sha256": generation["sha256"],
            "configuration_sha256": generation[
                "configuration_sha256"
            ],
            "source_train_sha256": pool_audit["source_train_sha256"],
            "source_commit": pool_audit["source_commit"],
        },
        generation["payload"],
    )


def _require_published_training_preflight(
    job: TrainingJob,
    *,
    repo: Path,
) -> dict[str, Any]:
    """Bind a child to the already-published execute preflight and its cell."""
    path = training_record_paths(repo)["preflight"]
    if not path.is_file() or path.is_symlink():
        raise _training_stop(
            "child fit requires the canonical published training preflight"
        )
    payload = _load_json(path)
    for key, expected in (
        ("schema_version", TRAINING_PREFLIGHT_SCHEMA),
        ("status", "validated_preflight"),
        ("technical_status", "T-PASS"),
        ("mode", "execute"),
        ("seed_axes_combined", False),
    ):
        if payload.get(key) != expected:
            raise _training_stop(f"published preflight {key} mismatch")
    source = payload.get("source_provenance")
    if not isinstance(source, Mapping):
        raise _training_stop("published preflight source record is missing")
    current_head = _git_output(repo, ("rev-parse", "HEAD")).stdout.strip()
    if source.get("head_commit") != current_head:
        raise _training_stop("source commit changed after training preflight")
    grid = payload.get("grid")
    if not isinstance(grid, list) or len(grid) != 25:
        raise _training_stop("published preflight grid is incomplete")
    matching = [
        cell
        for cell in grid
        if isinstance(cell, Mapping)
        and cell.get("construction_seed") == job.construction_seed
        and cell.get("pipeline_seed") == job.pipeline_seed
    ]
    if len(matching) != 1:
        raise _training_stop("published preflight lacks a unique child cell")
    cell = matching[0]
    command = cell.get("command")
    if not isinstance(command, Mapping) or command.get("argv") != list(
        job.command
    ):
        raise _training_stop("child command differs from published preflight")
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise _training_stop("published preflight environment is missing")
    if (
        environment.get("stable_projection_sha256")
        != EXPECTED_ENVIRONMENT_STABLE_SHA256
    ):
        raise _training_stop("published preflight environment hash mismatch")
    frozen_grid = build_training_grid(repo)
    try:
        grid_index = next(
            index
            for index, candidate in enumerate(frozen_grid)
            if (
                candidate.construction_seed,
                candidate.pipeline_seed,
            )
            == (job.construction_seed, job.pipeline_seed)
        )
    except StopIteration as exc:
        raise _training_stop(
            "child cell is absent from the canonical grid"
        ) from exc
    previous_jobs = frozen_grid[:grid_index]
    future_jobs = frozen_grid[grid_index:]
    for previous in previous_jobs:
        validate_training_bundle(previous, repo=repo)
    future_collisions = [
        target
        for candidate in future_jobs
        for target in (candidate.checkpoint_path, candidate.log_path)
        if path_lexists(target)
    ]
    if future_collisions:
        raise _training_stop(
            "training grid is not a strict fresh prefix; current/future "
            "target exists: "
            + ", ".join(str(path) for path in future_collisions)
        )
    try:
        snapshot = assert_training_snapshot_unchanged(
            payload,
            repo=repo,
            allowed_untracked=training_output_status_paths(
                repo, completed_jobs=previous_jobs
            ),
            stage="T-INCOMPLETE",
        )
    except RuntimeError as exc:
        raise _incomplete(str(exc)) from exc
    sampling = cell.get("sampling")
    if not isinstance(sampling, Mapping):
        raise _training_stop("preflight child sampling binding is missing")
    return {
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_commit": current_head,
        "source_snapshot": snapshot,
        "grid_index": grid_index,
        "previous_completed_fit_count": len(previous_jobs),
        "sampling_index_sha256": sampling.get("index_sha256"),
        "pool_file_sha256": sampling.get("pool_file_sha256"),
        "generation_log_sha256": sampling.get(
            "generation_log_sha256"
        ),
        "configuration_sha256": sampling.get("configuration_sha256"),
        "cell": {
            "construction_seed": job.construction_seed,
            "pipeline_seed": job.pipeline_seed,
            "command": command,
        },
    }


def run_child_fit(
    job: TrainingJob,
    *,
    generation_log_path: Path,
    repo: Path = REPO,
) -> dict[str, Any]:
    """Train and publish one fresh E15 fit; never skip an existing target."""
    repo = Path(repo).resolve()
    assert_training_targets_absent(
        (job.checkpoint_path, job.log_path)
    )
    if (
        job.pool_identifier != pool_identifier(job.construction_seed)
        or job.construction_seed not in CONSTRUCTION_SEEDS
        or job.pipeline_seed not in PIPELINE_SEEDS
    ):
        raise _training_stop("child job is outside the frozen 5 x 5 grid")
    preflight_binding = _require_published_training_preflight(
        job, repo=repo
    )
    _verify_runtime_inputs(repo)
    pool_binding, _ = _child_pool_binding(
        job, generation_log_path, repo=repo
    )
    for field in (
        "pool_file_sha256",
        "generation_log_sha256",
        "configuration_sha256",
    ):
        if preflight_binding.get(field) != pool_binding[field]:
            raise _training_stop(
                f"child {field} differs from published preflight"
            )
    environment = _environment_record(repo)

    with np.load(job.pool_path, allow_pickle=False) as pool:
        synthetic_x = np.asarray(pool["x"], dtype=np.float32)
        synthetic_y = np.asarray(pool["y_attack_type"], dtype=np.int64)
        indices, sampling = sample_indices_without_replacement(
            synthetic_y,
            SYNTHETIC_TOTAL,
            job.pipeline_seed + SAMPLING_SEED_OFFSET,
        )
        selected_x = synthetic_x[indices]
        selected_y = synthetic_y[indices]
    if (
        job.construction_seed == CONSTRUCTION_SEEDS[0]
        and sampling["index_sha256"]
        != ANCHOR_INDEX_SHA256[job.pipeline_seed]
    ):
        raise _training_stop("child anchor sampling hash mismatch")
    if (
        sampling["index_sha256"]
        != preflight_binding["sampling_index_sha256"]
    ):
        raise _training_stop(
            "child sampling vector differs from published preflight"
        )

    started = utc_now()
    start_time = time.monotonic()
    lc.set_seed(job.pipeline_seed)
    real_x, real_y, val_x, val_y = _load_real_train_val(repo)
    mean, std = lc.fit_standardizer(real_x)
    mean_le = np.ascontiguousarray(mean, dtype="<f4")
    std_le = np.ascontiguousarray(std, dtype="<f4")
    standardizer = {
        "mean": hashlib.sha256(mean_le.tobytes(order="C")).hexdigest(),
        "standard_deviation": hashlib.sha256(
            std_le.tobytes(order="C")
        ).hexdigest(),
        "mean_then_standard_deviation": hashlib.sha256(
            mean_le.tobytes(order="C") + std_le.tobytes(order="C")
        ).hexdigest(),
    }
    expected_standardizer = verify_derived_standardizer(repo=repo)["sha256"]
    if standardizer != expected_standardizer:
        raise _training_stop("child real-only standardizer hash mismatch")

    train_x = np.concatenate((real_x, selected_x), axis=0)
    train_y = np.concatenate((real_y, selected_y), axis=0)
    del real_x, selected_x, synthetic_x
    train_x = lc.standardize(train_x, mean, std)
    val_x = lc.standardize(val_x, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        lc.WindowDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        lc.WindowDataset(val_x, val_y),
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    model = CNN1D(in_channels=11, classes=5).to(device)
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(5), y=train_y
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    selected_checkpoint = 0
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    loader_iterator = iter(train_loader)
    interval_loss_sum = 0.0
    interval_examples = 0
    while optimizer_steps < MAX_STEPS:
        try:
            batch_x, batch_y = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(train_loader)
            batch_x, batch_y = next(loader_iterator)
        model.train()
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        interval_loss_sum += float(loss.detach().cpu()) * len(batch_y)
        interval_examples += len(batch_y)
        if optimizer_steps % VALIDATION_CADENCE == 0:
            validation_checkpoint = len(history) + 1
            val_true, val_probs = predict(model, val_loader, device)
            score = float(
                f1_score(
                    val_true,
                    val_probs.argmax(axis=1),
                    average="macro",
                    zero_division=0,
                )
            )
            train_loss = interval_loss_sum / interval_examples
            history.append(
                {
                    "validation_checkpoint": validation_checkpoint,
                    "optimizer_step": optimizer_steps,
                    "train_loss": float(train_loss),
                    "val_macro_f1": score,
                }
            )
            print(
                "E15 "
                f"construction={job.construction_seed} "
                f"pipeline={job.pipeline_seed} "
                f"step={optimizer_steps} "
                f"val_macro_f1={score:.8f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                selected_checkpoint = validation_checkpoint
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            interval_loss_sum = 0.0
            interval_examples = 0
    if (
        optimizer_steps != MAX_STEPS
        or len(history) != VALIDATION_CHECKPOINTS
        or best_state is None
    ):
        raise _incomplete("child training did not complete the frozen budget")
    model.load_state_dict(best_state)

    job.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        dir=job.checkpoint_path.parent,
        prefix=f".{job.checkpoint_path.name}.e15-stage-",
        suffix=".tmp",
    )
    staged_checkpoint = Path(staged_name)
    staged_log: Path | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(model.state_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint_hash = sha256_file(staged_checkpoint)
        previous_jobs = build_training_grid(repo)[
            :preflight_binding["grid_index"]
        ]
        base_allowed_status = training_output_status_paths(
            repo, completed_jobs=previous_jobs
        )
        try:
            snapshot_before_publication = assert_training_snapshot_unchanged(
                _load_json(training_record_paths(repo)["preflight"]),
                repo=repo,
                allowed_untracked={
                    *base_allowed_status,
                    repo_relative(staged_checkpoint, repo),
                },
                stage="T-INCOMPLETE",
            )
        except RuntimeError as exc:
            raise _incomplete(str(exc)) from exc
        log = {
            "schema_version": TRAINING_LOG_SCHEMA,
            "record_type": "e15_rule_construction_training_log",
            "status": "T-PASS",
            "started_utc": started,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "construction_seed": job.construction_seed,
            "pipeline_seed": job.pipeline_seed,
            "sampling_seed": job.pipeline_seed + SAMPLING_SEED_OFFSET,
            "seed_axes_combined": False,
            "source_commit": preflight_binding["source_commit"],
            "training_preflight": preflight_binding,
            "source_snapshot_before_publication":
                snapshot_before_publication,
            "environment": environment,
            "pool_identifier": job.pool_identifier,
            "pool_schema_version": POOL_SCHEMA_VERSION,
            "sampling_policy": SAMPLING_POLICY,
            "explicit_synthetic_pool_override": True,
            "fallback_used": False,
            "pool": pool_binding,
            "sampling_audit": sampling,
            "standardizer_sha256": standardizer,
            "architecture": {
                "name": "CNN1D",
                "expected_state_dict_shapes": {
                    key: list(shape)
                    for key, shape in EXPECTED_STATE_SHAPES.items()
                },
            },
            "optimizer": {
                "name": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            "loss": "balanced multiclass cross-entropy",
            "batch_size": BATCH_SIZE,
            "real_train_windows": REAL_TRAIN_WINDOWS,
            "synthetic_windows": SYNTHETIC_TOTAL,
            "train_windows": REAL_TRAIN_WINDOWS + SYNTHETIC_TOTAL,
            "optimizer_steps": optimizer_steps,
            "validation_cadence": VALIDATION_CADENCE,
            "validation_checkpoints": len(history),
            "checkpoint_selection":
                "first maximum validation multiclass macro-F1",
            "selected_validation_checkpoint": selected_checkpoint,
            "selected_optimizer_step":
                selected_checkpoint * VALIDATION_CADENCE,
            "best_val_macro_f1": float(best_score),
            "history": history,
            "device": str(device),
            "artifacts": {
                "checkpoint": {
                    "path": repo_relative(job.checkpoint_path, repo),
                    "bytes": staged_checkpoint.stat().st_size,
                    "sha256": checkpoint_hash,
                },
                "training_log": {
                    "path": repo_relative(job.log_path, repo),
                },
            },
            "scientific_outcome_inference_performed": False,
            "validation_forward_passes_performed": True,
        }
        staged_log = _stage_json(job.log_path, log)
        try:
            assert_training_snapshot_unchanged(
                _load_json(training_record_paths(repo)["preflight"]),
                repo=repo,
                allowed_untracked={
                    *base_allowed_status,
                    repo_relative(staged_checkpoint, repo),
                    repo_relative(staged_log, repo),
                },
                stage="T-INCOMPLETE",
            )
        except RuntimeError as exc:
            raise _incomplete(str(exc)) from exc
        atomic_publish_bundle(
            (
                (staged_checkpoint, job.checkpoint_path),
                (staged_log, job.log_path),
            )
        )
        staged_checkpoint.unlink()
        staged_log.unlink()
        return log
    except BaseException:
        # Leave private stages for diagnosis; do not retry this versioned cell.
        raise


def execute_training(
    preflight: Mapping[str, Any],
    *,
    repo: Path = REPO,
) -> dict[str, Any]:
    """Publish the preflight, launch 25 children, and publish final records."""
    if preflight.get("mode") != "execute":
        raise _training_stop("execute requires an execute-mode preflight")
    repo = Path(repo).resolve()
    grid = build_training_grid(repo)
    records = training_record_paths(repo)
    _target_absence(
        training_target_paths(grid) + tuple(records.values()), repo
    )
    staged_preflight = _stage_json(records["preflight"], preflight)
    atomic_publish_bundle(((staged_preflight, records["preflight"]),))
    staged_preflight.unlink()
    preflight_hash = sha256_file(records["preflight"])
    try:
        parent_snapshot_after_preflight = (
            assert_training_snapshot_unchanged(
                preflight,
                repo=repo,
                allowed_untracked=training_output_status_paths(repo),
                stage="T-INCOMPLETE",
            )
        )
    except RuntimeError as exc:
        raise _incomplete(str(exc)) from exc
    started = utc_now()
    bundles: list[dict[str, Any]] = []
    for job in grid:
        try:
            subprocess.run(
                list(job.command),
                cwd=repo,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise _incomplete(
                "fresh child fit failed; output version must not be resumed: "
                f"construction={job.construction_seed}, "
                f"pipeline={job.pipeline_seed}"
            ) from exc
        bundles.append(validate_training_bundle(job, repo=repo))
    if len(bundles) != 25:
        raise _incomplete("training grid did not produce 25 valid bundles")
    try:
        parent_snapshot_after_children = (
            assert_training_snapshot_unchanged(
                preflight,
                repo=repo,
                allowed_untracked=training_output_status_paths(
                    repo, completed_jobs=grid
                ),
                stage="T-INCOMPLETE",
            )
        )
    except RuntimeError as exc:
        raise _incomplete(str(exc)) from exc

    sampling_rows = [
        {
            "schema_version": SAMPLING_AUDIT_SCHEMA,
            "construction_seed": item["construction_seed"],
            "pipeline_seed": item["pipeline_seed"],
            "sampling_seed": item["pipeline_seed"] + SAMPLING_SEED_OFFSET,
            "sampling_policy": SAMPLING_POLICY,
            "requested": SYNTHETIC_TOTAL,
            "drawn": SYNTHETIC_TOTAL,
            "unique": SYNTHETIC_TOTAL,
            "repeated": 0,
            "pool_file_sha256": item["pool_file_sha256"],
            "generation_log_sha256": item["generation_log_sha256"],
            "configuration_sha256": item["configuration_sha256"],
            "index_sha256": item["sampling_index_sha256"],
        }
        for item in bundles
    ]
    manifest_rows = [
        {
            "schema_version": TRAINING_MANIFEST_SCHEMA,
            "construction_seed": item["construction_seed"],
            "pipeline_seed": item["pipeline_seed"],
            "pool_identifier": pool_identifier(item["construction_seed"]),
            "pool_file_sha256": item["pool_file_sha256"],
            "sampling_index_sha256": item["sampling_index_sha256"],
            "checkpoint_path": item["checkpoint"]["path"],
            "checkpoint_sha256": item["checkpoint"]["sha256"],
            "training_log_path": item["training_log"]["path"],
            "training_log_sha256": item["training_log"]["sha256"],
            "optimizer_steps": MAX_STEPS,
            "validation_checkpoints": VALIDATION_CHECKPOINTS,
            "selected_validation_checkpoint": item[
                "selected_validation_checkpoint"
            ],
            "selected_optimizer_step": item["selected_optimizer_step"],
            "best_val_macro_f1": item["best_val_macro_f1"],
        }
        for item in bundles
    ]
    staged_sampling = _stage_bytes(
        records["sampling_audit"], _csv_bytes(sampling_rows)
    )
    staged_manifest = _stage_bytes(
        records["training_manifest"], _csv_bytes(manifest_rows)
    )
    run = {
        "schema_version": TRAINING_RUN_SCHEMA,
        "record_type": "e15_rule_construction_training_run",
        "status": "T-PASS",
        "started_utc": started,
        "finished_utc": utc_now(),
        "source_commit": preflight["source_provenance"]["head_commit"],
        "preflight": {
            "path": repo_relative(records["preflight"], repo),
            "sha256": preflight_hash,
        },
        "source_snapshot_after_preflight":
            parent_snapshot_after_preflight,
        "source_snapshot_after_children":
            parent_snapshot_after_children,
        "environment_before": preflight["environment"],
        "environment_after": _environment_record(repo),
        "construction_axis": list(CONSTRUCTION_SEEDS),
        "pipeline_axis": list(PIPELINE_SEEDS),
        "seed_axes_combined": False,
        "fixed_execution_order":
            "construction order then pipeline order; separate processes",
        "fresh_fit_count": len(bundles),
        "bundles": bundles,
        "sampling_audit": {
            "path": repo_relative(records["sampling_audit"], repo),
            "bytes": staged_sampling.stat().st_size,
            "sha256": sha256_file(staged_sampling),
        },
        "training_manifest": {
            "path": repo_relative(records["training_manifest"], repo),
            "bytes": staged_manifest.stat().st_size,
            "sha256": sha256_file(staged_manifest),
        },
        "scientific_outcome_inference_performed": False,
        "validation_forward_passes_performed": True,
        "one_shot_complete": True,
    }
    staged_run = _stage_json(records["run"], run)
    try:
        assert_training_snapshot_unchanged(
            preflight,
            repo=repo,
            allowed_untracked={
                *training_output_status_paths(
                    repo, completed_jobs=grid
                ),
                repo_relative(staged_sampling, repo),
                repo_relative(staged_manifest, repo),
                repo_relative(staged_run, repo),
            },
            stage="T-INCOMPLETE",
        )
    except RuntimeError as exc:
        raise _incomplete(str(exc)) from exc
    atomic_publish_bundle(
        (
            (staged_sampling, records["sampling_audit"]),
            (staged_manifest, records["training_manifest"]),
            (staged_run, records["run"]),
        )
    )
    for staged in (staged_sampling, staged_manifest, staged_run):
        staged.unlink()
    return run


def _require_child_args(args: argparse.Namespace) -> TrainingJob:
    required = {
        "construction_seed": args.construction_seed,
        "pipeline_seed": args.pipeline_seed,
        "synthetic_pool": args.synthetic_pool,
        "pool_identifier": args.pool_identifier,
        "pool_schema": args.pool_schema,
        "generation_log": args.generation_log,
        "sampling_policy": args.sampling_policy,
        "checkpoint_output": args.checkpoint_output,
        "log_output": args.log_output,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise _training_stop(
            "child fit requires explicit arguments: " + ", ".join(missing)
        )
    if args.pool_schema != POOL_SCHEMA_VERSION:
        raise _training_stop("child pool schema override mismatch")
    if args.sampling_policy != SAMPLING_POLICY:
        raise _training_stop("child sampling policy override mismatch")
    construction_seed = int(args.construction_seed)
    pipeline_seed = int(args.pipeline_seed)
    pool_path = Path(args.synthetic_pool).resolve()
    checkpoint = Path(args.checkpoint_output).resolve()
    log = Path(args.log_output).resolve()
    expected_grid = {
        (job.construction_seed, job.pipeline_seed): job
        for job in build_training_grid(REPO)
    }
    key = (construction_seed, pipeline_seed)
    if key not in expected_grid:
        raise _training_stop(f"child seed cell is not registered: {key}")
    expected = expected_grid[key]
    for label, observed, frozen in (
        ("pool path", pool_path, expected.pool_path.resolve()),
        ("pool identifier", args.pool_identifier, expected.pool_identifier),
        ("checkpoint path", checkpoint, expected.checkpoint_path),
        ("log path", log, expected.log_path),
        (
            "generation log path",
            Path(args.generation_log).resolve(),
            _generation_log_path(construction_seed, REPO).resolve(),
        ),
    ):
        if observed != frozen:
            raise _training_stop(
                f"child {label} differs from frozen grid: {observed} != {frozen}"
            )
    return expected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    top_mode = parser.add_mutually_exclusive_group()
    top_mode.add_argument(
        "--execute",
        action="store_true",
        help="publish preflight and launch the one-shot 25-fit grid",
    )
    top_mode.add_argument(
        "--child-fit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--construction-seed", type=int)
    parser.add_argument("--pipeline-seed", type=int)
    parser.add_argument("--synthetic-pool", type=Path)
    parser.add_argument("--pool-identifier")
    parser.add_argument("--pool-schema")
    parser.add_argument("--generation-log", type=Path)
    parser.add_argument("--sampling-policy")
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--log-output", type=Path)
    parser.add_argument(
        "--allow-overwrite", action="store_true", help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.allow_overwrite:
        raise SystemExit("T-STOP-TRAINING: --allow-overwrite is forbidden")
    if args.child_fit:
        job = _require_child_args(args)
        run_child_fit(
            job,
            generation_log_path=Path(args.generation_log).resolve(),
            repo=REPO,
        )
        return 0
    child_only = (
        "construction_seed",
        "pipeline_seed",
        "synthetic_pool",
        "pool_identifier",
        "pool_schema",
        "generation_log",
        "sampling_policy",
        "checkpoint_output",
        "log_output",
    )
    supplied = [name for name in child_only if getattr(args, name) is not None]
    if supplied:
        raise SystemExit(
            "T-STOP-TRAINING: child-only arguments require --child-fit: "
            + ", ".join(supplied)
        )
    mode = "execute" if args.execute else "preflight"
    invocation = [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
    preflight = build_training_preflight(
        REPO, mode=mode, invocation_argv=invocation
    )
    if not args.execute:
        print(
            json.dumps(
                preflight,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return 0
    run = execute_training(preflight, repo=REPO)
    print(json.dumps(run, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
