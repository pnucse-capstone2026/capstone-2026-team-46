#!/usr/bin/env python3
"""Construct the five preregistered E15 Rule-pool realizations.

The public helpers in this module are intentionally usable without generating
data.  Importing the module and the default CLI mode are read-only.  Canonical
generation requires ``--execute`` and may start only after the preflight has
proved the frozen source, environment, and complete no-clobber target set.

This wrapper does not change the canonical E13 strict-v2 pool.  It calls the
audited Rule construction routine with an explicit construction seed and
publishes a separate E15 schema at separate versioned paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import lib_common as lc
from generate_rule_based_synthetic_v2 import (
    CLASS_NAMES,
    generate_pool,
    validate_train_source,
)
from generator_protocol import (
    summarize_condition_metadata,
    summarize_protocol_validity,
)
from run_e14_matched_real_training import (
    canonical_json_sha256,
    capture_environment,
    repo_relative,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[2]
E15_DIR = (
    REPO / "journal" / "experiments" / "e15_rule_construction_crossing"
)
TABLE_DIR = REPO / "journal" / "results" / "tables"
LOG_DIR = REPO / "journal" / "results" / "logs"
SYNTHETIC_DIR = REPO / "journal" / "datasets" / "synthetic"

CONSTRUCTION_SEEDS = (314159, 271828, 161803, 141421, 173205)
PIPELINE_SEEDS = (7, 42, 123, 2026, 3407)
OUTPUT_VERSION = "v2"
PER_ATTACK = 65_000
TOTAL_WINDOWS = 4 * PER_ATTACK
WINDOW_SIZE = 128
FEATURE_COUNT = 11
STRIDE = 32
POOL_SCHEMA_VERSION = "e15_rule_construction_pool_v1"
SAMPLING_POLICY = "e15_strict_without_replacement_0p30_v1"
GENERATION_CONFIGURATION_SCHEMA = "e15.rule_construction_configuration.v1"
PREFLIGHT_SCHEMA = "e15.pool_generation_preflight.v1"
RUN_SCHEMA = "e15.pool_generation_run.v1"
POOL_AUDIT_SCHEMA = "e15.rule_construction_pool_audit.v1"

TRAIN_SHA256 = (
    "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4"
)
LEGACY_POOL_SHA256 = (
    "4078ab00b8531f760b296b5fd293e3df914bef996d317c958ead4762137bafd5"
)
LEGACY_ORDERED_CONTENT_SHA256 = (
    "9c05045481fc235ea58d85709295996a99c6842c39053332507c6d7ebfbbbc43"
)
PREREG_SHA256 = (
    "5fedd45f2b81608e6358281e97843a0600f1d5676aacd4b1e909c64e08bc1905"
)
PREREG_COMMIT = "f30dd62f92a1d6928e485ef8f0a2850993a6557d"
EXPECTED_NUMPY_VERSION = "2.4.6"
EXPECTED_ENVIRONMENT_STABLE_BYTES = 5_059
EXPECTED_ENVIRONMENT_STABLE_SHA256 = (
    "0fa13120b151b1c8bcb5a0c68ea7a0846f3b4c19a8bcc7d895393e030d9c0cf9"
)

FROZEN_INPUTS: dict[str, str] = {
    "journal/datasets/windows/train_windows.npz": TRAIN_SHA256,
    "journal/datasets/synthetic/rule_based_windows.npz": LEGACY_POOL_SHA256,
    "wisa/scripts/generate_rule_based_synthetic.py":
        "df7fa574efa2b2bd94daa846970fbcd7beb7c6b6ecf13ecd2c6a5b2855cc5f69",
    "journal/scripts/generate_rule_based_synthetic_v2.py":
        "34ff0374af1856fd842da8226580d5ff0d47cbd16b36fb4273de3e06bcac2570",
    "journal/scripts/generator_protocol.py":
        "f23c52ad9f08673c365a0db158d8c6e2db8213b5d30c2584ac777e066a4f85c0",
    "journal/scripts/lib_common.py":
        "5848ff4b0063d4394cd1d6627faf49e072cb363d7796cc64fce3acc48b05639a",
    "journal/experiments/e15_rule_construction_crossing/PREREG.md":
        PREREG_SHA256,
    (
        "journal/experiments/e15_rule_construction_crossing/"
        "IMPLEMENTATION_AMENDMENT_2026-07-25_V2_PATH_IDENTITY.md"
    ):
        "e747c5c8877a5dfd4e4d6b5b23c54b0ec23f122cd40f145011fa33cb622afd12",
    (
        "journal/experiments/e15_rule_construction_crossing/"
        "training_failure_v1.json"
    ):
        "b5e4cde58d0e6ba207bf5ba91bf39a150c0485be66a27171810a83e3daadb828",
}

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
)

ARRAY_KEYS = (
    "x",
    "y_binary",
    "y_attack_type",
    "synthetic_type",
    "condition_label",
    "condition_name",
    "injection_count",
    "base_source_index",
    "construction_ordinal",
)
SCIENTIFIC_ANCHOR_KEYS = (
    "x",
    "y_binary",
    "y_attack_type",
    "injection_count",
    "synthetic_type",
    "feature_names",
    "window_size",
    "stride",
)


class E15PoolError(RuntimeError):
    """A preregistered E15 pool gate failed."""


def _stop(message: str) -> E15PoolError:
    return E15PoolError(f"T-STOP-POOL: {message}")


@dataclass(frozen=True)
class PoolTargets:
    """Canonical output paths for one construction realization."""

    construction_seed: int
    pool_identifier: str
    pool_path: Path
    statistics_path: Path
    generation_log_path: Path

    @property
    def stats_path(self) -> Path:
        """Compatibility alias used by audit/test callers."""
        return self.statistics_path

    @property
    def log_path(self) -> Path:
        """Compatibility alias used by audit/test callers."""
        return self.generation_log_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def pool_identifier(construction_seed: int) -> str:
    return (
        f"e15_rule_cseed{int(construction_seed)}_{OUTPUT_VERSION}"
    )


def pool_target_map(repo: Path = REPO) -> dict[int, PoolTargets]:
    repo = Path(repo)
    targets: dict[int, PoolTargets] = {}
    for seed in CONSTRUCTION_SEEDS:
        targets[seed] = PoolTargets(
            construction_seed=seed,
            pool_identifier=pool_identifier(seed),
            pool_path=(
                repo
                / "journal"
                / "datasets"
                / "synthetic"
                / f"rule_cseed{seed}_windows_{OUTPUT_VERSION}.npz"
            ),
            statistics_path=(
                repo
                / "journal"
                / "results"
                / "tables"
                / (
                    f"e15_rule_cseed{seed}_statistics_"
                    f"{OUTPUT_VERSION}.csv"
                )
            ),
            generation_log_path=(
                repo
                / "journal"
                / "results"
                / "logs"
                / (
                    f"e15_generate_rule_cseed{seed}_"
                    f"{OUTPUT_VERSION}.json"
                )
            ),
        )
    return targets


def generation_record_paths(repo: Path = REPO) -> dict[str, Path]:
    base = (
        Path(repo)
        / "journal"
        / "experiments"
        / "e15_rule_construction_crossing"
    )
    return {
        "preflight": (
            base / f"pool_generation_preflight_{OUTPUT_VERSION}.json"
        ),
        "run": base / f"pool_generation_run_{OUTPUT_VERSION}.json",
        "pool_audit": (
            Path(repo)
            / "journal"
            / "results"
            / "tables"
            / (
                "e15_rule_construction_pool_audit_"
                f"{OUTPUT_VERSION}.csv"
            )
        ),
    }


def generation_output_status_paths(repo: Path = REPO) -> set[str]:
    """Exact canonical generation paths allowed untracked after preflight."""
    repo = Path(repo).resolve()
    targets = pool_target_map(repo)
    paths = [
        path
        for target in targets.values()
        for path in (
            target.pool_path,
            target.statistics_path,
            target.generation_log_path,
        )
    ] + list(generation_record_paths(repo).values())
    return {repo_relative(path, repo) for path in paths}


def path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def validate_source_status(
    status_lines: Sequence[str],
    *,
    allowed_untracked: Iterable[str] = (),
    stage: str,
) -> dict[str, Any]:
    """Validate porcelain-v1 records against an exact untracked allowlist.

    Tracked/index changes are always rejected, including at an allowlisted
    path.  An allowlist grants only ``??`` status to that exact repository
    relative path.  This keeps the pool stage completely clean while allowing
    the training stage to see only its declared upstream generation outputs.
    """
    records = [str(record) for record in status_lines if str(record)]
    allowed = {Path(path).as_posix() for path in allowed_untracked}
    observed_allowed: list[str] = []
    rejected: list[str] = []
    for record in records:
        if len(record) < 4:
            rejected.append(record)
            continue
        status_code = record[:2]
        # Porcelain v1 puts a separating space at index two.
        relative = Path(record[3:]).as_posix()
        if status_code == "??" and relative in allowed:
            observed_allowed.append(record)
        else:
            rejected.append(record)
    if rejected:
        raise RuntimeError(
            f"{stage}: source status contains disallowed record(s): "
            + json.dumps(rejected, ensure_ascii=True)
        )
    return {
        "stage": stage,
        "status_records": records,
        "allowed_untracked_paths": sorted(allowed),
        "observed_allowed_untracked_records": observed_allowed,
        "rejected_status_records": [],
        "tracked_index_head_clean": True,
    }


def validate_hash_snapshot(
    expected_sha256: Mapping[str, str],
    observed_sha256: Mapping[str, str | None],
    *,
    stage: str,
) -> dict[str, Any]:
    """Pure exact hash-snapshot comparison used by long-running stages."""
    expected = {str(path): str(value) for path, value in expected_sha256.items()}
    observed = {
        str(path): (None if value is None else str(value))
        for path, value in observed_sha256.items()
    }
    missing_expectations = sorted(set(observed) - set(expected))
    missing_observations = sorted(set(expected) - set(observed))
    mismatches = {
        path: {
            "expected_sha256": expected[path],
            "observed_sha256": observed.get(path),
        }
        for path in expected
        if observed.get(path) != expected[path]
    }
    if missing_expectations or missing_observations or mismatches:
        raise RuntimeError(
            f"{stage}: source hash snapshot drift: "
            + json.dumps(
                {
                    "unexpected_observations": missing_expectations,
                    "missing_observations": missing_observations,
                    "mismatches": mismatches,
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        )
    return {
        "stage": stage,
        "file_count": len(expected),
        "all_present": True,
        "all_hashes_match": True,
        "sha256": {
            path: expected[path] for path in sorted(expected)
        },
    }


def _scalar(arrays: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(arrays[key])
    if value.shape != ():
        raise _stop(f"{key} must be a scalar, got {value.shape}")
    return value.item()


def _text_scalar(arrays: Mapping[str, Any], key: str) -> str:
    return str(_scalar(arrays, key))


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def ordered_content_digest(
    x: np.ndarray,
    y_attack: np.ndarray,
) -> str:
    """Return the frozen label-byte plus little-endian-row-digest stream hash."""
    x = np.asarray(x)
    y_attack = np.asarray(y_attack).reshape(-1)
    if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, FEATURE_COUNT):
        raise _stop(
            "ordered-content x shape must be "
            f"(N,{WINDOW_SIZE},{FEATURE_COUNT}), got {x.shape}"
        )
    if len(x) != len(y_attack):
        raise _stop("ordered-content arrays have different row counts")
    digest = hashlib.sha256()
    for row, raw_label in zip(x, y_attack, strict=True):
        label = int(raw_label)
        if label < 0 or label > 255:
            raise _stop(f"attack label is not one unsigned byte: {label}")
        row_bytes = np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")
        digest.update(bytes((label,)))
        digest.update(hashlib.sha256(row_bytes).digest())
    return digest.hexdigest()


def row_content_digests(x: np.ndarray) -> np.ndarray:
    """Return one exact 32-byte little-endian-float32 digest per stored row."""
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, FEATURE_COUNT):
        raise _stop(
            "row-digest x shape must be "
            f"(N,{WINDOW_SIZE},{FEATURE_COUNT}), got {x.shape}"
        )
    result = np.empty(len(x), dtype="|V32")
    for index, row in enumerate(x):
        row_bytes = np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")
        result[index] = np.void(hashlib.sha256(row_bytes).digest())
    return result


def cross_pool_row_overlap_audit(
    pools: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Report exact pairwise row-content overlap for all supplied pools.

    The audit hashes each row independently after explicit little-endian
    float32 normalization.  It is descriptive: a nonzero cross-pool overlap
    is allowed, while identical complete digest sets are rejected by the
    caller's distinct-complete-pool gate.
    """
    if len(pools) < 2:
        raise ValueError("cross-pool overlap requires at least two pools")
    ordered_seeds = tuple(int(seed) for seed in pools)
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("cross-pool construction seeds must be unique")
    digest_sets: dict[int, set[bytes]] = {}
    row_counts: dict[int, int] = {}
    unique_counts: dict[int, int] = {}
    for seed, x in pools.items():
        digests = row_content_digests(np.asarray(x))
        digest_set = {
            bytes(digest)
            for digest in digests
        }
        digest_sets[int(seed)] = digest_set
        row_counts[int(seed)] = int(len(digests))
        unique_counts[int(seed)] = int(len(digest_set))

    pairs: list[dict[str, Any]] = []
    for left_position, left_seed in enumerate(ordered_seeds):
        for right_seed in ordered_seeds[left_position + 1:]:
            overlap = len(
                digest_sets[left_seed].intersection(digest_sets[right_seed])
            )
            identical_complete_sets = (
                row_counts[left_seed] == row_counts[right_seed]
                and unique_counts[left_seed] == row_counts[left_seed]
                and unique_counts[right_seed] == row_counts[right_seed]
                and overlap == row_counts[left_seed]
            )
            pairs.append(
                {
                    "left_construction_seed": left_seed,
                    "right_construction_seed": right_seed,
                    "left_rows": row_counts[left_seed],
                    "right_rows": row_counts[right_seed],
                    "left_unique_row_digests": unique_counts[left_seed],
                    "right_unique_row_digests": unique_counts[right_seed],
                    "exact_row_content_overlap": int(overlap),
                    "left_overlap_fraction": (
                        float(overlap / row_counts[left_seed])
                        if row_counts[left_seed]
                        else 0.0
                    ),
                    "right_overlap_fraction": (
                        float(overlap / row_counts[right_seed])
                        if row_counts[right_seed]
                        else 0.0
                    ),
                    "identical_complete_digest_sets":
                        identical_complete_sets,
                }
            )
    expected_pairs = len(ordered_seeds) * (len(ordered_seeds) - 1) // 2
    if len(pairs) != expected_pairs:
        raise AssertionError("pairwise overlap audit is incomplete")
    return {
        "schema_version": "e15.cross_pool_row_overlap.v1",
        "serialization":
            "per-row-contiguous-little-endian-float32-C-sha256",
        "construction_order": list(ordered_seeds),
        "pool_count": len(ordered_seeds),
        "pair_count": len(pairs),
        "pairs": pairs,
        "descriptive_only": True,
    }


def content_uniqueness_audit(
    x: np.ndarray,
    y_attack: np.ndarray,
) -> dict[str, Any]:
    """Audit exact little-endian float32 content, globally and by class."""
    x = np.asarray(x)
    y_attack = np.asarray(y_attack).reshape(-1)
    if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, FEATURE_COUNT):
        raise _stop(f"invalid x shape for uniqueness audit: {x.shape}")
    if len(x) != len(y_attack):
        raise _stop("x/y row mismatch in uniqueness audit")

    global_seen: set[bytes] = set()
    per_class_seen: dict[int, set[bytes]] = {
        label: set() for label in CLASS_NAMES
    }
    per_class_repeated = {label: 0 for label in CLASS_NAMES}
    for row, raw_label in zip(x, y_attack, strict=True):
        label = int(raw_label)
        if label not in per_class_seen:
            raise _stop(f"unexpected attack label in content audit: {label}")
        row_digest = hashlib.sha256(
            np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")
        ).digest()
        if row_digest in per_class_seen[label]:
            per_class_repeated[label] += 1
        else:
            per_class_seen[label].add(row_digest)
        global_seen.add(row_digest)

    return {
        "serialization": "per-row-contiguous-little-endian-float32-C",
        "hash_algorithm": "sha256",
        "content_sha256": ordered_content_digest(x, y_attack),
        "total": {
            "windows": int(len(x)),
            "unique": int(len(global_seen)),
            "repeated": int(len(x) - len(global_seen)),
        },
        "per_class": {
            str(label): {
                "windows": int(np.count_nonzero(y_attack == label)),
                "unique": int(len(per_class_seen[label])),
                "repeated": int(per_class_repeated[label]),
            }
            for label in CLASS_NAMES
        },
    }


def _mapping_files(arrays: Mapping[str, Any]) -> set[str]:
    files = getattr(arrays, "files", None)
    return set(files if files is not None else arrays.keys())


def validate_pool_arrays(
    arrays: Mapping[str, Any],
    expected_seed: int,
    *,
    expected_per_attack: int = PER_ATTACK,
    expected_source_sha256: str | None = TRAIN_SHA256,
) -> dict[str, Any]:
    """Independently validate an in-memory E15 pool archive mapping."""
    expected_seed = int(expected_seed)
    expected_per_attack = int(expected_per_attack)
    required = set(ARRAY_KEYS) | {
        "feature_names",
        "window_size",
        "stride",
        "pool_schema_version",
        "pool_identifier",
        "downstream_sampling_policy",
        "construction_seed",
        "generator_configuration_json",
        "source_train_sha256",
        "source_commit",
        "ordered_content_sha256",
    }
    missing = sorted(required - _mapping_files(arrays))
    if missing:
        raise _stop(f"pool archive is missing keys: {missing}")

    total = expected_per_attack * len(CLASS_NAMES)
    x = np.asarray(arrays["x"])
    y_binary = np.asarray(arrays["y_binary"])
    y_attack = np.asarray(arrays["y_attack_type"])
    condition_label = np.asarray(arrays["condition_label"])
    injection_count = np.asarray(arrays["injection_count"])

    if x.shape != (total, WINDOW_SIZE, FEATURE_COUNT):
        raise _stop(f"pool x shape mismatch: {x.shape}")
    if x.dtype.str != "<f4":
        raise _stop(f"pool x dtype must be <f4, got {x.dtype.str}")
    for key, value in (
        ("y_binary", y_binary),
        ("y_attack_type", y_attack),
        ("condition_label", condition_label),
    ):
        if value.shape != (total,) or value.dtype.str != "|i1":
            raise _stop(
                f"{key} must have shape ({total},) and dtype |i1; "
                f"got {value.shape}/{value.dtype.str}"
            )
    if injection_count.shape != (total,) or injection_count.dtype.str != "<i2":
        raise _stop(
            "injection_count must have shape "
            f"({total},) and dtype <i2; got "
            f"{injection_count.shape}/{injection_count.dtype.str}"
        )
    for key in (
        "synthetic_type",
        "condition_name",
        "base_source_index",
        "construction_ordinal",
    ):
        if np.asarray(arrays[key]).shape != (total,):
            raise _stop(f"{key} row count mismatch")
    if not np.isfinite(x).all():
        raise _stop("pool contains non-finite features")
    if not np.all(y_binary == 1):
        raise _stop("E15 synthetic binary labels must all equal one")
    if not np.array_equal(condition_label, y_attack):
        raise _stop("condition_label and y_attack_type are not identical")

    if _text_scalar(arrays, "pool_schema_version") != POOL_SCHEMA_VERSION:
        raise _stop("pool schema mismatch")
    expected_identifier = pool_identifier(expected_seed)
    if _text_scalar(arrays, "pool_identifier") != expected_identifier:
        raise _stop("pool identifier/construction-seed mismatch")
    if _text_scalar(arrays, "downstream_sampling_policy") != SAMPLING_POLICY:
        raise _stop("pool downstream sampling policy mismatch")
    if int(_scalar(arrays, "construction_seed")) != expected_seed:
        raise _stop("pool construction seed mismatch")
    if int(_scalar(arrays, "window_size")) != WINDOW_SIZE:
        raise _stop("pool window_size mismatch")
    if int(_scalar(arrays, "stride")) != STRIDE:
        raise _stop("pool stride mismatch")
    feature_names = [
        str(value)
        for value in np.asarray(arrays["feature_names"]).reshape(-1)
    ]
    if feature_names != list(lc.FEATURE_NAMES):
        raise _stop(f"pool feature order mismatch: {feature_names!r}")
    source_sha256 = _text_scalar(arrays, "source_train_sha256")
    if (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256
    ):
        raise _stop(
            "pool train-source hash mismatch: "
            f"{source_sha256} != {expected_source_sha256}"
        )

    try:
        configuration = json.loads(
            _text_scalar(arrays, "generator_configuration_json")
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise _stop("invalid generator_configuration_json") from exc
    if not isinstance(configuration, dict):
        raise _stop("generator configuration must be an object")
    exact_configuration = {
        "schema_version": GENERATION_CONFIGURATION_SCHEMA,
        "family": "rule_based",
        "construction_seed": expected_seed,
        "per_attack": expected_per_attack,
        "total_windows": total,
        "class_order": ["DoS", "Fuzzy", "Gear", "RPM"],
        "source_normal_selection": "with_replacement",
        "final_pool_permutation": "one_after_all_classes",
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "rng": "numpy.random.default_rng/PCG64",
        "byte_order": "little",
        "x_dtype": "<f4",
        "label_dtype": "|i1",
        "injection_count_dtype": "<i2",
        "source_train_sha256": source_sha256,
    }
    if configuration != exact_configuration:
        raise _stop("pool generator configuration is not the frozen E15 law")

    counts = {
        str(label): int(np.count_nonzero(y_attack == label))
        for label in CLASS_NAMES
    }
    expected_counts = {
        str(label): expected_per_attack for label in CLASS_NAMES
    }
    if counts != expected_counts:
        raise _stop(f"pool class counts mismatch: {counts}")
    synthetic_values = {
        label: set(
            str(item)
            for item in np.asarray(arrays["synthetic_type"])[
                y_attack == label
            ]
        )
        for label in CLASS_NAMES
    }
    condition_values = {
        label: set(
            str(item)
            for item in np.asarray(arrays["condition_name"])[
                y_attack == label
            ]
        )
        for label in CLASS_NAMES
    }
    expected_names = {
        label: {name} for label, name in CLASS_NAMES.items()
    }
    if synthetic_values != expected_names or condition_values != expected_names:
        raise _stop("pool string condition metadata do not match exact labels")

    protocol = summarize_protocol_validity(x)
    condition = summarize_condition_metadata(
        y_binary,
        y_attack,
        np.asarray(arrays["synthetic_type"]),
        condition_label=condition_label,
        condition_name=np.asarray(arrays["condition_name"]),
    )
    if not protocol["protocol_valid"]:
        raise _stop("pool frame protocol validation failed")
    if not condition["condition_metadata_valid"]:
        raise _stop("pool condition metadata validation failed")
    uniqueness = content_uniqueness_audit(x, y_attack)
    if uniqueness["total"]["repeated"] != 0:
        raise _stop(
            "pool contains exact duplicate generated-window content: "
            f"{uniqueness['total']}"
        )
    declared_digest = _text_scalar(arrays, "ordered_content_sha256")
    if declared_digest != uniqueness["content_sha256"]:
        raise _stop("embedded ordered-content digest mismatch")

    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "pool_identifier": expected_identifier,
        "construction_seed": expected_seed,
        "per_attack": expected_per_attack,
        "total_windows": total,
        "shape": list(x.shape),
        "dtypes": {
            "x": x.dtype.str,
            "y_binary": y_binary.dtype.str,
            "y_attack_type": y_attack.dtype.str,
            "injection_count": injection_count.dtype.str,
        },
        "class_counts": counts,
        "source_train_sha256": source_sha256,
        "source_commit": _text_scalar(arrays, "source_commit"),
        "configuration": configuration,
        "configuration_sha256": canonical_json_sha256(configuration),
        "protocol": protocol,
        "condition_metadata": condition,
        "content_uniqueness": uniqueness,
        "ordered_content_sha256": uniqueness["content_sha256"],
    }


def validate_pool_archive(
    path: Path,
    expected_seed: int,
    *,
    expected_per_attack: int = PER_ATTACK,
    expected_source_sha256: str | None = TRAIN_SHA256,
) -> dict[str, Any]:
    """Load and independently validate one E15 NPZ without model inference."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise _stop(f"pool archive is missing or not a regular file: {path}")
    try:
        with np.load(path, allow_pickle=False) as arrays:
            audit = validate_pool_arrays(
                arrays,
                expected_seed,
                expected_per_attack=expected_per_attack,
                expected_source_sha256=expected_source_sha256,
            )
    except E15PoolError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise _stop(f"could not load pool archive {path}") from exc
    return {
        **audit,
        "path": str(path),
        "bytes": path.stat().st_size,
        "pool_file_sha256": sha256_file(path),
    }


def validate_anchor_arrays(
    candidate: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact scientific-array continuity with the historical anchor."""
    missing_candidate = sorted(
        set(SCIENTIFIC_ANCHOR_KEYS) - _mapping_files(candidate)
    )
    missing_historical = sorted(
        set(SCIENTIFIC_ANCHOR_KEYS) - _mapping_files(historical)
    )
    if missing_candidate or missing_historical:
        raise _stop(
            "anchor archive missing scientific arrays: "
            f"candidate={missing_candidate}, historical={missing_historical}"
        )

    comparisons: dict[str, bool] = {}
    for key in SCIENTIFIC_ANCHOR_KEYS:
        left = np.asarray(candidate[key])
        right = np.asarray(historical[key])
        if left.dtype.kind in {"U", "S", "O"} or right.dtype.kind in {
            "U",
            "S",
            "O",
        }:
            equal = np.array_equal(
                left.astype(str, copy=False),
                right.astype(str, copy=False),
            )
        else:
            equal = np.array_equal(left, right)
        comparisons[key] = bool(equal)
    mismatches = [key for key, equal in comparisons.items() if not equal]
    if mismatches:
        raise _stop(
            "historical-anchor scientific array mismatch: "
            + ", ".join(mismatches)
        )
    digest = ordered_content_digest(
        np.asarray(candidate["x"]),
        np.asarray(candidate["y_attack_type"]),
    )
    if digest != LEGACY_ORDERED_CONTENT_SHA256:
        raise _stop(
            "historical-anchor ordered-content digest mismatch: "
            f"{digest}"
        )
    return {
        "status": "anchor-array-equivalent",
        "compared_keys": list(SCIENTIFIC_ANCHOR_KEYS),
        "array_equal": comparisons,
        "ordered_content_sha256": digest,
        "expected_ordered_content_sha256": LEGACY_ORDERED_CONTENT_SHA256,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_inode(left: Path, right: Path) -> bool:
    try:
        left_stat = os.stat(left, follow_symlinks=False)
        right_stat = os.stat(right, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (
        left_stat.st_dev == right_stat.st_dev
        and left_stat.st_ino == right_stat.st_ino
    )


def atomic_publish_bundle(
    staged_targets: Iterable[tuple[Path, Path]],
) -> None:
    """Publish staged files via exclusive hard links and no-clobber rollback.

    Staged files are deliberately not deleted here.  A caller deletes them
    only after confirmed success; on failure they remain for diagnosis.
    """
    pairs = tuple((Path(staged), Path(target)) for staged, target in staged_targets)
    if not pairs:
        raise ValueError("atomic publication bundle must not be empty")
    targets = [target for _, target in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError("atomic publication targets must be unique")
    collisions = [target for target in targets if path_lexists(target)]
    if collisions:
        raise FileExistsError(
            "T-STOP-POOL: refusing to overwrite occupied target(s): "
            + ", ".join(str(path) for path in collisions)
        )
    for staged, target in pairs:
        if not staged.is_file() or staged.is_symlink():
            raise FileNotFoundError(f"invalid staged regular file: {staged}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if staged.stat().st_dev != target.parent.stat().st_dev:
            raise OSError(
                "staged file is not on the target filesystem: "
                f"{staged} -> {target}"
            )

    published: list[tuple[Path, Path]] = []
    try:
        for staged, target in pairs:
            try:
                os.link(staged, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"T-STOP-POOL: target appeared during publication: {target}"
                ) from exc
            published.append((staged, target))
        for directory in dict.fromkeys(target.parent for target in targets):
            _fsync_directory(directory)
    except BaseException:
        for staged, target in reversed(published):
            if _same_inode(staged, target):
                target.unlink()
        for directory in dict.fromkeys(
            target.parent for _, target in published
        ):
            _fsync_directory(directory)
        raise


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


def _stage_npz(target: Path, arrays: Mapping[str, Any]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.e15-stage-",
        suffix=".tmp",
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _environment_record(repo: Path) -> dict[str, Any]:
    environment = capture_environment(repo=repo)
    stable = {
        key: value
        for key, value in environment.items()
        if key not in {"process", "identity_sha256"}
    }
    encoded = _canonical_json_bytes(stable)
    stable_hash = hashlib.sha256(encoded).hexdigest()
    if len(encoded) != EXPECTED_ENVIRONMENT_STABLE_BYTES:
        raise _stop(
            "environment stable-projection byte count mismatch: "
            f"{len(encoded)}"
        )
    if stable_hash != EXPECTED_ENVIRONMENT_STABLE_SHA256:
        raise _stop(
            "environment stable-projection hash mismatch: "
            f"{stable_hash}"
        )
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise _stop(
            f"NumPy version must be {EXPECTED_NUMPY_VERSION}, got {np.__version__}"
        )
    if sys.byteorder != "little":
        raise _stop(f"host byte order must be little, got {sys.byteorder}")
    bit_generator = np.random.default_rng(0).bit_generator
    if bit_generator.__class__ is not np.random.PCG64:
        raise _stop(
            "default_rng must be backed exactly by numpy.random.PCG64"
        )
    return {
        "full": environment,
        "stable_projection": stable,
        "stable_projection_bytes": len(encoded),
        "stable_projection_sha256": stable_hash,
        "expected_stable_projection_sha256":
            EXPECTED_ENVIRONMENT_STABLE_SHA256,
        "numpy_version": np.__version__,
        "byteorder": sys.byteorder,
        "default_rng_bit_generator": (
            f"{type(bit_generator).__module__}.{type(bit_generator).__name__}"
        ),
        "matches_frozen_environment": True,
    }


def _run_git(repo: Path, arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise _stop(f"Git source gate failed: git {' '.join(arguments)}") from exc


def _source_record(repo: Path) -> dict[str, Any]:
    head = _run_git(repo, ("rev-parse", "HEAD")).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise _stop("frozen E15 preregistration commit is not an ancestor")
    status_raw = _run_git(
        repo,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    status = [item for item in status_raw.split("\0") if item]
    try:
        status_audit = validate_source_status(
            status,
            allowed_untracked=(),
            stage="T-STOP-POOL",
        )
    except RuntimeError as exc:
        raise _stop(
            "pool generation requires a completely clean source tree: "
            + str(exc)
        ) from exc
    implementation: dict[str, Any] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise _stop(f"required implementation is missing: {relative}")
        _run_git(repo, ("ls-files", "--error-unmatch", "--", relative))
        blob = _run_git(repo, ("rev-parse", f"HEAD:{relative}")).strip()
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        worktree_sha = sha256_file(path)
        committed_sha = hashlib.sha256(committed).hexdigest()
        if worktree_sha != committed_sha:
            raise _stop(f"implementation differs from HEAD: {relative}")
        implementation[relative] = {
            "path": relative,
            "sha256": worktree_sha,
            "bytes": path.stat().st_size,
            "git_blob_oid": blob,
            "matches_head": True,
        }
    return {
        "head_commit": head,
        "preregistration_commit": PREREG_COMMIT,
        "preregistration_commit_is_ancestor": True,
        "worktree_clean": True,
        "wisa_clean": True,
        "source_status": status_audit,
        "implementation": implementation,
    }


def assert_pool_generation_snapshot_unchanged(
    preflight: Mapping[str, Any],
    *,
    repo: Path = REPO,
    allowed_untracked: Iterable[str] = (),
    stage: str = "T-INCOMPLETE",
) -> dict[str, Any]:
    """Revalidate the published generation preflight against live source."""
    repo = Path(repo).resolve()
    source = preflight.get("source_provenance")
    frozen_inputs = preflight.get("frozen_inputs")
    environment = preflight.get("environment")
    if (
        not isinstance(source, Mapping)
        or not isinstance(frozen_inputs, Mapping)
        or not isinstance(environment, Mapping)
    ):
        raise RuntimeError(f"{stage}: preflight source snapshot is incomplete")
    expected_head = source.get("head_commit")
    observed_head = _run_git(repo, ("rev-parse", "HEAD")).strip()
    if observed_head != expected_head:
        raise RuntimeError(
            f"{stage}: source HEAD drifted: "
            f"{expected_head!r} -> {observed_head!r}"
        )

    expected_hashes: dict[str, str] = {}
    implementation = source.get("implementation")
    if not isinstance(implementation, Mapping):
        raise RuntimeError(
            f"{stage}: preflight implementation snapshot is missing"
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

    status_raw = _run_git(
        repo,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    status_records = [item for item in status_raw.split("\0") if item]
    status_audit = validate_source_status(
        status_records,
        allowed_untracked=allowed_untracked,
        stage=stage,
    )
    live_environment = _environment_record(repo)
    expected_stable = environment.get("stable_projection_sha256")
    if live_environment["stable_projection_sha256"] != expected_stable:
        raise RuntimeError(
            f"{stage}: environment stable projection drifted"
        )
    return {
        "stage": stage,
        "head_commit": observed_head,
        "hash_snapshot": hash_audit,
        "source_status": status_audit,
        "environment_stable_projection_sha256": expected_stable,
        "snapshot_unchanged": True,
    }


def _verify_frozen_inputs(repo: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    failures: list[str] = []
    for relative, expected in FROZEN_INPUTS.items():
        path = repo / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        observed = sha256_file(path)
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": observed,
            "expected_sha256": expected,
            "matches": observed == expected,
        }
        if observed != expected:
            failures.append(f"{relative}: {observed}")
    if failures:
        raise _stop("frozen input hash failure: " + "; ".join(failures))
    return records


def _target_absence(paths: Sequence[Path], repo: Path) -> dict[str, Any]:
    collisions = [path for path in paths if path_lexists(path)]
    if collisions:
        raise _stop(
            "canonical generation target collision(s): "
            + ", ".join(str(path) for path in collisions)
        )
    return {
        "expected_count": len(paths),
        "verified_absent_count": len(paths),
        "all_absent": True,
        "paths": [repo_relative(path, repo) for path in paths],
    }


def build_pool_generation_preflight(
    repo: Path = REPO,
    mode: str = "preflight",
    *,
    invocation_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the complete read-only E15 pool-generation preflight."""
    if mode == "dry-run":
        mode = "preflight"
    if mode not in {"preflight", "execute"}:
        raise ValueError("mode must be 'preflight' or 'execute'")
    repo = Path(repo).resolve()
    targets = pool_target_map(repo)
    records = generation_record_paths(repo)
    all_targets = [
        path
        for target in targets.values()
        for path in (
            target.pool_path,
            target.statistics_path,
            target.generation_log_path,
        )
    ] + list(records.values())
    absence = _target_absence(all_targets, repo)
    frozen = _verify_frozen_inputs(repo)
    source = _source_record(repo)
    environment = _environment_record(repo)
    train_path = repo / "journal" / "datasets" / "windows" / "train_windows.npz"
    with np.load(train_path, allow_pickle=True) as archive:
        train_x, normal_indices = validate_train_source(archive)
        train_shape = list(train_x.shape)
        normal_count = int(len(normal_indices))
    invocation = list(invocation_argv if invocation_argv is not None else sys.argv)
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "record_type": "e15_rule_pool_generation_preflight",
        "status": "validated_preflight",
        "technical_status": "T-PASS",
        "mode": mode,
        "created_utc": utc_now(),
        "pre_result": True,
        "model_inference_performed": False,
        "source_provenance": source,
        "frozen_inputs": frozen,
        "environment": environment,
        "invocation": {
            "argv": invocation,
            "cwd": str(Path.cwd()),
            "wrapper_path": repo_relative(Path(__file__), repo),
        },
        "construction_axis": {
            "name": "construction_seed",
            "ordered_values": list(CONSTRUCTION_SEEDS),
            "anchor": CONSTRUCTION_SEEDS[0],
            "new_primary": list(CONSTRUCTION_SEEDS[1:]),
            "controls_only_construction_rng": True,
        },
        "pipeline_axis": {
            "name": "pipeline_seed",
            "ordered_values": list(PIPELINE_SEEDS),
            "used_during_pool_generation": False,
            "combined_with_construction_seed": False,
        },
        "fixed_construction": {
            "per_attack": PER_ATTACK,
            "total_windows_per_pool": TOTAL_WINDOWS,
            "class_order": ["DoS", "Fuzzy", "Gear", "RPM"],
            "source_train_shape": train_shape,
            "source_normal_windows": normal_count,
            "source_normal_selection": "with_replacement",
            "final_pool_permutation": "one_after_all_classes",
            "pool_schema_version": POOL_SCHEMA_VERSION,
            "sampling_policy_for_downstream_consumer": SAMPLING_POLICY,
        },
        "targets": {
            str(seed): {
                "pool_identifier": target.pool_identifier,
                "pool_path": repo_relative(target.pool_path, repo),
                "statistics_path": repo_relative(target.statistics_path, repo),
                "generation_log_path": repo_relative(
                    target.generation_log_path, repo
                ),
            }
            for seed, target in targets.items()
        },
        "target_absence": absence,
        "one_shot_policy": (
            "publication of this execute-mode preflight starts a one-shot "
            "stage; failure is preserved and never resumed under "
            f"{OUTPUT_VERSION}"
        ),
    }


def _configuration(seed: int, source_commit: str) -> dict[str, Any]:
    del source_commit  # source commit is separate metadata, not construction law
    return {
        "schema_version": GENERATION_CONFIGURATION_SCHEMA,
        "family": "rule_based",
        "construction_seed": int(seed),
        "per_attack": PER_ATTACK,
        "total_windows": TOTAL_WINDOWS,
        "class_order": ["DoS", "Fuzzy", "Gear", "RPM"],
        "source_normal_selection": "with_replacement",
        "final_pool_permutation": "one_after_all_classes",
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "rng": "numpy.random.default_rng/PCG64",
        "byte_order": "little",
        "x_dtype": "<f4",
        "label_dtype": "|i1",
        "injection_count_dtype": "<i2",
        "source_train_sha256": TRAIN_SHA256,
    }


def _archive_arrays(
    generated_arrays: Mapping[str, Any],
    *,
    seed: int,
    source_commit: str,
    ordered_digest: str,
) -> dict[str, Any]:
    configuration = _configuration(seed, source_commit)
    return {
        **{
            key: np.asarray(generated_arrays[key])
            for key in ARRAY_KEYS
        },
        "x": np.ascontiguousarray(generated_arrays["x"], dtype="<f4"),
        "y_binary": np.asarray(generated_arrays["y_binary"], dtype="|i1"),
        "y_attack_type": np.asarray(
            generated_arrays["y_attack_type"], dtype="|i1"
        ),
        "condition_label": np.asarray(
            generated_arrays["condition_label"], dtype="|i1"
        ),
        "injection_count": np.asarray(
            generated_arrays["injection_count"], dtype="<i2"
        ),
        "feature_names": np.asarray(lc.FEATURE_NAMES, dtype="<U32"),
        "window_size": np.asarray(WINDOW_SIZE, dtype="<i8"),
        "stride": np.asarray(STRIDE, dtype="<i8"),
        "pool_schema_version": np.asarray(POOL_SCHEMA_VERSION),
        "pool_identifier": np.asarray(pool_identifier(seed)),
        "downstream_sampling_policy": np.asarray(SAMPLING_POLICY),
        "construction_seed": np.asarray(seed, dtype="<i8"),
        "generator_configuration_json": np.asarray(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        ),
        "source_train_sha256": np.asarray(TRAIN_SHA256),
        "source_commit": np.asarray(source_commit),
        "ordered_content_sha256": np.asarray(ordered_digest),
        "generation_log_path": np.asarray(
            "journal/results/logs/"
            f"e15_generate_rule_cseed{seed}_{OUTPUT_VERSION}.json"
        ),
    }


def _statistics_rows(
    audit: Mapping[str, Any],
    *,
    target: PoolTargets,
    pool_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configuration = audit["configuration"]
    for label, name in CLASS_NAMES.items():
        item = audit["content_uniqueness"]["per_class"][str(label)]
        rows.append(
            {
                "schema_version": "e15.rule_construction_statistics.v1",
                "pool_identifier": target.pool_identifier,
                "construction_seed": target.construction_seed,
                "synthetic_type": name,
                "attack_type_id": label,
                "windows": item["windows"],
                "generated_content_unique": item["unique"],
                "generated_content_repeated": item["repeated"],
                "protocol_valid": audit["protocol"]["protocol_valid"],
                "condition_metadata_valid": (
                    audit["condition_metadata"]["condition_metadata_valid"]
                ),
                "ordered_content_sha256": audit["ordered_content_sha256"],
                "pool_file_sha256": pool_sha256,
                "configuration_sha256": canonical_json_sha256(configuration),
                "source_train_sha256": TRAIN_SHA256,
            }
        )
    return rows


def execute_pool_generation(
    preflight: Mapping[str, Any],
    *,
    repo: Path = REPO,
) -> dict[str, Any]:
    """Execute and atomically publish the one-shot five-pool E15 stage."""
    if preflight.get("mode") != "execute":
        raise _stop("execute requires an execute-mode preflight")
    repo = Path(repo).resolve()
    targets = pool_target_map(repo)
    record_paths = generation_record_paths(repo)
    preflight_path = record_paths["preflight"]
    _target_absence(
        [
            path
            for target in targets.values()
            for path in (
                target.pool_path,
                target.statistics_path,
                target.generation_log_path,
            )
        ]
        + list(record_paths.values()),
        repo,
    )

    # Publishing this record starts the preregistered one-shot stage.
    staged_preflight = _stage_json(preflight_path, preflight)
    atomic_publish_bundle(((staged_preflight, preflight_path),))
    staged_preflight.unlink()
    preflight_hash = sha256_file(preflight_path)
    canonical_status_paths = generation_output_status_paths(repo)
    try:
        snapshot_after_preflight = assert_pool_generation_snapshot_unchanged(
            preflight,
            repo=repo,
            allowed_untracked=canonical_status_paths,
            stage="T-INCOMPLETE",
        )
    except RuntimeError as exc:
        raise E15PoolError(str(exc)) from exc

    started = utc_now()
    staged: list[Path] = []
    published_targets: list[Path] = []
    pool_records: list[dict[str, Any]] = []
    pool_rows_for_overlap: dict[int, np.ndarray] = {}
    try:
        train_path = (
            repo / "journal" / "datasets" / "windows" / "train_windows.npz"
        )
        legacy_path = (
            repo / "journal" / "datasets" / "synthetic"
            / "rule_based_windows.npz"
        )
        with np.load(train_path, allow_pickle=True) as train:
            train_x, normal_indices = validate_train_source(train)
            with np.load(legacy_path, allow_pickle=True) as historical:
                for seed in CONSTRUCTION_SEEDS:
                    target = targets[seed]
                    generated = generate_pool(
                        train_x,
                        normal_indices,
                        per_attack=PER_ATTACK,
                        construction_seed=seed,
                    )
                    raw_arrays = generated["arrays"]
                    digest = ordered_content_digest(
                        raw_arrays["x"], raw_arrays["y_attack_type"]
                    )
                    arrays = _archive_arrays(
                        raw_arrays,
                        seed=seed,
                        source_commit=preflight["source_provenance"][
                            "head_commit"
                        ],
                        ordered_digest=digest,
                    )
                    # Keep only the scientific rows needed for the bounded
                    # five-pool/ten-pair overlap audit (about 5.8 GiB if the
                    # raw arrays were retained, so store compact digests).
                    pool_rows_for_overlap[seed] = row_content_digests(
                        arrays["x"]
                    )
                    array_audit = validate_pool_arrays(arrays, seed)
                    if seed == CONSTRUCTION_SEEDS[0]:
                        anchor = validate_anchor_arrays(arrays, historical)
                    else:
                        anchor = None

                    staged_pool = _stage_npz(target.pool_path, arrays)
                    staged.append(staged_pool)
                    archive_audit = validate_pool_archive(
                        staged_pool, seed
                    )
                    archive_audit["path"] = repo_relative(
                        target.pool_path, repo
                    )
                    stats_rows = _statistics_rows(
                        archive_audit,
                        target=target,
                        pool_sha256=archive_audit["pool_file_sha256"],
                    )
                    staged_stats = _stage_bytes(
                        target.statistics_path, _csv_bytes(stats_rows)
                    )
                    staged.append(staged_stats)
                    log = {
                        "schema_version":
                            "e15.rule_construction_generation_log.v1",
                        "record_type": "e15_rule_pool_generation_log",
                        "status": "T-PASS",
                        "created_utc": utc_now(),
                        "construction_seed": seed,
                        "pipeline_seed": None,
                        "seed_axes_combined": False,
                        "pool_identifier": target.pool_identifier,
                        "pool_schema_version": POOL_SCHEMA_VERSION,
                        "sampling_policy": SAMPLING_POLICY,
                        "explicit_train_source_override": True,
                        "legacy_or_strict_v2_fallback_used": False,
                        "configuration": array_audit["configuration"],
                        "configuration_sha256": array_audit[
                            "configuration_sha256"
                        ],
                        "source_commit": preflight["source_provenance"][
                            "head_commit"
                        ],
                        "source_train_sha256": TRAIN_SHA256,
                        "pool": {
                            "path": repo_relative(target.pool_path, repo),
                            "bytes": staged_pool.stat().st_size,
                            "sha256": sha256_file(staged_pool),
                            "ordered_content_sha256": digest,
                        },
                        "statistics": {
                            "path": repo_relative(
                                target.statistics_path, repo
                            ),
                            "bytes": staged_stats.stat().st_size,
                            "sha256": sha256_file(staged_stats),
                        },
                        "audit": archive_audit,
                        "historical_anchor_equivalence": anchor,
                        "preflight": {
                            "path": repo_relative(preflight_path, repo),
                            "sha256": preflight_hash,
                        },
                        "model_inference_performed": False,
                    }
                    staged_log = _stage_json(
                        target.generation_log_path, log
                    )
                    staged.append(staged_log)
                    atomic_publish_bundle(
                        (
                            (staged_pool, target.pool_path),
                            (staged_stats, target.statistics_path),
                            (staged_log, target.generation_log_path),
                        )
                    )
                    for path in (staged_pool, staged_stats, staged_log):
                        path.unlink()
                        staged.remove(path)
                    published_targets.extend(
                        (
                            target.pool_path,
                            target.statistics_path,
                            target.generation_log_path,
                        )
                    )
                    pool_records.append(
                        {
                            "construction_seed": seed,
                            "pool_identifier": target.pool_identifier,
                            "pool_path": repo_relative(
                                target.pool_path, repo
                            ),
                            "pool_file_sha256": sha256_file(target.pool_path),
                            "pool_bytes": target.pool_path.stat().st_size,
                            "ordered_content_sha256": digest,
                            "statistics_path": repo_relative(
                                target.statistics_path, repo
                            ),
                            "statistics_sha256": sha256_file(
                                target.statistics_path
                            ),
                            "generation_log_path": repo_relative(
                                target.generation_log_path, repo
                            ),
                            "generation_log_sha256": sha256_file(
                                target.generation_log_path
                            ),
                            "anchor_equivalence": anchor,
                        }
                    )

        file_hashes = [item["pool_file_sha256"] for item in pool_records]
        content_hashes = [
            item["ordered_content_sha256"] for item in pool_records
        ]
        if len(set(file_hashes)) != len(CONSTRUCTION_SEEDS):
            raise _stop("all five pool file hashes are not distinct")
        if len(set(content_hashes)) != len(CONSTRUCTION_SEEDS):
            raise _stop("all five ordered-content hashes are not distinct")

        # ``cross_pool_row_overlap_audit`` accepts raw rows for unit-level use.
        # The canonical run already retained compact digest vectors, so apply
        # the same exact set-intersection law without reloading 1.3M windows.
        overlap_pairs: list[dict[str, Any]] = []
        overlap_sets = {
            seed: {bytes(digest) for digest in digests}
            for seed, digests in pool_rows_for_overlap.items()
        }
        for left_position, left_seed in enumerate(CONSTRUCTION_SEEDS):
            for right_seed in CONSTRUCTION_SEEDS[left_position + 1:]:
                overlap = len(
                    overlap_sets[left_seed].intersection(
                        overlap_sets[right_seed]
                    )
                )
                left_rows = len(pool_rows_for_overlap[left_seed])
                right_rows = len(pool_rows_for_overlap[right_seed])
                pair = {
                    "left_construction_seed": left_seed,
                    "right_construction_seed": right_seed,
                    "left_rows": left_rows,
                    "right_rows": right_rows,
                    "left_unique_row_digests": len(overlap_sets[left_seed]),
                    "right_unique_row_digests": len(overlap_sets[right_seed]),
                    "exact_row_content_overlap": overlap,
                    "left_overlap_fraction": overlap / left_rows,
                    "right_overlap_fraction": overlap / right_rows,
                    "identical_complete_digest_sets": (
                        overlap == left_rows
                        and overlap == right_rows
                        and len(overlap_sets[left_seed]) == left_rows
                        and len(overlap_sets[right_seed]) == right_rows
                    ),
                }
                if pair["identical_complete_digest_sets"]:
                    raise _stop(
                        "two complete E15 pools have identical row-digest sets: "
                        f"{left_seed}, {right_seed}"
                    )
                overlap_pairs.append(pair)
        if len(overlap_pairs) != 10:
            raise _stop("five-pool overlap audit must contain ten pairs")
        cross_pool_overlap = {
            "schema_version": "e15.cross_pool_row_overlap.v1",
            "serialization":
                "per-row-contiguous-little-endian-float32-C-sha256",
            "construction_order": list(CONSTRUCTION_SEEDS),
            "pool_count": 5,
            "pair_count": 10,
            "pairs": overlap_pairs,
            "descriptive_only": True,
        }

        audit_rows = [
            {
                "schema_version": POOL_AUDIT_SCHEMA,
                "record_scope": "pool",
                "construction_seed": item["construction_seed"],
                "role": (
                    "fresh_known_construction_anchor"
                    if item["construction_seed"] == CONSTRUCTION_SEEDS[0]
                    else "new_primary_construction"
                ),
                "pool_identifier": item["pool_identifier"],
                "pool_path": item["pool_path"],
                "pool_file_sha256": item["pool_file_sha256"],
                "ordered_content_sha256": item["ordered_content_sha256"],
                "generation_log_sha256": item["generation_log_sha256"],
                "per_attack": PER_ATTACK,
                "total_windows": TOTAL_WINDOWS,
                "exact_content_repeated": 0,
                "pool_hash_distinct": True,
                "content_hash_distinct": True,
                "left_construction_seed": "",
                "right_construction_seed": "",
                "left_rows": "",
                "right_rows": "",
                "exact_row_content_overlap": "",
                "left_overlap_fraction": "",
                "right_overlap_fraction": "",
                "identical_complete_digest_sets": "",
            }
            for item in pool_records
        ]
        audit_rows.extend(
            {
                "schema_version": POOL_AUDIT_SCHEMA,
                "record_scope": "cross_pool_pair",
                "construction_seed": "",
                "role": "descriptive_cross_pool_overlap",
                "pool_identifier": "",
                "pool_path": "",
                "pool_file_sha256": "",
                "ordered_content_sha256": "",
                "generation_log_sha256": "",
                "per_attack": "",
                "total_windows": "",
                "exact_content_repeated": "",
                "pool_hash_distinct": True,
                "content_hash_distinct": True,
                "left_construction_seed": pair["left_construction_seed"],
                "right_construction_seed": pair["right_construction_seed"],
                "left_rows": pair["left_rows"],
                "right_rows": pair["right_rows"],
                "exact_row_content_overlap":
                    pair["exact_row_content_overlap"],
                "left_overlap_fraction": pair["left_overlap_fraction"],
                "right_overlap_fraction": pair["right_overlap_fraction"],
                "identical_complete_digest_sets":
                    pair["identical_complete_digest_sets"],
            }
            for pair in overlap_pairs
        )
        staged_audit = _stage_bytes(
            record_paths["pool_audit"], _csv_bytes(audit_rows)
        )
        staged.append(staged_audit)
        try:
            snapshot_before_completion = (
                assert_pool_generation_snapshot_unchanged(
                    preflight,
                    repo=repo,
                    allowed_untracked={
                        *canonical_status_paths,
                        repo_relative(staged_audit, repo),
                    },
                    stage="T-INCOMPLETE",
                )
            )
        except RuntimeError as exc:
            raise E15PoolError(str(exc)) from exc
        finished = utc_now()
        run = {
            "schema_version": RUN_SCHEMA,
            "record_type": "e15_rule_pool_generation_run",
            "status": "T-PASS",
            "started_utc": started,
            "finished_utc": finished,
            "model_inference_performed": False,
            "source_commit": preflight["source_provenance"]["head_commit"],
            "environment": _environment_record(repo),
            "preflight": {
                "path": repo_relative(preflight_path, repo),
                "sha256": preflight_hash,
            },
            "source_snapshot_after_preflight": snapshot_after_preflight,
            "source_snapshot_before_completion":
                snapshot_before_completion,
            "construction_axis": list(CONSTRUCTION_SEEDS),
            "pipeline_axis_used": False,
            "seed_axes_combined": False,
            "pools": pool_records,
            "distinctness": {
                "five_distinct_pool_file_hashes": True,
                "five_distinct_ordered_content_hashes": True,
            },
            "cross_pool_row_overlap": cross_pool_overlap,
            "pool_audit": {
                "path": repo_relative(record_paths["pool_audit"], repo),
                "sha256": sha256_file(staged_audit),
                "bytes": staged_audit.stat().st_size,
            },
            "one_shot_complete": True,
        }
        staged_run = _stage_json(record_paths["run"], run)
        staged.append(staged_run)
        try:
            assert_pool_generation_snapshot_unchanged(
                preflight,
                repo=repo,
                allowed_untracked={
                    *canonical_status_paths,
                    repo_relative(staged_audit, repo),
                    repo_relative(staged_run, repo),
                },
                stage="T-INCOMPLETE",
            )
        except RuntimeError as exc:
            raise E15PoolError(str(exc)) from exc
        atomic_publish_bundle(
            (
                (staged_audit, record_paths["pool_audit"]),
                (staged_run, record_paths["run"]),
            )
        )
        for path in (staged_audit, staged_run):
            path.unlink()
            staged.remove(path)
        return run
    except BaseException:
        # Canonical artifacts and any private stages are intentionally kept.
        # Retrying this output version is forbidden after preflight publication.
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="publish the preflight and execute the one-shot canonical stage",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.allow_overwrite:
        raise SystemExit("T-STOP-POOL: --allow-overwrite is forbidden")
    mode = "execute" if args.execute else "preflight"
    invocation = [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
    preflight = build_pool_generation_preflight(
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
    run = execute_pool_generation(preflight, repo=REPO)
    print(json.dumps(run, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
