#!/usr/bin/env python3
"""Generate the versioned Rule pool used by strict-v2 augmentation runs.

This is a journal-native fork of the frozen WISA Rule construction.  The
attack injectors and default construction seed are unchanged, but the output
identity and safety gates are deliberately separate:

* the default pool has enough rows per class for the current +100% arm;
* required capacity is derived from the real training-window count;
* generated rows, labels, frame protocol, and exact-content uniqueness are
  validated before publication;
* the legacy ``rule_based_windows.npz`` path is protected even when explicit
  overwrite permission is supplied; and
* the NPZ, class statistics, and JSON log record portable provenance and
  cryptographic hashes.

The downstream strict-v2 sampler still draws *indices* without replacement.
Normal source windows are selected with replacement during construction, as
in the frozen Rule generator; exact generated-window uniqueness is audited
separately and is a hard pre-write gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import lib_common as lc
from generator_protocol import (summarize_condition_metadata,
                                summarize_protocol_validity)


CONSTRUCTION_SEED = 314159
DEFAULT_PER_ATTACK = 65_538
DEFAULT_TARGET_RATIO = 1.0
WINDOW_SIZE = 128
STRIDE = 32
POOL_IDENTIFIER = "rule_based_unique_pool_v2"
POOL_SCHEMA_VERSION = "rule_based_synthetic_pool_v2"
GENERATOR_PROTOCOL_VERSION = "dlc_payload_v1"
GENERATOR_CONFIGURATION = {
    "family": "rule_based",
    "construction_seed": CONSTRUCTION_SEED,
    "lineage": "legacy_rule_configuration_extended_without_replacement",
}
CLASS_NAMES = {1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}
ATTACK_IDS = {name: label for label, name in CLASS_NAMES.items()}


# These injectors are intentionally kept aligned with
# wisa/scripts/generate_rule_based_synthetic.py.  Do not import the frozen
# script: its path constants resolve inside the read-only WISA archive.
def choose_positions(rng, min_len, max_len, step_options):
    burst_len = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    step = int(rng.choice(step_options))
    return np.arange(start, start + burst_len, step)


def synth_dos(rng, x):
    pos = choose_positions(rng, 32, 90, [1, 2])
    x[pos, 0] = 0x000
    x[pos, 1] = 8
    x[pos, 2:10] = rng.choice(
        [0, 255], size=(len(pos), 8), p=[0.92, 0.08])
    x[pos, 10] = rng.uniform(0.00003, 0.002, size=len(pos))
    return len(pos)


def synth_fuzzy(rng, x):
    pos = choose_positions(rng, 24, 88, [1, 2, 3, 4])
    x[pos, 0] = rng.integers(0, 2048, size=len(pos))
    dlc = rng.choice([2, 5, 8], size=len(pos), p=[0.08, 0.12, 0.80])
    x[pos, 1] = dlc
    payload = rng.integers(0, 256, size=(len(pos), 8))
    for i, d in enumerate(dlc):
        if d < 8:
            payload[i, d:] = 0
    x[pos, 2:10] = payload
    x[pos, 10] = rng.uniform(0.00005, 0.02, size=len(pos))
    return len(pos)


def synth_spoof(rng, x, can_id, mode):
    pos = choose_positions(rng, 32, 90, [1, 2, 3])
    x[pos, 0] = can_id
    x[pos, 1] = 8
    base = x[pos, 2:10].copy()
    if mode == "gear":
        levels = rng.choice([0, 1, 2, 3, 4, 5], size=len(pos))
        base[:, 0] = np.clip(
            levels * 40 + rng.integers(0, 16, size=len(pos)), 0, 255)
        base[:, 1] = np.clip(
            255 - base[:, 0] + rng.integers(-8, 9, size=len(pos)), 0, 255)
    else:
        rpm = rng.integers(0, 8000, size=len(pos))
        base[:, 0] = (rpm // 32) % 256
        base[:, 1] = (rpm // 4) % 256
        base[:, 2] = np.clip(
            base[:, 2] + rng.integers(-50, 51, size=len(pos)), 0, 255)
    x[pos, 2:10] = base
    x[pos, 10] = np.maximum(x[pos, 10], 1e-5)
    return len(pos)


SPECS = (
    ("DoS", ATTACK_IDS["DoS"], synth_dos),
    ("Fuzzy", ATTACK_IDS["Fuzzy"], synth_fuzzy),
    (
        "Gear",
        ATTACK_IDS["Gear"],
        lambda rng, x: synth_spoof(rng, x, 0x43F, "gear"),
    ),
    (
        "RPM",
        ATTACK_IDS["RPM"],
        lambda rng, x: synth_spoof(rng, x, 0x316, "rpm"),
    ),
)


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError("construction_seed must be an integer")
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError(
            "construction_seed must be in the uint64 range")
    return value


def capacity_requirements(
        real_train_windows: int,
        target_ratio: float = DEFAULT_TARGET_RATIO,
) -> dict[str, Any]:
    """Return the exact class-prefix allocation used by strict-v2 sampling."""
    real_train_windows = _require_positive_int(
        real_train_windows, "real_train_windows")
    target_ratio = float(target_ratio)
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError("target_ratio must be finite and positive")

    total = round(real_train_windows * target_ratio)
    if total <= 0:
        raise ValueError(
            "target_ratio rounds to zero synthetic windows for this source")
    per_class, remainder = divmod(total, len(CLASS_NAMES))
    requests = {
        str(label): per_class + (position < remainder)
        for position, label in enumerate(CLASS_NAMES)
    }
    requests = {label: int(count) for label, count in requests.items()}
    return {
        "real_train_windows": real_train_windows,
        "target_augmentation_ratio": target_ratio,
        "synthetic_windows_requested": int(total),
        "remainder_assignment": "class_order_prefix",
        "per_class_requested": requests,
        "minimum_equal_pool_capacity_per_class": max(requests.values()),
    }


def require_pool_capacity(per_attack: int, capacity: dict[str, Any]) -> None:
    per_attack = _require_positive_int(per_attack, "per_attack")
    minimum = int(capacity["minimum_equal_pool_capacity_per_class"])
    if per_attack < minimum:
        details = ", ".join(
            f"class {label}: requested {requested}, available {per_attack}"
            for label, requested in capacity["per_class_requested"].items()
            if requested > per_attack
        )
        raise ValueError(
            "strict-v2 pool capacity is insufficient "
            f"({details}); use --per-attack >= {minimum}")


def _scalar(data, key):
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"source {key} must be a scalar, got {value.shape}")
    return value.item()


def validate_train_source(data) -> tuple[np.ndarray, np.ndarray]:
    """Validate the Car-Hacking train-window schema and return x/normal rows."""
    required = {
        "x",
        "y_binary",
        "y_attack_type",
        "feature_names",
        "window_size",
        "stride",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"source train archive is missing keys: {missing}")

    x = np.asarray(data["x"])
    y_binary = np.asarray(data["y_binary"]).reshape(-1)
    y_attack = np.asarray(data["y_attack_type"]).reshape(-1)
    if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, len(lc.FEATURE_NAMES)):
        raise ValueError(
            "source x must have shape "
            f"(N,{WINDOW_SIZE},{len(lc.FEATURE_NAMES)}), got {x.shape}")
    if len(x) == 0 or len(y_binary) != len(x) or len(y_attack) != len(x):
        raise ValueError("source x and label arrays must have one non-empty row set")
    if _scalar(data, "window_size") != WINDOW_SIZE:
        raise ValueError(f"source window_size must be {WINDOW_SIZE}")
    if _scalar(data, "stride") != STRIDE:
        raise ValueError(f"source stride must be {STRIDE}")

    feature_names = [str(name) for name in np.asarray(
        data["feature_names"]).reshape(-1)]
    if feature_names != list(lc.FEATURE_NAMES):
        raise ValueError(
            f"source feature_names mismatch: {feature_names!r}")
    if not np.isfinite(x).all():
        raise ValueError("source train windows contain non-finite values")

    normal_binary = y_binary == 0
    normal_attack = y_attack == 0
    if not np.array_equal(normal_binary, normal_attack):
        inconsistent = int(np.count_nonzero(normal_binary != normal_attack))
        raise ValueError(
            f"source binary/attack labels disagree on {inconsistent} rows")
    normal_indices = np.flatnonzero(normal_binary)
    if len(normal_indices) == 0:
        raise ValueError("source train archive has no normal windows")
    return x, normal_indices


def _content_uniqueness(
        x: np.ndarray,
        y_attack: np.ndarray,
) -> dict[str, Any]:
    """Hash exact float32 row bytes and report repeated generated content."""
    global_seen: set[bytes] = set()
    per_class_seen = {label: set() for label in CLASS_NAMES}
    per_class_repeated = {label: 0 for label in CLASS_NAMES}
    ordered_digest = hashlib.sha256()

    for row, label_raw in zip(x, y_attack, strict=True):
        label = int(label_raw)
        row_digest = hashlib.sha256(
            memoryview(np.ascontiguousarray(row)).cast("B")).digest()
        ordered_digest.update(bytes((label,)))
        ordered_digest.update(row_digest)
        if row_digest in per_class_seen[label]:
            per_class_repeated[label] += 1
        else:
            per_class_seen[label].add(row_digest)
        global_seen.add(row_digest)

    return {
        "hash_algorithm": "sha256",
        "content_sha256": ordered_digest.hexdigest(),
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


def generate_pool(
        train_x: np.ndarray,
        normal_indices: np.ndarray,
        *,
        per_attack: int,
        construction_seed: int = CONSTRUCTION_SEED,
) -> dict[str, Any]:
    """Construct, shuffle, and validate a Rule strict-v2 pool in memory."""
    per_attack = _require_positive_int(per_attack, "per_attack")
    construction_seed = _require_seed(construction_seed)
    train_x = np.asarray(train_x)
    normal_indices = np.asarray(normal_indices, dtype=np.int64).reshape(-1)
    if train_x.ndim != 3 or train_x.shape[1:] != (
            WINDOW_SIZE, len(lc.FEATURE_NAMES)):
        raise ValueError(f"invalid train_x shape: {train_x.shape}")
    if len(normal_indices) == 0:
        raise ValueError("normal_indices must not be empty")
    if normal_indices.min() < 0 or normal_indices.max() >= len(train_x):
        raise ValueError("normal_indices contain an out-of-range row")

    rng = np.random.default_rng(construction_seed)
    total = per_attack * len(SPECS)
    x = np.empty(
        (total, WINDOW_SIZE, len(lc.FEATURE_NAMES)), dtype=np.float32)
    y_binary = np.ones(total, dtype=np.int8)
    y_attack = np.empty(total, dtype=np.int8)
    synthetic_type = np.empty(total, dtype="<U5")
    condition_label = np.empty(total, dtype=np.int8)
    condition_name = np.empty(total, dtype="<U5")
    injection_count = np.empty(total, dtype=np.int16)
    base_source_index = np.empty(total, dtype=np.int64)
    construction_ordinal = np.empty(total, dtype=np.int64)
    per_class_audit: dict[str, Any] = {}

    for class_position, (name, label, injector) in enumerate(SPECS):
        start = class_position * per_attack
        stop = start + per_attack
        base = rng.choice(normal_indices, size=per_attack, replace=True)
        x[start:stop] = train_x[base].astype(np.float32, copy=False)
        counts = np.empty(per_attack, dtype=np.int16)
        for local_index in range(per_attack):
            counts[local_index] = injector(rng, x[start + local_index])

        y_attack[start:stop] = label
        synthetic_type[start:stop] = name
        condition_label[start:stop] = label
        condition_name[start:stop] = name
        injection_count[start:stop] = counts
        base_source_index[start:stop] = base
        construction_ordinal[start:stop] = np.arange(
            per_attack, dtype=np.int64)

        class_validity = summarize_protocol_validity(x[start:stop])
        if not class_validity["protocol_valid"]:
            raise RuntimeError(
                f"{name} construction failed frame-protocol validation")
        per_class_audit[name] = {
            "attack_type_id": label,
            "windows": per_attack,
            "injection_count_mean": float(counts.mean()),
            "injection_count_min": int(counts.min()),
            "injection_count_max": int(counts.max()),
            "base_source_rows_unique": int(np.unique(base).size),
            "base_source_rows_repeated": int(
                len(base) - np.unique(base).size),
            "protocol": class_validity,
        }

    order = rng.permutation(total)
    arrays = {
        "x": x[order],
        "y_binary": y_binary[order],
        "y_attack_type": y_attack[order],
        "synthetic_type": synthetic_type[order],
        "condition_label": condition_label[order],
        "condition_name": condition_name[order],
        "injection_count": injection_count[order],
        "base_source_index": base_source_index[order],
        "construction_ordinal": construction_ordinal[order],
    }

    protocol = summarize_protocol_validity(arrays["x"])
    condition = summarize_condition_metadata(
        arrays["y_binary"],
        arrays["y_attack_type"],
        arrays["synthetic_type"],
        condition_label=arrays["condition_label"],
        condition_name=arrays["condition_name"],
    )
    class_counts = {
        str(label): int(np.count_nonzero(arrays["y_attack_type"] == label))
        for label in CLASS_NAMES
    }
    expected_counts = {str(label): per_attack for label in CLASS_NAMES}
    uniqueness = _content_uniqueness(
        arrays["x"], arrays["y_attack_type"])

    if arrays["x"].shape != (
            total, WINDOW_SIZE, len(lc.FEATURE_NAMES)):
        raise RuntimeError(f"generated pool shape mismatch: {arrays['x'].shape}")
    if class_counts != expected_counts:
        raise RuntimeError(
            f"generated class counts mismatch: {class_counts}")
    if not protocol["protocol_valid"]:
        raise RuntimeError("generated pool failed frame-protocol validation")
    if not condition["condition_metadata_valid"]:
        raise RuntimeError("generated pool failed condition-metadata validation")
    if uniqueness["total"]["repeated"] != 0:
        raise RuntimeError(
            "generated pool contains exact duplicate window content: "
            f"{uniqueness['total']}")

    return {
        "arrays": arrays,
        "audit": {
            "shape": list(arrays["x"].shape),
            "class_counts": class_counts,
            "protocol": protocol,
            "condition_metadata": condition,
            "content_uniqueness": uniqueness,
            "per_class": per_class_audit,
        },
    }


def _journal_output_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = lc.ROOT / path
    # Keep the logical journal/datasets path rather than resolving its shared
    # datasets symlink; outputs still cannot lexically escape journal/.
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(lc.ROOT.absolute()):
        raise ValueError(f"output must remain under journal/: {path}")
    return path


def _portable_path(path: Path) -> str:
    normalized = Path(os.path.abspath(Path(path).expanduser()))
    try:
        return normalized.relative_to(lc.ROOT.parent.absolute()).as_posix()
    except ValueError:
        return str(normalized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    provenance = lc._source_provenance()
    return {
        **provenance,
        "worktree_dirty": not provenance["source_worktree_clean"],
    }


def _assert_output_targets(
        pool_out: Path,
        stats_out: Path,
        log_out: Path,
        *,
        allow_overwrite: bool,
) -> None:
    paths = [pool_out, stats_out, log_out]
    if len(set(paths)) != len(paths):
        raise ValueError("pool, statistics, and log outputs must be distinct")

    protected = {
        Path(os.path.abspath(lc.SYNTHETIC / "rule_based_windows.npz")),
        Path(os.path.abspath(
            lc.TABLES / "rule_based_synthetic_statistics.csv")),
    }
    selected_protected = [str(path) for path in paths if path in protected]
    if selected_protected:
        raise ValueError(
            "legacy Rule artifacts are protected and cannot be selected: "
            f"{selected_protected}")
    directories = [str(path) for path in paths if path.is_dir()]
    if directories:
        raise IsADirectoryError(
            f"output targets must be files, got directories: {directories}")
    existing = [str(path) for path in paths if path.exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(
            "refusing to overwrite existing outputs; choose new paths or "
            f"pass --allow-overwrite explicitly: {existing}")


def _validate_canonical_rule_request(
        pool_out,
        *,
        stats_out=None,
        log_out=None,
        construction_seed,
        per_attack,
        target_ratio):
    """Preflight the immutable Rule identity before expensive construction."""
    selected = _journal_output_path(pool_out)
    declared = lc.STRICT_V2_POOL_CONFIGS["rule"]
    declared_path = _journal_output_path(declared["path"])
    if selected != declared_path:
        return
    if stats_out is not None and (
        _journal_output_path(stats_out)
        != _journal_output_path(declared["statistics_path"])
    ):
        raise ValueError(
            "canonical strict-v2 Rule pool requires its declared "
            "statistics-output path"
        )
    if log_out is not None and (
        _journal_output_path(log_out)
        != _journal_output_path(declared["generation_log"])
    ):
        raise ValueError(
            "canonical strict-v2 Rule pool requires its declared log-output "
            "path"
        )
    expected_seed = int(
        declared["generator_configuration"]["construction_seed"])
    if int(construction_seed) != expected_seed:
        raise ValueError(
            f"canonical strict-v2 Rule pool requires construction seed "
            f"{expected_seed}, got {int(construction_seed)}"
        )
    minimum = int(declared["minimum_windows_per_class"])
    if int(per_attack) < minimum:
        raise ValueError(
            f"canonical strict-v2 Rule pool requires --per-attack >= "
            f"{minimum}, got {int(per_attack)}"
        )
    if not math.isclose(
            float(target_ratio), DEFAULT_TARGET_RATIO,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "canonical strict-v2 Rule pool requires --target-ratio 1.0; "
            "use a distinct output identity for another target ratio"
        )


def _atomic_savez_compressed(path: Path, **arrays) -> None:
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _statistics_csv(
        audit: dict[str, Any],
        *,
        pool_sha256: str,
        configuration_sha256: str,
) -> str:
    rows = []
    for label, name in CLASS_NAMES.items():
        item = audit["per_class"][name]
        rows.append({
            "generator": "rule_based",
            "pool_identifier": POOL_IDENTIFIER,
            "pool_schema_version": POOL_SCHEMA_VERSION,
            "synthetic_type": name,
            "windows": item["windows"],
            "binary_label": "Attack",
            "attack_type_id": label,
            "injection_count_mean": item["injection_count_mean"],
            "injection_count_min": item["injection_count_min"],
            "injection_count_max": item["injection_count_max"],
            "base_source_rows_unique": item["base_source_rows_unique"],
            "base_source_rows_repeated": item["base_source_rows_repeated"],
            "generated_content_unique": (
                audit["content_uniqueness"]["per_class"][str(label)]["unique"]),
            "generated_content_repeated": (
                audit["content_uniqueness"]["per_class"][str(label)]["repeated"]),
            "protocol_valid": item["protocol"]["protocol_valid"],
            "protocol_valid_frame_frac": (
                item["protocol"]["protocol_valid_frame_frac"]),
            "condition_metadata_valid": (
                audit["condition_metadata"]["condition_metadata_valid"]),
            "pool_content_sha256": (
                audit["content_uniqueness"]["content_sha256"]),
            "pool_file_sha256": pool_sha256,
            "configuration_sha256": configuration_sha256,
        })
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def generate_from_archive(
        train_path: Path,
        *,
        per_attack: int,
        target_ratio: float,
        construction_seed: int,
        pool_out: Path,
        stats_out: Path,
        log_out: Path,
        allow_overwrite: bool = False,
        expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the validated construction and publish all three versioned outputs."""
    train_path = Path(train_path).expanduser()
    if not train_path.is_absolute():
        train_path = lc.ROOT / train_path
    train_path = Path(os.path.abspath(train_path))
    if not train_path.is_file():
        raise FileNotFoundError(f"source train archive does not exist: {train_path}")

    source_sha256_before = _sha256_file(train_path)
    if (
        expected_source_sha256 is not None
        and source_sha256_before != expected_source_sha256
    ):
        raise ValueError(
            "canonical strict-v2 Rule generation requires the frozen "
            f"Car-Hacking train archive SHA-256 {expected_source_sha256}; "
            f"observed {source_sha256_before}"
        )
    pool_out = _journal_output_path(pool_out)
    stats_out = _journal_output_path(stats_out)
    log_out = _journal_output_path(log_out)
    _assert_output_targets(
        pool_out, stats_out, log_out, allow_overwrite=allow_overwrite)

    # Freeze source/code provenance before the expensive construction. A dirty
    # source tree must fail immediately, and code loaded by this process must
    # not change while the pool is being generated.
    git = _git_provenance()
    script_path = Path(__file__)
    code_paths = {
        "generator_script_sha256": script_path,
        "generator_protocol_sha256": script_path.with_name(
            "generator_protocol.py"),
        "lib_common_sha256": script_path.with_name("lib_common.py"),
    }
    code_hashes = {
        key: _sha256_file(path) for key, path in code_paths.items()
    }

    with np.load(train_path, allow_pickle=True) as train:
        train_x, normal_indices = validate_train_source(train)
        capacity = capacity_requirements(len(train_x), target_ratio)
        require_pool_capacity(per_attack, capacity)
        generated = generate_pool(
            train_x,
            normal_indices,
            per_attack=per_attack,
            construction_seed=construction_seed,
        )

    source_sha256_after = _sha256_file(train_path)
    if source_sha256_after != source_sha256_before:
        raise RuntimeError(
            "source train archive changed during generation; refusing to "
            "publish outputs")
    code_hashes_after = {
        key: _sha256_file(path) for key, path in code_paths.items()
    }
    if code_hashes_after != code_hashes:
        raise RuntimeError(
            "strict-v2 Rule generation code changed during construction; "
            "refusing to publish outputs"
        )
    source = {
        "path": _portable_path(train_path),
        "sha256": source_sha256_before,
        "windows": int(capacity["real_train_windows"]),
        "normal_windows": int(len(normal_indices)),
        "role": "Car-Hacking train normal windows only",
    }
    configuration = {
        "schema_version": "rule_based_generation_configuration_v2",
        "pool_identifier": POOL_IDENTIFIER,
        "pool_schema_version": POOL_SCHEMA_VERSION,
        "generator_configuration": {
            **GENERATOR_CONFIGURATION,
            "construction_seed": int(construction_seed),
            "source_train_sha256": source_sha256_before,
        },
        "per_attack": int(per_attack),
        "total_windows": int(per_attack * len(CLASS_NAMES)),
        "capacity": capacity,
        "source": source,
        "source_commit": git["source_commit"],
        "source_provenance": {
            key: value
            for key, value in git.items()
            if key != "worktree_dirty"
        },
        "code_hashes": code_hashes,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "feature_names": list(lc.FEATURE_NAMES),
        "downstream_sampling_policy": lc.STRICT_V2_SAMPLING_POLICY,
    }
    configuration_json = _canonical_json(configuration)
    configuration_sha256 = hashlib.sha256(
        configuration_json.encode("utf-8")).hexdigest()

    arrays = generated["arrays"]
    audit = generated["audit"]
    pool_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez_compressed(
        pool_out,
        **arrays,
        feature_names=np.asarray(lc.FEATURE_NAMES, dtype=np.str_),
        window_size=np.asarray(WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(STRIDE, dtype=np.int32),
        seed=np.asarray(construction_seed, dtype=np.uint64),
        construction_seed=np.asarray(construction_seed, dtype=np.uint64),
        generator=np.asarray("rule_based", dtype=np.str_),
        pool_identifier=np.asarray(POOL_IDENTIFIER, dtype=np.str_),
        pool_schema_version=np.asarray(POOL_SCHEMA_VERSION, dtype=np.str_),
        generator_protocol_version=np.asarray(
            GENERATOR_PROTOCOL_VERSION, dtype=np.str_),
        downstream_sampling_policy=np.asarray(
            lc.STRICT_V2_SAMPLING_POLICY, dtype=np.str_),
        source_train_path=np.asarray(source["path"], dtype=np.str_),
        source_train_sha256=np.asarray(source["sha256"], dtype=np.str_),
        source_commit=np.asarray(git["source_commit"], dtype=np.str_),
        source_worktree_clean=np.asarray(
            git["source_worktree_clean"], dtype=np.bool_),
        source_tracked_state_clean=np.asarray(
            git["source_tracked_state_clean"], dtype=np.bool_),
        source_allowed_untracked_artifacts_json=np.asarray(
            _canonical_json(git["source_allowed_untracked_artifacts"]),
            dtype=np.str_,
        ),
        generator_script_sha256=np.asarray(
            code_hashes["generator_script_sha256"], dtype=np.str_),
        pool_content_sha256=np.asarray(
            audit["content_uniqueness"]["content_sha256"], dtype=np.str_),
        generation_configuration_json=np.asarray(
            configuration_json, dtype=np.str_),
        generation_configuration_sha256=np.asarray(
            configuration_sha256, dtype=np.str_),
    )
    pool_sha256 = _sha256_file(pool_out)
    stats_text = _statistics_csv(
        audit,
        pool_sha256=pool_sha256,
        configuration_sha256=configuration_sha256,
    )
    _atomic_write_text(stats_out, stats_text)
    stats_sha256 = _sha256_file(stats_out)

    record = {
        "schema_version": "rule_based_generation_log_v2",
        "pool_identifier": POOL_IDENTIFIER,
        "outputs": {
            "pool": {
                "path": _portable_path(pool_out),
                "sha256": pool_sha256,
            },
            "statistics": {
                "path": _portable_path(stats_out),
                "sha256": stats_sha256,
            },
            "log": {"path": _portable_path(log_out)},
        },
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "worktree_dirty_at_generation": git["worktree_dirty"],
        "source_provenance": {
            key: value
            for key, value in git.items()
            if key != "worktree_dirty"
        },
        "audit": audit,
    }
    _atomic_write_text(log_out, json.dumps(record, indent=2, sort_keys=True))
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a journal-only, versioned Rule pool for strict-v2 "
            "without-replacement augmentation."))
    parser.add_argument(
        "--train-windows",
        type=Path,
        default=lc.WINDOWS / "train_windows.npz",
        help="Car-Hacking train_windows.npz source (default: journal dataset)",
    )
    parser.add_argument(
        "--per-attack",
        type=int,
        default=DEFAULT_PER_ATTACK,
        help=(
            "constructed rows per attack class; must cover the target-ratio "
            f"capacity (default: {DEFAULT_PER_ATTACK})"),
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=DEFAULT_TARGET_RATIO,
        help="largest strict-v2 augmentation ratio this pool must support",
    )
    parser.add_argument(
        "--construction-seed",
        type=int,
        default=CONSTRUCTION_SEED,
        help=f"Rule construction seed (default: {CONSTRUCTION_SEED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=lc.SYNTHETIC / "rule_based_windows_unique_pool_v2.npz",
        help="versioned NPZ output under journal/",
    )
    parser.add_argument(
        "--statistics-output",
        type=Path,
        default=(
            lc.TABLES
            / "rule_based_synthetic_statistics_unique_pool_v2.csv"),
        help="versioned per-class statistics CSV under journal/",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=(
            lc.LOGS / "generate_rule_based_synthetic_unique_pool_v2.log"),
        help="versioned generation/provenance JSON log under journal/",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "explicitly replace the selected v2 outputs; legacy Rule artifact "
            "paths remain protected"),
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_canonical_rule_request(
            args.output,
            stats_out=args.statistics_output,
            log_out=args.log_output,
            construction_seed=args.construction_seed,
            per_attack=args.per_attack,
            target_ratio=args.target_ratio,
        )
        record = generate_from_archive(
            args.train_windows,
            per_attack=args.per_attack,
            target_ratio=args.target_ratio,
            construction_seed=args.construction_seed,
            pool_out=args.output,
            stats_out=args.statistics_output,
            log_out=args.log_output,
            allow_overwrite=args.allow_overwrite,
            expected_source_sha256=(
                lc.STRICT_V2_REAL_SPLIT_SHA256["train"]
            ),
        )
    except (FileExistsError, FileNotFoundError, IsADirectoryError,
            TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "pool": record["outputs"]["pool"],
        "statistics": record["outputs"]["statistics"],
        "log": record["outputs"]["log"],
        "configuration_sha256": record["configuration_sha256"],
        "class_counts": record["audit"]["class_counts"],
        "content_uniqueness": record["audit"]["content_uniqueness"]["total"],
        "protocol_valid": record["audit"]["protocol"]["protocol_valid"],
    }, indent=2))


if __name__ == "__main__":
    main()
