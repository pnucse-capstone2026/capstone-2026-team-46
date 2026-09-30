#!/usr/bin/env python3
"""Evaluate matched-update checkpoints on raw-frame-disjoint variant subsets."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERA_ROOT = REPO_ROOT / "wisa_camera_ready"
DATA_ROOT = REPO_ROOT / "datasets"
TABLE_ROOT = CAMERA_ROOT / "results" / "tables"
LOG_ROOT = CAMERA_ROOT / "results" / "logs"
EXPERIMENT_ROOT = CAMERA_ROOT / "experiments" / "03_overlap_unit_audit"
PLAN_PATH = EXPERIMENT_ROOT / "PLAN.md"
FINAL_MANIFEST_PATH = EXPERIMENT_ROOT / "manifest_v1.json"
RESULTS_PATH = EXPERIMENT_ROOT / "RESULTS_v1.md"
SCRIPT_PATH = Path(__file__).resolve()

VERSION = "overlap_unit_audit_v1"
PLAN_COMMIT = "08f566108eeb6e84256800dff7ff7c59c9ef8f6d"
WINDOW_SIZE = 128
SEEDS = (7, 42, 123, 2026, 3407)
ARMS = ("real_only", "rule_0p30", "real_oversampling_0p30")
CONTRASTS = (
    ("rule_0p30_minus_real_only", "rule_0p30", "real_only"),
    (
        "rule_0p30_minus_real_oversampling_0p30",
        "rule_0p30",
        "real_oversampling_0p30",
    ),
)
CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}

TEST_PATH = DATA_ROOT / "windows" / "test_windows.npz"
FIXED_PATH = DATA_ROOT / "windows" / "variant_test_windows.npz"
SENSITIVITY_PATH = DATA_ROOT / "windows" / "variant_sensitivity_windows.npz"
TRAINING_MANIFEST_PATH = (
    CAMERA_ROOT / "experiments" / "02_matched_update" / "training_manifest_v1.json"
)
MATCHED_MANIFEST_PATH = (
    CAMERA_ROOT / "experiments" / "02_matched_update" / "manifest_v1.json"
)
MATCHED_SCRIPT_PATH = CAMERA_ROOT / "scripts" / "run_matched_update_v1.py"
MATCHED_BY_SEED_PATH = TABLE_ROOT / "matched_update_by_seed_v1.csv"
STANDARDIZER_PATH = (
    CAMERA_ROOT / "models" / "matched_update_v1" / "standardizer_source_real_train_v1.npz"
)

OUTPUTS = {
    "origin_summary": TABLE_ROOT / "overlap_unit_origin_summary_v1.csv",
    "by_seed": TABLE_ROOT / "overlap_unit_by_seed_v1.csv",
    "summary": TABLE_ROOT / "overlap_unit_summary_v1.csv",
    "by_source": TABLE_ROOT / "overlap_unit_by_source_v1.csv",
    "sensitivity": TABLE_ROOT / "overlap_unit_sensitivity_by_scenario_v1.csv",
    "log": LOG_ROOT / "overlap_unit_v1.log",
    "results": RESULTS_PATH,
}

EXPECTED_FILES = {
    "datasets/windows/test_windows.npz": (
        50_992_356,
        "4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7",
    ),
    "datasets/windows/variant_test_windows.npz": (
        60_445_693,
        "91bd4d5dca2bd73907fad11bb94ea6dec075f60bb3b24e3c517617bf2d50b2ba",
    ),
    "datasets/windows/variant_sensitivity_windows.npz": (
        29_966_923,
        "4266fb3bbb3e248ef1ab53aaca19cb4534a3815a18c9134dc3db446cdfcf5674",
    ),
    "wisa/scripts/generate_variant_test.py": (
        6_802,
        "fde7346a8f32a84c03c0c0340a9c892764ba5e2ef737d6afbab59b64194595bc",
    ),
    "wisa/scripts/generate_variant_sensitivity.py": (
        9_925,
        "6368717317c679f79f41ca05626fdbcc427db61ceeca419f60f662b67a916069",
    ),
    "wisa_camera_ready/experiments/02_matched_update/training_manifest_v1.json": (
        52_313,
        "c03419848737d29714381c84f84e8587756ec4b2685a26e9cb2be46b0a9fb228",
    ),
    "wisa_camera_ready/experiments/02_matched_update/manifest_v1.json": (
        13_050,
        "cc582638a9b086cff6f0cb0a7374ce72396205fd915ac168c554a939e0767671",
    ),
    "wisa_camera_ready/scripts/run_matched_update_v1.py": (
        83_564,
        "afacf325e8fca206d5ae15ac928fcc98a1fd8d70d77433962fbbac25deee6f71",
    ),
    "wisa_camera_ready/models/matched_update_v1/standardizer_source_real_train_v1.npz": (
        1_440,
        "14f5ee3f90429c04c08da93d00ac2cfc2f13ac7727347838f14af68cba731810",
    ),
}


sys.path.insert(0, str(CAMERA_ROOT / "scripts"))
import run_matched_update_v1 as matched  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_int64(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def file_entry(path: Path) -> dict:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def verify_expected_files() -> list[dict]:
    entries = []
    for name, (expected_bytes, expected_hash) in EXPECTED_FILES.items():
        path = REPO_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(name)
        entry = file_entry(path)
        if entry["bytes"] != expected_bytes or entry["sha256"] != expected_hash:
            raise ValueError(f"frozen input mismatch: {name}: {entry}")
        entries.append(entry)
    return entries


def verify_plan() -> dict:
    plan_blob = subprocess.run(
        ["git", "show", f"{PLAN_COMMIT}:{relative(PLAN_PATH)}"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    working = PLAN_PATH.read_bytes()
    if plan_blob != working:
        raise ValueError("working analysis plan differs from frozen plan commit")
    return {
        "path": relative(PLAN_PATH),
        "commit": PLAN_COMMIT,
        "sha256": hashlib.sha256(working).hexdigest(),
    }


def verify_script_state(*, allow_uncommitted: bool) -> dict:
    commit = run_git("rev-parse", "HEAD")
    if not allow_uncommitted:
        status = run_git("status", "--porcelain", "--untracked-files=all")
        if status:
            raise ValueError("evaluation requires a clean committed worktree")
        blob = subprocess.run(
            ["git", "show", f"HEAD:{relative(SCRIPT_PATH)}"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != sha256_file(SCRIPT_PATH):
            raise ValueError("working evaluator differs from committed evaluator")
    if run_git("status", "--porcelain", "--", "wisa"):
        raise ValueError("frozen wisa/ tree is dirty")
    return {
        "path": relative(SCRIPT_PATH),
        "commit": commit,
        "sha256": sha256_file(SCRIPT_PATH),
        "allow_uncommitted_preflight": allow_uncommitted,
    }


def matched_output_entry(manifest: dict, path: Path) -> dict:
    wanted = relative(path)
    for entry in manifest["outputs"]:
        if entry["path"] == wanted:
            if not path.is_file() or file_entry(path) != entry:
                raise ValueError(f"matched-output mismatch: {wanted}")
            return entry
    raise ValueError(f"matched-output absent from manifest: {wanted}")


def validate_training_lineage() -> tuple[dict, dict, list[dict]]:
    training = json.loads(TRAINING_MANIFEST_PATH.read_text())
    final = json.loads(MATCHED_MANIFEST_PATH.read_text())
    if training.get("version") != "matched_update_v1":
        raise ValueError("unexpected matched training-manifest version")
    if final.get("version") != "matched_update_v1":
        raise ValueError("unexpected matched final-manifest version")
    fits = training.get("fits", [])
    if len(fits) != 15:
        raise ValueError("matched training manifest must contain 15 fits")
    expected_cells = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed_cells = {
        (str(fit["arm"]), int(fit["pipeline_seed"])) for fit in fits
    }
    if observed_cells != expected_cells:
        raise ValueError("matched checkpoint cells are incomplete")
    checkpoint_entries = []
    for fit in fits:
        if int(fit["updates_run"]) != 7_992:
            raise ValueError("matched fit did not complete 7,992 updates")
        checkpoint = REPO_ROOT / fit["checkpoint"]["path"]
        entry = file_entry(checkpoint)
        if entry != fit["checkpoint"]:
            raise ValueError(f"checkpoint mismatch: {relative(checkpoint)}")
        checkpoint_entries.append(entry)
    matched_output_entry(final, MATCHED_BY_SEED_PATH)
    return training, final, checkpoint_entries


def count_rows(
    evaluation: str,
    group_type: str,
    group_values: np.ndarray,
    retained_mask: np.ndarray,
) -> list[dict]:
    rows = []
    values = np.asarray(group_values, dtype=object)
    for value in sorted(np.unique(values), key=str):
        mask = values == value
        full_count = int(mask.sum())
        retained_count = int((mask & retained_mask).sum())
        rows.append(
            {
                "evaluation": evaluation,
                "group_type": group_type,
                "group_key": str(value),
                "full_windows": full_count,
                "retained_windows": retained_count,
                "retained_fraction": retained_count / full_count,
            }
        )
    return rows


def validate_nonoverlap(
    source: np.ndarray,
    segment: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> dict:
    if len(start) == 0:
        raise ValueError("non-overlap subset is empty")
    records = list(zip(source.tolist(), segment.tolist(), start.tolist(), end.tolist()))
    if len(records) != len(set(records)):
        raise ValueError("retained provenance rows are not unique")
    groups = sorted({(str(src), int(seg)) for src, seg in zip(source, segment)})
    comparisons = 0
    for src, seg in groups:
        mask = (source == src) & (segment == seg)
        order = np.argsort(start[mask], kind="stable")
        starts = start[mask][order]
        ends = end[mask][order]
        if len(starts) > 1:
            comparisons += len(starts) - 1
            if np.any(starts[1:] <= ends[:-1]):
                raise ValueError(f"retained raw-frame overlap in {src} segment {seg}")
    return {
        "groups": len(groups),
        "adjacent_interval_comparisons": comparisons,
        "pairwise_overlap_detected": False,
    }


def load_frozen_generator(path: Path, module_name: str):
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen generator: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def require_array_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"{name} shape/dtype mismatch: "
            f"{actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}"
        )
    if not np.array_equal(actual, expected):
        raise ValueError(f"{name} is not bit-exact")


def replay_fixed(
    test_x: np.ndarray, normal_idx: np.ndarray, archive
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = load_frozen_generator(
        REPO_ROOT / "wisa" / "scripts" / "generate_variant_test.py",
        "frozen_generate_variant_test",
    )
    rng = np.random.default_rng(generator.SEED)
    selected = rng.choice(
        normal_idx,
        size=generator.NORMAL_WINDOWS + generator.PER_ATTACK * 4,
        replace=False,
    ).astype(np.int64, copy=False)
    normal_base = selected[: generator.NORMAL_WINDOWS]
    attack_base = selected[generator.NORMAL_WINDOWS :]

    xs = [test_x[normal_base].copy()]
    y_binary = [0] * generator.NORMAL_WINDOWS
    y_attack = [0] * generator.NORMAL_WINDOWS
    variant_type = ["Normal"] * generator.NORMAL_WINDOWS
    injection_counts = [0] * generator.NORMAL_WINDOWS
    attack_specs = [
        (
            "DoS",
            generator.ATTACK_IDS["DoS"],
            lambda r, x: generator.inject_dos(r, x),
        ),
        (
            "Fuzzy",
            generator.ATTACK_IDS["Fuzzy"],
            lambda r, x: generator.inject_fuzzy(r, x),
        ),
        (
            "Gear",
            generator.ATTACK_IDS["Gear"],
            lambda r, x: generator.inject_spoofing(r, x, 0x43F, "gear"),
        ),
        (
            "RPM",
            generator.ATTACK_IDS["RPM"],
            lambda r, x: generator.inject_spoofing(r, x, 0x316, "rpm"),
        ),
    ]
    cursor = 0
    for name, label, injector in attack_specs:
        base = attack_base[cursor : cursor + generator.PER_ATTACK]
        cursor += generator.PER_ATTACK
        x_part = test_x[base].copy()
        counts = [injector(rng, x_part[i]) for i in range(len(x_part))]
        xs.append(x_part)
        y_binary.extend([1] * generator.PER_ATTACK)
        y_attack.extend([label] * generator.PER_ATTACK)
        variant_type.extend([name] * generator.PER_ATTACK)
        injection_counts.extend(counts)

    replayed = {
        "x": np.concatenate(xs, axis=0).astype(np.float32),
        "y_binary": np.asarray(y_binary, dtype=np.int8),
        "y_attack_type": np.asarray(y_attack, dtype=np.int8),
        "variant_type": np.asarray(variant_type, dtype=object),
        "injection_count": np.asarray(injection_counts, dtype=np.int16),
    }
    order = rng.permutation(len(replayed["y_binary"])).astype(np.int64, copy=False)
    for key, values in replayed.items():
        require_array_equal(
            f"fixed_variant:{key}",
            np.asarray(archive[key]),
            np.asarray(values[order]),
        )
    ordered_selected = np.ascontiguousarray(selected[order], dtype=np.int64)
    del xs, replayed
    gc.collect()
    return selected, order, ordered_selected


def replay_sensitivity(
    test_x: np.ndarray, normal_idx: np.ndarray, archive
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = load_frozen_generator(
        REPO_ROOT / "wisa" / "scripts" / "generate_variant_sensitivity.py",
        "frozen_generate_variant_sensitivity",
    )
    rng = np.random.default_rng(generator.SEED)
    required = generator.NORMAL_WINDOWS + generator.PER_SCENARIO * len(
        generator.SCENARIOS
    )
    selected = rng.choice(normal_idx, size=required, replace=False).astype(
        np.int64, copy=False
    )
    normal_base = selected[: generator.NORMAL_WINDOWS]
    attack_base = selected[generator.NORMAL_WINDOWS :]

    xs = [test_x[normal_base].copy()]
    y_binary = [0] * generator.NORMAL_WINDOWS
    y_attack = [0] * generator.NORMAL_WINDOWS
    attack_type = ["Normal"] * generator.NORMAL_WINDOWS
    severity = ["normal"] * generator.NORMAL_WINDOWS
    scenario_id = ["Normal_normal"] * generator.NORMAL_WINDOWS
    burst_len_meta = [0] * generator.NORMAL_WINDOWS
    step_meta = [0] * generator.NORMAL_WINDOWS
    amplitude_meta = [0.0] * generator.NORMAL_WINDOWS
    injection_count = [0] * generator.NORMAL_WINDOWS

    cursor = 0
    for scenario in generator.SCENARIOS:
        base = attack_base[cursor : cursor + generator.PER_SCENARIO]
        cursor += generator.PER_SCENARIO
        x_part = test_x[base].copy()
        counts = []
        for i in range(len(x_part)):
            attack = scenario["attack_type"]
            if attack == "DoS":
                count = generator.inject_dos(
                    rng,
                    x_part[i],
                    scenario["burst_len"],
                    scenario["step"],
                    scenario["amplitude"],
                )
            elif attack == "Fuzzy":
                count = generator.inject_fuzzy(
                    rng,
                    x_part[i],
                    scenario["burst_len"],
                    scenario["step"],
                    scenario["amplitude"],
                )
            elif attack == "Gear":
                count = generator.inject_spoofing(
                    rng,
                    x_part[i],
                    0x43F,
                    "gear",
                    scenario["burst_len"],
                    scenario["step"],
                    scenario["amplitude"],
                )
            elif attack == "RPM":
                count = generator.inject_spoofing(
                    rng,
                    x_part[i],
                    0x316,
                    "rpm",
                    scenario["burst_len"],
                    scenario["step"],
                    scenario["amplitude"],
                )
            else:
                raise ValueError(attack)
            counts.append(count)
        sid = f"{scenario['attack_type']}_{scenario['severity']}"
        xs.append(x_part)
        y_binary.extend([1] * generator.PER_SCENARIO)
        y_attack.extend(
            [generator.ATTACK_IDS[scenario["attack_type"]]]
            * generator.PER_SCENARIO
        )
        attack_type.extend([scenario["attack_type"]] * generator.PER_SCENARIO)
        severity.extend([scenario["severity"]] * generator.PER_SCENARIO)
        scenario_id.extend([sid] * generator.PER_SCENARIO)
        burst_len_meta.extend([scenario["burst_len"]] * generator.PER_SCENARIO)
        step_meta.extend([scenario["step"]] * generator.PER_SCENARIO)
        amplitude_meta.extend([scenario["amplitude"]] * generator.PER_SCENARIO)
        injection_count.extend(counts)

    replayed = {
        "x": np.concatenate(xs, axis=0).astype(np.float32),
        "y_binary": np.asarray(y_binary, dtype=np.int8),
        "y_attack_type": np.asarray(y_attack, dtype=np.int8),
        "attack_type": np.asarray(attack_type, dtype=object),
        "severity": np.asarray(severity, dtype=object),
        "scenario_id": np.asarray(scenario_id, dtype=object),
        "burst_len": np.asarray(burst_len_meta, dtype=np.int16),
        "step": np.asarray(step_meta, dtype=np.int16),
        "amplitude": np.asarray(amplitude_meta, dtype=np.float32),
        "injection_count": np.asarray(injection_count, dtype=np.int16),
    }
    order = rng.permutation(len(replayed["y_binary"])).astype(np.int64, copy=False)
    for key, values in replayed.items():
        require_array_equal(
            f"sensitivity:{key}",
            np.asarray(archive[key]),
            np.asarray(values[order]),
        )
    ordered_selected = np.ascontiguousarray(selected[order], dtype=np.int64)
    del xs, replayed
    gc.collect()
    return selected, order, ordered_selected


def load_and_reconstruct() -> tuple[dict[str, dict], list[dict], dict]:
    with np.load(TEST_PATH, allow_pickle=True) as archive:
        test_y_binary = np.asarray(archive["y_binary"], dtype=np.int8)
        test_x = np.asarray(archive["x"], dtype=np.float32)
        source_all = np.asarray([str(v) for v in archive["source_file"]], dtype=object)
        segment_all = np.asarray(archive["segment_id"], dtype=np.int32)
        start_all = np.asarray(archive["start_index"], dtype=np.int64)
        end_all = np.asarray(archive["end_index"], dtype=np.int64)
        if int(archive["window_size"]) != 128 or int(archive["stride"]) != 32:
            raise ValueError("source-test window metadata mismatch")
    normal_idx = np.flatnonzero(test_y_binary == 0)
    if len(np.unique(normal_idx)) != len(normal_idx):
        raise ValueError("source-test normal indices are not unique")

    specifications = (
        {
            "name": "fixed_variant",
            "path": FIXED_PATH,
            "seed": 42,
            "count": 64_000,
            "normal_count": 32_000,
        },
        {
            "name": "sensitivity",
            "path": SENSITIVITY_PATH,
            "seed": 20_260_611,
            "count": 30_000,
            "normal_count": 12_000,
        },
    )
    reconstructed: dict[str, dict] = {}
    origin_rows: list[dict] = []
    reconstruction_manifest: dict[str, dict] = {}

    for spec in specifications:
        with np.load(spec["path"], allow_pickle=True) as archive:
            if int(archive["seed"]) != spec["seed"]:
                raise ValueError(f"{spec['name']} generator seed mismatch")
            x_all = np.asarray(archive["x"], dtype=np.float32)
            y_all = np.asarray(archive["y_attack_type"], dtype=np.int64)
            if len(x_all) != spec["count"] or len(y_all) != spec["count"]:
                raise ValueError(f"{spec['name']} row count mismatch")
            if spec["name"] == "fixed_variant":
                selected, order, ordered_selected = replay_fixed(
                    test_x, normal_idx, archive
                )
            else:
                selected, order, ordered_selected = replay_sensitivity(
                    test_x, normal_idx, archive
                )
            if len(np.unique(selected)) != spec["count"]:
                raise ValueError(f"{spec['name']} selected indices are not unique")
            metadata = {}
            if spec["name"] == "fixed_variant":
                metadata["variant_type"] = np.asarray(
                    archive["variant_type"], dtype=object
                )
            else:
                for key in ("attack_type", "severity", "scenario_id"):
                    metadata[key] = np.asarray(archive[key], dtype=object)

            source = source_all[ordered_selected]
            segment = segment_all[ordered_selected]
            start = start_all[ordered_selected]
            end = end_all[ordered_selected]
            retained_mask = (start % WINDOW_SIZE == 0) & (
                end - start + 1 == WINDOW_SIZE
            )
            retained_index = np.flatnonzero(retained_mask)
            if len(retained_index) == 0:
                raise ValueError(f"{spec['name']} retained no aligned windows")
            nonoverlap = validate_nonoverlap(
                source[retained_mask],
                segment[retained_mask],
                start[retained_mask],
                end[retained_mask],
            )
            retained_labels = y_all[retained_mask]
            if set(np.unique(retained_labels).tolist()) != {0, 1, 2, 3, 4}:
                raise ValueError(f"{spec['name']} retained labels are incomplete")

            origin_rows.append(
                {
                    "evaluation": spec["name"],
                    "group_type": "overall",
                    "group_key": "all",
                    "full_windows": len(y_all),
                    "retained_windows": len(retained_index),
                    "retained_fraction": len(retained_index) / len(y_all),
                }
            )
            origin_rows.extend(
                count_rows(
                    spec["name"],
                    "construction_label",
                    np.asarray([CLASS_NAMES[int(v)] for v in y_all], dtype=object),
                    retained_mask,
                )
            )
            origin_rows.extend(
                count_rows(
                    spec["name"],
                    "source_file",
                    source,
                    retained_mask,
                )
            )
            if spec["name"] == "sensitivity":
                origin_rows.extend(
                    count_rows(
                        spec["name"],
                        "scenario",
                        metadata["scenario_id"],
                        retained_mask,
                    )
                )

            reconstructed[spec["name"]] = {
                "x": np.ascontiguousarray(x_all[retained_mask], dtype=np.float32),
                "labels": np.ascontiguousarray(
                    retained_labels, dtype=np.int64
                ),
                "source_file": np.asarray(source[retained_mask], dtype=object),
                "segment_id": np.asarray(segment[retained_mask], dtype=np.int32),
                "start_index": np.asarray(start[retained_mask], dtype=np.int64),
                "end_index": np.asarray(end[retained_mask], dtype=np.int64),
                "metadata": {
                    key: np.asarray(values[retained_mask], dtype=object)
                    for key, values in metadata.items()
                },
            }
            reconstruction_manifest[spec["name"]] = {
                "generator_seed": spec["seed"],
                "selected_indices": len(selected),
                "selected_index_sha256_int64_c": sha256_int64(selected),
                "final_permutation_sha256_int64_c": sha256_int64(order),
                "ordered_base_index_sha256_int64_c": sha256_int64(
                    ordered_selected
                ),
                "full_generator_replay_bit_exact": True,
                "retained_windows": len(retained_index),
                "retained_fraction": len(retained_index) / len(selected),
                "retained_evaluation_index_sha256_int64_c": sha256_int64(
                    retained_index
                ),
                "nonoverlap_validation": nonoverlap,
            }
            del x_all, y_all
        gc.collect()

    del test_x
    gc.collect()
    return reconstructed, origin_rows, reconstruction_manifest


def make_eval_loaders(reconstructed: dict[str, dict]) -> dict[str, DataLoader]:
    with np.load(STANDARDIZER_PATH, allow_pickle=True) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float32)
        std = np.asarray(archive["std"], dtype=np.float32)
        feature_names = tuple(str(v) for v in archive["feature_names"].tolist())
    if feature_names != matched.FEATURE_NAMES:
        raise ValueError("standardizer feature order mismatch")
    loaders = {}
    for name, data in reconstructed.items():
        standardized = matched.standardize(data["x"], mean, std)
        dataset = matched.WindowDataset(standardized)
        loaders[name] = DataLoader(
            dataset,
            batch_size=matched.INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        del standardized
    return loaders


def build_source_rows(
    evaluation: str,
    arm: str,
    seed: int,
    labels: np.ndarray,
    predictions: np.ndarray,
    source: np.ndarray,
) -> list[dict]:
    rows = []
    for source_file in sorted(np.unique(source), key=str):
        mask = source == source_file
        rows.append(
            {
                "evaluation": evaluation,
                "arm": arm,
                "pipeline_seed": seed,
                "source_file": str(source_file),
                **matched.classification_metrics(labels[mask], predictions[mask]),
            }
        )
    return rows


def summarize(
    by_seed: Sequence[dict], full_by_seed: pd.DataFrame
) -> list[dict]:
    lookup = {
        (str(row["arm"]), int(row["pipeline_seed"]), str(row["evaluation"])): row
        for row in by_seed
    }
    full_lookup = {
        (str(row.arm), int(row.pipeline_seed), str(row.split)): row
        for row in full_by_seed.itertuples(index=False)
        if str(row.split) in ("fixed_variant", "sensitivity")
    }
    rows = []
    for evaluation in ("fixed_variant", "sensitivity"):
        for metric in matched.SUMMARY_METRICS:
            for arm in ARMS:
                values = [
                    float(lookup[(arm, seed, evaluation)][metric]) for seed in SEEDS
                ]
                full_values = [
                    float(getattr(full_lookup[(arm, seed, evaluation)], metric))
                    for seed in SEEDS
                ]
                row = {
                    "row_type": "arm_summary",
                    "evaluation": evaluation,
                    "metric": metric,
                    "arm": arm,
                    "contrast": "",
                    "n_pipeline_seeds": len(SEEDS),
                    "nonoverlap_mean": float(np.mean(values)),
                    "nonoverlap_std": sample_std(values),
                    "full_mean": float(np.mean(full_values)),
                    "full_std": sample_std(full_values),
                    "nonoverlap_minus_full_mean": float(
                        np.mean(np.asarray(values) - np.asarray(full_values))
                    ),
                    "positive_in_all_paired_seeds": "",
                }
                row.update(
                    {f"nonoverlap_seed_{seed}": value for seed, value in zip(SEEDS, values)}
                )
                row.update(
                    {f"full_seed_{seed}": value for seed, value in zip(SEEDS, full_values)}
                )
                rows.append(row)
            for contrast, minuend, subtrahend in CONTRASTS:
                values = [
                    float(lookup[(minuend, seed, evaluation)][metric])
                    - float(lookup[(subtrahend, seed, evaluation)][metric])
                    for seed in SEEDS
                ]
                full_values = [
                    float(getattr(full_lookup[(minuend, seed, evaluation)], metric))
                    - float(
                        getattr(full_lookup[(subtrahend, seed, evaluation)], metric)
                    )
                    for seed in SEEDS
                ]
                row = {
                    "row_type": "paired_difference",
                    "evaluation": evaluation,
                    "metric": metric,
                    "arm": "",
                    "contrast": contrast,
                    "n_pipeline_seeds": len(SEEDS),
                    "nonoverlap_mean": float(np.mean(values)),
                    "nonoverlap_std": sample_std(values),
                    "full_mean": float(np.mean(full_values)),
                    "full_std": sample_std(full_values),
                    "nonoverlap_minus_full_mean": float(
                        np.mean(np.asarray(values) - np.asarray(full_values))
                    ),
                    "positive_in_all_paired_seeds": all(value > 0 for value in values),
                }
                row.update(
                    {f"nonoverlap_seed_{seed}": value for seed, value in zip(SEEDS, values)}
                )
                row.update(
                    {f"full_seed_{seed}": value for seed, value in zip(SEEDS, full_values)}
                )
                rows.append(row)
    return rows


def validate_outputs(
    reconstructed: dict[str, dict],
    by_seed: Sequence[dict],
    summary: Sequence[dict],
    by_source: Sequence[dict],
    sensitivity_rows: Sequence[dict],
) -> dict:
    expected_cells = {
        (arm, seed, evaluation)
        for arm in ARMS
        for seed in SEEDS
        for evaluation in ("fixed_variant", "sensitivity")
    }
    observed_cells = {
        (str(row["arm"]), int(row["pipeline_seed"]), str(row["evaluation"]))
        for row in by_seed
    }
    if observed_cells != expected_cells:
        raise ValueError("by-seed cells are incomplete")
    for row in by_seed:
        for metric in matched.SUMMARY_METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {metric} in by-seed output")
    expected_summary = 2 * len(matched.SUMMARY_METRICS) * (
        len(ARMS) + len(CONTRASTS)
    )
    if len(summary) != expected_summary:
        raise ValueError("summary row count mismatch")
    seed_sensitivity = [
        row for row in sensitivity_rows if row["row_type"] == "seed"
    ]
    if len(seed_sensitivity) != 15 * 20 * 2:
        raise ValueError("sensitivity scenario seed-row count mismatch")
    if len(by_source) != 2 * 15 * 5:
        raise ValueError("source-file row count mismatch")
    return {
        "required_arm_seed_evaluation_cells_complete": True,
        "all_pooled_metrics_finite_and_bounded": True,
        "summary_row_count": len(summary),
        "source_file_row_count": len(by_source),
        "sensitivity_seed_row_count": len(seed_sensitivity),
        "fixed_nonoverlap_windows": len(reconstructed["fixed_variant"]["labels"]),
        "sensitivity_nonoverlap_windows": len(reconstructed["sensitivity"]["labels"]),
    }


def summary_row(
    rows: Sequence[dict], evaluation: str, metric: str, contrast: str
) -> dict:
    matches = [
        row
        for row in rows
        if row["row_type"] == "paired_difference"
        and row["evaluation"] == evaluation
        and row["metric"] == metric
        and row["contrast"] == contrast
    ]
    if len(matches) != 1:
        raise ValueError("primary summary row lookup failed")
    return matches[0]


def arm_summary_row(
    rows: Sequence[dict], evaluation: str, metric: str, arm: str
) -> dict:
    matches = [
        row
        for row in rows
        if row["row_type"] == "arm_summary"
        and row["evaluation"] == evaluation
        and row["metric"] == metric
        and row["arm"] == arm
    ]
    if len(matches) != 1:
        raise ValueError("arm summary row lookup failed")
    return matches[0]


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_results_markdown(
    origin_rows: Sequence[dict],
    summary_rows: Sequence[dict],
    sensitivity_rows: Sequence[dict],
) -> str:
    counts = {
        row["evaluation"]: row
        for row in origin_rows
        if row["group_type"] == "overall"
    }
    primary = []
    for evaluation in ("fixed_variant", "sensitivity"):
        for contrast, _, _ in CONTRASTS:
            primary.append(
                summary_row(
                    summary_rows, evaluation, "binary_attack_recall", contrast
                )
            )
    all_positive = all(
        bool(row["positive_in_all_paired_seeds"]) for row in primary
    )
    any_nonpositive_mean = any(float(row["nonoverlap_mean"]) <= 0 for row in primary)
    if any_nonpositive_mean:
        branch = (
            "At least one primary mean contrast is non-positive. The corresponding "
            "controlled gain is overlap-sensitive under the frozen interpretation."
        )
    elif all_positive:
        branch = (
            "All four primary paired contrasts remain positive in every pipeline "
            "seed. The controlled fixed/sensitivity advantage survives this "
            "deterministic raw-frame-disjoint evaluation sensitivity."
        )
    else:
        branch = (
            "All primary mean contrasts remain positive, but at least one paired "
            "seed is non-positive. The overlap sensitivity is positive but "
            "seed-mixed."
        )

    exact_counterexamples = [
        row
        for row in sensitivity_rows
        if row["row_type"] == "paired_difference"
        and row["contrast"] == "rule_0p30_minus_real_only"
        and row["metric"] == "exact_attack_recall"
        and row["granularity"] in ("family", "scenario")
        and float(row["mean"]) < 0
    ]
    counterexample_text = (
        ", ".join(
            f"{row['group_label']} ({fmt(row['mean'])})"
            for row in exact_counterexamples[:8]
        )
        if exact_counterexamples
        else "none"
    )

    lines = [
        "# Overlap-Aware Evaluation-Unit Audit v1 — Results",
        "",
        "## Evidential status",
        "",
        "This is a post-review evaluation-only sensitivity over the existing 15 "
        "matched-update checkpoints. No model was retrained, selected, or calibrated "
        "with these outcomes. The retained base windows are raw-frame-disjoint within "
        "each generated evaluation, but they are not independent captures or vehicles.",
        "",
        "## Retained evaluation support",
        "",
        "| Evaluation | Full windows | Non-overlap windows | Retained fraction |",
        "|---|---:|---:|---:|",
    ]
    for evaluation in ("fixed_variant", "sensitivity"):
        row = counts[evaluation]
        lines.append(
            f"| {evaluation} | {row['full_windows']} | "
            f"{row['retained_windows']} | {float(row['retained_fraction']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Primary outcomes",
            "",
            "| Evaluation | Contrast | Full mean | Non-overlap mean ± sample std | "
            "Non-overlap minus full | Positive in all five seeds |",
            "|---|---|---:|---:|---:|:---:|",
        ]
    )
    for row in primary:
        lines.append(
            f"| {row['evaluation']} | `{row['contrast']}` | "
            f"{fmt(row['full_mean'])} | {fmt(row['nonoverlap_mean'])} ± "
            f"{fmt(row['nonoverlap_std'])} | "
            f"{fmt(row['nonoverlap_minus_full_mean'])} | "
            f"{'yes' if row['positive_in_all_paired_seeds'] else 'no'} |"
        )
    lines.extend(["", branch, "", "Arm-level non-overlap binary attack recall:"])
    lines.extend(
        [
            "",
            "| Evaluation | Real only | Rule +30% | Real oversampling +30% |",
            "|---|---:|---:|---:|",
        ]
    )
    for evaluation in ("fixed_variant", "sensitivity"):
        values = [
            arm_summary_row(
                summary_rows, evaluation, "binary_attack_recall", arm
            )["nonoverlap_mean"]
            for arm in ARMS
        ]
        lines.append(
            f"| {evaluation} | {fmt(values[0])} | {fmt(values[1])} | "
            f"{fmt(values[2])} |"
        )
    lines.extend(
        [
            "",
            "## Secondary boundary",
            "",
            "Negative Rule-minus-real exact-family/scenario mean contrasts remain "
            f"visible: {counterexample_text}.",
            "",
            "The five repeats remain paired pipeline seeds conditional on one split "
            "and one Rule construction. This audit does not estimate capture-, "
            "vehicle-, or population-level uncertainty and does not establish a "
            "causal effect of overlap.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_existing_manifest() -> None:
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("existing overlap manifest version mismatch")
    for entry in manifest["outputs"]:
        path = REPO_ROOT / entry["path"]
        if not path.is_file() or file_entry(path) != entry:
            raise ValueError(f"existing output mismatch: {entry['path']}")


def run_preflight(*, allow_uncommitted: bool) -> None:
    started = time.monotonic()
    inputs = verify_expected_files()
    plan = verify_plan()
    implementation = verify_script_state(allow_uncommitted=allow_uncommitted)
    training, final, checkpoints = validate_training_lineage()
    reconstructed, origin_rows, reconstruction = load_and_reconstruct()
    result = {
        "status": "preflight_passed",
        "version": VERSION,
        "plan": plan,
        "implementation": implementation,
        "frozen_inputs_verified": len(inputs),
        "checkpoints_verified": len(checkpoints),
        "matched_contract_id": training["contract_id"],
        "matched_final_contract_id": final["contract_id"],
        "reconstruction": reconstruction,
        "origin_summary_rows": len(origin_rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    del reconstructed
    gc.collect()


def run_evaluation() -> None:
    if FINAL_MANIFEST_PATH.exists():
        verify_existing_manifest()
        print(
            json.dumps(
                {
                    "status": "already_complete_and_verified",
                    "manifest": relative(FINAL_MANIFEST_PATH),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    partial = [path for path in OUTPUTS.values() if path.exists()]
    if partial:
        raise ValueError(
            "partial v1 outputs exist without final manifest: "
            + ", ".join(relative(path) for path in partial)
        )

    started_utc = utc_now()
    started = time.monotonic()
    events: list[dict] = []

    def log(event: str, **payload) -> None:
        events.append({"utc": utc_now(), "event": event, **payload})

    inputs = verify_expected_files()
    plan = verify_plan()
    implementation = verify_script_state(allow_uncommitted=False)
    training, matched_final, checkpoints = validate_training_lineage()
    full_output_entry = matched_output_entry(matched_final, MATCHED_BY_SEED_PATH)
    log(
        "preflight_verified",
        frozen_inputs=len(inputs),
        checkpoints=len(checkpoints),
        implementation_commit=implementation["commit"],
    )

    reconstructed, origin_rows, reconstruction = load_and_reconstruct()
    log("origins_reconstructed", reconstruction=reconstruction)
    loaders = make_eval_loaders(reconstructed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_lookup = {
        (str(fit["arm"]), int(fit["pipeline_seed"])): fit
        for fit in training["fits"]
    }

    by_seed: list[dict] = []
    by_source: list[dict] = []
    sensitivity_seed_rows: list[dict] = []
    for seed in SEEDS:
        for arm in ARMS:
            fit = fit_lookup[(arm, seed)]
            checkpoint_path = REPO_ROOT / fit["checkpoint"]["path"]
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                str(payload["arm"]) != arm
                or int(payload["pipeline_seed"]) != seed
                or int(payload["selected_update"]) != int(fit["selected_update"])
            ):
                raise ValueError(f"checkpoint payload mismatch: {arm} seed {seed}")
            model = matched.CNN1D().to(device)
            model.load_state_dict(payload["state_dict"])
            predictions = {
                evaluation: matched.predict_labels(model, loaders[evaluation], device)
                for evaluation in ("fixed_variant", "sensitivity")
            }
            for evaluation in ("fixed_variant", "sensitivity"):
                data = reconstructed[evaluation]
                labels = data["labels"]
                prediction = predictions[evaluation]
                if len(prediction) != len(labels):
                    raise ValueError("prediction/label length mismatch")
                by_seed.append(
                    {
                        "arm": arm,
                        "pipeline_seed": seed,
                        "evaluation": evaluation,
                        "subset_policy": "base_start_index_mod_128_equals_0",
                        "selected_update": int(fit["selected_update"]),
                        **matched.classification_metrics(labels, prediction),
                    }
                )
                by_source.extend(
                    build_source_rows(
                        evaluation,
                        arm,
                        seed,
                        labels,
                        prediction,
                        data["source_file"],
                    )
                )
            sensitivity_seed_rows.extend(
                matched.build_sensitivity_seed_rows(
                    arm,
                    seed,
                    reconstructed["sensitivity"]["labels"],
                    predictions["sensitivity"],
                    reconstructed["sensitivity"]["metadata"],
                )
            )
            log(
                "checkpoint_evaluated",
                arm=arm,
                pipeline_seed=seed,
                selected_update=int(fit["selected_update"]),
                fixed_binary_attack_recall=by_seed[-2]["binary_attack_recall"],
                sensitivity_binary_attack_recall=by_seed[-1][
                    "binary_attack_recall"
                ],
            )
            del model, payload, predictions
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    full_by_seed = pd.read_csv(MATCHED_BY_SEED_PATH)
    summary_rows = summarize(by_seed, full_by_seed)
    sensitivity_rows = matched.summarize_sensitivity(sensitivity_seed_rows)
    validation = validate_outputs(
        reconstructed, by_seed, summary_rows, by_source, sensitivity_rows
    )
    results_text = build_results_markdown(
        origin_rows, summary_rows, sensitivity_rows
    )
    log("evaluation_completed", validation=validation)

    atomic_write_csv(OUTPUTS["origin_summary"], origin_rows)
    atomic_write_csv(OUTPUTS["by_seed"], by_seed)
    atomic_write_csv(OUTPUTS["summary"], summary_rows)
    atomic_write_csv(OUTPUTS["by_source"], by_source)
    atomic_write_csv(OUTPUTS["sensitivity"], sensitivity_rows)
    atomic_write_text(OUTPUTS["results"], results_text)
    atomic_write_text(
        OUTPUTS["log"],
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
    )
    elapsed = time.monotonic() - started
    output_entries = [
        file_entry(OUTPUTS[key])
        for key in (
            "origin_summary",
            "by_seed",
            "summary",
            "by_source",
            "sensitivity",
            "log",
            "results",
        )
    ]
    final_manifest = {
        "experiment": VERSION,
        "version": VERSION,
        "analysis_type": (
            "post-review evaluation-only raw-frame-disjoint window sensitivity"
        ),
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "runtime_seconds": elapsed,
        "analysis_plan": plan,
        "implementation": implementation,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "inputs": inputs + checkpoints + [full_output_entry],
        "matched_update_lineage": {
            "contract_id": training["contract_id"],
            "training_manifest": file_entry(TRAINING_MANIFEST_PATH),
            "final_manifest": file_entry(MATCHED_MANIFEST_PATH),
            "arms": list(ARMS),
            "paired_pipeline_seeds": list(SEEDS),
            "selected_checkpoint_count": len(checkpoints),
        },
        "origin_reconstruction": reconstruction,
        "subset_rule": {
            "window_size_frames": WINDOW_SIZE,
            "retain_if": (
                "start_index % 128 == 0 and "
                "end_index - start_index + 1 == 128"
            ),
            "alternate_offsets_searched": False,
            "labels_or_predictions_used_for_selection": False,
        },
        "validation_gates": validation,
        "outputs": output_entries,
        "warnings": [
            "The train/validation/test raw-range feasibility check was observed before the plan and is post-hoc.",
            "Generated NPZ files omit base indices; origins are reconstructed from frozen generator seeds and source-test order.",
            "The retained windows are raw-frame-disjoint within each evaluation, not independent captures or vehicles.",
            "The five repeats are paired pipeline seeds, not data-level or population replication.",
            "Full-set matched-update v1 remains primary; this is a reviewer-requested sensitivity.",
        ],
    }
    atomic_write_json(FINAL_MANIFEST_PATH, final_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": relative(FINAL_MANIFEST_PATH),
                "manifest_sha256": sha256_file(FINAL_MANIFEST_PATH),
                "runtime_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "evaluate"))
    parser.add_argument(
        "--allow-uncommitted-code",
        action="store_true",
        help="Only valid for preflight before committing the evaluator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_uncommitted_code and args.phase != "preflight":
        raise SystemExit("--allow-uncommitted-code is valid only for preflight")
    if args.phase == "preflight":
        run_preflight(allow_uncommitted=args.allow_uncommitted_code)
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
