#!/usr/bin/env python3
"""E8: evaluation-realization sensitivity for controlled CAN variants.

This audit re-instantiates L1, L2, and L4 on three mutually frame-disjoint
held-out-normal blocks.  The legacy default evaluates the preregistered frozen
WISA CNNs.  A journal-checkpoint rerun is opt-in and requires its own
pre-existing run record; it never trains or selects a model and refuses to
overwrite any output.

Prospective design:
  journal/experiments/e8_evaluation_realization/PREREG.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import t as student_t
from sklearn.metrics import f1_score

import lib_common as lc
from strict_v2_evaluation import (
    bundle_provenance,
    cnn_budget_mode_from_tag,
    evaluation_model_input_provenance,
    evaluator_source_and_input_provenance,
    validate_strict_v2_evaluation_scope,
)


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = JOURNAL_ROOT.parent
WISA_ROOT = REPO_ROOT / "wisa"
WINDOWS = JOURNAL_ROOT / "datasets" / "windows"
TABLES = JOURNAL_ROOT / "results" / "tables"
LOGS = JOURNAL_ROOT / "results" / "logs"
EXP = JOURNAL_ROOT / "experiments" / "e8_evaluation_realization"
JOURNAL_MODELS = JOURNAL_ROOT / "models" / "generator_extension"

DETECTOR_SEEDS = [7, 42, 123, 2026, 3407]
EVAL_SEEDS = [20260711, 20260712, 20260713]
MASTER_PARTITION_SEED = 20260710
BLOCK_SIZE = 12_000
SETTINGS = ["real_only", "rule_0p30", "rule_1p00"]
RUNGS = ["L1_fixed_variant", "L2_sensitivity", "L4_grammar_exit"]
CONTRASTS = [
    ("rule_0p30_minus_real_only", "rule_0p30", "real_only"),
    ("rule_1p00_minus_real_only", "rule_1p00", "real_only"),
]
METRICS = [
    "fpr",
    "normal_recall",
    "binary_attack_recall",
    "exact_attack_macro_recall",
]
CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}
DEFAULT_OUTPUT_TAG = "e8_v1"
MODEL_SOURCES = ["frozen-wisa", "journal"]
SAFE_MODEL_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
STRICT_V2_MARKERS = ("sampling_v2", "unique_pool_v2", "strict_v2", "strict-v2")


L2_SCENARIOS = [
    {"attack_type": "DoS", "severity": "low", "burst_len": 24, "step": 3, "amplitude": 0.25},
    {"attack_type": "DoS", "severity": "medium", "burst_len": 48, "step": 2, "amplitude": 0.50},
    {"attack_type": "DoS", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
    {"attack_type": "Fuzzy", "severity": "low", "burst_len": 24, "step": 4, "amplitude": 0.25},
    {"attack_type": "Fuzzy", "severity": "medium", "burst_len": 48, "step": 3, "amplitude": 0.50},
    {"attack_type": "Fuzzy", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
    {"attack_type": "Gear", "severity": "low", "burst_len": 24, "step": 4, "amplitude": 0.25},
    {"attack_type": "Gear", "severity": "medium", "burst_len": 48, "step": 2, "amplitude": 0.50},
    {"attack_type": "Gear", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
    {"attack_type": "RPM", "severity": "low", "burst_len": 24, "step": 4, "amplitude": 0.25},
    {"attack_type": "RPM", "severity": "medium", "burst_len": 48, "step": 2, "amplitude": 0.50},
    {"attack_type": "RPM", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
]

L4_SCENARIOS = [
    {"scenario_id": "Gear_payload_shift_canonical", "attack_type": "Gear", "target_id": 0x43F, "id_condition": "canonical", "mode": "gear"},
    {"scenario_id": "Gear_payload_shifted_0x440", "attack_type": "Gear", "target_id": 0x440, "id_condition": "shifted", "mode": "gear"},
    {"scenario_id": "RPM_payload_shift_canonical", "attack_type": "RPM", "target_id": 0x316, "id_condition": "canonical", "mode": "rpm"},
    {"scenario_id": "RPM_payload_shifted_0x329", "attack_type": "RPM", "target_id": 0x329, "id_condition": "shifted", "mode": "rpm"},
]


class CNN1D(nn.Module):
    """Architecture used by the frozen WISA CNN checkpoints."""

    def __init__(self, in_channels: int = 11, classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_publish_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Publish a CSV atomically and never replace a prior artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary output: {temporary}")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_publish_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary output: {temporary}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def frame_disjoint_indices(
    y_binary: np.ndarray,
    source_file: np.ndarray,
    segment_id: np.ndarray,
    start_index: np.ndarray,
    end_index: np.ndarray,
) -> np.ndarray:
    """Greedily retain non-overlapping normal frame intervals per capture."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(np.asarray(y_binary) == 0):
        groups[(str(source_file[index]), int(segment_id[index]))].append(int(index))
    retained: list[int] = []
    for key in sorted(groups):
        indices = sorted(
            groups[key],
            key=lambda i: (int(start_index[i]), int(end_index[i]), i),
        )
        last_end = -1
        for index in indices:
            start = int(start_index[index])
            end = int(end_index[index])
            if end < start:
                raise ValueError(f"invalid frame interval at test index {index}: {start}>{end}")
            if start > last_end:
                retained.append(index)
                last_end = end
    return np.asarray(retained, dtype=np.int64)


def assert_frame_disjoint(
    indices: Iterable[int],
    source_file: np.ndarray,
    segment_id: np.ndarray,
    start_index: np.ndarray,
    end_index: np.ndarray,
) -> None:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    indices = [int(i) for i in indices]
    if len(indices) != len(set(indices)):
        raise AssertionError("duplicate base-window indices across evaluation blocks")
    for index in indices:
        groups[(str(source_file[index]), int(segment_id[index]))].append(index)
    for key, members in groups.items():
        members.sort(key=lambda i: (int(start_index[i]), int(end_index[i]), i))
        previous_end = -1
        for index in members:
            start = int(start_index[index])
            if start <= previous_end:
                raise AssertionError(
                    f"frame overlap in {key}: start={start}, previous_end={previous_end}"
                )
            previous_end = int(end_index[index])


def partition_blocks(
    eligible_indices: np.ndarray,
    block_size: int = BLOCK_SIZE,
    eval_seeds: list[int] | None = None,
    master_seed: int = MASTER_PARTITION_SEED,
) -> dict[str, np.ndarray]:
    eval_seeds = EVAL_SEEDS if eval_seeds is None else eval_seeds
    required = block_size * len(eval_seeds)
    if len(eligible_indices) < required:
        raise ValueError(
            f"not enough frame-disjoint normal windows: need {required}, got {len(eligible_indices)}"
        )
    shuffled = np.random.default_rng(master_seed).permutation(np.asarray(eligible_indices))
    return {
        f"block_{position + 1:02d}": shuffled[position * block_size : (position + 1) * block_size]
        for position in range(len(eval_seeds))
    }


def choose_positions(rng: np.random.Generator, burst_len: int, step: int) -> np.ndarray:
    start = int(rng.integers(0, 128 - burst_len + 1))
    return np.arange(start, start + burst_len, step)


def inject_l1_dos(rng: np.random.Generator, x: np.ndarray) -> int:
    burst_len = int(rng.integers(48, 97))
    start = int(rng.integers(0, 128 - burst_len + 1))
    positions = np.arange(start, start + burst_len, int(rng.choice([1, 2])))
    x[positions, 0] = 0x000
    x[positions, 1] = 8
    x[positions, 2:10] = 0
    x[positions, 10] = np.maximum(x[positions, 10] * 0.25, 1e-5)
    return len(positions)


def inject_l1_fuzzy(rng: np.random.Generator, x: np.ndarray) -> int:
    burst_len = int(rng.integers(40, 97))
    start = int(rng.integers(0, 128 - burst_len + 1))
    positions = np.arange(start, start + burst_len, int(rng.choice([1, 2, 3])))
    x[positions, 0] = rng.integers(0, 2048, size=len(positions))
    x[positions, 1] = 8
    x[positions, 2:10] = rng.integers(0, 256, size=(len(positions), 8))
    x[positions, 10] = rng.uniform(0.00005, 0.015, size=len(positions))
    return len(positions)


def inject_l1_spoof(
    rng: np.random.Generator, x: np.ndarray, can_id: int, mode: str
) -> int:
    burst_len = int(rng.integers(48, 97))
    start = int(rng.integers(0, 128 - burst_len + 1))
    positions = np.arange(start, start + burst_len, 2)
    x[positions, 0] = can_id
    x[positions, 1] = 8
    ramp = np.linspace(0, 255, len(positions), dtype=np.float32)
    if mode == "gear":
        x[positions, 2] = ramp
        x[positions, 3] = 255 - ramp
        x[positions, 4] = np.clip(
            x[positions, 4] + rng.integers(-20, 21, size=len(positions)), 0, 255
        )
    else:
        wave = (
            127.5 + 127.5 * np.sin(np.linspace(0, 2 * np.pi, len(positions)))
        ).astype(np.float32)
        x[positions, 2] = wave
        x[positions, 3] = ramp
        x[positions, 5] = np.clip(
            x[positions, 5] + rng.integers(-30, 31, size=len(positions)), 0, 255
        )
    return len(positions)


def inject_l2(
    rng: np.random.Generator, x: np.ndarray, scenario: dict
) -> int:
    attack_type = scenario["attack_type"]
    burst_len = int(scenario["burst_len"])
    step = int(scenario["step"])
    amplitude = float(scenario["amplitude"])
    positions = choose_positions(rng, burst_len, step)
    if attack_type == "DoS":
        x[positions, 0] = 0x000
        x[positions, 1] = 8
        x[positions, 2:10] = 0
        x[positions, 10] = np.maximum(
            x[positions, 10] * (0.4 - 0.25 * amplitude), 1e-5
        )
    elif attack_type == "Fuzzy":
        x[positions, 0] = rng.integers(0, 2048, size=len(positions))
        x[positions, 1] = 8
        base = x[positions, 2:10].copy()
        random_payload = rng.integers(0, 256, size=(len(positions), 8))
        x[positions, 2:10] = np.clip(
            (1.0 - amplitude) * base + amplitude * random_payload, 0, 255
        )
        x[positions, 10] = rng.uniform(
            0.00005, 0.005 + 0.01 * amplitude, size=len(positions)
        )
    elif attack_type in {"Gear", "RPM"}:
        x[positions, 0] = 0x43F if attack_type == "Gear" else 0x316
        x[positions, 1] = 8
        scale = 255.0 * amplitude
        ramp = np.linspace(0, scale, len(positions), dtype=np.float32)
        base = x[positions, 2:10].copy()
        if attack_type == "Gear":
            base[:, 2] = np.clip(base[:, 2] * (1.0 - amplitude) + ramp, 0, 255)
            base[:, 3] = np.clip(
                base[:, 3] * (1.0 - amplitude) + (255 - ramp), 0, 255
            )
            base[:, 4] = np.clip(
                base[:, 4] + rng.integers(-10, 11, size=len(positions)) * amplitude,
                0,
                255,
            )
        else:
            wave = (
                127.5 + 127.5 * np.sin(np.linspace(0, 2 * np.pi, len(positions)))
            ).astype(np.float32) * amplitude
            base[:, 2] = np.clip(base[:, 2] * (1.0 - amplitude) + wave, 0, 255)
            base[:, 3] = np.clip(base[:, 3] * (1.0 - amplitude) + ramp, 0, 255)
            base[:, 5] = np.clip(
                base[:, 5] + rng.integers(-15, 16, size=len(positions)) * amplitude,
                0,
                255,
            )
        x[positions, 2:10] = base
        x[positions, 10] = np.maximum(x[positions, 10], 1e-5)
    else:
        raise ValueError(f"unknown L2 attack type: {attack_type}")
    return len(positions)


def inject_l4(
    rng: np.random.Generator, x: np.ndarray, target_id: int, mode: str
) -> int:
    positions = choose_positions(rng, 96, 3)
    x[positions, 0] = target_id
    x[positions, 1] = 8
    base = x[positions, 2:10].copy()
    ramp = np.linspace(0, 255, len(positions), dtype=np.float32)
    if mode == "gear":
        base[:, 6] = ramp
        base[:, 7] = np.clip(
            255 - ramp + rng.integers(-10, 11, size=len(positions)), 0, 255
        )
    else:
        wave = (
            127.5 + 127.5 * np.sin(np.linspace(0, 3 * np.pi, len(positions)))
        ).astype(np.float32)
        base[:, 4] = (np.arange(len(positions), dtype=np.float32) * 37) % 256
        base[:, 6] = wave
        base[:, 7] = np.clip(
            ramp + rng.integers(-12, 13, size=len(positions)), 0, 255
        )
    x[positions, 2:10] = base
    x[positions, 10] = np.maximum(x[positions, 10], 1e-5)
    return len(positions)


def generate_rung(
    x_test: np.ndarray,
    base_indices: np.ndarray,
    rung: str,
    eval_seed: int,
) -> dict[str, np.ndarray]:
    """Create one prospective controlled-variant realization in memory."""
    if len(base_indices) != BLOCK_SIZE:
        raise ValueError(f"expected {BLOCK_SIZE} base windows, got {len(base_indices)}")
    rng = np.random.default_rng(eval_seed)
    assigned = rng.permutation(np.asarray(base_indices))
    x = x_test[assigned].astype(np.float32, copy=True)
    y = np.zeros(BLOCK_SIZE, dtype=np.int8)
    scenario = np.full(BLOCK_SIZE, "Normal_normal", dtype=object)
    attack_type = np.full(BLOCK_SIZE, "Normal", dtype=object)
    detail = np.full(BLOCK_SIZE, "normal", dtype=object)
    injection_count = np.zeros(BLOCK_SIZE, dtype=np.int16)

    if rung == "L1_fixed_variant":
        cursor = 6_000
        specs = [
            ("DoS", 1, lambda z: inject_l1_dos(rng, z)),
            ("Fuzzy", 2, lambda z: inject_l1_fuzzy(rng, z)),
            ("Gear", 3, lambda z: inject_l1_spoof(rng, z, 0x43F, "gear")),
            ("RPM", 4, lambda z: inject_l1_spoof(rng, z, 0x316, "rpm")),
        ]
        for name, label, injector in specs:
            slc = slice(cursor, cursor + 1_500)
            y[slc] = label
            scenario[slc] = f"{name}_fixed"
            attack_type[slc] = name
            detail[slc] = "fixed"
            for index in range(slc.start, slc.stop):
                injection_count[index] = injector(x[index])
            cursor = slc.stop
    elif rung == "L2_sensitivity":
        cursor = 6_000
        for spec in L2_SCENARIOS:
            slc = slice(cursor, cursor + 500)
            name = str(spec["attack_type"])
            y[slc] = {"DoS": 1, "Fuzzy": 2, "Gear": 3, "RPM": 4}[name]
            scenario[slc] = f"{name}_{spec['severity']}"
            attack_type[slc] = name
            detail[slc] = str(spec["severity"])
            for index in range(slc.start, slc.stop):
                injection_count[index] = inject_l2(rng, x[index], spec)
            cursor = slc.stop
    elif rung == "L4_grammar_exit":
        cursor = 4_000
        for spec in L4_SCENARIOS:
            slc = slice(cursor, cursor + 2_000)
            name = str(spec["attack_type"])
            y[slc] = 3 if name == "Gear" else 4
            scenario[slc] = str(spec["scenario_id"])
            attack_type[slc] = name
            detail[slc] = str(spec["id_condition"])
            for index in range(slc.start, slc.stop):
                injection_count[index] = inject_l4(
                    rng, x[index], int(spec["target_id"]), str(spec["mode"])
                )
            cursor = slc.stop
    else:
        raise ValueError(f"unknown rung: {rung}")
    if cursor != BLOCK_SIZE:
        raise AssertionError(f"incomplete {rung} construction: cursor={cursor}")
    return {
        "x": x,
        "y_attack_type": y,
        "scenario_id": scenario,
        "attack_type": attack_type,
        "detail": detail,
        "injection_count": injection_count,
        "base_index": assigned,
    }


def parse_tagged_settings(
    model_tag: str, tagged_settings_value: str | None
) -> set[str]:
    if not model_tag:
        if tagged_settings_value:
            raise ValueError("--tagged-settings requires --model-tag")
        return set()
    if not SAFE_MODEL_TAG.fullmatch(model_tag):
        raise ValueError(
            "--model-tag must be a safe filename tag without path separators"
        )
    if tagged_settings_value:
        values = [value.strip() for value in tagged_settings_value.split(",")]
        if not values or any(not value for value in values):
            raise ValueError(
                "--tagged-settings must contain no empty setting"
            )
        if len(values) != len(set(values)):
            raise ValueError("--tagged-settings must not contain duplicates")
        tagged = set(values)
    else:
        tagged = set(SETTINGS)
    unknown = sorted(tagged - set(SETTINGS))
    if unknown:
        raise ValueError(f"unknown tagged setting(s): {unknown}")
    return tagged


def validate_model_selection(
    model_source: str,
    model_tag: str,
    tagged_settings: set[str],
    output_tag: str,
) -> None:
    strict_label = any(marker in output_tag for marker in STRICT_V2_MARKERS)
    if model_source == "frozen-wisa":
        if model_tag or tagged_settings:
            raise ValueError(
                "tagged checkpoints require --model-source journal"
            )
        if strict_label:
            raise ValueError(
                "strict-v2-labelled output cannot resolve to frozen WISA models"
            )
        return
    if model_source != "journal":
        raise ValueError(f"unknown model source: {model_source}")
    if not model_tag:
        if strict_label:
            raise ValueError(
                "strict-v2-labelled output requires an explicit --model-tag"
            )
        return
    if strict_label and not lc.is_strict_v2_tag(model_tag):
        raise ValueError(
            "strict-v2-labelled output requires a strict-v2 --model-tag"
        )
    if not tagged_settings:
        raise ValueError("--model-tag must select at least one tagged setting")
    if output_tag == DEFAULT_OUTPUT_TAG or model_tag not in output_tag:
        raise ValueError(
            "versioned checkpoints require an explicit --output-tag containing "
            "the --model-tag"
        )


def resolve_run_record(
    model_source: str, journal_run_record: str | None
) -> Path:
    """Resolve the declaration governing the selected model-input family."""
    legacy = (EXP / "PREREG.md").resolve()
    if model_source == "frozen-wisa":
        if journal_run_record:
            raise ValueError(
                "--journal-run-record is only valid with --model-source journal"
            )
        return legacy
    if not journal_run_record:
        raise ValueError(
            "--model-source journal requires a pre-existing "
            "--journal-run-record"
        )
    candidate = Path(journal_run_record)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    experiments_root = (JOURNAL_ROOT / "experiments").resolve()
    try:
        candidate.relative_to(experiments_root)
    except ValueError as exc:
        raise ValueError(
            "--journal-run-record must be under journal/experiments"
        ) from exc
    if candidate == legacy:
        raise ValueError(
            "journal checkpoints cannot reuse the frozen-WISA E8 preregistration"
        )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"journal run record does not exist: {candidate}"
        )
    return candidate


def effective_model_tag(
    setting: str,
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> str:
    if not model_tag:
        return ""
    if tagged_settings is not None and setting not in tagged_settings:
        return ""
    return model_tag


def model_paths(
    setting: str,
    detector_seed: int,
    model_source: str = "frozen-wisa",
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> tuple[Path, Path]:
    if setting not in SETTINGS:
        raise ValueError(f"undeclared setting: {setting}")
    if detector_seed not in DETECTOR_SEEDS:
        raise ValueError(f"undeclared detector seed: {detector_seed}")
    if model_source == "journal":
        tag = effective_model_tag(setting, model_tag, tagged_settings)
        suffix = f"_{tag}" if tag else ""
        return (
            JOURNAL_MODELS / f"cnn_{setting}{suffix}_seed{detector_seed}.pt",
            WINDOWS / "train_windows.npz",
        )
    if model_source != "frozen-wisa":
        raise ValueError(f"unknown model source: {model_source}")
    if model_tag:
        raise ValueError("frozen WISA model resolution does not accept a model tag")
    models = WISA_ROOT / "models"
    if detector_seed in {7, 42, 123}:
        if setting == "real_only":
            return (
                models / "baseline" / f"cnn1d_real_only_seed{detector_seed}.pt",
                models / "baseline" / "cnn_standardizer.npz",
            )
        ratio = {"rule_0p30": "0p30", "rule_1p00": "1p00"}[setting]
        slug = f"rule_ratio{ratio}_seed{detector_seed}"
        return (
            models / "ratio_sweep" / f"cnn1d_{slug}.pt",
            models / "ratio_sweep" / f"standardizer_{slug}.npz",
        )
    if detector_seed in {2026, 3407}:
        if setting == "real_only":
            slug = f"real_only_seed{detector_seed}"
        else:
            ratio = {"rule_0p30": "0p30", "rule_1p00": "1p00"}[setting]
            slug = f"rule_ratio{ratio}_seed{detector_seed}"
        return (
            models / "main_seed_extension" / f"cnn1d_{slug}.pt",
            models / "main_seed_extension" / f"standardizer_{slug}.npz",
        )
    raise AssertionError("unreachable detector-seed branch")


@lru_cache(maxsize=None)
def source_train_standardizer(path_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the journal CNN scaler from its immutable source-train cache."""
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as values:
        if "x" not in values.files:
            raise KeyError(f"standardizer source lacks x: {path}")
        x = values["x"]
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def preflight_model_inputs(
    model_source: str,
    model_tag: str,
    tagged_settings: set[str],
) -> list[dict]:
    """Require the full declared checkpoint matrix before creating a log."""
    missing: list[Path] = []
    checked: set[Path] = set()
    declared: list[tuple[str, int, Path]] = []
    for setting in SETTINGS:
        for detector_seed in DETECTOR_SEEDS:
            checkpoint, standardizer = model_paths(
                setting,
                detector_seed,
                model_source,
                model_tag,
                tagged_settings,
            )
            declared.append((setting, detector_seed, checkpoint))
            for path in (checkpoint, standardizer):
                if path in checked:
                    continue
                checked.add(path)
                if not path.is_file():
                    missing.append(path)
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"missing {len(missing)} declared model input(s): {preview}{suffix}"
        )
    provenance = []
    if model_source != "journal":
        return provenance
    for setting, detector_seed, checkpoint in declared:
        effective_tag = effective_model_tag(
            setting, model_tag, tagged_settings
        )
        if not lc.is_strict_v2_tag(effective_tag):
            continue
        training_log = (
            JOURNAL_ROOT
            / "results"
            / "logs"
            / f"train_cnn_{setting}_{effective_tag}_seed{detector_seed}.log"
        )
        artifacts = {"checkpoint": checkpoint}
        record = lc.validate_strict_v2_training_bundle(
            training_log,
            artifacts,
            {
                "family": "cnn",
                "setting": setting,
                "seed": detector_seed,
                "model_tag": effective_tag,
                "sampling_policy": "strict-v2",
                "budget_mode": cnn_budget_mode_from_tag(effective_tag),
            },
        )
        provenance.append(bundle_provenance(
            family="cnn",
            setting=setting,
            seed=detector_seed,
            tag=effective_tag,
            log_path=training_log,
            artifacts=artifacts,
            validated_record=record,
        ))
    return provenance


def load_model(
    setting: str,
    detector_seed: int,
    device: torch.device,
    model_source: str = "frozen-wisa",
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> tuple[CNN1D, np.ndarray, np.ndarray, Path, Path]:
    checkpoint, standardizer = model_paths(
        setting,
        detector_seed,
        model_source,
        model_tag,
        tagged_settings,
    )
    for path in [checkpoint, standardizer]:
        if not path.exists():
            label = (
                "frozen input"
                if model_source == "frozen-wisa"
                else "declared model input"
            )
            raise FileNotFoundError(f"missing {label}: {path}")
    model = CNN1D().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    if model_source == "frozen-wisa":
        values = np.load(standardizer)
        mean, std = values["mean"], values["std"]
    else:
        mean, std = source_train_standardizer(str(standardizer.resolve()))
    return model, mean, std, checkpoint, standardizer


def predict_classes(
    model: CNN1D,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    standardized = ((x - mean) / std).astype(np.float32)
    predictions = np.empty(len(standardized), dtype=np.int8)
    with torch.inference_mode():
        for start in range(0, len(standardized), batch_size):
            batch_array = standardized[start : start + batch_size].transpose(0, 2, 1).copy()
            batch = torch.from_numpy(batch_array).to(device)
            pred = model(batch).argmax(dim=1).cpu().numpy().astype(np.int8)
            predictions[start : start + len(pred)] = pred
    return predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    binary_true = y_true > 0
    binary_pred = y_pred > 0
    normal = ~binary_true
    attack = binary_true
    tn = int((normal & ~binary_pred).sum())
    fp = int((normal & binary_pred).sum())
    fn = int((attack & ~binary_pred).sum())
    tp = int((attack & binary_pred).sum())
    exact_recalls = []
    row: dict[str, float | int] = {
        "windows": int(len(y_true)),
        "normal_windows": int(normal.sum()),
        "attack_windows": int(attack.sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "normal_recall": float(tn / (fp + tn)) if (fp + tn) else np.nan,
        "binary_attack_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "binary_macro_f1": float(
            f1_score(binary_true.astype(np.int8), binary_pred.astype(np.int8), average="macro", zero_division=0)
        ),
        "exact_attack_micro_recall": float((y_pred[attack] == y_true[attack]).mean()) if attack.any() else np.nan,
    }
    for label in [1, 2, 3, 4]:
        mask = y_true == label
        value = float((y_pred[mask] == label).mean()) if mask.any() else np.nan
        row[f"exact_recall_{CLASS_NAMES[label]}"] = value
        if mask.any():
            exact_recalls.append(value)
    row["exact_attack_macro_recall"] = float(np.mean(exact_recalls)) if exact_recalls else np.nan
    return row


def scenario_metric_rows(
    generated: dict[str, np.ndarray], y_pred: np.ndarray
) -> list[dict]:
    y_true = generated["y_attack_type"]
    scenarios = generated["scenario_id"].astype(str)
    rows = []
    for scenario_id in sorted(np.unique(scenarios)):
        mask = scenarios == scenario_id
        true = y_true[mask]
        pred = y_pred[mask]
        is_attack = true > 0
        rows.append(
            {
                "scenario_id": scenario_id,
                "attack_type": str(generated["attack_type"][mask][0]),
                "detail": str(generated["detail"][mask][0]),
                "windows": int(mask.sum()),
                "injection_count_mean": float(generated["injection_count"][mask].mean()),
                "fpr": float((pred > 0).mean()) if not is_attack.any() else np.nan,
                "normal_recall": float((pred == 0).mean()) if not is_attack.any() else np.nan,
                "binary_attack_recall": float((pred > 0).mean()) if is_attack.any() else np.nan,
                "exact_class_recall": float((pred == true).mean()) if is_attack.any() else np.nan,
            }
        )
    return rows


def t_summary(values: Iterable[float], inferential: bool) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n_independent_units": 0, "mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "min": np.nan, "max": np.nan}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if inferential and len(array) > 1:
        half = float(student_t.ppf(0.975, len(array) - 1) * std / np.sqrt(len(array)))
        low, high = mean - half, mean + half
    else:
        low = high = np.nan
    return {
        "n_independent_units": int(len(array)),
        "mean": mean,
        "std": std,
        "ci_low": low,
        "ci_high": high,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def nested_arm_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate crossed cells without counting them as independent n=15."""
    rows = []
    for (setting, rung), group in cells.groupby(["setting", "rung"], sort=True):
        for metric in METRICS:
            detector_values = group.groupby("detector_seed", sort=True)[metric].mean()
            row = {
                "setting": setting,
                "rung": rung,
                "metric": metric,
                "analysis_level": "detector_seed_after_block_mean",
                "n_detector_seeds": int(group["detector_seed"].nunique()),
                "n_eval_blocks": int(group["block_id"].nunique()),
                "unit_values": ";".join(f"{index}:{value:.10g}" for index, value in detector_values.items()),
            }
            row.update(t_summary(detector_values, inferential=True))
            rows.append(row)
            block_values = group.groupby("block_id", sort=True)[metric].mean()
            row = {
                "setting": setting,
                "rung": rung,
                "metric": metric,
                "analysis_level": "evaluation_block_after_detector_mean_descriptive",
                "n_detector_seeds": int(group["detector_seed"].nunique()),
                "n_eval_blocks": int(group["block_id"].nunique()),
                "unit_values": ";".join(f"{index}:{value:.10g}" for index, value in block_values.items()),
            }
            row.update(t_summary(block_values, inferential=False))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = cells.set_index(
        ["setting", "rung", "detector_seed", "block_id"]
    ).sort_index()
    paired_rows = []
    for contrast, treatment, reference in CONTRASTS:
        for rung in RUNGS:
            treatment_rows = indexed.loc[(treatment, rung)]
            reference_rows = indexed.loc[(reference, rung)]
            if not treatment_rows.index.equals(reference_rows.index):
                treatment_rows, reference_rows = treatment_rows.align(reference_rows, join="inner")
            expected = len(DETECTOR_SEEDS) * len(EVAL_SEEDS)
            if len(treatment_rows) != expected:
                raise AssertionError(
                    f"incomplete paired cells for {contrast}/{rung}: {len(treatment_rows)} != {expected}"
                )
            for (detector_seed, block_id), treatment_row in treatment_rows.iterrows():
                reference_row = reference_rows.loc[(detector_seed, block_id)]
                row = {
                    "contrast": contrast,
                    "treatment": treatment,
                    "reference": reference,
                    "rung": rung,
                    "detector_seed": int(detector_seed),
                    "block_id": str(block_id),
                }
                for metric in METRICS:
                    row[f"delta_{metric}"] = float(treatment_row[metric] - reference_row[metric])
                paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    summary_rows = []
    for (contrast, treatment, reference, rung), group in paired.groupby(
        ["contrast", "treatment", "reference", "rung"], sort=True
    ):
        for metric in METRICS:
            delta = f"delta_{metric}"
            detector_values = group.groupby("detector_seed", sort=True)[delta].mean()
            row = {
                "contrast": contrast,
                "treatment": treatment,
                "reference": reference,
                "rung": rung,
                "metric": metric,
                "analysis_level": "paired_detector_seed_after_block_mean",
                "n_detector_seeds": int(group["detector_seed"].nunique()),
                "n_eval_blocks": int(group["block_id"].nunique()),
                "unit_values": ";".join(f"{index}:{value:.10g}" for index, value in detector_values.items()),
            }
            row.update(t_summary(detector_values, inferential=True))
            summary_rows.append(row)
            block_values = group.groupby("block_id", sort=True)[delta].mean()
            row = {
                "contrast": contrast,
                "treatment": treatment,
                "reference": reference,
                "rung": rung,
                "metric": metric,
                "analysis_level": "paired_evaluation_block_after_detector_mean_descriptive",
                "n_detector_seeds": int(group["detector_seed"].nunique()),
                "n_eval_blocks": int(group["block_id"].nunique()),
                "unit_values": ";".join(f"{index}:{value:.10g}" for index, value in block_values.items()),
            }
            row.update(t_summary(block_values, inferential=False))
            summary_rows.append(row)
    return paired, pd.DataFrame(summary_rows)


def output_paths(tag: str) -> dict[str, Path]:
    return {
        "blocks": TABLES / f"evaluation_realization_blocks_{tag}.csv",
        "cells": TABLES / f"evaluation_realization_by_cell_{tag}.csv",
        "scenarios": TABLES / f"evaluation_realization_by_scenario_{tag}.csv",
        "arm_summary": TABLES / f"evaluation_realization_nested_summary_{tag}.csv",
        "paired": TABLES / f"evaluation_realization_paired_by_cell_{tag}.csv",
        "paired_summary": TABLES / f"evaluation_realization_paired_summary_{tag}.csv",
        "record": EXP / f"run_{tag}.json",
        "log": LOGS / f"evaluation_realization_{tag}.log",
    }


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"evaluation_realization.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="x")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument(
        "--model-source",
        choices=MODEL_SOURCES,
        default="frozen-wisa",
        help=(
            "frozen-wisa preserves the preregistered legacy inputs; journal "
            "selects journal CNN checkpoints without fallback"
        ),
    )
    parser.add_argument(
        "--model-tag",
        default="",
        help=(
            "journal checkpoint suffix without leading '_' "
            "(for example sampling_v2)"
        ),
    )
    parser.add_argument(
        "--tagged-settings",
        default=None,
        help=(
            "comma-separated settings that use --model-tag; defaults to all "
            "three settings"
        ),
    )
    parser.add_argument(
        "--journal-run-record",
        default=None,
        help=(
            "pre-existing declaration under journal/experiments; required for "
            "--model-source journal and may not be the frozen-WISA E8 PREREG"
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="validate prospective block construction without writing outputs or evaluating models",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag):
        parser.error("--output-tag may contain only letters, digits, underscore, and hyphen")
    try:
        lc.validate_artifact_tag(args.model_tag, name="--model-tag")
        tagged_settings = parse_tagged_settings(
            args.model_tag, args.tagged_settings
        )
        validate_model_selection(
            args.model_source,
            args.model_tag,
            tagged_settings,
            args.output_tag,
        )
        validate_strict_v2_evaluation_scope(
            args.model_tag, args.output_tag, DETECTOR_SEEDS
        )
        prereg_path = resolve_run_record(
            args.model_source, args.journal_run_record
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")

    paths = output_paths(args.output_tag)
    if not args.integrity_only:
        collisions = [str(path) for path in paths.values() if path.exists()]
        if collisions:
            parser.error("refusing to overwrite existing output(s): " + ", ".join(collisions))
        try:
            strict_v2_bundles = preflight_model_inputs(
                args.model_source,
                args.model_tag,
                tagged_settings,
            )
            strict_v2_evaluator = None
            strict_v2_model_inputs = None
            if strict_v2_bundles:
                strict_v2_evaluator = (
                    evaluator_source_and_input_provenance(
                        __file__,
                        ["train_windows", "test_windows"],
                        training_bundles=strict_v2_bundles,
                    )
                )
                model_artifacts = {
                    (
                        f"cnn/{setting}/seed{detector_seed}/checkpoint"
                    ): model_paths(
                        setting,
                        detector_seed,
                        args.model_source,
                        args.model_tag,
                        tagged_settings,
                    )[0]
                    for setting in SETTINGS
                    for detector_seed in DETECTOR_SEEDS
                }
                model_artifacts["cnn/shared/source_train_standardizer"] = (
                    WINDOWS / "train_windows.npz"
                )
                strict_v2_model_inputs = (
                    evaluation_model_input_provenance(model_artifacts)
                )
        except (FileNotFoundError, RuntimeError) as exc:
            parser.error(str(exc))
        logger = setup_logger(paths["log"])
    else:
        logger = logging.getLogger("evaluation_realization.integrity")
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
        strict_v2_bundles = []
        strict_v2_evaluator = None
        strict_v2_model_inputs = None

    torch.set_num_threads(args.threads)
    test_path = WINDOWS / "test_windows.npz"
    if not prereg_path.exists():
        raise FileNotFoundError(f"prospective record missing: {prereg_path}")
    test = np.load(test_path, allow_pickle=True)
    required_keys = ["x", "y_binary", "source_file", "segment_id", "start_index", "end_index"]
    missing = [key for key in required_keys if key not in test.files]
    if missing:
        raise KeyError(f"test window metadata missing: {missing}")
    # NpzFile indexing decompresses an array on every access. Cache each
    # metadata array once before the 36,000-row manifest loop.
    y_binary = test["y_binary"]
    source_file = test["source_file"]
    segment_id = test["segment_id"]
    start_index = test["start_index"]
    end_index = test["end_index"]
    eligible = frame_disjoint_indices(
        y_binary, source_file, segment_id, start_index, end_index
    )
    blocks = partition_blocks(eligible)
    selected = np.concatenate(list(blocks.values()))
    assert_frame_disjoint(selected, source_file, segment_id, start_index, end_index)
    logger.info(
        "block integrity passed: eligible=%d selected=%d blocks=%d block_size=%d",
        len(eligible), len(selected), len(blocks), BLOCK_SIZE,
    )
    if args.integrity_only:
        return

    eval_seed_by_block = dict(zip(blocks, EVAL_SEEDS))
    block_rows = []
    for block_id, indices in blocks.items():
        for position, index in enumerate(indices):
            block_rows.append(
                {
                    "block_id": block_id,
                    "eval_seed": eval_seed_by_block[block_id],
                    "block_position": position,
                    "test_window_index": int(index),
                    "source_file": str(source_file[index]),
                    "segment_id": int(segment_id[index]),
                    "start_index": int(start_index[index]),
                    "end_index": int(end_index[index]),
                }
            )

    device = torch.device(args.device)
    x_test = test["x"].astype(np.float32, copy=False)
    cell_rows: list[dict] = []
    scenario_rows: list[dict] = []
    input_records: dict[str, dict] = {}
    for block_id, indices in blocks.items():
        eval_seed = eval_seed_by_block[block_id]
        for rung in RUNGS:
            logger.info("generate block=%s eval_seed=%d rung=%s", block_id, eval_seed, rung)
            generated = generate_rung(x_test, indices, rung, eval_seed)
            for setting in SETTINGS:
                for detector_seed in DETECTOR_SEEDS:
                    model, mean, std, checkpoint, standardizer = load_model(
                        setting,
                        detector_seed,
                        device,
                        args.model_source,
                        args.model_tag,
                        tagged_settings,
                    )
                    pred = predict_classes(
                        model, generated["x"], mean, std, device, args.batch_size
                    )
                    row = {
                        "setting": setting,
                        "detector_seed": detector_seed,
                        "block_id": block_id,
                        "eval_seed": eval_seed,
                        "rung": rung,
                    }
                    row.update(compute_metrics(generated["y_attack_type"], pred))
                    cell_rows.append(row)
                    for scenario_row in scenario_metric_rows(generated, pred):
                        scenario_rows.append(
                            {
                                "setting": setting,
                                "detector_seed": detector_seed,
                                "block_id": block_id,
                                "eval_seed": eval_seed,
                                "rung": rung,
                                **scenario_row,
                            }
                        )
                    key = f"{setting}|{detector_seed}"
                    if key not in input_records:
                        record = {
                            "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                            "checkpoint_sha256": sha256_file(checkpoint),
                        }
                        if args.model_source == "frozen-wisa":
                            record.update({
                                "standardizer": str(
                                    standardizer.relative_to(REPO_ROOT)
                                ),
                                "standardizer_sha256": sha256_file(standardizer),
                            })
                        else:
                            record.update({
                                "model_tag": effective_model_tag(
                                    setting,
                                    args.model_tag,
                                    tagged_settings,
                                ),
                                "standardizer_source": str(
                                    standardizer.relative_to(REPO_ROOT)
                                ),
                                "standardizer_source_sha256": lc.sha256_file(
                                    standardizer
                                ),
                                "standardizer_derivation": (
                                    "feature mean/std over axes (window,time); "
                                    "std<1e-6 replaced by 1.0"
                                ),
                            })
                        input_records[key] = record
                    del model
            logger.info("evaluated block=%s rung=%s cells=%d", block_id, rung, len(cell_rows))

    cells = pd.DataFrame(cell_rows)
    scenarios = pd.DataFrame(scenario_rows)
    expected_cells = len(SETTINGS) * len(DETECTOR_SEEDS) * len(EVAL_SEEDS) * len(RUNGS)
    if len(cells) != expected_cells or cells.duplicated(
        ["setting", "detector_seed", "block_id", "rung"]
    ).any():
        raise AssertionError(
            f"cell completeness failure: rows={len(cells)}, expected={expected_cells}"
        )
    arm_summary = nested_arm_summary(cells)
    paired, paired_summary = paired_tables(cells)
    record = {
        "status": "complete",
        "analysis_role": "evaluation-only robustness audit; no model selection",
        "prospective_record": str(prereg_path.relative_to(REPO_ROOT)),
        "prospective_record_sha256": sha256_file(prereg_path),
        "test_windows": str(test_path.relative_to(REPO_ROOT)),
        "test_windows_sha256": sha256_file(test_path),
        "master_partition_seed": MASTER_PARTITION_SEED,
        "evaluation_seeds": EVAL_SEEDS,
        "detector_seeds": DETECTOR_SEEDS,
        "settings": SETTINGS,
        "rungs": RUNGS,
        "eligible_frame_disjoint_normal_windows": int(len(eligible)),
        "selected_base_windows": int(len(selected)),
        "block_size": BLOCK_SIZE,
        "blocks_are_mutually_frame_disjoint": True,
        "same_blocks_reused_across_rungs_for_pairing": True,
        "crossed_cells_are_not_independent_replicates": True,
        "device": str(device),
        "batch_size": args.batch_size,
        "torch_threads": args.threads,
        "expected_cells": expected_cells,
        "observed_cells": int(len(cells)),
        "outputs": {name: str(path.relative_to(REPO_ROOT)) for name, path in paths.items()},
    }
    if args.model_source == "frozen-wisa":
        record["frozen_model_inputs"] = input_records
    else:
        record.update({
            "model_source": "journal",
            "model_tag": args.model_tag,
            "tagged_settings": sorted(tagged_settings),
            "untagged_settings": sorted(set(SETTINGS) - tagged_settings),
            "checkpoint_resolution": (
                "journal-only exact tagged/untagged paths; no WISA or "
                "missing-checkpoint fallback"
            ),
            "versioned_model_inputs": input_records,
        })
        if strict_v2_bundles:
            record["strict_v2_training_bundles"] = strict_v2_bundles
            record["strict_v2_evaluator_provenance"] = strict_v2_evaluator
            record["strict_v2_model_inputs"] = strict_v2_model_inputs
    atomic_publish_dataframe(paths["blocks"], pd.DataFrame(block_rows))
    atomic_publish_dataframe(paths["cells"], cells)
    atomic_publish_dataframe(paths["scenarios"], scenarios)
    atomic_publish_dataframe(paths["arm_summary"], arm_summary)
    atomic_publish_dataframe(paths["paired"], paired)
    atomic_publish_dataframe(paths["paired_summary"], paired_summary)
    atomic_publish_json(paths["record"], record)
    logger.info("complete: cells=%d outputs=%s", len(cells), args.output_tag)


if __name__ == "__main__":
    main()
