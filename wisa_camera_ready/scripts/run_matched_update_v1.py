#!/usr/bin/env python3
"""Run the WISA camera-ready matched optimizer-update sensitivity.

The command has deliberately separated training and evaluation phases:

* ``train`` may load only Car-Hacking train/validation and the frozen Rule pool.
* ``evaluate`` requires all 15 completed fits before it can load source test,
  fixed-variant, or sensitivity windows.

All writes are versioned under ``wisa_camera_ready``. The frozen ``wisa`` tree is
used only for hash-checked implementation/configuration provenance and for comparing
the real-train-fitted standardizer arrays.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, Sampler


VERSION = "matched_update_v1"
PLAN_COMMIT = "819686733ea87bf5a6cbdfb5f1c8402ddf46cf35"
PLAN_SHA256 = "5cbec7d60848e4ae17f8fc76edb14fc1d8993e9e45c35380ec3a5ccbb0dedd73"
ACCEPTED_SOURCE_COMMIT = "57880aa350ddf0f916b7e1bd2e940529d0fc6ea6"

CAMERA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMERA_ROOT.parent
WISA_ROOT = REPO_ROOT / "wisa"
DATA_ROOT = REPO_ROOT / "datasets"
PLAN_PATH = CAMERA_ROOT / "experiments" / "02_matched_update" / "PLAN.md"
EXP_ROOT = CAMERA_ROOT / "experiments" / "02_matched_update"
HISTORY_ROOT = EXP_ROOT / "histories"
RUN_ROOT = EXP_ROOT / "runs"
MODEL_ROOT = CAMERA_ROOT / "models" / VERSION
TABLE_ROOT = CAMERA_ROOT / "results" / "tables"
LOG_PATH = CAMERA_ROOT / "results" / "logs" / f"{VERSION}.log"
TRAINING_MANIFEST_PATH = EXP_ROOT / "training_manifest_v1.json"
FINAL_MANIFEST_PATH = EXP_ROOT / "manifest_v1.json"
STANDARDIZER_PATH = MODEL_ROOT / "standardizer_source_real_train_v1.npz"

OUTPUT_TABLES = {
    "checkpoint_selection": TABLE_ROOT
    / "matched_update_checkpoint_selection_v1.csv",
    "by_seed": TABLE_ROOT / "matched_update_by_seed_v1.csv",
    "summary": TABLE_ROOT / "matched_update_summary_v1.csv",
    "sensitivity": TABLE_ROOT
    / "matched_update_sensitivity_by_scenario_v1.csv",
}

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
FEATURE_NAMES = (
    "can_id",
    "dlc",
    "data0",
    "data1",
    "data2",
    "data3",
    "data4",
    "data5",
    "data6",
    "data7",
    "delta_t",
)

BATCH_SIZE = 512
INFERENCE_BATCH_SIZE = 1024
MAX_UPDATES = 7_992
VALIDATE_EVERY = 666
VALIDATION_UPDATES = tuple(range(VALIDATE_EVERY, MAX_UPDATES + 1, VALIDATE_EVERY))
ADDED_WINDOWS = 78_645
REAL_TRAIN_WINDOWS = 262_149
AUGMENTED_TRAIN_WINDOWS = 340_794
WINDOW_SIZE = 128
STRIDE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
TRAIN_WORKERS = 2
EVAL_WORKERS = 2
SAMPLER_SEED_OFFSET = 1_000_003
WORKER_SEED_OFFSET = 2_000_003

EXPECTED_FILES = {
    "datasets/windows/train_windows.npz": {
        "bytes": 70_962_600,
        "sha256": "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4",
    },
    "datasets/windows/val_windows.npz": {
        "bytes": 22_221_900,
        "sha256": "b4920e7f60fbf38b38e12976129f120ebff160a21a4c77093f3746f3c1aa218e",
    },
    "datasets/windows/test_windows.npz": {
        "bytes": 50_992_356,
        "sha256": "4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7",
    },
    "datasets/windows/variant_test_windows.npz": {
        "bytes": 60_445_693,
        "sha256": "91bd4d5dca2bd73907fad11bb94ea6dec075f60bb3b24e3c517617bf2d50b2ba",
    },
    "datasets/windows/variant_sensitivity_windows.npz": {
        "bytes": 29_966_923,
        "sha256": "4266fb3bbb3e248ef1ab53aaca19cb4534a3815a18c9134dc3db446cdfcf5674",
    },
    "datasets/synthetic/rule_based_windows.npz": {
        "bytes": 282_039_519,
        "sha256": "4078ab00b8531f760b296b5fd293e3df914bef996d317c958ead4762137bafd5",
    },
}

EXPECTED_DEPENDENCIES = {
    "wisa/scripts/train_real_only_baselines.py": (
        "5f4d423a07c573d69210034e752b2a847eb4b87486907cd8ea81aee06d9aef31"
    ),
    "wisa/scripts/train_rule_synthetic_ratio_sweep.py": (
        "846069660c8daf303587c11f05619c57c45d377153dbf8621aa076ed661e19dd"
    ),
    "wisa/scripts/train_oversampling_baseline.py": (
        "3f8f25448ab5d0b0646652a9bc86a7e0bb989a1a40d1dfea540d10d75c468d38"
    ),
    "wisa/scripts/run_seed_extension_stress.py": (
        "0c1fc8b40c4cb10d1fbabd1e2d0a47135e1a1479aa869795bdb84e35d117f3c2"
    ),
    "wisa/experiments/03_synthetic/rule_based_config.yaml": (
        "edfa574a1b9a325697563e627e3906976648f1806eee5047cc6a4dd8f8e49b15"
    ),
    "wisa/experiments/01_split/variant_generation_config.yaml": (
        "f4912d94083d5e7e5be7ad68c3175e6281ea1896594afec6497da689556b8add"
    ),
    "wisa/experiments/01_split/variant_sensitivity_config.yaml": (
        "f8d020c96afa8b3715823cf080a92b5276dda06d7eac943ddafb1001c42e1f57"
    ),
    "wisa/models/baseline/cnn_standardizer.npz": (
        "7ffaff380c0e4af079ca68f94bd9c1ec35e597be99266e3631d7284d8a05e389"
    ),
}

EXPECTED_LABEL_COUNTS = {
    "train": {0: 138_976, 1: 20_651, 2: 29_321, 3: 35_531, 4: 37_670},
    "val": {0: 46_224, 1: 7_007, 2: 5_617, 3: 12_134, 4: 13_908},
    "test": {0: 161_666, 1: 7_196, 2: 6_297, 3: 12_427, 4: 14_020},
    "rule_pool": {1: 65_000, 2: 65_000, 3: 65_000, 4: 65_000},
    "fixed_variant": {0: 32_000, 1: 8_000, 2: 8_000, 3: 8_000, 4: 8_000},
    "sensitivity": {0: 12_000, 1: 4_500, 2: 4_500, 3: 4_500, 4: 4_500},
}

EXPECTED_SAMPLE_HASHES = {
    ("rule_0p30", 7): (
        307,
        "4ff00984e1dd97a48447b2abac6c347eb8967b20e3c90782db081f9681c11581",
        "be7de8a2ee2090736ec3d10ae4bf5077b394edc9344ec46d0eb23b98d0483890",
    ),
    ("real_oversampling_0p30", 7): (
        408,
        "38874baf2d0d22afa2101e5850b28b6f7b7b7d010cdb30abe27b8a6d56403f26",
        "be7de8a2ee2090736ec3d10ae4bf5077b394edc9344ec46d0eb23b98d0483890",
    ),
    ("rule_0p30", 42): (
        342,
        "bdfcbdf15dcd5001122de17dd721508c5347353a022becd20df94c0e59a0f917",
        "fac21c3b58e98a5a3d69d115a72f530b2ce3893e6521bfadd83f9c17fbe18df6",
    ),
    ("real_oversampling_0p30", 42): (
        443,
        "ae9551ed910a8d31ec06daa82b3e53b6dc20924c787735fb6537b30bfeb15aca",
        "fac21c3b58e98a5a3d69d115a72f530b2ce3893e6521bfadd83f9c17fbe18df6",
    ),
    ("rule_0p30", 123): (
        423,
        "22bd0a549e33ad17662a6b5ac84065a8b229768aace1c5721ece4a2408255ba5",
        "53db276d4ce609b438b3e9d26010d8c0c1da3ae6ccf9ef760738badec8ff9dc1",
    ),
    ("real_oversampling_0p30", 123): (
        524,
        "c4a849ac1d069d22f534c9dc33d4bafe7330b37e1f916c8e94b2ec461e4090a7",
        "53db276d4ce609b438b3e9d26010d8c0c1da3ae6ccf9ef760738badec8ff9dc1",
    ),
    ("rule_0p30", 2026): (
        2326,
        "59a6f31755fb18e7f19c4ba9e819b050f4d5db617854e0f2c2e78f4ae31a3319",
        "88f4a265bc1983516751fdb346d790fa6e0cda0bf89bebaa22d805ad6b249ab3",
    ),
    ("real_oversampling_0p30", 2026): (
        2427,
        "0ef614d94468a8a91c9ebe04e2764dcc5940562d0308f7a7bb8844a36b6a85d3",
        "88f4a265bc1983516751fdb346d790fa6e0cda0bf89bebaa22d805ad6b249ab3",
    ),
    ("rule_0p30", 3407): (
        3707,
        "3ed08fb8b14d3796264fba3cdac76ca7563fd0d951ad295209d0798fd875e3bf",
        "6b173f6013ab3bfd8ae74bc9d9a08431912f67200973d9155c3bddf9e2297667",
    ),
    ("real_oversampling_0p30", 3407): (
        3808,
        "ed5f6263a7f011666b40413a8d167fc2f595a9fab8e22b6c4e9e45a3ed52c7b2",
        "6b173f6013ab3bfd8ae74bc9d9a08431912f67200973d9155c3bddf9e2297667",
    ),
}

EXPECTED_STANDARDIZER_ARRAY_HASHES = {
    "mean": "1e53f10b4dfb1de8522513ca20c7d0c2b5a19a8de8f76067e4273a93467da8a2",
    "std": "092f42e9a1019fe709233a91c6e876e976f8b3b8616807b42eb6e6da7b7dbb7c",
}

SUMMARY_METRICS = (
    "accuracy_binary",
    "macro_f1_binary",
    "fpr",
    "fnr",
    "normal_recall",
    "binary_attack_recall",
    "exact_attack_recall",
    "detection_recall_DoS",
    "detection_recall_Fuzzy",
    "detection_recall_Gear",
    "detection_recall_RPM",
    "exact_recall_DoS",
    "exact_recall_Fuzzy",
    "exact_recall_Gear",
    "exact_recall_RPM",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array_int64(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_array_raw(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def file_entry(path: Path) -> dict:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def log_event(event: str, **fields: object) -> None:
    record = {"utc": utc_now(), "event": event, **fields}
    line = json.dumps(record, sort_keys=True, ensure_ascii=True)
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    )


def atomic_write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(raw_temp)
    try:
        torch.save(value, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def label_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def json_label_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(key): value for key, value in label_counts(labels).items()}


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def environment_info() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
    }


def training_contract() -> dict:
    return {
        "version": VERSION,
        "accepted_source_commit": ACCEPTED_SOURCE_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "arms": list(ARMS),
        "paired_pipeline_seeds": list(SEEDS),
        "batch_size": BATCH_SIZE,
        "drop_last": False,
        "max_optimizer_updates": MAX_UPDATES,
        "validate_every_updates": VALIDATE_EVERY,
        "validation_updates": list(VALIDATION_UPDATES),
        "checkpoint_metric": "source validation five-class Macro-F1",
        "checkpoint_tie_rule": "strict improvement; earlier update wins exact tie",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "class_weight_policy": "balanced, recomputed from each arm training multiset",
        "standardizer_fit": "real Car-Hacking train only",
        "sampler": "deterministic cyclic torch.randperm; reshuffle at exhaustion",
        "sampler_seed_offset": SAMPLER_SEED_OFFSET,
        "worker_seed_offset": WORKER_SEED_OFFSET,
        "added_windows": ADDED_WINDOWS,
        "source_policy": "attack iff five-class argmax != class 0",
        "training_data_roles": [
            "Car-Hacking train",
            "Car-Hacking validation",
            "legacy Rule pool fitted from Car-Hacking train",
        ],
        "evaluation_data_roles": [
            "Car-Hacking source test",
            "fixed generated variant",
            "full generated sensitivity construction",
        ],
        "excluded_from_experiment": [
            "OTIDS",
            "can-train-and-test",
            "ROAD",
            "target-ID stress",
            "out-of-rule stress",
        ],
    }


CONTRACT_ID = canonical_digest(training_contract())


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 11, classes: int = 5) -> None:
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
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(128, classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(inputs))


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, len(FEATURE_NAMES)):
            raise ValueError(f"unexpected window tensor shape: {x.shape}")
        contiguous = np.ascontiguousarray(x.transpose(0, 2, 1), dtype=np.float32)
        self.x = torch.from_numpy(contiguous)
        self.y = (
            None
            if y is None
            else torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        )
        if self.y is not None and len(self.y) != len(self.x):
            raise ValueError("feature/label length mismatch")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        if self.y is None:
            return self.x[index]
        return self.x[index], self.y[index]


class CyclicBatchSampler(Sampler[list[int]]):
    """Yield a fixed number of update batches, reshuffling at each exhaustion."""

    def __init__(
        self, data_size: int, batch_size: int, updates: int, seed: int
    ) -> None:
        if data_size <= 0 or batch_size <= 0 or updates <= 0:
            raise ValueError("sampler sizes and updates must be positive")
        self.data_size = int(data_size)
        self.batch_size = int(batch_size)
        self.updates = int(updates)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.updates

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        yielded = 0
        while yielded < self.updates:
            permutation = torch.randperm(self.data_size, generator=generator)
            for start in range(0, self.data_size, self.batch_size):
                if yielded >= self.updates:
                    return
                yield permutation[start : start + self.batch_size].tolist()
                yielded += 1


@dataclass
class TrainingData:
    real_x: np.ndarray
    real_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    rule_x: np.ndarray
    rule_y: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: np.ndarray


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(
    dataset: Dataset,
    *,
    batch_sampler: Sampler[list[int]] | None = None,
    batch_size: int | None = None,
    workers: int,
    worker_seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(worker_seed)
    common = {
        "dataset": dataset,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "worker_init_fn": partial(seed_worker, base_seed=worker_seed),
        "generator": generator,
    }
    if workers > 0:
        common["prefetch_factor"] = 2
        common["persistent_workers"] = False
    if batch_sampler is not None:
        return DataLoader(batch_sampler=batch_sampler, **common)
    if batch_size is None:
        raise ValueError("batch_size is required without a batch_sampler")
    return DataLoader(batch_size=batch_size, shuffle=False, **common)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def balanced_indices(labels: np.ndarray, total_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = (1, 2, 3, 4)
    per_class = total_count // len(classes)
    remainder = total_count - per_class * len(classes)
    chunks: list[np.ndarray] = []
    for offset, attack_class in enumerate(classes):
        candidates = np.where(labels == attack_class)[0]
        draw_count = per_class + (1 if offset < remainder else 0)
        replace = draw_count > len(candidates)
        chunks.append(
            rng.choice(candidates, size=draw_count, replace=replace).astype(
                np.int64
            )
        )
    selected = np.concatenate(chunks)
    rng.shuffle(selected)
    return selected


def load_npz_fields(path: Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        missing = set(fields) - set(archive.files)
        if missing:
            raise ValueError(f"{relative(path)} missing fields: {sorted(missing)}")
        return {field: np.asarray(archive[field]) for field in fields}


def verify_window_metadata(
    path: Path,
    *,
    expected_count: int,
    expected_counts: dict[int, int],
    require_binary: bool = True,
) -> dict[str, np.ndarray]:
    fields = [
        "x",
        "y_attack_type",
        "feature_names",
        "window_size",
        "stride",
    ]
    if require_binary:
        fields.append("y_binary")
    arrays = load_npz_fields(path, fields)
    x = arrays["x"]
    y = arrays["y_attack_type"].astype(np.int64)
    if x.shape != (expected_count, WINDOW_SIZE, len(FEATURE_NAMES)):
        raise ValueError(f"{relative(path)} shape mismatch: {x.shape}")
    if label_counts(y) != expected_counts:
        raise ValueError(
            f"{relative(path)} class-count mismatch: {label_counts(y)}"
        )
    if tuple(str(value) for value in arrays["feature_names"].tolist()) != FEATURE_NAMES:
        raise ValueError(f"{relative(path)} feature order mismatch")
    if int(arrays["window_size"]) != WINDOW_SIZE or int(arrays["stride"]) != STRIDE:
        raise ValueError(f"{relative(path)} window/stride mismatch")
    if require_binary:
        expected_binary = (y > 0).astype(np.int8)
        if not np.array_equal(arrays["y_binary"].astype(np.int8), expected_binary):
            raise ValueError(f"{relative(path)} binary/type labels disagree")
    arrays["y_attack_type"] = y
    return arrays


def audit_frozen_files() -> tuple[list[dict], list[dict]]:
    if sha256_file(PLAN_PATH) != PLAN_SHA256:
        raise ValueError("analysis plan hash mismatch")
    if (
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", PLAN_COMMIT, "HEAD"],
            cwd=REPO_ROOT,
        )
        != 0
    ):
        raise ValueError("analysis plan commit is not an ancestor of HEAD")
    data_entries = []
    for raw_path, expected in EXPECTED_FILES.items():
        path = REPO_ROOT / raw_path
        entry = file_entry(path)
        if entry["bytes"] != expected["bytes"] or entry["sha256"] != expected["sha256"]:
            raise ValueError(f"frozen input mismatch: {raw_path}")
        data_entries.append(entry)
    dependency_entries = []
    for raw_path, expected_hash in EXPECTED_DEPENDENCIES.items():
        path = REPO_ROOT / raw_path
        entry = file_entry(path)
        if entry["sha256"] != expected_hash:
            raise ValueError(f"frozen dependency mismatch: {raw_path}")
        dependency_entries.append(entry)
    if git_text("status", "--short", "--", "wisa"):
        raise ValueError("frozen wisa tree is dirty")
    return data_entries, dependency_entries


def verify_script_state(allow_uncommitted: bool) -> tuple[str, str]:
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    if allow_uncommitted:
        return git_text("rev-parse", "HEAD"), script_hash
    tracked = subprocess.call(
        ["git", "ls-files", "--error-unmatch", relative(script_path)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dirty = subprocess.call(
        ["git", "diff", "--quiet", "HEAD", "--", relative(script_path)],
        cwd=REPO_ROOT,
    )
    if tracked != 0 or dirty != 0:
        raise ValueError("training implementation must be committed before execution")
    return git_text("rev-parse", "HEAD"), script_hash


def verify_sampling_contract(
    train_y: np.ndarray, rule_y: np.ndarray
) -> dict[tuple[str, int], dict]:
    attack_indices = np.where(train_y > 0)[0]
    records: dict[tuple[str, int], dict] = {}
    expected_added_counts = {1: 19_662, 2: 19_661, 3: 19_661, 4: 19_661}
    for seed in SEEDS:
        permutation = np.random.default_rng(seed).permutation(
            AUGMENTED_TRAIN_WINDOWS
        ).astype(np.int64)
        permutation_hash = sha256_array_int64(permutation)

        rule_indices = balanced_indices(rule_y, ADDED_WINDOWS, seed + 300)
        over_local = balanced_indices(
            train_y[attack_indices], ADDED_WINDOWS, seed + 401
        )
        over_indices = attack_indices[over_local].astype(np.int64)
        candidates = {
            "rule_0p30": (seed + 300, rule_indices, rule_y[rule_indices]),
            "real_oversampling_0p30": (
                seed + 401,
                over_indices,
                train_y[over_indices],
            ),
        }
        for arm, (sampling_seed, selected, selected_y) in candidates.items():
            expected_seed, expected_index_hash, expected_permutation_hash = (
                EXPECTED_SAMPLE_HASHES[(arm, seed)]
            )
            index_hash = sha256_array_int64(selected)
            if (
                sampling_seed != expected_seed
                or index_hash != expected_index_hash
                or permutation_hash != expected_permutation_hash
            ):
                raise ValueError(f"sampling lineage mismatch for {arm} seed {seed}")
            if len(np.unique(selected)) != ADDED_WINDOWS:
                raise ValueError(f"unexpected repeated added index for {arm} seed {seed}")
            if label_counts(selected_y) != expected_added_counts:
                raise ValueError(f"added class-count mismatch for {arm} seed {seed}")
            records[(arm, seed)] = {
                "sampling_seed": sampling_seed,
                "selected_index_sha256_int64_c": index_hash,
                "construction_permutation_sha256_int64_c": permutation_hash,
                "selected_unique_indices": int(len(np.unique(selected))),
                "selected_class_counts": json_label_counts(selected_y),
            }
    return records


def load_training_data() -> tuple[TrainingData, dict[tuple[str, int], dict]]:
    train = verify_window_metadata(
        DATA_ROOT / "windows" / "train_windows.npz",
        expected_count=REAL_TRAIN_WINDOWS,
        expected_counts=EXPECTED_LABEL_COUNTS["train"],
    )
    val = verify_window_metadata(
        DATA_ROOT / "windows" / "val_windows.npz",
        expected_count=84_890,
        expected_counts=EXPECTED_LABEL_COUNTS["val"],
    )
    rule = verify_window_metadata(
        DATA_ROOT / "synthetic" / "rule_based_windows.npz",
        expected_count=260_000,
        expected_counts=EXPECTED_LABEL_COUNTS["rule_pool"],
    )
    with np.load(
        DATA_ROOT / "synthetic" / "rule_based_windows.npz", allow_pickle=True
    ) as archive:
        if int(archive["seed"]) != 314_159:
            raise ValueError("Rule-pool generator seed mismatch")
    mean, std = fit_standardizer(train["x"])
    if sha256_array_raw(mean) != EXPECTED_STANDARDIZER_ARRAY_HASHES["mean"]:
        raise ValueError("real-train standardizer mean hash mismatch")
    if sha256_array_raw(std) != EXPECTED_STANDARDIZER_ARRAY_HASHES["std"]:
        raise ValueError("real-train standardizer std hash mismatch")
    with np.load(
        WISA_ROOT / "models" / "baseline" / "cnn_standardizer.npz"
    ) as frozen:
        if not np.array_equal(mean, frozen["mean"]) or not np.array_equal(
            std, frozen["std"]
        ):
            raise ValueError("recomputed standardizer differs from frozen baseline")
    sampling = verify_sampling_contract(train["y_attack_type"], rule["y_attack_type"])
    data = TrainingData(
        real_x=train["x"].astype(np.float32, copy=False),
        real_y=train["y_attack_type"].astype(np.int64, copy=False),
        val_x=val["x"].astype(np.float32, copy=False),
        val_y=val["y_attack_type"].astype(np.int64, copy=False),
        rule_x=rule["x"].astype(np.float32, copy=False),
        rule_y=rule["y_attack_type"].astype(np.int64, copy=False),
        mean=mean,
        std=std,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    return data, sampling


def run_unit_tests() -> None:
    if len(VALIDATION_UPDATES) != 12 or VALIDATION_UPDATES[-1] != MAX_UPDATES:
        raise AssertionError("validation update grid")
    sampler = CyclicBatchSampler(data_size=5, batch_size=2, updates=5, seed=19)
    batches = list(iter(sampler))
    if [len(batch) for batch in batches] != [2, 2, 1, 2, 2]:
        raise AssertionError("cyclic sampler final-batch retention")
    if sorted(index for batch in batches[:3] for index in batch) != list(range(5)):
        raise AssertionError("cyclic sampler first-cycle coverage")
    if sorted(index for batch in batches[3:] for index in batch) == list(range(5)):
        raise AssertionError("partial second cycle unexpectedly complete")
    first = list(
        CyclicBatchSampler(data_size=17, batch_size=4, updates=9, seed=77)
    )
    second = list(
        CyclicBatchSampler(data_size=17, batch_size=4, updates=9, seed=77)
    )
    if first != second:
        raise AssertionError("cyclic sampler determinism")
    tiny_y = np.array([0, 0, 1, 1], dtype=np.int64)
    tiny_pred = np.array([0, 1, 1, 2], dtype=np.int64)
    metrics = classification_metrics(tiny_y, tiny_pred)
    if (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) != (1, 1, 0, 2):
        raise AssertionError("binary confusion calculation")
    if not math.isclose(metrics["fpr"], 0.5):
        raise AssertionError("binary FPR calculation")
    best_update, best_value = select_best_checkpoint(
        [(666, 0.1), (1332, 0.2), (1998, 0.2)]
    )
    if best_update != 1332 or best_value != 0.2:
        raise AssertionError("earlier-tie checkpoint selection")
    model = CNN1D()
    if sum(parameter.numel() for parameter in model.parameters()) != 95_237:
        raise AssertionError("CNN architecture parameter count")
    with torch.no_grad():
        output = model(torch.zeros(2, len(FEATURE_NAMES), WINDOW_SIZE))
    if output.shape != (2, 5):
        raise AssertionError("CNN output shape")


def preflight(
    *, allow_uncommitted: bool
) -> tuple[
    TrainingData,
    dict[tuple[str, int], dict],
    list[dict],
    list[dict],
    str,
    str,
]:
    data_entries, dependency_entries = audit_frozen_files()
    implementation_commit, script_hash = verify_script_state(allow_uncommitted)
    run_unit_tests()
    data, sampling = load_training_data()
    return (
        data,
        sampling,
        data_entries,
        dependency_entries,
        implementation_commit,
        script_hash,
    )


def select_best_checkpoint(
    update_values: Sequence[tuple[int, float]]
) -> tuple[int, float]:
    best_update = -1
    best_value = -math.inf
    for update, value in update_values:
        if value > best_value:
            best_update = int(update)
            best_value = float(value)
    return best_update, best_value


def class_weights(labels: np.ndarray) -> np.ndarray:
    return compute_class_weight(
        class_weight="balanced", classes=np.arange(5), y=labels
    ).astype(np.float64)


def construct_training_arm(
    arm: str,
    seed: int,
    data: TrainingData,
    real_x_standardized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if arm == "real_only":
        labels = data.real_y
        return real_x_standardized, labels, {
            "training_windows": REAL_TRAIN_WINDOWS,
            "added_windows": 0,
            "sampling_seed": None,
            "selected_index_sha256_int64_c": None,
            "construction_permutation_sha256_int64_c": None,
            "selected_unique_indices": 0,
            "selected_class_counts": {},
        }

    permutation = np.random.default_rng(seed).permutation(
        AUGMENTED_TRAIN_WINDOWS
    ).astype(np.int64)
    if arm == "rule_0p30":
        selected = balanced_indices(data.rule_y, ADDED_WINDOWS, seed + 300)
        added_x = standardize(data.rule_x[selected], data.mean, data.std)
        added_y = data.rule_y[selected]
        sampling_seed = seed + 300
    elif arm == "real_oversampling_0p30":
        attack_indices = np.where(data.real_y > 0)[0]
        selected_local = balanced_indices(
            data.real_y[attack_indices], ADDED_WINDOWS, seed + 401
        )
        selected = attack_indices[selected_local].astype(np.int64)
        added_x = real_x_standardized[selected]
        added_y = data.real_y[selected]
        sampling_seed = seed + 401
    else:
        raise ValueError(f"unknown arm: {arm}")

    combined_x = np.concatenate([real_x_standardized, added_x], axis=0)
    combined_y = np.concatenate([data.real_y, added_y], axis=0).astype(
        np.int64, copy=False
    )
    ordered_x = np.ascontiguousarray(combined_x[permutation], dtype=np.float32)
    ordered_y = np.ascontiguousarray(combined_y[permutation], dtype=np.int64)
    del combined_x, combined_y, added_x
    record = {
        "training_windows": AUGMENTED_TRAIN_WINDOWS,
        "added_windows": ADDED_WINDOWS,
        "sampling_seed": sampling_seed,
        "selected_index_sha256_int64_c": sha256_array_int64(selected),
        "construction_permutation_sha256_int64_c": sha256_array_int64(permutation),
        "selected_unique_indices": int(len(np.unique(selected))),
        "selected_class_counts": json_label_counts(added_y),
    }
    expected = EXPECTED_SAMPLE_HASHES[(arm, seed)]
    if (
        record["sampling_seed"] != expected[0]
        or record["selected_index_sha256_int64_c"] != expected[1]
        or record["construction_permutation_sha256_int64_c"] != expected[2]
    ):
        raise ValueError(f"constructed arm lineage mismatch: {arm} seed {seed}")
    return ordered_x, ordered_y, record


def predict_labels(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            logits = model(
                features.to(device, non_blocking=torch.cuda.is_available())
            )
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite model logits")
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    if not predictions:
        raise ValueError("empty prediction loader")
    return np.concatenate(predictions).astype(np.int64, copy=False)


def validation_macro_f1(
    model: nn.Module,
    val_loader: DataLoader,
    val_y: np.ndarray,
    device: torch.device,
) -> float:
    predictions = predict_labels(model, val_loader, device)
    if len(predictions) != len(val_y):
        raise ValueError("validation prediction/label mismatch")
    return float(
        f1_score(val_y, predictions, average="macro", zero_division=0)
    )


def checkpoint_path(arm: str, seed: int) -> Path:
    return MODEL_ROOT / f"cnn1d_{arm}_seed{seed}_v1.pt"


def history_path(arm: str, seed: int) -> Path:
    return HISTORY_ROOT / f"{arm}_seed{seed}_v1.csv"


def run_metadata_path(arm: str, seed: int) -> Path:
    return RUN_ROOT / f"{arm}_seed{seed}_v1.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_completed_run(
    arm: str,
    seed: int,
    *,
    script_hash: str,
    implementation_commit: str,
) -> dict:
    metadata_path = run_metadata_path(arm, seed)
    model_path = checkpoint_path(arm, seed)
    hist_path = history_path(arm, seed)
    exists = [metadata_path.exists(), model_path.exists(), hist_path.exists()]
    if not any(exists):
        raise FileNotFoundError(f"no completed artifact for {arm} seed {seed}")
    if not all(exists):
        raise ValueError(f"incomplete artifact triplet for {arm} seed {seed}")
    metadata = load_json(metadata_path)
    expected_pairs = {
        "version": VERSION,
        "contract_id": CONTRACT_ID,
        "arm": arm,
        "pipeline_seed": seed,
        "implementation_commit": implementation_commit,
        "implementation_sha256": script_hash,
        "updates_run": MAX_UPDATES,
    }
    for key, expected in expected_pairs.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{arm} seed {seed} metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    if metadata["checkpoint"]["sha256"] != sha256_file(model_path):
        raise ValueError(f"checkpoint hash mismatch for {arm} seed {seed}")
    if metadata["history"]["sha256"] != sha256_file(hist_path):
        raise ValueError(f"history hash mismatch for {arm} seed {seed}")
    history = pd.read_csv(hist_path)
    if history["update"].astype(int).tolist() != list(VALIDATION_UPDATES):
        raise ValueError(f"history update grid mismatch for {arm} seed {seed}")
    if history["selected"].astype(int).sum() != 1:
        raise ValueError(f"history selected-row count mismatch for {arm} seed {seed}")
    selected = history.loc[history["selected"].astype(bool)].iloc[0]
    if int(selected["update"]) != int(metadata["selected_update"]):
        raise ValueError(f"selected update mismatch for {arm} seed {seed}")
    if not math.isclose(
        float(selected["val_macro_f1_multiclass"]),
        float(metadata["selected_val_macro_f1_multiclass"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"selected validation value mismatch for {arm} seed {seed}")
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"checkpoint contract mismatch for {arm} seed {seed}")
    if payload.get("arm") != arm or int(payload.get("pipeline_seed")) != seed:
        raise ValueError(f"checkpoint identity mismatch for {arm} seed {seed}")
    if int(payload.get("selected_update")) != int(metadata["selected_update"]):
        raise ValueError(f"checkpoint selected update mismatch for {arm} seed {seed}")
    metadata["run_metadata"] = file_entry(metadata_path)
    return metadata


def save_or_verify_standardizer(
    mean: np.ndarray, std: np.ndarray, feature_names: np.ndarray
) -> dict:
    if STANDARDIZER_PATH.exists():
        with np.load(STANDARDIZER_PATH, allow_pickle=True) as archive:
            if not np.array_equal(archive["mean"], mean) or not np.array_equal(
                archive["std"], std
            ):
                raise ValueError("existing camera-ready standardizer mismatch")
            if tuple(archive["feature_names"].tolist()) != FEATURE_NAMES:
                raise ValueError("existing camera-ready feature order mismatch")
    else:
        atomic_save_npz(
            STANDARDIZER_PATH,
            mean=mean,
            std=std,
            feature_names=feature_names,
            fit_role=np.asarray("Car-Hacking real train only"),
        )
    return file_entry(STANDARDIZER_PATH)


def train_one(
    arm: str,
    seed: int,
    data: TrainingData,
    real_x_standardized: np.ndarray,
    val_dataset: WindowDataset,
    *,
    implementation_commit: str,
    script_hash: str,
    input_inventory_digest: str,
) -> dict:
    existing = [
        checkpoint_path(arm, seed).exists(),
        history_path(arm, seed).exists(),
        run_metadata_path(arm, seed).exists(),
    ]
    if any(existing):
        metadata = validate_completed_run(
            arm,
            seed,
            script_hash=script_hash,
            implementation_commit=implementation_commit,
        )
        log_event(
            "training_fit_verified_and_skipped",
            arm=arm,
            pipeline_seed=seed,
            selected_update=metadata["selected_update"],
        )
        return metadata

    started_utc = utc_now()
    started = time.monotonic()
    log_event("training_fit_started", arm=arm, pipeline_seed=seed)
    set_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y, construction = construct_training_arm(
        arm, seed, data, real_x_standardized
    )
    expected_n = (
        REAL_TRAIN_WINDOWS if arm == "real_only" else AUGMENTED_TRAIN_WINDOWS
    )
    if len(train_y) != expected_n or label_counts(train_y).get(0) != 138_976:
        raise ValueError(f"training multiset mismatch for {arm} seed {seed}")
    weights = class_weights(train_y)
    train_counts = json_label_counts(train_y)

    train_dataset = WindowDataset(train_x, train_y)
    if arm != "real_only":
        del train_x
        gc.collect()
    batch_sampler = CyclicBatchSampler(
        data_size=len(train_dataset),
        batch_size=BATCH_SIZE,
        updates=MAX_UPDATES,
        seed=seed + SAMPLER_SEED_OFFSET,
    )
    train_loader = make_loader(
        train_dataset,
        batch_sampler=batch_sampler,
        workers=TRAIN_WORKERS,
        worker_seed=seed + WORKER_SEED_OFFSET,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        workers=EVAL_WORKERS,
        worker_seed=seed + WORKER_SEED_OFFSET + 17,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNN1D(in_channels=len(FEATURE_NAMES), classes=5).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history: list[dict] = []
    best_value = -math.inf
    best_update = -1
    best_state: dict[str, torch.Tensor] | None = None
    interval_loss_sum = 0.0
    interval_examples = 0
    cumulative_examples = 0

    for update, batch in enumerate(train_loader, start=1):
        features, labels = batch
        features = features.to(
            device, non_blocking=torch.cuda.is_available()
        )
        labels = labels.to(device, non_blocking=torch.cuda.is_available())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite training loss for {arm} seed {seed}")
        loss.backward()
        optimizer.step()
        batch_count = int(len(labels))
        interval_loss_sum += float(loss.detach().cpu()) * batch_count
        interval_examples += batch_count
        cumulative_examples += batch_count

        if update % VALIDATE_EVERY == 0:
            value = validation_macro_f1(
                model, val_loader, data.val_y, device
            )
            improved = value > best_value
            if improved:
                best_value = value
                best_update = update
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
            row = {
                "arm": arm,
                "pipeline_seed": seed,
                "update": update,
                "interval_examples": interval_examples,
                "cumulative_examples": cumulative_examples,
                "train_loss_interval_sample_mean": (
                    interval_loss_sum / interval_examples
                ),
                "val_macro_f1_multiclass": value,
                "strict_improvement_at_time": int(improved),
                "selected": 0,
            }
            history.append(row)
            log_event(
                "validation_completed",
                arm=arm,
                pipeline_seed=seed,
                update=update,
                train_loss_interval_sample_mean=row[
                    "train_loss_interval_sample_mean"
                ],
                val_macro_f1_multiclass=value,
                best_update=best_update,
                best_val_macro_f1_multiclass=best_value,
            )
            interval_loss_sum = 0.0
            interval_examples = 0

    if update != MAX_UPDATES:
        raise ValueError(f"fit stopped at update {update}, expected {MAX_UPDATES}")
    if best_state is None or best_update not in VALIDATION_UPDATES:
        raise ValueError(f"no valid checkpoint selected for {arm} seed {seed}")
    for row in history:
        row["selected"] = int(row["update"] == best_update)
    if len(history) != len(VALIDATION_UPDATES):
        raise ValueError(f"wrong validation-row count for {arm} seed {seed}")

    model.load_state_dict(best_state)
    verified_value = validation_macro_f1(
        model, val_loader, data.val_y, device
    )
    if not math.isclose(
        verified_value, best_value, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"selected checkpoint re-evaluation mismatch for {arm} seed {seed}"
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started

    checkpoint_payload = {
        "version": VERSION,
        "contract_id": CONTRACT_ID,
        "arm": arm,
        "pipeline_seed": seed,
        "selected_update": best_update,
        "selected_val_macro_f1_multiclass": best_value,
        "state_dict": best_state,
        "architecture": "unchanged five-class CNN1D",
        "source_policy": "attack iff argmax != class 0",
    }
    atomic_torch_save(checkpoint_path(arm, seed), checkpoint_payload)
    atomic_write_csv(history_path(arm, seed), history)
    checkpoint_entry = file_entry(checkpoint_path(arm, seed))
    history_entry = file_entry(history_path(arm, seed))

    metadata = {
        "version": VERSION,
        "contract_id": CONTRACT_ID,
        "arm": arm,
        "pipeline_seed": seed,
        "implementation_commit": implementation_commit,
        "implementation_sha256": script_hash,
        "input_inventory_digest": input_inventory_digest,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "runtime_seconds": elapsed,
        "device": str(device),
        "training_windows": len(train_y),
        "training_class_counts": train_counts,
        "class_weights_Normal_DoS_Fuzzy_Gear_RPM": [
            float(value) for value in weights
        ],
        "batch_size": BATCH_SIZE,
        "batches_per_cycle": math.ceil(len(train_y) / BATCH_SIZE),
        "sampler_seed": seed + SAMPLER_SEED_OFFSET,
        "worker_seed": seed + WORKER_SEED_OFFSET,
        "updates_run": MAX_UPDATES,
        "validation_updates": list(VALIDATION_UPDATES),
        "selected_update": best_update,
        "selected_val_macro_f1_multiclass": best_value,
        "selected_checkpoint_recomputed_val_macro_f1_multiclass": verified_value,
        "construction": construction,
        "checkpoint": checkpoint_entry,
        "history": history_entry,
    }
    atomic_write_json(run_metadata_path(arm, seed), metadata)
    metadata["run_metadata"] = file_entry(run_metadata_path(arm, seed))
    log_event(
        "training_fit_completed",
        arm=arm,
        pipeline_seed=seed,
        runtime_seconds=elapsed,
        selected_update=best_update,
        selected_val_macro_f1_multiclass=best_value,
        checkpoint_sha256=checkpoint_entry["sha256"],
    )

    del (
        model,
        optimizer,
        criterion,
        train_loader,
        train_dataset,
        val_loader,
        best_state,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def inventory_digest(entries: Sequence[dict]) -> str:
    normalized = [
        {
            "path": entry["path"],
            "bytes": int(entry["bytes"]),
            "sha256": entry["sha256"],
        }
        for entry in sorted(entries, key=lambda item: item["path"])
    ]
    return canonical_digest(normalized)


def validate_training_manifest(
    *,
    script_hash: str,
    implementation_commit: str,
) -> tuple[dict, list[dict]]:
    if not TRAINING_MANIFEST_PATH.exists():
        raise FileNotFoundError("training manifest is missing")
    manifest = load_json(TRAINING_MANIFEST_PATH)
    if manifest.get("version") != VERSION or manifest.get("contract_id") != CONTRACT_ID:
        raise ValueError("training manifest contract mismatch")
    if manifest["implementation"]["commit"] != implementation_commit:
        raise ValueError("training manifest implementation commit mismatch")
    if manifest["implementation"]["sha256"] != script_hash:
        raise ValueError("training manifest implementation hash mismatch")
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            runs.append(
                validate_completed_run(
                    arm,
                    seed,
                    script_hash=script_hash,
                    implementation_commit=implementation_commit,
                )
            )
    if len(runs) != 15:
        raise ValueError("training manifest does not cover 15 fits")
    return manifest, runs


def run_training(*, allow_uncommitted: bool = False) -> None:
    (
        data,
        sampling,
        data_entries,
        dependency_entries,
        implementation_commit,
        script_hash,
    ) = preflight(allow_uncommitted=allow_uncommitted)
    if allow_uncommitted:
        raise ValueError("full training refuses --allow-uncommitted-code")
    if FINAL_MANIFEST_PATH.exists():
        raise ValueError("completed v1 result manifest already exists")
    if TRAINING_MANIFEST_PATH.exists():
        validate_training_manifest(
            script_hash=script_hash,
            implementation_commit=implementation_commit,
        )
        log_event("training_manifest_verified", path=relative(TRAINING_MANIFEST_PATH))
        return

    training_started_utc = utc_now()
    training_started = time.monotonic()
    log_event(
        "training_phase_started",
        version=VERSION,
        contract_id=CONTRACT_ID,
        implementation_commit=implementation_commit,
        implementation_sha256=script_hash,
        run_order=[
            {"pipeline_seed": seed, "arm": arm}
            for seed in SEEDS
            for arm in ARMS
        ],
    )
    standardizer_entry = save_or_verify_standardizer(
        data.mean, data.std, data.feature_names
    )
    real_x_standardized = standardize(data.real_x, data.mean, data.std)
    val_x_standardized = standardize(data.val_x, data.mean, data.std)
    val_dataset = WindowDataset(val_x_standardized, data.val_y)
    del val_x_standardized
    gc.collect()

    all_inputs = data_entries + dependency_entries + [
        file_entry(PLAN_PATH),
        file_entry(Path(__file__).resolve()),
    ]
    input_digest = inventory_digest(all_inputs)
    run_records = []
    for seed in SEEDS:
        for arm in ARMS:
            record = train_one(
                arm,
                seed,
                data,
                real_x_standardized,
                val_dataset,
                implementation_commit=implementation_commit,
                script_hash=script_hash,
                input_inventory_digest=input_digest,
            )
            run_records.append(record)

    verified_runs = []
    for seed in SEEDS:
        for arm in ARMS:
            verified_runs.append(
                validate_completed_run(
                    arm,
                    seed,
                    script_hash=script_hash,
                    implementation_commit=implementation_commit,
                )
            )
    if len(verified_runs) != 15:
        raise ValueError("not all 15 fits completed")
    elapsed = time.monotonic() - training_started
    training_manifest = {
        "experiment": VERSION,
        "version": VERSION,
        "contract_id": CONTRACT_ID,
        "contract": training_contract(),
        "analysis_plan": {
            "path": relative(PLAN_PATH),
            "commit": PLAN_COMMIT,
            "sha256": PLAN_SHA256,
            "frozen_before_new_matched_update_outcomes": True,
        },
        "implementation": {
            "path": relative(Path(__file__).resolve()),
            "commit": implementation_commit,
            "sha256": script_hash,
        },
        "started_utc": training_started_utc,
        "completed_utc": utc_now(),
        "runtime_seconds": elapsed,
        "environment": environment_info(),
        "inputs": {
            "dataset_files": data_entries,
            "dependency_files": dependency_entries,
            "inventory_digest": input_digest,
            "standardizer": standardizer_entry,
        },
        "sampling_lineage": [
            {"arm": arm, "pipeline_seed": seed, **sampling[(arm, seed)]}
            for seed in SEEDS
            for arm in ("rule_0p30", "real_oversampling_0p30")
        ],
        "fits": verified_runs,
        "validation_gates": {
            "fits_expected": 15,
            "fits_completed": len(verified_runs),
            "all_updates_equal_7992": all(
                int(record["updates_run"]) == MAX_UPDATES
                for record in verified_runs
            ),
            "all_histories_have_12_validation_rows": True,
            "all_selected_updates_on_frozen_grid": all(
                int(record["selected_update"]) in VALIDATION_UPDATES
                for record in verified_runs
            ),
            "source_test_or_variant_loaded_during_training": False,
            "external_data_loaded": False,
            "wisa_tree_clean": not bool(
                git_text("status", "--short", "--", "wisa")
            ),
        },
        "warnings": [
            "Pipeline-seed variation is conditional on one fixed split and one fixed Rule pool.",
            "Arm-specific balanced class weights are retained from the accepted training policy.",
        ],
    }
    atomic_write_json(TRAINING_MANIFEST_PATH, training_manifest)
    log_event(
        "training_phase_completed",
        runtime_seconds=elapsed,
        fits_completed=15,
        training_manifest=relative(TRAINING_MANIFEST_PATH),
        training_manifest_sha256=sha256_file(TRAINING_MANIFEST_PATH),
    )
    del data, real_x_standardized, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classification_metrics(
    y_true_multiclass: np.ndarray, predictions: np.ndarray
) -> dict:
    y_true = np.asarray(y_true_multiclass, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if len(y_true) != len(pred) or len(y_true) == 0:
        raise ValueError("classification metric input mismatch")
    true_binary = (y_true > 0).astype(np.int8)
    pred_binary = (pred > 0).astype(np.int8)
    cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    normal_denominator = tn + fp
    attack_denominator = tp + fn
    attack_mask = y_true > 0
    row = {
        "windows": int(len(y_true)),
        "normal_windows": int((y_true == 0).sum()),
        "attack_windows": int(attack_mask.sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy_binary": float((true_binary == pred_binary).mean()),
        "macro_f1_binary": float(
            f1_score(
                true_binary,
                pred_binary,
                average="macro",
                zero_division=0,
            )
        ),
        "fpr": (
            float(fp / normal_denominator) if normal_denominator else math.nan
        ),
        "fnr": (
            float(fn / attack_denominator) if attack_denominator else math.nan
        ),
        "normal_recall": (
            float(tn / normal_denominator) if normal_denominator else math.nan
        ),
        "binary_attack_recall": (
            float(tp / attack_denominator) if attack_denominator else math.nan
        ),
        "exact_attack_recall": (
            float((pred[attack_mask] == y_true[attack_mask]).mean())
            if attack_mask.any()
            else math.nan
        ),
    }
    for attack_class in (1, 2, 3, 4):
        mask = y_true == attack_class
        name = CLASS_NAMES[attack_class]
        row[f"detection_recall_{name}"] = (
            float((pred[mask] > 0).mean()) if mask.any() else math.nan
        )
        row[f"exact_recall_{name}"] = (
            float((pred[mask] == attack_class).mean()) if mask.any() else math.nan
        )
    return row


def load_evaluation_data(
    mean: np.ndarray, std: np.ndarray
) -> tuple[dict[str, WindowDataset], dict[str, np.ndarray], dict[str, np.ndarray]]:
    val = verify_window_metadata(
        DATA_ROOT / "windows" / "val_windows.npz",
        expected_count=84_890,
        expected_counts=EXPECTED_LABEL_COUNTS["val"],
    )
    test = verify_window_metadata(
        DATA_ROOT / "windows" / "test_windows.npz",
        expected_count=201_606,
        expected_counts=EXPECTED_LABEL_COUNTS["test"],
    )
    fixed = verify_window_metadata(
        DATA_ROOT / "windows" / "variant_test_windows.npz",
        expected_count=64_000,
        expected_counts=EXPECTED_LABEL_COUNTS["fixed_variant"],
    )
    sensitivity = verify_window_metadata(
        DATA_ROOT / "windows" / "variant_sensitivity_windows.npz",
        expected_count=30_000,
        expected_counts=EXPECTED_LABEL_COUNTS["sensitivity"],
    )
    with np.load(
        DATA_ROOT / "windows" / "variant_test_windows.npz", allow_pickle=True
    ) as archive:
        if int(archive["seed"]) != 42:
            raise ValueError("fixed-variant seed mismatch")
    with np.load(
        DATA_ROOT / "windows" / "variant_sensitivity_windows.npz",
        allow_pickle=True,
    ) as archive:
        if int(archive["seed"]) != 20_260_611:
            raise ValueError("sensitivity seed mismatch")
        attack_type = np.asarray(archive["attack_type"], dtype=object)
        severity = np.asarray(archive["severity"], dtype=object)
        scenario_id = np.asarray(archive["scenario_id"], dtype=object)
    expected_scenarios = {
        f"{family}_{level}": 1_500
        for family in ("DoS", "Fuzzy", "Gear", "RPM")
        for level in ("low", "medium", "high")
    }
    scenario_counts = {
        str(key): int(value)
        for key, value in zip(*np.unique(scenario_id, return_counts=True))
        if str(key) != "Normal_normal"
    }
    if scenario_counts != expected_scenarios:
        raise ValueError(f"sensitivity scenario mismatch: {scenario_counts}")
    if int((scenario_id == "Normal_normal").sum()) != 12_000:
        raise ValueError("sensitivity normal count mismatch")

    raw = {"validation": val, "source_test": test, "fixed_variant": fixed, "sensitivity": sensitivity}
    datasets: dict[str, WindowDataset] = {}
    labels: dict[str, np.ndarray] = {}
    for name, arrays in raw.items():
        standardized = standardize(arrays["x"], mean, std)
        datasets[name] = WindowDataset(standardized)
        labels[name] = arrays["y_attack_type"].astype(np.int64, copy=False)
        del standardized
    sensitivity_meta = {
        "attack_type": attack_type,
        "severity": severity,
        "scenario_id": scenario_id,
    }
    return datasets, labels, sensitivity_meta


def build_eval_loaders(
    datasets: dict[str, WindowDataset], seed: int
) -> dict[str, DataLoader]:
    return {
        name: make_loader(
            dataset,
            batch_size=INFERENCE_BATCH_SIZE,
            workers=EVAL_WORKERS,
            worker_seed=seed + WORKER_SEED_OFFSET + 10_000 + offset,
            pin_memory=torch.cuda.is_available(),
        )
        for offset, (name, dataset) in enumerate(datasets.items())
    }


def sensitivity_groups(
    labels: np.ndarray, metadata: dict[str, np.ndarray]
) -> list[tuple[str, str, str, np.ndarray]]:
    attack_mask = labels > 0
    groups: list[tuple[str, str, str, np.ndarray]] = [
        ("overall", "all_attacks", "All attack windows", attack_mask)
    ]
    for level in ("low", "medium", "high"):
        groups.append(
            (
                "severity",
                level,
                f"{level} coupled severity",
                attack_mask & (metadata["severity"] == level),
            )
        )
    for family in ("DoS", "Fuzzy", "Gear", "RPM"):
        groups.append(
            (
                "family",
                family,
                family,
                attack_mask & (metadata["attack_type"] == family),
            )
        )
    for family in ("DoS", "Fuzzy", "Gear", "RPM"):
        for level in ("low", "medium", "high"):
            scenario = f"{family}_{level}"
            groups.append(
                (
                    "scenario",
                    scenario,
                    f"{family} {level}",
                    attack_mask & (metadata["scenario_id"] == scenario),
                )
            )
    return groups


def build_sensitivity_seed_rows(
    arm: str,
    seed: int,
    labels: np.ndarray,
    predictions: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for granularity, group_key, group_label, mask in sensitivity_groups(
        labels, metadata
    ):
        if not mask.any():
            raise ValueError(f"empty sensitivity group: {group_key}")
        binary_recall = float((predictions[mask] > 0).mean())
        exact_recall = float((predictions[mask] == labels[mask]).mean())
        for metric, value in (
            ("binary_attack_recall", binary_recall),
            ("exact_attack_recall", exact_recall),
        ):
            rows.append(
                {
                    "row_type": "seed",
                    "arm": arm,
                    "contrast": "",
                    "pipeline_seed": seed,
                    "granularity": granularity,
                    "group_key": group_key,
                    "group_label": group_label,
                    "metric": metric,
                    "attack_windows": int(mask.sum()),
                    "value": value,
                    "mean": "",
                    "std": "",
                    "n_pipeline_seeds": 1,
                }
            )
    return rows


def summarize_by_seed(rows: Sequence[dict]) -> list[dict]:
    summaries = []
    lookup = {
        (str(row["arm"]), int(row["pipeline_seed"]), str(row["split"])): row
        for row in rows
    }
    for split in ("source_test", "fixed_variant", "sensitivity"):
        for metric in SUMMARY_METRICS:
            for arm in ARMS:
                values = [
                    float(lookup[(arm, seed, split)][metric]) for seed in SEEDS
                ]
                summary = {
                    "row_type": "arm_summary",
                    "split": split,
                    "metric": metric,
                    "arm": arm,
                    "contrast": "",
                    "n_pipeline_seeds": len(values),
                    "mean": float(np.mean(values)),
                    "std": sample_std(values),
                }
                summary.update(
                    {f"seed_{seed}": value for seed, value in zip(SEEDS, values)}
                )
                summaries.append(summary)
            for contrast, minuend, subtrahend in CONTRASTS:
                differences = [
                    float(lookup[(minuend, seed, split)][metric])
                    - float(lookup[(subtrahend, seed, split)][metric])
                    for seed in SEEDS
                ]
                summary = {
                    "row_type": "paired_difference",
                    "split": split,
                    "metric": metric,
                    "arm": "",
                    "contrast": contrast,
                    "n_pipeline_seeds": len(differences),
                    "mean": float(np.mean(differences)),
                    "std": sample_std(differences),
                }
                summary.update(
                    {
                        f"seed_{seed}": value
                        for seed, value in zip(SEEDS, differences)
                    }
                )
                summaries.append(summary)
    return summaries


def summarize_sensitivity(seed_rows: Sequence[dict]) -> list[dict]:
    rows = [dict(row) for row in seed_rows]
    lookup = {
        (
            str(row["arm"]),
            int(row["pipeline_seed"]),
            str(row["granularity"]),
            str(row["group_key"]),
            str(row["metric"]),
        ): float(row["value"])
        for row in seed_rows
    }
    identities = []
    seen = set()
    for row in seed_rows:
        identity = (
            str(row["granularity"]),
            str(row["group_key"]),
            str(row["group_label"]),
            str(row["metric"]),
            int(row["attack_windows"]),
        )
        if identity not in seen:
            seen.add(identity)
            identities.append(identity)
    for granularity, group_key, group_label, metric, attack_windows in identities:
        for arm in ARMS:
            values = [
                lookup[(arm, seed, granularity, group_key, metric)]
                for seed in SEEDS
            ]
            row = {
                "row_type": "arm_summary",
                "arm": arm,
                "contrast": "",
                "pipeline_seed": "",
                "granularity": granularity,
                "group_key": group_key,
                "group_label": group_label,
                "metric": metric,
                "attack_windows": attack_windows,
                "value": "",
                "mean": float(np.mean(values)),
                "std": sample_std(values),
                "n_pipeline_seeds": len(values),
            }
            row.update(
                {f"seed_{seed}": value for seed, value in zip(SEEDS, values)}
            )
            rows.append(row)
        for contrast, minuend, subtrahend in CONTRASTS:
            differences = [
                lookup[(minuend, seed, granularity, group_key, metric)]
                - lookup[(subtrahend, seed, granularity, group_key, metric)]
                for seed in SEEDS
            ]
            row = {
                "row_type": "paired_difference",
                "arm": "",
                "contrast": contrast,
                "pipeline_seed": "",
                "granularity": granularity,
                "group_key": group_key,
                "group_label": group_label,
                "metric": metric,
                "attack_windows": attack_windows,
                "value": "",
                "mean": float(np.mean(differences)),
                "std": sample_std(differences),
                "n_pipeline_seeds": len(differences),
            }
            row.update(
                {
                    f"seed_{seed}": value
                    for seed, value in zip(SEEDS, differences)
                }
            )
            rows.append(row)
    return rows


def checkpoint_rows(runs: Sequence[dict]) -> list[dict]:
    rows = []
    for record in runs:
        history = pd.read_csv(REPO_ROOT / record["history"]["path"])
        for row in history.to_dict(orient="records"):
            rows.append(
                {
                    "arm": record["arm"],
                    "pipeline_seed": int(record["pipeline_seed"]),
                    "update": int(row["update"]),
                    "interval_examples": int(row["interval_examples"]),
                    "cumulative_examples": int(row["cumulative_examples"]),
                    "train_loss_interval_sample_mean": float(
                        row["train_loss_interval_sample_mean"]
                    ),
                    "val_macro_f1_multiclass": float(
                        row["val_macro_f1_multiclass"]
                    ),
                    "strict_improvement_at_time": int(
                        row["strict_improvement_at_time"]
                    ),
                    "selected": int(row["selected"]),
                }
            )
    if len(rows) != 15 * len(VALIDATION_UPDATES):
        raise ValueError("checkpoint selection row count mismatch")
    return rows


def validate_output_tables(
    by_seed: Sequence[dict],
    summary: Sequence[dict],
    sensitivity_rows: Sequence[dict],
    checkpoint_selection: Sequence[dict],
) -> dict:
    if len(by_seed) != 15 * 3:
        raise ValueError(f"by-seed row count mismatch: {len(by_seed)}")
    if {
        (row["arm"], int(row["pipeline_seed"]), row["split"]) for row in by_seed
    } != {
        (arm, seed, split)
        for arm in ARMS
        for seed in SEEDS
        for split in ("source_test", "fixed_variant", "sensitivity")
    }:
        raise ValueError("by-seed cells are incomplete")
    expected_summary_rows = 3 * len(SUMMARY_METRICS) * (
        len(ARMS) + len(CONTRASTS)
    )
    if len(summary) != expected_summary_rows:
        raise ValueError(f"summary row count mismatch: {len(summary)}")
    seed_sensitivity = [
        row for row in sensitivity_rows if row["row_type"] == "seed"
    ]
    if len(seed_sensitivity) != 15 * 20 * 2:
        raise ValueError(
            f"sensitivity seed-row count mismatch: {len(seed_sensitivity)}"
        )
    if len(checkpoint_selection) != 180:
        raise ValueError("checkpoint selection row count mismatch")
    finite_columns = [
        "accuracy_binary",
        "macro_f1_binary",
        "fpr",
        "fnr",
        "normal_recall",
        "binary_attack_recall",
        "exact_attack_recall",
    ]
    for row in by_seed:
        for column in finite_columns:
            if not math.isfinite(float(row[column])):
                raise ValueError(f"non-finite {column} in by-seed table")
            if not 0.0 <= float(row[column]) <= 1.0:
                raise ValueError(f"out-of-range {column} in by-seed table")
    return {
        "by_seed_rows": len(by_seed),
        "summary_rows": len(summary),
        "sensitivity_rows": len(sensitivity_rows),
        "sensitivity_seed_rows": len(seed_sensitivity),
        "checkpoint_selection_rows": len(checkpoint_selection),
        "all_required_arm_seed_split_cells_present": True,
        "all_primary_metrics_finite_and_bounded": True,
    }


def verify_existing_final_manifest() -> None:
    manifest = load_json(FINAL_MANIFEST_PATH)
    if manifest.get("version") != VERSION or manifest.get("contract_id") != CONTRACT_ID:
        raise ValueError("existing final manifest contract mismatch")
    for entry in manifest["outputs"]:
        path = REPO_ROOT / entry["path"]
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"existing final output mismatch: {entry['path']}")


def run_evaluation() -> None:
    data_entries, dependency_entries = audit_frozen_files()
    implementation_commit, script_hash = verify_script_state(False)
    run_unit_tests()
    if FINAL_MANIFEST_PATH.exists():
        verify_existing_final_manifest()
        print(
            json.dumps(
                {
                    "status": "already_complete_and_verified",
                    "manifest": relative(FINAL_MANIFEST_PATH),
                }
            ),
            flush=True,
        )
        return
    partial_final = [
        path
        for path in OUTPUT_TABLES.values()
        if path.exists()
    ]
    if partial_final:
        raise ValueError(
            "partial final tables exist without a final manifest: "
            + ", ".join(relative(path) for path in partial_final)
        )

    training_manifest, runs = validate_training_manifest(
        script_hash=script_hash,
        implementation_commit=implementation_commit,
    )
    started_utc = utc_now()
    started = time.monotonic()
    log_event(
        "evaluation_phase_started",
        fits_verified=len(runs),
        source_test_variant_or_sensitivity_first_loaded_after_all_fits=True,
    )
    with np.load(STANDARDIZER_PATH, allow_pickle=True) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float32)
        std = np.asarray(archive["std"], dtype=np.float32)
        if tuple(archive["feature_names"].tolist()) != FEATURE_NAMES:
            raise ValueError("camera-ready standardizer feature order mismatch")
    datasets, labels, sensitivity_metadata = load_evaluation_data(mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_seed: list[dict] = []
    sensitivity_seed_rows: list[dict] = []
    revalidation = []
    run_lookup = {
        (record["arm"], int(record["pipeline_seed"])): record for record in runs
    }
    for seed in SEEDS:
        loaders = build_eval_loaders(datasets, seed)
        for arm in ARMS:
            record = run_lookup[(arm, seed)]
            payload = torch.load(
                checkpoint_path(arm, seed),
                map_location="cpu",
                weights_only=False,
            )
            model = CNN1D().to(device)
            model.load_state_dict(payload["state_dict"])
            predictions = {
                name: predict_labels(model, loaders[name], device)
                for name in (
                    "validation",
                    "source_test",
                    "fixed_variant",
                    "sensitivity",
                )
            }
            val_value = float(
                f1_score(
                    labels["validation"],
                    predictions["validation"],
                    average="macro",
                    zero_division=0,
                )
            )
            if not math.isclose(
                val_value,
                float(record["selected_val_macro_f1_multiclass"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"post-training validation mismatch for {arm} seed {seed}"
                )
            revalidation.append(
                {
                    "arm": arm,
                    "pipeline_seed": seed,
                    "selected_update": int(record["selected_update"]),
                    "recorded_val_macro_f1_multiclass": float(
                        record["selected_val_macro_f1_multiclass"]
                    ),
                    "recomputed_val_macro_f1_multiclass": val_value,
                }
            )
            for split in ("source_test", "fixed_variant", "sensitivity"):
                row = {
                    "arm": arm,
                    "pipeline_seed": seed,
                    "split": split,
                    "updates_run": MAX_UPDATES,
                    "selected_update": int(record["selected_update"]),
                    "selected_val_macro_f1_multiclass": val_value,
                    **classification_metrics(
                        labels[split], predictions[split]
                    ),
                }
                by_seed.append(row)
            sensitivity_seed_rows.extend(
                build_sensitivity_seed_rows(
                    arm,
                    seed,
                    labels["sensitivity"],
                    predictions["sensitivity"],
                    sensitivity_metadata,
                )
            )
            log_event(
                "checkpoint_evaluated",
                arm=arm,
                pipeline_seed=seed,
                selected_update=int(record["selected_update"]),
                source_test_macro_f1_binary=by_seed[-3]["macro_f1_binary"],
                source_test_fpr=by_seed[-3]["fpr"],
                fixed_variant_binary_attack_recall=by_seed[-2][
                    "binary_attack_recall"
                ],
                sensitivity_binary_attack_recall=by_seed[-1][
                    "binary_attack_recall"
                ],
            )
            del model, predictions, payload
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del loaders
        gc.collect()

    summary = summarize_by_seed(by_seed)
    sensitivity_rows = summarize_sensitivity(sensitivity_seed_rows)
    checkpoint_selection = checkpoint_rows(runs)
    table_validation = validate_output_tables(
        by_seed, summary, sensitivity_rows, checkpoint_selection
    )

    atomic_write_csv(OUTPUT_TABLES["checkpoint_selection"], checkpoint_selection)
    atomic_write_csv(OUTPUT_TABLES["by_seed"], by_seed)
    atomic_write_csv(OUTPUT_TABLES["summary"], summary)
    atomic_write_csv(OUTPUT_TABLES["sensitivity"], sensitivity_rows)
    elapsed = time.monotonic() - started
    log_event(
        "evaluation_phase_completed",
        runtime_seconds=elapsed,
        by_seed_rows=len(by_seed),
        summary_rows=len(summary),
        sensitivity_rows=len(sensitivity_rows),
        checkpoint_selection_rows=len(checkpoint_selection),
    )

    output_entries = [
        file_entry(OUTPUT_TABLES[key])
        for key in ("checkpoint_selection", "by_seed", "summary", "sensitivity")
    ]
    output_entries.append(file_entry(LOG_PATH))
    input_entries = data_entries + dependency_entries
    final_manifest = {
        "experiment": VERSION,
        "version": VERSION,
        "contract_id": CONTRACT_ID,
        "analysis_type": "post-review matched optimizer-update training-policy sensitivity",
        "accepted_source_commit": ACCEPTED_SOURCE_COMMIT,
        "analysis_plan": {
            "path": relative(PLAN_PATH),
            "commit": PLAN_COMMIT,
            "sha256": PLAN_SHA256,
            "frozen_before_new_matched_update_outcomes": True,
        },
        "implementation": {
            "path": relative(Path(__file__).resolve()),
            "commit": implementation_commit,
            "sha256": script_hash,
            "commands": [
                "source .venv/bin/activate && python wisa_camera_ready/scripts/run_matched_update_v1.py train",
                "source .venv/bin/activate && python wisa_camera_ready/scripts/run_matched_update_v1.py evaluate",
            ],
        },
        "contract": training_contract(),
        "training_manifest": file_entry(TRAINING_MANIFEST_PATH),
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "evaluation_runtime_seconds": elapsed,
        "training_runtime_seconds": training_manifest["runtime_seconds"],
        "environment": environment_info(),
        "inputs": input_entries,
        "evaluation_counts": {
            split: json_label_counts(values)
            for split, values in labels.items()
            if split != "validation"
        },
        "sensitivity_scenario_counts": {
            str(key): int(value)
            for key, value in zip(
                *np.unique(
                    sensitivity_metadata["scenario_id"], return_counts=True
                )
            )
        },
        "selected_checkpoint_revalidation": revalidation,
        "validation_gates": {
            **table_validation,
            "fits_expected": 15,
            "fits_verified_before_evaluation": len(runs),
            "all_fits_ran_7992_updates": all(
                int(record["updates_run"]) == MAX_UPDATES for record in runs
            ),
            "all_selected_checkpoints_revalidated": len(revalidation) == 15,
            "evaluation_threshold_selected": False,
            "external_data_loaded": False,
            "wisa_tree_clean": not bool(
                git_text("status", "--short", "--", "wisa")
            ),
        },
        "outputs": output_entries,
        "warnings": [
            "The five repeats are paired pipeline seeds, not independent captures, vehicles, generator constructions, or population draws.",
            "Sensitivity low/medium/high rows are coupled parameter settings.",
            "Arm-specific balanced class weights are part of the retained accepted-paper policy.",
            "No external dataset is evaluated in this experiment.",
        ],
    }
    atomic_write_json(FINAL_MANIFEST_PATH, final_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": relative(FINAL_MANIFEST_PATH),
                "manifest_sha256": sha256_file(FINAL_MANIFEST_PATH),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_preflight(*, allow_uncommitted: bool) -> None:
    started = time.monotonic()
    (
        data,
        sampling,
        data_entries,
        dependency_entries,
        implementation_commit,
        script_hash,
    ) = preflight(allow_uncommitted=allow_uncommitted)
    result = {
        "status": "preflight_passed",
        "contract_id": CONTRACT_ID,
        "implementation_commit": implementation_commit,
        "implementation_sha256": script_hash,
        "data_files_verified": len(data_entries),
        "dependency_files_verified": len(dependency_entries),
        "sampling_cells_verified": len(sampling),
        "real_train_windows": len(data.real_y),
        "validation_windows": len(data.val_y),
        "rule_pool_windows": len(data.rule_y),
        "standardizer_equal_to_frozen": True,
        "unit_tests_passed": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("preflight", "train", "evaluate", "all")
    )
    parser.add_argument(
        "--allow-uncommitted-code",
        action="store_true",
        help="Only for pre-commit preflight; full training always refuses it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_uncommitted_code and args.phase != "preflight":
        raise SystemExit("--allow-uncommitted-code is valid only for preflight")
    if args.phase == "preflight":
        run_preflight(allow_uncommitted=args.allow_uncommitted_code)
    elif args.phase == "train":
        run_training()
    elif args.phase == "evaluate":
        run_evaluation()
    elif args.phase == "all":
        run_training()
        run_evaluation()
    else:
        raise AssertionError(args.phase)


if __name__ == "__main__":
    main()
