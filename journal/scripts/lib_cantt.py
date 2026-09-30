#!/usr/bin/env python3
"""can-train-and-test zip-streaming helpers for the journal extension pipeline.

Forked from wisa/scripts/run_cantt_tier1.py @ commit eb131df (selected helpers
only; behavior identical). External evaluation-only per AGENTS.md data rules.
"""
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]  # journal/
ZIP_PATH = ROOT / "datasets" / "can-train-and-test.zip"

FEATURE_NAMES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]
WINDOW_SIZE = 128
STRIDE = 32

VEHICLES = {
    "set_01": {"known": "2011 Chevrolet Impala", "unknown": "2016 Chevrolet Silverado"},
    "set_02": {"known": "2011 Chevrolet Traverse", "unknown": "2017 Subaru Forester"},
    "set_03": {"known": "2016 Chevrolet Silverado", "unknown": "2017 Subaru Forester"},
    "set_04": {"known": "2017 Subaru Forester", "unknown": "2011 Chevrolet Traverse"},
}


@dataclass(frozen=True)
class CanttFile:
    path: str
    set_id: str
    subset_id: str
    source_stem: str
    file_size: int
    vehicle_axis: str
    attack_axis: str
    vehicle: str


class MetricCounter:
    def __init__(self) -> None:
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y = y_true.astype(bool)
        p = y_pred.astype(bool)
        self.tn += int((~y & ~p).sum())
        self.fp += int((~y & p).sum())
        self.fn += int((y & ~p).sum())
        self.tp += int((y & p).sum())

    def row(self) -> dict:
        total = self.tn + self.fp + self.fn + self.tp
        y_true = np.array([0] * (self.tn + self.fp) + [1] * (self.fn + self.tp), dtype=np.int8)
        y_pred = np.array([0] * self.tn + [1] * self.fp + [0] * self.fn + [1] * self.tp, dtype=np.int8)
        return {
            "windows": total,
            "normal_windows": self.tn + self.fp,
            "attack_windows": self.fn + self.tp,
            "accuracy": (self.tn + self.tp) / total if total else "",
            "macro_f1_binary": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total else "",
            "normal_recall": self.tn / (self.tn + self.fp) if (self.tn + self.fp) else "",
            "fpr": self.fp / (self.tn + self.fp) if (self.tn + self.fp) else "",
            "attack_recall": self.tp / (self.tp + self.fn) if (self.tp + self.fn) else "",
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def parse_source_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"-\d+$", "", stem)


def parse_file_meta(info: zipfile.ZipInfo) -> CanttFile | None:
    parts = Path(info.filename).parts
    if len(parts) != 4 or not info.filename.endswith(".csv"):
        return None
    _, set_id, subset_id, filename = parts
    source_stem = parse_source_stem(filename)
    if subset_id == "train_01":
        vehicle_axis = "train"
        attack_axis = "train"
        vehicle = VEHICLES[set_id]["known"]
    else:
        vehicle_axis = "unknown_vehicle" if "unknown_vehicle" in subset_id else "known_vehicle"
        attack_axis = "unknown_attack" if "unknown_attack" in subset_id else "known_attack"
        vehicle = VEHICLES[set_id]["unknown" if vehicle_axis == "unknown_vehicle" else "known"]
    return CanttFile(info.filename, set_id, subset_id, source_stem, info.file_size, vehicle_axis, attack_axis, vehicle)


def list_cantt_files() -> list[CanttFile]:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        files = [parse_file_meta(info) for info in zf.infolist()]
    return sorted([f for f in files if f is not None], key=lambda f: f.path)


def payload_to_features(payloads: pd.Series) -> tuple[np.ndarray, np.ndarray, Counter]:
    values = payloads.fillna("").astype(str).to_numpy()
    n = len(values)
    data = np.zeros((n, 8), dtype=np.uint8)
    dlc = np.zeros(n, dtype=np.uint8)
    issues = Counter()
    for i, text in enumerate(values):
        s = text.strip()
        if len(s) % 2:
            issues["odd_length_payload"] += 1
            s = s[:-1]
        try:
            raw = bytes.fromhex(s)
        except ValueError:
            issues["malformed_payload"] += 1
            raw = b""
        if len(raw) > 8:
            issues["dlc_gt_8"] += 1
        dlc[i] = min(len(raw), 255)
        if raw:
            data[i, : min(len(raw), 8)] = np.frombuffer(raw[:8], dtype=np.uint8)
    return dlc, data, issues


def read_cantt_csv(zf: zipfile.ZipFile, meta: CanttFile) -> tuple[pd.DataFrame, np.ndarray, Counter]:
    with zf.open(meta.path) as fh:
        df = pd.read_csv(
            fh,
            dtype={"timestamp": "float64", "arbitration_id": "string", "data_field": "string", "attack": "int8"},
        )
    dlc, data, issues = payload_to_features(df["data_field"])
    can_id = np.array([int(str(v), 16) for v in df["arbitration_id"].fillna("0").to_numpy()], dtype=np.int32)
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    dt = np.zeros(len(df), dtype=np.float32)
    if len(df) > 1:
        diff = np.diff(timestamps)
        issues["timestamp_decrease"] += int((diff < 0).sum())
        dt[1:] = np.maximum(diff, 0.0).astype(np.float32)
    features = np.column_stack([can_id.astype(np.float32), dlc.astype(np.float32), data.astype(np.float32), dt])
    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "can_id": can_id,
            "dlc": dlc.astype(np.int16),
            "attack": df["attack"].to_numpy(dtype=np.int8),
        }
    )
    for i in range(8):
        out[f"data{i}"] = data[:, i]
    out["delta_t"] = dt
    return out, features.astype(np.float32), issues


def iter_windows(features: np.ndarray, frame_y: np.ndarray, batch_size: int = 4096):
    n = len(features)
    if n < WINDOW_SIZE:
        return
    starts = np.arange(0, n - WINDOW_SIZE + 1, STRIDE, dtype=np.int64)
    csum = np.concatenate([[0], np.cumsum(frame_y.astype(np.int32))])
    win_y = (csum[starts + WINDOW_SIZE] - csum[starts] > 0).astype(np.int8)
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        x = np.stack([features[s : s + WINDOW_SIZE] for s in batch_starts]).astype(np.float32)
        yield x, win_y[i : i + len(batch_starts)], batch_starts
