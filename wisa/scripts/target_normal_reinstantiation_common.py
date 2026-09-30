#!/usr/bin/env python3
"""Shared helpers for target-normal reinstantiation experiments."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from run_cantt_tier1 import list_cantt_files

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"

EXP = ROOT / "experiments" / "10_target_normal_reinstantiation"
SUPPORT_EXP = ROOT / "experiments" / "11_support_factor_decomposition"
MODELS = ROOT / "models" / "target_normal_reinstantiation"
SUPPORT_MODELS = ROOT / "models" / "support_factor_decomposition"

CALIB_NORMAL_CACHE = EXP / "calibration_normal_windows.npz"
CALIB_ATTACK_CACHE = EXP / "calibration_attack_windows_oracle.npz"
SPLIT_MANIFEST = EXP / "tnr_file_split_manifest.csv"
TARGET_SYNTH_PATH = SYNTHETIC / "target_normal_rule_windows.npz"

SEEDS = [7, 42, 123]
RULE_RATIO = 0.30
CALIB_NORMAL_CAP = 100_000
CALIB_ATTACK_CAP = 100_000
SUBSAMPLE_SEED = 2026

ARM_CONFIGS = {
    "source_real": {
        "label": "Source real",
        "group": "source_reference",
        "source_rule": False,
        "target_normal": False,
        "target_synth": False,
        "oracle_attack": False,
    },
    "source_rule30": {
        "label": "Source rule +30%",
        "group": "source_reference",
        "source_rule": True,
        "target_normal": False,
        "target_synth": False,
        "oracle_attack": False,
    },
    "target_normal": {
        "label": "+Target normal",
        "group": "target_diagnostic",
        "source_rule": False,
        "target_normal": True,
        "target_synth": False,
        "oracle_attack": False,
    },
    "target_synth_only": {
        "label": "+Target synthetic only",
        "group": "target_diagnostic",
        "source_rule": False,
        "target_normal": False,
        "target_synth": True,
        "oracle_attack": False,
    },
    "target_normal_target_synth": {
        "label": "+Target normal + target synthetic",
        "group": "target_diagnostic",
        "source_rule": False,
        "target_normal": True,
        "target_synth": True,
        "oracle_attack": False,
    },
    "target_normal_source_rule30": {
        "label": "+Target normal + source rule +30%",
        "group": "target_diagnostic",
        "source_rule": True,
        "target_normal": True,
        "target_synth": False,
        "oracle_attack": False,
    },
    "target_normal_real_attack_oracle": {
        "label": "+Target normal + real attack oracle",
        "group": "oracle_diagnostic",
        "source_rule": False,
        "target_normal": True,
        "target_synth": False,
        "oracle_attack": True,
    },
}

MAIN_ARMS = [
    "source_real",
    "source_rule30",
    "target_normal",
    "target_synth_only",
    "target_normal_target_synth",
    "target_normal_source_rule30",
]

ORACLE_ARMS = ["target_normal_real_attack_oracle"]

ATTACK_TYPE_IDS = {"DoS": 1, "Fuzzy": 2, "Gear": 3, "RPM": 4}
ORACLE_STEM_TO_ATTACK = {
    "DoS": 1,
    "fuzzing": 2,
    "rpm": 4,
    "force-neutral": 3,
    "rpm-accessory": 4,
}


def ensure_dirs() -> None:
    for path in [EXP, SUPPORT_EXP, MODELS, SUPPORT_MODELS, TABLES, FIGURES, LOGS, SYNTHETIC]:
        path.mkdir(parents=True, exist_ok=True)


def repo_rel(path: Path) -> str:
    """Return a repository-relative path for portable artifact logs."""
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_cantt_files():
    """Deterministic capture/file split over can-train test_* files."""
    test_files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    calibration = [f for i, f in enumerate(test_files) if i % 5 == 0]
    held_out = [f for i, f in enumerate(test_files) if i % 5 != 0]
    return calibration, held_out


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> list[dict]:
    rows: list[dict] = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        if "seed" in group:
            row["seeds"] = ";".join(str(int(s)) for s in sorted(group["seed"].dropna().unique()))
            row["n_seeds"] = int(group["seed"].nunique())
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals):
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def sample_rows(x: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, int]:
    total = len(x)
    if total <= cap:
        return x.astype(np.float32), total
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=cap, replace=False)
    idx.sort()
    return x[idx].astype(np.float32), total


def feature_validity(x: np.ndarray) -> dict:
    return {
        "can_id_min": float(x[:, :, 0].min()) if len(x) else "",
        "can_id_max": float(x[:, :, 0].max()) if len(x) else "",
        "dlc_min": float(x[:, :, 1].min()) if len(x) else "",
        "dlc_max": float(x[:, :, 1].max()) if len(x) else "",
        "payload_min": float(x[:, :, 2:10].min()) if len(x) else "",
        "payload_max": float(x[:, :, 2:10].max()) if len(x) else "",
        "delta_t_min": float(x[:, :, 10].min()) if len(x) else "",
        "delta_t_max": float(x[:, :, 10].max()) if len(x) else "",
        "invalid_can_id": int(((x[:, :, 0] < 0) | (x[:, :, 0] > 2047)).sum()) if len(x) else 0,
        "invalid_dlc": int(((x[:, :, 1] < 0) | (x[:, :, 1] > 8)).sum()) if len(x) else 0,
        "invalid_payload": int(((x[:, :, 2:10] < 0) | (x[:, :, 2:10] > 255)).sum()) if len(x) else 0,
        "invalid_delta_t": int((x[:, :, 10] < 0).sum()) if len(x) else 0,
    }
