#!/usr/bin/env python3
"""Post-review external ranking diagnostic for the WISA 2026 camera-ready.

This evaluator is intentionally read-only with respect to the frozen ``wisa/``
tree. It scores the accepted-paper CNN checkpoints on OTIDS,
can-train-and-test, and ROAD, then publishes a complete v1 result set
atomically under ``wisa_camera_ready/``.

External labels are used only for diagnostic evaluation. They do not enter
training, validation, checkpoint/scaler selection, or setting selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


CAMERA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMERA_ROOT.parent
WISA_ROOT = REPO_ROOT / "wisa"
WISA_SCRIPTS = WISA_ROOT / "scripts"
if str(WISA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(WISA_SCRIPTS))

import run_cantt_tier1 as cantt  # noqa: E402
from train_real_only_baselines import CNN1D  # noqa: E402


VERSION = "external_ranking_v1"
PLAN_COMMIT = "18b2176140a0458509eaf10c9e0bb1e129b15017"
ACCEPTED_SOURCE_COMMIT = "57880aa350ddf0f916b7e1bd2e940529d0fc6ea6"
EXPECTED_MODEL_SCALER_INVENTORY_SHA256 = (
    "9bfae8a7c58a639cc2f42e414061c6fea9058b447e67c1566a085c41c54f0e17"
)
EXPECTED_DATASET_HASHES = {
    "datasets/windows/otids_cross_windows.npz": (
        43_289_129,
        "b521f3546f633c6220e2696234abb6d7c37a03d64f19e71ec37bf36a100ed596",
    ),
    "datasets/can-train-and-test.zip": (
        1_507_455_719,
        "a9c607b38bd28f1768021ad01c29ffbfe4e82bb0ae5815ac3ce7ad74751ae061",
    ),
}
EXPECTED_ROAD_TREE_SHA256 = (
    "a2c0471df17466e7cbc0b9a5b997a6e49cf6a59c413259c39719313ed1510156"
)
EXPECTED_DEPENDENCY_HASHES = {
    "wisa/scripts/train_real_only_baselines.py": (
        "5f4d423a07c573d69210034e752b2a847eb4b87486907cd8ea81aee06d9aef31"
    ),
    "wisa/scripts/run_cantt_tier1.py": (
        "6f7fd11fe17c727889c5d0ec56fce199e1656a3fabf22a8f061fd217de9a114e"
    ),
    "wisa/results/tables/road_dataset_profile.csv": (
        "0d6202ae75e7e8e883ef5fb40fd205c205803b25307932c2b2bd9cf0eef7df83"
    ),
}

SEEDS = (7, 42, 123)
TARGET_FPRS = (0.001, 0.01)
BOOTSTRAP_SEED = 20_260_728
BOOTSTRAP_REPLICATES = 2_000
SCORE_DEFINITION = "1 - p(normal) from unchanged five-class CNN softmax"
SOURCE_POLICY = "attack iff five-class argmax != normal class 0"
WINDOW_SIZE = 128
STRIDE = 32
RAW_WINDOW_BATCH = 4_096
INFERENCE_BATCH = 1_024

SETTING_SPECS = (
    ("real_only", "Real only", "real_only", None),
    (
        "real_oversampling_0p50",
        "Real oversampling +50%",
        "oversampling",
        "0p50",
    ),
    ("rule_0p10", "Rule +10%", "rule", "0p10"),
    ("rule_0p30", "Rule +30%", "rule", "0p30"),
    ("rule_0p50", "Rule +50%", "rule", "0p50"),
    ("rule_1p00", "Rule +100%", "rule", "1p00"),
)

OUTPUT_PATHS = {
    "by_seed": CAMERA_ROOT
    / "results"
    / "tables"
    / "external_ranking_by_seed_v1.csv",
    "summary": CAMERA_ROOT
    / "results"
    / "tables"
    / "external_ranking_summary_v1.csv",
    "by_group": CAMERA_ROOT
    / "results"
    / "tables"
    / "external_low_fpr_by_group_v1.csv",
    "log": CAMERA_ROOT / "results" / "logs" / "external_ranking_v1.log",
    "manifest": CAMERA_ROOT
    / "experiments"
    / "01_external_ranking"
    / "manifest_v1.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def file_entry(path: Path) -> dict:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_inventory_digest(entries: list[dict]) -> str:
    lines = [
        f"{entry['path']}\t{entry['bytes']}\t{entry['sha256']}\n"
        for entry in sorted(entries, key=lambda item: item["path"])
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def model_and_scaler_paths(
    setting: str, family: str, ratio_tag: str | None, seed: int
) -> tuple[Path, Path]:
    if family == "real_only":
        return (
            WISA_ROOT / "models" / "baseline" / f"cnn1d_real_only_seed{seed}.pt",
            WISA_ROOT / "models" / "baseline" / "cnn_standardizer.npz",
        )
    if family == "oversampling":
        stem = f"real_oversampling_ratio0p50_seed{seed}"
        return (
            WISA_ROOT / "models" / "oversampling" / f"cnn1d_{stem}.pt",
            WISA_ROOT / "models" / "oversampling" / f"standardizer_{stem}.npz",
        )
    if family == "rule":
        stem = f"rule_ratio{ratio_tag}_seed{seed}"
        return (
            WISA_ROOT / "models" / "ratio_sweep" / f"cnn1d_{stem}.pt",
            WISA_ROOT / "models" / "ratio_sweep" / f"standardizer_{stem}.npz",
        )
    raise ValueError(f"unknown model family: {family} ({setting})")


@dataclass
class ModelInfo:
    setting: str
    label: str
    seed: int
    model_path: Path
    scaler_path: Path
    model: CNN1D
    mean: np.ndarray
    std: np.ndarray
    scaler_key: str

    @property
    def key(self) -> tuple[str, int]:
        return self.setting, self.seed


@dataclass
class Confusion:
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    def update(self, y_true: np.ndarray, predicted_attack: np.ndarray) -> None:
        y = np.asarray(y_true, dtype=bool)
        pred = np.asarray(predicted_attack, dtype=bool)
        if y.shape != pred.shape:
            raise ValueError(f"label/prediction shape mismatch: {y.shape} != {pred.shape}")
        self.tn += int((~y & ~pred).sum())
        self.fp += int((~y & pred).sum())
        self.fn += int((y & ~pred).sum())
        self.tp += int((y & pred).sum())

    def add(self, other: "Confusion") -> None:
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        self.tp += other.tp

    def row(self) -> dict:
        normal = self.tn + self.fp
        attack = self.fn + self.tp
        return {
            "source_policy_tn": self.tn,
            "source_policy_fp": self.fp,
            "source_policy_fn": self.fn,
            "source_policy_tp": self.tp,
            "source_policy_fpr": self.fp / normal if normal else math.nan,
            "source_policy_normal_recall": self.tn / normal if normal else math.nan,
            "source_policy_attack_recall": self.tp / attack if attack else math.nan,
        }


class EventLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, event: str, **fields: object) -> None:
        record = {"utc": utc_now(), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        self.lines.append(line)
        print(line, flush=True)


def load_models(device: torch.device, log: EventLog) -> tuple[list[ModelInfo], list[dict]]:
    models: list[ModelInfo] = []
    artifact_paths: set[Path] = set()
    for setting, label, family, ratio_tag in SETTING_SPECS:
        for seed in SEEDS:
            model_path, scaler_path = model_and_scaler_paths(
                setting, family, ratio_tag, seed
            )
            if not model_path.is_file() or not scaler_path.is_file():
                raise FileNotFoundError(
                    f"missing frozen input for {setting}, seed={seed}: "
                    f"{model_path}, {scaler_path}"
                )
            artifact_paths.update((model_path, scaler_path))
            scaler = np.load(scaler_path, allow_pickle=False)
            mean = np.asarray(scaler["mean"], dtype=np.float32)
            std = np.asarray(scaler["std"], dtype=np.float32)
            if mean.size != len(cantt.FEATURE_NAMES) or std.size != len(
                cantt.FEATURE_NAMES
            ):
                raise ValueError(f"unexpected scaler shape: {scaler_path}")
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise ValueError(f"non-finite scaler: {scaler_path}")
            if np.any(std <= 0):
                raise ValueError(f"non-positive scaler standard deviation: {scaler_path}")
            scaler_key = hashlib.sha256(
                mean.tobytes(order="C") + std.tobytes(order="C")
            ).hexdigest()
            model = CNN1D(in_channels=11, classes=5).to(device)
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            models.append(
                ModelInfo(
                    setting=setting,
                    label=label,
                    seed=seed,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    model=model,
                    mean=mean,
                    std=std,
                    scaler_key=scaler_key,
                )
            )

    entries = [file_entry(path) for path in sorted(artifact_paths)]
    inventory_digest = canonical_inventory_digest(entries)
    if inventory_digest != EXPECTED_MODEL_SCALER_INVENTORY_SHA256:
        raise RuntimeError(
            "checkpoint/scaler inventory hash mismatch: "
            f"{inventory_digest} != {EXPECTED_MODEL_SCALER_INVENTORY_SHA256}"
        )
    if len(models) != len(SETTING_SPECS) * len(SEEDS):
        raise AssertionError(f"expected 18 models, loaded {len(models)}")
    log.emit(
        "models_loaded",
        models=len(models),
        unique_artifacts=len(entries),
        unique_scalers=len({model.scaler_key for model in models}),
        inventory_sha256=inventory_digest,
        device=str(device),
    )
    return models, entries


def score_raw_batch(
    models: list[ModelInfo], x_raw: np.ndarray, device: torch.device
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    x = np.asarray(x_raw, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, len(cantt.FEATURE_NAMES)):
        raise ValueError(f"unexpected raw window shape: {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("raw feature batch contains non-finite values")

    by_scaler: dict[str, list[ModelInfo]] = defaultdict(list)
    for model in models:
        by_scaler[model.scaler_key].append(model)
    output: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for scaler_models in by_scaler.values():
            exemplar = scaler_models[0]
            x_std = ((x - exemplar.mean) / exemplar.std).astype(
                np.float32, copy=False
            )
            score_arrays = {
                model.key: np.empty(len(x_std), dtype=np.float32)
                for model in scaler_models
            }
            pred_arrays = {
                model.key: np.empty(len(x_std), dtype=np.int8)
                for model in scaler_models
            }
            for start in range(0, len(x_std), INFERENCE_BATCH):
                stop = min(start + INFERENCE_BATCH, len(x_std))
                tensor = torch.from_numpy(
                    np.ascontiguousarray(x_std[start:stop].transpose(0, 2, 1))
                ).to(device)
                for model in scaler_models:
                    probs = torch.softmax(model.model(tensor), dim=1)
                    score_arrays[model.key][start:stop] = (
                        1.0 - probs[:, 0]
                    ).cpu().numpy()
                    pred_arrays[model.key][start:stop] = (
                        probs.argmax(dim=1) != 0
                    ).to(torch.int8).cpu().numpy()
            for model in scaler_models:
                scores = score_arrays[model.key]
                if not np.isfinite(scores).all():
                    raise ValueError(
                        f"non-finite score for {model.setting}, seed={model.seed}"
                    )
                if np.any(scores < 0.0) or np.any(scores > 1.0):
                    raise ValueError(
                        f"score outside [0,1] for {model.setting}, seed={model.seed}"
                    )
                output[model.key] = scores, pred_arrays[model.key]
    if len(output) != len(models):
        raise AssertionError(f"expected {len(models)} score arrays, got {len(output)}")
    return output


def attainable_low_fpr_points(
    y_true: np.ndarray, scores: np.ndarray, targets: Iterable[float] = TARGET_FPRS
) -> dict[float, dict]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != score.shape:
        raise ValueError(f"label/score shape mismatch: {y.shape} != {score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("non-finite scores")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"non-binary labels: {np.unique(y)}")
    normal_scores = score[y == 0]
    attack_scores = score[y == 1]
    if not len(normal_scores):
        return {
            float(target): {
                "target_fpr": float(target),
                "max_allowed_fp": 0,
                "achieved_fp": 0,
                "achieved_fpr": math.nan,
                "threshold": math.nan,
                "tpr": math.nan,
                "next_excluded_score": math.nan,
                "next_excluded_normal_tie_count": 0,
                "next_excluded_attack_tie_count": 0,
                "unused_fp_capacity": 0,
                "tie_step_would_exceed_ceiling": "",
                "status": "undefined_no_normal_windows",
            }
            for target in targets
        }

    unique_asc, counts_asc = np.unique(normal_scores, return_counts=True)
    values_desc = unique_asc[::-1]
    counts_desc = counts_asc[::-1].astype(np.int64)
    cumulative = np.cumsum(counts_desc)
    result: dict[float, dict] = {}
    for target in targets:
        alpha = float(target)
        allowed = int(math.floor(alpha * len(normal_scores)))
        valid = np.flatnonzero(cumulative <= allowed)
        included_groups = int(valid[-1] + 1) if len(valid) else 0
        achieved_fp = int(cumulative[included_groups - 1]) if included_groups else 0

        if included_groups < len(values_desc):
            excluded_score = float(values_desc[included_groups])
            threshold = float(np.nextafter(excluded_score, math.inf))
            normal_tie = int(counts_desc[included_groups])
            attack_tie = int(np.count_nonzero(attack_scores == excluded_score))
            more_permissive_fp = achieved_fp + normal_tie
            tie_would_exceed = more_permissive_fp > allowed
            status = (
                "attainable_boundary_tie"
                if normal_tie > 1 or attack_tie > 0
                else "attainable"
            )
        else:
            excluded_score = math.nan
            threshold = float(-math.inf)
            normal_tie = 0
            attack_tie = 0
            tie_would_exceed = False
            status = "all_normal_scores_included"

        achieved_fpr = achieved_fp / len(normal_scores)
        if achieved_fpr > alpha + 1e-15:
            raise AssertionError(
                f"fixed-FPR ceiling violated: {achieved_fpr} > {alpha}"
            )
        tpr = (
            float(np.mean(attack_scores >= threshold))
            if len(attack_scores)
            else math.nan
        )
        result[alpha] = {
            "target_fpr": alpha,
            "max_allowed_fp": allowed,
            "achieved_fp": achieved_fp,
            "achieved_fpr": achieved_fpr,
            "threshold": threshold,
            "tpr": tpr,
            "next_excluded_score": excluded_score,
            "next_excluded_normal_tie_count": normal_tie,
            "next_excluded_attack_tie_count": attack_tie,
            "unused_fp_capacity": allowed - achieved_fp,
            "tie_step_would_exceed_ceiling": tie_would_exceed,
            "status": status,
        }

    ordered = [result[float(target)] for target in sorted(targets)]
    for lower, upper in zip(ordered, ordered[1:]):
        if lower["achieved_fpr"] > upper["achieved_fpr"] + 1e-15:
            raise AssertionError("achieved FPR is not monotonic across ceilings")
        if (
            math.isfinite(lower["tpr"])
            and math.isfinite(upper["tpr"])
            and lower["tpr"] > upper["tpr"] + 1e-15
        ):
            raise AssertionError("TPR is not monotonic across ceilings")
    return result


def ranking_row(
    y_true: np.ndarray, scores: np.ndarray, confusion: Confusion
) -> dict:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != score.shape:
        raise ValueError(f"label/score shape mismatch: {y.shape} != {score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("non-finite pooled score")
    normal = int((y == 0).sum())
    attack = int((y == 1).sum())
    row = {
        "windows": len(y),
        "normal_windows": normal,
        "attack_windows": attack,
        "attack_prevalence": attack / len(y) if len(y) else math.nan,
        **confusion.row(),
        "roc_auc": math.nan,
        "average_precision_pr_auc": math.nan,
        "ranking_status": "undefined_empty",
        "unique_score_count": int(np.unique(score).size) if len(score) else 0,
        "score_min": float(score.min()) if len(score) else math.nan,
        "score_max": float(score.max()) if len(score) else math.nan,
    }
    if normal and attack:
        roc_fpr, _roc_tpr, _ = roc_curve(y, score)
        if np.any(np.diff(roc_fpr) < -1e-15):
            raise AssertionError("ROC FPR input is not monotonic")
        _precision, pr_recall, _ = precision_recall_curve(y, score)
        if np.any(np.diff(pr_recall) > 1e-15):
            raise AssertionError("PR recall input is not monotonic")
        row["roc_auc"] = float(roc_auc_score(y, score))
        row["average_precision_pr_auc"] = float(
            average_precision_score(y, score)
        )
        row["ranking_status"] = "ok"
    elif normal:
        row["ranking_status"] = "undefined_no_attack_windows"
    elif attack:
        row["ranking_status"] = "undefined_no_normal_windows"

    points = attainable_low_fpr_points(y, score)
    for target, point in sorted(points.items()):
        tag = "0p001" if math.isclose(target, 0.001) else "0p01"
        for field, value in point.items():
            if field == "target_fpr":
                continue
            row[f"{field}_at_fpr_{tag}"] = value
    return row


def run_self_tests() -> dict:
    y = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    score = np.array([0.9, 0.8, 0.8, 0.1, 0.85, 0.8, 0.05])
    points = attainable_low_fpr_points(y, score, (0.1, 0.25, 0.75))
    assert points[0.1]["max_allowed_fp"] == 0
    assert points[0.1]["achieved_fp"] == 0
    assert points[0.1]["tpr"] == 0.0
    assert points[0.25]["max_allowed_fp"] == 1
    assert points[0.25]["achieved_fp"] == 1
    assert math.isclose(points[0.25]["achieved_fpr"], 0.25)
    assert math.isclose(points[0.25]["tpr"], 1 / 3)
    assert points[0.25]["next_excluded_normal_tie_count"] == 2
    assert points[0.25]["next_excluded_attack_tie_count"] == 1
    assert math.isclose(points[0.75]["achieved_fpr"], 0.75)
    assert math.isclose(points[0.75]["tpr"], 2 / 3)

    confusion = Confusion()
    confusion.update(y, score >= 0.5)
    row = ranking_row(y, score, confusion)
    assert row["source_policy_tn"] == 1
    assert row["source_policy_fp"] == 3
    assert row["source_policy_fn"] == 1
    assert row["source_policy_tp"] == 2
    assert 0.0 <= row["roc_auc"] <= 1.0
    assert 0.0 <= row["average_precision_pr_auc"] <= 1.0
    return {
        "status": "passed",
        "tests": [
            "exact tie-aware ceiling with zero-FP boundary",
            "indivisible normal/attack boundary tie accounting",
            "ceiling monotonicity",
            "source-policy confusion accounting",
            "ROC/AP range and curve monotonicity",
        ],
    }


def verify_frozen_inference_equivalence(
    models: list[ModelInfo], device: torch.device
) -> dict:
    """Check that the optimized scorer exactly matches the frozen evaluator."""
    data = np.load(
        REPO_ROOT / "datasets" / "windows" / "otids_cross_windows.npz",
        allow_pickle=True,
    )
    x = data["x"][:17]
    new_scores, new_predictions = score_raw_batch(models, x, device)[
        models[0].key
    ]
    frozen_info = {
        "setting": models[0].setting,
        "seed": models[0].seed,
        "model": models[0].model,
        "mean": models[0].mean,
        "std": models[0].std,
        "device": device,
    }
    frozen_predictions, frozen_scores = cantt.evaluate_batch(frozen_info, x)
    score_exact = np.array_equal(new_scores, frozen_scores)
    prediction_exact = np.array_equal(new_predictions, frozen_predictions)
    if not score_exact or not prediction_exact:
        raise AssertionError(
            "optimized inference differs from frozen accepted-paper evaluator"
        )
    return {
        "status": "passed",
        "windows": len(x),
        "checkpoint": relative(models[0].model_path),
        "score_exact": score_exact,
        "prediction_exact": prediction_exact,
    }


class DatasetAccumulator:
    def __init__(
        self,
        dataset: str,
        group_type: str,
        models: list[ModelInfo],
        device: torch.device,
        log: EventLog,
    ) -> None:
        self.dataset = dataset
        self.group_type = group_type
        self.models = models
        self.device = device
        self.log = log
        self.labels: list[np.ndarray] = []
        self.scores: dict[tuple[str, int], list[np.ndarray]] = {
            model.key: [] for model in models
        }
        self.confusions: dict[tuple[str, int], Confusion] = {
            model.key: Confusion() for model in models
        }
        self.group_rows: list[dict] = []
        self.group_count = 0
        self.windows = 0

    def evaluate_group(
        self,
        metadata: dict,
        batches: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        started = time.time()
        group_y_parts: list[np.ndarray] = []
        group_scores: dict[tuple[str, int], list[np.ndarray]] = {
            model.key: [] for model in self.models
        }
        group_confusions: dict[tuple[str, int], Confusion] = {
            model.key: Confusion() for model in self.models
        }
        for x_raw, y_raw in batches:
            y = np.asarray(y_raw, dtype=np.int8).reshape(-1)
            if len(y) != len(x_raw):
                raise ValueError(
                    f"group {metadata['group_id']} label/window mismatch: "
                    f"{len(y)} != {len(x_raw)}"
                )
            if not set(np.unique(y)).issubset({0, 1}):
                raise ValueError(
                    f"group {metadata['group_id']} has non-binary labels"
                )
            scored = score_raw_batch(self.models, x_raw, self.device)
            group_y_parts.append(y.copy())
            for key, (scores, predicted) in scored.items():
                if len(scores) != len(y):
                    raise ValueError(f"score/label mismatch for {key}")
                group_scores[key].append(scores)
                group_confusions[key].update(y, predicted)

        if group_y_parts:
            group_y = np.concatenate(group_y_parts)
        else:
            group_y = np.empty(0, dtype=np.int8)
        self.labels.append(group_y)
        self.group_count += 1
        self.windows += len(group_y)

        for model in self.models:
            key = model.key
            scores = (
                np.concatenate(group_scores[key])
                if group_scores[key]
                else np.empty(0, dtype=np.float32)
            )
            if len(scores) != len(group_y):
                raise AssertionError(
                    f"group score/label mismatch for {metadata['group_id']}, {key}"
                )
            self.scores[key].append(scores)
            self.confusions[key].add(group_confusions[key])
            self.group_rows.append(
                {
                    "row_type": "group",
                    "dataset": self.dataset,
                    "group_type": self.group_type,
                    **metadata,
                    "setting": model.setting,
                    "setting_label": model.label,
                    "seed": model.seed,
                    "score_definition": SCORE_DEFINITION,
                    "source_policy": SOURCE_POLICY,
                    **ranking_row(group_y, scores, group_confusions[key]),
                }
            )
        self.log.emit(
            "group_complete",
            dataset=self.dataset,
            group_index=self.group_count,
            group_id=metadata["group_id"],
            windows=len(group_y),
            normal_windows=int((group_y == 0).sum()),
            attack_windows=int((group_y == 1).sum()),
            elapsed_seconds=round(time.time() - started, 3),
        )

    def finalize(self) -> tuple[list[dict], list[dict]]:
        y = np.concatenate(self.labels) if self.labels else np.empty(0, dtype=np.int8)
        pooled_rows: list[dict] = []
        for model in self.models:
            scores = (
                np.concatenate(self.scores[model.key])
                if self.scores[model.key]
                else np.empty(0, dtype=np.float32)
            )
            if len(scores) != len(y):
                raise AssertionError(
                    f"pooled score/label mismatch for {self.dataset}, {model.key}"
                )
            pooled_rows.append(
                {
                    "row_type": "pooled_by_seed",
                    "dataset": self.dataset,
                    "group_type": self.group_type,
                    "group_count": self.group_count,
                    "setting": model.setting,
                    "setting_label": model.label,
                    "seed": model.seed,
                    "paired_pipeline_seed": True,
                    "score_definition": SCORE_DEFINITION,
                    "source_policy": SOURCE_POLICY,
                    "external_label_usage": "diagnostic_evaluation_only",
                    **ranking_row(y, scores, self.confusions[model.key]),
                }
            )
        if any(row["windows"] != self.windows for row in pooled_rows):
            raise AssertionError(f"pooled window count mismatch for {self.dataset}")
        return pooled_rows, self.group_rows


def otids_groups() -> Iterator[tuple[dict, Iterable[tuple[np.ndarray, np.ndarray]]]]:
    path = REPO_ROOT / "datasets" / "windows" / "otids_cross_windows.npz"
    data = np.load(path, allow_pickle=True)
    x = data["x"]
    y = data["y_binary"].astype(np.int8)
    source = np.asarray([str(value) for value in data["source_file"]])
    weak = data["is_weak_label"].astype(bool)
    if x.shape != (144_156, WINDOW_SIZE, len(cantt.FEATURE_NAMES)):
        raise ValueError(f"unexpected OTIDS window shape: {x.shape}")
    for source_file in sorted(np.unique(source)):
        indices = np.flatnonzero(source == source_file)
        local_y = y[indices]
        role = (
            "mixed"
            if np.any(local_y == 0) and np.any(local_y == 1)
            else ("normal_only" if np.any(local_y == 0) else "attack_only")
        )

        def batches(selected: np.ndarray = indices) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for start in range(0, len(selected), RAW_WINDOW_BATCH):
                take = selected[start : start + RAW_WINDOW_BATCH]
                yield x[take], y[take]

        metadata = {
            "group_id": source_file,
            "set_id": "",
            "subset_id": role,
            "vehicle_axis": "external_vehicle",
            "attack_axis": role,
            "source_stem": Path(source_file).stem,
            "role": role,
            "family": Path(source_file).stem,
            "masquerade": "",
            "bootstrap_stratum": role,
            "weak_label_windows": int(weak[indices].sum()),
            "group_limitation": (
                "scenario is single-class; attack scenarios use weak window labels"
            ),
        }
        yield metadata, batches()


def cantt_groups() -> Iterator[tuple[dict, Iterable[tuple[np.ndarray, np.ndarray]]]]:
    files = [
        meta
        for meta in cantt.list_cantt_files()
        if meta.subset_id.startswith("test_")
    ]
    if len(files) != 176:
        raise ValueError(f"expected 176 can-train test files, got {len(files)}")
    with zipfile.ZipFile(cantt.ZIP_PATH) as archive:
        for meta in files:
            frame_df, features, issues = cantt.read_cantt_csv(archive, meta)
            frame_y = frame_df["attack"].to_numpy(dtype=np.int8)

            def batches(
                raw: np.ndarray = features, labels: np.ndarray = frame_y
            ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
                for x_raw, y, _starts in cantt.iter_windows(
                    raw, labels, batch_size=RAW_WINDOW_BATCH
                ):
                    yield x_raw, y

            metadata = {
                "group_id": meta.path,
                "set_id": meta.set_id,
                "subset_id": meta.subset_id,
                "vehicle_axis": meta.vehicle_axis,
                "attack_axis": meta.attack_axis,
                "source_stem": meta.source_stem,
                "role": "test_file",
                "family": meta.source_stem,
                "masquerade": "",
                "bootstrap_stratum": f"{meta.set_id}|{meta.subset_id}",
                "weak_label_windows": 0,
                "group_limitation": "",
                "parse_issues": ";".join(
                    f"{key}:{value}" for key, value in sorted(issues.items()) if value
                ),
            }
            yield metadata, batches()


def road_groups() -> Iterator[tuple[dict, Iterable[tuple[np.ndarray, np.ndarray]]]]:
    frames_dir = REPO_ROOT / "datasets" / "processed" / "road_frames"
    paths = sorted(frames_dir.glob("*.parquet"))
    if len(paths) != 41:
        raise ValueError(f"expected 41 ROAD captures, got {len(paths)}")
    profile_path = WISA_ROOT / "results" / "tables" / "road_dataset_profile.csv"
    profile = pd.read_csv(profile_path).set_index("capture")
    for path in paths:
        if path.stem not in profile.index:
            raise ValueError(f"ROAD capture missing from frozen profile: {path.stem}")
        info = profile.loc[path.stem]
        frame_df = pd.read_parquet(
            path, columns=[*cantt.FEATURE_NAMES, "attack"]
        )
        features = frame_df[cantt.FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = frame_df["attack"].to_numpy(dtype=np.int8)
        role = str(info["role"])
        family = str(info["family"])
        masquerade = bool(int(info.get("masquerade", 0)))

        def batches(
            raw: np.ndarray = features, labels: np.ndarray = frame_y
        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for x_raw, y, _starts in cantt.iter_windows(
                raw, labels, batch_size=RAW_WINDOW_BATCH
            ):
                yield x_raw, y

        metadata = {
            "group_id": path.stem,
            "set_id": "",
            "subset_id": role,
            "vehicle_axis": "external_vehicle",
            "attack_axis": (
                "ambient"
                if role == "ambient"
                else f"{family}|{'masquerade' if masquerade else 'fabrication'}"
            ),
            "source_stem": path.stem,
            "role": role,
            "family": family,
            "masquerade": masquerade,
            "bootstrap_stratum": f"{role}|{family}|{int(masquerade)}",
            "weak_label_windows": 0,
            "group_limitation": "",
        }
        yield metadata, batches()


def evaluate_dataset(
    dataset: str,
    group_type: str,
    group_factory: Callable[
        [], Iterator[tuple[dict, Iterable[tuple[np.ndarray, np.ndarray]]]]
    ],
    expected_groups: int,
    models: list[ModelInfo],
    device: torch.device,
    log: EventLog,
) -> tuple[list[dict], list[dict]]:
    started = time.time()
    log.emit("dataset_start", dataset=dataset, expected_groups=expected_groups)
    accumulator = DatasetAccumulator(dataset, group_type, models, device, log)
    for metadata, batches in group_factory():
        accumulator.evaluate_group(metadata, batches)
    if accumulator.group_count != expected_groups:
        raise AssertionError(
            f"{dataset}: expected {expected_groups} groups, "
            f"got {accumulator.group_count}"
        )
    pooled, groups = accumulator.finalize()
    reference = pooled[0]
    log.emit(
        "dataset_complete",
        dataset=dataset,
        groups=accumulator.group_count,
        windows=reference["windows"],
        normal_windows=reference["normal_windows"],
        attack_windows=reference["attack_windows"],
        elapsed_seconds=round(time.time() - started, 3),
    )
    return pooled, groups


def bootstrap_mean_ci(
    values: np.ndarray,
    strata: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    strata = np.asarray(strata, dtype=str)
    finite = np.isfinite(values)
    values = values[finite]
    strata = strata[finite]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    unique_strata = sorted(np.unique(strata))
    boot = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        total = 0.0
        count = 0
        for stratum in unique_strata:
            local = values[strata == stratum]
            selected = local[rng.integers(0, len(local), size=len(local))]
            total += float(selected.sum())
            count += len(selected)
        boot[replicate] = total / count
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


MACRO_METRICS = (
    "source_policy_fpr",
    "source_policy_attack_recall",
    "attack_prevalence",
    "roc_auc",
    "average_precision_pr_auc",
    "tpr_at_fpr_0p001",
    "tpr_at_fpr_0p01",
)


def append_group_macro_rows(group_rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(group_rows)
    macro_rows: list[dict] = []
    for (dataset, group_type, setting, setting_label, seed), group in frame.groupby(
        ["dataset", "group_type", "setting", "setting_label", "seed"],
        sort=True,
    ):
        row: dict = {
            "row_type": "group_macro",
            "dataset": dataset,
            "group_type": group_type,
            "group_id": "__group_macro__",
            "setting": setting,
            "setting_label": setting_label,
            "seed": int(seed),
            "score_definition": SCORE_DEFINITION,
            "source_policy": SOURCE_POLICY,
            "group_count": len(group),
            "bootstrap_replicates": (
                0 if dataset == "OTIDS" else BOOTSTRAP_REPLICATES
            ),
            "bootstrap_base_seed": BOOTSTRAP_SEED,
            "bootstrap_rejected_draws": 0,
            "bootstrap_missing_class_policy": (
                "exclude groups where metric is undefined; retain and count them"
            ),
            "bootstrap_status": (
                "not_run_class_separated_scenarios"
                if dataset == "OTIDS"
                else "stratified_group_resampling"
            ),
        }
        for metric in MACRO_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=np.float64
            )
            eligible = np.isfinite(values)
            row[metric] = (
                float(values[eligible].mean()) if np.any(eligible) else math.nan
            )
            row[f"{metric}_eligible_groups"] = int(eligible.sum())
            row[f"{metric}_missing_groups"] = int((~eligible).sum())
            if dataset != "OTIDS" and int(eligible.sum()) >= 2:
                key = f"{dataset}|{setting}|{seed}|{metric}"
                metric_seed = BOOTSTRAP_SEED ^ zlib.crc32(key.encode("utf-8"))
                low, high = bootstrap_mean_ci(
                    values[eligible],
                    group.loc[eligible, "bootstrap_stratum"].astype(str).to_numpy(),
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=metric_seed,
                )
            else:
                low, high = math.nan, math.nan
            row[f"{metric}_bootstrap_ci_low"] = low
            row[f"{metric}_bootstrap_ci_high"] = high
        macro_rows.append(row)
    return group_rows + macro_rows


SUMMARY_METRICS = (
    "source_policy_fpr",
    "source_policy_normal_recall",
    "source_policy_attack_recall",
    "roc_auc",
    "average_precision_pr_auc",
    "attack_prevalence",
    "achieved_fpr_at_fpr_0p001",
    "tpr_at_fpr_0p001",
    "achieved_fpr_at_fpr_0p01",
    "tpr_at_fpr_0p01",
)


def summarize_across_pipeline_seeds(by_seed_rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(by_seed_rows)
    summary: list[dict] = []
    for (dataset, group_type, setting, setting_label), group in frame.groupby(
        ["dataset", "group_type", "setting", "setting_label"], sort=True
    ):
        if sorted(group["seed"].astype(int).tolist()) != list(SEEDS):
            raise AssertionError(
                f"missing pipeline seed for {dataset}, {setting}: "
                f"{group['seed'].tolist()}"
            )
        row = {
            "row_type": "setting_summary",
            "dataset": dataset,
            "group_type": group_type,
            "setting": setting,
            "setting_label": setting_label,
            "seeds": ";".join(str(seed) for seed in SEEDS),
            "pipeline_seed_count": len(SEEDS),
            "pipeline_seed_interpretation": (
                "paired fits; not capture, vehicle, construction, or population replication"
            ),
            "group_count": int(group["group_count"].iloc[0]),
            "windows": int(group["windows"].iloc[0]),
            "normal_windows": int(group["normal_windows"].iloc[0]),
            "attack_windows": int(group["attack_windows"].iloc[0]),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=np.float64
            )
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        summary.append(row)

    for dataset, dataset_frame in frame.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["setting"] == "real_only"].set_index(
            "seed"
        )
        rule = dataset_frame[dataset_frame["setting"] == "rule_0p30"].set_index(
            "seed"
        )
        if sorted(real.index.astype(int).tolist()) != list(SEEDS) or sorted(
            rule.index.astype(int).tolist()
        ) != list(SEEDS):
            raise AssertionError(f"primary contrast seed mismatch for {dataset}")
        row = {
            "row_type": "primary_paired_contrast",
            "dataset": dataset,
            "group_type": dataset_frame["group_type"].iloc[0],
            "setting": "rule_0p30_minus_real_only",
            "setting_label": "Rule +30% minus Real only",
            "seeds": ";".join(str(seed) for seed in SEEDS),
            "pipeline_seed_count": len(SEEDS),
            "pipeline_seed_interpretation": (
                "paired descriptive difference; supporting evidence only"
            ),
            "contrast_direction": (
                "positive favors Rule for recall/ranking; negative favors Rule for FPR"
            ),
            "group_count": int(real["group_count"].iloc[0]),
            "windows": int(real["windows"].iloc[0]),
            "normal_windows": int(real["normal_windows"].iloc[0]),
            "attack_windows": int(real["attack_windows"].iloc[0]),
        }
        for metric in SUMMARY_METRICS:
            differences = (
                pd.to_numeric(rule.loc[list(SEEDS), metric], errors="coerce").to_numpy()
                - pd.to_numeric(real.loc[list(SEEDS), metric], errors="coerce").to_numpy()
            )
            row[f"{metric}_mean"] = float(np.nanmean(differences))
            row[f"{metric}_std"] = float(np.nanstd(differences, ddof=1))
            row[f"{metric}_paired_values_seed7_42_123"] = ";".join(
                f"{float(value):.12g}" for value in differences
            )
        summary.append(row)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            clean = {
                key: (
                    ""
                    if isinstance(value, (float, np.floating))
                    and not math.isfinite(float(value))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(clean)


def verify_inputs() -> dict:
    dataset_entries: list[dict] = []
    for path_text, (expected_size, expected_hash) in EXPECTED_DATASET_HASHES.items():
        path = REPO_ROOT / path_text
        entry = file_entry(path)
        if entry["bytes"] != expected_size or entry["sha256"] != expected_hash:
            raise RuntimeError(f"dataset lineage mismatch: {entry}")
        dataset_entries.append(entry)

    road_paths = sorted(
        (REPO_ROOT / "datasets" / "processed" / "road_frames").glob("*.parquet")
    )
    road_entries = [file_entry(path) for path in road_paths]
    road_digest = canonical_inventory_digest(road_entries)
    if len(road_entries) != 41 or road_digest != EXPECTED_ROAD_TREE_SHA256:
        raise RuntimeError(
            f"ROAD lineage mismatch: files={len(road_entries)}, sha256={road_digest}"
        )

    dependency_entries = []
    for path_text, expected_hash in EXPECTED_DEPENDENCY_HASHES.items():
        entry = file_entry(REPO_ROOT / path_text)
        if entry["sha256"] != expected_hash:
            raise RuntimeError(f"dependency lineage mismatch: {entry}")
        dependency_entries.append(entry)

    otids = np.load(
        REPO_ROOT / "datasets" / "windows" / "otids_cross_windows.npz",
        allow_pickle=True,
    )
    otids_y = otids["y_binary"].astype(np.int8)
    otids_source = np.asarray([str(value) for value in otids["source_file"]])
    if (
        len(otids_y) != 144_156
        or int((otids_y == 0).sum()) != 74_040
        or int((otids_y == 1).sum()) != 70_116
        or len(np.unique(otids_source)) != 4
    ):
        raise RuntimeError("OTIDS label/group audit changed from frozen plan")

    test_files = [
        meta
        for meta in cantt.list_cantt_files()
        if meta.subset_id.startswith("test_")
    ]
    if len(test_files) != 176:
        raise RuntimeError("can-train test-file inventory changed from frozen plan")

    return {
        "dataset_files": dataset_entries,
        "road_files": road_entries,
        "road_tree_sha256": road_digest,
        "dependency_files": dependency_entries,
        "label_group_audit": {
            "OTIDS": {
                "groups": 4,
                "windows": 144_156,
                "normal_windows": 74_040,
                "attack_windows": 70_116,
                "weak_attack_windows": int(otids["is_weak_label"].sum()),
            },
            "can-train-and-test": {
                "groups": len(test_files),
                "sets": sorted({meta.set_id for meta in test_files}),
                "subsets": sorted({meta.subset_id for meta in test_files}),
            },
            "ROAD": {
                "groups": len(road_entries),
                "ambient_captures": 12,
                "attack_captures": 29,
            },
        },
    }


def refuse_overwrite() -> None:
    existing = [str(path) for path in OUTPUT_PATHS.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "v1 output already exists; refusing overwrite:\n" + "\n".join(existing)
        )


def publish_atomically(staged: dict[str, Path]) -> None:
    for key, destination in OUTPUT_PATHS.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged[key], destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic metric tests without loading checkpoints or data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(), indent=2, sort_keys=True))
        return

    refuse_overwrite()
    started_wall = time.time()
    started_utc = utc_now()
    log = EventLog()
    self_tests = run_self_tests()
    log.emit("self_tests", **self_tests)

    runtime_commit = git_text("rev-parse", "HEAD")
    dirty_before = git_text("status", "--porcelain", "--untracked-files=all")
    if dirty_before:
        raise RuntimeError(
            "evaluator requires a clean committed source state before inference:\n"
            + dirty_before
        )
    plan_in_commit = subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{PLAN_COMMIT}:wisa_camera_ready/experiments/01_external_ranking/PLAN.md",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    if not plan_in_commit:
        raise RuntimeError(f"analysis plan not found in frozen commit {PLAN_COMMIT}")

    input_audit = verify_inputs()
    log.emit(
        "input_audit_complete",
        datasets=3,
        otids_groups=4,
        cantt_groups=176,
        road_groups=41,
        road_tree_sha256=input_audit["road_tree_sha256"],
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, model_entries = load_models(device, log)
    inference_equivalence = verify_frozen_inference_equivalence(models, device)
    log.emit("frozen_inference_equivalence", **inference_equivalence)

    by_seed_rows: list[dict] = []
    group_rows: list[dict] = []
    dataset_specs = (
        ("OTIDS", "source_file_or_scenario", otids_groups, 4),
        ("ROAD", "capture", road_groups, 41),
        ("can-train-and-test", "test_file", cantt_groups, 176),
    )
    for dataset, group_type, factory, expected_groups in dataset_specs:
        pooled, groups = evaluate_dataset(
            dataset,
            group_type,
            factory,
            expected_groups,
            models,
            device,
            log,
        )
        by_seed_rows.extend(pooled)
        group_rows.extend(groups)

    expected_pooled = len(dataset_specs) * len(SETTING_SPECS) * len(SEEDS)
    expected_group_rows = sum(spec[3] for spec in dataset_specs) * len(
        SETTING_SPECS
    ) * len(SEEDS)
    if len(by_seed_rows) != expected_pooled:
        raise AssertionError(
            f"expected {expected_pooled} pooled rows, got {len(by_seed_rows)}"
        )
    if len(group_rows) != expected_group_rows:
        raise AssertionError(
            f"expected {expected_group_rows} group rows, got {len(group_rows)}"
        )

    all_group_rows = append_group_macro_rows(group_rows)
    summary_rows = summarize_across_pipeline_seeds(by_seed_rows)

    with tempfile.TemporaryDirectory(
        prefix="external_ranking_v1_", dir=CAMERA_ROOT
    ) as temp_name:
        temp = Path(temp_name)
        staged = {
            "by_seed": temp / OUTPUT_PATHS["by_seed"].name,
            "summary": temp / OUTPUT_PATHS["summary"].name,
            "by_group": temp / OUTPUT_PATHS["by_group"].name,
            "log": temp / OUTPUT_PATHS["log"].name,
            "manifest": temp / OUTPUT_PATHS["manifest"].name,
        }
        write_csv(staged["by_seed"], by_seed_rows)
        write_csv(staged["summary"], summary_rows)
        write_csv(staged["by_group"], all_group_rows)

        completed_utc = utc_now()
        log.emit(
            "analysis_complete",
            pooled_rows=len(by_seed_rows),
            setting_summary_and_contrast_rows=len(summary_rows),
            raw_group_rows=len(group_rows),
            group_macro_rows=len(all_group_rows) - len(group_rows),
            elapsed_seconds=round(time.time() - started_wall, 3),
        )
        staged["log"].write_text("\n".join(log.lines) + "\n", encoding="utf-8")

        staged_output_entries = {
            key: {
                "path": relative(OUTPUT_PATHS[key]),
                "bytes": staged[key].stat().st_size,
                "sha256": sha256_file(staged[key]),
            }
            for key in ("by_seed", "summary", "by_group", "log")
        }
        dataset_audit = {}
        by_seed_frame = pd.DataFrame(by_seed_rows)
        for dataset, group in by_seed_frame.groupby("dataset", sort=True):
            first = group.iloc[0]
            dataset_audit[dataset] = {
                "groups": int(first["group_count"]),
                "windows": int(first["windows"]),
                "normal_windows": int(first["normal_windows"]),
                "attack_windows": int(first["attack_windows"]),
                "settings": sorted(group["setting"].unique().tolist()),
                "seeds": sorted(int(seed) for seed in group["seed"].unique()),
            }

        manifest = {
            "schema_version": 1,
            "experiment": VERSION,
            "analysis_type": (
                "post-review target-label-dependent diagnostic evaluation"
            ),
            "external_data_role": "evaluation-only",
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "elapsed_seconds": time.time() - started_wall,
            "analysis_plan": {
                "path": relative(
                    CAMERA_ROOT
                    / "experiments"
                    / "01_external_ranking"
                    / "PLAN.md"
                ),
                "commit": PLAN_COMMIT,
                "frozen_before_inference": True,
            },
            "accepted_source_commit_at_plan_freeze": ACCEPTED_SOURCE_COMMIT,
            "evaluator": {
                "path": relative(Path(__file__)),
                "commit": runtime_commit,
                "sha256": sha256_file(Path(__file__)),
                "command": (
                    "source .venv/bin/activate && "
                    "python wisa_camera_ready/scripts/"
                    "evaluate_external_ranking_v1.py"
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else platform.processor()
                ),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "contract": {
                "settings": [spec[0] for spec in SETTING_SPECS],
                "paired_pipeline_seeds": list(SEEDS),
                "primary_contrast": "rule_0p30 - real_only",
                "datasets": [spec[0] for spec in dataset_specs],
                "score_definition": SCORE_DEFINITION,
                "source_policy": SOURCE_POLICY,
                "target_fpr_ceilings": list(TARGET_FPRS),
                "fixed_fpr_tie_rule": (
                    "most permissive threshold immediately above next excluded "
                    "normal-score tie; no interpolation or fractional tie breaking"
                ),
                "window_size": WINDOW_SIZE,
                "stride": STRIDE,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
            "inputs": {
                "checkpoint_and_scaler_inventory_sha256": (
                    canonical_inventory_digest(model_entries)
                ),
                "checkpoint_and_scaler_files": model_entries,
                **input_audit,
            },
            "observed_dataset_audit": dataset_audit,
            "validation_gates": {
                "metric_self_tests": self_tests,
                "frozen_inference_equivalence": inference_equivalence,
                "source_tree_clean_before_inference": True,
                "analysis_plan_present_in_frozen_commit": True,
                "input_hashes_match_frozen_plan": True,
                "models_loaded": len(models),
                "expected_models": 18,
                "finite_score_check": "passed for every batch and pooled row",
                "label_alignment_check": "passed for every batch, group, and dataset",
                "binary_label_check": "passed",
                "roc_pr_curve_monotonicity_check": "passed where both classes exist",
                "fixed_fpr_ceiling_and_monotonicity_check": "passed",
                "pooled_rows": len(by_seed_rows),
                "expected_pooled_rows": expected_pooled,
                "raw_group_rows": len(group_rows),
                "expected_raw_group_rows": expected_group_rows,
                "all_settings_and_seeds_reported": True,
                "bootstrap_rejected_draws": 0,
            },
            "outputs": staged_output_entries,
            "warnings": [
                (
                    "OTIDS source-file groups are class-separated; no scenario-local "
                    "ranking metric or group bootstrap interval is identifiable."
                ),
                (
                    "Pipeline-seed mean/std are descriptive paired-fit variability, "
                    "not capture-, vehicle-, construction-, or population-level uncertainty."
                ),
                (
                    "Target-dataset fixed-FPR thresholds are diagnostic attainable "
                    "points and are not deployment calibration or model selection."
                ),
            ],
        }
        staged["manifest"].write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_atomically(staged)

    print(
        json.dumps(
            {
                "status": "complete",
                "outputs": {
                    key: relative(path) for key, path in OUTPUT_PATHS.items()
                },
                "elapsed_seconds": round(time.time() - started_wall, 3),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
