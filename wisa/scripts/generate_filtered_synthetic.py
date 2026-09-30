#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from train_real_only_baselines import rf_features


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
EXP = ROOT / "experiments" / "06_filtering"
TABLES = ROOT / "results" / "tables"

SEED = 20260610
ATTACK_TYPES = {1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}
TEMPORAL_LOWER_Q = 0.05
TEMPORAL_UPPER_Q = 0.95
ARTIFACT_REMOVE_TOP_Q = 0.30


def window_timing_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = x[:, :, 10]
    return dt.mean(axis=1), dt.std(axis=1)


def temporal_keep_mask(real_x: np.ndarray, real_y: np.ndarray, syn_x: np.ndarray, syn_y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    syn_mean, syn_std = window_timing_features(syn_x)
    keep = np.zeros(len(syn_y), dtype=bool)
    rows = []
    for cls, name in ATTACK_TYPES.items():
        real_mask = real_y == cls
        syn_mask = syn_y == cls
        real_mean, real_std = window_timing_features(real_x[real_mask])
        mean_lo, mean_hi = np.quantile(real_mean, [TEMPORAL_LOWER_Q, TEMPORAL_UPPER_Q])
        std_lo, std_hi = np.quantile(real_std, [TEMPORAL_LOWER_Q, TEMPORAL_UPPER_Q])
        cls_keep = syn_mask & (syn_mean >= mean_lo) & (syn_mean <= mean_hi) & (syn_std >= std_lo) & (syn_std <= std_hi)
        keep |= cls_keep
        rows.append(
            {
                "attack_type": name,
                "real_windows": int(real_mask.sum()),
                "synthetic_windows": int(syn_mask.sum()),
                "mean_delta_t_lo": float(mean_lo),
                "mean_delta_t_hi": float(mean_hi),
                "std_delta_t_lo": float(std_lo),
                "std_delta_t_hi": float(std_hi),
                "kept_windows": int(cls_keep.sum()),
                "keep_ratio": float(cls_keep.sum() / max(syn_mask.sum(), 1)),
            }
        )
    return keep, rows


def artifact_scores(real_x: np.ndarray, syn_x: np.ndarray) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(SEED)
    n = min(60_000, len(real_x), len(syn_x))
    real_idx = rng.choice(len(real_x), size=n, replace=False)
    syn_idx = rng.choice(len(syn_x), size=n, replace=False)
    x = np.concatenate([rf_features(real_x[real_idx]), rf_features(syn_x[syn_idx])], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    clf.fit(x_train, y_train)
    val_acc = float((clf.predict(x_val) == y_val).mean())
    scores = clf.predict_proba(rf_features(syn_x))[:, 1]
    threshold = float(np.quantile(scores, 1.0 - ARTIFACT_REMOVE_TOP_Q))
    return scores, threshold, val_acc


def class_balanced_artifact_keep_mask(scores: np.ndarray, syn_y: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(scores), dtype=bool)
    for cls in ATTACK_TYPES:
        cls_idx = np.where(syn_y == cls)[0]
        keep_n = int(np.ceil(len(cls_idx) * (1.0 - ARTIFACT_REMOVE_TOP_Q)))
        # Scores can saturate at 1.0 when artifacts are very easy to detect.
        # Rank-based removal still enforces the intended retained fraction.
        ranked = cls_idx[np.argsort(scores[cls_idx], kind="mergesort")]
        keep[ranked[:keep_n]] = True
    return keep


def save_pool(name: str, mask: np.ndarray, syn: np.lib.npyio.NpzFile, artifact_score: np.ndarray) -> dict:
    path = SYNTHETIC / f"{name}_filtered_windows.npz"
    np.savez_compressed(
        path,
        x=syn["x"][mask].astype(np.float32),
        y_binary=syn["y_binary"][mask].astype(np.int8),
        y_attack_type=syn["y_attack_type"][mask].astype(np.int8),
        synthetic_type=syn["synthetic_type"][mask],
        injection_count=syn["injection_count"][mask].astype(np.int16),
        artifact_score=artifact_score[mask].astype(np.float32),
        feature_names=syn["feature_names"],
        window_size=syn["window_size"],
        stride=syn["stride"],
        seed=np.asarray(SEED, dtype=np.int32),
    )
    y = syn["y_attack_type"][mask]
    row = {
        "filter": name,
        "output": str(path.relative_to(ROOT)),
        "windows": int(mask.sum()),
        "keep_ratio": float(mask.mean()),
        "artifact_score_mean": float(artifact_score[mask].mean()) if mask.any() else float("nan"),
        "artifact_score_max": float(artifact_score[mask].max()) if mask.any() else float("nan"),
    }
    for cls, attack_name in ATTACK_TYPES.items():
        row[f"{attack_name}_windows"] = int((y == cls).sum())
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    syn = np.load(SYNTHETIC / "rule_based_windows.npz", allow_pickle=True)

    real_mask = train["y_binary"] == 1
    real_x = train["x"][real_mask].astype(np.float32)
    real_y = train["y_attack_type"][real_mask].astype(np.int64)
    syn_x = syn["x"].astype(np.float32)
    syn_y = syn["y_attack_type"].astype(np.int64)

    temporal_mask, temporal_rows = temporal_keep_mask(real_x, real_y, syn_x, syn_y)
    artifact_score, artifact_threshold, artifact_val_acc = artifact_scores(real_x, syn_x)
    artifact_mask = class_balanced_artifact_keep_mask(artifact_score, syn_y)
    combined_mask = temporal_mask & artifact_mask

    write_csv(EXP / "temporal_filter_thresholds.csv", temporal_rows)
    rows = [
        save_pool("temporal", temporal_mask, syn, artifact_score),
        save_pool("artifact", artifact_mask, syn, artifact_score),
        save_pool("temporal_artifact", combined_mask, syn, artifact_score),
    ]
    write_csv(TABLES / "filtered_synthetic_statistics.csv", rows)
    write_csv(EXP / "filtered_synthetic_statistics.csv", rows)

    config = {
        "seed": SEED,
        "source": "datasets/synthetic/rule_based_windows.npz",
        "temporal_filter": {
            "basis": "Car-Hacking train attack windows only",
            "features": ["delta_t_mean_per_window", "delta_t_std_per_window"],
            "lower_quantile": TEMPORAL_LOWER_Q,
            "upper_quantile": TEMPORAL_UPPER_Q,
        },
        "artifact_filter": {
            "basis": "RandomForest real-train-attack vs rule-synthetic discriminator",
            "remove_top_synthetic_score_quantile": ARTIFACT_REMOVE_TOP_Q,
            "artifact_score_threshold": artifact_threshold,
            "validation_accuracy": artifact_val_acc,
        },
        "outputs": rows,
    }
    (EXP / "filtering_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
