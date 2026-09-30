#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_real_only_baselines import CNN1D, WindowDataset, predict_cnn, standardize


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
TABLES = ROOT / "results" / "tables"
EXP = ROOT / "experiments" / "04_augmented_ids"
BASELINE_MODELS = ROOT / "models" / "baseline"
RATIO_MODELS = ROOT / "models" / "ratio_sweep"
OVERSAMPLING_MODELS = ROOT / "models" / "oversampling"

SEEDS = [42, 7, 123]
SETTINGS = [
    {"setting": "real_only", "family": "real_only", "ratio": 0.0, "label": "Real only"},
    {"setting": "real_oversampling_0p50", "family": "oversampling", "ratio": 0.5, "label": "Real oversampling +50%"},
    {"setting": "rule_0p30", "family": "rule", "ratio": 0.3, "label": "Rule +30%"},
    {"setting": "rule_0p50", "family": "rule", "ratio": 0.5, "label": "Rule +50%"},
    {"setting": "rule_1p00", "family": "rule", "ratio": 1.0, "label": "Rule +100%"},
]
VAL_FPR_TARGETS = [0.001, 0.01, 0.05]
OTIDS_FPR_TARGETS = [0.01, 0.05, 0.10]


def model_paths(setting: dict, seed: int) -> tuple[Path, Path]:
    family = setting["family"]
    if family == "real_only":
        return BASELINE_MODELS / f"cnn1d_real_only_seed{seed}.pt", BASELINE_MODELS / "cnn_standardizer.npz"
    if family == "oversampling":
        suffix = f"real_oversampling_ratio0p50_seed{seed}"
        return OVERSAMPLING_MODELS / f"cnn1d_{suffix}.pt", OVERSAMPLING_MODELS / f"standardizer_{suffix}.npz"
    if family == "rule":
        suffix = f"rule_ratio{setting['ratio']:.2f}_seed{seed}".replace(".", "p")
        return RATIO_MODELS / f"cnn1d_{suffix}.pt", RATIO_MODELS / f"standardizer_{suffix}.npz"
    raise ValueError(family)


def load_scores(setting: dict, seed: int, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_path, std_path = model_paths(setting, seed)
    stdz = np.load(std_path)
    x_std = standardize(x.astype(np.float32), stdz["mean"], stdz["std"])
    model = CNN1D(in_channels=x.shape[-1], classes=5).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    loader = DataLoader(WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(model, loader, device)
    return 1.0 - probs[:, 0]


def threshold_for_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    if len(normal_scores) == 0:
        return 1.0
    return float(np.quantile(normal_scores, 1.0 - target_fpr, method="higher"))


def metrics_at_threshold(scores: np.ndarray, y_binary: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    normal = y_binary == 0
    attack = y_binary == 1
    fp = int((normal & pred).sum())
    tn = int((normal & ~pred).sum())
    tp = int((attack & pred).sum())
    fn = int((attack & ~pred).sum())
    return {
        "threshold": float(threshold),
        "normal_recall": float(tn / max(tn + fp, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
        "attack_detection_recall": float(tp / max(tp + fn, 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["seeds"] = ";".join(str(s) for s in sorted(group["seed"].unique()))
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def add_metric_row(rows: list[dict], setting: dict, seed: int, policy: str, source: str, target_fpr: float | str, split: str, scores: np.ndarray, y_binary: np.ndarray, threshold: float) -> None:
    row = {
        "setting": setting["setting"],
        "label": setting["label"],
        "family": setting["family"],
        "ratio": setting["ratio"],
        "seed": seed,
        "threshold_policy": policy,
        "threshold_source": source,
        "target_fpr": target_fpr,
        "split": split,
    }
    row.update(metrics_at_threshold(scores, y_binary, threshold))
    rows.append(row)


def make_key_table(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[
        (summary["setting"].isin(["real_only", "rule_0p30", "rule_1p00"]))
        & (
            ((summary["threshold_policy"] == "validation_fpr") & (summary["target_fpr"].astype(str).isin(["0.001", "0.01"])))
            | ((summary["threshold_policy"] == "otids_normal_oracle") & (summary["target_fpr"].astype(str).isin(["0.01", "0.05"])))
        )
        & (summary["split"].isin(["variant_sensitivity", "otids_cross_binary"]))
    ].copy()

    rows = []
    for (setting, label, policy, target), group in selected.groupby(["setting", "label", "threshold_policy", "target_fpr"], sort=False):
        variant = group[group["split"] == "variant_sensitivity"]
        otids = group[group["split"] == "otids_cross_binary"]
        if variant.empty or otids.empty:
            continue
        rows.append(
            {
                "setting": setting,
                "label": label,
                "threshold_policy": policy,
                "target_fpr": target,
                "threshold_mean": float(group["threshold_mean"].mean()),
                "variant_attack_detection_recall_mean": float(variant["attack_detection_recall_mean"].iloc[0]),
                "variant_attack_detection_recall_std": float(variant["attack_detection_recall_std"].iloc[0]),
                "otids_fpr_mean": float(otids["fpr_mean"].iloc[0]),
                "otids_fpr_std": float(otids["fpr_std"].iloc[0]),
                "otids_normal_recall_mean": float(otids["normal_recall_mean"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    val = np.load(WINDOWS / "val_windows.npz", allow_pickle=True)
    sensitivity = np.load(WINDOWS / "variant_sensitivity_windows.npz", allow_pickle=True)
    otids = np.load(WINDOWS / "otids_cross_windows.npz", allow_pickle=True)
    val_y = (val["y_attack_type"].astype(np.int64) > 0).astype(np.int8)
    sens_y = (sensitivity["y_attack_type"].astype(np.int64) > 0).astype(np.int8)
    otids_y = otids["y_binary"].astype(np.int8)

    rows = []
    for setting in SETTINGS:
        for seed in SEEDS:
            print(f"threshold diagnostic {setting['setting']} seed={seed}")
            val_scores = load_scores(setting, seed, val["x"])
            sens_scores = load_scores(setting, seed, sensitivity["x"])
            otids_scores = load_scores(setting, seed, otids["x"])
            val_normal_scores = val_scores[val_y == 0]
            otids_normal_scores = otids_scores[otids_y == 0]

            for target in VAL_FPR_TARGETS:
                threshold = threshold_for_fpr(val_normal_scores, target)
                for split, scores, y in [
                    ("validation", val_scores, val_y),
                    ("variant_sensitivity", sens_scores, sens_y),
                    ("otids_cross_binary", otids_scores, otids_y),
                ]:
                    add_metric_row(rows, setting, seed, "validation_fpr", "car_hacking_val_normal", target, split, scores, y, threshold)

            for target in OTIDS_FPR_TARGETS:
                threshold = threshold_for_fpr(otids_normal_scores, target)
                for split, scores, y in [
                    ("variant_sensitivity", sens_scores, sens_y),
                    ("otids_cross_binary", otids_scores, otids_y),
                ]:
                    add_metric_row(rows, setting, seed, "otids_normal_oracle", "otids_normal", target, split, scores, y, threshold)

    detail = pd.DataFrame(rows)
    detail.to_csv(TABLES / "threshold_sensitivity_details.csv", index=False)
    detail.to_csv(EXP / "threshold_sensitivity_details.csv", index=False)
    summary = summarize(
        detail,
        ["setting", "label", "family", "ratio", "threshold_policy", "threshold_source", "target_fpr", "split"],
        ["threshold", "normal_recall", "fpr", "attack_detection_recall"],
    )
    summary.to_csv(TABLES / "threshold_sensitivity_summary.csv", index=False)
    summary.to_csv(EXP / "threshold_sensitivity_summary.csv", index=False)
    key = make_key_table(summary)
    key.to_csv(TABLES / "threshold_sensitivity_key_comparisons.csv", index=False)
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
