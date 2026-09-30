#!/usr/bin/env python3
import argparse
import csv
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from run_cantt_tier1 import (
    FEATURE_NAMES,
    LOGS,
    TABLES,
    ZIP_PATH,
    MetricCounter,
    evaluate_batch,
    iter_windows,
    list_cantt_files,
    load_models,
    read_cantt_csv,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed"
WINDOWS = ROOT / "datasets" / "windows"
SELECTED_SETTINGS = {"real_only", "rule_0p30"}


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


def car_hacking_train_normal_can_id_mean() -> float:
    values = []
    pf = pq.ParquetFile(PROCESSED / "car_hacking_train.parquet")
    for batch in pf.iter_batches(columns=["can_id", "binary_label_id"], batch_size=500_000):
        df = batch.to_pandas()
        normal = df.loc[df["binary_label_id"] == 0, "can_id"].to_numpy(dtype=np.float64)
        if len(normal):
            values.append((normal.sum(), len(normal)))
    total_sum = sum(v[0] for v in values)
    total_count = sum(v[1] for v in values)
    return float(total_sum / total_count)


def car_hacking_train_normal_window_feature_means() -> np.ndarray:
    data = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    x = data["x"]
    y = data["y_binary"]
    normal = x[y == 0]
    if len(normal) == 0:
        raise RuntimeError("No normal Car-Hacking train windows found.")
    return normal.mean(axis=(0, 1)).astype(np.float32)


def apply_variant(x: np.ndarray, variant: str, can_id_mean: float, feature_means: np.ndarray) -> np.ndarray:
    if variant == "can_id_masked":
        out = x.copy()
        out[:, :, 0] = np.float32(can_id_mean)
        return out
    if variant == "id_only_non_id_masked":
        out = x.copy()
        out[:, :, 1:] = feature_means[1:]
        return out
    raise ValueError(variant)


def aggregate_seed_rows(seed_rows: list[dict], original_summary: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in original_summary.iterrows():
        if row["setting"] not in SELECTED_SETTINGS:
            continue
        rows.append(
            {
                "diagnostic": "cnn_id_ablation",
                "variant": "original_existing_result",
                "setting": row["setting"],
                "label": row["label"],
                "seeds": row["seeds"],
                "normal_recall_mean": row["normal_recall_mean"],
                "normal_recall_std": row["normal_recall_std"],
                "fpr_mean": row["fpr_mean"],
                "fpr_std": row["fpr_std"],
                "attack_recall_mean": row["attack_recall_mean"],
                "attack_recall_std": row["attack_recall_std"],
                "source": "results/tables/cantt_external_eval_summary_mean_std.csv",
            }
        )

    df = pd.DataFrame(seed_rows)
    for (variant, setting, label), group in df.groupby(["variant", "setting", "label"], sort=True):
        rows.append(
            {
                "diagnostic": "cnn_id_ablation",
                "variant": variant,
                "setting": setting,
                "label": label,
                "seeds": ";".join(str(int(v)) for v in sorted(group["seed"].unique())),
                "normal_recall_mean": float(group["normal_recall"].mean()),
                "normal_recall_std": float(group["normal_recall"].std(ddof=1)) if len(group) > 1 else 0.0,
                "fpr_mean": float(group["fpr"].mean()),
                "fpr_std": float(group["fpr"].std(ddof=1)) if len(group) > 1 else 0.0,
                "attack_recall_mean": float(group["attack_recall"].mean()),
                "attack_recall_std": float(group["attack_recall"].std(ddof=1)) if len(group) > 1 else 0.0,
                "source": "streaming can-train external re-evaluation with input-channel masking",
            }
        )

    original_by_setting = {
        row["setting"]: row
        for row in rows
        if row["variant"] == "original_existing_result"
    }
    for row in rows:
        original = original_by_setting.get(row["setting"])
        if original and row["variant"] != "original_existing_result":
            row["fpr_delta_vs_original"] = row["fpr_mean"] - original["fpr_mean"]
            row["normal_recall_delta_vs_original"] = row["normal_recall_mean"] - original["normal_recall_mean"]
            row["attack_recall_delta_vs_original"] = row["attack_recall_mean"] - original["attack_recall_mean"]
        else:
            row["fpr_delta_vs_original"] = ""
            row["normal_recall_delta_vs_original"] = ""
            row["attack_recall_delta_vs_original"] = ""
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="can-train external CNN input-channel ablation.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["can_id_masked", "id_only_non_id_masked"],
        choices=["can_id_masked", "id_only_non_id_masked"],
    )
    args = parser.parse_args()

    start = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    can_id_mean = car_hacking_train_normal_can_id_mean()
    feature_means = car_hacking_train_normal_window_feature_means()
    files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    models = [m for m in load_models() if m["setting"] in SELECTED_SETTINGS]
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"id-ablation {idx}/{len(files)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                with torch.no_grad():
                    for variant in args.variants:
                        x_variant = apply_variant(x, variant, can_id_mean, feature_means)
                        for model_info in models:
                            pred, _scores = evaluate_batch(model_info, x_variant)
                            key = (variant, model_info["setting"], model_info["label"], model_info["seed"])
                            metrics[key].update(y, pred)

    seed_rows: list[dict] = []
    for (variant, setting, label, seed), counter in sorted(metrics.items()):
        row = {
            "diagnostic": "cnn_id_ablation",
            "variant": variant,
            "setting": setting,
            "label": label,
            "seed": seed,
        }
        row.update(counter.row())
        seed_rows.append(row)

    original_summary = pd.read_csv(TABLES / "cantt_external_eval_summary_mean_std.csv")
    summary_rows = aggregate_seed_rows(seed_rows, original_summary)
    write_csv(TABLES / "cantt_id_ablation_by_seed.csv", seed_rows)
    write_csv(TABLES / "cantt_id_ablation_summary.csv", summary_rows)

    log = {
        "diagnostic": "cnn_id_ablation",
        "selected_settings": sorted(SELECTED_SETTINGS),
        "seeds": sorted({int(m["seed"]) for m in models}),
        "variants": args.variants,
        "can_id_mask_value": can_id_mean,
        "can_id_mask_value_source": "datasets/processed/car_hacking_train.parquet frames with binary_label_id == 0",
        "feature_order": FEATURE_NAMES,
        "non_id_mask_values": {FEATURE_NAMES[i]: float(feature_means[i]) for i in range(1, len(FEATURE_NAMES))},
        "non_id_mask_value_source": "datasets/windows/train_windows.npz windows with y_binary == 0, overlap-weighted frame mean",
        "outputs": [
            "results/tables/cantt_id_ablation_summary.csv",
            "results/tables/cantt_id_ablation_by_seed.csv",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "cantt_id_ablation.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
