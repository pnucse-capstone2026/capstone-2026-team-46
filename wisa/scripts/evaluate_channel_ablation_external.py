#!/usr/bin/env python3
"""Tier 1 channel-ablation external evaluation.

Evaluates the retrained non_id_only / id_only CNNs (models/channel_ablation/)
on Car-Hacking test, OTIDS cross windows, and can-train-and-test external
test subsets, applying the same channel masking used during training before
each model's own standardizer.
"""
import csv
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_cantt_id_ablation import (
    car_hacking_train_normal_can_id_mean,
    car_hacking_train_normal_window_feature_means,
)
from run_cantt_tier1 import (
    LOGS,
    TABLES,
    ZIP_PATH,
    MetricCounter,
    evaluate_batch,
    iter_windows,
    list_cantt_files,
    read_cantt_csv,
)
from train_channel_ablation import SEEDS, SETTINGS, VARIANTS, mask_channels, write_csv
from train_real_only_baselines import CNN1D

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
MODELS = ROOT / "models" / "channel_ablation"

SETTING_LABELS = {"real_only": "Real only", "rule_0p30": "Rule +30%"}


def load_ablation_models() -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for variant in VARIANTS:
        for setting in SETTINGS:
            for seed in SEEDS:
                stdz = np.load(MODELS / f"standardizer_{variant}_{setting}_seed{seed}.npz")
                model = CNN1D(in_channels=11, classes=5).to(device)
                model.load_state_dict(
                    torch.load(MODELS / f"cnn1d_{variant}_{setting}_seed{seed}.pt", map_location=device)
                )
                model.eval()
                loaded.append(
                    {
                        "variant": variant,
                        "setting": setting,
                        "label": f"{SETTING_LABELS[setting]} ({variant})",
                        "seed": seed,
                        "model": model,
                        "mean": stdz["mean"].astype(np.float32),
                        "std": stdz["std"].astype(np.float32),
                        "device": device,
                    }
                )
    return loaded


def evaluate_local_npz(models, metrics, dataset_name, npz_name, can_id_mean, feature_means, chunk=8192):
    data = np.load(WINDOWS / npz_name, allow_pickle=True)
    x_all = data["x"]
    y_all = data["y_binary"].astype(np.int8)
    for start in range(0, len(x_all), chunk):
        x_raw = x_all[start : start + chunk].astype(np.float32)
        y = y_all[start : start + chunk]
        for variant in VARIANTS:
            x_masked = mask_channels(x_raw, variant, can_id_mean, feature_means)
            for info in models:
                if info["variant"] != variant:
                    continue
                pred, _scores = evaluate_batch(info, x_masked)
                metrics[(dataset_name, variant, info["setting"], info["seed"])].update(y, pred)


def evaluate_cantt(models, metrics, can_id_mean, feature_means):
    files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"channel-ablation cantt {idx}/{len(files)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            with torch.no_grad():
                for x, y, _starts in iter_windows(features, frame_y):
                    for variant in VARIANTS:
                        x_masked = mask_channels(x, variant, can_id_mean, feature_means)
                        for info in models:
                            if info["variant"] != variant:
                                continue
                            pred, _scores = evaluate_batch(info, x_masked)
                            metrics[("cantt_external", variant, info["setting"], info["seed"])].update(y, pred)


def summarize(seed_rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(seed_rows)
    rows = []
    for (dataset, variant, setting), group in df.groupby(["dataset", "variant", "setting"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "setting": setting,
                "label": f"{SETTING_LABELS[setting]} ({variant})",
                "seeds": ";".join(str(int(s)) for s in sorted(group["seed"].unique())),
                "normal_recall_mean": float(group["normal_recall"].mean()),
                "normal_recall_std": float(group["normal_recall"].std(ddof=1)) if len(group) > 1 else 0.0,
                "fpr_mean": float(group["fpr"].mean()),
                "fpr_std": float(group["fpr"].std(ddof=1)) if len(group) > 1 else 0.0,
                "attack_recall_mean": float(group["attack_recall"].mean()),
                "attack_recall_std": float(group["attack_recall"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return rows


def key_comparisons(summary_rows: list[dict]) -> list[dict]:
    rows = []
    original = pd.read_csv(TABLES / "cantt_external_eval_summary_mean_std.csv")
    masking = pd.read_csv(TABLES / "cantt_id_ablation_summary.csv")
    for setting in SETTINGS:
        row = {"setting": setting, "label": SETTING_LABELS[setting], "dataset": "cantt_external"}
        orig = original[original["setting"] == setting]
        row["fpr_original_full_input"] = float(orig["fpr_mean"].iloc[0]) if len(orig) else ""
        mask_sel = masking[(masking["setting"] == setting) & (masking["variant"] == "can_id_masked")]
        row["fpr_input_masking_can_id"] = float(mask_sel["fpr_mean"].iloc[0]) if len(mask_sel) else ""
        for variant in VARIANTS:
            match = [
                r
                for r in summary_rows
                if r["dataset"] == "cantt_external" and r["setting"] == setting and r["variant"] == variant
            ]
            row[f"fpr_retrained_{variant}"] = match[0]["fpr_mean"] if match else ""
        rows.append(row)
    return rows


def main() -> None:
    start = time.time()
    can_id_mean = car_hacking_train_normal_can_id_mean()
    feature_means = car_hacking_train_normal_window_feature_means()
    models = load_ablation_models()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)

    print("evaluating Car-Hacking test (masked, in-domain control)", flush=True)
    evaluate_local_npz(models, metrics, "car_hacking_test", "test_windows.npz", can_id_mean, feature_means)
    print("evaluating OTIDS cross windows (masked)", flush=True)
    evaluate_local_npz(models, metrics, "otids_cross", "otids_cross_windows.npz", can_id_mean, feature_means)
    evaluate_cantt(models, metrics, can_id_mean, feature_means)

    seed_rows = []
    for (dataset, variant, setting, seed), counter in sorted(metrics.items()):
        row = {"dataset": dataset, "variant": variant, "setting": setting, "seed": seed}
        row.update(counter.row())
        seed_rows.append(row)
    summary_rows = summarize(seed_rows)
    comparison_rows = key_comparisons(summary_rows)

    write_csv(TABLES / "channel_ablation_external_by_seed.csv", seed_rows)
    write_csv(TABLES / "channel_ablation_external_summary.csv", summary_rows)
    write_csv(TABLES / "channel_ablation_key_comparisons.csv", comparison_rows)

    log = {
        "diagnostic": "channel_ablation_external",
        "variants": VARIANTS,
        "settings": SETTINGS,
        "seeds": SEEDS,
        "datasets": ["car_hacking_test", "otids_cross", "cantt_external"],
        "masking": "same train-normal constants as training, applied before each model standardizer",
        "outputs": [
            "results/tables/channel_ablation_external_by_seed.csv",
            "results/tables/channel_ablation_external_summary.csv",
            "results/tables/channel_ablation_key_comparisons.csv",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "channel_ablation_external.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
