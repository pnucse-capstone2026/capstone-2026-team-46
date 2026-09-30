#!/usr/bin/env python3
"""Tier 1 channel-ablation retraining.

Retrains CNNs with channel groups masked to Car-Hacking train-normal constants
in both training and evaluation, closing the input-masking confound of
scripts/run_cantt_id_ablation.py:

- non_id_only: can_id (channel 0) masked; payload/timing information kept.
- id_only:     dlc/data0-7/delta_t (channels 1-10) masked; can_id kept.

New models are written only to models/channel_ablation/ per AGENTS.md.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_cantt_id_ablation import (
    car_hacking_train_normal_can_id_mean,
    car_hacking_train_normal_window_feature_means,
)
from train_real_only_baselines import (
    CNN1D,
    WindowDataset,
    compute_metrics,
    fit_standardizer,
    predict_cnn,
    set_seed,
    standardize,
    train_cnn,
)
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
MODELS = ROOT / "models" / "channel_ablation"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

VARIANTS = ["non_id_only", "id_only"]
SETTINGS = ["real_only", "rule_0p30"]
SEEDS = [7, 42, 123, 2026, 3407]
RULE_RATIO = 0.30


def mask_channels(x: np.ndarray, variant: str, can_id_mean: float, feature_means: np.ndarray) -> np.ndarray:
    out = x.copy()
    if variant == "non_id_only":
        out[:, :, 0] = np.float32(can_id_mean)
    elif variant == "id_only":
        out[:, :, 1:] = feature_means[1:]
    else:
        raise ValueError(variant)
    return out


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


def append_indomain_rows(rows: list[dict]) -> None:
    path = TABLES / "channel_ablation_indomain_by_seed.csv"
    existing: list[dict] = []
    if path.exists():
        with path.open() as f:
            existing = list(csv.DictReader(f))
        keys = {(r["variant"], r["setting"], str(r["seed"])) for r in rows}
        existing = [r for r in existing if (r["variant"], r["setting"], str(r["seed"])) not in keys]
    write_csv(path, existing + rows)


def run_one(variant: str, setting: str, seed: int, max_epochs: int) -> None:
    start = time.time()
    print(f"=== channel_ablation variant={variant} setting={setting} seed={seed} ===", flush=True)
    set_seed(seed)
    MODELS.mkdir(parents=True, exist_ok=True)

    can_id_mean = car_hacking_train_normal_can_id_mean()
    feature_means = car_hacking_train_normal_window_feature_means()

    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    val = np.load(WINDOWS / "val_windows.npz", allow_pickle=True)
    test = np.load(WINDOWS / "test_windows.npz", allow_pickle=True)

    train_x_raw = train["x"].astype(np.float32)
    train_y = train["y_attack_type"].astype(np.int64)
    if setting == "rule_0p30":
        syn = np.load(SYNTHETIC / "rule_based_windows.npz", allow_pickle=True)
        n_syn = int(round(len(train_x_raw) * RULE_RATIO))
        syn_idx = sample_synthetic_indices(syn["y_attack_type"], n_syn, seed + int(RULE_RATIO * 1000))
        train_x_raw = np.concatenate([train_x_raw, syn["x"][syn_idx].astype(np.float32)], axis=0)
        train_y = np.concatenate([train_y, syn["y_attack_type"][syn_idx].astype(np.int64)], axis=0)
    elif setting != "real_only":
        raise ValueError(setting)

    n_real_train = len(train["x"])
    train_x_raw = mask_channels(train_x_raw, variant, can_id_mean, feature_means)
    val_x_raw = mask_channels(val["x"].astype(np.float32), variant, can_id_mean, feature_means)
    test_x_raw = mask_channels(test["x"].astype(np.float32), variant, can_id_mean, feature_means)

    # Fit on masked real train only (same isolation rule as the ratio sweep).
    mean, std = fit_standardizer(train_x_raw[:n_real_train])
    np.savez(MODELS / f"standardizer_{variant}_{setting}_seed{seed}.npz", mean=mean, std=std)

    train_x = standardize(train_x_raw, mean, std)
    val_x = standardize(val_x_raw, mean, std)
    test_x = standardize(test_x_raw, mean, std)
    val_y = val["y_attack_type"].astype(np.int64)
    test_y = test["y_attack_type"].astype(np.int64)

    model, history, device = train_cnn(train_x, train_y, val_x, val_y, max_epochs=max_epochs)
    torch.save(model.state_dict(), MODELS / f"cnn1d_{variant}_{setting}_seed{seed}.pt")

    loader = DataLoader(WindowDataset(test_x, test_y), batch_size=1024, shuffle=False, num_workers=2)
    true, probs = predict_cnn(model, loader, device)
    pred = probs.argmax(axis=1)
    attack_scores = 1.0 - probs[:, 0]
    row = compute_metrics(f"CNN1D_{variant}", "test", test_y, attack_scores, pred, seed)
    row["variant"] = variant
    row["setting"] = setting
    append_indomain_rows([row])

    log = {
        "variant": variant,
        "setting": setting,
        "seed": seed,
        "epochs_run": len(history),
        "best_val_macro_f1": max(h["val_macro_f1_multiclass"] for h in history),
        "masked_channels": "can_id" if variant == "non_id_only" else "dlc,data0-7,delta_t",
        "can_id_mask_value": can_id_mean,
        "non_id_mask_values_source": "datasets/windows/train_windows.npz y_binary==0 frame means",
        "train_windows": int(len(train_x_raw)),
        "indomain_test_macro_f1": row["macro_f1_binary"],
        "indomain_test_attack_recall": row["attack_recall"],
        "indomain_test_fpr": row["fpr"],
        "elapsed_seconds": time.time() - start,
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"train_channel_ablation_{variant}_{setting}_seed{seed}.log").write_text(
        json.dumps(log, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train channel-ablation CNNs (Tier 1).")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--setting", choices=SETTINGS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip configs whose model file already exists.")
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()

    if args.all:
        for variant in VARIANTS:
            for setting in SETTINGS:
                for seed in SEEDS:
                    if args.skip_existing and (MODELS / f"cnn1d_{variant}_{setting}_seed{seed}.pt").exists():
                        print(f"skip existing {variant} {setting} seed{seed}", flush=True)
                        continue
                    run_one(variant, setting, seed, args.max_epochs)
        return
    if args.variant is None or args.setting is None or args.seed is None:
        raise SystemExit("Provide --variant, --setting and --seed, or use --all")
    run_one(args.variant, args.setting, args.seed, args.max_epochs)


if __name__ == "__main__":
    main()
