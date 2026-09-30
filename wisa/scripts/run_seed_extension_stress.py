#!/usr/bin/env python3
"""Five-seed extension for sensitivity and stress rungs.

This diagnostic intentionally avoids rewriting the canonical baseline,
ratio-sweep, and oversampling model files. Existing seeds 7/42/123 are read
from their original locations; new seeds are trained into
models/main_seed_extension/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from train_real_only_baselines import CNN1D, WindowDataset, fit_standardizer, predict_cnn, standardize
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "models"
EXT_MODELS = MODELS / "main_seed_extension"
EXP = ROOT / "experiments" / "14_main_seed_extension"

OLD_SEEDS = [7, 42, 123]
EXTRA_SEEDS = [2026, 3407]
ALL_SEEDS = [7, 42, 123, 2026, 3407]

SETTINGS = [
    {"setting": "real_only", "family": "real_only", "ratio": 0.0, "label": "Real only"},
    {"setting": "real_oversampling_0p50", "family": "oversampling", "ratio": 0.5, "label": "Real oversampling +50%"},
    {"setting": "rule_0p10", "family": "rule", "ratio": 0.1, "label": "Rule +10%"},
    {"setting": "rule_0p30", "family": "rule", "ratio": 0.3, "label": "Rule +30%"},
    {"setting": "rule_0p50", "family": "rule", "ratio": 0.5, "label": "Rule +50%"},
    {"setting": "rule_1p00", "family": "rule", "ratio": 1.0, "label": "Rule +100%"},
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ratio_suffix(ratio: float, seed: int) -> str:
    return f"rule_ratio{ratio:.2f}_seed{seed}".replace(".", "p")


def setting_slug(setting: dict, seed: int) -> str:
    if setting["family"] == "real_only":
        return f"real_only_seed{seed}"
    if setting["family"] == "oversampling":
        return f"real_oversampling_ratio0p50_seed{seed}"
    if setting["family"] == "rule":
        return ratio_suffix(setting["ratio"], seed)
    raise ValueError(setting["family"])


def original_model_paths(setting: dict, seed: int) -> tuple[Path, Path]:
    family = setting["family"]
    if family == "real_only":
        return MODELS / "baseline" / f"cnn1d_real_only_seed{seed}.pt", MODELS / "baseline" / "cnn_standardizer.npz"
    if family == "oversampling":
        suffix = f"real_oversampling_ratio0p50_seed{seed}"
        return MODELS / "oversampling" / f"cnn1d_{suffix}.pt", MODELS / "oversampling" / f"standardizer_{suffix}.npz"
    if family == "rule":
        suffix = ratio_suffix(setting["ratio"], seed)
        return MODELS / "ratio_sweep" / f"cnn1d_{suffix}.pt", MODELS / "ratio_sweep" / f"standardizer_{suffix}.npz"
    raise ValueError(family)


def extension_model_paths(setting: dict, seed: int) -> tuple[Path, Path]:
    slug = setting_slug(setting, seed)
    return EXT_MODELS / f"cnn1d_{slug}.pt", EXT_MODELS / f"standardizer_{slug}.npz"


def model_paths(setting: dict, seed: int) -> tuple[Path, Path]:
    if seed in OLD_SEEDS:
        return original_model_paths(setting, seed)
    return extension_model_paths(setting, seed)


def train_model(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, seed: int, max_epochs: int, patience: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        WindowDataset(train_x, train_y),
        batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(WindowDataset(val_x, val_y), batch_size=1024, shuffle=False, num_workers=2)
    model = CNN1D(in_channels=train_x.shape[-1], classes=5).to(device)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(5), y=train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = {"f1": -1.0, "state": None, "epoch": 0}
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y_batch)
        val_true, val_probs = predict_cnn(model, val_loader, device)
        val_pred = val_probs.argmax(axis=1)
        val_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y), "val_macro_f1_multiclass": val_f1})
        print(f"seed={seed} epoch={epoch} loss={history[-1]['train_loss']:.5f} val_macro_f1={val_f1:.5f}", flush=True)
        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best["state"])
    return model, history, device, best


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    yb = (y_true > 0).astype(np.int8)
    pb = (pred > 0).astype(np.int8)
    normal_mask = yb == 0
    attack_mask = yb == 1
    fp = int((normal_mask & (pb == 1)).sum())
    tn = int((normal_mask & (pb == 0)).sum())
    fn = int((attack_mask & (pb == 0)).sum())
    tp = int((attack_mask & (pb == 1)).sum())
    return {
        "accuracy": float((yb == pb).mean()),
        "macro_f1_binary": float(f1_score(yb, pb, average="macro", zero_division=0)),
        "attack_detection_recall": float(tp / max(tp + fn, 1)),
        "normal_recall": float(tn / max(tn + fp, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
        "auprc": float(average_precision_score(yb, 1.0 - probs[:, 0])) if len(np.unique(yb)) > 1 else "",
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def predict(setting: dict, seed: int, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_path, std_path = model_paths(setting, seed)
    if not state_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Missing model artifacts for {setting['setting']} seed {seed}: {state_path}, {std_path}")
    stdz = np.load(std_path)
    x_std = standardize(x, stdz["mean"], stdz["std"])
    model = CNN1D(in_channels=x.shape[-1], classes=5).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    loader = DataLoader(WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(model, loader, device)
    return probs


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["seeds"] = ";".join(str(s) for s in sorted(group["seed"].unique()))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals):
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_training_set(setting: dict, seed: int, train: np.lib.npyio.NpzFile, synthetic: np.lib.npyio.NpzFile | None) -> tuple[np.ndarray, np.ndarray, int]:
    train_x = train["x"]
    train_y = train["y_attack_type"].astype(np.int64)
    if setting["family"] == "real_only":
        return train_x.astype(np.float32), train_y, 0
    added = int(round(len(train_x) * setting["ratio"]))
    if setting["family"] == "oversampling":
        attack_idx = np.where(train_y > 0)[0]
        sampled_local = sample_synthetic_indices(train_y[attack_idx], added, seed + int(setting["ratio"] * 1000) + 101)
        sampled_idx = attack_idx[sampled_local]
        extra_x = train_x[sampled_idx]
        extra_y = train_y[sampled_idx]
    elif setting["family"] == "rule":
        if synthetic is None:
            raise ValueError("Synthetic windows are required for rule settings")
        syn_y = synthetic["y_attack_type"].astype(np.int64)
        syn_idx = sample_synthetic_indices(syn_y, added, seed + int(setting["ratio"] * 1000))
        extra_x = synthetic["x"][syn_idx]
        extra_y = syn_y[syn_idx]
    else:
        raise ValueError(setting["family"])
    out_x = np.concatenate([train_x, extra_x], axis=0).astype(np.float32)
    out_y = np.concatenate([train_y, extra_y], axis=0).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(out_y))
    return out_x[order], out_y[order], added


def train_missing(extra_seeds: list[int], max_epochs_real: int, max_epochs_aug: int) -> None:
    EXT_MODELS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    val = np.load(WINDOWS / "val_windows.npz", allow_pickle=True)
    variant = np.load(WINDOWS / "variant_test_windows.npz", allow_pickle=True)
    synthetic = np.load(SYNTHETIC / "rule_based_windows.npz", allow_pickle=True)
    val_y = val["y_attack_type"].astype(np.int64)
    rows = []
    for seed in extra_seeds:
        for setting in SETTINGS:
            model_path, std_path = extension_model_paths(setting, seed)
            if model_path.exists() and std_path.exists():
                print(f"skip existing {setting['setting']} seed={seed}", flush=True)
                continue
            print(f"=== train extension {setting['setting']} seed={seed} ===", flush=True)
            set_seed(seed)
            train_x_raw, train_y, added = build_training_set(setting, seed, train, synthetic)
            mean, std = fit_standardizer(train["x"])
            train_x = standardize(train_x_raw, mean, std)
            val_x = standardize(val["x"], mean, std)
            max_epochs = max_epochs_real if setting["family"] == "real_only" else max_epochs_aug
            patience = 4 if setting["family"] == "real_only" else 3
            model, history, device, best = train_model(train_x, train_y, val_x, val_y, seed, max_epochs, patience)
            torch.save(model.state_dict(), model_path)
            np.savez(std_path, mean=mean, std=std)
            write_csv(EXP / f"training_history_{setting_slug(setting, seed)}.csv", history)

            # Cheap sanity metric on fixed variant, used only as a run log.
            variant_x = standardize(variant["x"], mean, std)
            loader = DataLoader(WindowDataset(variant_x, variant["y_attack_type"].astype(np.int64)), batch_size=1024, shuffle=False, num_workers=2)
            _, probs = predict_cnn(model, loader, device)
            pred = probs.argmax(axis=1)
            y_attack = variant["y_attack_type"].astype(np.int64)
            metrics = binary_metrics(y_attack, probs)
            rows.append(
                {
                    "setting": setting["setting"],
                    "label": setting["label"],
                    "seed": seed,
                    "added_windows": added,
                    "best_epoch": best["epoch"],
                    "best_val_macro_f1_multiclass": best["f1"],
                    "variant_attack_recall": metrics["attack_detection_recall"],
                    "variant_exact_macro_f1_binary": metrics["macro_f1_binary"],
                    "variant_pred_non_normal_rate": float((pred > 0).mean()),
                }
            )
            write_csv(TABLES / "main_seed_extension_training_sanity.csv", rows)


def evaluate_variant_sensitivity(seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(WINDOWS / "variant_sensitivity_windows.npz", allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y_attack_type"].astype(np.int64)
    scenario = data["scenario_id"].astype(str)
    attack_type = data["attack_type"].astype(str)
    severity = data["severity"].astype(str)
    rows = []
    scenario_rows = []
    for setting in SETTINGS:
        for seed in seeds:
            print(f"eval sensitivity {setting['setting']} seed={seed}", flush=True)
            probs = predict(setting, seed, x)
            pred = probs.argmax(axis=1)
            row = {"setting": setting["setting"], "label": setting["label"], "family": setting["family"], "ratio": setting["ratio"], "seed": seed, "split": "variant_sensitivity"}
            row.update(binary_metrics(y, probs))
            rows.append(row)
            for sid in sorted(set(scenario)):
                mask = scenario == sid
                y_s = y[mask]
                pred_s = pred[mask]
                probs_s = probs[mask]
                is_attack = y_s > 0
                scenario_rows.append(
                    {
                        "setting": setting["setting"],
                        "label": setting["label"],
                        "family": setting["family"],
                        "ratio": setting["ratio"],
                        "seed": seed,
                        "scenario_id": sid,
                        "attack_type": str(attack_type[mask][0]),
                        "severity": str(severity[mask][0]),
                        "windows": int(mask.sum()),
                        "attack_detection_recall": float(((pred_s > 0) & is_attack).sum() / max(is_attack.sum(), 1)) if is_attack.any() else "",
                        "exact_class_recall": float((pred_s[is_attack] == y_s[is_attack]).sum() / max(is_attack.sum(), 1)) if is_attack.any() else "",
                        "normal_recall": float((pred_s == 0).mean()) if not is_attack.any() else "",
                        "mean_attack_score": float((1.0 - probs_s[:, 0]).mean()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(scenario_rows)


def evaluate_id_stress(filename: str, split: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(WINDOWS / filename, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y_attack_type"].astype(np.int64)
    scenario = data["scenario_id"].astype(str)
    attack_type = data["attack_type"].astype(str)
    id_condition = data["id_condition"].astype(str)
    rows = []
    scenario_rows = []
    for setting in SETTINGS:
        for seed in seeds:
            print(f"eval {split} {setting['setting']} seed={seed}", flush=True)
            probs = predict(setting, seed, x)
            pred = probs.argmax(axis=1)
            row = {"setting": setting["setting"], "label": setting["label"], "family": setting["family"], "ratio": setting["ratio"], "seed": seed, "split": split}
            row.update(binary_metrics(y, probs))
            rows.append(row)
            for sid in sorted(set(scenario)):
                mask = scenario == sid
                y_s = y[mask]
                pred_s = pred[mask]
                probs_s = probs[mask]
                is_attack = y_s > 0
                scenario_rows.append(
                    {
                        "setting": setting["setting"],
                        "label": setting["label"],
                        "family": setting["family"],
                        "ratio": setting["ratio"],
                        "seed": seed,
                        "scenario_id": sid,
                        "attack_type": str(attack_type[mask][0]),
                        "id_condition": str(id_condition[mask][0]),
                        "windows": int(mask.sum()),
                        "attack_detection_recall": float(((pred_s > 0) & is_attack).sum() / max(is_attack.sum(), 1)) if is_attack.any() else "",
                        "exact_class_recall": float((pred_s[is_attack] == y_s[is_attack]).sum() / max(is_attack.sum(), 1)) if is_attack.any() else "",
                        "normal_recall": float((pred_s == 0).mean()) if not is_attack.any() else "",
                        "mean_attack_score": float((1.0 - probs_s[:, 0]).mean()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(scenario_rows)


def write_id_key(prefix: str, scenario_summary: pd.DataFrame) -> pd.DataFrame:
    attack_rows = scenario_summary[scenario_summary["attack_type"].isin(["Gear", "RPM"])].copy()
    rows = []
    for (setting, label, family, ratio, attack), group in attack_rows.groupby(["setting", "label", "family", "ratio", "attack_type"], sort=False):
        can = group[group["id_condition"] == "canonical"]
        shifted = group[group["id_condition"] == "shifted"]
        if can.empty or shifted.empty:
            continue
        rows.append(
            {
                "setting": setting,
                "label": label,
                "family": family,
                "ratio": ratio,
                "attack_type": attack,
                "canonical_exact_recall_mean": float(can["exact_class_recall_mean"].iloc[0]),
                "canonical_exact_recall_std": float(can["exact_class_recall_std"].iloc[0]),
                "shifted_exact_recall_mean": float(shifted["exact_class_recall_mean"].iloc[0]),
                "shifted_exact_recall_std": float(shifted["exact_class_recall_std"].iloc[0]),
                "exact_recall_drop": float(can["exact_class_recall_mean"].iloc[0] - shifted["exact_class_recall_mean"].iloc[0]),
                "canonical_attack_detection_recall_mean": float(can["attack_detection_recall_mean"].iloc[0]),
                "canonical_attack_detection_recall_std": float(can["attack_detection_recall_std"].iloc[0]),
                "shifted_attack_detection_recall_mean": float(shifted["attack_detection_recall_mean"].iloc[0]),
                "shifted_attack_detection_recall_std": float(shifted["attack_detection_recall_std"].iloc[0]),
                "attack_detection_recall_drop": float(can["attack_detection_recall_mean"].iloc[0] - shifted["attack_detection_recall_mean"].iloc[0]),
                "seeds": str(can["seeds"].iloc[0]),
                "n_seeds": int(can["n_seeds"].iloc[0]),
            }
        )
    key = pd.DataFrame(rows)
    key.to_csv(TABLES / f"main_seed_extension_{prefix}_key_comparisons.csv", index=False)
    return key


def write_headline(
    sensitivity_summary: pd.DataFrame,
    sensitivity_scenario_summary: pd.DataFrame,
    target_key: pd.DataFrame,
    out_key: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for setting in ["real_only", "real_oversampling_0p50", "rule_0p10", "rule_0p30", "rule_0p50", "rule_1p00"]:
        row = sensitivity_summary[sensitivity_summary["setting"] == setting].iloc[0]
        rows.append(
            {
                "scope": "variant_sensitivity_overall",
                "setting": setting,
                "attack_type": "all",
                "condition": "all",
                "metric": "attack_detection_recall",
                "mean": row["attack_detection_recall_mean"],
                "std": row["attack_detection_recall_std"],
                "seeds": row["seeds"],
                "n_seeds": row["n_seeds"],
            }
        )
    low = sensitivity_scenario_summary[
        (sensitivity_scenario_summary["severity"] == "low")
        & (sensitivity_scenario_summary["attack_type"].isin(["Fuzzy", "Gear", "RPM"]))
        & (sensitivity_scenario_summary["setting"] == "rule_1p00")
    ]
    for _, row in low.iterrows():
        rows.append(
            {
                "scope": "low_intensity_rule_1p00",
                "setting": row["setting"],
                "attack_type": row["attack_type"],
                "condition": "low",
                "metric": "attack_detection_recall",
                "mean": row["attack_detection_recall_mean"],
                "std": row["attack_detection_recall_std"],
                "seeds": row["seeds"],
                "n_seeds": row["n_seeds"],
            }
        )
        rows.append(
            {
                "scope": "low_intensity_rule_1p00",
                "setting": row["setting"],
                "attack_type": row["attack_type"],
                "condition": "low",
                "metric": "exact_class_recall",
                "mean": row["exact_class_recall_mean"],
                "std": row["exact_class_recall_std"],
                "seeds": row["seeds"],
                "n_seeds": row["n_seeds"],
            }
        )
    for source, key in [("target_id_shift", target_key), ("out_of_generator", out_key)]:
        for _, row in key[key["setting"].isin(["real_only", "rule_0p30", "rule_1p00"])].iterrows():
            rows.append(
                {
                    "scope": source,
                    "setting": row["setting"],
                    "attack_type": row["attack_type"],
                    "condition": "canonical_exact_recall",
                    "metric": "exact_class_recall",
                    "mean": row["canonical_exact_recall_mean"],
                    "std": row["canonical_exact_recall_std"],
                    "seeds": row["seeds"],
                    "n_seeds": row["n_seeds"],
                }
            )
            rows.append(
                {
                    "scope": source,
                    "setting": row["setting"],
                    "attack_type": row["attack_type"],
                    "condition": "shifted_exact_recall",
                    "metric": "exact_class_recall",
                    "mean": row["shifted_exact_recall_mean"],
                    "std": row["shifted_exact_recall_std"],
                    "seeds": row["seeds"],
                    "n_seeds": row["n_seeds"],
                }
            )
    headline = pd.DataFrame(rows)
    headline.to_csv(TABLES / "main_seed_extension_headline.csv", index=False)
    return headline


def evaluate_all(seeds: list[int]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    sensitivity, sensitivity_scenario = evaluate_variant_sensitivity(seeds)
    sensitivity.to_csv(TABLES / "main_seed_extension_variant_sensitivity_metrics.csv", index=False)
    sensitivity_scenario.to_csv(TABLES / "main_seed_extension_variant_sensitivity_by_scenario_seed.csv", index=False)
    sensitivity_summary = summarize(
        sensitivity,
        ["setting", "label", "family", "ratio", "split"],
        ["accuracy", "macro_f1_binary", "attack_detection_recall", "normal_recall", "fpr", "auprc"],
    )
    sensitivity_scenario_summary = summarize(
        sensitivity_scenario,
        ["setting", "label", "family", "ratio", "scenario_id", "attack_type", "severity"],
        ["attack_detection_recall", "exact_class_recall", "normal_recall", "mean_attack_score"],
    )
    sensitivity_summary.to_csv(TABLES / "main_seed_extension_variant_sensitivity_summary.csv", index=False)
    sensitivity_scenario_summary.to_csv(TABLES / "main_seed_extension_variant_sensitivity_scenario_summary.csv", index=False)

    target, target_scenario = evaluate_id_stress("target_id_shift_stress_windows.npz", "target_id_shift_stress", seeds)
    target.to_csv(TABLES / "main_seed_extension_target_id_shift_metrics.csv", index=False)
    target_scenario.to_csv(TABLES / "main_seed_extension_target_id_shift_by_scenario_seed.csv", index=False)
    target_summary = summarize(
        target,
        ["setting", "label", "family", "ratio", "split"],
        ["accuracy", "macro_f1_binary", "attack_detection_recall", "normal_recall", "fpr", "auprc"],
    )
    target_scenario_summary = summarize(
        target_scenario,
        ["setting", "label", "family", "ratio", "scenario_id", "attack_type", "id_condition"],
        ["attack_detection_recall", "exact_class_recall", "normal_recall", "mean_attack_score"],
    )
    target_summary.to_csv(TABLES / "main_seed_extension_target_id_shift_summary.csv", index=False)
    target_scenario_summary.to_csv(TABLES / "main_seed_extension_target_id_shift_scenario_summary.csv", index=False)
    target_key = write_id_key("target_id_shift", target_scenario_summary)

    out, out_scenario = evaluate_id_stress("out_of_generator_stress_windows.npz", "out_of_generator_stress", seeds)
    out.to_csv(TABLES / "main_seed_extension_out_of_generator_metrics.csv", index=False)
    out_scenario.to_csv(TABLES / "main_seed_extension_out_of_generator_by_scenario_seed.csv", index=False)
    out_summary = summarize(
        out,
        ["setting", "label", "family", "ratio", "split"],
        ["accuracy", "macro_f1_binary", "attack_detection_recall", "normal_recall", "fpr", "auprc"],
    )
    out_scenario_summary = summarize(
        out_scenario,
        ["setting", "label", "family", "ratio", "scenario_id", "attack_type", "id_condition"],
        ["attack_detection_recall", "exact_class_recall", "normal_recall", "mean_attack_score"],
    )
    out_summary.to_csv(TABLES / "main_seed_extension_out_of_generator_summary.csv", index=False)
    out_scenario_summary.to_csv(TABLES / "main_seed_extension_out_of_generator_scenario_summary.csv", index=False)
    out_key = write_id_key("out_of_generator", out_scenario_summary)

    headline = write_headline(sensitivity_summary, sensitivity_scenario_summary, target_key, out_key)
    print(headline.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--extra-seeds", type=int, nargs="*", default=EXTRA_SEEDS)
    parser.add_argument("--seeds", type=int, nargs="*", default=ALL_SEEDS)
    parser.add_argument("--max-epochs-real", type=int, default=20)
    parser.add_argument("--max-epochs-aug", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_missing and not args.evaluate:
        args.train_missing = True
        args.evaluate = True
    config = {
        "extra_seeds": args.extra_seeds,
        "evaluation_seeds": args.seeds,
        "max_epochs_real": args.max_epochs_real,
        "max_epochs_aug": args.max_epochs_aug,
        "model_dir": str(EXT_MODELS.relative_to(ROOT)),
        "note": "Existing seeds are read from canonical model dirs; extra seeds are trained only under models/main_seed_extension.",
    }
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)
    if args.train_missing:
        train_missing(args.extra_seeds, args.max_epochs_real, args.max_epochs_aug)
    if args.evaluate:
        evaluate_all(args.seeds)


if __name__ == "__main__":
    main()
