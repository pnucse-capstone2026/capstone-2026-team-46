#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from train_real_only_baselines import CNN1D, compute_metrics, fit_standardizer, predict_cnn, standardize


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
EXP = ROOT / "experiments" / "04_augmented_ids"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "models" / "ratio_sweep"

RATIOS = [0.10, 0.30, 0.50, 1.00]
SEEDS = [42, 7, 123]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_npz(name, base=WINDOWS):
    return np.load(base / name, allow_pickle=True)


def sample_synthetic_indices(y_syn, total_count, seed):
    rng = np.random.default_rng(seed)
    classes = [1, 2, 3, 4]
    per_class = total_count // len(classes)
    remainder = total_count - per_class * len(classes)
    indices = []
    for i, cls in enumerate(classes):
        cls_idx = np.where(y_syn == cls)[0]
        n = per_class + (1 if i < remainder else 0)
        replace = n > len(cls_idx)
        indices.append(rng.choice(cls_idx, size=n, replace=replace))
    out = np.concatenate(indices)
    rng.shuffle(out)
    return out


def train_model(train_x, train_y, val_x, val_y, seed, max_epochs=12):
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
    patience = 3
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
        val_true, val_probs = predict_cnn(model, val_loader, device)
        val_pred = val_probs.argmax(axis=1)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y), "val_macro_f1_multiclass": val_f1})
        print(f"seed={seed} epoch={epoch} loss={history[-1]['train_loss']:.5f} val_macro_f1={val_f1:.5f}")
        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best["state"])
    return model, history, device, best


def evaluate(model, device, split_name, x, y_attack, seed, ratio, generator):
    loader = DataLoader(WindowDataset(x, y_attack), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(model, loader, device)
    pred = probs.argmax(axis=1)
    scores = 1.0 - probs[:, 0]
    row = compute_metrics(f"CNN1D+{generator}", split_name, y_attack, scores, pred, seed)
    row["ratio"] = ratio
    row["generator"] = generator
    return row


def write_csv(path, rows):
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


def aggregate_and_plot():
    files = sorted(TABLES.glob("ratio_sweep_rule_seed*.csv"))
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    df.to_csv(TABLES / "ratio_sweep_results.csv", index=False)
    metrics = ["accuracy", "macro_f1_binary", "attack_recall", "normal_recall", "fpr", "fnr", "auprc", "ece"]
    rows = []
    for (ratio, split), group in df.groupby(["ratio", "split"], sort=True):
        row = {"generator": "rule_based", "ratio": ratio, "split": split, "seeds": ";".join(str(s) for s in sorted(group["seed"].unique()))}
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "ratio_sweep_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    for split in ["test", "variant_test", "otids_cross_binary"]:
        sub = summary[summary["split"] == split].sort_values("ratio")
        plt.errorbar(sub["ratio"], sub["macro_f1_binary_mean"], yerr=sub["macro_f1_binary_std"], marker="o", capsize=4, label=split)
    plt.xlabel("Synthetic augmentation ratio")
    plt.ylabel("Binary Macro-F1")
    plt.title("Rule-based synthetic ratio sweep")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "macro_f1_by_ratio.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for split in ["test", "variant_test", "otids_cross_binary"]:
        sub = summary[summary["split"] == split].sort_values("ratio")
        plt.errorbar(sub["ratio"], sub["fpr_mean"], yerr=sub["fpr_std"], marker="o", capsize=4, label=split)
    plt.xlabel("Synthetic augmentation ratio")
    plt.ylabel("FPR")
    plt.title("FPR by rule-based synthetic ratio")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fpr_by_ratio.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for split in ["test", "variant_test", "otids_cross_binary"]:
        sub = summary[summary["split"] == split].sort_values("ratio")
        plt.errorbar(sub["ratio"], sub["ece_mean"], yerr=sub["ece_std"], marker="o", capsize=4, label=split)
    plt.xlabel("Synthetic augmentation ratio")
    plt.ylabel("ECE")
    plt.title("Calibration error by rule-based synthetic ratio")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "ece_by_ratio.png", dpi=180)
    plt.close()
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    if args.all:
        for seed in SEEDS:
            for ratio in RATIOS:
                run_one(seed, ratio, args.max_epochs)
        aggregate_and_plot()
    else:
        if args.seed is None or args.ratio is None:
            raise SystemExit("Provide --seed and --ratio, or use --all")
        run_one(args.seed, args.ratio, args.max_epochs)


def run_one(seed, ratio, max_epochs):
    set_seed(seed)
    print(f"=== rule ratio={ratio} seed={seed} ===")
    train = load_npz("train_windows.npz")
    val = load_npz("val_windows.npz")
    test = load_npz("test_windows.npz")
    cross = load_npz("otids_cross_windows.npz")
    variant = load_npz("variant_test_windows.npz")
    syn = load_npz("rule_based_windows.npz", SYNTHETIC)

    n_syn = int(round(len(train["x"]) * ratio))
    syn_idx = sample_synthetic_indices(syn["y_attack_type"], n_syn, seed + int(ratio * 1000))
    train_x_raw = np.concatenate([train["x"], syn["x"][syn_idx]], axis=0).astype(np.float32)
    train_y = np.concatenate([train["y_attack_type"], syn["y_attack_type"][syn_idx]], axis=0).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(train_y))
    train_x_raw = train_x_raw[order]
    train_y = train_y[order]

    mean, std = fit_standardizer(train["x"])  # fit on real train only to isolate augmentation effect.
    train_x = standardize(train_x_raw, mean, std)
    val_x = standardize(val["x"], mean, std)
    test_x = standardize(test["x"], mean, std)
    cross_x = standardize(cross["x"], mean, std)
    variant_x = standardize(variant["x"], mean, std)

    model, history, device, best = train_model(train_x, train_y, val_x, val["y_attack_type"].astype(np.int64), seed, max_epochs=max_epochs)
    suffix = f"rule_ratio{ratio:.2f}_seed{seed}".replace(".", "p")
    torch.save(model.state_dict(), MODELS / f"cnn1d_{suffix}.pt")
    np.savez(MODELS / f"standardizer_{suffix}.npz", mean=mean, std=std)
    write_csv(EXP / f"training_history_{suffix}.csv", history)

    rows = []
    rows.append(evaluate(model, device, "val", val_x, val["y_attack_type"].astype(np.int64), seed, ratio, "rule_based"))
    rows.append(evaluate(model, device, "test", test_x, test["y_attack_type"].astype(np.int64), seed, ratio, "rule_based"))
    rows.append(evaluate(model, device, "variant_test", variant_x, variant["y_attack_type"].astype(np.int64), seed, ratio, "rule_based"))
    rows.append(evaluate(model, device, "otids_cross_binary", cross_x, cross["y_binary"].astype(np.int8), seed, ratio, "rule_based"))
    for row in rows:
        row["synthetic_windows"] = n_syn
        row["best_epoch"] = best["epoch"]
        row["best_val_macro_f1_multiclass"] = best["f1"]
    write_csv(TABLES / f"ratio_sweep_rule_seed{seed}_ratio{ratio:.2f}.csv", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
