#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from train_real_only_baselines import CNN1D, compute_metrics, fit_standardizer, predict_cnn, standardize
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
EXP = ROOT / "experiments" / "02_baseline"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "models" / "oversampling"

SEEDS = [42, 7, 123]
RATIO = 0.50


def set_seed(seed: int) -> None:
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


def train_model(train_x, train_y, val_x, val_y, seed, max_epochs=12):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(WindowDataset(train_x, train_y), batch_size=512, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
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
        val_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
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


def evaluate(model, device, split_name, x, y_attack, seed, added_windows):
    loader = DataLoader(WindowDataset(x, y_attack), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(model, loader, device)
    pred = probs.argmax(axis=1)
    scores = 1.0 - probs[:, 0]
    row = compute_metrics("CNN1D+real_oversampling", split_name, y_attack, scores, pred, seed)
    row["ratio"] = RATIO
    row["oversampled_windows"] = added_windows
    return row


def run_one(seed: int, max_epochs: int) -> list[dict]:
    print(f"=== real oversampling ratio={RATIO} seed={seed} ===")
    set_seed(seed)
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    val = np.load(WINDOWS / "val_windows.npz", allow_pickle=True)
    test = np.load(WINDOWS / "test_windows.npz", allow_pickle=True)
    variant = np.load(WINDOWS / "variant_test_windows.npz", allow_pickle=True)
    cross = np.load(WINDOWS / "otids_cross_windows.npz", allow_pickle=True)

    attack_idx = np.where(train["y_attack_type"] > 0)[0]
    added = int(round(len(train["x"]) * RATIO))
    sampled_local = sample_synthetic_indices(train["y_attack_type"][attack_idx].astype(np.int64), added, seed + int(RATIO * 1000) + 101)
    sampled_idx = attack_idx[sampled_local]

    train_x_raw = np.concatenate([train["x"], train["x"][sampled_idx]], axis=0).astype(np.float32)
    train_y = np.concatenate([train["y_attack_type"], train["y_attack_type"][sampled_idx]], axis=0).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(train_y))
    train_x_raw = train_x_raw[order]
    train_y = train_y[order]

    mean, std = fit_standardizer(train["x"])
    train_x = standardize(train_x_raw, mean, std)
    val_x = standardize(val["x"], mean, std)
    test_x = standardize(test["x"], mean, std)
    variant_x = standardize(variant["x"], mean, std)
    cross_x = standardize(cross["x"], mean, std)

    model, history, device, best = train_model(train_x, train_y, val_x, val["y_attack_type"].astype(np.int64), seed, max_epochs=max_epochs)
    suffix = f"real_oversampling_ratio{RATIO:.2f}_seed{seed}".replace(".", "p")
    torch.save(model.state_dict(), MODELS / f"cnn1d_{suffix}.pt")
    np.savez(MODELS / f"standardizer_{suffix}.npz", mean=mean, std=std)
    write_csv(EXP / f"oversampling_training_history_seed{seed}.csv", history)

    rows = [
        evaluate(model, device, "val", val_x, val["y_attack_type"].astype(np.int64), seed, added),
        evaluate(model, device, "test", test_x, test["y_attack_type"].astype(np.int64), seed, added),
        evaluate(model, device, "variant_test", variant_x, variant["y_attack_type"].astype(np.int64), seed, added),
        evaluate(model, device, "otids_cross_binary", cross_x, cross["y_binary"].astype(np.int8), seed, added),
    ]
    for row in rows:
        row["best_epoch"] = best["epoch"]
        row["best_val_macro_f1_multiclass"] = best["f1"]
    write_csv(TABLES / f"oversampling_baseline_seed{seed}.csv", rows)
    return rows


def aggregate() -> pd.DataFrame:
    files = sorted(TABLES.glob("oversampling_baseline_seed*.csv"))
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    df.to_csv(EXP / "oversampling_baseline_results.csv", index=False)
    df.to_csv(TABLES / "oversampling_baseline_results.csv", index=False)
    metrics = ["accuracy", "macro_f1_binary", "attack_recall", "normal_recall", "fpr", "fnr", "auprc", "ece"]
    rows = []
    for split, group in df.groupby("split", sort=True):
        row = {"setting": "real_oversampling", "ratio": RATIO, "split": split, "seeds": ";".join(str(s) for s in sorted(group["seed"].unique()))}
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std(ddof=1)
        rows.append(row)
    over = pd.DataFrame(rows)

    synthetic = pd.read_csv(TABLES / "ratio_sweep_summary.csv")
    synthetic = synthetic[(synthetic["generator"] == "rule_based") & (np.isclose(synthetic["ratio"], RATIO))].copy()
    synthetic["setting"] = "rule_based_synthetic"
    synthetic = synthetic.rename(columns={"generator": "source"})
    cols = ["setting", "ratio", "split", "seeds"] + [f"{m}_{s}" for m in metrics for s in ["mean", "std"]]
    combined = pd.concat([over.reindex(columns=cols), synthetic.reindex(columns=cols)], ignore_index=True)
    combined.to_csv(TABLES / "oversampling_vs_synthetic.csv", index=False)
    return combined


def plot(combined: pd.DataFrame) -> None:
    labels = {"real_oversampling": "Real oversampling", "rule_based_synthetic": "Rule synthetic"}
    for metric, path, ylabel in [
        ("macro_f1_binary_mean", FIGURES / "oversampling_vs_synthetic_macro_f1.png", "Binary Macro-F1"),
        ("fpr_mean", FIGURES / "oversampling_vs_synthetic_fpr.png", "FPR"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
        for ax, split in zip(axes, ["test", "variant_test", "otids_cross_binary"]):
            sub = combined[combined["split"] == split].set_index("setting").reindex(["real_oversampling", "rule_based_synthetic"])
            x = np.arange(len(sub))
            ax.bar(x, sub[metric], yerr=sub[metric.replace("_mean", "_std")].fillna(0), capsize=3, color=["#6b7280", "#2563eb"])
            ax.set_title(split)
            ax.set_xticks(x)
            ax.set_xticklabels([labels[i] for i in sub.index], rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if args.all:
        for seed in SEEDS:
            run_one(seed, args.max_epochs)
    combined = aggregate()
    plot(combined)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
