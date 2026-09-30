#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
EXP = ROOT / "experiments" / "02_baseline"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "models" / "baseline"

CLASS_NAMES = {
    0: "Normal",
    1: "DoS",
    2: "Fuzzy",
    3: "Gear",
    4: "RPM",
    5: "Impersonation",
}

DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(split):
    return np.load(WINDOWS / f"{split}_windows.npz", allow_pickle=True)


def rf_features(x):
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    minv = x.min(axis=1)
    maxv = x.max(axis=1)
    delta = x[:, -1, :] - x[:, 0, :]
    return np.concatenate([mean, std, minv, maxv, delta], axis=1).astype(np.float32)


def fit_standardizer(x):
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def expected_calibration_error(y_true_binary, attack_scores, n_bins=15):
    confidence = np.maximum(attack_scores, 1.0 - attack_scores)
    pred = (attack_scores >= 0.5).astype(np.int8)
    correct = (pred == y_true_binary).astype(np.float32)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(model_name, split, y_attack_true, attack_scores, pred_attack_label, seed):
    y_binary_true = (y_attack_true > 0).astype(np.int8)
    y_binary_pred = (pred_attack_label > 0).astype(np.int8)
    cm = confusion_matrix(y_binary_true, y_binary_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    row = {
        "model": model_name,
        "seed": seed,
        "split": split,
        "accuracy": float((y_binary_true == y_binary_pred).mean()),
        "macro_f1_binary": float(f1_score(y_binary_true, y_binary_pred, average="macro", zero_division=0)),
        "attack_recall": float(recall_score(y_binary_true, y_binary_pred, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(y_binary_true, y_binary_pred, pos_label=0, zero_division=0)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auprc": float(average_precision_score(y_binary_true, attack_scores)) if len(np.unique(y_binary_true)) > 1 else "",
        "ece": expected_calibration_error(y_binary_true, attack_scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    for cls in [1, 2, 3, 4]:
        mask = y_attack_true == cls
        if np.any(mask):
            row[f"recall_{CLASS_NAMES[cls]}"] = float((pred_attack_label[mask] == cls).mean())
        else:
            row[f"recall_{CLASS_NAMES[cls]}"] = ""
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


def save_pr_curve(path, curves):
    plt.figure(figsize=(7, 5))
    for label, y_true, scores in curves:
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{label} AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Real-only IDS Precision-Recall")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion_matrix(path, title, y_true_binary, y_pred_binary):
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["Normal", "Attack"])
    plt.yticks([0, 1], ["Normal", "Attack"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CNN1D(nn.Module):
    def __init__(self, in_channels=11, classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, classes))

    def forward(self, x):
        return self.head(self.net(x))


def predict_cnn(model, loader, device):
    model.eval()
    probs = []
    y_true = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(probs)


def train_cnn(train_x, train_y, val_x, val_y, max_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = WindowDataset(train_x, train_y)
    val_ds = WindowDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = CNN1D(in_channels=train_x.shape[-1], classes=5).to(device)
    classes = np.arange(5)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = -1.0
    best_state = None
    history = []
    patience = 4
    stale = 0
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
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_ds), "val_macro_f1_multiclass": val_f1})
        print(f"cnn epoch={epoch} loss={history[-1]['train_loss']:.5f} val_macro_f1={val_f1:.5f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rf-trees", type=int, default=160)
    parser.add_argument("--max-epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    EXP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train = load_npz("train")
    val = load_npz("val")
    test = load_npz("test")
    cross = load_npz("otids_cross")

    train_y = train["y_attack_type"].astype(np.int64)
    val_y = val["y_attack_type"].astype(np.int64)
    test_y = test["y_attack_type"].astype(np.int64)
    cross_y_attack = cross["y_attack_type"].astype(np.int64)
    cross_y_binary = cross["y_binary"].astype(np.int8)

    rows = []
    pr_curves = []

    print("training random forest")
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    train_rf_x = rf_features(train["x"])
    rf.fit(train_rf_x, train_y)
    joblib.dump(rf, MODELS / f"random_forest_real_only_seed{args.seed}.joblib")

    for split_name, data, y_attack in [("val", val, val_y), ("test", test, test_y)]:
        probs = rf.predict_proba(rf_features(data["x"]))
        labels = rf.classes_
        full_probs = np.zeros((len(probs), 5), dtype=np.float32)
        for idx, label in enumerate(labels):
            if label < 5:
                full_probs[:, label] = probs[:, idx]
        pred = full_probs.argmax(axis=1)
        attack_scores = 1.0 - full_probs[:, 0]
        rows.append(compute_metrics("RandomForest", split_name, y_attack, attack_scores, pred, args.seed))
        pr_curves.append((f"RF {split_name}", (y_attack > 0).astype(np.int8), attack_scores))
        if split_name == "test":
            save_confusion_matrix(
                FIGURES / f"baseline_rf_confusion_matrix_seed{args.seed}.png",
                f"RF Real Test seed {args.seed}",
                (y_attack > 0),
                pred > 0,
            )

    cross_probs = rf.predict_proba(rf_features(cross["x"]))
    labels = rf.classes_
    full_cross_probs = np.zeros((len(cross_probs), 5), dtype=np.float32)
    for idx, label in enumerate(labels):
        if label < 5:
            full_cross_probs[:, label] = cross_probs[:, idx]
    cross_pred = full_cross_probs.argmax(axis=1)
    cross_scores = 1.0 - full_cross_probs[:, 0]
    rows.append(compute_metrics("RandomForest", "otids_cross_binary", cross_y_binary, cross_scores, cross_pred, args.seed))
    pr_curves.append(("RF OTIDS", cross_y_binary, cross_scores))

    print("training cnn")
    mean, std = fit_standardizer(train["x"])
    np.savez(MODELS / "cnn_standardizer.npz", mean=mean, std=std)
    train_x = standardize(train["x"], mean, std)
    val_x = standardize(val["x"], mean, std)
    test_x = standardize(test["x"], mean, std)
    cross_x = standardize(cross["x"], mean, std)
    cnn, history, device = train_cnn(train_x, train_y, val_x, val_y, max_epochs=args.max_epochs)
    torch.save(cnn.state_dict(), MODELS / f"cnn1d_real_only_seed{args.seed}.pt")
    write_csv(EXP / f"cnn_training_history_seed{args.seed}.csv", history)

    for split_name, x, y_attack in [("val", val_x, val_y), ("test", test_x, test_y)]:
        loader = DataLoader(WindowDataset(x, y_attack), batch_size=1024, shuffle=False, num_workers=2)
        true, probs = predict_cnn(cnn, loader, device)
        pred = probs.argmax(axis=1)
        attack_scores = 1.0 - probs[:, 0]
        rows.append(compute_metrics("CNN1D", split_name, y_attack, attack_scores, pred, args.seed))
        pr_curves.append((f"CNN {split_name}", (y_attack > 0).astype(np.int8), attack_scores))
        if split_name == "test":
            save_confusion_matrix(
                FIGURES / f"baseline_cnn_confusion_matrix_seed{args.seed}.png",
                f"CNN Real Test seed {args.seed}",
                (y_attack > 0),
                pred > 0,
            )

    cross_loader = DataLoader(WindowDataset(cross_x, cross_y_binary.astype(np.int64)), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(cnn, cross_loader, device)
    pred = probs.argmax(axis=1)
    attack_scores = 1.0 - probs[:, 0]
    rows.append(compute_metrics("CNN1D", "otids_cross_binary", cross_y_binary, attack_scores, pred, args.seed))
    pr_curves.append(("CNN OTIDS", cross_y_binary, attack_scores))

    write_csv(EXP / f"rf_real_only_results_seed{args.seed}.csv", [row for row in rows if row["model"] == "RandomForest"])
    write_csv(EXP / f"cnn_real_only_results_seed{args.seed}.csv", [row for row in rows if row["model"] == "CNN1D"])
    write_csv(TABLES / f"baseline_metrics_seed{args.seed}.csv", rows)
    save_pr_curve(FIGURES / f"baseline_pr_curve_seed{args.seed}.png", pr_curves)
    with (EXP / f"baseline_run_config_seed{args.seed}.json").open("w") as f:
        json.dump(
            {
                "seed": args.seed,
                "rf_features": "mean/std/min/max/last-minus-first over 11 frame features",
                "rf_trees": args.rf_trees,
                "cnn_input": "standardized 128x11 window tensor",
                "cnn_device": str(device),
                "cnn_epochs": len(history),
            },
            f,
            indent=2,
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
