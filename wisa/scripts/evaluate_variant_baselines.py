#!/usr/bin/env python3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_real_only_baselines import (
    CNN1D,
    WindowDataset,
    compute_metrics,
    predict_cnn,
    rf_features,
    standardize,
)


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
MODELS = ROOT / "models" / "baseline"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
EXP = ROOT / "experiments" / "02_baseline"
SEEDS = [42, 7, 123]


def evaluate_rf(seed, x, y_attack):
    rf = joblib.load(MODELS / f"random_forest_real_only_seed{seed}.joblib")
    probs = rf.predict_proba(rf_features(x))
    full_probs = np.zeros((len(probs), 5), dtype=np.float32)
    for idx, label in enumerate(rf.classes_):
        if label < 5:
            full_probs[:, label] = probs[:, idx]
    pred = full_probs.argmax(axis=1)
    attack_scores = 1.0 - full_probs[:, 0]
    return compute_metrics("RandomForest", "variant_test", y_attack, attack_scores, pred, seed)


def evaluate_cnn(seed, x, y_attack):
    stdz = np.load(MODELS / "cnn_standardizer.npz")
    x_std = standardize(x, stdz["mean"], stdz["std"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(in_channels=x.shape[-1], classes=5).to(device)
    state = torch.load(MODELS / f"cnn1d_real_only_seed{seed}.pt", map_location=device)
    model.load_state_dict(state)
    loader = DataLoader(WindowDataset(x_std, y_attack), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(model, loader, device)
    pred = probs.argmax(axis=1)
    attack_scores = 1.0 - probs[:, 0]
    return compute_metrics("CNN1D", "variant_test", y_attack, attack_scores, pred, seed)


def aggregate(rows):
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "variant_baseline_metrics.csv", index=False)
    metric_cols = [
        "accuracy",
        "macro_f1_binary",
        "attack_recall",
        "normal_recall",
        "fpr",
        "fnr",
        "auprc",
        "ece",
        "recall_DoS",
        "recall_Fuzzy",
        "recall_Gear",
        "recall_RPM",
    ]
    out = []
    for model, group in df.groupby("model"):
        row = {"model": model, "split": "variant_test", "seeds": ";".join(str(x) for x in sorted(group["seed"].unique()))}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(values):
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        out.append(row)
    summary = pd.DataFrame(out)
    summary.to_csv(TABLES / "variant_baseline_metrics_mean_std.csv", index=False)
    return df, summary


def plot(summary):
    labels = summary["model"].tolist()
    f1 = summary["macro_f1_binary_mean"].tolist()
    f1_err = summary["macro_f1_binary_std"].fillna(0).tolist()
    recall = summary["attack_recall_mean"].tolist()
    fpr = summary["fpr_mean"].tolist()
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width, f1, width, yerr=f1_err, capsize=4, label="Macro-F1")
    plt.bar(x, recall, width, label="Attack Recall")
    plt.bar(x + width, fpr, width, label="FPR")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.title("Real-only baselines on variant test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "variant_baseline_performance.png", dpi=180)
    plt.close()


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    data = np.load(WINDOWS / "variant_test_windows.npz", allow_pickle=True)
    x = data["x"].astype(np.float32)
    y_attack = data["y_attack_type"].astype(np.int64)
    rows = []
    for seed in SEEDS:
        print(f"evaluating seed {seed} RF")
        rows.append(evaluate_rf(seed, x, y_attack))
        print(f"evaluating seed {seed} CNN")
        rows.append(evaluate_cnn(seed, x, y_attack))
    df, summary = aggregate(rows)
    plot(summary)
    df[df["model"] == "RandomForest"].to_csv(EXP / "rf_variant_test_results.csv", index=False)
    df[df["model"] == "CNN1D"].to_csv(EXP / "cnn_variant_test_results.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
