#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from train_real_only_baselines import CNN1D, WindowDataset, predict_cnn, standardize


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
EXP = ROOT / "experiments" / "04_augmented_ids"
BASELINE_MODELS = ROOT / "models" / "baseline"
RATIO_MODELS = ROOT / "models" / "ratio_sweep"

ATTACK_TYPES = ["DoS", "Fuzzy", "Gear", "RPM"]
SPLITS = ["test", "variant_test"]
PLOT_SETTINGS = [
    ("real_only", 0.0, "Real only"),
    ("rule_based", 0.1, "+10% rule"),
    ("rule_based", 0.3, "+30% rule"),
    ("rule_based", 0.5, "+50% rule"),
    ("rule_based", 1.0, "+100% rule"),
]
CONFUSION_SETTINGS = [
    ("real_only", 0.0, "Real only"),
    ("rule_based", 0.1, "+10% rule"),
    ("rule_based", 0.3, "+30% rule"),
]
CLASS_LABELS = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]


def metric_rows_to_long() -> pd.DataFrame:
    baseline = pd.read_csv(TABLES / "baseline_metrics.csv")
    variant = pd.read_csv(TABLES / "variant_baseline_metrics.csv")
    ratio = pd.read_csv(TABLES / "ratio_sweep_results.csv")

    base = baseline[(baseline["model"] == "CNN1D") & (baseline["split"] == "test")].copy()
    base = pd.concat([base, variant[(variant["model"] == "CNN1D") & (variant["split"] == "variant_test")]], ignore_index=True)
    base["generator"] = "real_only"
    base["ratio"] = 0.0
    base["setting"] = "real_only_0p00"

    rule = ratio[(ratio["split"].isin(SPLITS)) & (ratio["generator"] == "rule_based")].copy()
    rule["setting"] = rule["generator"] + "_" + rule["ratio"].map(lambda x: f"{x:.2f}".replace(".", "p"))

    rows = []
    for df in [base, rule]:
        for _, row in df.iterrows():
            for attack_type in ATTACK_TYPES:
                value = pd.to_numeric(row.get(f"recall_{attack_type}"), errors="coerce")
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "generator": row["generator"],
                        "ratio": float(row["ratio"]),
                        "setting": row["setting"],
                        "split": row["split"],
                        "seed": int(row["seed"]),
                        "attack_type": attack_type,
                        "recall": float(value),
                    }
                )
    return pd.DataFrame(rows)


def summarize_recall(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby(["generator", "ratio", "setting", "split", "attack_type"], as_index=False)
        .agg(
            seeds=("seed", lambda s: ";".join(str(x) for x in sorted(set(s)))),
            recall_mean=("recall", "mean"),
            recall_std=("recall", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
        )
        .sort_values(["split", "attack_type", "ratio"])
    )
    baseline = summary[summary["generator"] == "real_only"][
        ["split", "attack_type", "recall_mean"]
    ].rename(columns={"recall_mean": "real_only_recall_mean"})
    summary = summary.merge(baseline, on=["split", "attack_type"], how="left")
    summary["delta_vs_real_only"] = summary["recall_mean"] - summary["real_only_recall_mean"]
    return summary


def plot_recall_delta(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    colors = {
        "DoS": "#2563eb",
        "Fuzzy": "#16a34a",
        "Gear": "#dc2626",
        "RPM": "#9333ea",
    }

    for ax, split in zip(axes, SPLITS):
        split_df = summary[summary["split"] == split].copy()
        x = np.arange(len(PLOT_SETTINGS))
        width = 0.18
        for idx, attack_type in enumerate(ATTACK_TYPES):
            values = []
            errors = []
            for generator, ratio, _ in PLOT_SETTINGS:
                row = split_df[(split_df["generator"] == generator) & (np.isclose(split_df["ratio"], ratio)) & (split_df["attack_type"] == attack_type)]
                values.append(float(row["recall_mean"].iloc[0]) if len(row) else np.nan)
                errors.append(float(row["recall_std"].iloc[0]) if len(row) else 0.0)
            ax.bar(
                x + (idx - 1.5) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=attack_type,
                color=colors[attack_type],
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Class recall")
        ax.set_title("Real Test" if split == "test" else "Variant Test")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, _, label in PLOT_SETTINGS], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "attack_type_recall_delta.png", dpi=200)
    plt.close(fig)


def load_model_predictions(generator: str, ratio: float, seed: int, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(in_channels=x.shape[-1], classes=5).to(device)
    if generator == "real_only":
        stdz = np.load(BASELINE_MODELS / "cnn_standardizer.npz")
        state_path = BASELINE_MODELS / f"cnn1d_real_only_seed{seed}.pt"
    else:
        suffix = f"rule_ratio{ratio:.2f}_seed{seed}".replace(".", "p")
        stdz = np.load(RATIO_MODELS / f"standardizer_{suffix}.npz")
        state_path = RATIO_MODELS / f"cnn1d_{suffix}.pt"
    x_std = standardize(x, stdz["mean"], stdz["std"])
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    loader = DataLoader(WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(model, loader, device)
    return probs.argmax(axis=1)


def plot_confusion_matrices(seed: int = 42) -> None:
    data = np.load(WINDOWS / "variant_test_windows.npz", allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y_attack_type"].astype(np.int64)

    fig, axes = plt.subplots(1, len(CONFUSION_SETTINGS), figsize=(14, 4.4), sharey=True)
    for ax, (generator, ratio, title) in zip(axes, CONFUSION_SETTINGS):
        pred = load_model_predictions(generator, ratio, seed, x)
        cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3, 4])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(5), CLASS_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(5), CLASS_LABELS)
        ax.set_xlabel("Predicted")
        for i in range(5):
            for j in range(5):
                text = f"{cm_norm[i, j]:.2f}" if cm[i, j] else "0"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("True")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Row-normalized count")
    fig.suptitle(f"Variant Test Multiclass Confusion Matrix, seed {seed}", y=1.02)
    fig.savefig(FIGURES / "class_confusion_matrix_by_generator.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    long_df = metric_rows_to_long()
    long_path = TABLES / "attack_type_recall_by_seed.csv"
    long_df.to_csv(long_path, index=False)
    summary = summarize_recall(long_df)
    summary_path = TABLES / "attack_type_recall_delta.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(EXP / "attack_type_recall_delta.csv", index=False)
    plot_recall_delta(summary)
    plot_confusion_matrices(seed=42)
    print(f"wrote {long_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {FIGURES / 'attack_type_recall_delta.png'}")
    print(f"wrote {FIGURES / 'class_confusion_matrix_by_generator.png'}")


if __name__ == "__main__":
    main()
