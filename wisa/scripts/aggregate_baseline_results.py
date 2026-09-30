#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
EXP = ROOT / "experiments" / "02_baseline"
FIGURES = ROOT / "results" / "figures"

METRIC_COLS = [
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


def main():
    files = sorted(TABLES.glob("baseline_metrics_seed*.csv"))
    if not files:
        raise SystemExit("No seed baseline metric files found.")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    df.to_csv(TABLES / "baseline_metrics.csv", index=False)
    rf = df[df["model"] == "RandomForest"]
    cnn = df[df["model"] == "CNN1D"]
    rf.to_csv(EXP / "rf_real_only_results.csv", index=False)
    cnn.to_csv(EXP / "cnn_real_only_results.csv", index=False)

    rows = []
    for (model, split), group in df.groupby(["model", "split"], sort=True):
        row = {"model": model, "split": split, "seeds": ";".join(str(x) for x in sorted(group["seed"].unique()))}
        for col in METRIC_COLS:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(values) == 0:
                continue
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "baseline_metrics_mean_std.csv", index=False)

    plot_cross_generalization(summary)
    print(summary.to_string(index=False))


def plot_cross_generalization(summary):
    FIGURES.mkdir(parents=True, exist_ok=True)
    metric = "macro_f1_binary_mean"
    rows = []
    for _, row in summary.iterrows():
        if row["split"] in ["test", "otids_cross_binary"]:
            rows.append(row)
    plot_df = pd.DataFrame(rows)
    labels = [f"{r.model}\n{r.split.replace('_binary', '')}" for r in plot_df.itertuples()]
    values = plot_df[metric].tolist()
    errors = plot_df.get("macro_f1_binary_std", pd.Series([0] * len(plot_df))).fillna(0).tolist()
    plt.figure(figsize=(8, 4.5))
    plt.bar(range(len(values)), values, yerr=errors, capsize=4, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"][: len(values)])
    plt.xticks(range(len(values)), labels)
    plt.ylabel("Binary Macro-F1")
    plt.ylim(0, 1.05)
    plt.title("Real-only baseline generalization")
    plt.tight_layout()
    plt.savefig(FIGURES / "baseline_generalization_gap.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
