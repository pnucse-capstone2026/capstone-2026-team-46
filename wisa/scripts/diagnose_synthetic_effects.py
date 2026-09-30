#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from train_real_only_baselines import CNN1D, WindowDataset, predict_cnn, standardize
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
EXP = ROOT / "experiments" / "05_diagnosis"
MODELS = ROOT / "models" / "ratio_sweep"

RATIOS = [0.10, 0.30, 0.50, 1.00]
SEEDS = [42, 7, 123]
CONFIDENCE_THRESHOLD = 0.90


def suffix_for(ratio: float, seed: int) -> str:
    return f"rule_ratio{ratio:.2f}_seed{seed}".replace(".", "p")


def synthetic_boundary_metrics() -> pd.DataFrame:
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    syn = np.load(SYNTHETIC / "rule_based_windows.npz", allow_pickle=True)
    syn_x = syn["x"].astype(np.float32)
    syn_y = syn["y_attack_type"].astype(np.int64)
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        for ratio in RATIOS:
            n_syn = int(round(len(train["x"]) * ratio))
            idx = sample_synthetic_indices(syn_y, n_syn, seed + int(ratio * 1000))
            suffix = suffix_for(ratio, seed)
            stdz = np.load(MODELS / f"standardizer_{suffix}.npz")
            model = CNN1D(in_channels=syn_x.shape[-1], classes=5).to(device)
            state = torch.load(MODELS / f"cnn1d_{suffix}.pt", map_location=device)
            model.load_state_dict(state)
            x_eval = standardize(syn_x[idx], stdz["mean"], stdz["std"])
            y_eval = syn_y[idx]
            loader = DataLoader(WindowDataset(x_eval, y_eval), batch_size=1024, shuffle=False, num_workers=2)
            _, probs = predict_cnn(model, loader, device)
            pred = probs.argmax(axis=1)
            max_prob = probs.max(axis=1)
            true_prob = probs[np.arange(len(y_eval)), y_eval]
            rows.append(
                {
                    "generator": "rule_based",
                    "ratio": ratio,
                    "seed": seed,
                    "synthetic_windows_used": n_syn,
                    "low_confidence_threshold": CONFIDENCE_THRESHOLD,
                    "low_confidence_synthetic_ratio": float((max_prob < CONFIDENCE_THRESHOLD).mean()),
                    "misleading_synthetic_ratio": float((pred != y_eval).mean()),
                    "mean_true_class_probability": float(true_prob.mean()),
                    "mean_max_probability": float(max_prob.mean()),
                }
            )
    return pd.DataFrame(rows)


def burst_mismatch_metrics() -> tuple[pd.DataFrame, float]:
    syn_stats = pd.read_csv(TABLES / "rule_based_synthetic_statistics.csv")
    variant_stats = pd.read_csv(TABLES / "variant_test_statistics.csv")
    variant_stats = variant_stats[variant_stats["variant_type"] != "Normal"]
    rows = []
    for _, syn_row in syn_stats.iterrows():
        attack_type = syn_row["synthetic_type"]
        var_row = variant_stats[variant_stats["variant_type"] == attack_type].iloc[0]
        rows.append(
            {
                "generator": "rule_based",
                "attack_type": attack_type,
                "synthetic_injection_count_mean": float(syn_row["injection_count_mean"]),
                "variant_injection_count_mean": float(var_row["injection_count_mean"]),
                "burst_count_abs_error_vs_variant": abs(float(syn_row["injection_count_mean"]) - float(var_row["injection_count_mean"])),
            }
        )
    df = pd.DataFrame(rows)
    return df, float(df["burst_count_abs_error_vs_variant"].mean())


def diagnostic_metrics(boundary: pd.DataFrame, burst_mae: float) -> pd.DataFrame:
    quality = pd.read_csv(TABLES / "synthetic_quality_metrics.csv").iloc[0].to_dict()
    grouped = (
        boundary.groupby(["generator", "ratio"], as_index=False)
        .agg(
            synthetic_windows_used=("synthetic_windows_used", "mean"),
            low_confidence_synthetic_ratio=("low_confidence_synthetic_ratio", "mean"),
            low_confidence_synthetic_ratio_std=("low_confidence_synthetic_ratio", "std"),
            misleading_synthetic_ratio=("misleading_synthetic_ratio", "mean"),
            misleading_synthetic_ratio_std=("misleading_synthetic_ratio", "std"),
            mean_true_class_probability=("mean_true_class_probability", "mean"),
            mean_max_probability=("mean_max_probability", "mean"),
        )
    )
    for key, value in quality.items():
        if key not in {"generator"}:
            grouped[key] = value
    grouped["burst_count_mae_vs_variant"] = burst_mae
    return grouped


def degradation_metrics(diag: pd.DataFrame) -> pd.DataFrame:
    summary = pd.read_csv(TABLES / "test_distribution_summary.csv")
    baseline = summary[summary["generator"] == "real_only"][
        [
            "split",
            "macro_f1_binary_mean",
            "attack_recall_mean",
            "fpr_mean",
            "ece_mean",
        ]
    ].rename(
        columns={
            "macro_f1_binary_mean": "baseline_macro_f1",
            "attack_recall_mean": "baseline_attack_recall",
            "fpr_mean": "baseline_fpr",
            "ece_mean": "baseline_ece",
        }
    )
    rule = summary[summary["generator"] == "rule_based"].merge(baseline, on="split", how="left")
    rule["macro_f1_degradation"] = rule["baseline_macro_f1"] - rule["macro_f1_binary_mean"]
    rule["attack_recall_degradation"] = rule["baseline_attack_recall"] - rule["attack_recall_mean"]
    rule["fpr_increase"] = rule["fpr_mean"] - rule["baseline_fpr"]
    rule["ece_change"] = rule["ece_mean"] - rule["baseline_ece"]

    real_by_ratio = rule[rule["split"] == "test"][["ratio", "macro_f1_binary_mean"]].rename(
        columns={"macro_f1_binary_mean": "same_setting_real_test_macro_f1"}
    )
    rule = rule.merge(real_by_ratio, on="ratio", how="left")
    rule["generalization_gap_from_real_test"] = rule["same_setting_real_test_macro_f1"] - rule["macro_f1_binary_mean"]
    merged = rule.merge(diag, on=["generator", "ratio"], how="left", suffixes=("", "_diagnostic"))
    return merged


def maybe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, str]:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return np.nan, np.nan, "insufficient_n"
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, "constant_metric"
    return float(pearsonr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic), float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic), "exploratory"


def correlations(deg: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("ratio", "macro_f1_degradation"),
        ("ratio", "fpr_increase"),
        ("ratio", "ece_change"),
        ("low_confidence_synthetic_ratio", "macro_f1_degradation"),
        ("low_confidence_synthetic_ratio", "fpr_increase"),
        ("misleading_synthetic_ratio", "macro_f1_degradation"),
        ("misleading_synthetic_ratio", "fpr_increase"),
        ("periodicity_mae", "macro_f1_degradation"),
        ("real_vs_synthetic_auc", "generalization_gap_from_real_test"),
        ("burst_count_mae_vs_variant", "macro_f1_degradation"),
    ]
    rows = []
    for split in ["test", "variant_test", "otids_cross_binary", "all"]:
        sub = deg if split == "all" else deg[deg["split"] == split]
        for x_col, y_col in pairs:
            pearson, spearman, note = maybe_corr(sub[x_col], sub[y_col])
            rows.append(
                {
                    "split": split,
                    "x_metric": x_col,
                    "y_metric": y_col,
                    "n": int(pd.concat([sub[x_col], sub[y_col]], axis=1).dropna().shape[0]),
                    "pearson_r": pearson,
                    "spearman_r": spearman,
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def plot_periodicity_vs_degradation(deg: pd.DataFrame) -> None:
    plt.figure(figsize=(7.2, 4.8))
    markers = {"test": "o", "variant_test": "s", "otids_cross_binary": "^"}
    for split, group in deg.groupby("split"):
        jitter = (group["ratio"] - group["ratio"].mean()) * 0.0002
        plt.scatter(group["periodicity_mae"] + jitter, group["macro_f1_degradation"], s=70, marker=markers.get(split, "o"), label=split)
        for _, row in group.iterrows():
            plt.annotate(f"{row['ratio']:.1f}", (row["periodicity_mae"] + (row["ratio"] - group["ratio"].mean()) * 0.0002, row["macro_f1_degradation"]), fontsize=8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Synthetic periodicity MAE")
    plt.ylabel("Macro-F1 degradation vs real-only")
    plt.title("Periodicity error vs performance degradation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "periodicity_error_vs_f1_degradation.png", dpi=200)
    plt.close()


def plot_artifact_auc_vs_gap(deg: pd.DataFrame) -> None:
    sub = deg[deg["split"].isin(["variant_test", "otids_cross_binary"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for split, group in sub.groupby("split"):
        jitter = (group["ratio"] - group["ratio"].mean()) * 0.00001
        ax.scatter(group["real_vs_synthetic_auc"] + jitter, group["generalization_gap_from_real_test"], s=70, label=split)
        for _, row in group.iterrows():
            ax.annotate(f"{row['ratio']:.1f}", (row["real_vs_synthetic_auc"] + (row["ratio"] - group["ratio"].mean()) * 0.00001, row["generalization_gap_from_real_test"]), fontsize=8)
    ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    ax.set_xlim(0.999975, 1.000005)
    ax.set_xlabel("Real-vs-synthetic discriminator AUC")
    ax.set_ylabel("Macro-F1 gap from same-setting real test")
    ax.set_title("Artifact detectability vs generalization gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "artifact_auc_vs_generalization_gap.png", dpi=200)
    plt.close(fig)


def plot_ratio_vs_fpr_increase(deg: pd.DataFrame) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for split, group in deg.groupby("split"):
        group = group.sort_values("ratio")
        plt.plot(group["ratio"], group["fpr_increase"], marker="o", label=split)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Synthetic augmentation ratio")
    plt.ylabel("FPR increase vs real-only")
    plt.title("Synthetic ratio vs FPR increase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "boundary_ratio_vs_fpr_increase.png", dpi=200)
    plt.close()


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    boundary = synthetic_boundary_metrics()
    boundary.to_csv(EXP / "synthetic_training_sample_diagnostics.csv", index=False)
    burst_by_type, burst_mae = burst_mismatch_metrics()
    burst_by_type.to_csv(EXP / "burst_mismatch_by_type.csv", index=False)

    diag = diagnostic_metrics(boundary, burst_mae)
    diag.to_csv(EXP / "diagnostic_metrics.csv", index=False)
    deg = degradation_metrics(diag)
    deg.to_csv(EXP / "degradation_metrics.csv", index=False)
    corr = correlations(deg)
    corr.to_csv(TABLES / "diagnosis_correlation.csv", index=False)

    plot_periodicity_vs_degradation(deg)
    plot_artifact_auc_vs_gap(deg)
    plot_ratio_vs_fpr_increase(deg)

    print(f"wrote {EXP / 'diagnostic_metrics.csv'}")
    print(f"wrote {EXP / 'degradation_metrics.csv'}")
    print(f"wrote {TABLES / 'diagnosis_correlation.csv'}")
    print(f"wrote {FIGURES / 'periodicity_error_vs_f1_degradation.png'}")
    print(f"wrote {FIGURES / 'artifact_auc_vs_generalization_gap.png'}")
    print(f"wrote {FIGURES / 'boundary_ratio_vs_fpr_increase.png'}")


if __name__ == "__main__":
    main()
