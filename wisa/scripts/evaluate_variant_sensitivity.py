#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader

from train_real_only_baselines import CNN1D, WindowDataset, predict_cnn, standardize


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
EXP = ROOT / "experiments" / "04_augmented_ids"
BASELINE_MODELS = ROOT / "models" / "baseline"
RATIO_MODELS = ROOT / "models" / "ratio_sweep"
OVERSAMPLING_MODELS = ROOT / "models" / "oversampling"

SEEDS = [42, 7, 123]
SETTINGS = [
    {"setting": "real_only", "family": "real_only", "ratio": 0.0, "label": "Real only"},
    {"setting": "real_oversampling_0p50", "family": "oversampling", "ratio": 0.5, "label": "Real oversampling +50%"},
    {"setting": "rule_0p10", "family": "rule", "ratio": 0.1, "label": "Rule +10%"},
    {"setting": "rule_0p30", "family": "rule", "ratio": 0.3, "label": "Rule +30%"},
    {"setting": "rule_0p50", "family": "rule", "ratio": 0.5, "label": "Rule +50%"},
    {"setting": "rule_1p00", "family": "rule", "ratio": 1.0, "label": "Rule +100%"},
]
SEVERITY_ORDER = ["low", "medium", "high"]
ATTACK_ORDER = ["DoS", "Fuzzy", "Gear", "RPM"]


def model_paths(setting: dict, seed: int) -> tuple[Path, Path]:
    family = setting["family"]
    if family == "real_only":
        return BASELINE_MODELS / f"cnn1d_real_only_seed{seed}.pt", BASELINE_MODELS / "cnn_standardizer.npz"
    if family == "oversampling":
        suffix = f"real_oversampling_ratio0p50_seed{seed}"
        return OVERSAMPLING_MODELS / f"cnn1d_{suffix}.pt", OVERSAMPLING_MODELS / f"standardizer_{suffix}.npz"
    if family == "rule":
        suffix = f"rule_ratio{setting['ratio']:.2f}_seed{seed}".replace(".", "p")
        return RATIO_MODELS / f"cnn1d_{suffix}.pt", RATIO_MODELS / f"standardizer_{suffix}.npz"
    raise ValueError(family)


def predict(setting: dict, seed: int, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_path, std_path = model_paths(setting, seed)
    stdz = np.load(std_path)
    x_std = standardize(x, stdz["mean"], stdz["std"])
    model = CNN1D(in_channels=x.shape[-1], classes=5).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    loader = DataLoader(WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)), batch_size=1024, shuffle=False, num_workers=2)
    _, probs = predict_cnn(model, loader, device)
    return probs


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
        "auprc": float(average_precision_score(yb, 1.0 - probs[:, 0])),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(WINDOWS / "variant_sensitivity_windows.npz", allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y_attack_type"].astype(np.int64)
    scenario = data["scenario_id"].astype(str)
    attack_type = data["attack_type"].astype(str)
    severity = data["severity"].astype(str)
    rows = []
    scenario_rows = []
    for setting in SETTINGS:
        for seed in SEEDS:
            print(f"evaluating {setting['setting']} seed={seed}")
            probs = predict(setting, seed, x)
            pred = probs.argmax(axis=1)
            overall = {
                "setting": setting["setting"],
                "label": setting["label"],
                "family": setting["family"],
                "ratio": setting["ratio"],
                "seed": seed,
                "split": "variant_sensitivity",
            }
            overall.update(binary_metrics(y, probs))
            rows.append(overall)

            for sid in sorted(set(scenario)):
                mask = scenario == sid
                y_s = y[mask]
                pred_s = pred[mask]
                probs_s = probs[mask]
                is_attack = y_s > 0
                row = {
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
                scenario_rows.append(row)
    overall_df = pd.DataFrame(rows)
    scenario_df = pd.DataFrame(scenario_rows)
    return overall_df, scenario_df


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["seeds"] = ";".join(str(s) for s in sorted(group["seed"].unique()))
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals):
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmaps(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected_settings = ["real_only", "real_oversampling_0p50", "rule_0p10", "rule_0p30"]
    titles = {
        "real_only": "Real only",
        "real_oversampling_0p50": "Real oversampling +50%",
        "rule_0p10": "Rule +10%",
        "rule_0p30": "Rule +30%",
    }
    for metric, path, title in [
        ("attack_detection_recall_mean", FIGURES / "variant_sensitivity_attack_detection.png", "Attack detection recall"),
        ("exact_class_recall_mean", FIGURES / "variant_sensitivity_exact_class_recall.png", "Exact class recall"),
    ]:
        fig, axes = plt.subplots(1, len(selected_settings), figsize=(15, 4), sharey=True)
        for ax, setting in zip(axes, selected_settings):
            sub = summary[summary["setting"] == setting]
            mat = np.full((len(ATTACK_ORDER), len(SEVERITY_ORDER)), np.nan)
            for i, attack in enumerate(ATTACK_ORDER):
                for j, sev in enumerate(SEVERITY_ORDER):
                    row = sub[(sub["attack_type"] == attack) & (sub["severity"] == sev)]
                    if len(row):
                        mat[i, j] = row[metric].iloc[0]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
            ax.set_title(titles[setting])
            ax.set_xticks(range(len(SEVERITY_ORDER)), SEVERITY_ORDER)
            ax.set_yticks(range(len(ATTACK_ORDER)), ATTACK_ORDER)
            for i in range(len(ATTACK_ORDER)):
                for j in range(len(SEVERITY_ORDER)):
                    val = mat[i, j]
                    ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "NA", ha="center", va="center", color="white" if np.isfinite(val) and val < 0.55 else "black", fontsize=8)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        fig.suptitle(title, y=1.03)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_curves(summary: pd.DataFrame) -> None:
    selected = summary[summary["attack_type"].isin(["Gear", "RPM"])].copy()
    selected["severity_num"] = selected["severity"].map({"low": 0, "medium": 1, "high": 2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, attack in zip(axes, ["Gear", "RPM"]):
        sub = selected[selected["attack_type"] == attack]
        for setting, label in [
            ("real_only", "Real only"),
            ("real_oversampling_0p50", "Oversampling +50%"),
            ("rule_0p10", "Rule +10%"),
            ("rule_0p30", "Rule +30%"),
            ("rule_0p50", "Rule +50%"),
            ("rule_1p00", "Rule +100%"),
        ]:
            g = sub[sub["setting"] == setting].sort_values("severity_num")
            ax.errorbar(g["severity_num"], g["exact_class_recall_mean"], yerr=g["exact_class_recall_std"], marker="o", capsize=3, label=label)
        ax.set_title(attack)
        ax.set_xticks([0, 1, 2], ["low", "medium", "high"])
        ax.set_xlabel("Variant severity")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Exact class recall")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "variant_sensitivity_spoofing_curves.png", dpi=220)
    plt.close(fig)


def write_key_comparisons(overall_summary: pd.DataFrame, scenario_summary: pd.DataFrame) -> None:
    rows = []
    for setting in ["real_only", "real_oversampling_0p50", "rule_0p10", "rule_0p30", "rule_1p00"]:
        row = overall_summary[overall_summary["setting"] == setting].iloc[0]
        rows.append(
            {
                "comparison_scope": "overall_variant_sensitivity",
                "setting": setting,
                "attack_type": "all",
                "severity": "all",
                "macro_f1_binary_mean": row["macro_f1_binary_mean"],
                "attack_detection_recall_mean": row["attack_detection_recall_mean"],
                "exact_class_recall_mean": "",
                "interpretation": "overall binary detection across all sensitivity scenarios",
            }
        )
    selected = scenario_summary[
        (scenario_summary["attack_type"].isin(["Fuzzy", "Gear", "RPM"]))
        & (scenario_summary["severity"].isin(["low", "medium", "high"]))
        & (scenario_summary["setting"].isin(["real_only", "real_oversampling_0p50", "rule_0p10", "rule_0p30", "rule_1p00"]))
    ]
    for _, row in selected.iterrows():
        rows.append(
            {
                "comparison_scope": "scenario_exact_class_recall",
                "setting": row["setting"],
                "attack_type": row["attack_type"],
                "severity": row["severity"],
                "macro_f1_binary_mean": "",
                "attack_detection_recall_mean": row.get("attack_detection_recall_mean", ""),
                "exact_class_recall_mean": row.get("exact_class_recall_mean", ""),
                "interpretation": "exact multiclass robustness under parameterized variant severity",
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "variant_sensitivity_key_comparisons.csv", index=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    overall, scenario = evaluate()
    overall.to_csv(TABLES / "variant_sensitivity_metrics.csv", index=False)
    scenario.to_csv(TABLES / "variant_sensitivity_by_scenario_seed.csv", index=False)
    overall_summary = summarize(
        overall,
        ["setting", "label", "family", "ratio", "split"],
        ["accuracy", "macro_f1_binary", "attack_detection_recall", "normal_recall", "fpr", "auprc"],
    )
    scenario_summary = summarize(
        scenario,
        ["setting", "label", "family", "ratio", "scenario_id", "attack_type", "severity"],
        ["attack_detection_recall", "exact_class_recall", "normal_recall", "mean_attack_score"],
    )
    overall_summary.to_csv(TABLES / "variant_sensitivity_summary.csv", index=False)
    scenario_summary.to_csv(TABLES / "variant_sensitivity_scenario_summary.csv", index=False)
    overall.to_csv(EXP / "variant_sensitivity_metrics.csv", index=False)
    scenario_summary.to_csv(EXP / "variant_sensitivity_scenario_summary.csv", index=False)
    write_key_comparisons(overall_summary, scenario_summary)
    plot_heatmaps(scenario_summary[scenario_summary["attack_type"] != "Normal"])
    plot_curves(scenario_summary)
    print(overall_summary.to_string(index=False))


if __name__ == "__main__":
    main()
