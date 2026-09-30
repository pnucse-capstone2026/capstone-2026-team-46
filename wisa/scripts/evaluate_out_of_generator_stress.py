#!/usr/bin/env python3
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


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(WINDOWS / "out_of_generator_stress_windows.npz", allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y_attack_type"].astype(np.int64)
    scenario = data["scenario_id"].astype(str)
    attack_type = data["attack_type"].astype(str)
    id_condition = data["id_condition"].astype(str)
    rows = []
    scenario_rows = []

    for setting in SETTINGS:
        for seed in SEEDS:
            print(f"evaluating {setting['setting']} seed={seed}")
            probs = predict(setting, seed, x)
            pred = probs.argmax(axis=1)
            row = {
                "setting": setting["setting"],
                "label": setting["label"],
                "family": setting["family"],
                "ratio": setting["ratio"],
                "seed": seed,
                "split": "out_of_generator_stress",
            }
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


def write_key_table(scenario_summary: pd.DataFrame) -> pd.DataFrame:
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
            }
        )
    key = pd.DataFrame(rows)
    key.to_csv(TABLES / "out_of_generator_key_comparisons.csv", index=False)
    return key


def plot_key(key: pd.DataFrame) -> None:
    selected = key[key["setting"].isin(["real_only", "real_oversampling_0p50", "rule_0p10", "rule_0p30", "rule_1p00"])].copy()
    labels = ["Real only", "Oversampling", "Rule +10%", "Rule +30%", "Rule +100%"]
    setting_order = ["real_only", "real_oversampling_0p50", "rule_0p10", "rule_0p30", "rule_1p00"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    width = 0.36
    x = np.arange(len(setting_order))
    for ax, attack in zip(axes, ["Gear", "RPM"]):
        sub = selected[selected["attack_type"] == attack].set_index("setting")
        canonical = [sub.loc[s, "canonical_exact_recall_mean"] for s in setting_order]
        shifted = [sub.loc[s, "shifted_exact_recall_mean"] for s in setting_order]
        canonical_err = [sub.loc[s, "canonical_exact_recall_std"] for s in setting_order]
        shifted_err = [sub.loc[s, "shifted_exact_recall_std"] for s in setting_order]
        ax.bar(x - width / 2, canonical, width, yerr=canonical_err, capsize=2, label="Canonical ID")
        ax.bar(x + width / 2, shifted, width, yerr=shifted_err, capsize=2, label="Shifted ID")
        ax.set_title(attack)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Exact class recall")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Out-of-generator payload-position stress test")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "out_of_generator_exact_recall.png", dpi=220)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    overall, scenario = evaluate()
    overall.to_csv(TABLES / "out_of_generator_metrics.csv", index=False)
    scenario.to_csv(TABLES / "out_of_generator_by_scenario_seed.csv", index=False)
    overall_summary = summarize(
        overall,
        ["setting", "label", "family", "ratio", "split"],
        ["accuracy", "macro_f1_binary", "attack_detection_recall", "normal_recall", "fpr", "auprc"],
    )
    scenario_summary = summarize(
        scenario,
        ["setting", "label", "family", "ratio", "scenario_id", "attack_type", "id_condition"],
        ["attack_detection_recall", "exact_class_recall", "normal_recall", "mean_attack_score"],
    )
    overall_summary.to_csv(TABLES / "out_of_generator_summary.csv", index=False)
    scenario_summary.to_csv(TABLES / "out_of_generator_scenario_summary.csv", index=False)
    overall.to_csv(EXP / "out_of_generator_metrics.csv", index=False)
    scenario_summary.to_csv(EXP / "out_of_generator_scenario_summary.csv", index=False)
    key = write_key_table(scenario_summary)
    plot_key(key)
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
