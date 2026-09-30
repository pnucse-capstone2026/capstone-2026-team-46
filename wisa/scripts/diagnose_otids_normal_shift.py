#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_real_only_baselines import CNN1D, WindowDataset, predict_cnn, standardize


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
EXP = ROOT / "experiments" / "07_ver3_diagnostics"
BASELINE_MODELS = ROOT / "models" / "baseline"
RATIO_MODELS = ROOT / "models" / "ratio_sweep"

SEEDS = [7, 42, 123]
RNG_SEED = 20260610
MAX_NORMAL_WINDOWS = 30_000
FEATURES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]

SETTINGS = [
    {"setting": "real_only", "family": "real_only", "ratio": 0.0, "label": "Real only"},
    {"setting": "rule_0p30", "family": "rule", "ratio": 0.3, "label": "Rule +30%"},
    {"setting": "rule_1p00", "family": "rule", "ratio": 1.0, "label": "Rule +100%"},
]

SCORE_SPLITS = [
    ("validation_normal", "val"),
    ("car_hacking_test_normal", "test"),
    ("otids_normal", "otids_cross"),
]


def model_paths(setting: dict, seed: int) -> tuple[Path, Path]:
    if setting["family"] == "real_only":
        return BASELINE_MODELS / f"cnn1d_real_only_seed{seed}.pt", BASELINE_MODELS / "cnn_standardizer.npz"
    suffix = f"rule_ratio{setting['ratio']:.2f}_seed{seed}".replace(".", "p")
    return RATIO_MODELS / f"cnn1d_{suffix}.pt", RATIO_MODELS / f"standardizer_{suffix}.npz"


def normal_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_binary" in data.files:
        return data["y_binary"].astype(np.int8) == 0
    return data["y_attack_type"].astype(np.int64) == 0


def load_normal_windows(split: str, max_windows: int | None = None) -> np.ndarray:
    data = np.load(WINDOWS / f"{split}_windows.npz", allow_pickle=True)
    idx = np.flatnonzero(normal_mask(data))
    if max_windows is not None and len(idx) > max_windows:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(idx, size=max_windows, replace=False)
    return data["x"][idx].astype(np.float32)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p]))
    kl_qm = np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q]))
    return float(0.5 * (kl_pm + kl_qm))


def entropy_bits(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def binned_count(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    return counts.astype(np.float64)


def window_distribution_summary(x: np.ndarray) -> dict:
    can_ids = np.clip(np.rint(x[:, :, 0]).astype(np.int64), 0, 2047).ravel()
    dlc = np.clip(np.rint(x[:, :, 1]).astype(np.int64), 0, 8).ravel()
    delta_t = np.clip(x[:, :, 10].astype(np.float64).ravel(), 0.0, None)
    payload = np.clip(np.rint(x[:, :, 2:10]).astype(np.int64), 0, 255).ravel()

    can_counts = np.bincount(can_ids, minlength=2048)
    dlc_counts = np.bincount(dlc, minlength=9)
    payload_counts = np.bincount(payload, minlength=256)
    log_dt = np.log10(delta_t + 1e-8)
    log_dt_bins = np.linspace(-8, -0.5, 80)
    log_dt_counts = binned_count(log_dt, log_dt_bins)

    top_order = np.argsort(can_counts)[::-1]
    top20 = top_order[:20]
    top50 = top_order[:50]
    frames = len(can_ids)
    return {
        "windows": int(len(x)),
        "frames": int(frames),
        "can_counts": can_counts,
        "dlc_counts": dlc_counts,
        "payload_counts": payload_counts,
        "log_dt_counts": log_dt_counts,
        "can_unique": int(np.count_nonzero(can_counts)),
        "can_entropy": entropy_bits(can_counts),
        "can_top20_ids": top20,
        "can_top20_coverage": float(can_counts[top20].sum() / frames),
        "can_top50_ids": set(int(v) for v in top50 if can_counts[v] > 0),
        "dlc_entropy": entropy_bits(dlc_counts),
        "dlc8_share": float(dlc_counts[8] / max(dlc_counts.sum(), 1)),
        "payload_entropy": entropy_bits(payload_counts),
        "delta_t_mean": float(delta_t.mean()),
        "delta_t_median": float(np.median(delta_t)),
        "delta_t_p95": float(np.quantile(delta_t, 0.95)),
        "delta_t_p99": float(np.quantile(delta_t, 0.99)),
    }


def write_distribution_diagnostics() -> None:
    car_x = load_normal_windows("test", MAX_NORMAL_WINDOWS)
    otids_x = load_normal_windows("otids_cross", MAX_NORMAL_WINDOWS)
    car = window_distribution_summary(car_x)
    otids = window_distribution_summary(otids_x)

    car_top20 = car["can_top20_ids"]
    otids_top20_coverage_by_car = float(otids["can_counts"][car_top20].sum() / max(otids["can_counts"].sum(), 1))
    top50_union = len(car["can_top50_ids"] | otids["can_top50_ids"])
    top50_intersection = len(car["can_top50_ids"] & otids["can_top50_ids"])
    top50_jaccard = float(top50_intersection / top50_union) if top50_union else 0.0

    summary_rows = []
    for split, stats in [("car_hacking_test_normal", car), ("otids_normal", otids)]:
        summary_rows.append(
            {
                "split": split,
                "sampled_windows": stats["windows"],
                "sampled_frames": stats["frames"],
                "can_unique": stats["can_unique"],
                "can_entropy_bits": stats["can_entropy"],
                "can_top20_coverage": stats["can_top20_coverage"],
                "dlc_entropy_bits": stats["dlc_entropy"],
                "dlc8_share": stats["dlc8_share"],
                "delta_t_mean": stats["delta_t_mean"],
                "delta_t_median": stats["delta_t_median"],
                "delta_t_p95": stats["delta_t_p95"],
                "delta_t_p99": stats["delta_t_p99"],
                "payload_entropy_bits": stats["payload_entropy"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "otids_normal_distribution_summary.csv", index=False)
    summary.to_csv(EXP / "otids_normal_distribution_summary.csv", index=False)

    key_rows = [
        {
            "diagnostic": "CAN ID support",
            "car_hacking_test_normal": f"{car['can_unique']} IDs; top-20 coverage {car['can_top20_coverage']:.4f}",
            "otids_normal": f"{otids['can_unique']} IDs; Car-Hacking top-20 coverage {otids_top20_coverage_by_car:.4f}",
            "shift_value": f"JS={js_divergence(car['can_counts'], otids['can_counts']):.4f}; top-50 Jaccard={top50_jaccard:.4f}",
            "interpretation": "External normal traffic uses a different CAN-ID support and frequency profile.",
        },
        {
            "diagnostic": "DLC distribution",
            "car_hacking_test_normal": f"DLC=8 share {car['dlc8_share']:.4f}; entropy {car['dlc_entropy']:.4f}",
            "otids_normal": f"DLC=8 share {otids['dlc8_share']:.4f}; entropy {otids['dlc_entropy']:.4f}",
            "shift_value": f"JS={js_divergence(car['dlc_counts'], otids['dlc_counts']):.4f}",
            "interpretation": "Frame-length mix changes between the in-dataset and external normal windows.",
        },
        {
            "diagnostic": "Inter-arrival time",
            "car_hacking_test_normal": f"median {car['delta_t_median']:.6g}; p99 {car['delta_t_p99']:.6g}",
            "otids_normal": f"median {otids['delta_t_median']:.6g}; p99 {otids['delta_t_p99']:.6g}",
            "shift_value": f"log-delta_t JS={js_divergence(car['log_dt_counts'], otids['log_dt_counts']):.4f}",
            "interpretation": "Timing support differs, so a validation-calibrated normal boundary need not transfer.",
        },
        {
            "diagnostic": "Payload byte distribution",
            "car_hacking_test_normal": f"entropy {car['payload_entropy']:.4f} bits",
            "otids_normal": f"entropy {otids['payload_entropy']:.4f} bits",
            "shift_value": f"JS={js_divergence(car['payload_counts'], otids['payload_counts']):.4f}",
            "interpretation": "Normal payload-byte statistics are not drawn from the same support.",
        },
    ]
    key = pd.DataFrame(key_rows)
    key.to_csv(TABLES / "otids_normal_shift_key_comparisons.csv", index=False)
    key.to_csv(EXP / "otids_normal_shift_key_comparisons.csv", index=False)


def score_windows(setting: dict, seed: int, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_path, std_path = model_paths(setting, seed)
    stdz = np.load(std_path)
    x_std = standardize(x, stdz["mean"], stdz["std"])
    model = CNN1D(in_channels=x.shape[-1], classes=5).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    loader = DataLoader(
        WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)),
        batch_size=1024,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    _, probs = predict_cnn(model, loader, device)
    return 1.0 - probs[:, 0]


def summarize_scores(scores: np.ndarray) -> dict:
    return {
        "normal_windows": int(len(scores)),
        "attack_score_mean": float(scores.mean()),
        "attack_score_q50": float(np.quantile(scores, 0.50)),
        "attack_score_q90": float(np.quantile(scores, 0.90)),
        "attack_score_q95": float(np.quantile(scores, 0.95)),
        "attack_score_q99": float(np.quantile(scores, 0.99)),
        "attack_score_max": float(scores.max()),
        "share_score_ge_0p5": float((scores >= 0.5).mean()),
        "share_score_ge_0p9": float((scores >= 0.9).mean()),
        "share_score_ge_0p99": float((scores >= 0.99).mean()),
    }


def summarize_group(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["seeds"] = ";".join(str(v) for v in sorted(group["seed"].unique()))
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_score_diagnostics() -> None:
    rows = []
    sampled_scores: dict[tuple[str, str], list[np.ndarray]] = {}
    rng = np.random.default_rng(RNG_SEED)
    for split_label, split in SCORE_SPLITS:
        x = load_normal_windows(split)
        for setting in SETTINGS:
            for seed in SEEDS:
                print(f"score diagnostic {setting['setting']} seed={seed} split={split_label}")
                scores = score_windows(setting, seed, x)
                row = {
                    "setting": setting["setting"],
                    "label": setting["label"],
                    "family": setting["family"],
                    "ratio": setting["ratio"],
                    "seed": seed,
                    "normal_split": split_label,
                }
                row.update(summarize_scores(scores))
                rows.append(row)
                key = (setting["setting"], split_label)
                sample_size = min(len(scores), 50_000)
                sampled_scores.setdefault(key, []).append(rng.choice(scores, size=sample_size, replace=False))

    detail = pd.DataFrame(rows)
    detail.to_csv(TABLES / "otids_normal_attack_score_details.csv", index=False)
    detail.to_csv(EXP / "otids_normal_attack_score_details.csv", index=False)
    summary = summarize_group(
        detail,
        ["setting", "label", "family", "ratio", "normal_split"],
        [
            "attack_score_mean",
            "attack_score_q50",
            "attack_score_q90",
            "attack_score_q95",
            "attack_score_q99",
            "attack_score_max",
            "share_score_ge_0p5",
            "share_score_ge_0p9",
            "share_score_ge_0p99",
        ],
    )
    summary.to_csv(TABLES / "otids_normal_attack_score_summary.csv", index=False)
    summary.to_csv(EXP / "otids_normal_attack_score_summary.csv", index=False)

    key = summary[
        summary["setting"].isin(["real_only", "rule_0p30", "rule_1p00"])
        & summary["normal_split"].isin(["validation_normal", "otids_normal"])
    ].copy()
    key.to_csv(TABLES / "otids_normal_attack_score_key_comparisons.csv", index=False)
    key.to_csv(EXP / "otids_normal_attack_score_key_comparisons.csv", index=False)

    plot_score_histograms(sampled_scores)


def plot_score_histograms(sampled_scores: dict[tuple[str, str], list[np.ndarray]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, 80)
    split_colors = {
        "validation_normal": "#2563eb",
        "car_hacking_test_normal": "#16a34a",
        "otids_normal": "#dc2626",
    }
    split_labels = {
        "validation_normal": "Car-Hacking val normal",
        "car_hacking_test_normal": "Car-Hacking test normal",
        "otids_normal": "OTIDS normal",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, setting in zip(axes, ["real_only", "rule_0p30", "rule_1p00"]):
        for split in ["validation_normal", "car_hacking_test_normal", "otids_normal"]:
            arrays = sampled_scores.get((setting, split), [])
            if not arrays:
                continue
            values = np.concatenate(arrays)
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=split_colors[split],
                label=split_labels[split],
            )
        ax.set_title(setting.replace("_", " "))
        ax.set_xlabel("Attack score")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "otids_normal_attack_score_histogram.png", dpi=220)
    plt.close(fig)


def write_rf_vs_cnn_diagnostic() -> None:
    baseline = pd.read_csv(TABLES / "baseline_metrics_mean_std.csv")
    main = pd.read_csv(TABLES / "main_results_mean_std.csv")
    rows = []

    for model in ["RandomForest", "CNN1D"]:
        for split in ["test", "otids_cross_binary"]:
            row = baseline[(baseline["model"] == model) & (baseline["split"] == split)].iloc[0]
            rows.append(
                {
                    "model_setting": f"{model} real-only",
                    "split": "Car-Hacking test" if split == "test" else "OTIDS cross-test",
                    "macro_f1_binary_mean": row["macro_f1_binary_mean"],
                    "macro_f1_binary_std": row["macro_f1_binary_std"],
                    "attack_recall_mean": row["attack_recall_mean"],
                    "attack_recall_std": row["attack_recall_std"],
                    "normal_recall_mean": row["normal_recall_mean"],
                    "normal_recall_std": row["normal_recall_std"],
                    "fpr_mean": row["fpr_mean"],
                    "fpr_std": row["fpr_std"],
                    "interpretation": "architecture contrast under real-only training",
                }
            )

    selected_main = main[
        (main["source_table"] == "test_distribution")
        & (main["generator"] == "rule_based")
        & (main["ratio"].round(2) == 0.50)
        & (main["split"].isin(["test", "otids_cross_binary"]))
    ]
    for _, row in selected_main.iterrows():
        rows.append(
            {
                "model_setting": "CNN1D Rule +50%",
                "split": "Car-Hacking test" if row["split"] == "test" else "OTIDS cross-test",
                "macro_f1_binary_mean": row["macro_f1_binary_mean"],
                "macro_f1_binary_std": row["macro_f1_binary_std"],
                "attack_recall_mean": row["attack_recall_mean"],
                "attack_recall_std": row["attack_recall_std"],
                "normal_recall_mean": row["normal_recall_mean"],
                "normal_recall_std": row["normal_recall_std"],
                "fpr_mean": row["fpr_mean"],
                "fpr_std": row["fpr_std"],
                "interpretation": "main augmented CNN setting; shows augmentation does not repair OTIDS normal FPR",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "rf_vs_cnn_cross_dataset_diagnostic.csv", index=False)
    out.to_csv(EXP / "rf_vs_cnn_cross_dataset_diagnostic.csv", index=False)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    write_distribution_diagnostics()
    write_score_diagnostics()
    write_rf_vs_cnn_diagnostic()


if __name__ == "__main__":
    main()
