#!/usr/bin/env python3
"""Tier 0 evaluation-ladder figure.

Loads existing result tables (no hardcoded numbers) and visualizes how
detection performance decomposes across evaluation rungs, from saturated
in-dataset testing down to external cross-dataset failure.

Panel (a) breaks the ladder line at metric boundaries (attack recall ->
detection recall -> exact class recall) and overlays hollow markers with
BINARY detection recall at the two exact-class rungs, so the exact-class
collapse is not misread as a binary-detection collapse.

Panel (b) adds an RF (real-only) series whose external FPR escapes the
CNN ceiling on OTIDS and can-train, giving the bars visual information.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PAPER_FIGURES = ROOT / "results" / "paper" / "figures"
LOGS = ROOT / "results" / "logs"

SETTINGS = [
    ("real_only", "Real only"),
    ("rule_0p30", "Rule +30%"),
]
SETTING_COLORS = {"real_only": "tab:blue", "rule_0p30": "tab:orange"}

# Rung indices per underlying metric (panel a).
ATTACK_RECALL_IDX = [0, 1]  # in-dataset test, fixed variant
DETECTION_IDX = [2]  # sensitivity sweep
EXACT_CLASS_IDX = [3, 4]  # target-ID shift, out-of-generator


def main_results_value(df: pd.DataFrame, setting: str, split: str) -> tuple[float, float]:
    sub = df[df["source_table"] == "test_distribution"]
    if setting == "real_only":
        sub = sub[(sub["generator"] == "real_only") & (sub["ratio"] == 0.0)]
    elif setting == "rule_0p30":
        sub = sub[(sub["generator"] == "rule_based") & (sub["ratio"] == 0.3)]
    else:
        raise ValueError(setting)
    row = sub[sub["split"] == split].iloc[0]
    return float(row["attack_recall_mean"]), float(row["attack_recall_std"])


def shifted_exact_recall(df: pd.DataFrame, setting: str) -> tuple[float, float]:
    sub = df[df["setting"] == setting]
    mean = float(sub["shifted_exact_recall_mean"].mean())
    std = float(np.sqrt((sub["shifted_exact_recall_std"] ** 2).mean()))
    return mean, std


def collect_rungs() -> tuple[list[str], dict, dict, dict, dict]:
    main_df = pd.read_csv(TABLES / "main_results_mean_std.csv")
    sens_df = pd.read_csv(TABLES / "variant_sensitivity_summary.csv")
    tid_df = pd.read_csv(TABLES / "target_id_shift_key_comparisons.csv")
    oog_df = pd.read_csv(TABLES / "out_of_generator_key_comparisons.csv")
    tid_sum_df = pd.read_csv(TABLES / "target_id_shift_summary.csv")
    oog_sum_df = pd.read_csv(TABLES / "out_of_generator_summary.csv")
    cantt_df = pd.read_csv(TABLES / "cantt_external_eval_summary_mean_std.csv")
    baseline_df = pd.read_csv(TABLES / "baseline_metrics_mean_std.csv")
    rf_ext_df = pd.read_csv(TABLES / "rf_external_summary.csv")
    road_path = TABLES / "road_external_summary.csv"
    road_df = pd.read_csv(road_path) if road_path.exists() else None

    # Per-rung metrics (attack/detection/exact recall) are stated in the
    # manuscript caption to keep tick labels legible at print size.
    rungs = [
        "In-dataset\ntest",
        "Fixed\nvariant",
        "Sensitivity\nsweep",
        "Target-ID\nshift",
        "Out-of-\ngenerator",
    ]
    perf: dict[str, tuple[list[float], list[float]]] = {}
    det_overlay: dict[str, tuple[list[float], list[float]]] = {}
    fpr: dict[str, dict[str, tuple[float, float]]] = {}
    for setting, _label in SETTINGS:
        means, stds = [], []
        for split in ["test", "variant_test"]:
            m, s = main_results_value(main_df, setting, split)
            means.append(m)
            stds.append(s)
        sens = sens_df[sens_df["setting"] == setting].iloc[0]
        means.append(float(sens["attack_detection_recall_mean"]))
        stds.append(float(sens["attack_detection_recall_std"]))
        for df in [tid_df, oog_df]:
            m, s = shifted_exact_recall(df, setting)
            means.append(m)
            stds.append(s)
        perf[setting] = (means, stds)

        # Hollow-marker overlay: BINARY detection recall on the same windows
        # as the two exact-class rungs (target-ID shift, out-of-generator).
        ov_means, ov_stds = [], []
        for sum_df in [tid_sum_df, oog_sum_df]:
            row = sum_df[sum_df["setting"] == setting].iloc[0]
            ov_means.append(float(row["attack_detection_recall_mean"]))
            ov_stds.append(float(row["attack_detection_recall_std"]))
        det_overlay[setting] = (ov_means, ov_stds)

        otids_m, otids_s = main_results_value(main_df, setting, "otids_cross_binary")
        # main_results stores attack_recall; FPR comes from its own columns.
        sub = main_df[main_df["source_table"] == "test_distribution"]
        if setting == "real_only":
            sub = sub[(sub["generator"] == "real_only") & (sub["ratio"] == 0.0)]
        else:
            sub = sub[(sub["generator"] == "rule_based") & (sub["ratio"] == 0.3)]
        otids_row = sub[sub["split"] == "otids_cross_binary"].iloc[0]
        cantt_row = cantt_df[cantt_df["setting"] == setting].iloc[0]
        fpr[setting] = {
            "OTIDS": (float(otids_row["fpr_mean"]), float(otids_row["fpr_std"])),
            "can-train": (float(cantt_row["fpr_mean"]), float(cantt_row["fpr_std"])),
        }
        if road_df is not None:
            road_row = road_df[
                (road_df["setting"] == setting) & (road_df["level"] == "all") & (road_df["axis"] == "all")
            ].iloc[0]
            fpr[setting]["ROAD"] = (float(road_row["fpr_mean"]), float(road_row["fpr_std"]))

    # RF (real-only) external FPR series for panel (b).
    rf_fpr: dict[str, tuple[float, float]] = {}
    rf_otids = baseline_df[
        (baseline_df["model"] == "RandomForest") & (baseline_df["split"] == "otids_cross_binary")
    ].iloc[0]
    rf_fpr["OTIDS"] = (float(rf_otids["fpr_mean"]), float(rf_otids["fpr_std"]))
    rf_ext = rf_ext_df[(rf_ext_df["model"] == "RandomForest_real_only") & (rf_ext_df["axis"] == "all")]
    rf_cantt = rf_ext[rf_ext["dataset"] == "cantt_external"].iloc[0]
    rf_fpr["can-train"] = (float(rf_cantt["fpr_mean"]), float(rf_cantt["fpr_std"]))
    rf_road = rf_ext[rf_ext["dataset"] == "road"].iloc[0]
    rf_fpr["ROAD"] = (float(rf_road["fpr_mean"]), float(rf_road["fpr_std"]))

    return rungs, perf, det_overlay, fpr, rf_fpr


def main() -> None:
    start = time.time()
    rungs, perf, det_overlay, fpr, rf_fpr = collect_rungs()

    # Sized for print: included at full LNCS text width (12.2 cm = 4.80 in),
    # so on-canvas font sizes are the printed font sizes.
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(4.8, 1.64), gridspec_kw={"width_ratios": [3.0, 1.45]}
    )
    xs = np.arange(len(rungs))

    # --- Panel (a): ladder with metric-boundary breaks ----------------------
    # Background shading per underlying metric, with tiny labels at the top.
    ax1.set_xlim(-0.5, 4.5)
    ax1.axvspan(1.5, 2.5, facecolor="0.88", alpha=0.55, zorder=0)
    ax1.axvspan(2.5, 4.5, facecolor="0.94", alpha=0.55, zorder=0)
    label_kw = dict(ha="center", va="center", fontsize=6, color="0.25", zorder=4)
    ax1.text(0.5, 1.115, "attack recall", **label_kw)
    ax1.text(2.0, 1.115, "detection", **label_kw)
    ax1.text(3.5, 1.115, "exact class", **label_kw)

    metric_segments = [ATTACK_RECALL_IDX, DETECTION_IDX, EXACT_CLASS_IDX]
    for setting, label in SETTINGS:
        means, stds = perf[setting]
        color = SETTING_COLORS[setting]
        for k, seg in enumerate(metric_segments):
            ax1.errorbar(
                [xs[i] for i in seg],
                [means[i] for i in seg],
                yerr=[stds[i] for i in seg],
                marker="o",
                markersize=3,
                linewidth=1,
                capsize=2.5,
                color=color,
                label=label if k == 0 else None,
                zorder=3,
            )
        # Hollow markers: binary detection recall at the exact-class rungs
        # (slightly x-offset; error bars omitted to avoid clutter).
        ov_means, _ov_stds = det_overlay[setting]
        ax1.plot(
            [xs[i] + 0.18 for i in EXACT_CLASS_IDX],
            ov_means,
            linestyle=":",
            linewidth=0.8,
            marker="o",
            markersize=3.4,
            markerfacecolor="none",
            markeredgewidth=0.8,
            color=color,
            zorder=3,
        )
    ax1.set_xticks(xs)
    ax1.set_xticklabels(rungs, fontsize=6)
    ax1.tick_params(axis="y", labelsize=6.5)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_ylabel("Recall", fontsize=7)
    ax1.set_ylim(0.0, 1.22)
    ax1.set_title("(a) Detection ladder", fontsize=7.5)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(
        Line2D(
            [], [], linestyle=":", linewidth=0.8, marker="o", markersize=3.4,
            markerfacecolor="none", markeredgewidth=0.8, color="0.35",
        )
    )
    labels.append("Binary detection")
    ax1.legend(handles, labels, fontsize=6, loc="lower left", framealpha=1.0,
               borderpad=0.3, handletextpad=0.5, handlelength=1.6, labelspacing=0.25,
               borderaxespad=0.15)
    ax1.grid(axis="y", linestyle=":", linewidth=0.5)

    # --- Panel (b): external FPR with RF (real-only) contrast ---------------
    series = [(setting, label, SETTING_COLORS[setting]) for setting, label in SETTINGS]
    datasets = [d for d in ["OTIDS", "can-train", "ROAD"] if d in fpr[SETTINGS[0][0]]]
    width = 0.26
    for i, (setting, label, color) in enumerate(series):
        values = [fpr[setting][d][0] for d in datasets]
        errs = [fpr[setting][d][1] for d in datasets]
        ax2.bar(
            [j + (i - 1) * width for j in range(len(datasets))],
            values,
            width=width,
            yerr=errs,
            capsize=1.5,
            color=color,
            error_kw={"linewidth": 0.7},
        )
    rf_values = [rf_fpr[d][0] for d in datasets]
    rf_errs = [rf_fpr[d][1] for d in datasets]
    ax2.bar(
        [j + width for j in range(len(datasets))],
        rf_values,
        width=width,
        yerr=rf_errs,
        capsize=1.5,
        color="0.55",
        error_kw={"linewidth": 0.7},
        label="RF (real-only)",
    )
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, fontsize=6)
    ax2.tick_params(axis="y", labelsize=6.5)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel("False positive rate", fontsize=7)
    ax2.set_ylim(0.0, 1.22)
    ax2.set_title("(b) External FPR", fontsize=7.5)
    # CNN bars reuse the panel-(a) colors/legend; only the RF series is new.
    ax2.legend(fontsize=6, loc="upper center", framealpha=1.0, borderpad=0.3,
               handletextpad=0.5, handlelength=1.2, borderaxespad=0.15)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "evaluation_ladder.png", dpi=300)
    fig.savefig(PAPER_FIGURES / "figure_37_evaluation_ladder.png", dpi=300)
    fig.savefig(PAPER_FIGURES / "figure_37_evaluation_ladder.pdf")
    plt.close(fig)

    log = {
        "rungs": [r.replace("\n", " ") for r in rungs],
        "binary_detection_overlay": {
            setting: dict(zip(["target_id_shift", "out_of_generator"], det_overlay[setting][0]))
            for setting, _ in SETTINGS
        },
        "rf_external_fpr": {d: rf_fpr[d][0] for d in rf_fpr},
        "sources": [
            "results/tables/main_results_mean_std.csv",
            "results/tables/variant_sensitivity_summary.csv",
            "results/tables/target_id_shift_key_comparisons.csv",
            "results/tables/out_of_generator_key_comparisons.csv",
            "results/tables/target_id_shift_summary.csv",
            "results/tables/out_of_generator_summary.csv",
            "results/tables/cantt_external_eval_summary_mean_std.csv",
            "results/tables/road_external_summary.csv",
            "results/tables/baseline_metrics_mean_std.csv",
            "results/tables/rf_external_summary.csv",
        ],
        "outputs": [
            "results/figures/evaluation_ladder.png",
            "results/paper/figures/figure_37_evaluation_ladder.png",
            "results/paper/figures/figure_37_evaluation_ladder.pdf",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "evaluation_ladder_figure.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
