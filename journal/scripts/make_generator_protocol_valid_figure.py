#!/usr/bin/env python3
"""Figure for protocol-corrected learned-generator detector reruns.

Controlled-rung paired estimates are separated by detector family.  The external panel
plots FPR and attack recall jointly so a low-FPR inert model or a high-recall
everything-is-attack model cannot look successful in isolation.

Sized for \\textwidth insertion (A4, 2.5 cm margins -> ~6.3 in).  The two
controlled panels occupy the first row and the joint external operating-point
panel spans the second row, avoiding the sub-7-pt labels produced by the old
three-panels-in-one-row layout.  Rung names follow the manuscript Table 1
Evaluation column ("fixed rule variant", "coupled (L2) sweep").
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib_common as lc


FILES = [
    "generator_extension_summary.csv",
    "generator_extension_summary_gan_protocol_valid.csv",
    "generator_extension_summary_arlm_protocol_valid.csv",
    "rf_generator_extension_summary.csv",
    "rf_generator_extension_summary_gan_protocol_valid.csv",
    "rf_generator_extension_summary_arlm_protocol_valid.csv",
]
SETTINGS = ["real_only", "rule_0p30", "ganvalid_0p30",
            "ganvalid_1p00", "arlmvalid_0p30"]
LABELS = {
    "real_only": "Real only",
    "rule_0p30": "Rule +30%",
    "ganvalid_0p30": "WGAN corrected +30%",
    "ganvalid_1p00": "WGAN corrected +100%",
    "arlmvalid_0p30": "AR-LM corrected +30%",
}
COLORS = {
    "real_only": "#7f7f7f",
    "rule_0p30": "#0072B2",
    "ganvalid_0p30": "#E69F00",
    "ganvalid_1p00": "#D55E00",
    "arlmvalid_0p30": "#009E73",
}
SEED_SET = "7;42;123;2026;3407"
SOFT_RED = "#FBE8E4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
})


def load_rows():
    parts = []
    for name in FILES:
        path = lc.TABLES / name
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_csv(path))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["setting"].isin(SETTINGS) & (df["subset"] == "all")].copy()
    # Corrected/specific files occur after canonical files and should win any
    # duplicate family/setting/rung rows.
    df = df.drop_duplicates(["family", "setting", "rung", "subset"], keep="last")
    plotted = df[df["rung"].isin(["L1_fixed_variant", "L2_sensitivity", "L5_otids"])]
    seed_sets = set(plotted["seeds"].astype(str))
    if seed_sets != {SEED_SET}:
        raise ValueError(f"expected the common five-seed set; found {seed_sets}")
    return df


def value(df, family, setting, rung, metric):
    row = df[(df["family"] == family) & (df["setting"] == setting)
             & (df["rung"] == rung)]
    if row.empty:
        return np.nan, 0.0
    mean = float(row[f"{metric}_mean"].iloc[0])
    std = float(row[f"{metric}_std"].iloc[0])
    return mean, std


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def controlled_panel(ax, df, family):
    y = np.arange(len(SETTINGS))[::-1]
    for ypos, setting in zip(y, SETTINGS):
        l1, l1_sd = value(df, family, setting, "L1_fixed_variant", "attack_recall")
        l2, l2_sd = value(df, family, setting, "L2_sensitivity", "attack_recall")
        ax.plot([l1, l2], [ypos, ypos], color=COLORS[setting], linewidth=1.1,
                alpha=0.55, zorder=1)
        ax.errorbar(
            l1,
            ypos,
            xerr=l1_sd,
            fmt="o",
            markersize=5.8,
            markerfacecolor="white",
            markeredgecolor=COLORS[setting],
            markeredgewidth=1.1,
            ecolor=COLORS[setting],
            elinewidth=0.8,
            capsize=1.8,
            zorder=3,
        )
        ax.errorbar(
            l2,
            ypos,
            xerr=l2_sd,
            fmt="s",
            markersize=5.5,
            markerfacecolor=COLORS[setting],
            markeredgecolor=COLORS[setting],
            markeredgewidth=1.0,
            ecolor=COLORS[setting],
            elinewidth=0.8,
            capsize=1.8,
            zorder=3,
        )
    ax.set_yticks(y)
    if family == "cnn":
        ax.set_yticklabels([LABELS[setting] for setting in SETTINGS], fontsize=7.8)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(0.42, 1.035)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(-0.55, len(SETTINGS) - 0.45)
    ax.set_xlabel("Binary attack recall")
    ax.set_title(f"({ 'a' if family == 'cnn' else 'b' }) {family.upper()} · controlled rungs",
                 loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    style_axis(ax)


def external_panel(ax, df):
    ax.axvspan(0.95, 1.03, color=SOFT_RED, zorder=0)
    ax.axvline(0.95, color="#D55E00", linewidth=0.8, linestyle=":")
    markers = {"cnn": "o", "rf": "s"}
    for family in ["cnn", "rf"]:
        for setting in SETTINGS:
            fpr, fpr_sd = value(df, family, setting, "L5_otids", "fpr")
            recall, rec_sd = value(df, family, setting, "L5_otids",
                                   "attack_recall")
            if not np.isfinite(fpr) or not np.isfinite(recall):
                continue
            # FPR and recall are bounded at [0, 1]; truncate SD whiskers.
            xerr = [[min(fpr_sd, fpr)], [min(fpr_sd, 1.0 - fpr)]]
            yerr = [[min(rec_sd, recall)], [min(rec_sd, 1.0 - recall)]]
            ax.errorbar(fpr, recall, xerr=xerr, yerr=yerr,
                        fmt=markers[family], markersize=5.5,
                        markerfacecolor=COLORS[setting],
                        markeredgecolor="black", markeredgewidth=0.4,
                        ecolor=COLORS[setting], elinewidth=0.7, capsize=1.5,
                        zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("OTIDS false-positive rate")
    ax.set_ylabel("OTIDS attack recall")
    ax.set_title("(c) OTIDS operating points · upper-left preferred",
                 loc="left", fontweight="bold")
    ax.grid(color="#dddddd", linewidth=0.5)
    style_axis(ax)
    family_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=5.5,
                   markerfacecolor="white", markeredgecolor="black", label="CNN"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=5.5,
                   markerfacecolor="white", markeredgecolor="black", label="RF"),
    ]
    # Lower left is empty; lower right would collide with the FPR band label.
    family_legend = ax.legend(handles=family_handles, loc="lower left",
                              frameon=False, fontsize=8, handletextpad=0.3)
    ax.add_artist(family_legend)
    # Panel (c) previously relied on panel (a)'s y-axis labels for the
    # color-to-arm mapping; make it self-contained with its own arm legend.
    arm_handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=5.5,
                   markerfacecolor=COLORS[setting],
                   markeredgecolor=COLORS[setting], label=LABELS[setting])
        for setting in SETTINGS
    ]
    ax.legend(handles=arm_handles, loc="lower center", ncol=2, frameon=False,
              fontsize=6.9, handletextpad=0.25, columnspacing=0.9,
              bbox_to_anchor=(0.54, 0.0))
    ax.text(0.935, 0.33, "FPR ≥ 0.95", rotation=90, ha="right", va="center",
            fontsize=7.5, color="#D55E00")


def main():
    df = load_rows()
    fig = plt.figure(figsize=(6.4, 4.65))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.08], hspace=0.56,
                            wspace=0.20)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    controlled_panel(axes[0], df, "cnn")
    controlled_panel(axes[1], df, "rf")
    external_panel(axes[2], df)
    rung_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=5.8,
                   markerfacecolor="white", markeredgecolor="#444444",
                   label="L1 fixed variant"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=5.5,
                   markerfacecolor="#444444", markeredgecolor="#444444",
                   label="L2 coupled sweep"),
    ]
    fig.legend(rung_handles, [h.get_label() for h in rung_handles],
               loc="center", ncol=2, frameon=False, fontsize=7.7,
               columnspacing=1.3, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.505))
    fig.subplots_adjust(left=0.22, right=0.985, top=0.96, bottom=0.10)
    lc.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(lc.FIGURES / f"generator_protocol_valid_ladder.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(lc.FIGURES / "generator_protocol_valid_ladder.png")


if __name__ == "__main__":
    main()
