#!/usr/bin/env python3
"""Main-text L5 figure: joint external FPR/attack-recall operating points.

One panel per external dataset (OTIDS, can-train, ROAD) plots the five-seed
mean operating point of every graded detector family under the real-only,
Rule +30%, and Rule +100% arms, with SD whiskers truncated at the [0, 1]
bounds. Gray connectors run real-only -> +30% -> +100% with a mid-segment
arrowhead on the longest segment, so the direction of the augmentation
response is visible; the shaded band marks the FPR >= 0.95
everything-is-attack region. Both Conv-AE threshold policies sit at exactly
(1, 1) in every panel and are drawn once (asserted below).

This is the scripted replacement for the interim Penpot-authored version of
the figure: values are read from the same summary tables as the ladder
overview heatmap instead of being baked into a hand-built SVG.

Inputs (read-only):
  journal/results/tables/generator_extension_summary.csv           (CNN)
  journal/results/tables/rf_generator_extension_summary.csv        (RF OTIDS)
  journal/results/tables/rf_generator_extension_external_summary.csv
  journal/results/tables/family_extension_summary.csv              (BiLSTM, AE)
  journal/results/tables/family_extension_transformer_summary.csv

Outputs:
  journal/results/figures/external_operating_points.(png|pdf)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import lib_common as lc


GRAY = "#666666"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
INK = "#222222"
GRID = "#DDDDDD"
SOFT_RED = "#FBE8E4"
CONNECTOR = "#AAAAAA"

ARMS = ["real_only", "rule_0p30", "rule_1p00"]
ARM_STYLE = {
    "real_only": dict(color=GRAY, open_face=True),
    "rule_0p30": dict(color=BLUE, open_face=False),
    "rule_1p00": dict(color=VERMILLION, open_face=False),
}
FAMILIES = [
    ("cnn", "o", "1D-CNN"),
    ("rf", "s", "RF"),
    ("lstm", "^", "BiLSTM"),
    ("transformer", "D", "Transf. (expl.)"),
]
AE_SETTINGS = ["ae_percentile", "ae_synthetic_calibrated"]
RUNGS = [("L5_otids", "(a) OTIDS"), ("L5_cantt", "(b) can-train"),
         ("L5_road", "(c) ROAD")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.8,
})


def load_combined() -> pd.DataFrame:
    cnn = pd.read_csv(lc.TABLES / "generator_extension_summary.csv")
    rf = pd.read_csv(lc.TABLES / "rf_generator_extension_summary.csv")
    rf_ext = pd.read_csv(lc.TABLES / "rf_generator_extension_external_summary.csv")
    if "family" not in rf_ext.columns:
        rf_ext = rf_ext.assign(family="rf")
    fam = pd.read_csv(lc.TABLES / "family_extension_summary.csv")
    tf = pd.read_csv(lc.TABLES / "family_extension_transformer_summary.csv")
    keep = ["family", "setting", "rung", "subset", "fpr_mean", "fpr_std",
            "attack_recall_mean", "attack_recall_std"]
    frames = [df[[c for c in keep if c in df.columns]]
              for df in (cnn, rf, rf_ext, fam, tf)]
    return pd.concat(frames, ignore_index=True)


def cell(combined: pd.DataFrame, family: str, setting: str,
         rung: str) -> tuple[float, float, float, float]:
    sub = combined[
        (combined["family"] == family)
        & (combined["setting"] == setting)
        & (combined["rung"] == rung)
        & (combined["subset"] == "all")
    ].drop_duplicates()
    if len(sub) != 1:
        raise ValueError(f"expected one row for {family}/{setting}/{rung}, "
                         f"found {len(sub)}")
    row = sub.iloc[0]
    return (float(row["fpr_mean"]), float(row["fpr_std"]),
            float(row["attack_recall_mean"]), float(row["attack_recall_std"]))


def _truncated_err(value: float, std: float) -> np.ndarray:
    return np.array([[min(std, value)], [min(std, 1.0 - value)]])


def _mid_arrow(ax: plt.Axes, p1: tuple[float, float],
               p2: tuple[float, float]) -> None:
    vec = np.array(p2) - np.array(p1)
    length = float(np.hypot(*vec))
    if length < 0.075:
        return
    mid = (np.array(p1) + np.array(p2)) / 2
    step = vec / length * 0.011
    ax.annotate(
        "", xy=mid + step, xytext=mid - step,
        arrowprops=dict(arrowstyle="-|>", color="#8A8A8A", linewidth=0.1,
                        mutation_scale=8),
        zorder=2.6, annotation_clip=True,
    )


def make_figure(
    combined: pd.DataFrame,
    *,
    families: list[tuple[str, str, str]] | None = None,
    ae_settings: list[str] | tuple[str, ...] | None = None,
) -> None:
    families = FAMILIES if families is None else families
    ae_settings = AE_SETTINGS if ae_settings is None else ae_settings
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.15), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.992, top=0.90, bottom=0.335,
                        wspace=0.10)

    road_points = []
    for ax, (rung, title) in zip(axes, RUNGS):
        ax.axvspan(0.95, 1.035, color=SOFT_RED, zorder=0)
        ax.axvline(0.95, color=VERMILLION, linewidth=0.8, linestyle=":",
                   zorder=1)
        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold",
                     pad=7)

        # Both AE threshold policies coincide at (1, 1); assert and draw once,
        # at the bottom of the corner pile.
        if ae_settings:
            ae_cells = [cell(combined, "ae", setting, rung)
                        for setting in ae_settings]
            for fpr, _fs, recall, _rs in ae_cells:
                if abs(fpr - 1.0) > 1e-9 or abs(recall - 1.0) > 1e-9:
                    raise ValueError(f"AE cell moved off (1, 1) for {rung}; "
                                     "update the figure and caption")
            ax.plot([1.0], [1.0], marker="v", markersize=5.2,
                    markerfacecolor="white", markeredgecolor=INK,
                    markeredgewidth=1.1, linestyle="none", zorder=3)

        for family, marker, _label in reversed(families):
            points = {arm: cell(combined, family, arm, rung) for arm in ARMS}
            xs = [points[arm][0] for arm in ARMS]
            ys = [points[arm][2] for arm in ARMS]
            ax.plot(xs, ys, color=CONNECTOR, linewidth=0.9, zorder=1.5)
            seg1 = ((xs[0], ys[0]), (xs[1], ys[1]))
            seg2 = ((xs[1], ys[1]), (xs[2], ys[2]))
            longest = max((seg1, seg2),
                          key=lambda s: np.hypot(s[1][0] - s[0][0],
                                                 s[1][1] - s[0][1]))
            _mid_arrow(ax, *longest)
            for arm in ARMS:
                fpr, fpr_std, recall, recall_std = points[arm]
                style = ARM_STYLE[arm]
                ax.errorbar(
                    fpr, recall,
                    xerr=_truncated_err(fpr, fpr_std),
                    yerr=_truncated_err(recall, recall_std),
                    fmt=marker, markersize=5.2,
                    markerfacecolor="white" if style["open_face"]
                    else style["color"],
                    markeredgecolor=style["color"] if style["open_face"]
                    else "black",
                    markeredgewidth=1.1 if style["open_face"] else 0.4,
                    ecolor=style["color"], elinewidth=0.7, capsize=1.5,
                    zorder=3,
                )
                if rung == "L5_road":
                    road_points.append((fpr, recall))

    # Direct labels (panel a) and the sub-ceiling RF label (panel b).
    label_kwargs = dict(fontsize=6.8, color="#333333", zorder=4)
    axes[0].text(0.03, 0.47, "Transf. (expl.)", ha="left", va="top",
                 **label_kwargs)
    axes[0].text(0.30, 0.655, "RF", ha="left", va="top", **label_kwargs)
    axes[0].text(0.385, 0.955, "BiLSTM", ha="left", va="top", **label_kwargs)
    axes[0].text(0.815, 0.80, "1D-CNN", ha="left", va="top", **label_kwargs)
    axes[1].text(0.42, 0.545, "RF · real only", ha="left", va="top",
                 **label_kwargs)

    # ROAD panel annotation, with data-driven bounds so the text cannot go
    # stale; the leader line points into the corner pile.
    road = np.array(road_points)
    if road[:, 0].min() < 0.999 or road[:, 1].min() < 0.998:
        raise ValueError("ROAD annotation bounds no longer hold; update text")
    axes[2].text(0.47, 0.55,
                 "every family and arm:\nFPR ≥ 0.999, recall ≥ 0.998",
                 ha="center", va="center", fontsize=7.0, color="#444444",
                 linespacing=1.45, zorder=4)
    axes[2].plot([0.655, 0.972], [0.665, 0.962], color="#999999",
                 linewidth=0.6, zorder=1.6)

    axes[0].set_ylabel("External attack recall")
    fig.text(0.53, 0.215,
             "Normal-window FPR — upper-left operating points preferred",
             ha="center", fontsize=8.5)

    # Shared legend: row 1 the family markers, row 2 the training-arm fills,
    # the FPR band swatch, and the whisker-truncation note. Entries are
    # column-major within ncol.
    def _family_glyph(marker):
        return Line2D([], [], color=INK, marker=marker, linestyle="none",
                      markersize=5.4, markerfacecolor="white",
                      markeredgecolor=INK, markeredgewidth=1.1)

    def _arm_glyph(color, open_face):
        return Line2D([], [], color=color, marker="s", linestyle="none",
                      markersize=5.4,
                      markerfacecolor="white" if open_face else color,
                      markeredgecolor=color if open_face else "black",
                      markeredgewidth=1.2 if open_face else 0.4)

    header = Line2D([], [], linestyle="none", marker=None)
    handles = [
        header, header,
        _family_glyph("o"), _arm_glyph(GRAY, True),
        _family_glyph("s"), _arm_glyph(BLUE, False),
        _family_glyph("^"), _arm_glyph(VERMILLION, False),
        _family_glyph("D"), Patch(facecolor=SOFT_RED, edgecolor="none"),
    ]
    legend_labels = [
        "Family:", "Training arm:",
        "1D-CNN", "Real only",
        "RF", "Rule +30%",
        "BiLSTM", "Rule +100%",
        "Transf. (expl.)", "FPR ≥ 0.95",
    ]
    if ae_settings:
        handles.extend([_family_glyph("v"), header])
        legend_labels.extend([
            "Conv-AE (both thresholds)",
            "SD whiskers truncated at [0, 1]",
        ])
    else:
        handles.extend([header, header])
        legend_labels.extend([
            "",
            "SD whiskers truncated at [0, 1]",
        ])
    legend = fig.legend(
        handles, legend_labels, loc="lower center",
        bbox_to_anchor=(0.53, -0.005), ncol=6, frameon=False, fontsize=6.9,
        handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
        labelspacing=0.5,
    )
    texts = legend.get_texts()
    for text in texts[:2]:
        text.set_fontweight("bold")
    texts[-1].set_fontstyle("italic")
    texts[-1].set_color("#666666")

    lc.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(lc.FIGURES / f"external_operating_points.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    combined = load_combined()
    make_figure(combined)
    print(f"wrote {lc.FIGURES / 'external_operating_points.png'}")
    print(f"wrote {lc.FIGURES / 'external_operating_points.pdf'}")


if __name__ == "__main__":
    main()
