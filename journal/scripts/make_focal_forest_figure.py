#!/usr/bin/env python3
"""Build the focal-comparison forest figure for the journal manuscript.

One horizontal forest panel shows every paired detector-seed contrast that the
claim ledger (main text Table `tab:claimledger`) reports numerically:

* the six declared focal comparisons K1--K6
  (``confirmatory_comparisons_holm_v1.csv``), and
* the two prospectively specified sensitivities outside that six-test family:
  the marginal-matched rule-minus-placebo L1 contrast (E10a) and the
  frame-disjoint L2 realization contrast (E8 v2).

Every row is a paired five-detector-seed delta with its two-sided Student-$t$
95% interval, exactly as stored in the source tables; nothing is recomputed
here.  Recall deltas and FPR deltas are separated because their reading
direction differs (a positive FPR delta is a cost, not a gain).

Inputs (read-only):
  journal/results/tables/confirmatory_comparisons_holm_v1.csv
  journal/results/tables/generator_extension_paired_e10a_placebo.csv
  journal/results/tables/evaluation_realization_paired_summary_e8_v2.csv

Outputs:
  journal/results/tables/focal_forest_ledger.csv
  journal/results/figures/focal_forest_ladder.(png|pdf)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import lib_common as lc


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Okabe-Ito roles shared with the other manuscript figures: blue for supported
# recall contrasts, vermillion for external-FPR contrasts, gray for
# inconclusive rows.  Marker shape redundantly separates the focal rows
# (filled circle) from the prospective sensitivities (open diamond).
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GRAY = "#666666"
INK = "#222222"
GRID = "#D9D9D9"

HOLM_SOURCE = "confirmatory_comparisons_holm_v1.csv"
PLACEBO_SOURCE = "generator_extension_paired_e10a_placebo.csv"
REALIZATION_SOURCE = "evaluation_realization_paired_summary_e8_v2.csv"


def _holm_row(holm: pd.DataFrame, comparison: str) -> pd.Series:
    match = holm[holm["comparison"] == comparison]
    if len(match) != 1:
        raise ValueError(f"{HOLM_SOURCE}: expected one row for {comparison}")
    return match.iloc[0]


def _placebo_row(placebo: pd.DataFrame) -> pd.Series:
    match = placebo[
        (placebo["rung"] == "L1_fixed_variant")
        & (placebo["subset"] == "all")
        & (placebo["metric"] == "attack_recall")
        & (placebo["arm_a"] == "rule_0p30")
        & (placebo["arm_b"] == "placebo_0p30")
    ]
    if len(match) != 1:
        raise ValueError(f"{PLACEBO_SOURCE}: expected one rule-minus-placebo L1 row")
    return match.iloc[0]


def _realization_row(realization: pd.DataFrame) -> pd.Series:
    match = realization[
        (realization["contrast"] == "rule_1p00_minus_real_only")
        & (realization["rung"] == "L2_sensitivity")
        & (realization["metric"] == "binary_attack_recall")
        & (realization["analysis_level"] == "paired_detector_seed_after_block_mean")
    ]
    if len(match) != 1:
        raise ValueError(f"{REALIZATION_SOURCE}: expected one frame-disjoint L2 row")
    return match.iloc[0]


def load_rows() -> list[dict[str, object]]:
    holm = pd.read_csv(lc.TABLES / HOLM_SOURCE)
    placebo = pd.read_csv(lc.TABLES / PLACEBO_SOURCE)
    realization = pd.read_csv(lc.TABLES / REALIZATION_SOURCE)

    rows: list[dict[str, object]] = []

    def add_focal(comparison: str, label: str, group: str) -> None:
        r = _holm_row(holm, comparison)
        rows.append({
            "row_id": comparison,
            "label": label,
            "group": group,
            "role": "focal",
            "metric": r["metric"],
            "delta": float(r["delta_mean"]),
            "ci_lo": float(r["primary_seed_t_ci_lo"]),
            "ci_hi": float(r["primary_seed_t_ci_hi"]),
            "sign_consistency": r["sign_consistency"],
            "decision": r["primary_decision"],
            "source_file": HOLM_SOURCE,
        })

    # Group 1: controlled attack-recall deltas (positive = gain).
    add_focal("K1_cnn_L1_rule_gain", "K1  CNN L1 · Rule +30%", "recall")
    add_focal("K3_lstm_L1_rule_gain", "K3  BiLSTM L1 · Rule +30%", "recall")
    add_focal("K6_ss_rf_L1_rule_gain",
              "K6  RF L1 · Rule +30% (second source)", "recall")
    add_focal("K2_cnn_L1_ganvalid_shift", "K2  CNN L1 · WGAN +100%", "recall")

    r = _realization_row(realization)
    rows.append({
        "row_id": "S_frame_disjoint_L2",
        "label": "CNN L2 · Rule +100% (frame-disjoint)",
        "group": "recall",
        "role": "sensitivity",
        "metric": "binary_attack_recall",
        "delta": float(r["mean"]),
        "ci_lo": float(r["ci_low"]),
        "ci_hi": float(r["ci_high"]),
        "sign_consistency": "",
        "decision": "prospective_sensitivity",
        "source_file": REALIZATION_SOURCE,
    })

    r = _placebo_row(placebo)
    rows.append({
        "row_id": "S_rule_minus_placebo_L1",
        "label": "CNN L1 · Rule − Placebo",
        "group": "recall",
        "role": "sensitivity",
        "metric": "attack_recall",
        "delta": float(r["delta_mean"]),
        "ci_lo": float(r["primary_seed_t_ci_lo"]),
        "ci_hi": float(r["primary_seed_t_ci_hi"]),
        "sign_consistency": (f"{int(r['positive_seeds'])}+/"
                             f"{int(r['zero_seeds'])}0/"
                             f"{int(r['negative_seeds'])}-"),
        "decision": r["primary_decision"],
        "source_file": PLACEBO_SOURCE,
    })

    # Group 2: external FPR deltas (positive = more false alarms).
    add_focal("K4_rf_otids_fpr_rule_push", "K4  RF OTIDS FPR · Rule +100%", "fpr")
    add_focal("K5_rf_otids_fpr_ganvalid_push",
              "K5  RF OTIDS FPR · WGAN +100%", "fpr")

    return rows


def make_figure(rows: list[dict[str, object]]) -> None:
    recall_rows = [r for r in rows if r["group"] == "recall"]
    fpr_rows = [r for r in rows if r["group"] == "fpr"]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    fig.subplots_adjust(left=0.40, right=0.985, top=0.985, bottom=0.155)

    # Row layout: recall block on top, one spacer row per header, FPR block
    # below.  y decreases downward.
    y_positions: list[float] = []
    y = 0.0
    header_rows: list[tuple[float, str]] = []
    header_rows.append((y, "Paired attack-recall deltas  (right of zero = recall gain)"))
    y -= 1.0
    for r in recall_rows:
        y_positions.append(y)
        y -= 1.0
    y -= 0.35
    header_rows.append((y, "Paired external-FPR deltas  (right of zero = more false alarms)"))
    y -= 1.0
    for r in fpr_rows:
        y_positions.append(y)
        y -= 1.0

    ordered = recall_rows + fpr_rows
    for ypos, r in zip(y_positions, ordered):
        inconclusive = "inconclusive" in str(r["decision"])
        if inconclusive:
            color = GRAY
        elif r["group"] == "fpr":
            color = VERMILLION
        else:
            color = BLUE
        open_marker = r["role"] == "sensitivity"
        ax.plot([r["ci_lo"], r["ci_hi"]], [ypos, ypos], color=color,
                linewidth=1.5, solid_capstyle="butt", zorder=2)
        for x_end in (r["ci_lo"], r["ci_hi"]):
            ax.plot([x_end, x_end], [ypos - 0.16, ypos + 0.16], color=color,
                    linewidth=1.1, zorder=2)
        ax.plot(
            r["delta"], ypos,
            marker="D" if open_marker else "o",
            markersize=5.4 if open_marker else 6.0,
            markerfacecolor="white" if open_marker else color,
            markeredgecolor=color, markeredgewidth=1.3, zorder=3,
        )
        label = f"+{r['delta']:.3f}" if r["delta"] >= 0 else f"−{abs(r['delta']):.3f}"
        if r["row_id"] == "S_rule_minus_placebo_L1":
            label = f"+{r['delta']:.6f}"
        ax.text(
            max(r["ci_hi"], r["delta"]) + 0.022, ypos, label,
            va="center", ha="left", fontsize=7.0, color=INK,
        )

    for ypos, text in header_rows:
        ax.text(-0.700, ypos, text, va="center", ha="left", fontsize=7.6,
                fontweight="bold", color=INK, clip_on=False)

    ax.axvline(0.0, color="#999999", linewidth=0.9, linestyle=":", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in ordered], fontsize=7.6)
    ax.set_ylim(min(y_positions) - 0.7, 0.6)
    ax.set_xlim(-0.12, 1.13)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Paired five-seed delta (augmented − comparison arm), 95% seed-t CI",
                  fontsize=8.2)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    lc.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(lc.FIGURES / f"focal_forest_ladder.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    lc.TABLES.mkdir(parents=True, exist_ok=True)
    ledger = lc.TABLES / "focal_forest_ledger.csv"
    pd.DataFrame.from_records(rows).to_csv(ledger, index=False, lineterminator="\n")
    make_figure(rows)
    print(f"wrote {ledger}")
    print(f"wrote {lc.FIGURES / 'focal_forest_ladder.png'}")
    print(f"wrote {lc.FIGURES / 'focal_forest_ladder.pdf'}")


if __name__ == "__main__":
    main()
