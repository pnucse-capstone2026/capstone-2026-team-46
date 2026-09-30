#!/usr/bin/env python3
"""Build the L4 grammar-exit figure for the journal manuscript.

One panel shows, per supervised detector family, the augmented arm's binary
attack recall next to its pooled exact-micro recall on the L4
payload-position exit.  The binary/exact gap is the RQ2 grammar boundary:
binary detection survives the exit while exact attack typing does not.

The values are the same five-seed summaries printed in the family matrix
(Table tab:familyrungs); the autoencoder threshold diagnostic has no
exact-type head and is therefore left to the table.

Inputs (read-only):
  journal/results/tables/generator_extension_summary.csv          (cnn)
  journal/results/tables/rf_generator_extension_summary.csv       (rf)
  journal/results/tables/family_extension_summary.csv             (lstm)
  journal/results/tables/family_extension_transformer_summary.csv (transformer)

Outputs:
  journal/results/tables/grammar_exit_figure.csv
  journal/results/figures/grammar_exit_ladder.(png|pdf)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

BLUE = "#0072B2"
VERMILION = "#D55E00"
GRID = "#D9D9D9"
INK = "#222222"

SEED_SET = "7;42;123;2026;3407"
RUNG = "L4_out_of_generator"
SETTING = "rule_0p30"

# Display order matches Table tab:familyrungs.
FAMILIES = [
    ("cnn", "1D-CNN", "generator_extension_summary.csv"),
    ("rf", "RF", "rf_generator_extension_summary.csv"),
    ("lstm", "BiLSTM", "family_extension_summary.csv"),
    ("transformer", "Transformer (expl.)", "family_extension_transformer_summary.csv"),
]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, label, source in FAMILIES:
        df = pd.read_csv(lc.TABLES / source)
        match = df[
            (df["family"] == family)
            & (df["setting"] == SETTING)
            & (df["rung"] == RUNG)
            & (df["subset"] == "all")
        ]
        if len(match) != 1:
            raise ValueError(
                f"{source}: expected one {family} row for rung={RUNG}, "
                f"setting={SETTING}, subset=all; found {len(match)}"
            )
        row = match.iloc[0]
        if str(row["seeds"]) != SEED_SET:
            raise ValueError(
                f"{source}: {family} L4 row uses seeds {row['seeds']}, "
                f"expected {SEED_SET}"
            )
        rows.append({
            "family": family,
            "label": label,
            "binary_mean": float(row["attack_recall_mean"]),
            "binary_std": float(row["attack_recall_std"]),
            "exact_mean": float(row["exact_recall_mean"]),
            "exact_std": float(row["exact_recall_std"]),
            "seeds": str(row["seeds"]),
            "source_file": source,
        })
    return rows


def make_figure(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.55))
    ys = list(range(len(rows) - 1, -1, -1))
    for y, row in zip(ys, rows):
        ax.plot(
            [row["exact_mean"], row["binary_mean"]],
            [y, y],
            color="#B7B7B7",
            linewidth=1.1,
            zorder=1,
        )
        # Recall is bounded at 0/1, so truncate SD whiskers at the bounds
        # instead of letting them run past the axis.
        binary_err = [[min(row["binary_std"], row["binary_mean"])],
                      [min(row["binary_std"], 1.0 - row["binary_mean"])]]
        exact_err = [[min(row["exact_std"], row["exact_mean"])],
                     [min(row["exact_std"], 1.0 - row["exact_mean"])]]
        ax.errorbar(
            row["binary_mean"], y, xerr=binary_err, fmt="o",
            markersize=6.2, markerfacecolor=BLUE, markeredgecolor=BLUE,
            ecolor=BLUE, elinewidth=0.8, capsize=2.0, zorder=3,
        )
        ax.errorbar(
            row["exact_mean"], y, xerr=exact_err, fmt="s",
            markersize=5.8, markerfacecolor="white", markeredgecolor=VERMILION,
            markeredgewidth=1.3, ecolor=VERMILION, elinewidth=0.8,
            capsize=2.0, zorder=3,
        )
    ax.set_yticks(ys, [row["label"] for row in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.0, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Recall on the L4 payload-position exit (augmented arm)",
                  fontsize=8.6)
    ax.tick_params(labelsize=8.4)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6.2,
               markerfacecolor=BLUE, markeredgecolor=BLUE,
               label="Binary attack recall"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=5.8,
               markerfacecolor="white", markeredgecolor=VERMILION,
               markeredgewidth=1.3, label="Pooled exact-micro recall"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, handletextpad=0.4, columnspacing=1.6,
               bbox_to_anchor=(0.56, 0.0))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.34)

    lc.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            lc.FIGURES / f"grammar_exit_ladder.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    lc.TABLES.mkdir(parents=True, exist_ok=True)
    source_table = lc.TABLES / "grammar_exit_figure.csv"
    pd.DataFrame(rows).to_csv(source_table, index=False, lineterminator="\n")
    make_figure(rows)
    print(f"wrote {source_table}")
    print(f"wrote {lc.FIGURES / 'grammar_exit_ladder.png'}")
    print(f"wrote {lc.FIGURES / 'grammar_exit_ladder.pdf'}")


if __name__ == "__main__":
    main()
