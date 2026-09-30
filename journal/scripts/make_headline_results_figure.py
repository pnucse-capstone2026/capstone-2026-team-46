#!/usr/bin/env python3
"""Build the two-panel headline evidence figure for the journal manuscript.

The figure deliberately uses one common, five-seed 1D-CNN family so that the
controlled rungs can be read as one audit rather than as a cross-family
comparison.  It visualizes:

* L1: real-only, the equal-budget real-attack duplication control, Rule +30%,
  and the marginal-matched placebo;
* L2: low/medium/high coupled severity tiers under real-only, Rule +30%, and
  Rule +100%.

The L4 binary-versus-exact separation has its own family-level figure
(make_grammar_exit_figure.py).  The L5 external cells all sit at or near the
FPR ceiling, so they are reported in prose and the claim ledger and shown
graphically by the attribution and sensitivity figures; this script still
loads and ledgers them (panel field ``text``) as the source for those prose
numbers.

Inputs (read-only):
  journal/results/tables/generator_extension_summary_e10a_placebo.csv
  journal/results/tables/generator_extension_summary.csv

Outputs:
  journal/results/tables/headline_shift_ladder_results.csv
  journal/results/figures/headline_shift_ladder_results.(png|pdf)

No external dataset is used for fitting or model selection by this script; it
only renders already-aggregated evaluation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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


# Okabe-Ito colors, with hatch/marker redundancy for grayscale and CVD use.
# Gray is the deliberate neutral for the real-content control arms; identity
# is always carried redundantly by row labels, markers, and line styles.
GRAY = "#666666"
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
INK = "#222222"
GRID = "#D9D9D9"

PLACEBO_SOURCE = "generator_extension_summary_e10a_placebo.csv"
LADDER_SOURCE = "generator_extension_summary.csv"


@dataclass(frozen=True)
class Value:
    mean: float
    std: float
    seeds: str


def _require_columns(df: pd.DataFrame, source: str) -> None:
    required = {
        "family",
        "setting",
        "rung",
        "subset",
        "seeds",
        "attack_recall_mean",
        "attack_recall_std",
        "fpr_mean",
        "fpr_std",
        "exact_recall_mean",
        "exact_recall_std",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")


def _row(
    df: pd.DataFrame,
    *,
    source: str,
    setting: str,
    rung: str,
    subset: str = "all",
) -> pd.Series:
    match = df[
        (df["family"] == "cnn")
        & (df["setting"] == setting)
        & (df["rung"] == rung)
        & (df["subset"] == subset)
    ]
    if len(match) != 1:
        raise ValueError(
            f"{source}: expected one CNN row for "
            f"setting={setting}, rung={rung}, subset={subset}; found {len(match)}"
        )
    return match.iloc[0]


def _value(row: pd.Series, metric: str) -> Value:
    mean = float(row[f"{metric}_mean"])
    std_raw = row.get(f"{metric}_std", 0.0)
    std = 0.0 if pd.isna(std_raw) else float(std_raw)
    if not (0.0 <= mean <= 1.0 and 0.0 <= std <= 1.0):
        raise ValueError(f"invalid {metric} summary: mean={mean}, std={std}")
    return Value(mean=mean, std=std, seeds=str(row["seeds"]))


def _source_record(
    *,
    panel: str,
    rung: str,
    arm: str,
    subset: str,
    dataset: str,
    primary_metric: str,
    primary: Value,
    source_file: str,
    secondary_metric: str = "",
    secondary: Value | None = None,
) -> dict[str, object]:
    return {
        "panel": panel,
        "rung": rung,
        "arm": arm,
        "subset": subset,
        "dataset": dataset,
        "primary_metric": primary_metric,
        "primary_mean": primary.mean,
        "primary_std": primary.std,
        "secondary_metric": secondary_metric,
        "secondary_mean": "" if secondary is None else secondary.mean,
        "secondary_std": "" if secondary is None else secondary.std,
        "seeds": primary.seeds,
        "source_file": source_file,
    }


def load_headline_data() -> tuple[dict[str, object], list[dict[str, object]]]:
    placebo = pd.read_csv(lc.TABLES / PLACEBO_SOURCE)
    ladder = pd.read_csv(lc.TABLES / LADDER_SOURCE)
    _require_columns(placebo, PLACEBO_SOURCE)
    _require_columns(ladder, LADDER_SOURCE)

    records: list[dict[str, object]] = []

    l1 = {}
    for setting, label in [
        ("real_only", "Real only"),
        ("over_0p30", "Real dup. +30%"),
        ("rule_0p30", "Rule +30%"),
        ("placebo_0p30", "Placebo +30%"),
    ]:
        row = _row(
            placebo,
            source=PLACEBO_SOURCE,
            setting=setting,
            rung="L1_fixed_variant",
        )
        value = _value(row, "attack_recall")
        l1[label] = value
        records.append(
            _source_record(
                panel="a",
                rung="L1_fixed_variant",
                arm=label,
                subset="all",
                dataset="Car-Hacking fixed variant",
                primary_metric="attack_recall",
                primary=value,
                source_file=PLACEBO_SOURCE,
            )
        )

    l2 = {}
    for setting, label in [
        ("real_only", "Real only"),
        ("rule_0p30", "Rule +30%"),
        ("rule_1p00", "Rule +100%"),
    ]:
        l2[label] = {}
        for severity in ("low", "medium", "high"):
            row = _row(
                ladder,
                source=LADDER_SOURCE,
                setting=setting,
                rung="L2_sensitivity",
                subset=f"severity:{severity}",
            )
            value = _value(row, "attack_recall")
            l2[label][severity] = value
            records.append(
                _source_record(
                    panel="b",
                    rung="L2_sensitivity",
                    arm=label,
                    subset=f"severity:{severity}",
                    dataset="Car-Hacking sensitivity sweep",
                    primary_metric="attack_recall",
                    primary=value,
                    source_file=LADDER_SOURCE,
                )
            )

    l5 = {}
    for setting, label in [
        ("real_only", "Real only"),
        ("rule_0p30", "Rule +30%"),
        ("rule_1p00", "Rule +100%"),
    ]:
        l5[label] = {}
        for rung, dataset in [
            ("L5_otids", "OTIDS"),
            ("L5_cantt", "can-train"),
            ("L5_road", "ROAD"),
        ]:
            row = _row(
                ladder,
                source=LADDER_SOURCE,
                setting=setting,
                rung=rung,
            )
            fpr = _value(row, "fpr")
            recall = _value(row, "attack_recall")
            l5[label][dataset] = {"fpr": fpr, "recall": recall}
            records.append(
                _source_record(
                    panel="text",
                    rung="L5_external",
                    arm=label,
                    subset="all",
                    dataset=dataset,
                    primary_metric="fpr",
                    primary=fpr,
                    secondary_metric="attack_recall",
                    secondary=recall,
                    source_file=LADDER_SOURCE,
                )
            )

    seed_sets = {record["seeds"] for record in records}
    if seed_sets != {"7;42;123;2026;3407"}:
        raise ValueError(f"headline figure requires the common five-seed set; found {seed_sets}")

    return {"l1": l1, "l2": l2, "l5": l5}, records


def load_focal_k1() -> tuple[float, float, float]:
    """Load the K1 paired delta and CI from the focal ledger (deduplicated)."""
    ledger = pd.read_csv(lc.TABLES / "focal_forest_ledger.csv")
    rows = ledger[ledger["row_id"] == "K1_cnn_L1_rule_gain"].drop_duplicates()
    if len(rows) != 1:
        raise ValueError(f"expected one deduplicated K1 ledger row, found {len(rows)}")
    row = rows.iloc[0]
    return float(row["delta"]), float(row["ci_lo"]), float(row["ci_hi"])


def _style_axis(ax: plt.Axes, *, ylabel: str = "Recall") -> None:
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(6.4, 3.15))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.94, 1.06], wspace=0.36)
    top_axes = [fig.add_subplot(outer[0, 0]), fig.add_subplot(outer[0, 1])]
    fig.subplots_adjust(
        left=0.185,
        right=0.985,
        top=0.908,
        bottom=0.30,
    )

    # (a) L1: real duplication does not close the gap; rule and the
    # marginal-matched placebo both do.
    ax = top_axes[0]
    arm_styles = [
        ("Real only", GRAY, "o", True),
        ("Real dup. +30%", GRAY, "v", True),
        ("Rule +30%", BLUE, "s", False),
        ("Placebo +30%", ORANGE, "D", False),
    ]
    labels = [label for label, _c, _m, _o in arm_styles]
    y = np.arange(len(labels))[::-1]
    for ypos, (label, color, marker, open_face) in zip(y, arm_styles):
        value = data["l1"][label].mean
        error = data["l1"][label].std
        ax.errorbar(
            value,
            ypos,
            xerr=error,
            fmt=marker,
            markersize=6.4,
            markerfacecolor="white" if open_face else color,
            markeredgecolor=color,
            markeredgewidth=1.2,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.5,
            zorder=3,
        )
        ax.text(min(value + 0.03, 1.025), ypos, f"{value:.3f}", va="center",
                ha="left", fontsize=7.2, color=INK)
    # Focal K1 annotation: the paired rule-minus-real delta with its 95% CI,
    # drawn as the distance from the real-only mean to the rule endpoint.
    k1_delta, k1_lo, k1_hi = load_focal_k1()
    real_mean = data["l1"]["Real only"].mean
    rule_mean = data["l1"]["Rule +30%"].mean
    if abs(k1_delta - (rule_mean - real_mean)) > 0.005:
        raise ValueError("focal K1 ledger delta disagrees with plotted arm means")
    y_rule = float(y[labels.index("Rule +30%")])
    y_real = float(y[labels.index("Real only")])
    y_arrow = y_rule + 0.38
    ax.plot([real_mean, real_mean], [y_arrow, y_real], color=GRAY,
            linewidth=0.8, linestyle=":", zorder=2)
    ax.annotate(
        "",
        xy=(rule_mean, y_arrow),
        xytext=(real_mean, y_arrow),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=1.1,
                        mutation_scale=9),
        zorder=2,
    )
    ax.text((real_mean + rule_mean) / 2, y_arrow + 0.14,
            f"focal K1: +{k1_delta:.3f} [{k1_lo:.3f}, {k1_hi:.3f}]",
            ha="center", va="bottom", fontsize=6.7, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.35, 1.06)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("Attack recall")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(a) L1 fixed variant", loc="left", fontsize=9.4,
                 fontweight="bold", pad=8)

    # (b) L2: coupled multi-axis sweep; both doses lift the aggregate response
    # while the low tier remains difficult.
    ax = top_axes[1]
    severities = ["low", "medium", "high"]
    x = np.arange(len(severities))
    # open_face mirrors panel (a) and the shared legend below: real-content
    # arms are hollow markers, synthetic-content arms are filled.
    l2_series = [
        ("Real only", GRAY, "o", "--", True),
        ("Rule +30%", BLUE, "s", "-", False),
        ("Rule +100%", VERMILLION, "^", "-", False),
    ]
    l2_values = {}
    for label, color, marker, linestyle, open_face in l2_series:
        values = [data["l2"][label][severity].mean for severity in severities]
        errors = [data["l2"][label][severity].std for severity in severities]
        l2_values[label] = values
        ax.errorbar(
            x,
            values,
            yerr=errors,
            label=label,
            color=color,
            marker=marker,
            markerfacecolor="white" if open_face else color,
            markeredgecolor=color,
            markeredgewidth=1.2,
            linestyle=linestyle,
            linewidth=1.6,
            markersize=5,
            capsize=2.5,
        )
    # Direct value labels, decluttered per severity tier: labels are stacked
    # in value order with a minimum on-page gap so close-together arms (e.g.
    # Rule +30% and Rule +100% at the medium tier) do not overlap. A tier
    # where every arm has converged (e.g. all reach 1.000 at the high tier)
    # collapses to one shared annotation instead of three coincident labels.
    MIN_LABEL_GAP_PT = 9.5
    for tier_index, severity in enumerate(severities):
        # The low tier sits right against the y-axis, so its labels need
        # extra horizontal clearance to avoid crowding the axis and markers.
        label_dx = -9 if tier_index == 0 else -7
        tier_points = sorted(
            ((l2_values[label][tier_index], color, label)
             for label, color, _marker, _ls, _open in l2_series),
            key=lambda item: item[0],
        )
        tier_values = [value for value, _color, _label in tier_points]
        if max(tier_values) - min(tier_values) < 0.004:
            ax.annotate(
                f"all arms = {tier_values[-1]:.3f}",
                xy=(tier_index, tier_values[-1]), xycoords="data",
                xytext=(label_dx, -9), textcoords="offset points",
                ha="right", va="top", fontsize=6.7, color=INK,
                fontweight="bold",
            )
            continue
        offsets = [0.0] * len(tier_points)
        for i in range(1, len(tier_points)):
            offsets[i] = max(offsets[i], offsets[i - 1] + MIN_LABEL_GAP_PT)
        center = offsets[len(offsets) // 2]
        offsets = [o - center for o in offsets]
        for (value, color, _label), dy in zip(tier_points, offsets):
            ax.annotate(
                f"{value:.3f}",
                xy=(tier_index, value), xycoords="data",
                xytext=(label_dx, dy), textcoords="offset points",
                ha="right", va="center", fontsize=6.7, color=color,
                fontweight="bold",
            )
    ax.set_xticks(x, [s.title() for s in severities], fontsize=8)
    # Extra left padding gives the low-tier value labels room between the
    # y-axis spine and the markers (see label_dx above).
    ax.set_xlim(-0.55, 2.3)
    ax.set_xlabel("Coupled severity tier", fontsize=8.2)
    _style_axis(ax)
    ax.set_title("(b) L2 coupled sweep", loc="left", fontsize=9.4,
                 fontweight="bold", pad=8)

    # Shared figure-level legend: row 1 lists the real-content arms (open gray
    # markers), row 2 the synthetic-content arms (filled colored markers), so
    # the arm encoding used by both panels (and later figures) is stated in
    # the figure itself. Entries are column-major within ncol.
    def _glyph(color, marker, open_face, linestyle="none"):
        return Line2D(
            [], [], color=color, marker=marker, linestyle=linestyle,
            linewidth=1.4, markersize=5.6,
            markerfacecolor="white" if open_face else color,
            markeredgecolor=color, markeredgewidth=1.2,
        )

    header = Line2D([], [], linestyle="none", marker=None)
    handles = [
        header, header,
        _glyph(GRAY, "o", True, "--"), _glyph(BLUE, "s", False, "-"),
        _glyph(GRAY, "v", True), _glyph(ORANGE, "D", False),
        header, _glyph(VERMILLION, "^", False, "-"),
    ]
    legend_labels = [
        "Real-content arms (open):", "Synthetic-content arms (filled):",
        "Real only", "Rule +30%",
        "Real dup. +30%", "Placebo +30%",
        "", "Rule +100%",
    ]
    legend = fig.legend(
        handles, legend_labels, loc="lower center", bbox_to_anchor=(0.53, 0.0),
        ncol=4, frameon=False, fontsize=7.2, handlelength=1.5,
        handletextpad=0.45, columnspacing=1.25, labelspacing=0.45,
    )
    for text in legend.get_texts()[:2]:
        text.set_fontweight("bold")

    lc.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            lc.FIGURES / f"headline_shift_ladder_results.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    data, records = load_headline_data()
    lc.TABLES.mkdir(parents=True, exist_ok=True)
    source_table = lc.TABLES / "headline_shift_ladder_results.csv"
    pd.DataFrame.from_records(records).to_csv(source_table, index=False, lineterminator="\n")
    make_figure(data)
    print(f"wrote {source_table}")
    print(f"wrote {lc.FIGURES / 'headline_shift_ladder_results.png'}")
    print(f"wrote {lc.FIGURES / 'headline_shift_ladder_results.pdf'}")


if __name__ == "__main__":
    main()
