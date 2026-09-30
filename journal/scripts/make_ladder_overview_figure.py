#!/usr/bin/env python3
"""Build the main-text ladder-overview heatmap.

One figure shows, for every graded detector family and training arm, how the
five-seed journal pipeline behaves as evaluation climbs the ladder:

* (a) controlled recall on L1 (fixed variant) and L2 (coupled sweep);
* (b) the compound L4 out-of-rule construction, which contains a
  payload-position exit plus span/step/waveform/jitter changes, using binary
  versus pooled exact-micro recall per family; and
* (c) external normal-window FPR on OTIDS, can-train, and ROAD.

This is the extended main-text version of the supplement family-ladder matrix
(``make_family_extension_figure.py``): it adds the RF rows and the L4
binary/exact columns so that the compound-construction separation and the
external FPR ceiling can be read in one place.  Every cell is the five-seed mean of the
journal pipeline; per-seed values and standard deviations remain in the source
tables and main-text Table `tab:familyrungs`.

Inputs (read-only):
  journal/results/tables/generator_extension_summary.csv           (CNN)
  journal/results/tables/rf_generator_extension_summary.csv        (RF)
  journal/results/tables/rf_generator_extension_external_summary.csv
  journal/results/tables/family_extension_summary.csv              (BiLSTM, AE)
  journal/results/tables/family_extension_transformer_summary.csv

Outputs:
  journal/results/figures/ladder_overview_heatmap.(png|pdf)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

JOURNAL_ROOT = Path(__file__).resolve().parents[1]
TABLES = JOURNAL_ROOT / "results" / "tables"
FIGURES = JOURNAL_ROOT / "results" / "figures"

RECALL_RUNGS = ["L1_fixed_variant", "L2_sensitivity"]
L4_RUNG = "L4_out_of_generator"
EXT_RUNGS = ["L5_otids", "L5_cantt", "L5_road"]

ARM_LABELS = {
    "real_only": "Real only",
    "rule_0p30": "Rule +30%",
    "rule_1p00": "Rule +100%*",
}


def load_combined() -> pd.DataFrame:
    cnn = pd.read_csv(TABLES / "generator_extension_summary.csv")
    cnn = cnn[cnn["setting"].isin(ARM_LABELS)]

    rf = pd.read_csv(TABLES / "rf_generator_extension_summary.csv")
    rf = rf[rf["setting"].isin(ARM_LABELS)]
    rf_ext = pd.read_csv(TABLES / "rf_generator_extension_external_summary.csv")
    rf_ext = rf_ext[rf_ext["setting"].isin(ARM_LABELS)]
    if "family" not in rf_ext.columns:
        rf_ext = rf_ext.assign(family="rf")

    fam = pd.read_csv(TABLES / "family_extension_summary.csv")
    tf_path = TABLES / "family_extension_transformer_summary.csv"
    if tf_path.exists():
        fam = pd.concat([fam, pd.read_csv(tf_path)], ignore_index=True)
    # AE rows encode the threshold policy in ``setting``; give them their own
    # family keys so they render as separate diagnostic rows.
    fam.loc[fam["family"] == "ae", "family"] = fam.loc[
        fam["family"] == "ae", "setting"
    ]

    keep = ["family", "setting", "rung", "subset", "attack_recall_mean",
            "fpr_mean", "exact_recall_mean", "seeds"]
    frames = []
    for df in (cnn, rf, rf_ext, fam):
        frames.append(df[[c for c in keep if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def build_series(*, include_ae: bool = True) -> list[tuple[str, str, str]]:
    series: list[tuple[str, str, str]] = []
    for fam, fam_label in [("cnn", "1D-CNN"), ("rf", "RF"), ("lstm", "BiLSTM"),
                           ("transformer", "Transf. (expl.)")]:
        for setting, arm_label in ARM_LABELS.items():
            series.append((fam, setting, f"{fam_label} · {arm_label}"))
    if include_ae:
        series.append(("ae_percentile", "ae_percentile", "Conv-AE · percentile thr."))
        series.append(("ae_synthetic_calibrated", "ae_synthetic_calibrated",
                       "Conv-AE · synth.-calib. thr."))
    return series


def cell(combined: pd.DataFrame, fam: str, setting: str, rung: str,
         metric: str) -> float:
    sub = combined[
        (combined["family"] == fam)
        & (combined["setting"] == setting)
        & (combined["rung"] == rung)
        & (combined["subset"] == "all")
    ]
    if sub.empty:
        return np.nan
    values = pd.to_numeric(sub[metric], errors="coerce").dropna()
    if values.empty:
        return np.nan
    if len(values.unique()) > 1:
        raise ValueError(
            f"ambiguous cell {fam}/{setting}/{rung}/{metric}: {values.tolist()}"
        )
    return float(values.iloc[0])


def make_figure(
    combined: pd.DataFrame,
    *,
    include_ae: bool = True,
    footer: str = (
        "Cell text is the five-pipeline-seed mean; "
        "* nominal +100% uses the pool-reuse sampler."
    ),
) -> None:
    series = build_series(include_ae=include_ae)

    controlled = np.array([
        [cell(combined, fam, setting, rung, "attack_recall_mean")
         for rung in RECALL_RUNGS]
        for fam, setting, _ in series
    ])
    grammar = np.array([
        [cell(combined, fam, setting, L4_RUNG, "attack_recall_mean"),
         cell(combined, fam, setting, L4_RUNG, "exact_recall_mean")]
        for fam, setting, _ in series
    ])
    external = np.array([
        [cell(combined, fam, setting, rung, "fpr_mean") for rung in EXT_RUNGS]
        for fam, setting, _ in series
    ])

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 4.45),
                             gridspec_kw={"width_ratios": [2, 2, 3]})
    panels = [
        (axes[0], controlled, ["L1 fixed\nvariant", "L2 coupled\nsweep"],
         "Blues", "(a) Controlled recall\n(higher is better)"),
        (axes[1], grammar, ["binary", "exact-\nmicro"],
         "Blues", "(b) Compound L4\nconstruction recall"),
        (axes[2], external, ["OTIDS", "can-train", "ROAD"],
         "OrRd", "(c) External FPR\n(lower is better)"),
    ]
    for ax, values, columns, cmap_name, title in panels:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#EFEFEF")
        ax.imshow(np.ma.masked_invalid(values), vmin=0.0, vmax=1.0, cmap=cmap,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(columns)), columns)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelsize=7.4, length=0, pad=4)
        ax.set_title(title, fontsize=8.8, loc="left", fontweight="bold", pad=11)
        ax.set_yticks(range(len(series)))
        ax.tick_params(axis="y", length=0)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isnan(value):
                    text, color = "—", "#777777"
                else:
                    text = f"{value:.3f}"
                    color = "white" if value >= 0.68 else "#222222"
                ax.text(col, row, text, ha="center", va="center",
                        fontsize=6.8, color=color,
                        fontweight="bold" if value >= 0.95 else "normal")
        for boundary in [2.5, 5.5, 8.5, 11.5]:
            ax.axhline(boundary, color="white", linewidth=2.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticklabels([label for _f, _s, label in series], fontsize=6.9)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    fig.text(0.63, 0.022, footer, ha="center", fontsize=6.8,
             color="#555555")
    fig.subplots_adjust(left=0.245, right=0.99, top=0.865, bottom=0.065,
                        wspace=0.08)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"ladder_overview_heatmap.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    combined = load_combined()
    make_figure(combined)
    print(f"wrote {FIGURES / 'ladder_overview_heatmap.png'}")
    print(f"wrote {FIGURES / 'ladder_overview_heatmap.pdf'}")


if __name__ == "__main__":
    main()
