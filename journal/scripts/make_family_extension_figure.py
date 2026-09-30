#!/usr/bin/env python3
"""Task A3: aggregate family-extension results against frozen WISA CNN references,
emit the key-comparisons table and the two-panel family-ladder figure.

Inputs (read-only): journal/results/tables/family_extension_summary.csv,
frozen WISA tables under wisa/results/tables/.
Outputs: journal/results/tables/family_extension_key_comparisons.csv,
journal/results/figures/family_extension_ladder.(png|pdf).  Updating the
curated human manifest requires the explicit ``--update-manifest`` flag.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

FAMILY_LABELS = {
    "cnn": "1D-CNN (conference)",
    "lstm": "BiLSTM (new)",
    "transformer": "Transformer (new)",
    "ae_percentile": "Conv-AE, percentile thr.",
    "ae_synthetic_calibrated": "Conv-AE, synthetic-calibrated thr.",
}


def wisa_cnn_reference():
    """Headline CNN numbers from frozen WISA tables, keyed like journal rows."""
    rows = []
    main = pd.read_csv(lc.WISA_TABLES / "main_results_mean_std.csv")
    gen_map = {("real_only", 0.0): "real_only", ("rule_based", 0.3): "rule_0p30", ("rule_based", 1.0): "rule_1p00"}
    for (gen, ratio), setting in gen_map.items():
        sub = main[(main["generator"] == gen) & (main["ratio"] == ratio)]
        for split, rung in [("test", "L0_source_test"), ("variant_test", "L1_fixed_variant"),
                            ("otids_cross_binary", "L5_otids")]:
            r = sub[sub["split"] == split]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"family": "cnn", "setting": setting, "rung": rung, "subset": "all",
                         "attack_recall_mean": r["attack_recall_mean"], "attack_recall_std": r["attack_recall_std"],
                         "fpr_mean": r["fpr_mean"], "fpr_std": r["fpr_std"],
                         "macro_f1_binary_mean": r["macro_f1_binary_mean"],
                         "macro_f1_binary_std": r["macro_f1_binary_std"], "seeds": r["seeds"]})
    sens = pd.read_csv(lc.WISA_TABLES / "variant_sensitivity_summary.csv")
    for setting in ["real_only", "rule_0p30", "rule_1p00"]:
        r = sens[sens["setting"] == setting]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({"family": "cnn", "setting": setting, "rung": "L2_sensitivity", "subset": "all",
                     "attack_recall_mean": r["attack_detection_recall_mean"],
                     "attack_recall_std": r["attack_detection_recall_std"],
                     "fpr_mean": r["fpr_mean"], "fpr_std": r["fpr_std"],
                     "macro_f1_binary_mean": r["macro_f1_binary_mean"],
                     "macro_f1_binary_std": r["macro_f1_binary_std"], "seeds": r["seeds"]})
    cantt = pd.read_csv(lc.WISA_TABLES / "cantt_external_eval_summary_mean_std.csv")
    for setting in ["real_only", "rule_0p30", "rule_1p00"]:
        r = cantt[cantt["setting"] == setting]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({"family": "cnn", "setting": setting, "rung": "L5_cantt", "subset": "all",
                     "attack_recall_mean": r["attack_recall_mean"], "attack_recall_std": r["attack_recall_std"],
                     "fpr_mean": r["fpr_mean"], "fpr_std": r["fpr_std"],
                     "macro_f1_binary_mean": r["macro_f1_binary_mean"],
                     "macro_f1_binary_std": r["macro_f1_binary_std"], "seeds": r["seeds"]})
    road = pd.read_csv(lc.WISA_TABLES / "road_external_summary.csv")
    road = road[(road["level"] == "all") & (road["axis"] == "all")]
    for setting in ["real_only", "rule_0p30", "rule_1p00"]:
        r = road[road["setting"] == setting]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({"family": "cnn", "setting": setting, "rung": "L5_road", "subset": "all",
                     "attack_recall_mean": r["attack_recall_mean"], "attack_recall_std": r["attack_recall_std"],
                     "fpr_mean": r["fpr_mean"], "fpr_std": r["fpr_std"],
                     "macro_f1_binary_mean": "", "macro_f1_binary_std": "", "seeds": r["seeds"]})
    return pd.DataFrame(rows)


def journal_cnn_rows():
    """Journal five-seed CNN rows (same pipeline and numbers as the main text).

    The figure previously reused the frozen three-seed WISA conference CNN
    numbers under a plain 1D-CNN label, which silently disagreed with the
    main text (e.g., L1 real-only 0.486 vs 0.468).  The heatmap now uses the
    journal extension pipeline for every plotted row; the WISA rows remain in
    the key-comparisons CSV as an explicitly labeled reference.
    """
    df = pd.read_csv(lc.TABLES / "generator_extension_summary.csv")
    df = df[
        (df["family"] == "cnn")
        & (df["subset"] == "all")
        & df["setting"].isin(["real_only", "rule_0p30", "rule_1p00"])
        & df["rung"].isin([
            "L0_source_test", "L1_fixed_variant", "L2_sensitivity",
            "L5_otids", "L5_cantt", "L5_road",
        ])
    ].copy()
    keep = ["family", "setting", "rung", "subset", "attack_recall_mean",
            "attack_recall_std", "fpr_mean", "fpr_std", "exact_recall_mean",
            "seeds"]
    return df[[c for c in keep if c in df.columns]]


def journal_rows():
    df = pd.read_csv(lc.TABLES / "family_extension_summary.csv")
    tf_path = lc.TABLES / "family_extension_transformer_summary.csv"
    if tf_path.exists():
        df = pd.concat([df, pd.read_csv(tf_path)], ignore_index=True)
    df = df[df["subset"] == "all"].copy()
    # AE settings already encode the policy; map family for coloring
    df.loc[df["family"] == "ae", "family"] = df.loc[df["family"] == "ae", "setting"].str.replace("ae_", "ae_", regex=False)
    keep = ["family", "setting", "rung", "subset", "attack_recall_mean", "attack_recall_std",
            "fpr_mean", "fpr_std", "macro_f1_binary_mean", "macro_f1_binary_std", "exact_recall_mean", "seeds"]
    return df[[c for c in keep if c in df.columns]]


def make_figure(
    combined,
    *,
    include_ae=True,
    footer=(
        "Cell text is the five-seed mean; "
        "color is a reading aid, not a second metric."
    ),
):
    gain_rungs = ["L1_fixed_variant", "L2_sensitivity"]
    ext_rungs = ["L5_otids", "L5_cantt", "L5_road"]
    gain_labels = ["L1 fixed\nvariant", "L2 coupled\nsweep"]
    ext_labels = ["OTIDS", "can-train", "ROAD"]

    series = []  # (family, setting, direct row label)
    for fam in ["cnn", "lstm", "transformer"]:
        for setting in ["real_only", "rule_0p30", "rule_1p00"]:
            family = {"cnn": "1D-CNN", "lstm": "BiLSTM", "transformer": "Transformer"}[fam]
            arm = {"real_only": "Real only", "rule_0p30": "Rule +30%", "rule_1p00": "Rule +100%"}[setting]
            series.append((fam, setting, f"{family} · {arm}"))
    if include_ae:
        for fam in ["ae_percentile", "ae_synthetic_calibrated"]:
            series.append((fam, fam, FAMILY_LABELS[fam]))

    def matrix(rungs, metric):
        values = []
        for fam, setting, _label in series:
            sub = combined[(combined["family"] == fam) & (combined["setting"] == setting)]
            row = []
            for rung in rungs:
                r = sub[sub["rung"] == rung]
                row.append(float(r[metric].iloc[0]) if len(r) and pd.notna(r[metric].iloc[0]) else np.nan)
            values.append(row)
        return np.asarray(values, dtype=float)

    controlled = matrix(gain_rungs, "attack_recall_mean")
    external = matrix(ext_rungs, "fpr_mean")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.15),
                             gridspec_kw={"width_ratios": [2, 3]})
    panels = [
        (axes[0], controlled, gain_labels, "Blues", "(a) Controlled recall\n(higher is better)"),
        (axes[1], external, ext_labels, "OrRd", "(b) External FPR\n(lower is better)"),
    ]
    for ax, values, columns, cmap_name, title in panels:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#EFEFEF")
        ax.imshow(values, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(range(len(columns)), columns)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelsize=7.7, length=0, pad=5)
        ax.set_title(title, fontsize=9.0, loc="left", fontweight="bold", pad=12)
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
                        fontsize=7.1, color=color,
                        fontweight="bold" if value >= 0.95 else "normal")
        for boundary in [2.5, 5.5, 8.5]:
            ax.axhline(boundary, color="white", linewidth=2.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticklabels([label for _fam, _setting, label in series], fontsize=7.1)
    axes[1].set_yticklabels([])
    fig.text(0.62, 0.025, footer, ha="center", fontsize=7.0,
             color="#555555")
    fig.subplots_adjust(left=0.34, right=0.985, top=0.86, bottom=0.08, wspace=0.10)
    figdir = lc.FIGURES
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(figdir / f"family_extension_ladder.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_manifest():
    # Append-if-missing: never truncate the shared manifest (other experiments
    # add their own rows).  This function is opt-in from the CLI.
    manifest = lc.ROOT / "results" / "paper_artifacts_manifest.md"
    own_rows = [
        "| `results/tables/family_extension_by_seed.csv` | `scripts/evaluate_family_extension.py` | family-extension models + shared windows |",
        "| `results/tables/family_extension_summary.csv` | `scripts/evaluate_family_extension.py` | by_seed aggregation |",
        "| `results/tables/family_extension_key_comparisons.csv` | `scripts/make_family_extension_figure.py` | summary + frozen WISA tables (read-only) |",
        "| `results/figures/family_extension_ladder.png/.pdf` | `scripts/make_family_extension_figure.py` | key_comparisons |",
    ]
    header = ("# Journal paper artifacts manifest\n\n"
              "| Artifact | Source script | Inputs |\n|---|---|---|\n")
    if manifest.exists():
        text = manifest.read_text()
        missing = [row for row in own_rows if row not in text]
        if missing:
            manifest.write_text(text.rstrip("\n") + "\n" + "\n".join(missing) + "\n")
    else:
        manifest.write_text(header + "\n".join(own_rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="explicitly append missing rows to the curated human manifest",
    )
    args = parser.parse_args()
    wisa_ref = wisa_cnn_reference()
    cnn_journal = journal_cnn_rows()
    ours = journal_rows()
    wisa_ref["pipeline"] = "wisa_conference_reference"
    cnn_journal["pipeline"] = "journal"
    ours["pipeline"] = "journal"
    combined = pd.concat([wisa_ref, cnn_journal, ours], ignore_index=True)
    out = lc.TABLES / "family_extension_key_comparisons.csv"
    combined.to_csv(out, index=False)
    # The plotted matrix uses journal-pipeline rows only, so every heatmap
    # cell matches the main text's five-seed numbers.
    make_figure(pd.concat([cnn_journal, ours], ignore_index=True))

    if args.update_manifest:
        update_manifest()
    print(json.dumps({"rows": len(combined),
                      "outputs": [str(out), "results/figures/family_extension_ladder.png"],
                      "manifest_updated": args.update_manifest}))


if __name__ == "__main__":
    main()
