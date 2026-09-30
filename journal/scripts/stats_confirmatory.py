#!/usr/bin/env python3
"""Backlog #10 (critique): focal-comparison statistics, stage 1.

Six focal comparisons — each restates a hypothesis that was either specified
in a version-controlled plan before its corresponding experiment ran (P2 in
the B-plan §7.2, question Q1 in the C-plan) or established in the WISA
submission and re-confirmed here at 5 seeds. The six-comparison family itself
was assembled before this multiplicity sensitivity was computed; it was not
independently preregistered as one joint family. Everything else in the paper
stays descriptive (declared policy). K2 and K5 use the protocol-valid WGAN reruns. If the legacy
protocol-invalid tables remain available, they are reported under separate
``legacy_sensitivity`` labels and are not substituted for the focal rows.
For each comparison this script reports:
  - per-seed paired deltas, mean±std, sign consistency ("n/5 seeds same sign")
  - a seed-level Student-t interval for the paired mean delta (PRIMARY)
  - Wilcoxon signed-rank p (n=5 -> min two-sided p 0.0625; SUPPORTING only)
  - per-arm worst-case Wilson 95% half-width over the seed cells (shows
    window-level binomial uncertainty is negligible next to seed variance)

Output: results/tables/confirmatory_comparisons.csv
"""
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon

import lib_common as lc

# (key, table, family, rung, subset, metric, count_cols, arm_a, arm_b,
#  hypothesis, analysis_role, protocol_status, required)
COMPARISONS = [
    ("K1_cnn_L1_rule_gain", "generator_extension_by_seed.csv", "cnn",
     "L1_fixed_variant", "all", "attack_recall", ("tp", "fn"),
     "rule_0p30", "real_only", "WISA-submission C1 replication at 5 seeds: rule arm lifts L1",
     "focal_primary", "not_applicable", True),
    ("K2_cnn_L1_ganvalid_shift", "generator_extension_by_seed_gan_protocol_valid.csv", "cnn",
     "L1_fixed_variant", "all", "attack_recall", ("tp", "fn"),
     "ganvalid_1p00", "real_only",
     "P2 diagnostic: estimate the protocol-valid WGAN L1 shift; no equivalence margin was pre-specified",
     "focal_primary", "protocol_valid", True),
    ("K3_lstm_L1_rule_gain", "family_extension_by_seed.csv", "lstm",
     "L1_fixed_variant", "all", "attack_recall", ("tp", "fn"),
     "rule_0p30", "real_only", "family generality of C1 (BiLSTM)",
     "focal_primary", "not_applicable", True),
    ("K4_rf_otids_fpr_rule_push", "rf_generator_extension_by_seed.csv", "rf",
     "L5_otids", "all", "fpr", ("fp", "tn"),
     "rule_1p00", "real_only", "augmentation pushes non-saturated external FPR up (rule)",
     "focal_primary", "not_applicable", True),
    ("K5_rf_otids_fpr_ganvalid_push", "rf_generator_extension_by_seed_gan_protocol_valid.csv", "rf",
     "L5_otids", "all", "fpr", ("fp", "tn"),
     "ganvalid_1p00", "real_only", "protocol-valid WGAN arm shifts external OTIDS FPR",
     "focal_primary", "protocol_valid", True),
    ("K6_ss_rf_L1_rule_gain", "second_source_rf_by_seed.csv", "rf",
     "C_L1_fixed_variant", "all", "attack_recall", ("tp", "fn"),
     "rule_0p30", "real_only", "C-plan Q1: C1 reproduces from the second source",
     "focal_primary", "not_applicable", True),
    ("S_K2_cnn_L1_legacy_gan_sensitivity", "generator_extension_by_seed.csv", "cnn",
     "L1_fixed_variant", "all", "attack_recall", ("tp", "fn"),
     "gan_1p00", "real_only",
     "legacy protocol-invalid WGAN sensitivity retained only to quantify the repair consequence",
     "legacy_sensitivity", "legacy_protocol_invalid", False),
    ("S_K5_rf_otids_fpr_legacy_gan_sensitivity", "rf_generator_extension_by_seed.csv", "rf",
     "L5_otids", "all", "fpr", ("fp", "tn"),
     "gan_1p00", "real_only",
     "legacy protocol-invalid WGAN sensitivity retained only to quantify the repair consequence",
     "legacy_sensitivity", "legacy_protocol_invalid", False),
]


def wilson_half_width(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    hw = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    _ = center
    return hw


def seed_mean_t_ci(values, alpha=0.05):
    """Two-sided CI over independent training-seed paired effects."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    critical = student_t.ppf(1 - alpha / 2, df=len(values) - 1)
    half = critical * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def cell_values(df, family, rung, subset, metric, count_cols, setting):
    sub = df[(df.get("family", pd.Series("rf", index=df.index)) == family)
             & (df["rung"] == rung) & (df["subset"] == subset)
             & (df["setting"] == setting)].sort_values("seed")
    if sub["seed"].duplicated().any():
        dup = sorted(sub.loc[sub["seed"].duplicated(keep=False), "seed"].unique())
        raise ValueError(f"duplicate cell rows for {family}/{rung}/{subset}/{setting}: seeds={dup}")
    vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy()
    seeds = sub["seed"].to_numpy()
    pos, neg = count_cols
    ns = (pd.to_numeric(sub[pos], errors="coerce") + pd.to_numeric(sub[neg], errors="coerce")).to_numpy()
    ks = pd.to_numeric(sub[pos], errors="coerce").to_numpy()
    hws = [wilson_half_width(k, n) for k, n in zip(ks, ns)]
    return seeds, vals, max(hws) if hws else float("nan")


def main():
    rows = []
    table_cache = {}
    expected_seeds = np.asarray(sorted(lc.SEEDS))
    # Resolve every required source before computing any row, so an in-flight
    # repaired-generator run cannot leave a misleading partial console report
    # or replace the prior canonical CSV.
    missing_required = sorted({
        table for (_key, table, _family, _rung, _subset, _metric,
                   _count_cols, _arm_a, _arm_b, _hyp, _analysis_role,
                   _protocol_status, required) in COMPARISONS
        if required and not (lc.TABLES / table).exists()
    })
    if missing_required:
        raise FileNotFoundError(
            "required focal source(s) are not complete/published: "
            + ", ".join(str(lc.TABLES / table) for table in missing_required)
            + "; canonical output was not touched")
    for table in {spec[1] for spec in COMPARISONS if (lc.TABLES / spec[1]).exists()}:
        table_cache[table] = pd.read_csv(lc.TABLES / table)
    for (key, table, family, rung, subset, metric, count_cols, arm_a, arm_b,
         _hyp, _analysis_role, _protocol_status, required) in COMPARISONS:
        if not required:
            continue
        df = table_cache[table]
        seeds_a, vals_a, _ = cell_values(
            df, family, rung, subset, metric, count_cols, arm_a)
        seeds_b, vals_b, _ = cell_values(
            df, family, rung, subset, metric, count_cols, arm_b)
        complete = (len(vals_a) == len(expected_seeds)
                    and len(vals_b) == len(expected_seeds)
                    and np.array_equal(np.sort(seeds_a), expected_seeds)
                    and np.array_equal(np.sort(seeds_b), expected_seeds)
                    and np.array_equal(seeds_a, seeds_b)
                    and np.isfinite(vals_a).all() and np.isfinite(vals_b).all())
        if not complete:
            raise RuntimeError(
                f"{key}: required focal source is incomplete; expected exactly one "
                f"finite cell for seeds {expected_seeds.tolist()} in both arms; "
                f"got A={seeds_a.tolist()} B={seeds_b.tolist()}; canonical output was not touched")
    for (key, table, family, rung, subset, metric, count_cols, arm_a, arm_b,
         hyp, analysis_role, protocol_status, required) in COMPARISONS:
        path = lc.TABLES / table
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"required focal source is not complete/published: {path}; "
                    "canonical output was not touched")
            print(f"SKIP optional sensitivity {key}: missing {path}", flush=True)
            continue
        df = table_cache[table]
        seeds_a, vals_a, hw_a = cell_values(df, family, rung, subset, metric, count_cols, arm_a)
        seeds_b, vals_b, hw_b = cell_values(df, family, rung, subset, metric, count_cols, arm_b)
        complete = (len(vals_a) == len(expected_seeds)
                    and len(vals_b) == len(expected_seeds)
                    and np.array_equal(np.sort(seeds_a), expected_seeds)
                    and np.array_equal(np.sort(seeds_b), expected_seeds)
                    and np.array_equal(seeds_a, seeds_b)
                    and np.isfinite(vals_a).all() and np.isfinite(vals_b).all())
        if not complete:
            message = (f"{key}: expected exactly one finite cell for seeds "
                       f"{expected_seeds.tolist()} in both arms; got A={seeds_a.tolist()} "
                       f"B={seeds_b.tolist()}")
            if required:
                raise RuntimeError(message + "; canonical output was not touched")
            print(f"SKIP optional sensitivity {message}", flush=True)
            continue
        deltas = vals_a - vals_b
        same_sign = int(max((deltas > 0).sum(), (deltas < 0).sum(), (deltas == 0).sum()))
        try:
            stat, p = wilcoxon(vals_a, vals_b, zero_method="wilcox")
        except ValueError:  # all deltas zero
            stat, p = 0.0, 1.0
        seed_ci_lo, seed_ci_hi = seed_mean_t_ci(deltas)
        if seed_ci_lo > 0:
            seed_decision = "positive_effect_supported"
        elif seed_ci_hi < 0:
            seed_decision = "negative_effect_supported"
        else:
            seed_decision = "inconclusive_not_equivalent"
        rows.append({
            "comparison": key, "hypothesis": hyp, "analysis_role": analysis_role,
            "protocol_status": protocol_status, "table": table,
            "family": family, "rung": rung, "metric": metric,
            "arm_a": arm_a, "arm_b": arm_b, "n_seeds": len(deltas),
            "mean_a": float(vals_a.mean()), "std_a": float(vals_a.std(ddof=1)),
            "mean_b": float(vals_b.mean()), "std_b": float(vals_b.std(ddof=1)),
            "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std(ddof=1)),
            "delta_min": float(deltas.min()), "delta_max": float(deltas.max()),
            "primary_seed_t_ci_lo": seed_ci_lo,
            "primary_seed_t_ci_hi": seed_ci_hi,
            "primary_decision": seed_decision,
            "equivalence_margin": "not_pre_specified",
            "sign_consistency": f"{same_sign}/{len(deltas)}",
            "wilcoxon_p": float(p),
            "max_wilson_halfwidth_a": hw_a, "max_wilson_halfwidth_b": hw_b,
        })
        print(f"{key}: delta={deltas.mean():+.4f}±{deltas.std(ddof=1):.4f} "
              f"seed95%CI=[{seed_ci_lo:+.4f},{seed_ci_hi:+.4f}] "
              f"decision={seed_decision} sign={same_sign}/{len(deltas)} p={p:.4f} "
              f"wilson_hw=({hw_a:.5f},{hw_b:.5f})", flush=True)

    lc.write_csv(lc.TABLES / "confirmatory_comparisons.csv", rows)
    lc.LOGS.mkdir(parents=True, exist_ok=True)
    (lc.LOGS / "stats_confirmatory.log").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
