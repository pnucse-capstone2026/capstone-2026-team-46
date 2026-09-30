#!/usr/bin/env python3
"""Multiplicity sensitivity for the declared focal seed-level comparisons.

This script does not redefine the primary estimands or overwrite the existing
confirmatory table.  It reads the six rows already labelled ``focal_primary``
in ``confirmatory_comparisons.csv``, reconstructs the two-sided one-sample
Student-t p-values from their paired-seed summaries, and applies Holm's
step-down family-wise correction as a supporting sensitivity analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "tables" / "confirmatory_comparisons.csv"
DEFAULT_OUTPUT = (
    ROOT / "results" / "tables" / "confirmatory_comparisons_holm_v1.csv"
)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Holm-adjusted p-values in the original row order."""
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or len(p_values) == 0:
        raise ValueError("p_values must be a non-empty one-dimensional array")
    if not np.isfinite(p_values).all() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("p_values must be finite and inside [0, 1]")

    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, float((m - rank) * p_values[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted


def build_table(source: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    focal = source.loc[source["analysis_role"].eq("focal_primary")].copy()
    if focal.empty:
        raise ValueError("input contains no analysis_role=focal_primary rows")

    required = {"comparison", "delta_mean", "delta_std", "n_seeds"}
    missing = required.difference(focal.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    if (focal["n_seeds"] < 2).any() or (focal["delta_std"] <= 0).any():
        raise ValueError("each focal row needs n_seeds >= 2 and delta_std > 0")

    standard_error = focal["delta_std"] / np.sqrt(focal["n_seeds"])
    focal["t_statistic"] = focal["delta_mean"] / standard_error
    focal["two_sided_p"] = 2.0 * student_t.sf(
        np.abs(focal["t_statistic"]), focal["n_seeds"] - 1
    )
    focal["holm_adjusted_p"] = holm_adjust(focal["two_sided_p"].to_numpy())
    focal["holm_reject_alpha_0p05"] = focal["holm_adjusted_p"] < alpha
    focal["multiplicity_family"] = "six_declared_focal_seed_level_comparisons"
    focal["multiplicity_role"] = (
        "supporting sensitivity; original paired seed-t intervals remain primary"
    )

    preferred = [
        "comparison",
        "hypothesis",
        "family",
        "rung",
        "metric",
        "arm_a",
        "arm_b",
        "n_seeds",
        "delta_mean",
        "primary_seed_t_ci_lo",
        "primary_seed_t_ci_hi",
        "sign_consistency",
        "t_statistic",
        "two_sided_p",
        "holm_adjusted_p",
        "holm_reject_alpha_0p05",
        "primary_decision",
        "multiplicity_family",
        "multiplicity_role",
    ]
    return focal[[c for c in preferred if c in focal.columns]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    result = build_table(pd.read_csv(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    result.to_csv(tmp, index=False)
    tmp.replace(args.out)
    print(f"wrote {args.out} ({len(result)} focal comparisons)")


if __name__ == "__main__":
    main()
