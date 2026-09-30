#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def write_low_intensity_table() -> None:
    src = pd.read_csv(TABLES / "variant_sensitivity_key_comparisons.csv")
    selected_settings = ["real_only", "real_oversampling_0p50", "rule_0p30", "rule_1p00"]
    selected = src[
        (src["comparison_scope"] == "scenario_exact_class_recall")
        & (src["severity"] == "low")
        & (src["attack_type"].isin(["Fuzzy", "Gear", "RPM"]))
        & (src["setting"].isin(selected_settings))
    ].copy()
    label_map = {
        "real_only": "Real only",
        "real_oversampling_0p50": "Real oversampling +50%",
        "rule_0p30": "Rule +30%",
        "rule_1p00": "Rule +100%",
    }
    selected["label"] = selected["setting"].map(label_map)
    selected = selected[
        [
            "setting",
            "label",
            "attack_type",
            "severity",
            "attack_detection_recall_mean",
            "exact_class_recall_mean",
        ]
    ]
    selected.to_csv(TABLES / "low_intensity_sensitivity_summary.csv", index=False)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_low_intensity_table()
    print(TABLES / "low_intensity_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
