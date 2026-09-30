#!/usr/bin/env python3
"""Build the compact support-shift summary table."""

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def row_value(df: pd.DataFrame, filters: dict, column: str) -> float:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if isinstance(value, float):
            mask &= df[key].round(4) == round(value, 4)
        else:
            mask &= df[key] == value
    rows = df[mask]
    if rows.empty:
        raise KeyError(f"no row for {filters}")
    return float(rows.iloc[0][column])


def diagnostic_row(df: pd.DataFrame, name: str) -> pd.Series:
    rows = df[df["diagnostic"] == name]
    if rows.empty:
        raise KeyError(name)
    return rows.iloc[0]


def parse_number(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"pattern {pattern!r} not found in {text!r}")
    return float(match.group(1))


def main() -> None:
    main_results = pd.read_csv(TABLES / "main_results_mean_std.csv")
    otids_shift = pd.read_csv(TABLES / "otids_normal_shift_key_comparisons.csv")
    otids_score = pd.read_csv(TABLES / "otids_normal_attack_score_key_comparisons.csv")
    cantt_overlap = pd.read_csv(TABLES / "cantt_can_id_overlap.csv")
    cantt_shift = pd.read_csv(TABLES / "cantt_vs_carhacking_scaled_feature_shift.csv")
    cantt_scores = pd.read_csv(TABLES / "cantt_external_score_quantiles.csv")
    cantt_external = pd.read_csv(TABLES / "cantt_external_eval_summary_mean_std.csv")
    road_z = pd.read_csv(TABLES / "road_zscore_diagnostic.csv")
    road_external = pd.read_csv(TABLES / "road_external_summary.csv")
    road_scores = pd.read_csv(TABLES / "road_calibration_probe_logit_summary.csv")

    id_row = diagnostic_row(otids_shift, "CAN ID support")
    dlc_row = diagnostic_row(otids_shift, "DLC distribution")
    timing_row = diagnostic_row(otids_shift, "Inter-arrival time")
    payload_row = diagnostic_row(otids_shift, "Payload byte distribution")

    car_top20 = parse_number(id_row["car_hacking_test_normal"], r"top-20 coverage ([0-9.]+)")
    otids_top20 = parse_number(id_row["otids_normal"], r"top-20 coverage ([0-9.]+)")

    car_fpr = row_value(
        main_results,
        {"generator": "rule_based", "ratio": 0.3, "split": "test"},
        "fpr_mean",
    )
    otids_fpr = row_value(
        main_results,
        {"generator": "rule_based", "ratio": 0.3, "split": "otids_cross_binary"},
        "fpr_mean",
    )
    car_score = row_value(
        otids_score,
        {"setting": "rule_0p30", "normal_split": "validation_normal"},
        "attack_score_q50_mean",
    )
    otids_score_q50 = row_value(
        otids_score,
        {"setting": "rule_0p30", "normal_split": "otids_normal"},
        "attack_score_q50_mean",
    )

    cantt_all = cantt_overlap[
        (cantt_overlap["set_id"] == "all")
        & (cantt_overlap["subset_id"] == "all")
        & (cantt_overlap["vehicle_axis"] == "all")
        & (cantt_overlap["attack_axis"] == "all")
    ].iloc[0]
    cantt_payload_abs_mean = cantt_shift[[f"data{i}_scaled_mean" for i in range(8)]].abs().mean(axis=1)
    cantt_score_q50 = cantt_scores[
        (cantt_scores["setting"] == "rule_0p30") & (cantt_scores["score_label"] == "normal")
    ]["score_p50"].median()
    cantt_fpr = row_value(cantt_external, {"setting": "rule_0p30"}, "fpr_mean")

    road_ambient = road_z[road_z["capture"].str.startswith("ambient")].copy()
    weights = road_ambient["frames"]

    def road_weighted(col: str) -> float:
        return float((road_ambient[col].abs() * weights).sum() / weights.sum())

    road_payload_abs = road_ambient[[f"z_mean_data{i}" for i in range(8)]].abs().mean(axis=1)
    road_payload_weighted = float((road_payload_abs * weights).sum() / weights.sum())
    road_fpr = row_value(road_external, {"setting": "rule_0p30", "level": "all", "axis": "all"}, "fpr_mean")
    road_saturation = row_value(
        road_scores,
        {"score": "softmax", "setting": "rule_0p30"},
        "saturation_frac_mean",
    )

    rows = [
        {
            "dataset": "Car-Hacking validation/test",
            "role": "source-domain reference",
            "id_support_indicator": f"self top-20 ID coverage={fmt(car_top20)}",
            "dlc_indicator": dlc_row["car_hacking_test_normal"],
            "payload_indicator": payload_row["car_hacking_test_normal"],
            "timing_indicator": timing_row["car_hacking_test_normal"],
            "rule_30_normal_score_indicator": f"validation normal median={car_score:.6f}",
            "rule_30_fpr": f"{car_fpr:.6f}",
            "takeaway": "source-domain normal support remains separable",
            "source_tables": "main_results_mean_std.csv; otids_normal_*",
        },
        {
            "dataset": "OTIDS normal",
            "role": "external scenario-level binary test",
            "id_support_indicator": f"Car-Hacking top-20 coverage={fmt(otids_top20)}; {id_row['shift_value']}",
            "dlc_indicator": dlc_row["shift_value"],
            "payload_indicator": payload_row["shift_value"],
            "timing_indicator": timing_row["shift_value"],
            "rule_30_normal_score_indicator": f"normal median={fmt(otids_score_q50, 4)}",
            "rule_30_fpr": f"{otids_fpr:.6f}",
            "takeaway": "external normals saturate as attack despite validation-calibrated source behavior",
            "source_tables": "main_results_mean_std.csv; otids_normal_shift_key_comparisons.csv; otids_normal_attack_score_key_comparisons.csv",
        },
        {
            "dataset": "can-train-and-test normal",
            "role": "external labeled vehicle/attack-axis test",
            "id_support_indicator": f"train-normal ID frame coverage={fmt(cantt_all['frame_coverage_by_car_hacking_ids'])}",
            "dlc_indicator": f"median |mean z|={fmt(cantt_shift['dlc_scaled_mean'].abs().median(), 3)}; max={fmt(cantt_shift['dlc_scaled_mean'].abs().max(), 3)}",
            "payload_indicator": f"median payload |mean z|={fmt(cantt_payload_abs_mean.median(), 3)}; max={fmt(cantt_payload_abs_mean.max(), 3)}",
            "timing_indicator": f"median |mean z|={fmt(cantt_shift['delta_t_scaled_mean'].abs().median(), 3)}; max={fmt(cantt_shift['delta_t_scaled_mean'].abs().max(), 3)}",
            "rule_30_normal_score_indicator": f"normal median={fmt(cantt_score_q50, 4)}",
            "rule_30_fpr": f"{cantt_fpr:.6f}",
            "takeaway": "low ID coverage and saturated scores repeat beyond OTIDS",
            "source_tables": "cantt_can_id_overlap.csv; cantt_vs_carhacking_scaled_feature_shift.csv; cantt_external_score_quantiles.csv; cantt_external_eval_summary_mean_std.csv",
        },
        {
            "dataset": "ROAD ambient",
            "role": "external physically verified-attack corpus ambient normal",
            "id_support_indicator": f"weighted train-normal ID coverage={fmt(road_weighted('train_normal_id_coverage'))}",
            "dlc_indicator": f"weighted |mean z|={fmt(road_weighted('z_mean_dlc'), 3)}",
            "payload_indicator": f"weighted payload |mean z|={fmt(road_payload_weighted, 3)}",
            "timing_indicator": f"weighted |mean z|={fmt(road_weighted('z_mean_delta_t'), 3)}",
            "rule_30_normal_score_indicator": f"softmax saturation fraction={fmt(road_saturation, 4)}",
            "rule_30_fpr": f"{road_fpr:.6f}",
            "takeaway": "real external ambient traffic also lands on the attack side of the source representation",
            "source_tables": "road_zscore_diagnostic.csv; road_external_summary.csv; road_calibration_probe_logit_summary.csv",
        },
    ]

    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table_path = TABLES / "support_shift_summary.csv"
    out.to_csv(table_path, index=False, lineterminator="\n")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
