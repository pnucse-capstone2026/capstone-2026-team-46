import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "external_operating_point_neural.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("external_operating_point_neural", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_source_threshold_records_tie_aware_achieved_fpr():
    scores = np.array([0.0] * 8 + [0.5, 0.5], dtype=np.float32)
    threshold, achieved = MODULE.source_threshold(scores, 0.1)
    assert threshold > 0.0
    assert achieved == 0.0


def test_count_reports_fpr_and_attack_recall_jointly():
    counter = MODULE.Count()
    counter.update(
        np.array([0, 0, 1, 1], dtype=np.int8),
        np.array([0, 1, 1, 0], dtype=np.int8),
    )
    row = counter.row()
    assert row["fpr"] == 0.5
    assert row["attack_recall"] == 0.5
    assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (1, 1, 1, 1)


def test_unit_macro_differs_from_pooled_and_keeps_unit_uncertainty():
    group = pd.DataFrame([
        {
            "dataset": "X", "unit_type": "file", "family": "cnn",
            "setting": "real_only", "seed": 7, "policy": "source_val_fpr_0.001",
            "val_fpr_target": 0.001, "threshold": 0.7, "val_fpr_achieved": 0.001,
            "windows": 20, "normal_windows": 10, "attack_windows": 10,
            "fp": 10, "tp": 5, "fpr": 1.0, "attack_recall": 0.5,
        },
        {
            "dataset": "X", "unit_type": "file", "family": "cnn",
            "setting": "real_only", "seed": 7, "policy": "source_val_fpr_0.001",
            "val_fpr_target": 0.001, "threshold": 0.7, "val_fpr_achieved": 0.001,
            "windows": 100, "normal_windows": 90, "attack_windows": 10,
            "fp": 0, "tp": 10, "fpr": 0.0, "attack_recall": 1.0,
        },
    ])
    row = MODULE.aggregate_unit_group(group, bootstrap_samples=200)
    assert row["pooled_fpr"] == 0.1
    assert row["unit_macro_fpr"] == 0.5
    assert row["pooled_attack_recall"] == 0.75
    assert row["unit_macro_attack_recall"] == 0.75
    assert row["unit_macro_fpr_ci_low"] <= 0.5 <= row["unit_macro_fpr_ci_high"]


def test_paired_summary_uses_within_seed_deltas():
    rows = []
    for seed, base, aug in [(7, 0.1, 0.4), (42, 0.2, 0.3)]:
        for setting, value in [("real_only", base), ("rule_0p30", aug)]:
            rows.append({
                "dataset": "X", "unit_type": "file", "family": "cnn",
                "setting": setting, "seed": seed, "policy": "source_val_fpr_0.001",
                "val_fpr_target": 0.001,
                "pooled_fpr": value, "pooled_attack_recall": value,
                "unit_macro_fpr": value, "unit_macro_attack_recall": value,
            })
    out = MODULE.paired_setting_summary(rows)
    assert len(out) == 1
    assert np.isclose(out[0]["delta_unit_macro_fpr_mean"], 0.2)
    assert out[0]["delta_unit_macro_fpr_positive_seeds"] == 2


def test_bootstrap_single_unit_does_not_invent_precision():
    low, high = MODULE.bootstrap_mean_ci([0.25], samples=2000, seed=1)
    assert low == high == 0.25
