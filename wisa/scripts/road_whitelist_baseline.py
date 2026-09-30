#!/usr/bin/env python3
"""ROAD ID-whitelist baseline (closes the "(not run on ROAD)" gap).

Applies the exact can-train-and-test lightweight whitelist baseline from
scripts/run_cantt_lightweight_baseline.py to ROAD: a frame is unknown if its
CAN ID is outside the Car-Hacking train-normal ID set, and a window
(128 frames, stride 32, per capture, no cross-capture windows) is flagged
attack if it contains any unknown frame. The whitelist definition and the
window-level flagging rule are imported unchanged from the can-train script,
so they are identical by construction.

ROAD is evaluation-only; this baseline is learning-free (no training,
validation, or model selection), so it is compliant with the data-usage rules.

Outputs a NEW table results/tables/road_whitelist_baseline.csv mirroring the
results/tables/road_external_summary.csv schema (setting/level/axis rows with
all / axis / capture levels), plus exact window counts appended. Existing
tables are never modified; run_cantt_lightweight_baseline.py and its default
outputs are untouched.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cantt_lightweight_baseline import (
    LOGS,
    PROCESSED,
    STRIDE,
    TABLES,
    WINDOW_SIZE,
    WINDOWS,
    MetricCounter,
    car_hacking_train_normal_can_ids,
    eval_npz_windows,
    window_binary_from_frames,
    write_csv,
)

FRAMES_DIR = PROCESSED / "road_frames"
RULE = "window_attack_if_any_can_id_not_in_car_hacking_train_normal_whitelist"
OUT_CSV = TABLES / "road_whitelist_baseline.csv"


def summary_row(level: str, axis: str, counter: MetricCounter) -> dict:
    """One row in the road_external_summary.csv column style.

    The whitelist baseline is deterministic, so seeds is marked accordingly
    and the std columns are 0 (nan where there are no attack/normal windows,
    matching the nan rendering of the existing summary).
    """
    m = counter.metrics()
    has_normal = m["normal_windows"] > 0
    has_attack = m["attack_windows"] > 0
    row = {
        "setting": "can_id_whitelist",
        "level": level,
        "axis": axis,
        "seeds": "deterministic",
        "windows": m["windows"],
        "normal_recall_mean": m["normal_recall"] if has_normal else float("nan"),
        "fpr_mean": m["fpr"] if has_normal else float("nan"),
        "fpr_std": 0.0 if has_normal else float("nan"),
        "attack_recall_mean": m["attack_recall"] if has_attack else float("nan"),
        "attack_recall_std": 0.0 if has_attack else float("nan"),
        # exact counts appended after the mirrored schema columns
        "normal_windows": m["normal_windows"],
        "attack_windows": m["attack_windows"],
        "tn": m["tn"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tp": m["tp"],
        "rule": RULE,
    }
    return row


def main() -> None:
    if OUT_CSV.exists():
        raise SystemExit(f"{OUT_CSV} already exists; refusing to overwrite an existing results table.")

    start = time.time()
    captures = sorted(FRAMES_DIR.glob("*.parquet"))
    if not captures:
        raise SystemExit("Run scripts/preprocess_road.py first.")

    whitelist = car_hacking_train_normal_can_ids()
    whitelist_list = list(whitelist)
    profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")

    metrics: dict[tuple[str, str], MetricCounter] = defaultdict(MetricCounter)
    attack_capture_normals = MetricCounter()  # FPR over normal windows within attack captures

    for idx, path in enumerate(captures, start=1):
        capture = path.stem
        info = profile.loc[capture]
        masquerade = "masquerade" if int(info.get("masquerade", 0)) else "fabrication"
        axis = "ambient" if info["role"] == "ambient" else f"{info['family']}|{masquerade}"
        df = pd.read_parquet(path, columns=["can_id", "attack"])
        can_ids = df["can_id"].to_numpy(dtype=np.int64)
        y = window_binary_from_frames(df["attack"].to_numpy(dtype=np.int8))
        unknown = (~np.isin(can_ids, whitelist_list)).astype(np.int8)
        pred = window_binary_from_frames(unknown)
        for key in [("all", "all"), ("axis", axis), ("capture", capture)]:
            metrics[key].update(y, pred)
        if axis != "ambient":
            normal_mask = y == 0
            attack_capture_normals.update(y[normal_mask], pred[normal_mask])
        print(f"road whitelist {idx}/{len(captures)} {capture} ({axis})", flush=True)

    rows = [summary_row(level, axis, counter) for (level, axis), counter in sorted(metrics.items())]
    write_csv(OUT_CSV, rows)

    # In-dataset sanity re-check (same eval path as the can-train script);
    # logged only, never written over existing tables.
    sanity = {}
    for dataset, path in [
        ("car_hacking_real_test", WINDOWS / "test_windows.npz"),
        ("otids_cross_binary", WINDOWS / "otids_cross_windows.npz"),
    ]:
        sanity[dataset] = eval_npz_windows(path, whitelist).metrics()

    overall = metrics[("all", "all")].metrics()
    ambient = metrics[("axis", "ambient")].metrics()
    log = {
        "method": "can_id_whitelist",
        "rule": RULE,
        "whitelist_source": "datasets/processed/car_hacking_train.parquet frames with binary_label_id == 0",
        "whitelist_size": len(whitelist),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "captures": len(captures),
        "road_overall": overall,
        "road_ambient": ambient,
        "road_attack_capture_normal_windows_fpr": (
            attack_capture_normals.fp / (attack_capture_normals.tn + attack_capture_normals.fp)
            if (attack_capture_normals.tn + attack_capture_normals.fp)
            else None
        ),
        "road_attack_capture_normal_windows": attack_capture_normals.tn + attack_capture_normals.fp,
        "in_dataset_sanity": sanity,
        "outputs": ["results/tables/road_whitelist_baseline.csv"],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "road_whitelist_baseline.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
