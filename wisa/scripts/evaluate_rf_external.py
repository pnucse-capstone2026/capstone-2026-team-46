#!/usr/bin/env python3
"""Model-family axis diagnostic: existing real-only random forests on the
external rungs (can-train-and-test, ROAD).

The RF baselines (models/baseline/random_forest_real_only_seed*.joblib) use
raw aggregate window features (mean/std/min/max/last-minus-first), no
standardizer. This measures whether the external normal-support failure is
CNN-specific or repeats for a different model family. Evaluation-only.
"""
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_cantt_tier1 import (
    FEATURE_NAMES,
    LOGS,
    TABLES,
    ZIP_PATH,
    MetricCounter,
    iter_windows,
    list_cantt_files,
    read_cantt_csv,
)
from train_channel_ablation import write_csv
from train_real_only_baselines import rf_features

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODELS = ROOT / "models" / "baseline"
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
SEEDS = [7, 42, 123]


def load_rfs() -> dict[int, object]:
    return {seed: joblib.load(BASELINE_MODELS / f"random_forest_real_only_seed{seed}.joblib") for seed in SEEDS}


def rf_binary_pred(rf, x: np.ndarray) -> np.ndarray:
    pred = rf.predict(rf_features(x))
    return (np.asarray(pred) > 0).astype(np.int8)


def main() -> None:
    start = time.time()
    rfs = load_rfs()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)

    road_profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")
    captures = sorted(ROAD_FRAMES.glob("*.parquet"))
    for idx, path in enumerate(captures, start=1):
        capture = path.stem
        role = road_profile.loc[capture]["role"]
        axis = "ambient" if role == "ambient" else "attack_captures"
        df = pd.read_parquet(path)
        features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        print(f"rf road {idx}/{len(captures)} {capture}", flush=True)
        for x, y, _starts in iter_windows(features, frame_y):
            for seed, rf in rfs.items():
                pred = rf_binary_pred(rf, x)
                metrics[("road", seed, "all")].update(y, pred)
                metrics[("road", seed, axis)].update(y, pred)

    files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"rf cantt {idx}/{len(files)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                for seed, rf in rfs.items():
                    pred = rf_binary_pred(rf, x)
                    metrics[("cantt_external", seed, "all")].update(y, pred)

    seed_rows = []
    for (dataset, seed, axis), counter in sorted(metrics.items()):
        row = {"model": "RandomForest_real_only", "dataset": dataset, "seed": seed, "axis": axis}
        row.update(counter.row())
        seed_rows.append(row)
    write_csv(TABLES / "rf_external_by_seed.csv", seed_rows)

    df = pd.DataFrame(seed_rows)
    summary_rows = []
    for (dataset, axis), group in df.groupby(["dataset", "axis"], sort=True):
        summary_rows.append(
            {
                "model": "RandomForest_real_only",
                "dataset": dataset,
                "axis": axis,
                "seeds": ";".join(str(int(s)) for s in sorted(group["seed"].unique())),
                "windows": int(group["windows"].iloc[0]),
                "normal_recall_mean": float(pd.to_numeric(group["normal_recall"], errors="coerce").mean()),
                "fpr_mean": float(pd.to_numeric(group["fpr"], errors="coerce").mean()),
                "fpr_std": float(pd.to_numeric(group["fpr"], errors="coerce").std(ddof=1)),
                "attack_recall_mean": float(pd.to_numeric(group["attack_recall"], errors="coerce").mean()),
                "attack_recall_std": float(pd.to_numeric(group["attack_recall"], errors="coerce").std(ddof=1)),
            }
        )
    write_csv(TABLES / "rf_external_summary.csv", summary_rows)

    log = {
        "diagnostic": "rf_external_eval",
        "models": [f"random_forest_real_only_seed{s}.joblib" for s in SEEDS],
        "datasets": ["road", "cantt_external"],
        "features": "raw aggregate window features (mean/std/min/max/last-minus-first), no standardizer",
        "outputs": [
            "results/tables/rf_external_by_seed.csv",
            "results/tables/rf_external_summary.csv",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "rf_external_eval.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
