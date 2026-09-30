#!/usr/bin/env python3
"""ROAD target-normal calibration probe (DIAGNOSTIC ONLY).

Question: would label-free calibration on a small amount of target-vehicle
normal traffic rescue the external failure? This is a constructive "what
would it take" probe, not a deployment procedure, and it never touches model
weights or model selection.

Protocol (capture-level split, no leakage):
- Calibration captures: ambient_dyno_drive_basic_long and
  ambient_highway_street_driving_long (one dynamometer, one street/highway).
- Held-out: the remaining 10 ambient captures (FPR) and all 29 attack
  captures (binary attack-window recall).
- Probes per existing CNN (real_only, rule_0p30; seeds 7/42/123):
  1) threshold: binary threshold = 99% quantile of calibration-normal attack
     scores under the original Car-Hacking standardizer (target FPR 1%).
  2) restandardize: refit input mean/std on calibration windows, keep the
     default argmax decision.
  3) restandardize+threshold: both.
- Baseline row: original standardizer + argmax on the same held-out split.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_cantt_tier1 import FEATURE_NAMES, LOGS, TABLES, MetricCounter, iter_windows, load_models
from train_channel_ablation import write_csv
from train_real_only_baselines import WindowDataset, predict_cnn, standardize
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
SELECTED_SETTINGS = {"real_only", "rule_0p30"}
CALIBRATION_CAPTURES = ["ambient_dyno_drive_basic_long", "ambient_highway_street_driving_long"]
TARGET_FPR = 0.01
CONDITIONS = ["original_argmax", "threshold", "restandardize", "restandardize_threshold"]


def capture_windows(path: Path):
    df = pd.read_parquet(path)
    features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    frame_y = df["attack"].to_numpy(dtype=np.int8)
    yield from iter_windows(features, frame_y)


def model_scores(info: dict, x_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)), batch_size=1024, shuffle=False, num_workers=0)
    _, probs = predict_cnn(info["model"], loader, info["device"])
    pred = (probs.argmax(axis=1) > 0).astype(np.int8)
    scores = 1.0 - probs[:, 0]
    return pred, scores


def main() -> None:
    start = time.time()
    models = [m for m in load_models() if m["setting"] in SELECTED_SETTINGS]
    profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")
    captures = sorted(ROAD_FRAMES.glob("*.parquet"))
    calib_paths = [p for p in captures if p.stem in CALIBRATION_CAPTURES]
    heldout_paths = [p for p in captures if p.stem not in CALIBRATION_CAPTURES]
    assert len(calib_paths) == 2

    # Pass 1: refit standardizer stats on calibration windows (raw).
    total, count = np.zeros(len(FEATURE_NAMES), dtype=np.float64), 0
    sq = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for path in calib_paths:
        for x, _y, _s in capture_windows(path):
            flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
            total += flat.sum(axis=0)
            sq += (flat ** 2).sum(axis=0)
            count += len(flat)
    refit_mean = (total / count).astype(np.float32).reshape(1, 1, -1)
    refit_var = sq / count - (total / count) ** 2
    refit_std = np.sqrt(np.maximum(refit_var, 0.0)).astype(np.float32).reshape(1, 1, -1)
    refit_std[refit_std < 1e-6] = 1.0

    # Pass 2: calibration scores under both standardizers -> thresholds.
    calib_scores_orig: dict[tuple, list] = defaultdict(list)
    calib_scores_refit: dict[tuple, list] = defaultdict(list)
    with torch.no_grad():
        for path in calib_paths:
            for x, _y, _s in capture_windows(path):
                for info in models:
                    key = (info["setting"], info["seed"])
                    _, s1 = model_scores(info, standardize(x, info["mean"], info["std"]))
                    _, s2 = model_scores(info, standardize(x, refit_mean, refit_std))
                    calib_scores_orig[key].append(s1)
                    calib_scores_refit[key].append(s2)
    thr_orig = {k: float(np.quantile(np.concatenate(v), 1.0 - TARGET_FPR)) for k, v in calib_scores_orig.items()}
    thr_refit = {k: float(np.quantile(np.concatenate(v), 1.0 - TARGET_FPR)) for k, v in calib_scores_refit.items()}

    # Pass 3: held-out evaluation under the four conditions.
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    with torch.no_grad():
        for idx, path in enumerate(heldout_paths, start=1):
            capture = path.stem
            axis = "ambient_heldout" if profile.loc[capture]["role"] == "ambient" else "attack_captures"
            print(f"probe {idx}/{len(heldout_paths)} {capture} ({axis})", flush=True)
            for x, y, _s in capture_windows(path):
                for info in models:
                    key = (info["setting"], info["seed"])
                    pred_o, score_o = model_scores(info, standardize(x, info["mean"], info["std"]))
                    pred_r, score_r = model_scores(info, standardize(x, refit_mean, refit_std))
                    preds = {
                        "original_argmax": pred_o,
                        "threshold": (score_o >= thr_orig[key]).astype(np.int8),
                        "restandardize": pred_r,
                        "restandardize_threshold": (score_r >= thr_refit[key]).astype(np.int8),
                    }
                    for cond in CONDITIONS:
                        metrics[(cond, *key, axis)].update(y, preds[cond])

    seed_rows = []
    for (cond, setting, seed, axis), counter in sorted(metrics.items()):
        row = {"condition": cond, "setting": setting, "seed": seed, "axis": axis}
        row.update(counter.row())
        seed_rows.append(row)
    write_csv(TABLES / "road_calibration_probe_by_seed.csv", seed_rows)

    df = pd.DataFrame(seed_rows)
    summary = []
    for (cond, setting, axis), group in df.groupby(["condition", "setting", "axis"], sort=True):
        summary.append(
            {
                "condition": cond,
                "setting": setting,
                "axis": axis,
                "seeds": ";".join(str(int(s)) for s in sorted(group["seed"].unique())),
                "windows": int(group["windows"].iloc[0]),
                "fpr_mean": float(pd.to_numeric(group["fpr"], errors="coerce").mean()),
                "fpr_std": float(pd.to_numeric(group["fpr"], errors="coerce").std(ddof=1)),
                "attack_recall_mean": float(pd.to_numeric(group["attack_recall"], errors="coerce").mean()),
                "attack_recall_std": float(pd.to_numeric(group["attack_recall"], errors="coerce").std(ddof=1)),
            }
        )
    write_csv(TABLES / "road_calibration_probe_summary.csv", summary)

    log = {
        "diagnostic": "road_calibration_probe",
        "note": "diagnostic only; no model weights or model selection touched",
        "calibration_captures": CALIBRATION_CAPTURES,
        "target_fpr": TARGET_FPR,
        "settings": sorted(SELECTED_SETTINGS),
        "thresholds_original_standardizer": {f"{k[0]}_seed{k[1]}": v for k, v in thr_orig.items()},
        "thresholds_refit_standardizer": {f"{k[0]}_seed{k[1]}": v for k, v in thr_refit.items()},
        "outputs": [
            "results/tables/road_calibration_probe_by_seed.csv",
            "results/tables/road_calibration_probe_summary.csv",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "road_calibration_probe.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
