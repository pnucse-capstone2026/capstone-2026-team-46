#!/usr/bin/env python3
"""Tier 2 ROAD external evaluation.

Evaluates the existing Car-Hacking-trained CNNs (real_only, oversampling,
rule ratios; seeds 7/42/123) on ROAD windows without retraining. ROAD is
evaluation-only. Reports, in this order per AGENTS.md:

1. Car-Hacking-scaler z-score diagnostics (especially delta_t) and CAN ID
   support overlap, before interpreting FPR.
2. FPR / normal recall on ambient captures, attack recall by attack family
   with fabrication vs masquerade separated. Binary detection only.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from run_cantt_tier1 import (
    FEATURE_NAMES,
    LOGS,
    TABLES,
    MetricCounter,
    evaluate_batch,
    iter_windows,
    load_models,
)
from train_channel_ablation import write_csv

ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / "datasets" / "processed" / "road_frames"
PROCESSED = ROOT / "datasets" / "processed"
BASELINE_STDZ = ROOT / "models" / "baseline" / "cnn_standardizer.npz"


def car_hacking_train_normal_id_set() -> set[int]:
    ids: set[int] = set()
    pf = pq.ParquetFile(PROCESSED / "car_hacking_train.parquet")
    for batch in pf.iter_batches(columns=["can_id", "binary_label_id"], batch_size=500_000):
        df = batch.to_pandas()
        ids.update(df.loc[df["binary_label_id"] == 0, "can_id"].unique().tolist())
    return ids


def zscore_and_overlap_diagnostics(captures: list[Path]) -> tuple[list[dict], dict]:
    stdz = np.load(BASELINE_STDZ)
    mean = stdz["mean"].reshape(-1)
    std = stdz["std"].reshape(-1)
    train_ids = car_hacking_train_normal_id_set()
    rows = []
    normal_total = 0
    normal_covered = 0
    for path in captures:
        df = pd.read_parquet(path)
        feats = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
        z = (feats - mean) / std
        normal_mask = df["attack"].to_numpy() == 0
        ids = df["can_id"].to_numpy()
        covered = np.isin(ids[normal_mask], list(train_ids)).sum()
        normal_total += int(normal_mask.sum())
        normal_covered += int(covered)
        row = {"capture": path.stem, "frames": len(df)}
        for i, name in enumerate(FEATURE_NAMES):
            row[f"z_mean_{name}"] = float(np.abs(z[:, i]).mean())
            row[f"z_gt3_frac_{name}"] = float((np.abs(z[:, i]) > 3).mean())
        row["train_normal_id_coverage"] = float(covered / max(normal_mask.sum(), 1))
        rows.append(row)
    overall = {
        "car_hacking_train_normal_id_coverage_on_road_normal_frames": normal_covered / max(normal_total, 1),
        "road_normal_frames": normal_total,
    }
    return rows, overall


def main() -> None:
    start = time.time()
    captures = sorted(FRAMES_DIR.glob("*.parquet"))
    if not captures:
        raise SystemExit("Run scripts/preprocess_road.py first.")

    profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")

    print("computing z-score and CAN ID overlap diagnostics", flush=True)
    z_rows, overlap = zscore_and_overlap_diagnostics(captures)
    write_csv(TABLES / "road_zscore_diagnostic.csv", z_rows)

    models = load_models()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)

    for idx, path in enumerate(captures, start=1):
        capture = path.stem
        info = profile.loc[capture]
        family = info["family"]
        masquerade = "masquerade" if int(info.get("masquerade", 0)) else "fabrication"
        axis = "ambient" if info["role"] == "ambient" else f"{family}|{masquerade}"
        df = pd.read_parquet(path)
        features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        print(f"road eval {idx}/{len(captures)} {capture} ({axis})", flush=True)
        with torch.no_grad():
            for x, y, _starts in iter_windows(features, frame_y):
                for model_info in models:
                    pred, _scores = evaluate_batch(model_info, x)
                    base = (model_info["setting"], model_info["seed"])
                    metrics[(*base, "all", "all")].update(y, pred)
                    metrics[(*base, "axis", axis)].update(y, pred)
                    metrics[(*base, "capture", capture)].update(y, pred)

    seed_rows = []
    for (setting, seed, level, axis), counter in sorted(metrics.items()):
        row = {"setting": setting, "seed": seed, "level": level, "axis": axis}
        row.update(counter.row())
        seed_rows.append(row)
    write_csv(TABLES / "road_external_by_seed.csv", seed_rows)

    df = pd.DataFrame(seed_rows)
    summary_rows = []
    for (setting, level, axis), group in df.groupby(["setting", "level", "axis"], sort=True):
        summary_rows.append(
            {
                "setting": setting,
                "level": level,
                "axis": axis,
                "seeds": ";".join(str(int(s)) for s in sorted(group["seed"].unique())),
                "windows": int(group["windows"].iloc[0]),
                "normal_recall_mean": float(pd.to_numeric(group["normal_recall"], errors="coerce").mean()),
                "fpr_mean": float(pd.to_numeric(group["fpr"], errors="coerce").mean()),
                "fpr_std": float(pd.to_numeric(group["fpr"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0,
                "attack_recall_mean": float(pd.to_numeric(group["attack_recall"], errors="coerce").mean()),
                "attack_recall_std": float(pd.to_numeric(group["attack_recall"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    write_csv(TABLES / "road_external_summary.csv", summary_rows)

    log = {
        "diagnostic": "road_external_eval",
        "models": "existing Car-Hacking-trained CNNs (no retraining)",
        "captures": len(captures),
        "id_overlap": overlap,
        "outputs": [
            "results/tables/road_zscore_diagnostic.csv",
            "results/tables/road_external_by_seed.csv",
            "results/tables/road_external_summary.csv",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "road_external_eval.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
