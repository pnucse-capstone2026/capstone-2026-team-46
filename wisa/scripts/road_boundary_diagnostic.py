#!/usr/bin/env python3
"""ROAD label-noise exposure diagnostic.

Quantifies how exposed window labels are to injection-interval boundary
noise: attack windows containing only a few injected frames are the ones
whose binary label could flip if frames near the interval boundary were
mislabeled. Reports the distribution of injected-frame counts per attack
window across all ROAD attack captures.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "datasets" / "processed" / "road_frames"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
W, S = 128, 32

profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")
counts_all = []
for path in sorted(FRAMES.glob("*.parquet")):
    if profile.loc[path.stem]["role"] != "attack":
        continue
    y = pd.read_parquet(path, columns=["attack"])["attack"].to_numpy(dtype=np.int32)
    if len(y) < W:
        continue
    starts = np.arange(0, len(y) - W + 1, S)
    csum = np.concatenate([[0], np.cumsum(y)])
    counts = csum[starts + W] - csum[starts]
    counts_all.append(counts[counts > 0])
counts_all = np.concatenate(counts_all)

rows = []
for k in [1, 2, 4, 8, 16]:
    rows.append({"injected_frames_leq": k, "attack_windows": int((counts_all <= k).sum()),
                 "fraction_of_attack_windows": float((counts_all <= k).mean())})
summary = pd.DataFrame(rows)
summary.to_csv(TABLES / "road_boundary_diagnostic.csv", index=False)
log = {
    "total_attack_windows": int(len(counts_all)),
    "median_injected_frames_per_attack_window": float(np.median(counts_all)),
    "fraction_leq2": float((counts_all <= 2).mean()),
    "output": "results/tables/road_boundary_diagnostic.csv",
}
(LOGS / "road_boundary_diagnostic.log").write_text(json.dumps(log, indent=2) + "\n")
print(json.dumps(log, indent=2))
print(summary.to_string(index=False))
