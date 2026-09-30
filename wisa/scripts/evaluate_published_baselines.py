#!/usr/bin/env python3
"""B3: published CAN IDS baselines through the evaluation ladder.

Faithful simplifications of two published detectors, profiled only on the
Car-Hacking training split and evaluated on the ladder rungs (in-dataset,
fixed variant, sensitivity, OTIDS, can-train external, ROAD). Evaluation-only;
deterministic (no seeds).

1) entropy  -- Mueter & Asaj (IEEE IV 2011): Shannon entropy of the CAN-ID
   distribution per 128-frame window; attack if outside the [0.1%, 99.9%]
   percentile band of train-normal window entropies. Vocabulary-free.
2) timing   -- Song, Kim & Kim (ICOIN 2016): per-ID mean inter-arrival rule;
   a window is an attack if any known ID with >=4 occurrences has mean
   in-window interval below half its train-normal median interval.
   Unseen-ID policies reported separately:
     timing_unseen_attack: any frame with an ID outside the train-normal ID
                           set marks the window as attack (whitelist-like).
     timing_unseen_ignore: unseen IDs are ignored (permissive).

Simplifications vs. the original papers are documented in the output log.
"""
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
PROCESSED = ROOT / "datasets" / "processed"
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"

ENTROPY_PCTL = (0.1, 99.9)
TIMING_MIN_OCC = 4
TIMING_FACTOR = 0.5
MAX_PROFILE_GAP_S = 10.0


# ---------------------------------------------------------------- profiles
def window_id_entropy(x: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of the CAN-ID distribution per window."""
    n, w, _ = x.shape
    ids = x[:, :, 0].astype(np.int64)
    win = np.repeat(np.arange(n, dtype=np.int64), w)
    key = win * (1 << 32) + ids.reshape(-1)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    group_win = np.zeros(counts.shape, dtype=np.int64)
    group_win[inverse] = win  # any member's window index
    p = counts / w
    contrib = -p * np.log2(p)
    return np.bincount(group_win, weights=contrib, minlength=n)


def build_entropy_profile(train_x: np.ndarray, train_y: np.ndarray) -> tuple[float, float]:
    ents = []
    normal_idx = np.where(train_y == 0)[0]
    for start in range(0, len(normal_idx), 16384):
        sel = normal_idx[start : start + 16384]
        ents.append(window_id_entropy(train_x[sel]))
    ents = np.concatenate(ents)
    return float(np.percentile(ents, ENTROPY_PCTL[0])), float(np.percentile(ents, ENTROPY_PCTL[1]))


def build_timing_profile() -> tuple[dict[int, float], set[int]]:
    """Per-ID median inter-arrival (seconds) from train-normal frames,
    grouped within (source_file, can_id) to avoid cross-file gaps."""
    pf = pq.ParquetFile(PROCESSED / "car_hacking_train.parquet")
    cols = ["source_file", "timestamp", "can_id", "binary_label_id"]
    diffs_by_id: dict[int, list] = defaultdict(list)
    known_ids: set[int] = set()
    for batch in pf.iter_batches(columns=cols, batch_size=2_000_000):
        df = batch.to_pandas()
        df = df[df["binary_label_id"] == 0]
        known_ids.update(int(v) for v in df["can_id"].unique())
        for (_src, cid), grp in df.groupby(["source_file", "can_id"], sort=False):
            d = np.diff(grp["timestamp"].to_numpy())
            d = d[(d > 0) & (d < MAX_PROFILE_GAP_S)]
            if len(d):
                diffs_by_id[int(cid)].append(d)
    profile = {cid: float(np.median(np.concatenate(parts))) for cid, parts in diffs_by_id.items()}
    return profile, known_ids


# ---------------------------------------------------------------- detectors
def predict_entropy(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    ent = window_id_entropy(x)
    return ((ent < lo) | (ent > hi)).astype(np.int8)


def predict_timing(
    x: np.ndarray, profile_ids: np.ndarray, profile_halves: np.ndarray, known_sorted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred_unseen_attack, pred_unseen_ignore)."""
    n, w, _ = x.shape
    ids = x[:, :, 0].astype(np.int64)
    t = np.cumsum(x[:, :, 10].astype(np.float64), axis=1)
    win = np.repeat(np.arange(n, dtype=np.int64), w)
    ids_flat = ids.reshape(-1)
    t_flat = t.reshape(-1)
    # frames are already in chronological order within each window
    order = np.lexsort((np.tile(np.arange(w), n), ids_flat, win))
    win_s, ids_s, t_s = win[order], ids_flat[order], t_flat[order]
    same = (win_s[1:] == win_s[:-1]) & (ids_s[1:] == ids_s[:-1])
    intervals = (t_s[1:] - t_s[:-1])[same]
    g_win = win_s[1:][same]
    g_ids = ids_s[1:][same]
    key = g_win * (1 << 32) + g_ids
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    mean_int = np.bincount(inverse, weights=intervals) / counts
    u_win = (uniq >> 32).astype(np.int64)
    u_ids = (uniq & 0xFFFFFFFF).astype(np.int64)
    pos = np.searchsorted(profile_ids, u_ids)
    pos_ok = (pos < len(profile_ids)) & (profile_ids[np.minimum(pos, len(profile_ids) - 1)] == u_ids)
    flagged_groups = pos_ok & (counts >= TIMING_MIN_OCC) & (mean_int < profile_halves[np.minimum(pos, len(profile_ids) - 1)])
    pred_known = np.zeros(n, dtype=np.int8)
    np.maximum.at(pred_known, u_win[flagged_groups], 1)
    unseen_frame = ~np.isin(ids_flat, known_sorted)
    has_unseen = np.zeros(n, dtype=np.int8)
    np.maximum.at(has_unseen, win[unseen_frame], 1)
    return np.maximum(pred_known, has_unseen), pred_known


# ---------------------------------------------------------------- evaluation
def main() -> None:
    start = time.time()
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    lo, hi = build_entropy_profile(train["x"], train["y_binary"].astype(np.int8))
    print(f"entropy band: [{lo:.4f}, {hi:.4f}]", flush=True)
    timing_profile, known_ids = build_timing_profile()
    profile_ids = np.array(sorted(timing_profile), dtype=np.int64)
    profile_halves = np.array([timing_profile[int(i)] * TIMING_FACTOR for i in profile_ids])
    known_sorted = np.array(sorted(known_ids), dtype=np.int64)
    print(f"timing profile ids: {len(profile_ids)}", flush=True)

    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)

    def eval_batch(rung: str, x: np.ndarray, y: np.ndarray) -> None:
        metrics[("entropy", "n/a", rung)].update(y, predict_entropy(x, lo, hi))
        p_attack, p_ignore = predict_timing(x, profile_ids, profile_halves, known_sorted)
        metrics[("timing", "unseen_attack", rung)].update(y, p_attack)
        metrics[("timing", "unseen_ignore", rung)].update(y, p_ignore)

    local_rungs = [
        ("indataset", "test_windows.npz"),
        ("fixed_variant", "variant_test_windows.npz"),
        ("sensitivity", "variant_sensitivity_windows.npz"),
        ("otids", "otids_cross_windows.npz"),
    ]
    for rung, npz_name in local_rungs:
        data = np.load(WINDOWS / npz_name, allow_pickle=True)
        x_all = data["x"]
        y_all = data["y_binary"].astype(np.int8)
        print(f"rung {rung}: {len(x_all)} windows", flush=True)
        for s in range(0, len(x_all), 16384):
            eval_batch(rung, x_all[s : s + 16384].astype(np.float32), y_all[s : s + 16384])

    profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")
    for idx, path in enumerate(sorted(ROAD_FRAMES.glob("*.parquet")), start=1):
        df = pd.read_parquet(path)
        features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        print(f"road {idx}/41 {path.stem}", flush=True)
        for x, y, _s in iter_windows(features, frame_y):
            eval_batch("road", x, y)

    files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"cantt {idx}/{len(files)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _s in iter_windows(features, frame_y):
                eval_batch("cantt_external", x, y)

    rows = []
    for (baseline, policy, rung), counter in sorted(metrics.items()):
        row = {"baseline": baseline, "policy": policy, "rung": rung}
        row.update(counter.row())
        rows.append(row)
    write_csv(TABLES / "published_baselines_summary.csv", rows)
    write_csv(
        TABLES / "published_baselines_profile.csv",
        [
            {"item": "entropy_band_lo", "value": lo},
            {"item": "entropy_band_hi", "value": hi},
            {"item": "entropy_percentiles", "value": f"{ENTROPY_PCTL[0]}/{ENTROPY_PCTL[1]}"},
            {"item": "timing_profile_id_count", "value": len(profile_ids)},
            {"item": "timing_rule", "value": f"mean in-window interval < {TIMING_FACTOR} * train-normal median, min {TIMING_MIN_OCC} occurrences"},
        ],
    )

    log = {
        "diagnostic": "published_baselines",
        "baselines": {
            "entropy": "Mueter & Asaj 2011, windowed CAN-ID Shannon entropy, percentile band on train-normal",
            "timing": "Song, Kim & Kim 2016, per-ID interval rule; window-level transplant with two unseen-ID policies",
        },
        "simplifications": [
            "entropy: percentile band [0.1,99.9] replaces the original sliding range check",
            "timing: per-ID mean in-window interval vs half the train-normal median (original uses message-level interval halving); min 4 occurrences",
            "profiles use Car-Hacking train-normal only; deterministic, no seeds",
        ],
        "outputs": [
            "results/tables/published_baselines_summary.csv",
            "results/tables/published_baselines_profile.csv",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "published_baselines.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
