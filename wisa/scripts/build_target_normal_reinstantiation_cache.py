#!/usr/bin/env python3
"""Build can-train target-normal caches for target-normal reinstantiation."""
from __future__ import annotations

import argparse
import json
import time
import zipfile

import numpy as np

from run_cantt_tier1 import ZIP_PATH, iter_windows, read_cantt_csv
from target_normal_reinstantiation_common import (
    CALIB_ATTACK_CACHE,
    CALIB_ATTACK_CAP,
    CALIB_NORMAL_CACHE,
    CALIB_NORMAL_CAP,
    EXP,
    LOGS,
    ORACLE_STEM_TO_ATTACK,
    SPLIT_MANIFEST,
    SUBSAMPLE_SEED,
    ensure_dirs,
    sample_rows,
    split_cantt_files,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-normal reinstantiation caches.")
    parser.add_argument("--force", action="store_true", help="Rebuild caches even if they already exist.")
    parser.add_argument("--include-oracle-attacks", action="store_true", help="Also cache calibration attack windows for oracle-only diagnostics.")
    args = parser.parse_args()

    ensure_dirs()
    if CALIB_NORMAL_CACHE.exists() and (CALIB_ATTACK_CACHE.exists() or not args.include_oracle_attacks) and not args.force:
        print(f"cache exists: {CALIB_NORMAL_CACHE}", flush=True)
        return

    start = time.time()
    calibration, held_out = split_cantt_files()
    manifest_rows = [
        {
            "role": "calibration",
            "path": f.path,
            "set_id": f.set_id,
            "subset_id": f.subset_id,
            "source_stem": f.source_stem,
            "vehicle_axis": f.vehicle_axis,
            "attack_axis": f.attack_axis,
            "vehicle": f.vehicle,
        }
        for f in calibration
    ] + [
        {
            "role": "held_out",
            "path": f.path,
            "set_id": f.set_id,
            "subset_id": f.subset_id,
            "source_stem": f.source_stem,
            "vehicle_axis": f.vehicle_axis,
            "attack_axis": f.attack_axis,
            "vehicle": f.vehicle,
        }
        for f in held_out
    ]
    write_csv(SPLIT_MANIFEST, manifest_rows)

    normal_chunks: list[np.ndarray] = []
    attack_chunks: list[np.ndarray] = []
    attack_labels: list[np.ndarray] = []
    file_rows: list[dict] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(calibration, start=1):
            df, features, issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            normal_count = 0
            attack_count = 0
            oracle_attack_count = 0
            oracle_label = ORACLE_STEM_TO_ATTACK.get(meta.source_stem)
            for x, y, _starts in iter_windows(features, frame_y):
                normal_mask = y == 0
                attack_mask = y == 1
                normal_count += int(normal_mask.sum())
                attack_count += int(attack_mask.sum())
                if normal_mask.any():
                    normal_chunks.append(x[normal_mask].astype(np.float32))
                if args.include_oracle_attacks and oracle_label is not None and attack_mask.any():
                    attack_chunks.append(x[attack_mask].astype(np.float32))
                    attack_labels.append(np.full(int(attack_mask.sum()), oracle_label, dtype=np.int64))
                    oracle_attack_count += int(attack_mask.sum())
            row = {
                "path": meta.path,
                "source_stem": meta.source_stem,
                "subset_id": meta.subset_id,
                "vehicle_axis": meta.vehicle_axis,
                "attack_axis": meta.attack_axis,
                "normal_windows": normal_count,
                "attack_windows_excluded_from_main_training": attack_count,
                "oracle_attack_windows_cached": oracle_attack_count,
                "oracle_attack_type_id": oracle_label if oracle_label is not None else "",
                "parse_issues": ";".join(f"{k}:{v}" for k, v in sorted(issues.items())),
            }
            file_rows.append(row)
            print(
                f"calibration {idx}/{len(calibration)} {meta.path} "
                f"normal={normal_count} attack={attack_count} oracle={oracle_attack_count}",
                flush=True,
            )

    if not normal_chunks:
        raise RuntimeError("No calibration normal windows were collected.")
    all_normal = np.concatenate(normal_chunks, axis=0)
    normal_sample, normal_total = sample_rows(all_normal, CALIB_NORMAL_CAP, SUBSAMPLE_SEED)
    np.savez_compressed(
        CALIB_NORMAL_CACHE,
        x=normal_sample.astype(np.float32),
        total_normal_windows_before_cap=np.asarray(normal_total, dtype=np.int64),
        cap=np.asarray(CALIB_NORMAL_CAP, dtype=np.int64),
        subsample_seed=np.asarray(SUBSAMPLE_SEED, dtype=np.int64),
    )

    attack_total = 0
    attack_used = 0
    if args.include_oracle_attacks and attack_chunks:
        all_attack = np.concatenate(attack_chunks, axis=0)
        all_attack_y = np.concatenate(attack_labels, axis=0)
        attack_total = len(all_attack)
        if attack_total > CALIB_ATTACK_CAP:
            rng = np.random.default_rng(SUBSAMPLE_SEED + 1)
            idx = rng.choice(attack_total, size=CALIB_ATTACK_CAP, replace=False)
            idx.sort()
            all_attack = all_attack[idx]
            all_attack_y = all_attack_y[idx]
        attack_used = len(all_attack)
        np.savez_compressed(
            CALIB_ATTACK_CACHE,
            x=all_attack.astype(np.float32),
            y_attack_type=all_attack_y.astype(np.int64),
            total_attack_windows_before_cap=np.asarray(attack_total, dtype=np.int64),
            cap=np.asarray(CALIB_ATTACK_CAP, dtype=np.int64),
            subsample_seed=np.asarray(SUBSAMPLE_SEED + 1, dtype=np.int64),
        )

    write_csv(EXP / "calibration_window_counts.csv", file_rows)
    log = {
        "role": "target_normal_reinstantiation_cache",
        "calibration_files": len(calibration),
        "held_out_files": len(held_out),
        "normal_windows_before_cap": int(normal_total),
        "normal_windows_used": int(len(normal_sample)),
        "normal_cap": CALIB_NORMAL_CAP,
        "oracle_attack_windows_before_cap": int(attack_total),
        "oracle_attack_windows_used": int(attack_used),
        "include_oracle_attacks": bool(args.include_oracle_attacks),
        "outputs": [
            str(CALIB_NORMAL_CACHE.relative_to(EXP.parents[1])),
            str(SPLIT_MANIFEST.relative_to(EXP.parents[1])),
        ],
        "elapsed_seconds": time.time() - start,
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "target_normal_reinstantiation_cache.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    (EXP / "cache_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
