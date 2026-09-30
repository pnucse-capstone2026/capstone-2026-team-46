#!/usr/bin/env python3
"""Generate target-normal-anchored rule synthetic attacks from can-train normals."""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from target_normal_reinstantiation_common import (
    ATTACK_TYPE_IDS,
    CALIB_NORMAL_CACHE,
    EXP,
    RULE_RATIO,
    SYNTHETIC,
    TABLES,
    TARGET_SYNTH_PATH,
    WINDOWS,
    ensure_dirs,
    feature_validity,
    write_csv,
)

SEED = 271828
WINDOW_SIZE = 128


def choose_positions(rng: np.random.Generator, min_len: int, max_len: int, step_options: list[int]) -> np.ndarray:
    burst_len = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    step = int(rng.choice(step_options))
    return np.arange(start, start + burst_len, step)


def target_id_support(x: np.ndarray) -> np.ndarray:
    ids = np.unique(x[:, :, 0].astype(np.int32))
    ids = ids[(ids >= 0) & (ids <= 2047)]
    if len(ids) == 0:
        return np.arange(0, 2048, dtype=np.int32)
    return ids.astype(np.int32)


def synth_dos(rng: np.random.Generator, x: np.ndarray, _support_ids: np.ndarray) -> int:
    pos = choose_positions(rng, 32, 90, [1, 2])
    x[pos, 0] = 0x000
    x[pos, 1] = 8
    x[pos, 2:10] = rng.choice([0, 255], size=(len(pos), 8), p=[0.92, 0.08])
    x[pos, 10] = rng.uniform(0.00003, 0.002, size=len(pos))
    return len(pos)


def synth_fuzzy(rng: np.random.Generator, x: np.ndarray, support_ids: np.ndarray, full_id_range: bool = False) -> int:
    pos = choose_positions(rng, 24, 88, [1, 2, 3, 4])
    if full_id_range:
        x[pos, 0] = rng.integers(0, 2048, size=len(pos))
    else:
        x[pos, 0] = rng.choice(support_ids, size=len(pos), replace=True)
    dlc = rng.choice([2, 5, 8], size=len(pos), p=[0.08, 0.12, 0.80])
    x[pos, 1] = dlc
    payload = rng.integers(0, 256, size=(len(pos), 8))
    for i, d in enumerate(dlc):
        if d < 8:
            payload[i, int(d) :] = 0
    x[pos, 2:10] = payload
    x[pos, 10] = rng.uniform(0.00005, 0.02, size=len(pos))
    return len(pos)


def synth_spoof(rng: np.random.Generator, x: np.ndarray, support_ids: np.ndarray, mode: str) -> int:
    pos = choose_positions(rng, 32, 90, [1, 2, 3])
    x[pos, 0] = rng.choice(support_ids, size=len(pos), replace=True)
    x[pos, 1] = 8
    base = x[pos, 2:10].copy()
    if mode == "gear":
        levels = rng.choice([0, 1, 2, 3, 4, 5], size=len(pos))
        base[:, 0] = np.clip(levels * 40 + rng.integers(0, 16, size=len(pos)), 0, 255)
        base[:, 1] = np.clip(255 - base[:, 0] + rng.integers(-8, 9, size=len(pos)), 0, 255)
        base[:, 6] = np.clip(base[:, 6] + rng.integers(-16, 17, size=len(pos)), 0, 255)
    elif mode == "rpm":
        rpm = rng.integers(0, 8000, size=len(pos))
        base[:, 0] = (rpm // 32) % 256
        base[:, 1] = (rpm // 4) % 256
        base[:, 2] = np.clip(base[:, 2] + rng.integers(-50, 51, size=len(pos)), 0, 255)
        base[:, 5] = np.clip(base[:, 5] + rng.integers(-24, 25, size=len(pos)), 0, 255)
    else:
        raise ValueError(mode)
    x[pos, 2:10] = base
    x[pos, 10] = np.maximum(x[pos, 10], 1e-5)
    return len(pos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate target-normal rule synthetic windows.")
    parser.add_argument("--ratio", type=float, default=RULE_RATIO, help="Synthetic count relative to Car-Hacking train windows.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fuzzy-full-id-range", action="store_true", help="Use 0..2047 IDs for Fuzzy instead of target normal ID support.")
    args = parser.parse_args()

    ensure_dirs()
    if not CALIB_NORMAL_CACHE.exists():
        raise SystemExit("Run scripts/build_target_normal_reinstantiation_cache.py first.")
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    calib = np.load(CALIB_NORMAL_CACHE, allow_pickle=True)
    calib_x = calib["x"].astype(np.float32)
    support_ids = target_id_support(calib_x)
    rng = np.random.default_rng(args.seed)

    n_total = int(round(len(train["x"]) * args.ratio))
    classes = [
        ("DoS", ATTACK_TYPE_IDS["DoS"], lambda r, x: synth_dos(r, x, support_ids)),
        ("Fuzzy", ATTACK_TYPE_IDS["Fuzzy"], lambda r, x: synth_fuzzy(r, x, support_ids, args.fuzzy_full_id_range)),
        ("Gear", ATTACK_TYPE_IDS["Gear"], lambda r, x: synth_spoof(r, x, support_ids, "gear")),
        ("RPM", ATTACK_TYPE_IDS["RPM"], lambda r, x: synth_spoof(r, x, support_ids, "rpm")),
    ]
    per_class = n_total // len(classes)
    remainder = n_total - per_class * len(classes)

    xs: list[np.ndarray] = []
    y_binary: list[int] = []
    y_attack: list[int] = []
    synth_type: list[str] = []
    counts: list[int] = []
    rows: list[dict] = []
    for i, (name, label, fn) in enumerate(classes):
        n = per_class + (1 if i < remainder else 0)
        base_idx = rng.choice(len(calib_x), size=n, replace=n > len(calib_x))
        x_part = calib_x[base_idx].astype(np.float32).copy()
        local_counts = [fn(rng, x_part[j]) for j in range(n)]
        xs.append(x_part)
        y_binary.extend([1] * n)
        y_attack.extend([label] * n)
        synth_type.extend([name] * n)
        counts.extend(local_counts)
        rows.append(
            {
                "generator": "target_normal_rule",
                "synthetic_type": name,
                "windows": n,
                "binary_label": "Attack",
                "attack_type_id": label,
                "injection_count_mean": float(np.mean(local_counts)),
                "injection_count_min": int(np.min(local_counts)),
                "injection_count_max": int(np.max(local_counts)),
            }
        )

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y_binary_arr = np.asarray(y_binary, dtype=np.int8)
    y_attack_arr = np.asarray(y_attack, dtype=np.int8)
    type_arr = np.asarray(synth_type, dtype=object)
    count_arr = np.asarray(counts, dtype=np.int16)
    order = rng.permutation(len(y_binary_arr))
    validity = feature_validity(x)

    np.savez_compressed(
        TARGET_SYNTH_PATH,
        x=x[order],
        y_binary=y_binary_arr[order],
        y_attack_type=y_attack_arr[order],
        synthetic_type=type_arr[order],
        injection_count=count_arr[order],
        source="can-train calibration normal windows",
        seed=np.asarray(args.seed, dtype=np.int64),
        ratio=np.asarray(args.ratio, dtype=np.float32),
        fuzzy_id_policy=np.asarray("full_11bit" if args.fuzzy_full_id_range else "target_normal_support", dtype=object),
    )
    for row in rows:
        row.update(validity)
        row["target_normal_support_unique_ids"] = int(len(support_ids))
        row["fuzzy_id_policy"] = "full_11bit" if args.fuzzy_full_id_range else "target_normal_support"
    write_csv(TABLES / "target_normal_synthetic_statistics.csv", rows)

    config = {
        "seed": args.seed,
        "ratio": args.ratio,
        "source": "can-train calibration normal windows only",
        "output": "datasets/synthetic/target_normal_rule_windows.npz",
        "total_windows": int(len(y_binary_arr)),
        "target_normal_support_unique_ids": int(len(support_ids)),
        "fuzzy_id_policy": "full_11bit" if args.fuzzy_full_id_range else "target_normal_support",
        "validity": validity,
        "usage": "diagnostic target-normal synthetic attack training arm",
    }
    (EXP / "target_normal_rule_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"output": str(TARGET_SYNTH_PATH), "total_windows": int(len(y_binary_arr)), "validity": validity}, indent=2))


if __name__ == "__main__":
    main()
