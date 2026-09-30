#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
EXP_SPLIT = ROOT / "experiments" / "01_split"
TABLES = ROOT / "results" / "tables"

SEED = 42
PER_ATTACK = 8000
NORMAL_WINDOWS = PER_ATTACK * 4
WINDOW_SIZE = 128
FEATURES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]
ATTACK_IDS = {"Normal": 0, "DoS": 1, "Fuzzy": 2, "Gear": 3, "RPM": 4}


def choose_burst(rng, min_len=40, max_len=96):
    burst_len = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    return start, start + burst_len


def inject_dos(rng, x):
    start, end = choose_burst(rng, 48, 96)
    step = int(rng.choice([1, 2]))
    positions = np.arange(start, end, step)
    x[positions, 0] = 0x000
    x[positions, 1] = 8
    x[positions, 2:10] = 0
    x[positions, 10] = np.maximum(x[positions, 10] * 0.25, 1e-5)
    return len(positions)


def inject_fuzzy(rng, x):
    start, end = choose_burst(rng, 40, 96)
    step = int(rng.choice([1, 2, 3]))
    positions = np.arange(start, end, step)
    x[positions, 0] = rng.integers(0, 2048, size=len(positions))
    x[positions, 1] = 8
    x[positions, 2:10] = rng.integers(0, 256, size=(len(positions), 8))
    x[positions, 10] = rng.uniform(0.00005, 0.015, size=len(positions))
    return len(positions)


def inject_spoofing(rng, x, can_id, mode):
    start, end = choose_burst(rng, 48, 96)
    positions = np.arange(start, end, 2)
    x[positions, 0] = can_id
    x[positions, 1] = 8
    ramp = np.linspace(0, 255, len(positions), dtype=np.float32)
    if mode == "gear":
        x[positions, 2] = ramp
        x[positions, 3] = 255 - ramp
        x[positions, 4] = np.clip(x[positions, 4] + rng.integers(-20, 21, size=len(positions)), 0, 255)
    else:
        wave = (127.5 + 127.5 * np.sin(np.linspace(0, 2 * np.pi, len(positions)))).astype(np.float32)
        x[positions, 2] = wave
        x[positions, 3] = ramp
        x[positions, 5] = np.clip(x[positions, 5] + rng.integers(-30, 31, size=len(positions)), 0, 255)
    return len(positions)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = np.random.default_rng(SEED)
    test = np.load(WINDOWS / "test_windows.npz", allow_pickle=True)
    x_test = test["x"].astype(np.float32)
    normal_idx = np.where(test["y_binary"] == 0)[0]
    if len(normal_idx) < NORMAL_WINDOWS + PER_ATTACK * 4:
        raise RuntimeError("not enough normal windows for variant generation")
    selected = rng.choice(normal_idx, size=NORMAL_WINDOWS + PER_ATTACK * 4, replace=False)
    normal_base = selected[:NORMAL_WINDOWS]
    attack_base = selected[NORMAL_WINDOWS:]

    xs = []
    y_binary = []
    y_attack = []
    variant_type = []
    injection_counts = []

    xs.append(x_test[normal_base].copy())
    y_binary.extend([0] * NORMAL_WINDOWS)
    y_attack.extend([0] * NORMAL_WINDOWS)
    variant_type.extend(["Normal"] * NORMAL_WINDOWS)
    injection_counts.extend([0] * NORMAL_WINDOWS)

    attack_specs = [
        ("DoS", ATTACK_IDS["DoS"], lambda r, x: inject_dos(r, x)),
        ("Fuzzy", ATTACK_IDS["Fuzzy"], lambda r, x: inject_fuzzy(r, x)),
        ("Gear", ATTACK_IDS["Gear"], lambda r, x: inject_spoofing(r, x, 0x43F, "gear")),
        ("RPM", ATTACK_IDS["RPM"], lambda r, x: inject_spoofing(r, x, 0x316, "rpm")),
    ]

    cursor = 0
    for name, label, injector in attack_specs:
        base = attack_base[cursor : cursor + PER_ATTACK]
        cursor += PER_ATTACK
        x_part = x_test[base].copy()
        counts = []
        for i in range(len(x_part)):
            counts.append(injector(rng, x_part[i]))
        xs.append(x_part)
        y_binary.extend([1] * PER_ATTACK)
        y_attack.extend([label] * PER_ATTACK)
        variant_type.extend([name] * PER_ATTACK)
        injection_counts.extend(counts)

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y_binary = np.asarray(y_binary, dtype=np.int8)
    y_attack = np.asarray(y_attack, dtype=np.int8)
    variant_type = np.asarray(variant_type, dtype=object)
    injection_counts = np.asarray(injection_counts, dtype=np.int16)

    order = rng.permutation(len(y_binary))
    out = WINDOWS / "variant_test_windows.npz"
    np.savez_compressed(
        out,
        x=x[order],
        y_binary=y_binary[order],
        y_attack_type=y_attack[order],
        variant_type=variant_type[order],
        injection_count=injection_counts[order],
        feature_names=np.asarray(FEATURES, dtype=object),
        window_size=np.asarray(WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(32, dtype=np.int32),
        seed=np.asarray(SEED, dtype=np.int32),
    )

    rows = []
    for name in ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]:
        mask = variant_type == name
        rows.append(
            {
                "variant_type": name,
                "windows": int(mask.sum()),
                "binary_label": "Attack" if name != "Normal" else "Normal",
                "attack_type_id": ATTACK_IDS[name],
                "injection_count_mean": float(injection_counts[mask].mean()) if mask.any() else 0.0,
                "injection_count_min": int(injection_counts[mask].min()) if mask.any() else 0,
                "injection_count_max": int(injection_counts[mask].max()) if mask.any() else 0,
            }
        )
    write_csv(TABLES / "variant_test_statistics.csv", rows)

    config = {
        "seed": SEED,
        "source": "held-out Car-Hacking Real Test normal windows only",
        "output": "datasets/windows/variant_test_windows.npz",
        "normal_windows": NORMAL_WINDOWS,
        "per_attack_windows": PER_ATTACK,
        "window_size": WINDOW_SIZE,
        "stride": 32,
        "variants": {
            "DoS": "set burst positions to CAN ID 0x000 with zero payload and shortened delta_t",
            "Fuzzy": "random CAN IDs in 0..2047 with random 8-byte payload and variable delta_t",
            "Gear": "CAN ID 0x43f with gradual payload byte transition while preserving timing",
            "RPM": "CAN ID 0x316 with sinusoidal/ramp payload transition while preserving timing",
        },
        "usage": "evaluation only; never used for training or synthetic generator fitting",
    }
    EXP_SPLIT.mkdir(parents=True, exist_ok=True)
    (EXP_SPLIT / "variant_generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"output": str(out), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
