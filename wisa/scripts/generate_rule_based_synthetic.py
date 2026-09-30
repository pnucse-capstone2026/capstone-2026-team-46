#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
EXP = ROOT / "experiments" / "03_synthetic"
TABLES = ROOT / "results" / "tables"

SEED = 314159
PER_ATTACK = 65_000
WINDOW_SIZE = 128
STRIDE = 32
FEATURES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]
ATTACK_IDS = {"DoS": 1, "Fuzzy": 2, "Gear": 3, "RPM": 4}


def choose_positions(rng, min_len, max_len, step_options):
    burst_len = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    step = int(rng.choice(step_options))
    return np.arange(start, start + burst_len, step)


def synth_dos(rng, x):
    pos = choose_positions(rng, 32, 90, [1, 2])
    # Use the canonical DoS ID, but vary payload and timing slightly so this is
    # not identical to the variant test generator.
    x[pos, 0] = 0x000
    x[pos, 1] = 8
    x[pos, 2:10] = rng.choice([0, 255], size=(len(pos), 8), p=[0.92, 0.08])
    x[pos, 10] = rng.uniform(0.00003, 0.002, size=len(pos))
    return len(pos)


def synth_fuzzy(rng, x):
    pos = choose_positions(rng, 24, 88, [1, 2, 3, 4])
    x[pos, 0] = rng.integers(0, 2048, size=len(pos))
    dlc = rng.choice([2, 5, 8], size=len(pos), p=[0.08, 0.12, 0.80])
    x[pos, 1] = dlc
    payload = rng.integers(0, 256, size=(len(pos), 8))
    for i, d in enumerate(dlc):
        if d < 8:
            payload[i, d:] = 0
    x[pos, 2:10] = payload
    x[pos, 10] = rng.uniform(0.00005, 0.02, size=len(pos))
    return len(pos)


def synth_spoof(rng, x, can_id, mode):
    pos = choose_positions(rng, 32, 90, [1, 2, 3])
    x[pos, 0] = can_id
    x[pos, 1] = 8
    base = x[pos, 2:10].copy()
    if mode == "gear":
        levels = rng.choice([0, 1, 2, 3, 4, 5], size=len(pos))
        base[:, 0] = np.clip(levels * 40 + rng.integers(0, 16, size=len(pos)), 0, 255)
        base[:, 1] = np.clip(255 - base[:, 0] + rng.integers(-8, 9, size=len(pos)), 0, 255)
    else:
        rpm = rng.integers(0, 8000, size=len(pos))
        base[:, 0] = (rpm // 32) % 256
        base[:, 1] = (rpm // 4) % 256
        base[:, 2] = np.clip(base[:, 2] + rng.integers(-50, 51, size=len(pos)), 0, 255)
    x[pos, 2:10] = base
    # Spoofing keeps timing normal-like.
    x[pos, 10] = np.maximum(x[pos, 10], 1e-5)
    return len(pos)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate(x):
    return {
        "can_id_min": float(x[:, :, 0].min()),
        "can_id_max": float(x[:, :, 0].max()),
        "dlc_min": float(x[:, :, 1].min()),
        "dlc_max": float(x[:, :, 1].max()),
        "payload_min": float(x[:, :, 2:10].min()),
        "payload_max": float(x[:, :, 2:10].max()),
        "delta_t_min": float(x[:, :, 10].min()),
        "delta_t_max": float(x[:, :, 10].max()),
        "invalid_can_id": int(((x[:, :, 0] < 0) | (x[:, :, 0] > 2047)).sum()),
        "invalid_dlc": int(((x[:, :, 1] < 0) | (x[:, :, 1] > 8)).sum()),
        "invalid_payload": int(((x[:, :, 2:10] < 0) | (x[:, :, 2:10] > 255)).sum()),
        "invalid_delta_t": int((x[:, :, 10] < 0).sum()),
    }


def main():
    rng = np.random.default_rng(SEED)
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    normal_idx = np.where(train["y_binary"] == 0)[0]
    if len(normal_idx) == 0:
        raise RuntimeError("no train normal windows available")

    xs = []
    y_binary = []
    y_attack = []
    synthetic_type = []
    injection_counts = []

    specs = [
        ("DoS", ATTACK_IDS["DoS"], synth_dos),
        ("Fuzzy", ATTACK_IDS["Fuzzy"], synth_fuzzy),
        ("Gear", ATTACK_IDS["Gear"], lambda r, x: synth_spoof(r, x, 0x43F, "gear")),
        ("RPM", ATTACK_IDS["RPM"], lambda r, x: synth_spoof(r, x, 0x316, "rpm")),
    ]
    for name, label, fn in specs:
        base = rng.choice(normal_idx, size=PER_ATTACK, replace=True)
        x_part = train["x"][base].astype(np.float32).copy()
        counts = []
        for i in range(PER_ATTACK):
            counts.append(fn(rng, x_part[i]))
        xs.append(x_part)
        y_binary.extend([1] * PER_ATTACK)
        y_attack.extend([label] * PER_ATTACK)
        synthetic_type.extend([name] * PER_ATTACK)
        injection_counts.extend(counts)

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y_binary = np.asarray(y_binary, dtype=np.int8)
    y_attack = np.asarray(y_attack, dtype=np.int8)
    synthetic_type = np.asarray(synthetic_type, dtype=object)
    injection_counts = np.asarray(injection_counts, dtype=np.int16)
    order = rng.permutation(len(y_binary))

    out = SYNTHETIC / "rule_based_windows.npz"
    np.savez_compressed(
        out,
        x=x[order],
        y_binary=y_binary[order],
        y_attack_type=y_attack[order],
        synthetic_type=synthetic_type[order],
        injection_count=injection_counts[order],
        feature_names=np.asarray(FEATURES, dtype=object),
        window_size=np.asarray(WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(STRIDE, dtype=np.int32),
        seed=np.asarray(SEED, dtype=np.int32),
    )

    rows = []
    for name, label, _ in specs:
        mask = synthetic_type == name
        rows.append(
            {
                "generator": "rule_based",
                "synthetic_type": name,
                "windows": int(mask.sum()),
                "binary_label": "Attack",
                "attack_type_id": label,
                "injection_count_mean": float(injection_counts[mask].mean()),
                "injection_count_min": int(injection_counts[mask].min()),
                "injection_count_max": int(injection_counts[mask].max()),
            }
        )
    validity = validate(x)
    for row in rows:
        row.update(validity)
    write_csv(TABLES / "rule_based_synthetic_statistics.csv", rows)

    config = {
        "seed": SEED,
        "source": "Car-Hacking train normal windows only",
        "output": "datasets/synthetic/rule_based_windows.npz",
        "per_attack_windows": PER_ATTACK,
        "total_windows": int(len(y_binary)),
        "usage": "training augmentation only",
        "variants": {
            "DoS": "CAN ID 0x000 burst with mostly zero payload and shortened delta_t",
            "Fuzzy": "random CAN ID, variable DLC, random payload, variable delta_t",
            "Gear": "CAN ID 0x43f payload level mutation with normal-like timing",
            "RPM": "CAN ID 0x316 payload mutation based on random RPM-like values with normal-like timing",
        },
        "validity": validity,
    }
    (EXP / "rule_based_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"output": str(out), "total_windows": int(len(y_binary)), "validity": validity}, indent=2))


if __name__ == "__main__":
    main()
