#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
EXP = ROOT / "experiments" / "01_split"
TABLES = ROOT / "results" / "tables"

SEED = 20260613
WINDOW_SIZE = 128
NORMAL_WINDOWS = 8000
PER_SCENARIO = 4000
FEATURES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]
ATTACK_IDS = {"Normal": 0, "Gear": 3, "RPM": 4}

SCENARIOS = [
    {
        "scenario_id": "Gear_payload_shift_canonical",
        "attack_type": "Gear",
        "target_id": 0x43F,
        "id_condition": "canonical",
        "mode": "gear",
    },
    {
        "scenario_id": "Gear_payload_shifted_0x440",
        "attack_type": "Gear",
        "target_id": 0x440,
        "id_condition": "shifted",
        "mode": "gear",
    },
    {
        "scenario_id": "RPM_payload_shift_canonical",
        "attack_type": "RPM",
        "target_id": 0x316,
        "id_condition": "canonical",
        "mode": "rpm",
    },
    {
        "scenario_id": "RPM_payload_shifted_0x329",
        "attack_type": "RPM",
        "target_id": 0x329,
        "id_condition": "shifted",
        "mode": "rpm",
    },
]


def choose_positions(rng: np.random.Generator) -> np.ndarray:
    burst_len = 96
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    return np.arange(start, start + burst_len, 3)


def inject_payload_shift(rng: np.random.Generator, x: np.ndarray, target_id: int, mode: str) -> int:
    pos = choose_positions(rng)
    x[pos, 0] = target_id
    x[pos, 1] = 8
    base = x[pos, 2:10].copy()
    ramp = np.linspace(0, 255, len(pos), dtype=np.float32)
    inv_ramp = 255 - ramp
    if mode == "gear":
        base[:, 6] = ramp
        base[:, 7] = np.clip(inv_ramp + rng.integers(-10, 11, size=len(pos)), 0, 255)
    else:
        wave = (127.5 + 127.5 * np.sin(np.linspace(0, 3 * np.pi, len(pos)))).astype(np.float32)
        saw = (np.arange(len(pos), dtype=np.float32) * 37) % 256
        base[:, 4] = saw
        base[:, 6] = wave
        base[:, 7] = np.clip(ramp + rng.integers(-12, 13, size=len(pos)), 0, 255)
    x[pos, 2:10] = base
    x[pos, 10] = np.maximum(x[pos, 10], 1e-5)
    return len(pos)


def main() -> None:
    rng = np.random.default_rng(SEED)
    test = np.load(WINDOWS / "test_windows.npz", allow_pickle=True)
    x_test = test["x"].astype(np.float32)
    normal_idx = np.where(test["y_binary"] == 0)[0]
    required = NORMAL_WINDOWS + PER_SCENARIO * len(SCENARIOS)
    if len(normal_idx) < required:
        raise RuntimeError(f"not enough held-out normal windows: need {required}, got {len(normal_idx)}")

    selected = rng.choice(normal_idx, size=required, replace=False)
    normal_base = selected[:NORMAL_WINDOWS]
    attack_base = selected[NORMAL_WINDOWS:]

    xs = [x_test[normal_base].copy()]
    y_binary = [0] * NORMAL_WINDOWS
    y_attack = [0] * NORMAL_WINDOWS
    attack_type = ["Normal"] * NORMAL_WINDOWS
    scenario_id = ["Normal_normal"] * NORMAL_WINDOWS
    id_condition = ["normal"] * NORMAL_WINDOWS
    target_id = [0] * NORMAL_WINDOWS
    injection_count = [0] * NORMAL_WINDOWS
    rows = [
        {
            "scenario_id": "Normal_normal",
            "attack_type": "Normal",
            "id_condition": "normal",
            "target_id_hex": "",
            "windows": NORMAL_WINDOWS,
            "injection_count_mean": 0.0,
        }
    ]

    cursor = 0
    for scenario in SCENARIOS:
        base = attack_base[cursor : cursor + PER_SCENARIO]
        cursor += PER_SCENARIO
        x_part = x_test[base].copy()
        counts = [inject_payload_shift(rng, x_part[i], scenario["target_id"], scenario["mode"]) for i in range(len(x_part))]
        xs.append(x_part)
        y_binary.extend([1] * PER_SCENARIO)
        y_attack.extend([ATTACK_IDS[scenario["attack_type"]]] * PER_SCENARIO)
        attack_type.extend([scenario["attack_type"]] * PER_SCENARIO)
        scenario_id.extend([scenario["scenario_id"]] * PER_SCENARIO)
        id_condition.extend([scenario["id_condition"]] * PER_SCENARIO)
        target_id.extend([scenario["target_id"]] * PER_SCENARIO)
        injection_count.extend(counts)
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "attack_type": scenario["attack_type"],
                "id_condition": scenario["id_condition"],
                "target_id_hex": f"0x{scenario['target_id']:03x}",
                "windows": PER_SCENARIO,
                "injection_count_mean": float(np.mean(counts)),
            }
        )

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y_binary = np.asarray(y_binary, dtype=np.int8)
    y_attack = np.asarray(y_attack, dtype=np.int8)
    order = rng.permutation(len(y_binary))

    out = WINDOWS / "out_of_generator_stress_windows.npz"
    np.savez_compressed(
        out,
        x=x[order],
        y_binary=y_binary[order],
        y_attack_type=y_attack[order],
        attack_type=np.asarray(attack_type, dtype=object)[order],
        scenario_id=np.asarray(scenario_id, dtype=object)[order],
        id_condition=np.asarray(id_condition, dtype=object)[order],
        target_id=np.asarray(target_id, dtype=np.int32)[order],
        injection_count=np.asarray(injection_count, dtype=np.int16)[order],
        feature_names=np.asarray(FEATURES, dtype=object),
        window_size=np.asarray(WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(32, dtype=np.int32),
        seed=np.asarray(SEED, dtype=np.int32),
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "out_of_generator_stress_statistics.csv", index=False)
    config = {
        "seed": SEED,
        "source": "held-out Car-Hacking Real Test normal windows only",
        "output": "datasets/windows/out_of_generator_stress_windows.npz",
        "normal_windows": NORMAL_WINDOWS,
        "per_scenario_windows": PER_SCENARIO,
        "usage": "evaluation only; payload-position stress test outside the disclosed training and fixed/sensitivity rules",
        "scenarios": [
            {**s, "target_id_hex": f"0x{s['target_id']:03x}", "target_id": int(s["target_id"])}
            for s in SCENARIOS
        ],
        "payload_shift": "Gear uses bytes 6/7; RPM uses bytes 4/6/7. These byte positions are outside the Gear/RPM training synthetic payload positions.",
    }
    (EXP / "out_of_generator_stress_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {out} windows={len(y_binary)}")


if __name__ == "__main__":
    main()
