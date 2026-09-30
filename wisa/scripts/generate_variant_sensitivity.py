#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
EXP = ROOT / "experiments" / "01_split"
TABLES = ROOT / "results" / "tables"

SEED = 20260611
WINDOW_SIZE = 128
PER_SCENARIO = 1500
NORMAL_WINDOWS = 12000
FEATURES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]
ATTACK_IDS = {"Normal": 0, "DoS": 1, "Fuzzy": 2, "Gear": 3, "RPM": 4}

SCENARIOS = [
    {"attack_type": "DoS", "severity": "low", "burst_len": 24, "step": 3, "amplitude": 0.25},
    {"attack_type": "DoS", "severity": "medium", "burst_len": 48, "step": 2, "amplitude": 0.50},
    {"attack_type": "DoS", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
    {"attack_type": "Fuzzy", "severity": "low", "burst_len": 24, "step": 4, "amplitude": 0.25},
    {"attack_type": "Fuzzy", "severity": "medium", "burst_len": 48, "step": 3, "amplitude": 0.50},
    {"attack_type": "Fuzzy", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
    {"attack_type": "Gear", "severity": "low", "burst_len": 24, "step": 4, "amplitude": 0.25},
    {"attack_type": "Gear", "severity": "medium", "burst_len": 48, "step": 2, "amplitude": 0.50},
    {"attack_type": "Gear", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
    {"attack_type": "RPM", "severity": "low", "burst_len": 24, "step": 4, "amplitude": 0.25},
    {"attack_type": "RPM", "severity": "medium", "burst_len": 48, "step": 2, "amplitude": 0.50},
    {"attack_type": "RPM", "severity": "high", "burst_len": 96, "step": 1, "amplitude": 1.00},
]


def choose_positions(rng: np.random.Generator, burst_len: int, step: int) -> np.ndarray:
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    return np.arange(start, start + burst_len, step)


def inject_dos(rng: np.random.Generator, x: np.ndarray, burst_len: int, step: int, amplitude: float) -> int:
    pos = choose_positions(rng, burst_len, step)
    x[pos, 0] = 0x000
    x[pos, 1] = 8
    x[pos, 2:10] = 0
    x[pos, 10] = np.maximum(x[pos, 10] * (0.4 - 0.25 * amplitude), 1e-5)
    return len(pos)


def inject_fuzzy(rng: np.random.Generator, x: np.ndarray, burst_len: int, step: int, amplitude: float) -> int:
    pos = choose_positions(rng, burst_len, step)
    x[pos, 0] = rng.integers(0, 2048, size=len(pos))
    x[pos, 1] = 8
    base = x[pos, 2:10].copy()
    random_payload = rng.integers(0, 256, size=(len(pos), 8))
    x[pos, 2:10] = np.clip((1.0 - amplitude) * base + amplitude * random_payload, 0, 255)
    x[pos, 10] = rng.uniform(0.00005, 0.005 + 0.01 * amplitude, size=len(pos))
    return len(pos)


def inject_spoofing(
    rng: np.random.Generator,
    x: np.ndarray,
    can_id: int,
    mode: str,
    burst_len: int,
    step: int,
    amplitude: float,
) -> int:
    pos = choose_positions(rng, burst_len, step)
    x[pos, 0] = can_id
    x[pos, 1] = 8
    scale = 255.0 * amplitude
    ramp = np.linspace(0, scale, len(pos), dtype=np.float32)
    base = x[pos, 2:10].copy()
    if mode == "gear":
        base[:, 2] = np.clip(base[:, 2] * (1.0 - amplitude) + ramp, 0, 255)
        base[:, 3] = np.clip(base[:, 3] * (1.0 - amplitude) + (255 - ramp), 0, 255)
        base[:, 4] = np.clip(base[:, 4] + rng.integers(-10, 11, size=len(pos)) * amplitude, 0, 255)
    else:
        wave = (127.5 + 127.5 * np.sin(np.linspace(0, 2 * np.pi, len(pos)))).astype(np.float32) * amplitude
        base[:, 2] = np.clip(base[:, 2] * (1.0 - amplitude) + wave, 0, 255)
        base[:, 3] = np.clip(base[:, 3] * (1.0 - amplitude) + ramp, 0, 255)
        base[:, 5] = np.clip(base[:, 5] + rng.integers(-15, 16, size=len(pos)) * amplitude, 0, 255)
    x[pos, 2:10] = base
    x[pos, 10] = np.maximum(x[pos, 10], 1e-5)
    return len(pos)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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
    severity = ["normal"] * NORMAL_WINDOWS
    scenario_id = ["Normal_normal"] * NORMAL_WINDOWS
    burst_len_meta = [0] * NORMAL_WINDOWS
    step_meta = [0] * NORMAL_WINDOWS
    amplitude_meta = [0.0] * NORMAL_WINDOWS
    injection_count = [0] * NORMAL_WINDOWS

    rows = []
    cursor = 0
    for scenario in SCENARIOS:
        base = attack_base[cursor : cursor + PER_SCENARIO]
        cursor += PER_SCENARIO
        x_part = x_test[base].copy()
        counts = []
        for i in range(len(x_part)):
            if scenario["attack_type"] == "DoS":
                count = inject_dos(rng, x_part[i], scenario["burst_len"], scenario["step"], scenario["amplitude"])
            elif scenario["attack_type"] == "Fuzzy":
                count = inject_fuzzy(rng, x_part[i], scenario["burst_len"], scenario["step"], scenario["amplitude"])
            elif scenario["attack_type"] == "Gear":
                count = inject_spoofing(rng, x_part[i], 0x43F, "gear", scenario["burst_len"], scenario["step"], scenario["amplitude"])
            elif scenario["attack_type"] == "RPM":
                count = inject_spoofing(rng, x_part[i], 0x316, "rpm", scenario["burst_len"], scenario["step"], scenario["amplitude"])
            else:
                raise ValueError(scenario["attack_type"])
            counts.append(count)
        sid = f"{scenario['attack_type']}_{scenario['severity']}"
        xs.append(x_part)
        y_binary.extend([1] * PER_SCENARIO)
        y_attack.extend([ATTACK_IDS[scenario["attack_type"]]] * PER_SCENARIO)
        attack_type.extend([scenario["attack_type"]] * PER_SCENARIO)
        severity.extend([scenario["severity"]] * PER_SCENARIO)
        scenario_id.extend([sid] * PER_SCENARIO)
        burst_len_meta.extend([scenario["burst_len"]] * PER_SCENARIO)
        step_meta.extend([scenario["step"]] * PER_SCENARIO)
        amplitude_meta.extend([scenario["amplitude"]] * PER_SCENARIO)
        injection_count.extend(counts)
        rows.append(
            {
                "scenario_id": sid,
                "attack_type": scenario["attack_type"],
                "severity": scenario["severity"],
                "windows": PER_SCENARIO,
                "burst_len": scenario["burst_len"],
                "step": scenario["step"],
                "amplitude": scenario["amplitude"],
                "injection_count_mean": float(np.mean(counts)),
                "injection_count_min": int(np.min(counts)),
                "injection_count_max": int(np.max(counts)),
            }
        )

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y_binary = np.asarray(y_binary, dtype=np.int8)
    y_attack = np.asarray(y_attack, dtype=np.int8)
    attack_type = np.asarray(attack_type, dtype=object)
    severity = np.asarray(severity, dtype=object)
    scenario_id = np.asarray(scenario_id, dtype=object)
    burst_len_meta = np.asarray(burst_len_meta, dtype=np.int16)
    step_meta = np.asarray(step_meta, dtype=np.int16)
    amplitude_meta = np.asarray(amplitude_meta, dtype=np.float32)
    injection_count = np.asarray(injection_count, dtype=np.int16)
    order = rng.permutation(len(y_binary))

    out = WINDOWS / "variant_sensitivity_windows.npz"
    np.savez_compressed(
        out,
        x=x[order],
        y_binary=y_binary[order],
        y_attack_type=y_attack[order],
        attack_type=attack_type[order],
        severity=severity[order],
        scenario_id=scenario_id[order],
        burst_len=burst_len_meta[order],
        step=step_meta[order],
        amplitude=amplitude_meta[order],
        injection_count=injection_count[order],
        feature_names=np.asarray(FEATURES, dtype=object),
        window_size=np.asarray(WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(32, dtype=np.int32),
        seed=np.asarray(SEED, dtype=np.int32),
    )

    rows.insert(
        0,
        {
            "scenario_id": "Normal_normal",
            "attack_type": "Normal",
            "severity": "normal",
            "windows": NORMAL_WINDOWS,
            "burst_len": 0,
            "step": 0,
            "amplitude": 0.0,
            "injection_count_mean": 0.0,
            "injection_count_min": 0,
            "injection_count_max": 0,
        },
    )
    write_csv(TABLES / "variant_sensitivity_statistics.csv", rows)

    config = {
        "seed": SEED,
        "source": "held-out Car-Hacking Real Test normal windows only",
        "output": "datasets/windows/variant_sensitivity_windows.npz",
        "normal_windows": NORMAL_WINDOWS,
        "per_scenario_windows": PER_SCENARIO,
        "window_size": WINDOW_SIZE,
        "stride": 32,
        "scenarios": SCENARIOS,
        "usage": "evaluation only; never used for training, generator fitting, threshold fitting, or model selection",
    }
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "variant_sensitivity_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {out}")
    print(f"total windows={len(y_binary)} scenarios={len(rows)}")


if __name__ == "__main__":
    main()
