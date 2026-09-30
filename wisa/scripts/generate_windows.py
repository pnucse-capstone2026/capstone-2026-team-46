#!/usr/bin/env python3
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed"
WINDOWS = ROOT / "datasets" / "windows"
OUT_TABLES = ROOT / "results" / "tables"
OUT_LOGS = ROOT / "results" / "logs"

FEATURE_COLUMNS = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"]
PARQUET_COLUMNS = [
    "source_file",
    "segment_id",
    "timestamp",
    *FEATURE_COLUMNS,
    "binary_label_id",
    "attack_type_id",
    "is_weak_label",
]

INPUTS = [
    ("train", PROCESSED / "car_hacking_train.parquet", WINDOWS / "train_windows.npz"),
    ("val", PROCESSED / "car_hacking_val.parquet", WINDOWS / "val_windows.npz"),
    ("test", PROCESSED / "car_hacking_test.parquet", WINDOWS / "test_windows.npz"),
    ("otids_cross", PROCESSED / "otids_cross_test.parquet", WINDOWS / "otids_cross_windows.npz"),
]


def load_config():
    with (ROOT / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    return cfg["windowing"]["window_size"], cfg["windowing"]["stride"]


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_groups(path):
    pf = pq.ParquetFile(path)
    current_key = None
    buffers = defaultdict(list)

    for batch in pf.iter_batches(columns=PARQUET_COLUMNS, batch_size=500_000):
        data = batch.to_pydict()
        n = len(data["timestamp"])
        for i in range(n):
            key = (data["source_file"][i], data["segment_id"][i])
            if current_key is None:
                current_key = key
            if key != current_key:
                yield current_key, buffers
                buffers = defaultdict(list)
                current_key = key
            for col in PARQUET_COLUMNS:
                buffers[col].append(data[col][i])
    if current_key is not None:
        yield current_key, buffers


def group_to_arrays(group):
    timestamps = np.asarray(group["timestamp"], dtype=np.float64)
    base_features = np.column_stack([np.asarray(group[col]) for col in FEATURE_COLUMNS]).astype(np.float32)
    delta_t = np.zeros((len(timestamps), 1), dtype=np.float32)
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        dt = np.maximum(dt, 0.0)
        delta_t[1:, 0] = dt.astype(np.float32)
    features = np.concatenate([base_features, delta_t], axis=1)
    binary = np.asarray(group["binary_label_id"], dtype=np.int8)
    attack = np.asarray(group["attack_type_id"], dtype=np.int8)
    weak = np.asarray(group["is_weak_label"], dtype=np.bool_)
    return timestamps, features, binary, attack, weak


def majority_attack_label(attack_labels):
    attack_only = attack_labels[attack_labels > 0]
    if len(attack_only) == 0:
        return 0
    values, counts = np.unique(attack_only, return_counts=True)
    return int(values[np.argmax(counts)])


def make_windows_for_group(split_name, key, group, window_size, stride):
    source_file, segment_id = key
    timestamps, features, binary, attack, weak = group_to_arrays(group)
    n = len(features)
    if n < window_size:
        return None

    xs = []
    y_binary = []
    y_attack = []
    starts = []
    ends = []
    start_ts = []
    end_ts = []
    weak_window = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        win_binary = binary[start:end]
        win_attack = attack[start:end]
        label_binary = int(np.any(win_binary == 1))
        label_attack = majority_attack_label(win_attack) if label_binary else 0
        xs.append(features[start:end])
        y_binary.append(label_binary)
        y_attack.append(label_attack)
        starts.append(start)
        ends.append(end - 1)
        start_ts.append(timestamps[start])
        end_ts.append(timestamps[end - 1])
        weak_window.append(bool(np.any(weak[start:end])))

    if not xs:
        return None
    return {
        "x": np.stack(xs).astype(np.float32),
        "y_binary": np.asarray(y_binary, dtype=np.int8),
        "y_attack_type": np.asarray(y_attack, dtype=np.int8),
        "source_file": np.asarray([source_file] * len(xs), dtype=object),
        "segment_id": np.asarray([segment_id] * len(xs), dtype=np.int32),
        "start_index": np.asarray(starts, dtype=np.int64),
        "end_index": np.asarray(ends, dtype=np.int64),
        "start_timestamp": np.asarray(start_ts, dtype=np.float64),
        "end_timestamp": np.asarray(end_ts, dtype=np.float64),
        "is_weak_label": np.asarray(weak_window, dtype=np.bool_),
    }


def concat_parts(parts):
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def generate_one(split_name, input_path, output_path, window_size, stride):
    parts = []
    stats = Counter()
    source_stats = defaultdict(Counter)
    groups_seen = 0
    groups_skipped_short = 0

    for key, group in iter_groups(input_path):
        groups_seen += 1
        part = make_windows_for_group(split_name, key, group, window_size, stride)
        if part is None:
            groups_skipped_short += 1
            continue
        parts.append(part)
        n = len(part["y_binary"])
        stats["windows"] += n
        stats["weak_windows"] += int(part["is_weak_label"].sum())
        stats["binary_normal"] += int((part["y_binary"] == 0).sum())
        stats["binary_attack"] += int((part["y_binary"] == 1).sum())
        for label, count in zip(*np.unique(part["y_attack_type"], return_counts=True)):
            stats[f"attack_type_{int(label)}"] += int(count)
        source_stats[key[0]]["windows"] += n
        source_stats[key[0]]["binary_normal"] += int((part["y_binary"] == 0).sum())
        source_stats[key[0]]["binary_attack"] += int((part["y_binary"] == 1).sum())
        print(f"{split_name}: {key[0]} segment={key[1]} windows={n}")

    if not parts:
        raise RuntimeError(f"no windows generated for {input_path}")

    arrays = concat_parts(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        feature_names=np.asarray([*FEATURE_COLUMNS, "delta_t"], dtype=object),
        window_size=np.asarray(window_size, dtype=np.int32),
        stride=np.asarray(stride, dtype=np.int32),
    )
    return stats, source_stats, groups_seen, groups_skipped_short


def attack_distribution(counter):
    names = {0: "Normal", 1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM", 5: "Impersonation"}
    values = []
    for i in range(0, 6):
        key = f"attack_type_{i}"
        if counter[key]:
            values.append(f"{names[i]}:{counter[key]}")
    return ";".join(values)


def main():
    window_size, stride = load_config()
    WINDOWS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    rows = []
    source_rows = []
    manifest = {
        "window_size": window_size,
        "stride": stride,
        "feature_names": [*FEATURE_COLUMNS, "delta_t"],
        "label_rule": "binary attack if any attack frame; multiclass majority over attack frames",
        "inputs": {},
    }

    for split_name, input_path, output_path in INPUTS:
        stats, source_stats, groups_seen, groups_skipped_short = generate_one(
            split_name, input_path, output_path, window_size, stride
        )
        rows.append(
            {
                "split": split_name,
                "windows": stats["windows"],
                "binary_normal": stats["binary_normal"],
                "binary_attack": stats["binary_attack"],
                "weak_windows": stats["weak_windows"],
                "attack_type_distribution": attack_distribution(stats),
                "groups_seen": groups_seen,
                "groups_skipped_short": groups_skipped_short,
                "output": str(output_path.relative_to(ROOT)),
            }
        )
        for source, counter in sorted(source_stats.items()):
            source_rows.append(
                {
                    "split": split_name,
                    "source_file": source,
                    "windows": counter["windows"],
                    "binary_normal": counter["binary_normal"],
                    "binary_attack": counter["binary_attack"],
                }
            )
        manifest["inputs"][split_name] = {
            "input": str(input_path.relative_to(ROOT)),
            "output": str(output_path.relative_to(ROOT)),
            "windows": stats["windows"],
            "binary_normal": stats["binary_normal"],
            "binary_attack": stats["binary_attack"],
            "weak_windows": stats["weak_windows"],
        }

    write_csv(OUT_TABLES / "window_statistics.csv", rows)
    write_csv(OUT_TABLES / "window_source_statistics.csv", source_rows)
    (WINDOWS / "window_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
