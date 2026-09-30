#!/usr/bin/env python3
import bisect
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed"
OUT_SPLIT = ROOT / "experiments" / "01_split"
OUT_TABLES = ROOT / "results" / "tables"

CAR_PATH = PROCESSED / "car_hacking.parquet"
OTIDS_PATH = PROCESSED / "otids.parquet"

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
ATTACK_EPISODE_GAP_SECONDS = 1.0
BATCH_SIZE = 250_000


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_source_totals(path):
    totals = Counter()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["source_file"], batch_size=BATCH_SIZE):
        totals.update(batch.column(0).to_pylist())
    return totals


def discover_attack_episodes(path):
    pf = pq.ParquetFile(path)
    current = {}
    episodes = defaultdict(list)
    prev_attack_ts = {}
    prev_attack_row = {}

    for batch in pf.iter_batches(columns=["source_file", "binary_label", "row_in_source", "timestamp"], batch_size=BATCH_SIZE):
        data = batch.to_pydict()
        for source, label, row, ts in zip(data["source_file"], data["binary_label"], data["row_in_source"], data["timestamp"]):
            if label != "Attack":
                continue
            start_new = source not in current or (ts - prev_attack_ts[source]) > ATTACK_EPISODE_GAP_SECONDS
            if start_new:
                if source in current:
                    episodes[source].append(current[source])
                current[source] = {
                    "source_file": source,
                    "episode_index": len(episodes[source]),
                    "start_row": row,
                    "end_row": row,
                    "start_timestamp": ts,
                    "end_timestamp": ts,
                    "attack_rows": 1,
                }
            else:
                current[source]["end_row"] = row
                current[source]["end_timestamp"] = ts
                current[source]["attack_rows"] += 1
            prev_attack_ts[source] = ts
            prev_attack_row[source] = row

    for source, episode in current.items():
        episodes[source].append(episode)
    return episodes


def episode_split_lookup(episodes):
    lookups = {}
    episode_rows = []
    for source, source_episodes in episodes.items():
        n = len(source_episodes)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        starts = [ep["start_row"] for ep in source_episodes]
        for i, ep in enumerate(source_episodes):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"
            ep = dict(ep)
            ep["split"] = split
            episode_rows.append(ep)
            source_episodes[i]["split"] = split
        lookups[source] = {"starts": starts, "episodes": source_episodes}
    return lookups, episode_rows


def assign_split(source, row, source_totals, episode_lookups):
    if source in episode_lookups:
        starts = episode_lookups[source]["starts"]
        episodes = episode_lookups[source]["episodes"]
        idx = bisect.bisect_right(starts, row) - 1
        if idx < 0:
            idx = 0
        if idx >= len(episodes):
            idx = len(episodes) - 1
        return episodes[idx]["split"]

    total = source_totals[source]
    frac = row / total
    if frac < TRAIN_RATIO:
        return "train"
    if frac < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


def output_schema(base_schema):
    return base_schema.append(pa.field("split", pa.string()))


def write_batch(path, rows, schema, writer):
    table = pa.Table.from_pylist(rows, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(path, schema, compression="zstd")
    writer.write_table(table)
    return writer


def split_car_hacking(source_totals, episode_lookups):
    pf = pq.ParquetFile(CAR_PATH)
    schema = output_schema(pf.schema_arrow)
    paths = {
        "train": PROCESSED / "car_hacking_train.parquet",
        "val": PROCESSED / "car_hacking_val.parquet",
        "test": PROCESSED / "car_hacking_test.parquet",
    }
    writers = {split: None for split in paths}
    buffers = {split: [] for split in paths}
    stats = defaultdict(Counter)
    source_stats = defaultdict(Counter)

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        rows = batch.to_pylist()
        for row in rows:
            split = assign_split(row["source_file"], row["row_in_source"], source_totals, episode_lookups)
            row["split"] = split
            buffers[split].append(row)
            stats[split]["rows"] += 1
            stats[split][f"binary:{row['binary_label']}"] += 1
            stats[split][f"attack_type:{row['attack_type']}"] += 1
            source_stats[(split, row["source_file"])]["rows"] += 1
            source_stats[(split, row["source_file"])][f"binary:{row['binary_label']}"] += 1
            if len(buffers[split]) >= BATCH_SIZE:
                writers[split] = write_batch(paths[split], buffers[split], schema, writers[split])
                buffers[split].clear()

    for split in paths:
        if buffers[split]:
            writers[split] = write_batch(paths[split], buffers[split], schema, writers[split])
            buffers[split].clear()
        if writers[split] is not None:
            writers[split].close()

    return stats, source_stats


def write_otids_cross_test():
    pf = pq.ParquetFile(OTIDS_PATH)
    schema = output_schema(pf.schema_arrow)
    out = PROCESSED / "otids_cross_test.parquet"
    writer = None
    stats = Counter()
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        rows = batch.to_pylist()
        for row in rows:
            row["split"] = "cross_test"
            stats["rows"] += 1
            stats[f"binary:{row['binary_label']}"] += 1
            stats[f"attack_type:{row['attack_type']}"] += 1
        writer = write_batch(out, rows, schema, writer)
    if writer is not None:
        writer.close()
    return stats


def stats_to_rows(stats, source_stats, otids_stats):
    rows = []
    for split in ["train", "val", "test"]:
        counter = stats[split]
        rows.append(
            {
                "dataset": "Car-Hacking",
                "split": split,
                "source_file": "ALL",
                "rows": counter["rows"],
                "normal_rows": counter["binary:Normal"],
                "attack_rows": counter["binary:Attack"],
                "attack_type_distribution": compact_prefixed(counter, "attack_type:"),
            }
        )
    for (split, source), counter in sorted(source_stats.items()):
        rows.append(
            {
                "dataset": "Car-Hacking",
                "split": split,
                "source_file": source,
                "rows": counter["rows"],
                "normal_rows": counter["binary:Normal"],
                "attack_rows": counter["binary:Attack"],
                "attack_type_distribution": "",
            }
        )
    rows.append(
        {
            "dataset": "OTIDS",
            "split": "cross_test",
            "source_file": "ALL",
            "rows": otids_stats["rows"],
            "normal_rows": otids_stats["binary:Normal"],
            "attack_rows": otids_stats["binary:Attack"],
            "attack_type_distribution": compact_prefixed(otids_stats, "attack_type:"),
        }
    )
    return rows


def compact_prefixed(counter, prefix):
    items = []
    for key, value in sorted(counter.items()):
        if key.startswith(prefix):
            items.append(f"{key[len(prefix):]}:{value}")
    return ";".join(items)


def main():
    OUT_SPLIT.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    source_totals = discover_source_totals(CAR_PATH)
    episodes = discover_attack_episodes(CAR_PATH)
    episode_lookups, episode_rows = episode_split_lookup(episodes)

    split_stats, source_stats = split_car_hacking(source_totals, episode_lookups)
    otids_stats = write_otids_cross_test()
    split_rows = stats_to_rows(split_stats, source_stats, otids_stats)

    write_csv(
        OUT_SPLIT / "attack_episode_summary.csv",
        episode_rows,
        fieldnames=[
            "source_file",
            "episode_index",
            "start_row",
            "end_row",
            "start_timestamp",
            "end_timestamp",
            "attack_rows",
            "split",
        ],
    )
    write_csv(OUT_TABLES / "split_statistics.csv", split_rows)

    manifest = {
        "strategy": "source-local chronological attack-episode split",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "attack_episode_gap_seconds": ATTACK_EPISODE_GAP_SECONDS,
        "normal_only_source_strategy": "source-local contiguous row split",
        "otids_strategy": "entire dataset copied to cross_test only",
        "car_hacking_source_totals": dict(source_totals),
        "attack_episode_counts": {source: len(eps) for source, eps in episodes.items()},
        "outputs": {
            "train": "datasets/processed/car_hacking_train.parquet",
            "val": "datasets/processed/car_hacking_val.parquet",
            "test": "datasets/processed/car_hacking_test.parquet",
            "cross_test": "datasets/processed/otids_cross_test.parquet",
        },
    }
    (OUT_SPLIT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
