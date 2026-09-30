#!/usr/bin/env python3
import csv
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed"
OUT_TABLES = ROOT / "results" / "tables"


def compact(counter):
    return ";".join(f"{k}:{v}" for k, v in sorted(counter.items(), key=lambda item: str(item[0])))


def validate_file(path):
    pf = pq.ParquetFile(path)
    rows = []
    source_counts = Counter()
    binary_counts = Counter()
    attack_counts = Counter()
    weak_counts = Counter()
    mismatch_count = 0
    invalid_dlc_count = 0

    for batch in pf.iter_batches(
        columns=[
            "source_file",
            "binary_label",
            "attack_type",
            "is_weak_label",
            "payload_len_mismatch",
            "invalid_dlc",
        ],
        batch_size=250_000,
    ):
        data = batch.to_pydict()
        source_counts.update(data["source_file"])
        binary_counts.update(data["binary_label"])
        attack_counts.update(data["attack_type"])
        weak_counts.update(str(v) for v in data["is_weak_label"])
        mismatch_count += sum(1 for v in data["payload_len_mismatch"] if v)
        invalid_dlc_count += sum(1 for v in data["invalid_dlc"] if v)

    rows.append(
        {
            "parquet_file": str(path.relative_to(ROOT)),
            "rows": pf.metadata.num_rows,
            "row_groups": pf.metadata.num_row_groups,
            "columns": pf.metadata.num_columns,
            "source_file_distribution": compact(source_counts),
            "binary_distribution": compact(binary_counts),
            "attack_type_distribution": compact(attack_counts),
            "is_weak_label_distribution": compact(weak_counts),
            "payload_len_mismatch": mismatch_count,
            "invalid_dlc": invalid_dlc_count,
        }
    )
    return rows


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in [PROCESSED / "car_hacking.parquet", PROCESSED / "otids.parquet"]:
        rows.extend(validate_file(path))

    out = OUT_TABLES / "preprocessing_validation.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
