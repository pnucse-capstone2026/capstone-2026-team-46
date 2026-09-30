#!/usr/bin/env python3
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
PROCESSED = DATASETS / "processed"
OUT_PROFILE = ROOT / "experiments" / "00_profile"
OUT_SPLIT = ROOT / "experiments" / "01_split"
OUT_TABLES = ROOT / "results" / "tables"

CAR_ZIP = DATASETS / "9) Car-Hacking Dataset.zip"
OTIDS_ZIP = DATASETS / "10) CAN-Intrusion Dataset.zip"

BATCH_SIZE = 200_000

CAR_FILES = [
    ("DoS_dataset.csv", "DoS", "car_attack_csv"),
    ("Fuzzy_dataset.csv", "Fuzzy", "car_attack_csv"),
    ("gear_dataset.csv", "Gear", "car_attack_csv"),
    ("RPM_dataset.csv", "RPM", "car_attack_csv"),
    ("normal_run_data/normal_run_data.txt", "Normal", "txt_all_normal"),
]

OTIDS_FILES = [
    ("Attack_free_dataset.txt", "Normal", "txt_all_normal"),
    ("DoS_attack_dataset.txt", "DoS", "txt_scenario_attack"),
    ("Fuzzy_attack_dataset.txt", "Fuzzy", "txt_scenario_attack"),
    ("Impersonation_attack_dataset.txt", "Impersonation", "txt_scenario_attack"),
]

SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("source_file", pa.string()),
        ("label_mode", pa.string()),
        ("timestamp", pa.float64()),
        ("can_id", pa.uint16()),
        ("dlc", pa.uint8()),
        ("data0", pa.uint8()),
        ("data1", pa.uint8()),
        ("data2", pa.uint8()),
        ("data3", pa.uint8()),
        ("data4", pa.uint8()),
        ("data5", pa.uint8()),
        ("data6", pa.uint8()),
        ("data7", pa.uint8()),
        ("payload_len", pa.uint8()),
        ("payload_len_mismatch", pa.bool_()),
        ("invalid_dlc", pa.bool_()),
        ("binary_label", pa.string()),
        ("binary_label_id", pa.int8()),
        ("attack_type", pa.string()),
        ("attack_type_id", pa.int8()),
        ("original_label", pa.string()),
        ("is_weak_label", pa.bool_()),
        ("segment_id", pa.int32()),
        ("row_in_source", pa.int64()),
        ("row_in_segment", pa.int64()),
    ]
)

ATTACK_TYPE_ID = {
    "Normal": 0,
    "DoS": 1,
    "Fuzzy": 2,
    "Gear": 3,
    "RPM": 4,
    "Impersonation": 5,
    "Unknown": -1,
}

BINARY_LABEL_ID = {
    "Normal": 0,
    "Attack": 1,
    "Unknown": -1,
}


def parse_car_csv_line(line):
    parts = line.rstrip(b"\r\n").split(b",")
    if len(parts) < 5:
        raise ValueError("too few csv fields")
    timestamp = float(parts[0])
    can_id = int(parts[1], 16)
    dlc = int(parts[2])
    payload = [parse_hex_byte(x) for x in parts[3:-1]]
    raw_label = parts[-1].decode("ascii").strip()
    return timestamp, can_id, dlc, payload, raw_label


def parse_txt_line(line):
    parts = line.decode("ascii", errors="strict").strip().split()
    timestamp = float(parts[1])
    can_id = int(parts[3], 16)
    dlc_idx = parts.index("DLC:")
    dlc = int(parts[dlc_idx + 1])
    payload_tokens = parts[dlc_idx + 2 : dlc_idx + 2 + max(dlc, 0)]
    payload = [parse_hex_byte(x.encode("ascii")) for x in payload_tokens]
    return timestamp, can_id, dlc, payload


def parse_hex_byte(value):
    if isinstance(value, bytes):
        text = value.decode("ascii").strip()
    else:
        text = str(value).strip()
    parsed = int(text, 16)
    if parsed < 0 or parsed > 255:
        raise ValueError(f"payload byte out of range: {text}")
    return parsed


def canonical_payload(payload):
    out = list(payload[:8])
    while len(out) < 8:
        out.append(0)
    return out


class Summary:
    def __init__(self, dataset, source_file, label_mode):
        self.dataset = dataset
        self.source_file = source_file
        self.label_mode = label_mode
        self.rows_written = 0
        self.parse_errors = 0
        self.invalid_dlc = 0
        self.payload_len_mismatch = 0
        self.monotonic_breaks = 0
        self.max_segment_id = 0
        self.binary_counts = Counter()
        self.attack_counts = Counter()
        self.original_label_counts = Counter()
        self.dlc_counts = Counter()
        self.segment_rows = Counter()

    def row(self):
        return {
            "dataset": self.dataset,
            "source_file": self.source_file,
            "label_mode": self.label_mode,
            "rows_written": self.rows_written,
            "parse_errors": self.parse_errors,
            "invalid_dlc": self.invalid_dlc,
            "payload_len_mismatch": self.payload_len_mismatch,
            "monotonic_breaks": self.monotonic_breaks,
            "segments": self.max_segment_id + 1 if self.rows_written else 0,
            "binary_distribution": compact(self.binary_counts),
            "attack_type_distribution": compact(self.attack_counts),
            "original_label_distribution": compact(self.original_label_counts),
            "dlc_distribution": compact(self.dlc_counts),
        }


def compact(counter):
    return ";".join(f"{k}:{v}" for k, v in sorted(counter.items(), key=lambda item: str(item[0])))


def rows_to_table(rows):
    columns = {name: [row[name] for row in rows] for name in SCHEMA.names}
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_zip(zip_path, file_specs, dataset_name, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    batch = []
    summaries = []
    parse_errors = []
    invalid_rows = []
    segment_rows = []

    with zipfile.ZipFile(zip_path) as zf:
        for source_file, scenario_attack_type, mode in file_specs:
            summary = Summary(dataset_name, source_file, mode)
            prev_ts = None
            segment_id = 0
            row_in_segment = 0

            with zf.open(source_file) as fh:
                for line_no, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        if mode == "car_attack_csv":
                            timestamp, can_id, dlc, payload, raw_label = parse_car_csv_line(line)
                            is_weak_label = False
                            if raw_label == "T":
                                binary_label = "Attack"
                                attack_type = scenario_attack_type
                            elif raw_label == "R":
                                binary_label = "Normal"
                                attack_type = "Normal"
                            else:
                                binary_label = "Unknown"
                                attack_type = "Unknown"
                        elif mode == "txt_all_normal":
                            timestamp, can_id, dlc, payload = parse_txt_line(line)
                            raw_label = "scenario_normal"
                            binary_label = "Normal"
                            attack_type = "Normal"
                            is_weak_label = False
                        elif mode == "txt_scenario_attack":
                            timestamp, can_id, dlc, payload = parse_txt_line(line)
                            raw_label = f"scenario_{scenario_attack_type}"
                            binary_label = "Attack"
                            attack_type = scenario_attack_type
                            is_weak_label = True
                        else:
                            raise ValueError(f"unknown mode: {mode}")
                    except Exception as exc:
                        summary.parse_errors += 1
                        parse_errors.append(
                            {
                                "dataset": dataset_name,
                                "source_file": source_file,
                                "line_no": line_no,
                                "error": str(exc),
                                "raw_prefix": line[:160].decode("ascii", errors="replace").rstrip(),
                            }
                        )
                        continue

                    if prev_ts is not None and timestamp < prev_ts:
                        segment_rows.append(
                            {
                                "dataset": dataset_name,
                                "source_file": source_file,
                                "segment_id": segment_id,
                                "rows": row_in_segment,
                                "end_line_no": line_no - 1,
                                "end_reason": "timestamp_decrease",
                            }
                        )
                        segment_id += 1
                        row_in_segment = 0
                        summary.monotonic_breaks += 1
                    prev_ts = timestamp

                    payload_len = len(payload)
                    invalid_dlc = dlc < 0 or dlc > 8
                    payload_len_mismatch = payload_len != dlc
                    if invalid_dlc:
                        summary.invalid_dlc += 1
                    if payload_len_mismatch:
                        summary.payload_len_mismatch += 1
                        invalid_rows.append(
                            {
                                "dataset": dataset_name,
                                "source_file": source_file,
                                "line_no": line_no,
                                "reason": "payload_len_mismatch",
                                "dlc": dlc,
                                "payload_len": payload_len,
                            }
                        )
                    if invalid_dlc:
                        invalid_rows.append(
                            {
                                "dataset": dataset_name,
                                "source_file": source_file,
                                "line_no": line_no,
                                "reason": "invalid_dlc",
                                "dlc": dlc,
                                "payload_len": payload_len,
                            }
                        )

                    payload8 = canonical_payload(payload)
                    row = {
                        "dataset": dataset_name,
                        "source_file": source_file,
                        "label_mode": mode,
                        "timestamp": timestamp,
                        "can_id": can_id,
                        "dlc": max(0, min(dlc, 255)),
                        "data0": payload8[0],
                        "data1": payload8[1],
                        "data2": payload8[2],
                        "data3": payload8[3],
                        "data4": payload8[4],
                        "data5": payload8[5],
                        "data6": payload8[6],
                        "data7": payload8[7],
                        "payload_len": min(payload_len, 255),
                        "payload_len_mismatch": payload_len_mismatch,
                        "invalid_dlc": invalid_dlc,
                        "binary_label": binary_label,
                        "binary_label_id": BINARY_LABEL_ID.get(binary_label, -1),
                        "attack_type": attack_type,
                        "attack_type_id": ATTACK_TYPE_ID.get(attack_type, -1),
                        "original_label": raw_label,
                        "is_weak_label": is_weak_label,
                        "segment_id": segment_id,
                        "row_in_source": line_no,
                        "row_in_segment": row_in_segment,
                    }
                    batch.append(row)

                    summary.rows_written += 1
                    summary.max_segment_id = max(summary.max_segment_id, segment_id)
                    summary.binary_counts[binary_label] += 1
                    summary.attack_counts[attack_type] += 1
                    summary.original_label_counts[raw_label] += 1
                    summary.dlc_counts[dlc] += 1
                    summary.segment_rows[segment_id] += 1
                    row_in_segment += 1

                    if len(batch) >= BATCH_SIZE:
                        writer = write_batch(output_path, batch, writer)
                        batch.clear()

            if summary.rows_written:
                segment_rows.append(
                    {
                        "dataset": dataset_name,
                        "source_file": source_file,
                        "segment_id": segment_id,
                        "rows": row_in_segment,
                        "end_line_no": line_no,
                        "end_reason": "file_end",
                    }
                )
            summaries.append(summary)
            print(
                f"preprocessed {dataset_name}: {source_file} "
                f"rows={summary.rows_written} parse_errors={summary.parse_errors} "
                f"mismatch={summary.payload_len_mismatch}"
            )

    if batch:
        writer = write_batch(output_path, batch, writer)
        batch.clear()
    if writer is not None:
        writer.close()

    return summaries, parse_errors, invalid_rows, segment_rows


def write_batch(output_path, rows, writer):
    table = rows_to_table(rows)
    if writer is None:
        writer = pq.ParquetWriter(output_path, SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


def write_config_files():
    OUT_SPLIT.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": 42,
        "raw_inputs": {
            "main_dataset": "9) Car-Hacking Dataset.zip",
            "cross_test_dataset": "10) CAN-Intrusion Dataset.zip",
        },
        "preprocessing": {
            "payload_padding": "right_pad_zero_to_8_bytes",
            "payload_over_8_bytes": "keep_first_8_in_canonical_columns_and_log",
            "payload_len_mismatch": "preserve_row_with_flag_and_log",
            "invalid_dlc": "preserve_row_with_flag_and_log",
            "parse_error": "exclude_row_and_log",
            "segment_rule": "new_segment_on_timestamp_decrease_within_source_file",
            "normal_run_label": "all_normal",
            "otids_attack_label": "scenario_level_weak_attack",
            "normalization": "not_applied_in_preprocessing",
            "encoding_fit": "not_applied_in_preprocessing",
        },
    }
    label_mapping = {
        "binary_label_id": BINARY_LABEL_ID,
        "attack_type_id": ATTACK_TYPE_ID,
        "car_hacking": {
            "R": {"binary_label": "Normal", "attack_type": "Normal"},
            "T": "attack_type_from_source_file",
        },
        "otids": {
            "Attack_free_dataset.txt": {"binary_label": "Normal", "attack_type": "Normal", "is_weak_label": False},
            "DoS_attack_dataset.txt": {"binary_label": "Attack", "attack_type": "DoS", "is_weak_label": True},
            "Fuzzy_attack_dataset.txt": {"binary_label": "Attack", "attack_type": "Fuzzy", "is_weak_label": True},
            "Impersonation_attack_dataset.txt": {
                "binary_label": "Attack",
                "attack_type": "Impersonation",
                "is_weak_label": True,
            },
        },
    }
    (OUT_SPLIT / "preprocessing_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (OUT_SPLIT / "label_mapping.yaml").write_text(yaml.safe_dump(label_mapping, sort_keys=False))

    raw_report = """# Raw Format Report

## Car-Hacking attack CSV

Format: `timestamp,can_id,dlc,payload...,R/T`.

The payload field count follows DLC and is not always 8. The parser reads all
columns between DLC and the final label as payload bytes, then pads canonical
`data0`-`data7` columns with zero.

## Car-Hacking normal_run TXT

Format: `Timestamp: <ts> ID: <hex> 000 DLC: <dlc> <payload...>`.

No per-frame label is present, so all parsed frames are labeled Normal.

## OTIDS/CAN-Intrusion TXT

Format: `Timestamp: <ts> ID: <hex> 000 DLC: <dlc> <payload...>`.

Attack files do not contain Car-Hacking-style per-frame `R/T` labels. They are
stored as scenario-level weak attack labels and must not be used as main
multiclass training labels.
"""
    (OUT_PROFILE / "raw_format_report.md").write_text(raw_report)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    write_config_files()

    car_summary, car_errors, car_invalid, car_segments = preprocess_zip(
        CAR_ZIP, CAR_FILES, "Car-Hacking", PROCESSED / "car_hacking.parquet"
    )
    otids_summary, otids_errors, otids_invalid, otids_segments = preprocess_zip(
        OTIDS_ZIP, OTIDS_FILES, "OTIDS", PROCESSED / "otids.parquet"
    )

    summary_rows = [s.row() for s in car_summary + otids_summary]
    parse_errors = car_errors + otids_errors
    invalid_rows = car_invalid + otids_invalid
    segment_rows = car_segments + otids_segments

    write_csv(OUT_PROFILE / "parse_error_log.csv", parse_errors, fieldnames=["dataset", "source_file", "line_no", "error", "raw_prefix"])
    write_csv(OUT_PROFILE / "invalid_rows.csv", invalid_rows, fieldnames=["dataset", "source_file", "line_no", "reason", "dlc", "payload_len"])
    write_csv(
        OUT_PROFILE / "segment_summary.csv",
        segment_rows,
        fieldnames=["dataset", "source_file", "segment_id", "rows", "end_line_no", "end_reason"],
    )
    write_csv(OUT_TABLES / "preprocessing_summary.csv", summary_rows)

    total = {
        "car_hacking_rows": sum(s.rows_written for s in car_summary),
        "otids_rows": sum(s.rows_written for s in otids_summary),
        "parse_errors": sum(s.parse_errors for s in car_summary + otids_summary),
        "payload_len_mismatch": sum(s.payload_len_mismatch for s in car_summary + otids_summary),
        "invalid_dlc": sum(s.invalid_dlc for s in car_summary + otids_summary),
        "monotonic_breaks": sum(s.monotonic_breaks for s in car_summary + otids_summary),
    }
    (OUT_PROFILE / "preprocessing_totals.json").write_text(json.dumps(total, indent=2))
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
