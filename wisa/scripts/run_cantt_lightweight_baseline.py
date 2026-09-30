#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
WINDOWS = ROOT / "datasets" / "windows"
PROCESSED = ROOT / "datasets" / "processed"
CANTT_ZIP = ROOT / "datasets" / "can-train-and-test.zip"

WINDOW_SIZE = 128
STRIDE = 32


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class MetricCounter:
    def __init__(self) -> None:
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y = y_true.astype(bool)
        p = y_pred.astype(bool)
        self.tn += int((~y & ~p).sum())
        self.fp += int((~y & p).sum())
        self.fn += int((y & ~p).sum())
        self.tp += int((y & p).sum())

    def metrics(self) -> dict:
        total = self.tn + self.fp + self.fn + self.tp
        normal = self.tn + self.fp
        attack = self.fn + self.tp
        return {
            "windows": total,
            "normal_windows": normal,
            "attack_windows": attack,
            "normal_recall": self.tn / normal if normal else "",
            "fpr": self.fp / normal if normal else "",
            "attack_recall": self.tp / attack if attack else "",
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def car_hacking_train_normal_can_ids() -> set[int]:
    path = PROCESSED / "car_hacking_train.parquet"
    ids: set[int] = set()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["can_id", "binary_label_id"], batch_size=500_000):
        df = batch.to_pandas()
        ids.update(int(v) for v in df.loc[df["binary_label_id"] == 0, "can_id"].unique())
    return ids


def eval_npz_windows(path: Path, whitelist: set[int]) -> MetricCounter:
    data = np.load(path, allow_pickle=True)
    x = data["x"]
    y = data["y_binary"].astype(np.int8)
    can_ids = np.rint(x[:, :, 0]).astype(np.int64)
    known = np.isin(can_ids, list(whitelist))
    pred = (~known.all(axis=1)).astype(np.int8)
    counter = MetricCounter()
    counter.update(y, pred)
    return counter


def parse_cantt_source_stem(filename: str) -> str:
    return re.sub(r"-\d+$", "", Path(filename).stem)


def iter_cantt_csv_files() -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(CANTT_ZIP) as zf:
        infos = [
            info
            for info in zf.infolist()
            if info.filename.endswith(".csv")
            and len(Path(info.filename).parts) == 4
            and Path(info.filename).parts[2] != "train_01"
        ]
    return sorted(infos, key=lambda x: x.filename)


def window_binary_from_frames(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n < WINDOW_SIZE:
        return np.zeros(0, dtype=np.int8)
    starts = np.arange(0, n - WINDOW_SIZE + 1, STRIDE, dtype=np.int64)
    prefix = np.concatenate([[0], values.astype(np.int64).cumsum()])
    counts = prefix[starts + WINDOW_SIZE] - prefix[starts]
    return (counts > 0).astype(np.int8)


def eval_cantt_external(whitelist: set[int]) -> tuple[MetricCounter, list[dict], Counter]:
    overall = MetricCounter()
    by_subset: dict[str, MetricCounter] = {}
    source_rows: list[dict] = []
    issue_counter: Counter = Counter()

    with zipfile.ZipFile(CANTT_ZIP) as zf:
        for info in iter_cantt_csv_files():
            parts = Path(info.filename).parts
            set_id, subset_id, filename = parts[1], parts[2], parts[3]
            source_stem = parse_cantt_source_stem(filename)
            with zf.open(info.filename) as fh:
                df = pd.read_csv(
                    fh,
                    usecols=["arbitration_id", "attack"],
                    dtype={"arbitration_id": "string", "attack": "int8"},
                )
            try:
                can_ids = np.array([int(str(v), 16) for v in df["arbitration_id"].fillna("0")], dtype=np.int64)
            except ValueError:
                parsed = []
                for value in df["arbitration_id"].fillna("0"):
                    try:
                        parsed.append(int(str(value), 16))
                    except ValueError:
                        parsed.append(0)
                        issue_counter["invalid_arbitration_id"] += 1
                can_ids = np.asarray(parsed, dtype=np.int64)
            y = window_binary_from_frames(df["attack"].to_numpy(dtype=np.int8))
            unknown = (~np.isin(can_ids, list(whitelist))).astype(np.int8)
            pred = window_binary_from_frames(unknown)
            counter = MetricCounter()
            counter.update(y, pred)
            overall.update(y, pred)
            by_subset.setdefault(subset_id, MetricCounter()).update(y, pred)
            row = {
                "dataset": "can_train_and_test",
                "set_id": set_id,
                "subset_id": subset_id,
                "source_stem": source_stem,
                "source_file": info.filename,
            }
            row.update(counter.metrics())
            source_rows.append(row)

    subset_rows = []
    for subset_id, counter in sorted(by_subset.items()):
        row = {"dataset": "can_train_and_test", "set_id": "all", "subset_id": subset_id, "source_stem": "all"}
        row.update(counter.metrics())
        subset_rows.append(row)
    return overall, subset_rows + source_rows, issue_counter


def main() -> None:
    parser = argparse.ArgumentParser(description="CAN ID whitelist baseline for Car-Hacking, OTIDS, and can-train external evaluation.")
    parser.parse_args()

    start = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    whitelist = car_hacking_train_normal_can_ids()
    rows: list[dict] = []
    detail_rows: list[dict] = []

    for dataset, path in [
        ("car_hacking_real_test", WINDOWS / "test_windows.npz"),
        ("otids_cross_binary", WINDOWS / "otids_cross_windows.npz"),
    ]:
        counter = eval_npz_windows(path, whitelist)
        row = {
            "method": "can_id_whitelist",
            "rule": "window_attack_if_any_can_id_not_in_car_hacking_train_normal_whitelist",
            "dataset": dataset,
            "set_id": "all",
            "subset_id": "all",
            "source_stem": "all",
        }
        row.update(counter.metrics())
        rows.append(row)

    cantt_counter, cantt_details, issues = eval_cantt_external(whitelist)
    cantt_row = {
        "method": "can_id_whitelist",
        "rule": "window_attack_if_any_can_id_not_in_car_hacking_train_normal_whitelist",
        "dataset": "can_train_and_test_external",
        "set_id": "all",
        "subset_id": "all",
        "source_stem": "all",
    }
    cantt_row.update(cantt_counter.metrics())
    rows.append(cantt_row)
    for row in cantt_details:
        row = {
            "method": "can_id_whitelist",
            "rule": "window_attack_if_any_can_id_not_in_car_hacking_train_normal_whitelist",
            **row,
        }
        detail_rows.append(row)

    write_csv(TABLES / "cantt_lightweight_baseline_summary.csv", rows)
    write_csv(TABLES / "cantt_lightweight_baseline_by_cantt_source.csv", detail_rows)
    elapsed = time.time() - start
    log = {
        "method": "can_id_whitelist",
        "whitelist_source": "datasets/processed/car_hacking_train.parquet frames with binary_label_id == 0",
        "whitelist_size": len(whitelist),
        "whitelist_min_can_id": min(whitelist) if whitelist else None,
        "whitelist_max_can_id": max(whitelist) if whitelist else None,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "rule": "window_attack_if_any_can_id_not_in_car_hacking_train_normal_whitelist",
        "outputs": [
            "results/tables/cantt_lightweight_baseline_summary.csv",
            "results/tables/cantt_lightweight_baseline_by_cantt_source.csv",
        ],
        "parse_issues": dict(issues),
        "elapsed_seconds": elapsed,
    }
    (LOGS / "cantt_lightweight_baseline.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
