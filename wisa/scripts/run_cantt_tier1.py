#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from train_real_only_baselines import CNN1D, WindowDataset, predict_cnn, standardize


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "datasets" / "can-train-and-test.zip"
RAW_DIR = ROOT / "datasets" / "raw" / "can_train_and_test"
WINDOWS = ROOT / "datasets" / "windows"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
EXP_EXTERNAL = ROOT / "experiments" / "cantt_external"
EXP_COMMON = ROOT / "experiments" / "cantt_common_family"
BASELINE_MODELS = ROOT / "models" / "baseline"
RATIO_MODELS = ROOT / "models" / "ratio_sweep"
OVERSAMPLING_MODELS = ROOT / "models" / "oversampling"

FEATURE_NAMES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]
WINDOW_SIZE = 128
STRIDE = 32
SEEDS = [42, 7, 123]
SETTINGS = [
    {"setting": "real_only", "family": "real_only", "ratio": 0.0, "label": "Real only"},
    {"setting": "real_oversampling_0p50", "family": "oversampling", "ratio": 0.5, "label": "Real oversampling +50%"},
    {"setting": "rule_0p10", "family": "rule", "ratio": 0.1, "label": "Rule +10%"},
    {"setting": "rule_0p30", "family": "rule", "ratio": 0.3, "label": "Rule +30%"},
    {"setting": "rule_0p50", "family": "rule", "ratio": 0.5, "label": "Rule +50%"},
    {"setting": "rule_1p00", "family": "rule", "ratio": 1.0, "label": "Rule +100%"},
]

VEHICLES = {
    "set_01": {"known": "2011 Chevrolet Impala", "unknown": "2016 Chevrolet Silverado"},
    "set_02": {"known": "2011 Chevrolet Traverse", "unknown": "2017 Subaru Forester"},
    "set_03": {"known": "2016 Chevrolet Silverado", "unknown": "2017 Subaru Forester"},
    "set_04": {"known": "2017 Subaru Forester", "unknown": "2011 Chevrolet Traverse"},
}

COMMON_FAMILY_MAP = {
    "DoS": ("DoS", "direct_name"),
    "fuzzing": ("Fuzzy", "direct_name"),
    "rpm": ("RPM", "direct_name"),
    "force-neutral": ("Gear-like", "tentative_semantic"),
    "rpm-accessory": ("RPM/accessory", "binary_only"),
    "speed-accessory": ("Speed/accessory", "binary_only"),
}


@dataclass(frozen=True)
class CanttFile:
    path: str
    set_id: str
    subset_id: str
    source_stem: str
    file_size: int
    vehicle_axis: str
    attack_axis: str
    vehicle: str


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

    def row(self) -> dict:
        total = self.tn + self.fp + self.fn + self.tp
        y_true = np.array([0] * (self.tn + self.fp) + [1] * (self.fn + self.tp), dtype=np.int8)
        y_pred = np.array([0] * self.tn + [1] * self.fp + [0] * self.fn + [1] * self.tp, dtype=np.int8)
        return {
            "windows": total,
            "normal_windows": self.tn + self.fp,
            "attack_windows": self.fn + self.tp,
            "accuracy": (self.tn + self.tp) / total if total else "",
            "macro_f1_binary": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total else "",
            "normal_recall": self.tn / (self.tn + self.fp) if (self.tn + self.fp) else "",
            "fpr": self.fp / (self.tn + self.fp) if (self.tn + self.fp) else "",
            "attack_recall": self.tp / (self.tp + self.fn) if (self.tp + self.fn) else "",
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


class ScoreHist:
    def __init__(self, bins: int = 1000) -> None:
        self.edges = np.linspace(0.0, 1.0, bins + 1)
        self.hist = np.zeros(bins, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        if len(values):
            hist, _ = np.histogram(values, bins=self.edges)
            self.hist += hist.astype(np.int64)

    def quantile(self, q: float) -> float | str:
        total = int(self.hist.sum())
        if total == 0:
            return ""
        target = max(0, min(total - 1, int(math.ceil(q * total) - 1)))
        cum = np.cumsum(self.hist)
        idx = int(np.searchsorted(cum, target, side="left"))
        return float((self.edges[idx] + self.edges[idx + 1]) / 2)

    def row(self) -> dict:
        return {
            "score_count": int(self.hist.sum()),
            "score_p01": self.quantile(0.01),
            "score_p05": self.quantile(0.05),
            "score_p50": self.quantile(0.50),
            "score_p95": self.quantile(0.95),
            "score_p99": self.quantile(0.99),
        }


def ensure_dirs() -> None:
    for path in [RAW_DIR, TABLES, FIGURES, LOGS, EXP_EXTERNAL, EXP_COMMON]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_source_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"-\d+$", "", stem)


def parse_file_meta(info: zipfile.ZipInfo) -> CanttFile | None:
    parts = Path(info.filename).parts
    if len(parts) != 4 or not info.filename.endswith(".csv"):
        return None
    _, set_id, subset_id, filename = parts
    source_stem = parse_source_stem(filename)
    if subset_id == "train_01":
        vehicle_axis = "train"
        attack_axis = "train"
        vehicle = VEHICLES[set_id]["known"]
    else:
        vehicle_axis = "unknown_vehicle" if "unknown_vehicle" in subset_id else "known_vehicle"
        attack_axis = "unknown_attack" if "unknown_attack" in subset_id else "known_attack"
        vehicle = VEHICLES[set_id]["unknown" if vehicle_axis == "unknown_vehicle" else "known"]
    return CanttFile(info.filename, set_id, subset_id, source_stem, info.file_size, vehicle_axis, attack_axis, vehicle)


def list_cantt_files() -> list[CanttFile]:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        files = [parse_file_meta(info) for info in zf.infolist()]
    return sorted([f for f in files if f is not None], key=lambda f: f.path)


def write_download_md() -> None:
    ensure_dirs()
    size = ZIP_PATH.stat().st_size
    digest = sha256(ZIP_PATH)
    text = f"""# can-train-and-test Download Record

Recorded: 2026-06-10

## Source

- Dataset: can-train-and-test
- Version used here: DTU Data v1 DOI `10.11583/DTU.24805533.v1`
- DTU Data landing page: <https://data.dtu.dk/articles/dataset/can-train-and-test/24805533>
- Paper/arXiv description: <https://arxiv.org/abs/2308.04972>
- Local file: `datasets/can-train-and-test.zip`
- License recorded in project checklist: CC BY 4.0
- Access note: public landing page was discoverable on 2026-06-10; direct automated fetch of the DTU page returned HTTP 403, so the local zip checksum below is the reproducibility anchor.

## Local Artifact

- File size: `{size}` bytes
- SHA256: `{digest}`
- Zip CSV count: `{len(list_cantt_files())}`
- CSV schema observed locally: `timestamp,arbitration_id,data_field,attack`
- DLC policy: derive from `data_field` byte length and zero-pad payload bytes to eight.

## Data Handling

- Preserve the original zip and do not edit it.
- Do not commit raw data in a future git repository.
- Use can-train-and-test first as external evaluation/profiling data; native training and transfer are separate diagnostic branches.
"""
    (RAW_DIR / "DOWNLOAD.md").write_text(text)


def write_file_manifest() -> None:
    rows = [f.__dict__ for f in list_cantt_files()]
    write_csv(TABLES / "cantt_file_manifest.csv", rows)
    with (LOGS / "cantt_file_manifest.txt").open("w") as out:
        for row in rows:
            out.write(
                f"{row['set_id']}\t{row['subset_id']}\t{row['source_stem']}\t{row['file_size']}\t{row['path']}\n"
            )


def model_paths(setting: dict, seed: int) -> tuple[Path, Path]:
    family = setting["family"]
    if family == "real_only":
        return BASELINE_MODELS / f"cnn1d_real_only_seed{seed}.pt", BASELINE_MODELS / "cnn_standardizer.npz"
    if family == "oversampling":
        suffix = f"real_oversampling_ratio0p50_seed{seed}"
        return OVERSAMPLING_MODELS / f"cnn1d_{suffix}.pt", OVERSAMPLING_MODELS / f"standardizer_{suffix}.npz"
    if family == "rule":
        suffix = f"rule_ratio{setting['ratio']:.2f}_seed{seed}".replace(".", "p")
        return RATIO_MODELS / f"cnn1d_{suffix}.pt", RATIO_MODELS / f"standardizer_{suffix}.npz"
    raise ValueError(family)


def run_gate0() -> None:
    ensure_dirs()
    write_download_md()
    write_file_manifest()
    rows = []
    log_lines = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for setting in SETTINGS:
        for seed in SEEDS:
            model_path, scaler_path = model_paths(setting, seed)
            row = {
                "setting": setting["setting"],
                "label": setting["label"],
                "seed": seed,
                "model_path": str(model_path.relative_to(ROOT)),
                "model_exists": model_path.exists(),
                "scaler_path": str(scaler_path.relative_to(ROOT)),
                "scaler_exists": scaler_path.exists(),
                "load_status": "not_run",
            }
            try:
                stdz = np.load(scaler_path)
                row["scaler_mean_shape"] = str(stdz["mean"].shape)
                row["scaler_std_shape"] = str(stdz["std"].shape)
                model = CNN1D(in_channels=11, classes=5).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                row["load_status"] = "ok"
            except Exception as exc:
                row["load_status"] = f"error: {exc}"
            rows.append(row)
            log_lines.append(json.dumps(row, ensure_ascii=False))

    write_csv(TABLES / "cantt_gate0_model_artifact_inventory.csv", rows)
    (LOGS / "cantt_gate0_model_artifact_check.log").write_text("\n".join(log_lines) + "\n")


def payload_to_features(payloads: pd.Series) -> tuple[np.ndarray, np.ndarray, Counter]:
    values = payloads.fillna("").astype(str).to_numpy()
    n = len(values)
    data = np.zeros((n, 8), dtype=np.uint8)
    dlc = np.zeros(n, dtype=np.uint8)
    issues = Counter()
    for i, text in enumerate(values):
        s = text.strip()
        if len(s) % 2:
            issues["odd_length_payload"] += 1
            s = s[:-1]
        try:
            raw = bytes.fromhex(s)
        except ValueError:
            issues["malformed_payload"] += 1
            raw = b""
        if len(raw) > 8:
            issues["dlc_gt_8"] += 1
        dlc[i] = min(len(raw), 255)
        if raw:
            data[i, : min(len(raw), 8)] = np.frombuffer(raw[:8], dtype=np.uint8)
    return dlc, data, issues


def read_cantt_csv(zf: zipfile.ZipFile, meta: CanttFile) -> tuple[pd.DataFrame, np.ndarray, Counter]:
    with zf.open(meta.path) as fh:
        df = pd.read_csv(
            fh,
            dtype={"timestamp": "float64", "arbitration_id": "string", "data_field": "string", "attack": "int8"},
        )
    dlc, data, issues = payload_to_features(df["data_field"])
    can_id = np.array([int(str(v), 16) for v in df["arbitration_id"].fillna("0").to_numpy()], dtype=np.int32)
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    dt = np.zeros(len(df), dtype=np.float32)
    if len(df) > 1:
        diff = np.diff(timestamps)
        issues["timestamp_decrease"] += int((diff < 0).sum())
        dt[1:] = np.maximum(diff, 0.0).astype(np.float32)
    features = np.column_stack([can_id.astype(np.float32), dlc.astype(np.float32), data.astype(np.float32), dt])
    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "can_id": can_id,
            "dlc": dlc.astype(np.int16),
            "attack": df["attack"].to_numpy(dtype=np.int8),
        }
    )
    for i in range(8):
        out[f"data{i}"] = data[:, i]
    out["delta_t"] = dt
    return out, features.astype(np.float32), issues


def car_hacking_train_normal_can_ids() -> set[int]:
    ids: set[int] = set()
    pf = pq.ParquetFile(ROOT / "datasets" / "processed" / "car_hacking_train.parquet")
    for batch in pf.iter_batches(columns=["can_id", "binary_label_id"], batch_size=500_000):
        df = batch.to_pandas()
        ids.update(int(v) for v in df.loc[df["binary_label_id"] == 0, "can_id"].unique())
    return ids


def scaled_stats(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    scaled = (features - mean.reshape(-1)) / std.reshape(-1)
    row = {}
    for idx, name in enumerate(FEATURE_NAMES):
        vals = scaled[:, idx]
        row[f"{name}_scaled_mean"] = float(np.mean(vals))
        row[f"{name}_scaled_std"] = float(np.std(vals))
        row[f"{name}_scaled_p05"] = float(np.quantile(vals, 0.05))
        row[f"{name}_scaled_p50"] = float(np.quantile(vals, 0.50))
        row[f"{name}_scaled_p95"] = float(np.quantile(vals, 0.95))
        row[f"{name}_scaled_p99"] = float(np.quantile(vals, 0.99))
    return row


def run_profile() -> None:
    ensure_dirs()
    run_gate0()
    files = list_cantt_files()
    ch_ids = car_hacking_train_normal_can_ids()
    stdz = np.load(BASELINE_MODELS / "cnn_standardizer.npz")
    mean = stdz["mean"]
    std = stdz["std"]
    dataset_rows = []
    attack_rows = []
    vehicle_rows = []
    overlap_rows = []
    scaled_rows = []
    issue_counter = Counter()
    group_counts: dict[tuple, Counter] = defaultdict(Counter)
    can_ids_by_group: dict[tuple, Counter] = defaultdict(Counter)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"profile {idx}/{len(files)} {meta.path}")
            df, features, issues = read_cantt_csv(zf, meta)
            issue_counter.update(issues)
            rows = len(df)
            attacks = int(df["attack"].sum())
            normals = rows - attacks
            attack_rows.append(
                {
                    "set_id": meta.set_id,
                    "subset_id": meta.subset_id,
                    "vehicle_axis": meta.vehicle_axis,
                    "attack_axis": meta.attack_axis,
                    "vehicle": meta.vehicle,
                    "source_stem": meta.source_stem,
                    "source_file": meta.path,
                    "rows": rows,
                    "normal_frames": normals,
                    "attack_frames": attacks,
                    "attack_frame_rate": attacks / rows if rows else 0.0,
                    "min_can_id": int(df["can_id"].min()) if rows else "",
                    "max_can_id": int(df["can_id"].max()) if rows else "",
                    "unique_can_ids": int(df["can_id"].nunique()),
                    "extended_id_frames": int((df["can_id"] > 0x7FF).sum()),
                    "dlc_distribution": ";".join(f"{k}:{v}" for k, v in sorted(Counter(df["dlc"]).items())),
                    "delta_t_p50": float(df["delta_t"].quantile(0.50)) if rows else "",
                    "delta_t_p95": float(df["delta_t"].quantile(0.95)) if rows else "",
                    "delta_t_p99": float(df["delta_t"].quantile(0.99)) if rows else "",
                    "parse_issues": ";".join(f"{k}:{v}" for k, v in sorted(issues.items())),
                }
            )
            for key in [
                (meta.set_id, meta.subset_id, meta.vehicle_axis, meta.attack_axis, meta.vehicle),
                (meta.set_id, "all", "all", "all", "all"),
                ("all", "all", "all", "all", "all"),
            ]:
                c = group_counts[key]
                c["rows"] += rows
                c["normal_frames"] += normals
                c["attack_frames"] += attacks
                can_ids_by_group[key].update(Counter(df.loc[df["attack"] == 0, "can_id"].astype(int)))
            normal_features = features[df["attack"].to_numpy(dtype=np.int8) == 0]
            if len(normal_features):
                row = {
                    "set_id": meta.set_id,
                    "subset_id": meta.subset_id,
                    "vehicle_axis": meta.vehicle_axis,
                    "attack_axis": meta.attack_axis,
                    "vehicle": meta.vehicle,
                    "source_stem": meta.source_stem,
                    "source_file": meta.path,
                    "normal_frames": len(normal_features),
                }
                row.update(scaled_stats(normal_features, mean, std))
                scaled_rows.append(row)

    for key, counts in sorted(group_counts.items()):
        set_id, subset_id, vehicle_axis, attack_axis, vehicle = key
        ids = set(can_ids_by_group[key])
        overlap = ids & ch_ids
        total = sum(can_ids_by_group[key].values())
        covered = sum(v for k, v in can_ids_by_group[key].items() if k in ch_ids)
        vehicle_rows.append(
            {
                "set_id": set_id,
                "subset_id": subset_id,
                "vehicle_axis": vehicle_axis,
                "attack_axis": attack_axis,
                "vehicle": vehicle,
                "rows": counts["rows"],
                "normal_frames": counts["normal_frames"],
                "attack_frames": counts["attack_frames"],
                "attack_frame_rate": counts["attack_frames"] / counts["rows"] if counts["rows"] else 0.0,
            }
        )
        overlap_rows.append(
            {
                "set_id": set_id,
                "subset_id": subset_id,
                "vehicle_axis": vehicle_axis,
                "attack_axis": attack_axis,
                "vehicle": vehicle,
                "cantt_normal_unique_can_ids": len(ids),
                "car_hacking_train_normal_unique_can_ids": len(ch_ids),
                "intersection_unique_can_ids": len(overlap),
                "unique_overlap_ratio_vs_cantt": len(overlap) / len(ids) if ids else "",
                "frame_coverage_by_car_hacking_ids": covered / total if total else "",
                "cantt_top20_normal_can_ids": ";".join(f"{k}:{v}" for k, v in can_ids_by_group[key].most_common(20)),
            }
        )

    dataset_rows.append(
        {
            "dataset": "can_train_and_test",
            "zip_path": str(ZIP_PATH.relative_to(ROOT)),
            "zip_size_bytes": ZIP_PATH.stat().st_size,
            "zip_sha256": sha256(ZIP_PATH),
            "csv_files": len(files),
            "total_csv_uncompressed_bytes": sum(f.file_size for f in files),
            "parse_issues": ";".join(f"{k}:{v}" for k, v in sorted(issue_counter.items())),
        }
    )
    outlier_rows = []
    for row in scaled_rows:
        for name in FEATURE_NAMES:
            if abs(float(row[f"{name}_scaled_mean"])) > 3.0 or abs(float(row[f"{name}_scaled_p99"])) > 10.0:
                outlier_rows.append(
                    {
                        "source_file": row["source_file"],
                        "set_id": row["set_id"],
                        "subset_id": row["subset_id"],
                        "vehicle_axis": row["vehicle_axis"],
                        "source_stem": row["source_stem"],
                        "feature": name,
                        "scaled_mean": row[f"{name}_scaled_mean"],
                        "scaled_p95": row[f"{name}_scaled_p95"],
                        "scaled_p99": row[f"{name}_scaled_p99"],
                    }
                )
    write_csv(TABLES / "cantt_dataset_profile.csv", dataset_rows)
    write_csv(TABLES / "cantt_attack_counts.csv", attack_rows)
    write_csv(TABLES / "cantt_vehicle_subset_counts.csv", vehicle_rows)
    write_csv(TABLES / "cantt_can_id_overlap.csv", overlap_rows)
    write_csv(TABLES / "cantt_vs_carhacking_scaled_feature_shift.csv", scaled_rows)
    write_csv(TABLES / "cantt_scaled_zscore_outliers.csv", outlier_rows)
    (LOGS / "cantt_delta_t_policy_check.log").write_text(
        "delta_t policy: per source CSV, first row 0, negative timestamp deltas clipped to 0; source files are never joined.\n"
        f"observed issues: {dict(issue_counter)}\n"
    )
    (LOGS / "cantt_can_id_representation_check.log").write_text(
        "can_train arbitration_id parsed as hexadecimal integer into the same raw numeric can_id feature used by Car-Hacking windows.\n"
        "Interpret high external FPR together with can_id overlap and scaled can_id shift because raw integer IDs are vehicle-specific.\n"
    )
    (LOGS / "cantt_preprocess_profile.log").write_text(json.dumps({"parse_issues": dict(issue_counter)}, indent=2))


def load_models() -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for setting in SETTINGS:
        for seed in SEEDS:
            model_path, scaler_path = model_paths(setting, seed)
            stdz = np.load(scaler_path)
            model = CNN1D(in_channels=11, classes=5).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            loaded.append(
                {
                    **setting,
                    "seed": seed,
                    "model": model,
                    "mean": stdz["mean"].astype(np.float32),
                    "std": stdz["std"].astype(np.float32),
                    "device": device,
                }
            )
    return loaded


def iter_windows(features: np.ndarray, frame_y: np.ndarray, batch_size: int = 4096):
    n = len(features)
    if n < WINDOW_SIZE:
        return
    starts = np.arange(0, n - WINDOW_SIZE + 1, STRIDE, dtype=np.int64)
    csum = np.concatenate([[0], np.cumsum(frame_y.astype(np.int32))])
    win_y = (csum[starts + WINDOW_SIZE] - csum[starts] > 0).astype(np.int8)
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        x = np.stack([features[s : s + WINDOW_SIZE] for s in batch_starts]).astype(np.float32)
        yield x, win_y[i : i + len(batch_starts)], batch_starts


def update_metric_groups(metrics: dict, model_info: dict, meta: CanttFile, y: np.ndarray, pred: np.ndarray) -> None:
    base = (model_info["setting"], model_info["label"], model_info["seed"])
    keys = [
        ("summary", *base, "all", "all", "all", "all"),
        ("subset", *base, meta.set_id, meta.subset_id, meta.vehicle_axis, meta.attack_axis),
        ("vehicle_axis", *base, "all", "all", meta.vehicle_axis, "all"),
        ("attack_axis", *base, "all", "all", "all", meta.attack_axis),
        ("stem", *base, meta.set_id, meta.source_stem, meta.vehicle_axis, meta.attack_axis),
    ]
    for key in keys:
        metrics[key].update(y, pred)


def update_score_hists(hists: dict, model_info: dict, meta: CanttFile, y: np.ndarray, scores: np.ndarray) -> None:
    base = (model_info["setting"], model_info["label"], model_info["seed"])
    for label_name, mask in [("normal", y == 0), ("attack", y == 1), ("all", np.ones(len(y), dtype=bool))]:
        if np.any(mask):
            key = (*base, label_name, meta.set_id, meta.subset_id, meta.vehicle_axis, meta.attack_axis)
            hists[key].update(scores[mask])


def evaluate_batch(model_info: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_std = standardize(x, model_info["mean"], model_info["std"])
    loader = DataLoader(WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)), batch_size=1024, shuffle=False, num_workers=0)
    _, probs = predict_cnn(model_info["model"], loader, model_info["device"])
    pred = (probs.argmax(axis=1) > 0).astype(np.int8)
    scores = 1.0 - probs[:, 0]
    return pred, scores


def grammar_distance_rows(profile_path: Path) -> list[dict]:
    df = pd.read_csv(profile_path)
    rows = []
    for stem, group in df[df["attack_frames"] > 0].groupby("source_stem"):
        candidate, confidence = COMMON_FAMILY_MAP.get(stem, ("other_attack", "binary_only"))
        top_ids = Counter()
        dlc = Counter()
        for _, row in group.iterrows():
            if isinstance(row.get("dlc_distribution"), str):
                for part in row["dlc_distribution"].split(";"):
                    if ":" in part:
                        k, v = part.split(":", 1)
                        dlc[k] += int(v)
            top_field = str(row.get("min_can_id", "")) + "-" + str(row.get("max_can_id", ""))
            top_ids[top_field] += int(row["attack_frames"])
        rows.append(
            {
                "source_stem": stem,
                "car_hacking_synthetic_family_candidate": candidate,
                "mapping_confidence": confidence,
                "target_can_id_overlap_interpretation": "not_claimed_from_name_only",
                "observed_can_id_range_weighted": ";".join(f"{k}:{v}" for k, v in top_ids.most_common(10)),
                "dlc_distribution_from_files": ";".join(f"{k}:{v}" for k, v in sorted(dlc.items())),
                "primary_metric_policy": "binary_detection_recall",
            }
        )
    return rows


def run_evaluate() -> None:
    ensure_dirs()
    if not (TABLES / "cantt_attack_counts.csv").exists():
        run_profile()
    files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    models = load_models()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    hists: dict[tuple, ScoreHist] = defaultdict(ScoreHist)
    window_rows = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"evaluate {idx}/{len(files)} {meta.path}")
            df, features, issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            file_windows = 0
            file_attack_windows = 0
            for x, y, starts in iter_windows(features, frame_y):
                file_windows += len(y)
                file_attack_windows += int(y.sum())
                with torch.no_grad():
                    for model_info in models:
                        pred, scores = evaluate_batch(model_info, x)
                        update_metric_groups(metrics, model_info, meta, y, pred)
                        update_score_hists(hists, model_info, meta, y, scores)
            window_rows.append(
                {
                    "set_id": meta.set_id,
                    "subset_id": meta.subset_id,
                    "vehicle_axis": meta.vehicle_axis,
                    "attack_axis": meta.attack_axis,
                    "vehicle": meta.vehicle,
                    "source_stem": meta.source_stem,
                    "source_file": meta.path,
                    "windows": file_windows,
                    "normal_windows": file_windows - file_attack_windows,
                    "attack_windows": file_attack_windows,
                    "parse_issues": ";".join(f"{k}:{v}" for k, v in sorted(issues.items())),
                }
            )
    write_csv(TABLES / "cantt_window_counts.csv", window_rows)
    (LOGS / "cantt_window_generation.log").write_text(
        f"Generated test windows by streaming zip CSV files with window_size={WINDOW_SIZE}, stride={STRIDE}; no cross-file windows.\n"
    )

    rows_by_kind: dict[str, list[dict]] = defaultdict(list)
    for key, counter in metrics.items():
        kind, setting, label, seed, set_id, subset_or_stem, vehicle_axis, attack_axis = key
        row = {
            "setting": setting,
            "label": label,
            "seed": seed,
            "set_id": set_id,
            "subset_or_stem": subset_or_stem,
            "vehicle_axis": vehicle_axis,
            "attack_axis": attack_axis,
        }
        row.update(counter.row())
        rows_by_kind[kind].append(row)
    write_csv(TABLES / "cantt_external_eval_summary.csv", rows_by_kind["summary"])
    write_csv(TABLES / "cantt_external_eval_by_subset.csv", rows_by_kind["subset"])
    write_csv(TABLES / "cantt_external_eval_by_axis.csv", rows_by_kind["vehicle_axis"] + rows_by_kind["attack_axis"])
    write_csv(TABLES / "cantt_external_eval_by_stem.csv", rows_by_kind["stem"])
    write_csv(EXP_EXTERNAL / "cantt_external_eval_summary.csv", rows_by_kind["summary"])
    write_csv(EXP_EXTERNAL / "cantt_external_eval_by_subset.csv", rows_by_kind["subset"])
    write_csv(EXP_EXTERNAL / "cantt_external_eval_by_axis.csv", rows_by_kind["vehicle_axis"] + rows_by_kind["attack_axis"])
    write_csv(EXP_EXTERNAL / "cantt_external_eval_by_stem.csv", rows_by_kind["stem"])

    quantile_rows = []
    for key, hist in hists.items():
        setting, label, seed, score_label, set_id, subset_id, vehicle_axis, attack_axis = key
        row = {
            "setting": setting,
            "label": label,
            "seed": seed,
            "score_label": score_label,
            "set_id": set_id,
            "subset_id": subset_id,
            "vehicle_axis": vehicle_axis,
            "attack_axis": attack_axis,
        }
        row.update(hist.row())
        quantile_rows.append(row)
    write_csv(TABLES / "cantt_external_score_quantiles.csv", quantile_rows)

    common = pd.DataFrame(rows_by_kind["stem"])
    common = common[common["subset_or_stem"].isin(COMMON_FAMILY_MAP.keys())].copy()
    common.to_csv(TABLES / "cantt_common_family_external_stress.csv", index=False)
    common.to_csv(EXP_COMMON / "cantt_common_family_external_stress.csv", index=False)
    grammar_rows = grammar_distance_rows(TABLES / "cantt_attack_counts.csv")
    write_csv(TABLES / "cantt_common_family_grammar_distance.csv", grammar_rows)
    write_csv(EXP_COMMON / "cantt_common_family_grammar_distance.csv", grammar_rows)
    plot_score_histograms(TABLES / "cantt_external_score_quantiles.csv")


def plot_score_histograms(path: Path) -> None:
    df = pd.read_csv(path)
    selected = df[
        (df["seed"] == 42)
        & (df["score_label"] == "normal")
        & (df["setting"].isin(["real_only", "rule_0p30", "rule_1p00"]))
    ].copy()
    if selected.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    x = np.arange(len(selected))
    ax.scatter(x, selected["score_p50"], label="p50", s=18)
    ax.scatter(x, selected["score_p95"], label="p95", s=18)
    ax.scatter(x, selected["score_p99"], label="p99", s=18)
    for _, row in selected.iterrows():
        labels.append(f"{row['setting']}\n{row['set_id']} {row['subset_id'].replace('test_', 't')}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Attack score quantile on can-train normal windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cantt_external_score_histograms.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["gate0", "profile", "evaluate", "all"])
    args = parser.parse_args()
    if args.command == "gate0":
        run_gate0()
    elif args.command == "profile":
        run_profile()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "all":
        run_profile()
        run_evaluate()


if __name__ == "__main__":
    main()
