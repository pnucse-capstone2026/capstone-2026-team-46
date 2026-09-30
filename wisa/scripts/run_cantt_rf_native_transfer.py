#!/usr/bin/env python3
"""Diagnostic RF native/transfer matrix for can-train-and-test.

This is a sanity/transfer diagnostic, not a new main detector:
- train a binary RF on can-train train_01 windows only;
- evaluate it on can-train test_* windows, Car-Hacking test, OTIDS, and ROAD;
- combine the result with existing Car-Hacking-trained RF rows to expose
  transfer directionality.

The training set is capped by default for runtime control and logged explicitly.
No existing model or table is overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from run_cantt_tier1 import (
    FEATURE_NAMES,
    ZIP_PATH,
    LOGS,
    TABLES,
    MetricCounter,
    iter_windows,
    list_cantt_files,
    read_cantt_csv,
)
from train_real_only_baselines import compute_metrics, rf_features

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
ROAD_PROFILE = TABLES / "road_dataset_profile.csv"
EXP = ROOT / "experiments" / "13_cantt_rf_native_transfer"
MODELS = ROOT / "models" / "cantt_rf_native_transfer"

DEFAULT_SEEDS = [7, 42, 123]
DEFAULT_TRAIN_CAP = 300_000
DEFAULT_TREES = 160

METRIC_COLS = [
    "accuracy",
    "macro_f1_binary",
    "attack_recall",
    "normal_recall",
    "fpr",
    "fnr",
    "auprc",
    "ece",
]


def ensure_dirs() -> None:
    for path in [EXP, MODELS, TABLES, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


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


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_cantt_rf_features(files, role: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Collect aggregate RF features from can-train files."""
    feats_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    rows: list[dict] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"collect {role} {idx}/{len(files)} {meta.path}", flush=True)
            df, features, issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            file_windows = 0
            file_attack = 0
            for x, y, _starts in iter_windows(features, frame_y):
                agg = rf_features(x)
                feats_parts.append(agg)
                y_parts.append(y.astype(np.int8))
                file_windows += int(len(y))
                file_attack += int(y.sum())
            rows.append(
                {
                    "role": role,
                    "source_file": meta.path,
                    "set_id": meta.set_id,
                    "subset_id": meta.subset_id,
                    "vehicle_axis": meta.vehicle_axis,
                    "attack_axis": meta.attack_axis,
                    "source_stem": meta.source_stem,
                    "windows": file_windows,
                    "attack_windows": file_attack,
                    "normal_windows": file_windows - file_attack,
                    "parse_issues": ";".join(f"{k}:{v}" for k, v in sorted(issues.items())),
                }
            )
    if not feats_parts:
        raise RuntimeError(f"No windows collected for role={role}")
    return np.concatenate(feats_parts, axis=0), np.concatenate(y_parts, axis=0), rows


def stratified_cap(x: np.ndarray, y: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if cap <= 0 or len(y) <= cap:
        return x, y, {"cap_applied": False, "train_cap": cap, "selected_windows": int(len(y))}
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    target1 = min(len(idx1), cap // 2)
    target0 = min(len(idx0), cap - target1)
    # If attacks are scarce, fill remaining budget with normals.
    remaining = cap - target0 - target1
    if remaining > 0 and target0 < len(idx0):
        target0 = min(len(idx0), target0 + remaining)
    chosen = np.concatenate(
        [
            rng.choice(idx0, size=target0, replace=False),
            rng.choice(idx1, size=target1, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return (
        x[chosen],
        y[chosen],
        {
            "cap_applied": True,
            "train_cap": cap,
            "selected_windows": int(len(chosen)),
            "selected_normal_windows": int((y[chosen] == 0).sum()),
            "selected_attack_windows": int((y[chosen] == 1).sum()),
            "available_windows": int(len(y)),
            "available_normal_windows": int(len(idx0)),
            "available_attack_windows": int(len(idx1)),
        },
    )


def binary_proba(rf, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = rf.predict_proba(feats)
    attack = np.zeros(len(feats), dtype=np.float32)
    for idx, label in enumerate(rf.classes_):
        if int(label) == 1:
            attack = probs[:, idx].astype(np.float32)
            break
    pred = (attack >= 0.5).astype(np.int8)
    return pred, attack


def metric_row(model: str, seed: int, train_domain: str, eval_set: str, y: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict:
    # compute_metrics accepts multiclass y, but binary y works for the binary metrics
    # and leaves exact-class recall fields blank.
    row = compute_metrics(model, eval_set, y.astype(np.int64), scores, pred.astype(np.int64), seed)
    row.update({"setting": model, "train_domain": train_domain, "evaluation_set": eval_set, "level": "all", "axis": "all"})
    return row


def evaluate_npz(rf, seed: int) -> list[dict]:
    rows: list[dict] = []
    for split, y_key in [
        ("test", "y_attack_type"),
        ("variant_test", "y_attack_type"),
        ("variant_sensitivity", "y_attack_type"),
        ("otids_cross", "y_binary"),
    ]:
        path = WINDOWS / f"{split}_windows.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        feats = rf_features(data["x"])
        y_raw = data[y_key]
        y = (y_raw > 0).astype(np.int8)
        pred, scores = binary_proba(rf, feats)
        rows.append(metric_row("RF_cantt_native", seed, "can-train train_01", split, y, pred, scores))
    return rows


def evaluate_cantt_stream(rf_by_seed: dict[int, object], files) -> list[dict]:
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"eval cantt {idx}/{len(files)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                feats = rf_features(x)
                for seed, rf in rf_by_seed.items():
                    pred, _scores = binary_proba(rf, feats)
                    base = ("RF_cantt_native", seed)
                    for key in [
                        (*base, "cantt_test_all", "all", "all"),
                        (*base, "cantt_test_by_vehicle", "vehicle_axis", meta.vehicle_axis),
                        (*base, "cantt_test_by_attack", "attack_axis", meta.attack_axis),
                        (*base, "cantt_test_by_set", "set_id", meta.set_id),
                    ]:
                        metrics[key].update(y, pred)
    rows = []
    for (model, seed, evaluation_set, level, axis), counter in sorted(metrics.items()):
        row = {"setting": model, "train_domain": "can-train train_01", "seed": seed, "evaluation_set": evaluation_set, "level": level, "axis": axis}
        row.update(counter.row())
        rows.append(row)
    return rows


def evaluate_road_stream(rf_by_seed: dict[int, object]) -> list[dict]:
    if not ROAD_FRAMES.exists() or not ROAD_PROFILE.exists():
        return []
    profile = pd.read_csv(ROAD_PROFILE).set_index("capture")
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    captures = sorted(ROAD_FRAMES.glob("*.parquet"))
    for idx, path in enumerate(captures, start=1):
        capture = path.stem
        role = profile.loc[capture]["role"]
        axis = "ambient" if role == "ambient" else "attack_captures"
        print(f"eval road {idx}/{len(captures)} {capture}", flush=True)
        df = pd.read_parquet(path)
        features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        for x, y, _starts in iter_windows(features, frame_y):
            feats = rf_features(x)
            for seed, rf in rf_by_seed.items():
                pred, _scores = binary_proba(rf, feats)
                base = ("RF_cantt_native", seed)
                metrics[(*base, "road", "all", "all")].update(y, pred)
                metrics[(*base, "road_by_role", "role", axis)].update(y, pred)
    rows = []
    for (model, seed, evaluation_set, level, axis), counter in sorted(metrics.items()):
        row = {"setting": model, "train_domain": "can-train train_01", "seed": seed, "evaluation_set": evaluation_set, "level": level, "axis": axis}
        row.update(counter.row())
        rows.append(row)
    return rows


def summarize(rows: list[dict], group_cols: list[str]) -> list[dict]:
    df = pd.DataFrame(rows)
    out = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["seeds"] = ";".join(str(int(s)) for s in sorted(group["seed"].dropna().unique()))
        row["n_seeds"] = int(group["seed"].nunique())
        if "windows" in group and pd.to_numeric(group["windows"], errors="coerce").notna().all():
            vals = pd.to_numeric(group["windows"], errors="coerce")
            if vals.nunique() == 1:
                row["windows"] = int(vals.iloc[0])
        for metric in METRIC_COLS:
            if metric not in group:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals):
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def source_rf_matrix_rows() -> list[dict]:
    rows: list[dict] = []
    baseline = TABLES / "baseline_metrics_mean_std.csv"
    rf_ext = TABLES / "rf_external_summary.csv"
    if baseline.exists():
        df = pd.read_csv(baseline)
        for split, label in [("test", "Car-Hacking test"), ("otids_cross_binary", "OTIDS")]:
            sub = df[(df["model"] == "RandomForest") & (df["split"] == split)]
            if len(sub):
                r = sub.iloc[0]
                rows.append(
                    {
                        "train_domain": "Car-Hacking train",
                        "model": "RF_source_real",
                        "evaluation": label,
                        "fpr_mean": r.get("fpr_mean", ""),
                        "attack_recall_mean": r.get("attack_recall_mean", ""),
                        "normal_recall_mean": r.get("normal_recall_mean", ""),
                        "interpretation": "source RF reference",
                    }
                )
    if rf_ext.exists():
        df = pd.read_csv(rf_ext)
        for dataset, label in [("cantt_external", "can-train external"), ("road", "ROAD")]:
            sub = df[(df["model"] == "RandomForest_real_only") & (df["dataset"] == dataset) & (df["axis"] == "all")]
            if len(sub):
                r = sub.iloc[0]
                rows.append(
                    {
                        "train_domain": "Car-Hacking train",
                        "model": "RF_source_real",
                        "evaluation": label,
                        "fpr_mean": r.get("fpr_mean", ""),
                        "attack_recall_mean": r.get("attack_recall_mean", ""),
                        "normal_recall_mean": r.get("normal_recall_mean", ""),
                        "interpretation": "source-to-target diagnostic",
                    }
                )
    return rows


def native_matrix_rows(summary_rows: list[dict]) -> list[dict]:
    rows = []
    df = pd.DataFrame(summary_rows)
    mapping = [
        ("cantt_test_all", "all", "all", "can-train held-out"),
        ("test", "all", "all", "Car-Hacking test"),
        ("otids_cross", "all", "all", "OTIDS"),
        ("road", "all", "all", "ROAD"),
    ]
    for evaluation_set, level, axis, label in mapping:
        sub = df[(df["evaluation_set"] == evaluation_set) & (df["level"] == level) & (df["axis"] == axis)]
        if len(sub):
            r = sub.iloc[0]
            rows.append(
                {
                    "train_domain": "can-train train_01",
                    "model": "RF_cantt_native",
                    "evaluation": label,
                    "fpr_mean": r.get("fpr_mean", ""),
                    "attack_recall_mean": r.get("attack_recall_mean", ""),
                    "normal_recall_mean": r.get("normal_recall_mean", ""),
                    "interpretation": "native sanity" if label == "can-train held-out" else "target-to-source/other transfer diagnostic",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run can-train native RF and transfer diagnostic matrix.")
    parser.add_argument("--train-cap", type=int, default=DEFAULT_TRAIN_CAP)
    parser.add_argument("--rf-trees", type=int, default=DEFAULT_TREES)
    parser.add_argument("--seeds", default=";".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    start = time.time()
    seeds = [int(s) for s in args.seeds.replace(",", ";").split(";") if s.strip()]

    all_files = list_cantt_files()
    train_files = [f for f in all_files if f.subset_id == "train_01"]
    test_files = [f for f in all_files if f.subset_id.startswith("test_")]
    train_x, train_y, train_manifest = collect_cantt_rf_features(train_files, "train_01")
    write_csv(EXP / "cantt_rf_native_train_window_manifest.csv", train_manifest)

    rows: list[dict] = []
    rf_by_seed: dict[int, object] = {}
    train_logs: list[dict] = []
    for seed in seeds:
        model_path = MODELS / f"random_forest_cantt_native_seed{seed}.joblib"
        log_path = EXP / f"train_log_seed{seed}.json"
        if args.skip_existing and model_path.exists():
            rf = joblib.load(model_path)
            log = json.loads(log_path.read_text()) if log_path.exists() else {"seed": seed}
        else:
            x_sel, y_sel, cap_info = stratified_cap(train_x, train_y, args.train_cap, seed)
            rf = RandomForestClassifier(
                n_estimators=args.rf_trees,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
            t0 = time.time()
            rf.fit(x_sel, y_sel)
            fit_seconds = time.time() - t0
            joblib.dump(rf, model_path)
            log = {
                "seed": seed,
                "train_domain": "can-train train_01",
                "model": repo_rel(model_path),
                "rf_trees": args.rf_trees,
                "rf_features": "mean/std/min/max/last-minus-first over 11 frame features",
                "fit_seconds": fit_seconds,
                **cap_info,
            }
            log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
        print(json.dumps(log, indent=2, sort_keys=True), flush=True)
        rf_by_seed[seed] = rf
        train_logs.append(log)
        rows.extend(evaluate_npz(rf, seed))

    rows.extend(evaluate_cantt_stream(rf_by_seed, test_files))
    rows.extend(evaluate_road_stream(rf_by_seed))

    for row in rows:
        log = next((l for l in train_logs if int(l.get("seed", -1)) == int(row.get("seed", -2))), {})
        for key in [
            "train_cap",
            "cap_applied",
            "selected_windows",
            "selected_normal_windows",
            "selected_attack_windows",
            "available_windows",
            "available_normal_windows",
            "available_attack_windows",
        ]:
            row[key] = log.get(key, "")

    by_seed_path = TABLES / "cantt_rf_native_transfer_by_seed.csv"
    summary_path = TABLES / "cantt_rf_native_transfer_summary.csv"
    matrix_path = TABLES / "cantt_rf_native_transfer_matrix.csv"
    write_csv(by_seed_path, rows)
    write_csv(EXP / by_seed_path.name, rows)
    summary_rows = summarize(rows, ["setting", "train_domain", "evaluation_set", "level", "axis"])
    write_csv(summary_path, summary_rows)
    write_csv(EXP / summary_path.name, summary_rows)
    matrix_rows = source_rf_matrix_rows() + native_matrix_rows(summary_rows)
    write_csv(matrix_path, matrix_rows)
    write_csv(EXP / matrix_path.name, matrix_rows)

    log = {
        "role": "cantt_rf_native_transfer",
        "diagnostic_only": True,
        "seeds": seeds,
        "train_files": len(train_files),
        "test_files": len(test_files),
        "outputs": [repo_rel(by_seed_path), repo_rel(summary_path), repo_rel(matrix_path)],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "cantt_rf_native_transfer.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    (EXP / "run_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(matrix_rows).to_string(index=False), flush=True)
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
