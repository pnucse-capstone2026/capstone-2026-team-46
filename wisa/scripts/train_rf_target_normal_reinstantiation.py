#!/usr/bin/env python3
"""Selected RF replication for target-normal reinstantiation.

This mirrors the target-normal CNN diagnostic with a second model family while
keeping can-train calibration files out of evaluation.
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluate_variant_sensitivity import binary_metrics
from run_cantt_tier1 import FEATURE_NAMES, ZIP_PATH, MetricCounter, iter_windows, read_cantt_csv
from target_normal_reinstantiation_common import (
    CALIB_NORMAL_CACHE,
    LOGS,
    ROOT,
    RULE_RATIO,
    SEEDS,
    SPLIT_MANIFEST,
    TABLES,
    TARGET_SYNTH_PATH,
    WINDOWS,
    ensure_dirs,
    repo_rel,
    split_cantt_files,
    summarize,
    write_csv,
)
from train_real_only_baselines import compute_metrics, rf_features
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices

MODELS = ROOT / "models" / "rf_target_normal"
EXP = ROOT / "experiments" / "12_rf_target_normal_reinstantiation"
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
ROAD_PROFILE = TABLES / "road_dataset_profile.csv"
PAPER_TABLES = ROOT / "results" / "paper" / "tables"
MANIFEST = ROOT / "results" / "paper" / "paper_artifacts_manifest.md"

ARMS = {
    "source_real": {"label": "RF source real", "target_normal": False, "target_synth": False},
    "target_normal": {"label": "RF +target normal", "target_normal": True, "target_synth": False},
    "target_normal_target_synth": {
        "label": "RF +target normal+synth",
        "target_normal": True,
        "target_synth": True,
    },
}

METRIC_COLS = [
    "accuracy",
    "macro_f1_binary",
    "attack_recall",
    "attack_detection_recall",
    "normal_recall",
    "fpr",
    "fnr",
    "auprc",
    "ece",
    "recall_DoS",
    "recall_Fuzzy",
    "recall_Gear",
    "recall_RPM",
]


def rf_full_probs(rf, feats: np.ndarray) -> np.ndarray:
    probs = rf.predict_proba(feats)
    full = np.zeros((len(probs), 5), dtype=np.float32)
    for idx, label in enumerate(rf.classes_):
        if label < 5:
            full[:, label] = probs[:, idx]
    return full


def load_training_features() -> dict[str, np.ndarray]:
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    real_feats = rf_features(train["x"])
    real_y = train["y_attack_type"].astype(np.int64)
    out = {"real_feats": real_feats, "real_y": real_y}
    if not CALIB_NORMAL_CACHE.exists():
        raise SystemExit("Run scripts/build_target_normal_reinstantiation_cache.py first.")
    calib = np.load(CALIB_NORMAL_CACHE, allow_pickle=True)
    out["calib_feats"] = rf_features(calib["x"])
    out["calib_y"] = np.zeros(len(calib["x"]), dtype=np.int64)
    if not TARGET_SYNTH_PATH.exists():
        raise SystemExit("Run scripts/generate_target_normal_synthetic.py first.")
    target_syn = np.load(TARGET_SYNTH_PATH, allow_pickle=True)
    out["target_syn_feats"] = rf_features(target_syn["x"])
    out["target_syn_y"] = target_syn["y_attack_type"].astype(np.int64)
    return out


def build_arm_features(arm: str, seed: int, feats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = ARMS[arm]
    parts_x = [feats["real_feats"]]
    parts_y = [feats["real_y"]]
    info = {
        "real_windows": int(len(feats["real_y"])),
        "target_normal_windows": 0,
        "target_synthetic_windows": 0,
    }
    if cfg["target_normal"]:
        parts_x.append(feats["calib_feats"])
        parts_y.append(feats["calib_y"])
        info["target_normal_windows"] = int(len(feats["calib_y"]))
    if cfg["target_synth"]:
        n_syn = int(round(len(feats["real_y"]) * RULE_RATIO))
        idx = sample_synthetic_indices(feats["target_syn_y"], n_syn, seed + 600)
        parts_x.append(feats["target_syn_feats"][idx])
        parts_y.append(feats["target_syn_y"][idx])
        info["target_synthetic_windows"] = int(len(idx))
    x = np.concatenate(parts_x, axis=0)
    y = np.concatenate(parts_y, axis=0).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(y))
    info["train_windows_total"] = int(len(y))
    return x[order], y[order], info


def train_one(arm: str, seed: int, feats: dict[str, np.ndarray], skip_existing: bool) -> tuple[object, dict]:
    model_path = MODELS / f"random_forest_{arm}_seed{seed}.joblib"
    log_path = EXP / f"train_log_{arm}_seed{seed}.json"
    if skip_existing and model_path.exists():
        rf = joblib.load(model_path)
        log = json.loads(log_path.read_text()) if log_path.exists() else {"arm": arm, "seed": seed}
        return rf, log
    x, y, info = build_arm_features(arm, seed, feats)
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    start = time.time()
    rf.fit(x, y)
    fit_seconds = time.time() - start
    MODELS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)
    log = {
        "arm": arm,
        "label": ARMS[arm]["label"],
        "seed": seed,
        **info,
        "features": "rf_features mean/std/min/max/last-minus-first over 11 frame features",
        "rf_params": {
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": seed,
        },
        "fit_seconds": fit_seconds,
        "model": repo_rel(model_path),
    }
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)
    return rf, log


def load_eval_sets() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    test = np.load(WINDOWS / "test_windows.npz", allow_pickle=True)
    variant = np.load(WINDOWS / "variant_test_windows.npz", allow_pickle=True)
    sens = np.load(WINDOWS / "variant_sensitivity_windows.npz", allow_pickle=True)
    otids = np.load(WINDOWS / "otids_cross_windows.npz", allow_pickle=True)
    eval_sets = {
        "test": (rf_features(test["x"]), test["y_attack_type"].astype(np.int64)),
        "variant_test": (rf_features(variant["x"]), variant["y_attack_type"].astype(np.int64)),
        "variant_sensitivity": (rf_features(sens["x"]), sens["y_attack_type"].astype(np.int64)),
        "otids": (rf_features(otids["x"]), otids["y_binary"].astype(np.int8)),
    }
    sens_meta = {
        "scenario_id": sens["scenario_id"].astype(str),
        "attack_type": sens["attack_type"].astype(str),
        "severity": sens["severity"].astype(str),
    }
    return eval_sets, sens_meta


def evaluate_npz(arm: str, label: str, seed: int, rf, eval_sets: dict[str, tuple[np.ndarray, np.ndarray]], sens_meta: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for split, (feats, y) in eval_sets.items():
        probs = rf_full_probs(rf, feats)
        pred = probs.argmax(axis=1)
        scores = 1.0 - probs[:, 0]
        if split == "variant_sensitivity":
            row = {"setting": arm, "label": label, "seed": seed, "evaluation_set": split, "level": "all", "axis": "all"}
            row.update(binary_metrics(y, probs))
        else:
            row = compute_metrics(arm, split, y, scores, pred, seed)
            row.update({"setting": arm, "label": label, "evaluation_set": split, "level": "all", "axis": "all"})
        rows.append(row)
    return rows


def evaluate_cantt(rfs: dict[tuple[str, int], object]) -> list[dict]:
    _calibration, held_out = split_cantt_files()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(held_out, start=1):
            print(f"rf_tnr cantt {idx}/{len(held_out)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                feats = rf_features(x)
                for (arm, seed), rf in rfs.items():
                    pred = (np.asarray(rf.predict(feats)) > 0).astype(np.int8)
                    base = (arm, ARMS[arm]["label"], seed)
                    metrics[(*base, "all", "all")].update(y, pred)
                    metrics[(*base, "vehicle_axis", meta.vehicle_axis)].update(y, pred)
                    metrics[(*base, "attack_axis", meta.attack_axis)].update(y, pred)
    rows = []
    for (arm, label, seed, level, axis), counter in sorted(metrics.items()):
        row = {"setting": arm, "label": label, "seed": seed, "evaluation_set": "cantt_holdout", "level": level, "axis": axis}
        row.update(counter.row())
        rows.append(row)
    return rows


def evaluate_road(rfs: dict[tuple[str, int], object]) -> list[dict]:
    if not ROAD_FRAMES.exists() or not ROAD_PROFILE.exists():
        return []
    profile = pd.read_csv(ROAD_PROFILE).set_index("capture")
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    captures = sorted(ROAD_FRAMES.glob("*.parquet"))
    for idx, path in enumerate(captures, start=1):
        capture = path.stem
        p = profile.loc[capture]
        family = p["family"]
        masquerade = "masquerade" if int(p.get("masquerade", 0)) else "fabrication"
        axis = "ambient" if p["role"] == "ambient" else f"{family}|{masquerade}"
        df = pd.read_parquet(path)
        features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        print(f"rf_tnr road {idx}/{len(captures)} {capture}", flush=True)
        for x, y, _starts in iter_windows(features, frame_y):
            feats = rf_features(x)
            for (arm, seed), rf in rfs.items():
                pred = (np.asarray(rf.predict(feats)) > 0).astype(np.int8)
                base = (arm, ARMS[arm]["label"], seed)
                metrics[(*base, "all", "all")].update(y, pred)
                metrics[(*base, "axis", axis)].update(y, pred)
    rows = []
    for (arm, label, seed, level, axis), counter in sorted(metrics.items()):
        row = {"setting": arm, "label": label, "seed": seed, "evaluation_set": "road", "level": level, "axis": axis}
        row.update(counter.row())
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], logs: list[dict], elapsed: float) -> None:
    for row in rows:
        match = next((log for log in logs if log["arm"] == row["setting"] and int(log["seed"]) == int(row["seed"])), {})
        for key in ["real_windows", "target_normal_windows", "target_synthetic_windows", "train_windows_total", "fit_seconds"]:
            row[key] = match.get(key, "")
    write_csv(TABLES / "rf_target_normal_reinstantiation_by_seed.csv", rows)
    write_csv(EXP / "rf_target_normal_reinstantiation_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    summary = summarize(df, ["setting", "label", "evaluation_set", "level", "axis"], METRIC_COLS)
    write_csv(TABLES / "rf_target_normal_reinstantiation_summary.csv", summary)
    write_csv(EXP / "rf_target_normal_reinstantiation_summary.csv", summary)
    key = df[(df["level"] == "all") & (df["axis"] == "all")]
    key_summary = summarize(key, ["setting", "label", "evaluation_set"], METRIC_COLS)
    write_csv(TABLES / "rf_target_normal_reinstantiation_key_comparisons.csv", key_summary)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    copied_table = PAPER_TABLES / "table_50_rf_target_normal_reinstantiation_key_comparisons.csv"
    shutil.copy2(TABLES / "rf_target_normal_reinstantiation_key_comparisons.csv", copied_table)
    if MANIFEST.exists():
        marker = "## RF Target-Normal Reinstantiation Addendum"
        text = MANIFEST.read_text()
        block = f"""

{marker}

- **RF-TNR**: Selected random-forest replication of source-real, target-normal, and target-normal+target-synthetic arms, using the same can-train held-out split as the CNN target-normal diagnostic.
  - Tables: {copied_table.name}
"""
        if marker in text:
            text = text.split(marker)[0].rstrip() + block
        else:
            text = text.rstrip() + block
        MANIFEST.write_text(text + "\n")
    log = {
        "role": "rf_target_normal_reinstantiation",
        "arms": list(ARMS),
        "seeds": SEEDS,
        "split_manifest": repo_rel(SPLIT_MANIFEST),
        "rows": len(rows),
        "outputs": [
            "results/tables/rf_target_normal_reinstantiation_by_seed.csv",
            "results/tables/rf_target_normal_reinstantiation_summary.csv",
            "results/tables/rf_target_normal_reinstantiation_key_comparisons.csv",
        ],
        "elapsed_seconds": elapsed,
    }
    (EXP / "run_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    (LOGS / "rf_target_normal_reinstantiation.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(key_summary).to_string(index=False), flush=True)
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected RF target-normal reinstantiation replication.")
    parser.add_argument("command", choices=["all"], nargs="?", default="all")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing RF model files when present.")
    args = parser.parse_args()

    ensure_dirs()
    MODELS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    start = time.time()
    feats = load_training_features()
    eval_sets, sens_meta = load_eval_sets()
    rows = []
    logs = []
    rfs: dict[tuple[str, int], object] = {}
    for arm in ARMS:
        for seed in SEEDS:
            print(f"=== rf target-normal arm={arm} seed={seed} ===", flush=True)
            rf, log = train_one(arm, seed, feats, skip_existing=args.skip_existing)
            rfs[(arm, seed)] = rf
            logs.append(log)
            rows.extend(evaluate_npz(arm, ARMS[arm]["label"], seed, rf, eval_sets, sens_meta))
    rows.extend(evaluate_cantt(rfs))
    rows.extend(evaluate_road(rfs))
    write_outputs(rows, logs, elapsed=time.time() - start)


if __name__ == "__main__":
    main()
