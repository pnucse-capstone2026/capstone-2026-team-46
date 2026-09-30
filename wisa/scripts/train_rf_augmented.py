#!/usr/bin/env python3
"""Augmented Random-Forest arm: Car-Hacking real train windows + rule-based
synthetic windows at +30% of the real training window count.

Purpose: test whether the rule-based augmentation effects observed on the CNN
(fixed-variant robustness gain) reproduce in a second model family, and whether
the real-only RF's comparatively low can-train external FPR survives
augmentation.

Policy compliance:
- Training data: Car-Hacking real train windows + rule-based synthetic windows
  only. OTIDS / can-train-and-test / ROAD remain evaluation-only.
- Synthetic pool sampling replicates scripts/train_rule_synthetic_ratio_sweep.py
  exactly: class-balanced sample of round(n_real * 0.30) windows with
  rng seed = seed + int(ratio * 1000), then a default_rng(seed) permutation of
  the concatenated training set.
- RF features and hyperparameters replicate
  scripts/train_real_only_baselines.py (rf_features aggregation; 160 trees,
  unlimited depth, min_samples_leaf=2, class_weight=balanced_subsample,
  random_state=seed).
- New artifacts only: models/rf_augmented/, results/tables/rf_augmented_*.csv.
  No existing model or table is modified.

CPU only; run with CUDA_VISIBLE_DEVICES="".
"""
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluate_variant_sensitivity import binary_metrics
from run_cantt_tier1 import (
    LOGS,
    TABLES,
    ZIP_PATH,
    FEATURE_NAMES,
    MetricCounter,
    iter_windows,
    list_cantt_files,
    read_cantt_csv,
    write_csv,
)
from train_real_only_baselines import compute_metrics, rf_features
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
MODELS = ROOT / "models" / "rf_augmented"

MODEL_NAME = "RandomForest_rule_0p30"
SEEDS = [7, 42, 123]
RATIO = 0.30

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
    """Map predict_proba onto the fixed 5-class layout, as in the baseline scripts."""
    probs = rf.predict_proba(feats)
    full = np.zeros((len(probs), 5), dtype=np.float32)
    for idx, label in enumerate(rf.classes_):
        if label < 5:
            full[:, label] = probs[:, idx]
    return full


def train_one(seed: int, real_feats: np.ndarray, real_y: np.ndarray, syn_feats: np.ndarray, syn_y: np.ndarray) -> tuple:
    n_syn = int(round(len(real_y) * RATIO))
    # Same sampling rng convention as train_rule_synthetic_ratio_sweep.run_one.
    syn_idx = sample_synthetic_indices(syn_y, n_syn, seed + int(RATIO * 1000))
    feats = np.concatenate([real_feats, syn_feats[syn_idx]], axis=0)
    y = np.concatenate([real_y, syn_y[syn_idx]], axis=0).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(y))
    feats = feats[order]
    y = y[order]
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    t0 = time.time()
    rf.fit(feats, y)
    fit_seconds = time.time() - t0
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODELS / f"random_forest_rule_0p30_seed{seed}.joblib")
    return rf, n_syn, fit_seconds


def evaluate_indomain(rf, seed: int, eval_sets: dict, sens_meta: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    scenario_rows = []
    # Car-Hacking test / fixed variant test / OTIDS: same code path as
    # train_real_only_baselines / evaluate_variant_baselines (compute_metrics).
    for split, (feats, y_attack) in eval_sets.items():
        full_probs = rf_full_probs(rf, feats)
        pred = full_probs.argmax(axis=1)
        attack_scores = 1.0 - full_probs[:, 0]
        if split == "variant_sensitivity":
            # Same code path as evaluate_variant_sensitivity (binary_metrics + scenario rows).
            row = {"model": MODEL_NAME, "seed": seed, "split": split}
            row.update(binary_metrics(y_attack, full_probs))
            rows.append(row)
            scenario = sens_meta["scenario_id"]
            for sid in sorted(set(scenario)):
                mask = scenario == sid
                y_s = y_attack[mask]
                pred_s = pred[mask]
                scores_s = attack_scores[mask]
                is_attack = y_s > 0
                scenario_rows.append(
                    {
                        "model": MODEL_NAME,
                        "seed": seed,
                        "scenario_id": sid,
                        "attack_type": str(sens_meta["attack_type"][mask][0]),
                        "severity": str(sens_meta["severity"][mask][0]),
                        "windows": int(mask.sum()),
                        "attack_detection_recall": float(((pred_s > 0) & is_attack).sum() / max(is_attack.sum(), 1)) if is_attack.any() else "",
                        "exact_class_recall": float((pred_s[is_attack] == y_s[is_attack]).sum() / max(is_attack.sum(), 1)) if is_attack.any() else "",
                        "normal_recall": float((pred_s == 0).mean()) if not is_attack.any() else "",
                        "mean_attack_score": float(scores_s.mean()),
                    }
                )
        else:
            row = compute_metrics(MODEL_NAME, split, y_attack, attack_scores, pred, seed)
            rows.append(row)
    return rows, scenario_rows


def evaluate_external(rfs: dict) -> list[dict]:
    """Same code path as evaluate_rf_external.py (rf.predict, MetricCounter)."""
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)

    road_profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")
    captures = sorted(ROAD_FRAMES.glob("*.parquet"))
    for idx, path in enumerate(captures, start=1):
        capture = path.stem
        role = road_profile.loc[capture]["role"]
        axis = "ambient" if role == "ambient" else "attack_captures"
        df = pd.read_parquet(path)
        features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        print(f"rf_aug road {idx}/{len(captures)} {capture}", flush=True)
        for x, y, _starts in iter_windows(features, frame_y):
            feats = rf_features(x)
            for seed, rf in rfs.items():
                pred = (np.asarray(rf.predict(feats)) > 0).astype(np.int8)
                metrics[("road", seed, "all")].update(y, pred)
                metrics[("road", seed, axis)].update(y, pred)

    files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(files, start=1):
            print(f"rf_aug cantt {idx}/{len(files)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                feats = rf_features(x)
                for seed, rf in rfs.items():
                    pred = (np.asarray(rf.predict(feats)) > 0).astype(np.int8)
                    metrics[("cantt_external", seed, "all")].update(y, pred)

    rows = []
    for (dataset, seed, axis), counter in sorted(metrics.items()):
        row = {"model": MODEL_NAME, "seed": seed, "split": dataset, "axis": axis}
        row.update(counter.row())
        rows.append(row)
    return rows


def summarize(by_seed: list[dict]) -> list[dict]:
    df = pd.DataFrame(by_seed)
    if "axis" not in df.columns:
        df["axis"] = ""
    df["axis"] = df["axis"].fillna("")
    rows = []
    for (split, axis), group in df.groupby(["split", "axis"], sort=True):
        row = {
            "model": MODEL_NAME,
            "split": split,
            "axis": axis,
            "seeds": ";".join(str(int(s)) for s in sorted(group["seed"].unique())),
        }
        if "windows" in group.columns and pd.to_numeric(group["windows"], errors="coerce").notna().all():
            row["windows"] = int(group["windows"].iloc[0])
        for metric in METRIC_COLS:
            if metric not in group.columns:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals):
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def main() -> None:
    start = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("featurizing real train windows", flush=True)
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    real_feats = rf_features(train["x"])
    real_y = train["y_attack_type"].astype(np.int64)
    del train

    print("featurizing rule-based synthetic pool", flush=True)
    syn = np.load(SYNTHETIC / "rule_based_windows.npz", allow_pickle=True)
    syn_feats = rf_features(syn["x"])
    syn_y = syn["y_attack_type"].astype(np.int64)
    del syn

    print("featurizing evaluation sets", flush=True)
    test = np.load(WINDOWS / "test_windows.npz", allow_pickle=True)
    variant = np.load(WINDOWS / "variant_test_windows.npz", allow_pickle=True)
    sens = np.load(WINDOWS / "variant_sensitivity_windows.npz", allow_pickle=True)
    cross = np.load(WINDOWS / "otids_cross_windows.npz", allow_pickle=True)
    eval_sets = {
        "test": (rf_features(test["x"]), test["y_attack_type"].astype(np.int64)),
        "variant_test": (rf_features(variant["x"]), variant["y_attack_type"].astype(np.int64)),
        "variant_sensitivity": (rf_features(sens["x"]), sens["y_attack_type"].astype(np.int64)),
        "otids_cross_binary": (rf_features(cross["x"]), cross["y_binary"].astype(np.int8)),
    }
    sens_meta = {
        "scenario_id": sens["scenario_id"].astype(str),
        "attack_type": sens["attack_type"].astype(str),
        "severity": sens["severity"].astype(str),
    }
    del test, variant, sens, cross

    by_seed_rows = []
    scenario_rows = []
    rfs = {}
    fit_log = {}
    n_syn = None
    for seed in SEEDS:
        print(f"=== training RF rule +30% seed={seed} ===", flush=True)
        rf, n_syn, fit_seconds = train_one(seed, real_feats, real_y, syn_feats, syn_y)
        fit_log[seed] = fit_seconds
        rfs[seed] = rf
        print(f"seed={seed} fit done in {fit_seconds:.1f}s; evaluating in-domain splits", flush=True)
        rows, srows = evaluate_indomain(rf, seed, eval_sets, sens_meta)
        by_seed_rows.extend(rows)
        scenario_rows.extend(srows)

    print("evaluating external rungs (ROAD, can-train-and-test)", flush=True)
    by_seed_rows.extend(evaluate_external(rfs))

    write_csv(TABLES / "rf_augmented_by_seed.csv", by_seed_rows)
    write_csv(TABLES / "rf_augmented_sensitivity_by_scenario.csv", scenario_rows)

    summary_rows = summarize(by_seed_rows)
    write_csv(TABLES / "rf_augmented_summary.csv", summary_rows)

    scen_df = pd.DataFrame(scenario_rows)
    scen_summary = []
    for (sid, attack, sev), group in scen_df.groupby(["scenario_id", "attack_type", "severity"], sort=True):
        row = {
            "model": MODEL_NAME,
            "scenario_id": sid,
            "attack_type": attack,
            "severity": sev,
            "windows": int(group["windows"].iloc[0]),
            "seeds": ";".join(str(int(s)) for s in sorted(group["seed"].unique())),
        }
        for metric in ["attack_detection_recall", "exact_class_recall", "normal_recall", "mean_attack_score"]:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals):
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        scen_summary.append(row)
    write_csv(TABLES / "rf_augmented_sensitivity_scenario_summary.csv", scen_summary)

    log = {
        "experiment": "rf_augmented_rule_0p30",
        "training_data": "Car-Hacking real train windows + rule-based synthetic windows (+30% of real count)",
        "n_real_windows": int(len(real_y)),
        "n_synthetic_windows": int(n_syn),
        "sampling": "sample_synthetic_indices(y_syn, n_syn, seed + int(0.30 * 1000)); default_rng(seed) permutation",
        "features": "rf_features: mean/std/min/max/last-minus-first over 11 frame features, no standardizer",
        "rf_params": {
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": "seed",
        },
        "seeds": SEEDS,
        "models": [f"models/rf_augmented/random_forest_rule_0p30_seed{s}.joblib" for s in SEEDS],
        "fit_seconds_by_seed": fit_log,
        "outputs": [
            "results/tables/rf_augmented_by_seed.csv",
            "results/tables/rf_augmented_summary.csv",
            "results/tables/rf_augmented_sensitivity_by_scenario.csv",
            "results/tables/rf_augmented_sensitivity_scenario_summary.csv",
        ],
        "evaluation_only_datasets": ["otids_cross", "cantt_external", "road"],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "rf_augmented_train_eval.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
