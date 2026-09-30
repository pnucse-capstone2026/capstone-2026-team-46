#!/usr/bin/env python3
"""Train and evaluate target-normal reinstantiation CNN arms."""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_cantt_tier1 import FEATURE_NAMES, ZIP_PATH, MetricCounter, iter_windows, read_cantt_csv
from target_normal_reinstantiation_common import (
    ARM_CONFIGS,
    CALIB_ATTACK_CACHE,
    CALIB_NORMAL_CACHE,
    MAIN_ARMS,
    MODELS,
    ORACLE_ARMS,
    RULE_RATIO,
    SEEDS,
    SYNTHETIC,
    TABLES,
    TARGET_SYNTH_PATH,
    WINDOWS,
    ensure_dirs,
    split_cantt_files,
    summarize,
    write_csv,
)
from train_real_only_baselines import CNN1D, compute_metrics, fit_standardizer, standardize
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices, train_model

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "10_target_normal_reinstantiation"
LOGS = ROOT / "results" / "logs"
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
ROAD_PROFILE = ROOT / "results" / "tables" / "road_dataset_profile.csv"
EVAL_BATCH_SIZE = 8192


def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def build_training_set(arm: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = ARM_CONFIGS[arm]
    train = load_npz(WINDOWS / "train_windows.npz")
    real_x = train["x"].astype(np.float32)
    real_y = train["y_attack_type"].astype(np.int64)
    parts_x = [real_x]
    parts_y = [real_y]
    info = {
        "real_windows": int(len(real_x)),
        "source_synthetic_windows": 0,
        "target_normal_windows": 0,
        "target_synthetic_windows": 0,
        "oracle_attack_windows": 0,
    }

    n_syn = int(round(len(real_x) * RULE_RATIO))
    if cfg["source_rule"]:
        source_syn = load_npz(SYNTHETIC / "rule_based_windows.npz")
        idx = sample_synthetic_indices(source_syn["y_attack_type"], n_syn, seed + int(RULE_RATIO * 1000))
        parts_x.append(source_syn["x"][idx].astype(np.float32))
        parts_y.append(source_syn["y_attack_type"][idx].astype(np.int64))
        info["source_synthetic_windows"] = int(len(idx))

    if cfg["target_normal"]:
        if not CALIB_NORMAL_CACHE.exists():
            raise SystemExit("Run scripts/build_target_normal_reinstantiation_cache.py first.")
        calib = load_npz(CALIB_NORMAL_CACHE)
        x = calib["x"].astype(np.float32)
        parts_x.append(x)
        parts_y.append(np.zeros(len(x), dtype=np.int64))
        info["target_normal_windows"] = int(len(x))

    if cfg["target_synth"]:
        if not TARGET_SYNTH_PATH.exists():
            raise SystemExit("Run scripts/generate_target_normal_synthetic.py first.")
        target_syn = load_npz(TARGET_SYNTH_PATH)
        idx = sample_synthetic_indices(target_syn["y_attack_type"], n_syn, seed + 600)
        parts_x.append(target_syn["x"][idx].astype(np.float32))
        parts_y.append(target_syn["y_attack_type"][idx].astype(np.int64))
        info["target_synthetic_windows"] = int(len(idx))

    if cfg["oracle_attack"]:
        if not CALIB_ATTACK_CACHE.exists():
            raise SystemExit("Run cache build with --include-oracle-attacks before using oracle arm.")
        oracle = load_npz(CALIB_ATTACK_CACHE)
        parts_x.append(oracle["x"].astype(np.float32))
        parts_y.append(oracle["y_attack_type"].astype(np.int64))
        info["oracle_attack_windows"] = int(len(oracle["x"]))

    x_raw = np.concatenate(parts_x, axis=0).astype(np.float32)
    y = np.concatenate(parts_y, axis=0).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(y))
    info["train_windows_total"] = int(len(y))
    return x_raw[order], y[order], info


def train_arm(arm: str, seed: int, max_epochs: int, skip_existing: bool) -> None:
    model_path = MODELS / f"cnn1d_{arm}_seed{seed}.pt"
    if skip_existing and model_path.exists():
        print(f"skip existing {arm} seed={seed}", flush=True)
        return
    start = time.time()
    print(f"=== target-normal arm={arm} seed={seed} ===", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    val = load_npz(WINDOWS / "val_windows.npz")
    val_x_raw = val["x"].astype(np.float32)
    val_y = val["y_attack_type"].astype(np.int64)
    train_x_raw, train_y, info = build_training_set(arm, seed)
    mean, std = fit_standardizer(train_x_raw)
    np.savez(MODELS / f"standardizer_{arm}_seed{seed}.npz", mean=mean, std=std)
    train_x = standardize(train_x_raw, mean, std)
    val_x = standardize(val_x_raw, mean, std)
    del train_x_raw

    model, history, _device, best = train_model(train_x, train_y, val_x, val_y, seed, max_epochs=max_epochs)
    torch.save(model.state_dict(), model_path)
    write_csv(EXP / f"training_history_{arm}_seed{seed}.csv", history)
    log = {
        "arm": arm,
        "seed": seed,
        **info,
        "standardizer_policy": "fit on full arm training windows",
        "epochs_run": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1_multiclass": float(best["f1"]),
        "max_epochs": max_epochs,
        "elapsed_seconds": time.time() - start,
    }
    (EXP / f"train_log_{arm}_seed{seed}.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


def load_models(arms: list[str]) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded: list[dict] = []
    for arm in arms:
        cfg = ARM_CONFIGS[arm]
        for seed in SEEDS:
            stdz = np.load(MODELS / f"standardizer_{arm}_seed{seed}.npz")
            model = CNN1D(in_channels=11, classes=5).to(device)
            model.load_state_dict(torch.load(MODELS / f"cnn1d_{arm}_seed{seed}.pt", map_location=device))
            model.eval()
            loaded.append(
                {
                    "setting": arm,
                    "label": cfg["label"],
                    "group": cfg["group"],
                    "seed": seed,
                    "model": model,
                    "mean": stdz["mean"].astype(np.float32),
                    "std": stdz["std"].astype(np.float32),
                    "device": device,
                }
            )
    return loaded


def predict_info(info: dict, x_raw: np.ndarray, y_for_loader: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = standardize(x_raw.astype(np.float32), info["mean"], info["std"])
    preds: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    model = info["model"]
    device = info["device"]
    with torch.no_grad():
        for start in range(0, len(x), EVAL_BATCH_SIZE):
            batch = x[start : start + EVAL_BATCH_SIZE].transpose(0, 2, 1)
            xb = torch.from_numpy(batch).to(device)
            probs = torch.softmax(model(xb), dim=1).detach().cpu().numpy()
            preds.append(probs.argmax(axis=1))
            scores.append(1.0 - probs[:, 0])
    return np.concatenate(preds), np.concatenate(scores)


def evaluate_npz(models: list[dict], split: str, path: Path, y_key: str = "y_attack_type") -> list[dict]:
    data = load_npz(path)
    x = data["x"].astype(np.float32)
    y = data[y_key].astype(np.int64)
    rows: list[dict] = []
    for info in models:
        print(f"eval {split} {info['setting']} seed={info['seed']}", flush=True)
        pred, scores = predict_info(info, x)
        row = compute_metrics(info["setting"], split, y, scores, pred, info["seed"])
        row.update({"setting": info["setting"], "label": info["label"], "group": info["group"], "evaluation_set": split, "level": "all", "axis": "all"})
        rows.append(row)
    return rows


def evaluate_cantt_holdout(models: list[dict]) -> list[dict]:
    _calibration, held_out = split_cantt_files()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(held_out, start=1):
            print(f"cantt holdout {idx}/{len(held_out)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                for info in models:
                    pred, _scores = predict_info(info, x)
                    base = (info["setting"], info["label"], info["group"], info["seed"])
                    metrics[(*base, "all", "all")].update(y, pred > 0)
                    metrics[(*base, "vehicle_axis", meta.vehicle_axis)].update(y, pred > 0)
                    metrics[(*base, "attack_axis", meta.attack_axis)].update(y, pred > 0)
                    metrics[(*base, "source_stem", meta.source_stem)].update(y, pred > 0)
    rows = []
    for (setting, label, group, seed, level, axis), counter in sorted(metrics.items()):
        row = {
            "setting": setting,
            "label": label,
            "group": group,
            "seed": seed,
            "evaluation_set": "cantt_holdout",
            "level": level,
            "axis": axis,
        }
        row.update(counter.row())
        rows.append(row)
    return rows


def evaluate_road(models: list[dict]) -> list[dict]:
    if not ROAD_FRAMES.exists() or not ROAD_PROFILE.exists():
        print("skip ROAD evaluation: preprocessed ROAD artifacts not found", flush=True)
        return []
    captures = sorted(ROAD_FRAMES.glob("*.parquet"))
    if not captures:
        return []
    profile = pd.read_csv(ROAD_PROFILE).set_index("capture")
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    for idx, path in enumerate(captures, start=1):
        capture = path.stem
        p = profile.loc[capture]
        family = p["family"]
        masquerade = "masquerade" if int(p.get("masquerade", 0)) else "fabrication"
        axis = "ambient" if p["role"] == "ambient" else f"{family}|{masquerade}"
        df = pd.read_parquet(path)
        features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        print(f"road eval {idx}/{len(captures)} {capture} {axis}", flush=True)
        for x, y, _starts in iter_windows(features, frame_y):
            for info in models:
                pred, _scores = predict_info(info, x)
                base = (info["setting"], info["label"], info["group"], info["seed"])
                metrics[(*base, "all", "all")].update(y, pred > 0)
                metrics[(*base, "axis", axis)].update(y, pred > 0)
    rows = []
    for (setting, label, group, seed, level, axis), counter in sorted(metrics.items()):
        row = {
            "setting": setting,
            "label": label,
            "group": group,
            "seed": seed,
            "evaluation_set": "road",
            "level": level,
            "axis": axis,
        }
        row.update(counter.row())
        rows.append(row)
    return rows


def add_training_log_columns(rows: list[dict]) -> None:
    logs: dict[tuple, dict] = {}
    for arm in ARM_CONFIGS:
        for seed in SEEDS:
            path = EXP / f"train_log_{arm}_seed{seed}.json"
            if path.exists():
                logs[(arm, seed)] = json.loads(path.read_text())
    for row in rows:
        log = logs.get((row["setting"], int(row["seed"])), {})
        for key in [
            "real_windows",
            "source_synthetic_windows",
            "target_normal_windows",
            "target_synthetic_windows",
            "oracle_attack_windows",
            "train_windows_total",
            "epochs_run",
            "best_epoch",
            "best_val_macro_f1_multiclass",
        ]:
            row[key] = log.get(key, "")


def write_outputs(rows: list[dict]) -> None:
    add_training_log_columns(rows)
    write_csv(TABLES / "target_normal_reinstantiation_by_seed.csv", rows)
    write_csv(EXP / "target_normal_reinstantiation_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "macro_f1_binary",
        "attack_recall",
        "normal_recall",
        "fpr",
        "fnr",
        "auprc",
        "ece",
    ]
    summary_rows = summarize(df, ["setting", "label", "group", "evaluation_set", "level", "axis"], metric_cols)
    write_csv(TABLES / "target_normal_reinstantiation_summary.csv", summary_rows)
    write_csv(EXP / "target_normal_reinstantiation_summary.csv", summary_rows)

    key_sets = ["test", "variant_test", "variant_sensitivity", "out_of_generator", "cantt_holdout", "otids", "road"]
    key = df[(df["level"] == "all") & (df["axis"] == "all") & (df["evaluation_set"].isin(key_sets))].copy()
    key_summary = summarize(key, ["setting", "label", "group", "evaluation_set"], metric_cols)
    write_csv(TABLES / "target_normal_reinstantiation_key_comparisons.csv", key_summary)

    axis = df[df["evaluation_set"].isin(["cantt_holdout", "road"]) & (df["level"] != "all")].copy()
    axis_summary = summarize(axis, ["setting", "label", "group", "evaluation_set", "level", "axis"], metric_cols)
    write_csv(TABLES / "target_normal_reinstantiation_by_axis.csv", axis_summary)


def run_training(arms: list[str], max_epochs: int, skip_existing: bool) -> None:
    for arm in arms:
        for seed in SEEDS:
            train_arm(arm, seed, max_epochs=max_epochs, skip_existing=skip_existing)


def run_evaluation(arms: list[str], skip_road: bool) -> None:
    start = time.time()
    models = load_models(arms)
    rows: list[dict] = []
    rows.extend(evaluate_npz(models, "test", WINDOWS / "test_windows.npz"))
    rows.extend(evaluate_npz(models, "variant_test", WINDOWS / "variant_test_windows.npz"))
    rows.extend(evaluate_npz(models, "variant_sensitivity", WINDOWS / "variant_sensitivity_windows.npz"))
    rows.extend(evaluate_npz(models, "out_of_generator", WINDOWS / "out_of_generator_stress_windows.npz"))
    rows.extend(evaluate_npz(models, "otids", WINDOWS / "otids_cross_windows.npz", y_key="y_binary"))
    rows.extend(evaluate_cantt_holdout(models))
    if not skip_road:
        rows.extend(evaluate_road(models))
    write_outputs(rows)
    log = {
        "role": "target_normal_reinstantiation_evaluation",
        "arms": arms,
        "rows": len(rows),
        "skip_road": skip_road,
        "elapsed_seconds": time.time() - start,
    }
    (EXP / "evaluation_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    (LOGS / "target_normal_reinstantiation_eval.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


def parse_arms(include_oracle: bool) -> list[str]:
    arms = list(MAIN_ARMS)
    if include_oracle:
        arms.extend(ORACLE_ARMS)
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate target-normal reinstantiation arms.")
    parser.add_argument("command", choices=["train", "evaluate", "all"])
    parser.add_argument("--arm", choices=list(ARM_CONFIGS), help="Run only one arm.")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="With --arm and command=train, run only one seed.")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-road", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()
    ensure_dirs()

    if args.arm:
        arms = [args.arm]
    else:
        arms = parse_arms(args.include_oracle)

    if args.command in ("train", "all"):
        if args.arm and args.seed is not None:
            train_arm(args.arm, args.seed, max_epochs=args.max_epochs, skip_existing=args.skip_existing)
        else:
            run_training(arms, max_epochs=args.max_epochs, skip_existing=args.skip_existing)
    if args.command in ("evaluate", "all"):
        run_evaluation(arms, skip_road=args.skip_road)


if __name__ == "__main__":
    main()
