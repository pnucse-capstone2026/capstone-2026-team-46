#!/usr/bin/env python3
"""Support-factor decomposition for non-ID external failure attribution."""
from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_cantt_id_ablation import car_hacking_train_normal_window_feature_means
from run_cantt_tier1 import FEATURE_NAMES, ZIP_PATH, MetricCounter, iter_windows, read_cantt_csv
from target_normal_reinstantiation_common import (
    CALIB_NORMAL_CACHE,
    FIGURES,
    LOGS,
    ROOT,
    SEEDS,
    SUPPORT_EXP,
    SUPPORT_MODELS,
    TABLES,
    WINDOWS,
    ensure_dirs,
    repo_rel,
    split_cantt_files,
    summarize,
    write_csv,
)
from train_real_only_baselines import CNN1D, compute_metrics, fit_standardizer, standardize
from train_rule_synthetic_ratio_sweep import train_model

ARMS = {
    "payload_only": {"label": "Payload only", "keep": [2, 3, 4, 5, 6, 7, 8, 9]},
    "timing_only": {"label": "Timing only", "keep": [10]},
    "dlc_timing": {"label": "DLC + timing", "keep": [1, 10]},
    "id_payload": {"label": "ID + payload", "keep": [0, 2, 3, 4, 5, 6, 7, 8, 9]},
    "payload_timing": {"label": "Payload + timing", "keep": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
}
EVAL_BATCH_SIZE = 8192
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
ROAD_PROFILE = ROOT / "results" / "tables" / "road_dataset_profile.csv"
PAPER_TABLES = ROOT / "results" / "paper" / "tables"
PAPER_FIGURES = ROOT / "results" / "paper" / "figures"
MANIFEST = ROOT / "results" / "paper" / "paper_artifacts_manifest.md"


def mask_to_kept_channels(x: np.ndarray, keep: list[int], constants: np.ndarray) -> np.ndarray:
    out = x.copy()
    keep_set = set(keep)
    for idx in range(out.shape[-1]):
        if idx not in keep_set:
            out[:, :, idx] = constants[idx]
    return out


def build_training_set(arm: str) -> tuple[np.ndarray, np.ndarray, dict]:
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    if not CALIB_NORMAL_CACHE.exists():
        raise SystemExit("Run scripts/build_target_normal_reinstantiation_cache.py first.")
    calib = np.load(CALIB_NORMAL_CACHE, allow_pickle=True)
    x = np.concatenate([train["x"].astype(np.float32), calib["x"].astype(np.float32)], axis=0)
    y = np.concatenate(
        [
            train["y_attack_type"].astype(np.int64),
            np.zeros(len(calib["x"]), dtype=np.int64),
        ]
    )
    return x, y, {
        "real_windows": int(len(train["x"])),
        "target_normal_windows": int(len(calib["x"])),
        "train_windows_total": int(len(y)),
    }


def train_arm(arm: str, seed: int, max_epochs: int, skip_existing: bool) -> None:
    model_path = SUPPORT_MODELS / f"cnn1d_{arm}_seed{seed}.pt"
    if skip_existing and model_path.exists():
        print(f"skip existing support arm={arm} seed={seed}", flush=True)
        return
    start = time.time()
    print(f"=== support-factor arm={arm} seed={seed} ===", flush=True)
    SUPPORT_MODELS.mkdir(parents=True, exist_ok=True)
    SUPPORT_EXP.mkdir(parents=True, exist_ok=True)
    constants = car_hacking_train_normal_window_feature_means().astype(np.float32)
    keep = ARMS[arm]["keep"]

    val = np.load(WINDOWS / "val_windows.npz", allow_pickle=True)
    train_x_raw, train_y, info = build_training_set(arm)
    train_x_raw = mask_to_kept_channels(train_x_raw, keep, constants)
    val_x_raw = mask_to_kept_channels(val["x"].astype(np.float32), keep, constants)
    order = np.random.default_rng(seed).permutation(len(train_y))
    train_x_raw = train_x_raw[order]
    train_y = train_y[order]

    mean, std = fit_standardizer(train_x_raw)
    np.savez(SUPPORT_MODELS / f"standardizer_{arm}_seed{seed}.npz", mean=mean, std=std)
    train_x = standardize(train_x_raw, mean, std)
    val_x = standardize(val_x_raw, mean, std)
    model, history, _device, best = train_model(train_x, train_y, val_x, val["y_attack_type"].astype(np.int64), seed, max_epochs=max_epochs)
    torch.save(model.state_dict(), model_path)
    write_csv(SUPPORT_EXP / f"training_history_{arm}_seed{seed}.csv", history)
    log = {
        "arm": arm,
        "seed": seed,
        "label": ARMS[arm]["label"],
        "kept_channels": keep,
        "constant_policy": "Car-Hacking train-normal feature means",
        **info,
        "epochs_run": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1_multiclass": float(best["f1"]),
        "elapsed_seconds": time.time() - start,
    }
    (SUPPORT_EXP / f"train_log_{arm}_seed{seed}.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True), flush=True)


def load_models(arms: list[str]) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for arm in arms:
        for seed in SEEDS:
            stdz = np.load(SUPPORT_MODELS / f"standardizer_{arm}_seed{seed}.npz")
            model = CNN1D(in_channels=11, classes=5).to(device)
            model.load_state_dict(torch.load(SUPPORT_MODELS / f"cnn1d_{arm}_seed{seed}.pt", map_location=device))
            model.eval()
            loaded.append(
                {
                    "setting": arm,
                    "label": ARMS[arm]["label"],
                    "seed": seed,
                    "model": model,
                    "mean": stdz["mean"].astype(np.float32),
                    "std": stdz["std"].astype(np.float32),
                    "device": device,
                    "keep": ARMS[arm]["keep"],
                }
            )
    return loaded


def predict_info(info: dict, x_raw: np.ndarray, constants: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_masked = mask_to_kept_channels(x_raw.astype(np.float32), info["keep"], constants)
    x = standardize(x_masked, info["mean"], info["std"])
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


def evaluate_npz(models: list[dict], split: str, path, constants: np.ndarray, y_key: str = "y_attack_type") -> list[dict]:
    data = np.load(path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data[y_key].astype(np.int64)
    rows = []
    for info in models:
        print(f"support eval {split} {info['setting']} seed={info['seed']}", flush=True)
        pred, scores = predict_info(info, x, constants)
        row = compute_metrics(info["setting"], split, y, scores, pred, info["seed"])
        row.update({"setting": info["setting"], "label": info["label"], "evaluation_set": split, "level": "all", "axis": "all"})
        rows.append(row)
    return rows


def evaluate_cantt(models: list[dict], constants: np.ndarray) -> list[dict]:
    _calibration, held_out = split_cantt_files()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(held_out, start=1):
            print(f"support cantt {idx}/{len(held_out)} {meta.path}", flush=True)
            df, features, _issues = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                for info in models:
                    pred, _scores = predict_info(info, x, constants)
                    base = (info["setting"], info["label"], info["seed"])
                    metrics[(*base, "all", "all")].update(y, pred > 0)
                    metrics[(*base, "vehicle_axis", meta.vehicle_axis)].update(y, pred > 0)
                    metrics[(*base, "attack_axis", meta.attack_axis)].update(y, pred > 0)
    rows = []
    for (setting, label, seed, level, axis), counter in sorted(metrics.items()):
        row = {"setting": setting, "label": label, "seed": seed, "evaluation_set": "cantt_holdout", "level": level, "axis": axis}
        row.update(counter.row())
        rows.append(row)
    return rows


def evaluate_road(models: list[dict], constants: np.ndarray) -> list[dict]:
    if not ROAD_FRAMES.exists() or not ROAD_PROFILE.exists():
        print("skip support ROAD evaluation: preprocessed ROAD artifacts not found", flush=True)
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
        print(f"support road {idx}/{len(captures)} {capture} {axis}", flush=True)
        for x, y, _starts in iter_windows(features, frame_y):
            for info in models:
                pred, _scores = predict_info(info, x, constants)
                base = (info["setting"], info["label"], info["seed"])
                metrics[(*base, "all", "all")].update(y, pred > 0)
                metrics[(*base, "axis", axis)].update(y, pred > 0)
    rows = []
    for (setting, label, seed, level, axis), counter in sorted(metrics.items()):
        row = {"setting": setting, "label": label, "seed": seed, "evaluation_set": "road", "level": level, "axis": axis}
        row.update(counter.row())
        rows.append(row)
    return rows


def plot_support_fpr(key_summary: list[dict]) -> Path:
    df = pd.DataFrame(key_summary)
    sub = df[df["evaluation_set"].isin(["otids", "cantt_holdout", "road"])].copy()
    if sub.empty:
        raise RuntimeError("No external support-factor rows found.")
    order = list(ARMS)
    eval_order = ["otids", "cantt_holdout", "road"]
    labels = {"otids": "OTIDS", "cantt_holdout": "can-train", "road": "ROAD"}
    width = 0.24
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, eval_set in enumerate(eval_order):
        vals = []
        errs = []
        for arm in order:
            row = sub[(sub["setting"] == arm) & (sub["evaluation_set"] == eval_set)]
            vals.append(float(row["fpr_mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["fpr_std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (i - 1) * width, vals, width=width, yerr=errs, capsize=2, label=labels[eval_set])
    ax.set_xticks(x, [ARMS[a]["label"] for a in order], rotation=25, ha="right")
    ax.set_ylabel("False positive rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = FIGURES / "support_factor_decomposition_fpr.png"
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def promote_support_artifacts(key_summary: list[dict]) -> None:
    fig = plot_support_fpr(key_summary)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    table = TABLES / "support_factor_decomposition_key_comparisons.csv"
    copied_table = PAPER_TABLES / "table_49_support_factor_decomposition_key_comparisons.csv"
    copied_fig = PAPER_FIGURES / "figure_40_support_factor_decomposition_fpr.png"
    shutil.copy2(table, copied_table)
    shutil.copy2(fig, copied_fig)
    if MANIFEST.exists():
        marker = "## Support-Factor Decomposition Addendum"
        text = MANIFEST.read_text()
        block = f"""

{marker}

- **SFD**: Support-factor decomposition masks non-selected feature channels to Car-Hacking train-normal means after target-normal reinstantiation, testing which non-ID support factors carry the external false-positive ceiling.
  - Tables: {copied_table.name}
  - Figures: {copied_fig.name}
"""
        if marker in text:
            text = text.split(marker)[0].rstrip() + block
        else:
            text = text.rstrip() + block
        MANIFEST.write_text(text + "\n")
    log = {"outputs": [repo_rel(table), repo_rel(fig), repo_rel(copied_table), repo_rel(copied_fig)]}
    (SUPPORT_EXP / "figure_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")


def run_training(arms: list[str], max_epochs: int, skip_existing: bool) -> None:
    for arm in arms:
        for seed in SEEDS:
            train_arm(arm, seed, max_epochs=max_epochs, skip_existing=skip_existing)


def run_evaluation(arms: list[str]) -> None:
    constants = car_hacking_train_normal_window_feature_means().astype(np.float32)
    models = load_models(arms)
    rows = []
    rows.extend(evaluate_npz(models, "test", WINDOWS / "test_windows.npz", constants))
    rows.extend(evaluate_npz(models, "variant_test", WINDOWS / "variant_test_windows.npz", constants))
    rows.extend(evaluate_npz(models, "otids", WINDOWS / "otids_cross_windows.npz", constants, y_key="y_binary"))
    rows.extend(evaluate_cantt(models, constants))
    rows.extend(evaluate_road(models, constants))
    write_csv(TABLES / "support_factor_decomposition_by_seed.csv", rows)
    write_csv(SUPPORT_EXP / "support_factor_decomposition_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    metric_cols = ["accuracy", "macro_f1_binary", "attack_recall", "normal_recall", "fpr", "fnr", "auprc", "ece"]
    summary = summarize(df, ["setting", "label", "evaluation_set", "level", "axis"], metric_cols)
    write_csv(TABLES / "support_factor_decomposition_summary.csv", summary)
    key = df[(df["level"] == "all") & (df["axis"] == "all")]
    key_summary = summarize(key, ["setting", "label", "evaluation_set"], metric_cols)
    write_csv(TABLES / "support_factor_decomposition_key_comparisons.csv", key_summary)
    promote_support_artifacts(key_summary)
    log = {"role": "support_factor_decomposition_evaluation", "arms": arms, "rows": len(rows)}
    (SUPPORT_EXP / "evaluation_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    (LOGS / "support_factor_decomposition_eval.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(key_summary).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate support-factor decomposition arms.")
    parser.add_argument("command", choices=["train", "evaluate", "all"])
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()
    ensure_dirs()
    arms = [args.arm] if args.arm else list(ARMS)
    if args.command in ("train", "all"):
        if args.arm and args.seed is not None:
            train_arm(args.arm, args.seed, args.max_epochs, args.skip_existing)
        else:
            run_training(arms, args.max_epochs, args.skip_existing)
    if args.command in ("evaluate", "all"):
        run_evaluation(arms)


if __name__ == "__main__":
    main()
