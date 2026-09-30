#!/usr/bin/env python3
"""Diagnostic adaptation probe: can-train-and-test target-domain retraining.

DIAGNOSTIC ADAPTATION PROBE - EXPLICITLY DIAGNOSTIC USE OF can-train-and-test.

Purpose
-------
The paper concludes (by elimination) that repairing the external transfer
failure (can-train FPR ~= 1.0) "appears to require retraining-level
intervention on target-domain data". This probe tests that conclusion
directly: does retraining with a modest amount of can-train target NORMAL
data actually remove the external FPR ~= 1.0 failure, and at what cost to
attack recall and in-dataset (Car-Hacking) performance?

This uses the AGENTS.md exception that allows can-train-and-test for tuning
only when the experiment is "explicitly diagnostic and labeled as such".
All artifacts of this probe are diagnostic-only and must never be presented
as a deployment adaptation procedure or as cross-dataset robustness.
ROAD and OTIDS are NOT used for any training or tuning here.

Capture(file)-level calibration/held-out split
----------------------------------------------
- Population: the 176 can-train-and-test `test_*` CSV files (the same file
  population used by scripts/run_cantt_tier1.py external evaluation).
  `train_01` native-training files are excluded so the held-out evaluation
  stays a clean subset of the existing external benchmark.
- Deterministic rule: sort files by zip path (as in run_cantt_tier1
  list_cantt_files); every 5th file (sorted index % 5 == 0) is assigned to
  CALIBRATION (~20%, 36 files); the rest are HELD-OUT (140 files).
- Both normal-only and attack-containing files get assigned by this rule,
  but ONLY normal windows (no attack frame anywhere in the 128-frame
  window) from calibration files enter training. Held-out files are used
  exclusively for evaluation.
- Calibration normal windows are capped at 100,000 via a deterministic
  subsample (np.random.default_rng(2026) choice without replacement over
  the global window index, then sorted).

Windows use the canonical schema: 128 frames, stride 32, the same 11
features as run_cantt_tier1.py, no cross-file windows.

Settings (x 3 seeds: 7/42/123)
------------------------------
- adapt_real:   Car-Hacking real train windows + calibration normals.
- adapt_rule30: Car-Hacking real train + rule-based +30% synthetic pool
                (sampled exactly as in train_rule_synthetic_ratio_sweep:
                n_syn = round(0.30 * len(real_train)), sampling seed =
                seed + 300) + calibration normals.

Standardizer policy (intentional deviation, documented): the standardizer
is REFIT on each setting's full combined training windows (real [+ synth]
+ calibration normals). The ratio sweep fits on real train only to isolate
the augmentation effect; here the refit on target-domain-containing data
IS the retraining-level intervention under test.

Training matches the augmented arm: CNN1D, <=12 epochs, patience 3, AdamW
lr 1e-3 / weight decay 1e-4, batch 512, class-weighted CE, model selection
by validation multiclass Macro-F1 on the existing Car-Hacking validation
split (validation stays Car-Hacking-only; no can-train data in validation).

Evaluation (adapted models + pre-adaptation reference models real_only /
rule_0p30 from models/baseline and models/ratio_sweep, read-only):
 (i)   held-out can-train normal windows  -> FPR
 (ii)  held-out can-train attack windows  -> binary attack recall
 (iii) Car-Hacking test split             -> binary Macro-F1 / FPR / recall
 (iv)  variant sensitivity windows (rung-3 set of
       evaluate_variant_sensitivity.py)   -> binary attack detection recall

Artifacts (new files only, per task rules)
------------------------------------------
- models/adaptation/cnn1d_{setting}_seed{seed}.pt
- models/adaptation/standardizer_{setting}_seed{seed}.npz
- experiments/09_adaptation/ (calibration cache, split manifest, training
  histories, run log)
- results/tables/adaptation_probe_by_seed.csv
- results/tables/adaptation_probe_summary.csv
"""
import argparse
import csv
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_cantt_tier1 import (
    MetricCounter,
    ZIP_PATH,
    evaluate_batch,
    iter_windows,
    list_cantt_files,
    model_paths,
    read_cantt_csv,
)
from train_real_only_baselines import (
    CNN1D,
    WindowDataset,
    compute_metrics,
    fit_standardizer,
    predict_cnn,
    standardize,
)
from train_rule_synthetic_ratio_sweep import sample_synthetic_indices, train_model
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
MODELS = ROOT / "models" / "adaptation"
EXP = ROOT / "experiments" / "09_adaptation"
TABLES = ROOT / "results" / "tables"

SEEDS = [7, 42, 123]
ADAPT_SETTINGS = ["adapt_real", "adapt_rule30"]
REFERENCE_SETTINGS = [
    {"setting": "real_only", "family": "real_only", "ratio": 0.0, "label": "Real only (pre-adaptation ref)"},
    {"setting": "rule_0p30", "family": "rule", "ratio": 0.3, "label": "Rule +30% (pre-adaptation ref)"},
]
RULE_RATIO = 0.30
CALIB_CAP = 100_000
CALIB_SUBSAMPLE_SEED = 2026
CALIB_CACHE = EXP / "calibration_normal_windows.npz"
SPLIT_MANIFEST = EXP / "adaptation_file_split_manifest.csv"


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


def split_files() -> tuple[list, list]:
    """Deterministic capture(file)-level split over sorted test_* files."""
    test_files = [f for f in list_cantt_files() if f.subset_id.startswith("test_")]
    calibration = [f for i, f in enumerate(test_files) if i % 5 == 0]
    held_out = [f for i, f in enumerate(test_files) if i % 5 != 0]
    return calibration, held_out


def build_calibration(force: bool = False) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    calibration, held_out = split_files()
    manifest_rows = [
        {"role": "calibration", "path": f.path, "set_id": f.set_id, "subset_id": f.subset_id, "source_stem": f.source_stem}
        for f in calibration
    ] + [
        {"role": "held_out", "path": f.path, "set_id": f.set_id, "subset_id": f.subset_id, "source_stem": f.source_stem}
        for f in held_out
    ]
    write_csv(SPLIT_MANIFEST, manifest_rows)
    if CALIB_CACHE.exists() and not force:
        print(f"calibration cache exists: {CALIB_CACHE}", flush=True)
        return

    start = time.time()
    chunks: list[np.ndarray] = []
    file_rows = []
    for idx, meta in enumerate(calibration, start=1):
        with zipfile.ZipFile(ZIP_PATH) as zf:
            df, features, _ = read_cantt_csv(zf, meta)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        n_norm = 0
        n_attack = 0
        for x, y, _starts in iter_windows(features, frame_y):
            normal_mask = y == 0
            n_norm += int(normal_mask.sum())
            n_attack += int((~normal_mask).sum())
            if normal_mask.any():
                chunks.append(x[normal_mask])
        file_rows.append(
            {
                "path": meta.path,
                "subset_id": meta.subset_id,
                "source_stem": meta.source_stem,
                "normal_windows": n_norm,
                "attack_windows_excluded": n_attack,
            }
        )
        print(f"calibration {idx}/{len(calibration)} {meta.path} normal={n_norm} attack_excluded={n_attack}", flush=True)

    all_normal = np.concatenate(chunks, axis=0).astype(np.float32)
    chunks.clear()
    total = len(all_normal)
    if total > CALIB_CAP:
        rng = np.random.default_rng(CALIB_SUBSAMPLE_SEED)
        sel = rng.choice(total, size=CALIB_CAP, replace=False)
        sel.sort()
        sampled = all_normal[sel]
    else:
        sampled = all_normal
    np.savez(CALIB_CACHE, x=sampled, total_normal_windows_before_cap=np.int64(total))
    write_csv(EXP / "calibration_window_counts.csv", file_rows)
    log = {
        "role": "diagnostic adaptation probe calibration build",
        "calibration_files": len(calibration),
        "held_out_files": len(held_out),
        "normal_windows_before_cap": int(total),
        "attack_windows_excluded": int(sum(r["attack_windows_excluded"] for r in file_rows)),
        "calibration_windows_used": int(len(sampled)),
        "cap": CALIB_CAP,
        "subsample_seed": CALIB_SUBSAMPLE_SEED,
        "elapsed_seconds": time.time() - start,
    }
    (EXP / "calibration_build_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2), flush=True)


def build_training_set(setting: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    calib = np.load(CALIB_CACHE, allow_pickle=True)
    real_x = train["x"].astype(np.float32)
    real_y = train["y_attack_type"].astype(np.int64)
    calib_x = calib["x"].astype(np.float32)
    calib_y = np.zeros(len(calib_x), dtype=np.int64)  # calibration normals -> class 0

    parts_x = [real_x]
    parts_y = [real_y]
    n_syn = 0
    if setting == "adapt_rule30":
        syn = np.load(SYNTHETIC / "rule_based_windows.npz", allow_pickle=True)
        n_syn = int(round(len(real_x) * RULE_RATIO))
        syn_idx = sample_synthetic_indices(syn["y_attack_type"], n_syn, seed + int(RULE_RATIO * 1000))
        parts_x.append(syn["x"][syn_idx].astype(np.float32))
        parts_y.append(syn["y_attack_type"][syn_idx].astype(np.int64))
    elif setting != "adapt_real":
        raise ValueError(setting)
    parts_x.append(calib_x)
    parts_y.append(calib_y)

    train_x_raw = np.concatenate(parts_x, axis=0)
    train_y = np.concatenate(parts_y, axis=0)
    order = np.random.default_rng(seed).permutation(len(train_y))
    info = {
        "real_windows": int(len(real_x)),
        "synthetic_windows": int(n_syn),
        "calibration_windows": int(len(calib_x)),
        "train_windows_total": int(len(train_y)),
    }
    return train_x_raw[order], train_y[order], info


def run_training(skip_existing: bool = False, max_epochs: int = 12) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    val = np.load(WINDOWS / "val_windows.npz", allow_pickle=True)
    val_x_raw = val["x"].astype(np.float32)
    val_y = val["y_attack_type"].astype(np.int64)

    for setting in ADAPT_SETTINGS:
        for seed in SEEDS:
            model_path = MODELS / f"cnn1d_{setting}_seed{seed}.pt"
            if skip_existing and model_path.exists():
                print(f"skip existing {setting} seed{seed}", flush=True)
                continue
            start = time.time()
            print(f"=== adaptation probe setting={setting} seed={seed} ===", flush=True)
            train_x_raw, train_y, info = build_training_set(setting, seed)
            # Refit standardizer on the full combined training windows
            # (real [+ synthetic] + can-train calibration normals). This is
            # the retraining-level intervention under test; it intentionally
            # deviates from the ratio-sweep real-only scaler policy.
            mean, std = fit_standardizer(train_x_raw)
            np.savez(MODELS / f"standardizer_{setting}_seed{seed}.npz", mean=mean, std=std)
            train_x = standardize(train_x_raw, mean, std)
            del train_x_raw
            val_x = standardize(val_x_raw, mean, std)

            model, history, _device, best = train_model(train_x, train_y, val_x, val_y, seed, max_epochs=max_epochs)
            torch.save(model.state_dict(), model_path)
            write_csv(EXP / f"training_history_{setting}_seed{seed}.csv", history)
            log = {
                "role": "diagnostic adaptation probe training",
                "setting": setting,
                "seed": seed,
                **info,
                "epochs_run": len(history),
                "best_epoch": best["epoch"],
                "best_val_macro_f1_multiclass": best["f1"],
                "max_epochs": max_epochs,
                "standardizer_policy": "refit on combined training windows incl. can-train calibration normals",
                "elapsed_seconds": time.time() - start,
            }
            (EXP / f"train_log_{setting}_seed{seed}.json").write_text(json.dumps(log, indent=2) + "\n")
            print(json.dumps(log, indent=2), flush=True)


def load_eval_models() -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for setting in ADAPT_SETTINGS:
        for seed in SEEDS:
            stdz = np.load(MODELS / f"standardizer_{setting}_seed{seed}.npz")
            model = CNN1D(in_channels=11, classes=5).to(device)
            model.load_state_dict(torch.load(MODELS / f"cnn1d_{setting}_seed{seed}.pt", map_location=device))
            model.eval()
            loaded.append(
                {
                    "setting": setting,
                    "label": {"adapt_real": "Adapt real (+calib normals)", "adapt_rule30": "Adapt rule+30% (+calib normals)"}[setting],
                    "group": "adapted",
                    "seed": seed,
                    "model": model,
                    "mean": stdz["mean"].astype(np.float32),
                    "std": stdz["std"].astype(np.float32),
                    "device": device,
                }
            )
    for ref in REFERENCE_SETTINGS:
        for seed in SEEDS:
            model_path, scaler_path = model_paths(ref, seed)
            stdz = np.load(scaler_path)
            model = CNN1D(in_channels=11, classes=5).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            loaded.append(
                {
                    "setting": f"ref_{ref['setting']}",
                    "label": ref["label"],
                    "group": "pre_adaptation_reference",
                    "seed": seed,
                    "model": model,
                    "mean": stdz["mean"].astype(np.float32),
                    "std": stdz["std"].astype(np.float32),
                    "device": device,
                }
            )
    return loaded


def evaluate_holdout(models: list[dict]) -> dict[tuple, dict]:
    _calibration, held_out = split_files()
    metrics: dict[tuple, MetricCounter] = defaultdict(MetricCounter)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for idx, meta in enumerate(held_out, start=1):
            print(f"holdout eval {idx}/{len(held_out)} {meta.path}", flush=True)
            df, features, _ = read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)
            for x, y, _starts in iter_windows(features, frame_y):
                with torch.no_grad():
                    for info in models:
                        pred, _scores = evaluate_batch(info, x)
                        metrics[(info["setting"], info["seed"])].update(y, pred)
    return {key: counter.row() for key, counter in metrics.items()}


def evaluate_car_hacking_test(models: list[dict]) -> dict[tuple, dict]:
    test = np.load(WINDOWS / "test_windows.npz", allow_pickle=True)
    test_x_raw = test["x"].astype(np.float32)
    test_y = test["y_attack_type"].astype(np.int64)
    out = {}
    for info in models:
        x = standardize(test_x_raw, info["mean"], info["std"])
        loader = DataLoader(WindowDataset(x, test_y), batch_size=1024, shuffle=False, num_workers=2)
        _, probs = predict_cnn(info["model"], loader, info["device"])
        pred = probs.argmax(axis=1)
        scores = 1.0 - probs[:, 0]
        out[(info["setting"], info["seed"])] = compute_metrics(info["setting"], "test", test_y, scores, pred, info["seed"])
        print(f"car-hacking test {info['setting']} seed={info['seed']} macro_f1={out[(info['setting'], info['seed'])]['macro_f1_binary']:.5f}", flush=True)
    return out


def evaluate_variant_sensitivity(models: list[dict]) -> dict[tuple, dict]:
    data = np.load(WINDOWS / "variant_sensitivity_windows.npz", allow_pickle=True)
    x_raw = data["x"].astype(np.float32)
    y = data["y_attack_type"].astype(np.int64)
    attack_mask = y > 0
    normal_mask = ~attack_mask
    out = {}
    for info in models:
        x = standardize(x_raw, info["mean"], info["std"])
        loader = DataLoader(WindowDataset(x, y), batch_size=1024, shuffle=False, num_workers=2)
        _, probs = predict_cnn(info["model"], loader, info["device"])
        pred_binary = (probs.argmax(axis=1) > 0).astype(np.int8)
        row = {
            "attack_windows": int(attack_mask.sum()),
            "attack_detection_recall": float(pred_binary[attack_mask].mean()) if attack_mask.any() else "",
            "normal_windows": int(normal_mask.sum()),
            "fpr": float(pred_binary[normal_mask].mean()) if normal_mask.any() else "",
        }
        out[(info["setting"], info["seed"])] = row
        print(f"variant sensitivity {info['setting']} seed={info['seed']} detection_recall={row['attack_detection_recall']:.5f}", flush=True)
    return out


def run_evaluation() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    start = time.time()
    models = load_eval_models()
    ch_rows = evaluate_car_hacking_test(models)
    vs_rows = evaluate_variant_sensitivity(models)
    holdout_rows = evaluate_holdout(models)

    train_logs = {}
    for setting in ADAPT_SETTINGS:
        for seed in SEEDS:
            path = EXP / f"train_log_{setting}_seed{seed}.json"
            if path.exists():
                train_logs[(setting, seed)] = json.loads(path.read_text())

    by_seed = []
    for info in models:
        key = (info["setting"], info["seed"])
        hold = holdout_rows[key]
        ch = ch_rows[key]
        vs = vs_rows[key]
        log = train_logs.get(key, {})
        by_seed.append(
            {
                "setting": info["setting"],
                "label": info["label"],
                "group": info["group"],
                "seed": info["seed"],
                "cantt_holdout_normal_windows": hold["normal_windows"],
                "cantt_holdout_attack_windows": hold["attack_windows"],
                "cantt_holdout_fpr": hold["fpr"],
                "cantt_holdout_normal_recall": hold["normal_recall"],
                "cantt_holdout_attack_recall": hold["attack_recall"],
                "cantt_holdout_macro_f1_binary": hold["macro_f1_binary"],
                "ch_test_macro_f1_binary": ch["macro_f1_binary"],
                "ch_test_fpr": ch["fpr"],
                "ch_test_attack_recall": ch["attack_recall"],
                "ch_test_accuracy": ch["accuracy"],
                "variant_sens_attack_detection_recall": vs["attack_detection_recall"],
                "variant_sens_fpr": vs["fpr"],
                "train_windows_total": log.get("train_windows_total", ""),
                "calibration_windows": log.get("calibration_windows", ""),
                "synthetic_windows": log.get("synthetic_windows", ""),
                "epochs_run": log.get("epochs_run", ""),
                "best_epoch": log.get("best_epoch", ""),
                "best_val_macro_f1_multiclass": log.get("best_val_macro_f1_multiclass", ""),
            }
        )
    write_csv(TABLES / "adaptation_probe_by_seed.csv", by_seed)
    write_csv(EXP / "adaptation_probe_by_seed.csv", by_seed)

    calibration, held_out = split_files()
    calib_cache = np.load(CALIB_CACHE, allow_pickle=True)
    df = pd.DataFrame(by_seed)
    metric_cols = [
        "cantt_holdout_fpr",
        "cantt_holdout_attack_recall",
        "cantt_holdout_macro_f1_binary",
        "ch_test_macro_f1_binary",
        "ch_test_fpr",
        "ch_test_attack_recall",
        "variant_sens_attack_detection_recall",
        "variant_sens_fpr",
        "best_val_macro_f1_multiclass",
    ]
    summary_rows = []
    for (setting, label, group), grp in df.groupby(["setting", "label", "group"], sort=False):
        row = {
            "setting": setting,
            "label": label,
            "group": group,
            "seeds": ";".join(str(s) for s in sorted(grp["seed"])),
            "n_seeds": len(grp),
            "calibration_files": len(calibration),
            "held_out_files": len(held_out),
            "calibration_windows": int(len(calib_cache["x"])),
        }
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals):
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)
    write_csv(TABLES / "adaptation_probe_summary.csv", summary_rows)
    write_csv(EXP / "adaptation_probe_summary.csv", summary_rows)
    (EXP / "evaluation_log.json").write_text(
        json.dumps(
            {
                "role": "diagnostic adaptation probe evaluation",
                "models_evaluated": [(m["setting"], m["seed"]) for m in models],
                "held_out_files": len(held_out),
                "elapsed_seconds": time.time() - start,
            },
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic adaptation probe (can-train target-normal retraining).")
    parser.add_argument("command", choices=["calib", "train", "evaluate", "all"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-calib", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()
    if args.command in ("calib", "all"):
        build_calibration(force=args.force_calib)
    if args.command in ("train", "all"):
        run_training(skip_existing=args.skip_existing, max_epochs=args.max_epochs)
    if args.command in ("evaluate", "all"):
        run_evaluation()


if __name__ == "__main__":
    main()
