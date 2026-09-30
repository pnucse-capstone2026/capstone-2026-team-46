#!/usr/bin/env python3
"""Evaluate family-extension models (LSTM classifier, Conv-AE) on ladder rungs L0-L5.

Rungs: L0 source test, L1 fixed variant, L2 sensitivity sweep, L3 target-ID shift,
L4 out-of-generator payload stress, L5 OTIDS / can-train-and-test / ROAD externals.
External reporting is FPR-first per AGENTS.md. AE rows appear as two threshold
policies (percentile vs synthetic_calibrated) sharing one trained model per seed.

Outputs:
  journal/results/tables/family_extension_by_seed.csv
  journal/results/tables/family_extension_summary.csv
  journal/results/logs/evaluate_family_extension.log
"""
import argparse
import json
import time
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import lib_common as lc
import lib_cantt as lt
from strict_v2_evaluation import (
    bundle_provenance,
    evaluator_source_and_input_provenance,
    load_identical_saved_standardizer,
    validate_strict_v2_evaluation_scope,
)
from train_family_extension_ae import ConvAE, reconstruction_errors
from train_family_extension_lstm import (
    LSTMClassifier,
    checkpoint_path,
    log_path,
    standardizer_path,
)

LSTM_SETTINGS = ["real_only", "rule_0p30", "rule_1p00"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE_PERCENTILE = 99.5
AE_SYN_RATIO = 0.30  # synthetic calibration sample size as a fraction of val windows


def binary_row(y_true_binary, y_pred_binary):
    c = lt.MetricCounter()
    c.update(y_true_binary, y_pred_binary)
    return c.row()


def load_standardizer():
    train = lc.load_npz("train")
    mean, std = lc.fit_standardizer(train["x"])
    return mean, std


def load_lstm_models(settings=None, seeds=None, model_tag=""):
    settings = LSTM_SETTINGS if settings is None else settings
    seeds = lc.SEEDS if seeds is None else seeds
    models = {}
    for setting in settings:
        for seed in seeds:
            path = checkpoint_path(setting, seed, model_tag)
            if not path.exists():
                print(f"WARN missing {path.name}; skipping", flush=True)
                continue
            model = LSTMClassifier()
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE).eval()
            models[(setting, seed)] = model
    return models


def validate_strict_v2_lstm_bundles(settings, seeds, model_tag):
    if not lc.is_strict_v2_tag(model_tag):
        return []
    provenance = []
    for setting in settings:
        for seed in seeds:
            checkpoint = checkpoint_path(setting, seed, model_tag)
            standardizer = standardizer_path(setting, seed, model_tag)
            training_log = log_path(setting, seed, model_tag)
            artifacts = {
                "checkpoint": checkpoint,
                "standardizer": standardizer,
            }
            record = lc.validate_strict_v2_training_bundle(
                training_log,
                artifacts,
                {
                    "family": "lstm",
                    "setting": setting,
                    "seed": seed,
                    "model_tag": model_tag,
                    "sampling_policy": "strict-v2",
                },
            )
            provenance.append(bundle_provenance(
                family="lstm",
                setting=setting,
                seed=seed,
                tag=model_tag,
                log_path=training_log,
                artifacts=artifacts,
                validated_record=record,
            ))
    return provenance


def load_ae_models():
    models = {}
    for seed in lc.SEEDS:
        path = lc.MODELS / f"ae_seed{seed}.pt"
        if not path.exists():
            print(f"WARN missing {path.name}; skipping", flush=True)
            continue
        model = ConvAE()
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        models[seed] = model
    return models


def ae_thresholds(ae_models, mean, std):
    """Per-seed thresholds: val-normal percentile and synthetic-calibrated best-F1."""
    val = lc.load_npz("val")
    val_x = lc.standardize(val["x"], mean, std)
    val_normal = val_x[val["y_attack_type"] == 0]
    syn_x, syn_y = lc.load_synthetic_pool()
    thresholds = {}
    for seed, model in ae_models.items():
        normal_err = reconstruction_errors(model, val_normal, DEVICE)
        thr_pct = float(np.percentile(normal_err, AE_PERCENTILE))
        n_syn = round(len(val_x) * AE_SYN_RATIO)
        idx = lc.sample_synthetic_indices(syn_y, n_syn, seed + int(AE_SYN_RATIO * 1000))
        syn_err = reconstruction_errors(model, lc.standardize(syn_x[idx], mean, std), DEVICE)
        errs = np.concatenate([normal_err, syn_err])
        labels = np.concatenate([np.zeros(len(normal_err), np.int8), np.ones(len(syn_err), np.int8)])
        candidates = np.quantile(errs, np.linspace(0.0, 1.0, 512))
        best_f1, best_thr = -1.0, thr_pct
        pos = labels.sum()
        for thr in candidates:
            pred = errs > thr
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            fn = int(pos - tp)
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        thresholds[seed] = {"percentile": thr_pct, "synthetic_calibrated": best_thr,
                            "synthetic_calibration_f1": best_f1}
        print(f"ae seed={seed} thr_pct={thr_pct:.5f} thr_syn={best_thr:.5f} calib_f1={best_f1:.4f}", flush=True)
    return thresholds


def lstm_predict_batches(model, x_std, batch_size=2048):
    preds = np.empty(len(x_std), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(x_std), batch_size):
            batch = torch.from_numpy(x_std[i : i + batch_size]).to(DEVICE)
            preds[i : i + len(batch)] = model(batch).argmax(dim=1).cpu().numpy()
    return preds


def eval_npz_rungs(rows, lstm_models, ae_models, ae_thr, mean, std):
    rung_specs = [
        ("L0_source_test", "test"),
        ("L1_fixed_variant", "variant_test"),
        ("L2_sensitivity", "variant_sensitivity"),
        ("L3_target_id", "target_id_shift_stress"),
        ("L4_out_of_generator", "out_of_generator_stress"),
        ("L5_otids", "otids_cross"),
    ]
    for rung, split in rung_specs:
        data = lc.load_npz(split)
        x = lc.standardize(data["x"], mean, std)
        y_type = data["y_attack_type"]
        y_bin = (y_type > 0).astype(np.int8)
        subsets = {"all": np.ones(len(x), dtype=bool)}
        if rung == "L2_sensitivity":
            for sev in ["low", "medium", "high"]:
                subsets[f"severity:{sev}"] = np.asarray(data["severity"]) == sev
        if rung in ("L3_target_id", "L4_out_of_generator"):
            attack = np.asarray(data["attack_type"])
            cond = np.asarray(data["id_condition"])
            for atk in ["Gear", "RPM"]:
                for c in ["canonical", "shifted"]:
                    subsets[f"{atk}:{c}"] = (attack == atk) & (cond == c)
        print(f"eval {rung}: {len(x)} windows, {len(subsets)} subsets", flush=True)

        for (setting, seed), model in lstm_models.items():
            pred = lstm_predict_batches(model, x)
            pred_bin = (pred > 0).astype(np.int8)
            for name, mask in subsets.items():
                row = {"family": "lstm", "setting": setting, "seed": seed, "rung": rung, "subset": name}
                row.update(binary_row(y_bin[mask], pred_bin[mask]))
                atk_mask = mask & (y_type > 0)
                row["exact_recall"] = float((pred[atk_mask] == y_type[atk_mask]).mean()) if atk_mask.any() else ""
                rows.append(row)

        for seed, model in ae_models.items():
            err = reconstruction_errors(model, x, DEVICE)
            for policy in ["percentile", "synthetic_calibrated"]:
                pred_bin = (err > ae_thr[seed][policy]).astype(np.int8)
                for name, mask in subsets.items():
                    row = {"family": "ae", "setting": f"ae_{policy}", "seed": seed, "rung": rung, "subset": name}
                    row.update(binary_row(y_bin[mask], pred_bin[mask]))
                    row["exact_recall"] = ""
                    rows.append(row)


def eval_cantt(rows, lstm_models, ae_models, ae_thr, mean, std):
    files = [f for f in lt.list_cantt_files() if f.subset_id.startswith("test_")]
    counters = defaultdict(lt.MetricCounter)
    t0 = time.time()
    with zipfile.ZipFile(lt.ZIP_PATH) as zf:
        for i, meta in enumerate(files, 1):
            df, features, _issues = lt.read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(np.int8)
            for x_raw, y, _starts in lt.iter_windows(features, frame_y):
                x = lc.standardize(x_raw, mean, std)
                for (setting, seed), model in lstm_models.items():
                    pred_bin = (lstm_predict_batches(model, x) > 0).astype(np.int8)
                    for key in [("all", "all"), ("vehicle_axis", meta.vehicle_axis), ("attack_axis", meta.attack_axis)]:
                        counters[("lstm", setting, seed, *key)].update(y, pred_bin)
                for seed, model in ae_models.items():
                    err = reconstruction_errors(model, x, DEVICE)
                    for policy in ["percentile", "synthetic_calibrated"]:
                        pred_bin = (err > ae_thr[seed][policy]).astype(np.int8)
                        for key in [("all", "all"), ("vehicle_axis", meta.vehicle_axis), ("attack_axis", meta.attack_axis)]:
                            counters[("ae", f"ae_{policy}", seed, *key)].update(y, pred_bin)
            if i % 20 == 0:
                print(f"cantt {i}/{len(files)} files elapsed={time.time()-t0:.0f}s", flush=True)
    for (family, setting, seed, level, axis), counter in sorted(counters.items()):
        subset = "all" if level == "all" else f"{level}:{axis}"
        row = {"family": family, "setting": setting, "seed": seed, "rung": "L5_cantt", "subset": subset}
        row.update(counter.row())
        row["exact_recall"] = ""
        rows.append(row)


def eval_road(rows, lstm_models, ae_models, ae_thr, mean, std):
    frames_dir = lc.ROOT / "datasets" / "processed" / "road_frames"
    profile = pd.read_csv(lc.WISA_TABLES / "road_dataset_profile.csv").set_index("capture")  # read-only reference
    captures = sorted(frames_dir.glob("*.parquet"))
    counters = defaultdict(lt.MetricCounter)
    t0 = time.time()
    for i, path in enumerate(captures, 1):
        info = profile.loc[path.stem]
        masq = "masquerade" if int(info.get("masquerade", 0)) else "fabrication"
        axis = "ambient" if info["role"] == "ambient" else f"{info['family']}|{masq}"
        df = pd.read_parquet(path)
        features = df[lt.FEATURE_NAMES].to_numpy(dtype=np.float32)
        frame_y = df["attack"].to_numpy(dtype=np.int8)
        for x_raw, y, _starts in lt.iter_windows(features, frame_y):
            x = lc.standardize(x_raw, mean, std)
            for (setting, seed), model in lstm_models.items():
                pred_bin = (lstm_predict_batches(model, x) > 0).astype(np.int8)
                counters[("lstm", setting, seed, "all", "all")].update(y, pred_bin)
                counters[("lstm", setting, seed, "axis", axis)].update(y, pred_bin)
            for seed, model in ae_models.items():
                err = reconstruction_errors(model, x, DEVICE)
                for policy in ["percentile", "synthetic_calibrated"]:
                    pred_bin = (err > ae_thr[seed][policy]).astype(np.int8)
                    counters[("ae", f"ae_{policy}", seed, "all", "all")].update(y, pred_bin)
                    counters[("ae", f"ae_{policy}", seed, "axis", axis)].update(y, pred_bin)
        if i % 10 == 0:
            print(f"road {i}/{len(captures)} captures elapsed={time.time()-t0:.0f}s", flush=True)
    for (family, setting, seed, level, axis), counter in sorted(counters.items()):
        subset = "all" if level == "all" else f"axis:{axis}"
        row = {"family": family, "setting": setting, "seed": seed, "rung": "L5_road", "subset": subset}
        row.update(counter.row())
        row["exact_recall"] = ""
        rows.append(row)


def summarize(rows):
    df = pd.DataFrame(rows)
    num_cols = ["fpr", "normal_recall", "attack_recall", "macro_f1_binary", "exact_recall"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = []
    for (family, setting, rung, subset), g in df.groupby(["family", "setting", "rung", "subset"], sort=True):
        row = {"family": family, "setting": setting, "rung": rung, "subset": subset,
               "seeds": ";".join(str(s) for s in sorted(g["seed"].unique())),
               "windows": int(g["windows"].iloc[0])}
        for col in num_cols:
            vals = g[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else ""
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) else "")
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-npz", action="store_true")
    parser.add_argument("--skip-cantt", action="store_true")
    parser.add_argument("--skip-road", action="store_true")
    parser.add_argument("--skip-ae", action="store_true",
                        help="evaluate only the requested BiLSTM checkpoints")
    parser.add_argument("--lstm-settings", default=None,
                        help="comma-separated BiLSTM settings (default: all)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated pipeline seeds (default: all)")
    parser.add_argument("--lstm-model-tag", default="",
                        help="checkpoint suffix without leading '_'")
    parser.add_argument("--out-suffix", default="",
                        help="versioned result/log suffix beginning with '_'")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless every requested BiLSTM seed checkpoint exists",
    )
    args = parser.parse_args()

    lstm_settings = (
        args.lstm_settings.split(",") if args.lstm_settings else LSTM_SETTINGS
    )
    seeds = (
        [int(value) for value in args.seeds.split(",")]
        if args.seeds else lc.SEEDS
    )
    unknown = sorted(set(lstm_settings) - set(LSTM_SETTINGS))
    if unknown:
        parser.error(f"unknown BiLSTM setting(s): {unknown}")
    if args.out_suffix and not args.out_suffix.startswith("_"):
        parser.error("--out-suffix must start with '_'")
    try:
        lc.validate_artifact_tag(
            args.lstm_model_tag, name="--lstm-model-tag"
        )
        lc.validate_artifact_tag(
            args.out_suffix,
            name="--out-suffix",
            leading_underscore=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_strict_v2_evaluation_scope(
            args.lstm_model_tag,
            args.out_suffix,
            seeds,
            reduced_scope=(
                lstm_settings != LSTM_SETTINGS
                or seeds != lc.SEEDS
                or args.skip_npz
                or args.skip_cantt
                or args.skip_road
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if lc.is_strict_v2_tag(args.lstm_model_tag):
        if not args.require_complete:
            parser.error("strict-v2 model tags require --require-complete")
        if not args.skip_ae:
            parser.error(
                "strict-v2 BiLSTM evaluation requires --skip-ae; the legacy "
                "AE and its synthetic-calibrated threshold are outside E13"
            )
        if args.lstm_model_tag not in args.out_suffix:
            parser.error(
                "strict-v2 --out-suffix must contain --lstm-model-tag"
            )
    non_primary = (
        lstm_settings != LSTM_SETTINGS
        or seeds != lc.SEEDS
        or bool(args.lstm_model_tag)
        or args.skip_ae
    )
    if non_primary and not args.out_suffix:
        parser.error(
            "non-primary checkpoint/selection runs require --out-suffix"
        )
    outputs = [
        lc.TABLES / f"family_extension_by_seed{args.out_suffix}.csv",
        lc.TABLES / f"family_extension_summary{args.out_suffix}.csv",
        lc.LOGS / f"evaluate_family_extension{args.out_suffix}.log",
    ]
    collisions = [str(path) for path in outputs if path.exists()]
    if collisions:
        parser.error("refusing to overwrite existing output(s): " + ", ".join(collisions))

    t0 = time.time()
    try:
        strict_v2_bundles = validate_strict_v2_lstm_bundles(
            lstm_settings, seeds, args.lstm_model_tag
        )
        strict_v2_evaluator = None
        if strict_v2_bundles:
            input_keys = ["train_windows", "val_windows"]
            if not args.skip_npz:
                input_keys.extend([
                    "test_windows",
                    "variant_test_windows",
                    "variant_sensitivity_windows",
                    "target_id_shift_stress_windows",
                    "out_of_generator_stress_windows",
                    "otids_cross_windows",
                ])
            if not args.skip_cantt:
                input_keys.append("can_train_and_test_archive")
            if not args.skip_road:
                input_keys.extend([
                    "road_archive",
                    "road_dataset_profile",
                ])
            strict_v2_evaluator = evaluator_source_and_input_provenance(
                __file__,
                input_keys,
                include_road_frames=not args.skip_road,
                training_bundles=strict_v2_bundles,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    if strict_v2_bundles:
        try:
            mean, std, strict_v2_inference_standardizer = (
                load_identical_saved_standardizer(
                    standardizer_path(
                        setting, seed, args.lstm_model_tag
                    )
                    for setting in lstm_settings
                    for seed in seeds
                )
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    else:
        mean, std = load_standardizer()
        strict_v2_inference_standardizer = None
    lstm_models = load_lstm_models(
        lstm_settings, seeds, args.lstm_model_tag
    )
    expected_lstm_models = len(lstm_settings) * len(seeds)
    if args.require_complete and len(lstm_models) != expected_lstm_models:
        parser.error(
            "incomplete BiLSTM checkpoint set: expected "
            f"{expected_lstm_models}, found {len(lstm_models)}"
        )
    ae_models = {} if args.skip_ae else load_ae_models()
    print(f"loaded {len(lstm_models)} lstm models, {len(ae_models)} ae models on {DEVICE}", flush=True)
    if not lstm_models and not ae_models:
        parser.error("no requested checkpoints were found")
    ae_thr = ae_thresholds(ae_models, mean, std) if ae_models else {}

    rows = []
    if not args.skip_npz:
        eval_npz_rungs(rows, lstm_models, ae_models, ae_thr, mean, std)
    if not args.skip_cantt:
        eval_cantt(rows, lstm_models, ae_models, ae_thr, mean, std)
    if not args.skip_road:
        eval_road(rows, lstm_models, ae_models, ae_thr, mean, std)

    sfx = args.out_suffix
    lc.write_csv(lc.TABLES / f"family_extension_by_seed{sfx}.csv", rows)
    lc.write_csv(lc.TABLES / f"family_extension_summary{sfx}.csv", summarize(rows))
    log = {
        "lstm_models": len(lstm_models), "ae_models": len(ae_models),
        "lstm_settings": lstm_settings, "seeds": seeds,
        "lstm_model_tag": args.lstm_model_tag,
        "require_complete": args.require_complete,
        "ae_thresholds": ae_thr, "rows": len(rows),
        "elapsed_seconds": time.time() - t0,
        "outputs": [f"results/tables/family_extension_by_seed{sfx}.csv",
                    f"results/tables/family_extension_summary{sfx}.csv"],
    }
    if strict_v2_bundles:
        log["strict_v2_training_bundles"] = strict_v2_bundles
        log["strict_v2_evaluator_provenance"] = strict_v2_evaluator
        log["strict_v2_inference_standardizer"] = (
            strict_v2_inference_standardizer
        )
    lc.LOGS.mkdir(parents=True, exist_ok=True)
    (lc.LOGS / f"evaluate_family_extension{sfx}.log").write_text(json.dumps(log, indent=2))
    print(f"done rows={len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
