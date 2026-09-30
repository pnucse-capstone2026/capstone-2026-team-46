#!/usr/bin/env python3
"""Track B/D: evaluate generator-extension CNN arms on ladder rungs L0-L5.

Forked from journal/scripts/evaluate_family_extension.py (same rung structure,
subset definitions, cantt/road iteration and FPR-first reporting) with the
LSTM/AE families replaced by the 1D-CNN across five training arms:
real_only / rule_0p30 / rule_1p00 / gan_0p30 / gan_1p00 (5 seeds each).

Outputs:
  journal/results/tables/generator_extension_by_seed.csv
  journal/results/tables/generator_extension_summary.csv
  journal/results/logs/evaluate_generator_extension.log
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
    cnn_budget_mode_from_tag,
    evaluator_source_and_input_provenance,
    source_standardizer_provenance,
    validate_strict_v2_evaluation_scope,
)
from train_generator_extension_cnn import (
    CNN1D,
    MODELS,
    SETTINGS,
    VALID_SETTINGS,
    checkpoint_path,
    training_log_path,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_row(y_true_binary, y_pred_binary):
    c = lt.MetricCounter()
    c.update(y_true_binary, y_pred_binary)
    return c.row()


def load_standardizer():
    train = lc.load_npz("train")
    mean, std = lc.fit_standardizer(train["x"])
    return mean, std


def load_cnn_models(settings, model_tag=""):
    models = {}
    tag = f"_{model_tag}" if model_tag else ""
    for setting in settings:
        for seed in lc.SEEDS:
            path = MODELS / f"cnn_{setting}{tag}_seed{seed}.pt"
            if not path.exists():
                print(f"WARN missing {path.name}; skipping", flush=True)
                continue
            model = CNN1D()
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE).eval()
            models[(setting, seed)] = model
    return models


def validate_strict_v2_bundles(settings, model_tag):
    if not lc.is_strict_v2_tag(model_tag):
        return []
    provenance = []
    for setting in settings:
        for seed in lc.SEEDS:
            checkpoint = checkpoint_path(setting, seed, model_tag)
            training_log = training_log_path(setting, seed, model_tag)
            artifacts = {"checkpoint": checkpoint}
            record = lc.validate_strict_v2_training_bundle(
                training_log,
                artifacts,
                {
                    "family": "cnn",
                    "setting": setting,
                    "seed": seed,
                    "model_tag": model_tag,
                    "sampling_policy": "strict-v2",
                    "budget_mode": cnn_budget_mode_from_tag(model_tag),
                },
            )
            provenance.append(bundle_provenance(
                family="cnn",
                setting=setting,
                seed=seed,
                tag=model_tag,
                log_path=training_log,
                artifacts=artifacts,
                validated_record=record,
            ))
    return provenance


def cnn_predict_batches(model, x_std, batch_size=4096):
    """x_std: (N, 128, 11) standardized -> class predictions."""
    preds = np.empty(len(x_std), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(x_std), batch_size):
            batch = torch.from_numpy(
                x_std[i : i + batch_size].transpose(0, 2, 1).copy()
            ).to(DEVICE)
            preds[i : i + len(batch)] = model(batch).argmax(dim=1).cpu().numpy()
    return preds


def eval_npz_rungs(rows, models, mean, std):
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

        for (setting, seed), model in models.items():
            pred = cnn_predict_batches(model, x)
            pred_bin = (pred > 0).astype(np.int8)
            for name, mask in subsets.items():
                row = {"family": "cnn", "setting": setting, "seed": seed, "rung": rung, "subset": name}
                row.update(binary_row(y_bin[mask], pred_bin[mask]))
                atk_mask = mask & (y_type > 0)
                row["exact_recall"] = float((pred[atk_mask] == y_type[atk_mask]).mean()) if atk_mask.any() else ""
                rows.append(row)


def eval_cantt(rows, models, mean, std):
    files = [f for f in lt.list_cantt_files() if f.subset_id.startswith("test_")]
    counters = defaultdict(lt.MetricCounter)
    t0 = time.time()
    with zipfile.ZipFile(lt.ZIP_PATH) as zf:
        for i, meta in enumerate(files, 1):
            df, features, _issues = lt.read_cantt_csv(zf, meta)
            frame_y = df["attack"].to_numpy(np.int8)
            for x_raw, y, _starts in lt.iter_windows(features, frame_y):
                x = lc.standardize(x_raw, mean, std)
                for (setting, seed), model in models.items():
                    pred_bin = (cnn_predict_batches(model, x) > 0).astype(np.int8)
                    for key in [("all", "all"), ("vehicle_axis", meta.vehicle_axis), ("attack_axis", meta.attack_axis)]:
                        counters[(setting, seed, *key)].update(y, pred_bin)
            if i % 20 == 0:
                print(f"cantt {i}/{len(files)} files elapsed={time.time()-t0:.0f}s", flush=True)
    for (setting, seed, level, axis), counter in sorted(counters.items()):
        subset = "all" if level == "all" else f"{level}:{axis}"
        row = {"family": "cnn", "setting": setting, "seed": seed, "rung": "L5_cantt", "subset": subset}
        row.update(counter.row())
        row["exact_recall"] = ""
        rows.append(row)


def eval_road(rows, models, mean, std):
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
            for (setting, seed), model in models.items():
                pred_bin = (cnn_predict_batches(model, x) > 0).astype(np.int8)
                counters[(setting, seed, "all", "all")].update(y, pred_bin)
                counters[(setting, seed, "axis", axis)].update(y, pred_bin)
        if i % 10 == 0:
            print(f"road {i}/{len(captures)} captures elapsed={time.time()-t0:.0f}s", flush=True)
    for (setting, seed, level, axis), counter in sorted(counters.items()):
        subset = "all" if level == "all" else f"axis:{axis}"
        row = {"family": "cnn", "setting": setting, "seed": seed, "rung": "L5_road", "subset": subset}
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
    parser.add_argument("--settings", type=str, help="comma-separated subset of settings")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="suffix for output table names (parallel sharded runs; "
                             "merge shards with merge_generator_extension_tables.py)")
    parser.add_argument("--model-tag", default="",
                        help="checkpoint suffix, e.g. matchedsteps")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail unless all five declared seeds exist for every requested setting")
    args = parser.parse_args()

    t0 = time.time()
    if args.skip_npz and args.skip_cantt and args.skip_road:
        parser.error("at least one evaluation group must remain enabled")
    settings = args.settings.split(",") if args.settings else SETTINGS
    unknown = sorted(set(settings) - set(VALID_SETTINGS))
    if unknown:
        parser.error(f"unknown setting(s): {unknown}; choose from {VALID_SETTINGS}")
    if args.out_suffix and not args.out_suffix.startswith("_"):
        parser.error("--out-suffix must start with '_' (for example, _matchedsteps)")
    try:
        lc.validate_artifact_tag(args.model_tag, name="--model-tag")
        lc.validate_artifact_tag(
            args.out_suffix,
            name="--out-suffix",
            leading_underscore=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_strict_v2_evaluation_scope(
            args.model_tag,
            args.out_suffix,
            lc.SEEDS,
            reduced_scope=(
                settings != SETTINGS
                or args.skip_npz
                or args.skip_cantt
                or args.skip_road
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if lc.is_strict_v2_tag(args.model_tag):
        if not args.require_complete:
            parser.error("strict-v2 model tags require --require-complete")
        if args.model_tag not in args.out_suffix:
            parser.error(
                "strict-v2 --out-suffix must contain the --model-tag"
            )
    if not args.out_suffix and (settings != SETTINGS or args.model_tag):
        parser.error("non-primary setting/model-tag runs require --out-suffix to preserve the primary tables")
    sfx = args.out_suffix
    outputs = [
        lc.TABLES / f"generator_extension_by_seed{sfx}.csv",
        lc.TABLES / f"generator_extension_summary{sfx}.csv",
        lc.LOGS / f"evaluate_generator_extension{sfx}.log",
    ]
    collisions = [str(path) for path in outputs if path.exists()]
    if collisions:
        parser.error("refusing to overwrite existing output(s): " + ", ".join(collisions))
    try:
        strict_v2_bundles = validate_strict_v2_bundles(
            settings, args.model_tag
        )
        strict_v2_evaluator = None
        if strict_v2_bundles:
            input_keys = ["train_windows"]
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
    strict_v2_standardizer = (
        source_standardizer_provenance() if strict_v2_bundles else None
    )
    mean, std = load_standardizer()
    models = load_cnn_models(settings, args.model_tag)
    if not models:
        parser.error("no requested checkpoints were found")
    expected_models = len(settings) * len(lc.SEEDS)
    if args.require_complete and len(models) != expected_models:
        parser.error(f"incomplete checkpoint set: expected {expected_models}, found {len(models)}")
    print(f"loaded {len(models)} cnn models on {DEVICE}", flush=True)

    rows = []
    if not args.skip_npz:
        eval_npz_rungs(rows, models, mean, std)
    if not args.skip_cantt:
        eval_cantt(rows, models, mean, std)
    if not args.skip_road:
        eval_road(rows, models, mean, std)

    lc.write_csv(lc.TABLES / f"generator_extension_by_seed{sfx}.csv", rows)
    lc.write_csv(lc.TABLES / f"generator_extension_summary{sfx}.csv", summarize(rows))
    log = {
        "cnn_models": len(models), "settings": settings, "model_tag": args.model_tag,
        "skip_npz": args.skip_npz, "skip_cantt": args.skip_cantt,
        "skip_road": args.skip_road, "out_suffix": args.out_suffix,
        "require_complete": args.require_complete,
        "rows": len(rows),
        "elapsed_seconds": time.time() - t0,
        "outputs": [f"results/tables/generator_extension_by_seed{sfx}.csv",
                    f"results/tables/generator_extension_summary{sfx}.csv"],
    }
    if strict_v2_bundles:
        log["strict_v2_training_bundles"] = strict_v2_bundles
        log["standardizer_provenance"] = strict_v2_standardizer
        log["strict_v2_evaluator_provenance"] = strict_v2_evaluator
    lc.LOGS.mkdir(parents=True, exist_ok=True)
    (lc.LOGS / f"evaluate_generator_extension{sfx}.log").write_text(json.dumps(log, indent=2))
    print(f"done rows={len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
