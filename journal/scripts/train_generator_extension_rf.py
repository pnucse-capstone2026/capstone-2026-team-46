#!/usr/bin/env python3
"""Track B (amendment #7): Random-Forest arms for the generator extension,
expanding the generator x family matrix beyond the CNN (2x2: {rule,gan} x
{cnn,rf}). Train + evaluate in one pass (RF is cheap; CPU only).

Conventions replicate wisa/scripts/train_rf_augmented.py and
train_real_only_baselines.py @ eb131df exactly:
- rf_features window aggregation (mean/std/min/max/last-minus-first, 55-dim),
  computed on raw (un-standardized) windows.
- RandomForestClassifier(160 trees, unlimited depth, min_samples_leaf=2,
  class_weight=balanced_subsample, random_state=seed).
- Synthetic sampling identical to the CNN arms (sample_synthetic_indices with
  rng seed = seed + int(ratio*1000)), then default_rng(seed) permutation.

Rungs this session: npz-based only (L0 source test, L1 fixed variant,
L2 sensitivity, L3 target-ID, L4 out-of-generator, L5 OTIDS). can-train/ROAD
external RF rows are deferred (frozen WISA tables cover the rule arms there).

Outputs:
  journal/models/generator_extension/rf_{setting}_seed{seed}.joblib
  journal/results/tables/rf_generator_extension_by_seed.csv
  journal/results/tables/rf_generator_extension_summary.csv
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

import lib_cantt as lt
import lib_common as lc
from strict_v2_evaluation import validate_strict_v2_evaluation_scope

MODELS = lc.ROOT / "models" / "generator_extension"
SETTINGS = ["real_only", "rule_0p30", "rule_1p00", "gan_0p30", "gan_1p00"]
VALID_SETTINGS = SETTINGS + ["ganvalid_0p30", "ganvalid_1p00",
                             "ganvalidb_1p00", "ganvalidc_1p00",
                             "arlmvalid_0p30",
                             # E10: matched real duplication + marginal-matched
                             # placebo arms (e10_.../PREREG.md §1 — the RF
                             # harness previously lacked the over_* settings)
                             "over_0p30", "placebo_0p30"]


def rf_features(x):
    """Forked from wisa/scripts/train_real_only_baselines.py @ eb131df."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    minv = x.min(axis=1)
    maxv = x.max(axis=1)
    delta = x[:, -1, :] - x[:, 0, :]
    return np.concatenate([mean, std, minv, maxv, delta], axis=1).astype(np.float32)


def binary_row(y_true_binary, y_pred_binary):
    c = lt.MetricCounter()
    c.update(y_true_binary, y_pred_binary)
    return c.row()


def checkpoint_path(setting, seed, model_tag=""):
    tag = f"_{model_tag}" if model_tag else ""
    return MODELS / f"rf_{setting}{tag}_seed{seed}.joblib"


def training_log_path(setting, seed, model_tag=""):
    tag = f"_{model_tag}" if model_tag else ""
    return lc.LOGS / f"train_rf_{setting}{tag}_seed{seed}.log"


def train_one(setting, seed, model_tag="", sampling_policy="legacy"):
    from sklearn.ensemble import RandomForestClassifier

    model_path = checkpoint_path(setting, seed, model_tag)
    strict_log = training_log_path(setting, seed, model_tag)
    collisions = [model_path]
    if sampling_policy == "strict-v2":
        collisions.append(strict_log)
    existing = [path for path in collisions if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing training artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    t0 = time.time()
    if sampling_policy == "strict-v2":
        train_x, train_y, _val_x, _val_y, sampling_audit = (
            lc.build_train_arrays_v2(setting, seed)
        )
    elif sampling_policy == "legacy":
        train_x, train_y, _val_x, _val_y = lc.build_train_arrays(setting, seed)
        sampling_audit = None
    else:
        raise ValueError(f"unknown sampling policy: {sampling_policy}")
    feats = rf_features(train_x)
    y = train_y.astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(y))
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(feats[order], y[order])
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)
    if sampling_policy == "strict-v2":
        record = {
            "family": "rf",
            "setting": setting,
            "seed": seed,
            "model_tag": model_tag,
            "sampling_policy": sampling_policy,
            "sampling_audit": sampling_audit,
            "train_windows": int(len(y)),
            "elapsed_seconds": time.time() - t0,
        }
        record.update(lc.strict_v2_artifact_record({
            "checkpoint": model_path,
        }))
        lc.write_json_atomic(strict_log, record)
    print(f"rf {setting} seed={seed} trained on {len(y)} windows "
          f"in {time.time()-t0:.0f}s", flush=True)
    return rf, sampling_audit


def eval_rungs(rows, rf, setting, seed, eval_cache):
    for rung, split in [
        ("L0_source_test", "test"),
        ("L1_fixed_variant", "variant_test"),
        ("L2_sensitivity", "variant_sensitivity"),
        ("L3_target_id", "target_id_shift_stress"),
        ("L4_out_of_generator", "out_of_generator_stress"),
        ("L5_otids", "otids_cross"),
    ]:
        feats, y_type, subsets = eval_cache[split]
        pred = rf.predict(feats)
        pred_bin = (pred > 0).astype(np.int8)
        y_bin = (y_type > 0).astype(np.int8)
        for name, mask in subsets.items():
            row = {"family": "rf", "setting": setting, "seed": seed, "rung": rung, "subset": name}
            row.update(binary_row(y_bin[mask], pred_bin[mask]))
            atk_mask = mask & (y_type > 0)
            row["exact_recall"] = float((pred[atk_mask] == y_type[atk_mask]).mean()) if atk_mask.any() else ""
            rows.append(row)


def build_eval_cache():
    cache = {}
    for split in ["test", "variant_test", "variant_sensitivity",
                  "target_id_shift_stress", "out_of_generator_stress", "otids_cross"]:
        data = lc.load_npz(split)
        feats = rf_features(data["x"])
        y_type = data["y_attack_type"]
        subsets = {"all": np.ones(len(y_type), dtype=bool)}
        if split == "variant_sensitivity":
            for sev in ["low", "medium", "high"]:
                subsets[f"severity:{sev}"] = np.asarray(data["severity"]) == sev
        if split in ("target_id_shift_stress", "out_of_generator_stress"):
            attack = np.asarray(data["attack_type"])
            cond = np.asarray(data["id_condition"])
            for atk in ["Gear", "RPM"]:
                for c in ["canonical", "shifted"]:
                    subsets[f"{atk}:{c}"] = (attack == atk) & (cond == c)
        cache[split] = (feats, y_type, subsets)
        print(f"eval cache {split}: {len(y_type)} windows", flush=True)
    return cache


def summarize(rows):
    df = pd.DataFrame(rows)
    num_cols = [c for c in ["fpr", "normal_recall", "attack_recall", "macro_f1_binary", "exact_recall"]
                if c in df.columns]
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
    parser.add_argument("--settings", type=str)
    parser.add_argument("--seeds", type=str)
    parser.add_argument("--out-suffix", default="",
                        help="versioned output suffix; corrected-pool runs must not overwrite legacy tables")
    parser.add_argument("--model-tag", default="",
                        help="version suffix for RF checkpoints (without leading '_')")
    parser.add_argument(
        "--sampling-policy",
        choices=["legacy", "strict-v2"],
        default="legacy",
        help="strict-v2 opts into versioned pools and no-replacement sampling",
    )
    args = parser.parse_args()
    settings = args.settings.split(",") if args.settings else SETTINGS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else lc.SEEDS
    model_tag = args.model_tag or (
        "sampling_v2" if args.sampling_policy == "strict-v2" else ""
    )
    try:
        lc.validate_artifact_tag(model_tag, name="--model-tag")
        lc.validate_artifact_tag(
            args.out_suffix,
            name="--out-suffix",
            leading_underscore=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    unknown = sorted(set(settings) - set(VALID_SETTINGS))
    if unknown:
        raise SystemExit(f"unknown setting(s): {unknown}; choose from {VALID_SETTINGS}")
    if args.out_suffix and not args.out_suffix.startswith("_"):
        parser.error("--out-suffix must start with '_' (for example, _protocol_valid)")
    if lc.is_strict_v2_tag(model_tag) and args.sampling_policy != "strict-v2":
        parser.error(
            "a strict-v2 model tag requires --sampling-policy strict-v2"
        )
    if args.sampling_policy == "strict-v2":
        if "matchedsteps" in model_tag.lower():
            parser.error(
                "RF strict-v2 has no matched-step policy; "
                "--model-tag may not contain 'matchedsteps'"
            )
        unsupported = sorted(
            {setting.split("_")[0] for setting in settings}
            - set(lc.STRICT_V2_POOL_CONFIGS)
        )
        if unsupported:
            parser.error(
                "strict-v2 supports only declared versioned pools; "
                f"unsupported: {unsupported}"
            )
        if "sampling_v2" not in model_tag and "unique_pool_v2" not in model_tag:
            parser.error(
                "strict-v2 --model-tag must include 'sampling_v2' or "
                "'unique_pool_v2'"
            )
        try:
            lc.validate_strict_v2_training_scope(
                family="rf",
                settings=settings,
                seeds=seeds,
                tag=model_tag,
            )
        except ValueError as exc:
            parser.error(str(exc))
        canonical_strict_settings = [
            f"{pool}_1p00" for pool in lc.STRICT_V2_POOL_CONFIGS
        ]
        try:
            validate_strict_v2_evaluation_scope(
                model_tag,
                args.out_suffix,
                seeds,
                reduced_scope=(
                    settings != canonical_strict_settings
                    or seeds != lc.SEEDS
                ),
            )
        except ValueError as exc:
            parser.error(str(exc))
        if model_tag not in args.out_suffix:
            parser.error(
                "strict-v2 --out-suffix must contain the --model-tag"
            )
    if not args.out_suffix and (
            settings != SETTINGS or seeds != lc.SEEDS or model_tag):
        parser.error("non-primary setting/seed runs require --out-suffix to preserve the primary tables")

    sfx = args.out_suffix
    outputs = [
        lc.TABLES / f"rf_generator_extension_by_seed{sfx}.csv",
        lc.TABLES / f"rf_generator_extension_summary{sfx}.csv",
        lc.LOGS / f"train_generator_extension_rf{sfx}.log",
    ]
    collisions = [str(path) for path in outputs if path.exists()]
    if collisions:
        parser.error("refusing to overwrite existing output(s): " + ", ".join(collisions))

    t0 = time.time()
    eval_cache = build_eval_cache()
    rows = []
    sampling_audits = []
    for setting in settings:
        for seed in seeds:
            model_path = checkpoint_path(setting, seed, model_tag)
            if model_path.exists():
                if args.sampling_policy == "strict-v2":
                    existing_record = lc.validate_strict_v2_training_bundle(
                        training_log_path(setting, seed, model_tag),
                        {"checkpoint": model_path},
                        {
                            "family": "rf",
                            "setting": setting,
                            "seed": seed,
                            "model_tag": model_tag,
                            "sampling_policy": "strict-v2",
                        },
                    )
                    sampling_audits.append(
                        existing_record["sampling_audit"]
                    )
                rf = joblib.load(model_path)
                print(f"loaded existing {model_path.name}", flush=True)
            else:
                rf, sampling_audit = train_one(
                    setting, seed, model_tag, args.sampling_policy
                )
                if sampling_audit is not None:
                    sampling_audits.append(sampling_audit)
            eval_rungs(rows, rf, setting, seed, eval_cache)

    lc.write_csv(lc.TABLES / f"rf_generator_extension_by_seed{sfx}.csv", rows)
    lc.write_csv(lc.TABLES / f"rf_generator_extension_summary{sfx}.csv", summarize(rows))
    lc.LOGS.mkdir(parents=True, exist_ok=True)
    (lc.LOGS / f"train_generator_extension_rf{sfx}.log").write_text(json.dumps({
        "settings": settings, "seeds": seeds, "rows": len(rows),
        "model_tag": model_tag, "sampling_policy": args.sampling_policy,
        "sampling_audits": sampling_audits,
        "elapsed_seconds": time.time() - t0}, indent=2))
    print(f"done rows={len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
