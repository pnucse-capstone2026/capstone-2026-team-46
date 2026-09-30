#!/usr/bin/env python3
"""Source-calibrated, file/capture-aware neural external diagnostic.

This is an inference-only follow-up to the pooled L5 tables.  It evaluates the
existing CNN, BiLSTM, and Transformer checkpoints at thresholds selected only
from Car-Hacking validation-normal windows.  External data are never used for
fitting, threshold selection, or model selection.

The focused default comparison is ``real_only`` versus ``rule_0p30`` over all
five detector seeds.  Two source-validation operating points are retained
(``FPR=0.001`` and ``FPR=0.01``), and FPR and binary attack recall are always
reported together.  Aggregation preserves the physical external unit:

* OTIDS: source capture/file (four source files), and
* can-train-and-test: each ``test_*`` CSV (176 files).

The per-seed table contains both pooled-window and unit-macro metrics.  Unit
bootstrap intervals are descriptive uncertainty across captures/files; they
do not make overlapping windows independent.  The paired table reports the
within-detector-seed ``rule_0p30 - real_only`` change.  This analysis is
diagnostic only and cannot establish operational transfer.

No output is overwritten.  Defaults (tag ``neural_primary_v1``):

  results/tables/external_operating_neural_by_unit_neural_primary_v1.csv
  results/tables/external_operating_neural_by_seed_neural_primary_v1.csv
  results/tables/external_operating_neural_summary_neural_primary_v1.csv
  results/tables/external_operating_neural_paired_neural_primary_v1.csv
  results/logs/external_operating_neural_neural_primary_v1.log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

import lib_cantt as lt
import lib_common as lc
from strict_v2_evaluation import (
    bundle_provenance,
    cnn_budget_mode_from_tag,
    evaluation_model_input_provenance,
    evaluator_source_and_input_provenance,
    load_identical_saved_standardizer,
    source_standardizer_provenance,
    transformer_expected_extras,
    validate_strict_v2_evaluation_scope,
)
from train_family_extension_lstm import (
    LSTMClassifier,
    log_path as lstm_training_log_path,
)
from train_family_extension_transformer import (
    TransformerClassifier,
    training_artifact_paths as transformer_training_artifact_paths,
)
from train_generator_extension_cnn import (
    CNN1D,
    MODELS as CNN_MODELS,
    training_log_path as cnn_training_log_path,
)


DEFAULT_FAMILIES = ["cnn", "lstm", "transformer"]
DEFAULT_SETTINGS = ["real_only", "rule_0p30"]
VALID_SETTINGS = ["real_only", "rule_0p30", "rule_1p00"]
DEFAULT_DATASETS = ["otids", "cantt"]
VAL_FPRS = [0.001, 0.01]
DEFAULT_TAG = "neural_primary_v1"
SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
STRICT_V2_MARKERS = ("sampling_v2", "unique_pool_v2", "strict_v2", "strict-v2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAMILY_BATCH = {"cnn": 4096, "lstm": 1024, "transformer": 1024}


def parse_csv_list(value: str, cast=str) -> list:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ValueError("comma-separated arguments must contain no empty item")
    out = [cast(item) for item in items]
    if len(out) != len(set(out)):
        raise ValueError("comma-separated arguments must not contain duplicates")
    return out


def output_paths(tag: str) -> tuple[Path, Path, Path, Path, Path]:
    stem = f"external_operating_neural_{{kind}}_{tag}.csv"
    return (
        lc.TABLES / stem.format(kind="by_unit"),
        lc.TABLES / stem.format(kind="by_seed"),
        lc.TABLES / stem.format(kind="summary"),
        lc.TABLES / stem.format(kind="paired"),
        lc.LOGS / f"external_operating_neural_{tag}.log",
    )


def resolve_tagged_settings(
    settings: list[str], model_tag: str, tagged_settings_value: str | None
) -> set[str]:
    """Return the requested settings whose checkpoints carry ``model_tag``."""
    if not model_tag:
        if tagged_settings_value:
            raise ValueError("--tagged-settings requires --model-tag")
        return set()
    if not SAFE_TAG.fullmatch(model_tag):
        raise ValueError(
            "--model-tag must be a safe filename tag without path separators"
        )
    tagged = (
        set(parse_csv_list(tagged_settings_value))
        if tagged_settings_value
        else set(settings)
    )
    unknown = sorted(tagged - set(settings))
    if unknown:
        raise ValueError(
            f"--tagged-settings contains unrequested setting(s): {unknown}"
        )
    return tagged


def validate_model_output_tag(
    model_tag: str,
    tagged_settings: set[str],
    out_tag: str,
    allow_missing: bool = False,
) -> None:
    strict_label = any(marker in out_tag for marker in STRICT_V2_MARKERS)
    if not model_tag:
        if strict_label:
            raise ValueError(
                "strict-v2-labelled output requires an explicit --model-tag"
            )
        return
    if strict_label and not lc.is_strict_v2_tag(model_tag):
        raise ValueError(
            "strict-v2-labelled output requires a strict-v2 --model-tag"
        )
    if not tagged_settings:
        raise ValueError("--model-tag must select at least one tagged setting")
    if allow_missing:
        raise ValueError(
            "versioned checkpoint evaluation forbids --allow-missing"
        )
    if out_tag == DEFAULT_TAG or model_tag not in out_tag:
        raise ValueError(
            "versioned checkpoints require an explicit --out-tag containing "
            "the --model-tag"
        )


def effective_model_tag(
    setting: str,
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> str:
    if not model_tag:
        return ""
    if tagged_settings is not None and setting not in tagged_settings:
        return ""
    return model_tag


def checkpoint_path(
    family: str,
    setting: str,
    seed: int,
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> Path:
    tag = effective_model_tag(setting, model_tag, tagged_settings)
    if family == "cnn":
        suffix = f"_{tag}" if tag else ""
        return CNN_MODELS / f"cnn_{setting}{suffix}_seed{seed}.pt"
    if family == "lstm":
        suffix = f"_{tag}" if tag else ""
        return lc.MODELS / f"lstm_{setting}{suffix}_seed{seed}.pt"
    if family == "transformer":
        suffix = f"_{tag}" if tag else ""
        return lc.MODELS / f"transformer_{setting}_seed{seed}{suffix}.pt"
    raise ValueError(f"unknown family: {family}")


def standardizer_path(
    family: str,
    setting: str,
    seed: int,
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> Path | None:
    tag = effective_model_tag(setting, model_tag, tagged_settings)
    suffix = f"_{tag}" if tag else ""
    if family == "cnn":
        return None
    if family == "lstm":
        return lc.MODELS / f"lstm_standardizer_{setting}{suffix}_seed{seed}.npz"
    if family == "transformer":
        return (
            lc.MODELS
            / f"transformer_standardizer_{setting}_seed{seed}{suffix}.npz"
        )
    raise ValueError(f"unknown family: {family}")


def validate_strict_v2_neural_bundles(
    families: list[str],
    settings: list[str],
    seeds: list[int],
    model_tag: str,
    tagged_settings: set[str],
) -> list[dict]:
    if not lc.is_strict_v2_tag(model_tag):
        return []
    provenance = []
    for family in families:
        for setting in settings:
            if setting not in tagged_settings:
                continue
            for seed in seeds:
                checkpoint = checkpoint_path(
                    family, setting, seed, model_tag, tagged_settings
                )
                artifacts = {"checkpoint": checkpoint}
                if family == "cnn":
                    training_log = cnn_training_log_path(
                        setting, seed, model_tag
                    )
                    expected = {
                        "family": "cnn",
                        "setting": setting,
                        "seed": seed,
                        "model_tag": model_tag,
                        "sampling_policy": "strict-v2",
                        "budget_mode": cnn_budget_mode_from_tag(model_tag),
                    }
                elif family == "lstm":
                    standardizer = standardizer_path(
                        family,
                        setting,
                        seed,
                        model_tag,
                        tagged_settings,
                    )
                    assert standardizer is not None
                    artifacts["standardizer"] = standardizer
                    training_log = lstm_training_log_path(
                        setting, seed, model_tag
                    )
                    expected = {
                        "family": "lstm",
                        "setting": setting,
                        "seed": seed,
                        "model_tag": model_tag,
                        "sampling_policy": "strict-v2",
                    }
                elif family == "transformer":
                    trainer_tag = f"_{model_tag}"
                    _checkpoint, standardizer, training_log = (
                        transformer_training_artifact_paths(
                            setting, seed, trainer_tag
                        )
                    )
                    if _checkpoint != checkpoint:
                        raise RuntimeError(
                            "transformer evaluator/trainer checkpoint path "
                            f"mismatch: {checkpoint} vs {_checkpoint}"
                        )
                    artifacts["standardizer"] = standardizer
                    expected = {
                        "family": "transformer",
                        "setting": setting,
                        "seed": seed,
                        "tag": trainer_tag,
                        "sampling_policy": "strict-v2",
                    }
                    expected.update(transformer_expected_extras(trainer_tag))
                else:
                    raise ValueError(f"unknown family: {family}")
                record = lc.validate_strict_v2_training_bundle(
                    training_log,
                    artifacts,
                    expected,
                )
                provenance.append(bundle_provenance(
                    family=family,
                    setting=setting,
                    seed=seed,
                    tag=model_tag,
                    log_path=training_log,
                    artifacts=artifacts,
                    validated_record=record,
                ))
    return provenance


def new_model(family: str) -> torch.nn.Module:
    if family == "cnn":
        return CNN1D()
    if family == "lstm":
        return LSTMClassifier()
    if family == "transformer":
        return TransformerClassifier()
    raise ValueError(f"unknown family: {family}")


def load_models(
    families: list[str],
    settings: list[str],
    seeds: list[int],
    *,
    allow_missing: bool,
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> dict[tuple[str, str, int], torch.nn.Module]:
    models = {}
    missing = []
    for family in families:
        for setting in settings:
            for seed in seeds:
                path = checkpoint_path(
                    family, setting, seed, model_tag, tagged_settings
                )
                if not path.exists():
                    missing.append(path)
                    continue
                model = new_model(family)
                model.load_state_dict(
                    torch.load(path, map_location=DEVICE, weights_only=True)
                )
                model.to(DEVICE).eval()
                models[(family, setting, seed)] = model
    if missing and not allow_missing:
        names = ", ".join(str(path) for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"missing {len(missing)} checkpoint(s): {names}{suffix}")
    for path in missing:
        print(f"WARN missing {path}; skipping", flush=True)
    return models


def verify_saved_standardizers(
    models: dict,
    mean: np.ndarray,
    std: np.ndarray,
    model_tag: str = "",
    tagged_settings: set[str] | None = None,
) -> dict:
    """Verify recurrent/attention checkpoints use the source-train scaler.

    CNN training also fits this scaler but does not persist a per-arm copy.
    """
    checked = 0
    for family, setting, seed in models:
        if family == "cnn":
            continue
        path = standardizer_path(
            family, setting, seed, model_tag, tagged_settings
        )
        assert path is not None
        if not path.exists():
            raise FileNotFoundError(path)
        saved = np.load(path)
        if not (np.allclose(saved["mean"], mean) and np.allclose(saved["std"], std)):
            raise RuntimeError(f"saved standardizer differs from source train: {path}")
        checked += 1
    return {
        "derived_from": "Car-Hacking train windows only",
        "saved_recurrent_attention_standardizers_checked": checked,
        "cnn_training_convention": "same source-train scaler; no per-arm scaler file",
    }


def score_models(models: dict, x_std: np.ndarray) -> dict[tuple[str, str, int], np.ndarray]:
    """Score one raw-window batch, sharing each family tensor transfer."""
    output = {}
    by_family = defaultdict(list)
    for key, model in models.items():
        by_family[key[0]].append((key, model))
    with torch.inference_mode():
        for family, family_models in sorted(by_family.items()):
            batch_size = FAMILY_BATCH[family]
            family_scores = {
                key: np.empty(len(x_std), dtype=np.float32)
                for key, _model in family_models
            }
            for start in range(0, len(x_std), batch_size):
                stop = min(start + batch_size, len(x_std))
                xb = torch.from_numpy(x_std[start:stop]).to(DEVICE)
                if family == "cnn":
                    xb = xb.transpose(1, 2).contiguous()
                for key, model in family_models:
                    probs = torch.softmax(model(xb), dim=1)
                    family_scores[key][start:stop] = (
                        1.0 - probs[:, 0]
                    ).cpu().numpy()
            output.update(family_scores)
    return output


def source_threshold(score: np.ndarray, target_fpr: float) -> tuple[float, float]:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("source-normal scores must be non-empty and finite")
    threshold = float(np.quantile(values, 1.0 - target_fpr))
    # Strict > mirrors the existing operating-point diagnostics.  The achieved
    # rate is retained because score ties can make it lower than the target.
    achieved = float((values > threshold).mean())
    return threshold, achieved


def calibrate_thresholds(models: dict, mean: np.ndarray, std: np.ndarray) -> dict:
    val = lc.load_npz("val")
    normal = val["x"][val["y_attack_type"] == 0]
    score_chunks = {key: [] for key in models}
    for start in range(0, len(normal), 4096):
        x_std = lc.standardize(normal[start:start + 4096], mean, std)
        batch_scores = score_models(models, x_std)
        for key, scores in batch_scores.items():
            score_chunks[key].append(scores)
    thresholds = {}
    for key, chunks in sorted(score_chunks.items()):
        scores = np.concatenate(chunks)
        thresholds[key] = {}
        for target in VAL_FPRS:
            threshold, achieved = source_threshold(scores, target)
            thresholds[key][target] = {
                "threshold": threshold,
                "val_fpr_achieved": achieved,
            }
        print(
            f"calibrated {key[0]} {key[1]} seed={key[2]} "
            f"achieved@0.001={thresholds[key][0.001]['val_fpr_achieved']:.6f}",
            flush=True,
        )
    return thresholds


class Count:
    def __init__(self) -> None:
        self.tn = self.fp = self.fn = self.tp = 0

    def update(self, y_true: np.ndarray, alarm: np.ndarray) -> None:
        y = np.asarray(y_true, dtype=bool)
        pred = np.asarray(alarm, dtype=bool)
        if y.shape != pred.shape:
            raise ValueError(f"label/prediction shape mismatch: {y.shape} vs {pred.shape}")
        self.tn += int((~y & ~pred).sum())
        self.fp += int((~y & pred).sum())
        self.fn += int((y & ~pred).sum())
        self.tp += int((y & pred).sum())

    def row(self) -> dict:
        normal = self.tn + self.fp
        attack = self.fn + self.tp
        return {
            "windows": normal + attack,
            "normal_windows": normal,
            "attack_windows": attack,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "fpr": self.fp / normal if normal else np.nan,
            "normal_recall": self.tn / normal if normal else np.nan,
            "attack_recall": self.tp / attack if attack else np.nan,
        }


def evaluate_batches(
    batches, models: dict, thresholds: dict
) -> dict[tuple[str, str, int, float], Count]:
    counters = {
        (*key, target): Count()
        for key in models
        for target in VAL_FPRS
    }
    for x_raw, y in batches:
        x_std = lc.standardize(x_raw, evaluate_batches.mean, evaluate_batches.std)
        scores = score_models(models, x_std)
        for key, values in scores.items():
            for target in VAL_FPRS:
                threshold = thresholds[key][target]["threshold"]
                counters[(*key, target)].update(y, values > threshold)
    return counters


# Assigned by main after fitting the source-only scaler.  Keeping the public
# evaluate_batches signature small also makes unit-stream adapters identical.
evaluate_batches.mean = None
evaluate_batches.std = None


def rows_for_unit(
    dataset: str,
    unit_type: str,
    unit_id: str,
    metadata: dict,
    counters: dict,
    thresholds: dict,
) -> list[dict]:
    rows = []
    for (family, setting, seed, target), counter in sorted(counters.items()):
        calibration = thresholds[(family, setting, seed)][target]
        rows.append({
            "dataset": dataset,
            "unit_type": unit_type,
            "unit_id": unit_id,
            **metadata,
            "family": family,
            "setting": setting,
            "seed": seed,
            "policy": f"source_val_fpr_{target:g}",
            "val_fpr_target": target,
            "threshold": calibration["threshold"],
            "val_fpr_achieved": calibration["val_fpr_achieved"],
            **counter.row(),
        })
    return rows


def evaluate_otids(models: dict, thresholds: dict) -> list[dict]:
    data = lc.load_npz("otids_cross")
    source = np.asarray([str(value) for value in data["source_file"]])
    rows = []
    started = time.time()
    for index, unit_id in enumerate(sorted(np.unique(source)), 1):
        idx = np.flatnonzero(source == unit_id)

        def batches():
            for start in range(0, len(idx), 4096):
                selected = idx[start:start + 4096]
                yield data["x"][selected], data["y_binary"][selected].astype(np.int8)

        counters = evaluate_batches(batches(), models, thresholds)
        y = data["y_binary"][idx]
        normal, attack = int((y == 0).sum()), int((y > 0).sum())
        role = "mixed" if normal and attack else ("normal_only" if normal else "attack_only")
        rows.extend(rows_for_unit(
            "OTIDS", "source_file", unit_id,
            {
                "set_id": "",
                "subset_id": role,
                "vehicle": "OTIDS vehicle",
                "vehicle_axis": "external_vehicle",
                "attack_axis": role,
                "source_stem": Path(unit_id).stem,
            },
            counters, thresholds,
        ))
        print(
            f"OTIDS {index}/4 {unit_id} windows={len(idx)} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
    return rows


def evaluate_cantt(models: dict, thresholds: dict) -> list[dict]:
    files = [meta for meta in lt.list_cantt_files() if meta.subset_id.startswith("test_")]
    if not files:
        raise FileNotFoundError(f"no can-train test files in {lt.ZIP_PATH}")
    rows = []
    started = time.time()
    with zipfile.ZipFile(lt.ZIP_PATH) as archive:
        for index, meta in enumerate(files, 1):
            df, raw, _issues = lt.read_cantt_csv(archive, meta)
            frame_y = df["attack"].to_numpy(dtype=np.int8)

            def batches():
                for x_raw, y, _starts in lt.iter_windows(raw, frame_y):
                    yield x_raw, y

            counters = evaluate_batches(batches(), models, thresholds)
            rows.extend(rows_for_unit(
                "can-train-and-test", "test_file", meta.path,
                {
                    "set_id": meta.set_id,
                    "subset_id": meta.subset_id,
                    "vehicle": meta.vehicle,
                    "vehicle_axis": meta.vehicle_axis,
                    "attack_axis": meta.attack_axis,
                    "source_stem": meta.source_stem,
                },
                counters, thresholds,
            ))
            if index % 10 == 0 or index == len(files):
                print(
                    f"can-train {index}/{len(files)} files "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    return rows


def bootstrap_mean_ci(values, *, samples: int, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    if len(x) == 1 or samples <= 0:
        mean = float(x.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    # Chunk draws to avoid a large samples-by-units allocation.
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        draw = rng.integers(0, len(x), size=(stop - start, len(x)))
        boot[start:stop] = x[draw].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def aggregate_unit_group(group: pd.DataFrame, *, bootstrap_samples: int) -> dict:
    normal = int(group["normal_windows"].sum())
    attack = int(group["attack_windows"].sum())
    fp, tp = int(group["fp"].sum()), int(group["tp"].sum())
    fpr = pd.to_numeric(
        group.loc[group["normal_windows"] > 0, "fpr"], errors="coerce"
    ).dropna().to_numpy()
    recall = pd.to_numeric(
        group.loc[group["attack_windows"] > 0, "attack_recall"], errors="coerce"
    ).dropna().to_numpy()
    key_text = "|".join(str(group[column].iloc[0]) for column in [
        "dataset", "family", "setting", "seed", "policy"
    ])
    base_seed = zlib.crc32(key_text.encode("utf-8"))
    fpr_low, fpr_high = bootstrap_mean_ci(
        fpr, samples=bootstrap_samples, seed=base_seed
    )
    recall_low, recall_high = bootstrap_mean_ci(
        recall, samples=bootstrap_samples, seed=base_seed ^ 0xA5A5A5A5
    )
    return {
        "dataset": group["dataset"].iloc[0],
        "unit_type": group["unit_type"].iloc[0],
        "family": group["family"].iloc[0],
        "setting": group["setting"].iloc[0],
        "seed": int(group["seed"].iloc[0]),
        "policy": group["policy"].iloc[0],
        "val_fpr_target": float(group["val_fpr_target"].iloc[0]),
        "threshold": float(group["threshold"].iloc[0]),
        "val_fpr_achieved": float(group["val_fpr_achieved"].iloc[0]),
        "units": int(len(group)),
        "units_with_normal_windows": int(len(fpr)),
        "units_with_attack_windows": int(len(recall)),
        "windows": int(group["windows"].sum()),
        "normal_windows": normal,
        "attack_windows": attack,
        "pooled_fpr": fp / normal if normal else np.nan,
        "pooled_attack_recall": tp / attack if attack else np.nan,
        "unit_macro_fpr": float(fpr.mean()) if len(fpr) else np.nan,
        "unit_macro_fpr_sd": float(fpr.std(ddof=1)) if len(fpr) > 1 else (
            0.0 if len(fpr) else np.nan
        ),
        "unit_macro_fpr_ci_low": fpr_low,
        "unit_macro_fpr_ci_high": fpr_high,
        "unit_fpr_min": float(fpr.min()) if len(fpr) else np.nan,
        "unit_fpr_max": float(fpr.max()) if len(fpr) else np.nan,
        "unit_fraction_fpr_ge_0p99": float((fpr >= 0.99).mean()) if len(fpr) else np.nan,
        "unit_macro_attack_recall": float(recall.mean()) if len(recall) else np.nan,
        "unit_macro_attack_recall_sd": (
            float(recall.std(ddof=1)) if len(recall) > 1 else
            (0.0 if len(recall) else np.nan)
        ),
        "unit_macro_attack_recall_ci_low": recall_low,
        "unit_macro_attack_recall_ci_high": recall_high,
        "unit_attack_recall_min": float(recall.min()) if len(recall) else np.nan,
        "unit_attack_recall_max": float(recall.max()) if len(recall) else np.nan,
    }


def build_by_seed(unit_rows: list[dict], *, bootstrap_samples: int) -> list[dict]:
    frame = pd.DataFrame(unit_rows)
    keys = ["dataset", "family", "setting", "seed", "policy"]
    return [
        aggregate_unit_group(group, bootstrap_samples=bootstrap_samples)
        for _values, group in frame.groupby(keys, sort=True)
    ]


SEED_SUMMARY_METRICS = [
    "threshold",
    "val_fpr_achieved",
    "pooled_fpr",
    "pooled_attack_recall",
    "unit_macro_fpr",
    "unit_macro_attack_recall",
    "unit_fraction_fpr_ge_0p99",
]


def summarize_seeds(by_seed_rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(by_seed_rows)
    keys = ["dataset", "unit_type", "family", "setting", "policy", "val_fpr_target"]
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ";".join(str(seed) for seed in sorted(group["seed"].unique()))
        for column in [
            "units", "units_with_normal_windows", "units_with_attack_windows",
            "windows", "normal_windows", "attack_windows",
        ]:
            distinct = group[column].unique()
            if len(distinct) != 1:
                raise RuntimeError(f"{column} differs across detector seeds for {values}")
            row[column] = int(distinct[0])
        for metric in SEED_SUMMARY_METRICS:
            x = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(x.mean()) if len(x) else np.nan
            row[f"{metric}_std"] = (
                float(x.std(ddof=1)) if len(x) > 1 else
                (0.0 if len(x) else np.nan)
            )
        rows.append(row)
    return rows


def mean_t_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    if len(x) > 1 and sd > 0:
        half = float(student_t.ppf(0.975, len(x) - 1) * sd / math.sqrt(len(x)))
        return mean, sd, mean - half, mean + half
    return mean, sd, mean, mean


def paired_setting_summary(
    by_seed_rows: list[dict], *, augmented: str = "rule_0p30", baseline: str = "real_only"
) -> list[dict]:
    frame = pd.DataFrame(by_seed_rows)
    if augmented not in set(frame["setting"]) or baseline not in set(frame["setting"]):
        return []
    index = ["dataset", "unit_type", "family", "policy", "val_fpr_target", "seed"]
    metrics = [
        "pooled_fpr", "pooled_attack_recall",
        "unit_macro_fpr", "unit_macro_attack_recall",
    ]
    pivot = frame.pivot(index=index, columns="setting", values=metrics)
    rows = []
    group_levels = index[:-1]
    for values, group in pivot.groupby(level=group_levels, sort=True):
        row = dict(zip(group_levels, values))
        seeds = group.index.get_level_values("seed").to_numpy()
        row.update({
            "comparison": f"{augmented}-{baseline}",
            "n_paired_seeds": int(len(seeds)),
            "seeds": ";".join(str(seed) for seed in sorted(seeds)),
        })
        for metric in metrics:
            delta = (
                group[(metric, augmented)].to_numpy(dtype=np.float64)
                - group[(metric, baseline)].to_numpy(dtype=np.float64)
            )
            if not np.isfinite(delta).all():
                raise RuntimeError(f"non-finite paired delta for {values} {metric}")
            mean, sd, low, high = mean_t_ci(delta)
            row[f"delta_{metric}_mean"] = mean
            row[f"delta_{metric}_std"] = sd
            row[f"delta_{metric}_ci_low"] = low
            row[f"delta_{metric}_ci_high"] = high
            row[f"delta_{metric}_positive_seeds"] = int((delta > 0).sum())
            row[f"delta_{metric}_negative_seeds"] = int((delta < 0).sum())
        rows.append(row)
    return rows


def validate_results(unit_rows: list[dict], models: dict, datasets: list[str]) -> None:
    if not unit_rows:
        raise RuntimeError("external operating-point evaluation produced no rows")
    frame = pd.DataFrame(unit_rows)
    keys = [
        "dataset", "unit_type", "unit_id", "family", "setting", "seed", "policy"
    ]
    if frame.duplicated(keys, keep=False).any():
        raise RuntimeError("duplicate unit/model/policy rows")
    expected = {
        (*key, f"source_val_fpr_{target:g}")
        for key in models
        for target in VAL_FPRS
    }
    dataset_names = {"otids": "OTIDS", "cantt": "can-train-and-test"}
    for short_name in datasets:
        name = dataset_names[short_name]
        subset = frame[frame["dataset"] == name]
        actual = set(zip(
            subset["family"], subset["setting"], subset["seed"].astype(int), subset["policy"]
        ))
        if actual != expected:
            raise RuntimeError(
                f"incomplete {name} model/policy matrix: missing={sorted(expected - actual)}"
            )


def stage_and_publish(
    paths: tuple[Path, ...], tables: tuple[list[dict], ...], log: dict
) -> None:
    collisions = [path for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in collisions)
        )
    results_root = lc.ROOT / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".external-neural-publish-", dir=results_root))
    published = []
    try:
        staged = [staging / path.name for path in paths]
        for path, rows in zip(staged[:-1], tables):
            lc.write_csv(path, rows)
            if len(pd.read_csv(path)) != len(rows):
                raise RuntimeError(f"staged row-count mismatch for {path.name}")
        staged[-1].write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
        for source, destination in zip(staged, paths):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            published.append(destination)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--settings", default=",".join(DEFAULT_SETTINGS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in lc.SEEDS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out-tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--model-tag",
        default="",
        help=(
            "checkpoint suffix without leading '_' (for example sampling_v2)"
        ),
    )
    parser.add_argument(
        "--tagged-settings",
        default=None,
        help=(
            "comma-separated settings that use --model-tag; defaults to every "
            "requested setting"
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="legacy exploratory runs only; forbidden with --model-tag",
    )
    args = parser.parse_args()

    try:
        lc.validate_artifact_tag(args.model_tag, name="--model-tag")
        families = parse_csv_list(args.families)
        settings = parse_csv_list(args.settings)
        seeds = parse_csv_list(args.seeds, int)
        datasets = parse_csv_list(args.datasets)
        tagged_settings = resolve_tagged_settings(
            settings, args.model_tag, args.tagged_settings
        )
        validate_model_output_tag(
            args.model_tag,
            tagged_settings,
            args.out_tag,
            args.allow_missing,
        )
        validate_strict_v2_evaluation_scope(
            args.model_tag,
            args.out_tag,
            seeds,
            reduced_scope=(
                families != DEFAULT_FAMILIES
                or settings != DEFAULT_SETTINGS
                or seeds != lc.SEEDS
                or datasets != DEFAULT_DATASETS
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    for values, allowed, label in [
        (families, set(DEFAULT_FAMILIES), "families"),
        (settings, set(VALID_SETTINGS), "settings"),
        (datasets, set(DEFAULT_DATASETS), "datasets"),
    ]:
        unknown = sorted(set(values) - allowed)
        if unknown:
            parser.error(f"unknown {label}: {unknown}; choose from {sorted(allowed)}")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    if not SAFE_TAG.fullmatch(args.out_tag):
        parser.error("--out-tag must be a safe filename tag without path separators")
    paths = output_paths(args.out_tag)
    collisions = [path for path in paths if path.exists()]
    if collisions:
        parser.error(
            "refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in collisions)
        )
    try:
        strict_v2_bundles = validate_strict_v2_neural_bundles(
            families,
            settings,
            seeds,
            args.model_tag,
            tagged_settings,
        )
        strict_v2_evaluator = None
        strict_v2_model_inputs = None
        if strict_v2_bundles:
            input_keys = ["train_windows", "val_windows"]
            if "otids" in datasets:
                input_keys.append("otids_cross_windows")
            if "cantt" in datasets:
                input_keys.append("can_train_and_test_archive")
            strict_v2_evaluator = evaluator_source_and_input_provenance(
                __file__,
                input_keys,
                training_bundles=strict_v2_bundles,
            )
            model_artifacts = {}
            for family in families:
                for setting in settings:
                    for seed in seeds:
                        prefix = f"{family}/{setting}/seed{seed}"
                        model_artifacts[f"{prefix}/checkpoint"] = (
                            checkpoint_path(
                                family,
                                setting,
                                seed,
                                args.model_tag,
                                tagged_settings,
                            )
                        )
                        scaler = standardizer_path(
                            family,
                            setting,
                            seed,
                            args.model_tag,
                            tagged_settings,
                        )
                        if scaler is not None:
                            model_artifacts[f"{prefix}/standardizer"] = scaler
            strict_v2_model_inputs = evaluation_model_input_provenance(
                model_artifacts
            )
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    strict_v2_standardizer = (
        source_standardizer_provenance() if strict_v2_bundles else None
    )

    started = time.time()
    models = load_models(
        families,
        settings,
        seeds,
        allow_missing=args.allow_missing,
        model_tag=args.model_tag,
        tagged_settings=tagged_settings,
    )
    if not models:
        parser.error("no requested checkpoints were found")
    print(f"loaded {len(models)} checkpoints on {DEVICE}", flush=True)

    strict_v2_inference_standardizer = None
    saved_scaler_paths = [
        path
        for family, setting, seed in models
        if (
            path := standardizer_path(
                family,
                setting,
                seed,
                args.model_tag,
                tagged_settings,
            )
        ) is not None
    ]
    if strict_v2_bundles and saved_scaler_paths:
        try:
            mean, std, strict_v2_inference_standardizer = (
                load_identical_saved_standardizer(saved_scaler_paths)
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    else:
        train = lc.load_npz("train")
        mean, std = lc.fit_standardizer(train["x"])
        del train
    standardizer_audit = verify_saved_standardizers(
        models,
        mean,
        std,
        args.model_tag,
        tagged_settings,
    )
    evaluate_batches.mean, evaluate_batches.std = mean, std
    thresholds = calibrate_thresholds(models, mean, std)

    unit_rows = []
    if "otids" in datasets:
        unit_rows.extend(evaluate_otids(models, thresholds))
    if "cantt" in datasets:
        unit_rows.extend(evaluate_cantt(models, thresholds))
    validate_results(unit_rows, models, datasets)
    by_seed_rows = build_by_seed(unit_rows, bootstrap_samples=args.bootstrap_samples)
    summary_rows = summarize_seeds(by_seed_rows)
    paired_rows = paired_setting_summary(by_seed_rows)
    elapsed = time.time() - started
    log = {
        "analysis_role": "diagnostic_only",
        "external_data_role": "evaluation_only",
        "claim_scope": "does not establish operational transfer",
        "score": "1 - P(Normal)",
        "threshold_selection": "Car-Hacking validation-normal only; strict score > threshold",
        "unit_uncertainty": (
            "descriptive percentile bootstrap across physical files/captures; "
            "overlapping windows are not treated as independent"
        ),
        "families": families,
        "settings": settings,
        "seeds": seeds,
        "datasets": datasets,
        "val_fpr_targets": VAL_FPRS,
        "bootstrap_samples": args.bootstrap_samples,
        "models": len(models),
        "standardizer_audit": standardizer_audit,
        "unit_rows": len(unit_rows),
        "by_seed_rows": len(by_seed_rows),
        "summary_rows": len(summary_rows),
        "paired_rows": len(paired_rows),
        "elapsed_seconds": elapsed,
        "device": str(DEVICE),
        "outputs": [str(path.relative_to(lc.ROOT)) for path in paths],
        "publication": "staged_then_atomic_no_clobber",
    }
    if args.model_tag:
        log.update({
            "model_tag": args.model_tag,
            "tagged_settings": sorted(tagged_settings),
            "untagged_settings": sorted(set(settings) - tagged_settings),
            "checkpoint_resolution": (
                "exact tagged/untagged paths; no missing-checkpoint fallback"
            ),
        })
    if strict_v2_bundles:
        log["strict_v2_training_bundles"] = strict_v2_bundles
        log["standardizer_provenance"] = strict_v2_standardizer
        log["strict_v2_evaluator_provenance"] = strict_v2_evaluator
        log["strict_v2_model_inputs"] = strict_v2_model_inputs
        log["strict_v2_inference_standardizer"] = (
            strict_v2_inference_standardizer
            if strict_v2_inference_standardizer is not None
            else {
                "policy": (
                    "CNN-only strict evaluation recomputes the committed "
                    "training convention from the frozen train_windows input; "
                    "CNN trainers do not persist per-arm scaler files"
                ),
                "source": strict_v2_standardizer,
            }
        )
    try:
        stage_and_publish(
            paths,
            (unit_rows, by_seed_rows, summary_rows, paired_rows),
            log,
        )
    except FileExistsError as exc:
        parser.error(str(exc))
    print(
        f"done unit_rows={len(unit_rows)} summary_rows={len(summary_rows)} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
