#!/usr/bin/env python3
"""Shared utilities for the journal extension pipeline.

Forked from wisa/scripts/train_real_only_baselines.py and
wisa/scripts/train_rule_synthetic_ratio_sweep.py @ commit eb131df.
Behavior is intentionally identical; only path constants resolve to journal/.
Do not import from wisa/scripts (its ROOT points into the frozen archive).
"""
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]  # journal/
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
MODELS = ROOT / "models" / "family_extension"

# Fonts vendored for the conceptual-figure SVG->PDF exports.  The Figma masters
# are laid out in Inter with per-run absolute coordinates, so a substituted
# metric-incompatible face silently overflows the fitted boxes.  Vendoring keeps
# the export deterministic on machines that do not have Inter installed.
FONTS = ROOT / "assets" / "fonts"

# Read-only reference to frozen WISA result tables (for key-comparison columns).
WISA_TABLES = ROOT.parent / "wisa" / "results" / "tables"

FEATURE_NAMES = ["can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "delta_t"]
CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}
SEEDS = [7, 42, 123, 2026, 3407]
SYNTHETIC_POOL = SYNTHETIC / "rule_based_windows.npz"
SYNTHETIC_POOLS = {
    "rule": SYNTHETIC / "rule_based_windows.npz",
    "gan": SYNTHETIC / "gan_windows.npz",
    # Protocol-repaired learned pools.  These keys intentionally do not
    # replace the legacy pools, so every downstream rerun is a new arm.
    "ganvalid": SYNTHETIC / "gan_protocol_valid_windows.npz",
    # Protocol-valid resamples of the two pre-existing WGAN generator-seed
    # sensitivity checkpoints.  The legacy ``ganb``/``ganc`` pools above were
    # generated before the DLC-conditioned padding audit and must not be used
    # as evidence for corrected generator-seed sensitivity.
    "ganvalidb": SYNTHETIC / "gan_gseed271828_protocol_valid_windows.npz",
    "ganvalidc": SYNTHETIC / "gan_gseed161803_protocol_valid_windows.npz",
    # generator-seed sensitivity pools (critique #2); pool keys must stay
    # underscore-free for build_train_arrays' "<pool>_<ratio>" setting parser
    "ganb": SYNTHETIC / "gan_gseed271828_windows.npz",
    "ganc": SYNTHETIC / "gan_gseed161803_windows.npz",
    # ID-marginal-calibrated contrast pool (tautology defense, B-plan §8)
    "gansnap": SYNTHETIC / "gan_idsnap_windows.npz",
    # E1: on-grammar learned generator (discrete-ID autoregressive LM;
    # experiments/e1_arlm_ongrammar/PREREG.md)
    "arlm": SYNTHETIC / "arlm_windows.npz",
    "arlmvalid": SYNTHETIC / "arlm_protocol_valid_windows.npz",
    # E12: AR-LM generator-seed sensitivity pools, protocol-valid sampling of
    # newly trained seed-271828/161803 generators (e12_arlm_gseed/PREREG.md)
    "arlmvalidb": SYNTHETIC / "arlm_gseed271828_protocol_valid_windows.npz",
    "arlmvalidc": SYNTHETIC / "arlm_gseed161803_protocol_valid_windows.npz",
    # E10: marginal-matched placebo twin of the rule pool — identical
    # positions/counts/IDs/DLC/timing/per-channel value multisets, only the
    # Gear/RPM inter-byte relational structure broken by constrained data1
    # permutation (experiments/e10_background_grammar_crossover/PREREG.md §2)
    "placebo": SYNTHETIC / "rule_placebo_windows.npz",
}

# The clean +100% rerun must not silently reuse the 65,000/class legacy pools.
# These are intentionally separate, not-yet-generated pool identities.  The
# strict-v2 builder below fails clearly until each versioned pool exists.
STRICT_V2_SAMPLING_POLICY = "strict_without_replacement_v2"
STRICT_V2_PLUS_100_REAL_TRAIN_WINDOWS = 262_149
STRICT_V2_MIN_PER_CLASS_CAPACITY = (
    STRICT_V2_PLUS_100_REAL_TRAIN_WINDOWS + 3
) // 4
# Frozen Car-Hacking source splits used by every canonical strict-v2 detector.
# These match wisa/results/logs/processed_dataset_hashes.txt; the arrays are
# gitignored, so Git cleanliness alone cannot establish their identity.
STRICT_V2_REAL_SPLIT_SHA256 = {
    "train": "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4",
    "val": "b4920e7f60fbf38b38e12976129f120ebff160a21a4c77093f3746f3c1aa218e",
}
STRICT_V2_ALLOWED_UNTRACKED_ARTIFACT_ROOTS = (
    "journal/results/",
    "journal/models/",
    "datasets/synthetic/",
)
STRICT_V2_ALLOWED_UNTRACKED_ARTIFACT_SUFFIXES = frozenset({
    ".csv",
    ".joblib",
    ".json",
    ".log",
    ".npz",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".stdout",
    ".svg",
    ".txt",
})
STRICT_V2_TAG_MARKERS = (
    "sampling_v2",
    "unique_pool_v2",
    "strict_v2",
    "strict-v2",
)
SAFE_ARTIFACT_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
STRICT_V2_POOL_CONFIGS = {
    "rule": {
        "identifier": "rule_based_unique_pool_v2",
        "path": SYNTHETIC / "rule_based_windows_unique_pool_v2.npz",
        "pool_schema_version": "rule_based_synthetic_pool_v2",
        "statistics_path": (
            TABLES / "rule_based_synthetic_statistics_unique_pool_v2.csv"
        ),
        "generation_log": (
            LOGS / "generate_rule_based_synthetic_unique_pool_v2.log"
        ),
        "minimum_windows_per_class": STRICT_V2_MIN_PER_CLASS_CAPACITY,
        "generator_configuration": {
            "family": "rule_based",
            "construction_seed": 314159,
            "source_train_sha256": (
                "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9"
                "eda81a34faec7cb4"
            ),
            "lineage": "legacy_rule_configuration_extended_without_replacement",
        },
    },
    "ganvalid": {
        "identifier": "gan_protocol_valid_unique_pool_v2",
        "path": SYNTHETIC / "gan_protocol_valid_windows_unique_pool_v2.npz",
        "pool_schema_version": "gan_protocol_valid_unique_pool_v2",
        "statistics_path": (
            TABLES
            / "gan_synthetic_statistics_protocol_valid_unique_pool_v2.csv"
        ),
        "generation_log": (
            LOGS / "generate_gan_synthetic_protocol_valid_unique_pool_v2.log"
        ),
        "minimum_windows_per_class": STRICT_V2_MIN_PER_CLASS_CAPACITY,
        "generator_configuration": {
            "family": "wgan_gp_protocol_valid",
            "generator_training_seed": 314159,
            "pool_sampling_seed": 314159,
            "generator_checkpoint_sha256": (
                "2a83647aafc9ae62df8d390ac5c896a4c6307e825e083b4c"
                "33bf09b5bef04517"
            ),
            "standardizer_sha256": (
                "7ffaff380c0e4af079ca68f94bd9c1ec35e597be99266e36"
                "31d7284d8a05e389"
            ),
            "lineage": "existing_corrected_generator_extended_without_replacement",
        },
    },
    "ganvalidb": {
        "identifier": "gan_gseed271828_protocol_valid_unique_pool_v2",
        "path": SYNTHETIC / "gan_gseed271828_protocol_valid_windows_unique_pool_v2.npz",
        "pool_schema_version": "gan_protocol_valid_unique_pool_v2",
        "statistics_path": (
            TABLES
            / "gan_synthetic_statistics_gseed271828_protocol_valid_unique_pool_v2.csv"
        ),
        "generation_log": (
            LOGS
            / "generate_gan_synthetic_gseed271828_protocol_valid_unique_pool_v2.log"
        ),
        "minimum_windows_per_class": STRICT_V2_MIN_PER_CLASS_CAPACITY,
        "generator_configuration": {
            "family": "wgan_gp_protocol_valid",
            "generator_training_seed": 271828,
            "pool_sampling_seed": 314159,
            "generator_checkpoint_sha256": (
                "bc1003d286bab53f2b9ad9a534391f626a94c8e55e90084e"
                "03e01a23714886c8"
            ),
            "standardizer_sha256": (
                "7ffaff380c0e4af079ca68f94bd9c1ec35e597be99266e36"
                "31d7284d8a05e389"
            ),
            "lineage": "existing_corrected_generator_extended_without_replacement",
        },
    },
    "ganvalidc": {
        "identifier": "gan_gseed161803_protocol_valid_unique_pool_v2",
        "path": SYNTHETIC / "gan_gseed161803_protocol_valid_windows_unique_pool_v2.npz",
        "pool_schema_version": "gan_protocol_valid_unique_pool_v2",
        "statistics_path": (
            TABLES
            / "gan_synthetic_statistics_gseed161803_protocol_valid_unique_pool_v2.csv"
        ),
        "generation_log": (
            LOGS
            / "generate_gan_synthetic_gseed161803_protocol_valid_unique_pool_v2.log"
        ),
        "minimum_windows_per_class": STRICT_V2_MIN_PER_CLASS_CAPACITY,
        "generator_configuration": {
            "family": "wgan_gp_protocol_valid",
            "generator_training_seed": 161803,
            "pool_sampling_seed": 314159,
            "generator_checkpoint_sha256": (
                "a13c88185dc0c6a3123c8540f8dff13f002770c599c634d"
                "d1a47911318beea01"
            ),
            "standardizer_sha256": (
                "7ffaff380c0e4af079ca68f94bd9c1ec35e597be99266e36"
                "31d7284d8a05e389"
            ),
            "lineage": "existing_corrected_generator_extended_without_replacement",
        },
    },
}


def figure_font_env(staging):
    """Stage the vendored figure fonts and return an env that exposes them.

    fontconfig picks up ``$XDG_DATA_HOME/fonts``, so the faces are copied into
    ``staging/fonts`` and XDG_DATA_HOME is redirected there.  Pointing at a
    scratch directory instead of the repository keeps the caller's own XDG data
    untouched and leaves the font cache out of the working tree.
    """
    staging = Path(staging)
    font_dir = staging / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    faces = sorted(FONTS.glob("*.ttf"))
    if not faces:
        raise FileNotFoundError(f"no vendored fonts found under {FONTS}")
    for face in faces:
        shutil.copyfile(face, font_dir / face.name)
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(staging.resolve())
    return env


def is_strict_v2_tag(value):
    value = str(value or "")
    return any(marker in value for marker in STRICT_V2_TAG_MARKERS)


def validate_artifact_tag(
        value, *, name="tag", leading_underscore=False, allow_empty=True):
    """Validate one filename-only model/result tag."""
    value = str(value or "")
    if not value and allow_empty:
        return
    candidate = value
    if leading_underscore:
        if not value.startswith("_"):
            raise ValueError(f"{name} must start with '_'")
        candidate = value[1:]
    if not candidate or not SAFE_ARTIFACT_TAG.fullmatch(candidate):
        raise ValueError(
            f"{name} must contain only letters, digits, dot, underscore, "
            "or hyphen and may not contain path separators"
        )


def validate_strict_v2_training_scope(
        *, family, settings, seeds, tag, budget_mode=None):
    """Bind E13's exact canonical tags to its prespecified arm matrix."""
    normalized_tag = str(tag or "").strip("_")
    if normalized_tag not in {
        "sampling_v2",
        "matchedsteps_sampling_v2",
    }:
        return

    requested_seeds = [int(seed) for seed in seeds]
    if (
        len(requested_seeds) != len(set(requested_seeds))
        or not set(requested_seeds).issubset(SEEDS)
    ):
        raise ValueError(
            f"canonical {normalized_tag} jobs require unique seeds drawn "
            f"from {SEEDS}"
        )
    requested_settings = list(settings)
    if not requested_settings or len(requested_settings) != len(
            set(requested_settings)):
        raise ValueError(
            f"canonical {normalized_tag} jobs require a nonempty unique "
            "setting list"
        )

    if normalized_tag == "matchedsteps_sampling_v2":
        if family != "cnn" or budget_mode != "matched_steps":
            raise ValueError(
                "canonical matchedsteps_sampling_v2 is prespecified only "
                "for the CNN matched-step arm"
            )
        allowed_settings = {"ganvalid_1p00"}
    else:
        allowed_by_family = {
            "cnn": {
                "rule_1p00",
                "ganvalid_1p00",
                "ganvalidb_1p00",
                "ganvalidc_1p00",
            },
            "rf": {
                "rule_1p00",
                "ganvalid_1p00",
                "ganvalidb_1p00",
                "ganvalidc_1p00",
            },
            "lstm": {"rule_1p00"},
            "transformer": {"rule_1p00"},
        }
        allowed_settings = allowed_by_family.get(family)
        if allowed_settings is None:
            raise ValueError(
                f"canonical sampling_v2 has no declared {family!r} arm"
            )
    unsupported = sorted(set(requested_settings) - allowed_settings)
    if unsupported:
        raise ValueError(
            f"canonical {normalized_tag} {family} scope excludes settings: "
            f"{unsupported}"
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(split):
    return np.load(WINDOWS / f"{split}_windows.npz", allow_pickle=True)


def fit_standardizer(x):
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def expected_calibration_error(y_true_binary, attack_scores, n_bins=15):
    confidence = np.maximum(attack_scores, 1.0 - attack_scores)
    pred = (attack_scores >= 0.5).astype(np.int8)
    correct = (pred == y_true_binary).astype(np.float32)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(model_name, split, y_attack_true, attack_scores, pred_attack_label, seed):
    y_binary_true = (y_attack_true > 0).astype(np.int8)
    y_binary_pred = (pred_attack_label > 0).astype(np.int8)
    cm = confusion_matrix(y_binary_true, y_binary_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    row = {
        "model": model_name,
        "seed": seed,
        "split": split,
        "accuracy": float((y_binary_true == y_binary_pred).mean()),
        "macro_f1_binary": float(f1_score(y_binary_true, y_binary_pred, average="macro", zero_division=0)),
        "attack_recall": float(recall_score(y_binary_true, y_binary_pred, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(y_binary_true, y_binary_pred, pos_label=0, zero_division=0)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auprc": float(average_precision_score(y_binary_true, attack_scores)) if len(np.unique(y_binary_true)) > 1 else "",
        "ece": expected_calibration_error(y_binary_true, attack_scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    for cls in [1, 2, 3, 4]:
        mask = y_attack_true == cls
        if np.any(mask):
            row[f"recall_{CLASS_NAMES[cls]}"] = float((pred_attack_label[mask] == cls).mean())
        else:
            row[f"recall_{CLASS_NAMES[cls]}"] = ""
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        # Keep generated text artifacts platform-stable and friendly to
        # repository diff checks.  The csv module otherwise emits CRLF even
        # on Linux when ``newline=''`` is used.
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class WindowDataset(Dataset):
    """(N,128,11) float32 -> (N,11,128) tensors, labels int64."""

    def __init__(self, x, y):
        self.x = torch.from_numpy(x.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SequenceDataset(Dataset):
    """(N,128,11) kept time-major for recurrent models (batch_first LSTM)."""

    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def sample_synthetic_indices(y_syn, total_count, seed):
    """Class-balanced synthetic sampling. Seed rule must stay `seed + int(ratio*1000)`
    at call sites to reproduce the WISA training-set construction."""
    rng = np.random.default_rng(seed)
    classes = [1, 2, 3, 4]
    per_class = total_count // len(classes)
    remainder = total_count - per_class * len(classes)
    indices = []
    for i, cls in enumerate(classes):
        cls_idx = np.where(y_syn == cls)[0]
        n = per_class + (1 if i < remainder else 0)
        replace = n > len(cls_idx)
        indices.append(rng.choice(cls_idx, size=n, replace=replace))
    out = np.concatenate(indices)
    rng.shuffle(out)
    return out


def sample_synthetic_indices_v2(y_syn, total_count, seed, classes=(1, 2, 3, 4)):
    """Strict class-balanced sampling without replacement.

    This is a separate API so the legacy sampler above remains bit-for-bit
    available for reproducing the historical arms.  Capacity is checked for
    every class before any draw is made; an undersized pool is an error rather
    than a trigger for implicit replacement.

    Returns ``(indices, audit)``.  The audit is JSON-serializable and records
    requested/drawn/unique/repeated counts per class.
    """
    y_syn = np.asarray(y_syn).reshape(-1)
    if isinstance(total_count, (bool, np.bool_)) or not isinstance(
            total_count, (int, np.integer)):
        raise TypeError("total_count must be an integer")
    total_count = int(total_count)
    if total_count < 0:
        raise ValueError("total_count must be non-negative")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    classes = tuple(int(cls) for cls in classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("classes must be a non-empty sequence of unique labels")

    per_class = total_count // len(classes)
    remainder = total_count - per_class * len(classes)
    requests = {
        cls: per_class + (1 if position < remainder else 0)
        for position, cls in enumerate(classes)
    }
    candidates = {cls: np.flatnonzero(y_syn == cls) for cls in classes}
    shortages = {
        cls: {"requested": requests[cls], "available": int(len(candidates[cls]))}
        for cls in classes
        if requests[cls] > len(candidates[cls])
    }
    if shortages:
        details = ", ".join(
            f"class {cls}: requested {counts['requested']}, "
            f"available {counts['available']}"
            for cls, counts in shortages.items()
        )
        raise ValueError(
            f"{STRICT_V2_SAMPLING_POLICY} capacity exceeded ({details})"
        )

    rng = np.random.default_rng(seed)
    chunks = []
    class_audit = {}
    for cls in classes:
        requested = requests[cls]
        drawn = rng.choice(candidates[cls], size=requested, replace=False)
        unique = int(np.unique(drawn).size)
        class_audit[str(cls)] = {
            "available": int(len(candidates[cls])),
            "requested": int(requested),
            "drawn": int(len(drawn)),
            "unique": unique,
            "repeated": int(len(drawn) - unique),
        }
        chunks.append(drawn.astype(np.int64, copy=False))

    out = (np.concatenate(chunks) if chunks
           else np.empty(0, dtype=np.int64))
    rng.shuffle(out)
    unique_total = int(np.unique(out).size)
    audit = {
        "schema_version": "synthetic_sampling_indices_v2",
        "sampling_policy": STRICT_V2_SAMPLING_POLICY,
        "sampling_seed": seed,
        "classes": list(classes),
        "remainder_assignment": "class_order_prefix",
        "total": {
            "requested": total_count,
            "drawn": int(len(out)),
            "unique": unique_total,
            "repeated": int(len(out) - unique_total),
        },
        "per_class": class_audit,
        "indices_sha256": hashlib.sha256(
            np.asarray(out, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    return out, audit


def load_synthetic_pool(pool="rule"):
    data = np.load(SYNTHETIC_POOLS[pool], allow_pickle=True)
    return data["x"], data["y_attack_type"]


def build_train_arrays(setting, seed):
    """Reproduce the WISA ratio-sweep training-set construction for a setting.

    setting: 'real_only' | '<pool>_<ratio>' where pool is a SYNTHETIC_POOLS key,
    e.g. 'rule_0p30', 'gan_1p00'. Returns (train_x, train_y, val_x, val_y)
    raw (un-standardized) arrays. The synthetic sampling seed rule
    `seed + int(ratio*1000)` is shared across pools so gan arms mirror the
    rule-arm training-set construction exactly.
    """
    train = load_npz("train")
    val = load_npz("val")
    train_x, train_y = train["x"], train["y_attack_type"]
    val_x, val_y = val["x"], val["y_attack_type"]
    if setting == "real_only":
        return train_x, train_y, val_x, val_y
    pool = setting.split("_")[0]
    if pool == "over":
        # equal-budget real-attack oversampling null (critique #7): the
        # "synthetic" pool is the real training attacks themselves.
        ratio = float(setting.split("_")[1].replace("p", "."))
        atk = train_y > 0
        syn_x, syn_y = train_x[atk], train_y[atk]
    elif pool in SYNTHETIC_POOLS:
        ratio = float(setting.split("_")[1].replace("p", "."))
        syn_x, syn_y = load_synthetic_pool(pool)
    else:
        raise ValueError(f"unknown setting: {setting}")
    n_syn = round(len(train_x) * ratio)
    idx = sample_synthetic_indices(syn_y, n_syn, seed + int(ratio * 1000))
    aug_x = np.concatenate([train_x, syn_x[idx]], axis=0)
    aug_y = np.concatenate([train_y, syn_y[idx]], axis=0)
    return aug_x, aug_y, val_x, val_y


@lru_cache(maxsize=None)
def _sha256_file_version(path_string, size, mtime_ns):
    """Hash one immutable file version; size/mtime make cache reuse explicit."""
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    path = Path(path)
    stat = path.stat()
    return _sha256_file_version(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _strict_v2_npz_scalar(archive, key, pool_path):
    """Read one required scalar from a declared strict-v2 NPZ."""
    if key not in archive.files:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} is missing metadata key "
            f"{key!r}"
        )
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(
            f"declared strict-v2 pool {pool_path} metadata {key!r} must be "
            f"a scalar, got shape {value.shape}"
        )
    return value.item()


def _require_sha256_metadata(value, key, pool_path):
    value = str(value)
    if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(
            f"declared strict-v2 pool {pool_path} metadata {key!r} must be "
            "a 64-character SHA-256 hex digest"
        )
    return value.lower()


def _validate_declared_strict_v2_pool_metadata(
        archive, pool_key, declared, pool_path):
    """Validate that a declared path contains its declared pool identity.

    Explicit, fully specified ``build_train_arrays_v2`` overrides deliberately
    bypass this check.  Official pool paths do not: a filename cannot stand in
    for the generator seed, generator inputs, or construction configuration
    embedded in the archive.
    """
    expected_identifier = declared["identifier"]
    expected_generator = declared["generator_configuration"]
    observed_identifier = str(_strict_v2_npz_scalar(
        archive, "pool_identifier", pool_path))
    if observed_identifier != expected_identifier:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} pool_identifier mismatch: "
            f"expected {expected_identifier!r}, observed "
            f"{observed_identifier!r}"
        )

    sampling_policy = str(_strict_v2_npz_scalar(
        archive, "downstream_sampling_policy", pool_path))
    if sampling_policy != STRICT_V2_SAMPLING_POLICY:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} sampling policy mismatch: "
            f"expected {STRICT_V2_SAMPLING_POLICY!r}, observed "
            f"{sampling_policy!r}"
        )

    observed_schema = str(_strict_v2_npz_scalar(
        archive, "pool_schema_version", pool_path))
    expected_schema = declared["pool_schema_version"]
    if observed_schema != expected_schema:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} schema mismatch: expected "
            f"{expected_schema!r}, observed {observed_schema!r}"
        )

    configuration_json = str(_strict_v2_npz_scalar(
        archive, "generation_configuration_json", pool_path))
    configuration_sha256 = _require_sha256_metadata(
        _strict_v2_npz_scalar(
            archive, "generation_configuration_sha256", pool_path),
        "generation_configuration_sha256",
        pool_path,
    )
    actual_configuration_sha256 = hashlib.sha256(
        configuration_json.encode("utf-8")).hexdigest()
    if configuration_sha256 != actual_configuration_sha256:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} generation configuration "
            "SHA-256 mismatch"
        )
    try:
        generation_configuration = json.loads(configuration_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} has invalid generation "
            "configuration JSON"
        ) from exc
    if not isinstance(generation_configuration, dict):
        raise ValueError(
            f"declared strict-v2 pool {pool_path} generation configuration "
            "must be a JSON object"
        )
    embedded_generator = generation_configuration.get(
        "generator_configuration")
    if embedded_generator != expected_generator:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} generator configuration "
            f"mismatch for {pool_key!r}: expected "
            f"{json.dumps(expected_generator, sort_keys=True)}, observed "
            f"{json.dumps(embedded_generator, sort_keys=True)}"
        )
    configured_identifier = generation_configuration.get("pool_identifier")
    if configured_identifier != expected_identifier:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} generation configuration "
            "pool_identifier mismatch"
        )
    configured_policy = generation_configuration.get(
        "downstream_sampling_policy")
    if configured_policy != sampling_policy:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} generation configuration "
            "sampling policy mismatch"
        )

    family = expected_generator.get("family")
    observed_generator = str(_strict_v2_npz_scalar(
        archive, "generator", pool_path))
    metadata = {
        "pool_identifier": observed_identifier,
        "pool_schema_version": observed_schema,
        "downstream_sampling_policy": sampling_policy,
        "generation_configuration_sha256": configuration_sha256,
        "generator": observed_generator,
    }
    if family == "rule_based":
        if observed_generator != "rule_based":
            raise ValueError(
                f"declared strict-v2 Rule pool {pool_path} generator "
                f"mismatch: observed {observed_generator!r}"
            )
        observed_seed = int(_strict_v2_npz_scalar(
            archive, "construction_seed", pool_path))
        expected_seed = int(expected_generator["construction_seed"])
        if observed_seed != expected_seed:
            raise ValueError(
                f"declared strict-v2 Rule pool {pool_path} construction seed "
                f"mismatch: expected {expected_seed}, observed {observed_seed}"
            )
        source_train_sha256 = _require_sha256_metadata(
            _strict_v2_npz_scalar(
                archive, "source_train_sha256", pool_path),
            "source_train_sha256",
            pool_path,
        )
        expected_source_sha256 = expected_generator[
            "source_train_sha256"
        ]
        configured_source = generation_configuration.get("source")
        configured_source_sha256 = (
            configured_source.get("sha256")
            if isinstance(configured_source, dict)
            else None
        )
        if (
            source_train_sha256 != expected_source_sha256
            or configured_source_sha256 != expected_source_sha256
        ):
            raise ValueError(
                f"declared strict-v2 Rule pool {pool_path} source train "
                "SHA-256 mismatch"
            )
        source_commit = str(_strict_v2_npz_scalar(
            archive, "source_commit", pool_path)).strip()
        source_tracked_state_clean = bool(_strict_v2_npz_scalar(
            archive, "source_tracked_state_clean", pool_path))
        source_worktree_clean = bool(_strict_v2_npz_scalar(
            archive, "source_worktree_clean", pool_path))
        try:
            source_allowed_artifacts = json.loads(str(
                _strict_v2_npz_scalar(
                    archive,
                    "source_allowed_untracked_artifacts_json",
                    pool_path,
                )
            ))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"declared strict-v2 Rule pool {pool_path} has invalid "
                "allowed-artifact provenance JSON"
            ) from exc
        configured_provenance = generation_configuration.get(
            "source_provenance")
        provenance_mismatches = {}
        if not source_commit:
            provenance_mismatches["source_commit"] = {
                "expected": "nonempty",
                "observed": source_commit,
            }
        compare_values = {
            "source_commit": source_commit,
            "source_tracked_state_clean": source_tracked_state_clean,
            "source_worktree_clean": source_worktree_clean,
            "source_allowed_untracked_artifacts": source_allowed_artifacts,
        }
        configured_values = {
            "source_commit": generation_configuration.get("source_commit"),
            "source_tracked_state_clean": (
                configured_provenance.get("source_tracked_state_clean")
                if isinstance(configured_provenance, dict)
                else None
            ),
            "source_worktree_clean": (
                configured_provenance.get("source_worktree_clean")
                if isinstance(configured_provenance, dict)
                else None
            ),
            "source_allowed_untracked_artifacts": (
                configured_provenance.get(
                    "source_allowed_untracked_artifacts")
                if isinstance(configured_provenance, dict)
                else None
            ),
        }
        for key, expected_value in compare_values.items():
            if configured_values[key] != expected_value:
                provenance_mismatches[key] = {
                    "expected": expected_value,
                    "observed": configured_values[key],
                }
        if source_tracked_state_clean is not True:
            provenance_mismatches["source_tracked_state_clean"] = {
                "expected": True,
                "observed": source_tracked_state_clean,
            }
        if provenance_mismatches:
            raise ValueError(
                f"declared strict-v2 Rule pool {pool_path} source "
                "provenance mismatch: "
                f"{json.dumps(provenance_mismatches, sort_keys=True)}"
            )
        metadata.update({
            "construction_seed": observed_seed,
            "source_train_sha256": source_train_sha256,
            "source_commit": source_commit,
            "source_tracked_state_clean": source_tracked_state_clean,
            "source_worktree_clean": source_worktree_clean,
            "source_allowed_untracked_artifacts": source_allowed_artifacts,
        })
    elif family == "wgan_gp_protocol_valid":
        if observed_generator != "conditional_wgan_gp_aux":
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} generator "
                f"mismatch: observed {observed_generator!r}"
            )
        observed_training_seed = int(_strict_v2_npz_scalar(
            archive, "generator_training_seed", pool_path))
        expected_training_seed = int(
            expected_generator["generator_training_seed"])
        if observed_training_seed != expected_training_seed:
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} generator "
                f"training seed mismatch: expected {expected_training_seed}, "
                f"observed {observed_training_seed}"
            )
        observed_sampling_seed = int(_strict_v2_npz_scalar(
            archive, "pool_sampling_seed", pool_path))
        expected_sampling_seed = int(expected_generator["pool_sampling_seed"])
        if observed_sampling_seed != expected_sampling_seed:
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} pool sampling seed "
                f"mismatch: expected {expected_sampling_seed}, observed "
                f"{observed_sampling_seed}"
            )
        checkpoint_sha256 = _require_sha256_metadata(
            _strict_v2_npz_scalar(
                archive, "generator_checkpoint_sha256", pool_path),
            "generator_checkpoint_sha256",
            pool_path,
        )
        standardizer_sha256 = _require_sha256_metadata(
            _strict_v2_npz_scalar(
                archive, "standardizer_sha256", pool_path),
            "standardizer_sha256",
            pool_path,
        )
        if (
            checkpoint_sha256
            != expected_generator["generator_checkpoint_sha256"]
        ):
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} frozen generator "
                "checkpoint SHA-256 mismatch"
            )
        if (
            standardizer_sha256
            != expected_generator["standardizer_sha256"]
        ):
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} frozen "
                "standardizer SHA-256 mismatch"
            )
        source_commit = str(_strict_v2_npz_scalar(
            archive, "source_commit", pool_path)).strip()
        source_tracked_state_clean = bool(_strict_v2_npz_scalar(
            archive, "source_tracked_state_clean", pool_path))
        if not source_commit:
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} source_commit "
                "must be nonempty"
            )
        configured_provenance = generation_configuration.get(
            "input_provenance")
        if not isinstance(configured_provenance, dict):
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} generation "
                "configuration is missing input_provenance"
            )
        configured_values = {
            "generator_training_seed": observed_training_seed,
            "pool_sampling_seed": observed_sampling_seed,
            "generator_checkpoint_sha256": checkpoint_sha256,
            "standardizer_sha256": standardizer_sha256,
            "source_commit": source_commit,
        }
        provenance_mismatches = {
            key: {
                "expected": value,
                "observed": configured_provenance.get(key),
            }
            for key, value in configured_values.items()
            if configured_provenance.get(key) != value
        }
        if configured_provenance.get("source_tracked_state_clean") is not True:
            provenance_mismatches["source_tracked_state_clean"] = {
                "expected": True,
                "observed": configured_provenance.get(
                    "source_tracked_state_clean"),
            }
        if source_tracked_state_clean is not True:
            provenance_mismatches["archive_source_tracked_state_clean"] = {
                "expected": True,
                "observed": source_tracked_state_clean,
            }
        if provenance_mismatches:
            raise ValueError(
                f"declared strict-v2 WGAN pool {pool_path} input provenance "
                "mismatch: "
                f"{json.dumps(provenance_mismatches, sort_keys=True)}"
            )
        metadata.update({
            "generator_training_seed": observed_training_seed,
            "pool_sampling_seed": observed_sampling_seed,
            "generator_checkpoint_sha256": checkpoint_sha256,
            "standardizer_sha256": standardizer_sha256,
            "source_commit": source_commit,
            "source_tracked_state_clean": source_tracked_state_clean,
        })
    else:
        raise ValueError(
            f"declared strict-v2 pool {pool_key!r} has unsupported generator "
            f"family {family!r}"
        )
    return metadata


def _strict_v2_pool_content_audit(archive, pool_path):
    """Re-run the declared strict-v2 pool's consumer-side integrity gates."""
    required = {
        "x",
        "y_binary",
        "y_attack_type",
        "synthetic_type",
        "condition_label",
        "condition_name",
        "feature_names",
        "window_size",
        "stride",
        "generator_protocol_version",
        "pool_content_sha256",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} is missing content metadata "
            f"keys: {missing}"
        )

    x = np.asarray(archive["x"])
    y_binary = np.asarray(archive["y_binary"]).reshape(-1)
    y_attack = np.asarray(archive["y_attack_type"]).reshape(-1)
    synthetic_type = np.asarray(archive["synthetic_type"]).reshape(-1)
    condition_label = np.asarray(archive["condition_label"]).reshape(-1)
    condition_name = np.asarray(archive["condition_name"]).reshape(-1)
    if x.dtype != np.float32:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} x must be float32, got "
            f"{x.dtype}"
        )
    if x.ndim != 3 or x.shape[1:] != (128, len(FEATURE_NAMES)):
        raise ValueError(
            f"declared strict-v2 pool {pool_path} x must have shape "
            f"(N,128,{len(FEATURE_NAMES)}), got {x.shape}"
        )
    aligned = {
        "y_binary": len(y_binary),
        "y_attack_type": len(y_attack),
        "synthetic_type": len(synthetic_type),
        "condition_label": len(condition_label),
        "condition_name": len(condition_name),
    }
    if not len(x) or any(length != len(x) for length in aligned.values()):
        raise ValueError(
            f"declared strict-v2 pool {pool_path} has unaligned content "
            f"arrays: x={len(x)}, metadata={aligned}"
        )
    if int(_strict_v2_npz_scalar(
            archive, "window_size", pool_path)) != 128:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} window_size must be 128"
        )
    if int(_strict_v2_npz_scalar(
            archive, "stride", pool_path)) != 32:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} stride must be 32"
        )
    feature_names = [
        str(name) for name in np.asarray(
            archive["feature_names"]).reshape(-1)
    ]
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} feature_names mismatch"
        )
    protocol_version = str(_strict_v2_npz_scalar(
        archive, "generator_protocol_version", pool_path))
    if protocol_version != "dlc_payload_v1":
        raise ValueError(
            f"declared strict-v2 pool {pool_path} generator protocol "
            f"version mismatch: {protocol_version!r}"
        )

    observed_labels = {int(value) for value in np.unique(y_attack)}
    if observed_labels != {1, 2, 3, 4}:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} must contain exactly attack "
            f"labels 1..4, observed {sorted(observed_labels)}"
        )

    # Imported lazily so legacy training paths do not pay for or depend on
    # the stricter generation audit.
    from generator_protocol import (  # pylint: disable=import-outside-toplevel
        summarize_condition_metadata,
        summarize_protocol_validity,
    )

    protocol = summarize_protocol_validity(x)
    if not protocol["protocol_valid"]:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} failed consumer-side frame "
            "protocol validation"
        )
    condition = summarize_condition_metadata(
        y_binary,
        y_attack,
        synthetic_type,
        condition_label=condition_label,
        condition_name=condition_name,
    )
    if not condition["condition_metadata_valid"]:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} failed consumer-side "
            "condition-metadata validation"
        )

    ordered_digest = hashlib.sha256()
    global_seen = set()
    per_class_seen = {label: set() for label in (1, 2, 3, 4)}
    per_class_repeated = {label: 0 for label in (1, 2, 3, 4)}
    for row, label_raw in zip(x, y_attack, strict=True):
        label = int(label_raw)
        row_digest = hashlib.sha256(
            memoryview(np.ascontiguousarray(row)).cast("B")
        ).digest()
        ordered_digest.update(bytes((label,)))
        ordered_digest.update(row_digest)
        if row_digest in per_class_seen[label]:
            per_class_repeated[label] += 1
        else:
            per_class_seen[label].add(row_digest)
        global_seen.add(row_digest)

    content_sha256 = ordered_digest.hexdigest()
    embedded_sha256 = _require_sha256_metadata(
        _strict_v2_npz_scalar(
            archive, "pool_content_sha256", pool_path),
        "pool_content_sha256",
        pool_path,
    )
    if embedded_sha256 != content_sha256:
        raise ValueError(
            f"declared strict-v2 pool {pool_path} content SHA-256 mismatch"
        )
    total_repeated = len(x) - len(global_seen)
    if total_repeated or any(per_class_repeated.values()):
        raise ValueError(
            f"declared strict-v2 pool {pool_path} contains exact duplicate "
            "window content"
        )

    return {
        "consumer_content_audit": {
            "pool_content_sha256": content_sha256,
            "total_windows": int(len(x)),
            "total_unique": int(len(global_seen)),
            "total_repeated": int(total_repeated),
            "per_class_windows": {
                str(label): int(np.count_nonzero(y_attack == label))
                for label in (1, 2, 3, 4)
            },
            "per_class_repeated": {
                str(label): int(per_class_repeated[label])
                for label in (1, 2, 3, 4)
            },
            "protocol_valid": True,
            "condition_metadata_valid": True,
        }
    }


def _strict_v2_generation_log_binding(
        pool_path, pool_key, declared, pool_sha256, pool_metadata):
    """Bind a declared pool byte-for-byte to its completed generation log."""
    log_path = Path(declared["generation_log"])
    if not log_path.is_file():
        raise FileNotFoundError(
            f"declared strict-v2 pool generation log is missing: {log_path}"
        )
    try:
        record = json.loads(log_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"declared strict-v2 pool generation log is invalid: {log_path}"
        ) from exc

    family = declared["generator_configuration"]["family"]
    if family == "rule_based":
        schema = "rule_based_generation_log_v2"
        recorded_pool_sha256 = (
            record.get("outputs", {}).get("pool", {}).get("sha256")
        )
        recorded_configuration = record.get("configuration")
        recorded_configuration_sha256 = record.get("configuration_sha256")
        recorded_content_sha256 = (
            record.get("audit", {})
            .get("content_uniqueness", {})
            .get("content_sha256")
        )
    elif family == "wgan_gp_protocol_valid":
        schema = "gan_pool_generation_log_v2"
        recorded_pool_sha256 = record.get("output_sha256")
        recorded_configuration = record.get("generation_configuration")
        recorded_configuration_sha256 = record.get(
            "generation_configuration_sha256"
        )
        recorded_content_sha256 = (
            record.get("content_uniqueness", {}).get("content_sha256")
        )
    else:
        raise ValueError(
            f"unsupported strict-v2 generation-log family: {family!r}"
        )

    configuration_sha256 = pool_metadata[
        "generation_configuration_sha256"
    ]
    content_sha256 = pool_metadata[
        "consumer_content_audit"
    ]["pool_content_sha256"]
    canonical_recorded_configuration_sha256 = (
        hashlib.sha256(
            json.dumps(
                recorded_configuration,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if isinstance(recorded_configuration, dict)
        else None
    )
    mismatches = {}
    checks = {
        "schema_version": (schema, record.get("schema_version")),
        "pool_identifier": (
            declared["identifier"],
            record.get("pool_identifier"),
        ),
        "pool_sha256": (pool_sha256, recorded_pool_sha256),
        "generation_configuration_sha256": (
            configuration_sha256,
            recorded_configuration_sha256,
        ),
        "generation_configuration_content": (
            configuration_sha256,
            canonical_recorded_configuration_sha256,
        ),
        "pool_content_sha256": (
            content_sha256,
            recorded_content_sha256,
        ),
    }
    if family == "wgan_gp_protocol_valid":
        checks["strict_v2_pool_key"] = (
            pool_key,
            record.get("strict_v2_pool_key"),
        )
    for key, (expected, observed) in checks.items():
        if observed != expected:
            mismatches[key] = {
                "expected": expected,
                "observed": observed,
            }
    if mismatches:
        raise ValueError(
            f"declared strict-v2 pool generation-log mismatch for "
            f"{pool_path}: {json.dumps(mismatches, sort_keys=True)}"
        )
    return {
        "generation_log": {
            "path": _audit_path(log_path),
            "sha256": sha256_file(log_path),
            "schema_version": schema,
            "pool_sha256_verified": pool_sha256,
        }
    }


def strict_v2_artifact_record(artifacts):
    """Return portable paths and hashes for a completed training bundle."""
    normalized = {str(name): Path(path) for name, path in artifacts.items()}
    missing = [str(path) for path in normalized.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot record incomplete strict-v2 artifact bundle: "
            + ", ".join(missing)
        )
    return {
        "artifact_paths": {
            name: _audit_path(path) for name, path in normalized.items()
        },
        "artifact_sha256": {
            name: sha256_file(path) for name, path in normalized.items()
        },
    }


def write_json_atomic(path, payload):
    """Publish one JSON record atomically without replacing it implicitly."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_strict_v2_sampling_audit(audit, record):
    """Return field-level mismatches for one canonical training-array audit."""
    mismatches = {}

    def compare(key, expected_value, observed_value):
        if observed_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "observed": observed_value,
            }

    if not isinstance(audit, dict):
        return {
            "sampling_audit": {
                "expected": "canonical strict-v2 sampling audit object",
                "observed": audit,
            }
        }

    setting = record.get("setting")
    seed = record.get("seed")
    compare(
        "sampling_audit.schema_version",
        "synthetic_training_configuration_v2",
        audit.get("schema_version"),
    )
    compare(
        "sampling_audit.sampling_policy",
        STRICT_V2_SAMPLING_POLICY,
        audit.get("sampling_policy"),
    )
    compare("sampling_audit.setting", setting, audit.get("setting"))
    compare(
        "sampling_audit.detector_training_pipeline_seed",
        seed,
        audit.get("detector_training_pipeline_seed"),
    )
    compare("sampling_audit.pool_override", False, audit.get("pool_override"))

    parts = str(setting or "").split("_")
    if len(parts) != 2 or parts[0] not in STRICT_V2_POOL_CONFIGS:
        mismatches["sampling_audit.setting_contract"] = {
            "expected": "declared '<pool>_<ratio>' strict-v2 setting",
            "observed": setting,
        }
        return mismatches
    pool_key, ratio_text = parts
    try:
        ratio = float(ratio_text.replace("p", "."))
    except ValueError:
        ratio = None
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        mismatches["sampling_audit.ratio"] = {
            "expected": "finite positive ratio encoded by setting",
            "observed": audit.get("ratio"),
        }
        return mismatches
    compare("sampling_audit.ratio", ratio, audit.get("ratio"))

    declared = STRICT_V2_POOL_CONFIGS[pool_key]
    compare(
        "sampling_audit.pool_identifier",
        declared["identifier"],
        audit.get("pool_identifier"),
    )
    compare(
        "sampling_audit.pool_path",
        _audit_path(declared["path"]),
        audit.get("pool_path"),
    )
    compare(
        "sampling_audit.generator_configuration",
        declared["generator_configuration"],
        audit.get("generator_configuration"),
    )

    pool_sha256 = audit.get("pool_sha256")
    if not _is_sha256(pool_sha256):
        mismatches["sampling_audit.pool_sha256"] = {
            "expected": "64-character SHA-256",
            "observed": pool_sha256,
        }
    pool_path = Path(declared["path"])
    if not pool_path.is_file():
        mismatches["sampling_audit.current_pool"] = {
            "expected": str(pool_path),
            "observed": "missing",
        }
    elif _is_sha256(pool_sha256):
        current_pool_sha256 = sha256_file(pool_path)
        compare(
            "sampling_audit.current_pool_sha256",
            pool_sha256,
            current_pool_sha256,
        )

    pool_metadata = audit.get("pool_metadata")
    if not isinstance(pool_metadata, dict):
        mismatches["sampling_audit.pool_metadata"] = {
            "expected": "consumer-validated declared pool metadata",
            "observed": pool_metadata,
        }
    else:
        compare(
            "sampling_audit.pool_metadata.pool_identifier",
            declared["identifier"],
            pool_metadata.get("pool_identifier"),
        )
        compare(
            "sampling_audit.pool_metadata.pool_schema_version",
            declared["pool_schema_version"],
            pool_metadata.get("pool_schema_version"),
        )
        compare(
            "sampling_audit.pool_metadata.downstream_sampling_policy",
            STRICT_V2_SAMPLING_POLICY,
            pool_metadata.get("downstream_sampling_policy"),
        )
        content = pool_metadata.get("consumer_content_audit")
        if not isinstance(content, dict):
            mismatches[
                "sampling_audit.pool_metadata.consumer_content_audit"
            ] = {
                "expected": "completed consumer-side content audit",
                "observed": content,
            }
        else:
            if not _is_sha256(content.get("pool_content_sha256")):
                mismatches[
                    "sampling_audit.pool_metadata.pool_content_sha256"
                ] = {
                    "expected": "64-character SHA-256",
                    "observed": content.get("pool_content_sha256"),
                }
            compare(
                "sampling_audit.pool_metadata.total_repeated",
                0,
                content.get("total_repeated"),
            )
            compare(
                "sampling_audit.pool_metadata.protocol_valid",
                True,
                content.get("protocol_valid"),
            )
            compare(
                "sampling_audit.pool_metadata.condition_metadata_valid",
                True,
                content.get("condition_metadata_valid"),
            )
            repeated = content.get("per_class_repeated")
            compare(
                "sampling_audit.pool_metadata.per_class_repeated",
                {str(label): 0 for label in (1, 2, 3, 4)},
                repeated,
            )

        generation_log = pool_metadata.get("generation_log")
        expected_log_path = Path(declared["generation_log"])
        if not isinstance(generation_log, dict):
            mismatches[
                "sampling_audit.pool_metadata.generation_log"
            ] = {
                "expected": "generation-log/NPZ binding",
                "observed": generation_log,
            }
        else:
            compare(
                "sampling_audit.pool_metadata.generation_log.path",
                _audit_path(expected_log_path),
                generation_log.get("path"),
            )
            compare(
                "sampling_audit.pool_metadata.generation_log.pool_sha256",
                pool_sha256,
                generation_log.get("pool_sha256_verified"),
            )
            recorded_log_sha256 = generation_log.get("sha256")
            if not _is_sha256(recorded_log_sha256):
                mismatches[
                    "sampling_audit.pool_metadata.generation_log.sha256"
                ] = {
                    "expected": "64-character SHA-256",
                    "observed": recorded_log_sha256,
                }
            if not expected_log_path.is_file():
                mismatches["sampling_audit.current_generation_log"] = {
                    "expected": str(expected_log_path),
                    "observed": "missing",
                }
            elif _is_sha256(recorded_log_sha256):
                compare(
                    "sampling_audit.current_generation_log_sha256",
                    recorded_log_sha256,
                    sha256_file(expected_log_path),
                )
            if (
                pool_path.is_file()
                and _is_sha256(pool_sha256)
                and isinstance(content, dict)
            ):
                try:
                    rebound = _strict_v2_generation_log_binding(
                        pool_path,
                        pool_key,
                        declared,
                        pool_sha256,
                        pool_metadata,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    mismatches[
                        "sampling_audit.current_generation_log_binding"
                    ] = {
                        "expected": "current log binds current pool",
                        "observed": str(exc),
                    }
                else:
                    compare(
                        "sampling_audit.rebound_generation_log",
                        generation_log,
                        rebound["generation_log"],
                    )

    sampling = audit.get("sampling")
    total_requested = audit.get("synthetic_windows_requested")
    real_train_windows = audit.get("real_train_windows")
    expected_sampling_seed = (
        int(seed) + int(ratio * 1000)
        if isinstance(seed, int)
        else None
    )
    compare(
        "sampling_audit.sampling_seed",
        expected_sampling_seed,
        audit.get("sampling_seed"),
    )
    if (
        isinstance(real_train_windows, int)
        and not isinstance(real_train_windows, bool)
    ):
        compare(
            "sampling_audit.synthetic_windows_requested",
            round(real_train_windows * ratio),
            total_requested,
        )
        compare(
            "sampling_audit.augmented_train_windows",
            real_train_windows + round(real_train_windows * ratio),
            audit.get("augmented_train_windows"),
        )
        if "train_windows" in record:
            train_subset = record.get("train_subset")
            tag = str(record.get("tag") or record.get("model_tag") or "")
            if train_subset is None:
                compare(
                    "training_log.train_windows",
                    audit.get("augmented_train_windows"),
                    record.get("train_windows"),
                )
            elif (
                "smoke" in tag.lower()
                and isinstance(train_subset, (int, float))
                and not isinstance(train_subset, bool)
                and 0 < float(train_subset) <= 1
            ):
                observed_train_windows = record.get("train_windows")
                if (
                    not isinstance(observed_train_windows, int)
                    or observed_train_windows <= 0
                    or observed_train_windows
                    > audit.get("augmented_train_windows")
                ):
                    mismatches["training_log.train_windows"] = {
                        "expected": (
                            "positive smoke subset no larger than the full "
                            "augmented training array"
                        ),
                        "observed": observed_train_windows,
                    }
            else:
                mismatches["training_log.train_subset"] = {
                    "expected": (
                        "None for canonical runs or a 0<f<=1 subset with a "
                        "smoke tag"
                    ),
                    "observed": train_subset,
                }
    else:
        mismatches["sampling_audit.real_train_windows"] = {
            "expected": "positive integer",
            "observed": real_train_windows,
        }

    if not isinstance(sampling, dict):
        mismatches["sampling_audit.sampling"] = {
            "expected": "strict no-replacement draw audit",
            "observed": sampling,
        }
    else:
        compare(
            "sampling_audit.sampling.schema_version",
            "synthetic_sampling_indices_v2",
            sampling.get("schema_version"),
        )
        compare(
            "sampling_audit.sampling.sampling_policy",
            STRICT_V2_SAMPLING_POLICY,
            sampling.get("sampling_policy"),
        )
        compare(
            "sampling_audit.sampling.sampling_seed",
            expected_sampling_seed,
            sampling.get("sampling_seed"),
        )
        compare(
            "sampling_audit.sampling.classes",
            [1, 2, 3, 4],
            sampling.get("classes"),
        )
        compare(
            "sampling_audit.sampling.remainder_assignment",
            "class_order_prefix",
            sampling.get("remainder_assignment"),
        )
        if not _is_sha256(sampling.get("indices_sha256")):
            mismatches["sampling_audit.sampling.indices_sha256"] = {
                "expected": "64-character SHA-256",
                "observed": sampling.get("indices_sha256"),
            }
        total = sampling.get("total")
        expected_total = {
            "requested": total_requested,
            "drawn": total_requested,
            "unique": total_requested,
            "repeated": 0,
        }
        compare(
            "sampling_audit.sampling.total",
            expected_total,
            total,
        )
        per_class = sampling.get("per_class")
        if not isinstance(per_class, dict) or not isinstance(
                total_requested, int):
            mismatches["sampling_audit.sampling.per_class"] = {
                "expected": "complete class 1..4 audit",
                "observed": per_class,
            }
        else:
            base, remainder = divmod(total_requested, 4)
            pool_counts = audit.get("pool_class_counts")
            for position, label in enumerate((1, 2, 3, 4)):
                requested = base + (position < remainder)
                observed = per_class.get(str(label))
                if not isinstance(observed, dict):
                    mismatches[
                        f"sampling_audit.sampling.per_class.{label}"
                    ] = {
                        "expected": "class draw audit",
                        "observed": observed,
                    }
                    continue
                compare(
                    f"sampling_audit.sampling.per_class.{label}.requested",
                    int(requested),
                    observed.get("requested"),
                )
                compare(
                    f"sampling_audit.sampling.per_class.{label}.drawn",
                    int(requested),
                    observed.get("drawn"),
                )
                compare(
                    f"sampling_audit.sampling.per_class.{label}.unique",
                    int(requested),
                    observed.get("unique"),
                )
                compare(
                    f"sampling_audit.sampling.per_class.{label}.repeated",
                    0,
                    observed.get("repeated"),
                )
                available = observed.get("available")
                if not isinstance(available, int) or available < requested:
                    mismatches[
                        f"sampling_audit.sampling.per_class.{label}.available"
                    ] = {
                        "expected": f">={requested}",
                        "observed": available,
                    }
                if isinstance(pool_counts, dict):
                    compare(
                        f"sampling_audit.pool_class_counts.{label}",
                        available,
                        pool_counts.get(str(label)),
                    )

    config_keys = (
        "schema_version",
        "setting",
        "ratio",
        "sampling_policy",
        "detector_training_pipeline_seed",
        "sampling_seed",
        "generator_configuration",
        "pool_identifier",
        "pool_path",
        "pool_sha256",
        "pool_override",
        "pool_metadata",
        "real_data_provenance",
        "source_commit",
        "source_worktree_clean",
        "source_tracked_state_clean",
        "source_allowed_untracked_artifacts",
        "source_provenance_policy",
    )
    if any(key not in audit for key in config_keys):
        mismatches["sampling_audit.configuration_fields"] = {
            "expected": list(config_keys),
            "observed": sorted(audit),
        }
    else:
        configuration = {key: audit[key] for key in config_keys}
        actual_config_sha256 = hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        compare(
            "sampling_audit.config_sha256",
            actual_config_sha256,
            audit.get("config_sha256"),
        )
    compare(
        "sampling_audit.source_tracked_state_clean",
        True,
        audit.get("source_tracked_state_clean"),
    )
    if not str(audit.get("source_commit") or "").strip():
        mismatches["sampling_audit.source_commit"] = {
            "expected": "non-empty clean source commit",
            "observed": audit.get("source_commit"),
        }
    return mismatches


def validate_strict_v2_training_bundle(log_path, artifacts, expected):
    """Validate a resumable strict-v2 model/scaler/audit bundle.

    A checkpoint without its companion audit log is deliberately not
    resumable.  The caller must use a new version tag after inspecting any
    partial artifacts instead of silently accepting them.
    """
    log_path = Path(log_path)
    normalized = {str(name): Path(path) for name, path in artifacts.items()}
    missing = [
        str(path)
        for path in [*normalized.values(), log_path]
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "incomplete strict-v2 training bundle; inspect the partial "
            "artifacts and rerun with a new tag: " + ", ".join(missing)
        )
    try:
        record = json.loads(log_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid strict-v2 training audit log: {log_path}"
        ) from exc
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    audit = record.get("sampling_audit")
    mismatches.update(_validate_strict_v2_sampling_audit(audit, record))
    if isinstance(audit, dict):
        recorded_data = audit.get("real_data_provenance")
        if not isinstance(recorded_data, dict) or (
            recorded_data.get("policy") != "frozen_source_split_sha256"
        ):
            mismatches["sampling_audit.real_data_provenance"] = {
                "expected": "frozen_source_split_sha256",
                "observed": recorded_data,
            }
        else:
            try:
                current_data = _strict_v2_real_split_file_provenance()
            except (FileNotFoundError, RuntimeError) as exc:
                mismatches["current_real_data_provenance"] = {
                    "expected": "frozen train/val split hashes",
                    "observed": str(exc),
                }
            else:
                for split in ("train", "val"):
                    recorded_split = recorded_data.get(split)
                    expected_split = current_data[split]
                    if not isinstance(recorded_split, dict) or (
                        recorded_split.get("sha256")
                        != expected_split["sha256"]
                    ):
                        mismatches[
                            f"sampling_audit.real_data_provenance.{split}"
                        ] = {
                            "expected": expected_split,
                            "observed": recorded_split,
                        }
    recorded_paths = record.get("artifact_paths")
    expected_paths = {
        name: _audit_path(path) for name, path in normalized.items()
    }
    if recorded_paths != expected_paths:
        mismatches["artifact_paths"] = {
            "expected": expected_paths,
            "observed": recorded_paths,
        }
    recorded_hashes = record.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict):
        mismatches["artifact_sha256"] = {
            "expected": sorted(normalized),
            "observed": recorded_hashes,
        }
    elif set(recorded_hashes) != set(normalized):
        mismatches["artifact_sha256.keys"] = {
            "expected": sorted(normalized),
            "observed": sorted(recorded_hashes),
        }
    else:
        for name, path in normalized.items():
            observed = recorded_hashes.get(name)
            actual = sha256_file(path)
            if observed != actual:
                mismatches[f"artifact_sha256.{name}"] = {
                    "expected": actual,
                    "observed": observed,
                }
    if mismatches:
        raise RuntimeError(
            "strict-v2 training bundle audit mismatch; do not reuse this "
            f"tag: {json.dumps(mismatches, sort_keys=True)}"
        )
    return record


def _strict_v2_status_records(status_output):
    """Parse porcelain-v1 status records, including the test-friendly form."""
    if "\0" in status_output:
        return [record for record in status_output.split("\0") if record]
    return [record for record in status_output.splitlines() if record]


def _strict_v2_allowed_untracked_artifact(status_record):
    if not status_record.startswith("?? "):
        return False
    relative_path = status_record[3:].replace("\\", "/")
    if not any(
            relative_path.startswith(root)
            for root in STRICT_V2_ALLOWED_UNTRACKED_ARTIFACT_ROOTS):
        return False
    return (
        Path(relative_path).suffix.lower()
        in STRICT_V2_ALLOWED_UNTRACKED_ARTIFACT_SUFFIXES
    )


@lru_cache(maxsize=1)
def _source_commit():
    """Return a commit whose tracked/source state is reproducibly clean.

    Strict-v2 runs are intentionally sequential and no-clobber.  Untracked
    generated artifacts under the narrow results/models/synthetic roots may
    remain between processes; tracked modifications and untracked source,
    tests, documentation, or experiment definitions remain hard failures.
    The accepted status is cached before the process writes any new output.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "strict-v2 provenance requires an accessible Git worktree"
        ) from exc
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("strict-v2 provenance could not resolve HEAD")
    records = _strict_v2_status_records(status.stdout)
    allowed_artifacts = sorted(
        record[3:]
        for record in records
        if _strict_v2_allowed_untracked_artifact(record)
    )
    blocking = [
        record
        for record in records
        if not _strict_v2_allowed_untracked_artifact(record)
    ]
    if blocking:
        raise RuntimeError(
            "strict-v2 provenance requires a clean Git worktree outside "
            "no-clobber generated artifact paths; commit or otherwise "
            "resolve source changes before generation/training: "
            + json.dumps(blocking, ensure_ascii=True)
        )
    _source_commit.last_provenance = {
        "source_commit": commit,
        "source_worktree_clean": not allowed_artifacts,
        "source_tracked_state_clean": True,
        "source_allowed_untracked_artifacts": allowed_artifacts,
        "source_provenance_policy": (
            "tracked_source_clean_allow_untracked_no_clobber_artifacts"
        ),
    }
    return commit


def _source_provenance():
    """Return the cached strict-v2 source-state audit for this process."""
    commit = _source_commit()
    provenance = getattr(_source_commit, "last_provenance", None)
    if not isinstance(provenance, dict) or (
            provenance.get("source_commit") != commit):
        raise RuntimeError("strict-v2 source provenance cache is unavailable")
    return {
        **provenance,
        "source_allowed_untracked_artifacts": list(
            provenance["source_allowed_untracked_artifacts"]),
    }


def _audit_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.parent.resolve()))
    except ValueError:
        return str(path)


def _strict_v2_real_split_file_provenance():
    """Hash the gitignored frozen train/validation inputs used by strict-v2."""
    provenance = {"policy": "frozen_source_split_sha256"}
    for split, expected_sha256 in STRICT_V2_REAL_SPLIT_SHA256.items():
        path = WINDOWS / f"{split}_windows.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"strict-v2 frozen source split is missing: {path}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"strict-v2 {split} split SHA-256 mismatch: expected "
                f"{expected_sha256}, observed {actual_sha256}"
            )
        provenance[split] = {
            "path": _audit_path(path),
            "sha256": actual_sha256,
        }
    return provenance


def build_train_arrays_v2(
        setting,
        seed,
        *,
        pool_path=None,
        pool_identifier=None,
        generator_configuration=None,
        source_commit=None):
    """Build one explicitly strict-v2 augmented training arm.

    ``build_train_arrays`` remains the legacy entry point.  Callers must opt in
    to this function and must still use the historical arm name (for example,
    ``rule_1p00``); the distinct policy is recorded in the returned audit and
    should also be carried in versioned checkpoint/output tags.

    The optional pool arguments exist for tests and controlled handoffs.  In a
    production rerun, omit them so the declared versioned pool identity in
    ``STRICT_V2_POOL_CONFIGS`` is used.

    Returns ``(train_x, train_y, val_x, val_y, audit)``.
    """
    parts = setting.split("_")
    if len(parts) != 2:
        raise ValueError(
            "strict-v2 setting must be exactly '<pool>_<ratio>', "
            f"got {setting!r}"
        )
    pool, ratio_text = parts
    if pool not in STRICT_V2_POOL_CONFIGS:
        supported = ", ".join(sorted(STRICT_V2_POOL_CONFIGS))
        raise ValueError(
            f"strict-v2 pool {pool!r} is not declared; supported pools: {supported}"
        )
    pool_override = any(
        value is not None
        for value in (pool_path, pool_identifier, generator_configuration)
    )
    if pool_override and not all(
        value is not None
        for value in (pool_path, pool_identifier, generator_configuration)
    ):
        raise ValueError(
            "strict-v2 pool override requires pool_path, pool_identifier, "
            "and generator_configuration together"
        )
    try:
        ratio = float(ratio_text.replace("p", "."))
    except ValueError as exc:
        raise ValueError(f"invalid strict-v2 ratio in setting {setting!r}") from exc
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("strict-v2 ratio must be finite and positive")

    declared = STRICT_V2_POOL_CONFIGS[pool]
    selected_path = Path(pool_path) if pool_path is not None else Path(declared["path"])
    selected_identifier = pool_identifier or declared["identifier"]
    selected_generator = (
        generator_configuration
        if generator_configuration is not None
        else declared["generator_configuration"]
    )
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"strict-v2 pool does not exist: {selected_path}; generate the "
            "versioned expanded pool without replacing the legacy pool"
        )

    train = load_npz("train")
    val = load_npz("val")
    train_x, train_y = train["x"], train["y_attack_type"]
    val_x, val_y = val["x"], val["y_attack_type"]
    if source_commit is None:
        real_data_provenance = _strict_v2_real_split_file_provenance()
        real_data_provenance["train"].update({
            "windows": int(len(train_y)),
            "x_shape": list(train_x.shape),
            "y_shape": list(train_y.shape),
        })
        real_data_provenance["val"].update({
            "windows": int(len(val_y)),
            "x_shape": list(val_x.shape),
            "y_shape": list(val_y.shape),
        })
    else:
        # Explicit source_commit is reserved for unit tests and controlled
        # handoffs; such bundles are not accepted as canonical by
        # validate_strict_v2_training_bundle.
        real_data_provenance = {
            "policy": "explicit_test_or_controlled_override",
            "train": {
                "windows": int(len(train_y)),
                "x_shape": list(train_x.shape),
                "y_shape": list(train_y.shape),
            },
            "val": {
                "windows": int(len(val_y)),
                "x_shape": list(val_x.shape),
                "y_shape": list(val_y.shape),
            },
        }
    with np.load(selected_path, allow_pickle=True) as synthetic:
        required = {"x", "y_attack_type"}
        missing = sorted(required - set(synthetic.files))
        if missing:
            raise ValueError(
                f"strict-v2 pool {selected_path} is missing keys: {missing}"
            )
        pool_metadata = (
            None
            if pool_override
            else _validate_declared_strict_v2_pool_metadata(
                synthetic, pool, declared, selected_path)
        )
        if pool_metadata is not None:
            pool_metadata.update(
                _strict_v2_pool_content_audit(synthetic, selected_path)
            )
        syn_x = synthetic["x"]
        syn_y = synthetic["y_attack_type"]
    if len(syn_x) != len(syn_y):
        raise ValueError(
            f"strict-v2 pool x/y length mismatch: {len(syn_x)} != {len(syn_y)}"
        )

    n_syn = round(len(train_x) * ratio)
    sampling_seed = int(seed) + int(ratio * 1000)
    idx, sampling_audit = sample_synthetic_indices_v2(
        syn_y, n_syn, sampling_seed
    )
    pool_digest = sha256_file(selected_path)
    if pool_metadata is not None:
        pool_metadata.update(
            _strict_v2_generation_log_binding(
                selected_path,
                pool,
                declared,
                pool_digest,
                pool_metadata,
            )
        )
    source_provenance = (
        None
        if source_commit is not None
        else _source_provenance()
    )
    commit = (
        source_commit
        if source_commit is not None
        else source_provenance["source_commit"]
    )
    config = {
        "schema_version": "synthetic_training_configuration_v2",
        "setting": setting,
        "ratio": ratio,
        "sampling_policy": STRICT_V2_SAMPLING_POLICY,
        "detector_training_pipeline_seed": int(seed),
        "sampling_seed": sampling_seed,
        "generator_configuration": selected_generator,
        "pool_identifier": selected_identifier,
        "pool_path": _audit_path(selected_path),
        "pool_sha256": pool_digest,
        "pool_override": pool_override,
        "pool_metadata": pool_metadata,
        "real_data_provenance": real_data_provenance,
        "source_commit": commit,
        "source_worktree_clean": (
            source_provenance["source_worktree_clean"]
            if source_provenance is not None
            else None
        ),
        "source_tracked_state_clean": (
            source_provenance["source_tracked_state_clean"]
            if source_provenance is not None
            else None
        ),
        "source_allowed_untracked_artifacts": (
            source_provenance["source_allowed_untracked_artifacts"]
            if source_provenance is not None
            else None
        ),
        "source_provenance_policy": (
            source_provenance["source_provenance_policy"]
            if source_provenance is not None
            else "explicit_test_or_controlled_override"
        ),
    }
    config_sha256 = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        **config,
        "config_sha256": config_sha256,
        "pool_windows": int(len(syn_y)),
        "pool_class_counts": {
            str(cls): int(np.count_nonzero(syn_y == cls))
            for cls in (1, 2, 3, 4)
        },
        "real_train_windows": int(len(train_x)),
        "synthetic_windows_requested": int(n_syn),
        "sampling": sampling_audit,
    }

    aug_x = np.concatenate([train_x, syn_x[idx]], axis=0)
    aug_y = np.concatenate([train_y, syn_y[idx]], axis=0)
    audit["augmented_train_windows"] = int(len(aug_y))
    return aug_x, aug_y, val_x, val_y, audit
