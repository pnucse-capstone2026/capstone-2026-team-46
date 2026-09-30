#!/usr/bin/env python3
"""E14 matched L4 construction-counterfactual factorial evaluator.

The prospective contract is frozen in
``journal/experiments/e14_l4_counterfactual_factorial/PREREG.md``.  This
module deliberately keeps Stage A (``--prepare-only``) free of checkpoint
deserialization and model inference.  PyTorch is imported lazily inside the
Stage-B scoring functions only.

The public construction helpers are pure NumPy functions so the registered
factorial can be regression-tested on toy windows without touching production
inputs or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = JOURNAL_ROOT.parent
TABLES = JOURNAL_ROOT / "results" / "tables"
LOGS = JOURNAL_ROOT / "results" / "logs"
MODELS = JOURNAL_ROOT / "models" / "generator_extension"
WINDOWS = JOURNAL_ROOT / "datasets" / "windows"
EXP = JOURNAL_ROOT / "experiments" / "e14_l4_counterfactual_factorial"

PREREG = EXP / "PREREG.md"
AMENDMENT = EXP / "IMPLEMENTATION_AMENDMENT_2026-07-24.md"
POST_TRAINING_AMENDMENT = (
    EXP
    / "IMPLEMENTATION_AMENDMENT_2026-07-24_STAGE_A_TRANSCRIPT.md"
)
# Descriptive aliases retained for the transition checks and their tests.
TRANSCRIPT_AMENDMENT = POST_TRAINING_AMENDMENT
BLOCK_MANIFEST = TABLES / "evaluation_realization_blocks_e13_sampling_v2.csv"
E8_RUN_RECORD = (
    JOURNAL_ROOT / "experiments" / "e8_evaluation_realization"
    / "run_e13_sampling_v2.json"
)
TEST_WINDOWS = WINDOWS / "test_windows.npz"
TRAIN_WINDOWS = WINDOWS / "train_windows.npz"
MATCHED_REAL_PREFLIGHT = EXP / "matched_real_training_preflight_v1.json"
MATCHED_REAL_RUN = EXP / "matched_real_training_run_v1.json"
MATCHED_REAL_TRANSCRIPT = EXP / "matched_real_training_child_output_v1.log"
FAILURE_DIR = EXP / "failures"
TRAINING_SOURCE_COMMIT = "ef7afd2da361718c3a6fe4ad75cd82068430c61c"
AMENDMENT_SOURCE_COMMIT = "ee21d150277fe83b0bcbc188829920de164b2b01"
TRAINING_IMPLEMENTATION_COMMIT = TRAINING_SOURCE_COMMIT
TRANSCRIPT_AMENDMENT_COMMIT = AMENDMENT_SOURCE_COMMIT
POST_TRAINING_AMENDMENT_SHA256 = (
    "bae44420c2d8745363b8baf8cb0d0d1652b6bea9fc980b9b32196be41b445b54"
)
MATCHED_REAL_RECORD_HASHES = {
    MATCHED_REAL_PREFLIGHT:
        "3b895fe3c84e5188968fedf0ef0450f2ace03c98e3fe1e3ea11aa3a8f98e1e14",
    MATCHED_REAL_RUN:
        "53865949ecf4383fa6f9a21c224b76c289baeb9c88daa7fbf80eddfacfe0de28",
    MATCHED_REAL_TRANSCRIPT:
        "e362b71ca4cdff7755e27b16ca614ca2f1e574c12c8fc95777aeb095a631e0e5",
}
MATCHED_REAL_RECORD_BYTES = {
    MATCHED_REAL_PREFLIGHT: 21_223,
    MATCHED_REAL_RUN: 37_000,
    MATCHED_REAL_TRANSCRIPT: 4_977,
}
MATCHED_REAL_TRAINING_RECORD_KEYS = frozenset(
    {"preflight", "run", "child_transcript"}
)
IMPLEMENTATION_SOURCES = {
    "training_wrapper": JOURNAL_ROOT
    / "scripts"
    / "run_e14_matched_real_training.py",
    "evaluator": Path(__file__).resolve(),
    "analyzer": JOURNAL_ROOT
    / "scripts"
    / "analyze_l4_counterfactual_factorial.py",
    "tests": JOURNAL_ROOT
    / "tests"
    / "test_l4_counterfactual_factorial.py",
    "amendment": AMENDMENT,
    "transcript_amendment": POST_TRAINING_AMENDMENT,
}

TRAINING_TO_AMENDMENT_PATHS = (
    "journal/experiments/e14_l4_counterfactual_factorial/"
    "IMPLEMENTATION_AMENDMENT_2026-07-24_STAGE_A_TRANSCRIPT.md",
)
AMENDMENT_TO_PREPARE_PATHS = (
    "journal/scripts/evaluate_l4_counterfactual_factorial.py",
    "journal/scripts/analyze_l4_counterfactual_factorial.py",
    "journal/tests/test_l4_counterfactual_factorial.py",
)
AMENDMENT_ONLY_PATHS = frozenset(
    ("A", path) for path in TRAINING_TO_AMENDMENT_PATHS
)
IMPLEMENTATION_CORRECTION_PATHS = frozenset(
    ("M", path) for path in AMENDMENT_TO_PREPARE_PATHS
)
FULL_TRANSITION_PATHS = AMENDMENT_ONLY_PATHS | IMPLEMENTATION_CORRECTION_PATHS

PREPARE_PATHS = {
    "base_manifest": TABLES / "e14_l4_factorial_base_manifest_v1.csv",
    "latent_manifest": TABLES / "e14_l4_factorial_latent_manifest_v1.csv",
    "manipulation_checks": TABLES
    / "e14_l4_factorial_manipulation_checks_v1.csv",
    "prepare_record": EXP / "prepare_v1.json",
}
SCORE_PATHS = {
    "normal_by_seed": TABLES / "e14_l4_factorial_normal_by_seed_v1.csv",
    "by_cell": TABLES / "e14_l4_factorial_by_cell_v1.csv",
    "by_scenario": TABLES / "e14_l4_factorial_by_scenario_v1.csv",
    "augmentation_delta": TABLES
    / "e14_l4_factorial_augmentation_delta_v1.csv",
    "log": LOGS / "e14_l4_factorial_v1.log",
    "run_record": EXP / "run_v1.json",
}

FEATURE_NAMES = [
    "can_id", "dlc", "data0", "data1", "data2", "data3", "data4",
    "data5", "data6", "data7", "delta_t",
]
WINDOW_SHAPE = (128, 11)
SEEDS = (7, 42, 123, 2026, 3407)
BLOCK_IDS = ("block_01", "block_02", "block_03")
BASES_PER_BLOCK = 2_000
MASTER_CONSTRUCTION_SEED = 2_026_071_400
SCORING_BATCH_SIZE = 4_096
TORCH_THREADS = 8
TRANSFORM_SERIALIZATION_VERSION = "e14-transform-instance-le-f4-v1"
LATENT_SERIALIZATION_VERSION = "e14-latent-json-and-le-f4-v1"

EXPECTED_HASHES = {
    PREREG: "f0465fea952b9f7d5fd1b20a7864a96192099b9d3346f6c81e4274ce0fe0424a",
    AMENDMENT:
        "f6ecf1c56a08ef9d6fc02f3a2f9f9277b6c4e7cf4c972dd4314bbcc4bb431b62",
    POST_TRAINING_AMENDMENT: POST_TRAINING_AMENDMENT_SHA256,
    BLOCK_MANIFEST: "9e18b3844e20f259938c732a10bc0251767f0d2eaae9c731ba8da7ef087f5efb",
    E8_RUN_RECORD: "fae2ca82bb26ab24a5699fce9f331951be23746093fef9e28d0b240fd2c3b2db",
    TEST_WINDOWS: "4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7",
    TRAIN_WINDOWS: "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4",
    JOURNAL_ROOT / "datasets" / "synthetic" / "rule_based_windows.npz":
        "4078ab00b8531f760b296b5fd293e3df914bef996d317c958ead4762137bafd5",
    JOURNAL_ROOT / "datasets" / "synthetic" / "rule_placebo_windows.npz":
        "2b336ed46bb8f3248d10831c80fdfa6db25c3e6eb44f4fd680f78546a63ec4fa",
    LOGS / "generate_rule_placebo_twin.log":
        "5ca8287c5f7cef322808f73759fa1ad69e5b6d1c80182ab4b795809d0bfaddc6",
    JOURNAL_ROOT / "scripts" / "evaluate_evaluation_realization.py":
        "eadf93df237946b7f670a74e316f0b80fc1d1920d4cc7072c7434c699a2ff2bb",
    JOURNAL_ROOT / "scripts" / "generate_rule_based_synthetic_v2.py":
        "34ff0374af1856fd842da8226580d5ff0d47cbd16b36fb4273de3e06bcac2570",
    JOURNAL_ROOT / "scripts" / "e10a_crossover.py":
        "a1156ce1228b1975906fcfe8aad1c888e307944e7e88b6242aa32e5c3713432f",
    JOURNAL_ROOT / "scripts" / "train_generator_extension_cnn.py":
        "a5d88bcb579da77e234e991a9b050c040df21d2f75e971c0c952c6226749db61",
    JOURNAL_ROOT / "scripts" / "lib_common.py":
        "5848ff4b0063d4394cd1d6627faf49e072cb363d7796cc64fce3acc48b05639a",
}
EXPECTED_STANDARDIZER_HASHES = {
    "mean": "1e53f10b4dfb1de8522513ca20c7d0c2b5a19a8de8f76067e4273a93467da8a2",
    "std": "092f42e9a1019fe709233a91c6e876e976f8b3b8616807b42eb6e6da7b7dbb7c",
    "mean_then_std":
        "9019b1a91c983859e8dd998717eb5c7b612ed03dc2a65ec63402fd54ab42ef4e",
}

ATTACK_SPECS = {
    3: {
        "attack": "Gear",
        "canonical_id": 0x43F,
        "shifted_id": 0x440,
        "p0_columns": (2, 3),
        "p1_columns": (8, 9),
    },
    4: {
        "attack": "RPM",
        "canonical_id": 0x316,
        "shifted_id": 0x329,
        "p0_columns": (2, 3, 4),
        "p1_columns": (6, 8, 9),
    },
}
ID_STRATA = ("canonical", "shifted")
CELLS = tuple((p, s, d) for p in (0, 1) for s in (0, 1) for d in (0, 1))
ARMS = ("real_ms", "rule_ms", "placebo_ms", "real_std", "rule_std", "placebo_std")
CONTRASTS = (
    ("rule_ms-real_ms", "rule_ms", "real_ms"),
    ("placebo_ms-real_ms", "placebo_ms", "real_ms"),
    ("rule_ms-placebo_ms", "rule_ms", "placebo_ms"),
    ("rule_std-real_std", "rule_std", "real_std"),
    ("placebo_std-real_std", "placebo_std", "real_std"),
    ("rule_std-placebo_std", "rule_std", "placebo_std"),
)

EXISTING_CHECKPOINT_HASHES = {
    ("real_std", 7): "a16f97511c74629db2e76075eab94d82f0d1a3f169737a09369967261c36dac3",
    ("rule_ms", 7): "27c37e443ea6880761163bc2ea10ed5a86ab3ab1a6a6969770a284e2fd0d1182",
    ("placebo_ms", 7): "60bcf0a0d36a9b7725d64239a7e9cabe639ebc13248b62a436f05b7fa770869f",
    ("rule_std", 7): "77e90846fb1ab3282ef8b48a3000c2f13dbb5d37dbb3ad68208ea892f5bd6947",
    ("placebo_std", 7): "ac7fe4a64dcb9205425db8865b1bcd8d6f4faade46e42496b729a879bdbb2919",
    ("real_std", 42): "a6a06376f0348fcbbb058047eaf8e64b488c5695054265e8f5ef28b6a4388d0e",
    ("rule_ms", 42): "36616d6c1450337e51a172ab97a63e702c4147ab0da1411b0ced9e359a326364",
    ("placebo_ms", 42): "c0baf3e93be225dc1d08bbb1efbcd82be242f63d6b4719b619f269169eb6ea3f",
    ("rule_std", 42): "3ad60e856c38cd2c94ea6fcbdb5156b3ab82222648e48ce84aebdc100775511d",
    ("placebo_std", 42): "640970486f0581fd24b4a4ff9f2b5c73ae8893a665129357b0f62db35bce7d02",
    ("real_std", 123): "b395d863bb67c382b66f002cdc17bfe96e30e4dadebe90005266f113ac139149",
    ("rule_ms", 123): "d68719b70933d1b91d9cec34ddc8300e55c6710df622627c9702db9b6e2932e0",
    ("placebo_ms", 123): "6eae99dfd6ac4899e2f5a6c56207389fec653962b303a96aadc6bceab50337df",
    ("rule_std", 123): "2f2bf9b2f9bfa75ec903656864eaee2eb42b454c966dbe6872c1533d0d2f6193",
    ("placebo_std", 123): "938fca913ce50778e8f7df0f31c90671aac0437ce007034b923ec9a99870074e",
    ("real_std", 2026): "5bc1f150bea0e2f76e407ea7f4df80423a54db871255e23dd268fc690d1ff792",
    ("rule_ms", 2026): "3623d62c3bc7d4f56771a000f02c08eec36ea91f2dccecd8588599b2490118f9",
    ("placebo_ms", 2026): "7bf9a3ce69e98915c7c989e26cb3a0decd6fba12901ab348a45e09ea029ea8ec",
    ("rule_std", 2026): "fed3aa9ff056f2f82dfe24ba67b37c25b4d8fb4f3a2ac2eda093d7e0af0c023f",
    ("placebo_std", 2026): "5673e6137a82e39294f05b073c1fb2e3f474382d6b741f29581fb4ee1b1079cb",
    ("real_std", 3407): "d57c5cde43323fff38bb594bb7716b965bc171a3c598b4bddfb94d1f4c2307d9",
    ("rule_ms", 3407): "2248b276a2ac4c380faf890ed8fc6ff2cc8e27bdd90d3d41b60dcf3beee4a5cf",
    ("placebo_ms", 3407): "cfe3a9def49cb35af4c18e8c5a03ab807c4694f8a48021dd7e6a2b5ff368a7a8",
    ("rule_std", 3407): "e23120af83183f5d967d4761ab92f5f31b98833fed19289569d77122d95efabb",
    ("placebo_std", 3407): "3b36a329f78d1098defa6ceda13ab58952364a505b5375405281392760fa41bf",
}


@dataclass(frozen=True)
class FactorLatent:
    """Order-independent latent variables for one base × attack pair."""

    block_id: str
    block_number: int
    block_position: int
    test_window_index: int
    attack_label: int
    l4_child_seed: int
    rule_child_seed: int
    start: int
    s0_positions: np.ndarray
    s1_positions: np.ndarray
    d0_roles: tuple[np.ndarray, ...]
    d1_roles: tuple[np.ndarray, ...]
    d0_driver: np.ndarray
    d0_jitters: tuple[np.ndarray, ...]
    d1_jitter: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    return str(Path(os.path.abspath(path)).relative_to(REPO_ROOT.absolute()))


def _f4(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype="<f4"))


def array_sha256(values: Any) -> str:
    array = _f4(values)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def derive_child_seed(
    block_number: int,
    test_window_index: int,
    attack_label: int,
    stream_code: int,
    master_seed: int = MASTER_CONSTRUCTION_SEED,
) -> int:
    """Derive the registered order-independent uint64 child seed."""
    if block_number not in (1, 2, 3):
        raise ValueError(f"invalid block number: {block_number}")
    if test_window_index < 0:
        raise ValueError("test_window_index must be non-negative")
    if attack_label not in ATTACK_SPECS:
        raise ValueError(f"invalid E14 attack label: {attack_label}")
    if stream_code not in (1, 2):
        raise ValueError(f"invalid E14 stream code: {stream_code}")
    sequence = np.random.SeedSequence(
        [master_seed, block_number, test_window_index, attack_label, stream_code]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def build_latent(
    base: np.ndarray,
    *,
    block_id: str,
    block_number: int,
    block_position: int,
    test_window_index: int,
    attack_label: int,
) -> FactorLatent:
    """Build the registered shared latent for one base × Gear/RPM attack."""
    base = np.asarray(base)
    if base.shape != WINDOW_SHAPE:
        raise ValueError(f"base must have shape {WINDOW_SHAPE}, got {base.shape}")
    if base.dtype != np.float32 or not np.isfinite(base).all():
        raise ValueError("base must be finite float32")

    l4_seed = derive_child_seed(
        block_number, test_window_index, attack_label, 1
    )
    rule_seed = derive_child_seed(
        block_number, test_window_index, attack_label, 2
    )
    l4_rng = np.random.default_rng(l4_seed)
    start = int(l4_rng.integers(0, 33))
    s0 = np.arange(start, start + 64, 2, dtype=np.int16)
    s1 = np.arange(start, start + 96, 3, dtype=np.int16)
    if len(s0) != 32 or len(s1) != 32:
        raise AssertionError("registered masks must each contain 32 frames")

    rank = np.arange(32)
    ramp = np.linspace(0, 255, 32, dtype=np.float32)
    if attack_label == 3:
        d1_jitter = l4_rng.integers(-10, 11, size=32)
        d1_roles = (
            _f4(ramp),
            _f4(np.clip(255 - ramp + d1_jitter, 0, 255)),
        )
    else:
        d1_jitter = l4_rng.integers(-12, 13, size=32)
        d1_roles = (
            _f4((rank.astype(np.float32) * 37) % 256),
            _f4(
                127.5
                + 127.5 * np.sin(np.linspace(0, 3 * np.pi, 32))
            ),
            _f4(np.clip(ramp + d1_jitter, 0, 255)),
        )

    rule_rng = np.random.default_rng(rule_seed)
    if attack_label == 3:
        driver = rule_rng.choice([0, 1, 2, 3, 4, 5], size=32)
        jitter0 = rule_rng.integers(0, 16, size=32)
        role0 = np.clip(driver * 40 + jitter0, 0, 255)
        jitter1 = rule_rng.integers(-8, 9, size=32)
        role1 = np.clip(255 - role0 + jitter1, 0, 255)
        d0_roles = (_f4(role0), _f4(role1))
        d0_jitters = (np.asarray(jitter0), np.asarray(jitter1))
    else:
        driver = rule_rng.integers(0, 8000, size=32)
        jitter2 = rule_rng.integers(-50, 51, size=32)
        d0_roles = (
            _f4((driver // 32) % 256),
            _f4((driver // 4) % 256),
            _f4(np.clip(base[s0, 4] + jitter2, 0, 255)),
        )
        d0_jitters = (np.asarray(jitter2),)

    latent = FactorLatent(
        block_id=block_id,
        block_number=int(block_number),
        block_position=int(block_position),
        test_window_index=int(test_window_index),
        attack_label=int(attack_label),
        l4_child_seed=l4_seed,
        rule_child_seed=rule_seed,
        start=start,
        s0_positions=np.asarray(s0, dtype=np.int16),
        s1_positions=np.asarray(s1, dtype=np.int16),
        d0_roles=tuple(_f4(role) for role in d0_roles),
        d1_roles=tuple(_f4(role) for role in d1_roles),
        d0_driver=np.asarray(driver),
        d0_jitters=tuple(np.asarray(value) for value in d0_jitters),
        d1_jitter=np.asarray(d1_jitter),
    )
    validate_latent(latent)
    return latent


def validate_latent(latent: FactorLatent) -> None:
    expected_roles = 2 if latent.attack_label == 3 else 3
    if latent.start < 0 or latent.start > 32:
        raise AssertionError("latent start outside registered common support")
    expected_s0 = latent.start + 2 * np.arange(32)
    expected_s1 = latent.start + 3 * np.arange(32)
    if not np.array_equal(latent.s0_positions, expected_s0):
        raise AssertionError("S0 mask violates registered formula")
    if not np.array_equal(latent.s1_positions, expected_s1):
        raise AssertionError("S1 mask violates registered formula")
    for family in (latent.d0_roles, latent.d1_roles):
        if len(family) != expected_roles:
            raise AssertionError("role-count mismatch")
        for role in family:
            if role.shape != (32,) or role.dtype != np.dtype("<f4"):
                raise AssertionError("roles must be 32-vector little-endian float32")
            if not np.isfinite(role).all() or (role < 0).any() or (role > 255).any():
                raise AssertionError("payload role outside finite byte range")


def apply_factorial_transform(
    base: np.ndarray,
    latent: FactorLatent,
    *,
    attack_label: int,
    id_stratum: str,
    p: int,
    s: int,
    d: int,
) -> np.ndarray:
    """Apply one registered P×S×D cell to a base window."""
    if attack_label != latent.attack_label or attack_label not in ATTACK_SPECS:
        raise ValueError("attack/latent mismatch")
    if id_stratum not in ID_STRATA or (p, s, d) not in CELLS:
        raise ValueError("undeclared E14 stratum or factorial cell")
    base = np.asarray(base)
    if base.shape != WINDOW_SHAPE or base.dtype != np.float32:
        raise ValueError("base must be float32 (128,11)")
    result = base.copy()
    spec = ATTACK_SPECS[attack_label]
    positions = latent.s0_positions if s == 0 else latent.s1_positions
    columns = spec["p0_columns"] if p == 0 else spec["p1_columns"]
    roles = latent.d0_roles if d == 0 else latent.d1_roles
    target_id = spec[f"{id_stratum}_id"]
    result[positions, 0] = target_id
    result[positions, 1] = 8
    for column, role in zip(columns, roles, strict=True):
        result[positions, column] = role
    result[positions, 10] = np.maximum(result[positions, 10], 1e-5)
    return np.ascontiguousarray(result, dtype=np.float32)


def cell_name(p: int, s: int, d: int) -> str:
    if (p, s, d) not in CELLS:
        raise ValueError("undeclared E14 cell")
    return f"{p}{s}{d}"


def transformation_key(
    latent: FactorLatent, id_stratum: str, p: int, s: int, d: int
) -> tuple[str, int, int, int, str, int, int, int]:
    return (
        latent.block_id,
        latent.block_position,
        latent.test_window_index,
        latent.attack_label,
        id_stratum,
        p,
        s,
        d,
    )


def transformation_key_bytes(key: Sequence[Any]) -> bytes:
    if len(key) != 8:
        raise ValueError("transformation key must contain eight fields")
    encoded = json.dumps(
        list(key), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return struct.pack("<Q", len(encoded)) + encoded


def update_transform_digest(
    digest: Any, key: Sequence[Any], window: np.ndarray
) -> None:
    array = np.asarray(window)
    if array.shape != WINDOW_SHAPE:
        raise ValueError("digest window shape mismatch")
    payload = np.ascontiguousarray(array, dtype="<f4")
    digest.update(transformation_key_bytes(key))
    digest.update(b"<f4")
    digest.update(struct.pack("<II", *WINDOW_SHAPE))
    digest.update(memoryview(payload).cast("B"))


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _array_json(values: np.ndarray) -> list[Any]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return [int(value) for value in array]
    return [float(value) for value in array]


def latent_to_row(latent: FactorLatent) -> dict[str, Any]:
    """Serialize every registered latent array to one CSV-safe row."""
    spec = ATTACK_SPECS[latent.attack_label]
    return {
        "block_id": latent.block_id,
        "block_number": latent.block_number,
        "block_position": latent.block_position,
        "test_window_index": latent.test_window_index,
        "attack": spec["attack"],
        "attack_label": latent.attack_label,
        "l4_child_seed": str(latent.l4_child_seed),
        "rule_child_seed": str(latent.rule_child_seed),
        "start": latent.start,
        "s0_positions_json": _compact_json(_array_json(latent.s0_positions)),
        "s1_positions_json": _compact_json(_array_json(latent.s1_positions)),
        "d0_roles_json": _compact_json(
            [_array_json(role) for role in latent.d0_roles]
        ),
        "d1_roles_json": _compact_json(
            [_array_json(role) for role in latent.d1_roles]
        ),
        "d0_driver_json": _compact_json(_array_json(latent.d0_driver)),
        "d0_jitters_json": _compact_json(
            [_array_json(jitter) for jitter in latent.d0_jitters]
        ),
        "d1_jitter_json": _compact_json(_array_json(latent.d1_jitter)),
        "serialization_version": LATENT_SERIALIZATION_VERSION,
    }


def latent_from_row(row: Mapping[str, Any]) -> FactorLatent:
    def arrays(name: str, dtype: Any) -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(values, dtype=dtype)
            for values in json.loads(str(row[name]))
        )

    latent = FactorLatent(
        block_id=str(row["block_id"]),
        block_number=int(row["block_number"]),
        block_position=int(row["block_position"]),
        test_window_index=int(row["test_window_index"]),
        attack_label=int(row["attack_label"]),
        l4_child_seed=int(str(row["l4_child_seed"])),
        rule_child_seed=int(str(row["rule_child_seed"])),
        start=int(row["start"]),
        s0_positions=np.asarray(
            json.loads(str(row["s0_positions_json"])), dtype=np.int16
        ),
        s1_positions=np.asarray(
            json.loads(str(row["s1_positions_json"])), dtype=np.int16
        ),
        d0_roles=arrays("d0_roles_json", "<f4"),
        d1_roles=arrays("d1_roles_json", "<f4"),
        d0_driver=np.asarray(
            json.loads(str(row["d0_driver_json"])), dtype=np.int64
        ),
        d0_jitters=arrays("d0_jitters_json", np.int64),
        d1_jitter=np.asarray(
            json.loads(str(row["d1_jitter_json"])), dtype=np.int64
        ),
    )
    if str(row.get("serialization_version")) != LATENT_SERIALIZATION_VERSION:
        raise ValueError("latent serialization version mismatch")
    validate_latent(latent)
    return latent


def latent_semantic_digest(latents: Sequence[FactorLatent]) -> str:
    """Hash keys, seeds, masks, drivers, jitters, and float32 role bytes."""
    digest = hashlib.sha256()
    previous: tuple[Any, ...] | None = None
    for latent in latents:
        key = (
            latent.block_id,
            latent.block_position,
            latent.test_window_index,
            latent.attack_label,
        )
        if previous is not None and key <= previous:
            raise AssertionError("latent rows are not strictly lexicographically sorted")
        previous = key
        payload = _compact_json(
            [
                *key,
                latent.block_number,
                str(latent.l4_child_seed),
                str(latent.rule_child_seed),
                latent.start,
            ]
        ).encode("ascii")
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
        for values in (
            latent.s0_positions,
            latent.s1_positions,
            latent.d0_driver,
            *latent.d0_jitters,
            latent.d1_jitter,
        ):
            array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
            digest.update(struct.pack("<Q", array.size))
            digest.update(memoryview(array).cast("B"))
        for role in (*latent.d0_roles, *latent.d1_roles):
            array = _f4(role)
            digest.update(struct.pack("<Q", array.size))
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        missing = {
            "x", "y_binary", "y_attack_type", "source_file", "segment_id",
            "start_index", "end_index", "feature_names", "window_size",
        } - set(archive.files)
        if missing:
            raise KeyError(f"{path} lacks required arrays: {sorted(missing)}")
        values = {name: archive[name] for name in archive.files}
    if values["x"].dtype != np.float32 or values["x"].shape[1:] != WINDOW_SHAPE:
        raise ValueError(f"{path} has invalid x representation")
    feature_names = [str(value) for value in values["feature_names"].reshape(-1)]
    if feature_names != FEATURE_NAMES or int(values["window_size"]) != 128:
        raise ValueError(f"{path} feature/window metadata mismatch")
    return values


def _assert_frame_disjoint(frame: pd.DataFrame) -> None:
    for key, group in frame.groupby(["source_file", "segment_id"], sort=True):
        previous_end = -1
        for row in group.sort_values(
            ["start_index", "end_index", "test_window_index"]
        ).itertuples(index=False):
            if int(row.start_index) <= previous_end:
                raise AssertionError(
                    f"selected base frame overlap for {key}: "
                    f"{row.start_index} <= {previous_end}"
                )
            previous_end = int(row.end_index)


def build_base_manifest(
    manifest: pd.DataFrame, test: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate and materialize the frozen first-2,000-per-block base panel."""
    required = {
        "block_id", "block_position", "test_window_index", "source_file",
        "segment_id", "start_index", "end_index",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"E8 block manifest lacks columns: {sorted(missing)}")
    selected = manifest.loc[
        manifest["block_position"].astype(np.int64) < BASES_PER_BLOCK
    ].copy()
    selected = selected.sort_values(
        ["block_id", "block_position", "test_window_index"], kind="stable"
    ).reset_index(drop=True)
    if set(selected["block_id"]) != set(BLOCK_IDS):
        raise AssertionError("E14 requires exactly block_01..block_03")
    for block_id, group in selected.groupby("block_id", sort=True):
        if len(group) != BASES_PER_BLOCK:
            raise AssertionError(f"{block_id} does not contain 2,000 bases")
        if not np.array_equal(
            group["block_position"].to_numpy(dtype=np.int64),
            np.arange(BASES_PER_BLOCK),
        ):
            raise AssertionError(f"{block_id} block positions are not 0..1999")
    if len(selected) != 6_000:
        raise AssertionError("E14 base manifest must contain exactly 6,000 rows")
    indices = selected["test_window_index"].to_numpy(dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("test_window_index is not unique across E14 bases")
    if indices.min() < 0 or indices.max() >= len(test["x"]):
        raise IndexError("E14 manifest contains an out-of-range test index")

    checks = (
        ("source_file", lambda x: str(x)),
        ("segment_id", int),
        ("start_index", int),
        ("end_index", int),
    )
    for column, normalize in checks:
        observed = [normalize(value) for value in selected[column]]
        expected = [normalize(test[column][index]) for index in indices]
        if observed != expected:
            raise AssertionError(f"E8/test metadata mismatch for {column}")
    y_binary = np.asarray(test["y_binary"])[indices]
    y_attack = np.asarray(test["y_attack_type"])[indices]
    if np.any(y_binary != 0) or np.any(y_attack != 0):
        raise AssertionError("all E14 bases must be normal under both labels")
    triples = selected[["source_file", "start_index", "end_index"]]
    if triples.duplicated().any():
        raise AssertionError("(source_file,start_index,end_index) is not unique")
    _assert_frame_disjoint(selected)

    bases = np.ascontiguousarray(test["x"][indices], dtype=np.float32)
    if not np.isfinite(bases).all():
        raise AssertionError("selected bases contain non-finite values")
    selected.insert(
        1,
        "block_number",
        selected["block_id"].str.extract(r"(\d+)$")[0].astype(int),
    )
    selected["y_binary"] = 0
    selected["y_attack_type"] = 0
    selected["window_sha256"] = [
        hashlib.sha256(
            memoryview(np.ascontiguousarray(window, dtype="<f4")).cast("B")
        ).hexdigest()
        for window in bases
    ]
    selected["base_order"] = np.arange(len(selected), dtype=np.int64)
    return selected, bases


def build_latent_manifest(
    base_manifest: pd.DataFrame, bases: np.ndarray
) -> tuple[pd.DataFrame, list[FactorLatent]]:
    if len(base_manifest) != len(bases):
        raise ValueError("base manifest/window length mismatch")
    latents: list[FactorLatent] = []
    for row, base in zip(
        base_manifest.itertuples(index=False), bases, strict=True
    ):
        for attack_label in sorted(ATTACK_SPECS):
            latents.append(
                build_latent(
                    base,
                    block_id=str(row.block_id),
                    block_number=int(row.block_number),
                    block_position=int(row.block_position),
                    test_window_index=int(row.test_window_index),
                    attack_label=attack_label,
                )
            )
    if len(latents) != 12_000:
        raise AssertionError("E14 must contain exactly 12,000 latent rows")
    frame = pd.DataFrame(latent_to_row(latent) for latent in latents)
    keys = ["block_id", "block_position", "test_window_index", "attack_label"]
    if frame.duplicated(keys).any():
        raise AssertionError("latent key is not unique")
    return frame, latents


def frozen_l4_oracle(
    base: np.ndarray, *, child_seed: int, attack_label: int, target_id: int
) -> np.ndarray:
    """Independent byte-level transcription of frozen E8 ``inject_l4``."""
    result = np.asarray(base, dtype=np.float32).copy()
    rng = np.random.default_rng(child_seed)
    start = int(rng.integers(0, 33))
    positions = np.arange(start, start + 96, 3)
    result[positions, 0] = target_id
    result[positions, 1] = 8
    payload = result[positions, 2:10].copy()
    ramp = np.linspace(0, 255, len(positions), dtype=np.float32)
    if attack_label == 3:
        payload[:, 6] = ramp
        payload[:, 7] = np.clip(
            255 - ramp + rng.integers(-10, 11, size=len(positions)), 0, 255
        )
    elif attack_label == 4:
        wave = (
            127.5
            + 127.5 * np.sin(np.linspace(0, 3 * np.pi, len(positions)))
        ).astype(np.float32)
        payload[:, 4] = (
            np.arange(len(positions), dtype=np.float32) * 37
        ) % 256
        payload[:, 6] = wave
        payload[:, 7] = np.clip(
            ramp + rng.integers(-12, 13, size=len(positions)), 0, 255
        )
    else:
        raise ValueError("oracle supports Gear and RPM only")
    result[positions, 2:10] = payload
    result[positions, 10] = np.maximum(result[positions, 10], 1e-5)
    return np.ascontiguousarray(result, dtype=np.float32)


def conditional_rule_anchor_d0_oracle(
    base: np.ndarray,
    *,
    block_number: int,
    test_window_index: int,
    attack_label: int,
    target_id: int,
) -> np.ndarray:
    """Independently construct the registered conditional Rule-side anchor.

    This oracle intentionally does not consume ``FactorLatent`` masks, roles,
    drivers, or jitters.  It replays both child-seed streams and the frozen Rule
    RNG order directly from the registered base identity, thereby making cell
    ``000`` a genuine byte-level regression of the D0 formula rather than a
    self-comparison against values already stored in the latent object.
    """
    source = np.asarray(base)
    if source.shape != WINDOW_SHAPE or source.dtype != np.float32:
        raise ValueError("Rule-anchor oracle base must be float32 (128,11)")
    if attack_label not in ATTACK_SPECS:
        raise ValueError("Rule-anchor oracle supports Gear and RPM only")

    l4_seed = derive_child_seed(
        block_number, test_window_index, attack_label, 1
    )
    l4_rng = np.random.default_rng(l4_seed)
    start = int(l4_rng.integers(0, 33))
    positions = start + 2 * np.arange(32)

    rule_seed = derive_child_seed(
        block_number, test_window_index, attack_label, 2
    )
    rule_rng = np.random.default_rng(rule_seed)
    result = source.copy()
    result[positions, 0] = target_id
    result[positions, 1] = 8
    if attack_label == 3:
        levels = rule_rng.choice([0, 1, 2, 3, 4, 5], size=32)
        role0 = np.clip(
            levels * 40 + rule_rng.integers(0, 16, size=32), 0, 255
        )
        role1 = np.clip(
            255 - role0 + rule_rng.integers(-8, 9, size=32), 0, 255
        )
        result[positions, 2] = role0
        result[positions, 3] = role1
    else:
        rpm = rule_rng.integers(0, 8000, size=32)
        result[positions, 2] = (rpm // 32) % 256
        result[positions, 3] = (rpm // 4) % 256
        result[positions, 4] = np.clip(
            source[positions, 4]
            + rule_rng.integers(-50, 51, size=32),
            0,
            255,
        )
    result[positions, 10] = np.maximum(result[positions, 10], 1e-5)
    return np.ascontiguousarray(result, dtype=np.float32)


def _assert_changes_only(
    left: np.ndarray, right: np.ndarray, allowed: np.ndarray, label: str
) -> None:
    if allowed.shape != WINDOW_SHAPE or allowed.dtype != bool:
        raise TypeError("allowed-change mask must be bool (128,11)")
    if not np.array_equal(left[~allowed], right[~allowed]):
        first = np.argwhere((left != right) & ~allowed)[0].tolist()
        raise AssertionError(f"{label} changed a prohibited field at {first}")


def _validate_window_representation(window: np.ndarray) -> None:
    if window.shape != WINDOW_SHAPE or window.dtype != np.float32:
        raise AssertionError("transformed window representation mismatch")
    if not np.isfinite(window).all():
        raise AssertionError("transformed window contains non-finite values")
    ids, dlc, payload, delta_t = (
        window[:, 0], window[:, 1], window[:, 2:10], window[:, 10]
    )
    if (
        (ids < 0).any() or (ids > 0x7FF).any()
        or not np.array_equal(ids, np.rint(ids))
    ):
        raise AssertionError("invalid CAN ID representation")
    if (
        (dlc < 0).any() or (dlc > 8).any()
        or not np.array_equal(dlc, np.rint(dlc))
    ):
        raise AssertionError("invalid DLC representation")
    if (payload < 0).any() or (payload > 255).any():
        raise AssertionError("payload outside registered feature range")
    if (delta_t < 0).any():
        raise AssertionError("negative delta_t")
    dlc_int = dlc.astype(np.int8)
    for byte_index in range(8):
        padding = dlc_int <= byte_index
        if np.any(payload[padding, byte_index] != 0):
            raise AssertionError("nonzero DLC-conditioned padding")


def validate_factor_relations(
    base: np.ndarray, latent: FactorLatent
) -> dict[tuple[str, int, int, int], np.ndarray]:
    """Exhaustively validate one base/attack's 16 generated transforms."""
    generated = {
        (id_stratum, p, s, d): apply_factorial_transform(
            base,
            latent,
            attack_label=latent.attack_label,
            id_stratum=id_stratum,
            p=p,
            s=s,
            d=d,
        )
        for id_stratum in ID_STRATA
        for p, s, d in CELLS
    }
    spec = ATTACK_SPECS[latent.attack_label]
    for (id_stratum, p, s, d), window in generated.items():
        _validate_window_representation(window)
        positions = latent.s0_positions if s == 0 else latent.s1_positions
        columns = spec["p0_columns"] if p == 0 else spec["p1_columns"]
        roles = latent.d0_roles if d == 0 else latent.d1_roles
        outside = np.ones(128, dtype=bool)
        outside[positions] = False
        if not np.array_equal(window[outside], base[outside]):
            raise AssertionError("transform changed a non-injected frame")
        if np.any(window[positions, 0] != spec[f"{id_stratum}_id"]):
            raise AssertionError("target CAN ID mismatch")
        if np.any(window[positions, 1] != 8):
            raise AssertionError("injected DLC mismatch")
        if np.any(window[positions, 10] < 1e-5):
            raise AssertionError("injected timing floor mismatch")
        for column, role in zip(columns, roles, strict=True):
            if not np.array_equal(window[positions, column], role):
                raise AssertionError("rank-ordered payload role mismatch")

    for p, s, d in CELLS:
        left = generated[("canonical", p, s, d)]
        right = generated[("shifted", p, s, d)]
        allowed = np.zeros(WINDOW_SHAPE, dtype=bool)
        positions = latent.s0_positions if s == 0 else latent.s1_positions
        allowed[positions, 0] = True
        _assert_changes_only(left, right, allowed, "ID stratum")

    for id_stratum in ID_STRATA:
        for s in (0, 1):
            positions = latent.s0_positions if s == 0 else latent.s1_positions
            for d in (0, 1):
                allowed = np.zeros(WINDOW_SHAPE, dtype=bool)
                union_columns = set(spec["p0_columns"]) | set(spec["p1_columns"])
                for column in union_columns:
                    allowed[positions, column] = True
                _assert_changes_only(
                    generated[(id_stratum, 0, s, d)],
                    generated[(id_stratum, 1, s, d)],
                    allowed,
                    "P",
                )
        for p in (0, 1):
            columns = spec["p0_columns"] if p == 0 else spec["p1_columns"]
            union_positions = np.union1d(
                latent.s0_positions, latent.s1_positions
            )
            for d in (0, 1):
                allowed = np.zeros(WINDOW_SHAPE, dtype=bool)
                for column in (0, 1, 10, *columns):
                    allowed[union_positions, column] = True
                _assert_changes_only(
                    generated[(id_stratum, p, 0, d)],
                    generated[(id_stratum, p, 1, d)],
                    allowed,
                    "S",
                )
            for s in (0, 1):
                positions = latent.s0_positions if s == 0 else latent.s1_positions
                allowed = np.zeros(WINDOW_SHAPE, dtype=bool)
                for column in columns:
                    allowed[positions, column] = True
                low = generated[(id_stratum, p, s, 0)]
                high = generated[(id_stratum, p, s, 1)]
                _assert_changes_only(low, high, allowed, "D")
                if not np.array_equal(low[:, 10], high[:, 10]):
                    raise AssertionError("payload dynamics changed delta_t")

    for id_stratum in ID_STRATA:
        observed_anchor = generated[(id_stratum, 0, 0, 0)]
        expected_anchor = conditional_rule_anchor_d0_oracle(
            base,
            block_number=latent.block_number,
            test_window_index=latent.test_window_index,
            attack_label=latent.attack_label,
            target_id=int(spec[f"{id_stratum}_id"]),
        )
        if not np.array_equal(observed_anchor, expected_anchor):
            raise AssertionError(
                "cell 000 is not byte-exact to conditional Rule D0 oracle"
            )

        observed = generated[(id_stratum, 1, 1, 1)]
        expected = frozen_l4_oracle(
            base,
            child_seed=latent.l4_child_seed,
            attack_label=latent.attack_label,
            target_id=int(spec[f"{id_stratum}_id"]),
        )
        if not np.array_equal(observed, expected):
            raise AssertionError("cell 111 is not byte-exact to frozen L4 oracle")
    return generated


def _transform_pass(
    base_manifest: pd.DataFrame,
    bases: np.ndarray,
    latents: Sequence[FactorLatent],
    *,
    validate_relations: bool,
) -> tuple[str, str, int, dict[tuple[Any, ...], int]]:
    latent_lookup = {
        (
            latent.block_id,
            latent.block_position,
            latent.test_window_index,
            latent.attack_label,
        ): latent
        for latent in latents
    }
    transform_digest = hashlib.sha256()
    scenario_digest = hashlib.sha256()
    previous_key: tuple[Any, ...] | None = None
    count = 0
    group_counts: dict[tuple[Any, ...], int] = {}
    for row, base in zip(
        base_manifest.itertuples(index=False), bases, strict=True
    ):
        for attack_label in sorted(ATTACK_SPECS):
            latent = latent_lookup[
                (
                    str(row.block_id),
                    int(row.block_position),
                    int(row.test_window_index),
                    attack_label,
                )
            ]
            generated = (
                validate_factor_relations(base, latent)
                if validate_relations
                else None
            )
            for id_stratum in ID_STRATA:
                for p, s, d in CELLS:
                    key = transformation_key(latent, id_stratum, p, s, d)
                    if previous_key is not None and key <= previous_key:
                        raise AssertionError(
                            "scenario keys are not strictly lexicographically sorted"
                        )
                    previous_key = key
                    window = (
                        generated[(id_stratum, p, s, d)]
                        if generated is not None
                        else apply_factorial_transform(
                            base,
                            latent,
                            attack_label=attack_label,
                            id_stratum=id_stratum,
                            p=p,
                            s=s,
                            d=d,
                        )
                    )
                    key_bytes = transformation_key_bytes(key)
                    scenario_digest.update(key_bytes)
                    update_transform_digest(transform_digest, key, window)
                    count += 1
                    group = (
                        latent.block_id,
                        ATTACK_SPECS[attack_label]["attack"],
                        id_stratum,
                        p,
                        s,
                        d,
                    )
                    group_counts[group] = group_counts.get(group, 0) + 1
    return (
        transform_digest.hexdigest(),
        scenario_digest.hexdigest(),
        count,
        group_counts,
    )


def manipulation_audit(
    base_manifest: pd.DataFrame,
    bases: np.ndarray,
    latents: Sequence[FactorLatent],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    first = _transform_pass(
        base_manifest, bases, latents, validate_relations=True
    )
    second = _transform_pass(
        base_manifest, bases, latents, validate_relations=False
    )
    if first[:3] != second[:3] or first[3] != second[3]:
        raise AssertionError("repeated transform regeneration is not deterministic")
    transform_sha, scenario_sha, count, group_counts = first
    if count != 192_000 or len(group_counts) != 96:
        raise AssertionError(
            f"factorial completeness mismatch: instances={count}, "
            f"groups={len(group_counts)}"
        )
    rows = []
    for group, windows in sorted(group_counts.items()):
        block_id, attack, id_stratum, p, s, d = group
        if windows != BASES_PER_BLOCK:
            raise AssertionError(f"incomplete manipulation group: {group}")
        rows.append(
            {
                "scope": "block_cell_stratum",
                "block_id": block_id,
                "attack": attack,
                "id_stratum": id_stratum,
                "P": p,
                "S": s,
                "D": d,
                "cell": cell_name(p, s, d),
                "base_windows": windows,
                "injected_frames_per_window": 32,
                "injected_frames": windows * 32,
                "target_id_dlc_timing_pass": True,
                "feature_range_finite_padding_pass": True,
                "role_formula_and_order_pass": True,
                "conditional_rule_anchor_000_oracle_pass": True,
                "one_factor_pairing_pass": True,
                "canonical_shifted_pairing_pass": True,
                "l4_oracle_111_pass": True,
                "status": "PASS",
            }
        )
    gates = {
        "base_rows": int(len(base_manifest)),
        "latent_rows": int(len(latents)),
        "scenario_instances": count,
        "unique_scenario_keys": count,
        "groups": len(group_counts),
        "instances_per_group": BASES_PER_BLOCK,
        "injected_frames_per_transform": 32,
        "factor_cells": [cell_name(*cell) for cell in CELLS],
        "id_strata": list(ID_STRATA),
        "attack_labels": sorted(ATTACK_SPECS),
        "transform_serialization_version": TRANSFORM_SERIALIZATION_VERSION,
        "transformation_sha256": transform_sha,
        "scenario_key_sha256": scenario_sha,
        "repeated_regeneration_match": True,
        "conditional_rule_anchor_000_oracle_pass": True,
        "all_manipulation_gates_pass": True,
    }
    return pd.DataFrame(rows), gates


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish_bundle_no_clobber(
    artifacts: Sequence[tuple[Path, bytes]],
) -> None:
    """Stage a multi-file bundle and link each target without replacement.

    The final run/prepare record must be passed last and therefore acts as the
    completed-bundle marker.  If publication raises synchronously, only links
    created by this invocation are rolled back and the isolated staged bytes
    are retained for diagnosis.
    """
    targets = [Path(path) for path, _ in artifacts]
    if len(targets) != len(set(targets)):
        raise ValueError("bundle contains duplicate target paths")
    for target in targets:
        absolute = Path(os.path.abspath(target))
        if not absolute.is_relative_to(JOURNAL_ROOT.absolute()):
            raise ValueError(f"E14 output escapes journal/: {target}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite E14 output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    EXP.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".e14_staged_", dir=EXP))
    staged: list[tuple[Path, Path]] = []
    linked: list[tuple[Path, Path]] = []
    success = False
    try:
        for ordinal, (target, payload) in enumerate(artifacts):
            staged_path = stage / f"{ordinal:02d}-{target.name}"
            with staged_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((staged_path, target))
        for _, target in staged:
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"E14 output appeared during publication: {target}"
                )
        for staged_path, target in staged:
            os.link(staged_path, target)
            linked.append((staged_path, target))
        success = True
    except BaseException:
        for staged_path, target in reversed(linked):
            try:
                target_status = target.stat(follow_symlinks=False)
                staged_status = staged_path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (
                target_status.st_dev == staged_status.st_dev
                and target_status.st_ino == staged_status.st_ino
            ):
                target.unlink()
        raise
    finally:
        if success:
            shutil.rmtree(stage)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON input: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _assert_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing regular frozen input: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"frozen input hash mismatch for {path}: "
            f"expected={expected}, observed={observed}"
        )
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def _matched_real_record(path: Path) -> dict[str, Any]:
    """Validate one byte-frozen production training record."""
    record = _assert_hash(path, MATCHED_REAL_RECORD_HASHES[path])
    expected_bytes = MATCHED_REAL_RECORD_BYTES[path]
    if record["bytes"] != expected_bytes:
        raise ValueError(
            f"matched-real record byte count mismatch for {path}: "
            f"expected={expected_bytes}, observed={record['bytes']}"
        )
    return record


def parse_matched_real_transcript(payload: bytes) -> dict[str, Any]:
    """Strictly parse and validate the frozen wrapper transcript bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("matched-real transcript payload must be bytes")
    magic = b"E14 matched-real frozen trainer transcript v1\n"
    if not payload.startswith(magic):
        raise ValueError("matched-real transcript magic header mismatch")
    cursor = len(magic)

    def consume_line(field: bytes) -> bytes:
        nonlocal cursor
        end = payload.find(b"\n", cursor)
        if end < 0:
            raise ValueError(
                "matched-real transcript has an unterminated metadata line"
            )
        line = payload[cursor:end]
        cursor = end + 1
        prefix = field + b"="
        if not line.startswith(prefix):
            raise ValueError(
                "matched-real transcript metadata order/name mismatch"
            )
        return line[len(prefix):]

    if consume_line(b"returncode") != b"0":
        raise ValueError("matched-real transcript child return code is not zero")

    def canonical_size(field: bytes) -> int:
        raw = consume_line(field)
        if (
            not raw
            or not raw.isdigit()
            or (len(raw) > 1 and raw.startswith(b"0"))
        ):
            raise ValueError(
                f"matched-real transcript has invalid {field.decode('ascii')}"
            )
        return int(raw)

    stdout_bytes = canonical_size(b"stdout_bytes")
    stderr_bytes = canonical_size(b"stderr_bytes")
    stdout_marker = b"\n--- stdout ---\n"
    if payload[cursor:cursor + len(stdout_marker)] != stdout_marker:
        raise ValueError("matched-real transcript stdout delimiter mismatch")
    cursor += len(stdout_marker)
    stdout_end = cursor + stdout_bytes
    if stdout_end > len(payload):
        raise ValueError("matched-real transcript stdout is truncated")
    stdout = payload[cursor:stdout_end]
    cursor = stdout_end
    stderr_marker = b"\n--- stderr ---\n"
    if payload[cursor:cursor + len(stderr_marker)] != stderr_marker:
        raise ValueError("matched-real transcript stderr delimiter mismatch")
    cursor += len(stderr_marker)
    stderr_end = cursor + stderr_bytes
    if stderr_end != len(payload):
        raise ValueError(
            "matched-real transcript stderr length/trailing bytes mismatch"
        )
    stderr = payload[cursor:stderr_end]
    skip_lines = [
        line
        for line in stdout.decode("utf-8", errors="replace").splitlines()
        if line.startswith("skip existing ")
    ]
    if skip_lines:
        raise ValueError(
            "matched-real transcript reports pre-existing target skips: "
            + _compact_json(skip_lines)
        )
    return {
        "format": "e14.matched_real_training_child_output.v1",
        "returncode": 0,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "existing_target_skip_lines": skip_lines,
    }


# Private alias kept for focused regression tests.
_parse_matched_real_transcript = parse_matched_real_transcript


def _validate_matched_real_transcript(
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the actual transcript bytes and their unchanged run cross-link."""
    file_record = _matched_real_record(MATCHED_REAL_TRANSCRIPT)
    parsed = parse_matched_real_transcript(MATCHED_REAL_TRANSCRIPT.read_bytes())
    declared = run.get("child_transcript")
    if not isinstance(declared, Mapping):
        raise ValueError("matched-real run lacks a child transcript record")
    expected_declared = {
        "path": file_record["path"],
        "bytes": file_record["bytes"],
        "sha256": file_record["sha256"],
        "format": parsed["format"],
        "stdout_bytes": parsed["stdout_bytes"],
        "stderr_bytes": parsed["stderr_bytes"],
        "streams_preserved_separately": True,
        "existing_target_skip_lines": parsed["existing_target_skip_lines"],
    }
    if dict(declared) != expected_declared:
        raise ValueError("matched-real run/transcript cross-link mismatch")
    validation = run.get("validation", {})
    if (
        validation.get("child_output_persisted") is not True
        or validation.get("child_reported_existing_target_skips") is not False
    ):
        raise ValueError("matched-real run transcript validation gates failed")
    return {
        **file_record,
        **parsed,
        "streams_preserved_separately": True,
    }


validate_matched_real_transcript = _validate_matched_real_transcript


def _git_bytes(args: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Git provenance command failed: {args}") from exc
    return completed.stdout


def _git_output(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Git provenance command failed: {args}") from exc
    return completed.stdout


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Git ancestry command failed") from exc
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            "Git ancestry command failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return completed.returncode == 0


def _git_name_status(
    from_commit: str,
    to_commit: str,
) -> frozenset[tuple[str, str]]:
    raw = _git_bytes(
        [
            "diff",
            "--name-status",
            "--no-renames",
            "-z",
            from_commit,
            to_commit,
            "--",
        ]
    )
    fields = raw.split(b"\0")
    if fields[-1:] == [b""]:
        fields.pop()
    if len(fields) % 2:
        raise RuntimeError("malformed NUL-delimited Git name-status output")
    records: set[tuple[str, str]] = set()
    for index in range(0, len(fields), 2):
        status = fields[index].decode("ascii")
        path = os.fsdecode(fields[index + 1])
        if not status or (status, path) in records:
            raise RuntimeError("invalid or duplicate Git name-status record")
        records.add((status, path))
    return frozenset(records)


def _validate_source_transition(prepare_commit: str) -> dict[str, Any]:
    """Verify the exact two one-commit legs authorized by amendment 2."""
    if not isinstance(prepare_commit, str) or len(prepare_commit) != 40:
        raise ValueError("prepare source commit must be a full 40-hex commit ID")
    try:
        int(prepare_commit, 16)
    except ValueError as exc:
        raise ValueError(
            "prepare source commit must be a full 40-hex commit ID"
        ) from exc
    resolved = _git_output(
        ["rev-parse", f"{prepare_commit}^{{commit}}"]
    ).strip()
    if resolved != prepare_commit:
        raise ValueError("prepare source commit did not resolve exactly")

    ancestry = {
        "training_to_amendment": _git_is_ancestor(
            TRAINING_SOURCE_COMMIT, AMENDMENT_SOURCE_COMMIT
        ),
        "amendment_to_prepare": _git_is_ancestor(
            AMENDMENT_SOURCE_COMMIT, prepare_commit
        ),
    }
    if not all(ancestry.values()):
        raise ValueError("E14 implementation transition ancestry mismatch")

    legs = (
        (
            "training_to_amendment",
            TRAINING_SOURCE_COMMIT,
            AMENDMENT_SOURCE_COMMIT,
            AMENDMENT_ONLY_PATHS,
        ),
        (
            "amendment_to_prepare",
            AMENDMENT_SOURCE_COMMIT,
            prepare_commit,
            IMPLEMENTATION_CORRECTION_PATHS,
        ),
    )
    records: dict[str, Any] = {}
    for name, from_commit, to_commit, expected_changes in legs:
        count = int(
            _git_output(
                ["rev-list", "--count", f"{from_commit}..{to_commit}"]
            ).strip()
        )
        changes = _git_name_status(from_commit, to_commit)
        if count != 1:
            raise ValueError(
                f"E14 {name} transition must contain exactly one commit"
            )
        if changes != expected_changes:
            raise ValueError(
                f"E14 {name} changed an unauthorized path/status: "
                f"expected={sorted(expected_changes)}, observed={sorted(changes)}"
            )
        records[name] = {
            "from_commit": from_commit,
            "to_commit": to_commit,
            "commit_count": count,
            "exactly_one_commit": True,
            "changes": [
                {"status": status, "path": path}
                for status, path in sorted(changes)
            ],
            "expected_changes_match": True,
        }

    amendment_bytes = _git_bytes(
        [
            "show",
            f"{AMENDMENT_SOURCE_COMMIT}:"
            f"{TRAINING_TO_AMENDMENT_PATHS[0]}",
        ]
    )
    if hashlib.sha256(amendment_bytes).hexdigest() != POST_TRAINING_AMENDMENT_SHA256:
        raise ValueError("committed transcript amendment hash mismatch")
    return {
        "schema_version": "e14.implementation_transition.v1",
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "amendment_source_commit": AMENDMENT_SOURCE_COMMIT,
        "prepare_source_commit": prepare_commit,
        "ancestry": ancestry,
        **records,
    }


_implementation_transition_for_prepare = _validate_source_transition


def _validate_implementation_transition(
    registered: Mapping[str, Any],
    *,
    require_prepare_head: bool,
) -> None:
    if not isinstance(registered, Mapping):
        raise ValueError("prepare implementation transition is missing")
    prepare_commit = registered.get("prepare_source_commit")
    if not isinstance(prepare_commit, str):
        raise ValueError("prepare implementation transition commit is missing")
    expected = _validate_source_transition(prepare_commit)
    if dict(registered) != expected:
        raise ValueError("prepare implementation transition record mismatch")
    head = _git_output(["rev-parse", "HEAD"]).strip()
    if require_prepare_head:
        if head != prepare_commit:
            raise ValueError("Stage A source HEAD is not its prepare commit")
    elif not _git_is_ancestor(prepare_commit, head):
        raise ValueError("prepare implementation commit is not an ancestor of HEAD")


def implementation_source_records() -> dict[str, dict[str, Any]]:
    """Require each registered E14 implementation file to equal its HEAD blob."""
    records: dict[str, dict[str, Any]] = {}
    for role, path in IMPLEMENTATION_SOURCES.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"missing regular E14 implementation source ({role}): {path}"
            )
        relative = portable_path(path)
        try:
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                timeout=30,
            )
            head = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"E14 implementation source is not available from HEAD: {relative}"
            ) from exc
        working_bytes = path.read_bytes()
        if working_bytes != head:
            raise RuntimeError(
                f"E14 implementation source differs from HEAD: {relative}"
            )
        records[role] = {
            "path": relative,
            "bytes": len(working_bytes),
            "sha256": hashlib.sha256(working_bytes).hexdigest(),
            "tracked": True,
            "head_blob_matches": True,
        }
    return records


def _verify_implementation_source_records(
    registered: Mapping[str, Any],
) -> None:
    observed = implementation_source_records()
    if set(registered) != set(IMPLEMENTATION_SOURCES):
        raise ValueError("prepare record implementation-source roles mismatch")
    for role, record in registered.items():
        current = observed[role]
        for field in ("path", "bytes", "sha256"):
            if record.get(field) != current[field]:
                raise ValueError(
                    f"E14 implementation source changed after preparation: "
                    f"{role}/{field}"
                )


def _allowed_untracked_paths_for_prepare() -> set[str]:
    return {
        portable_path(MATCHED_REAL_PREFLIGHT),
        portable_path(MATCHED_REAL_RUN),
        portable_path(MATCHED_REAL_TRANSCRIPT),
        *{
            portable_path(
                MODELS / f"cnn_real_only_matchedsteps_e14_v1_seed{seed}.pt"
            )
            for seed in SEEDS
        },
        *{
            portable_path(
                LOGS / f"train_cnn_real_only_matchedsteps_e14_v1_seed{seed}.log"
            )
            for seed in SEEDS
        },
    }


def source_provenance_for_prepare() -> dict[str, Any]:
    """Require clean tracked source while allowing only new training outputs."""
    head = _git_output(["rev-parse", "HEAD"]).strip()
    existing_outputs = sorted(
        portable_path(path)
        for path in (*PREPARE_PATHS.values(), *SCORE_PATHS.values())
        if path.exists() or path.is_symlink()
    )
    if existing_outputs:
        raise RuntimeError(
            "canonical E14 preparation requires all Stage-A/Stage-B outputs "
            "to be absent: " + _compact_json(existing_outputs)
        )
    tracked = sorted(
        set(
            filter(
                None,
                (
                    _git_output(["diff", "--name-only"])
                    + _git_output(["diff", "--cached", "--name-only"])
                ).splitlines(),
            )
        )
    )
    if tracked:
        raise RuntimeError(
            "canonical E14 preparation requires clean tracked state: "
            + _compact_json(tracked)
        )
    untracked = sorted(
        filter(
            None,
            _git_output(["ls-files", "--others", "--exclude-standard"]).splitlines(),
        )
    )
    allowed = _allowed_untracked_paths_for_prepare()
    blocking = sorted(set(untracked) - allowed)
    if blocking:
        raise RuntimeError(
            "canonical E14 preparation found undeclared untracked files: "
            + _compact_json(blocking)
        )
    if _git_output(["status", "--short", "--", "wisa"]).strip():
        raise RuntimeError("frozen wisa/ archive is not clean")
    implementation_transition = _validate_source_transition(head)
    return {
        "source_commit": head,
        "source_tracked_state_clean": True,
        "canonical_stage_a_stage_b_outputs_absent": True,
        "allowed_untracked_training_artifacts": untracked,
        "implementation_transition": implementation_transition,
        "policy": (
            "clean tracked source; only preregistered no-clobber matched-real "
            "training artifacts may be untracked"
        ),
    }


def source_provenance_for_score() -> dict[str, Any]:
    status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"]
    ).strip()
    if status:
        raise RuntimeError(
            "canonical E14 scoring requires a fully clean source/worktree "
            "after preparation is committed"
        )
    if _git_output(["status", "--short", "--", "wisa"]).strip():
        raise RuntimeError("frozen wisa/ archive is not clean")
    return {
        "source_commit": _git_output(["rev-parse", "HEAD"]).strip(),
        "source_worktree_clean": True,
        "wisa_clean": True,
    }


def checkpoint_path(arm: str, seed: int) -> Path:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("undeclared E14 arm or seed")
    names = {
        "real_ms": f"cnn_real_only_matchedsteps_e14_v1_seed{seed}.pt",
        "rule_ms": f"cnn_rule_0p30_matchedsteps_seed{seed}.pt",
        "placebo_ms": f"cnn_placebo_0p30_matchedsteps_seed{seed}.pt",
        "real_std": f"cnn_real_only_seed{seed}.pt",
        "rule_std": f"cnn_rule_0p30_seed{seed}.pt",
        "placebo_std": f"cnn_placebo_0p30_seed{seed}.pt",
    }
    return MODELS / names[arm]


def training_log_path(arm: str, seed: int) -> Path:
    names = {
        "real_ms": f"train_cnn_real_only_matchedsteps_e14_v1_seed{seed}.log",
        "rule_ms": f"train_cnn_rule_0p30_matchedsteps_seed{seed}.log",
        "placebo_ms": f"train_cnn_placebo_0p30_matchedsteps_seed{seed}.log",
        "real_std": f"train_cnn_real_only_seed{seed}.log",
        "rule_std": f"train_cnn_rule_0p30_seed{seed}.log",
        "placebo_std": f"train_cnn_placebo_0p30_seed{seed}.log",
    }
    if arm not in names or seed not in SEEDS:
        raise ValueError("undeclared E14 arm or seed")
    return LOGS / names[arm]


def _validate_matched_real_records() -> dict[int, dict[str, Any]]:
    _matched_real_record(MATCHED_REAL_PREFLIGHT)
    _matched_real_record(MATCHED_REAL_RUN)
    preflight = _read_json(MATCHED_REAL_PREFLIGHT)
    run = _read_json(MATCHED_REAL_RUN)
    if (
        preflight.get("schema_version")
        != "e14.matched_real_training_preflight.v1"
        or preflight.get("status") != "T-PASS-PREFLIGHT"
        or preflight.get("source_provenance", {}).get("head_commit")
        != TRAINING_SOURCE_COMMIT
    ):
        raise ValueError("matched-real preflight schema/status mismatch")
    if (
        run.get("schema_version") != "e14.matched_real_training_run.v1"
        or run.get("status") != "T-PASS-TRAINING"
        or run.get("child_exit_status") != 0
        or run.get("environment_matches_preflight") is not True
        or run.get("source_commit") != TRAINING_SOURCE_COMMIT
    ):
        raise ValueError("matched-real run schema/status mismatch")
    link = run.get("preflight", {})
    if (
        link.get("path") != portable_path(MATCHED_REAL_PREFLIGHT)
        or link.get("sha256") != sha256_file(MATCHED_REAL_PREFLIGHT)
        or link.get("bytes") != MATCHED_REAL_PREFLIGHT.stat().st_size
        or link.get("schema_version") != preflight["schema_version"]
    ):
        raise ValueError("matched-real run/preflight cross-link mismatch")
    validation = run.get("validation", {})
    required_true = (
        "all_logs_valid",
        "all_checkpoints_weights_only_load",
        "all_checkpoint_architectures_compatible",
        "all_checkpoint_tensors_finite",
        "all_artifact_hashes_recorded",
        "preflight_cross_linked",
        "no_e14_factorial_inference",
    )
    if (
        validation.get("validated_seed_count") != 5
        or validation.get("seeds") != list(SEEDS)
        or any(validation.get(name) is not True for name in required_true)
    ):
        raise ValueError("matched-real run validation gates did not all pass")
    _validate_matched_real_transcript(run)
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 5:
        raise ValueError("matched-real run must declare five artifacts")
    by_seed: dict[int, dict[str, Any]] = {}
    for artifact in artifacts:
        seed = int(artifact["seed"])
        if seed in by_seed or seed not in SEEDS:
            raise ValueError("matched-real artifact seed mismatch")
        checkpoint = artifact["checkpoint"]
        log = artifact["log"]
        expected_checkpoint = checkpoint_path("real_ms", seed)
        expected_log = training_log_path("real_ms", seed)
        for record, path in (
            (checkpoint, expected_checkpoint),
            (log, expected_log),
        ):
            if (
                record.get("path") != portable_path(path)
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != sha256_file(path)
            ):
                raise ValueError(
                    f"matched-real artifact record mismatch for seed {seed}"
                )
        checkpoint_validation = checkpoint.get("validation", {})
        if (
            checkpoint_validation.get("weights_only_load") is not True
            or checkpoint_validation.get("state_dict_keys") != 23
            or checkpoint_validation.get("expected_state_dict_keys") != 23
            or checkpoint_validation.get("architecture_compatible") is not True
            or checkpoint_validation.get("all_tensors_finite") is not True
        ):
            raise ValueError(
                f"matched-real checkpoint validation failed for seed {seed}"
            )
        by_seed[seed] = artifact
    if set(by_seed) != set(SEEDS):
        raise ValueError("matched-real artifact seed set mismatch")
    return by_seed


def _matched_real_training_record_bundle() -> dict[str, dict[str, Any]]:
    """Return the exact three-record training evidence bundle for Stage A."""
    _validate_matched_real_records()
    run = _read_json(MATCHED_REAL_RUN)
    records = {
        "preflight": _matched_real_record(MATCHED_REAL_PREFLIGHT),
        "run": _matched_real_record(MATCHED_REAL_RUN),
        "child_transcript": _validate_matched_real_transcript(run),
    }
    if set(records) != MATCHED_REAL_TRAINING_RECORD_KEYS:
        raise AssertionError("internal matched-real evidence key mismatch")
    return records


def _validate_training_log(arm: str, seed: int, path: Path) -> dict[str, Any]:
    log = _read_json(path)
    setting_by_arm = {
        "real_ms": "real_only",
        "rule_ms": "rule_0p30",
        "placebo_ms": "placebo_0p30",
        "real_std": "real_only",
        "rule_std": "rule_0p30",
        "placebo_std": "placebo_0p30",
    }
    if (
        log.get("family") != "cnn"
        or int(log.get("seed", -1)) != seed
        or log.get("setting") != setting_by_arm[arm]
    ):
        raise ValueError(f"training-log identity mismatch: {path}")
    score = log.get("best_val_macro_f1")
    if score is None or not math.isfinite(float(score)):
        raise ValueError(f"training log lacks a finite selected score: {path}")
    best_score = float(score)
    if not 0.0 <= best_score <= 1.0:
        raise ValueError(f"training log selected score is out of range: {path}")

    def positive_int(name: str) -> int:
        value = log.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training log has invalid {name}: {path}")
        return int(value)

    train_windows = positive_int("train_windows")
    epochs_run = positive_int("epochs_run")
    selected_epoch = positive_int("selected_epoch")
    history = log.get("history")
    if not isinstance(history, list) or len(history) != epochs_run:
        raise ValueError(f"training-log history/epoch count mismatch: {path}")
    if selected_epoch > len(history):
        raise ValueError(f"training-log selected epoch is out of range: {path}")
    history_scores: list[float] = []
    for position, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"training-log history row is not an object: {path}")
        if int(row.get("epoch", -1)) != position:
            raise ValueError(f"training-log history epoch ordering mismatch: {path}")
        row_score = row.get("val_macro_f1")
        if row_score is None or not math.isfinite(float(row_score)):
            raise ValueError(f"training-log history score is non-finite: {path}")
        converted_score = float(row_score)
        if not 0.0 <= converted_score <= 1.0:
            raise ValueError(f"training-log history score is out of range: {path}")
        history_scores.append(converted_score)
    expected_selected_epoch = (
        max(range(len(history_scores)), key=history_scores.__getitem__) + 1
    )
    if selected_epoch != expected_selected_epoch:
        raise ValueError(f"training-log selected checkpoint is inconsistent: {path}")
    if best_score != history_scores[selected_epoch - 1]:
        raise ValueError(f"training-log best score/history mismatch: {path}")

    is_matched = arm in {"real_ms", "rule_ms", "placebo_ms"}
    consistency_gates = {
        "identity_matches_arm_seed": True,
        "history_rows_equal_epochs_run": True,
        "history_epochs_are_sequential": True,
        "selected_epoch_is_first_maximum": True,
        "best_score_matches_selected_history_row": True,
    }
    if is_matched:
        expected_model_tag = (
            "matchedsteps_e14_v1" if arm == "real_ms" else "matchedsteps"
        )
        if (
            log.get("budget_mode") != "matched_steps"
            or log.get("model_tag") != expected_model_tag
            or positive_int("optimizer_steps") != 6_156
            or positive_int("validation_checkpoints") != 12
            or epochs_run != 12
        ):
            raise ValueError(f"matched-budget training-log mismatch: {path}")
        history_steps = []
        for position, row in enumerate(history, start=1):
            value = row.get("optimizer_step")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"matched training log lacks recorded history steps: {path}"
                )
            if value != position * 513:
                raise ValueError(
                    f"matched training-log validation cadence mismatch: {path}"
                )
            history_steps.append(int(value))
        total_optimizer_steps = positive_int("optimizer_steps")
        if history_steps[-1] != total_optimizer_steps:
            raise ValueError(
                f"matched training-log final history step mismatch: {path}"
            )
        selected_optimizer_step = history_steps[selected_epoch - 1]
        total_source = "recorded_optimizer_steps"
        selected_source = "recorded_selected_history_optimizer_step"
        derivation = (
            "matched_steps: total from optimizer_steps; selected from the "
            "recorded optimizer_step at selected validation checkpoint"
        )
        steps_per_epoch: int | None = None
        consistency_gates.update(
            {
                "matched_budget_mode_and_model_tag": True,
                "matched_history_steps_recorded": True,
                "matched_validation_cadence_513": True,
                "matched_final_history_step_equals_total": True,
            }
        )
    else:
        if log.get("budget_mode") not in (None, "", "legacy"):
            raise ValueError(f"legacy training log has unexpected budget mode: {path}")
        steps_per_epoch = math.ceil(train_windows / 512)
        derived_total = steps_per_epoch * epochs_run
        derived_selected = steps_per_epoch * selected_epoch

        recorded_total = log.get("optimizer_steps")
        if recorded_total is None:
            total_optimizer_steps = derived_total
            total_source = "derived_legacy_epochs"
        else:
            if (
                isinstance(recorded_total, bool)
                or not isinstance(recorded_total, int)
                or recorded_total != derived_total
            ):
                raise ValueError(
                    f"legacy training-log total optimizer steps mismatch: {path}"
                )
            total_optimizer_steps = int(recorded_total)
            total_source = "recorded_optimizer_steps_verified_by_legacy_derivation"

        history_step_presence = [
            row.get("optimizer_step") is not None for row in history
        ]
        if any(history_step_presence) and not all(history_step_presence):
            raise ValueError(
                f"legacy training log has partially recorded history steps: {path}"
            )
        if all(history_step_presence):
            history_steps = []
            for position, row in enumerate(history, start=1):
                value = row.get("optimizer_step")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value != position * steps_per_epoch
                ):
                    raise ValueError(
                        f"legacy training-log history step mismatch: {path}"
                    )
                history_steps.append(int(value))
            selected_optimizer_step = history_steps[selected_epoch - 1]
            if selected_optimizer_step != derived_selected:
                raise ValueError(
                    f"legacy selected optimizer step mismatch: {path}"
                )
            selected_source = (
                "recorded_selected_history_step_verified_by_legacy_derivation"
            )
        else:
            history_steps = []
            selected_optimizer_step = derived_selected
            selected_source = "derived_legacy_selected_epoch"
        derivation = (
            "legacy: steps_per_epoch=ceil(train_windows/512); "
            "total=steps_per_epoch*epochs_run; "
            "selected=steps_per_epoch*selected_epoch"
        )
        consistency_gates.update(
            {
                "legacy_steps_per_epoch_positive": steps_per_epoch > 0,
                "legacy_total_matches_epoch_derivation": (
                    total_optimizer_steps == derived_total
                ),
                "legacy_selected_matches_epoch_derivation": (
                    selected_optimizer_step == derived_selected
                ),
                "legacy_history_step_recording_is_all_or_none": True,
            }
        )

    if not all(consistency_gates.values()):
        raise ValueError(f"training-log optimizer-step consistency failed: {path}")
    return {
        "budget_mode": "matched_steps" if is_matched else "legacy",
        "train_windows": train_windows,
        "epochs_run": epochs_run,
        "selected_epoch": selected_epoch,
        "best_val_macro_f1": best_score,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "selected_optimizer_step": selected_optimizer_step,
        "total_optimizer_steps_source": total_source,
        "selected_optimizer_step_source": selected_source,
        "optimizer_step_derivation": derivation,
        "history_optimizer_steps": history_steps,
        "consistency_gates": consistency_gates,
    }


def validate_checkpoint_bundle_without_loading() -> list[dict[str, Any]]:
    """Validate paths, bytes, hashes, logs, and wrapper tensor attestations.

    This function intentionally contains no PyTorch import and no checkpoint
    deserialization.  The wrapper's weights-only tensor validation is
    cross-linked for new real-ms fits; Stage B independently deserializes and
    validates all 30 checkpoints immediately before scoring.
    """
    matched_real = _validate_matched_real_records()
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            checkpoint = checkpoint_path(arm, seed)
            log_path = training_log_path(arm, seed)
            if (
                not checkpoint.is_file()
                or checkpoint.is_symlink()
                or not log_path.is_file()
                or log_path.is_symlink()
            ):
                raise FileNotFoundError(
                    f"incomplete regular checkpoint/log bundle: {arm}/{seed}"
                )
            checkpoint_sha = sha256_file(checkpoint)
            if arm == "real_ms":
                expected_sha = matched_real[seed]["checkpoint"]["sha256"]
                validation_source = "matched_real_training_run_v1"
            else:
                expected_sha = EXISTING_CHECKPOINT_HASHES[(arm, seed)]
                validation_source = "frozen_prereg_hash_and_prior_training_bundle"
            if checkpoint_sha != expected_sha:
                raise ValueError(
                    f"checkpoint hash mismatch for {arm}/{seed}: "
                    f"{checkpoint_sha} != {expected_sha}"
                )
            log_validation = _validate_training_log(arm, seed, log_path)
            if arm == "real_ms":
                wrapper_artifact = matched_real[seed]
                if (
                    int(wrapper_artifact["selected_optimizer_step"])
                    != log_validation["selected_optimizer_step"]
                    or int(wrapper_artifact["optimizer_steps"])
                    != log_validation["total_optimizer_steps"]
                ):
                    raise ValueError(
                        f"matched-real wrapper/log optimizer-step mismatch: {seed}"
                    )
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": {
                        "path": portable_path(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                        "sha256": checkpoint_sha,
                    },
                    "training_log": {
                        "path": portable_path(log_path),
                        "bytes": log_path.stat().st_size,
                        "sha256": sha256_file(log_path),
                        "selected_validation_score": log_validation[
                            "best_val_macro_f1"
                        ],
                        "total_optimizer_steps": log_validation[
                            "total_optimizer_steps"
                        ],
                        "selected_optimizer_step": log_validation[
                            "selected_optimizer_step"
                        ],
                        "optimizer_step_validation": log_validation,
                    },
                    "total_optimizer_steps": log_validation[
                        "total_optimizer_steps"
                    ],
                    "selected_optimizer_step": log_validation[
                        "selected_optimizer_step"
                    ],
                    "optimizer_step_derivation": log_validation[
                        "optimizer_step_derivation"
                    ],
                    "prepare_checkpoint_validation": (
                        "byte hash and cross-linked tensor attestation only; "
                        "checkpoint not deserialized by prepare-only"
                    ),
                    "tensor_validation_source": validation_source,
                }
            )
    if len(records) != 30:
        raise AssertionError("E14 requires exactly 30 checkpoint bundles")
    return records


def fit_registered_standardizer() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    with np.load(TRAIN_WINDOWS, allow_pickle=False) as archive:
        if "x" not in archive.files:
            raise KeyError("train archive lacks x")
        x = archive["x"]
    mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    mean = _f4(mean)
    std = _f4(std)
    combined = hashlib.sha256()
    combined.update(memoryview(mean).cast("B"))
    combined.update(memoryview(std).cast("B"))
    hashes = {
        "mean": array_sha256(mean),
        "std": array_sha256(std),
        "mean_then_std": combined.hexdigest(),
    }
    if hashes != EXPECTED_STANDARDIZER_HASHES:
        raise ValueError(
            "registered standardizer hashes changed: "
            f"expected={EXPECTED_STANDARDIZER_HASHES}, observed={hashes}"
        )
    return mean, std, hashes


def environment_without_torch_import() -> dict[str, Any]:
    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch_distribution": version("torch"),
        "scikit_learn_distribution": version("scikit-learn"),
        "canonical_scoring_device": "cpu",
        "canonical_torch_threads": TORCH_THREADS,
        "canonical_batch_size": SCORING_BATCH_SIZE,
        "torch_imported_by_prepare_only": False,
    }


def _assert_prepare_torch_absent() -> None:
    if "torch" in sys.modules:
        raise RuntimeError(
            "E14 --prepare-only requires a fresh process with torch absent; "
            "checkpoint deserialization and inference remain prohibited"
        )


def _run_prepare_only_success() -> None:
    """Execute Stage A only; no code reachable here imports or loads PyTorch."""
    _assert_prepare_torch_absent()
    for path in PREPARE_PATHS.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite E14 prepare output: {path}")
    source = source_provenance_for_prepare()
    implementation_sources = implementation_source_records()
    frozen_inputs = {
        portable_path(path): _assert_hash(path, expected)
        for path, expected in EXPECTED_HASHES.items()
    }
    matched_real_training_records = _matched_real_training_record_bundle()
    checkpoint_bundles = validate_checkpoint_bundle_without_loading()
    _, _, standardizer_hashes = fit_registered_standardizer()

    test = _load_npz(TEST_WINDOWS)
    block_manifest = pd.read_csv(BLOCK_MANIFEST)
    base_manifest, bases = build_base_manifest(block_manifest, test)
    latent_manifest, latents = build_latent_manifest(base_manifest, bases)
    latent_sha = latent_semantic_digest(latents)
    _, regenerated_latents = build_latent_manifest(base_manifest, bases)
    regenerated_latent_sha = latent_semantic_digest(regenerated_latents)
    if latent_sha != regenerated_latent_sha:
        raise AssertionError("repeated latent regeneration digest mismatch")
    checks, manipulation = manipulation_audit(base_manifest, bases, latents)
    manipulation["latent_semantic_sha256"] = latent_sha
    manipulation["repeated_latent_regeneration_match"] = True

    base_bytes = _csv_bytes(base_manifest)
    latent_bytes = _csv_bytes(latent_manifest)
    checks_bytes = _csv_bytes(checks)
    prepare = {
        "schema_version": "e14.l4_factorial_prepare.v1",
        "status": "T-PASS-MANIPULATION",
        "created_utc": utc_now(),
        "prospective_record": {
            "path": portable_path(PREREG),
            "sha256": EXPECTED_HASHES[PREREG],
        },
        "source_provenance": source,
        "environment": environment_without_torch_import(),
        "implementation_sources": implementation_sources,
        # Kept as a direct alias for simple consumers; the authoritative map
        # above also freezes analyzer and tests before any score exists.
        "evaluator": implementation_sources["evaluator"],
        "technical_amendments": {
            "implementation": implementation_sources["amendment"],
            "stage_a_transcript":
                implementation_sources["transcript_amendment"],
        },
        "frozen_inputs": frozen_inputs,
        "matched_real_training_records": matched_real_training_records,
        "checkpoint_bundles": checkpoint_bundles,
        "standardizer": {
            "source": portable_path(TRAIN_WINDOWS),
            "derivation": (
                "mean/std over complete real train x axes (window,time); "
                "std<1e-6 replaced by 1.0; float32"
            ),
            "hashes": standardizer_hashes,
        },
        "construction": {
            "master_seed": MASTER_CONSTRUCTION_SEED,
            "stream_codes": {"l4": 1, "rule": 2},
            "cells": [cell_name(*cell) for cell in CELLS],
            "attacks": {
                str(label): {
                    key: (
                        list(value) if isinstance(value, tuple) else value
                    )
                    for key, value in spec.items()
                }
                for label, spec in ATTACK_SPECS.items()
            },
        },
        "manipulation_gates": manipulation,
        "checkpoint_deserialization_performed": False,
        "model_inference_performed": False,
        "outputs": {
            "base_manifest": {
                "path": portable_path(PREPARE_PATHS["base_manifest"]),
                "bytes": len(base_bytes),
                "sha256": _sha256_bytes(base_bytes),
                "rows": len(base_manifest),
            },
            "latent_manifest": {
                "path": portable_path(PREPARE_PATHS["latent_manifest"]),
                "bytes": len(latent_bytes),
                "sha256": _sha256_bytes(latent_bytes),
                "rows": len(latent_manifest),
                "semantic_sha256": latent_sha,
            },
            "manipulation_checks": {
                "path": portable_path(PREPARE_PATHS["manipulation_checks"]),
                "bytes": len(checks_bytes),
                "sha256": _sha256_bytes(checks_bytes),
                "rows": len(checks),
            },
        },
    }
    prepare_bytes = _json_bytes(prepare)
    _assert_prepare_torch_absent()
    publish_bundle_no_clobber(
        [
            (PREPARE_PATHS["base_manifest"], base_bytes),
            (PREPARE_PATHS["latent_manifest"], latent_bytes),
            (PREPARE_PATHS["manipulation_checks"], checks_bytes),
            (PREPARE_PATHS["prepare_record"], prepare_bytes),
        ]
    )
    _assert_prepare_torch_absent()
    print(
        _compact_json(
            {
                "status": prepare["status"],
                "transformation_sha256": manipulation["transformation_sha256"],
                "outputs": {
                    key: value["path"] for key, value in prepare["outputs"].items()
                },
            }
        )
    )


def _validate_prepare_record() -> dict[str, Any]:
    prepare_path = PREPARE_PATHS["prepare_record"]
    prepare = _read_json(prepare_path)
    if (
        prepare.get("schema_version") != "e14.l4_factorial_prepare.v1"
        or prepare.get("status") != "T-PASS-MANIPULATION"
        or prepare.get("checkpoint_deserialization_performed") is not False
        or prepare.get("model_inference_performed") is not False
    ):
        raise ValueError("E14 prepare record schema/status mismatch")
    if prepare.get("prospective_record", {}).get("sha256") != EXPECTED_HASHES[PREREG]:
        raise ValueError("E14 prepare record points to the wrong preregistration")
    _verify_implementation_source_records(
        prepare.get("implementation_sources", {})
    )
    source_provenance = prepare.get("source_provenance", {})
    if not isinstance(source_provenance, Mapping):
        raise ValueError("prepare source provenance is missing")
    implementation_transition = source_provenance.get(
        "implementation_transition"
    )
    _validate_implementation_transition(
        implementation_transition,
        require_prepare_head=False,
    )
    if (
        source_provenance.get("source_commit")
        != implementation_transition["prepare_source_commit"]
    ):
        raise ValueError("prepare source commit/transition disagreement")
    evaluator = prepare.get("evaluator", {})
    registered_evaluator = prepare["implementation_sources"]["evaluator"]
    if any(
        evaluator.get(field) != registered_evaluator[field]
        for field in ("path", "bytes", "sha256")
    ):
        raise ValueError("prepare evaluator alias disagrees with source map")
    expected_amendments = {
        "implementation": prepare["implementation_sources"]["amendment"],
        "stage_a_transcript":
            prepare["implementation_sources"]["transcript_amendment"],
    }
    if prepare.get("technical_amendments") != expected_amendments:
        raise ValueError("prepare technical-amendment aliases disagree")
    for path, expected in EXPECTED_HASHES.items():
        _assert_hash(path, expected)
    frozen_inputs = prepare.get("frozen_inputs", {})
    expected_frozen_keys = {portable_path(path) for path in EXPECTED_HASHES}
    if set(frozen_inputs) != expected_frozen_keys:
        raise ValueError("prepare frozen-input set mismatch")
    for relative, record in frozen_inputs.items():
        path = Path(os.path.abspath(REPO_ROOT / relative))
        if not path.is_relative_to(REPO_ROOT.absolute()):
            raise ValueError(f"prepare frozen input escapes repository: {relative}")
        if (
            record.get("path") != relative
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"prepare frozen input changed: {relative}")
    training_records = prepare.get("matched_real_training_records", {})
    if (
        not isinstance(training_records, Mapping)
        or set(training_records) != MATCHED_REAL_TRAINING_RECORD_KEYS
    ):
        raise ValueError("prepare matched-real training-record set mismatch")
    if dict(training_records) != _matched_real_training_record_bundle():
        raise ValueError("matched-real training evidence changed after prepare")
    _reverify_checkpoint_bundle_records(prepare)
    outputs = prepare.get("outputs", {})
    for name in ("base_manifest", "latent_manifest", "manipulation_checks"):
        record = outputs.get(name, {})
        path = PREPARE_PATHS[name]
        if (
            record.get("path") != portable_path(path)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"prepared artifact cross-link mismatch: {name}")
    gates = prepare.get("manipulation_gates", {})
    if (
        gates.get("all_manipulation_gates_pass") is not True
        or gates.get("repeated_regeneration_match") is not True
        or gates.get("repeated_latent_regeneration_match") is not True
        or gates.get("conditional_rule_anchor_000_oracle_pass") is not True
        or gates.get("scenario_instances") != 192_000
    ):
        raise ValueError("Stage-A manipulation gates did not all pass")
    return prepare


def _reverify_checkpoint_bundle_records(prepare: Mapping[str, Any]) -> None:
    records = prepare.get("checkpoint_bundles")
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("prepare checkpoint/log matrix must contain 30 records")
    observed_keys: set[tuple[str, int]] = set()
    for record in records:
        arm = str(record.get("arm"))
        seed = int(record.get("seed", -1))
        key = (arm, seed)
        if key in observed_keys or arm not in ARMS or seed not in SEEDS:
            raise ValueError("prepare checkpoint/log key mismatch")
        observed_keys.add(key)
        for field, expected_path in (
            ("checkpoint", checkpoint_path(arm, seed)),
            ("training_log", training_log_path(arm, seed)),
        ):
            artifact = record.get(field, {})
            if (
                artifact.get("path") != portable_path(expected_path)
                or artifact.get("bytes") != expected_path.stat().st_size
                or artifact.get("sha256") != sha256_file(expected_path)
            ):
                raise ValueError(
                    f"checkpoint/log changed after prepare: {arm}/{seed}/{field}"
                )
        optimizer_validation = _validate_training_log(
            arm, seed, training_log_path(arm, seed)
        )
        training_record = record["training_log"]
        for field in (
            "total_optimizer_steps",
            "selected_optimizer_step",
            "optimizer_step_validation",
        ):
            expected = (
                optimizer_validation
                if field == "optimizer_step_validation"
                else optimizer_validation[field]
            )
            if training_record.get(field) != expected:
                raise ValueError(
                    f"prepare optimizer-step audit changed: "
                    f"{arm}/{seed}/{field}"
                )
        for field in (
            "total_optimizer_steps",
            "selected_optimizer_step",
            "optimizer_step_derivation",
        ):
            expected = (
                optimizer_validation["optimizer_step_derivation"]
                if field == "optimizer_step_derivation"
                else optimizer_validation[field]
            )
            if record.get(field) != expected:
                raise ValueError(
                    f"prepare bundle optimizer-step field changed: "
                    f"{arm}/{seed}/{field}"
                )
    if observed_keys != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("prepare checkpoint/log key set is incomplete")


def load_prepared_inputs(
    prepare: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, list[FactorLatent]]:
    base_manifest = pd.read_csv(PREPARE_PATHS["base_manifest"])
    latent_manifest = pd.read_csv(
        PREPARE_PATHS["latent_manifest"],
        dtype={"l4_child_seed": str, "rule_child_seed": str},
    )
    if len(base_manifest) != 6_000 or len(latent_manifest) != 12_000:
        raise AssertionError("prepared base/latent row count mismatch")
    test = _load_npz(TEST_WINDOWS)
    indices = base_manifest["test_window_index"].to_numpy(dtype=np.int64)
    bases = np.ascontiguousarray(test["x"][indices], dtype=np.float32)
    observed_hashes = [
        hashlib.sha256(
            memoryview(np.ascontiguousarray(window, dtype="<f4")).cast("B")
        ).hexdigest()
        for window in bases
    ]
    if observed_hashes != base_manifest["window_sha256"].astype(str).tolist():
        raise ValueError("prepared base manifest no longer matches test-window bytes")
    if np.any(np.asarray(test["y_binary"])[indices] != 0) or np.any(
        np.asarray(test["y_attack_type"])[indices] != 0
    ):
        raise ValueError("prepared E14 base labels are no longer normal")
    latents = [
        latent_from_row(row)
        for row in latent_manifest.to_dict(orient="records")
    ]
    semantic_sha = latent_semantic_digest(latents)
    expected_semantic = prepare["manipulation_gates"][
        "latent_semantic_sha256"
    ]
    if semantic_sha != expected_semantic:
        raise ValueError("prepared latent semantic digest mismatch")
    return base_manifest, bases, latents


def _make_cnn(torch: Any) -> Any:
    """Construct the frozen 23-key CNN architecture (Stage B only)."""
    nn = torch.nn

    class CNN1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(11, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 5)
            )

        def forward(self, values: Any) -> Any:
            return self.head(self.net(values))

    return CNN1D()


def load_and_validate_models(
    prepare: Mapping[str, Any],
) -> tuple[Any, dict[tuple[str, int], Any], list[dict[str, Any]]]:
    """Lazily import torch and weights-only validate all 30 checkpoints."""
    import torch  # pylint: disable=import-outside-toplevel

    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # May already be fixed by an embedding process.  The observed value is
        # recorded and must still equal one below.
        pass
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("could not establish registered torch thread count")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("could not establish one inter-op torch thread")

    prepared = {
        (str(record["arm"]), int(record["seed"])): record
        for record in prepare["checkpoint_bundles"]
    }
    if set(prepared) != {
        (arm, seed) for arm in ARMS for seed in SEEDS
    }:
        raise ValueError("prepare record checkpoint matrix is incomplete")
    models: dict[tuple[str, int], Any] = {}
    validation_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            path = checkpoint_path(arm, seed)
            observed_sha = sha256_file(path)
            if observed_sha != prepared[(arm, seed)]["checkpoint"]["sha256"]:
                raise ValueError(f"checkpoint changed after preparation: {arm}/{seed}")
            model = _make_cnn(torch)
            expected_state = model.state_dict()
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, Mapping):
                raise TypeError(f"checkpoint is not a state dict: {path}")
            if set(state) != set(expected_state) or len(state) != 23:
                raise ValueError(f"checkpoint key mismatch: {path}")
            tensors = {}
            for key, expected in expected_state.items():
                value = state[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != tuple(expected.shape)
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError(f"invalid checkpoint tensor {key}: {path}")
                tensors[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": True,
                }
            model.load_state_dict(state, strict=True)
            model.eval()
            models[(arm, seed)] = model
            validation_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": portable_path(path),
                    "sha256": observed_sha,
                    "weights_only_load": True,
                    "state_dict_keys": len(state),
                    "architecture_compatible": True,
                    "all_tensors_finite": True,
                    "tensors": tensors,
                }
            )
    return torch, models, validation_rows


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = ((values - mean) / std).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("non-finite standardized model input")
    return standardized


def predict_standardized_logits(
    torch: Any,
    model: Any,
    standardized: np.ndarray,
    *,
    batch_size: int = SCORING_BATCH_SIZE,
) -> np.ndarray:
    """Return CPU logits for already-standardized float32 windows."""
    values = np.asarray(standardized)
    if values.ndim != 3 or values.shape[1:] != WINDOW_SHAPE:
        raise ValueError("standardized model input has the wrong shape")
    logits = np.empty((len(values), 5), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch_array = np.ascontiguousarray(
                values[start : start + batch_size].transpose(0, 2, 1),
                dtype=np.float32,
            )
            batch = torch.from_numpy(batch_array)
            output = model(batch).detach().cpu().numpy().astype(np.float32)
            if output.shape != (len(batch_array), 5) or not np.isfinite(output).all():
                raise ValueError("model produced invalid E14 logits")
            logits[start : start + len(output)] = output
    return logits


def exact_margin(logits: np.ndarray, attack_label: int) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("logits must have shape (N,5)")
    competitors = np.delete(logits, attack_label, axis=1)
    return (
        logits[:, attack_label] - np.max(competitors, axis=1)
    ).astype(np.float32)


def binary_margin(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("logits must have shape (N,5)")
    return (np.max(logits[:, 1:], axis=1) - logits[:, 0]).astype(np.float32)


def unscaled_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _build_by_cell(scenarios: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "arm", "seed", "block_id", "id_stratum", "P", "S", "D", "cell"
    ]
    rows = []
    for key, group in scenarios.groupby(group_columns, sort=True):
        if len(group) != 2 or set(group["attack"]) != {"Gear", "RPM"}:
            raise AssertionError("by-cell macro requires exactly Gear and RPM")
        exact_status = (
            "ok"
            if set(group["exact_standardized_margin_status"]) == {"ok"}
            else "no clean-margin dynamic range"
        )
        binary_status = (
            "ok"
            if set(group["binary_standardized_margin_status"]) == {"ok"}
            else "no clean-margin dynamic range"
        )
        row = dict(zip(group_columns, key, strict=True))
        row.update(
            {
                "attacks": 2,
                "n_per_attack": BASES_PER_BLOCK,
                "macro_weighting": "equal Gear/RPM",
                "exact_recall": float(group["exact_recall"].mean()),
                "binary_recall": float(group["binary_recall"].mean()),
                "exact_standardized_margin_shift": (
                    float(group["exact_standardized_margin_shift"].mean())
                    if exact_status == "ok"
                    else np.nan
                ),
                "binary_standardized_margin_shift": (
                    float(group["binary_standardized_margin_shift"].mean())
                    if binary_status == "ok"
                    else np.nan
                ),
                "exact_standardized_margin_status": exact_status,
                "binary_standardized_margin_status": binary_status,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 1_440 or result.duplicated(group_columns).any():
        raise AssertionError("by-cell completeness/uniqueness failure")
    return result


def _build_augmentation_delta(scenarios: pd.DataFrame) -> pd.DataFrame:
    scenario_keys = [
        "seed", "block_id", "attack", "attack_label", "id_stratum",
        "P", "S", "D", "cell",
    ]
    by_arm = {}
    for arm, frame in scenarios.groupby("arm", sort=False):
        indexed = frame.set_index(scenario_keys)
        if not indexed.index.is_unique:
            raise AssertionError(f"duplicate scenario keys for arm {arm}")
        by_arm[arm] = indexed
    rows: list[dict[str, Any]] = []
    endpoint_columns = (
        "exact_recall",
        "binary_recall",
        "exact_standardized_margin_shift",
        "binary_standardized_margin_shift",
    )
    for contrast, numerator, denominator in CONTRASTS:
        upper = by_arm[numerator].sort_index()
        lower = by_arm[denominator].sort_index()
        if not upper.index.equals(lower.index):
            raise AssertionError(f"unpaired scenario rows for {contrast}")
        for key, upper_row, lower_row in zip(
            upper.index,
            upper.itertuples(index=False),
            lower.itertuples(index=False),
            strict=True,
        ):
            upper_values = upper.loc[key]
            lower_values = lower.loc[key]
            row = {
                "contrast": contrast,
                "numerator_arm": numerator,
                "denominator_arm": denominator,
                **dict(zip(scenario_keys, key, strict=True)),
                "n": BASES_PER_BLOCK,
                "value_semantics": "numerator_arm_minus_denominator_arm",
            }
            for endpoint in endpoint_columns:
                upper_value = float(upper_values[endpoint])
                lower_value = float(lower_values[endpoint])
                row[endpoint] = (
                    upper_value - lower_value
                    if math.isfinite(upper_value) and math.isfinite(lower_value)
                    else np.nan
                )
            row["exact_standardized_margin_status"] = (
                "ok"
                if (
                    upper_values["exact_standardized_margin_status"] == "ok"
                    and lower_values["exact_standardized_margin_status"] == "ok"
                )
                else "no clean-margin dynamic range"
            )
            row["binary_standardized_margin_status"] = (
                "ok"
                if (
                    upper_values["binary_standardized_margin_status"] == "ok"
                    and lower_values["binary_standardized_margin_status"] == "ok"
                )
                else "no clean-margin dynamic range"
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    key = ["contrast", *scenario_keys]
    if len(result) != 2_880 or result.duplicated(key).any():
        raise AssertionError("augmentation-delta completeness/uniqueness failure")
    return result


def _validate_scoring_tables(
    normal: pd.DataFrame,
    scenarios: pd.DataFrame,
    cells: pd.DataFrame,
    deltas: pd.DataFrame,
) -> dict[str, int]:
    normal_key = ["arm", "seed", "block_id"]
    scenario_key = [
        "arm", "seed", "block_id", "attack", "id_stratum", "P", "S", "D"
    ]
    if len(normal) != 90 or normal.duplicated(normal_key).any():
        raise AssertionError("normal-control completeness failure")
    if len(scenarios) != 2_880 or scenarios.duplicated(scenario_key).any():
        raise AssertionError("scenario completeness failure")
    for column in ("normal_recall", "fpr"):
        values = normal[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise AssertionError(f"invalid normal-control metric: {column}")
    for column in ("exact_recall", "binary_recall"):
        values = scenarios[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise AssertionError(f"invalid scenario metric: {column}")
    if len(cells) != 1_440 or len(deltas) != 2_880:
        raise AssertionError("derived scoring-table row count mismatch")
    primary = scenarios[scenarios["arm"].isin({"real_ms", "rule_ms", "placebo_ms"})]
    canonical_primary = primary[primary["id_stratum"] == "canonical"]
    if len(primary) != 1_440 or len(canonical_primary) != 720:
        raise AssertionError("primary-panel completeness failure")
    transformed_evaluations = int(scenarios["n"].sum())
    clean_evaluations = int(normal["n"].sum())
    if transformed_evaluations != 5_760_000 or clean_evaluations != 180_000:
        raise AssertionError("forward-evaluation count mismatch")
    return {
        "checkpoint_evaluations": 30,
        "normal_rows": len(normal),
        "scenario_rows": len(scenarios),
        "primary_scenario_rows": len(primary),
        "canonical_primary_scenario_rows": len(canonical_primary),
        "by_cell_rows": len(cells),
        "augmentation_delta_rows": len(deltas),
        "transformed_forward_evaluations": transformed_evaluations,
        "clean_forward_evaluations": clean_evaluations,
    }


def _run_score_success() -> None:
    """Execute the complete registered Stage-B CPU scoring matrix."""
    for path in SCORE_PATHS.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite E14 score output: {path}")
    started_utc = utc_now()
    source = source_provenance_for_score()
    prepare = _validate_prepare_record()
    base_manifest, bases, latents = load_prepared_inputs(prepare)

    transform_sha, scenario_key_sha, instances, _ = _transform_pass(
        base_manifest, bases, latents, validate_relations=False
    )
    gates = prepare["manipulation_gates"]
    if (
        instances != 192_000
        or transform_sha != gates["transformation_sha256"]
        or scenario_key_sha != gates["scenario_key_sha256"]
    ):
        raise RuntimeError(
            "Stage-B transform regeneration does not match frozen Stage A"
        )

    mean, std, standardizer_hashes = fit_registered_standardizer()
    if standardizer_hashes != prepare["standardizer"]["hashes"]:
        raise ValueError("standardizer changed after Stage A")
    torch, models, checkpoint_validation = load_and_validate_models(prepare)

    standardized_bases = _standardize(bases, mean, std)
    block_orders = {
        block_id: base_manifest.index[
            base_manifest["block_id"] == block_id
        ].to_numpy(dtype=np.int64)
        for block_id in BLOCK_IDS
    }
    if any(len(values) != BASES_PER_BLOCK for values in block_orders.values()):
        raise AssertionError("prepared block order is incomplete")

    clean_logits: dict[tuple[str, int], np.ndarray] = {}
    clean_exact_margin: dict[tuple[str, int, int], np.ndarray] = {}
    clean_binary_margin: dict[tuple[str, int, int], np.ndarray] = {}
    clean_mad: dict[tuple[str, int, int, str], float] = {}
    normal_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            key = (arm, seed)
            logits = predict_standardized_logits(
                torch, models[key], standardized_bases
            )
            clean_logits[key] = logits
            predictions = logits.argmax(axis=1)
            for attack_label in sorted(ATTACK_SPECS):
                exact_values = exact_margin(logits, attack_label)
                binary_values = binary_margin(logits)
                clean_exact_margin[(arm, seed, attack_label)] = exact_values
                clean_binary_margin[(arm, seed, attack_label)] = binary_values
                clean_mad[(arm, seed, attack_label, "exact")] = unscaled_mad(
                    exact_values
                )
                clean_mad[(arm, seed, attack_label, "binary")] = unscaled_mad(
                    binary_values
                )
            for block_id, orders in block_orders.items():
                correct = int(np.count_nonzero(predictions[orders] == 0))
                normal_recall = correct / len(orders)
                normal_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "block_id": block_id,
                        "control_scope": "Car-Hacking untouched normal base",
                        "n": len(orders),
                        "normal_correct": correct,
                        "normal_recall": normal_recall,
                        "fpr": 1.0 - normal_recall,
                        "checkpoint_sha256": sha256_file(
                            checkpoint_path(arm, seed)
                        ),
                    }
                )
    normal = pd.DataFrame(normal_rows)

    latent_lookup = {
        (
            latent.block_id,
            latent.block_position,
            latent.test_window_index,
            latent.attack_label,
        ): latent
        for latent in latents
    }
    scenario_rows: list[dict[str, Any]] = []
    for block_id in BLOCK_IDS:
        orders = block_orders[block_id]
        block_rows = base_manifest.loc[orders]
        block_bases = bases[orders]
        for attack_label in sorted(ATTACK_SPECS):
            spec = ATTACK_SPECS[attack_label]
            block_latents = [
                latent_lookup[
                    (
                        block_id,
                        int(row.block_position),
                        int(row.test_window_index),
                        attack_label,
                    )
                ]
                for row in block_rows.itertuples(index=False)
            ]
            for id_stratum in ID_STRATA:
                for p, s, d in CELLS:
                    transformed = np.stack(
                        [
                            apply_factorial_transform(
                                base,
                                latent,
                                attack_label=attack_label,
                                id_stratum=id_stratum,
                                p=p,
                                s=s,
                                d=d,
                            )
                            for base, latent in zip(
                                block_bases, block_latents, strict=True
                            )
                        ]
                    ).astype(np.float32, copy=False)
                    standardized = _standardize(transformed, mean, std)
                    for arm in ARMS:
                        for seed in SEEDS:
                            logits = predict_standardized_logits(
                                torch, models[(arm, seed)], standardized
                            )
                            predictions = logits.argmax(axis=1)
                            exact_correct = int(
                                np.count_nonzero(predictions == attack_label)
                            )
                            binary_correct = int(
                                np.count_nonzero(predictions != 0)
                            )
                            transformed_exact = exact_margin(
                                logits, attack_label
                            )
                            transformed_binary = binary_margin(logits)
                            clean_exact = clean_exact_margin[
                                (arm, seed, attack_label)
                            ][orders]
                            clean_binary = clean_binary_margin[
                                (arm, seed, attack_label)
                            ][orders]
                            exact_shift = transformed_exact - clean_exact
                            binary_shift = transformed_binary - clean_binary
                            exact_mad = clean_mad[
                                (arm, seed, attack_label, "exact")
                            ]
                            binary_mad = clean_mad[
                                (arm, seed, attack_label, "binary")
                            ]
                            exact_ok = exact_mad > 1e-6
                            binary_ok = binary_mad > 1e-6
                            scenario_rows.append(
                                {
                                    "arm": arm,
                                    "seed": seed,
                                    "block_id": block_id,
                                    "attack": spec["attack"],
                                    "attack_label": attack_label,
                                    "id_stratum": id_stratum,
                                    "P": p,
                                    "S": s,
                                    "D": d,
                                    "cell": cell_name(p, s, d),
                                    "n": BASES_PER_BLOCK,
                                    "exact_correct": exact_correct,
                                    "binary_correct": binary_correct,
                                    "exact_recall": (
                                        exact_correct / BASES_PER_BLOCK
                                    ),
                                    "binary_recall": (
                                        binary_correct / BASES_PER_BLOCK
                                    ),
                                    "exact_margin_shift_mean": float(
                                        np.mean(exact_shift)
                                    ),
                                    "binary_margin_shift_mean": float(
                                        np.mean(binary_shift)
                                    ),
                                    "exact_clean_margin_mad": exact_mad,
                                    "binary_clean_margin_mad": binary_mad,
                                    "exact_standardized_margin_shift": (
                                        float(np.mean(exact_shift / exact_mad))
                                        if exact_ok
                                        else np.nan
                                    ),
                                    "binary_standardized_margin_shift": (
                                        float(np.mean(binary_shift / binary_mad))
                                        if binary_ok
                                        else np.nan
                                    ),
                                    "exact_standardized_margin_status": (
                                        "ok"
                                        if exact_ok
                                        else "no clean-margin dynamic range"
                                    ),
                                    "binary_standardized_margin_status": (
                                        "ok"
                                        if binary_ok
                                        else "no clean-margin dynamic range"
                                    ),
                                }
                            )
    scenarios = pd.DataFrame(scenario_rows)
    cells = _build_by_cell(scenarios)
    deltas = _build_augmentation_delta(scenarios)
    counts = _validate_scoring_tables(normal, scenarios, cells, deltas)

    normal_bytes = _csv_bytes(normal)
    cells_bytes = _csv_bytes(cells)
    scenarios_bytes = _csv_bytes(scenarios)
    deltas_bytes = _csv_bytes(deltas)
    completed_utc = utc_now()
    log = {
        "schema_version": "e14.l4_factorial_score_log.v1",
        "status": "T-PASS",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "batch_size": SCORING_BATCH_SIZE,
        "counts": counts,
        "transformation_sha256": transform_sha,
        "normal_control_reported_before_attack_results": True,
        "exact_and_binary_endpoints_separate": True,
        "effects_and_scientific_verdict_deferred_to_registered_analyzer": True,
    }
    log_bytes = _json_bytes(log)
    table_payloads = {
        "normal_by_seed": normal_bytes,
        "by_cell": cells_bytes,
        "by_scenario": scenarios_bytes,
        "augmentation_delta": deltas_bytes,
        "log": log_bytes,
    }
    run = {
        "schema_version": "e14.l4_factorial_run.v1",
        "status": "T-PASS",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "analysis_role": (
            "evaluation-only construction factorial; no training, threshold "
            "selection, external data, or scientific verdict"
        ),
        "source_provenance": source,
        "prospective_record": {
            "path": portable_path(PREREG),
            "sha256": EXPECTED_HASHES[PREREG],
        },
        "prepare_record": {
            "path": portable_path(PREPARE_PATHS["prepare_record"]),
            "sha256": sha256_file(PREPARE_PATHS["prepare_record"]),
        },
        "evaluator": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "batch_size": SCORING_BATCH_SIZE,
        },
        "standardizer_hashes": standardizer_hashes,
        "checkpoint_validation": checkpoint_validation,
        "construction": {
            "transformation_instances": instances,
            "transformation_sha256": transform_sha,
            "scenario_key_sha256": scenario_key_sha,
            "matches_prepare": True,
            "serialization_version": TRANSFORM_SERIALIZATION_VERSION,
        },
        "completeness": counts,
        "outputs": {
            name: {
                "path": portable_path(SCORE_PATHS[name]),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in table_payloads.items()
        },
        "effects_table_published": False,
        "artifact_manifest_published": False,
        "next_required_stage": (
            "registered analyzer publishes effects and final artifact manifest"
        ),
    }
    run_bytes = _json_bytes(run)
    publish_bundle_no_clobber(
        [
            (SCORE_PATHS["normal_by_seed"], normal_bytes),
            (SCORE_PATHS["by_cell"], cells_bytes),
            (SCORE_PATHS["by_scenario"], scenarios_bytes),
            (SCORE_PATHS["augmentation_delta"], deltas_bytes),
            (SCORE_PATHS["log"], log_bytes),
            (SCORE_PATHS["run_record"], run_bytes),
        ]
    )
    print(
        _compact_json(
            {
                "status": "T-PASS",
                "run_record": portable_path(SCORE_PATHS["run_record"]),
                "transformation_sha256": transform_sha,
                "counts": counts,
            }
        )
    )


def _path_presence_audit(path: Path) -> dict[str, Any]:
    """Describe one canonical path without following a symlink."""
    exists = os.path.lexists(os.fspath(path))
    record: dict[str, Any] = {
        "path": portable_path(path),
        "lexists": exists,
        "is_symlink": bool(path.is_symlink()) if exists else False,
        "is_regular_file": False,
        "bytes": None,
        "sha256": None,
        "audit_error": None,
    }
    if not exists or record["is_symlink"]:
        return record
    try:
        record["is_regular_file"] = path.is_file()
        if record["is_regular_file"]:
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
    except OSError as exc:
        record["audit_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _failure_output_presence(stage: str) -> dict[str, Any]:
    prepare = {
        name: _path_presence_audit(path)
        for name, path in PREPARE_PATHS.items()
    }
    score = {
        name: _path_presence_audit(path)
        for name, path in SCORE_PATHS.items()
    }
    try:
        staged = sorted(
            portable_path(path)
            for path in EXP.glob(".e14_staged_*")
            if path.is_dir()
        )
    except OSError:
        staged = []
    return {
        "failed_stage": stage,
        "upstream_prepare_outputs": prepare,
        "score_outputs": score,
        "prepare_outputs_present": sum(
            int(record["lexists"]) for record in prepare.values()
        ),
        "score_outputs_present": sum(
            int(record["lexists"]) for record in score.values()
        ),
        "isolated_staging_directories": staged,
    }


def _versioned_failure_path(stage: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return (
        FAILURE_DIR
        / f"{stage}_failure_{stamp}_pid{os.getpid()}_{time.time_ns()}.json"
    )


def publish_failure_record(
    *, stage: str, status: str, error: BaseException
) -> Path:
    """Atomically publish a non-scientific, versioned technical failure record."""
    if stage not in {"prepare", "score"}:
        raise ValueError(f"unknown E14 failure stage: {stage}")
    expected_status = {
        "prepare": "T-STOP-MANIPULATION",
        "score": "T-INCOMPLETE",
    }[stage]
    if status != expected_status:
        raise ValueError(f"invalid {stage} failure status: {status}")
    target = _versioned_failure_path(stage)
    try:
        head_commit = _git_output(["rev-parse", "HEAD"]).strip()
    except Exception as provenance_error:  # best-effort failure diagnostics
        head_commit = None
        provenance_note = (
            f"{type(provenance_error).__name__}: {provenance_error}"
        )
    else:
        provenance_note = None
    record = {
        "schema_version": "e14.l4_factorial_failure.v1",
        "record_type": "technical_failure",
        "status": status,
        "stage": stage,
        "created_utc": utc_now(),
        "source_commit": head_commit,
        "source_provenance_error": provenance_note,
        "process": {
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
            "argv": list(sys.argv),
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "repr": repr(error),
            "traceback": "".join(
                traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            ),
        },
        "output_presence_audit": _failure_output_presence(stage),
        "failure_handler_published_scientific_outputs": False,
        "scientific_verdict_issued": False,
        "canonical_success_record_published_by_failure_handler": False,
        "recovery_policy": (
            "preserve all pre-existing and isolated partial artifacts; do not "
            "reuse, overwrite, or issue a scientific verdict without the "
            "registered amendment/version procedure"
        ),
        "record_path": portable_path(target),
    }
    publish_bundle_no_clobber([(target, _json_bytes(record))])
    return target


def _run_with_failure_record(
    *, stage: str, status: str, operation: Any
) -> None:
    try:
        operation()
    except BaseException as error:
        try:
            failure_path = publish_failure_record(
                stage=stage, status=status, error=error
            )
        except BaseException as publication_error:
            print(
                "E14 failure-record publication also failed: "
                f"{type(publication_error).__name__}: {publication_error}",
                file=sys.stderr,
            )
        else:
            print(
                f"E14 technical failure recorded at {portable_path(failure_path)}",
                file=sys.stderr,
            )
        raise


def run_prepare_only() -> None:
    _run_with_failure_record(
        stage="prepare",
        status="T-STOP-MANIPULATION",
        operation=_run_prepare_only_success,
    )


def run_score() -> None:
    _run_with_failure_record(
        stage="score",
        status="T-INCOMPLETE",
        operation=_run_score_success,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or score the preregistered E14 matched L4 factorial. "
            "Exactly one stage must be selected."
        )
    )
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "run manipulation/provenance gates and publish preparation records; "
            "this path never deserializes a checkpoint or runs inference"
        ),
    )
    stages.add_argument(
        "--score",
        action="store_true",
        help="run the complete frozen CPU scoring matrix after Stage A",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_only:
        run_prepare_only()
    else:
        run_score()


if __name__ == "__main__":
    main()
