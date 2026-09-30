#!/usr/bin/env python3
"""Fit the 245 preregistered E18 Random-Forest arms on the joint-structure map.

Registered by ``journal/experiments/e18_relation_visible_rf/PREREG.md``
sections 5, 6, 14.2, 15.3 and 15.5.

Why E18 exists
--------------
E17 fit a Random Forest on the 55-dimensional ``rf_features`` marginal map.
``mean``, ``std``, ``min`` and ``max`` of a channel are invariant under a
within-window permutation of that channel, and the E16 placebo twin *is* such a
permutation of ``data1`` with the per-window multiset preserved.  54 of E17's 55
features were therefore provably blind to the manipulation, and only 2.54% of
raw-differing windows produced any feature-vector change.  E17 is a valid
negative control; it is not evidence about the relation.

E18 changes exactly one thing: the feature map.  It adds the 28 within-window
Pearson correlations between every unordered pair of the eight payload bytes
``data0..data7``, giving 83 features.  The addition is deliberately generic
cross-byte joint structure -- not a Gear- or RPM-specific indicator -- so the
question is "does a detector able to represent cross-byte joint structure
respond to the manipulation?", not "does a bespoke Gear detector detect Gear?".

Grid (PREREG section 6)
-----------------------
* 20 generator realizations x 2 arms (Rule, placebo) x 5 pipeline seeds = 200
* 4 bridge realizations x 2 arms x 5 pipeline seeds                    =  40
* 5 shared real-only references                                        =   5

Stages
------
``(default)``        read-only preflight; performs no fitting.
``--verify-pools``   stage S2; hashes the 48 consumed pool slots, inherits the
                     frozen manipulation checks, publishes the audit.  No fit.
``--visibility``     stage S2b; gate G17.  Measures, for all 24 pool pairs, how
                     much of the manipulation the 83-dimensional map perceives,
                     publishes the table **regardless of outcome**, then applies
                     the registered floor.  No fit.
``--visibility-v2``  stage S2b-v2; the same gate measured in float64 and split
                     by attack, under amendment 2026-08-07 v2.  Additive: the
                     v1 table is preserved.  No fit.
``--execute``        stages S3 and S4; 5 reference fits then 240 paired fits.

Pools, draws, seeds, evaluation material, endpoint, aggregation order, margin,
decision rule and inferential unit are inherited from E16/E17 verbatim; the pool
inheritance machinery and the post-fit gates are imported from the E17 runner
rather than re-implemented.  Importing this module performs no fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib_common as lc  # noqa: E402
import generate_e16_rule_placebo_pairs as e16gen  # noqa: E402
import run_e17_rf_paired_training as e17train  # noqa: E402
import train_generator_extension_rf as rfmod  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _stage_bytes,
    _stage_json,
)
from train_rule_construction_seed_crossing import (  # noqa: E402
    SAMPLING_SEED_OFFSET,
    SYNTHETIC_TOTAL,
    _load_real_train_val,
    sample_indices_without_replacement,
    sampling_indices_sha256,
)

REPO = e17train.REPO
EXPERIMENT_DIR = REPO / "journal" / "experiments" / "e18_relation_visible_rf"
SYNTHETIC_DIR = e17train.SYNTHETIC_DIR
MODEL_DIR = e17train.MODEL_DIR
LOG_DIR = e17train.LOG_DIR
TABLE_DIR = e17train.TABLE_DIR

# --- registration provenance (PREREG header + freeze commit) ----------------

PREREG_PATH = "journal/experiments/e18_relation_visible_rf/PREREG.md"
#: Commit that froze the E18 registration.  It is a single-file commit and is
#: required to be a strict ancestor of the commit that introduced this runner
#: (PREREG section 15.1, gate G0).
PREREG_COMMIT = "f2f9930"
PREREG_SOURCE_COMMIT = "93dbdc4"
IMPLEMENTATION_PATHS = (
    "journal/scripts/run_e18_relation_visible_rf_training.py",
    "journal/scripts/evaluate_e18_relation_visible_rf.py",
    "journal/scripts/analyze_e18_relation_visible_rf.py",
)
#: Dated prospective amendments, in order, each committed before the stage it
#: affects was rerun (PREREG section 15.4).
AMENDMENTS = (
    ("journal/experiments/e18_relation_visible_rf/"
     "AMENDMENT_2026-08-07_V1_FLOAT32_REDUCTION_ARTIFACT.md", "73aa169"),
    ("journal/experiments/e18_relation_visible_rf/"
     "AMENDMENT_2026-08-07_V2_FLOAT64_AND_PER_ATTACK_VISIBILITY.md",
     "bd86500"),
)
AMENDMENT_PATH = AMENDMENTS[0][0]
AMENDMENT_COMMIT = AMENDMENTS[0][1]

# --- frozen design constants (PREREG sections 3, 5, 6) ----------------------

REGISTERED_CONSTRUCTION_SEEDS = e17train.REGISTERED_CONSTRUCTION_SEEDS
REGISTERED_BRIDGE_SEEDS = e17train.REGISTERED_BRIDGE_SEEDS
CONSTRUCTION_SEEDS = e16gen.CONSTRUCTION_SEEDS
BRIDGE_SEEDS = e16gen.BRIDGE_SEEDS
PIPELINE_SEEDS = e17train.PIPELINE_SEEDS
ARMS = e17train.ARMS

OUTPUT_VERSION = "v1"
MODEL_TAG = "e18_v1"
PER_ATTACK_CAP = e17train.PER_ATTACK_CAP

#: PREREG section 5.1.  ``lib_common.FEATURE_NAMES`` is
#: ``[can_id, dlc, data0..data7, delta_t]``, so the eight payload bytes are raw
#: channel indices 2..9.
PAYLOAD_CHANNEL_SLICE = slice(2, 10)
PAYLOAD_CHANNELS = 8
#: Frozen pair enumeration order: lexicographic over ``data`` indices, 28 pairs.
CORRELATION_PAIRS = tuple(itertools.combinations(range(PAYLOAD_CHANNELS), 2))
MARGINAL_DIMENSION = e17train.FEATURE_DIMENSION          # 55
CORRELATION_DIMENSION = len(CORRELATION_PAIRS)           # 28
FEATURE_DIMENSION = MARGINAL_DIMENSION + CORRELATION_DIMENSION   # 83
#: Rows per chunk when featurizing; the map is row-wise so this is exact.
FEATURE_CHUNK = 16_384

ESTIMATOR = e17train.ESTIMATOR
ESTIMATOR_FIDELITY_KEYS = e17train.ESTIMATOR_FIDELITY_KEYS

# --- gate G17 (PREREG section 15.5), frozen before measurement --------------

G17_POOLED_FLOOR = 0.50
G17_PER_PAIR_FLOOR = 0.25
#: The manipulated raw channel: ``lib_common.FEATURE_NAMES[3] == "data1"``, i.e.
#: payload byte ``data1`` (payload-local index 1).
MANIPULATED_RAW_CHANNEL = 3
MANIPULATED_PAYLOAD_INDEX = 1
#: The five ``rf_features`` statistic blocks, in concatenation order.
MARGINAL_STATISTICS = ("mean", "std", "min", "max", "delta")
RAW_CHANNELS = MARGINAL_DIMENSION // len(MARGINAL_STATISTICS)   # 11


def _marginal_index(statistic: str, raw_channel: int) -> int:
    return MARGINAL_STATISTICS.index(statistic) * RAW_CHANNELS + raw_channel


def _correlation_index(payload_pair: tuple[int, int]) -> int:
    return MARGINAL_DIMENSION + CORRELATION_PAIRS.index(payload_pair)


#: PREREG section 5.1 as amended 2026-08-07 v1, defined structurally from the
#: channel identity of ``data1`` and the frozen pair list, never from a
#: magnitude threshold and never from an observed outcome.
#:
#: SUBSTANTIVE — features whose *exact-arithmetic* value can change under the
#: registered manipulation: ``delta(data1)`` plus the seven ``r(data1, .)``.
SUBSTANTIVE_FEATURE_INDICES = tuple(sorted(
    [_marginal_index("delta", MANIPULATED_RAW_CHANNEL)]
    + [_correlation_index(pair) for pair in CORRELATION_PAIRS
       if MANIPULATED_PAYLOAD_INDEX in pair]))
#: ARTIFACT — provably invariant in exact arithmetic; a float32 reduction over
#: the 128-frame axis is order-sensitive and the placebo permutes that order, so
#: ``mean`` and ``std`` of the manipulated channel may differ at float32 epsilon
#: scale.  ``min``/``max`` are order-independent selections and ``delta`` is a
#: two-element difference, so neither can produce an artifact.
ARTIFACT_FEATURE_INDICES = tuple(sorted(
    _marginal_index(statistic, MANIPULATED_RAW_CHANNEL)
    for statistic in ("mean", "std")))
#: Anything outside the union is an implementation error and is a technical stop.
ADMISSIBLE_FEATURE_INDICES = frozenset(
    SUBSTANTIVE_FEATURE_INDICES + ARTIFACT_FEATURE_INDICES)
#: PREREG section 1.1, the E17 substantive-metric reference figures, reproduced
#: inside E18's own audit rather than quoted.
E17_REFERENCE_VISIBLE_FRACTION = 0.0254
E17_REFERENCE_SUBSTANTIVE_FEATURES = 1

POOL_INHERITANCE_SCHEMA = "e18.pool_inheritance.v1"
VISIBILITY_SCHEMA = "e18.manipulation_visibility.v1"
TRAINING_PREFLIGHT_SCHEMA = "e18.training_preflight.v1"
TRAINING_RUN_SCHEMA = "e18.training_run.v1"
TRAINING_LOG_SCHEMA = "e18.rf_training_log.v1"

EXPECTED_PRIMARY_FITS = e17train.EXPECTED_PRIMARY_FITS
EXPECTED_BRIDGE_FITS = e17train.EXPECTED_BRIDGE_FITS
EXPECTED_REFERENCE_FITS = e17train.EXPECTED_REFERENCE_FITS
EXPECTED_FIT_COUNT = e17train.EXPECTED_FIT_COUNT

publish_and_cleanup = e17train.publish_and_cleanup
_sha256_file = e17train._sha256_file
_git = e17train._git
_dump_forest = e17train._dump_forest
_forbidden_standardizer = e17train._forbidden_standardizer
#: Imported, never re-implemented (PREREG section 14.1, gate G14).
rf_features = rfmod.rf_features


class E18TrainingError(RuntimeError):
    """Raised on any registered E18 training gate violation."""


class E18BranchE(RuntimeError):
    """Gate G17 floor failure: inconclusive-by-construction, not a stop."""


def _stop(message: str) -> E18TrainingError:
    return E18TrainingError(f"T-STOP-E18-TRAIN: {message}")


def _incomplete(message: str) -> E18TrainingError:
    return E18TrainingError(f"T-INCOMPLETE-E18-TRAIN: {message}")


def utc_now() -> str:
    return e16gen.utc_now()


# ---------------------------------------------------------------------------
# The registered feature map (PREREG section 5.1)
# ---------------------------------------------------------------------------


def _payload_correlations_float64(x: np.ndarray) -> np.ndarray:
    """The 28 within-window Pearson correlations over ``data0..data7`` pairs.

    Population (``ddof = 0``) covariance and standard deviation over the 128
    frames, computed in ``float64`` on raw un-standardized windows.  A pair with
    a constant channel has an undefined correlation and is defined as ``0.0``
    by the registered convention -- the value that encodes "no linear
    association was observable" -- which keeps the matrix finite and NaN-free
    without imputation.
    """
    payload = np.asarray(x, dtype=np.float64)[:, :, PAYLOAD_CHANNEL_SLICE]
    if payload.shape[2] != PAYLOAD_CHANNELS:
        raise _stop(
            f"payload slice yielded {payload.shape[2]} channels, expected "
            f"{PAYLOAD_CHANNELS}")
    frames = payload.shape[1]
    centered = payload - payload.mean(axis=1, keepdims=True)
    covariance = np.einsum("nti,ntj->nij", centered, centered) / frames
    variance = np.einsum("nii->ni", covariance)
    deviation = np.sqrt(np.maximum(variance, 0.0))
    left = np.fromiter((p[0] for p in CORRELATION_PAIRS), dtype=np.intp,
                       count=CORRELATION_DIMENSION)
    right = np.fromiter((p[1] for p in CORRELATION_PAIRS), dtype=np.intp,
                        count=CORRELATION_DIMENSION)
    numerator = covariance[:, left, right]
    denominator = deviation[:, left] * deviation[:, right]
    correlations = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=correlations, where=denominator > 0.0)
    return correlations


def payload_correlation_features(x: np.ndarray) -> np.ndarray:
    """The registered float32 correlation block."""
    return _payload_correlations_float64(x).astype(np.float32)


def e18_features(x: np.ndarray) -> np.ndarray:
    """The registered 83-dimensional map: ``rf_features`` (55) then 28 correlations."""
    windows = np.asarray(x, dtype=np.float32)
    if windows.ndim != 3:
        raise _stop(f"expected (N, frames, channels) windows, got {windows.shape}")
    if len(windows) == 0:
        return np.zeros((0, FEATURE_DIMENSION), dtype=np.float32)
    parts: list[np.ndarray] = []
    for start in range(0, len(windows), FEATURE_CHUNK):
        block = windows[start:start + FEATURE_CHUNK]
        parts.append(np.concatenate(
            (rf_features(block), payload_correlation_features(block)), axis=1))
    features = np.concatenate(parts, axis=0)
    if features.shape[1] != FEATURE_DIMENSION:
        raise _stop(f"realized feature dimension {features.shape[1]}")
    return features.astype(np.float32, copy=False)


# --- float64 measurement mirrors (amendment 2026-08-07 v2) ------------------
#
# Measurement instruments only.  Every E18 checkpoint is fit on the registered
# float32 map above; nothing below ever touches a fit.


def marginal_features_float64(x: np.ndarray) -> np.ndarray:
    """float64 mirror of the frozen ``rf_features``: same five statistics."""
    windows = np.asarray(x, dtype=np.float64)
    return np.concatenate([
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        windows[:, -1, :] - windows[:, 0, :],
    ], axis=1)


def e18_features_float64(x: np.ndarray) -> np.ndarray:
    """The 83-dimensional map evaluated end to end in float64."""
    windows = np.asarray(x, dtype=np.float32)
    if len(windows) == 0:
        return np.zeros((0, FEATURE_DIMENSION), dtype=np.float64)
    parts = [
        np.concatenate((marginal_features_float64(block),
                        _payload_correlations_float64(block)), axis=1)
        for block in (windows[i:i + FEATURE_CHUNK]
                      for i in range(0, len(windows), FEATURE_CHUNK))
    ]
    return np.concatenate(parts, axis=0)


def assert_float64_mirror_fidelity() -> dict[str, Any]:
    """The mirror is the frozen map, not a second feature design.

    Amendment v2 section 2.1 asserted the mirror "reproduce[s] the frozen
    function exactly when cast back to float32".  That sentence is **too
    strong and is corrected here**, disclosed rather than quietly satisfied:
    ``mean``, ``min``, ``max`` and ``delta`` are exact in both precisions --
    the raw channels are integers in [0, 255], so their 128-frame sums and
    the mean are exactly representable in float32 -- but ``std`` accumulates
    squared deviations, which are not, so the frozen float32 reduction and the
    float64 mirror differ there at float32 epsilon.  That difference is the
    very artifact amendment v1 diagnosed and amendment v2 exists to remove, so
    requiring bit-equality on it would have been self-contradictory.  The
    realized check is therefore: bit-exact on the 44 order-independent
    marginals, and bounded by float32 epsilon on the 11 ``std`` marginals.
    """
    rng = np.random.default_rng(20260807)
    probe = rng.integers(0, 256, size=(64, 128, 11)).astype(np.float32)
    frozen = rf_features(probe)
    mirror = marginal_features_float64(probe).astype(np.float32)
    std_block = slice(MARGINAL_STATISTICS.index("std") * RAW_CHANNELS,
                      (MARGINAL_STATISTICS.index("std") + 1) * RAW_CHANNELS)
    exact = np.ones(MARGINAL_DIMENSION, dtype=bool)
    exact[std_block] = False
    if not np.array_equal(frozen[:, exact], mirror[:, exact]):
        raise _stop(
            "the float64 marginal mirror is not bit-identical to the frozen "
            "rf_features on the order-independent statistics")
    std_gap = float(np.abs(frozen[:, std_block].astype(np.float64)
                           - mirror[:, std_block].astype(np.float64)).max())
    tolerance = 1e-3
    if std_gap > tolerance:
        raise _stop(
            f"the float64 marginal mirror disagrees with the frozen "
            f"rf_features std block by {std_gap}, above float32 rounding")
    correlations = _payload_correlations_float64(probe)
    if not np.array_equal(correlations.astype(np.float32),
                          payload_correlation_features(probe)):
        raise _stop("the float64 correlation block is not the registered one")
    return {
        "check": "float64_mirror_fidelity",
        "passed": True,
        "order_independent_marginals_bit_exact": True,
        "order_independent_marginals": int(np.count_nonzero(exact)),
        "std_marginals": int(RAW_CHANNELS),
        "std_max_abs_gap_vs_frozen_float32": std_gap,
        "std_gap_tolerance": tolerance,
        "correlation_block_identical_after_cast": True,
        "disclosed_deviation": (
            "amendment v2 section 2.1 says the mirror reproduces the frozen "
            "function 'exactly when cast back to float32'.  That is true for "
            "44 of the 55 marginals and false for the 11 std marginals, "
            "because the float32 std reduction is order-sensitive -- which is "
            "the artifact amendment v2 exists to remove.  The assertion is "
            "implemented as bit-exactness on the 44 plus a float32-epsilon "
            "bound on the 11, and the overstatement is disclosed here rather "
            "than silently weakened."),
        "fits_affected": "none; every E18 checkpoint is fit on the registered "
                         "float32 map",
    }


# ---------------------------------------------------------------------------
# Registration provenance (gate G0)
# ---------------------------------------------------------------------------


def assert_registration_committed() -> dict[str, Any]:
    """S0/G0 — registration committed, and committed *before* the runner."""
    record: dict[str, Any] = {
        "gate": "G0_registration_committed",
        "passed": True,
        "preregistration_path": PREREG_PATH,
        "registration_source_commit_in_header": PREREG_SOURCE_COMMIT,
    }
    path = REPO / PREREG_PATH
    if not path.exists():
        raise _stop(f"missing registration document: {PREREG_PATH}")
    if _git("status", "--porcelain", "--", PREREG_PATH):
        raise _stop(f"registration has uncommitted changes: {PREREG_PATH}")
    try:
        resolved = _git("rev-parse", PREREG_COMMIT)
        subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
             PREREG_COMMIT, "HEAD"], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise _stop(
            f"registration commit {PREREG_COMMIT} is not an ancestor of HEAD"
        ) from exc
    record["registration_commit"] = resolved
    record["registration_commit_is_ancestor"] = True
    record["registration_last_modified_commit"] = _git(
        "log", "-1", "--format=%H", "--", PREREG_PATH)
    record["registration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if record["registration_last_modified_commit"] != resolved:
        raise _stop(
            "the registration has been modified after its freeze commit "
            f"{PREREG_COMMIT}")
    files_in_freeze = _git("show", "--name-only", "--format=", PREREG_COMMIT)
    record["registration_commit_files"] = [
        line for line in files_in_freeze.splitlines() if line.strip()]
    if record["registration_commit_files"] != [PREREG_PATH]:
        raise _stop(
            "the registration freeze is not a single-file commit: "
            f"{record['registration_commit_files']}")

    # PREREG section 15.4: every amendment is committed before the stage it
    # affects is rerun.
    amendments: list[dict[str, str]] = []
    for rel_path, commit in AMENDMENTS:
        amendment_path = REPO / rel_path
        if not amendment_path.exists():
            raise _stop(f"missing amendment document: {rel_path}")
        if _git("status", "--porcelain", "--", rel_path):
            raise _stop(f"amendment has uncommitted changes: {rel_path}")
        try:
            amendment_resolved = _git("rev-parse", commit)
            subprocess.run(
                ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                 commit, "HEAD"], check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise _stop(
                f"amendment commit {commit} is not an ancestor of HEAD"
            ) from exc
        amendments.append({
            "path": rel_path,
            "commit": amendment_resolved,
            "commit_is_ancestor": "true",
            "sha256": hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
        })
    record["amendments"] = amendments

    # PREREG section 15.1: registration strictly before implementation.
    implementation: dict[str, str] = {}
    for rel_path in IMPLEMENTATION_PATHS:
        if not (REPO / rel_path).exists():
            raise _stop(f"missing implementation file: {rel_path}")
        if _git("status", "--porcelain", "--", rel_path):
            raise _stop(f"implementation has uncommitted changes: {rel_path}")
        commit = _git("log", "-1", "--format=%H", "--", rel_path)
        if not commit:
            raise _stop(f"implementation file is not committed: {rel_path}")
        if commit == resolved:
            raise _stop(
                f"{rel_path} was committed in the registration freeze commit; "
                "the registration must be a single-file commit that precedes "
                "the implementation")
        try:
            subprocess.run(
                ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                 resolved, commit], check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise _stop(
                f"registration commit {PREREG_COMMIT} is not an ancestor of "
                f"the commit {commit[:7]} that introduced {rel_path}; the "
                "registered commit ordering is violated") from exc
        implementation[rel_path] = commit
    record["implementation_commits"] = implementation
    record["registration_precedes_implementation"] = True

    frozen = _git("status", "--porcelain", "--", "wisa")
    if frozen:
        raise _stop(f"the frozen wisa/ tree is dirty: {frozen.splitlines()[:3]}")
    record["wisa_tree_clean"] = True
    return record


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E18Job:
    construction_seed: int
    arm: str
    pipeline_seed: int
    role: str

    @property
    def setting(self) -> str:
        if self.arm == "real_only":
            return "real_only"
        return f"{self.arm}_0p30_cseed{self.construction_seed}"

    @property
    def pool_path(self) -> Path | None:
        """Read-only pool consumed by this fit; ``None`` for the reference."""
        return e17train.E17Job(
            self.construction_seed, self.arm, self.pipeline_seed,
            self.role).pool_path

    @property
    def checkpoint_path(self) -> Path:
        return rfmod.checkpoint_path(self.setting, self.pipeline_seed, MODEL_TAG)

    @property
    def log_path(self) -> Path:
        return rfmod.training_log_path(
            self.setting, self.pipeline_seed, MODEL_TAG)

    @property
    def key(self) -> str:
        return f"{self.construction_seed}:{self.arm}:{self.pipeline_seed}"


def build_training_grid() -> tuple[E18Job, ...]:
    """Deterministic registered job order: references first, then the grid."""
    jobs = [E18Job(j.construction_seed, j.arm, j.pipeline_seed, j.role)
            for j in e17train.build_training_grid()]
    if len(jobs) != EXPECTED_FIT_COUNT:
        raise _stop(f"grid size {len(jobs)} != registered {EXPECTED_FIT_COUNT}")
    if len({job.key for job in jobs}) != len(jobs):
        raise _stop("duplicate job key in the training grid")
    return tuple(jobs)


build_pool_groups = e17train.build_pool_groups


def record_paths() -> dict[str, Path]:
    return {
        "pool_inheritance": EXPERIMENT_DIR
        / f"pool_inheritance_{OUTPUT_VERSION}.json",
        "pool_audit": TABLE_DIR
        / f"e18_pool_inheritance_audit_{OUTPUT_VERSION}.csv",
        "visibility_record": EXPERIMENT_DIR
        / f"manipulation_visibility_{OUTPUT_VERSION}.json",
        "visibility": TABLE_DIR
        / f"e18_manipulation_visibility_{OUTPUT_VERSION}.csv",
        "visibility_record_v2": EXPERIMENT_DIR
        / "manipulation_visibility_v2.json",
        "visibility_v2": TABLE_DIR / "e18_manipulation_visibility_v2.csv",
        "preflight": EXPERIMENT_DIR
        / f"training_preflight_{OUTPUT_VERSION}.json",
        "run": EXPERIMENT_DIR / f"training_run_{OUTPUT_VERSION}.json",
        "manifest": TABLE_DIR / f"e18_training_manifest_{OUTPUT_VERSION}.csv",
        "sampling_audit": TABLE_DIR
        / f"e18_sampling_audit_{OUTPUT_VERSION}.csv",
    }


def training_target_paths(jobs: Sequence[E18Job]) -> list[Path]:
    records = record_paths()
    paths = [job.checkpoint_path for job in jobs]
    paths += [job.log_path for job in jobs]
    paths += [records[key] for key in
              ("preflight", "run", "manifest", "sampling_audit")]
    return paths


def assert_targets_absent(paths: Sequence[Path]) -> dict[str, Any]:
    """Gate G10a — no-clobber preflight, mirroring the frozen RF harness."""
    present = [str(p.relative_to(REPO)) for p in paths
               if p.exists() or p.is_symlink()]
    if present:
        raise _stop(f"refusing to overwrite existing targets: {present[:8]}"
                    + (f" (+{len(present) - 8} more)" if len(present) > 8
                       else ""))
    return {"gate": "G10a_no_clobber", "passed": True, "checked": len(paths)}


# ---------------------------------------------------------------------------
# Gate G14 — feature-map and estimator contract fidelity
# ---------------------------------------------------------------------------


def verify_feature_contract() -> dict[str, Any]:
    """Gate G14 — the 83-dimensional map is exactly what Section 5.1 registers."""
    if rf_features is not rfmod.rf_features:
        raise _stop(
            "the runner's marginal feature function is not the imported "
            "train_generator_extension_rf.rf_features object")
    if CORRELATION_PAIRS != tuple(
            itertools.combinations(range(PAYLOAD_CHANNELS), 2)):
        raise _stop("correlation pair order drifted from PREREG section 5.1")

    rng = np.random.default_rng(0)
    probe = rng.integers(0, 256, size=(7, 128, 11)).astype(np.float32)
    #: a window whose payload bytes are all constant exercises the registered
    #: constant-channel convention.
    probe[3, :, PAYLOAD_CHANNEL_SLICE] = 17.0
    #: an exactly collinear pair must return +/- 1 and pin the pair ordering.
    probe[4, :, 2] = np.arange(128, dtype=np.float32)
    probe[4, :, 3] = 3.0 * np.arange(128, dtype=np.float32) + 5.0
    probe[4, :, 9] = -2.0 * np.arange(128, dtype=np.float32)

    features = e18_features(probe)
    if features.shape != (len(probe), FEATURE_DIMENSION):
        raise _stop(f"realized feature shape {features.shape}")
    if features.dtype != np.float32:
        raise _stop(f"feature dtype {features.dtype} is not float32")
    if not np.isfinite(features).all():
        raise _stop("the feature map produced a non-finite value")

    marginal = rf_features(probe)
    if not np.array_equal(features[:, :MARGINAL_DIMENSION], marginal):
        raise _stop(
            "the first 55 columns are not bit-identical to the imported "
            "rf_features output")
    correlations = features[:, MARGINAL_DIMENSION:]
    if not np.array_equal(correlations[3], np.zeros(CORRELATION_DIMENSION,
                                                    dtype=np.float32)):
        raise _stop(
            "the registered constant-channel convention did not return 0.0 for "
            "an all-constant payload window")
    pair_index = {pair: k for k, pair in enumerate(CORRELATION_PAIRS)}
    if abs(float(correlations[4, pair_index[(0, 1)]]) - 1.0) > 1e-5:
        raise _stop("perfectly collinear payload pair (data0, data1) != +1")
    if abs(float(correlations[4, pair_index[(0, 7)]]) + 1.0) > 1e-5:
        raise _stop("perfectly anticollinear payload pair (data0, data7) != -1")

    upstream = e17train._upstream_estimator_keywords()
    mismatched = {
        key: (ESTIMATOR[key], upstream.get(key, "<absent>"))
        for key in ESTIMATOR_FIDELITY_KEYS
        if upstream.get(key, "<absent>") != ESTIMATOR[key]
    }
    if mismatched:
        raise _stop(
            f"frozen RF harness estimator differs from PREREG section 5.2: "
            f"{mismatched}")
    if upstream.get("random_state") != "seed":
        raise _stop(
            "frozen RF harness does not seed random_state from the pipeline "
            f"seed: random_state={upstream.get('random_state')!r}")
    return {
        "gate": "G14_feature_map_and_estimator_contract",
        "passed": True,
        "marginal_feature_function": f"{rfmod.__name__}.rf_features",
        "marginal_feature_function_is_imported_object": True,
        "marginal_dimension": MARGINAL_DIMENSION,
        "correlation_dimension": CORRELATION_DIMENSION,
        "feature_dimension": FEATURE_DIMENSION,
        "correlation_pairs": [list(p) for p in CORRELATION_PAIRS],
        "payload_channels": "data0..data7 (raw channel indices 2..9)",
        "constant_channel_convention": 0.0,
        "features_computed_on_raw_windows": True,
        "standardizer_constructed": False,
        "estimator": {k: ESTIMATOR[k] for k in ESTIMATOR_FIDELITY_KEYS},
        "estimator_matches_frozen_harness_source": True,
        "upstream_keywords": upstream,
    }


# ---------------------------------------------------------------------------
# Stage S2 — pool inheritance verification (machinery inherited from E17)
# ---------------------------------------------------------------------------


def run_pool_verification() -> dict[str, Any]:
    """Stage S2.  No fitting occurs."""
    started = time.monotonic()
    records = record_paths()
    assert_targets_absent([records["pool_inheritance"], records["pool_audit"]])
    record: dict[str, Any] = {
        "schema_version": POOL_INHERITANCE_SCHEMA,
        "stage": "e18_pool_inheritance_verification",
        "started_utc": utc_now(),
        "model_fitting_performed": False,
        "registration": assert_registration_committed(),
        "environment": _environment_record(REPO),
        "seed_axis": e17train.verify_seed_axis(),
        "machinery": (
            "gates G1i, G2i, G3i-G8i are the E17 implementations, imported "
            "from journal/scripts/run_e17_rf_paired_training.py and not "
            "re-implemented; the consumed pools and their reference digests are "
            "identical to E17's"),
    }
    rows, gate = e17train.verify_pool_inheritance()
    record["pool_hash_inheritance"] = gate
    record["completed_utc"] = utc_now()
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record["status"] = "T-PASS"
    publish_and_cleanup([
        (_stage_bytes(records["pool_audit"], _csv_bytes(rows)),
         records["pool_audit"]),
        (_stage_json(records["pool_inheritance"], record),
         records["pool_inheritance"]),
    ])
    return record


# ---------------------------------------------------------------------------
# Stage S2b — gate G17, the manipulation-visibility gate (PREREG section 15.5)
# ---------------------------------------------------------------------------


def measure_pool_pair_visibility(construction_seed: int, role: str
                                 ) -> dict[str, Any]:
    """How much of the placebo manipulation each feature map can perceive.

    Two metrics per map, both published (PREREG section 15.5 as amended
    2026-08-07 v1):

    * *mechanical* — exact ``float32`` inequality over every feature, i.e. the
      values the estimator actually consumes;
    * *substantive* — restricted to the features whose exact-arithmetic value
      can change under the manipulation, so that a ``float32`` reduction
      artifact in ``mean``/``std`` of the manipulated channel cannot inflate it.
    """
    rule_path = E18Job(construction_seed, "rule", PIPELINE_SEEDS[0],
                       role).pool_path
    placebo_path = E18Job(construction_seed, "placebo", PIPELINE_SEEDS[0],
                          role).pool_path
    with np.load(rule_path, allow_pickle=False) as pool:
        rule_x = np.asarray(pool["x"], dtype=np.float32)
    with np.load(placebo_path, allow_pickle=False) as pool:
        placebo_x = np.asarray(pool["x"], dtype=np.float32)
    if rule_x.shape != placebo_x.shape:
        raise _stop(
            f"{construction_seed}: arm pools are not aligned "
            f"{rule_x.shape} vs {placebo_x.shape}")

    raw_differing = np.any(rule_x != placebo_x, axis=(1, 2))
    raw_channels = np.flatnonzero(np.any(rule_x != placebo_x, axis=(0, 1)))
    if raw_channels.tolist() != [MANIPULATED_RAW_CHANNEL]:
        raise _stop(
            f"{construction_seed}: raw channels {raw_channels.tolist()} differ "
            f"between the arms; the registered manipulation touches only "
            f"channel {MANIPULATED_RAW_CHANNEL} (data1)")
    windows = int(len(rule_x))
    raw_differing_count = int(np.count_nonzero(raw_differing))

    row: dict[str, Any] = {
        "construction_seed": construction_seed,
        "role": role,
        "rule_pool": str(rule_path.relative_to(REPO)),
        "placebo_pool": str(placebo_path.relative_to(REPO)),
        "windows": windows,
        "raw_differing_windows": raw_differing_count,
    }
    for label, dimension, function in (
            ("e18_83", FEATURE_DIMENSION, e18_features),
            ("e17_55", MARGINAL_DIMENSION, rf_features),
    ):
        rule_features = function(rule_x)
        placebo_features = function(placebo_x)
        unequal = rule_features != placebo_features
        per_feature = np.any(unequal, axis=0)
        per_window = np.any(unequal, axis=1)
        leaked = int(np.count_nonzero(per_window & ~raw_differing))
        if leaked:
            raise _stop(
                f"{construction_seed}: {leaked} window(s) with identical raw "
                f"data produced differing {label} features")
        differing = np.flatnonzero(per_feature)
        inadmissible = [int(k) for k in differing
                        if int(k) not in ADMISSIBLE_FEATURE_INDICES]
        if inadmissible:
            raise _stop(
                f"{construction_seed}: {label} feature index/indices "
                f"{inadmissible} differ between the arms but are provably "
                "invariant under the registered manipulation; PREREG section "
                "5.1 as amended makes this an implementation error")
        substantive = [int(k) for k in differing
                       if int(k) in SUBSTANTIVE_FEATURE_INDICES]
        artifact = [int(k) for k in differing
                    if int(k) in ARTIFACT_FEATURE_INDICES]
        substantive_window = (
            np.any(unequal[:, substantive], axis=1) if substantive
            else np.zeros(len(unequal), dtype=bool))
        artifact_max = 0.0
        for k in artifact:
            artifact_max = max(artifact_max, float(np.abs(
                rule_features[:, k].astype(np.float64)
                - placebo_features[:, k].astype(np.float64)).max()))
        visible = int(np.count_nonzero(per_window & raw_differing))
        substantive_visible = int(
            np.count_nonzero(substantive_window & raw_differing))
        row[f"{label}_features"] = int(dimension)
        row[f"{label}_features_ever_differing"] = int(len(differing))
        row[f"{label}_differing_feature_indices"] = ";".join(
            str(int(k)) for k in differing)
        row[f"{label}_substantive_features_ever_differing"] = len(substantive)
        row[f"{label}_substantive_feature_indices"] = ";".join(
            str(k) for k in substantive)
        row[f"{label}_artifact_features_ever_differing"] = len(artifact)
        row[f"{label}_artifact_feature_indices"] = ";".join(
            str(k) for k in artifact)
        row[f"{label}_artifact_max_abs_difference"] = artifact_max
        row[f"{label}_visible_windows"] = visible
        row[f"{label}_visible_fraction"] = (
            visible / raw_differing_count if raw_differing_count else 0.0)
        row[f"{label}_substantive_visible_windows"] = substantive_visible
        row[f"{label}_substantive_visible_fraction"] = (
            substantive_visible / raw_differing_count
            if raw_differing_count else 0.0)
        del rule_features, placebo_features, unequal
    del rule_x, placebo_x
    return row


def run_manipulation_visibility() -> dict[str, Any]:
    """Stage S2b.  Publishes the G17 table regardless of outcome, then gates."""
    started = time.monotonic()
    records = record_paths()
    assert_targets_absent(
        [records["visibility"], records["visibility_record"]])
    if not records["pool_audit"].exists():
        raise _stop(
            "stage S2 has not run: missing "
            f"{records['pool_audit'].relative_to(REPO)}")

    rows: list[dict[str, Any]] = []
    slots = ([(g, "primary") for g in CONSTRUCTION_SEEDS]
             + [(g, "bridge_sensitivity") for g in BRIDGE_SEEDS])
    for position, (construction_seed, role) in enumerate(slots, start=1):
        row = measure_pool_pair_visibility(construction_seed, role)
        rows.append(row)
        print(f"  [{position}/{len(slots)}] {construction_seed} ({role}): "
              f"83-dim sees {row['e18_83_visible_fraction']:.6f} "
              f"(substantive {row['e18_83_substantive_visible_fraction']:.6f}) "
              f"of {row['raw_differing_windows']} altered windows via "
              f"{row['e18_83_substantive_features_ever_differing']} "
              f"substantive feature(s); 55-dim sees "
              f"{row['e17_55_visible_fraction']:.6f} "
              f"(substantive {row['e17_55_substantive_visible_fraction']:.6f})",
              flush=True)

    primary = [r for r in rows if r["role"] == "primary"]
    pooled: dict[str, Any] = {"construction_seed": "", "role": "pooled_primary",
                              "rule_pool": "", "placebo_pool": ""}
    pooled["windows"] = sum(int(r["windows"]) for r in primary)
    pooled["raw_differing_windows"] = sum(
        int(r["raw_differing_windows"]) for r in primary)
    denominator = pooled["raw_differing_windows"]
    for label, dimension in (("e18_83", FEATURE_DIMENSION),
                             ("e17_55", MARGINAL_DIMENSION)):
        pooled[f"{label}_features"] = int(dimension)
        for kind, key in (("", "differing_feature_indices"),
                          ("substantive_", "substantive_feature_indices"),
                          ("artifact_", "artifact_feature_indices")):
            indices = sorted({
                int(i) for r in primary
                for i in str(r[f"{label}_{key}"]).split(";") if i != ""})
            pooled[f"{label}_{kind}features_ever_differing"] = len(indices)
            pooled[f"{label}_{key}"] = ";".join(str(i) for i in indices)
        pooled[f"{label}_artifact_max_abs_difference"] = max(
            float(r[f"{label}_artifact_max_abs_difference"]) for r in primary)
        for kind in ("", "substantive_"):
            visible = sum(
                int(r[f"{label}_{kind}visible_windows"]) for r in primary)
            pooled[f"{label}_{kind}visible_windows"] = visible
            pooled[f"{label}_{kind}visible_fraction"] = (
                visible / denominator if denominator else 0.0)
    rows.append(pooled)

    metrics: dict[str, Any] = {}
    passed = True
    for metric, column in (("mechanical", "e18_83_visible_fraction"),
                           ("substantive",
                            "e18_83_substantive_visible_fraction")):
        pooled_value = float(pooled[column])
        per_pair = [float(r[column]) for r in primary]
        pooled_ok = pooled_value >= G17_POOLED_FLOOR
        per_pair_ok = min(per_pair) >= G17_PER_PAIR_FLOOR
        passed = passed and pooled_ok and per_pair_ok
        metrics[metric] = {
            "pooled_visible_fraction": pooled_value,
            "pooled_floor_passed": bool(pooled_ok),
            "minimum_per_pair_visible_fraction": min(per_pair),
            "maximum_per_pair_visible_fraction": max(per_pair),
            "per_pair_floor_passed": bool(per_pair_ok),
            "e17_55_pooled_visible_fraction": float(pooled[
                column.replace("e18_83", "e17_55")]),
        }
    passed = bool(passed)

    gate = {
        "gate": "G17_manipulation_visibility",
        "passed": passed,
        "pooled_floor": G17_POOLED_FLOOR,
        "per_pair_floor": G17_PER_PAIR_FLOOR,
        "both_metrics_required": True,
        "metrics": metrics,
        # The registered headline figures are the substantive ones: they are
        # the metric a float32 reduction artifact cannot inflate.
        "pooled_visible_fraction": metrics["substantive"][
            "pooled_visible_fraction"],
        "pooled_mechanical_visible_fraction": metrics["mechanical"][
            "pooled_visible_fraction"],
        "minimum_per_pair_visible_fraction": metrics["substantive"][
            "minimum_per_pair_visible_fraction"],
        "primary_pairs": len(primary),
        "pooled_raw_differing_windows": int(denominator),
        "pooled_visible_windows": int(
            pooled["e18_83_substantive_visible_windows"]),
        "features_ever_differing": int(
            pooled["e18_83_features_ever_differing"]),
        "substantive_features_ever_differing": int(
            pooled["e18_83_substantive_features_ever_differing"]),
        "artifact_features_ever_differing": int(
            pooled["e18_83_artifact_features_ever_differing"]),
        "artifact_max_abs_difference": float(
            pooled["e18_83_artifact_max_abs_difference"]),
        "admissible_feature_indices": {
            "substantive": list(SUBSTANTIVE_FEATURE_INDICES),
            "artifact": list(ARTIFACT_FEATURE_INDICES),
        },
        "e17_55_substantive_features_ever_differing": int(
            pooled["e17_55_substantive_features_ever_differing"]),
        "e17_55_pooled_substantive_visible_fraction": float(
            pooled["e17_55_substantive_visible_fraction"]),
        "e17_55_pooled_mechanical_visible_fraction": float(
            pooled["e17_55_visible_fraction"]),
        "e17_reference_visible_fraction_in_prereg":
            E17_REFERENCE_VISIBLE_FRACTION,
        "e17_reference_substantive_features_in_prereg":
            E17_REFERENCE_SUBSTANTIVE_FEATURES,
        "improvement_factor_over_e17": (
            float(pooled["e18_83_substantive_visible_fraction"])
            / float(pooled["e17_55_substantive_visible_fraction"])
            if float(pooled["e17_55_substantive_visible_fraction"]) > 0
            else None),
        "amendment": AMENDMENT_PATH,
        "branch_e_if_failed": (
            "inconclusive-by-construction; no fit is run and no inferential "
            "claim is made (PREREG sections 13 and 15.5)"),
    }

    record = {
        "schema_version": VISIBILITY_SCHEMA,
        "stage": "e18_manipulation_visibility",
        "started_utc": utc_now(),
        "model_fitting_performed": False,
        "registration": assert_registration_committed(),
        "environment": _environment_record(REPO),
        "feature_contract": verify_feature_contract(),
        "measurement": (
            "per pool pair: W = windows whose raw x differs; V = those windows "
            "whose float32 feature vector differs; visibility = V / W.  Two "
            "metrics are published: mechanical, over all features by exact "
            "float32 inequality (the values the estimator consumes), and "
            "substantive, restricted to the features whose exact-arithmetic "
            "value can change under the manipulation.  Both must clear both "
            "floors (PREREG section 15.5 as amended 2026-08-07 v1)."),
        "visibility_gate": gate,
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": "T-PASS" if passed else "BRANCH-E",
    }
    # Published before the floor is enforced, so a failing measurement cannot
    # be discarded (PREREG section 15.5).
    publish_and_cleanup([
        (_stage_bytes(records["visibility"], _csv_bytes(rows)),
         records["visibility"]),
        (_stage_json(records["visibility_record"], record),
         records["visibility_record"]),
    ])
    if not passed:
        raise E18BranchE(
            f"BRANCH-E-E18: gate G17 floor not met — substantive pooled "
            f"visibility "
            f"{metrics['substantive']['pooled_visible_fraction']:.6f} / "
            f"mechanical "
            f"{metrics['mechanical']['pooled_visible_fraction']:.6f} against "
            f"the pooled floor {G17_POOLED_FLOOR}; minimum per-pair "
            f"{metrics['substantive']['minimum_per_pair_visible_fraction']:.6f}"
            f" / {metrics['mechanical']['minimum_per_pair_visible_fraction']:.6f}"
            f" against the per-pair floor {G17_PER_PAIR_FLOOR}. E18 is "
            f"inconclusive-by-construction; the table is published at "
            f"{records['visibility'].relative_to(REPO)} and no fit is run.")
    return record


# ---------------------------------------------------------------------------
# Stage S2b-v2 — float64 and per-attack visibility (amendment 2026-08-07 v2)
# ---------------------------------------------------------------------------


#: ``lib_common.CLASS_NAMES`` — the two spoof attacks the manipulation touches.
ATTACK_LABELS = (("Gear", 3), ("RPM", 4))
#: (metric label, feature-index restriction, precision)
V2_METRICS = ("mechanical_f32", "substantive_f32", "float64")


def _visibility_counts(unequal: np.ndarray, raw_differing: np.ndarray,
                       attack: np.ndarray) -> dict[str, int]:
    """Visible-window counts for the pooled scope and each attack scope."""
    per_window = np.any(unequal, axis=1) & raw_differing
    counts = {"all": int(np.count_nonzero(per_window))}
    for name, label in ATTACK_LABELS:
        counts[name] = int(np.count_nonzero(per_window & (attack == label)))
    return counts


def measure_pool_pair_visibility_v2(construction_seed: int, role: str
                                    ) -> dict[str, Any]:
    """Three metrics x three scopes per feature map, for one pool pair."""
    rule_path = E18Job(construction_seed, "rule", PIPELINE_SEEDS[0],
                       role).pool_path
    placebo_path = E18Job(construction_seed, "placebo", PIPELINE_SEEDS[0],
                          role).pool_path
    with np.load(rule_path, allow_pickle=False) as pool:
        rule_x = np.asarray(pool["x"], dtype=np.float32)
        attack = np.asarray(pool["y_attack_type"], dtype=np.int64)
    with np.load(placebo_path, allow_pickle=False) as pool:
        placebo_x = np.asarray(pool["x"], dtype=np.float32)
    raw_differing = np.any(rule_x != placebo_x, axis=(1, 2))

    row: dict[str, Any] = {
        "construction_seed": construction_seed,
        "role": role,
        "windows": int(len(rule_x)),
        "raw_differing_windows": int(np.count_nonzero(raw_differing)),
    }
    for name, label in ATTACK_LABELS:
        row[f"raw_differing_windows_{name}"] = int(
            np.count_nonzero(raw_differing & (attack == label)))

    for map_label, f32, f64, dimension in (
            ("e18_83", e18_features, e18_features_float64, FEATURE_DIMENSION),
            ("e17_55", rf_features,
             lambda x: marginal_features_float64(x), MARGINAL_DIMENSION),
    ):
        unequal32 = f32(rule_x) != f32(placebo_x)
        unequal64 = f64(rule_x) != f64(placebo_x)
        substantive = [k for k in range(dimension)
                       if k in SUBSTANTIVE_FEATURE_INDICES]
        restricted = np.zeros_like(unequal32)
        if substantive:
            restricted[:, substantive] = unequal32[:, substantive]
        row[f"{map_label}_features"] = int(dimension)
        for metric, unequal in (("mechanical_f32", unequal32),
                                ("substantive_f32", restricted),
                                ("float64", unequal64)):
            differing = np.flatnonzero(np.any(unequal, axis=0))
            row[f"{map_label}_{metric}_features_ever_differing"] = int(
                len(differing))
            row[f"{map_label}_{metric}_feature_indices"] = ";".join(
                str(int(k)) for k in differing)
            counts = _visibility_counts(unequal, raw_differing, attack)
            for scope, visible in counts.items():
                denominator = row[
                    "raw_differing_windows" if scope == "all"
                    else f"raw_differing_windows_{scope}"]
                row[f"{map_label}_{metric}_{scope}_visible_windows"] = visible
                row[f"{map_label}_{metric}_{scope}_visible_fraction"] = (
                    visible / denominator if denominator else 0.0)
        del unequal32, unequal64, restricted
    del rule_x, placebo_x
    return row


def run_manipulation_visibility_v2() -> dict[str, Any]:
    """Stage S2b-v2.  Publishes regardless of outcome, then applies the floor."""
    started = time.monotonic()
    records = record_paths()
    assert_targets_absent(
        [records["visibility_v2"], records["visibility_record_v2"]])
    if not records["visibility"].exists():
        raise _stop(
            "the v1 visibility measurement has not run: missing "
            f"{records['visibility'].relative_to(REPO)}")

    rows: list[dict[str, Any]] = []
    slots = ([(g, "primary") for g in CONSTRUCTION_SEEDS]
             + [(g, "bridge_sensitivity") for g in BRIDGE_SEEDS])
    for position, (construction_seed, role) in enumerate(slots, start=1):
        row = measure_pool_pair_visibility_v2(construction_seed, role)
        rows.append(row)
        print(f"  [{position}/{len(slots)}] {construction_seed} ({role}): "
              f"83-dim float64 all="
              f"{row['e18_83_float64_all_visible_fraction']:.6f} "
              f"Gear={row['e18_83_float64_Gear_visible_fraction']:.6f} "
              f"RPM={row['e18_83_float64_RPM_visible_fraction']:.6f}; "
              f"55-dim float64 all="
              f"{row['e17_55_float64_all_visible_fraction']:.6f} "
              f"Gear={row['e17_55_float64_Gear_visible_fraction']:.6f} "
              f"RPM={row['e17_55_float64_RPM_visible_fraction']:.6f}",
              flush=True)

    primary = [r for r in rows if r["role"] == "primary"]
    pooled: dict[str, Any] = {"construction_seed": "", "role": "pooled_primary"}
    for key in ("windows", "raw_differing_windows",
                *[f"raw_differing_windows_{n}" for n, _ in ATTACK_LABELS]):
        pooled[key] = sum(int(r[key]) for r in primary)
    for map_label, dimension in (("e18_83", FEATURE_DIMENSION),
                                 ("e17_55", MARGINAL_DIMENSION)):
        pooled[f"{map_label}_features"] = int(dimension)
        for metric in V2_METRICS:
            indices = sorted({
                int(i) for r in primary
                for i in str(r[f"{map_label}_{metric}_feature_indices"]
                             ).split(";") if i != ""})
            pooled[f"{map_label}_{metric}_features_ever_differing"] = len(
                indices)
            pooled[f"{map_label}_{metric}_feature_indices"] = ";".join(
                str(i) for i in indices)
            for scope in ("all", *[n for n, _ in ATTACK_LABELS]):
                visible = sum(
                    int(r[f"{map_label}_{metric}_{scope}_visible_windows"])
                    for r in primary)
                denominator = pooled[
                    "raw_differing_windows" if scope == "all"
                    else f"raw_differing_windows_{scope}"]
                pooled[f"{map_label}_{metric}_{scope}_visible_windows"] = visible
                pooled[f"{map_label}_{metric}_{scope}_visible_fraction"] = (
                    visible / denominator if denominator else 0.0)
    rows.append(pooled)

    metrics: dict[str, Any] = {}
    passed = True
    for metric in V2_METRICS:
        column = f"e18_83_{metric}_all_visible_fraction"
        per_pair = [float(r[column]) for r in primary]
        pooled_value = float(pooled[column])
        pooled_ok = pooled_value >= G17_POOLED_FLOOR
        per_pair_ok = min(per_pair) >= G17_PER_PAIR_FLOOR
        passed = passed and pooled_ok and per_pair_ok
        metrics[metric] = {
            "pooled_visible_fraction": pooled_value,
            "pooled_floor_passed": bool(pooled_ok),
            "minimum_per_pair_visible_fraction": min(per_pair),
            "maximum_per_pair_visible_fraction": max(per_pair),
            "per_pair_floor_passed": bool(per_pair_ok),
            "features_ever_differing": int(
                pooled[f"e18_83_{metric}_features_ever_differing"]),
            "e17_55_pooled_visible_fraction": float(
                pooled[f"e17_55_{metric}_all_visible_fraction"]),
            "e17_55_features_ever_differing": int(
                pooled[f"e17_55_{metric}_features_ever_differing"]),
        }
    passed = bool(passed)

    per_attack: dict[str, Any] = {}
    for map_label in ("e18_83", "e17_55"):
        for metric in V2_METRICS:
            entry = {
                scope: float(
                    pooled[f"{map_label}_{metric}_{scope}_visible_fraction"])
                for scope in ("all", *[n for n, _ in ATTACK_LABELS])
            }
            entry["Gear_minus_RPM"] = entry["Gear"] - entry["RPM"]
            per_attack[f"{map_label}_{metric}"] = entry
    headline = per_attack["e18_83_float64"]
    differential = abs(headline["Gear_minus_RPM"]) >= 0.05

    gate = {
        "gate": "G17_manipulation_visibility_v2",
        "passed": passed,
        "amendment": AMENDMENTS[1][0],
        "precision_note": (
            "the float64 metric is exact for this data: the raw channels are "
            "integers in [0, 255], so every partial sum of the 128-frame "
            "reduction is exactly representable in float64 and std is exactly "
            "permutation-invariant, which is why the float32 reduction "
            "artifact of amendment v1 does not arise here"),
        "pooled_floor": G17_POOLED_FLOOR,
        "per_pair_floor": G17_PER_PAIR_FLOOR,
        "all_three_metrics_required": True,
        "metrics": metrics,
        "per_attack_visible_fraction": per_attack,
        "per_attack_differential": differential,
        "per_attack_interpretive_constraint": (
            "amendment v2 section 2.2: no Gear-versus-RPM difference in any E18 "
            "outcome may be attributed to differential visibility unless these "
            "figures differ materially between the attacks"
            + ("; they do NOT, so any Gear/RPM asymmetry must be reported as "
               "dispersion or as manipulation completeness, never as the "
               "detector seeing one attack better than the other"
               if not differential else
               "; they DO differ, so the attribution may be examined")),
        "float64_mirror": assert_float64_mirror_fidelity(),
        "branch_e_if_failed": (
            "inconclusive-by-construction; the fits are reported as "
            "uninterpretable and no inferential claim is made"),
    }

    record = {
        "schema_version": "e18.manipulation_visibility.v2",
        "stage": "e18_manipulation_visibility_v2",
        "started_utc": utc_now(),
        "model_fitting_performed": False,
        "supersedes": None,
        "relation_to_v1": (
            "additive; the v1 table and record are preserved verbatim and are "
            "not recomputed"),
        "registration": assert_registration_committed(),
        "environment": _environment_record(REPO),
        "measurement": (
            "per pool pair and per attack: W = windows whose raw x differs; "
            "V = those whose feature vector differs.  Three metrics: "
            "mechanical_f32 (exact float32 inequality over all features, the "
            "values the estimator consumes), substantive_f32 (restricted to "
            "the features whose exact-arithmetic value can change), and "
            "float64 (the whole map evaluated in float64).  All three must "
            "clear both registered floors on the pooled scope; the per-attack "
            "figures are reported, not gated."),
        "visibility_gate": gate,
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": "T-PASS" if passed else "BRANCH-E",
    }
    publish_and_cleanup([
        (_stage_bytes(records["visibility_v2"], _csv_bytes(rows)),
         records["visibility_v2"]),
        (_stage_json(records["visibility_record_v2"], record),
         records["visibility_record_v2"]),
    ])
    if not passed:
        raise E18BranchE(
            "BRANCH-E-E18: gate G17 v2 floor not met; "
            f"{json.dumps(metrics, sort_keys=True)}")
    return record


# ---------------------------------------------------------------------------
# Preflight (stages S3/S4)
# ---------------------------------------------------------------------------


def build_preflight() -> dict[str, Any]:
    jobs = build_training_grid()
    records = record_paths()
    for key, stage in (("pool_audit", "S2"), ("visibility", "S2b")):
        if not records[key].exists():
            raise _stop(
                f"stage {stage} has not run: missing "
                f"{records[key].relative_to(REPO)}")
    visibility = json.loads(records["visibility_record"].read_text())
    if not visibility["visibility_gate"]["passed"]:
        raise _stop(
            "gate G17 did not pass; E18 is inconclusive-by-construction and no "
            "fit may be run (PREREG section 15.5)")
    return {
        "schema_version": TRAINING_PREFLIGHT_SCHEMA,
        "stage": "e18_relation_visible_rf_training",
        "built_utc": utc_now(),
        "expected_fits": EXPECTED_FIT_COUNT,
        "grid": {
            "generator_realizations": len(CONSTRUCTION_SEEDS),
            "bridge_realizations": len(BRIDGE_SEEDS),
            "arms": list(ARMS),
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "primary_fits": EXPECTED_PRIMARY_FITS,
            "bridge_fits": EXPECTED_BRIDGE_FITS,
            "reference_fits": EXPECTED_REFERENCE_FITS,
        },
        "protocol": {
            "detector_family": "random_forest",
            "feature_map": "joint_structure_83",
            "matching_contract": "identical_training_rows",
            "optimizer_update_budget": None,
            "early_stopping": None,
            "minibatch_order": None,
            "standardizer": None,
            "synthetic_total": SYNTHETIC_TOTAL,
            "sampling_seed_offset": SAMPLING_SEED_OFFSET,
            "per_attack_cap": PER_ATTACK_CAP,
            "estimator": ESTIMATOR,
            "feature_dimension": FEATURE_DIMENSION,
        },
        "registration": assert_registration_committed(),
        "environment": _environment_record(REPO),
        "seed_axis": e17train.verify_seed_axis(),
        "feature_contract": verify_feature_contract(),
        "pool_inheritance_record": {
            "path": str(records["pool_inheritance"].relative_to(REPO)),
            "sha256": _sha256_file(records["pool_inheritance"]),
            "audit_path": str(records["pool_audit"].relative_to(REPO)),
            "audit_sha256": _sha256_file(records["pool_audit"]),
        },
        "manipulation_visibility_record": {
            "path": str(records["visibility_record"].relative_to(REPO)),
            "sha256": _sha256_file(records["visibility_record"]),
            "table_path": str(records["visibility"].relative_to(REPO)),
            "table_sha256": _sha256_file(records["visibility"]),
            "gate": visibility["visibility_gate"],
        },
        "e16_training_manifest": {
            "path": str(e17train.E16_TRAINING_MANIFEST.relative_to(REPO)),
            "sha256": _sha256_file(e17train.E16_TRAINING_MANIFEST),
        },
        "target_absence": assert_targets_absent(training_target_paths(jobs)),
    }


# ---------------------------------------------------------------------------
# Child fits — one pool, five pipeline seeds
# ---------------------------------------------------------------------------


def _real_training_rows() -> tuple[np.ndarray, np.ndarray]:
    real_x, real_y, _val_x, _val_y = _load_real_train_val(REPO)
    features = e18_features(real_x)
    del real_x
    return features, real_y.astype(np.int64)


def _fit_and_publish(
        job: E18Job,
        train_features: np.ndarray,
        train_y: np.ndarray,
        *,
        pool_record: Mapping[str, Any],
        sampling: Mapping[str, Any] | None,
        index_digest: str,
) -> dict[str, Any]:
    """Fit one Random Forest and publish its bundle atomically."""
    from sklearn.ensemble import RandomForestClassifier

    assert_targets_absent([job.checkpoint_path, job.log_path])
    if train_features.shape[1] != FEATURE_DIMENSION:
        raise _stop(
            f"{job.key}: feature dimension {train_features.shape[1]} != "
            f"{FEATURE_DIMENSION}")
    if len(train_features) != len(train_y):
        raise _stop(f"{job.key}: feature/label row mismatch")

    started = utc_now()
    start_time = time.monotonic()
    order = np.random.default_rng(job.pipeline_seed).permutation(len(train_y))
    forest = RandomForestClassifier(
        n_estimators=ESTIMATOR["n_estimators"],
        max_depth=ESTIMATOR["max_depth"],
        min_samples_leaf=ESTIMATOR["min_samples_leaf"],
        class_weight=ESTIMATOR["class_weight"],
        n_jobs=ESTIMATOR["n_jobs"],
        random_state=job.pipeline_seed,
    )
    forest.fit(train_features[order], train_y[order])
    elapsed = round(time.monotonic() - start_time, 3)

    realized = {
        "n_estimators": int(forest.n_estimators),
        "max_depth": forest.max_depth,
        "min_samples_leaf": int(forest.min_samples_leaf),
        "class_weight": forest.class_weight,
        "random_state": int(forest.random_state),
        "n_features_in": int(forest.n_features_in_),
        "classes": [int(c) for c in forest.classes_],
    }
    for key in ESTIMATOR_FIDELITY_KEYS:
        if realized[key] != ESTIMATOR[key]:
            raise _stop(
                f"{job.key}: realized {key}={realized[key]!r} != registered "
                f"{ESTIMATOR[key]!r}")
    if realized["n_features_in"] != FEATURE_DIMENSION:
        raise _stop(f"{job.key}: fitted on {realized['n_features_in']} features")
    if realized["random_state"] != job.pipeline_seed:
        raise _stop(f"{job.key}: random_state is not the pipeline seed")

    log_record = {
        "schema_version": TRAINING_LOG_SCHEMA,
        "experiment": "e18_relation_visible_rf",
        "family": "rf",
        "feature_map": "joint_structure_83",
        "setting": job.setting,
        "construction_seed": job.construction_seed,
        "arm": job.arm,
        "pipeline_seed": job.pipeline_seed,
        "role": job.role,
        "model_tag": MODEL_TAG,
        "pool": pool_record,
        "sampling": sampling,
        "sampling_index_sha256": index_digest,
        "permutation_seed": job.pipeline_seed,
        "permutation": "numpy.random.default_rng(seed).permutation(len(y))",
        "standardizer": None,
        "features_on_raw_windows": True,
        "feature_dimension": FEATURE_DIMENSION,
        "estimator": realized,
        "train_windows": int(len(train_y)),
        "device": "cpu",
        "started_utc": started,
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "checkpoint": str(job.checkpoint_path.relative_to(REPO)),
    }

    staged_checkpoint = _dump_forest(forest, job.checkpoint_path.parent,
                                     job.checkpoint_path.name)
    staged_log = _stage_json(job.log_path, log_record)
    publish_and_cleanup([
        (staged_checkpoint, job.checkpoint_path),
        (staged_log, job.log_path),
    ])
    log_record["checkpoint_sha256"] = _sha256_file(job.checkpoint_path)
    print(f"  fit {job.key} rows={len(train_y)} {elapsed:.0f}s", flush=True)
    return log_record


def run_child_group(construction_seed: int, arm: str, role: str
                    ) -> list[dict[str, Any]]:
    """Fit one pool's five pipeline seeds, extracting features once."""
    lc.fit_standardizer = _forbidden_standardizer  # gate G14
    verify_feature_contract()

    real_features, real_y = _real_training_rows()
    jobs = [E18Job(construction_seed, arm, pipeline_seed, role)
            for pipeline_seed in PIPELINE_SEEDS]
    assert_targets_absent(
        [j.checkpoint_path for j in jobs] + [j.log_path for j in jobs])

    if arm == "real_only":
        pool_record = {"path": None, "sha256": None,
                       "policy": "real_only_no_synthetic_draw"}
        return [
            _fit_and_publish(job, real_features, real_y,
                             pool_record=pool_record, sampling=None,
                             index_digest="")
            for job in jobs
        ]

    pool_path = jobs[0].pool_path
    if pool_path is None or not pool_path.exists():
        raise _stop(f"missing pool for {construction_seed}:{arm}")
    pool_digest = _sha256_file(pool_path)
    with np.load(pool_path, allow_pickle=False) as pool:
        pool_construction = int(pool["construction_seed"])
        pool_y = np.asarray(pool["y_attack_type"], dtype=np.int64)
        # Extract once per pool: the feature map is row-wise, so selecting rows
        # of the feature matrix is identical to featurizing the selected rows.
        pool_features = e18_features(np.asarray(pool["x"], dtype=np.float32))
    if pool_construction != construction_seed:
        raise _stop(
            f"pool construction seed {pool_construction} != "
            f"{construction_seed}")
    pool_record = {
        "path": str(pool_path.relative_to(REPO)),
        "sha256": pool_digest,
        "windows": int(len(pool_y)),
        "read_only": True,
    }

    records: list[dict[str, Any]] = []
    for job in jobs:
        indices, sampling = sample_indices_without_replacement(
            pool_y, SYNTHETIC_TOTAL, job.pipeline_seed + SAMPLING_SEED_OFFSET)
        e17train._assert_draw_integrity(job, sampling)
        train_features = np.concatenate(
            (real_features, pool_features[indices]), axis=0)
        train_y = np.concatenate((real_y, pool_y[indices]), axis=0)
        records.append(_fit_and_publish(
            job, train_features, train_y, pool_record=pool_record,
            sampling=sampling,
            index_digest=sampling_indices_sha256(indices)))
        del train_features, train_y
    return records


# ---------------------------------------------------------------------------
# Gate G15 — fit determinism
# ---------------------------------------------------------------------------


def _evaluation_probe_features() -> np.ndarray:
    """The frozen E14 held-out normal base panel, as E18 features."""
    import evaluate_l4_counterfactual_factorial as e14

    prepare_path = (REPO / "journal" / "experiments"
                    / "e14_l4_counterfactual_factorial" / "prepare_v1.json")
    _manifest, bases, _latents = e14.load_prepared_inputs(
        json.loads(prepare_path.read_text()))
    return e18_features(np.asarray(bases, dtype=np.float32))


def run_determinism_child(construction_seed: int, arm: str, role: str,
                          pipeline_seed: int, out_path: Path) -> dict[str, Any]:
    """Refit one job in this fresh interpreter and write it outside the tree."""
    from sklearn.ensemble import RandomForestClassifier

    lc.fit_standardizer = _forbidden_standardizer
    job = E18Job(construction_seed, arm, pipeline_seed, role)
    real_features, real_y = _real_training_rows()
    with np.load(job.pool_path, allow_pickle=False) as pool:
        pool_y = np.asarray(pool["y_attack_type"], dtype=np.int64)
        pool_features = e18_features(np.asarray(pool["x"], dtype=np.float32))
    indices, _sampling = sample_indices_without_replacement(
        pool_y, SYNTHETIC_TOTAL, pipeline_seed + SAMPLING_SEED_OFFSET)
    train_features = np.concatenate(
        (real_features, pool_features[indices]), axis=0)
    train_y = np.concatenate((real_y, pool_y[indices]), axis=0)
    order = np.random.default_rng(pipeline_seed).permutation(len(train_y))
    forest = RandomForestClassifier(
        n_estimators=ESTIMATOR["n_estimators"],
        max_depth=ESTIMATOR["max_depth"],
        min_samples_leaf=ESTIMATOR["min_samples_leaf"],
        class_weight=ESTIMATOR["class_weight"],
        n_jobs=ESTIMATOR["n_jobs"],
        random_state=pipeline_seed,
    )
    forest.fit(train_features[order], train_y[order])
    del train_features, train_y, pool_features, real_features

    staged = _dump_forest(forest, out_path.parent, out_path.name)
    Path(staged).replace(out_path)
    probe = _evaluation_probe_features()
    predictions = forest.predict(probe)
    return {
        "key": job.key,
        "repeat_checkpoint": str(out_path),
        "repeat_checkpoint_sha256": _sha256_file(out_path),
        "prediction_rows": int(len(predictions)),
        "prediction_sha256": hashlib.sha256(np.ascontiguousarray(
            predictions.astype("<i8")).tobytes()).hexdigest(),
    }


def verify_fit_determinism() -> dict[str, Any]:
    """Gate G15 — one Rule and one placebo fit repeat byte-identically."""
    import joblib

    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="e18-determinism-") as scratch:
        for arm in ARMS:
            job = E18Job(CONSTRUCTION_SEEDS[0], arm, PIPELINE_SEEDS[0],
                         "primary")
            if not job.checkpoint_path.exists():
                raise _incomplete(
                    f"G15 cannot run: missing published checkpoint for "
                    f"{job.key}")
            out_path = Path(scratch) / f"repeat_{arm}.joblib"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--determinism-child",
                "--construction-seed", str(job.construction_seed),
                "--arm", job.arm,
                "--pipeline-seed", str(job.pipeline_seed),
                "--role", job.role,
                "--out", str(out_path),
            ]
            completed = subprocess.run(command, check=False,
                                       capture_output=True, text=True)
            if completed.returncode != 0:
                raise _stop(
                    f"G15 determinism child for {job.key} exited "
                    f"{completed.returncode}: {completed.stderr[-800:]}")
            child = json.loads(completed.stdout.strip().splitlines()[-1])
            published_digest = _sha256_file(job.checkpoint_path)
            byte_identical = (
                child["repeat_checkpoint_sha256"] == published_digest)
            forest = joblib.load(job.checkpoint_path)
            probe = _evaluation_probe_features()
            predictions = forest.predict(probe)
            prediction_digest = hashlib.sha256(np.ascontiguousarray(
                predictions.astype("<i8")).tobytes()).hexdigest()
            predictions_identical = (
                prediction_digest == child["prediction_sha256"])
            del forest
            if not byte_identical or not predictions_identical:
                raise _stop(
                    f"G15 fit determinism FAILED for {job.key}: "
                    f"byte_identical={byte_identical} "
                    f"predictions_identical={predictions_identical}")
            checks.append({
                "key": job.key,
                "arm": arm,
                "published_checkpoint_sha256": published_digest,
                "repeated_checkpoint_sha256": child[
                    "repeat_checkpoint_sha256"],
                "joblib_payload_byte_identical": True,
                "evaluation_prediction_rows": child["prediction_rows"],
                "evaluation_predictions_identical": True,
                "prediction_sha256": prediction_digest,
            })
    return {
        "gate": "G15_fit_determinism",
        "passed": True,
        "repeated_in_fresh_child_interpreter": True,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------


def _child_command(construction_seed: int, arm: str, role: str) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--construction-seed", str(construction_seed),
        "--arm", arm,
        "--role", role,
    ]


def _manifest_row(record: Mapping[str, Any], job: E18Job) -> dict[str, Any]:
    estimator = record["estimator"]
    pool = record["pool"]
    return {
        "construction_seed": job.construction_seed,
        "arm": job.arm,
        "pipeline_seed": job.pipeline_seed,
        "role": job.role,
        "setting": record["setting"],
        "model_tag": MODEL_TAG,
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "pool_path": pool.get("path") or "",
        "pool_sha256": pool.get("sha256") or "",
        "sampling_index_sha256": record["sampling_index_sha256"],
        "feature_dimension": record["feature_dimension"],
        "n_estimators": estimator["n_estimators"],
        "max_depth": "" if estimator["max_depth"] is None
                     else estimator["max_depth"],
        "min_samples_leaf": estimator["min_samples_leaf"],
        "class_weight": estimator["class_weight"],
        "permutation_seed": record["permutation_seed"],
        "train_windows": record["train_windows"],
        "elapsed_seconds": record["elapsed_seconds"],
        "device": record["device"],
    }


def execute(*, limit: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    preflight = build_preflight()
    records = record_paths()
    publish_and_cleanup([(_stage_json(records["preflight"], preflight),
                          records["preflight"])])

    groups = build_pool_groups()
    if limit is not None:
        groups = groups[:limit]

    manifest_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    for position, (construction_seed, arm, role) in enumerate(groups, start=1):
        print(f"[{position}/{len(groups)}] {construction_seed}:{arm} ({role})",
              flush=True)
        completed = subprocess.run(
            _child_command(construction_seed, arm, role), check=False)
        if completed.returncode != 0:
            raise _stop(
                f"child group {construction_seed}:{arm} exited "
                f"{completed.returncode}; stage halted with no outcome "
                "interpretation")
        for pipeline_seed in PIPELINE_SEEDS:
            job = E18Job(construction_seed, arm, pipeline_seed, role)
            if not job.checkpoint_path.exists() or not job.log_path.exists():
                raise _incomplete(f"child fit {job.key} published no bundle")
            record = json.loads(job.log_path.read_text())
            record.setdefault("checkpoint_sha256",
                              _sha256_file(job.checkpoint_path))
            manifest_rows.append(_manifest_row(record, job))
            sampling = record.get("sampling")
            if sampling is None:
                continue
            totals = sampling["total"]
            sampling_rows.append({
                "construction_seed": job.construction_seed,
                "arm": job.arm,
                "pipeline_seed": job.pipeline_seed,
                "role": job.role,
                "requested": totals["requested"],
                "drawn": totals["drawn"],
                "unique": totals["unique"],
                "repeated": totals["repeated"],
                "per_class_max_requested": max(
                    int(e["requested"]) for e in sampling["per_class"].values()),
                "per_attack_cap": PER_ATTACK_CAP,
                "index_sha256": record["sampling_index_sha256"],
            })

    internal_pairing = e17train.verify_internal_pairing(manifest_rows)
    cross_family = e17train.verify_cross_family_inheritance(manifest_rows)
    completeness = e17train.verify_completeness(manifest_rows)
    determinism = verify_fit_determinism()

    run_record = {
        "schema_version": TRAINING_RUN_SCHEMA,
        "stage": "e18_relation_visible_rf_training",
        "started_utc": preflight["built_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "fit_count": len(manifest_rows),
        "expected_fit_count": EXPECTED_FIT_COUNT,
        "registration": preflight["registration"],
        "feature_contract_gate": preflight["feature_contract"],
        "manipulation_visibility_gate":
            preflight["manipulation_visibility_record"]["gate"],
        "draw_integrity_gate": {
            "gate": "G9i_draw_integrity",
            "passed": True,
            "fits_checked": len(sampling_rows),
            "requested_equals_drawn_equals_unique": True,
            "repeated": 0,
            "per_attack_cap": PER_ATTACK_CAP,
        },
        "internal_pairing_gate": internal_pairing,
        "cross_family_inheritance_gate": cross_family,
        "completeness_gate": completeness,
        "determinism_gate": determinism,
        "matching_contract": (
            "identical training rows, inherited from the frozen E16 draws; the "
            "only difference between the two arms of a realization is the "
            "data1 permutation itself, read through the 83-dimensional map"),
        "preflight_path": str(records["preflight"].relative_to(REPO)),
        "status": "T-PASS",
        "technical_status": "T-PASS",
    }
    publish_and_cleanup([
        (_stage_bytes(records["manifest"], _csv_bytes(manifest_rows)),
         records["manifest"]),
        (_stage_bytes(records["sampling_audit"], _csv_bytes(sampling_rows)),
         records["sampling_audit"]),
        (_stage_json(records["run"], run_record), records["run"]),
    ])
    return run_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-pools", action="store_true",
                        help="stage S2: hash the consumed pools; no fitting")
    parser.add_argument("--visibility", action="store_true",
                        help="stage S2b: gate G17; publishes its table always")
    parser.add_argument("--visibility-v2", action="store_true",
                        help="stage S2b-v2: float64 and per-attack visibility "
                             "(amendment 2026-08-07 v2); publishes always")
    parser.add_argument("--execute", action="store_true",
                        help="stages S3/S4: run the canonical 245-fit grid")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--determinism-child", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--construction-seed", type=int)
    parser.add_argument("--arm", choices=(*ARMS, "real_only"))
    parser.add_argument("--pipeline-seed", type=int)
    parser.add_argument("--role", default="primary")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N pool groups (smoke only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.determinism_child:
        print(json.dumps(run_determinism_child(
            args.construction_seed, args.arm, args.role, args.pipeline_seed,
            args.out)))
        return 0
    if args.child:
        records = run_child_group(args.construction_seed, args.arm, args.role)
        print(json.dumps({"fits": len(records)}))
        return 0
    if args.verify_pools:
        print(json.dumps(run_pool_verification(), indent=2, sort_keys=True))
        return 0
    if args.visibility:
        print(json.dumps(run_manipulation_visibility(), indent=2,
                         sort_keys=True))
        return 0
    if args.visibility_v2:
        print(json.dumps(run_manipulation_visibility_v2(), indent=2,
                         sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps(build_preflight(), indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(limit=args.limit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
