#!/usr/bin/env python3
"""Construct the twenty preregistered E16 Rule/placebo construction pairs.

Registered by ``journal/experiments/e16_relational_response_decomposition/PREREG.md``.

The public helpers in this module are usable without generating data.  Importing
the module and the default CLI mode are read-only.  Canonical generation
requires ``--execute`` and may start only after the preflight has proved the
frozen source, environment, and complete no-clobber target set.

Design (PREREG sections 5.1--5.4)
---------------------------------
Each construction produces two pools that are array-for-array identical except
for the ``data1`` channel on injected spoof frames:

* the **Rule** arm is the E15 construction law at a fresh construction seed;
* the **placebo** arm destroys the implemented Gear/RPM inter-byte relation by a
  constrained within-window permutation of ``data1`` over injected frames only,
  while preserving the per-window ``data1`` multiset.

``generate_rule_based_synthetic_v2.generate_pool`` does not return injected
frame positions, and the placebo needs them.  This module therefore mirrors that
routine with position-recording injector forks whose RNG consumption order is
byte-for-byte identical, and then **proves** the mirror by requiring the
resulting Rule arrays to hash-match the canonical ``generate_pool`` output for
the same construction seed (gate G1a).  For the four E15 bridge constructions
the Rule arm must additionally reproduce the frozen E15 v2 pool array-for-array
(gate G7).

This wrapper never modifies the canonical E13 strict-v2 pool, the E14/E15 pools,
or anything under ``wisa/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib_common as lc  # noqa: E402
from generate_rule_based_synthetic_v2 import (  # noqa: E402
    CLASS_NAMES,
    WINDOW_SIZE,
    generate_pool,
    validate_train_source,
)
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _source_record,
    _stage_bytes,
    _stage_json,
    _stage_npz,
    atomic_publish_bundle,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    REPO / "journal" / "experiments" / "e16_relational_response_decomposition"
)
SYNTHETIC_DIR = REPO / "journal" / "datasets" / "synthetic"
TABLE_DIR = REPO / "journal" / "results" / "tables"
LOG_DIR = REPO / "journal" / "results" / "logs"

# --- frozen design constants (PREREG sections 3.1, 3.2, 5.1) ----------------

META_SEED = 20260725
SEED_FRAME = (100_000, 999_999)
E15_LINEAGE = (314159, 271828, 161803, 141421, 173205)
N_FRESH = 20

CONSTRUCTION_SEEDS = (
    415637, 748203, 560657, 459336, 457191,
    999191, 715822, 298581, 922089, 864178,
    583909, 467232, 550427, 485791, 820632,
    837943, 222537, 685012, 192753, 739263,
)

#: E15 constructions reused as the post-hoc bridge sensitivity.  Only the
#: placebo arm is published for these; the Rule arm already exists as frozen
#: E15 v2 pools and is regenerated solely to satisfy gate G7.
BRIDGE_SEEDS = (271828, 161803, 141421, 173205)

PER_ATTACK = 65_000
MAX_POSITIONS = 90
DATA0, DATA1 = 2, 3
GEAR_RESIDUAL_TOLERANCE = 8
GEAR_RESIDUAL_MAX_FRACTION = 0.01

#: Commit that froze journal/experiments/e16_relational_response_decomposition/PREREG.md.
#: The reused E15 source helper reports the *E15* preregistration commit, so E16
#: records its own registration provenance explicitly.
PREREG_COMMIT = "7371443"
PREREG_PATH = "journal/experiments/e16_relational_response_decomposition/PREREG.md"

OUTPUT_VERSION = "v2"
POOL_SCHEMA_VERSION = "e16_rule_placebo_pair_pool_v2"
PREFLIGHT_SCHEMA = "e16.pool_generation_preflight.v1"
RUN_SCHEMA = "e16.pool_generation_run.v1"
POOL_AUDIT_SCHEMA = "e16.rule_placebo_pool_audit.v1"
CHECKS_SCHEMA = "e16.manipulation_checks.v1"
EXPECTED_NUMPY_VERSION = "2.4.6"

SPOOF_LABELS = (3, 4)  # Gear, RPM
GEAR_LABEL, RPM_LABEL = 3, 4

ARRAY_KEYS = (
    "x",
    "y_binary",
    "y_attack_type",
    "synthetic_type",
    "condition_label",
    "condition_name",
    "injection_count",
    "base_source_index",
    "construction_ordinal",
)


class E16PoolError(RuntimeError):
    """Raised on any registered gate violation; halts the stage."""


def _stop(message: str) -> E16PoolError:
    return E16PoolError(f"E16 pool stage halted: {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Frozen seed axes (PREREG sections 3.1, 3.2)
# ---------------------------------------------------------------------------


def draw_construction_seeds(
        *,
        meta_seed: int = META_SEED,
        frame: tuple[int, int] = SEED_FRAME,
        excluded: Sequence[int] = E15_LINEAGE,
        count: int = N_FRESH,
) -> tuple[int, ...]:
    """Replay the registered deterministic construction-seed draw."""
    rng = np.random.default_rng(meta_seed)
    seen, seeds = set(excluded), []
    while len(seeds) < count:
        candidate = int(rng.integers(frame[0], frame[1] + 1))
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)
    return tuple(seeds)


def placebo_seed(construction_seed: int) -> int:
    """Registered placebo-stream derivation (PREREG section 3.2)."""
    digest = hashlib.sha256(f"e16-placebo|{construction_seed}".encode()).digest()
    return int.from_bytes(digest[-4:], "big")


def assert_seed_axis() -> dict[str, Any]:
    """Gate G2 — seed integrity."""
    drawn = draw_construction_seeds()
    if drawn != CONSTRUCTION_SEEDS:
        raise _stop(
            "construction seed list does not match the registered draw; "
            f"replay produced {drawn}")
    if len(set(CONSTRUCTION_SEEDS)) != N_FRESH:
        raise _stop("construction seeds are not unique")
    placebo = [placebo_seed(g) for g in CONSTRUCTION_SEEDS]
    if len(set(placebo)) != N_FRESH:
        raise _stop("derived placebo seeds are not unique")
    overlap = set(CONSTRUCTION_SEEDS) & set(E15_LINEAGE)
    if overlap:
        raise _stop(f"construction seeds intersect the E15 lineage: {overlap}")
    return {
        "gate": "G2_seed_integrity",
        "passed": True,
        "meta_seed": META_SEED,
        "frame": list(SEED_FRAME),
        "excluded": list(E15_LINEAGE),
        "construction_seeds": list(CONSTRUCTION_SEEDS),
        "placebo_seeds": placebo,
    }


# ---------------------------------------------------------------------------
# Position-recording injector forks
#
# Forked from generate_rule_based_synthetic_v2 with exactly one change: the
# injected positions are returned alongside the count.  The rng call sequence is
# untouched so the mirror reproduces the canonical pool bit-for-bit.
# ---------------------------------------------------------------------------


def choose_positions(rng, min_len, max_len, step_options):
    burst_len = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    step = int(rng.choice(step_options))
    return np.arange(start, start + burst_len, step)


def synth_dos_pos(rng, x):
    pos = choose_positions(rng, 32, 90, [1, 2])
    x[pos, 0] = 0x000
    x[pos, 1] = 8
    x[pos, 2:10] = rng.choice([0, 255], size=(len(pos), 8), p=[0.92, 0.08])
    x[pos, 10] = rng.uniform(0.00003, 0.002, size=len(pos))
    return len(pos), pos


def synth_fuzzy_pos(rng, x):
    pos = choose_positions(rng, 24, 88, [1, 2, 3, 4])
    x[pos, 0] = rng.integers(0, 2048, size=len(pos))
    dlc = rng.choice([2, 5, 8], size=len(pos), p=[0.08, 0.12, 0.80])
    x[pos, 1] = dlc
    payload = rng.integers(0, 256, size=(len(pos), 8))
    for i, d in enumerate(dlc):
        if d < 8:
            payload[i, d:] = 0
    x[pos, 2:10] = payload
    x[pos, 10] = rng.uniform(0.00005, 0.02, size=len(pos))
    return len(pos), pos


def synth_spoof_pos(rng, x, can_id, mode):
    pos = choose_positions(rng, 32, 90, [1, 2, 3])
    x[pos, 0] = can_id
    x[pos, 1] = 8
    base = x[pos, 2:10].copy()
    if mode == "gear":
        levels = rng.choice([0, 1, 2, 3, 4, 5], size=len(pos))
        base[:, 0] = np.clip(
            levels * 40 + rng.integers(0, 16, size=len(pos)), 0, 255)
        base[:, 1] = np.clip(
            255 - base[:, 0] + rng.integers(-8, 9, size=len(pos)), 0, 255)
    else:
        rpm = rng.integers(0, 8000, size=len(pos))
        base[:, 0] = (rpm // 32) % 256
        base[:, 1] = (rpm // 4) % 256
        base[:, 2] = np.clip(
            base[:, 2] + rng.integers(-50, 51, size=len(pos)), 0, 255)
    x[pos, 2:10] = base
    x[pos, 10] = np.maximum(x[pos, 10], 1e-5)
    return len(pos), pos


SPECS_POS = (
    ("DoS", 1, synth_dos_pos),
    ("Fuzzy", 2, synth_fuzzy_pos),
    ("Gear", GEAR_LABEL, lambda rng, x: synth_spoof_pos(rng, x, 0x43F, "gear")),
    ("RPM", RPM_LABEL, lambda rng, x: synth_spoof_pos(rng, x, 0x316, "rpm")),
)


# ---------------------------------------------------------------------------
# Constrained data1 permutations (placebo stream only)
# ---------------------------------------------------------------------------


def permute_gear_data1(prng, d0, d1):
    """Off-bin rotation over the six ``data0 // 40`` level bins.

    Returns ``(new_d1, offbin_count, n)``.  Every off-bin assignment
    deterministically violates ``|255 - data0 - data1| <= 8``.
    """
    n = len(d1)
    bins = (d0.astype(np.int64) // 40).clip(0, 5)
    order = np.lexsort((prng.random(n), bins))  # grouped by bin, random inside
    largest = int(np.bincount(bins, minlength=6).max())
    shift = largest if 2 * largest <= n else n - largest
    if shift == 0 and n > 1:
        shift = 1
    src = order[(np.arange(n) + shift) % n]
    new_d1 = np.empty_like(d1)
    new_d1[order] = d1[src]
    offbin = int((bins[order] != bins[src]).sum())
    return new_d1, offbin, n


def rpm_coupling_class(d0: np.ndarray) -> np.ndarray:
    """Classes within which the RPM coupling survives any reassignment.

    The generator writes ``data0 = (rpm // 32) % 256`` and
    ``data1 = (rpm // 4) % 256``, so ``data1 = 8*data0 + r`` with
    ``r = (rpm // 4) % 8``.  Giving frame ``i`` the ``data1`` of frame ``j``
    leaves ``(data1_j - 8*data0_i) mod 256 = (8*(data0_j - data0_i) + r_j) mod 256``,
    which stays below 8 exactly when ``data0_j == data0_i (mod 32)``.  The
    coupling is therefore destroyed by any off-class assignment.
    """
    return d0.astype(np.int64) % 32


def rpm_structural_minimum(d0: np.ndarray) -> int:
    """Residual matches no permutation of this window can avoid.

    If one class holds more than half the injected frames, at least
    ``2*largest - n`` frames must receive a same-class value.
    """
    classes = rpm_coupling_class(d0)
    n = len(classes)
    largest = int(np.bincount(classes).max()) if n else 0
    return max(0, 2 * largest - n)


def permute_rpm_data1(prng, d0, d1):
    """Off-class rotation, mirroring the Gear construction.

    Deterministic given the placebo stream and optimal: it attains the
    structural minimum of Section 5.2 for every window.  Pure rejection
    resampling cannot, because a window whose largest coupling class exceeds
    half its injected frames admits no zero-residual permutation at all.

    Returns ``(new_d1, residual_matches, structural_minimum, n)``.
    """
    n = len(d1)
    classes = rpm_coupling_class(d0)
    order = np.lexsort((prng.random(n), classes))  # grouped, random inside
    counts = np.bincount(classes)
    largest = int(counts.max()) if n else 0
    shift = largest if 2 * largest <= n else n - largest
    if shift == 0 and n > 1:
        shift = 1
    src = order[(np.arange(n) + shift) % n]
    new_d1 = np.empty_like(d1)
    new_d1[order] = d1[src]
    residual = int(
        (((new_d1.astype(np.int64) - 8 * d0.astype(np.int64)) % 256) < 8).sum())
    return new_d1, residual, max(0, 2 * largest - n), n


# ---------------------------------------------------------------------------
# Paired construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairResult:
    construction_seed: int
    placebo_seed: int
    rule: dict[str, Any]
    placebo_x: np.ndarray
    positions: np.ndarray
    position_count: np.ndarray
    stats: dict[str, Any]


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype == object or array.dtype.kind in "US":
        payload = json.dumps(
            [str(item) for item in array.ravel().tolist()],
            separators=(",", ":")).encode()
    else:
        payload = array.tobytes()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(payload)
    return digest.hexdigest()


def arrays_digest(arrays: Mapping[str, Any]) -> dict[str, str]:
    return {key: _array_digest(np.asarray(arrays[key])) for key in ARRAY_KEYS}


def generate_pair(
        train_x: np.ndarray,
        normal_indices: np.ndarray,
        *,
        per_attack: int,
        construction_seed: int,
) -> PairResult:
    """Mirror ``generate_pool`` with positions, then derive the placebo arm.

    The RNG consumption order is identical to ``generate_pool``: for every class
    a base draw, then one injector call per window, and after all four classes a
    single permutation.  Gate G1a proves this by digest comparison.
    """
    train_x = np.asarray(train_x)
    normal_indices = np.asarray(normal_indices, dtype=np.int64).reshape(-1)
    feature_count = len(lc.FEATURE_NAMES)

    rng = np.random.default_rng(construction_seed)
    prng = np.random.default_rng(placebo_seed(construction_seed))

    total = per_attack * len(SPECS_POS)
    x = np.empty((total, WINDOW_SIZE, feature_count), dtype=np.float32)
    x_placebo = np.empty_like(x)
    y_binary = np.ones(total, dtype=np.int8)
    y_attack = np.empty(total, dtype=np.int8)
    synthetic_type = np.empty(total, dtype="<U5")
    condition_label = np.empty(total, dtype=np.int8)
    condition_name = np.empty(total, dtype="<U5")
    injection_count = np.empty(total, dtype=np.int16)
    base_source_index = np.empty(total, dtype=np.int64)
    construction_ordinal = np.empty(total, dtype=np.int64)
    positions = np.full((total, MAX_POSITIONS), -1, dtype=np.int16)
    position_count = np.zeros(total, dtype=np.int16)

    gear = {"offbin": 0, "frames": 0, "residual_within_tolerance": 0}
    rpm = {"residual_matches": 0, "structural_minimum": 0, "frames": 0,
           "structurally_constrained_windows": 0, "suboptimal_windows": 0}

    for class_position, (name, label, injector) in enumerate(SPECS_POS):
        start = class_position * per_attack
        stop = start + per_attack
        base = rng.choice(normal_indices, size=per_attack, replace=True)
        x[start:stop] = train_x[base].astype(np.float32, copy=False)
        counts = np.empty(per_attack, dtype=np.int16)
        class_positions: list[np.ndarray] = []
        for local_index in range(per_attack):
            count, pos = injector(rng, x[start + local_index])
            counts[local_index] = count
            positions[start + local_index, :len(pos)] = pos
            position_count[start + local_index] = len(pos)
            class_positions.append(pos)

        y_attack[start:stop] = label
        synthetic_type[start:stop] = name
        condition_label[start:stop] = label
        condition_name[start:stop] = name
        injection_count[start:stop] = counts
        base_source_index[start:stop] = base
        construction_ordinal[start:stop] = np.arange(per_attack, dtype=np.int64)

        # placebo arm: identical everywhere, then data1 permuted on spoof frames
        x_placebo[start:stop] = x[start:stop]
        if label in SPOOF_LABELS:
            for local_index in range(per_attack):
                row = start + local_index
                pos = class_positions[local_index]
                d0 = x_placebo[row][pos, DATA0]
                d1 = x_placebo[row][pos, DATA1]
                if label == GEAR_LABEL:
                    new_d1, offbin, n = permute_gear_data1(prng, d0, d1)
                    gear["offbin"] += offbin
                    gear["frames"] += n
                    residual = np.abs(255.0 - d0 - new_d1)
                    gear["residual_within_tolerance"] += int(
                        (residual <= GEAR_RESIDUAL_TOLERANCE).sum())
                else:
                    new_d1, violations, minimum, n = permute_rpm_data1(
                        prng, d0, d1)
                    rpm["residual_matches"] += violations
                    rpm["structural_minimum"] += minimum
                    rpm["frames"] += n
                    if minimum > 0:
                        rpm["structurally_constrained_windows"] += 1
                    if violations > minimum:
                        rpm["suboptimal_windows"] += 1
                x_placebo[row][pos, DATA1] = new_d1

    order = rng.permutation(total)
    rule_arrays = {
        "x": x[order],
        "y_binary": y_binary[order],
        "y_attack_type": y_attack[order],
        "synthetic_type": synthetic_type[order],
        "condition_label": condition_label[order],
        "condition_name": condition_name[order],
        "injection_count": injection_count[order],
        "base_source_index": base_source_index[order],
        "construction_ordinal": construction_ordinal[order],
    }
    return PairResult(
        construction_seed=construction_seed,
        placebo_seed=placebo_seed(construction_seed),
        rule=rule_arrays,
        placebo_x=x_placebo[order],
        positions=positions[order],
        position_count=position_count[order],
        stats={"gear": gear, "rpm": rpm},
    )


# ---------------------------------------------------------------------------
# Registered gates (PREREG section 15.2)
# ---------------------------------------------------------------------------


def _check(name: str, gate: str, value: Any, threshold: Any, passed: bool
           ) -> dict[str, Any]:
    return {"check": name, "gate": gate, "value": value,
            "threshold": threshold, "passed": passed}


def evaluate_pair_gates(
        pair: PairResult,
        canonical_digests: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Gates G1, G3--G6 for one construction pair."""
    checks: list[dict[str, Any]] = []
    g = pair.construction_seed

    # --- G1a: the position-recording mirror reproduces generate_pool ---------
    mirror = arrays_digest(pair.rule)
    mismatched = sorted(k for k in ARRAY_KEYS if mirror[k] != canonical_digests[k])
    checks.append(_check(
        "G1a_mirror_reproduces_generate_pool", "abort",
        ",".join(mismatched) if mismatched else "", "", not mismatched))

    x_rule = pair.rule["x"]
    x_plac = pair.placebo_x
    y_attack = pair.rule["y_attack_type"]
    spoof = np.isin(y_attack, SPOOF_LABELS)
    other = ~spoof

    # --- G1b: arms differ only in data1, and only on injected spoof frames ---
    other_cols = [c for c in range(x_rule.shape[2]) if c != DATA1]
    non_data1_equal = np.array_equal(
        x_rule[spoof][:, :, other_cols], x_plac[spoof][:, :, other_cols])
    checks.append(_check(
        "G1b_non_data1_channels_identical", "abort", float(non_data1_equal),
        1.0, bool(non_data1_equal)))

    # data1 outside injected positions must be untouched
    spoof_positions = pair.positions[spoof]
    valid = spoof_positions >= 0
    injected = np.zeros((int(spoof.sum()), WINDOW_SIZE), dtype=bool)
    injected[np.repeat(np.arange(len(spoof_positions)), valid.sum(axis=1)),
             spoof_positions[valid].astype(np.int64)] = True
    d1_rule = x_rule[spoof][:, :, DATA1]
    d1_plac = x_plac[spoof][:, :, DATA1]
    off_frame_equal = np.array_equal(d1_rule[~injected], d1_plac[~injected])
    checks.append(_check(
        "G1c_data1_untouched_off_injected_frames", "abort",
        float(off_frame_equal), 1.0, bool(off_frame_equal)))

    # the permutation must actually move values on injected frames
    moved = int((d1_rule[injected] != d1_plac[injected]).sum())
    checks.append(_check(
        "G1d_injected_data1_moved_count", "report", float(moved), "", ""))

    # --- G3: per-window data1 multiset preserved on injected frames ----------
    multiset_ok = np.array_equal(
        np.sort(x_rule[spoof][:, :, DATA1], axis=1),
        np.sort(x_plac[spoof][:, :, DATA1], axis=1))
    checks.append(_check(
        "G3_data1_multiset_identical", "abort", float(multiset_ok), 1.0,
        bool(multiset_ok)))

    # --- G4: label/metadata internal consistency ----------------------------
    # Both arms serialize the same metadata objects, so the meaningful check is
    # that those objects are internally coherent and agree with the recorded
    # injection geometry.
    labels_ok = bool(
        np.array_equal(pair.rule["condition_label"], y_attack)
        and np.array_equal(
            np.asarray(pair.rule["condition_name"], dtype="<U5"),
            np.asarray(pair.rule["synthetic_type"], dtype="<U5"))
        and np.array_equal(pair.rule["injection_count"], pair.position_count)
        and np.all(pair.rule["y_binary"] == 1))
    checks.append(_check(
        "G4_metadata_internally_consistent", "abort", float(labels_ok), 1.0,
        labels_ok))

    counts_ok = {int(label): int(np.count_nonzero(y_attack == label))
                 for label in CLASS_NAMES}
    expected_per_class = len(y_attack) // len(CLASS_NAMES)
    quota_ok = all(count == expected_per_class for count in counts_ok.values())
    checks.append(_check(
        "G4_per_class_quota", "abort", json.dumps(counts_ok, sort_keys=True),
        expected_per_class, bool(quota_ok)))

    # --- G5: relation destruction -------------------------------------------
    gear_frames = max(pair.stats["gear"]["frames"], 1)
    gear_fraction = pair.stats["gear"]["residual_within_tolerance"] / gear_frames
    checks.append(_check(
        "G5_gear_residual_within_tolerance_fraction", "abort", gear_fraction,
        GEAR_RESIDUAL_MAX_FRACTION, bool(gear_fraction < GEAR_RESIDUAL_MAX_FRACTION)))
    rpm_residual = pair.stats["rpm"]["residual_matches"]
    rpm_minimum = pair.stats["rpm"]["structural_minimum"]
    # Gate on optimality, not on zero: a window whose largest coupling class
    # exceeds half its injected frames admits no zero-residual permutation.
    checks.append(_check(
        "G5_rpm_residual_attains_structural_minimum", "abort",
        float(rpm_residual), float(rpm_minimum),
        bool(rpm_residual == rpm_minimum)))
    rpm_frames = max(pair.stats["rpm"]["frames"], 1)
    checks.append(_check(
        "G5_rpm_residual_fraction", "abort", rpm_residual / rpm_frames,
        GEAR_RESIDUAL_MAX_FRACTION,
        bool(rpm_residual / rpm_frames < GEAR_RESIDUAL_MAX_FRACTION)))
    checks.append(_check(
        "G5_rpm_structurally_constrained_windows", "report",
        float(pair.stats["rpm"]["structurally_constrained_windows"]), "", ""))
    checks.append(_check(
        "G5_gear_offbin_fraction", "report",
        pair.stats["gear"]["offbin"] / gear_frames, "", ""))

    # --- G6: DoS/Fuzzy bit identity -----------------------------------------
    dos_fuzzy_ok = np.array_equal(x_rule[other], x_plac[other])
    checks.append(_check(
        "G6_dos_fuzzy_bit_identical", "abort", float(dos_fuzzy_ok), 1.0,
        bool(dos_fuzzy_ok)))

    for row in checks:
        row["construction_seed"] = g
    return checks


def failed_gates(checks: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(c["check"]) for c in checks
            if c["gate"] == "abort" and c["passed"] is not True]


# ---------------------------------------------------------------------------
# Output identity (PREREG section 5.4)
# ---------------------------------------------------------------------------


def rule_pool_path(construction_seed: int, repo: Path = REPO) -> Path:
    return (repo / "journal" / "datasets" / "synthetic"
            / f"e16_rule_cseed{construction_seed}_windows_{OUTPUT_VERSION}.npz")


def placebo_pool_path(construction_seed: int, repo: Path = REPO) -> Path:
    return (repo / "journal" / "datasets" / "synthetic"
            / f"e16_placebo_cseed{construction_seed}_windows_{OUTPUT_VERSION}.npz")


def target_paths(repo: Path = REPO) -> list[Path]:
    """Every canonical artifact this stage publishes."""
    paths = [rule_pool_path(g, repo) for g in CONSTRUCTION_SEEDS]
    paths += [placebo_pool_path(g, repo) for g in CONSTRUCTION_SEEDS]
    paths += [placebo_pool_path(g, repo) for g in BRIDGE_SEEDS]
    paths += [
        repo / "journal" / "results" / "tables"
        / f"e16_pool_audit_{OUTPUT_VERSION}.csv",
        repo / "journal" / "results" / "tables"
        / f"e16_manipulation_checks_{OUTPUT_VERSION}.csv",
        repo / "journal" / "experiments"
        / "e16_relational_response_decomposition"
        / f"pool_generation_run_{OUTPUT_VERSION}.json",
    ]
    return paths


def assert_registration_committed(repo: Path = REPO) -> dict[str, Any]:
    """The E16 preregistration must be committed and an ancestor of HEAD.

    ``_source_record`` is reused from the E15 module and reports the E15
    registration commit, so E16 verifies its own registration separately.
    """
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True, capture_output=True, text=True).stdout.strip()

    prereg = repo / PREREG_PATH
    if not prereg.exists():
        raise _stop(f"missing preregistration: {PREREG_PATH}")
    if git("status", "--porcelain", "--", PREREG_PATH):
        raise _stop(f"preregistration has uncommitted changes: {PREREG_PATH}")
    try:
        resolved = git("rev-parse", PREREG_COMMIT)
        subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor",
             PREREG_COMMIT, "HEAD"], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise _stop(
            f"registration commit {PREREG_COMMIT} is not an ancestor of HEAD"
        ) from exc
    last_touch = git("log", "-1", "--format=%H", "--", PREREG_PATH)
    return {
        "gate": "registration_committed",
        "passed": True,
        "preregistration_path": PREREG_PATH,
        "registration_commit": resolved,
        "registration_commit_is_ancestor": True,
        "preregistration_last_modified_commit": last_touch,
        "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
    }


def assert_targets_absent(repo: Path = REPO) -> dict[str, Any]:
    """No-clobber preflight."""
    present = [str(p.relative_to(repo)) for p in target_paths(repo)
               if p.exists() or p.is_symlink()]
    if present:
        raise _stop(f"refusing to overwrite existing outputs: {present}")
    return {"gate": "no_clobber_preflight", "passed": True,
            "checked": len(target_paths(repo))}


def pool_arrays(pair: PairResult, arm: str) -> dict[str, Any]:
    """Serializable arrays for one arm."""
    if arm not in ("rule", "placebo"):
        raise ValueError(f"unknown arm: {arm}")
    arrays = dict(pair.rule)
    if arm == "placebo":
        arrays["x"] = pair.placebo_x
    arrays.update({
        "positions": pair.positions,
        "position_count": pair.position_count,
        "feature_names": np.asarray(lc.FEATURE_NAMES, dtype=object),
        "window_size": np.asarray(WINDOW_SIZE, dtype=np.int32),
        "stride": np.asarray(32, dtype=np.int32),
        "construction_seed": np.asarray(pair.construction_seed, dtype=np.int64),
        "placebo_seed": np.asarray(pair.placebo_seed, dtype=np.int64),
        "arm": np.asarray(arm, dtype=object),
        "pool_schema_version": np.asarray(POOL_SCHEMA_VERSION, dtype=object),
    })
    return arrays


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def build_preflight(repo: Path = REPO) -> dict[str, Any]:
    """Read-only preflight; performs no generation."""
    record: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "stage": "e16_pool_generation",
        "built_utc": utc_now(),
        "per_attack": PER_ATTACK,
        "output_version": OUTPUT_VERSION,
        "construction_count": len(CONSTRUCTION_SEEDS),
        "bridge_count": len(BRIDGE_SEEDS),
    }
    record["seed_axis"] = assert_seed_axis()
    record["environment"] = _environment_record(repo)
    numpy_version = (record["environment"].get("full", {})
                     .get("packages", {}).get("numpy"))
    if numpy_version != EXPECTED_NUMPY_VERSION:
        raise _stop(
            f"NumPy {numpy_version} != registered {EXPECTED_NUMPY_VERSION}")
    record["numpy_version"] = numpy_version
    record["source"] = _source_record(repo)
    record["registration"] = assert_registration_committed(repo)
    record["target_absence"] = assert_targets_absent(repo)
    record["targets"] = [str(p.relative_to(repo)) for p in target_paths(repo)]
    return record


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _load_train_source() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(lc.WINDOWS / "train_windows.npz", allow_pickle=True)
    return validate_train_source(data)


def _frozen_e15_pool(construction_seed: int) -> Path:
    return SYNTHETIC_DIR / f"rule_cseed{construction_seed}_windows_v2.npz"


def _bridge_lineage_check(pair: PairResult) -> dict[str, Any]:
    """Gate G7 — reproduce the frozen E15 v2 Rule pool array-for-array."""
    reference_path = _frozen_e15_pool(pair.construction_seed)
    if not reference_path.exists():
        raise _stop(f"missing frozen E15 reference pool: {reference_path}")
    reference = np.load(reference_path, allow_pickle=True)
    mine = arrays_digest(pair.rule)
    theirs = {}
    for key in ARRAY_KEYS:
        if key not in reference.files:
            raise _stop(f"E15 reference pool lacks array '{key}'")
        theirs[key] = _array_digest(np.asarray(reference[key]))
    mismatched = sorted(k for k in ARRAY_KEYS if mine[k] != theirs[k])
    return _check(
        "G7_e15_lineage_reproduction", "abort",
        ",".join(mismatched) if mismatched else "", "", not mismatched
    ) | {"construction_seed": pair.construction_seed}


def execute(repo: Path = REPO, *, limit: int | None = None) -> dict[str, Any]:
    """Generate, gate, and atomically publish every registered pair."""
    started = time.time()
    preflight = build_preflight(repo)
    train_x, normal_indices = _load_train_source()

    fresh = CONSTRUCTION_SEEDS if limit is None else CONSTRUCTION_SEEDS[:limit]
    bridge = BRIDGE_SEEDS if limit is None else BRIDGE_SEEDS[:max(limit // 5, 1)]

    all_checks: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    staged: list[tuple[Path, Path]] = []

    for construction_seed in tuple(fresh) + tuple(bridge):
        is_bridge = construction_seed in BRIDGE_SEEDS
        t0 = time.time()
        pair = generate_pair(
            train_x, normal_indices,
            per_attack=PER_ATTACK, construction_seed=construction_seed)

        canonical = generate_pool(
            train_x, normal_indices,
            per_attack=PER_ATTACK, construction_seed=construction_seed)
        checks = evaluate_pair_gates(pair, arrays_digest(canonical["arrays"]))
        if is_bridge:
            checks.append(_bridge_lineage_check(pair))
        del canonical

        failures = failed_gates(checks)
        all_checks.extend(checks)
        if failures:
            raise _stop(
                f"construction {construction_seed} failed gate(s): {failures}")

        arms = ("placebo",) if is_bridge else ("rule", "placebo")
        for arm in arms:
            target = (placebo_pool_path(construction_seed, repo) if arm == "placebo"
                      else rule_pool_path(construction_seed, repo))
            arrays = pool_arrays(pair, arm)
            staged.append((_stage_npz(target, arrays), target))
            audit_rows.append({
                "construction_seed": construction_seed,
                "placebo_seed": pair.placebo_seed,
                "arm": arm,
                "role": "bridge_sensitivity" if is_bridge else "primary",
                "windows": int(len(pair.rule["y_binary"])),
                "per_attack": PER_ATTACK,
                "x_digest": _array_digest(
                    pair.placebo_x if arm == "placebo" else pair.rule["x"]),
                "target": str(target.relative_to(repo)),
            })
        print(f"  {construction_seed} ({'bridge' if is_bridge else 'primary'}): "
              f"gates passed, {time.time()-t0:.0f}s", flush=True)
        del pair

    checks_target = TABLE_DIR / f"e16_manipulation_checks_{OUTPUT_VERSION}.csv"
    audit_target = TABLE_DIR / f"e16_pool_audit_{OUTPUT_VERSION}.csv"
    run_target = EXPERIMENT_DIR / f"pool_generation_run_{OUTPUT_VERSION}.json"

    run_record = {
        "schema_version": RUN_SCHEMA,
        "stage": "e16_pool_generation",
        "started_utc": preflight["built_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.time() - started, 3),
        "preflight": preflight,
        "constructions_generated": list(fresh),
        "bridge_generated": list(bridge),
        "gate_summary": {
            "total_checks": len(all_checks),
            "abort_checks": sum(1 for c in all_checks if c["gate"] == "abort"),
            "failures": [],
        },
        "pool_audit_rows": len(audit_rows),
        "status": "T-PASS",
        "technical_status": "T-PASS",
    }

    staged.append((_stage_bytes(checks_target, _csv_bytes(all_checks)), checks_target))
    staged.append((_stage_bytes(audit_target, _csv_bytes(audit_rows)), audit_target))
    staged.append((_stage_json(run_target, run_record), run_target))
    publish_and_cleanup(staged)
    return run_record



def publish_and_cleanup(staged) -> None:
    """Publish atomically, then remove the staged hard-link sources.

    ``atomic_publish_bundle`` links staged files into place and deliberately
    leaves them behind so a failed bundle can be diagnosed.  On success the
    caller owns that cleanup; skipping it leaves dot-prefixed temporaries that
    a later stage's clean-source gate will reject.
    """
    atomic_publish_bundle(staged)
    for source, _ in staged:
        try:
            Path(source).unlink()
        except FileNotFoundError:
            pass

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="run canonical generation (default is a read-only preflight)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="generate only the first N constructions (smoke checks only; "
             "never used for the canonical run)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.execute:
        record = build_preflight()
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    record = execute(limit=args.limit)
    print(json.dumps(
        {k: v for k, v in record.items() if k != "preflight"},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
