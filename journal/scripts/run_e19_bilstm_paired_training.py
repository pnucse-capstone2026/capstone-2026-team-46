#!/usr/bin/env python3
"""Train the 245 preregistered E19 BiLSTM fits.

Registered by ``journal/experiments/e19_bilstm_relational_crossing/PREREG.md``
(commit ``25f5dfa``), sections 3--5.

Grid (PREREG section 5)
-----------------------
* 5 shared BiLSTM real-only references                                =   5
* 20 realizations x 2 arms (Rule, placebo) x 5 pipeline seeds        = 200
* 4 bridge realizations x 2 arms x 5 pipeline seeds                  =  40

E19 changes exactly one axis relative to E16: the detector architecture.  The
``LSTMClassifier`` is imported from ``train_family_extension_lstm`` and every
training constant is imported from ``train_rule_construction_seed_crossing``;
nothing is retuned.  Within a realization and pipeline seed the Rule and
placebo fits share the initialization seed, the synthetic row indices, the
minibatch RNG convention, and the 6,156-update budget.

The identical-training-row contract (PREREG 4.3) is verified after fitting:
every fit with a frozen E16 counterpart must reproduce the E16
``sampling_index_sha256``.  The 20 bridge Rule fits have no counterpart and
are recorded as exempt, mirroring E17.

Every fit runs in a fresh child interpreter.  Importing this module and the
default CLI mode are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib_common as lc  # noqa: E402
import generate_e16_rule_placebo_pairs as e16gen  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _source_record,
    _stage_bytes,
    _stage_json,
    atomic_publish_bundle,
)
from train_family_extension_lstm import LSTMClassifier, predict  # noqa: E402
from train_rule_construction_seed_crossing import (  # noqa: E402
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_STEPS,
    SAMPLING_SEED_OFFSET,
    SYNTHETIC_TOTAL,
    VALIDATION_BATCH_SIZE,
    VALIDATION_CADENCE,
    VALIDATION_CHECKPOINTS,
    WEIGHT_DECAY,
    _load_real_train_val,
    sample_indices_without_replacement,
    sampling_indices_sha256,
    verify_derived_standardizer,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "journal" / "experiments" / "e19_bilstm_relational_crossing"
SYNTHETIC_DIR = REPO / "journal" / "datasets" / "synthetic"
MODEL_DIR = REPO / "journal" / "models" / "generator_extension"
LOG_DIR = REPO / "journal" / "results" / "logs"
TABLE_DIR = REPO / "journal" / "results" / "tables"

# --- registration provenance ------------------------------------------------

PREREG_PATH = "journal/experiments/e19_bilstm_relational_crossing/PREREG.md"
PREREG_COMMIT = "25f5dfa"

# --- frozen design constants (PREREG sections 3, 5) -------------------------

#: PREREG section 3.1, transcribed from the registration text.  The gate fires
#: if either this literal or the E16 module list drifts.
REGISTERED_CONSTRUCTION_SEEDS = (
    415637, 748203, 560657, 459336, 457191,
    999191, 715822, 298581, 922089, 864178,
    583909, 467232, 550427, 485791, 820632,
    837943, 222537, 685012, 192753, 739263,
)
REGISTERED_BRIDGE_SEEDS = (271828, 161803, 141421, 173205)

CONSTRUCTION_SEEDS = e16gen.CONSTRUCTION_SEEDS
BRIDGE_SEEDS = e16gen.BRIDGE_SEEDS
PIPELINE_SEEDS = (7, 42, 123, 2026, 3407)
PAIRED_ARMS = ("rule", "placebo")
REAL_ARM = "real_only"

OUTPUT_VERSION = "v1"
MODEL_TAG = f"matchedsteps_e19_{OUTPUT_VERSION}"

E16_SAMPLING_AUDIT = TABLE_DIR / "e16_sampling_audit_v2.csv"

TRAINING_PREFLIGHT_SCHEMA = "e19.training_preflight.v1"
TRAINING_RUN_SCHEMA = "e19.training_run.v1"
TRAINING_LOG_SCHEMA = "e19.paired_training_log.v1"

EXPECTED_REFERENCE_FITS = len(PIPELINE_SEEDS)
EXPECTED_PRIMARY_FITS = (
    len(CONSTRUCTION_SEEDS) * len(PAIRED_ARMS) * len(PIPELINE_SEEDS))
EXPECTED_BRIDGE_FITS = len(BRIDGE_SEEDS) * len(PAIRED_ARMS) * len(PIPELINE_SEEDS)
EXPECTED_FIT_COUNT = (
    EXPECTED_REFERENCE_FITS + EXPECTED_PRIMARY_FITS + EXPECTED_BRIDGE_FITS)


class E19TrainingError(RuntimeError):
    """Raised on any registered training gate violation."""


def _stop(message: str) -> E19TrainingError:
    return E19TrainingError(f"T-STOP-E19-TRAIN: {message}")


def _incomplete(message: str) -> E19TrainingError:
    return E19TrainingError(f"T-INCOMPLETE-E19-TRAIN: {message}")


def utc_now() -> str:
    return e16gen.utc_now()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_and_cleanup(staged: Sequence[tuple[Path, Path]]) -> None:
    atomic_publish_bundle(staged)
    for source, _ in staged:
        try:
            Path(source).unlink()
        except FileNotFoundError:
            pass


def assert_registration_committed() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args],
            check=True, capture_output=True, text=True).stdout.strip()

    prereg = REPO / PREREG_PATH
    if not prereg.exists():
        raise _stop(f"missing preregistration: {PREREG_PATH}")
    if git("status", "--porcelain", "--", PREREG_PATH):
        raise _stop(f"preregistration has uncommitted changes: {PREREG_PATH}")
    try:
        resolved = git("rev-parse", PREREG_COMMIT)
        subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
             PREREG_COMMIT, "HEAD"], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise _stop(
            f"registration commit {PREREG_COMMIT} is not an ancestor of HEAD"
        ) from exc
    if CONSTRUCTION_SEEDS != REGISTERED_CONSTRUCTION_SEEDS:
        raise _stop("E16 module construction seeds drifted from the "
                    "registered literals")
    if BRIDGE_SEEDS != REGISTERED_BRIDGE_SEEDS:
        raise _stop("E16 module bridge seeds drifted from the registered "
                    "literals")
    return {
        "gate": "registration_committed",
        "passed": True,
        "preregistration_path": PREREG_PATH,
        "registration_commit": resolved,
        "preregistration_sha256":
            hashlib.sha256(prereg.read_bytes()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def _frozen_e15_rule_pool(construction_seed: int) -> Path:
    return SYNTHETIC_DIR / f"rule_cseed{construction_seed}_windows_v2.npz"


@dataclass(frozen=True)
class E19Job:
    construction_seed: int
    arm: str
    pipeline_seed: int
    role: str

    @property
    def pool_path(self) -> Path | None:
        if self.arm == REAL_ARM:
            return None
        if self.arm == "placebo":
            return e16gen.placebo_pool_path(self.construction_seed, REPO)
        if self.role == "bridge_sensitivity":
            return _frozen_e15_rule_pool(self.construction_seed)
        return e16gen.rule_pool_path(self.construction_seed, REPO)

    @property
    def checkpoint_path(self) -> Path:
        if self.arm == REAL_ARM:
            return MODEL_DIR / (
                f"bilstm_real_only_{MODEL_TAG}_seed{self.pipeline_seed}.pt")
        return MODEL_DIR / (
            f"bilstm_{self.arm}_0p30_cseed{self.construction_seed}_"
            f"{MODEL_TAG}_seed{self.pipeline_seed}.pt")

    @property
    def log_path(self) -> Path:
        if self.arm == REAL_ARM:
            return LOG_DIR / (
                f"train_bilstm_real_only_{MODEL_TAG}_"
                f"seed{self.pipeline_seed}.log")
        return LOG_DIR / (
            f"train_bilstm_{self.arm}_0p30_cseed{self.construction_seed}_"
            f"{MODEL_TAG}_seed{self.pipeline_seed}.log")

    @property
    def key(self) -> str:
        return f"{self.construction_seed}:{self.arm}:{self.pipeline_seed}"


def build_training_grid() -> tuple[E19Job, ...]:
    """Registered order: references first, then primary, then bridge."""
    jobs: list[E19Job] = []
    for pipeline_seed in PIPELINE_SEEDS:
        jobs.append(E19Job(0, REAL_ARM, pipeline_seed, "shared_reference"))
    for construction_seed in CONSTRUCTION_SEEDS:
        for arm in PAIRED_ARMS:
            for pipeline_seed in PIPELINE_SEEDS:
                jobs.append(E19Job(
                    construction_seed, arm, pipeline_seed, "primary"))
    for construction_seed in BRIDGE_SEEDS:
        for arm in PAIRED_ARMS:
            for pipeline_seed in PIPELINE_SEEDS:
                jobs.append(E19Job(
                    construction_seed, arm, pipeline_seed,
                    "bridge_sensitivity"))
    if len(jobs) != EXPECTED_FIT_COUNT:
        raise _stop(f"grid size {len(jobs)} != registered {EXPECTED_FIT_COUNT}")
    if len({job.key for job in jobs}) != len(jobs):
        raise _stop("duplicate job key in the training grid")
    return tuple(jobs)


def training_record_paths() -> dict[str, Path]:
    return {
        "preflight": EXPERIMENT_DIR / f"training_preflight_{OUTPUT_VERSION}.json",
        "run": EXPERIMENT_DIR / f"training_run_{OUTPUT_VERSION}.json",
        "manifest": TABLE_DIR / f"e19_training_manifest_{OUTPUT_VERSION}.csv",
        "sampling_audit": TABLE_DIR / f"e19_sampling_audit_{OUTPUT_VERSION}.csv",
        "pool_inheritance": EXPERIMENT_DIR
        / f"pool_inheritance_{OUTPUT_VERSION}.json",
    }


def target_paths(jobs: Sequence[E19Job]) -> list[Path]:
    paths = [job.checkpoint_path for job in jobs]
    paths += [job.log_path for job in jobs]
    paths += list(training_record_paths().values())
    return paths


def assert_targets_absent(paths: Sequence[Path]) -> dict[str, Any]:
    present = [str(p.relative_to(REPO)) for p in paths
               if p.exists() or p.is_symlink()]
    if present:
        raise _stop(
            f"refusing to overwrite existing targets: {present[:8]}"
            + (f" (+{len(present) - 8} more)" if len(present) > 8 else ""))
    return {"gate": "no_clobber", "passed": True, "checked": len(paths)}


def _pool_label_digest(path: Path) -> str:
    with np.load(path, allow_pickle=True) as pool:
        labels = np.asarray(pool["y_attack_type"], dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()


def assert_pools_present_and_paired() -> dict[str, Any]:
    """Pools exist and paired arms carry identical label vectors."""
    rows: list[dict[str, Any]] = []
    for construction_seed in CONSTRUCTION_SEEDS + BRIDGE_SEEDS:
        role = ("bridge_sensitivity" if construction_seed in BRIDGE_SEEDS
                else "primary")
        rule = (_frozen_e15_rule_pool(construction_seed)
                if role == "bridge_sensitivity"
                else e16gen.rule_pool_path(construction_seed, REPO))
        placebo = e16gen.placebo_pool_path(construction_seed, REPO)
        for path in (rule, placebo):
            if not path.exists():
                raise _stop(f"missing pool: {path.relative_to(REPO)}")
        rule_labels = _pool_label_digest(rule)
        placebo_labels = _pool_label_digest(placebo)
        if rule_labels != placebo_labels:
            raise _stop(
                f"realization {construction_seed}: paired arms have different "
                "label vectors, so the registered draw cannot be identical")
        rows.append({
            "construction_seed": construction_seed,
            "role": role,
            "rule_pool": str(rule.relative_to(REPO)),
            "rule_pool_sha256": _sha256_file(rule),
            "placebo_pool": str(placebo.relative_to(REPO)),
            "placebo_pool_sha256": _sha256_file(placebo),
            "label_vector_sha256": rule_labels,
        })
    checks = TABLE_DIR / "e16_manipulation_checks_v2.csv"
    if not checks.exists():
        raise _stop("missing inherited manipulation-check record")
    return {
        "gate": "pool_inheritance",
        "passed": True,
        "constructions": len(rows),
        "inherited_manipulation_checks": str(checks.relative_to(REPO)),
        "inherited_manipulation_checks_sha256": _sha256_file(checks),
        "rows": rows,
    }


def build_preflight() -> dict[str, Any]:
    jobs = build_training_grid()
    record: dict[str, Any] = {
        "schema_version": TRAINING_PREFLIGHT_SCHEMA,
        "stage": "e19_paired_training",
        "built_utc": utc_now(),
        "expected_fits": EXPECTED_FIT_COUNT,
        "grid": {
            "references": EXPECTED_REFERENCE_FITS,
            "primary": EXPECTED_PRIMARY_FITS,
            "bridge": EXPECTED_BRIDGE_FITS,
            "pipeline_seeds": list(PIPELINE_SEEDS),
        },
        "protocol": {
            "architecture": "LSTMClassifier(train_family_extension_lstm)",
            "max_optimizer_updates": MAX_STEPS,
            "validation_cadence": VALIDATION_CADENCE,
            "validation_checkpoints": VALIDATION_CHECKPOINTS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "synthetic_total": SYNTHETIC_TOTAL,
            "sampling_seed_offset": SAMPLING_SEED_OFFSET,
        },
    }
    record["registration"] = assert_registration_committed()
    record["environment"] = _environment_record(REPO)
    record["source"] = _source_record(REPO)
    record["pools"] = assert_pools_present_and_paired()
    record["standardizer"] = verify_derived_standardizer(repo=REPO)
    if not E16_SAMPLING_AUDIT.exists():
        raise _stop("missing frozen E16 sampling audit for the "
                    "identical-training-row contract")
    record["e16_sampling_audit_sha256"] = _sha256_file(E16_SAMPLING_AUDIT)
    record["target_absence"] = assert_targets_absent(target_paths(jobs))
    return record


# ---------------------------------------------------------------------------
# Child fit
# ---------------------------------------------------------------------------


def run_child_fit(job: E19Job) -> dict[str, Any]:
    """Train and atomically publish exactly one E19 fit."""
    assert_targets_absent([job.checkpoint_path, job.log_path])

    started = utc_now()
    start_time = time.monotonic()

    sampling: dict[str, Any] | None = None
    sampling_sha: str = ""
    pool_sha: str = ""
    if job.arm == REAL_ARM:
        selected_x = None
        selected_y = None
    else:
        pool_path = job.pool_path
        if pool_path is None or not pool_path.exists():
            raise _stop(f"missing pool for {job.key}: {pool_path}")
        with np.load(pool_path, allow_pickle=True) as pool:
            synthetic_x = np.asarray(pool["x"], dtype=np.float32)
            synthetic_y = np.asarray(pool["y_attack_type"], dtype=np.int64)
            pool_construction = int(pool["construction_seed"])
        if pool_construction != job.construction_seed:
            raise _stop(
                f"pool construction seed {pool_construction} != job "
                f"{job.construction_seed}")
        pool_sha = _sha256_file(pool_path)
        indices, sampling = sample_indices_without_replacement(
            synthetic_y, SYNTHETIC_TOTAL,
            job.pipeline_seed + SAMPLING_SEED_OFFSET)
        totals = sampling["total"]
        if (totals["repeated"] != 0
                or totals["drawn"] != totals["unique"]
                or totals["requested"] != totals["drawn"]
                or totals["requested"] != SYNTHETIC_TOTAL):
            raise _stop(f"{job.key}: draw is not the registered "
                        f"no-replacement draw: {totals}")
        sampling_sha = sampling_indices_sha256(indices)
        selected_x = synthetic_x[indices]
        selected_y = synthetic_y[indices]
        del synthetic_x

    lc.set_seed(job.pipeline_seed)
    real_x, real_y, val_x, val_y = _load_real_train_val(REPO)
    mean, std = lc.fit_standardizer(real_x)
    standardizer = verify_derived_standardizer(repo=REPO)

    if selected_x is None:
        train_x, train_y = real_x, real_y
    else:
        train_x = np.concatenate((real_x, selected_x), axis=0)
        train_y = np.concatenate((real_y, selected_y), axis=0)
        del real_x, selected_x
    train_x = lc.standardize(train_x, mean, std)
    val_x = lc.standardize(val_x, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The BiLSTM consumes (N, 128, 11) sequences; SequenceDataset keeps that
    # layout, unlike the CNN's channel-first WindowDataset.
    train_loader = DataLoader(
        lc.SequenceDataset(train_x, train_y), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        lc.SequenceDataset(val_x, val_y), batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = LSTMClassifier(in_features=11, hidden=128, classes=5).to(device)
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(5), y=train_y)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    selected_checkpoint = 0
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    loader_iterator = iter(train_loader)
    interval_loss_sum = 0.0
    interval_examples = 0
    while optimizer_steps < MAX_STEPS:
        try:
            batch_x, batch_y = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(train_loader)
            batch_x, batch_y = next(loader_iterator)
        model.train()
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        interval_loss_sum += float(loss.detach().cpu()) * len(batch_y)
        interval_examples += len(batch_y)
        if optimizer_steps % VALIDATION_CADENCE == 0:
            validation_checkpoint = len(history) + 1
            val_true, val_probs = predict(model, val_loader, device)
            score = float(f1_score(
                val_true, val_probs.argmax(axis=1),
                average="macro", zero_division=0))
            history.append({
                "validation_checkpoint": validation_checkpoint,
                "optimizer_step": optimizer_steps,
                "train_loss": float(interval_loss_sum / interval_examples),
                "val_macro_f1": score,
            })
            print(f"E19 construction={job.construction_seed} arm={job.arm} "
                  f"pipeline={job.pipeline_seed} step={optimizer_steps} "
                  f"val_macro_f1={score:.8f}", flush=True)
            if score > best_score:
                best_score = score
                selected_checkpoint = validation_checkpoint
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            interval_loss_sum = 0.0
            interval_examples = 0

    if (optimizer_steps != MAX_STEPS
            or len(history) != VALIDATION_CHECKPOINTS
            or best_state is None):
        raise _incomplete(
            f"{job.key}: fit did not complete the frozen budget "
            f"(steps={optimizer_steps}, checkpoints={len(history)})")
    model.load_state_dict(best_state)

    log_record = {
        "schema_version": TRAINING_LOG_SCHEMA,
        "experiment": "e19_bilstm_relational_crossing",
        "construction_seed": job.construction_seed,
        "arm": job.arm,
        "pipeline_seed": job.pipeline_seed,
        "role": job.role,
        "pool": ("" if job.pool_path is None
                 else str(job.pool_path.relative_to(REPO))),
        "pool_sha256": pool_sha,
        "sampling": sampling,
        "sampling_index_sha256": sampling_sha,
        "standardizer": standardizer,
        "max_optimizer_updates": MAX_STEPS,
        "optimizer_updates_performed": optimizer_steps,
        "validation_cadence": VALIDATION_CADENCE,
        "validation_checkpoints": len(history),
        "selected_validation_checkpoint": selected_checkpoint,
        "best_val_macro_f1": best_score,
        "history": history,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "started_utc": started,
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "checkpoint": str(job.checkpoint_path.relative_to(REPO)),
    }

    job.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        dir=job.checkpoint_path.parent,
        prefix=f".{job.checkpoint_path.name}.e19-stage-", suffix=".tmp")
    os.close(descriptor)
    staged_checkpoint = Path(staged_name)
    with open(staged_checkpoint, "wb") as handle:
        torch.save(model.state_dict(), handle)
        handle.flush()
        os.fsync(handle.fileno())
    staged_log = _stage_json(job.log_path, log_record)
    atomic_publish_bundle([
        (staged_checkpoint, job.checkpoint_path),
        (staged_log, job.log_path),
    ])
    log_record["checkpoint_sha256"] = _sha256_file(job.checkpoint_path)
    return log_record


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------


def _child_command(job: E19Job) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--construction-seed", str(job.construction_seed),
        "--arm", job.arm,
        "--pipeline-seed", str(job.pipeline_seed),
        "--role", job.role,
    ]


def verify_sampling_contract(rows: Sequence[Mapping[str, Any]]
                             ) -> dict[str, Any]:
    """PREREG 4.3 — identical-training-row contract against frozen E16."""
    frozen = pd.read_csv(E16_SAMPLING_AUDIT)
    frozen_map = {
        (int(r["construction_seed"]), str(r["arm"]), int(r["pipeline_seed"])):
            str(r["index_sha256"])
        for r in frozen.to_dict(orient="records")
    }
    mismatched: list[str] = []
    exempt: list[str] = []
    checked = 0
    for row in rows:
        if row["arm"] == REAL_ARM:
            continue
        key = (int(row["construction_seed"]), str(row["arm"]),
               int(row["pipeline_seed"]))
        frozen_sha = frozen_map.get(key)
        if frozen_sha is None:
            exempt.append(f"{key[0]}:{key[1]}:{key[2]}")
            continue
        checked += 1
        if str(row["sampling_index_sha256"]) != frozen_sha:
            mismatched.append(f"{key[0]}:{key[1]}:{key[2]}")
    expected_exempt = len(BRIDGE_SEEDS) * len(PIPELINE_SEEDS)  # bridge rule
    if mismatched:
        raise _stop(
            "identical-training-row contract FAILED: "
            f"{len(mismatched)} fits drew different rows than frozen E16: "
            f"{mismatched[:5]}")
    if len(exempt) != expected_exempt:
        raise _stop(
            f"{len(exempt)} fits lack a frozen E16 counterpart; registered "
            f"exemption covers exactly {expected_exempt} bridge Rule fits: "
            f"{exempt[:8]}")
    return {"gate": "identical_training_row_contract", "passed": True,
            "checked": checked, "exempt_bridge_rule_fits": len(exempt)}


def verify_pairing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired arms drew identical synthetic rows."""
    by_pair: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if row["arm"] == REAL_ARM:
            continue
        key = (int(row["construction_seed"]), int(row["pipeline_seed"]))
        by_pair.setdefault(key, {})[str(row["arm"])] = str(
            row["sampling_index_sha256"])
    mismatched = [
        f"{c}:{p}" for (c, p), arms in by_pair.items()
        if len(arms) == 2 and arms.get("rule") != arms.get("placebo")]
    if mismatched:
        raise _stop(
            f"paired arms drew different synthetic rows: {mismatched[:5]}")
    incomplete = [f"{c}:{p}" for (c, p), arms in by_pair.items()
                  if len(arms) != 2]
    return {"gate": "paired_draw_identity", "passed": True,
            "pairs_checked": len(by_pair), "incomplete_pairs": incomplete}


def verify_completeness(rows: Sequence[Mapping[str, Any]], *, expected: int
                        ) -> dict[str, Any]:
    if len(rows) != expected:
        raise _incomplete(f"{len(rows)} fits present, expected {expected}")
    bad_budget = [r for r in rows if int(r["optimizer_updates"]) != MAX_STEPS]
    bad_checkpoints = [
        r for r in rows
        if int(r["validation_checkpoints"]) != VALIDATION_CHECKPOINTS]
    if bad_budget or bad_checkpoints:
        raise _incomplete(
            f"{len(bad_budget)} fits off budget, "
            f"{len(bad_checkpoints)} fits off validation cadence")
    devices = sorted({str(r["device"]) for r in rows})
    return {"gate": "fit_completeness", "passed": True,
            "fits": len(rows), "devices": devices,
            "optimizer_updates": MAX_STEPS,
            "validation_checkpoints": VALIDATION_CHECKPOINTS}


def execute(*, limit: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    preflight = build_preflight()
    records = training_record_paths()
    publish_and_cleanup([
        (_stage_json(records["preflight"], preflight), records["preflight"]),
        (_stage_json(records["pool_inheritance"], preflight["pools"]),
         records["pool_inheritance"]),
    ])

    jobs = build_training_grid()
    if limit is not None:
        jobs = jobs[:limit]

    manifest_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    for position, job in enumerate(jobs, start=1):
        print(f"[{position}/{len(jobs)}] {job.key} ({job.role})", flush=True)
        completed = subprocess.run(_child_command(job), check=False)
        if completed.returncode != 0:
            print(f"child fit {job.key} exited {completed.returncode}; "
                  "retrying once per PREREG section 9", flush=True)
            completed = subprocess.run(_child_command(job), check=False)
            if completed.returncode != 0:
                raise _stop(
                    f"child fit {job.key} failed twice; stage halted with no "
                    "outcome interpretation")
        if not job.checkpoint_path.exists() or not job.log_path.exists():
            raise _incomplete(f"child fit {job.key} published no bundle")
        record = json.loads(job.log_path.read_text())
        manifest_rows.append({
            "construction_seed": job.construction_seed,
            "arm": job.arm,
            "pipeline_seed": job.pipeline_seed,
            "role": job.role,
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": _sha256_file(job.checkpoint_path),
            "pool_sha256": record["pool_sha256"],
            "sampling_index_sha256": record["sampling_index_sha256"],
            "optimizer_updates": record["optimizer_updates_performed"],
            "validation_checkpoints": record["validation_checkpoints"],
            "selected_validation_checkpoint": record[
                "selected_validation_checkpoint"],
            "best_val_macro_f1": record["best_val_macro_f1"],
            "device": record["device"],
            "elapsed_seconds": record["elapsed_seconds"],
        })
        if job.arm != REAL_ARM:
            totals = record["sampling"]["total"]
            sampling_rows.append({
                "construction_seed": job.construction_seed,
                "arm": job.arm,
                "pipeline_seed": job.pipeline_seed,
                "role": job.role,
                "requested": totals["requested"],
                "drawn": totals["drawn"],
                "unique": totals["unique"],
                "repeated": totals["repeated"],
                "index_sha256": record["sampling_index_sha256"],
            })

    pairing = verify_pairing(manifest_rows)
    contract = verify_sampling_contract(manifest_rows)
    completeness = verify_completeness(manifest_rows, expected=len(jobs))

    run_record = {
        "schema_version": TRAINING_RUN_SCHEMA,
        "stage": "e19_paired_training",
        "started_utc": preflight["built_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "fit_count": len(manifest_rows),
        "expected_fit_count": len(jobs),
        "pairing_gate": pairing,
        "identical_training_row_gate": contract,
        "completeness_gate": completeness,
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
    parser.add_argument("--execute", action="store_true",
                        help="run the canonical 245-fit grid")
    parser.add_argument("--child", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--construction-seed", type=int, default=0)
    parser.add_argument("--arm", choices=PAIRED_ARMS + (REAL_ARM,))
    parser.add_argument("--pipeline-seed", type=int)
    parser.add_argument("--role", default="primary")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N jobs (smoke checks only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.child:
        job = E19Job(args.construction_seed, args.arm,
                     args.pipeline_seed, args.role)
        record = run_child_fit(job)
        print(json.dumps(
            {k: v for k, v in record.items()
             if k not in ("history", "sampling")},
            indent=2, sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps(build_preflight(), indent=2, sort_keys=True,
                         default=str))
        return 0
    record = execute(limit=args.limit)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
