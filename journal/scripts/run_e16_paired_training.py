#!/usr/bin/env python3
"""Train the 220 preregistered E16 paired Rule/placebo fits.

Registered by ``journal/experiments/e16_relational_response_decomposition/PREREG.md``
sections 6, 14.2, and 15.3.

Grid
----
* 20 fresh constructions x 2 arms (Rule, placebo) x 5 pipeline seeds = 200
* 4 E15 bridge constructions x 1 arm (placebo only) x 5 pipeline seeds = 20

The five E14 matched-real checkpoints are the shared reference and are **not**
retrained here.

Pairing (PREREG section 6)
-------------------------
Within a construction and pipeline seed, the Rule and placebo fits share the
detector initialization seed, the synthetic row indices, the minibatch RNG
convention, and the optimizer-update budget.  Because the two arms of a pair are
array-aligned, their ``y_attack_type`` vectors are identical, so the registered
no-replacement draw returns the identical index vector.  Gate G10c asserts this
rather than assuming it.

Every fit runs in a fresh child interpreter so that the RNG state, the CUDA
context, and the 1.4 GiB pool allocation cannot leak across fits.  Importing
this module and the default CLI mode are read-only.
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
from train_generator_extension_cnn import CNN1D, predict  # noqa: E402
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
EXPERIMENT_DIR = (
    REPO / "journal" / "experiments" / "e16_relational_response_decomposition"
)
MODEL_DIR = REPO / "journal" / "models" / "generator_extension"
LOG_DIR = REPO / "journal" / "results" / "logs"
TABLE_DIR = REPO / "journal" / "results" / "tables"

CONSTRUCTION_SEEDS = e16gen.CONSTRUCTION_SEEDS
BRIDGE_SEEDS = e16gen.BRIDGE_SEEDS
PIPELINE_SEEDS = (7, 42, 123, 2026, 3407)
ARMS = ("rule", "placebo")

OUTPUT_VERSION = e16gen.OUTPUT_VERSION
MODEL_TAG = f"matchedsteps_e16_{OUTPUT_VERSION}"
SHARED_REAL_TEMPLATE = "cnn_real_only_matchedsteps_e14_v1_seed{pipeline}.pt"

TRAINING_PREFLIGHT_SCHEMA = "e16.training_preflight.v1"
TRAINING_RUN_SCHEMA = "e16.training_run.v1"
TRAINING_LOG_SCHEMA = "e16.paired_training_log.v1"
TRAINING_MANIFEST_SCHEMA = "e16.paired_training_manifest.v1"

EXPECTED_FIT_COUNT = len(CONSTRUCTION_SEEDS) * len(ARMS) * len(PIPELINE_SEEDS) \
    + len(BRIDGE_SEEDS) * len(PIPELINE_SEEDS)


class E16TrainingError(RuntimeError):
    """Raised on any registered training gate violation."""


def _stop(message: str) -> E16TrainingError:
    return E16TrainingError(f"T-STOP-E16-TRAIN: {message}")


def _incomplete(message: str) -> E16TrainingError:
    return E16TrainingError(f"T-INCOMPLETE-E16-TRAIN: {message}")


def utc_now() -> str:
    return e16gen.utc_now()


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E16Job:
    construction_seed: int
    arm: str
    pipeline_seed: int
    role: str

    @property
    def pool_path(self) -> Path:
        if self.arm == "rule":
            return e16gen.rule_pool_path(self.construction_seed, REPO)
        return e16gen.placebo_pool_path(self.construction_seed, REPO)

    @property
    def checkpoint_path(self) -> Path:
        return MODEL_DIR / (
            f"cnn_{self.arm}_0p30_cseed{self.construction_seed}_"
            f"{MODEL_TAG}_seed{self.pipeline_seed}.pt")

    @property
    def log_path(self) -> Path:
        return LOG_DIR / (
            f"train_cnn_{self.arm}_0p30_cseed{self.construction_seed}_"
            f"{MODEL_TAG}_seed{self.pipeline_seed}.log")

    @property
    def key(self) -> str:
        return f"{self.construction_seed}:{self.arm}:{self.pipeline_seed}"


def build_training_grid() -> tuple[E16Job, ...]:
    """Deterministic registered job order."""
    jobs: list[E16Job] = []
    for construction_seed in CONSTRUCTION_SEEDS:
        for arm in ARMS:
            for pipeline_seed in PIPELINE_SEEDS:
                jobs.append(E16Job(
                    construction_seed, arm, pipeline_seed, "primary"))
    for construction_seed in BRIDGE_SEEDS:
        for pipeline_seed in PIPELINE_SEEDS:
            jobs.append(E16Job(
                construction_seed, "placebo", pipeline_seed,
                "bridge_sensitivity"))
    if len(jobs) != EXPECTED_FIT_COUNT:
        raise _stop(f"grid size {len(jobs)} != registered {EXPECTED_FIT_COUNT}")
    if len({job.key for job in jobs}) != len(jobs):
        raise _stop("duplicate job key in the training grid")
    return tuple(jobs)


def shared_real_checkpoint(pipeline_seed: int) -> Path:
    return MODEL_DIR / SHARED_REAL_TEMPLATE.format(pipeline=pipeline_seed)


def training_record_paths() -> dict[str, Path]:
    return {
        "preflight": EXPERIMENT_DIR / f"training_preflight_{OUTPUT_VERSION}.json",
        "run": EXPERIMENT_DIR / f"training_run_{OUTPUT_VERSION}.json",
        "manifest": TABLE_DIR / f"e16_training_manifest_{OUTPUT_VERSION}.csv",
        "sampling_audit": TABLE_DIR / f"e16_sampling_audit_{OUTPUT_VERSION}.csv",
    }


def target_paths(jobs: Sequence[E16Job]) -> list[Path]:
    paths = [job.checkpoint_path for job in jobs]
    paths += [job.log_path for job in jobs]
    paths += list(training_record_paths().values())
    return paths


def assert_targets_absent(paths: Sequence[Path]) -> dict[str, Any]:
    present = [str(p.relative_to(REPO)) for p in paths
               if p.exists() or p.is_symlink()]
    if present:
        raise _stop(f"refusing to overwrite existing targets: {present[:8]}"
                    + (f" (+{len(present) - 8} more)" if len(present) > 8 else ""))
    return {"gate": "G10a_no_clobber", "passed": True, "checked": len(paths)}


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_label_digest(path: Path) -> str:
    with np.load(path, allow_pickle=False) as pool:
        labels = np.asarray(pool["y_attack_type"], dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()


def assert_pools_present_and_paired() -> dict[str, Any]:
    """Gate G10b/G10c — pools exist and paired arms are draw-identical."""
    rows: list[dict[str, Any]] = []
    for construction_seed in CONSTRUCTION_SEEDS:
        rule = e16gen.rule_pool_path(construction_seed, REPO)
        placebo = e16gen.placebo_pool_path(construction_seed, REPO)
        for path in (rule, placebo):
            if not path.exists():
                raise _stop(f"missing E16 pool: {path.relative_to(REPO)}")
        rule_labels = _pool_label_digest(rule)
        placebo_labels = _pool_label_digest(placebo)
        if rule_labels != placebo_labels:
            raise _stop(
                f"construction {construction_seed}: paired arms have different "
                "label vectors, so the registered draw cannot be identical")
        rows.append({
            "construction_seed": construction_seed,
            "role": "primary",
            "rule_pool_sha256": _sha256_file(rule),
            "placebo_pool_sha256": _sha256_file(placebo),
            "label_vector_sha256": rule_labels,
        })
    for construction_seed in BRIDGE_SEEDS:
        placebo = e16gen.placebo_pool_path(construction_seed, REPO)
        if not placebo.exists():
            raise _stop(f"missing bridge placebo pool: {placebo}")
        rows.append({
            "construction_seed": construction_seed,
            "role": "bridge_sensitivity",
            "rule_pool_sha256": "",
            "placebo_pool_sha256": _sha256_file(placebo),
            "label_vector_sha256": _pool_label_digest(placebo),
        })
    return {"gate": "G10bc_pools_present_and_paired", "passed": True,
            "constructions": len(rows), "rows": rows}


def assert_shared_real_present() -> dict[str, Any]:
    entries = {}
    for pipeline_seed in PIPELINE_SEEDS:
        path = shared_real_checkpoint(pipeline_seed)
        if not path.exists():
            raise _stop(f"missing shared E14 matched-real checkpoint: {path}")
        entries[path.name] = _sha256_file(path)
    return {"gate": "shared_real_reference", "passed": True,
            "checkpoints": entries}


def build_preflight() -> dict[str, Any]:
    jobs = build_training_grid()
    record: dict[str, Any] = {
        "schema_version": TRAINING_PREFLIGHT_SCHEMA,
        "stage": "e16_paired_training",
        "built_utc": utc_now(),
        "expected_fits": EXPECTED_FIT_COUNT,
        "grid": {
            "constructions": len(CONSTRUCTION_SEEDS),
            "arms": list(ARMS),
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "bridge_constructions": len(BRIDGE_SEEDS),
        },
        "protocol": {
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
    record["registration"] = e16gen.assert_registration_committed(REPO)
    record["environment"] = _environment_record(REPO)
    record["source"] = _source_record(REPO)
    record["pools"] = assert_pools_present_and_paired()
    record["shared_real"] = assert_shared_real_present()
    record["standardizer"] = verify_derived_standardizer(repo=REPO)
    record["target_absence"] = assert_targets_absent(target_paths(jobs))
    return record


# ---------------------------------------------------------------------------
# Child fit
# ---------------------------------------------------------------------------


def run_child_fit(job: E16Job) -> dict[str, Any]:
    """Train and atomically publish exactly one E16 fit."""
    assert_targets_absent([job.checkpoint_path, job.log_path])
    if not job.pool_path.exists():
        raise _stop(f"missing pool for {job.key}: {job.pool_path}")

    started = utc_now()
    start_time = time.monotonic()

    with np.load(job.pool_path, allow_pickle=False) as pool:
        synthetic_x = np.asarray(pool["x"], dtype=np.float32)
        synthetic_y = np.asarray(pool["y_attack_type"], dtype=np.int64)
        pool_construction = int(pool["construction_seed"])
    if pool_construction != job.construction_seed:
        raise _stop(
            f"pool construction seed {pool_construction} != job "
            f"{job.construction_seed}")

    indices, sampling = sample_indices_without_replacement(
        synthetic_y, SYNTHETIC_TOTAL, job.pipeline_seed + SAMPLING_SEED_OFFSET)
    totals = sampling["total"]
    if (totals["repeated"] != 0
            or totals["drawn"] != totals["unique"]
            or totals["requested"] != totals["drawn"]
            or totals["requested"] != SYNTHETIC_TOTAL):
        raise _stop(f"{job.key}: draw is not the registered no-replacement "
                    f"draw: {totals}")
    selected_x = synthetic_x[indices]
    selected_y = synthetic_y[indices]

    lc.set_seed(job.pipeline_seed)
    real_x, real_y, val_x, val_y = _load_real_train_val(REPO)
    mean, std = lc.fit_standardizer(real_x)
    standardizer = verify_derived_standardizer(repo=REPO)

    train_x = np.concatenate((real_x, selected_x), axis=0)
    train_y = np.concatenate((real_y, selected_y), axis=0)
    del real_x, selected_x, synthetic_x
    train_x = lc.standardize(train_x, mean, std)
    val_x = lc.standardize(val_x, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        lc.WindowDataset(train_x, train_y), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        lc.WindowDataset(val_x, val_y), batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = CNN1D(in_channels=11, classes=5).to(device)
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
            print(f"E16 construction={job.construction_seed} arm={job.arm} "
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
        "experiment": "e16_relational_response_decomposition",
        "construction_seed": job.construction_seed,
        "arm": job.arm,
        "pipeline_seed": job.pipeline_seed,
        "role": job.role,
        "pool": str(job.pool_path.relative_to(REPO)),
        "pool_sha256": _sha256_file(job.pool_path),
        "sampling": sampling,
        "sampling_index_sha256": sampling_indices_sha256(indices),
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
        prefix=f".{job.checkpoint_path.name}.e16-stage-", suffix=".tmp")
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

def _child_command(job: E16Job) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--construction-seed", str(job.construction_seed),
        "--arm", job.arm,
        "--pipeline-seed", str(job.pipeline_seed),
        "--role", job.role,
    ]


def execute(*, limit: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    preflight = build_preflight()
    records = training_record_paths()
    staged = [(_stage_json(records["preflight"], preflight), records["preflight"])]
    publish_and_cleanup(staged)

    jobs = build_training_grid()
    if limit is not None:
        jobs = jobs[:limit]

    manifest_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    for position, job in enumerate(jobs, start=1):
        print(f"[{position}/{len(jobs)}] {job.key} ({job.role})", flush=True)
        completed = subprocess.run(_child_command(job), check=False)
        if completed.returncode != 0:
            raise _stop(
                f"child fit {job.key} exited {completed.returncode}; "
                "stage halted with no outcome interpretation")
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
        totals = record["sampling"]["total"]
        sampling_rows.append({
            "construction_seed": job.construction_seed,
            "arm": job.arm,
            "pipeline_seed": job.pipeline_seed,
            "requested": totals["requested"],
            "drawn": totals["drawn"],
            "unique": totals["unique"],
            "repeated": totals["repeated"],
            "index_sha256": record["sampling_index_sha256"],
        })

    pairing = verify_pairing(manifest_rows)
    completeness = verify_completeness(manifest_rows, expected=len(jobs))

    run_record = {
        "schema_version": TRAINING_RUN_SCHEMA,
        "stage": "e16_paired_training",
        "started_utc": preflight["built_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "fresh_fit_count": len(manifest_rows),
        "expected_fit_count": len(jobs),
        "pairing_gate": pairing,
        "completeness_gate": completeness,
        "preflight_path": str(records["preflight"].relative_to(REPO)),
        "status": "T-PASS",
        "technical_status": "T-PASS",
    }
    staged = [
        (_stage_bytes(records["manifest"], _csv_bytes(manifest_rows)),
         records["manifest"]),
        (_stage_bytes(records["sampling_audit"], _csv_bytes(sampling_rows)),
         records["sampling_audit"]),
        (_stage_json(records["run"], run_record), records["run"]),
    ]
    publish_and_cleanup(staged)
    return run_record


def verify_pairing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate G10c — paired arms drew identical synthetic rows."""
    by_pair: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if row["role"] != "primary":
            continue
        key = (int(row["construction_seed"]), int(row["pipeline_seed"]))
        by_pair.setdefault(key, {})[str(row["arm"])] = str(
            row["sampling_index_sha256"])
    mismatched = [
        f"{c}:{p}" for (c, p), arms in by_pair.items()
        if len(arms) == 2 and arms.get("rule") != arms.get("placebo")]
    incomplete = [f"{c}:{p}" for (c, p), arms in by_pair.items()
                  if len(arms) != 2]
    if mismatched:
        raise _stop(
            f"paired arms drew different synthetic rows: {mismatched[:5]}")
    return {"gate": "G10c_paired_draw_identity", "passed": True,
            "pairs_checked": len(by_pair), "incomplete_pairs": incomplete}


def verify_completeness(rows: Sequence[Mapping[str, Any]], *, expected: int
                        ) -> dict[str, Any]:
    """Gate G10d — every registered fit completed the frozen budget."""
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
    return {"gate": "G10d_fit_completeness", "passed": True,
            "fits": len(rows), "devices": devices,
            "optimizer_updates": MAX_STEPS,
            "validation_checkpoints": VALIDATION_CHECKPOINTS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="run the canonical 220-fit grid")
    parser.add_argument("--child", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--construction-seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--pipeline-seed", type=int)
    parser.add_argument("--role", default="primary")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N jobs (smoke checks only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.child:
        job = E16Job(args.construction_seed, args.arm,
                     args.pipeline_seed, args.role)
        record = run_child_fit(job)
        print(json.dumps(
            {k: v for k, v in record.items() if k != "history"},
            indent=2, sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps(build_preflight(), indent=2, sort_keys=True))
        return 0
    record = execute(limit=args.limit)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
