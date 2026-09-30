#!/usr/bin/env python3
"""Score the E19 BiLSTM grid on the frozen E14 L4 factorial.

Registered by ``journal/experiments/e19_bilstm_relational_crossing/PREREG.md``
section 6.  Scoring device is CUDA, declared in the registration: no frozen
BiLSTM integer record exists, so no cross-run integer-equality gate binds the
device.  Completeness gates replace the E16 G12 continuity gate: exactly 245
identities, 96 scenario rows per identity, untouched-base normal outcomes
reported first.

Importing this module performs no inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_l4_counterfactual_factorial as e14  # noqa: E402
import evaluate_e16_relational_response as e16eval  # noqa: E402
import run_e19_bilstm_paired_training as e19train  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _source_record,
    _stage_bytes,
    _stage_json,
    atomic_publish_bundle,
)

REPO = e19train.REPO
EXPERIMENT_DIR = e19train.EXPERIMENT_DIR
TABLE_DIR = e19train.TABLE_DIR

BLOCKS = e16eval.BLOCKS
ID_STRATA = e16eval.ID_STRATA
FACTOR_CELLS = e16eval.FACTOR_CELLS
PIPELINE_SEEDS = e19train.PIPELINE_SEEDS

OUTPUT_VERSION = e19train.OUTPUT_VERSION
PREPARE_SCHEMA = "e19.evaluation_prepare.v1"
RUN_SCHEMA = "e19.evaluation_run.v1"

REAL_ARM_LABEL = "real_bilstm"
SCORING_DEVICE = "cuda"
SCORING_BATCH_SIZE = 4_096


class E19EvaluationError(RuntimeError):
    """Raised on any registered evaluation gate violation."""


def _stop(message: str) -> E19EvaluationError:
    return E19EvaluationError(f"T-STOP-E19-EVAL: {message}")


def _incomplete(message: str) -> E19EvaluationError:
    return E19EvaluationError(f"T-INCOMPLETE-E19-EVAL: {message}")


def utc_now() -> str:
    return e19train.utc_now()


_sha256_file = e19train._sha256_file
publish_and_cleanup = e19train.publish_and_cleanup


def output_paths() -> dict[str, Path]:
    return {
        "prepare": EXPERIMENT_DIR / f"prepare_{OUTPUT_VERSION}.json",
        "run": EXPERIMENT_DIR / f"run_{OUTPUT_VERSION}.json",
        "by_scenario": TABLE_DIR / f"e19_by_scenario_{OUTPUT_VERSION}.csv",
        "by_cell": TABLE_DIR / f"e19_by_cell_{OUTPUT_VERSION}.csv",
        "normal": TABLE_DIR / f"e19_normal_by_arm_{OUTPUT_VERSION}.csv",
    }


def build_identities() -> list[dict[str, Any]]:
    """Every checkpoint scored by this stage, in registered order."""
    identities: list[dict[str, Any]] = []
    for job in e19train.build_training_grid():
        arm = REAL_ARM_LABEL if job.arm == e19train.REAL_ARM else job.arm
        identities.append({
            "model_key": (arm, job.construction_seed, job.pipeline_seed),
            "arm": arm,
            "construction_seed": job.construction_seed,
            "pipeline_seed": job.pipeline_seed,
            "role": ("shared_reference" if job.arm == e19train.REAL_ARM
                     else job.role),
            "checkpoint": job.checkpoint_path,
        })
    keys = [identity["model_key"] for identity in identities]
    if len(set(keys)) != len(keys):
        raise _stop("duplicate model key among evaluation identities")
    if len(identities) != e19train.EXPECTED_FIT_COUNT:
        raise _stop(f"{len(identities)} identities, registered "
                    f"{e19train.EXPECTED_FIT_COUNT}")
    return identities


def assert_checkpoints_present(identities: Sequence[Mapping[str, Any]]
                               ) -> dict[str, Any]:
    missing = [str(i["checkpoint"].relative_to(REPO)) for i in identities
               if not i["checkpoint"].exists()]
    if missing:
        raise _stop(f"missing {len(missing)} checkpoint(s): {missing[:5]}")
    return {"gate": "checkpoints_present", "passed": True,
            "count": len(identities)}


def _configure_torch() -> Any:
    import torch

    if SCORING_DEVICE == "cuda" and not torch.cuda.is_available():
        raise _stop("registered scoring device cuda is unavailable")
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def _load_model(torch: Any, path: Path) -> Any:
    from train_family_extension_lstm import LSTMClassifier

    model = LSTMClassifier(in_features=11, hidden=128, classes=5)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(SCORING_DEVICE)


def _predict_classes(torch: Any, model: Any, windows: np.ndarray,
                     mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = e14._standardize(windows, mean, std)
    predictions = np.empty(len(standardized), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(standardized), SCORING_BATCH_SIZE):
            batch = torch.from_numpy(np.ascontiguousarray(
                standardized[start:start + SCORING_BATCH_SIZE]))
            batch = batch.to(SCORING_DEVICE, non_blocking=True)
            logits = model(batch)
            predictions[start:start + len(batch)] = (
                logits.argmax(dim=1).cpu().numpy())
    return predictions


def score_identities(
        torch: Any,
        identities: Sequence[Mapping[str, Any]],
        models: Mapping[Any, Any],
        base_manifest: pd.DataFrame,
        bases: np.ndarray,
        latents: Sequence[Any],
        mean: np.ndarray,
        std: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror of the E16 scoring loop with a device-aware predictor."""
    block_orders = {
        block: base_manifest.index[
            base_manifest["block_id"].astype(str).eq(block)].to_numpy(np.int64)
        for block in BLOCKS
    }
    if any(len(order) != 2_000 for order in block_orders.values()):
        raise _incomplete("L4 block identity is incomplete")
    latent_lookup = {
        (latent.block_id, latent.block_position,
         latent.test_window_index, latent.attack_label): latent
        for latent in latents
    }

    normal_rows: list[dict[str, Any]] = []
    for identity in identities:
        predictions = _predict_classes(
            torch, models[identity["model_key"]], bases, mean, std)
        for block, order in block_orders.items():
            correct = int(np.count_nonzero(predictions[order] == 0))
            normal_rows.append({
                "arm": identity["arm"],
                "construction_seed": identity["construction_seed"],
                "pipeline_seed": identity["pipeline_seed"],
                "role": identity["role"],
                "block_id": block,
                "n": len(order),
                "normal_correct": correct,
                "normal_recall": correct / len(order),
                "fpr": 1.0 - correct / len(order),
                "checkpoint_sha256": identity["checkpoint_sha256"],
            })

    scenario_rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        order = block_orders[block]
        block_manifest = base_manifest.loc[order]
        block_bases = bases[order]
        for attack_label in sorted(e14.ATTACK_SPECS):
            attack = str(e14.ATTACK_SPECS[attack_label]["attack"])
            block_latents = [
                latent_lookup[(block, int(row.block_position),
                               int(row.test_window_index), attack_label)]
                for row in block_manifest.itertuples(index=False)
            ]
            for id_stratum in ID_STRATA:
                for p, s, d in FACTOR_CELLS:
                    transformed = np.stack([
                        e14.apply_factorial_transform(
                            base, latent, attack_label=attack_label,
                            id_stratum=id_stratum, p=p, s=s, d=d)
                        for base, latent in zip(
                            block_bases, block_latents, strict=True)
                    ]).astype(np.float32, copy=False)
                    for identity in identities:
                        predictions = _predict_classes(
                            torch, models[identity["model_key"]], transformed,
                            mean, std)
                        exact_correct = int(
                            np.count_nonzero(predictions == attack_label))
                        binary_correct = int(
                            np.count_nonzero(predictions != 0))
                        scenario_rows.append({
                            "arm": identity["arm"],
                            "construction_seed": identity["construction_seed"],
                            "pipeline_seed": identity["pipeline_seed"],
                            "role": identity["role"],
                            "block_id": block,
                            "attack": attack,
                            "attack_label": attack_label,
                            "id_stratum": id_stratum,
                            "P": p, "S": s, "D": d, "cell": f"{p}{s}{d}",
                            "n": len(transformed),
                            "exact_correct": exact_correct,
                            "binary_correct": binary_correct,
                            "exact_recall": exact_correct / len(transformed),
                            "binary_recall": binary_correct / len(transformed),
                            "checkpoint_sha256": identity["checkpoint_sha256"],
                        })
        print(f"  scored {block} ({len(scenario_rows)} scenario rows)",
              flush=True)
    return pd.DataFrame(normal_rows), pd.DataFrame(scenario_rows)


def run_prepare() -> dict[str, Any]:
    identities = build_identities()
    record: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "stage": "e19_evaluation_prepare",
        "created_utc": utc_now(),
        "model_inference_performed": False,
        "identities": len(identities),
        "scoring": {
            "device": SCORING_DEVICE,
            "batch_size": SCORING_BATCH_SIZE,
        },
    }
    record["registration"] = e19train.assert_registration_committed()
    record["environment"] = _environment_record(REPO)
    record["source"] = _source_record(REPO)
    record["checkpoints"] = assert_checkpoints_present(identities)
    present = [str(p.relative_to(REPO)) for p in output_paths().values()
               if p.exists() and p.name != f"prepare_{OUTPUT_VERSION}.json"]
    if present:
        raise _stop(f"refusing to overwrite existing outputs: {present}")
    return record


def execute() -> dict[str, Any]:
    started = time.monotonic()
    prepare = run_prepare()
    paths = output_paths()
    publish_and_cleanup([(_stage_json(paths["prepare"], prepare),
                          paths["prepare"])])

    torch = _configure_torch()
    identities = build_identities()
    for identity in identities:
        identity["checkpoint_sha256"] = _sha256_file(identity["checkpoint"])

    base_manifest, bases, latents = e14.load_prepared_inputs(
        json.loads((e14.EXP / "prepare_v1.json").read_text()))
    mean, std, standardizer = e14.fit_registered_standardizer()

    print(f"scoring {len(identities)} E19 checkpoints on {SCORING_DEVICE}",
          flush=True)
    models = {i["model_key"]: _load_model(torch, i["checkpoint"])
              for i in identities}
    normal, scenario = score_identities(
        torch, identities, models, base_manifest, bases, latents, mean, std)
    by_cell = e16eval.build_by_cell(scenario)

    expected_scenario = (
        len(identities) * len(BLOCKS) * len(e14.ATTACK_SPECS)
        * len(ID_STRATA) * len(FACTOR_CELLS))
    if len(scenario) != expected_scenario:
        raise _incomplete(
            f"{len(scenario)} scenario rows, expected {expected_scenario}")

    run_record = {
        "schema_version": RUN_SCHEMA,
        "stage": "e19_evaluation",
        "started_utc": prepare["created_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "device": SCORING_DEVICE,
        "batch_size": SCORING_BATCH_SIZE,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "standardizer": standardizer,
        "exact_and_binary_endpoints_separate": True,
        "normal_control_reported_before_attack_results": True,
        "identities": len(identities),
        "completeness": {
            "scenario_rows": len(scenario),
            "expected_scenario_rows": expected_scenario,
            "normal_rows": len(normal),
            "by_cell_rows": len(by_cell),
        },
        "status": "T-PASS",
        "technical_status": "T-PASS",
    }

    publish_and_cleanup([
        (_stage_bytes(paths["normal"], normal.to_csv(index=False).encode()),
         paths["normal"]),
        (_stage_bytes(paths["by_scenario"],
                      scenario.to_csv(index=False).encode()),
         paths["by_scenario"]),
        (_stage_bytes(paths["by_cell"], by_cell.to_csv(index=False).encode()),
         paths["by_cell"]),
        (_stage_json(paths["run"], run_record), paths["run"]),
    ])
    return run_record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    record = execute() if args.execute else run_prepare()
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
