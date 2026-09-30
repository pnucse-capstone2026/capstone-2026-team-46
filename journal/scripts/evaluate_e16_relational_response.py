#!/usr/bin/env python3
"""Score the E16 paired Rule/placebo grid on the frozen E14 L4 factorial.

Registered by ``journal/experiments/e16_relational_response_decomposition/PREREG.md``
sections 4.3, 7, and 15.3.

Stages
------
``--prepare-only``  read-only: bind the frozen E14 evaluation material, the
                    E16 checkpoints, and the shared E14 matched-real
                    references.  Performs no inference.
``--execute``       gate G12 (shared-real runtime continuity) first, then score
                    every E16 checkpoint and publish exact/binary tables.

Gate G12 is mandatory and blocking: the five shared E14 matched-real
checkpoints are re-scored under this evaluator and every integer count must
equal the frozen E14 record across all 495 continuity rows.  No E16 checkpoint
is scored until it passes.  Scoring runs on CPU because the E14/E15 records were
produced on CPU and integer-count equality is the gate; the device is not
changed.

Importing this module performs no inference and does not import torch.
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
import generate_e16_rule_placebo_pairs as e16gen  # noqa: E402
import run_e16_paired_training as e16train  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _source_record,
    _stage_bytes,
    _stage_json,
    atomic_publish_bundle,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    REPO / "journal" / "experiments" / "e16_relational_response_decomposition"
)
E14_DIR = REPO / "journal" / "experiments" / "e14_l4_counterfactual_factorial"
TABLE_DIR = REPO / "journal" / "results" / "tables"

BLOCKS = ("block_01", "block_02", "block_03")
ID_STRATA = ("canonical", "shifted")
FACTOR_CELLS = tuple(
    (p, s, d) for p in (0, 1) for s in (0, 1) for d in (0, 1))
PIPELINE_SEEDS = e16train.PIPELINE_SEEDS

OUTPUT_VERSION = e16gen.OUTPUT_VERSION
PREPARE_SCHEMA = "e16.evaluation_prepare.v1"
RUN_SCHEMA = "e16.evaluation_run.v1"
ENVIRONMENT_SCHEMA = "e16.evaluation_environment.v1"

SCORING_DEVICE = "cpu"
SCORING_BATCH_SIZE = e14.SCORING_BATCH_SIZE
TORCH_THREADS = e14.TORCH_THREADS
TORCH_INTEROP_THREADS = 1

CONTINUITY_SCENARIO_ROWS = 480
CONTINUITY_NORMAL_ROWS = 15
CONTINUITY_ROWS = CONTINUITY_SCENARIO_ROWS + CONTINUITY_NORMAL_ROWS

E14_SCENARIO_TABLE = TABLE_DIR / "e14_l4_factorial_by_scenario_v1.csv"
E14_NORMAL_TABLE = TABLE_DIR / "e14_l4_factorial_normal_by_seed_v1.csv"


class E16EvaluationError(RuntimeError):
    """Raised on any registered evaluation gate violation."""


def _stop(message: str) -> E16EvaluationError:
    return E16EvaluationError(f"T-STOP-E16-EVAL: {message}")


def _incomplete(message: str) -> E16EvaluationError:
    return E16EvaluationError(f"T-INCOMPLETE-E16-EVAL: {message}")


def utc_now() -> str:
    return e16gen.utc_now()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()



def publish_and_cleanup(staged: Sequence[tuple[Path, Path]]) -> None:
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

def output_paths() -> dict[str, Path]:
    return {
        "prepare": EXPERIMENT_DIR / f"prepare_{OUTPUT_VERSION}.json",
        "run": EXPERIMENT_DIR / f"run_{OUTPUT_VERSION}.json",
        "by_scenario": TABLE_DIR / f"e16_by_scenario_{OUTPUT_VERSION}.csv",
        "by_cell": TABLE_DIR / f"e16_by_cell_{OUTPUT_VERSION}.csv",
        "normal": TABLE_DIR / f"e16_normal_by_arm_{OUTPUT_VERSION}.csv",
        "continuity": TABLE_DIR
        / f"e16_real_runtime_continuity_{OUTPUT_VERSION}.csv",
    }


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def e15_bridge_rule_checkpoint(construction_seed: int, pipeline_seed: int
                               ) -> Path:
    """Frozen E15 v2 Rule checkpoint reused for the bridge sensitivity.

    The bridge arm's Rule outcomes are already observed (PREREG section 12.9),
    which is exactly why the bridge is a post-hoc control and not part of the
    primary.  These checkpoints are reused, never retrained, and their hashes
    are recorded before scoring.
    """
    return (REPO / "journal" / "models" / "generator_extension"
            / f"cnn_rule_0p30_cseed{construction_seed}_matchedsteps_e15_v2_"
              f"seed{pipeline_seed}.pt")


def build_identities() -> list[dict[str, Any]]:
    """Every checkpoint scored by this stage, in registered order."""
    identities: list[dict[str, Any]] = []
    for pipeline_seed in PIPELINE_SEEDS:
        path = e16train.shared_real_checkpoint(pipeline_seed)
        identities.append({
            "model_key": ("real_ms", 0, pipeline_seed),
            "arm": "real_ms",
            "construction_seed": 0,
            "pipeline_seed": pipeline_seed,
            "role": "shared_reference",
            "checkpoint": path,
        })
    for job in e16train.build_training_grid():
        identities.append({
            "model_key": (job.arm, job.construction_seed, job.pipeline_seed),
            "arm": job.arm,
            "construction_seed": job.construction_seed,
            "pipeline_seed": job.pipeline_seed,
            "role": job.role,
            "checkpoint": job.checkpoint_path,
        })
    # The bridge placebo fits are E16; their Rule counterparts are the frozen
    # E15 checkpoints.  Without them the bridge sensitivity has no Rule arm and
    # theta cannot be formed for those four constructions.
    for construction_seed in e16train.BRIDGE_SEEDS:
        for pipeline_seed in PIPELINE_SEEDS:
            identities.append({
                "model_key": ("rule", construction_seed, pipeline_seed),
                "arm": "rule",
                "construction_seed": construction_seed,
                "pipeline_seed": pipeline_seed,
                "role": "bridge_sensitivity",
                "checkpoint": e15_bridge_rule_checkpoint(
                    construction_seed, pipeline_seed),
            })
    keys = [identity["model_key"] for identity in identities]
    if len(set(keys)) != len(keys):
        raise _stop("duplicate model key among evaluation identities")
    return identities


def assert_checkpoints_present(identities: Sequence[Mapping[str, Any]]
                               ) -> dict[str, Any]:
    missing = [str(i["checkpoint"].relative_to(REPO)) for i in identities
               if not i["checkpoint"].exists()]
    if missing:
        raise _stop(f"missing {len(missing)} checkpoint(s): {missing[:5]}")
    return {"gate": "checkpoints_present", "passed": True,
            "count": len(identities)}


# ---------------------------------------------------------------------------
# Torch configuration
# ---------------------------------------------------------------------------


def _configure_torch() -> Any:
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def _load_model(torch: Any, path: Path) -> Any:
    model = e14._make_cnn(torch)
    state = torch.load(path, map_location=SCORING_DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class InferenceLedger:
    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def record(self, rows: int) -> None:
        self.calls += 1
        self.rows += rows

    def as_dict(self) -> dict[str, int]:
        return {"inference_calls": self.calls, "prediction_rows": self.rows}


def _predict_classes(torch: Any, model: Any, windows: np.ndarray,
                     mean: np.ndarray, std: np.ndarray,
                     ledger: InferenceLedger) -> np.ndarray:
    standardized = e14._standardize(windows, mean, std)
    logits = e14.predict_standardized_logits(
        torch, model, standardized, batch_size=SCORING_BATCH_SIZE)
    ledger.record(len(windows))
    return np.asarray(logits).argmax(axis=1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_identities(
        torch: Any,
        identities: Sequence[Mapping[str, Any]],
        models: Mapping[Any, Any],
        base_manifest: pd.DataFrame,
        bases: np.ndarray,
        latents: Sequence[Any],
        mean: np.ndarray,
        std: np.ndarray,
        ledger: InferenceLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score untouched bases and every registered L4 transformation."""
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
            torch, models[identity["model_key"]], bases, mean, std, ledger)
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
                            mean, std, ledger)
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


# ---------------------------------------------------------------------------
# Gate G12 — shared-real runtime continuity
# ---------------------------------------------------------------------------


def build_runtime_continuity(normal: pd.DataFrame, scenario: pd.DataFrame
                             ) -> pd.DataFrame:
    """Compare re-scored shared-real counts against the frozen E14 record."""
    e14_scenario = pd.read_csv(E14_SCENARIO_TABLE)
    e14_scenario = e14_scenario[e14_scenario["arm"].astype(str) == "real_ms"]
    e14_normal = pd.read_csv(E14_NORMAL_TABLE)
    e14_normal = e14_normal[e14_normal["arm"].astype(str) == "real_ms"]

    mine_scenario = scenario[scenario["arm"].astype(str) == "real_ms"].copy()
    mine_normal = normal[normal["arm"].astype(str) == "real_ms"].copy()

    scenario_keys = ["block_id", "attack", "id_stratum", "P", "S", "D"]
    merged = mine_scenario.merge(
        e14_scenario, left_on=scenario_keys + ["pipeline_seed"],
        right_on=scenario_keys + ["seed"], suffixes=("_e16", "_e14"),
        how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise _incomplete(
            "shared-real continuity grid is incomplete: "
            f"{merged['_merge'].value_counts().to_dict()}")

    rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        for field in ("exact_correct", "binary_correct"):
            rows.append({
                "row_type": "scenario",
                "pipeline_seed": record["pipeline_seed"],
                "block_id": record["block_id"],
                "attack": record["attack"],
                "id_stratum": record["id_stratum"],
                "cell": f"{record['P']}{record['S']}{record['D']}",
                "metric": field,
                "e14_value": int(record[f"{field}_e14"]),
                "e16_value": int(record[f"{field}_e16"]),
                "difference": int(record[f"{field}_e16"])
                - int(record[f"{field}_e14"]),
                "denominator": int(record["n_e16"]),
            })

    normal_merged = mine_normal.merge(
        e14_normal, left_on=["block_id", "pipeline_seed"],
        right_on=["block_id", "seed"], suffixes=("_e16", "_e14"),
        how="outer", indicator=True)
    if not (normal_merged["_merge"] == "both").all():
        raise _incomplete("shared-real normal continuity grid is incomplete")
    for record in normal_merged.to_dict(orient="records"):
        rows.append({
            "row_type": "normal",
            "pipeline_seed": record["pipeline_seed"],
            "block_id": record["block_id"],
            "attack": "", "id_stratum": "", "cell": "",
            "metric": "normal_correct",
            "e14_value": int(record["normal_correct_e14"]),
            "e16_value": int(record["normal_correct_e16"]),
            "difference": int(record["normal_correct_e16"])
            - int(record["normal_correct_e14"]),
            "denominator": int(record["n_e16"]),
        })
    return pd.DataFrame(rows)


def require_runtime_continuity(continuity: pd.DataFrame) -> dict[str, Any]:
    """Gate G12 — blocking; every integer count difference must be zero."""
    scenario_rows = int((continuity["row_type"] == "scenario").sum()) // 2
    normal_rows = int((continuity["row_type"] == "normal").sum())
    total_rows = scenario_rows + normal_rows
    if scenario_rows != CONTINUITY_SCENARIO_ROWS or normal_rows != CONTINUITY_NORMAL_ROWS:
        raise _incomplete(
            f"continuity grid has {scenario_rows} scenario and {normal_rows} "
            f"normal rows; registered {CONTINUITY_SCENARIO_ROWS}/"
            f"{CONTINUITY_NORMAL_ROWS}")
    nonzero = continuity[continuity["difference"] != 0]
    if len(nonzero):
        raise _stop(
            f"shared-real runtime continuity FAILED: {len(nonzero)} of "
            f"{len(continuity)} integer counts differ from the E14 record; "
            "E16 checkpoint scoring is blocked")
    return {
        "gate": "G12_shared_real_runtime_continuity",
        "passed": True,
        "status": "T-PASS",
        "rows": total_rows,
        "scenario_rows": scenario_rows,
        "normal_rows": normal_rows,
        "integer_comparisons": len(continuity),
        "all_integer_count_differences_zero": True,
        "device": SCORING_DEVICE,
    }


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def build_by_cell(scenario: pd.DataFrame) -> pd.DataFrame:
    """Gear/RPM equal-macro per (arm, construction, pipeline, stratum, cell)."""
    grouped = (
        scenario.groupby(
            ["arm", "construction_seed", "pipeline_seed", "role",
             "id_stratum", "P", "S", "D", "cell", "attack"],
            as_index=False)
        .agg(exact_recall=("exact_recall", "mean"),
             binary_recall=("binary_recall", "mean"),
             n=("n", "sum"))
    )
    macro = (
        grouped.groupby(
            ["arm", "construction_seed", "pipeline_seed", "role",
             "id_stratum", "P", "S", "D", "cell"], as_index=False)
        .agg(exact_recall=("exact_recall", "mean"),
             binary_recall=("binary_recall", "mean"))
    )
    macro["attack_scope"] = "equal_macro"
    per_attack = grouped.rename(columns={"attack": "attack_scope"})
    return pd.concat([macro, per_attack], ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _load_e14_prepare() -> dict[str, Any]:
    path = E14_DIR / "prepare_v1.json"
    if not path.exists():
        raise _stop(f"missing frozen E14 prepare record: {path}")
    return json.loads(path.read_text())


def run_prepare() -> dict[str, Any]:
    """Read-only: bind inputs and identities; performs no inference."""
    identities = build_identities()
    record: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "stage": "e16_evaluation_prepare",
        "created_utc": utc_now(),
        "model_inference_performed": False,
        "identities": len(identities),
        "grid": {
            "blocks": list(BLOCKS),
            "id_strata": list(ID_STRATA),
            "factor_cells": ["".join(map(str, c)) for c in FACTOR_CELLS],
            "pipeline_seeds": list(PIPELINE_SEEDS),
        },
        "scoring": {
            "device": SCORING_DEVICE,
            "batch_size": SCORING_BATCH_SIZE,
            "torch_threads": TORCH_THREADS,
            "torch_interop_threads": TORCH_INTEROP_THREADS,
        },
    }
    record["registration"] = e16gen.assert_registration_committed(REPO)
    record["environment"] = _environment_record(REPO)
    record["source"] = _source_record(REPO)
    record["checkpoints"] = assert_checkpoints_present(identities)
    record["e14_prepare_sha256"] = _sha256_file(E14_DIR / "prepare_v1.json")
    record["e14_continuity_tables"] = {
        str(E14_SCENARIO_TABLE.relative_to(REPO)): _sha256_file(E14_SCENARIO_TABLE),
        str(E14_NORMAL_TABLE.relative_to(REPO)): _sha256_file(E14_NORMAL_TABLE),
    }
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

    base_manifest, bases, latents = e14.load_prepared_inputs(_load_e14_prepare())
    mean, std, standardizer = e14.fit_registered_standardizer()

    # --- gate G12 first: shared real only ---------------------------------
    shared = [i for i in identities if i["role"] == "shared_reference"]
    print(f"gate G12: re-scoring {len(shared)} shared-real checkpoints on "
          f"{SCORING_DEVICE}", flush=True)
    ledger = InferenceLedger()
    shared_models = {i["model_key"]: _load_model(torch, i["checkpoint"])
                     for i in shared}
    shared_normal, shared_scenario = score_identities(
        torch, shared, shared_models, base_manifest, bases, latents,
        mean, std, ledger)
    continuity = build_runtime_continuity(shared_normal, shared_scenario)
    continuity_gate = require_runtime_continuity(continuity)
    print(f"gate G12 PASS: {continuity_gate['integer_comparisons']} integer "
          "counts identical to the E14 record", flush=True)
    del shared_models

    # --- E16 checkpoints ---------------------------------------------------
    e16_identities = [i for i in identities if i["role"] != "shared_reference"]
    print(f"scoring {len(e16_identities)} E16 checkpoints", flush=True)
    e16_models = {i["model_key"]: _load_model(torch, i["checkpoint"])
                  for i in e16_identities}
    e16_normal, e16_scenario = score_identities(
        torch, e16_identities, e16_models, base_manifest, bases, latents,
        mean, std, ledger)

    normal = pd.concat([shared_normal, e16_normal], ignore_index=True)
    scenario = pd.concat([shared_scenario, e16_scenario], ignore_index=True)
    by_cell = build_by_cell(scenario)

    expected_scenario = (
        len(identities) * len(BLOCKS) * len(e14.ATTACK_SPECS)
        * len(ID_STRATA) * len(FACTOR_CELLS))
    if len(scenario) != expected_scenario:
        raise _incomplete(
            f"{len(scenario)} scenario rows, expected {expected_scenario}")

    run_record = {
        "schema_version": RUN_SCHEMA,
        "stage": "e16_evaluation",
        "started_utc": prepare["created_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "environment": {
            "schema_version": ENVIRONMENT_SCHEMA,
            "device": SCORING_DEVICE,
            "batch_size": SCORING_BATCH_SIZE,
            "torch_threads": TORCH_THREADS,
            "torch_interop_threads": TORCH_INTEROP_THREADS,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "standardizer": standardizer,
        "runtime_continuity_gate": continuity_gate,
        "e16_inference_started_after_runtime_continuity": True,
        "exact_and_binary_endpoints_separate": True,
        "normal_control_reported_before_attack_results": True,
        "identities": len(identities),
        "completeness": {
            "scenario_rows": len(scenario),
            "expected_scenario_rows": expected_scenario,
            "normal_rows": len(normal),
            "by_cell_rows": len(by_cell),
        },
        "inference_ledger": ledger.as_dict(),
        "status": "T-PASS",
        "technical_status": "T-PASS",
    }

    staged = [
        (_stage_bytes(paths["continuity"],
                      continuity.to_csv(index=False).encode()),
         paths["continuity"]),
        (_stage_bytes(paths["normal"], normal.to_csv(index=False).encode()),
         paths["normal"]),
        (_stage_bytes(paths["by_scenario"],
                      scenario.to_csv(index=False).encode()),
         paths["by_scenario"]),
        (_stage_bytes(paths["by_cell"], by_cell.to_csv(index=False).encode()),
         paths["by_cell"]),
        (_stage_json(paths["run"], run_record), paths["run"]),
    ]
    publish_and_cleanup(staged)
    return run_record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true",
                      help="read-only binding stage; performs no inference")
    mode.add_argument("--execute", action="store_true",
                      help="run gate G12 then score the E16 grid")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute:
        record = execute()
    else:
        record = run_prepare()
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
