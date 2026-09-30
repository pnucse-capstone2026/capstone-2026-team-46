#!/usr/bin/env python3
"""Score the E17 Random-Forest grid on the frozen E14 L4 factorial.

Registered by ``journal/experiments/e17_rf_relational_replication/PREREG.md``
sections 4.4, 7, and 15.3.

Stages
------
``--prepare-only``  read-only: bind the frozen E14 evaluation material and the
                    245 E17 checkpoints.  Performs no inference.
``--execute``       stage S6.  Gate G12 first: the five shared Random-Forest
                    real-only references are scored **before** any Rule or
                    placebo checkpoint, their per-cell integer counts are
                    frozen, and after the full grid has been scored they are
                    re-scored and required to be byte-identical.  Any drift
                    halts S6.

Evaluation material is the frozen E14 construction, reused unchanged: three
frame-disjoint blocks, 6,000 held-out normal bases, four ID strata, and the
eight ``(P, S, D)`` factorial cells.  E17 introduces no new evaluation
construction; the only new axis is the detector family.

Random Forests consume **raw** windows through the 55-dimensional
``rf_features`` map.  No standardizer is fitted or applied anywhere in this
module, which is the registered contrast with the CNN evaluator.  Scoring is
CPU-only and imports no torch.
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
import run_e17_rf_paired_training as e17train  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _environment_record,
    _stage_bytes,
    _stage_json,
)

REPO = e17train.REPO
EXPERIMENT_DIR = e17train.EXPERIMENT_DIR
E14_DIR = REPO / "journal" / "experiments" / "e14_l4_counterfactual_factorial"
TABLE_DIR = e17train.TABLE_DIR

BLOCKS = ("block_01", "block_02", "block_03")
ID_STRATA = ("canonical", "shifted")
FACTOR_CELLS = tuple((p, s, d) for p in (0, 1) for s in (0, 1) for d in (0, 1))
PIPELINE_SEEDS = e17train.PIPELINE_SEEDS
BASES_PER_BLOCK = 2_000
TOTAL_BASES = len(BLOCKS) * BASES_PER_BLOCK

OUTPUT_VERSION = e17train.OUTPUT_VERSION
PREPARE_SCHEMA = "e17.evaluation_prepare.v1"
RUN_SCHEMA = "e17.evaluation_run.v1"
ENVIRONMENT_SCHEMA = "e17.evaluation_environment.v1"

SCORING_DEVICE = "cpu"
REAL_ARM = "real_rf"

#: Same shape as the E16 continuity grid, so the two are directly comparable.
CONTINUITY_SCENARIO_ROWS = 480
CONTINUITY_NORMAL_ROWS = 15

publish_and_cleanup = e17train.publish_and_cleanup
_sha256_file = e17train._sha256_file
rf_features = e17train.rf_features


class E17EvaluationError(RuntimeError):
    """Raised on any registered E17 evaluation gate violation."""


def _stop(message: str) -> E17EvaluationError:
    return E17EvaluationError(f"T-STOP-E17-EVAL: {message}")


def _incomplete(message: str) -> E17EvaluationError:
    return E17EvaluationError(f"T-INCOMPLETE-E17-EVAL: {message}")


def utc_now() -> str:
    return e17train.utc_now()


def output_paths() -> dict[str, Path]:
    return {
        "prepare": EXPERIMENT_DIR / f"prepare_{OUTPUT_VERSION}.json",
        "run": EXPERIMENT_DIR / f"run_{OUTPUT_VERSION}.json",
        "by_scenario": TABLE_DIR / f"e17_by_scenario_{OUTPUT_VERSION}.csv",
        "by_cell": TABLE_DIR / f"e17_by_cell_{OUTPUT_VERSION}.csv",
        "normal": TABLE_DIR / f"e17_normal_by_arm_{OUTPUT_VERSION}.csv",
        "continuity": TABLE_DIR
        / f"e17_real_only_reference_continuity_{OUTPUT_VERSION}.csv",
    }


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def build_identities() -> list[dict[str, Any]]:
    """Every checkpoint scored by this stage, references first."""
    identities: list[dict[str, Any]] = []
    for job in e17train.build_training_grid():
        arm = REAL_ARM if job.arm == "real_only" else job.arm
        identities.append({
            "model_key": (arm, job.construction_seed, job.pipeline_seed),
            "arm": arm,
            "construction_seed": job.construction_seed,
            "pipeline_seed": job.pipeline_seed,
            "role": job.role,
            "checkpoint": job.checkpoint_path,
        })
    keys = [identity["model_key"] for identity in identities]
    if len(set(keys)) != len(keys):
        raise _stop("duplicate model key among evaluation identities")
    if len(identities) != e17train.EXPECTED_FIT_COUNT:
        raise _stop(
            f"{len(identities)} identities, registered "
            f"{e17train.EXPECTED_FIT_COUNT}")
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
# Evaluation feature panel
# ---------------------------------------------------------------------------


def build_evaluation_panel(
        base_manifest: pd.DataFrame,
        bases: np.ndarray,
        latents: Sequence[Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Featurize the untouched bases and every registered L4 transformation.

    The transformations are the frozen E14 construction; only the *reading* of
    them changes, from standardized CNN tensors to raw ``rf_features`` rows.
    Every model sees one stacked matrix and is predicted exactly once, which is
    both faster and removes any batching axis from the comparison.
    """
    block_orders = {
        block: base_manifest.index[
            base_manifest["block_id"].astype(str).eq(block)].to_numpy(np.int64)
        for block in BLOCKS
    }
    if any(len(order) != BASES_PER_BLOCK for order in block_orders.values()):
        raise _incomplete("L4 block identity is incomplete")
    latent_lookup = {
        (latent.block_id, latent.block_position,
         latent.test_window_index, latent.attack_label): latent
        for latent in latents
    }

    parts: list[np.ndarray] = [rf_features(np.asarray(bases, np.float32))]
    segments: list[dict[str, Any]] = [
        {"kind": "normal", "start": 0, "stop": len(bases)}]
    cursor = len(bases)
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
                    parts.append(rf_features(transformed))
                    segments.append({
                        "kind": "scenario", "block_id": block,
                        "attack": attack, "attack_label": attack_label,
                        "id_stratum": id_stratum, "P": p, "S": s, "D": d,
                        "start": cursor, "stop": cursor + len(transformed),
                    })
                    cursor += len(transformed)
        print(f"  featurized {block}", flush=True)
    panel = np.concatenate(parts, axis=0)
    expected = TOTAL_BASES + (
        len(BLOCKS) * len(e14.ATTACK_SPECS) * len(ID_STRATA)
        * len(FACTOR_CELLS) * BASES_PER_BLOCK)
    if len(panel) != expected or panel.shape[1] != e17train.FEATURE_DIMENSION:
        raise _incomplete(
            f"evaluation panel is {panel.shape}, expected "
            f"({expected}, {e17train.FEATURE_DIMENSION})")
    return panel, block_orders, segments


class InferenceLedger:
    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def record(self, rows: int) -> None:
        self.calls += 1
        self.rows += rows

    def as_dict(self) -> dict[str, int]:
        return {"inference_calls": self.calls, "prediction_rows": self.rows}


def score_identities(
        identities: Sequence[Mapping[str, Any]],
        panel: np.ndarray,
        block_orders: Mapping[str, np.ndarray],
        segments: Sequence[Mapping[str, Any]],
        ledger: InferenceLedger,
        *,
        label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each checkpoint once over the whole panel and split the counts."""
    import joblib

    normal_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for position, identity in enumerate(identities, start=1):
        forest = joblib.load(identity["checkpoint"])
        if int(forest.n_features_in_) != e17train.FEATURE_DIMENSION:
            raise _stop(
                f"{identity['model_key']}: checkpoint expects "
                f"{forest.n_features_in_} features")
        predictions = np.asarray(forest.predict(panel), dtype=np.int64)
        del forest
        ledger.record(len(panel))
        common = {
            "arm": identity["arm"],
            "construction_seed": identity["construction_seed"],
            "pipeline_seed": identity["pipeline_seed"],
            "role": identity["role"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
        }
        for segment in segments:
            window = predictions[segment["start"]:segment["stop"]]
            if segment["kind"] == "normal":
                for block, order in block_orders.items():
                    correct = int(np.count_nonzero(window[order] == 0))
                    normal_rows.append({
                        **common, "block_id": block, "n": len(order),
                        "normal_correct": correct,
                        "normal_recall": correct / len(order),
                        "fpr": 1.0 - correct / len(order),
                    })
                continue
            attack_label = segment["attack_label"]
            exact_correct = int(np.count_nonzero(window == attack_label))
            binary_correct = int(np.count_nonzero(window != 0))
            scenario_rows.append({
                **common,
                "block_id": segment["block_id"],
                "attack": segment["attack"],
                "attack_label": attack_label,
                "id_stratum": segment["id_stratum"],
                "P": segment["P"], "S": segment["S"], "D": segment["D"],
                "cell": f"{segment['P']}{segment['S']}{segment['D']}",
                "n": len(window),
                "exact_correct": exact_correct,
                "binary_correct": binary_correct,
                "exact_recall": exact_correct / len(window),
                "binary_recall": binary_correct / len(window),
            })
        if position % 20 == 0 or position == len(identities):
            print(f"  {label}: scored {position}/{len(identities)}", flush=True)
    return pd.DataFrame(normal_rows), pd.DataFrame(scenario_rows)


# ---------------------------------------------------------------------------
# Gate G12 — shared real-only reference continuity
# ---------------------------------------------------------------------------


def build_reference_continuity(
        first_normal: pd.DataFrame, first_scenario: pd.DataFrame,
        second_normal: pd.DataFrame, second_scenario: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the pre-grid and post-grid reference counts, integer for integer."""
    scenario_keys = ["pipeline_seed", "block_id", "attack", "id_stratum",
                     "P", "S", "D"]
    merged = first_scenario.merge(
        second_scenario, on=scenario_keys, suffixes=("_first", "_second"),
        how="outer", indicator=True, validate="1:1")
    if not (merged["_merge"] == "both").all():
        raise _incomplete(
            "reference continuity grid is incomplete: "
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
                "pre_grid_value": int(record[f"{field}_first"]),
                "post_grid_value": int(record[f"{field}_second"]),
                "difference": int(record[f"{field}_second"])
                - int(record[f"{field}_first"]),
                "denominator": int(record["n_first"]),
            })
    normal_merged = first_normal.merge(
        second_normal, on=["pipeline_seed", "block_id"],
        suffixes=("_first", "_second"), how="outer", indicator=True,
        validate="1:1")
    if not (normal_merged["_merge"] == "both").all():
        raise _incomplete("reference normal continuity grid is incomplete")
    for record in normal_merged.to_dict(orient="records"):
        rows.append({
            "row_type": "normal",
            "pipeline_seed": record["pipeline_seed"],
            "block_id": record["block_id"],
            "attack": "", "id_stratum": "", "cell": "",
            "metric": "normal_correct",
            "pre_grid_value": int(record["normal_correct_first"]),
            "post_grid_value": int(record["normal_correct_second"]),
            "difference": int(record["normal_correct_second"])
            - int(record["normal_correct_first"]),
            "denominator": int(record["n_first"]),
        })
    return pd.DataFrame(rows)


def require_reference_continuity(continuity: pd.DataFrame) -> dict[str, Any]:
    """Gate G12 — blocking; every integer count difference must be zero."""
    scenario_rows = int((continuity["row_type"] == "scenario").sum()) // 2
    normal_rows = int((continuity["row_type"] == "normal").sum())
    if (scenario_rows != CONTINUITY_SCENARIO_ROWS
            or normal_rows != CONTINUITY_NORMAL_ROWS):
        raise _incomplete(
            f"continuity grid has {scenario_rows} scenario and {normal_rows} "
            f"normal rows; registered {CONTINUITY_SCENARIO_ROWS}/"
            f"{CONTINUITY_NORMAL_ROWS}")
    nonzero = continuity[continuity["difference"] != 0]
    if len(nonzero):
        raise _stop(
            f"shared real-only reference continuity FAILED: {len(nonzero)} of "
            f"{len(continuity)} integer counts drifted between the pre-grid "
            "and post-grid scoring passes")
    return {
        "gate": "G12_real_only_reference_continuity",
        "passed": True,
        "status": "T-PASS",
        "rows": scenario_rows + normal_rows,
        "scenario_rows": scenario_rows,
        "normal_rows": normal_rows,
        "integer_comparisons": len(continuity),
        "all_integer_count_differences_zero": True,
        "references_scored_before_any_augmented_arm": True,
        "device": SCORING_DEVICE,
    }


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def build_by_cell(scenario: pd.DataFrame) -> pd.DataFrame:
    """Gear/RPM equal-macro per (arm, realization, pipeline, stratum, cell)."""
    grouped = (
        scenario.groupby(
            ["arm", "construction_seed", "pipeline_seed", "role",
             "id_stratum", "P", "S", "D", "cell", "attack"], as_index=False)
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
    training_run = EXPERIMENT_DIR / f"training_run_{OUTPUT_VERSION}.json"
    if not training_run.exists():
        raise _stop(f"stage S4 has not completed: missing {training_run}")
    record: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "stage": "e17_evaluation_prepare",
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
            "detector_family": "random_forest",
            "standardizer": None,
            "feature_dimension": e17train.FEATURE_DIMENSION,
            "windows": "raw, un-standardized",
        },
        "registration": e17train.assert_registration_committed(),
        "environment": _environment_record(REPO),
        "checkpoints": assert_checkpoints_present(identities),
        "e14_prepare_sha256": _sha256_file(E14_DIR / "prepare_v1.json"),
        "training_run_sha256": _sha256_file(training_run),
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

    identities = build_identities()
    for identity in identities:
        identity["checkpoint_sha256"] = _sha256_file(identity["checkpoint"])

    base_manifest, bases, latents = e14.load_prepared_inputs(_load_e14_prepare())
    print("building the frozen E14 evaluation panel", flush=True)
    panel, block_orders, segments = build_evaluation_panel(
        base_manifest, bases, latents)
    del bases, latents
    ledger = InferenceLedger()

    # --- gate G12 pass one: the shared references, before any augmented arm --
    shared = [i for i in identities if i["role"] == "shared_reference"]
    augmented = [i for i in identities if i["role"] != "shared_reference"]
    print(f"gate G12 pass 1: scoring {len(shared)} shared real-only "
          f"references on {SCORING_DEVICE}", flush=True)
    first_normal, first_scenario = score_identities(
        shared, panel, block_orders, segments, ledger, label="reference-pre")

    print(f"scoring {len(augmented)} E17 Rule/placebo checkpoints", flush=True)
    arm_normal, arm_scenario = score_identities(
        augmented, panel, block_orders, segments, ledger, label="grid")

    print("gate G12 pass 2: re-scoring the shared real-only references",
          flush=True)
    second_normal, second_scenario = score_identities(
        shared, panel, block_orders, segments, ledger, label="reference-post")
    continuity = build_reference_continuity(
        first_normal, first_scenario, second_normal, second_scenario)
    continuity_gate = require_reference_continuity(continuity)
    print(f"gate G12 PASS: {continuity_gate['integer_comparisons']} integer "
          "counts identical across the scoring session", flush=True)

    normal = pd.concat([first_normal, arm_normal], ignore_index=True)
    scenario = pd.concat([first_scenario, arm_scenario], ignore_index=True)
    by_cell = build_by_cell(scenario)

    expected_scenario = (
        len(identities) * len(BLOCKS) * len(e14.ATTACK_SPECS)
        * len(ID_STRATA) * len(FACTOR_CELLS))
    if len(scenario) != expected_scenario:
        raise _incomplete(
            f"{len(scenario)} scenario rows, expected {expected_scenario}")

    run_record = {
        "schema_version": RUN_SCHEMA,
        "stage": "e17_evaluation",
        "started_utc": prepare["created_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "environment": {
            "schema_version": ENVIRONMENT_SCHEMA,
            "device": SCORING_DEVICE,
            "detector_family": "random_forest",
            "standardizer_applied": False,
            "feature_dimension": e17train.FEATURE_DIMENSION,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "reference_continuity_gate": continuity_gate,
        "augmented_scoring_started_after_reference_pass": True,
        "exact_and_binary_endpoints_separate": True,
        "normal_control_reported_before_attack_results": True,
        "identities": len(identities),
        "completeness": {
            "scenario_rows": len(scenario),
            "expected_scenario_rows": expected_scenario,
            "normal_rows": len(normal),
            "by_cell_rows": len(by_cell),
            "evaluation_panel_rows": int(len(panel)),
        },
        "inference_ledger": ledger.as_dict(),
        "status": "T-PASS",
        "technical_status": "T-PASS",
    }

    publish_and_cleanup([
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
    ])
    return run_record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true",
                      help="read-only binding stage; performs no inference")
    mode.add_argument("--execute", action="store_true",
                      help="run gate G12 then score the E17 grid")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    record = execute() if args.execute else run_prepare()
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
