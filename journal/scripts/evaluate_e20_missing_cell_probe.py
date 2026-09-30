#!/usr/bin/env python3
"""E20 missing-cell probe: evaluation-only, over the frozen E16 checkpoints.

Specified by ``journal/experiments/e20_missing_cell_probe/DESIGN.md``
(prospective exploratory design, commit ``5e2b4ae``).  No model is trained or
modified; the frozen E14 evaluation material and the frozen E16 CNN
checkpoints are consumed by hash.

Variant construction (DESIGN section 2)
---------------------------------------
For attack in {Gear, RPM}, sparsity ``k`` in {2, 4, 8, 16, 32}, and relation
state in {intact, broken}: inject the first ``k`` entries of the registered
``S = 0`` mask at the ``P = 0`` (trained, in-grammar) byte destinations with
the registered ``D = 0`` Rule-side role values, canonical CAN ID.  ``broken``
permutes the ``data1`` role entries across the ``k`` injected positions with
the same constrained rotations the E16 training placebo uses, so the
within-window ``data1`` multiset is preserved and only the rowwise coupling is
destroyed.  ``k = 32`` intact is bit-identical to frozen factorial cell
``000`` canonical, which supplies the blocking continuity gate.

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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_l4_counterfactual_factorial as e14  # noqa: E402
import evaluate_e16_relational_response as e16eval  # noqa: E402
import run_e16_paired_training as e16train  # noqa: E402
import generate_e16_rule_placebo_pairs as e16gen  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _source_record,
    _stage_bytes,
    _stage_json,
    atomic_publish_bundle,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "journal" / "experiments" / "e20_missing_cell_probe"
TABLE_DIR = REPO / "journal" / "results" / "tables"

OUTPUT_VERSION = "v1"
RUN_SCHEMA = "e20.run.v1"

DESIGN_PATH = "journal/experiments/e20_missing_cell_probe/DESIGN.md"
DESIGN_COMMIT = "5e2b4ae"

SPARSITY_LEVELS = (2, 4, 8, 16, 32)
RELATION_STATES = ("intact", "broken")
ID_STRATUM = "canonical"
BLOCKS = e16eval.BLOCKS
PIPELINE_SEEDS = e16train.PIPELINE_SEEDS

SCORING_DEVICE = "cpu"
SCORING_BATCH_SIZE = e14.SCORING_BATCH_SIZE
TORCH_THREADS = e14.TORCH_THREADS

E16_SCENARIO_TABLE = TABLE_DIR / "e16_by_scenario_v2.csv"


class E20Error(RuntimeError):
    """Raised on any design-gate violation."""


def _stop(message: str) -> E20Error:
    return E20Error(f"T-STOP-E20: {message}")


def utc_now() -> str:
    return e16gen.utc_now()


_sha256_file = e16train._sha256_file
publish_and_cleanup = e16eval.publish_and_cleanup


def output_paths() -> dict[str, Path]:
    return {
        "run": EXPERIMENT_DIR / f"run_{OUTPUT_VERSION}.json",
        "by_scenario": TABLE_DIR
        / f"e20_missing_cell_by_scenario_{OUTPUT_VERSION}.csv",
        "summary": TABLE_DIR
        / f"e20_missing_cell_summary_{OUTPUT_VERSION}.csv",
        "manipulation": TABLE_DIR
        / f"e20_missing_cell_manipulation_{OUTPUT_VERSION}.csv",
    }


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------


def _perm_rng(block_id: str, test_window_index: int, attack_label: int,
              k: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"e20-evalperm|{block_id}|{test_window_index}|{attack_label}|{k}"
        .encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[-4:], "big"))


def apply_missing_cell_transform(
        base: np.ndarray,
        latent: Any,
        *,
        attack_label: int,
        k: int,
        relation: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """In-grammar sparse injection; returns the window and manipulation stats."""
    spec = e14.ATTACK_SPECS[attack_label]
    result = np.asarray(base).copy()
    positions = latent.s0_positions[:k]
    columns = spec["p0_columns"]
    roles = [np.asarray(role[:k]).copy() for role in latent.d0_roles]
    stats_row = {"frames": k, "off_relation": 0, "structural_minimum": 0}
    if relation == "broken":
        prng = _perm_rng(latent.block_id, latent.test_window_index,
                         attack_label, k)
        d0 = roles[0]
        d1 = roles[1]
        if attack_label == 3:
            new_d1, offbin, n = e16gen.permute_gear_data1(prng, d0, d1)
            residual = int((np.abs(255.0 - d0 - new_d1)
                            <= e16gen.GEAR_RESIDUAL_TOLERANCE).sum())
            stats_row["off_relation"] = n - residual
        else:
            new_d1, residual, minimum, n = e16gen.permute_rpm_data1(
                prng, d0, d1)
            stats_row["off_relation"] = n - residual
            stats_row["structural_minimum"] = minimum
        roles[1] = np.asarray(new_d1, dtype=np.float32)
    result[positions, 0] = spec["canonical_id"]
    result[positions, 1] = 8
    for column, role in zip(columns, roles, strict=True):
        result[positions, column] = role
    result[positions, 10] = np.maximum(result[positions, 10], 1e-5)
    return np.ascontiguousarray(result, dtype=np.float32), stats_row


# ---------------------------------------------------------------------------
# Identities — frozen E16 primary checkpoints plus the shared real references
# ---------------------------------------------------------------------------


def build_identities() -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for pipeline_seed in PIPELINE_SEEDS:
        identities.append({
            "model_key": ("real_ms", 0, pipeline_seed),
            "arm": "real_ms",
            "construction_seed": 0,
            "pipeline_seed": pipeline_seed,
            "role": "shared_reference",
            "checkpoint": e16train.shared_real_checkpoint(pipeline_seed),
        })
    for job in e16train.build_training_grid():
        if job.role != "primary":
            continue
        identities.append({
            "model_key": (job.arm, job.construction_seed, job.pipeline_seed),
            "arm": job.arm,
            "construction_seed": job.construction_seed,
            "pipeline_seed": job.pipeline_seed,
            "role": "primary",
            "checkpoint": job.checkpoint_path,
        })
    if len(identities) != 205:
        raise _stop(f"{len(identities)} identities, design says 205")
    missing = [str(i["checkpoint"]) for i in identities
               if not i["checkpoint"].exists()]
    if missing:
        raise _stop(f"missing checkpoints: {missing[:4]}")
    return identities


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _configure_torch() -> Any:
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def score_all(torch: Any, identities: Sequence[Mapping[str, Any]],
              models: Mapping[Any, Any], base_manifest: pd.DataFrame,
              bases: np.ndarray, latents: Sequence[Any],
              mean: np.ndarray, std: np.ndarray
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_orders = {
        block: base_manifest.index[
            base_manifest["block_id"].astype(str).eq(block)].to_numpy(np.int64)
        for block in BLOCKS
    }
    latent_lookup = {
        (latent.block_id, latent.block_position,
         latent.test_window_index, latent.attack_label): latent
        for latent in latents
    }
    scenario_rows: list[dict[str, Any]] = []
    manipulation_rows: list[dict[str, Any]] = []
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
            for k in SPARSITY_LEVELS:
                for relation in RELATION_STATES:
                    manipulation = {"frames": 0, "off_relation": 0,
                                    "structural_minimum": 0}
                    windows = np.empty(
                        (len(block_bases),) + e14.WINDOW_SHAPE,
                        dtype=np.float32)
                    for index, (base, latent) in enumerate(
                            zip(block_bases, block_latents, strict=True)):
                        window, stats_row = apply_missing_cell_transform(
                            base, latent, attack_label=attack_label,
                            k=k, relation=relation)
                        windows[index] = window
                        for key in manipulation:
                            manipulation[key] += stats_row[key]
                    if relation == "broken":
                        manipulation_rows.append({
                            "block_id": block, "attack": attack, "k": k,
                            **manipulation,
                            "off_relation_fraction":
                                manipulation["off_relation"]
                                / max(manipulation["frames"], 1),
                        })
                    standardized = e14._standardize(windows, mean, std)
                    for identity in identities:
                        logits = e14.predict_standardized_logits(
                            torch, models[identity["model_key"]],
                            standardized, batch_size=SCORING_BATCH_SIZE)
                        predictions = np.asarray(logits).argmax(axis=1)
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
                            "id_stratum": ID_STRATUM,
                            "k": k,
                            "relation": relation,
                            "n": len(windows),
                            "exact_correct": exact_correct,
                            "binary_correct": binary_correct,
                            "exact_recall": exact_correct / len(windows),
                            "binary_recall": binary_correct / len(windows),
                        })
            print(f"  scored {block}/{attack} "
                  f"({len(scenario_rows)} scenario rows)", flush=True)
    return pd.DataFrame(scenario_rows), pd.DataFrame(manipulation_rows)


# ---------------------------------------------------------------------------
# Continuity gate — k = 32 intact must equal frozen cell 000 canonical
# ---------------------------------------------------------------------------


def require_continuity(scenario: pd.DataFrame) -> dict[str, Any]:
    frozen = pd.read_csv(E16_SCENARIO_TABLE)
    frozen = frozen[(frozen["P"] == 0) & (frozen["S"] == 0)
                    & (frozen["D"] == 0)]
    frozen = frozen[(frozen["id_stratum"] == ID_STRATUM)
                    & (frozen["role"] != "bridge_sensitivity")]
    mine = scenario[(scenario["k"] == 32)
                    & (scenario["relation"] == "intact")]
    keys = ["arm", "construction_seed", "pipeline_seed", "block_id", "attack"]
    merged = mine.merge(
        frozen[keys + ["exact_correct", "binary_correct"]],
        on=keys, how="left", suffixes=("", "_frozen"), validate="1:1")
    if merged["exact_correct_frozen"].isna().any():
        raise _stop("continuity join incomplete against the frozen E16 record")
    bad = merged[
        (merged["exact_correct"] != merged["exact_correct_frozen"])
        | (merged["binary_correct"] != merged["binary_correct_frozen"])]
    if len(bad):
        raise _stop(
            f"continuity FAILED: {len(bad)} of {len(merged)} k=32-intact "
            "integer counts differ from the frozen cell 000 record")
    return {"gate": "k32_intact_equals_frozen_cell000", "passed": True,
            "rows_compared": int(len(merged)),
            "all_integer_count_differences_zero": True}


# ---------------------------------------------------------------------------
# Summary — fixed by DESIGN section 4
# ---------------------------------------------------------------------------


def _t_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return math_nan, math_nan
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)) * se
    return mean - half, mean + half


math_nan = float("nan")


def build_summary(scenario: pd.DataFrame) -> pd.DataFrame:
    """Realization-level curves, differentials, and theta_mc per k."""
    frame = scenario.copy()
    keys = ["arm", "construction_seed", "pipeline_seed", "role", "block_id",
            "k", "relation"]
    macro = frame.groupby(keys, as_index=False)[
        ["exact_recall", "binary_recall"]].mean()          # attack equal-macro
    keys = ["arm", "construction_seed", "pipeline_seed", "role", "k",
            "relation"]
    by_pipeline = macro.groupby(keys, as_index=False)[
        ["exact_recall", "binary_recall"]].mean()          # block mean
    keys = ["arm", "construction_seed", "role", "k", "relation"]
    by_realization = by_pipeline.groupby(keys, as_index=False)[
        ["exact_recall", "binary_recall"]].mean()          # pipeline mean

    rows: list[dict[str, Any]] = []
    for endpoint in ("exact_recall", "binary_recall"):
        wide = by_realization.pivot_table(
            index=["arm", "construction_seed", "role", "k"],
            columns="relation", values=endpoint).reset_index()
        wide["differential"] = wide["intact"] - wide["broken"]
        for (arm, k), group in wide.groupby(["arm", "k"]):
            rows.append({
                "endpoint": endpoint, "arm": arm, "k": int(k),
                "quantity": "recall_intact",
                "mean": float(group["intact"].mean()),
                "sd": float(group["intact"].std(ddof=1))
                if len(group) > 1 else math_nan,
                "n_realizations": int(len(group)),
            })
            rows.append({
                "endpoint": endpoint, "arm": arm, "k": int(k),
                "quantity": "recall_broken",
                "mean": float(group["broken"].mean()),
                "sd": float(group["broken"].std(ddof=1))
                if len(group) > 1 else math_nan,
                "n_realizations": int(len(group)),
            })
            values = group["differential"].to_numpy(dtype=float)
            low90, high90 = _t_interval(values, 0.90)
            low95, high95 = _t_interval(values, 0.95)
            rows.append({
                "endpoint": endpoint, "arm": arm, "k": int(k),
                "quantity": "differential_intact_minus_broken",
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1))
                if len(values) > 1 else math_nan,
                "ci90_low": low90, "ci90_high": high90,
                "ci95_low": low95, "ci95_high": high95,
                "positive_signs": int((values > 0).sum()),
                "negative_signs": int((values < 0).sum()),
                "n_realizations": int(len(values)),
            })
        # theta_mc(k): rule differential minus placebo differential per
        # realization (shared-reference arm has one pseudo-realization and is
        # reported only through its own differential above).
        primary = wide[wide["role"] == "primary"]
        rule = primary[primary["arm"] == "rule"].set_index(
            ["construction_seed", "k"])["differential"]
        placebo = primary[primary["arm"] == "placebo"].set_index(
            ["construction_seed", "k"])["differential"]
        for k in SPARSITY_LEVELS:
            theta = (rule.xs(k, level="k")
                     - placebo.xs(k, level="k")).to_numpy(dtype=float)
            low90, high90 = _t_interval(theta, 0.90)
            low95, high95 = _t_interval(theta, 0.95)
            rows.append({
                "endpoint": endpoint, "arm": "rule_minus_placebo",
                "k": int(k), "quantity": "theta_missing_cell",
                "mean": float(np.mean(theta)),
                "sd": float(np.std(theta, ddof=1)),
                "ci90_low": low90, "ci90_high": high90,
                "ci95_low": low95, "ci95_high": high95,
                "positive_signs": int((theta > 0).sum()),
                "negative_signs": int((theta < 0).sum()),
                "n_realizations": int(len(theta)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def execute() -> dict[str, Any]:
    started = time.monotonic()
    paths = output_paths()
    present = [str(p.relative_to(REPO)) for p in paths.values() if p.exists()]
    if present:
        raise _stop(f"refusing to overwrite existing outputs: {present}")
    if not E16_SCENARIO_TABLE.exists():
        raise _stop("missing frozen E16 scenario table for the continuity gate")

    # The frozen environment projection pins the torch thread counts, so the
    # environment and source records must be captured before
    # ``_configure_torch`` changes them (mirrors the E16 evaluator order).
    environment_record = _environment_record(REPO)
    source_record = _source_record(REPO)

    identities = build_identities()
    torch = _configure_torch()
    base_manifest, bases, latents = e14.load_prepared_inputs(
        json.loads((e14.EXP / "prepare_v1.json").read_text()))
    mean, std, standardizer = e14.fit_registered_standardizer()

    print(f"loading {len(identities)} frozen checkpoints on CPU", flush=True)
    models = {}
    for identity in identities:
        model = e14._make_cnn(torch)
        state = torch.load(identity["checkpoint"], map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models[identity["model_key"]] = model

    scenario, manipulation = score_all(
        torch, identities, models, base_manifest, bases, latents, mean, std)
    continuity = require_continuity(scenario)
    print(f"continuity gate PASS: {continuity['rows_compared']} integer "
          "comparisons identical to the frozen record", flush=True)
    summary = build_summary(scenario)

    run_record = {
        "schema_version": RUN_SCHEMA,
        "stage": "e20_missing_cell_probe",
        "design_path": DESIGN_PATH,
        "design_commit": DESIGN_COMMIT,
        "started_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "class": "prospective exploratory sensitivity; no margin, no verdict",
        "environment": environment_record,
        "source": source_record,
        "standardizer": standardizer,
        "identities": len(identities),
        "sparsity_levels": list(SPARSITY_LEVELS),
        "continuity_gate": continuity,
        "scenario_rows": int(len(scenario)),
        "e16_scenario_table_sha256": _sha256_file(E16_SCENARIO_TABLE),
        "status": "T-PASS",
    }
    publish_and_cleanup([
        (_stage_bytes(paths["by_scenario"],
                      scenario.to_csv(index=False).encode()),
         paths["by_scenario"]),
        (_stage_bytes(paths["manipulation"],
                      manipulation.to_csv(index=False).encode()),
         paths["manipulation"]),
        (_stage_bytes(paths["summary"], summary.to_csv(index=False).encode()),
         paths["summary"]),
        (_stage_json(paths["run"], run_record), paths["run"]),
    ])
    return run_record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        identities = build_identities()
        print(f"read-only preflight: {len(identities)} identities present; "
              "no inference performed")
        return 0
    record = execute()
    print(json.dumps({k: record[k] for k in (
        "continuity_gate", "scenario_rows", "elapsed_seconds", "status")},
        indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
