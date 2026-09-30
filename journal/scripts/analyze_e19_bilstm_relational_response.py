#!/usr/bin/env python3
"""Registered analysis of the E19 BiLSTM relational crossing.

Registered by ``journal/experiments/e19_bilstm_relational_crossing/PREREG.md``
sections 7--8.  Everything inferential is inherited from E16/E17 verbatim --
endpoint, aggregation order, inferential unit, margin, decision order,
bootstrap count and seed -- and the corresponding functions are **imported**
from ``analyze_e17_rf_relational_response`` rather than re-implemented, so the
registrations cannot drift apart.  E19 changes only the detector family, which
is upstream of this module.

The E17 module names its shared real-only arm ``real_rf`` through a
module-level constant.  E19's shared references are BiLSTM fits labeled
``real_bilstm`` in every published E19 table; this module points the imported
machinery at that label by assigning ``e17ana.REAL_ARM`` before any call.  The
assignment changes no inferential constant and is recorded in the analysis
output.

This module computes no outcome on import and writes nothing until
``--execute``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_e17_rf_relational_response as e17ana  # noqa: E402
import evaluate_e19_bilstm_relational_response as e19eval  # noqa: E402
import run_e19_bilstm_paired_training as e19train  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _stage_bytes,
    _stage_json,
)

REPO = e19train.REPO
EXPERIMENT_DIR = e19train.EXPERIMENT_DIR
TABLE_DIR = e19train.TABLE_DIR
OUTPUT_VERSION = e19train.OUTPUT_VERSION
ANALYSIS_SCHEMA = "e19.registered_analysis.v1"

publish_and_cleanup = e19train.publish_and_cleanup
_sha256_file = e19train._sha256_file

#: Point the imported E17 machinery at the E19 shared-reference arm label.
e17ana.REAL_ARM = e19eval.REAL_ARM_LABEL

DELTA = e17ana.DELTA
SECONDARY_EXACT_FAMILY = e17ana.SECONDARY_EXACT_FAMILY
EFFECTS = e17ana.EFFECTS
PRIMARY_ENDPOINT = e17ana.PRIMARY_ENDPOINT
COMPANION_ENDPOINT = e17ana.COMPANION_ENDPOINT


class E19AnalysisError(RuntimeError):
    """Raised on any registered analysis gate violation."""


def _stop(message: str) -> E19AnalysisError:
    return E19AnalysisError(f"T-STOP-E19-ANALYSIS: {message}")


def utc_now() -> str:
    return e19train.utc_now()


def output_paths() -> dict[str, Path]:
    return {
        "analysis": EXPERIMENT_DIR / f"analysis_{OUTPUT_VERSION}.json",
        "decomposition": TABLE_DIR / f"e19_decomposition_{OUTPUT_VERSION}.csv",
        "decision": TABLE_DIR / f"e19_registered_decision_{OUTPUT_VERSION}.json",
        "margin": TABLE_DIR / f"e19_margin_sensitivity_{OUTPUT_VERSION}.csv",
        "cells": TABLE_DIR / f"e19_cell_comparison_{OUTPUT_VERSION}.csv",
        "secondary_exact": TABLE_DIR
        / f"e19_secondary_exact_{OUTPUT_VERSION}.csv",
        "secondary_binary": TABLE_DIR
        / f"e19_secondary_binary_{OUTPUT_VERSION}.csv",
        "heterogeneity": TABLE_DIR
        / f"e19_attack_heterogeneity_{OUTPUT_VERSION}.csv",
        "dispersion": TABLE_DIR / f"e19_crossed_dispersion_{OUTPUT_VERSION}.csv",
        "bridge": TABLE_DIR / f"e19_bridge_{OUTPUT_VERSION}.csv",
    }


def analyze(scenario: pd.DataFrame) -> dict[str, Any]:
    """The E17 registered analysis sequence over the E19 scored table."""
    deltas = e17ana.build_augmentation_deltas(scenario)
    per_pipeline, per_realization = e17ana.build_effects_by_realization(deltas)
    decomposition = e17ana.build_decomposition(per_realization, effect="P")
    closure = e17ana.assert_decomposition_closure(decomposition)
    integrity = e17ana.verify_output_integrity(scenario, decomposition)

    primary_rows = decomposition[decomposition["role"] == "primary"]
    theta = primary_rows["theta_rule_minus_placebo"].to_numpy(dtype=float)
    if len(theta) != len(e19train.CONSTRUCTION_SEEDS):
        raise _stop(
            f"primary has {len(theta)} realizations, registered "
            f"{len(e19train.CONSTRUCTION_SEEDS)}")
    decision = e17ana.primary_decision(
        theta, delta=DELTA, margin_label="registered")
    share = e17ana.relational_share(primary_rows)

    p1_table, p1_summary = e17ana.build_cell_comparison(scenario, p_level=1)
    p0_table, p0_summary = e17ana.build_cell_comparison(scenario, p_level=0)
    saturation = e17ana.determine_p0_saturation(p0_summary, decomposition)
    margin_table, margin_record = e17ana.build_margin_sensitivity(
        theta, p1_summary, decision)

    exact_family = e17ana.build_secondary_family(
        per_realization, effects=SECONDARY_EXACT_FAMILY,
        label="exact_rule_minus_placebo_secondary", endpoint=PRIMARY_ENDPOINT)
    binary_deltas = e17ana.build_augmentation_deltas(
        scenario, endpoint=COMPANION_ENDPOINT)
    _, binary_per_realization = e17ana.build_effects_by_realization(
        binary_deltas)
    binary_family = e17ana.build_secondary_family(
        binary_per_realization, effects=EFFECTS,
        label="binary_rule_minus_placebo_companion",
        endpoint=COMPANION_ENDPOINT)
    heterogeneity = e17ana.build_attack_heterogeneity(scenario)
    dispersion = e17ana.build_crossed_dispersion(per_pipeline)

    bridge = decomposition[
        decomposition["role"] == "bridge_sensitivity"].copy()
    bridge["status"] = (
        "bridge sensitivity attached to the E15 lineage; never pooled with "
        "the primary twenty, never changes n or df")

    return {
        "per_pipeline": per_pipeline,
        "per_realization": per_realization,
        "decomposition": decomposition,
        "closure_gate": closure,
        "integrity_gate": integrity,
        "decision": decision,
        "relational_share": share,
        "p1_table": p1_table,
        "p1_summary": p1_summary,
        "p0_table": p0_table,
        "p0_summary": p0_summary,
        "p0_saturation": saturation,
        "margin_table": margin_table,
        "margin_record": margin_record,
        "secondary_exact": exact_family,
        "secondary_binary": binary_family,
        "heterogeneity": heterogeneity,
        "dispersion": dispersion,
        "bridge": bridge,
        "marginal_component": e17ana.build_arm_summary(
            per_realization, arm=e17ana.PLACEBO_ARM),
        "total_response": e17ana.build_arm_summary(
            per_realization, arm=e17ana.RULE_ARM),
    }


def _load_scenario() -> pd.DataFrame:
    path = TABLE_DIR / f"e19_by_scenario_{OUTPUT_VERSION}.csv"
    if not path.exists():
        raise _stop(f"missing scored table: {path}")
    return pd.read_csv(path)


def execute() -> dict[str, Any]:
    started = time.monotonic()
    paths = output_paths()
    present = [str(p.relative_to(REPO)) for p in paths.values() if p.exists()]
    if present:
        raise _stop(f"refusing to overwrite existing outputs: {present}")

    registration = e19train.assert_registration_committed()
    scenario = _load_scenario()
    result = analyze(scenario)

    decision = result["decision"]
    margin = result["margin_record"]
    record: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "stage": "e19_registered_analysis",
        "created_utc": utc_now(),
        "registration": registration,
        "environment": _environment_record(REPO),
        "real_arm_label": e19eval.REAL_ARM_LABEL,
        "inherited_from": "analyze_e17_rf_relational_response",
        "inferential_unit": {
            "unit": "generator realization", "n": decision["n"],
            "degrees_of_freedom": decision["degrees_of_freedom"],
            "never_units": ["block", "cell", "attack", "pipeline seed",
                            "window", "fit"],
        },
        "matching_contract": (
            "the E16 matched-budget protocol verbatim: shared initialization "
            "seed, shared synthetic row indices, shared minibatch RNG "
            "convention, and the 6,156-update budget, with only the detector "
            "architecture changed (PREREG 4.2)"),
        "closure_gate": result["closure_gate"],
        "output_integrity_gate": result["integrity_gate"],
        "primary": decision,
        "relational_share": result["relational_share"],
        "p1_cell_summary": result["p1_summary"],
        "p0_cell_summary": result["p0_summary"],
        "p0_saturation": result["p0_saturation"],
        "marginal_exposure_component": result["marginal_component"],
        "total_augmentation_response": result["total_response"],
        "verdict": decision["verdict"],
        "verdict_strict_margin": margin["verdict_strict_margin"],
        "margin_disagreement": margin["margin_disagreement"],
        "g_placebo": margin["g_placebo"],
        "delta": margin["delta"],
        "delta_strict": margin["delta_strict"],
        "delta_strict_over_delta": margin["delta_strict_over_delta"],
        "strict_margin_decision": margin["strict_decision"],
        "registered_margin_sentence": margin["registered_sentence"],
        "scientific_verdict": decision["verdict"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": "analysis_complete",
    }
    record["dual_margin_gate"] = e17ana.verify_dual_margin_reporting(record)

    cells = pd.concat(
        [result["p1_table"].assign(p_level=1),
         result["p0_table"].assign(p_level=0)],
        ignore_index=True, sort=False)
    staged = [
        (_stage_bytes(paths["decomposition"],
                      result["decomposition"].to_csv(index=False).encode()),
         paths["decomposition"]),
        (_stage_json(paths["decision"], result["decision"]),
         paths["decision"]),
        (_stage_bytes(paths["margin"],
                      result["margin_table"].to_csv(index=False).encode()),
         paths["margin"]),
        (_stage_bytes(paths["cells"], cells.to_csv(index=False).encode()),
         paths["cells"]),
        (_stage_bytes(paths["secondary_exact"],
                      result["secondary_exact"].to_csv(index=False).encode()),
         paths["secondary_exact"]),
        (_stage_bytes(paths["secondary_binary"],
                      result["secondary_binary"].to_csv(index=False).encode()),
         paths["secondary_binary"]),
        (_stage_bytes(paths["heterogeneity"],
                      result["heterogeneity"].to_csv(index=False).encode()),
         paths["heterogeneity"]),
        (_stage_bytes(paths["dispersion"],
                      result["dispersion"].to_csv(index=False).encode()),
         paths["dispersion"]),
        (_stage_bytes(paths["bridge"],
                      result["bridge"].to_csv(index=False).encode()),
         paths["bridge"]),
        (_stage_json(paths["analysis"], record), paths["analysis"]),
    ]
    publish_and_cleanup(staged)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        scenario = _load_scenario()
        print(f"scenario rows: {len(scenario)} (read-only; no output written)")
        return 0
    record = execute()
    print(json.dumps(
        {k: record[k] for k in ("primary", "relational_share", "verdict",
                                "verdict_strict_margin", "margin_disagreement",
                                "status")},
        indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
