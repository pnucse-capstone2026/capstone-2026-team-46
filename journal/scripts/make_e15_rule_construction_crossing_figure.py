#!/usr/bin/env python3
"""Render the frozen E15 Rule-construction crossing figure.

The three panels keep the registered evidence layers separate:

* panel (a) shows the primary construction-first canonical exact effects.
  Four new construction seeds are the inferential units; the fresh 314159
  anchor is descriptive and excluded from the mean, interval, and tests;
* panel (b) shows the supporting Gear/RPM construction summaries for the two
  effects carrying the registered attack-heterogeneity modifier;
* panel (c) shows descriptive finite-grid dispersion across the crossed
  construction, pipeline, and interaction axes.  These shares are not
  random-effects variance components or population variance explained.

All effects are matched-budget Rule-minus-real augmentation-pipeline
contrasts under one fixed Rule construction law.  They do not identify
physical CAN mechanisms, realism, real-attack replacement, or external
transfer.

Inputs (read-only, frozen by SHA-256):
  journal/results/tables/e15_l4_factorial_hierarchical_summary_v2.csv
  journal/results/tables/e15_l4_factorial_crossed_dispersion_v2.csv
  journal/results/logs/e15_rule_construction_crossing_artifact_manifest_v2.json

Outputs (versioned, staged, and published without overwrite):
  journal/results/figures/e15_rule_construction_crossing_v2.pdf
  journal/results/figures/e15_rule_construction_crossing_v2.png
  journal/results/logs/e15_rule_construction_crossing_figure_provenance_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Keep PDF metadata stable across independent reruns.
os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
JOURNAL_ROOT = SCRIPT.parents[1]
REPO_ROOT = JOURNAL_ROOT.parent
RESULTS_ROOT = JOURNAL_ROOT / "results"

HIERARCHICAL_PATH = (
    RESULTS_ROOT / "tables" / "e15_l4_factorial_hierarchical_summary_v2.csv"
)
DISPERSION_PATH = (
    RESULTS_ROOT / "tables" / "e15_l4_factorial_crossed_dispersion_v2.csv"
)
ANCHOR_CONTINUITY_PATH = (
    RESULTS_ROOT / "tables" / "e15_anchor_outcome_continuity_v2.csv"
)
UPSTREAM_MANIFEST_PATH = (
    RESULTS_ROOT
    / "logs"
    / "e15_rule_construction_crossing_artifact_manifest_v2.json"
)
PDF_OUTPUT = (
    RESULTS_ROOT / "figures" / "e15_rule_construction_crossing_v2.pdf"
)
PNG_OUTPUT = (
    RESULTS_ROOT / "figures" / "e15_rule_construction_crossing_v2.png"
)
PROVENANCE_OUTPUT = (
    RESULTS_ROOT
    / "logs"
    / "e15_rule_construction_crossing_figure_provenance_v2.json"
)

EXPECTED_HIERARCHICAL_BYTES = 598_903
EXPECTED_HIERARCHICAL_SHA256 = (
    "d43cbea09ea9956de9add362787bf0e90016596318accb5bae8149372ed3e0f3"
)
EXPECTED_DISPERSION_BYTES = 121_730
EXPECTED_DISPERSION_SHA256 = (
    "376dedb87e1bf418a962b5728f224b2d9448b5cafbcb341f3d0c1585341a3829"
)
EXPECTED_ANCHOR_CONTINUITY_BYTES = 134_995
EXPECTED_ANCHOR_CONTINUITY_SHA256 = (
    "ec2702c53316b8f49035069f2c34fb8ebcac3dff43b65eeeb56e7c3290a873ac"
)
EXPECTED_MANIFEST_BYTES = 313_466
EXPECTED_MANIFEST_SHA256 = (
    "2f91d784e3d0e001f47a1fb3c4318fc7d08bd98eb86c2725398850e1ec22169d"
)
EXPECTED_HIERARCHICAL_ROWS = 2_160
EXPECTED_DISPERSION_ROWS = 288
EXPECTED_ANCHOR_CONTINUITY_ROWS = 495
EXPECTED_NEW_CONSTRUCTIONS = 4
EXPECTED_PIPELINE_SEEDS = 5
EXPECTED_DF = 3
EXPECTED_VERDICT = (
    "C-MAIN-AND-INTERACTION+ATTACK-HETEROGENEOUS(S,PS)"
)
EXPECTED_P_DIRECTION = "P-BOTH-MARGINS-CONCORDANT"
EXPECTED_EXACT_HOLM_FAMILY = "primary_exact_construction_new4"
EXPECTED_BINARY_HOLM_FAMILY = "companion_binary_construction_new4"
EFFECTS = ("P", "S", "D", "PS", "PD", "SD", "PSD")
HETEROGENEOUS_EFFECTS = ("S", "PS")
NEW_CONSTRUCTION_SEEDS = ("271828", "161803", "141421", "173205")
ANCHOR_SEED = "314159"

EFFECT_LABELS = {
    "P": "P — payload-role destination",
    "S": "S — frame-mask layout/span",
    "D": "D — payload dynamics",
    "PS": "P×S",
    "PD": "P×D",
    "SD": "S×D",
    "PSD": "P×S×D",
}

HIERARCHICAL_COLUMNS = {
    "endpoint",
    "id_scope",
    "attack_scope",
    "effect",
    "row_type",
    "unit_id",
    "unit_role",
    "value",
    "status",
    "n_available",
    "mean",
    "sd",
    "ci_low",
    "ci_high",
    "degrees_of_freedom",
    "raw_p",
    "holm_adjusted_p",
    "p_status",
    "holm_family",
    "degenerate",
    "strict_positive",
    "strict_negative",
    "strict_zero",
    "construction_resolved",
    "construction_sign_heterogeneous",
    "pipeline_margin_directionally_concordant",
    "pipeline_margin_heterogeneous",
    "attack_heterogeneous",
    "shared_attack_direction_eligible",
    "endpoint_relation",
    "p_direction_status",
    "scientific_verdict",
}

DISPERSION_COLUMNS = {
    "endpoint",
    "id_scope",
    "attack_scope",
    "effect",
    "scope",
    "G",
    "P",
    "grid_cells",
    "grand_mean",
    "share_construction",
    "share_pipeline",
    "share_interaction",
    "closure_error",
    "status",
    "interpretation",
}

ANCHOR_CONTINUITY_COLUMNS = {
    "row_scope",
    "pipeline_seed",
    "block_id",
    "n",
    "exact_correct_difference",
    "binary_correct_difference",
    "normal_correct_difference",
    "status",
    "checkpoint_bytes_identical",
}

# Okabe-Ito colors, with marker/line redundancy for grayscale reproduction.
BLUE = "#0072B2"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
GRAY = "#777777"
LIGHT_GRAY = "#D5D5D5"
INK = "#222222"
GRID = "#D9D9D9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


@dataclass(frozen=True)
class PrimaryEffect:
    """One validated construction-first primary effect."""

    effect: str
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    raw_p: float
    holm_p: float
    construction_values: tuple[float, ...]
    anchor_value: float
    binary_mean: float
    binary_sd: float
    binary_ci_low: float
    binary_ci_high: float
    binary_raw_p: float
    binary_holm_p: float
    binary_construction_values: tuple[float, ...]
    binary_anchor_value: float
    endpoint_discordant: bool


@dataclass(frozen=True)
class SupportingEffect:
    """One non-confirmatory attack-specific construction summary."""

    attack: str
    effect: str
    mean: float
    ci_low: float
    ci_high: float
    strict_positive: int
    strict_negative: int


@dataclass(frozen=True)
class DispersionShare:
    """One descriptive crossed-grid dispersion partition."""

    effect: str
    construction: float
    pipeline: float
    interaction: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes repository: {path}")
    return resolved.relative_to(root).as_posix()


def artifact_record(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required regular artifact is missing: {path}")
    return {
        "path": repo_relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_frozen_artifact(
    record: Mapping[str, object],
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    if (
        record["bytes"] != expected_bytes
        or record["sha256"] != expected_sha256
    ):
        raise ValueError(
            "frozen E15 input identity mismatch for "
            f"{record['path']}: expected bytes={expected_bytes}, "
            f"sha256={expected_sha256}; observed bytes={record['bytes']}, "
            f"sha256={record['sha256']}"
        )


def strict_float(row: pd.Series, field: str) -> float:
    raw = str(row[field])
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{row.get('attack_scope', '')}/{row.get('effect', '')}: "
            f"{field} is not numeric: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"{row.get('attack_scope', '')}/{row.get('effect', '')}: "
            f"{field} is not finite"
        )
    return value


def strict_int(row: pd.Series, field: str) -> int:
    raw = str(row[field])
    try:
        numeric = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{row.get('attack_scope', '')}/{row.get('effect', '')}: "
            f"{field} is not an integer: {raw!r}"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"{row.get('attack_scope', '')}/{row.get('effect', '')}: "
            f"{field} is not an integer: {raw!r}"
        )
    return int(numeric)


def strict_bool(row: pd.Series, field: str) -> bool:
    raw = str(row[field])
    if raw not in {"True", "False"}:
        raise ValueError(
            f"{row.get('attack_scope', '')}/{row.get('effect', '')}: "
            f"{field} is not a strict boolean: {raw!r}"
        )
    return raw == "True"


def validate_upstream_manifest(
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_record = artifact_record(UPSTREAM_MANIFEST_PATH)
    require_frozen_artifact(
        manifest_record,
        expected_bytes=EXPECTED_MANIFEST_BYTES,
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    payload = json.loads(UPSTREAM_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "e15.rule_construction_artifact_manifest.v1"
    ):
        raise ValueError("unexpected E15 artifact-manifest schema")
    if (
        payload.get("technical_status") != "T-PASS"
        or payload.get("stage_status") != "analysis_complete"
        or payload.get("scientific_verdict_status") != "complete"
        or payload.get("completion_marker") is not True
    ):
        raise ValueError("E15 machine verdict is not technically complete")

    verdict = payload.get("scientific_verdict", {})
    if (
        verdict.get("scientific_verdict") != EXPECTED_VERDICT
        or verdict.get("p_direction_status") != EXPECTED_P_DIRECTION
        or verdict.get("resolved_exact_effects") != list(EFFECTS)
        or verdict.get("attack_heterogeneous_effects")
        != list(HETEROGENEOUS_EFFECTS)
        or verdict.get("construction_sign_heterogeneous_effects") != []
        or verdict.get("pipeline_margin_heterogeneous_effects") != []
    ):
        raise ValueError("E15 scientific verdict changed")

    units = verdict.get("inferential_units", {})
    if units != {
        "blocks_averaged": 3,
        "degrees_of_freedom": EXPECTED_DF,
        "historical_anchor_excluded": True,
        "new_constructions": EXPECTED_NEW_CONSTRUCTIONS,
        "pipeline_seeds_fixed": EXPECTED_PIPELINE_SEEDS,
    }:
        raise ValueError("E15 inferential-unit declaration changed")

    contract = payload.get("analysis_contract", {})
    required_contract = {
        "anchor_excluded_from_primary_n4": True,
        "blocks_averaged_before_construction_inference": True,
        "dispersion_is_descriptive_finite_grid": True,
        "exact_and_binary_holm_families_separate": True,
        "external_dataset_evaluation": False,
        "l2_p_values_or_verdicts": False,
        "shared_real_reference_counted_once": True,
    }
    if any(contract.get(key) is not value for key, value in required_contract.items()):
        raise ValueError("E15 analysis contract changed")
    if payload.get("fresh_anchor_outcome_continuity", {}).get("status") != (
        "anchor-outcome-drift"
    ):
        raise ValueError("E15 fresh-anchor drift disclosure changed")
    if payload.get("next_permitted_step") != (
        "create a figure, update the paper-artifact manifest, and revise "
        "the manuscript only from this complete machine-readable verdict"
    ):
        raise ValueError("E15 next-permitted-step declaration changed")
    return payload, manifest_record


def read_frozen_table(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_rows: int,
    required_columns: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    record = artifact_record(path)
    require_frozen_artifact(
        record,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(
            f"{path.name} row count changed: expected {expected_rows}, "
            f"observed {len(frame)}"
        )
    return frame, record


def select_primary(
    frame: pd.DataFrame,
) -> list[PrimaryEffect]:
    exact_family = frame[
        frame["endpoint"].eq("exact_recall")
        & frame["id_scope"].eq("canonical")
        & frame["attack_scope"].eq("equal_macro")
        & frame["effect"].isin(EFFECTS)
    ].copy()
    binary_family = frame[
        frame["endpoint"].eq("binary_recall")
        & frame["id_scope"].eq("canonical")
        & frame["attack_scope"].eq("equal_macro")
        & frame["effect"].isin(EFFECTS)
    ].copy()
    summaries = exact_family[
        exact_family["row_type"].eq("summary_new4")
    ].copy()
    binary_summaries = binary_family[
        binary_family["row_type"].eq("summary_new4")
    ].copy()
    constructions = exact_family[
        exact_family["row_type"].eq("construction")
    ].copy()
    binary_constructions = binary_family[
        binary_family["row_type"].eq("construction")
    ].copy()
    if (
        len(summaries) != len(EFFECTS)
        or set(summaries["effect"]) != set(EFFECTS)
        or summaries["effect"].duplicated().any()
        or len(binary_summaries) != len(EFFECTS)
        or set(binary_summaries["effect"]) != set(EFFECTS)
        or binary_summaries["effect"].duplicated().any()
    ):
        raise ValueError(
            "expected one exact and one binary new4 summary per effect"
        )

    indexed = summaries.set_index("effect", drop=False)
    binary_indexed = binary_summaries.set_index("effect", drop=False)
    output: list[PrimaryEffect] = []
    for effect in EFFECTS:
        row = indexed.loc[effect]
        if (
            row["unit_id"] != "new4"
            or row["unit_role"] != "primary_new_constructions"
            or row["status"] != "complete"
            or row["p_status"] != "inferential_p"
            or row["holm_family"] != EXPECTED_EXACT_HOLM_FAMILY
            or strict_int(row, "n_available") != EXPECTED_NEW_CONSTRUCTIONS
            or strict_int(row, "degrees_of_freedom") != EXPECTED_DF
            or strict_bool(row, "degenerate")
            or not strict_bool(row, "construction_resolved")
            or strict_bool(row, "construction_sign_heterogeneous")
            or strict_bool(row, "pipeline_margin_heterogeneous")
        ):
            raise ValueError(f"{effect}: invalid primary E15 decision metadata")

        mean = strict_float(row, "mean")
        if not math.isclose(
            mean, strict_float(row, "value"), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"{effect}: primary value/mean mismatch")
        ci_low = strict_float(row, "ci_low")
        ci_high = strict_float(row, "ci_high")
        holm_p = strict_float(row, "holm_adjusted_p")
        if (
            not ci_low <= mean <= ci_high
            or holm_p >= 0.05
            or strict_float(row, "raw_p") > holm_p + 1e-15
        ):
            raise ValueError(f"{effect}: invalid primary interval or p-value")

        sign = 1 if mean > 0.0 else -1
        expected_positive = EXPECTED_NEW_CONSTRUCTIONS if sign > 0 else 0
        expected_negative = EXPECTED_NEW_CONSTRUCTIONS if sign < 0 else 0
        if (
            strict_int(row, "strict_positive") != expected_positive
            or strict_int(row, "strict_negative") != expected_negative
            or strict_int(row, "strict_zero") != 0
        ):
            raise ValueError(f"{effect}: construction sign counts changed")

        effect_rows = constructions[constructions["effect"].eq(effect)]
        new_rows = effect_rows[
            effect_rows["unit_role"].eq("new_construction_primary")
        ]
        anchor_rows = effect_rows[
            effect_rows["unit_role"].eq("fresh_known_construction_anchor")
        ]
        if (
            len(new_rows) != EXPECTED_NEW_CONSTRUCTIONS
            or set(new_rows["unit_id"]) != set(NEW_CONSTRUCTION_SEEDS)
            or len(anchor_rows) != 1
            or str(anchor_rows.iloc[0]["unit_id"]) != ANCHOR_SEED
        ):
            raise ValueError(f"{effect}: construction-unit grid changed")
        ordered = new_rows.set_index("unit_id")
        construction_values = tuple(
            strict_float(ordered.loc[seed], "value")
            for seed in NEW_CONSTRUCTION_SEEDS
        )
        if not math.isclose(
            sum(construction_values) / len(construction_values),
            mean,
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError(f"{effect}: construction values do not average to mean")

        binary_row = binary_indexed.loc[effect]
        if (
            binary_row["unit_id"] != "new4"
            or binary_row["unit_role"] != "primary_new_constructions"
            or binary_row["status"] != "complete"
            or binary_row["p_status"] != "inferential_p"
            or binary_row["holm_family"] != EXPECTED_BINARY_HOLM_FAMILY
            or strict_int(binary_row, "n_available")
            != EXPECTED_NEW_CONSTRUCTIONS
            or strict_int(binary_row, "degrees_of_freedom") != EXPECTED_DF
            or strict_bool(binary_row, "degenerate")
            or not strict_bool(
                binary_row, "binary_construction_resolved"
            )
            or strict_bool(
                binary_row, "construction_sign_heterogeneous"
            )
        ):
            raise ValueError(
                f"{effect}: invalid binary companion decision metadata"
            )
        binary_mean = strict_float(binary_row, "mean")
        if not math.isclose(
            binary_mean,
            strict_float(binary_row, "value"),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{effect}: binary value/mean mismatch")
        binary_ci_low = strict_float(binary_row, "ci_low")
        binary_ci_high = strict_float(binary_row, "ci_high")
        binary_holm_p = strict_float(binary_row, "holm_adjusted_p")
        if (
            not binary_ci_low <= binary_mean <= binary_ci_high
            or binary_holm_p >= 0.05
            or strict_float(binary_row, "raw_p")
            > binary_holm_p + 1e-15
        ):
            raise ValueError(
                f"{effect}: invalid binary interval or p-value"
            )
        binary_sign = 1 if binary_mean > 0.0 else -1
        binary_positive = (
            EXPECTED_NEW_CONSTRUCTIONS if binary_sign > 0 else 0
        )
        binary_negative = (
            EXPECTED_NEW_CONSTRUCTIONS if binary_sign < 0 else 0
        )
        if (
            strict_int(binary_row, "strict_positive") != binary_positive
            or strict_int(binary_row, "strict_negative") != binary_negative
            or strict_int(binary_row, "strict_zero") != 0
        ):
            raise ValueError(
                f"{effect}: binary construction sign counts changed"
            )
        binary_effect_rows = binary_constructions[
            binary_constructions["effect"].eq(effect)
        ]
        binary_new_rows = binary_effect_rows[
            binary_effect_rows["unit_role"].eq(
                "new_construction_primary"
            )
        ]
        binary_anchor_rows = binary_effect_rows[
            binary_effect_rows["unit_role"].eq(
                "fresh_known_construction_anchor"
            )
        ]
        if (
            len(binary_new_rows) != EXPECTED_NEW_CONSTRUCTIONS
            or set(binary_new_rows["unit_id"])
            != set(NEW_CONSTRUCTION_SEEDS)
            or len(binary_anchor_rows) != 1
            or str(binary_anchor_rows.iloc[0]["unit_id"]) != ANCHOR_SEED
        ):
            raise ValueError(
                f"{effect}: binary construction-unit grid changed"
            )
        binary_ordered = binary_new_rows.set_index("unit_id")
        binary_construction_values = tuple(
            strict_float(binary_ordered.loc[seed], "value")
            for seed in NEW_CONSTRUCTION_SEEDS
        )
        if not math.isclose(
            sum(binary_construction_values)
            / len(binary_construction_values),
            binary_mean,
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError(
                f"{effect}: binary constructions do not average to mean"
            )
        endpoint_discordant = effect == "SD"
        expected_relation = (
            "endpoint-discordant"
            if endpoint_discordant
            else "endpoint-directionally-concordant"
        )
        if (
            row["endpoint_relation"] != expected_relation
            or binary_row["endpoint_relation"] != expected_relation
        ):
            raise ValueError(f"{effect}: endpoint relation changed")

        output.append(PrimaryEffect(
            effect=effect,
            mean=mean,
            sd=strict_float(row, "sd"),
            ci_low=ci_low,
            ci_high=ci_high,
            raw_p=strict_float(row, "raw_p"),
            holm_p=holm_p,
            construction_values=construction_values,
            anchor_value=strict_float(anchor_rows.iloc[0], "value"),
            binary_mean=binary_mean,
            binary_sd=strict_float(binary_row, "sd"),
            binary_ci_low=binary_ci_low,
            binary_ci_high=binary_ci_high,
            binary_raw_p=strict_float(binary_row, "raw_p"),
            binary_holm_p=binary_holm_p,
            binary_construction_values=binary_construction_values,
            binary_anchor_value=strict_float(
                binary_anchor_rows.iloc[0], "value"
            ),
            endpoint_discordant=endpoint_discordant,
        ))

    verdict_rows = summaries[
        summaries["scientific_verdict"].eq(EXPECTED_VERDICT)
    ]
    if len(verdict_rows) != 1 or verdict_rows.iloc[0]["effect"] != "P":
        raise ValueError("expected one canonical E15 verdict carrier on P")
    if indexed.loc["P", "p_direction_status"] != EXPECTED_P_DIRECTION:
        raise ValueError("P construction/pipeline concordance changed")
    return output


def select_supporting(
    frame: pd.DataFrame,
) -> dict[str, list[SupportingEffect]]:
    selected = frame[
        frame["endpoint"].eq("exact_recall")
        & frame["id_scope"].eq("canonical")
        & frame["attack_scope"].isin(("Gear", "RPM"))
        & frame["effect"].isin(HETEROGENEOUS_EFFECTS)
        & frame["row_type"].eq("summary_new4")
    ].copy()
    if len(selected) != 4:
        raise ValueError("expected four Gear/RPM supporting summaries")

    output: dict[str, list[SupportingEffect]] = {}
    for attack in ("Gear", "RPM"):
        attack_rows = selected[selected["attack_scope"].eq(attack)]
        indexed = attack_rows.set_index("effect", drop=False)
        if set(indexed.index) != set(HETEROGENEOUS_EFFECTS):
            raise ValueError(f"{attack}: incomplete supporting effect set")
        output[attack] = []
        for effect in HETEROGENEOUS_EFFECTS:
            row = indexed.loc[effect]
            if (
                row["status"] != "complete"
                or row["p_status"] != "not_tested_supporting"
                or strict_int(row, "n_available")
                != EXPECTED_NEW_CONSTRUCTIONS
                or str(row["raw_p"]) != ""
                or str(row["holm_adjusted_p"]) != ""
            ):
                raise ValueError(
                    f"{attack}/{effect}: supporting row was promoted or changed"
                )
            mean = strict_float(row, "mean")
            output[attack].append(SupportingEffect(
                attack=attack,
                effect=effect,
                mean=mean,
                ci_low=strict_float(row, "ci_low"),
                ci_high=strict_float(row, "ci_high"),
                strict_positive=strict_int(row, "strict_positive"),
                strict_negative=strict_int(row, "strict_negative"),
            ))

    by_attack = {
        attack: {row.effect: row for row in rows}
        for attack, rows in output.items()
    }
    for effect in HETEROGENEOUS_EFFECTS:
        gear = by_attack["Gear"][effect]
        rpm = by_attack["RPM"][effect]
        if (
            gear.mean <= 0.0
            or gear.strict_positive != EXPECTED_NEW_CONSTRUCTIONS
            or gear.strict_negative != 0
            or rpm.mean >= 0.0
            or rpm.strict_positive != 2
            or rpm.strict_negative != 2
            or not rpm.ci_low < 0.0 < rpm.ci_high
        ):
            raise ValueError(f"{effect}: registered attack heterogeneity changed")
    return output


def select_dispersion(frame: pd.DataFrame) -> list[DispersionShare]:
    selected = frame[
        frame["endpoint"].eq("exact_recall")
        & frame["id_scope"].eq("canonical")
        & frame["attack_scope"].eq("equal_macro")
        & frame["effect"].isin(EFFECTS)
        & frame["scope"].eq("new4")
    ].copy()
    if (
        len(selected) != len(EFFECTS)
        or set(selected["effect"]) != set(EFFECTS)
        or selected["effect"].duplicated().any()
    ):
        raise ValueError("expected one new4 dispersion row per primary effect")
    indexed = selected.set_index("effect", drop=False)
    output: list[DispersionShare] = []
    expected_interpretation = (
        "descriptive finite-grid dispersion; not a random-effects variance "
        "component or population variance explained"
    )
    for effect in EFFECTS:
        row = indexed.loc[effect]
        if (
            row["status"] != "complete"
            or row["interpretation"] != expected_interpretation
            or strict_int(row, "G") != EXPECTED_NEW_CONSTRUCTIONS
            or strict_int(row, "P") != EXPECTED_PIPELINE_SEEDS
            or strict_int(row, "grid_cells")
            != EXPECTED_NEW_CONSTRUCTIONS * EXPECTED_PIPELINE_SEEDS
        ):
            raise ValueError(f"{effect}: crossed-grid declaration changed")
        shares = tuple(
            strict_float(row, field)
            for field in (
                "share_construction",
                "share_pipeline",
                "share_interaction",
            )
        )
        if any(value < 0.0 or value > 1.0 for value in shares):
            raise ValueError(f"{effect}: dispersion share is outside [0,1]")
        if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{effect}: dispersion shares do not close")
        if abs(strict_float(row, "closure_error")) > 1e-12:
            raise ValueError(f"{effect}: crossed-grid closure error changed")
        output.append(DispersionShare(
            effect=effect,
            construction=shares[0],
            pipeline=shares[1],
            interaction=shares[2],
        ))
    return output


def validate_anchor_continuity(
    frame: pd.DataFrame,
) -> dict[str, object]:
    if len(frame) != EXPECTED_ANCHOR_CONTINUITY_ROWS:
        raise ValueError("fresh-anchor continuity row count changed")
    scope_counts = frame["row_scope"].value_counts().to_dict()
    status_counts = frame["status"].value_counts().to_dict()
    if scope_counts != {"scenario": 480, "normal": 15}:
        raise ValueError("fresh-anchor scenario/normal partition changed")
    if status_counts != {
        "anchor-outcome-drift": 291,
        "anchor-outcome-identical": 204,
    }:
        raise ValueError("fresh-anchor drift/identity counts changed")
    if set(frame["pipeline_seed"]) != {"7", "42", "123", "2026", "3407"}:
        raise ValueError("fresh-anchor pipeline grid changed")
    if set(frame["block_id"]) != {"block_01", "block_02", "block_03"}:
        raise ValueError("fresh-anchor block grid changed")
    if any(strict_int(row, "n") != 2_000 for _, row in frame.iterrows()):
        raise ValueError("fresh-anchor denominator changed")
    if set(frame["checkpoint_bytes_identical"]) != {"False"}:
        raise ValueError("fresh/frozen checkpoint identity status changed")

    def nonzero_count(field: str) -> int:
        values = pd.to_numeric(frame[field], errors="coerce")
        return int((values.fillna(0.0) != 0.0).sum())

    counts = {
        "exact_correct_difference_nonzero_rows": nonzero_count(
            "exact_correct_difference"
        ),
        "binary_correct_difference_nonzero_rows": nonzero_count(
            "binary_correct_difference"
        ),
        "normal_correct_difference_nonzero_rows": nonzero_count(
            "normal_correct_difference"
        ),
    }
    expected = {
        "exact_correct_difference_nonzero_rows": 263,
        "binary_correct_difference_nonzero_rows": 184,
        "normal_correct_difference_nonzero_rows": 4,
    }
    if counts != expected:
        raise ValueError(
            f"fresh-anchor nonzero difference counts changed: {counts}"
        )
    return {
        "status": "anchor-outcome-drift",
        "rows": EXPECTED_ANCHOR_CONTINUITY_ROWS,
        "scenario_rows": 480,
        "normal_rows": 15,
        "drift_rows": 291,
        "identical_rows": 204,
        **counts,
        "changes_primary_new4_test": False,
        "exact_reproduction_claim_allowed": False,
    }


def load_validated_evidence() -> tuple[
    list[PrimaryEffect],
    dict[str, list[SupportingEffect]],
    list[DispersionShare],
    dict[str, object],
    dict[str, object],
]:
    manifest, manifest_record = validate_upstream_manifest()
    hierarchical, hierarchical_record = read_frozen_table(
        HIERARCHICAL_PATH,
        expected_bytes=EXPECTED_HIERARCHICAL_BYTES,
        expected_sha256=EXPECTED_HIERARCHICAL_SHA256,
        expected_rows=EXPECTED_HIERARCHICAL_ROWS,
        required_columns=HIERARCHICAL_COLUMNS,
    )
    dispersion, dispersion_record = read_frozen_table(
        DISPERSION_PATH,
        expected_bytes=EXPECTED_DISPERSION_BYTES,
        expected_sha256=EXPECTED_DISPERSION_SHA256,
        expected_rows=EXPECTED_DISPERSION_ROWS,
        required_columns=DISPERSION_COLUMNS,
    )
    anchor_continuity, anchor_continuity_record = read_frozen_table(
        ANCHOR_CONTINUITY_PATH,
        expected_bytes=EXPECTED_ANCHOR_CONTINUITY_BYTES,
        expected_sha256=EXPECTED_ANCHOR_CONTINUITY_SHA256,
        expected_rows=EXPECTED_ANCHOR_CONTINUITY_ROWS,
        required_columns=ANCHOR_CONTINUITY_COLUMNS,
    )
    if manifest["output_tables"]["hierarchical_summary"] != {
        **hierarchical_record,
        "rows": EXPECTED_HIERARCHICAL_ROWS,
        "unique_key": [
            "endpoint",
            "id_scope",
            "attack_scope",
            "effect",
            "row_type",
            "unit_id",
        ],
    }:
        raise ValueError("machine manifest does not bind the hierarchical input")
    if manifest["output_tables"]["crossed_dispersion"] != {
        **dispersion_record,
        "rows": EXPECTED_DISPERSION_ROWS,
        "unique_key": [
            "endpoint",
            "id_scope",
            "attack_scope",
            "effect",
            "scope",
        ],
    }:
        raise ValueError("machine manifest does not bind the dispersion input")
    expected_anchor_record = {
        **anchor_continuity_record,
        "rows": EXPECTED_ANCHOR_CONTINUITY_ROWS,
    }
    if (
        manifest["evaluator_handoff"]["outputs"]["anchor_continuity"]
        != expected_anchor_record
    ):
        raise ValueError(
            "machine manifest does not bind the anchor-continuity input"
        )

    inputs = {
        "hierarchical_summary": hierarchical_record,
        "crossed_dispersion": dispersion_record,
        "anchor_outcome_continuity": anchor_continuity_record,
        "upstream_artifact_manifest": manifest_record,
        "upstream_manifest_schema": manifest["schema_version"],
        "upstream_source_commit": manifest["source_commit"],
    }
    return (
        select_primary(hierarchical),
        select_supporting(hierarchical),
        select_dispersion(dispersion),
        inputs,
        validate_anchor_continuity(anchor_continuity),
    )


def style_zero_axis(ax: plt.Axes) -> None:
    ax.axvline(0.0, color=INK, linewidth=0.85, linestyle="--", zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_figure(
    primary: Sequence[PrimaryEffect],
    supporting: Mapping[str, Sequence[SupportingEffect]],
    dispersion: Sequence[DispersionShare],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.45, 5.25))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.38, 1.0),
        height_ratios=(0.88, 1.12),
        wspace=0.42,
        hspace=0.56,
    )
    left = fig.add_subplot(grid[:, 0])
    top_right = fig.add_subplot(grid[0, 1])
    bottom_right = fig.add_subplot(grid[1, 1])

    positions = list(reversed(range(len(EFFECTS))))
    jitter = (-0.045, -0.015, 0.015, 0.045)
    for position, summary in zip(positions, primary):
        if summary.endpoint_discordant:
            left.axhspan(
                position - 0.40,
                position + 0.40,
                color="#F2F2F2",
                zorder=0,
            )
        exact_y = position + 0.11
        binary_y = position - 0.11
        for offset, value in zip(jitter, summary.construction_values):
            left.scatter(
                value,
                exact_y + offset,
                marker="o",
                s=12,
                facecolor="white",
                edgecolor=BLUE,
                linewidth=0.75,
                zorder=3,
            )
        for offset, value in zip(
            jitter, summary.binary_construction_values
        ):
            left.scatter(
                value,
                binary_y + offset,
                marker="s",
                s=11,
                facecolor="white",
                edgecolor=ORANGE,
                linewidth=0.75,
                zorder=3,
            )
        left.scatter(
            summary.anchor_value,
            exact_y + 0.15,
            marker="D",
            s=23,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1.0,
            zorder=3,
        )
        left.scatter(
            summary.binary_anchor_value,
            binary_y - 0.15,
            marker="s",
            s=23,
            facecolor="white",
            edgecolor=ORANGE,
            linewidth=1.0,
            zorder=3,
        )
        left.errorbar(
            summary.mean,
            exact_y,
            xerr=[
                [summary.mean - summary.ci_low],
                [summary.ci_high - summary.mean],
            ],
            fmt="D",
            color=BLUE,
            ecolor=BLUE,
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            markersize=4.5,
            elinewidth=1.1,
            capsize=2.2,
            capthick=0.9,
            zorder=4,
        )
        left.errorbar(
            summary.binary_mean,
            binary_y,
            xerr=[
                [summary.binary_mean - summary.binary_ci_low],
                [summary.binary_ci_high - summary.binary_mean],
            ],
            fmt="s",
            color=ORANGE,
            ecolor=ORANGE,
            markerfacecolor=ORANGE,
            markeredgecolor=ORANGE,
            markersize=4.3,
            elinewidth=1.1,
            capsize=2.2,
            capthick=0.9,
            zorder=4,
        )
    left.set_yticks(positions)
    left.set_yticklabels(
        [EFFECT_LABELS[effect] for effect in EFFECTS],
        fontsize=7.8,
    )
    left.set_ylim(-0.55, len(EFFECTS) - 0.35)
    left.set_xlim(-1.02, 0.18)
    left.set_xticks((-1.0, -0.8, -0.6, -0.4, -0.2, 0.0))
    left.set_title(
        "(a) Construction-first exact and binary effects",
        loc="left",
        fontsize=9.3,
        fontweight="bold",
        pad=7,
    )
    left.set_xlabel(
        "Rule − matched-real recall effect",
        fontsize=8.0,
    )
    style_zero_axis(left)
    left.legend(
        handles=[
            Line2D(
                [], [], marker="D", linestyle="-", color=BLUE,
                markerfacecolor=BLUE, markersize=4.5,
                label="exact mean + nominal CI",
            ),
            Line2D(
                [], [], marker="s", linestyle="-", color=ORANGE,
                markerfacecolor=ORANGE, markersize=4.3,
                label="binary mean + nominal CI",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", color=GRAY,
                markerfacecolor="white", markersize=3.8,
                label="4 construction marginals",
            ),
            Line2D(
                [], [], marker="D", linestyle="none", color=GRAY,
                markerfacecolor="white", markersize=4.6,
                label="fresh 314159 anchor (excluded)",
            ),
        ],
        loc="lower left",
        fontsize=6.8,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.45,
        borderaxespad=0.35,
    )

    support_map = {
        attack: {row.effect: row for row in rows}
        for attack, rows in supporting.items()
    }
    support_positions = {"S": 1, "PS": 0}
    attack_style = {
        "Gear": (BLUE, "o", 0.10),
        "RPM": (ORANGE, "s", -0.10),
    }
    for attack in ("Gear", "RPM"):
        color, marker, offset = attack_style[attack]
        for effect in HETEROGENEOUS_EFFECTS:
            row = support_map[attack][effect]
            position = support_positions[effect] + offset
            top_right.errorbar(
                row.mean,
                position,
                xerr=[
                    [row.mean - row.ci_low],
                    [row.ci_high - row.mean],
                ],
                fmt=marker,
                color=color,
                ecolor=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=4.5,
                elinewidth=1.05,
                capsize=2.0,
                zorder=3,
            )
    top_right.set_yticks((1, 0))
    top_right.set_yticklabels(("S", "P×S"))
    top_right.set_ylim(-0.45, 1.45)
    top_right.set_xlim(-0.08, 0.28)
    top_right.set_xticks((-0.05, 0.0, 0.1, 0.2))
    top_right.set_title(
        "(b) Attack-specific support",
        loc="left",
        fontsize=9.3,
        fontweight="bold",
        pad=7,
    )
    top_right.set_xlabel("Exact-macro effect", fontsize=8.0)
    style_zero_axis(top_right)
    top_right.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", color=BLUE,
                markerfacecolor=BLUE, markersize=4.5, label="Gear",
            ),
            Line2D(
                [], [], marker="s", linestyle="none", color=ORANGE,
                markerfacecolor=ORANGE, markersize=4.3, label="RPM",
            ),
        ],
        loc="lower right",
        fontsize=7.1,
        frameon=False,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
        borderaxespad=0.25,
    )
    top_right.text(
        0.01,
        0.98,
        "Supporting only; RPM CIs cross zero",
        transform=top_right.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
        color=GRAY,
    )

    dispersion_map = {row.effect: row for row in dispersion}
    d_positions = list(reversed(range(len(EFFECTS))))
    construction_values = [
        dispersion_map[effect].construction for effect in EFFECTS
    ]
    pipeline_values = [
        dispersion_map[effect].pipeline for effect in EFFECTS
    ]
    interaction_values = [
        dispersion_map[effect].interaction for effect in EFFECTS
    ]
    bottom_right.barh(
        d_positions,
        construction_values,
        color=SKY,
        edgecolor="white",
        linewidth=0.35,
        height=0.68,
        label="construction",
    )
    bottom_right.barh(
        d_positions,
        pipeline_values,
        left=construction_values,
        color=GREEN,
        edgecolor="white",
        linewidth=0.35,
        height=0.68,
        label="pipeline",
    )
    starts = [
        construction + pipeline
        for construction, pipeline in zip(
            construction_values, pipeline_values
        )
    ]
    bottom_right.barh(
        d_positions,
        interaction_values,
        left=starts,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.35,
        height=0.68,
        label="interaction",
    )
    bottom_right.set_yticks(d_positions)
    bottom_right.set_yticklabels(EFFECTS)
    bottom_right.set_xlim(0.0, 1.0)
    bottom_right.set_xticks((0.0, 0.5, 1.0))
    bottom_right.set_xticklabels(("0%", "50%", "100%"))
    bottom_right.set_title(
        "(c) Crossed-grid dispersion",
        loc="left",
        fontsize=9.3,
        fontweight="bold",
        pad=7,
    )
    bottom_right.set_xlabel("Descriptive share", fontsize=8.0)
    bottom_right.tick_params(axis="both", labelsize=7.8)
    bottom_right.spines["top"].set_visible(False)
    bottom_right.spines["right"].set_visible(False)
    bottom_right.legend(
        handles=[
            Patch(facecolor=SKY, label="construction"),
            Patch(facecolor=GREEN, label="pipeline"),
            Patch(facecolor=ORANGE, label="interaction"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        fontsize=6.8,
        frameon=False,
        ncol=3,
        columnspacing=0.65,
        handlelength=0.9,
        handletextpad=0.25,
    )
    fig.subplots_adjust(
        left=0.205,
        right=0.985,
        bottom=0.13,
        top=0.94,
    )
    pdf_metadata = {
        "Title": "E15 Rule construction-seed crossing",
        "Author": "WISA CAN IDS journal extension",
        "Subject": (
            "Construction-first matched-budget factorial effects and "
            "descriptive crossed-grid dispersion"
        ),
        "Creator": "make_e15_rule_construction_crossing_figure.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata=pdf_metadata,
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={
            "Software": "make_e15_rule_construction_crossing_figure.py"
        },
    )
    plt.close(fig)


def validate_staged_outputs(paths: Iterable[Path]) -> None:
    paths = list(paths)
    if len(paths) != 2:
        raise ValueError("expected exactly one PDF and one PNG")
    for path in paths:
        if not path.is_file() or path.stat().st_size < 10_000:
            raise ValueError(f"staged figure is missing or too small: {path}")
    pdf = next(path for path in paths if path.suffix == ".pdf")
    png = next(path for path in paths if path.suffix == ".png")
    if pdf.read_bytes()[:5] != b"%PDF-":
        raise ValueError("staged PDF signature is invalid")
    with png.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("staged PNG signature is invalid")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1_800 or height < 1_200:
        raise ValueError(
            f"staged PNG resolution is unexpectedly small: {width}x{height}"
        )


def git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "worktree_clean": not status_lines,
        "status_lines": status_lines,
    }


def publish_no_clobber(pairs: Sequence[tuple[Path, Path]]) -> None:
    collisions = [
        str(destination)
        for _source, destination in pairs
        if destination.exists() or destination.is_symlink()
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite E15 figure artifacts: "
            f"{collisions}"
        )
    published: list[tuple[Path, Path]] = []
    try:
        for source, destination in pairs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            published.append((source, destination))
    except BaseException:
        for source, destination in reversed(published):
            try:
                source_stat = source.stat(follow_symlinks=False)
                destination_stat = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (
                source_stat.st_dev == destination_stat.st_dev
                and source_stat.st_ino == destination_stat.st_ino
            ):
                destination.unlink()
        raise


def build(*, check_only: bool = False) -> dict[str, object]:
    (
        primary,
        supporting,
        dispersion,
        provenance_inputs,
        anchor_continuity,
    ) = (
        load_validated_evidence()
    )
    if check_only:
        return {
            "status": "inputs_valid",
            "primary_effects": list(EFFECTS),
            "primary_construction_n": EXPECTED_NEW_CONSTRUCTIONS,
            "fixed_pipeline_seeds": EXPECTED_PIPELINE_SEEDS,
            "scientific_verdict": EXPECTED_VERDICT,
        }

    destinations = (PDF_OUTPUT, PNG_OUTPUT, PROVENANCE_OUTPUT)
    frozen_wisa = (REPO_ROOT / "wisa").resolve()
    for destination in destinations:
        resolved = destination.resolve()
        if resolved == frozen_wisa or resolved.is_relative_to(frozen_wisa):
            raise ValueError("refusing to publish under frozen wisa/")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite E15 figure artifact: {destination}"
            )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    source_state = git_metadata()
    temporary = Path(tempfile.mkdtemp(
        prefix=".e15-rule-construction-figure-",
        dir=RESULTS_ROOT,
    ))
    published = False
    try:
        stage_figures = temporary / "figures"
        stage_logs = temporary / "logs"
        stage_figures.mkdir()
        stage_logs.mkdir()
        staged_pdf = stage_figures / PDF_OUTPUT.name
        staged_png = stage_figures / PNG_OUTPUT.name
        repeat_pdf = stage_figures / f"repeat_{PDF_OUTPUT.name}"
        repeat_png = stage_figures / f"repeat_{PNG_OUTPUT.name}"
        staged_provenance = stage_logs / PROVENANCE_OUTPUT.name

        render_figure(
            primary,
            supporting,
            dispersion,
            pdf_path=staged_pdf,
            png_path=staged_png,
        )
        validate_staged_outputs((staged_pdf, staged_png))
        render_figure(
            primary,
            supporting,
            dispersion,
            pdf_path=repeat_pdf,
            png_path=repeat_png,
        )
        validate_staged_outputs((repeat_pdf, repeat_png))
        determinism_check = {
            "render_count": 2,
            "pdf_byte_identical": (
                sha256_file(staged_pdf) == sha256_file(repeat_pdf)
            ),
            "png_byte_identical": (
                sha256_file(staged_png) == sha256_file(repeat_png)
            ),
        }
        if not all(
            determinism_check[key]
            for key in ("pdf_byte_identical", "png_byte_identical")
        ):
            raise ValueError("repeated E15 figure render is not byte-identical")
        repeat_pdf.unlink()
        repeat_png.unlink()

        output_records = [
            {
                "path": repo_relative(destination),
                "bytes": staged.stat().st_size,
                "sha256": sha256_file(staged),
            }
            for staged, destination in (
                (staged_pdf, PDF_OUTPUT),
                (staged_png, PNG_OUTPUT),
            )
        ]
        provenance = {
            "schema_version": (
                "e15_rule_construction_crossing_figure_provenance_v1"
            ),
            "output_version": "v2",
            "record_type": "e15_figure_publication",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": "E15 Rule construction-seed crossing",
            "technical_status": "T-PASS",
            "scientific_verdict": EXPECTED_VERDICT,
            "scientific_scope": {
                "contrast": "rule_ms-real_ms",
                "training_budget": "matched_6156_update_maximum",
                "primary_endpoints": ["exact_recall", "binary_recall"],
                "id_scope": "canonical",
                "primary_attack_scope": "equal_macro",
                "primary_effects": list(EFFECTS),
                "primary_interval": (
                    "nominal_two_sided_95pct_student_t_df3"
                ),
                "primary_multiplicity": (
                    "separate seven-effect exact and binary Holm families"
                ),
                "endpoint_discordant_effects": ["SD"],
                "primary_construction_seeds": [
                    int(seed) for seed in NEW_CONSTRUCTION_SEEDS
                ],
                "historical_anchor_seed": int(ANCHOR_SEED),
                "historical_anchor_excluded_from_primary": True,
                "historical_anchor_status": (
                    "fresh_anchor_outcome_drift_descriptive_only"
                ),
                "fixed_pipeline_seeds": [7, 42, 123, 2026, 3407],
                "blocks_averaged_before_construction_inference": 3,
                "shared_real_references": True,
                "supporting_attack_scopes": ["Gear", "RPM"],
                "supporting_effects": list(HETEROGENEOUS_EFFECTS),
                "supporting_inference": (
                    "nominal_intervals_only_not_separately_tested"
                ),
                "dispersion_interpretation": (
                    "descriptive finite selected 4x5 grid; not random-effects "
                    "variance components or population variance explained"
                ),
                "estimand_boundary": (
                    "construction-marginal equal-budget augmentation-pipeline "
                    "association under one fixed Rule construction law; not "
                    "a physical mechanism, realism, real-attack replacement, "
                    "or external transfer"
                ),
            },
            "selected_rows": {
                "primary_summary_rows": 2 * len(primary),
                "primary_new_construction_rows": sum(
                    len(row.construction_values)
                    + len(row.binary_construction_values)
                    for row in primary
                ),
                "descriptive_anchor_rows": 2 * len(primary),
                "supporting_summary_rows": sum(
                    len(rows) for rows in supporting.values()
                ),
                "dispersion_rows": len(dispersion),
            },
            "anchor_outcome_continuity": anchor_continuity,
            "inputs": provenance_inputs,
            "script": artifact_record(SCRIPT),
            "rendering_environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "matplotlib": matplotlib.__version__,
                "pandas": pd.__version__,
                "backend": matplotlib.get_backend(),
                "source_date_epoch": os.environ["SOURCE_DATE_EPOCH"],
            },
            "determinism_check": determinism_check,
            "git": source_state,
            "outputs": output_records,
            "publication": {
                "method": (
                    "same-filesystem staging followed by no-clobber hard links"
                ),
                "overwrites_allowed": False,
            },
        }
        staged_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        json.loads(staged_provenance.read_text(encoding="utf-8"))
        publish_no_clobber((
            (staged_pdf, PDF_OUTPUT),
            (staged_png, PNG_OUTPUT),
            (staged_provenance, PROVENANCE_OUTPUT),
        ))
        published = True
    finally:
        if published:
            shutil.rmtree(temporary)
        elif temporary.exists():
            # Preserve a failed staging directory for forensic inspection.
            pass

    return {
        "status": "published",
        "outputs": [
            repo_relative(PDF_OUTPUT),
            repo_relative(PNG_OUTPUT),
        ],
        "provenance": repo_relative(PROVENANCE_OUTPUT),
        "scientific_verdict": EXPECTED_VERDICT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen E15 construction-crossing figure."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate frozen inputs and selected evidence without publishing",
    )
    args = parser.parse_args()
    try:
        result = build(check_only=args.check_only)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
