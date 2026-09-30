#!/usr/bin/env python3
"""Build the versioned E13 strict-v2 analysis tables.

This builder is deliberately separate from the legacy table generators.  It
combines the strict no-replacement-v2 per-seed evaluations with only the
retained controls fixed in the E13 prospective record, while preserving every
row's source file, source SHA-256, and sampling lineage.  Legacy ``+100%`` rows
are retained under a distinct arm identity and are used only for the explicitly
labelled joint corrective sensitivity:

    pool re-instantiation + no-replacement

That contrast must not be interpreted as the causal effect of replacement
policy alone.

Every selected evaluation cell must contain exactly the five prespecified
pipeline seeds and no duplicate seed rows.  All outputs are new, versioned,
no-clobber files.  They are staged first, linked into place without overwrite,
and committed by publishing the provenance JSON last.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t


SCRIPT_PATH = Path(__file__).resolve()
JOURNAL_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = JOURNAL_ROOT.parent
DEFAULT_TABLES = JOURNAL_ROOT / "results" / "tables"
DEFAULT_LOGS = JOURNAL_ROOT / "results" / "logs"
DEFAULT_PREREG = (
    JOURNAL_ROOT / "experiments" / "e13_strict_v2_sampling" / "PREREG.md"
)

EXPECTED_SEEDS = (7, 42, 123, 2026, 3407)
SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
METRICS = (
    "fpr",
    "normal_recall",
    "attack_recall",
    "macro_f1_binary",
    "exact_recall",
)
ROW_KEY = (
    "family",
    "training_schedule",
    "analysis_arm",
    "seed",
    "rung",
    "subset",
)
CELL_KEY = ("family", "training_schedule", "analysis_arm", "rung", "subset")
SAMPLE_SIZE_COLUMNS = ("windows", "normal_windows", "attack_windows")

ORIGINAL_SCHEDULE = "original_recipe_unmatched_optimizer_steps"
MATCHED_SCHEDULE = "matched_optimizer_steps"
SECOND_SOURCE_SCHEDULE = "retained_second_source_original_recipe"

STRICT_ROLE = "strict_v2_plus100"
STRICT_LINEAGE = "strict_without_replacement_v2"
RETAINED_ROLE = "retained_unaffected_control"
RETAINED_LINEAGE = "retained_untagged_control_unaffected_by_e13"
RETAINED_MATCHED_ROLE = "retained_unaffected_matched_step_control"
RETAINED_MATCHED_LINEAGE = (
    "retained_matched_step_control_unaffected_by_e13"
)
LEGACY_ROLE = "legacy_plus100_joint_sensitivity"
LEGACY_LINEAGE = "legacy_pool_reuse_sensitivity"

JOINT_CORRECTIVE_ESTIMAND = (
    "joint_pool_reinstantiation_plus_no_replacement_corrective_sensitivity"
)
JOINT_CORRECTIVE_BOUNDARY = (
    "does_not_isolate_the_causal_effect_of_replacement_policy"
)


@dataclass(frozen=True)
class SelectionSpec:
    """A disjoint row selection from one immutable source CSV."""

    name: str
    file_name: str
    family: str
    settings: tuple[str, ...]
    training_schedule: str
    analysis_role: str
    sampling_lineage: str
    allow_missing_exact_recall: bool = False


@dataclass(frozen=True)
class PairSpec:
    """A declared comparison between two uniquely identified analysis arms."""

    comparison: str
    family: str
    training_schedule: str
    arm_a: str
    arm_b: str
    estimand: str
    interpretation_boundary: str
    cell_policy: str = "exact"


@dataclass(frozen=True)
class FocalSpec:
    comparison: str
    hypothesis: str
    family: str
    training_schedule: str
    rung: str
    subset: str
    metric: str
    arm_a: str
    arm_b: str
    protocol_status: str


def arm_id(role: str, setting: str) -> str:
    """Return a stable arm identifier that cannot collapse legacy and v2 rows."""

    prefix = {
        STRICT_ROLE: "strict_v2",
        RETAINED_ROLE: "retained",
        RETAINED_MATCHED_ROLE: "retained_matched_steps",
        LEGACY_ROLE: "legacy_pool_reuse",
    }.get(role)
    if prefix is None:
        raise ValueError(f"unknown analysis role: {role}")
    return f"{prefix}::{setting}"


SELECTION_SPECS = (
    # CNN original-recipe retained controls and legacy +100 comparators.
    SelectionSpec(
        "cnn_retained_primary",
        "generator_extension_by_seed.csv",
        "cnn",
        ("real_only", "rule_0p30"),
        ORIGINAL_SCHEDULE,
        RETAINED_ROLE,
        RETAINED_LINEAGE,
    ),
    SelectionSpec(
        "cnn_retained_equal_budget_real",
        "generator_extension_by_seed_over.csv",
        "cnn",
        ("over_1p00",),
        ORIGINAL_SCHEDULE,
        RETAINED_ROLE,
        RETAINED_LINEAGE,
    ),
    SelectionSpec(
        "cnn_legacy_rule_plus100",
        "generator_extension_by_seed.csv",
        "cnn",
        ("rule_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "cnn_legacy_wgan_primary_plus100",
        "generator_extension_by_seed_gan_protocol_valid.csv",
        "cnn",
        ("ganvalid_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "cnn_legacy_wgan_gseed_plus100",
        "generator_extension_by_seed_gseed_protocol_valid_sensitivity.csv",
        "cnn",
        ("ganvalidb_1p00", "ganvalidc_1p00"),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "cnn_strict_v2_plus100",
        "generator_extension_by_seed_sampling_v2_strict_1p00.csv",
        "cnn",
        (
            "rule_1p00",
            "ganvalid_1p00",
            "ganvalidb_1p00",
            "ganvalidc_1p00",
        ),
        ORIGINAL_SCHEDULE,
        STRICT_ROLE,
        STRICT_LINEAGE,
    ),
    # CNN matched-step sensitivity and its unaffected retained controls.
    SelectionSpec(
        "cnn_retained_matched_step_controls",
        "generator_extension_by_seed_matchedsteps_controls.csv",
        "cnn",
        ("rule_0p30", "over_1p00"),
        MATCHED_SCHEDULE,
        RETAINED_MATCHED_ROLE,
        RETAINED_MATCHED_LINEAGE,
    ),
    SelectionSpec(
        "cnn_legacy_matched_step_wgan_plus100",
        "generator_extension_by_seed_matchedsteps_controls.csv",
        "cnn",
        ("ganvalid_1p00",),
        MATCHED_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "cnn_strict_v2_matched_step_wgan_plus100",
        "generator_extension_by_seed_matchedsteps_sampling_v2_ganvalid_1p00.csv",
        "cnn",
        ("ganvalid_1p00",),
        MATCHED_SCHEDULE,
        STRICT_ROLE,
        STRICT_LINEAGE,
    ),
    # RF original-recipe controls, legacy comparators, and strict arms.
    SelectionSpec(
        "rf_retained_primary",
        "rf_generator_extension_by_seed.csv",
        "rf",
        ("real_only", "rule_0p30"),
        ORIGINAL_SCHEDULE,
        RETAINED_ROLE,
        RETAINED_LINEAGE,
    ),
    SelectionSpec(
        "rf_legacy_rule_plus100",
        "rf_generator_extension_by_seed.csv",
        "rf",
        ("rule_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "rf_legacy_wgan_primary_plus100",
        "rf_generator_extension_by_seed_gan_protocol_valid.csv",
        "rf",
        ("ganvalid_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "rf_legacy_wgan_gseed_plus100",
        "rf_generator_extension_by_seed_gseed_protocol_valid_sensitivity.csv",
        "rf",
        ("ganvalidb_1p00", "ganvalidc_1p00"),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "rf_strict_v2_plus100",
        "rf_generator_extension_by_seed_sampling_v2.csv",
        "rf",
        (
            "rule_1p00",
            "ganvalid_1p00",
            "ganvalidb_1p00",
            "ganvalidc_1p00",
        ),
        ORIGINAL_SCHEDULE,
        STRICT_ROLE,
        STRICT_LINEAGE,
    ),
    # RF CANTT/ROAD evaluation is emitted by a separate streamed-external
    # evaluator.  These rows extend the same arm identities without
    # duplicating the local/OTIDS cells above.  Exact attack type is not
    # available for these external datasets and is represented as structural
    # NA for all five seeds.
    SelectionSpec(
        "rf_retained_streamed_external",
        "rf_generator_extension_external_by_seed.csv",
        "rf",
        ("real_only", "rule_0p30"),
        ORIGINAL_SCHEDULE,
        RETAINED_ROLE,
        RETAINED_LINEAGE,
        True,
    ),
    SelectionSpec(
        "rf_legacy_rule_streamed_external",
        "rf_generator_extension_external_by_seed.csv",
        "rf",
        ("rule_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
        True,
    ),
    SelectionSpec(
        "rf_legacy_wgan_primary_streamed_external",
        "rf_generator_extension_external_by_seed_protocol_valid.csv",
        "rf",
        ("ganvalid_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
        True,
    ),
    SelectionSpec(
        "rf_strict_v2_streamed_external",
        "rf_generator_extension_external_by_seed_sampling_v2_strict_1p00.csv",
        "rf",
        (
            "rule_1p00",
            "ganvalid_1p00",
            "ganvalidb_1p00",
            "ganvalidc_1p00",
        ),
        ORIGINAL_SCHEDULE,
        STRICT_ROLE,
        STRICT_LINEAGE,
        True,
    ),
    # Neural-family Rule arms.  AE is intentionally excluded by E13.
    SelectionSpec(
        "lstm_retained_primary",
        "family_extension_by_seed.csv",
        "lstm",
        ("real_only", "rule_0p30"),
        ORIGINAL_SCHEDULE,
        RETAINED_ROLE,
        RETAINED_LINEAGE,
    ),
    SelectionSpec(
        "lstm_legacy_rule_plus100",
        "family_extension_by_seed.csv",
        "lstm",
        ("rule_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "lstm_strict_v2_rule_plus100",
        "family_extension_by_seed_sampling_v2_rule_1p00.csv",
        "lstm",
        ("rule_1p00",),
        ORIGINAL_SCHEDULE,
        STRICT_ROLE,
        STRICT_LINEAGE,
    ),
    SelectionSpec(
        "transformer_retained_primary",
        "family_extension_transformer_by_seed.csv",
        "transformer",
        ("real_only", "rule_0p30"),
        ORIGINAL_SCHEDULE,
        RETAINED_ROLE,
        RETAINED_LINEAGE,
    ),
    SelectionSpec(
        "transformer_legacy_rule_plus100",
        "family_extension_transformer_by_seed.csv",
        "transformer",
        ("rule_1p00",),
        ORIGINAL_SCHEDULE,
        LEGACY_ROLE,
        LEGACY_LINEAGE,
    ),
    SelectionSpec(
        "transformer_strict_v2_rule_plus100",
        "family_extension_transformer_by_seed_sampling_v2_rule_1p00.csv",
        "transformer",
        ("rule_1p00",),
        ORIGINAL_SCHEDULE,
        STRICT_ROLE,
        STRICT_LINEAGE,
    ),
)

SECOND_SOURCE_SPEC = SelectionSpec(
    "rf_retained_second_source_focal",
    "second_source_rf_by_seed.csv",
    "rf",
    ("real_only", "rule_0p30"),
    SECOND_SOURCE_SCHEDULE,
    RETAINED_ROLE,
    RETAINED_LINEAGE,
    True,
)


def strict_arm(setting: str) -> str:
    return arm_id(STRICT_ROLE, setting)


def retained_arm(setting: str) -> str:
    return arm_id(RETAINED_ROLE, setting)


def retained_matched_arm(setting: str) -> str:
    return arm_id(RETAINED_MATCHED_ROLE, setting)


def legacy_arm(setting: str) -> str:
    return arm_id(LEGACY_ROLE, setting)


def _general_pair_specs() -> tuple[PairSpec, ...]:
    specs: list[PairSpec] = []
    strict_settings = {
        "cnn": (
            "rule_1p00",
            "ganvalid_1p00",
            "ganvalidb_1p00",
            "ganvalidc_1p00",
        ),
        "rf": (
            "rule_1p00",
            "ganvalid_1p00",
            "ganvalidb_1p00",
            "ganvalidc_1p00",
        ),
        "lstm": ("rule_1p00",),
        "transformer": ("rule_1p00",),
    }
    for family, settings in strict_settings.items():
        for setting in settings:
            specs.append(
                PairSpec(
                    f"{family}_{setting}_strict_v2_vs_real_only",
                    family,
                    ORIGINAL_SCHEDULE,
                    strict_arm(setting),
                    retained_arm("real_only"),
                    "strict_v2_arm_vs_retained_real_only",
                    (
                        "paired_pipeline_seed_contrast; external_results_are_"
                        "interpreted_fpr_and_normal_recall_first"
                    ),
                )
            )
        specs.append(
            PairSpec(
                f"{family}_rule_1p00_strict_v2_vs_rule_0p30",
                family,
                ORIGINAL_SCHEDULE,
                strict_arm("rule_1p00"),
                retained_arm("rule_0p30"),
                "strict_v2_plus100_vs_retained_plus30",
                (
                    "dose_contrast_combines_dose_and_new_pool_realization; "
                    "not_a_replacement_policy_effect"
                ),
            )
        )
    # The equal-added-count real-attack control has only the local/OTIDS rung
    # scope.  Its source-cell subset is explicit in the provenance coverage.
    for setting in (
        "rule_1p00",
        "ganvalid_1p00",
        "ganvalidb_1p00",
        "ganvalidc_1p00",
    ):
        specs.append(
            PairSpec(
                f"cnn_{setting}_strict_v2_vs_over_1p00",
                "cnn",
                ORIGINAL_SCHEDULE,
                strict_arm(setting),
                retained_arm("over_1p00"),
                "strict_v2_synthetic_vs_equal_added_real_attack_control",
                (
                    "joint_content_distribution_contrast; "
                    "not_a_pure_grammar_or_replacement_effect"
                ),
                "b_subset_of_a",
            )
        )
    specs.append(
        PairSpec(
            "cnn_ganvalid_1p00_strict_v2_matched_steps_vs_over_1p00",
            "cnn",
            MATCHED_SCHEDULE,
            strict_arm("ganvalid_1p00"),
            retained_matched_arm("over_1p00"),
            "strict_v2_wgan_vs_equal_added_real_at_matched_optimizer_steps",
            (
                "joint_content_distribution_contrast_at_matched_steps; "
                "not_a_pure_grammar_or_replacement_effect"
            ),
            "b_subset_of_a",
        )
    )
    return tuple(specs)


def _corrective_pair_specs() -> tuple[PairSpec, ...]:
    specs: list[PairSpec] = []
    settings_by_family = {
        "cnn": (
            "rule_1p00",
            "ganvalid_1p00",
            "ganvalidb_1p00",
            "ganvalidc_1p00",
        ),
        "rf": (
            "rule_1p00",
            "ganvalid_1p00",
            "ganvalidb_1p00",
            "ganvalidc_1p00",
        ),
        "lstm": ("rule_1p00",),
        "transformer": ("rule_1p00",),
    }
    for family, settings in settings_by_family.items():
        for setting in settings:
            # The historical b/c generator-seed sensitivity stopped at OTIDS;
            # the strict table additionally has CANTT/ROAD.  Require that the
            # legacy scope is an exact subset instead of silently intersecting.
            policy = (
                "b_subset_of_a"
                if family in {"cnn", "rf"}
                and setting in {"ganvalidb_1p00", "ganvalidc_1p00"}
                else "exact"
            )
            specs.append(
                PairSpec(
                    f"{family}_{setting}_strict_v2_minus_legacy",
                    family,
                    ORIGINAL_SCHEDULE,
                    strict_arm(setting),
                    legacy_arm(setting),
                    JOINT_CORRECTIVE_ESTIMAND,
                    JOINT_CORRECTIVE_BOUNDARY,
                    policy,
                )
            )
    specs.append(
        PairSpec(
            "cnn_ganvalid_1p00_strict_v2_minus_legacy_matched_steps",
            "cnn",
            MATCHED_SCHEDULE,
            strict_arm("ganvalid_1p00"),
            legacy_arm("ganvalid_1p00"),
            JOINT_CORRECTIVE_ESTIMAND,
            JOINT_CORRECTIVE_BOUNDARY,
            "b_subset_of_a",
        )
    )
    return tuple(specs)


FOCAL_SPECS = (
    FocalSpec(
        "K1_cnn_L1_rule_gain",
        "WISA-submission C1 replication at five seeds: retained Rule+30 lifts L1",
        "cnn",
        ORIGINAL_SCHEDULE,
        "L1_fixed_variant",
        "all",
        "attack_recall",
        retained_arm("rule_0p30"),
        retained_arm("real_only"),
        "not_applicable",
    ),
    FocalSpec(
        "K2_cnn_L1_ganvalid_shift_e13",
        (
            "E13 diagnostic: estimate the strict-v2 protocol-valid WGAN L1 "
            "shift; no equivalence margin was prespecified"
        ),
        "cnn",
        ORIGINAL_SCHEDULE,
        "L1_fixed_variant",
        "all",
        "attack_recall",
        strict_arm("ganvalid_1p00"),
        retained_arm("real_only"),
        "protocol_valid_strict_v2",
    ),
    FocalSpec(
        "K3_lstm_L1_rule_gain",
        "Family generality of the retained Rule+30 C1 contrast (BiLSTM)",
        "lstm",
        ORIGINAL_SCHEDULE,
        "L1_fixed_variant",
        "all",
        "attack_recall",
        retained_arm("rule_0p30"),
        retained_arm("real_only"),
        "not_applicable",
    ),
    FocalSpec(
        "K4_rf_otids_fpr_rule_push_e13",
        "E13 strict-v2 Rule+100 shifts non-saturated external OTIDS FPR",
        "rf",
        ORIGINAL_SCHEDULE,
        "L5_otids",
        "all",
        "fpr",
        strict_arm("rule_1p00"),
        retained_arm("real_only"),
        "strict_v2_rule",
    ),
    FocalSpec(
        "K5_rf_otids_fpr_ganvalid_push_e13",
        "E13 strict-v2 protocol-valid WGAN+100 shifts external OTIDS FPR",
        "rf",
        ORIGINAL_SCHEDULE,
        "L5_otids",
        "all",
        "fpr",
        strict_arm("ganvalid_1p00"),
        retained_arm("real_only"),
        "protocol_valid_strict_v2",
    ),
)

SECOND_SOURCE_FOCAL = FocalSpec(
    "K6_ss_rf_L1_rule_gain",
    "C-plan Q1: retained Rule+30 C1 contrast reproduces from the second source",
    "rf",
    SECOND_SOURCE_SCHEDULE,
    "C_L1_fixed_variant",
    "all",
    "attack_recall",
    retained_arm("rule_0p30"),
    retained_arm("real_only"),
    "not_applicable",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _exact_seed_string(seeds: Iterable[int]) -> str:
    return ";".join(str(seed) for seed in sorted(int(seed) for seed in seeds))


def _validate_selected_cells(
    frame: pd.DataFrame,
    spec: SelectionSpec,
    required_metrics: Sequence[str] = METRICS,
) -> None:
    """Reject missing, duplicate, nonfinite, or non-five-seed selected cells."""

    required_columns = {
        "family",
        "setting",
        "seed",
        "rung",
        "subset",
        *SAMPLE_SIZE_COLUMNS,
        *required_metrics,
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{spec.name}: source is missing columns {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError(f"{spec.name}: selected no rows")
    found_families = sorted(frame["family"].astype(str).unique())
    if found_families != [spec.family]:
        raise ValueError(
            f"{spec.name}: expected family {spec.family}, got {found_families}"
        )
    found_settings = set(frame["setting"].astype(str).unique())
    if found_settings != set(spec.settings):
        raise ValueError(
            f"{spec.name}: expected settings {sorted(spec.settings)}, "
            f"got {sorted(found_settings)}"
        )

    numeric_seed = pd.to_numeric(frame["seed"], errors="coerce")
    if numeric_seed.isna().any() or (numeric_seed % 1 != 0).any():
        raise ValueError(f"{spec.name}: seed contains non-integer values")
    frame["seed"] = numeric_seed.astype(int)

    source_row_key = ["family", "setting", "seed", "rung", "subset"]
    duplicated = frame.duplicated(source_row_key, keep=False)
    if duplicated.any():
        examples = (
            frame.loc[duplicated, source_row_key]
            .drop_duplicates()
            .head(8)
            .to_dict("records")
        )
        raise ValueError(
            f"{spec.name}: duplicate selected per-seed rows: {examples}"
        )

    expected = set(EXPECTED_SEEDS)
    bad_cells: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["family", "setting", "rung", "subset"],
        sort=True,
        dropna=False,
    ):
        seeds = set(int(seed) for seed in group["seed"])
        if seeds != expected or len(group) != len(expected):
            bad_cells.append(
                {
                    "cell": tuple(str(value) for value in keys),
                    "seeds": sorted(seeds),
                    "rows": int(len(group)),
                }
            )
    if bad_cells:
        raise ValueError(
            f"{spec.name}: every selected cell requires exactly seeds "
            f"{list(EXPECTED_SEEDS)}; bad cells={bad_cells[:8]}"
        )

    counts = frame[list(SAMPLE_SIZE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(counts.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{spec.name}: selected sample sizes are nonfinite")
    if (counts < 0).any().any():
        raise ValueError(f"{spec.name}: selected sample sizes must be nonnegative")
    # Some diagnostic subsets contain only attack or only normal windows, and
    # some external sets lack exact attack-type labels.  Their inapplicable
    # metrics are intentionally blank.  Missingness must nevertheless be
    # structural and identical across all five seeds in a cell.
    for metric in required_metrics:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if np.isinf(values.to_numpy(dtype=np.float64)).any():
            raise ValueError(f"{spec.name}: {metric} contains infinite values")
        check = frame[["family", "setting", "rung", "subset"]].copy()
        check["finite"] = np.isfinite(values.to_numpy(dtype=np.float64))
        finite_counts = check.groupby(
            ["family", "setting", "rung", "subset"], dropna=False
        )["finite"].sum()
        partial = finite_counts.loc[
            ~finite_counts.isin((0, len(EXPECTED_SEEDS)))
        ]
        if not partial.empty:
            raise ValueError(
                f"{spec.name}: {metric} is defined for only some seeds in "
                f"cells {partial.head(8).to_dict()}"
            )


def load_selection(
    tables_dir: Path,
    spec: SelectionSpec,
    repo_root: Path,
    cache: dict[Path, tuple[pd.DataFrame, str]] | None = None,
    required_metrics: Sequence[str] = METRICS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and annotate one declared source selection."""

    cache = {} if cache is None else cache
    path = (tables_dir / spec.file_name).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{spec.name}: missing required source {path}")
    if path not in cache:
        cache[path] = (pd.read_csv(path), sha256_file(path))
    source, source_sha256 = cache[path]
    selected = source.loc[
        source["family"].astype(str).eq(spec.family)
        & source["setting"].astype(str).isin(spec.settings)
    ].copy()
    if spec.allow_missing_exact_recall and "exact_recall" not in selected:
        selected["exact_recall"] = np.nan
    _validate_selected_cells(selected, spec, required_metrics)

    selected["training_schedule"] = spec.training_schedule
    selected["analysis_role"] = spec.analysis_role
    selected["sampling_lineage"] = spec.sampling_lineage
    selected["analysis_arm"] = selected["setting"].map(
        lambda setting: arm_id(spec.analysis_role, str(setting))
    )
    selected["source_selection"] = spec.name
    selected["source_file"] = display_path(path, repo_root)
    selected["source_sha256"] = source_sha256
    selected["analysis_version"] = ""

    record = {
        **asdict(spec),
        "path": display_path(path, repo_root),
        "sha256": source_sha256,
        "bytes": path.stat().st_size,
        "selected_rows": int(len(selected)),
        "selected_cells": int(
            selected.groupby(["family", "setting", "rung", "subset"]).ngroups
        ),
    }
    return selected, record


def build_merged_by_seed(
    tables_dir: Path,
    repo_root: Path,
    version: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[Path, tuple[pd.DataFrame, str]]]:
    """Build the lineage-explicit unified E13 per-seed table."""

    cache: dict[Path, tuple[pd.DataFrame, str]] = {}
    pieces: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for spec in SELECTION_SPECS:
        selected, record = load_selection(tables_dir, spec, repo_root, cache)
        selected["analysis_version"] = version
        pieces.append(selected)
        records.append(record)
    merged = pd.concat(pieces, ignore_index=True, sort=False)

    duplicated = merged.duplicated(list(ROW_KEY), keep=False)
    if duplicated.any():
        examples = (
            merged.loc[duplicated, list(ROW_KEY) + ["source_file"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "merged table has duplicate lineage-aware per-seed rows: "
            f"{examples}"
        )
    # Keep the evaluator schema first, then make lineage unavoidable.
    lineage_columns = [
        "analysis_version",
        "training_schedule",
        "analysis_arm",
        "analysis_role",
        "sampling_lineage",
        "source_selection",
        "source_file",
        "source_sha256",
    ]
    original_columns = [
        column for column in pieces[0].columns if column not in lineage_columns
    ]
    merged = merged[original_columns + lineage_columns].sort_values(
        list(ROW_KEY), kind="stable"
    )
    return merged.reset_index(drop=True), records, cache


def build_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Report arm-level means and sample SDs over the five pipeline seeds."""

    grouping = [
        "analysis_version",
        "family",
        "training_schedule",
        "analysis_arm",
        "setting",
        "analysis_role",
        "sampling_lineage",
        "rung",
        "subset",
        "source_selection",
        "source_file",
        "source_sha256",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in merged.groupby(grouping, sort=True, dropna=False):
        seeds = sorted(int(seed) for seed in group["seed"])
        if tuple(seeds) != EXPECTED_SEEDS or len(group) != len(EXPECTED_SEEDS):
            raise ValueError(
                f"summary cell {keys}: expected exactly {list(EXPECTED_SEEDS)}, "
                f"got {seeds}"
            )
        row = dict(zip(grouping, keys))
        row["n_seeds"] = len(seeds)
        row["seeds"] = _exact_seed_string(seeds)
        for column in SAMPLE_SIZE_COLUMNS:
            values = pd.to_numeric(group[column], errors="raise").to_numpy()
            if len(np.unique(values)) != 1:
                raise ValueError(
                    f"summary cell {keys}: {column} changes across seeds"
                )
            row[column] = int(values[0])
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(
                dtype=np.float64
            )
            finite = np.isfinite(values)
            if finite.all():
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            elif not finite.any():
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
            else:
                raise ValueError(
                    f"summary cell {keys}: {metric} is defined for only "
                    "some seeds"
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["family", "training_schedule", "analysis_arm", "rung", "subset"],
        kind="stable",
    ).reset_index(drop=True)


def paired_seed_statistics(
    values_a: Sequence[float],
    values_b: Sequence[float],
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> dict[str, object]:
    """Return five-seed paired statistics with an untruncated Student-t CI."""

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("paired arrays must be one-dimensional and equal length")
    if len(a) != len(seeds) or tuple(int(seed) for seed in seeds) != EXPECTED_SEEDS:
        raise ValueError(
            f"paired statistics require exact seeds {list(EXPECTED_SEEDS)}"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired values must be finite")
    delta = a - b
    delta_mean = float(delta.mean())
    delta_std = float(delta.std(ddof=1))
    half_width = float(
        student_t.ppf(0.975, len(delta) - 1)
        * delta_std
        / math.sqrt(len(delta))
    )
    ci_lo = delta_mean - half_width
    ci_hi = delta_mean + half_width
    if ci_lo > 0:
        decision = "positive_effect_supported"
    elif ci_hi < 0:
        decision = "negative_effect_supported"
    else:
        decision = "inconclusive_not_equivalent"
    if delta_std == 0.0:
        if delta_mean == 0.0:
            t_statistic, two_sided_p = 0.0, 1.0
        else:
            t_statistic, two_sided_p = math.copysign(math.inf, delta_mean), 0.0
    else:
        t_statistic = delta_mean / (delta_std / math.sqrt(len(delta)))
        two_sided_p = float(
            2.0 * student_t.sf(abs(t_statistic), len(delta) - 1)
        )
    seed_delta = {
        str(int(seed)): float(value) for seed, value in zip(seeds, delta)
    }
    same_sign = max(
        int((delta > 0).sum()),
        int((delta < 0).sum()),
        int((delta == 0).sum()),
    )
    return {
        "n_paired_seeds": len(delta),
        "paired_seeds": _exact_seed_string(seeds),
        "mean_a": float(a.mean()),
        "std_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()),
        "std_b": float(b.std(ddof=1)),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "primary_seed_t_ci_lo": float(ci_lo),
        "primary_seed_t_ci_hi": float(ci_hi),
        "primary_decision": decision,
        "positive_seeds": int((delta > 0).sum()),
        "negative_seeds": int((delta < 0).sum()),
        "zero_seeds": int((delta == 0).sum()),
        "sign_consistency": f"{same_sign}/{len(delta)}",
        "t_statistic": float(t_statistic),
        "two_sided_seed_t_p": two_sided_p,
        "paired_seed_deltas_json": json.dumps(
            seed_delta, sort_keys=True, separators=(",", ":")
        ),
    }


def _arm_rows(
    merged: pd.DataFrame,
    family: str,
    schedule: str,
    analysis_arm: str,
) -> pd.DataFrame:
    rows = merged.loc[
        merged["family"].eq(family)
        & merged["training_schedule"].eq(schedule)
        & merged["analysis_arm"].eq(analysis_arm)
    ].copy()
    if rows.empty:
        raise ValueError(
            f"missing comparison arm {family}/{schedule}/{analysis_arm}"
        )
    return rows


def _cell_set(rows: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(rung), str(subset))
        for rung, subset in rows[["rung", "subset"]].itertuples(index=False)
    }


def _validate_cell_policy(
    spec: PairSpec,
    cells_a: set[tuple[str, str]],
    cells_b: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    if spec.cell_policy == "exact":
        valid = cells_a == cells_b
    elif spec.cell_policy == "b_subset_of_a":
        valid = cells_b.issubset(cells_a)
    else:
        raise ValueError(
            f"{spec.comparison}: unknown cell policy {spec.cell_policy}"
        )
    if not valid:
        raise ValueError(
            f"{spec.comparison}: cell policy {spec.cell_policy} failed; "
            f"A-only={sorted(cells_a - cells_b)}, "
            f"B-only={sorted(cells_b - cells_a)}"
        )
    return cells_a & cells_b


def _pair_one_cell(
    rows_a: pd.DataFrame,
    rows_b: pd.DataFrame,
    metric: str,
    comparison_fields: Mapping[str, object],
) -> dict[str, object]:
    """Pair one already selected family/schedule/rung/subset cell."""

    if metric not in METRICS and metric not in rows_a:
        raise ValueError(f"missing requested metric {metric}")
    duplicate_key = ["seed"]
    if rows_a.duplicated(duplicate_key).any() or rows_b.duplicated(
        duplicate_key
    ).any():
        raise ValueError(f"{comparison_fields}: duplicate seed in paired cell")
    a = rows_a.set_index("seed").sort_index()
    b = rows_b.set_index("seed").sort_index()
    if tuple(int(seed) for seed in a.index) != EXPECTED_SEEDS or tuple(
        int(seed) for seed in b.index
    ) != EXPECTED_SEEDS:
        raise ValueError(
            f"{comparison_fields}: paired cell requires exact seeds "
            f"{list(EXPECTED_SEEDS)}; A={a.index.tolist()} B={b.index.tolist()}"
        )
    for column in SAMPLE_SIZE_COLUMNS:
        values_a = pd.to_numeric(a[column], errors="raise").to_numpy()
        values_b = pd.to_numeric(b[column], errors="raise").to_numpy()
        if not np.array_equal(values_a, values_b):
            raise ValueError(
                f"{comparison_fields}: paired cell {column} differs by arm"
            )

    result = dict(comparison_fields)
    result.update(
        {
            "arm_a_setting": str(a["setting"].iloc[0]),
            "arm_b_setting": str(b["setting"].iloc[0]),
            "arm_a_analysis_role": str(a["analysis_role"].iloc[0]),
            "arm_b_analysis_role": str(b["analysis_role"].iloc[0]),
            "arm_a_sampling_lineage": str(a["sampling_lineage"].iloc[0]),
            "arm_b_sampling_lineage": str(b["sampling_lineage"].iloc[0]),
            "arm_a_source_file": str(a["source_file"].iloc[0]),
            "arm_b_source_file": str(b["source_file"].iloc[0]),
            "arm_a_source_sha256": str(a["source_sha256"].iloc[0]),
            "arm_b_source_sha256": str(b["source_sha256"].iloc[0]),
            "windows": int(a["windows"].iloc[0]),
            "normal_windows": int(a["normal_windows"].iloc[0]),
            "attack_windows": int(a["attack_windows"].iloc[0]),
        }
    )
    result.update(
        paired_seed_statistics(
            pd.to_numeric(a[metric], errors="raise").to_numpy(),
            pd.to_numeric(b[metric], errors="raise").to_numpy(),
            tuple(int(seed) for seed in a.index),
        )
    )
    return result


def build_pair_table(
    merged: pd.DataFrame,
    specs: Sequence[PairSpec],
    metrics: Sequence[str] = METRICS,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Build declared paired rows and explicit cell-coverage provenance."""

    rows: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for spec in specs:
        arm_a = _arm_rows(
            merged, spec.family, spec.training_schedule, spec.arm_a
        )
        arm_b = _arm_rows(
            merged, spec.family, spec.training_schedule, spec.arm_b
        )
        cells_a = _cell_set(arm_a)
        cells_b = _cell_set(arm_b)
        paired_cells = _validate_cell_policy(spec, cells_a, cells_b)
        coverage.append(
            {
                **asdict(spec),
                "arm_a_cells": len(cells_a),
                "arm_b_cells": len(cells_b),
                "paired_cells": len(paired_cells),
                "arm_a_only_cells": [
                    {"rung": rung, "subset": subset}
                    for rung, subset in sorted(cells_a - cells_b)
                ],
                "arm_b_only_cells": [
                    {"rung": rung, "subset": subset}
                    for rung, subset in sorted(cells_b - cells_a)
                ],
            }
        )
        for rung, subset in sorted(paired_cells):
            a_cell = arm_a.loc[
                arm_a["rung"].astype(str).eq(rung)
                & arm_a["subset"].astype(str).eq(subset)
            ]
            b_cell = arm_b.loc[
                arm_b["rung"].astype(str).eq(rung)
                & arm_b["subset"].astype(str).eq(subset)
            ]
            for metric in metrics:
                finite_a = np.isfinite(
                    pd.to_numeric(a_cell[metric], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                )
                finite_b = np.isfinite(
                    pd.to_numeric(b_cell[metric], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                )
                if not finite_a.any() and not finite_b.any():
                    continue
                if not finite_a.all() or not finite_b.all():
                    raise ValueError(
                        f"{spec.comparison}/{rung}/{subset}/{metric}: metric "
                        "availability differs by seed or arm"
                    )
                rows.append(
                    _pair_one_cell(
                        a_cell,
                        b_cell,
                        metric,
                        {
                            "comparison": spec.comparison,
                            "family": spec.family,
                            "training_schedule": spec.training_schedule,
                            "rung": rung,
                            "subset": subset,
                            "metric": metric,
                            "arm_a": spec.arm_a,
                            "arm_b": spec.arm_b,
                            "estimand": spec.estimand,
                            "interpretation_boundary": (
                                spec.interpretation_boundary
                            ),
                            "cell_policy": spec.cell_policy,
                            "arm_a_cells": len(cells_a),
                            "arm_b_cells": len(cells_b),
                            "paired_cells": len(paired_cells),
                        },
                    )
                )
    table = pd.DataFrame(rows)
    analysis_versions = sorted(merged["analysis_version"].astype(str).unique())
    if len(analysis_versions) != 1:
        raise ValueError(
            f"paired analysis requires one version, got {analysis_versions}"
        )
    table.insert(0, "analysis_version", analysis_versions[0])
    duplicate_key = [
        "comparison",
        "family",
        "training_schedule",
        "rung",
        "subset",
        "metric",
    ]
    if table.duplicated(duplicate_key).any():
        raise ValueError("paired output contains duplicate comparison cells")
    return (
        table.sort_values(duplicate_key, kind="stable").reset_index(drop=True),
        coverage,
    )


def _focal_row(merged: pd.DataFrame, spec: FocalSpec) -> dict[str, object]:
    arm_a = _arm_rows(
        merged, spec.family, spec.training_schedule, spec.arm_a
    )
    arm_b = _arm_rows(
        merged, spec.family, spec.training_schedule, spec.arm_b
    )
    a_cell = arm_a.loc[
        arm_a["rung"].eq(spec.rung) & arm_a["subset"].eq(spec.subset)
    ]
    b_cell = arm_b.loc[
        arm_b["rung"].eq(spec.rung) & arm_b["subset"].eq(spec.subset)
    ]
    if a_cell.empty or b_cell.empty:
        raise ValueError(
            f"{spec.comparison}: missing focal cell {spec.rung}/{spec.subset}"
        )
    versions = sorted(merged["analysis_version"].astype(str).unique())
    if len(versions) != 1:
        raise ValueError(
            f"{spec.comparison}: focal source has versions {versions}"
        )
    result = _pair_one_cell(
        a_cell,
        b_cell,
        spec.metric,
        {
            "analysis_version": versions[0],
            "comparison": spec.comparison,
            "hypothesis": spec.hypothesis,
            "analysis_role": "focal_primary",
            "protocol_status": spec.protocol_status,
            "family": spec.family,
            "training_schedule": spec.training_schedule,
            "rung": spec.rung,
            "subset": spec.subset,
            "metric": spec.metric,
            "arm_a": spec.arm_a,
            "arm_b": spec.arm_b,
            "estimand": "paired_pipeline_seed_effect",
            "interpretation_boundary": (
                "seed_t_interval_is_primary; no_equivalence_margin_was_"
                "prespecified; Holm_is_supporting_only"
            ),
            "cell_policy": "single_prespecified_cell",
        },
    )
    result["n_seeds"] = result["n_paired_seeds"]
    return result


def build_focal_table(
    merged: pd.DataFrame,
    second_source: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the existing six-comparison supporting family with E13 arms."""

    rows = [_focal_row(merged, spec) for spec in FOCAL_SPECS]
    rows.append(_focal_row(second_source, SECOND_SOURCE_FOCAL))
    result = pd.DataFrame(rows)
    if set(result["comparison"]) != {
        spec.comparison for spec in (*FOCAL_SPECS, SECOND_SOURCE_FOCAL)
    }:
        raise ValueError("focal comparison family is incomplete")
    if len(result) != 6 or result["comparison"].duplicated().any():
        raise ValueError("focal comparison family must contain exactly six rows")
    return result.sort_values("comparison", kind="stable").reset_index(drop=True)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return stable, monotone Holm-adjusted p-values in original row order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must be finite and inside [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, float((len(values) - rank) * values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def build_holm_table(
    focal: pd.DataFrame,
    focal_file: str,
    focal_sha256: str,
) -> pd.DataFrame:
    if len(focal) != 6 or not focal["analysis_role"].eq(
        "focal_primary"
    ).all():
        raise ValueError("supporting Holm input must be the six focal rows")
    result = focal[
        [
            "analysis_version",
            "comparison",
            "hypothesis",
            "family",
            "training_schedule",
            "rung",
            "subset",
            "metric",
            "arm_a",
            "arm_b",
            "n_paired_seeds",
            "n_seeds",
            "delta_mean",
            "delta_std",
            "primary_seed_t_ci_lo",
            "primary_seed_t_ci_hi",
            "sign_consistency",
            "t_statistic",
            "two_sided_seed_t_p",
            "primary_decision",
            "arm_a_source_file",
            "arm_a_source_sha256",
            "arm_a_sampling_lineage",
            "arm_b_source_file",
            "arm_b_source_sha256",
            "arm_b_sampling_lineage",
        ]
    ].copy()
    result["holm_adjusted_p"] = holm_adjust(
        result["two_sided_seed_t_p"].to_numpy()
    )
    result["holm_reject_alpha_0p05"] = result["holm_adjusted_p"] < 0.05
    result["multiplicity_family"] = (
        "six_declared_focal_seed_level_comparisons"
    )
    result["multiplicity_role"] = (
        "supporting_sensitivity; paired_seed_t_intervals_remain_primary"
    )
    result["focal_input_file"] = focal_file
    result["focal_input_sha256"] = focal_sha256
    return result


def dataframe_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _git_output(
    repo_root: Path, *args: str, allow_failure: bool = False
) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=not allow_failure,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    # Preserve the status column's leading space in porcelain output.  A
    # generic strip() changes a first-line worktree deletion from `` D`` to
    # ``D `` and makes the path parser drop the first filename character.
    return completed.stdout.rstrip("\r\n")


def source_state(repo_root: Path) -> dict[str, object]:
    commit = _git_output(repo_root, "rev-parse", "HEAD", allow_failure=True)
    status_text = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        allow_failure=True,
    )
    status = status_text.splitlines() if status_text else []
    artifact_prefixes = (
        "journal/results/",
        "journal/models/",
        "datasets/synthetic/",
    )
    source_definition_status: list[str] = []
    artifact_status: list[str] = []
    for line in status:
        path = line[3:] if len(line) >= 4 else line
        # Rename records have "old -> new"; neither side is an allowed output
        # unless the entire record is under an artifact root.
        if path.startswith(artifact_prefixes):
            artifact_status.append(line)
        else:
            source_definition_status.append(line)
    return {
        "commit": commit or None,
        "source_definition_clean": not source_definition_status,
        "source_definition_status": source_definition_status,
        "artifact_status_count": len(artifact_status),
        "artifact_status_sha256": sha256_bytes(
            "\n".join(artifact_status).encode("utf-8")
        ),
    }


def _output_paths(
    tables_dir: Path, logs_dir: Path, version: str
) -> dict[str, Path]:
    stem = f"e13_sampling_v2"
    return {
        "by_seed": tables_dir / f"{stem}_by_seed_{version}.csv",
        "summary": tables_dir / f"{stem}_summary_{version}.csv",
        "paired": tables_dir / f"{stem}_paired_{version}.csv",
        "legacy_joint_corrective": (
            tables_dir / f"{stem}_legacy_joint_corrective_{version}.csv"
        ),
        "focal_comparisons": (
            tables_dir / f"{stem}_focal_comparisons_{version}.csv"
        ),
        "supporting_holm": (
            tables_dir / f"{stem}_supporting_holm_{version}.csv"
        ),
        "provenance": (
            logs_dir / f"{stem}_analysis_provenance_{version}.json"
        ),
    }


def publish_outputs_atomic(
    output_bytes: Mapping[Path, bytes],
    commit_marker: Path,
) -> None:
    """Publish no-clobber files from one staging area; marker is linked last.

    Each destination appears atomically through ``link(2)`` and can never
    replace an existing file.  The provenance JSON is the transaction commit
    marker.  A caught publication failure removes files linked by this call.
    Consumers must require the marker, which is never visible before all data
    tables.
    """

    paths = list(output_bytes)
    if commit_marker not in output_bytes:
        raise ValueError("commit marker must be one of the staged outputs")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate output destinations")
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing E13 output(s): "
            + ", ".join(str(path) for path in existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    common_parent = Path(
        os.path.commonpath([str(path.parent.resolve()) for path in paths])
    )
    publish_order = [path for path in paths if path != commit_marker] + [
        commit_marker
    ]
    linked: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=".e13_sampling_v2_analysis.", dir=common_parent
    ) as temporary:
        staging = Path(temporary)
        staged: dict[Path, Path] = {}
        for index, destination in enumerate(publish_order):
            stage_path = staging / f"{index:02d}.stage"
            with stage_path.open("xb") as handle:
                handle.write(output_bytes[destination])
                handle.flush()
                os.fsync(handle.fileno())
            staged[destination] = stage_path
        try:
            for destination in publish_order:
                os.link(staged[destination], destination)
                linked.append(destination)
        except BaseException:
            for destination in reversed(linked):
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            raise


def build_outputs(
    tables_dir: Path,
    logs_dir: Path,
    prereg_path: Path,
    repo_root: Path,
    version: str,
    require_clean_source: bool = False,
    input_tables_dir: Path | None = None,
) -> dict[str, Path]:
    """Build, validate, serialize, and publish the complete E13 analysis."""

    input_tables_dir = (
        tables_dir if input_tables_dir is None else input_tables_dir
    )
    if not SAFE_VERSION.fullmatch(version):
        raise ValueError(
            "version must match [A-Za-z0-9][A-Za-z0-9_.-]*"
        )
    output_paths = _output_paths(tables_dir, logs_dir, version)
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing E13 output(s): "
            + ", ".join(str(path) for path in existing)
        )
    if not prereg_path.is_file():
        raise FileNotFoundError(f"missing E13 prospective record: {prereg_path}")
    prereg_sha256 = sha256_file(prereg_path)
    prereg_text = prereg_path.read_text(encoding="utf-8")
    required_prereg_phrases = (
        "strict no-replacement sampling v2",
        "pool-reinstantiation plus no-replacement corrective sensitivity",
        "7, 42, 123, 2026, 3407",
    )
    missing_phrases = [
        phrase for phrase in required_prereg_phrases if phrase not in prereg_text
    ]
    if missing_phrases:
        raise ValueError(
            f"E13 prospective record is missing fixed boundary text: "
            f"{missing_phrases}"
        )

    state = source_state(repo_root)
    if require_clean_source and not state["source_definition_clean"]:
        raise RuntimeError(
            "source-definition tree is not clean: "
            f"{state['source_definition_status']}"
        )

    merged, input_records, cache = build_merged_by_seed(
        input_tables_dir, repo_root, version
    )
    summary = build_summary(merged)
    paired, paired_coverage = build_pair_table(
        merged, _general_pair_specs()
    )
    corrective, corrective_coverage = build_pair_table(
        merged, _corrective_pair_specs()
    )
    corrective.insert(
        1,
        "corrective_sensitivity_label",
        JOINT_CORRECTIVE_ESTIMAND,
    )

    second_source, second_source_record = load_selection(
        input_tables_dir,
        SECOND_SOURCE_SPEC,
        repo_root,
        cache,
        required_metrics=("attack_recall",),
    )
    second_source["analysis_version"] = version
    focal = build_focal_table(merged, second_source)

    csv_frames = {
        "by_seed": merged,
        "summary": summary,
        "paired": paired,
        "legacy_joint_corrective": corrective,
        "focal_comparisons": focal,
    }
    serialized: dict[str, bytes] = {
        name: dataframe_bytes(frame) for name, frame in csv_frames.items()
    }
    focal_display_path = display_path(
        output_paths["focal_comparisons"], repo_root
    )
    holm = build_holm_table(
        focal,
        focal_display_path,
        sha256_bytes(serialized["focal_comparisons"]),
    )
    csv_frames["supporting_holm"] = holm
    serialized["supporting_holm"] = dataframe_bytes(holm)

    unique_input_records: dict[str, dict[str, object]] = {}
    for record in [*input_records, second_source_record]:
        path = str(record["path"])
        target = unique_input_records.setdefault(
            path,
            {
                "path": path,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "selections": [],
            },
        )
        if target["sha256"] != record["sha256"]:
            raise RuntimeError(f"source changed while building analysis: {path}")
        target["selections"].append(
            {
                key: record[key]
                for key in (
                    "name",
                    "family",
                    "settings",
                    "training_schedule",
                    "analysis_role",
                    "sampling_lineage",
                    "allow_missing_exact_recall",
                    "selected_rows",
                    "selected_cells",
                )
            }
        )

    # Re-hash every immutable source immediately before serialization.  A
    # concurrent writer must not leave a table whose recorded digest describes
    # bytes different from those used to compute it.
    for path, (_frame, initial_sha256) in cache.items():
        final_sha256 = sha256_file(path)
        if final_sha256 != initial_sha256:
            raise RuntimeError(
                f"source changed while building analysis: "
                f"{display_path(path, repo_root)}"
            )
    if sha256_file(prereg_path) != prereg_sha256:
        raise RuntimeError("E13 prospective record changed while building")

    output_records = {}
    for name, frame in csv_frames.items():
        data = serialized[name]
        output_records[name] = {
            "path": display_path(output_paths[name], repo_root),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

    provenance = {
        "schema_version": "e13_sampling_v2_analysis_provenance_v1",
        "analysis_version": version,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_state": state,
        "analysis_script": {
            "path": display_path(SCRIPT_PATH, repo_root),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "prospective_record": {
            "path": display_path(prereg_path, repo_root),
            "sha256": prereg_sha256,
        },
        "expected_pipeline_seeds": list(EXPECTED_SEEDS),
        "fixed_exclusions": [
            "conv_ae_has_no_plus100_arm_and_is_not_merged",
            "arlm_has_no_plus100_arm_and_is_not_rerun",
            "all_plus30_arms_are_retained_not_retrained",
        ],
        "statistics": {
            "inferential_unit": "detector_training_pipeline_seed",
            "arm_summary": "arithmetic_mean_and_sample_sd_ddof_1",
            "paired_interval": (
                "two_sided_untruncated_95_percent_Student_t_df_4"
            ),
            "supporting_multiplicity": (
                "Holm_step_down_over_the_existing_six_focal_seed_level_"
                "comparison_family"
            ),
            "primary_vs_supporting": (
                "paired_seed_t_intervals_are_primary; Holm_is_supporting_only"
            ),
        },
        "corrective_sensitivity": {
            "estimand": JOINT_CORRECTIVE_ESTIMAND,
            "interpretation_boundary": JOINT_CORRECTIVE_BOUNDARY,
            "legacy_plus100_rows_retained": True,
        },
        "input_sources": sorted(
            unique_input_records.values(), key=lambda record: str(record["path"])
        ),
        "pairing_coverage": {
            "paired": paired_coverage,
            "legacy_joint_corrective": corrective_coverage,
        },
        "outputs": output_records,
        "publication": {
            "no_clobber": True,
            "method": "staged_same_filesystem_hard_links",
            "commit_marker": display_path(
                output_paths["provenance"], repo_root
            ),
            "commit_marker_published_last": True,
        },
        "software": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    provenance_bytes = (
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    publish_map = {
        output_paths[name]: data for name, data in serialized.items()
    }
    publish_map[output_paths["provenance"]] = provenance_bytes
    publish_outputs_atomic(publish_map, output_paths["provenance"])
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-tables-dir",
        type=Path,
        default=DEFAULT_TABLES,
        help="directory containing immutable evaluator/retained-control CSVs",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES,
        help="destination directory for new versioned analysis CSVs",
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--version",
        default="v1",
        help="safe filename version token; every output refuses overwrite",
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help=(
            "require no Git-visible source-definition changes outside the "
            "declared artifact roots"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_outputs(
        args.tables_dir.resolve(),
        args.logs_dir.resolve(),
        args.prereg.resolve(),
        args.repo_root.resolve(),
        args.version,
        args.require_clean_source,
        args.input_tables_dir.resolve(),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
