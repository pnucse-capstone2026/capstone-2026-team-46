#!/usr/bin/env python3
"""Build the E13 strict-v2 figure set from an explicit versioned input map.

This driver deliberately does not replace the legacy paper figures.  It
stages every companion in an isolated directory, validates the complete
five-seed input matrix, records input and output SHA-256 values, and then
publishes versioned companions with no-clobber hard links.

The E13 family/ladder/external views contain the four supervised detector
families only.  They exclude the legacy Conv-AE threshold rows, as fixed in
the E13 prospective record.  The learned-generator view retains the
protocol-valid WGAN/AR-LM +30% rows as explicit historical context while
replacing only the WGAN +100% row with the strict-v2 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import lib_common as lc
import make_external_operating_figure as external_figure
import make_family_extension_figure as family_figure
import make_focal_forest_figure as focal_figure
import make_generator_protocol_valid_figure as generator_figure
import make_headline_results_figure as headline_figure
import make_ladder_overview_figure as ladder_figure


SCRIPT = Path(__file__).resolve()
JOURNAL_ROOT = SCRIPT.parents[1]
REPO_ROOT = JOURNAL_ROOT.parent
DEFAULT_INPUT_MAP = (
    JOURNAL_ROOT
    / "experiments"
    / "e13_strict_v2_sampling"
    / "figure_input_map_v1.json"
)
DEFAULT_OUTPUT_TAG = "e13_sampling_v2"
EXPECTED_SEEDS = "7;42;123;2026;3407"
SUPERVISED_FAMILIES = ("cnn", "rf", "lstm", "transformer")
MAIN_ARMS = ("real_only", "rule_0p30", "rule_1p00")
MAIN_RUNGS = (
    "L1_fixed_variant",
    "L2_sensitivity",
    "L4_out_of_generator",
    "L5_otids",
    "L5_cantt",
    "L5_road",
)
REQUIRED_INPUTS = {
    "analysis_provenance",
    "analysis_summary",
    "analysis_supporting_holm",
    "arlm_cnn_retained",
    "arlm_rf_retained",
    "gan_cnn_retained",
    "gan_rf_retained",
    "placebo_paired_retained",
    "placebo_summary_retained",
    "realization_paired_e13",
}
OUTPUT_STEMS = (
    "headline_shift_ladder_results",
    "ladder_overview_heatmap",
    "external_operating_points",
    "generator_protocol_valid_ladder",
    "family_extension_ladder",
    "focal_forest_ladder",
)
E13_TO_RENDERER_FOCAL_IDS = {
    "K2_cnn_L1_ganvalid_shift_e13": "K2_cnn_L1_ganvalid_shift",
    "K4_rf_otids_fpr_rule_push_e13": "K4_rf_otids_fpr_rule_push",
    "K5_rf_otids_fpr_ganvalid_push_e13": (
        "K5_rf_otids_fpr_ganvalid_push"
    ),
}
RENDERER_TO_E13_FOCAL_IDS = {
    value: key for key, value in E13_TO_RENDERER_FOCAL_IDS.items()
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def root_relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def validate_output_tag(tag: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", tag):
        raise ValueError(
            "output tag must contain only lower-case letters, digits, and "
            "single underscore separators"
        )
    if "sampling_v2" not in tag:
        raise ValueError("E13 figure output tag must include 'sampling_v2'")
    return tag


def load_input_map(
    path: Path,
    *,
    input_root: Path = REPO_ROOT,
) -> tuple[dict, dict[str, Path]]:
    path = path.resolve()
    input_root = input_root.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.is_relative_to(JOURNAL_ROOT.resolve()):
        raise ValueError(f"input map must live under journal/: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported E13 figure input-map schema")
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, dict):
        raise ValueError("input map must contain an inputs object")
    missing = sorted(REQUIRED_INPUTS.difference(raw_inputs))
    extra = sorted(set(raw_inputs).difference(REQUIRED_INPUTS))
    if missing or extra:
        raise ValueError(
            f"input-map keys do not match the fixed E13 set: "
            f"missing={missing}, extra={extra}"
        )

    resolved: dict[str, Path] = {}
    journal = JOURNAL_ROOT.resolve()
    for key, raw in raw_inputs.items():
        candidate = (input_root / str(raw)).resolve()
        # Preview analysis tables can live in a mirror under --input-root,
        # while retained legacy context remains read-only in the checkout.
        # The exact path selected for every key is hashed into provenance.
        if not candidate.is_file() and input_root != REPO_ROOT.resolve():
            candidate = (REPO_ROOT / str(raw)).resolve()
        allowed_preview = candidate.is_relative_to(input_root)
        allowed_checkout = candidate.is_relative_to(journal)
        if not (allowed_preview or allowed_checkout):
            raise ValueError(
                f"{key}: input escapes the preview root and journal/: "
                f"{candidate}"
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"{key}: {candidate}")
        resolved[key] = candidate
    return payload, resolved


def _original_schedule_mask(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower()
    return values.str.startswith("original")


def load_canonical_summary(
    path: Path,
    expected_analysis_version: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "analysis_version",
        "analysis_role",
        "training_schedule",
        "family",
        "setting",
        "rung",
        "subset",
        "seeds",
        "attack_recall_mean",
        "attack_recall_std",
        "fpr_mean",
        "fpr_std",
        "exact_recall_mean",
        "exact_recall_std",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing E13 summary columns: {missing}")

    original = frame[_original_schedule_mask(frame["training_schedule"])].copy()
    legacy_plus100 = original["analysis_role"].astype(str).str.contains(
        "legacy_plus100", case=False, na=False
    )
    canonical = original[~legacy_plus100].copy()
    if canonical.empty:
        raise ValueError("E13 summary has no canonical original-schedule rows")
    versions = set(canonical["analysis_version"].astype(str))
    if len(versions) != 1:
        raise ValueError(f"unexpected E13 analysis versions: {sorted(versions)}")
    if (
        expected_analysis_version is not None
        and versions != {expected_analysis_version}
    ):
        raise ValueError(
            "E13 summary analysis version does not match the input map: "
            f"map={expected_analysis_version!r}, table={sorted(versions)}"
        )

    keys = ["family", "setting", "rung", "subset"]
    duplicated = canonical[canonical.duplicated(keys, keep=False)]
    if not duplicated.empty:
        examples = duplicated[keys + ["analysis_role"]].head(8).to_dict("records")
        raise ValueError(f"ambiguous canonical E13 summary rows: {examples}")
    return canonical


def validate_main_matrix(frame: pd.DataFrame) -> None:
    missing = []
    for family in SUPERVISED_FAMILIES:
        for setting in MAIN_ARMS:
            for rung in MAIN_RUNGS:
                rows = frame[
                    (frame["family"] == family)
                    & (frame["setting"] == setting)
                    & (frame["rung"] == rung)
                    & (frame["subset"] == "all")
                ]
                if len(rows) != 1:
                    missing.append(
                        f"{family}/{setting}/{rung}/all ({len(rows)} rows)"
                    )
                elif str(rows.iloc[0]["seeds"]) != EXPECTED_SEEDS:
                    raise ValueError(
                        f"{family}/{setting}/{rung}: expected seeds "
                        f"{EXPECTED_SEEDS}, found {rows.iloc[0]['seeds']}"
                    )
    for setting in MAIN_ARMS:
        for severity in ("low", "medium", "high"):
            rows = frame[
                (frame["family"] == "cnn")
                & (frame["setting"] == setting)
                & (frame["rung"] == "L2_sensitivity")
                & (frame["subset"] == f"severity:{severity}")
            ]
            if len(rows) != 1:
                missing.append(
                    f"cnn/{setting}/L2_sensitivity/severity:{severity} "
                    f"({len(rows)} rows)"
                )
    for family in ("cnn", "rf"):
        for rung in ("L1_fixed_variant", "L2_sensitivity", "L5_otids"):
            rows = frame[
                (frame["family"] == family)
                & (frame["setting"] == "ganvalid_1p00")
                & (frame["rung"] == rung)
                & (frame["subset"] == "all")
            ]
            if len(rows) != 1:
                missing.append(
                    f"{family}/ganvalid_1p00/{rung}/all ({len(rows)} rows)"
                )
    if missing:
        raise ValueError(
            "incomplete canonical E13 figure matrix: " + "; ".join(missing[:20])
        )
    if (frame["family"] == "ae").any():
        raise ValueError("canonical E13 figure matrix must not contain Conv-AE rows")


def validate_generator_figure_matrix(frame: pd.DataFrame) -> None:
    missing = []
    for family in ("cnn", "rf"):
        for setting in generator_figure.SETTINGS:
            for rung in (
                "L1_fixed_variant",
                "L2_sensitivity",
                "L5_otids",
            ):
                rows = frame[
                    (frame["family"] == family)
                    & (frame["setting"] == setting)
                    & (frame["rung"] == rung)
                    & (frame["subset"] == "all")
                ]
                if len(rows) != 1:
                    missing.append(
                        f"{family}/{setting}/{rung} ({len(rows)} rows)"
                    )
    if missing:
        raise ValueError(
            "incomplete E13 learned-generator figure matrix: "
            + "; ".join(missing)
        )


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _retained_rows(path: Path, setting: str, family: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = frame[
        (frame["family"] == family)
        & (frame["setting"] == setting)
    ].copy()
    if rows.empty:
        raise ValueError(f"{path.name}: no retained {family}/{setting} rows")
    seed_sets = set(rows["seeds"].astype(str))
    if seed_sets != {EXPECTED_SEEDS}:
        raise ValueError(
            f"{path.name}: retained {family}/{setting} seed sets are {seed_sets}"
        )
    return rows


def stage_inputs(
    tables: Path,
    canonical: pd.DataFrame,
    inputs: dict[str, Path],
) -> None:
    cnn = canonical[canonical["family"] == "cnn"].copy()
    rf = canonical[canonical["family"] == "rf"].copy()
    lstm = canonical[canonical["family"] == "lstm"].copy()
    transformer = canonical[canonical["family"] == "transformer"].copy()

    _write(cnn, tables / "generator_extension_summary.csv")
    _write(
        rf[~rf["rung"].isin(["L5_cantt", "L5_road"])],
        tables / "rf_generator_extension_summary.csv",
    )
    _write(
        rf[rf["rung"].isin(["L5_cantt", "L5_road"])],
        tables / "rf_generator_extension_external_summary.csv",
    )
    _write(lstm, tables / "family_extension_summary.csv")
    _write(
        transformer,
        tables / "family_extension_transformer_summary.csv",
    )

    # The corrected learned-generator figure keeps only the unaffected +30%
    # context from the legacy protocol-valid tables.  The canonical E13
    # summary above supplies the strict-v2 WGAN +100% rows.
    _write(
        _retained_rows(
            inputs["gan_cnn_retained"], "ganvalid_0p30", "cnn"
        ),
        tables / "generator_extension_summary_gan_protocol_valid.csv",
    )
    _write(
        _retained_rows(
            inputs["arlm_cnn_retained"], "arlmvalid_0p30", "cnn"
        ),
        tables / "generator_extension_summary_arlm_protocol_valid.csv",
    )
    _write(
        _retained_rows(inputs["gan_rf_retained"], "ganvalid_0p30", "rf"),
        tables / "rf_generator_extension_summary_gan_protocol_valid.csv",
    )
    _write(
        _retained_rows(inputs["arlm_rf_retained"], "arlmvalid_0p30", "rf"),
        tables / "rf_generator_extension_summary_arlm_protocol_valid.csv",
    )

    shutil.copy2(
        inputs["placebo_summary_retained"],
        tables / headline_figure.PLACEBO_SOURCE,
    )
    shutil.copy2(
        inputs["placebo_paired_retained"],
        tables / focal_figure.PLACEBO_SOURCE,
    )
    holm = pd.read_csv(inputs["analysis_supporting_holm"])
    holm["comparison"] = holm["comparison"].replace(
        E13_TO_RENDERER_FOCAL_IDS
    )
    expected_renderer_ids = {
        "K1_cnn_L1_rule_gain",
        "K2_cnn_L1_ganvalid_shift",
        "K3_lstm_L1_rule_gain",
        "K4_rf_otids_fpr_rule_push",
        "K5_rf_otids_fpr_ganvalid_push",
        "K6_ss_rf_L1_rule_gain",
    }
    if set(holm["comparison"].astype(str)) != expected_renderer_ids:
        raise ValueError(
            "E13 supporting Holm table does not contain the fixed six focal "
            f"comparisons: {sorted(holm['comparison'].astype(str))}"
        )
    _write(holm, tables / focal_figure.HOLM_SOURCE)
    shutil.copy2(
        inputs["realization_paired_e13"],
        tables / focal_figure.REALIZATION_SOURCE,
    )


def _render_staged_figures(
    stage_tables: Path,
    stage_figures: Path,
    canonical: pd.DataFrame,
    inputs: dict[str, Path],
) -> None:
    stage_inputs(stage_tables, canonical, inputs)
    original_tables, original_figures = lc.TABLES, lc.FIGURES
    original_ladder_tables = ladder_figure.TABLES
    original_ladder_figures = ladder_figure.FIGURES
    original_arm_labels = dict(ladder_figure.ARM_LABELS)
    original_generator_labels = dict(generator_figure.LABELS)
    try:
        lc.TABLES = stage_tables
        lc.FIGURES = stage_figures
        ladder_figure.TABLES = stage_tables
        ladder_figure.FIGURES = stage_figures
        ladder_figure.ARM_LABELS["rule_1p00"] = "Rule +100%"
        generator_figure.LABELS.update({
            "rule_0p30": "Rule +30% (retained)",
            "ganvalid_0p30": "WGAN +30% (retained)",
            "ganvalid_1p00": "WGAN strict-v2 +100%",
            "arlmvalid_0p30": "AR-LM +30% (retained)",
        })

        focal_rows = focal_figure.load_rows()
        for row in focal_rows:
            row["row_id"] = RENDERER_TO_E13_FOCAL_IDS.get(
                str(row["row_id"]), row["row_id"]
            )
            labels = {
                "K2_cnn_L1_ganvalid_shift_e13": (
                    "K2  CNN L1 · WGAN strict-v2 +100%"
                ),
                "K4_rf_otids_fpr_rule_push_e13": (
                    "K4  RF OTIDS FPR · Rule strict-v2 +100%"
                ),
                "K5_rf_otids_fpr_ganvalid_push_e13": (
                    "K5  RF OTIDS FPR · WGAN strict-v2 +100%"
                ),
                "S_frame_disjoint_L2": (
                    "CNN L2 · Rule strict-v2 +100% (frame-disjoint)"
                ),
            }
            row["label"] = labels.get(str(row["row_id"]), row["label"])
        _write(
            pd.DataFrame.from_records(focal_rows),
            stage_tables / "focal_forest_ledger.csv",
        )
        focal_figure.make_figure(focal_rows)

        headline_data, headline_records = headline_figure.load_headline_data()
        _write(
            pd.DataFrame.from_records(headline_records),
            stage_tables / "headline_shift_ladder_results.csv",
        )
        headline_figure.make_figure(headline_data)

        combined_ladder = ladder_figure.load_combined()
        ladder_figure.make_figure(
            combined_ladder,
            include_ae=False,
            footer=(
                "Cell text is the five-pipeline-seed mean; "
                "+100% is the E13 joint pool-re-instantiation/"
                "no-replacement arm."
            ),
        )

        combined_external = external_figure.load_combined()
        external_figure.make_figure(
            combined_external,
            families=list(external_figure.FAMILIES),
            ae_settings=(),
        )

        cnn_family = family_figure.journal_cnn_rows()
        other_families = family_figure.journal_rows()
        family_figure.make_figure(
            pd.concat([cnn_family, other_families], ignore_index=True),
            include_ae=False,
            footer=(
                "Cell text is the five-seed mean; "
                "+100% is the E13 joint pool-re-instantiation/"
                "no-replacement arm."
            ),
        )

        validate_generator_figure_matrix(generator_figure.load_rows())
        generator_figure.main()
    finally:
        lc.TABLES, lc.FIGURES = original_tables, original_figures
        ladder_figure.TABLES = original_ladder_tables
        ladder_figure.FIGURES = original_ladder_figures
        ladder_figure.ARM_LABELS.clear()
        ladder_figure.ARM_LABELS.update(original_arm_labels)
        generator_figure.LABELS.clear()
        generator_figure.LABELS.update(original_generator_labels)


def staged_output_paths(stage_tables: Path, stage_figures: Path) -> list[Path]:
    paths = [
        stage_tables / "headline_shift_ladder_results.csv",
        stage_tables / "focal_forest_ledger.csv",
    ]
    for stem in OUTPUT_STEMS:
        paths.extend([
            stage_figures / f"{stem}.png",
            stage_figures / f"{stem}.pdf",
        ])
    return paths


def validate_staged_outputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"figure build did not create companions: {missing}")
    for path in paths:
        if path.suffix == ".png":
            if path.stat().st_size < 10_000 or path.read_bytes()[:8] != (
                b"\x89PNG\r\n\x1a\n"
            ):
                raise RuntimeError(f"invalid PNG companion: {path}")
        elif path.suffix == ".pdf":
            if path.stat().st_size < 5_000 or not path.read_bytes().startswith(
                b"%PDF"
            ):
                raise RuntimeError(f"invalid PDF companion: {path}")
        elif path.suffix == ".csv":
            if pd.read_csv(path).empty:
                raise RuntimeError(f"empty figure ledger: {path}")


def versioned_destination(path: Path, tag: str) -> Path:
    return versioned_destination_under(
        path,
        tag,
        tables_root=lc.TABLES,
        figures_root=lc.FIGURES,
    )


def versioned_destination_under(
    path: Path,
    tag: str,
    *,
    tables_root: Path,
    figures_root: Path,
) -> Path:
    if path.parent.name == "tables":
        root = tables_root
    elif path.parent.name == "figures":
        root = figures_root
    else:
        raise ValueError(f"unexpected staged output directory: {path}")
    return root / f"{path.stem}_{tag}{path.suffix}"


def git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    artifact_prefixes = (
        "journal/results/",
        "journal/models/",
        "datasets/synthetic/",
    )
    source_definition_status = []
    artifact_status = []
    for line in status:
        relative = line[3:] if len(line) > 3 else line
        if relative.startswith(artifact_prefixes):
            artifact_status.append(line)
        else:
            source_definition_status.append(line)
    return {
        "commit": commit,
        "source_definition_clean": not source_definition_status,
        "source_definition_status": source_definition_status,
        "artifact_status_count": len(artifact_status),
        "artifact_status_sha256": hashlib.sha256(
            "\n".join(artifact_status).encode("utf-8")
        ).hexdigest(),
    }


def publish_no_clobber(pairs: list[tuple[Path, Path]]) -> None:
    collisions = [str(destination) for _source, destination in pairs
                  if destination.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite existing E13 figure artifacts: {collisions}"
        )
    published: list[Path] = []
    try:
        for source, destination in pairs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            published.append(destination)
    except Exception:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise


def build(
    input_map_path: Path,
    output_tag: str,
    *,
    input_root: Path = REPO_ROOT,
    output_root: Path = REPO_ROOT,
    check_only: bool = False,
    require_clean_source: bool = False,
) -> dict:
    output_tag = validate_output_tag(output_tag)
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    frozen_wisa = (REPO_ROOT / "wisa").resolve()
    if output_root == frozen_wisa or output_root.is_relative_to(frozen_wisa):
        raise ValueError("refusing to write E13 figures under frozen wisa/")
    map_payload, inputs = load_input_map(
        input_map_path,
        input_root=input_root,
    )
    # Parse the upstream provenance now, rather than merely hashing an opaque
    # sidecar.  Its detailed schema remains owned by the analysis builder.
    analysis_provenance = json.loads(
        inputs["analysis_provenance"].read_text()
    )
    state = git_metadata()
    if require_clean_source:
        if not state["source_definition_clean"]:
            raise RuntimeError(
                "source-definition tree is not clean: "
                f"{state['source_definition_status']}"
            )
        upstream_state = analysis_provenance.get("source_state", {})
        if not upstream_state.get("source_definition_clean"):
            raise RuntimeError(
                "upstream E13 analysis was not built from a clean "
                "source-definition tree"
            )
        if upstream_state.get("commit") != state["commit"]:
            raise RuntimeError(
                "upstream E13 analysis commit does not match the figure "
                f"builder commit: analysis={upstream_state.get('commit')} "
                f"figure={state['commit']}"
            )
    canonical = load_canonical_summary(
        inputs["analysis_summary"],
        str(map_payload["analysis_version"]),
    )
    validate_main_matrix(canonical)
    if check_only:
        return {
            "status": "inputs_valid",
            "analysis_version": map_payload["analysis_version"],
            "canonical_rows": len(canonical),
            "output_tag": output_tag,
        }

    output_journal = output_root / "journal"
    output_tables = output_journal / "results" / "tables"
    output_figures = output_journal / "results" / "figures"
    output_logs = output_journal / "results" / "logs"
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    output_logs.mkdir(parents=True, exist_ok=True)
    audit_destination = (
        output_logs / f"{output_tag}_figure_provenance_v1.json"
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output_tag}-figure-build-",
        dir=output_journal / "results",
    ) as temporary:
        stage = Path(temporary)
        stage_tables = stage / "tables"
        stage_figures = stage / "figures"
        stage_tables.mkdir()
        stage_figures.mkdir()
        staged_paths = staged_output_paths(stage_tables, stage_figures)
        destinations = [
            versioned_destination_under(
                path,
                output_tag,
                tables_root=output_tables,
                figures_root=output_figures,
            )
            for path in staged_paths
        ]
        collisions = [
            str(path) for path in [*destinations, audit_destination]
            if path.exists()
        ]
        if collisions:
            raise FileExistsError(
                "refusing to overwrite existing E13 figure artifacts: "
                f"{collisions}"
            )

        _render_staged_figures(
            stage_tables, stage_figures, canonical, inputs
        )
        validate_staged_outputs(staged_paths)

        input_records = {
            key: {
                "path": (
                    root_relative(path, input_root)
                    if path.is_relative_to(input_root)
                    else repo_relative(path)
                ),
                "source_root": (
                    "input_root"
                    if path.is_relative_to(input_root)
                    else "repository"
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in sorted(inputs.items())
        }
        input_records["figure_input_map"] = {
            "path": repo_relative(input_map_path),
            "bytes": input_map_path.stat().st_size,
            "sha256": sha256(input_map_path),
        }
        output_records = []
        for source, destination in zip(staged_paths, destinations):
            output_records.append({
                "path": root_relative(destination, output_root),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            })
        helper_scripts = [
            SCRIPT,
            Path(headline_figure.__file__),
            Path(ladder_figure.__file__),
            Path(external_figure.__file__),
            Path(generator_figure.__file__),
            Path(family_figure.__file__),
            Path(focal_figure.__file__),
        ]
        audit = {
            "schema_version": 1,
            "analysis_version": map_payload["analysis_version"],
            "output_tag": output_tag,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git": state,
            "seed_set": EXPECTED_SEEDS,
            "canonical_summary_rows": len(canonical),
            "inputs": input_records,
            "outputs": output_records,
            "scripts": {
                repo_relative(path): sha256(path)
                for path in helper_scripts
            },
            "lineage_rules": {
                "main_family_views": (
                    "retained real-only/+30% controls and strict-v2 +100%; "
                    "legacy Conv-AE rows excluded"
                ),
                "generator_view": (
                    "retained protocol-valid WGAN/AR-LM +30% context; "
                    "strict-v2 primary WGAN +100%"
                ),
                "headline_placebo": (
                    "retained E10a real-duplication/placebo controls; "
                    "strict-v2 Rule +100% in L2"
                ),
                "legacy_v2_interpretation": (
                    "joint pool re-instantiation plus no-replacement "
                    "corrective sensitivity, not a replacement-only effect"
                ),
            },
            "publication": {
                "method": "staged companions followed by no-clobber hard links",
                "overwrites_allowed": False,
            },
        }
        staged_audit = stage / audit_destination.name
        staged_audit.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        json.loads(staged_audit.read_text())
        publish_no_clobber([
            *list(zip(staged_paths, destinations)),
            (staged_audit, audit_destination),
        ])

    return {
        "status": "published",
        "analysis_version": map_payload["analysis_version"],
        "output_tag": output_tag,
        "outputs": [root_relative(path, output_root) for path in destinations],
        "provenance": root_relative(audit_destination, output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-map", type=Path, default=DEFAULT_INPUT_MAP)
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "repository-layout mirror containing preview E13 analysis files; "
            "missing retained inputs fall back to the read-only checkout"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT,
        help="repository-layout root for versioned outputs",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the mapped E13 inputs and complete five-seed matrix",
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help=(
            "require clean source definitions and the same clean commit as "
            "the upstream E13 analysis provenance"
        ),
    )
    args = parser.parse_args()
    try:
        result = build(
            args.input_map,
            args.output_tag,
            input_root=args.input_root,
            output_root=args.output_root,
            check_only=args.check_only,
            require_clean_source=args.require_clean_source,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
