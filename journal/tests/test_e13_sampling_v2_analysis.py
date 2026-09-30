from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_e13_sampling_v2_analysis as e13  # noqa: E402


def test_source_state_preserves_first_porcelain_status_column(
    tmp_path, monkeypatch
):
    def fake_run(command, **_kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout="abc123\n")
        return SimpleNamespace(
            returncode=0,
            stdout=" D journal/results/tables/result.csv\n",
        )

    monkeypatch.setattr(e13.subprocess, "run", fake_run)
    state = e13.source_state(tmp_path)

    assert state["commit"] == "abc123"
    assert state["source_definition_clean"] is True
    assert state["source_definition_status"] == []
    assert state["artifact_status_count"] == 1


def _cell_frame(seeds=e13.EXPECTED_SEEDS):
    rows = []
    for index, seed in enumerate(seeds):
        rows.append(
            {
                "family": "cnn",
                "setting": "real_only",
                "seed": seed,
                "rung": "L1_fixed_variant",
                "subset": "all",
                "windows": 100,
                "normal_windows": 50,
                "attack_windows": 50,
                "accuracy": 0.8 + index * 0.001,
                "macro_f1_binary": 0.7 + index * 0.001,
                "normal_recall": 0.9 - index * 0.001,
                "fpr": 0.1 + index * 0.001,
                "attack_recall": 0.6 + index * 0.001,
                "tn": 45,
                "fp": 5,
                "fn": 20,
                "tp": 30,
                "exact_recall": 0.5 + index * 0.001,
            }
        )
    return pd.DataFrame(rows)


def _unit_spec():
    return e13.SelectionSpec(
        "unit",
        "unit.csv",
        "cnn",
        ("real_only",),
        e13.ORIGINAL_SCHEDULE,
        e13.RETAINED_ROLE,
        e13.RETAINED_LINEAGE,
    )


def test_exact_five_seed_validation_rejects_missing_seed():
    with pytest.raises(ValueError, match="requires exactly seeds"):
        e13._validate_selected_cells(
            _cell_frame(e13.EXPECTED_SEEDS[:-1]), _unit_spec()
        )


def test_exact_five_seed_validation_rejects_duplicate_row():
    frame = _cell_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate selected per-seed"):
        e13._validate_selected_cells(frame, _unit_spec())


def test_structurally_undefined_metric_requires_all_five_seeds():
    frame = _cell_frame()
    frame["exact_recall"] = np.nan
    e13._validate_selected_cells(frame, _unit_spec())

    frame.loc[0, "exact_recall"] = 0.5
    with pytest.raises(ValueError, match="defined for only some seeds"):
        e13._validate_selected_cells(frame, _unit_spec())


def test_declared_external_selection_fills_only_missing_exact_recall(tmp_path):
    source = _cell_frame().drop(columns=["exact_recall"])
    source.to_csv(tmp_path / "external.csv", index=False)
    declared = e13.SelectionSpec(
        "external",
        "external.csv",
        "cnn",
        ("real_only",),
        e13.ORIGINAL_SCHEDULE,
        e13.RETAINED_ROLE,
        e13.RETAINED_LINEAGE,
        True,
    )
    selected, _ = e13.load_selection(tmp_path, declared, tmp_path)
    assert selected["exact_recall"].isna().all()

    undeclared = e13.SelectionSpec(
        "external",
        "external.csv",
        "cnn",
        ("real_only",),
        e13.ORIGINAL_SCHEDULE,
        e13.RETAINED_ROLE,
        e13.RETAINED_LINEAGE,
    )
    with pytest.raises(ValueError, match="missing columns"):
        e13.load_selection(tmp_path, undeclared, tmp_path)


def test_paired_interval_is_sample_sd_and_untruncated():
    result = e13.paired_seed_statistics(
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert result["delta_std"] == pytest.approx(
        np.std([1.0, 1.0, 1.0, 1.0, 0.0], ddof=1)
    )
    assert result["primary_seed_t_ci_hi"] > 1.0
    assert result["paired_seeds"] == "7;42;123;2026;3407"


def test_holm_adjust_is_monotone_in_sorted_p_order():
    p = np.asarray([0.04, 0.001, 0.02, 0.5])
    adjusted = e13.holm_adjust(p)
    order = np.argsort(p, kind="stable")

    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= p) & (adjusted <= 1.0))


def test_atomic_publisher_is_no_clobber(tmp_path):
    first = tmp_path / "tables" / "first.csv"
    second = tmp_path / "tables" / "second.csv"
    marker = tmp_path / "logs" / "provenance.json"
    payloads = {
        first: b"a\n1\n",
        second: b"b\n2\n",
        marker: b'{"complete":true}\n',
    }

    e13.publish_outputs_atomic(payloads, marker)
    before = {path: path.read_bytes() for path in payloads}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e13.publish_outputs_atomic(payloads, marker)
    assert {path: path.read_bytes() for path in payloads} == before


def test_atomic_publisher_rolls_back_caught_link_failure(tmp_path, monkeypatch):
    first = tmp_path / "tables" / "first.csv"
    second = tmp_path / "tables" / "second.csv"
    marker = tmp_path / "logs" / "provenance.json"
    payloads = {first: b"1", second: b"2", marker: b"{}"}
    real_link = e13.os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return real_link(source, destination)

    monkeypatch.setattr(e13.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected"):
        e13.publish_outputs_atomic(payloads, marker)
    assert not any(path.exists() for path in payloads)


def _setting_offset(setting: str) -> float:
    settings = {
        "real_only": 0.00,
        "rule_0p30": 0.08,
        "over_1p00": 0.04,
        "rule_1p00": 0.14,
        "ganvalid_1p00": 0.10,
        "ganvalidb_1p00": 0.06,
        "ganvalidc_1p00": 0.02,
    }
    return settings[setting]


def _write_full_input_fixture(tables: Path):
    selections: dict[tuple[str, str], set[str]] = {}
    for spec in e13.SELECTION_SPECS:
        selections.setdefault((spec.file_name, spec.family), set()).update(
            spec.settings
        )

    strict_names = {
        spec.file_name
        for spec in e13.SELECTION_SPECS
        if spec.analysis_role == e13.STRICT_ROLE
    }
    for (file_name, family), settings in selections.items():
        rows = []
        file_offset = 0.03 if file_name in strict_names else 0.0
        rungs = (
            ("L5_cantt", "L5_road")
            if "rf_generator_extension_external" in file_name
            else ("L1_fixed_variant", "L5_otids")
        )
        for setting in sorted(settings):
            setting_offset = _setting_offset(setting)
            slope = (sorted(settings).index(setting) + 1) * 0.0002
            for rung in rungs:
                for seed_index, seed in enumerate(e13.EXPECTED_SEEDS):
                    value = (
                        0.30
                        + setting_offset
                        + file_offset
                        + slope * seed_index
                    )
                    rows.append(
                        {
                            "family": family,
                            "setting": setting,
                            "seed": seed,
                            "rung": rung,
                            "subset": "all",
                            "windows": 100,
                            "normal_windows": 50,
                            "attack_windows": 50,
                            "accuracy": value,
                            "macro_f1_binary": value,
                            "normal_recall": 1.0 - value / 2.0,
                            "fpr": value / 2.0,
                            "attack_recall": value,
                            "tn": 40,
                            "fp": 10,
                            "fn": 20,
                            "tp": 30,
                            "exact_recall": value - 0.01,
                        }
                    )
        pd.DataFrame(rows).to_csv(tables / file_name, index=False)

    second_rows = []
    for setting in e13.SECOND_SOURCE_SPEC.settings:
        for seed_index, seed in enumerate(e13.EXPECTED_SEEDS):
            value = (
                0.2
                + _setting_offset(setting)
                + (0.0003 if setting == "rule_0p30" else 0.0001)
                * seed_index
            )
            second_rows.append(
                {
                    "family": "rf",
                    "setting": setting,
                    "seed": seed,
                    "rung": "C_L1_fixed_variant",
                    "subset": "all",
                    "windows": 100,
                    "normal_windows": 50,
                    "attack_windows": 50,
                    "accuracy": value,
                    "macro_f1_binary": value,
                    "normal_recall": 1.0 - value / 2.0,
                    "fpr": value / 2.0,
                    "attack_recall": value,
                    "tn": 40,
                    "fp": 10,
                    "fn": 20,
                    "tp": 30,
                }
            )
    pd.DataFrame(second_rows).to_csv(
        tables / e13.SECOND_SOURCE_SPEC.file_name, index=False
    )


def test_integrated_builder_merges_lineage_and_publishes_versioned_outputs(
    tmp_path,
):
    repo = tmp_path / "repo"
    input_tables = repo / "journal" / "results" / "source_tables"
    tables = repo / "journal" / "results" / "tables"
    logs = repo / "journal" / "results" / "logs"
    prereg = (
        repo
        / "journal"
        / "experiments"
        / "e13_strict_v2_sampling"
        / "PREREG.md"
    )
    input_tables.mkdir(parents=True)
    prereg.parent.mkdir(parents=True)
    prereg.write_text(
        "strict no-replacement sampling v2\n"
        "pool-reinstantiation plus no-replacement corrective sensitivity\n"
        "7, 42, 123, 2026, 3407\n",
        encoding="utf-8",
    )
    _write_full_input_fixture(input_tables)

    paths = e13.build_outputs(
        tables,
        logs,
        prereg,
        repo,
        "unit_v1",
        input_tables_dir=input_tables,
    )

    assert set(paths) == {
        "by_seed",
        "summary",
        "paired",
        "legacy_joint_corrective",
        "focal_comparisons",
        "supporting_holm",
        "provenance",
    }
    assert all(path.is_file() for path in paths.values())

    merged = pd.read_csv(paths["by_seed"])
    assert {
        e13.STRICT_ROLE,
        e13.RETAINED_ROLE,
        e13.RETAINED_MATCHED_ROLE,
        e13.LEGACY_ROLE,
    }.issubset(set(merged["analysis_role"]))
    assert merged["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not merged.duplicated(list(e13.ROW_KEY)).any()

    summary = pd.read_csv(paths["summary"])
    assert summary["n_seeds"].eq(5).all()
    assert summary["seeds"].eq("7;42;123;2026;3407").all()

    corrective = pd.read_csv(paths["legacy_joint_corrective"])
    assert corrective["corrective_sensitivity_label"].eq(
        e13.JOINT_CORRECTIVE_ESTIMAND
    ).all()
    assert corrective["interpretation_boundary"].eq(
        e13.JOINT_CORRECTIVE_BOUNDARY
    ).all()
    assert corrective["n_paired_seeds"].eq(5).all()

    focal = pd.read_csv(paths["focal_comparisons"])
    holm = pd.read_csv(paths["supporting_holm"])
    assert len(focal) == len(holm) == 6
    assert focal["n_seeds"].eq(5).all()
    assert holm["multiplicity_role"].str.contains("supporting").all()

    provenance = json.loads(paths["provenance"].read_text())
    assert provenance["expected_pipeline_seeds"] == list(e13.EXPECTED_SEEDS)
    assert provenance["corrective_sensitivity"]["interpretation_boundary"] == (
        e13.JOINT_CORRECTIVE_BOUNDARY
    )
    for name, record in provenance["outputs"].items():
        assert e13.sha256_file(paths[name]) == record["sha256"]

    before = {name: path.read_bytes() for name, path in paths.items()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e13.build_outputs(
            tables,
            logs,
            prereg,
            repo,
            "unit_v1",
            input_tables_dir=input_tables,
        )
    assert {name: path.read_bytes() for name, path in paths.items()} == before
