import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_rule_based_synthetic_v2 as rule_v2  # noqa: E402


def _normal_windows(count):
    x = np.zeros(
        (count, rule_v2.WINDOW_SIZE, len(rule_v2.lc.FEATURE_NAMES)),
        dtype=np.float32,
    )
    for row in range(count):
        x[row, :, 0] = 100 + row
        x[row, :, 1] = 8
        x[row, :, 2:10] = row % 251
        x[row, :, 10] = 0.001 + row * 1e-6
    return x


def _write_train_archive(path, count):
    x = _normal_windows(count)
    np.savez_compressed(
        path,
        x=x,
        y_binary=np.zeros(count, dtype=np.int8),
        y_attack_type=np.zeros(count, dtype=np.int8),
        feature_names=np.asarray(rule_v2.lc.FEATURE_NAMES, dtype=object),
        window_size=np.asarray(rule_v2.WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(rule_v2.STRIDE, dtype=np.int32),
    )


def test_current_plus_100_capacity_matches_strict_class_prefix_split():
    capacity = rule_v2.capacity_requirements(262_149, 1.0)

    assert capacity["synthetic_windows_requested"] == 262_149
    assert capacity["per_class_requested"] == {
        "1": 65_538,
        "2": 65_537,
        "3": 65_537,
        "4": 65_537,
    }
    assert capacity["minimum_equal_pool_capacity_per_class"] == 65_538
    rule_v2.require_pool_capacity(65_538, capacity)
    with pytest.raises(
            ValueError,
            match=r"class 1: requested 65538, available 65537"):
        rule_v2.require_pool_capacity(65_537, capacity)


def test_canonical_rule_identity_rejects_wrong_seed_or_scope(
        tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    declared = {
        **rule_v2.lc.STRICT_V2_POOL_CONFIGS["rule"],
        "path": (
            journal
            / "datasets"
            / "synthetic"
            / "rule_based_windows_unique_pool_v2.npz"
        ),
        "statistics_path": (
            journal
            / "results"
            / "tables"
            / "rule_based_synthetic_statistics_unique_pool_v2.csv"
        ),
        "generation_log": (
            journal
            / "results"
            / "logs"
            / "generate_rule_based_synthetic_unique_pool_v2.log"
        ),
    }
    monkeypatch.setattr(rule_v2.lc, "ROOT", journal)
    monkeypatch.setitem(
        rule_v2.lc.STRICT_V2_POOL_CONFIGS, "rule", declared)

    with pytest.raises(ValueError, match="requires construction seed"):
        rule_v2._validate_canonical_rule_request(
            declared["path"],
            construction_seed=271828,
            per_attack=65_538,
            target_ratio=1.0,
        )
    with pytest.raises(ValueError, match="requires --per-attack"):
        rule_v2._validate_canonical_rule_request(
            declared["path"],
            construction_seed=314159,
            per_attack=65_000,
            target_ratio=1.0,
        )
    with pytest.raises(ValueError, match="requires --target-ratio 1.0"):
        rule_v2._validate_canonical_rule_request(
            declared["path"],
            construction_seed=314159,
            per_attack=65_538,
            target_ratio=0.3,
        )
    with pytest.raises(ValueError, match="declared log-output path"):
        rule_v2._validate_canonical_rule_request(
            declared["path"],
            stats_out=declared["statistics_path"],
            log_out=journal / "results" / "logs" / "wrong.log",
            construction_seed=314159,
            per_attack=65_538,
            target_ratio=1.0,
        )


def test_small_rule_pool_is_seed_reproducible_unique_and_protocol_valid():
    train_x = _normal_windows(9)
    normal_indices = np.arange(len(train_x), dtype=np.int64)

    first = rule_v2.generate_pool(
        train_x,
        normal_indices,
        per_attack=3,
        construction_seed=rule_v2.CONSTRUCTION_SEED,
    )
    replay = rule_v2.generate_pool(
        train_x,
        normal_indices,
        per_attack=3,
        construction_seed=rule_v2.CONSTRUCTION_SEED,
    )

    for key in first["arrays"]:
        np.testing.assert_array_equal(
            first["arrays"][key], replay["arrays"][key])
    assert first["audit"] == replay["audit"]
    assert first["audit"]["shape"] == [12, 128, 11]
    assert first["audit"]["class_counts"] == {
        "1": 3, "2": 3, "3": 3, "4": 3,
    }
    assert first["audit"]["protocol"]["protocol_valid"] is True
    assert (
        first["audit"]["condition_metadata"]["condition_metadata_valid"]
        is True
    )
    assert first["audit"]["content_uniqueness"]["total"] == {
        "windows": 12,
        "unique": 12,
        "repeated": 0,
    }


def test_train_source_schema_rejects_binary_attack_label_disagreement(
        tmp_path):
    source = tmp_path / "train_windows.npz"
    x = _normal_windows(2)
    np.savez(
        source,
        x=x,
        y_binary=np.asarray([0, 0], dtype=np.int8),
        y_attack_type=np.asarray([0, 1], dtype=np.int8),
        feature_names=np.asarray(rule_v2.lc.FEATURE_NAMES, dtype=object),
        window_size=np.asarray(rule_v2.WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(rule_v2.STRIDE, dtype=np.int32),
    )

    with np.load(source, allow_pickle=True) as data:
        with pytest.raises(ValueError, match="labels disagree on 1 rows"):
            rule_v2.validate_train_source(data)


def test_canonical_rule_generation_rejects_wrong_source_hash(tmp_path):
    source = tmp_path / "train_windows.npz"
    _write_train_archive(source, 2)

    with pytest.raises(
        ValueError,
        match="requires the frozen Car-Hacking train archive SHA-256",
    ):
        rule_v2.generate_from_archive(
            source,
            per_attack=1,
            target_ratio=1.0,
            construction_seed=rule_v2.CONSTRUCTION_SEED,
            pool_out=tmp_path / "pool.npz",
            stats_out=tmp_path / "stats.csv",
            log_out=tmp_path / "generation.log",
            expected_source_sha256="0" * 64,
        )


def test_legacy_pool_is_protected_even_with_overwrite_permission(
        tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    synthetic = journal / "datasets" / "synthetic"
    tables = journal / "results" / "tables"
    logs = journal / "results" / "logs"
    monkeypatch.setattr(rule_v2.lc, "ROOT", journal)
    monkeypatch.setattr(rule_v2.lc, "SYNTHETIC", synthetic)
    monkeypatch.setattr(rule_v2.lc, "TABLES", tables)

    with pytest.raises(ValueError, match="legacy Rule artifacts are protected"):
        rule_v2._assert_output_targets(
            synthetic / "rule_based_windows.npz",
            tables / "new.csv",
            logs / "new.log",
            allow_overwrite=True,
        )


def test_git_provenance_reports_artifact_only_dirty_state(monkeypatch):
    monkeypatch.setattr(
        rule_v2.lc,
        "_source_provenance",
        lambda: {
            "source_commit": "deadbeef",
            "source_worktree_clean": False,
            "source_tracked_state_clean": True,
            "source_allowed_untracked_artifacts": [
                "journal/results/tables/prior.csv"],
            "source_provenance_policy": (
                "tracked_source_clean_allow_untracked_no_clobber_artifacts"),
        },
    )

    provenance = rule_v2._git_provenance()

    assert provenance["source_commit"] == "deadbeef"
    assert provenance["source_tracked_state_clean"] is True
    assert provenance["worktree_dirty"] is True


def test_end_to_end_small_archive_records_hashes_and_refuses_rewrite(
        tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    synthetic = journal / "datasets" / "synthetic"
    tables = journal / "results" / "tables"
    logs = journal / "results" / "logs"
    windows = journal / "datasets" / "windows"
    windows.mkdir(parents=True)
    source = windows / "train_windows.npz"
    _write_train_archive(source, 13)

    monkeypatch.setattr(rule_v2.lc, "ROOT", journal)
    monkeypatch.setattr(rule_v2.lc, "WINDOWS", windows)
    monkeypatch.setattr(rule_v2.lc, "SYNTHETIC", synthetic)
    monkeypatch.setattr(rule_v2.lc, "TABLES", tables)
    monkeypatch.setattr(rule_v2.lc, "LOGS", logs)
    monkeypatch.setattr(
        rule_v2,
        "_git_provenance",
        lambda: {
            "source_commit": "deadbeef",
            "source_worktree_clean": False,
            "source_tracked_state_clean": True,
            "source_allowed_untracked_artifacts": [
                "journal/results/logs/prior_sampling_v2.log"],
            "source_provenance_policy": (
                "tracked_source_clean_allow_untracked_no_clobber_artifacts"),
            "worktree_dirty": True,
        },
    )

    pool = synthetic / "rule_based_windows_unique_pool_v2.npz"
    stats = tables / "rule_based_synthetic_statistics_unique_pool_v2.csv"
    log = logs / "generate_rule_based_synthetic_unique_pool_v2.log"
    record = rule_v2.generate_from_archive(
        source,
        per_attack=4,
        target_ratio=1.0,
        construction_seed=rule_v2.CONSTRUCTION_SEED,
        pool_out=pool,
        stats_out=stats,
        log_out=log,
    )

    assert pool.is_file() and stats.is_file() and log.is_file()
    expected_pool_hash = hashlib.sha256(pool.read_bytes()).hexdigest()
    assert record["outputs"]["pool"]["sha256"] == expected_pool_hash
    assert record["configuration"]["source"]["role"] == (
        "Car-Hacking train normal windows only")
    assert record["configuration"]["generator_configuration"][
        "source_train_sha256"
    ] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["configuration"]["capacity"]["per_class_requested"] == {
        "1": 4, "2": 3, "3": 3, "4": 3,
    }
    assert len(record["configuration_sha256"]) == 64

    with np.load(pool, allow_pickle=False) as generated:
        assert generated["x"].shape == (16, 128, 11)
        assert generated["pool_identifier"].item() == (
            rule_v2.POOL_IDENTIFIER)
        assert generated["pool_schema_version"].item() == (
            rule_v2.POOL_SCHEMA_VERSION)
        assert generated["construction_seed"].item() == (
            rule_v2.CONSTRUCTION_SEED)
        assert generated["source_train_sha256"].item() == hashlib.sha256(
            source.read_bytes()).hexdigest()
        assert generated["source_worktree_clean"].item() is False
        assert generated["source_tracked_state_clean"].item() is True
        assert json.loads(
            generated["source_allowed_untracked_artifacts_json"].item()
        ) == ["journal/results/logs/prior_sampling_v2.log"]
        assert generated["generation_configuration_sha256"].item() == (
            record["configuration_sha256"])
        assert generated["downstream_sampling_policy"].item() == (
            rule_v2.lc.STRICT_V2_SAMPLING_POLICY)
        assert len(np.unique(generated["base_source_index"])) <= 13

    with stats.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["synthetic_type"] for row in rows] == [
        "DoS", "Fuzzy", "Gear", "RPM"]
    assert all(row["generated_content_repeated"] == "0" for row in rows)
    log_record = json.loads(log.read_text())
    assert log_record["outputs"]["pool"]["sha256"] == expected_pool_hash
    assert log_record["configuration"]["source_commit"] == "deadbeef"
    assert log_record["worktree_dirty_at_generation"] is True
    assert log_record["source_provenance"][
        "source_tracked_state_clean"] is True

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rule_v2.generate_from_archive(
            source,
            per_attack=4,
            target_ratio=1.0,
            construction_seed=rule_v2.CONSTRUCTION_SEED,
            pool_out=pool,
            stats_out=stats,
            log_out=log,
        )
