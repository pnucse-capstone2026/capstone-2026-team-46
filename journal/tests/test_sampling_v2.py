from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import lib_common as lc  # noqa: E402
import train_family_extension_lstm as lstm_train  # noqa: E402
import train_family_extension_transformer as transformer_train  # noqa: E402
import train_generator_extension_cnn as cnn_train  # noqa: E402
import train_generator_extension_rf as rf_train  # noqa: E402


def _balanced_labels(per_class):
    return np.concatenate([
        np.full(per_class, cls, dtype=np.int8)
        for cls in (1, 2, 3, 4)
    ])


def _canonical_rule_sampling_audit(tmp_path, monkeypatch, real_data):
    pool_path = tmp_path / "rule_based_windows_unique_pool_v2.npz"
    generation_log_path = tmp_path / "rule_generation.log"
    pool_path.write_bytes(b"canonical-rule-pool")
    declared = {
        **lc.STRICT_V2_POOL_CONFIGS["rule"],
        "path": pool_path,
        "generation_log": generation_log_path,
        "generator_configuration": dict(
            lc.STRICT_V2_POOL_CONFIGS["rule"][
                "generator_configuration"
            ]
        ),
    }
    monkeypatch.setitem(lc.STRICT_V2_POOL_CONFIGS, "rule", declared)

    generation_configuration = {
        "schema_version": "unit_generation_configuration_v2",
        "pool_identifier": declared["identifier"],
        "generator_configuration": declared["generator_configuration"],
        "downstream_sampling_policy": lc.STRICT_V2_SAMPLING_POLICY,
        "source": {
            "sha256": declared["generator_configuration"][
                "source_train_sha256"
            ],
        },
    }
    generation_configuration_sha256 = hashlib.sha256(
        json.dumps(
            generation_configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    content_sha256 = "c" * 64
    pool_sha256 = lc.sha256_file(pool_path)
    generation_log_path.write_text(json.dumps({
        "schema_version": "rule_based_generation_log_v2",
        "pool_identifier": declared["identifier"],
        "outputs": {"pool": {"sha256": pool_sha256}},
        "configuration": generation_configuration,
        "configuration_sha256": generation_configuration_sha256,
        "audit": {
            "content_uniqueness": {
                "content_sha256": content_sha256,
            },
        },
    }))
    pool_metadata = {
        "pool_identifier": declared["identifier"],
        "pool_schema_version": declared["pool_schema_version"],
        "downstream_sampling_policy": lc.STRICT_V2_SAMPLING_POLICY,
        "generation_configuration_sha256": (
            generation_configuration_sha256
        ),
        "generator": "rule_based",
        "construction_seed": 314159,
        "source_train_sha256": declared["generator_configuration"][
            "source_train_sha256"
        ],
        "consumer_content_audit": {
            "pool_content_sha256": content_sha256,
            "total_windows": 4,
            "total_unique": 4,
            "total_repeated": 0,
            "per_class_windows": {
                str(label): 1 for label in (1, 2, 3, 4)
            },
            "per_class_repeated": {
                str(label): 0 for label in (1, 2, 3, 4)
            },
            "protocol_valid": True,
            "condition_metadata_valid": True,
        },
        "generation_log": {
            "path": lc._audit_path(generation_log_path),
            "sha256": lc.sha256_file(generation_log_path),
            "schema_version": "rule_based_generation_log_v2",
            "pool_sha256_verified": pool_sha256,
        },
    }
    configuration = {
        "schema_version": "synthetic_training_configuration_v2",
        "setting": "rule_1p00",
        "ratio": 1.0,
        "sampling_policy": lc.STRICT_V2_SAMPLING_POLICY,
        "detector_training_pipeline_seed": 7,
        "sampling_seed": 1007,
        "generator_configuration": declared["generator_configuration"],
        "pool_identifier": declared["identifier"],
        "pool_path": lc._audit_path(pool_path),
        "pool_sha256": pool_sha256,
        "pool_override": False,
        "pool_metadata": pool_metadata,
        "real_data_provenance": real_data,
        "source_commit": "deadbeef",
        "source_worktree_clean": True,
        "source_tracked_state_clean": True,
        "source_allowed_untracked_artifacts": [],
        "source_provenance_policy": (
            "tracked_source_clean_allow_untracked_no_clobber_artifacts"
        ),
    }
    _, sampling = lc.sample_synthetic_indices_v2(
        np.asarray([1, 2, 3, 4], dtype=np.int8),
        4,
        1007,
    )
    return {
        **configuration,
        "config_sha256": hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        "pool_windows": 4,
        "pool_class_counts": {
            str(label): 1 for label in (1, 2, 3, 4)
        },
        "real_train_windows": 4,
        "synthetic_windows_requested": 4,
        "sampling": sampling,
        "augmented_train_windows": 8,
    }


@pytest.mark.parametrize(
    ("total", "expected_requested"),
    [
        (19, [5, 5, 5, 4]),  # per-class boundary N-1 overall
        (20, [5, 5, 5, 5]),  # exact per-class capacity N
    ],
)
def test_strict_v2_capacity_boundary_allows_n_minus_one_and_n(
        total, expected_requested):
    labels = _balanced_labels(5)
    indices, audit = lc.sample_synthetic_indices_v2(labels, total, seed=307)

    assert len(indices) == total
    assert len(np.unique(indices)) == total
    assert audit["total"] == {
        "requested": total,
        "drawn": total,
        "unique": total,
        "repeated": 0,
    }
    assert [
        audit["per_class"][str(cls)]["requested"]
        for cls in (1, 2, 3, 4)
    ] == expected_requested
    assert all(
        audit["per_class"][str(cls)]["repeated"] == 0
        for cls in (1, 2, 3, 4)
    )


def test_strict_v2_capacity_boundary_rejects_n_plus_one():
    labels = _balanced_labels(5)
    with pytest.raises(
            ValueError,
            match=r"capacity exceeded .*class 1: requested 6, available 5"):
        lc.sample_synthetic_indices_v2(labels, 21, seed=307)


def test_strict_v2_is_reproducible_seeded_and_class_balanced():
    labels = _balanced_labels(20)
    first, first_audit = lc.sample_synthetic_indices_v2(labels, 31, seed=307)
    replay, replay_audit = lc.sample_synthetic_indices_v2(labels, 31, seed=307)
    other, other_audit = lc.sample_synthetic_indices_v2(labels, 31, seed=308)

    np.testing.assert_array_equal(first, replay)
    assert first_audit == replay_audit
    assert not np.array_equal(first, other)
    assert first_audit["indices_sha256"] != other_audit["indices_sha256"]
    assert [np.count_nonzero(labels[first] == cls) for cls in (1, 2, 3, 4)] == [
        8, 8, 8, 7
    ]
    assert first_audit["sampling_policy"] == lc.STRICT_V2_SAMPLING_POLICY


def test_strict_v2_does_not_change_legacy_replacement_behavior():
    labels = _balanced_labels(2)
    legacy = lc.sample_synthetic_indices(labels, total_count=12, seed=307)
    assert len(legacy) == 12
    assert len(np.unique(legacy)) < len(legacy)

    with pytest.raises(ValueError, match="capacity exceeded"):
        lc.sample_synthetic_indices_v2(labels, total_count=12, seed=307)


def test_build_train_arrays_v2_records_complete_audit(
        tmp_path, monkeypatch):
    train_x = np.arange(17 * 2 * 3, dtype=np.float32).reshape(17, 2, 3)
    train_y = np.zeros(17, dtype=np.int8)
    val_x = np.zeros((3, 2, 3), dtype=np.float32)
    val_y = np.zeros(3, dtype=np.int8)

    def fake_load_npz(split):
        if split == "train":
            return {"x": train_x, "y_attack_type": train_y}
        if split == "val":
            return {"x": val_x, "y_attack_type": val_y}
        raise AssertionError(split)

    monkeypatch.setattr(lc, "load_npz", fake_load_npz)
    pool = tmp_path / "rule_unique_pool_v2.npz"
    pool_y = _balanced_labels(5)
    pool_x = np.arange(20 * 2 * 3, dtype=np.float32).reshape(20, 2, 3)
    np.savez(pool, x=pool_x, y_attack_type=pool_y)
    expected_pool_hash = hashlib.sha256(pool.read_bytes()).hexdigest()
    generator = {
        "family": "rule_based",
        "construction_seed": 314159,
        "lineage": "unit_test",
    }

    built = lc.build_train_arrays_v2(
        "rule_1p00",
        seed=7,
        pool_path=pool,
        pool_identifier="unit_rule_unique_pool_v2",
        generator_configuration=generator,
        source_commit="deadbeef",
    )
    aug_x, aug_y, returned_val_x, returned_val_y, audit = built

    assert aug_x.shape == (34, 2, 3)
    assert aug_y.shape == (34,)
    np.testing.assert_array_equal(returned_val_x, val_x)
    np.testing.assert_array_equal(returned_val_y, val_y)
    assert audit["sampling_policy"] == lc.STRICT_V2_SAMPLING_POLICY
    assert audit["detector_training_pipeline_seed"] == 7
    assert audit["sampling_seed"] == 1007
    assert audit["generator_configuration"] == generator
    assert audit["pool_identifier"] == "unit_rule_unique_pool_v2"
    assert audit["pool_sha256"] == expected_pool_hash
    assert audit["pool_override"] is True
    assert audit["source_commit"] == "deadbeef"
    assert audit["source_worktree_clean"] is None
    assert (
        audit["source_provenance_policy"]
        == "explicit_test_or_controlled_override"
    )
    assert len(audit["config_sha256"]) == 64
    assert audit["pool_class_counts"] == {
        "1": 5, "2": 5, "3": 5, "4": 5,
    }
    assert audit["sampling"]["total"] == {
        "requested": 17,
        "drawn": 17,
        "unique": 17,
        "repeated": 0,
    }
    assert [
        audit["sampling"]["per_class"][str(cls)]["requested"]
        for cls in (1, 2, 3, 4)
    ] == [5, 4, 4, 4]

    replay_audit = lc.build_train_arrays_v2(
        "rule_1p00",
        seed=7,
        pool_path=pool,
        pool_identifier="unit_rule_unique_pool_v2",
        generator_configuration=generator,
        source_commit="deadbeef",
    )[-1]
    assert replay_audit["config_sha256"] == audit["config_sha256"]
    assert (
        replay_audit["sampling"]["indices_sha256"]
        == audit["sampling"]["indices_sha256"]
    )


def test_build_train_arrays_v2_requires_declared_versioned_pool(tmp_path):
    with pytest.raises(ValueError, match="is not declared"):
        lc.build_train_arrays_v2(
            "gan_1p00",
            seed=7,
            pool_path=tmp_path / "missing.npz",
            source_commit="deadbeef",
        )


def test_build_train_arrays_v2_rejects_partial_pool_override(tmp_path):
    with pytest.raises(ValueError, match="requires pool_path, pool_identifier"):
        lc.build_train_arrays_v2(
            "rule_1p00",
            seed=7,
            pool_path=tmp_path / "rule.npz",
            source_commit="deadbeef",
        )


def test_strict_v2_source_commit_requires_clean_worktree(monkeypatch):
    lc._source_commit.cache_clear()
    responses = iter([
        SimpleNamespace(stdout="deadbeef\n"),
        SimpleNamespace(stdout=" M journal/scripts/lib_common.py\n"),
    ])
    monkeypatch.setattr(lc.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    with pytest.raises(RuntimeError, match="requires a clean Git worktree"):
        lc._source_commit()
    lc._source_commit.cache_clear()


def test_strict_v2_source_commit_is_cached_before_outputs(monkeypatch):
    lc._source_commit.cache_clear()
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="deadbeef\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    assert lc._source_commit() == "deadbeef"
    assert lc._source_commit() == "deadbeef"
    assert len(calls) == 2
    lc._source_commit.cache_clear()


def test_strict_v2_training_bundle_validates_log_and_artifact_hash(
        tmp_path, monkeypatch):
    windows = tmp_path / "windows"
    windows.mkdir()
    train_source = windows / "train_windows.npz"
    val_source = windows / "val_windows.npz"
    train_source.write_bytes(b"frozen-train-source")
    val_source.write_bytes(b"frozen-val-source")
    monkeypatch.setattr(lc, "WINDOWS", windows)
    monkeypatch.setattr(
        lc,
        "STRICT_V2_REAL_SPLIT_SHA256",
        {
            "train": hashlib.sha256(
                train_source.read_bytes()).hexdigest(),
            "val": hashlib.sha256(val_source.read_bytes()).hexdigest(),
        },
    )

    checkpoint = tmp_path / "model.pt"
    log = tmp_path / "train.log"
    checkpoint.write_bytes(b"model")
    real_data = lc._strict_v2_real_split_file_provenance()
    record = {
        "family": "cnn",
        "setting": "rule_1p00",
        "seed": 7,
        "model_tag": "sampling_v2",
        "sampling_policy": "strict-v2",
        "sampling_audit": _canonical_rule_sampling_audit(
            tmp_path, monkeypatch, real_data
        ),
        "train_windows": 8,
    }
    record.update(lc.strict_v2_artifact_record({
        "checkpoint": checkpoint,
    }))
    lc.write_json_atomic(log, record)

    validated = lc.validate_strict_v2_training_bundle(
        log,
        {"checkpoint": checkpoint},
        {
            "family": "cnn",
            "setting": "rule_1p00",
            "seed": 7,
            "model_tag": "sampling_v2",
            "sampling_policy": "strict-v2",
        },
    )
    assert validated["artifact_sha256"]["checkpoint"] == hashlib.sha256(
        b"model"
    ).hexdigest()

    smoke_record = {
        "setting": "rule_1p00",
        "seed": 7,
        "tag": "_smoke_sampling_v2",
        "train_subset": 0.25,
        "train_windows": 2,
    }
    smoke_mismatches = lc._validate_strict_v2_sampling_audit(
        record["sampling_audit"], smoke_record)
    assert "training_log.train_windows" not in smoke_mismatches
    assert "training_log.train_subset" not in smoke_mismatches

    invalid_subset_record = {
        **smoke_record,
        "tag": "_sampling_v2",
    }
    invalid_subset = lc._validate_strict_v2_sampling_audit(
        record["sampling_audit"], invalid_subset_record)
    assert "training_log.train_subset" in invalid_subset

    incomplete_record = json.loads(log.read_text())
    incomplete_record["sampling_audit"].pop("pool_metadata")
    log.write_text(json.dumps(incomplete_record))
    with pytest.raises(RuntimeError, match="pool_metadata"):
        lc.validate_strict_v2_training_bundle(
            log,
            {"checkpoint": checkpoint},
            {
                "family": "cnn",
                "setting": "rule_1p00",
                "seed": 7,
                "model_tag": "sampling_v2",
                "sampling_policy": "strict-v2",
            },
        )
    log.write_text(json.dumps(record))

    train_source.write_bytes(b"tampered-frozen-train-source")
    with pytest.raises(RuntimeError, match="current_real_data_provenance"):
        lc.validate_strict_v2_training_bundle(
            log,
            {"checkpoint": checkpoint},
            {
                "family": "cnn",
                "setting": "rule_1p00",
                "seed": 7,
                "model_tag": "sampling_v2",
                "sampling_policy": "strict-v2",
            },
        )
    train_source.write_bytes(b"frozen-train-source")

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact_sha256.checkpoint"):
        lc.validate_strict_v2_training_bundle(
            log,
            {"checkpoint": checkpoint},
            {
                "family": "cnn",
                "setting": "rule_1p00",
                "seed": 7,
                "model_tag": "sampling_v2",
                "sampling_policy": "strict-v2",
            },
        )


def test_strict_v2_training_bundle_rejects_checkpoint_without_audit(tmp_path):
    checkpoint = tmp_path / "partial.pt"
    checkpoint.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="incomplete strict-v2 training bundle"):
        lc.validate_strict_v2_training_bundle(
            tmp_path / "missing.log",
            {"checkpoint": checkpoint},
            {"sampling_policy": "strict-v2"},
        )


def test_artifact_tags_reject_path_components():
    with pytest.raises(ValueError, match="path separators"):
        lc.validate_artifact_tag("../sampling_v2", name="--model-tag")
    with pytest.raises(ValueError, match="path separators"):
        lc.validate_artifact_tag(
            "_sampling_v2/part",
            name="--out-suffix",
            leading_underscore=True,
        )


def test_canonical_strict_training_tags_enforce_e13_scope():
    lc.validate_strict_v2_training_scope(
        family="cnn",
        settings=["rule_1p00", "ganvalid_1p00"],
        seeds=[7, 42],
        tag="sampling_v2",
        budget_mode="legacy",
    )
    with pytest.raises(ValueError, match="excludes settings"):
        lc.validate_strict_v2_training_scope(
            family="cnn",
            settings=["rule_0p30"],
            seeds=[7],
            tag="sampling_v2",
            budget_mode="legacy",
        )
    with pytest.raises(ValueError, match="require unique seeds"):
        lc.validate_strict_v2_training_scope(
            family="lstm",
            settings=["rule_1p00"],
            seeds=[999],
            tag="sampling_v2",
        )
    with pytest.raises(ValueError, match="excludes settings"):
        lc.validate_strict_v2_training_scope(
            family="cnn",
            settings=["rule_1p00"],
            seeds=[7],
            tag="matchedsteps_sampling_v2",
            budget_mode="matched_steps",
        )
    # A visibly noncanonical smoke tag is allowed for preflight work; the
    # canonical evaluator still refuses to merge it into the full matrix.
    lc.validate_strict_v2_training_scope(
        family="cnn",
        settings=["rule_0p30"],
        seeds=[999],
        tag="smoke_rule0p30_sampling_v2",
        budget_mode="legacy",
    )


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (
            lstm_train,
            [
                "train_family_extension_lstm.py",
                "--setting", "rule_1p00",
                "--seed", "7",
                "--sampling-policy", "strict-v2",
                "--model-tag", "matchedsteps_sampling_v2",
            ],
        ),
        (
            transformer_train,
            [
                "train_family_extension_transformer.py",
                "--setting", "rule_1p00",
                "--seed", "7",
                "--sampling-policy", "strict-v2",
                "--tag", "_matchedsteps_sampling_v2",
            ],
        ),
        (
            rf_train,
            [
                "train_generator_extension_rf.py",
                "--settings",
                "rule_1p00,ganvalid_1p00,ganvalidb_1p00,ganvalidc_1p00",
                "--seeds", "7,42,123,2026,3407",
                "--sampling-policy", "strict-v2",
                "--model-tag", "matchedsteps_sampling_v2",
                "--out-suffix", "_matchedsteps_sampling_v2",
            ],
        ),
    ],
)
def test_non_cnn_strict_trainers_reject_matchedsteps_tags(
        module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        module.main()


def test_rf_strict_output_suffix_must_contain_model_tag(
        monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_generator_extension_rf.py",
            "--settings",
            "rule_1p00,ganvalid_1p00,ganvalidb_1p00,ganvalidc_1p00",
            "--seeds", "7,42,123,2026,3407",
            "--sampling-policy", "strict-v2",
            "--model-tag", "alternate_sampling_v2",
            "--out-suffix", "_sampling_v2",
        ],
    )
    with pytest.raises(SystemExit):
        rf_train.main()


def test_cnn_strict_cli_selects_v2_and_versioned_checkpoint(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cnn_train, "MODELS", tmp_path / "models")
    monkeypatch.setattr(cnn_train.lc, "LOGS", tmp_path / "logs")
    calls = []
    monkeypatch.setattr(cnn_train, "train_one", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_generator_extension_cnn.py",
            "--settings", "rule_1p00",
            "--seeds", "7",
            "--sampling-policy", "strict-v2",
        ],
    )

    cnn_train.main()

    assert len(calls) == 1
    assert calls[0][-2:] == ("sampling_v2", "strict-v2")
    assert (
        cnn_train.checkpoint_path("rule_1p00", 7, "sampling_v2").name
        == "cnn_rule_1p00_sampling_v2_seed7.pt"
    )


def test_rf_strict_cli_versions_models_results_and_audit(
        tmp_path, monkeypatch):
    monkeypatch.setattr(rf_train, "MODELS", tmp_path / "models")
    monkeypatch.setattr(rf_train.lc, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(rf_train.lc, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(rf_train, "build_eval_cache", lambda: {})
    calls = []

    def fake_train(setting, seed, model_tag, sampling_policy):
        calls.append((setting, seed, model_tag, sampling_policy))
        return object(), {"sampling_policy": lc.STRICT_V2_SAMPLING_POLICY}

    monkeypatch.setattr(rf_train, "train_one", fake_train)
    monkeypatch.setattr(rf_train, "eval_rungs", lambda *_args: None)
    monkeypatch.setattr(rf_train, "summarize", lambda _rows: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_generator_extension_rf.py",
            "--settings", "rule_1p00",
            "--seeds", "7",
            "--sampling-policy", "strict-v2",
            "--model-tag", "smoke_sampling_v2",
            "--out-suffix", "_smoke_sampling_v2_rule",
        ],
    )

    rf_train.main()

    assert calls == [
        ("rule_1p00", 7, "smoke_sampling_v2", "strict-v2")
    ]
    assert (
        rf_train.checkpoint_path(
            "rule_1p00", 7, "smoke_sampling_v2"
        ).name
        == "rf_rule_1p00_smoke_sampling_v2_seed7.joblib"
    )
    log = (
        tmp_path
        / "logs"
        / "train_generator_extension_rf_smoke_sampling_v2_rule.log"
    )
    assert lc.STRICT_V2_SAMPLING_POLICY in log.read_text()


@pytest.mark.parametrize(
    ("module", "program", "tag_option", "path_function", "expected_name"),
    [
        (
            lstm_train,
            "train_family_extension_lstm.py",
            "--model-tag",
            lstm_train.checkpoint_path,
            "lstm_rule_1p00_sampling_v2_seed7.pt",
        ),
        (
            transformer_train,
            "train_family_extension_transformer.py",
            "--tag",
            lambda setting, seed, tag: transformer_train.training_artifact_paths(
                setting, seed, tag
            )[0],
            "transformer_rule_1p00_seed7_sampling_v2.pt",
        ),
    ],
)
def test_neural_family_strict_cli_uses_versioned_artifacts(
        module, program, tag_option, path_function, expected_name,
        tmp_path, monkeypatch):
    monkeypatch.setattr(module.lc, "MODELS", tmp_path / "models")
    monkeypatch.setattr(module.lc, "LOGS", tmp_path / "logs")
    calls = []
    monkeypatch.setattr(module, "train_one", lambda *args: calls.append(args))
    argv = [
        program,
        "--setting", "rule_1p00",
        "--seed", "7",
        "--sampling-policy", "strict-v2",
    ]
    if tag_option == "--tag":
        expected_tag = "_sampling_v2"
    else:
        expected_tag = "sampling_v2"
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert len(calls) == 1
    assert calls[0][-1] == "strict-v2"
    assert calls[0][3] == expected_tag
    assert path_function("rule_1p00", 7, expected_tag).name == expected_name


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (
            cnn_train,
            [
                "train_generator_extension_cnn.py",
                "--settings", "rule_1p00",
                "--seeds", "7",
                "--model-tag", "sampling_v2",
            ],
        ),
        (
            rf_train,
            [
                "train_generator_extension_rf.py",
                "--settings", "rule_1p00",
                "--seeds", "7",
                "--model-tag", "sampling_v2",
                "--out-suffix", "_sampling_v2",
            ],
        ),
        (
            lstm_train,
            [
                "train_family_extension_lstm.py",
                "--setting", "rule_1p00",
                "--seed", "7",
                "--model-tag", "sampling_v2",
            ],
        ),
        (
            transformer_train,
            [
                "train_family_extension_transformer.py",
                "--setting", "rule_1p00",
                "--seed", "7",
                "--tag", "_sampling_v2",
            ],
        ),
    ],
)
def test_strict_v2_tag_rejects_legacy_sampler(
        module, argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert (
        "a strict-v2 model tag requires --sampling-policy strict-v2"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("module", "argv", "message"),
    [
        (
            cnn_train,
            [
                "train_generator_extension_cnn.py",
                "--settings", "rule_1p00",
                "--seeds", "7",
                "--sampling-policy", "strict-v2",
                "--budget-mode", "matched_steps",
                "--model-tag", "sampling_v2",
            ],
            "strict-v2 budget/tag mismatch",
        ),
        (
            cnn_train,
            [
                "train_generator_extension_cnn.py",
                "--settings", "rule_1p00",
                "--seeds", "7",
                "--sampling-policy", "strict-v2",
                "--max-epochs", "1",
            ],
            "strict-v2 budget overrides require a model tag containing 'smoke'",
        ),
        (
            lstm_train,
            [
                "train_family_extension_lstm.py",
                "--setting", "rule_1p00",
                "--seed", "7",
                "--sampling-policy", "strict-v2",
                "--max-epochs", "1",
            ],
            "strict-v2 --max-epochs requires a model tag containing 'smoke'",
        ),
        (
            transformer_train,
            [
                "train_family_extension_transformer.py",
                "--setting", "rule_1p00",
                "--seed", "7",
                "--sampling-policy", "strict-v2",
                "--train-subset", "0.1",
            ],
            "strict-v2 --max-epochs/--train-subset overrides require a tag",
        ),
    ],
)
def test_strict_v2_canonical_tags_reject_smoke_or_budget_mismatch(
        module, argv, message, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
