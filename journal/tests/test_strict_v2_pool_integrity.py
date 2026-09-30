import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import lib_common as lc  # noqa: E402


def _balanced_labels(per_class=5):
    return np.concatenate([
        np.full(per_class, label, dtype=np.int8)
        for label in (1, 2, 3, 4)
    ])


def _write_declared_pool(
        path,
        declared,
        *,
        generator_configuration=None,
        scalar_overrides=None,
):
    expected = declared["generator_configuration"]
    generator_configuration = (
        dict(expected)
        if generator_configuration is None
        else dict(generator_configuration)
    )
    generation_configuration = {
        "schema_version": "unit_generation_configuration_v2",
        "pool_identifier": declared["identifier"],
        "generator_configuration": generator_configuration,
        "downstream_sampling_policy": lc.STRICT_V2_SAMPLING_POLICY,
    }
    if expected["family"] == "wgan_gp_protocol_valid":
        generation_configuration["input_provenance"] = {
            "generator_training_seed": expected["generator_training_seed"],
            "pool_sampling_seed": expected["pool_sampling_seed"],
            "generator_checkpoint_sha256": (
                expected["generator_checkpoint_sha256"]
            ),
            "standardizer_sha256": expected["standardizer_sha256"],
            "source_commit": "deadbeef",
            "source_tracked_state_clean": True,
        }
    else:
        generation_configuration["source"] = {
            "sha256": expected["source_train_sha256"],
        }
        generation_configuration["source_commit"] = "deadbeef"
        generation_configuration["source_provenance"] = {
            "source_commit": "deadbeef",
            "source_worktree_clean": True,
            "source_tracked_state_clean": True,
            "source_allowed_untracked_artifacts": [],
        }
    configuration_json = json.dumps(
        generation_configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    y_attack = _balanced_labels()
    x = np.zeros((len(y_attack), 128, len(lc.FEATURE_NAMES)),
                 dtype=np.float32)
    x[:, 0, 0] = np.arange(len(y_attack), dtype=np.float32)
    class_names = np.asarray(
        [lc.CLASS_NAMES[int(label)] for label in y_attack],
        dtype=np.str_,
    )
    ordered_digest = hashlib.sha256()
    for row, label in zip(x, y_attack, strict=True):
        row_digest = hashlib.sha256(
            memoryview(np.ascontiguousarray(row)).cast("B")
        ).digest()
        ordered_digest.update(bytes((int(label),)))
        ordered_digest.update(row_digest)
    arrays = {
        "x": x,
        "y_binary": np.ones(len(y_attack), dtype=np.int8),
        "y_attack_type": y_attack,
        "synthetic_type": class_names,
        "condition_label": y_attack.copy(),
        "condition_name": class_names.copy(),
        "feature_names": np.asarray(lc.FEATURE_NAMES, dtype=np.str_),
        "window_size": np.asarray(128, dtype=np.int32),
        "stride": np.asarray(32, dtype=np.int32),
        "generator_protocol_version": np.asarray(
            "dlc_payload_v1", dtype=np.str_),
        "pool_content_sha256": np.asarray(
            ordered_digest.hexdigest(), dtype=np.str_),
        "pool_identifier": np.asarray(declared["identifier"], dtype=np.str_),
        "pool_schema_version": np.asarray(
            declared["pool_schema_version"], dtype=np.str_),
        "downstream_sampling_policy": np.asarray(
            lc.STRICT_V2_SAMPLING_POLICY, dtype=np.str_),
        "generation_configuration_json": np.asarray(
            configuration_json, dtype=np.str_),
        "generation_configuration_sha256": np.asarray(
            hashlib.sha256(configuration_json.encode("utf-8")).hexdigest(),
            dtype=np.str_,
        ),
    }
    if expected["family"] == "rule_based":
        arrays.update({
            "generator": np.asarray("rule_based", dtype=np.str_),
            "construction_seed": np.asarray(
                expected["construction_seed"], dtype=np.uint64),
            "source_train_sha256": np.asarray(
                expected["source_train_sha256"], dtype=np.str_),
            "source_commit": np.asarray("deadbeef", dtype=np.str_),
            "source_worktree_clean": np.asarray(True, dtype=np.bool_),
            "source_tracked_state_clean": np.asarray(True, dtype=np.bool_),
            "source_allowed_untracked_artifacts_json": np.asarray(
                "[]", dtype=np.str_),
        })
    else:
        arrays.update({
            "generator": np.asarray(
                "conditional_wgan_gp_aux", dtype=np.str_),
            "generator_training_seed": np.asarray(
                expected["generator_training_seed"], dtype=np.int32),
            "pool_sampling_seed": np.asarray(
                expected["pool_sampling_seed"], dtype=np.int32),
            "generator_checkpoint_sha256": np.asarray(
                expected["generator_checkpoint_sha256"], dtype=np.str_),
            "standardizer_sha256": np.asarray(
                expected["standardizer_sha256"], dtype=np.str_),
            "source_commit": np.asarray("deadbeef", dtype=np.str_),
            "source_tracked_state_clean": np.asarray(True, dtype=np.bool_),
        })
    arrays.update(scalar_overrides or {})
    np.savez(path, **arrays)


def _write_generation_log(path, pool_path, declared, pool_key):
    with np.load(pool_path, allow_pickle=True) as archive:
        configuration_json = str(
            archive["generation_configuration_json"].item()
        )
        configuration = json.loads(configuration_json)
        configuration_sha256 = str(
            archive["generation_configuration_sha256"].item()
        )
        content_sha256 = str(archive["pool_content_sha256"].item())
    pool_sha256 = lc.sha256_file(pool_path)
    family = declared["generator_configuration"]["family"]
    if family == "rule_based":
        record = {
            "schema_version": "rule_based_generation_log_v2",
            "pool_identifier": declared["identifier"],
            "outputs": {"pool": {"sha256": pool_sha256}},
            "configuration": configuration,
            "configuration_sha256": configuration_sha256,
            "audit": {
                "content_uniqueness": {
                    "content_sha256": content_sha256,
                },
            },
        }
    else:
        record = {
            "schema_version": "gan_pool_generation_log_v2",
            "pool_identifier": declared["identifier"],
            "strict_v2_pool_key": pool_key,
            "output_sha256": pool_sha256,
            "generation_configuration": configuration,
            "generation_configuration_sha256": configuration_sha256,
            "content_uniqueness": {
                "content_sha256": content_sha256,
            },
        }
    path.write_text(json.dumps(record, indent=2))


def _fake_real_splits():
    train_x = np.zeros(
        (17, 128, len(lc.FEATURE_NAMES)), dtype=np.float32)
    train_y = np.zeros(17, dtype=np.int8)
    val_x = np.zeros(
        (3, 128, len(lc.FEATURE_NAMES)), dtype=np.float32)
    val_y = np.zeros(3, dtype=np.int8)

    def load(split):
        if split == "train":
            return {"x": train_x, "y_attack_type": train_y}
        if split == "val":
            return {"x": val_x, "y_attack_type": val_y}
        raise AssertionError(split)

    return load


def test_declared_rule_pool_metadata_is_validated_and_audited(
        tmp_path, monkeypatch):
    pool_path = tmp_path / "rule_based_windows_unique_pool_v2.npz"
    declared = {
        **lc.STRICT_V2_POOL_CONFIGS["rule"],
        "path": pool_path,
        "generation_log": tmp_path / "rule_generation.log",
        "generator_configuration": dict(
            lc.STRICT_V2_POOL_CONFIGS["rule"]["generator_configuration"]),
    }
    _write_declared_pool(pool_path, declared)
    _write_generation_log(
        declared["generation_log"], pool_path, declared, "rule")
    monkeypatch.setitem(lc.STRICT_V2_POOL_CONFIGS, "rule", declared)
    monkeypatch.setattr(lc, "load_npz", _fake_real_splits())

    audit = lc.build_train_arrays_v2(
        "rule_1p00", seed=7, source_commit="deadbeef")[-1]

    assert audit["pool_override"] is False
    assert audit["pool_metadata"]["pool_identifier"] == declared["identifier"]
    assert audit["pool_metadata"]["generator"] == "rule_based"
    assert audit["pool_metadata"]["construction_seed"] == 314159
    assert audit["pool_metadata"]["consumer_content_audit"][
        "total_repeated"] == 0
    assert audit["pool_metadata"]["generation_log"]["pool_sha256_verified"] == (
        lc.sha256_file(pool_path)
    )

    generation_record = json.loads(
        declared["generation_log"].read_text()
    )
    generation_record["outputs"]["pool"]["sha256"] = "0" * 64
    declared["generation_log"].write_text(json.dumps(generation_record))
    with pytest.raises(ValueError, match="generation-log mismatch"):
        lc.build_train_arrays_v2(
            "rule_1p00", seed=7, source_commit="deadbeef")


def test_declared_pool_consumer_recomputes_content_hash(tmp_path):
    pool_path = tmp_path / "rule_based_windows_unique_pool_v2.npz"
    declared = lc.STRICT_V2_POOL_CONFIGS["rule"]
    _write_declared_pool(
        pool_path,
        declared,
        scalar_overrides={
            "pool_content_sha256": np.asarray("0" * 64, dtype=np.str_),
        },
    )

    with np.load(pool_path, allow_pickle=True) as archive:
        with pytest.raises(ValueError, match="content SHA-256 mismatch"):
            lc._strict_v2_pool_content_audit(archive, pool_path)


@pytest.mark.parametrize(
    ("scalar_overrides", "generator_configuration", "message"),
    [
        (
            {"pool_identifier": np.asarray("wrong_rule_pool", dtype=np.str_)},
            None,
            "pool_identifier mismatch",
        ),
        (
            {"construction_seed": np.asarray(271828, dtype=np.uint64)},
            None,
            "construction seed mismatch",
        ),
        (
            {"source_train_sha256": np.asarray("0" * 64, dtype=np.str_)},
            None,
            "source train SHA-256 mismatch",
        ),
        (
            None,
            {
                "family": "rule_based",
                "construction_seed": 271828,
                "lineage": (
                    "legacy_rule_configuration_extended_without_replacement"),
            },
            "generator configuration mismatch",
        ),
    ],
)
def test_declared_rule_pool_rejects_identity_seed_or_config_mismatch(
        tmp_path, scalar_overrides, generator_configuration, message):
    pool_path = tmp_path / "rule_based_windows_unique_pool_v2.npz"
    declared = lc.STRICT_V2_POOL_CONFIGS["rule"]
    _write_declared_pool(
        pool_path,
        declared,
        scalar_overrides=scalar_overrides,
        generator_configuration=generator_configuration,
    )

    with np.load(pool_path, allow_pickle=True) as archive:
        with pytest.raises(ValueError, match=message):
            lc._validate_declared_strict_v2_pool_metadata(
                archive, "rule", declared, pool_path)


def test_declared_wgan_pool_validates_input_hashes_and_provenance(tmp_path):
    pool_path = tmp_path / (
        "gan_gseed271828_protocol_valid_windows_unique_pool_v2.npz")
    declared = lc.STRICT_V2_POOL_CONFIGS["ganvalidb"]
    _write_declared_pool(pool_path, declared)

    with np.load(pool_path, allow_pickle=True) as archive:
        metadata = lc._validate_declared_strict_v2_pool_metadata(
            archive, "ganvalidb", declared, pool_path)

    assert metadata["generator_training_seed"] == 271828
    assert metadata["pool_sampling_seed"] == 314159
    assert metadata["generator_checkpoint_sha256"] == (
        declared["generator_configuration"]["generator_checkpoint_sha256"]
    )
    assert metadata["standardizer_sha256"] == (
        declared["generator_configuration"]["standardizer_sha256"]
    )
    assert metadata["source_commit"] == "deadbeef"


@pytest.mark.parametrize(
    ("scalar_overrides", "message"),
    [
        (
            {"pool_identifier": np.asarray(
                "gan_protocol_valid_unique_pool_v2", dtype=np.str_)},
            "pool_identifier mismatch",
        ),
        (
            {"generator_training_seed": np.asarray(314159, dtype=np.int32)},
            "generator training seed mismatch",
        ),
        (
            {"pool_sampling_seed": np.asarray(7, dtype=np.int32)},
            "pool sampling seed mismatch",
        ),
        (
            {"generator_checkpoint_sha256": np.asarray(
                "not-a-sha256", dtype=np.str_)},
            "generator_checkpoint_sha256.*64-character",
        ),
        (
            {"generator_checkpoint_sha256": np.asarray(
                "0" * 64, dtype=np.str_)},
            "frozen generator checkpoint SHA-256 mismatch",
        ),
        (
            {"standardizer_sha256": np.asarray("", dtype=np.str_)},
            "standardizer_sha256.*64-character",
        ),
        (
            {"standardizer_sha256": np.asarray(
                "0" * 64, dtype=np.str_)},
            "frozen standardizer SHA-256 mismatch",
        ),
        (
            {"source_commit": np.asarray("", dtype=np.str_)},
            "source_commit must be nonempty",
        ),
    ],
)
def test_declared_wgan_pool_rejects_wrong_seed_or_provenance(
        tmp_path, scalar_overrides, message):
    pool_path = tmp_path / (
        "gan_gseed271828_protocol_valid_windows_unique_pool_v2.npz")
    declared = lc.STRICT_V2_POOL_CONFIGS["ganvalidb"]
    _write_declared_pool(
        pool_path, declared, scalar_overrides=scalar_overrides)

    with np.load(pool_path, allow_pickle=True) as archive:
        with pytest.raises(ValueError, match=message):
            lc._validate_declared_strict_v2_pool_metadata(
                archive, "ganvalidb", declared, pool_path)


def test_source_provenance_allows_only_untracked_generated_artifacts(
        monkeypatch):
    lc._source_commit.cache_clear()
    responses = iter([
        SimpleNamespace(stdout="deadbeef\n"),
        SimpleNamespace(stdout=(
            "?? journal/results/logs/train_sampling_v2.log\0"
            "?? journal/results/tables/summary_sampling_v2.csv\0"
        )),
    ])
    monkeypatch.setattr(
        lc.subprocess,
        "run",
        lambda *_args, **_kwargs: next(responses),
    )

    provenance = lc._source_provenance()

    assert provenance["source_commit"] == "deadbeef"
    assert provenance["source_worktree_clean"] is False
    assert provenance["source_tracked_state_clean"] is True
    assert provenance["source_allowed_untracked_artifacts"] == [
        "journal/results/logs/train_sampling_v2.log",
        "journal/results/tables/summary_sampling_v2.csv",
    ]
    assert provenance["source_provenance_policy"] == (
        "tracked_source_clean_allow_untracked_no_clobber_artifacts")
    lc._source_commit.cache_clear()


@pytest.mark.parametrize(
    "status_record",
    [
        "?? journal/scripts/untracked_source.py\0",
        "?? journal/tests/untracked_test.py\0",
        "?? journal/manuscript/untracked_note.txt\0",
        " M journal/results/tables/tracked_output.csv\0",
    ],
)
def test_source_provenance_rejects_source_or_tracked_output_changes(
        monkeypatch, status_record):
    lc._source_commit.cache_clear()
    responses = iter([
        SimpleNamespace(stdout="deadbeef\n"),
        SimpleNamespace(stdout=status_record),
    ])
    monkeypatch.setattr(
        lc.subprocess,
        "run",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="outside no-clobber"):
        lc._source_provenance()
    lc._source_commit.cache_clear()
