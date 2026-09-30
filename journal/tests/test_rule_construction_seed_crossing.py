from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_rule_construction_seed_crossing as analyzer  # noqa: E402
import evaluate_rule_construction_seed_crossing as evaluator  # noqa: E402
import generate_rule_construction_sensitivity as generator  # noqa: E402
import train_rule_construction_seed_crossing as training  # noqa: E402


CONSTRUCTION_SEEDS = (314159, 271828, 161803, 141421, 173205)
NEW_CONSTRUCTION_SEEDS = (271828, 161803, 141421, 173205)
PIPELINE_SEEDS = (7, 42, 123, 2026, 3407)
ANCHOR_INDEX_SHA256 = {
    7: "4ff00984e1dd97a48447b2abac6c347eb8967b20e3c90782db081f9681c11581",
    42: "bdfcbdf15dcd5001122de17dd721508c5347353a022becd20df94c0e59a0f917",
    123: "22bd0a549e33ad17662a6b5ac84065a8b229768aace1c5721ece4a2408255ba5",
    2026: "59a6f31755fb18e7f19c4ba9e819b050f4d5db617854e0f2c2e78f4ae31a3319",
    3407: "3ed08fb8b14d3796264fba3cdac76ca7563fd0d951ad295209d0798fd875e3bf",
}


def _tiny_pool_arrays(
    *,
    construction_seed: int = 314159,
    per_attack: int = 2,
) -> dict[str, np.ndarray]:
    labels = np.repeat(
        np.asarray([1, 2, 3, 4], dtype=np.int8),
        per_attack,
    )
    x = np.zeros((len(labels), 128, 11), dtype="<f4")
    for row in range(len(labels)):
        x[row, :, 0] = np.float32(0x100 + row)
        x[row, :, 1] = np.float32(8)
        x[row, :, 2:10] = np.float32(row + 1)
        x[row, :, 10] = np.float32(0.001 + row * 1e-5)
    synthetic_names = np.asarray(
        [
            {1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}[int(label)]
            for label in labels
        ],
        dtype=np.str_,
    )
    arrays = {
        "x": x,
        "y_binary": np.ones(len(labels), dtype=np.int8),
        "y_attack_type": labels,
        "injection_count": np.full(len(labels), 4, dtype="<i2"),
        "synthetic_type": synthetic_names,
        "condition_label": labels.copy(),
        "condition_name": synthetic_names.copy(),
        "base_source_index": np.arange(len(labels), dtype="<i8"),
        "construction_ordinal": np.arange(len(labels), dtype="<i8"),
        "feature_names": np.asarray(
            [
                "can_id",
                "dlc",
                "data0",
                "data1",
                "data2",
                "data3",
                "data4",
                "data5",
                "data6",
                "data7",
                "delta_t",
            ],
            dtype=np.str_,
        ),
        "window_size": np.asarray(128, dtype="<i4"),
        "stride": np.asarray(32, dtype="<i4"),
        "pool_schema_version": np.asarray(
            generator.POOL_SCHEMA_VERSION,
            dtype=np.str_,
        ),
        "pool_identifier": np.asarray(
            (
                f"e15_rule_cseed{construction_seed}_"
                f"{generator.OUTPUT_VERSION}"
            ),
            dtype=np.str_,
        ),
        "construction_seed": np.asarray(construction_seed, dtype="<u8"),
        "downstream_sampling_policy": np.asarray(
            generator.SAMPLING_POLICY,
            dtype=np.str_,
        ),
        "generator_configuration_json": np.asarray(
            json.dumps(
                {
                    "schema_version": (
                        generator.GENERATION_CONFIGURATION_SCHEMA
                    ),
                    "family": "rule_based",
                    "construction_seed": construction_seed,
                    "per_attack": per_attack,
                    "total_windows": len(labels),
                    "class_order": ["DoS", "Fuzzy", "Gear", "RPM"],
                    "source_normal_selection": "with_replacement",
                    "final_pool_permutation": "one_after_all_classes",
                    "numpy_version": generator.EXPECTED_NUMPY_VERSION,
                    "rng": "numpy.random.default_rng/PCG64",
                    "byte_order": "little",
                    "x_dtype": "<f4",
                    "label_dtype": "|i1",
                    "injection_count_dtype": "<i2",
                    "source_train_sha256": generator.TRAIN_SHA256,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        ),
        "source_train_sha256": np.asarray(generator.TRAIN_SHA256),
        "source_commit": np.asarray("deadbeef"),
    }
    arrays["ordered_content_sha256"] = np.asarray(
        generator.ordered_content_digest(x, labels)
    )
    return arrays


def _replace_array(
    arrays: dict[str, np.ndarray],
    key: str,
    value: np.ndarray,
) -> dict[str, np.ndarray]:
    return {**arrays, key: value}


def _continuity_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    fresh_scenario = pd.DataFrame(
        [
            {
                "pipeline_seed": 7,
                "block_id": "block_01",
                "attack": "Gear",
                "id_stratum": "canonical",
                "P": 0,
                "S": 0,
                "D": 0,
                "n": 2000,
                "exact_correct": 123,
                "binary_correct": 456,
            }
        ]
    )
    frozen_scenario = fresh_scenario.rename(
        columns={"pipeline_seed": "seed"}
    ).copy()
    fresh_normal = pd.DataFrame(
        [
            {
                "pipeline_seed": 7,
                "block_id": "block_01",
                "n": 2000,
                "normal_correct": 1990,
            }
        ]
    )
    frozen_normal = fresh_normal.rename(
        columns={"pipeline_seed": "seed"}
    ).copy()
    return fresh_scenario, frozen_scenario, fresh_normal, frozen_normal


def _registered_runtime_continuity() -> pd.DataFrame:
    rows = []
    for pipeline_seed, block_id, attack, id_stratum, cell in (
        itertools.product(
            evaluator.PIPELINE_SEEDS,
            evaluator.BLOCKS,
            evaluator.ATTACKS,
            evaluator.ID_STRATA,
            evaluator.CELLS,
        )
    ):
        rows.append(
            {
                "row_scope": "scenario",
                "pipeline_seed": pipeline_seed,
                "block_id": block_id,
                "attack": attack,
                "id_stratum": id_stratum,
                "cell": cell,
                "n": 2_000,
                "exact_correct_difference": 0,
                "binary_correct_difference": 0,
                "normal_correct_difference": 0,
                "status": "runtime-identical",
            }
        )
    for pipeline_seed, block_id in itertools.product(
        evaluator.PIPELINE_SEEDS,
        evaluator.BLOCKS,
    ):
        rows.append(
            {
                "row_scope": "normal",
                "pipeline_seed": pipeline_seed,
                "block_id": block_id,
                "attack": "not_applicable",
                "id_stratum": "not_applicable",
                "cell": "not_applicable",
                "n": 2_000,
                "exact_correct_difference": 0,
                "binary_correct_difference": 0,
                "normal_correct_difference": 0,
                "status": "runtime-identical",
            }
        )
    result = pd.DataFrame(rows)
    assert len(result) == evaluator.EXPECTED_RUNTIME_CONTINUITY_ROWS
    return result


def _primary_rows(
    exact_p: list[float],
    binary_p: list[float],
) -> pd.DataFrame:
    rows = []
    for endpoint, p_values in (
        ("exact_recall", exact_p),
        ("binary_recall", binary_p),
    ):
        for effect, p_value in zip(analyzer.FACTORIAL_EFFECTS, p_values, strict=True):
            rows.append(
                {
                    "endpoint": endpoint,
                    "effect": effect,
                    "row_type": "summary_new4",
                    "id_scope": "canonical",
                    "attack_scope": "equal_macro",
                    "holm_input_p": p_value,
                    "mean": -0.2 if effect == "P" else 0.1,
                    "degenerate": False,
                    "strict_positive": 0 if effect == "P" else 4,
                    "strict_negative": 4 if effect == "P" else 0,
                }
            )
    return pd.DataFrame(rows)


def test_seed_axes_and_frozen_constant_contracts_are_distinct():
    assert tuple(generator.CONSTRUCTION_SEEDS) == CONSTRUCTION_SEEDS
    assert tuple(generator.PIPELINE_SEEDS) == PIPELINE_SEEDS
    assert tuple(training.CONSTRUCTION_SEEDS) == CONSTRUCTION_SEEDS
    assert tuple(training.PIPELINE_SEEDS) == PIPELINE_SEEDS
    assert tuple(analyzer.NEW_CONSTRUCTION_SEEDS) == NEW_CONSTRUCTION_SEEDS
    assert tuple(evaluator.CONSTRUCTION_SEEDS) == CONSTRUCTION_SEEDS
    assert (
        generator.OUTPUT_VERSION
        == training.OUTPUT_VERSION
        == evaluator.OUTPUT_VERSION
        == analyzer.OUTPUT_VERSION
        == "v2"
    )

    assert generator.PER_ATTACK == 65_000
    assert generator.POOL_SCHEMA_VERSION == (
        "e15_rule_construction_pool_v1"
    )
    assert generator.SAMPLING_POLICY == (
        "e15_strict_without_replacement_0p30_v1"
    )
    assert generator.LEGACY_ORDERED_CONTENT_SHA256 == (
        "9c05045481fc235ea58d85709295996a99c6842c39053332507c6d7ebfbbbc43"
    )
    assert training.REAL_TRAIN_WINDOWS == 262_149
    assert training.SYNTHETIC_TOTAL == 78_645
    assert training.PER_CLASS_REQUESTED == {
        1: 19_662,
        2: 19_661,
        3: 19_661,
        4: 19_661,
    }
    assert training.ANCHOR_INDEX_SHA256 == ANCHOR_INDEX_SHA256


def test_v2_amendment_and_v1_failure_evidence_are_hash_bound():
    amendment_relative = (
        "journal/experiments/e15_rule_construction_crossing/"
        "IMPLEMENTATION_AMENDMENT_2026-07-25_V2_PATH_IDENTITY.md"
    )
    failure_relative = (
        "journal/experiments/e15_rule_construction_crossing/"
        "training_failure_v1.json"
    )
    expected = {
        amendment_relative:
            "e747c5c8877a5dfd4e4d6b5b23c54b0ec23f122cd40f145011fa33cb622afd12",
        failure_relative:
            "b5e4cde58d0e6ba207bf5ba91bf39a150c0485be66a27171810a83e3daadb828",
    }

    for relative, digest in expected.items():
        path = generator.REPO / relative
        assert generator.FROZEN_INPUTS[relative] == digest
        assert generator.sha256_file(path) == digest
    assert evaluator.FROZEN_HASHES[evaluator.AMENDMENT] == (
        expected[amendment_relative]
    )
    assert evaluator.FROZEN_HASHES[evaluator.V1_TRAINING_FAILURE] == (
        expected[failure_relative]
    )


def test_cli_defaults_are_read_only_and_overwrite_is_never_exposed():
    generation_args = generator.parse_args([])
    training_args = training.parse_args([])
    assert generation_args.execute is False
    assert training_args.execute is False
    assert training_args.child_fit is False

    with pytest.raises(SystemExit, match="T-STOP-POOL"):
        generator.main(["--allow-overwrite"])
    with pytest.raises(SystemExit, match="T-STOP-TRAINING"):
        training.main(["--allow-overwrite"])
    with pytest.raises(SystemExit) as missing_stage:
        evaluator.parse_args([])
    assert missing_stage.value.code == 2


def test_training_grid_keeps_construction_and_pipeline_axes_separate(tmp_path):
    grid = training.build_training_grid(repo=tmp_path)
    expected_pairs = list(
        itertools.product(CONSTRUCTION_SEEDS, PIPELINE_SEEDS)
    )
    observed_pairs = {
        (job.construction_seed, job.pipeline_seed)
        for job in grid
    }

    assert len(grid) == 25
    assert observed_pairs == set(expected_pairs)
    assert [
        (job.construction_seed, job.pipeline_seed)
        for job in grid
    ] == expected_pairs
    for job in grid:
        argv = tuple(str(part) for part in job.command)
        flag_values = {
            flag: argv[argv.index(flag) + 1]
            for flag in (
                "--construction-seed",
                "--pipeline-seed",
                "--synthetic-pool",
                "--pool-identifier",
                "--pool-schema",
                "--generation-log",
                "--sampling-policy",
                "--checkpoint-output",
                "--log-output",
            )
        }
        assert flag_values["--construction-seed"] == str(
            job.construction_seed
        )
        assert flag_values["--pipeline-seed"] == str(job.pipeline_seed)
        assert flag_values["--synthetic-pool"] == str(job.pool_path)
        assert flag_values["--pool-identifier"] == job.pool_identifier
        assert flag_values["--pool-schema"] == (
            generator.POOL_SCHEMA_VERSION
        )
        assert flag_values["--sampling-policy"] == (
            generator.SAMPLING_POLICY
        )
        assert flag_values["--checkpoint-output"] == str(
            job.checkpoint_path
        )
        assert flag_values["--log-output"] == str(job.log_path)
        assert flag_values["--generation-log"].endswith(
            f"e15_generate_rule_cseed{job.construction_seed}_v2.json"
        )
        assert job.pool_identifier == (
            f"e15_rule_cseed{job.construction_seed}_v2"
        )
        assert job.sampling_seed == job.pipeline_seed + 300
        assert "--allow-overwrite" not in argv
        assert "rule_based_windows.npz" not in argv
        assert "rule_based_windows_unique_pool_v2.npz" not in argv

    first = next(
        job for job in grid
        if (job.construction_seed, job.pipeline_seed) == (314159, 7)
    )
    other_construction = next(
        job for job in grid
        if (job.construction_seed, job.pipeline_seed) == (271828, 7)
    )
    assert first.sampling_seed == other_construction.sampling_seed == 307
    assert first.pool_path != other_construction.pool_path
    assert first.pool_identifier != other_construction.pool_identifier


def test_child_command_accepts_pool_path_through_repository_symlink():
    job = training.build_training_grid(repo=training.REPO)[0]
    assert job.pool_path != job.pool_path.resolve()

    args = training.parse_args(list(job.command)[2:])
    validated = training._require_child_args(args)

    assert validated.construction_seed == job.construction_seed
    assert validated.pipeline_seed == job.pipeline_seed
    assert validated.pool_path == job.pool_path
    assert validated.pool_path.resolve() == Path(args.synthetic_pool).resolve()


def test_pool_target_map_is_complete_versioned_and_has_no_legacy_alias(
    tmp_path,
):
    targets = generator.pool_target_map(repo=tmp_path)

    assert tuple(targets) == CONSTRUCTION_SEEDS
    assert len(
        {
            path
            for bundle in targets.values()
            for path in (
                bundle.pool_path,
                bundle.statistics_path,
                bundle.generation_log_path,
            )
        }
    ) == 15
    for construction_seed, bundle in targets.items():
        assert bundle.pool_path.name == (
            f"rule_cseed{construction_seed}_windows_v2.npz"
        )
        assert bundle.statistics_path.name == (
            f"e15_rule_cseed{construction_seed}_statistics_v2.csv"
        )
        assert bundle.generation_log_path.name == (
            f"e15_generate_rule_cseed{construction_seed}_v2.json"
        )
        assert "rule_based_windows.npz" not in {
            bundle.pool_path.name,
            bundle.statistics_path.name,
            bundle.generation_log_path.name,
        }


def test_v1_and_v2_canonical_generation_targets_are_disjoint(tmp_path):
    v2_targets = {
        path
        for bundle in generator.pool_target_map(repo=tmp_path).values()
        for path in (
            bundle.pool_path,
            bundle.statistics_path,
            bundle.generation_log_path,
        )
    }
    v2_targets.update(
        generator.generation_record_paths(repo=tmp_path).values()
    )
    v1_targets = {
        Path(str(path).replace("_v2", "_v1"))
        for path in v2_targets
    }

    assert len(v2_targets) == 18
    assert len(v1_targets) == 18
    assert v1_targets.isdisjoint(v2_targets)
    assert all("_v2" in path.name for path in v2_targets)
    assert all("_v1" in path.name for path in v1_targets)

    for path in v1_targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserved-v1-evidence")
    absence = generator._target_absence(
        sorted(v2_targets),
        tmp_path,
    )
    assert absence["all_absent"] is True
    assert absence["verified_absent_count"] == 18

    collision = sorted(v2_targets)[0]
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_bytes(b"v2-collision")
    with pytest.raises(generator.E15PoolError, match="target collision"):
        generator._target_absence(sorted(v2_targets), tmp_path)


def test_all_e15_owned_v2_output_identities_are_disjoint_from_v1():
    generation_targets = {
        path
        for bundle in generator.pool_target_map().values()
        for path in (
            bundle.pool_path,
            bundle.statistics_path,
            bundle.generation_log_path,
        )
    }
    generation_targets.update(generator.generation_record_paths().values())
    training_grid = training.build_training_grid()
    training_targets = {
        path
        for job in training_grid
        for path in (job.checkpoint_path, job.log_path)
    }
    training_targets.update(training.training_record_paths().values())
    evaluator_targets = {
        evaluator.PREPARE_RECORD,
        evaluator.RUN_RECORD,
        *evaluator.OUTPUT_PATHS.values(),
    }
    v2_targets = {
        *generation_targets,
        *training_targets,
        *evaluator_targets,
        *analyzer.ANALYSIS_OUTPUTS,
    }
    v1_targets = {
        Path(str(path).replace("_v2", "_v1"))
        for path in v2_targets
    }

    assert len(v2_targets) == 88
    assert len(v1_targets) == 88
    assert all("_v2" in path.name for path in v2_targets)
    assert all("_v1" in path.name for path in v1_targets)
    assert v2_targets.isdisjoint(v1_targets)


def test_source_status_allowlist_never_permits_tracked_or_extra_paths(
    tmp_path,
):
    clean = generator.validate_source_status(
        (),
        stage="pool generation",
    )
    assert clean["tracked_index_head_clean"] is True
    assert clean["observed_allowed_untracked_records"] == []

    allowed = generator.generation_output_status_paths(tmp_path)
    assert len(allowed) == 18
    selected = sorted(allowed)[0]
    with pytest.raises(RuntimeError, match="source|status|allowed"):
        generator.validate_source_status(
            (f"?? {selected}",),
            stage="pool generation",
        )
    audit = generator.validate_source_status(
        (f"?? {selected}",),
        allowed_untracked=allowed,
        stage="training",
    )
    assert audit["observed_allowed_untracked_records"] == [
        f"?? {selected}"
    ]

    for records in (
        (" M journal/scripts/train_rule_construction_seed_crossing.py",),
        ("A  journal/scripts/uncommitted.py",),
        ("?? journal/results/tables/extra.csv",),
        (f" M {selected}",),
    ):
        with pytest.raises(RuntimeError, match="source|status|allowed"):
            generator.validate_source_status(
                records,
                allowed_untracked=allowed,
                stage="training",
            )


def test_long_stage_hash_snapshot_and_progress_allowlists_are_exact(tmp_path):
    expected = {
        "journal/scripts/a.py": "a" * 64,
        "journal/datasets/input.npz": "b" * 64,
    }
    audit = generator.validate_hash_snapshot(
        expected,
        dict(expected),
        stage="toy stage",
    )
    assert audit["file_count"] == 2
    assert audit["all_present"] is True
    assert audit["all_hashes_match"] is True

    with pytest.raises(RuntimeError, match="snapshot drift"):
        generator.validate_hash_snapshot(
            expected,
            {
                "journal/scripts/a.py": "c" * 64,
                "journal/datasets/input.npz": "b" * 64,
            },
            stage="toy stage",
        )
    with pytest.raises(RuntimeError, match="snapshot drift"):
        generator.validate_hash_snapshot(
            expected,
            {"journal/scripts/a.py": "a" * 64},
            stage="toy stage",
        )

    grid = training.build_training_grid(repo=tmp_path)
    before_children = training.training_output_status_paths(
        tmp_path,
        completed_jobs=(),
    )
    after_two = training.training_output_status_paths(
        tmp_path,
        completed_jobs=grid[:2],
    )
    assert len(before_children) == 19
    assert len(after_two) == 23
    assert after_two - before_children == {
        str(grid[0].checkpoint_path.relative_to(tmp_path)),
        str(grid[0].log_path.relative_to(tmp_path)),
        str(grid[1].checkpoint_path.relative_to(tmp_path)),
        str(grid[1].log_path.relative_to(tmp_path)),
    }


def test_prepare_allowlist_contains_only_declared_generation_and_training():
    allowed = evaluator._declared_prepare_upstream_paths()

    assert len(allowed) == 72
    assert (
        "journal/experiments/e15_rule_construction_crossing/"
        "pool_generation_preflight_v2.json"
    ) in allowed
    assert (
        "journal/experiments/e15_rule_construction_crossing/"
        "training_run_v2.json"
    ) in allowed
    assert all(
        path.startswith(
            (
                "journal/datasets/synthetic/",
                "journal/experiments/e15_rule_construction_crossing/",
                "journal/models/generator_extension/",
                "journal/results/logs/",
                "journal/results/tables/",
            )
        )
        for path in allowed
    )
    selected = sorted(allowed)[-1]
    generator.validate_source_status(
        (f"?? {selected}",),
        allowed_untracked=allowed,
        stage="prepare",
    )
    for record in (
        " M journal/scripts/evaluate_rule_construction_seed_crossing.py",
        "?? journal/results/tables/undeclared.csv",
    ):
        with pytest.raises(RuntimeError, match="source|status|allowed"):
            generator.validate_source_status(
                (record,),
                allowed_untracked=allowed,
                stage="prepare",
            )


def _install_prepare_snapshot_collectors(monkeypatch):
    expected = {
        "source": {
            "source_commit": "head-commit",
            "implementation": {"evaluator.py": {"sha256": "1" * 64}},
            "observed_allowed_untracked_records": [],
        },
        "frozen_inputs": {"frozen.csv": "2" * 64},
        "upstream_lineage": {"training_run": "3" * 64},
        "environment": {
            "stage": "prepare_only",
            "identity_sha256": "4" * 64,
        },
        "standardizer_hashes": {"mean_sha256": "5" * 64},
    }
    live = dict(expected)
    monkeypatch.setattr(
        evaluator,
        "source_provenance_for_prepare",
        lambda: live["source"],
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_frozen_hashes",
        lambda: live["frozen_inputs"],
    )
    monkeypatch.setattr(
        evaluator,
        "validate_pool_training_provenance",
        lambda: live["upstream_lineage"],
    )
    monkeypatch.setattr(
        evaluator,
        "evaluation_environment_record",
        lambda *, stage, torch_module=None: (
            live["environment"]
            if stage == "prepare_only" and torch_module is None
            else pytest.fail("unexpected Stage-A environment collector call")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluator.e14,
        "fit_registered_standardizer",
        lambda: (None, None, live["standardizer_hashes"]),
    )
    return expected, live


def test_stage_a_final_snapshot_helper_accepts_exact_live_replay(monkeypatch):
    expected, _ = _install_prepare_snapshot_collectors(monkeypatch)

    audit = evaluator.assert_prepare_snapshot_unchanged(**expected)

    assert audit["snapshot_unchanged"] is True


@pytest.mark.parametrize("drift", ("frozen_inputs", "upstream_lineage"))
def test_stage_a_final_snapshot_helper_rejects_artifact_drift(
    monkeypatch,
    drift,
):
    expected, live = _install_prepare_snapshot_collectors(monkeypatch)
    live[drift] = {f"changed_{drift}": "f" * 64}

    with pytest.raises(RuntimeError, match="T-STOP-MANIPULATION"):
        evaluator.assert_prepare_snapshot_unchanged(**expected)


def test_stage_a_rechecks_complete_snapshot_before_prepare_publication():
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    prepare = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_prepare_only"
    )
    snapshot_calls = [
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_prepare_snapshot_unchanged"
    ]
    publish_calls = [
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_bundle_no_clobber"
    ]
    assert len(snapshot_calls) == 1
    assert len(publish_calls) == 1
    assert snapshot_calls[0].lineno < publish_calls[0].lineno
    assert {
        keyword.arg
        for keyword in snapshot_calls[0].keywords
        if keyword.arg is not None
    } == {
        "source",
        "frozen_inputs",
        "upstream_lineage",
        "environment",
        "standardizer_hashes",
    }


def test_analysis_allowlist_accepts_only_registered_evaluator_outputs():
    evaluator_outputs = analyzer.allowed_evaluator_status_paths()

    assert len(evaluator_outputs) == 10
    selected = sorted(evaluator_outputs)[0]
    audit = analyzer.validate_analysis_source_status(
        (f"?? {selected}",),
        allowed_untracked=evaluator_outputs,
    )
    assert audit["observed_allowed_untracked_records"] == [
        f"?? {selected}"
    ]
    for record in (
        " M journal/scripts/analyze_rule_construction_seed_crossing.py",
        "?? journal/experiments/e15_rule_construction_crossing/"
        "pool_generation_run_v2.json",
        "?? journal/experiments/e15_rule_construction_crossing/"
        "training_run_v2.json",
        "?? journal/experiments/e15_rule_construction_crossing/"
        "prepare_v2.json",
        "?? journal/experiments/e15_rule_construction_crossing/"
        "pool_generation_run_v1.json",
        "?? journal/results/tables/undeclared-analysis.csv",
    ):
        with pytest.raises(RuntimeError, match="source|status|allowlist"):
            analyzer.validate_analysis_source_status(
                (record,),
                allowed_untracked=evaluator_outputs,
            )


@pytest.mark.parametrize(
    "environment",
    (None, {"stage": "prepare_only", "identity_sha256": "a" * 64}),
    ids=("missing", "non_score"),
)
def test_evaluator_handoff_rejects_missing_or_non_score_environment(
    monkeypatch,
    tmp_path,
    environment,
):
    run_record = tmp_path / "run_v1.json"
    run_record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(analyzer, "RUN_RECORD", run_record)
    monkeypatch.setattr(
        analyzer,
        "_validate_path_binding",
        lambda binding, *, path, label, expected_rows=None: {
            "path": str(path),
            "rows": expected_rows,
        },
    )
    monkeypatch.setattr(analyzer, "sha256_file", lambda path: "b" * 64)
    output_names = (
        "l4_normal",
        "l4_scenario",
        "l4_cell",
        "l4_delta",
        "runtime_continuity",
        "anchor_continuity",
        "l2_scenario",
        "l2_delta",
        "log",
    )
    run = {
        "schema_version": "e15.rule_construction_crossing_run.v1",
        "status": "T-PASS",
        "stage_status": "evaluation_complete",
        "scientific_verdict_status": "pending_registered_analyzer",
        "source_commit": "head-commit",
        "rule_inference_started_after_runtime_continuity": True,
        "runtime_continuity_gate": {"status": "PASS", "rows": 495},
        "prepare": {},
        "outputs": {name: {} for name in output_names},
        "environment": environment,
    }

    with pytest.raises(ValueError, match="environment|score"):
        analyzer.validate_evaluator_run(
            run,
            source_commit="head-commit",
        )


def test_final_manifest_rejects_missing_evaluation_environment(monkeypatch):
    pool_generation_run = {
        "pools": [
            {
                "construction_seed": seed,
                "ordered_content_sha256": f"{position:x}" * 64,
            }
            for position, seed in enumerate(
                analyzer.CONSTRUCTION_SEEDS,
                start=1,
            )
        ],
        "environment": {"stage": "pool_generation"},
    }
    records = {
        analyzer.TRAINING_RUN: {
            "bundles": [],
            "environment_before": {"stage": "training"},
            "environment_after": {"stage": "training"},
        },
        analyzer.POOL_GENERATION_RUN: pool_generation_run,
        analyzer.PREPARE_RECORD: {
            "l4_transformation": {"identity": "l4"},
            "l2_transformation": {"identity": "l2"},
        },
        analyzer.RUN_RECORD: {
            "transformations": {"match_prepare": True},
        },
    }
    monkeypatch.setattr(
        analyzer,
        "_read_json_object",
        lambda path: records[path],
    )
    monkeypatch.setattr(analyzer, "_lineage_artifact_paths", lambda: [])
    monkeypatch.setattr(
        analyzer,
        "_sampling_vector_bindings",
        lambda training_run: [],
    )

    with pytest.raises(ValueError, match="evaluation.*environment|environment"):
        analyzer.build_final_manifest(
            source={
                "source_commit": "head-commit",
                "implementation": {},
                "source_status": {},
                "wisa_clean": True,
            },
            evaluator_handoff={
                "handoff_complete": True,
                "environment": None,
            },
            verdict={
                "verdict_count": 1,
                "scientific_verdict": "C-UNRESOLVED",
            },
            runtime_continuity={"status": "PASS"},
            anchor_continuity={"status": "anchor-outcome-identical"},
            output_records={
                name: {}
                for name in (
                    "crossed_effects",
                    "hierarchical_summary",
                    "crossed_dispersion",
                    "l2_summary",
                )
            },
        )


def test_sampling_is_exact_no_replacement_and_reproducible():
    labels = np.repeat(np.asarray([1, 2, 3, 4], dtype=np.int8), 10)
    first, first_audit = training.sample_indices_without_replacement(
        labels,
        total_count=13,
        sampling_seed=307,
    )
    replay, replay_audit = training.sample_indices_without_replacement(
        labels.copy(),
        total_count=13,
        sampling_seed=307,
    )
    changed, _ = training.sample_indices_without_replacement(
        labels,
        total_count=13,
        sampling_seed=342,
    )

    np.testing.assert_array_equal(first, replay)
    assert first_audit == replay_audit
    assert not np.array_equal(first, changed)
    assert first.dtype == np.dtype("<i8")
    assert len(first) == len(np.unique(first)) == 13
    assert first_audit["total"] == {
        "requested": 13,
        "drawn": 13,
        "unique": 13,
        "repeated": 0,
    }
    assert {
        label: first_audit["per_class"][str(label)]["requested"]
        for label in (1, 2, 3, 4)
    } == {1: 4, 2: 3, 3: 3, 4: 3}
    assert all(
        first_audit["per_class"][str(label)]["repeated"] == 0
        for label in (1, 2, 3, 4)
    )


def test_sampling_rejects_insufficient_class_capacity():
    labels = np.asarray([1, 2, 2, 3, 3, 4, 4, 4], dtype=np.int8)
    with pytest.raises(ValueError, match="capacity"):
        training.sample_indices_without_replacement(
            labels,
            total_count=8,
            sampling_seed=307,
        )


def test_sampling_index_hash_is_explicit_little_endian_signed_int64():
    indices = np.asarray([0, 256, 65_535], dtype=">i8")
    expected = hashlib.sha256(
        np.ascontiguousarray(indices, dtype="<i8").tobytes(order="C")
    ).hexdigest()

    assert training.sampling_indices_sha256(indices) == expected


def test_ordered_content_digest_uses_label_byte_and_little_endian_row_hash():
    x = np.arange(2 * 128 * 11, dtype=">f4").reshape(2, 128, 11)
    labels = np.asarray([1, 4], dtype=np.int8)
    expected = hashlib.sha256()
    for row, label in zip(x, labels, strict=True):
        expected.update(bytes((int(label),)))
        expected.update(
            hashlib.sha256(
                np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")
            ).digest()
        )

    assert generator.ordered_content_digest(x, labels) == expected.hexdigest()


def test_cross_pool_overlap_is_complete_and_strictly_descriptive():
    x = _tiny_pool_arrays()["x"]
    second = x[:2].copy()
    second[1, 0, 0] += np.float32(100)
    observed = generator.cross_pool_row_overlap_audit(
        {
            314159: x[:2],
            271828: second,
            161803: x[2:4],
            141421: x[4:6],
            173205: x[6:8],
        }
    )

    assert observed["pool_count"] == 5
    assert observed["pair_count"] == 10
    assert bool(observed["descriptive_only"])
    anchor_pair = next(
        pair
        for pair in observed["pairs"]
        if (
            pair["left_construction_seed"],
            pair["right_construction_seed"],
        ) == (314159, 271828)
    )
    assert anchor_pair["exact_row_content_overlap"] == 1
    assert anchor_pair["left_overlap_fraction"] == pytest.approx(0.5)
    assert anchor_pair["right_overlap_fraction"] == pytest.approx(0.5)
    assert not bool(anchor_pair["identical_complete_digest_sets"])


def test_tiny_pool_schema_dtype_content_and_identity_gates():
    arrays = _tiny_pool_arrays()
    audit = generator.validate_pool_arrays(
        arrays,
        expected_seed=314159,
        expected_per_attack=2,
    )

    assert audit["class_counts"] == {
        "1": 2,
        "2": 2,
        "3": 2,
        "4": 2,
    }
    assert audit["content_uniqueness"]["total"]["repeated"] == 0
    assert audit["ordered_content_sha256"] == (
        generator.ordered_content_digest(
            arrays["x"],
            arrays["y_attack_type"],
        )
    )

    wrong_schema = _replace_array(
        arrays,
        "pool_schema_version",
        np.asarray("legacy_rule_pool", dtype=np.str_),
    )
    with pytest.raises(generator.E15PoolError, match="T-STOP-POOL.*schema"):
        generator.validate_pool_arrays(
            wrong_schema,
            expected_seed=314159,
            expected_per_attack=2,
        )

    wrong_seed = _replace_array(
        arrays,
        "construction_seed",
        np.asarray(271828, dtype="<u8"),
    )
    with pytest.raises(generator.E15PoolError, match="T-STOP-POOL.*seed"):
        generator.validate_pool_arrays(
            wrong_seed,
            expected_seed=314159,
            expected_per_attack=2,
        )

    wrong_dtype = _replace_array(
        arrays,
        "x",
        arrays["x"].astype("<f8"),
    )
    with pytest.raises(generator.E15PoolError, match="T-STOP-POOL.*dtype"):
        generator.validate_pool_arrays(
            wrong_dtype,
            expected_seed=314159,
            expected_per_attack=2,
        )

    duplicate = arrays["x"].copy()
    duplicate[1] = duplicate[0]
    duplicate_arrays = _replace_array(arrays, "x", duplicate)
    with pytest.raises(
        generator.E15PoolError,
        match="T-STOP-POOL.*duplicate",
    ):
        generator.validate_pool_arrays(
            duplicate_arrays,
            expected_seed=314159,
            expected_per_attack=2,
        )


@pytest.mark.parametrize(
    ("key", "dtype"),
    [
        ("y_binary", "<i2"),
        ("y_attack_type", "<i2"),
        ("condition_label", "<i2"),
        ("injection_count", "<i4"),
    ],
)
def test_tiny_pool_rejects_every_nonfrozen_numeric_dtype(key, dtype):
    arrays = _tiny_pool_arrays()
    arrays[key] = arrays[key].astype(dtype)

    with pytest.raises(generator.E15PoolError, match="T-STOP-POOL.*dtype"):
        generator.validate_pool_arrays(
            arrays,
            expected_seed=314159,
            expected_per_attack=2,
        )


def test_anchor_gate_is_array_exact_but_string_dtype_agnostic(monkeypatch):
    historical = _tiny_pool_arrays()
    historical["synthetic_type"] = historical["synthetic_type"].astype(object)
    historical["feature_names"] = historical["feature_names"].astype(object)
    candidate = _tiny_pool_arrays()
    monkeypatch.setattr(
        generator,
        "LEGACY_ORDERED_CONTENT_SHA256",
        generator.ordered_content_digest(
            candidate["x"],
            candidate["y_attack_type"],
        ),
    )

    result = generator.validate_anchor_arrays(candidate, historical)
    assert result["status"] == "anchor-array-equivalent"

    drifted = {key: value.copy() for key, value in candidate.items()}
    drifted["x"][0, 0, 2] += np.float32(1)
    with pytest.raises(generator.E15PoolError, match="T-STOP-POOL.*anchor"):
        generator.validate_anchor_arrays(drifted, historical)


def test_pool_validation_requires_explicit_e15_identity_without_fallback():
    arrays = _tiny_pool_arrays()
    for missing in (
        "pool_identifier",
        "pool_schema_version",
        "construction_seed",
        "downstream_sampling_policy",
    ):
        incomplete = {
            key: value
            for key, value in arrays.items()
            if key != missing
        }
        with pytest.raises(generator.E15PoolError, match="T-STOP-POOL"):
            generator.validate_pool_arrays(
                incomplete,
                expected_seed=314159,
                expected_per_attack=2,
            )


def test_atomic_bundle_uses_lexists_for_dangling_symlink(tmp_path):
    staged_pool = tmp_path / "staged-pool"
    staged_stats = tmp_path / "staged-stats"
    staged_pool.write_bytes(b"pool")
    staged_stats.write_bytes(b"stats")
    pool_target = tmp_path / "pool.npz"
    stats_target = tmp_path / "stats.csv"
    pool_target.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError, match="overwrite|exists"):
        generator.atomic_publish_bundle(
            (
                (staged_pool, pool_target),
                (staged_stats, stats_target),
            )
        )

    assert pool_target.is_symlink()
    assert not os.path.lexists(stats_target)


def test_atomic_bundle_rolls_back_only_its_own_links_on_race(
    tmp_path,
    monkeypatch,
):
    staged_pool = tmp_path / "staged-pool"
    staged_stats = tmp_path / "staged-stats"
    staged_pool.write_bytes(b"pool")
    staged_stats.write_bytes(b"stats")
    pool_target = tmp_path / "pool.npz"
    stats_target = tmp_path / "stats.csv"
    real_link = os.link
    calls = 0

    def racing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(target).write_bytes(b"racer")
        return real_link(source, target)

    monkeypatch.setattr(generator.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="appeared|exists"):
        generator.atomic_publish_bundle(
            (
                (staged_pool, pool_target),
                (staged_stats, stats_target),
            )
        )

    assert not os.path.lexists(pool_target)
    assert stats_target.read_bytes() == b"racer"


def test_training_grid_has_exactly_50_absent_targets(tmp_path):
    grid = training.build_training_grid(repo=tmp_path)
    targets = training.training_target_paths(grid)

    assert len(targets) == 50
    assert len(set(targets)) == 50
    assert sum(path.suffix == ".pt" for path in targets) == 25
    assert sum(path.suffix == ".log" for path in targets) == 25
    training.assert_training_targets_absent(targets)

    collision = targets[-1]
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="50|target|exists"):
        training.assert_training_targets_absent(targets)


def test_prepare_only_guard_prohibits_inference_and_predictions():
    evaluator.assert_prepare_only_inference_guard(
        inference_calls=0,
        prediction_rows=0,
    )
    with pytest.raises(RuntimeError, match="T-STOP-MANIPULATION"):
        evaluator.assert_prepare_only_inference_guard(
            inference_calls=1,
            prediction_rows=0,
        )
    with pytest.raises(RuntimeError, match="T-STOP-MANIPULATION"):
        evaluator.assert_prepare_only_inference_guard(
            inference_calls=0,
            prediction_rows=1,
        )


def test_prepare_only_import_and_local_call_graph_are_inference_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(SCRIPTS)!r});"
                "import evaluate_rule_construction_seed_crossing;"
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=True,
    )
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = {
        name: {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
        }
        for name, node in functions.items()
    }
    reachable: set[str] = set()
    frontier = ["run_prepare_only"]
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        frontier.extend(calls.get(name, set()) & functions.keys())

    assert {
        "run_score",
        "_run_score_production",
        "predict",
        "_predict",
        "load_rule_checkpoint_for_inference",
    }.isdisjoint(reachable)
    for name in reachable:
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(
                node.func,
                ast.Attribute,
            ):
                assert node.func.attr not in {
                    "record_prediction",
                    "record_rule_load",
                    "predict",
                    "forward",
                }


def test_rule_checkpoint_load_is_blocked_before_runtime_continuity():
    ledger = evaluator.InferenceLedger()

    with pytest.raises(RuntimeError, match="T-STOP-RUNTIME"):
        ledger.record_rule_load()

    assert ledger.rule_checkpoint_loads == 0
    assert ledger.inference_calls == 0
    assert ledger.prediction_rows == 0


def test_rule_loader_does_not_touch_checkpoint_before_runtime_gate(
    monkeypatch,
):
    calls = []

    def fake_load(torch, path, expected_sha256):
        calls.append((torch, path, expected_sha256))
        return "loaded-model"

    monkeypatch.setattr(
        evaluator,
        "_load_checkpoint_for_inference",
        fake_load,
    )
    ledger = evaluator.InferenceLedger()
    binding = {"sha256": "a" * 64}

    with pytest.raises(RuntimeError, match="T-STOP-RUNTIME"):
        evaluator.load_rule_checkpoint_for_inference(
            object(),
            construction_seed=271828,
            pipeline_seed=7,
            binding=binding,
            ledger=ledger,
        )
    assert calls == []
    assert ledger.rule_checkpoint_loads == 0

    ledger.runtime_gate_passed = True
    observed = evaluator.load_rule_checkpoint_for_inference(
        "torch-sentinel",
        construction_seed=271828,
        pipeline_seed=7,
        binding=binding,
        ledger=ledger,
    )
    assert observed == "loaded-model"
    assert ledger.rule_checkpoint_loads == 1
    assert calls[0][0] == "torch-sentinel"
    assert calls[0][2] == "a" * 64


def test_real_runtime_continuity_is_integer_exact_and_hard_stops():
    inputs = _continuity_inputs()
    continuity = evaluator.build_real_runtime_continuity(*inputs)

    assert len(continuity) == 2
    assert set(continuity["row_scope"]) == {"scenario", "normal"}
    assert not continuity.duplicated(list(evaluator.CONTINUITY_KEY)).any()
    for column in (
        "exact_correct_difference",
        "binary_correct_difference",
        "normal_correct_difference",
    ):
        assert (
            pd.to_numeric(continuity[column], errors="coerce")
            .dropna()
            .eq(0)
            .all()
        )
    evaluator.require_real_runtime_continuity(continuity)

    drifted_frozen = inputs[1].copy()
    drifted_frozen.loc[0, "exact_correct"] += 1
    drifted = evaluator.build_real_runtime_continuity(
        inputs[0],
        drifted_frozen,
        inputs[2],
        inputs[3],
    )
    with pytest.raises(RuntimeError, match="T-STOP-RUNTIME"):
        evaluator.require_real_runtime_continuity(drifted)


def test_runtime_first_orchestrator_never_calls_rule_operation_on_drift():
    inputs = _continuity_inputs()
    continuity = evaluator.build_real_runtime_continuity(*inputs)
    drifted = continuity.copy()
    scenario = drifted["row_scope"].eq("scenario")
    drifted.loc[scenario, "exact_correct_difference"] = 1
    drifted.loc[scenario, "status"] = "runtime-drift"
    failed_ledger = evaluator.InferenceLedger()
    failed_events = []

    with pytest.raises(RuntimeError, match="T-STOP-RUNTIME"):
        evaluator.run_rule_after_runtime_continuity(
            drifted,
            ledger=failed_ledger,
            expected_rows=2,
            continuity_publish=lambda: failed_events.append("publish"),
            rule_operation=lambda: failed_events.append("rule"),
        )
    assert failed_events == []
    assert failed_ledger.runtime_gate_passed is False
    assert failed_ledger.rule_checkpoint_loads == 0

    passed_ledger = evaluator.InferenceLedger()
    passed_events = []

    def publish():
        passed_events.append("publish")

    def rule_operation():
        assert passed_ledger.runtime_gate_passed is True
        passed_ledger.record_rule_load()
        passed_events.append("rule")
        return "rule-result"

    result, gate = evaluator.run_rule_after_runtime_continuity(
        continuity,
        ledger=passed_ledger,
        expected_rows=2,
        continuity_publish=publish,
        rule_operation=rule_operation,
    )
    assert result == "rule-result"
    assert gate["status"] == "T-PASS"
    assert passed_events == ["publish", "rule"]
    assert passed_ledger.rule_checkpoint_loads == 1


def test_production_runtime_gate_accepts_only_full_registered_495_grid():
    continuity = _registered_runtime_continuity()
    ledger = evaluator.InferenceLedger()
    events = []

    result, gate = evaluator.run_rule_after_runtime_continuity(
        continuity,
        ledger=ledger,
        expected_rows=evaluator.EXPECTED_RUNTIME_CONTINUITY_ROWS,
        continuity_publish=lambda: events.append("publish"),
        rule_operation=lambda: events.append("rule") or "complete",
    )

    assert result == "complete"
    assert gate["rows"] == 495
    assert gate["scenario_rows"] == 480
    assert gate["normal_rows"] == 15
    assert events == ["publish", "rule"]


@pytest.mark.parametrize(
    "defect",
    ("wrong_scope_composition", "duplicate_axis_substitution"),
)
def test_production_runtime_grid_defects_stop_before_rule_operation(defect):
    continuity = _registered_runtime_continuity()
    scenario_index = continuity.index[
        continuity["row_scope"].eq("scenario")
    ][-1]
    if defect == "wrong_scope_composition":
        continuity.loc[scenario_index, "row_scope"] = "normal"
    else:
        continuity.loc[scenario_index, "pipeline_seed"] = (
            evaluator.PIPELINE_SEEDS[0]
        )
        continuity.loc[scenario_index, "block_id"] = "block_substitute"

    ledger = evaluator.InferenceLedger()
    events = []
    with pytest.raises(RuntimeError, match="T-STOP-RUNTIME"):
        evaluator.run_rule_after_runtime_continuity(
            continuity,
            ledger=ledger,
            expected_rows=evaluator.EXPECTED_RUNTIME_CONTINUITY_ROWS,
            continuity_publish=lambda: events.append("publish"),
            rule_operation=lambda: events.append("rule"),
        )
    assert events == []
    assert ledger.runtime_gate_passed is False
    assert ledger.rule_checkpoint_loads == 0


def test_production_scorer_routes_rule_loads_through_runtime_first_boundary():
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    production = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_score_production"
    )
    rule_phase = next(
        node
        for node in production.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "score_rule_phase"
    )

    boundary_calls = [
        node
        for node in ast.walk(production)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_rule_after_runtime_continuity"
    ]
    assert len(boundary_calls) == 1
    boundary = boundary_calls[0]
    assert len(boundary.args) == 1
    assert isinstance(boundary.args[0], ast.Name)
    assert boundary.args[0].id == "continuity"

    keywords = {
        keyword.arg: keyword.value
        for keyword in boundary.keywords
        if keyword.arg is not None
    }
    assert set(keywords) == {
        "ledger",
        "rule_operation",
        "expected_rows",
        "continuity_publish",
    }
    assert isinstance(keywords["ledger"], ast.Name)
    assert keywords["ledger"].id == "ledger"
    assert isinstance(keywords["rule_operation"], ast.Name)
    assert keywords["rule_operation"].id == "score_rule_phase"
    assert isinstance(keywords["expected_rows"], ast.Name)
    assert (
        keywords["expected_rows"].id
        == "EXPECTED_RUNTIME_CONTINUITY_ROWS"
    )
    assert isinstance(keywords["continuity_publish"], ast.Lambda)
    publish_calls = [
        node
        for node in ast.walk(keywords["continuity_publish"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_bundle_no_clobber"
    ]
    assert len(publish_calls) == 1
    assert "runtime_continuity" in ast.unparse(publish_calls[0])

    rule_load_calls = [
        node
        for node in ast.walk(production)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_rule_models"
    ]
    assert len(rule_load_calls) == 1
    rule_load = rule_load_calls[0]
    assert rule_phase.lineno < rule_load.lineno <= rule_phase.end_lineno
    assert [
        argument.id
        for argument in rule_load.args
        if isinstance(argument, ast.Name)
    ] == ["torch", "prepare", "ledger"]
    assert not [
        node
        for node in ast.walk(production)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "score_rule_phase"
    ]
    assert boundary.lineno > rule_phase.end_lineno

    run_assignments = [
        node
        for node in production.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "run"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    assert len(run_assignments) == 1
    run_payload = {
        key.value: value
        for key, value in zip(
            run_assignments[0].value.keys,
            run_assignments[0].value.values,
            strict=True,
        )
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
    }
    assert "environment" in run_payload
    environment_value = run_payload["environment"]
    assert not (
        isinstance(environment_value, ast.Constant)
        and environment_value.value is None
    )

    score_environment_calls = [
        node
        for node in ast.walk(production)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluation_environment_record"
        and any(
            keyword.arg == "stage"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "score"
            for keyword in node.keywords
        )
    ]
    assert len(score_environment_calls) == 1
    environment_rechecks = [
        node
        for node in ast.walk(production)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_score_environment_unchanged"
    ]
    assert len(environment_rechecks) == 1
    if isinstance(environment_value, ast.Name):
        environment_assignments = [
            node
            for node in production.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == environment_value.id
                for target in node.targets
            )
        ]
        assert len(environment_assignments) == 1
        assert any(
            call in list(ast.walk(environment_assignments[0].value))
            for call in score_environment_calls
        )
    else:
        assert any(
            call in list(ast.walk(environment_value))
            for call in score_environment_calls
        )


def test_registered_l4_l2_row_counts_and_unique_keys_are_exact():
    assert evaluator.EXPECTED_L4_ROWS == {
        "normal": 75,
        "scenario": 2_400,
        "cell": 1_200,
        "delta": 2_400,
        "crossed_effects": 10_800,
        "hierarchical": 2_160,
        "crossed_dispersion": 288,
        "runtime_continuity": 495,
        "anchor_continuity": 495,
    }
    assert evaluator.EXPECTED_L2_ROWS == {
        "by_scenario": 1_170,
        "delta": 900,
        "summary": 360,
    }
    assert analyzer.EXPECTED_CROSSED_EFFECT_ROWS == 10_800
    assert analyzer.EXPECTED_HIERARCHICAL_ROWS == 2_160
    assert analyzer.EXPECTED_CROSSED_DISPERSION_ROWS == 288
    assert analyzer.EXPECTED_L2_SUMMARY_ROWS == 360
    assert tuple(analyzer.L2_SUMMARY_KEY) == (
        "endpoint",
        "setting",
        "attack_scope",
        "row_type",
        "unit_id",
    )
    assert tuple(evaluator.CONTINUITY_KEY) == (
        "row_scope",
        "pipeline_seed",
        "block_id",
        "attack",
        "id_stratum",
        "cell",
    )

    unique = pd.DataFrame(
        [
            {"construction_seed": 271828, "pipeline_seed": 7},
            {"construction_seed": 271828, "pipeline_seed": 42},
        ]
    )
    evaluator.require_unique(
        unique,
        ("construction_seed", "pipeline_seed"),
        expected=2,
        label="toy crossed keys",
    )
    duplicate = pd.concat([unique, unique.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate|unique"):
        evaluator.require_unique(
            duplicate,
            ("construction_seed", "pipeline_seed"),
            expected=3,
            label="toy crossed keys",
        )


def test_l4_validator_accepts_only_the_complete_registered_cartesian_grid():
    normal = pd.DataFrame(
        [
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "block_id": block,
                "n": 2000,
                "normal_recall": 0.99,
                "fpr": 0.01,
            }
            for construction, pipeline, block in itertools.product(
                CONSTRUCTION_SEEDS,
                PIPELINE_SEEDS,
                evaluator.BLOCKS,
            )
        ]
    )
    scenario = pd.DataFrame(
        [
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "block_id": block,
                "attack": attack,
                "id_stratum": id_stratum,
                "P": p,
                "S": s,
                "D": d,
                "n": 2000,
                "exact_recall": 0.55,
                "binary_recall": 0.75,
            }
            for (
                construction,
                pipeline,
                block,
                attack,
                id_stratum,
                p,
                s,
                d,
            ) in itertools.product(
                CONSTRUCTION_SEEDS,
                PIPELINE_SEEDS,
                evaluator.BLOCKS,
                evaluator.ATTACKS,
                evaluator.ID_STRATA,
                (0, 1),
                (0, 1),
                (0, 1),
            )
        ]
    )
    cell = pd.DataFrame(
        [
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "block_id": block,
                "id_stratum": id_stratum,
                "P": p,
                "S": s,
                "D": d,
                "n": 2000,
                "exact_recall": 0.55,
                "binary_recall": 0.75,
            }
            for (
                construction,
                pipeline,
                block,
                id_stratum,
                p,
                s,
                d,
            ) in itertools.product(
                CONSTRUCTION_SEEDS,
                PIPELINE_SEEDS,
                evaluator.BLOCKS,
                evaluator.ID_STRATA,
                (0, 1),
                (0, 1),
                (0, 1),
            )
        ]
    )
    delta = scenario.copy()
    delta["exact_recall"] = -0.20
    delta["binary_recall"] = -0.10

    counts = evaluator.validate_l4_scoring_tables(
        normal,
        scenario,
        cell,
        delta,
    )
    assert counts == {
        "checkpoint_evaluations": 25,
        "normal_rows": 75,
        "scenario_rows": 2_400,
        "cell_rows": 1_200,
        "delta_rows": 2_400,
        "transformed_rule_forward_evaluations": 4_800_000,
        "clean_rule_forward_evaluations": 150_000,
    }

    with pytest.raises(ValueError, match="row count|Cartesian"):
        evaluator.validate_l4_scoring_tables(
            normal,
            scenario.iloc[:-1].copy(),
            cell,
            delta,
        )


def test_l2_validator_accepts_all_30_checkpoints_and_three_settings():
    scenario_ids = ("Normal_normal",) + tuple(
        f"{attack}_{setting}"
        for attack in evaluator.L2_ATTACKS
        for setting in ("low", "medium", "high")
    )
    rows = []
    for construction, pipeline, block, scenario_id in itertools.product(
        CONSTRUCTION_SEEDS,
        PIPELINE_SEEDS,
        evaluator.BLOCKS,
        scenario_ids,
    ):
        is_normal = scenario_id == "Normal_normal"
        rows.append(
            {
                "arm": "rule",
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "block_id": block,
                "scenario_id": scenario_id,
                "n": 6000 if is_normal else 500,
                "normal_recall": 0.99 if is_normal else np.nan,
                "fpr": 0.01 if is_normal else np.nan,
                "exact_recall": np.nan if is_normal else 0.50,
                "binary_recall": np.nan if is_normal else 0.70,
            }
        )
    for pipeline, block, scenario_id in itertools.product(
        PIPELINE_SEEDS,
        evaluator.BLOCKS,
        scenario_ids,
    ):
        is_normal = scenario_id == "Normal_normal"
        rows.append(
            {
                "arm": "shared_real",
                "construction_seed": "shared_reference",
                "pipeline_seed": pipeline,
                "block_id": block,
                "scenario_id": scenario_id,
                "n": 6000 if is_normal else 500,
                "normal_recall": 0.98 if is_normal else np.nan,
                "fpr": 0.02 if is_normal else np.nan,
                "exact_recall": np.nan if is_normal else 0.45,
                "binary_recall": np.nan if is_normal else 0.65,
            }
        )
    by_scenario = pd.DataFrame(rows)
    delta = pd.DataFrame(
        [
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "block_id": block,
                "attack": attack,
                "setting": setting,
                "exact_recall": 0.05,
                "binary_recall": 0.05,
            }
            for (
                construction,
                pipeline,
                block,
                attack,
                setting,
            ) in itertools.product(
                CONSTRUCTION_SEEDS,
                PIPELINE_SEEDS,
                evaluator.BLOCKS,
                evaluator.L2_ATTACKS,
                ("low", "medium", "high"),
            )
        ]
    )

    counts = evaluator.validate_l2_scoring_tables(by_scenario, delta)
    assert counts == {
        "checkpoint_evaluations": 30,
        "scenario_rows": 1_170,
        "delta_rows": 900,
        "forward_evaluations": 1_080_000,
    }

    duplicated = pd.concat(
        [by_scenario.iloc[:-1], by_scenario.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate|scenario"):
        evaluator.validate_l2_scoring_tables(duplicated, delta)


def test_n4_t_summary_uses_df3_and_untruncated_interval():
    values = np.asarray([0.0, 0.1, 0.2, 0.3])
    observed = analyzer.t_summary_n4(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=3))
    half_width = critical * sd / np.sqrt(4)
    expected_p = float(
        2 * student_t.sf(abs(mean / (sd / np.sqrt(4))), df=3)
    )

    assert observed["n_available"] == 4
    assert observed["mean"] == pytest.approx(mean)
    assert observed["sd"] == pytest.approx(sd)
    assert observed["ci_low"] == pytest.approx(mean - half_width)
    assert observed["ci_high"] == pytest.approx(mean + half_width)
    assert observed["ci_low"] < 0
    assert observed["test_p"] == pytest.approx(expected_p)
    assert observed["holm_input_p"] == pytest.approx(expected_p)
    assert observed["strict_positive"] == 3
    assert observed["strict_negative"] == 0

    with pytest.raises(ValueError, match="four|n=4"):
        analyzer.t_summary_n4([0.0, 0.1, 0.2, 0.3, 0.4])


def test_factorial_operators_match_the_frozen_e14_scaling():
    cube = {
        (p, s, d): (
            2 * p
            + 3 * s
            + 4 * d
            + 5 * p * s
            + 6 * p * d
            + 7 * s * d
            + 8 * p * s * d
        )
        for p, s, d in itertools.product((0, 1), repeat=3)
    }
    expected = {
        "P": 9.5,
        "S": 11.0,
        "D": 12.5,
        "PS": 9.0,
        "PD": 10.0,
        "SD": 11.0,
        "PSD": 8.0,
        "corner_111_minus_000": 35.0,
    }

    assert {
        effect: analyzer.factorial_effect(cube, effect)
        for effect in analyzer.EFFECTS
    } == pytest.approx(expected)


def test_degenerate_n4_summary_is_retained_for_holm_bookkeeping_only():
    observed = analyzer.t_summary_n4([0.25, 0.25, 0.25, 0.25])

    assert observed["mean"] == pytest.approx(0.25)
    assert bool(observed["degenerate"])
    assert observed["p_status"] == "bookkeeping_p"
    assert observed["holm_input_p"] == pytest.approx(1.0)
    assert np.isnan(observed["sd"])
    assert np.isnan(observed["ci_low"])
    assert np.isnan(observed["ci_high"])
    assert np.isnan(observed["test_p"])


def test_exact_and_binary_use_separate_seven_effect_holm_families():
    exact_p = [0.001, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060]
    binary_p = [0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800]
    adjusted = analyzer.apply_primary_inference(
        _primary_rows(exact_p, binary_p)
    )
    exact = adjusted[
        adjusted["endpoint"].eq("exact_recall")
    ].set_index("effect").loc[list(analyzer.FACTORIAL_EFFECTS)]
    binary = adjusted[
        adjusted["endpoint"].eq("binary_recall")
    ].set_index("effect").loc[list(analyzer.FACTORIAL_EFFECTS)]

    np.testing.assert_allclose(
        exact["holm_adjusted_p"],
        analyzer.holm_adjust(exact_p),
    )
    np.testing.assert_allclose(
        binary["holm_adjusted_p"],
        analyzer.holm_adjust(binary_p),
    )
    assert exact["holm_family"].nunique() == 1
    assert binary["holm_family"].nunique() == 1
    assert exact["holm_family"].iloc[0] != binary["holm_family"].iloc[0]
    assert exact["holm_adjusted_p"].iloc[0] == pytest.approx(0.007)


def test_verdict_and_heterogeneity_modifiers_follow_frozen_rules():
    adjusted = analyzer.apply_primary_inference(
        _primary_rows(
            [0.0001, 0.5, 0.5, 0.0002, 0.5, 0.5, 0.5],
            [0.5] * 7,
        )
    )
    adjusted["pipeline_margin_directionally_concordant"] = False
    p_mask = (
        adjusted["endpoint"].eq("exact_recall")
        & adjusted["effect"].eq("P")
    )
    adjusted.loc[p_mask, "pipeline_margin_directionally_concordant"] = True
    verdict = analyzer.build_scientific_verdict(
        adjusted,
        construction_heterogeneous_effects=("PS",),
        pipeline_heterogeneous_effects=("PS",),
        attack_heterogeneous_effects=("PS",),
    )

    assert verdict["verdict_count"] == 1
    assert verdict["base_verdict"] == "C-MAIN-AND-INTERACTION"
    assert verdict["p_direction_status"] == "P-BOTH-MARGINS-CONCORDANT"
    assert verdict["resolved_exact_effects"] == ["P", "PS"]
    assert verdict["construction_sign_heterogeneous_effects"] == ["PS"]
    assert verdict["pipeline_margin_heterogeneous_effects"] == ["PS"]
    assert verdict["attack_heterogeneous_effects"] == ["PS"]
    assert verdict["scientific_verdict"] == (
        "C-MAIN-AND-INTERACTION"
        "+CONSTRUCTION-SIGN-HETEROGENEOUS(PS)"
        "+PIPELINE-MARGIN-HETEROGENEOUS(PS)"
        "+ATTACK-HETEROGENEOUS(PS)"
    )

    mixed_input = _primary_rows(
        [0.0001, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5] * 7,
    )
    mixed_p = (
        mixed_input["endpoint"].eq("exact_recall")
        & mixed_input["effect"].eq("P")
    )
    mixed_input.loc[mixed_p, ["strict_positive", "strict_negative"]] = [2, 2]
    mixed = analyzer.build_scientific_verdict(
        analyzer.apply_primary_inference(mixed_input)
    )
    assert mixed["base_verdict"] == "C-MIXED"
    assert mixed["p_direction_status"] == "P-MIXED"
    assert mixed["resolved_exact_effects"] == []


def test_finite_grid_dispersion_matches_frozen_crossed_formulas():
    rows = [
        {
            "construction_seed": construction,
            "pipeline_seed": pipeline,
            "value": 10.0 * construction_index + pipeline_index,
        }
        for construction_index, construction in enumerate(
            NEW_CONSTRUCTION_SEEDS
        )
        for pipeline_index, pipeline in enumerate(PIPELINE_SEEDS)
    ]
    observed = analyzer.finite_grid_dispersion(
        pd.DataFrame(rows),
        scope="new4",
    )

    assert observed["G"] == 4
    assert observed["P"] == 5
    assert observed["ss_construction"] == pytest.approx(2500.0)
    assert observed["ss_pipeline"] == pytest.approx(40.0)
    assert observed["ss_interaction"] == pytest.approx(0.0, abs=1e-12)
    assert observed["ss_total"] == pytest.approx(2540.0)
    assert observed["rms_interaction"] == pytest.approx(0.0, abs=1e-12)
    assert observed["share_construction"] == pytest.approx(2500 / 2540)
    assert observed["share_pipeline"] == pytest.approx(40 / 2540)
    assert observed["share_interaction"] == pytest.approx(0.0, abs=1e-12)
    assert (
        observed["share_construction"]
        + observed["share_pipeline"]
        + observed["share_interaction"]
    ) == pytest.approx(1.0)


def test_finite_grid_degenerate_total_dispersion_has_na_shares():
    frame = pd.DataFrame(
        [
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "value": 1.0,
            }
            for construction in NEW_CONSTRUCTION_SEEDS
            for pipeline in PIPELINE_SEEDS
        ]
    )
    observed = analyzer.finite_grid_dispersion(frame, scope="new4")

    assert observed["status"] == "degenerate_total_dispersion"
    assert observed["ss_total"] == pytest.approx(0.0)
    assert np.isnan(observed["share_construction"])
    assert np.isnan(observed["share_pipeline"])
    assert np.isnan(observed["share_interaction"])


def test_finite_grid_rejects_duplicate_or_incomplete_cells():
    frame = pd.DataFrame(
        [
            {
                "construction_seed": construction,
                "pipeline_seed": pipeline,
                "value": 1.0,
            }
            for construction in NEW_CONSTRUCTION_SEEDS
            for pipeline in PIPELINE_SEEDS
        ]
    )
    with pytest.raises(ValueError, match="duplicate|grid"):
        analyzer.finite_grid_dispersion(
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            scope="new4",
        )
    with pytest.raises(ValueError, match="complete|grid|20"):
        analyzer.finite_grid_dispersion(
            frame.iloc[:-1].copy(),
            scope="new4",
        )
