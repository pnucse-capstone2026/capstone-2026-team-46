from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_l4_counterfactual_factorial as evaluator  # noqa: E402
import evaluate_evaluation_realization as realization  # noqa: E402
import analyze_l4_counterfactual_factorial as analyzer  # noqa: E402
import run_e14_matched_real_training as training_wrapper  # noqa: E402
import train_generator_extension_cnn as trainer  # noqa: E402


def raw_matched_real_transcript(
    *,
    stdout: bytes = b"ordinary output\n",
    stderr: bytes = b"",
    returncode: int = 0,
) -> bytes:
    return b"".join(
        (
            b"E14 matched-real frozen trainer transcript v1\n",
            f"returncode={returncode}\n".encode("ascii"),
            f"stdout_bytes={len(stdout)}\n".encode("ascii"),
            f"stderr_bytes={len(stderr)}\n".encode("ascii"),
            b"\n--- stdout ---\n",
            stdout,
            b"\n--- stderr ---\n",
            stderr,
        )
    )


def transcript_parsers():
    return (
        evaluator.parse_matched_real_transcript,
        analyzer._parse_matched_real_transcript,
    )


def toy_base() -> np.ndarray:
    base = np.zeros(evaluator.WINDOW_SHAPE, dtype=np.float32)
    base[:, 0] = 0x123
    base[:, 1] = 8
    ranks = np.arange(128, dtype=np.float32)
    for payload_index, column in enumerate(range(2, 10)):
        base[:, column] = (ranks * (payload_index + 3) + payload_index) % 256
    base[:, 10] = np.linspace(0, 0.01, 128, dtype=np.float32)
    return base


def latent_for(attack_label: int = 3):
    return evaluator.build_latent(
        toy_base(),
        block_id="block_01",
        block_number=1,
        block_position=17,
        test_window_index=1234,
        attack_label=attack_label,
    )


def transform(
    latent,
    *,
    p: int,
    s: int,
    d: int,
    id_stratum: str = "canonical",
) -> np.ndarray:
    return evaluator.apply_factorial_transform(
        toy_base(),
        latent,
        attack_label=latent.attack_label,
        id_stratum=id_stratum,
        p=p,
        s=s,
        d=d,
    )


def test_child_seed_is_order_independent_and_stream_specific():
    first = evaluator.derive_child_seed(1, 1234, 3, 1)
    repeated = evaluator.derive_child_seed(1, 1234, 3, 1)
    other_stream = evaluator.derive_child_seed(1, 1234, 3, 2)
    other_attack = evaluator.derive_child_seed(1, 1234, 4, 1)

    assert first == repeated
    assert len({first, other_stream, other_attack}) == 3
    assert 0 <= first <= np.iinfo(np.uint64).max


@pytest.mark.parametrize(
    ("attack_label", "mode", "target_id"),
    [(3, "gear", 0x43F), (4, "rpm", 0x316)],
)
def test_111_is_byte_exact_to_frozen_l4_oracle(
    attack_label, mode, target_id
):
    base = toy_base()
    latent = latent_for(attack_label)
    observed = evaluator.apply_factorial_transform(
        base,
        latent,
        attack_label=attack_label,
        id_stratum="canonical",
        p=1,
        s=1,
        d=1,
    )

    expected = base.copy()
    rng = np.random.default_rng(latent.l4_child_seed)
    count = realization.inject_l4(rng, expected, target_id, mode)

    assert count == 32
    assert np.array_equal(observed, expected)


@pytest.mark.parametrize("attack_label", [3, 4])
@pytest.mark.parametrize("id_stratum", evaluator.ID_STRATA)
def test_000_is_byte_exact_to_independent_rule_d0_oracle(
    attack_label, id_stratum
):
    base = toy_base()
    latent = latent_for(attack_label)
    observed = evaluator.apply_factorial_transform(
        base,
        latent,
        attack_label=attack_label,
        id_stratum=id_stratum,
        p=0,
        s=0,
        d=0,
    )
    spec = evaluator.ATTACK_SPECS[attack_label]
    expected = evaluator.conditional_rule_anchor_d0_oracle(
        base,
        block_number=latent.block_number,
        test_window_index=latent.test_window_index,
        attack_label=attack_label,
        target_id=spec[f"{id_stratum}_id"],
    )

    assert np.array_equal(observed, expected)


def test_independent_rule_d0_oracle_rejects_mutated_latent_role():
    latent = latent_for(3)
    mutated_role = latent.d0_roles[0].copy()
    mutated_role[0] = (mutated_role[0] + 1) % 256
    mutated = replace(
        latent,
        d0_roles=(mutated_role, *latent.d0_roles[1:]),
    )

    with pytest.raises(
        AssertionError,
        match="cell 000 is not byte-exact",
    ):
        evaluator.validate_factor_relations(toy_base(), mutated)


@pytest.mark.parametrize("attack_label", [3, 4])
def test_canonical_shifted_pair_differs_only_in_injected_can_id(attack_label):
    latent = latent_for(attack_label)
    canonical = transform(latent, p=1, s=1, d=1, id_stratum="canonical")
    shifted = transform(latent, p=1, s=1, d=1, id_stratum="shifted")
    positions = latent.s1_positions

    difference = canonical != shifted
    expected = np.zeros_like(difference)
    expected[positions, 0] = True
    assert np.array_equal(difference, expected)


@pytest.mark.parametrize("attack_label", [3, 4])
@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize("s", [0, 1])
@pytest.mark.parametrize("d", [0, 1])
def test_every_cell_uses_shared_rank_roles_and_exactly_32_frames(
    attack_label, p, s, d
):
    base = toy_base()
    latent = latent_for(attack_label)
    observed = evaluator.apply_factorial_transform(
        base,
        latent,
        attack_label=attack_label,
        id_stratum="canonical",
        p=p,
        s=s,
        d=d,
    )
    spec = evaluator.ATTACK_SPECS[attack_label]
    positions = latent.s0_positions if s == 0 else latent.s1_positions
    columns = spec["p0_columns"] if p == 0 else spec["p1_columns"]
    roles = latent.d0_roles if d == 0 else latent.d1_roles

    assert len(np.unique(positions)) == 32
    for column, role in zip(columns, roles, strict=True):
        np.testing.assert_array_equal(observed[positions, column], role)
    np.testing.assert_array_equal(
        observed[positions, 10],
        np.maximum(base[positions, 10], np.float32(1e-5)),
    )


@pytest.mark.parametrize("attack_label", [3, 4])
def test_one_factor_p_and_d_changes_are_field_local(attack_label):
    latent = latent_for(attack_label)
    spec = evaluator.ATTACK_SPECS[attack_label]
    positions = latent.s0_positions

    p0 = transform(latent, p=0, s=0, d=0)
    p1 = transform(latent, p=1, s=0, d=0)
    allowed_p = np.zeros_like(p0, dtype=bool)
    allowed_p[
        np.ix_(
            positions,
            np.asarray(
                sorted(set(spec["p0_columns"]) | set(spec["p1_columns"]))
            ),
        )
    ] = True
    assert not np.any((p0 != p1) & ~allowed_p)

    d0 = transform(latent, p=0, s=0, d=0)
    d1 = transform(latent, p=0, s=0, d=1)
    allowed_d = np.zeros_like(d0, dtype=bool)
    allowed_d[np.ix_(positions, np.asarray(spec["p0_columns"]))] = True
    assert not np.any((d0 != d1) & ~allowed_d)


def test_rpm_d0_third_role_is_frozen_from_s0_data2():
    base = toy_base()
    latent = latent_for(4)
    jitter = latent.d0_jitters[0]
    expected = np.clip(
        base[latent.s0_positions, 4] + jitter, 0, 255
    ).astype(np.float32)

    np.testing.assert_array_equal(latent.d0_roles[2], expected)
    for s in (0, 1):
        observed = evaluator.apply_factorial_transform(
            base,
            latent,
            attack_label=4,
            id_stratum="canonical",
            p=1,
            s=s,
            d=0,
        )
        positions = latent.s0_positions if s == 0 else latent.s1_positions
        np.testing.assert_array_equal(observed[positions, 9], expected)


def test_transform_digest_is_repeatable_and_key_sensitive():
    latent = latent_for(3)
    window = transform(latent, p=1, s=1, d=1)
    key = evaluator.transformation_key(latent, "canonical", 1, 1, 1)

    first = hashlib.sha256()
    evaluator.update_transform_digest(first, key, window)
    second = hashlib.sha256()
    evaluator.update_transform_digest(second, key, window.copy())
    changed = hashlib.sha256()
    changed_key = (*key[:-3], 0, 1, 1)
    evaluator.update_transform_digest(changed, changed_key, window)

    assert first.hexdigest() == second.hexdigest()
    assert first.hexdigest() != changed.hexdigest()


def test_prepare_module_does_not_import_torch():
    command = [
        sys.executable,
        "-c",
        (
            "import sys;"
            f"sys.path.insert(0,{str(SCRIPTS)!r});"
            "import evaluate_l4_counterfactual_factorial;"
            "assert 'torch' not in sys.modules"
        ),
    ]
    subprocess.run(command, check=True)


def test_stage_a_local_call_graph_cannot_reach_torch_or_checkpoint_loading():
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
    reachable = set()
    frontier = ["_run_prepare_only_success"]
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        frontier.extend(calls.get(name, set()) & functions.keys())

    assert {
        "load_and_validate_models",
        "_make_cnn",
        "_predict",
        "_run_score_success",
    }.isdisjoint(reachable)
    for name in reachable:
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("torch")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("torch")


def test_raw_matched_real_transcript_parsers_accept_exact_stream_format():
    stdout = b"seed=7 complete\nseed=42 complete\n"
    stderr = b""
    payload = raw_matched_real_transcript(stdout=stdout, stderr=stderr)

    for parser in transcript_parsers():
        parsed = parser(payload)
        assert parsed["format"] == (
            "e14.matched_real_training_child_output.v1"
        )
        assert parsed["returncode"] == 0
        assert parsed["stdout_bytes"] == len(stdout)
        assert parsed["stderr_bytes"] == len(stderr)
        assert parsed["existing_target_skip_lines"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "header",
        "stdout_length",
        "stderr_delimiter",
        "returncode",
        "skip_line",
    ),
)
def test_raw_matched_real_transcript_parsers_reject_invalid_bytes(mutation):
    stdout = b"ordinary output\n"
    payload = raw_matched_real_transcript(stdout=stdout)
    if mutation == "header":
        payload = payload.replace(
            b"E14 matched-real frozen trainer transcript v1",
            b"E14 altered trainer transcript v1",
            1,
        )
    elif mutation == "stdout_length":
        payload = payload.replace(
            f"stdout_bytes={len(stdout)}".encode("ascii"),
            b"stdout_bytes=999",
            1,
        )
    elif mutation == "stderr_delimiter":
        payload = payload.replace(
            b"\n--- stderr ---\n",
            b"\n--- altered ---\n",
            1,
        )
    elif mutation == "returncode":
        payload = raw_matched_real_transcript(
            stdout=stdout,
            returncode=1,
        )
    elif mutation == "skip_line":
        payload = raw_matched_real_transcript(
            stdout=b"skip existing protected.pt\n",
        )
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(mutation)

    for parser in transcript_parsers():
        with pytest.raises(ValueError):
            parser(payload)


def valid_training_log(seed: int = 7) -> dict:
    history = [
        {
            "epoch": position,
            "optimizer_step": position * training_wrapper.VALIDATION_CADENCE,
            "train_loss": 1.0 / position,
            "val_macro_f1": 0.8 + position / 1000,
        }
        for position in range(1, 13)
    ]
    return {
        "family": "cnn",
        "setting": "real_only",
        "seed": seed,
        "budget_mode": "matched_steps",
        "model_tag": training_wrapper.MODEL_TAG,
        "sampling_policy": "legacy",
        "sampling_audit": None,
        "optimizer_steps": 6156,
        "real_train_windows": 262149,
        "train_windows": 262149,
        "validation_checkpoints": 12,
        "epochs_run": 12,
        "selected_epoch": 12,
        "epochs_run_semantics":
            "validation_checkpoints_not_augmented_data_epochs",
        "selected_epoch_semantics": "validation_checkpoint",
        "best_val_macro_f1": history[-1]["val_macro_f1"],
        "elapsed_seconds": 1.0,
        "history": history,
    }


def test_training_wrapper_requires_explicit_execute_and_fixed_child_argv():
    default = training_wrapper.parse_args([])
    explicit = training_wrapper.parse_args(["--execute"])

    assert default.execute is False
    assert explicit.execute is True
    assert training_wrapper.fixed_child_command() == [
        sys.executable,
        "journal/scripts/train_generator_extension_cnn.py",
        "--settings",
        "real_only",
        "--seeds",
        "7,42,123,2026,3407",
        "--budget-mode",
        "matched_steps",
        "--max-steps",
        "6156",
        "--model-tag",
        "matchedsteps_e14_v1",
    ]


def test_training_wrapper_uses_exact_versioned_target_paths(tmp_path):
    for seed in training_wrapper.SEEDS:
        assert training_wrapper.checkpoint_path(seed, tmp_path) == (
            tmp_path
            / "journal/models/generator_extension"
            / f"cnn_real_only_matchedsteps_e14_v1_seed{seed}.pt"
        )
        assert training_wrapper.training_log_path(seed, tmp_path) == (
            tmp_path
            / "journal/results/logs"
            / f"train_cnn_real_only_matchedsteps_e14_v1_seed{seed}.log"
        )
    assert len(set(training_wrapper.training_artifact_targets(tmp_path))) == 10


def test_training_log_contract_rejects_wrong_budget():
    payload = valid_training_log()
    validated = training_wrapper.validate_training_log(payload, 7)
    assert validated["selected_optimizer_step"] == 6156

    payload["optimizer_steps"] = 6155
    with pytest.raises(training_wrapper.E14TrainingError, match="optimizer_steps"):
        training_wrapper.validate_training_log(payload, 7)


def test_training_record_publication_is_atomic_and_no_clobber(tmp_path):
    path = tmp_path / "record.json"
    training_wrapper.atomic_publish_json(path, {"status": "first"})
    before = path.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite"):
        training_wrapper.atomic_publish_json(path, {"status": "second"})
    assert path.read_bytes() == before
    assert json.loads(path.read_text())["status"] == "first"


def test_amendment_is_frozen_by_every_e14_stage():
    amendment = training_wrapper.REPO / training_wrapper.AMENDMENT_PATH
    observed = hashlib.sha256(amendment.read_bytes()).hexdigest()

    assert observed == training_wrapper.AMENDMENT_SHA256
    assert (
        training_wrapper.FROZEN_REGISTERED_INPUTS[
            training_wrapper.AMENDMENT_PATH
        ]
        == observed
    )
    assert training_wrapper.AMENDMENT_PATH in (
        training_wrapper.IMPLEMENTATION_SOURCE_PATHS
    )
    assert evaluator.EXPECTED_HASHES[evaluator.AMENDMENT] == observed
    assert evaluator.IMPLEMENTATION_SOURCES["amendment"] == evaluator.AMENDMENT
    assert analyzer.AMENDMENT_SHA256 == observed
    assert analyzer.IMPLEMENTATION_SOURCES["amendment"] == analyzer.AMENDMENT
    assert analyzer.AMENDMENT in analyzer.CANONICAL_MANIFEST_INPUTS


def test_transcript_amendment_is_frozen_by_prepare_and_analyzer():
    amendment = evaluator.POST_TRAINING_AMENDMENT
    observed = hashlib.sha256(amendment.read_bytes()).hexdigest()

    assert amendment == evaluator.TRANSCRIPT_AMENDMENT
    assert amendment == analyzer.TRANSCRIPT_AMENDMENT
    assert observed == evaluator.POST_TRAINING_AMENDMENT_SHA256
    assert evaluator.EXPECTED_HASHES[amendment] == observed
    assert (
        evaluator.IMPLEMENTATION_SOURCES["transcript_amendment"]
        == amendment
    )
    assert observed == analyzer.TRANSCRIPT_AMENDMENT_SHA256
    assert (
        analyzer.IMPLEMENTATION_SOURCES["transcript_amendment"]
        == amendment
    )
    assert amendment in analyzer.CANONICAL_MANIFEST_INPUTS
    assert evaluator.MATCHED_REAL_TRANSCRIPT == analyzer.MATCHED_REAL_TRANSCRIPT
    assert evaluator.MATCHED_REAL_TRANSCRIPT in (
        analyzer.CANONICAL_MANIFEST_INPUTS
    )


def test_post_training_transition_constants_and_path_sets_are_exact():
    training_commit = "ef7afd2da361718c3a6fe4ad75cd82068430c61c"
    amendment_commit = "ee21d150277fe83b0bcbc188829920de164b2b01"
    amendment_path = (
        "journal/experiments/e14_l4_counterfactual_factorial/"
        "IMPLEMENTATION_AMENDMENT_2026-07-24_STAGE_A_TRANSCRIPT.md"
    )
    correction_paths = {
        "journal/scripts/evaluate_l4_counterfactual_factorial.py",
        "journal/scripts/analyze_l4_counterfactual_factorial.py",
        "journal/tests/test_l4_counterfactual_factorial.py",
    }

    assert evaluator.TRAINING_SOURCE_COMMIT == training_commit
    assert evaluator.TRAINING_IMPLEMENTATION_COMMIT == training_commit
    assert evaluator.AMENDMENT_SOURCE_COMMIT == amendment_commit
    assert evaluator.TRANSCRIPT_AMENDMENT_COMMIT == amendment_commit
    assert analyzer.TRAINING_SOURCE_COMMIT == training_commit
    assert analyzer.TRANSCRIPT_AMENDMENT_COMMIT == amendment_commit
    assert analyzer.AMENDMENT_SOURCE_COMMIT == amendment_commit

    assert evaluator.TRAINING_TO_AMENDMENT_PATHS == (amendment_path,)
    assert set(evaluator.AMENDMENT_TO_PREPARE_PATHS) == correction_paths
    assert evaluator.AMENDMENT_ONLY_PATHS == frozenset(
        {("A", amendment_path)}
    )
    assert evaluator.IMPLEMENTATION_CORRECTION_PATHS == frozenset(
        {("M", path) for path in correction_paths}
    )
    assert evaluator.FULL_TRANSITION_PATHS == (
        evaluator.AMENDMENT_ONLY_PATHS
        | evaluator.IMPLEMENTATION_CORRECTION_PATHS
    )
    assert analyzer.TRAINING_TO_AMENDMENT_CHANGES == (
        {"status": "A", "path": amendment_path},
    )
    assert set(analyzer.AMENDMENT_TO_PREPARE_PATHS) == correction_paths
    assert {
        (record["status"], record["path"])
        for record in analyzer.AMENDMENT_TO_PREPARE_CHANGES
    } == {("M", path) for path in correction_paths}

    repo = Path(__file__).resolve().parents[2]
    parent = subprocess.run(
        ["git", "rev-parse", f"{amendment_commit}^"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    count = subprocess.run(
        [
            "git",
            "rev-list",
            "--count",
            f"{training_commit}..{amendment_commit}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changes = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--no-renames",
            training_commit,
            amendment_commit,
            "--",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert parent == training_commit
    assert count == "1"
    assert changes == [f"A\t{amendment_path}"]


def test_frozen_matched_real_training_record_schema_is_exact():
    expected_hashes = {
        evaluator.MATCHED_REAL_PREFLIGHT:
            "3b895fe3c84e5188968fedf0ef0450f2ace03c98e3fe1e3ea11aa3a8f98e1e14",
        evaluator.MATCHED_REAL_RUN:
            "53865949ecf4383fa6f9a21c224b76c289baeb9c88daa7fbf80eddfacfe0de28",
        evaluator.MATCHED_REAL_TRANSCRIPT:
            "e362b71ca4cdff7755e27b16ca614ca2f1e574c12c8fc95777aeb095a631e0e5",
    }
    expected_bytes = {
        evaluator.MATCHED_REAL_PREFLIGHT: 21_223,
        evaluator.MATCHED_REAL_RUN: 37_000,
        evaluator.MATCHED_REAL_TRANSCRIPT: 4_977,
    }
    expected_roles = frozenset(
        {"preflight", "run", "child_transcript"}
    )

    assert evaluator.MATCHED_REAL_RECORD_HASHES == expected_hashes
    assert evaluator.MATCHED_REAL_RECORD_BYTES == expected_bytes
    assert evaluator.MATCHED_REAL_TRAINING_RECORD_KEYS == expected_roles
    assert analyzer.MATCHED_REAL_TRAINING_RECORD_KEYS == expected_roles
    assert analyzer.MATCHED_REAL_RECORD_HASHES == expected_hashes
    assert analyzer.MATCHED_REAL_RECORD_BYTES == expected_bytes
    assert (
        analyzer.TRAINING_PREFLIGHT_SHA256
        == expected_hashes[evaluator.MATCHED_REAL_PREFLIGHT]
    )
    assert (
        analyzer.TRAINING_RUN_SHA256
        == expected_hashes[evaluator.MATCHED_REAL_RUN]
    )
    assert (
        analyzer.MATCHED_REAL_TRANSCRIPT_SHA256
        == expected_hashes[evaluator.MATCHED_REAL_TRANSCRIPT]
    )
    assert analyzer.MATCHED_REAL_TRANSCRIPT_FORMAT == (
        "e14.matched_real_training_child_output.v1"
    )


def test_matched_real_training_record_bundle_schema_is_exact(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path
    exp = repo / "journal/experiments/e14"
    exp.mkdir(parents=True)
    preflight_path = exp / "preflight.json"
    run_path = exp / "run.json"
    transcript_path = exp / "transcript.log"
    preflight_path.write_bytes(b'{"record":"preflight"}\n')
    transcript = raw_matched_real_transcript(
        stdout=b"seed=7 complete\n",
    )
    transcript_path.write_bytes(transcript)
    transcript_sha = hashlib.sha256(transcript).hexdigest()
    transcript_record = {
        "path": transcript_path.relative_to(repo).as_posix(),
        "bytes": len(transcript),
        "sha256": transcript_sha,
        "format": "e14.matched_real_training_child_output.v1",
        "stdout_bytes": len(b"seed=7 complete\n"),
        "stderr_bytes": 0,
        "streams_preserved_separately": True,
        "existing_target_skip_lines": [],
    }
    run_path.write_text(
        json.dumps(
            {
                "child_transcript": transcript_record,
                "validation": {
                    "child_output_persisted": True,
                    "child_reported_existing_target_skips": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    frozen_paths = (preflight_path, run_path, transcript_path)
    monkeypatch.setattr(evaluator, "REPO_ROOT", repo)
    monkeypatch.setattr(evaluator, "MATCHED_REAL_PREFLIGHT", preflight_path)
    monkeypatch.setattr(evaluator, "MATCHED_REAL_RUN", run_path)
    monkeypatch.setattr(evaluator, "MATCHED_REAL_TRANSCRIPT", transcript_path)
    monkeypatch.setattr(
        evaluator,
        "MATCHED_REAL_RECORD_HASHES",
        {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frozen_paths
        },
    )
    monkeypatch.setattr(
        evaluator,
        "MATCHED_REAL_RECORD_BYTES",
        {path: path.stat().st_size for path in frozen_paths},
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_matched_real_records",
        lambda: {},
    )

    records = evaluator._matched_real_training_record_bundle()
    assert set(records) == evaluator.MATCHED_REAL_TRAINING_RECORD_KEYS
    assert set(records["preflight"]) == {"path", "bytes", "sha256"}
    assert set(records["run"]) == {"path", "bytes", "sha256"}
    assert set(records["child_transcript"]) == {
        "path",
        "bytes",
        "sha256",
        "format",
        "returncode",
        "stdout_bytes",
        "stderr_bytes",
        "stdout_sha256",
        "stderr_sha256",
        "streams_preserved_separately",
        "existing_target_skip_lines",
    }
    assert records["child_transcript"]["bytes"] == len(transcript)
    assert records["child_transcript"]["sha256"] == transcript_sha
    assert records["child_transcript"]["stdout_bytes"] == len(
        b"seed=7 complete\n"
    )
    assert records["child_transcript"]["stderr_bytes"] == 0
    assert records["child_transcript"]["returncode"] == 0
    assert (
        records["child_transcript"]["streams_preserved_separately"] is True
    )
    assert records["child_transcript"]["existing_target_skip_lines"] == []


def test_stage_a_untracked_allowlist_is_exact():
    expected = {
        evaluator.portable_path(evaluator.MATCHED_REAL_PREFLIGHT),
        evaluator.portable_path(evaluator.MATCHED_REAL_RUN),
        evaluator.portable_path(evaluator.MATCHED_REAL_TRANSCRIPT),
    }
    for seed in evaluator.SEEDS:
        expected.add(
            evaluator.portable_path(
                evaluator.checkpoint_path("real_ms", seed)
            )
        )
        expected.add(
            evaluator.portable_path(
                evaluator.training_log_path("real_ms", seed)
            )
        )

    assert evaluator._allowed_untracked_paths_for_prepare() == expected


def test_evaluator_reports_matched_and_derived_legacy_optimizer_steps(tmp_path):
    matched_path = tmp_path / "matched.log"
    matched_path.write_text(json.dumps(valid_training_log()), encoding="utf-8")
    matched = evaluator._validate_training_log(
        "real_ms", 7, matched_path
    )
    assert matched["total_optimizer_steps"] == 6_156
    assert matched["selected_optimizer_step"] == 6_156
    assert matched["total_optimizer_steps_source"] == "recorded_optimizer_steps"

    legacy_payload = {
        "family": "cnn",
        "setting": "real_only",
        "seed": 7,
        "train_windows": 1_025,
        "epochs_run": 3,
        "selected_epoch": 2,
        "best_val_macro_f1": 0.9,
        "history": [
            {"epoch": 1, "val_macro_f1": 0.8},
            {"epoch": 2, "val_macro_f1": 0.9},
            {"epoch": 3, "val_macro_f1": 0.85},
        ],
    }
    legacy_path = tmp_path / "legacy.log"
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy = evaluator._validate_training_log("real_std", 7, legacy_path)
    assert legacy["steps_per_epoch"] == 3
    assert legacy["total_optimizer_steps"] == 9
    assert legacy["selected_optimizer_step"] == 6
    assert legacy["total_optimizer_steps_source"] == "derived_legacy_epochs"
    assert (
        legacy["selected_optimizer_step_source"]
        == "derived_legacy_selected_epoch"
    )


def test_trainer_rejects_dangling_symlink_target(tmp_path):
    staged_model = tmp_path / "staged-model"
    staged_log = tmp_path / "staged-log"
    staged_model.write_bytes(b"model")
    staged_log.write_bytes(b"log")
    model_target = tmp_path / "model.pt"
    log_target = tmp_path / "train.log"
    model_target.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError, match="overwrite"):
        trainer._publish_staged_training_bundle(
            ((staged_model, model_target), (staged_log, log_target))
        )

    assert model_target.is_symlink()
    assert not os.path.lexists(log_target)


def test_trainer_rolls_back_own_link_when_second_target_races(
    tmp_path, monkeypatch
):
    staged_model = tmp_path / "staged-model"
    staged_log = tmp_path / "staged-log"
    staged_model.write_bytes(b"model")
    staged_log.write_bytes(b"log")
    model_target = tmp_path / "model.pt"
    log_target = tmp_path / "train.log"
    real_link = os.link
    calls = 0

    def racing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(target).write_bytes(b"racer")
        return real_link(source, target)

    monkeypatch.setattr(trainer.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="appeared during publication"):
        trainer._publish_staged_training_bundle(
            ((staged_model, model_target), (staged_log, log_target))
        )

    assert not os.path.lexists(model_target)
    assert log_target.read_bytes() == b"racer"


def test_training_wrapper_preserves_separate_child_streams(tmp_path, capsys):
    completed = subprocess.CompletedProcess(
        args=["trainer"],
        returncode=1,
        stdout=b"ordinary output\nskip existing protected.pt\n",
        stderr=b"failure detail\n",
    )
    transcript_path = tmp_path / "child.log"
    transcript = training_wrapper.persist_child_transcript(
        completed,
        transcript_path,
        repo=tmp_path,
    )
    captured = capsys.readouterr()

    assert "ordinary output" in captured.out
    assert "failure detail" in captured.err
    assert transcript["streams_preserved_separately"] is True
    assert transcript["existing_target_skip_lines"] == [
        "skip existing protected.pt"
    ]
    payload = transcript_path.read_bytes()
    assert b"--- stdout ---" in payload
    assert b"--- stderr ---" in payload


def test_evaluator_rollback_preserves_racing_replacement(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluator, "JOURNAL_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "EXP", tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    real_link = os.link
    calls = 0

    def racing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            first.unlink()
            first.write_bytes(b"replacement")
            second.write_bytes(b"racer")
        return real_link(source, target)

    monkeypatch.setattr(evaluator.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        evaluator.publish_bundle_no_clobber(
            ((first, b"ours-first"), (second, b"ours-second"))
        )

    assert first.read_bytes() == b"replacement"
    assert second.read_bytes() == b"racer"
    retained = list(tmp_path.glob(".e14_staged_*"))
    assert len(retained) == 1


def test_evaluator_records_keyboard_interrupt_before_reraising(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "JOURNAL_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "EXP", tmp_path)
    monkeypatch.setattr(evaluator, "FAILURE_DIR", tmp_path / "failures")
    monkeypatch.setattr(
        evaluator,
        "PREPARE_PATHS",
        {"prepare_record": tmp_path / "prepare.json"},
    )
    monkeypatch.setattr(
        evaluator,
        "SCORE_PATHS",
        {"run_record": tmp_path / "run.json"},
    )

    def interrupt():
        raise KeyboardInterrupt("operator stop")

    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        evaluator._run_with_failure_record(
            stage="prepare",
            status="T-STOP-MANIPULATION",
            operation=interrupt,
        )

    records = list((tmp_path / "failures").glob("prepare_failure_*.json"))
    assert len(records) == 1
    failure = json.loads(records[0].read_text(encoding="utf-8"))
    assert failure["status"] == "T-STOP-MANIPULATION"
    assert failure["error"]["type"] == "KeyboardInterrupt"
    assert failure["scientific_verdict_issued"] is False


@pytest.fixture(scope="module")
def toy_scoring_tables():
    arm_coefficient = {
        "real_ms": 0.40,
        "rule_ms": 1.10,
        "placebo_ms": 0.70,
        "real_std": 0.35,
        "rule_std": 0.90,
        "placebo_std": 0.60,
    }
    seed_scale = dict(
        zip(analyzer.SEEDS, (0.94, 0.98, 1.00, 1.03, 1.07), strict=True)
    )
    rows = []
    for (
        arm,
        seed,
        block_id,
        attack,
        id_stratum,
        p,
        s,
        d,
    ) in itertools.product(
        analyzer.ARMS,
        analyzer.SEEDS,
        analyzer.BLOCKS,
        analyzer.ATTACKS,
        analyzer.ID_LEVELS,
        analyzer.BITS,
        analyzer.BITS,
        analyzer.BITS,
    ):
        block_index = analyzer.BLOCKS.index(block_id)
        attack_label = 3 if attack == "Gear" else 4
        attack_scale = 1.0 if attack == "Gear" else 0.8
        id_scale = 1.0 if id_stratum == "canonical" else 0.65
        factorial_signal = (
            0.010 * p
            - 0.008 * s
            + 0.006 * d
            + 0.005 * p * s
            - 0.004 * p * d
            + 0.003 * s * d
            + 0.002 * p * s * d
        )
        response = (
            arm_coefficient[arm]
            * seed_scale[seed]
            * attack_scale
            * id_scale
            * factorial_signal
        )
        common = (
            0.35
            + 0.002 * block_index
            + (0.004 if attack == "RPM" else 0.0)
            + (0.003 if id_stratum == "shifted" else 0.0)
        )
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "block_id": block_id,
                "attack": attack,
                "attack_label": attack_label,
                "id_stratum": id_stratum,
                "P": p,
                "S": s,
                "D": d,
                "cell": evaluator.cell_name(p, s, d),
                "n": evaluator.BASES_PER_BLOCK,
                "exact_recall": common + response,
                "binary_recall": common + 0.20 + 0.75 * response,
                "exact_standardized_margin_shift": 2.0 * response,
                "binary_standardized_margin_shift": 1.5 * response,
                "exact_standardized_margin_status": "ok",
                "binary_standardized_margin_status": "ok",
            }
        )
    scenarios = pd.DataFrame(rows)
    cells = evaluator._build_by_cell(scenarios)
    deltas = evaluator._build_augmentation_delta(scenarios)

    normal_rows = []
    for arm, seed, block_id in itertools.product(
        analyzer.ARMS, analyzer.SEEDS, analyzer.BLOCKS
    ):
        normal_recall = (
            0.99
            - 0.002 * analyzer.ARMS.index(arm)
            - 0.0001 * analyzer.SEEDS.index(seed)
        )
        normal_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "block_id": block_id,
                "n": evaluator.BASES_PER_BLOCK,
                "normal_recall": normal_recall,
                "fpr": 1.0 - normal_recall,
            }
        )
    return pd.DataFrame(normal_rows), scenarios, cells, deltas


def test_evaluator_and_analyzer_table_schemas_agree(toy_scoring_tables):
    normal, scenarios, cells, deltas = toy_scoring_tables

    evaluator_counts = evaluator._validate_scoring_tables(
        normal, scenarios, cells, deltas
    )
    analyzer_counts = analyzer.validate_complete_tables(
        normal, scenarios, cells, deltas
    )

    assert evaluator_counts["scenario_rows"] == 2_880
    assert evaluator_counts["augmentation_delta_rows"] == 2_880
    assert analyzer_counts["normal_rows"] == 90
    assert analyzer_counts["cell_rows"] == 1_440
    assert analyzer_counts["delta_rows"] == 2_880


@pytest.fixture(scope="module")
def toy_effects(toy_scoring_tables):
    return analyzer.build_effect_table(toy_scoring_tables[3])


def test_analyzer_builds_exact_registered_effect_cartesian_product(toy_effects):
    assert len(toy_effects) == analyzer.EXPECTED_EFFECT_ROWS == 15_552
    assert not toy_effects.duplicated(list(analyzer.EFFECT_KEY)).any()
    assert set(toy_effects["row_type"]) == {"seed", "block", "summary"}
    assert set(toy_effects.loc[toy_effects["row_type"] == "seed", "unit_id"]) == {
        str(seed) for seed in analyzer.SEEDS
    }
    assert set(toy_effects.loc[toy_effects["row_type"] == "block", "unit_id"]) == {
        *analyzer.BLOCKS
    }


def test_analyzer_uses_only_the_three_registered_holm_families(toy_effects):
    adjusted = toy_effects[toy_effects["holm_family"].astype(str).ne("")]
    assert set(adjusted["holm_family"]) == set(analyzer.PRIMARY_HOLM_FAMILIES)
    assert len(adjusted) == 3 * len(analyzer.FACTORIAL_EFFECTS)
    assert adjusted["row_type"].eq("summary").all()
    assert adjusted["holm_adjusted_p"].notna().all()

    supporting = toy_effects[
        toy_effects["row_type"].eq("summary")
        & toy_effects["holm_family"].astype(str).eq("")
    ]
    assert supporting["test_p"].isna().all()
    assert supporting["holm_input_p"].isna().all()
    assert supporting["raw_p"].isna().all()
    assert supporting["p_status"].eq("not_tested_supporting").all()


def test_analyzer_reports_registered_block_descriptives(toy_effects):
    summaries = toy_effects[toy_effects["row_type"].eq("summary")]

    assert summaries["block_n_available"].eq(3).all()
    for column in ("block_mean", "block_sd", "block_min", "block_max"):
        assert summaries[column].notna().all()
    assert (
        summaries["block_min"] <= summaries["block_mean"] + 1e-15
    ).all()
    assert (
        summaries["block_mean"] <= summaries["block_max"] + 1e-15
    ).all()


def test_zero_legacy_mean_is_unresolved_not_budget_discordant():
    assert analyzer.legacy_budget_classification(
        primary_mean=0.1,
        legacy_mean=0.0,
        legacy_ci_low=-0.1,
        legacy_ci_high=0.1,
    ) == "budget-direction-unresolved-zero"


def test_analyzer_margin_missing_seed_has_no_reduced_n_interval():
    summary = analyzer.t_summary([0.1, 0.2, np.nan, 0.3, 0.4])

    assert summary["n_available"] == 4
    assert summary["status"] == "margin_n_available_only"
    assert np.isnan(summary["ci_low"])
    assert np.isnan(summary["ci_high"])
    assert np.isnan(summary["test_p"])


def test_analyzer_retains_staged_bundle_if_publication_races(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(analyzer, "ROOT", tmp_path)
    inputs = (tmp_path / "input-a", tmp_path / "input-b")
    for index, path in enumerate(inputs):
        path.write_text(f"input-{index}", encoding="utf-8")
    monkeypatch.setattr(analyzer, "CANONICAL_MANIFEST_INPUTS", inputs)

    effects_path = tmp_path / "tables" / "effects.csv"
    manifest_path = tmp_path / "logs" / "manifest.json"
    effects_path.parent.mkdir()
    manifest_path.parent.mkdir()
    effects = pd.DataFrame(
        [{"row_type": "summary", "localization_verdict": "F-MAIN"}]
    )
    input_records = [analyzer._artifact_record(path) for path in inputs]
    provenance = {
        "status": "T-PASS-PROVENANCE",
        "analysis_source_state": {"tracked_state_clean": True},
        "canonical_manifest_inputs": input_records,
    }
    real_link = os.link
    calls = 0

    def racing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            effects_path.unlink()
            effects_path.write_text("replacement", encoding="utf-8")
            Path(target).write_text("racer", encoding="utf-8")
        return real_link(source, target)

    monkeypatch.setattr(analyzer.os, "link", racing_link)
    with pytest.raises(RuntimeError, match="staged bundle retained at") as exc:
        analyzer.publish_analysis_bundle(
            effects,
            effects_path,
            manifest_path,
            inputs,
            {},
            provenance,
        )

    retained = Path(str(exc.value).rsplit(" at ", 1)[1])
    assert retained.is_dir()
    assert sorted(path.name for path in retained.iterdir()) == [
        "effects.csv",
        "manifest.json",
    ]
    assert effects_path.read_text(encoding="utf-8") == "replacement"
    assert manifest_path.read_text(encoding="utf-8") == "racer"
    shutil.rmtree(retained)
