#!/usr/bin/env python3
"""Safely launch the five preregistered E14 matched-budget real-only fits.

The generic CNN trainer intentionally supports many settings and historically
skipped an existing checkpoint in its command-line loop.  This wrapper narrows
that interface to the one E14 command, proves that all ten model/log targets
are absent, freezes provenance before the child starts, and validates the
complete five-seed bundle before publishing a success record.

The default mode is a read-only dry run.  Training requires the explicit
``--execute`` flag.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch


REPO = Path(__file__).resolve().parents[2]
E14_DIR = REPO / "journal" / "experiments" / "e14_l4_counterfactual_factorial"
MODEL_DIR = REPO / "journal" / "models" / "generator_extension"
LOG_DIR = REPO / "journal" / "results" / "logs"

SEEDS = (7, 42, 123, 2026, 3407)
SETTING = "real_only"
MODEL_TAG = "matchedsteps_e14_v1"
MAX_STEPS = 6_156
VALIDATION_CADENCE = 513
EXPECTED_VALIDATION_CHECKPOINTS = 12
EXPECTED_REAL_TRAIN_WINDOWS = 262_149

PREREG_COMMIT = "686f4aaae8d35e1b807480c56abd02ead7a13c27"
PREREG_PATH = (
    "journal/experiments/e14_l4_counterfactual_factorial/PREREG.md"
)
PREREG_SHA256 = (
    "f0465fea952b9f7d5fd1b20a7864a96192099b9d3346f6c81e4274ce0fe0424a"
)
AMENDMENT_PATH = (
    "journal/experiments/e14_l4_counterfactual_factorial/"
    "IMPLEMENTATION_AMENDMENT_2026-07-24.md"
)
AMENDMENT_SHA256 = (
    "f6ecf1c56a08ef9d6fc02f3a2f9f9277b6c4e7cf4c972dd4314bbcc4bb431b62"
)

PREFLIGHT_PATH = E14_DIR / "matched_real_training_preflight_v1.json"
RUN_PATH = E14_DIR / "matched_real_training_run_v1.json"
FAILURE_PATH = E14_DIR / "matched_real_training_failure_v1.json"
TRANSCRIPT_PATH = E14_DIR / "matched_real_training_child_output_v1.log"

# Every Section 3.1 file is checked, even when it is not read by the real-only
# child.  This keeps the E14 implementation freeze tied to the registered
# evidence base rather than only to the minimum trainer dependency closure.
FROZEN_REGISTERED_INPUTS: dict[str, str] = {
    "journal/results/tables/evaluation_realization_blocks_e13_sampling_v2.csv":
        "9e18b3844e20f259938c732a10bc0251767f0d2eaae9c731ba8da7ef087f5efb",
    "journal/experiments/e8_evaluation_realization/run_e13_sampling_v2.json":
        "fae2ca82bb26ab24a5699fce9f331951be23746093fef9e28d0b240fd2c3b2db",
    "journal/datasets/windows/test_windows.npz":
        "4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d73629a591c56b54b7",
    "journal/datasets/windows/train_windows.npz":
        "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9eda81a34faec7cb4",
    "journal/datasets/synthetic/rule_based_windows.npz":
        "4078ab00b8531f760b296b5fd293e3df914bef996d317c958ead4762137bafd5",
    "journal/datasets/synthetic/rule_placebo_windows.npz":
        "2b336ed46bb8f3248d10831c80fdfa6db25c3e6eb44f4fd680f78546a63ec4fa",
    "journal/results/logs/generate_rule_placebo_twin.log":
        "5ca8287c5f7cef322808f73759fa1ad69e5b6d1c80182ab4b795809d0bfaddc6",
    "journal/scripts/evaluate_evaluation_realization.py":
        "eadf93df237946b7f670a74e316f0b80fc1d1920d4cc7072c7434c699a2ff2bb",
    "journal/scripts/generate_rule_based_synthetic_v2.py":
        "34ff0374af1856fd842da8226580d5ff0d47cbd16b36fb4273de3e06bcac2570",
    "journal/scripts/e10a_crossover.py":
        "a1156ce1228b1975906fcfe8aad1c888e307944e7e88b6242aa32e5c3713432f",
    "journal/scripts/train_generator_extension_cnn.py":
        "a5d88bcb579da77e234e991a9b050c040df21d2f75e971c0c952c6226749db61",
    "journal/scripts/lib_common.py":
        "5848ff4b0063d4394cd1d6627faf49e072cb363d7796cc64fce3acc48b05639a",
    PREREG_PATH: PREREG_SHA256,
    AMENDMENT_PATH: AMENDMENT_SHA256,
}

# The frozen generic trainer also reads validation windows.  Section 3.1 did
# not list that path separately, so the implementation commit adds an explicit
# runtime-input freeze rather than leaving the actual selection data unhashed.
ADDITIONAL_RUNTIME_INPUTS: dict[str, str] = {
    "journal/datasets/windows/val_windows.npz":
        "b4920e7f60fbf38b38e12976129f120ebff160a21a4c77093f3746f3c1aa218e",
}

EXPECTED_STANDARDIZER_SHA256 = {
    "mean": "1e53f10b4dfb1de8522513ca20c7d0c2b5a19a8de8f76067e4273a93467da8a2",
    "standard_deviation":
        "092f42e9a1019fe709233a91c6e876e976f8b3b8616807b42eb6e6da7b7dbb7c",
    "mean_then_standard_deviation":
        "9019b1a91c983859e8dd998717eb5c7b612ed03dc2a65ec63402fd54ab42ef4e",
}

IMPLEMENTATION_SOURCE_PATHS = (
    "journal/scripts/run_e14_matched_real_training.py",
    "journal/scripts/evaluate_l4_counterfactual_factorial.py",
    "journal/scripts/analyze_l4_counterfactual_factorial.py",
    "journal/tests/test_l4_counterfactual_factorial.py",
    PREREG_PATH,
    AMENDMENT_PATH,
)

THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
)

# Frozen CNN1D state-dict contract from the registered trainer.
EXPECTED_STATE_SHAPES: dict[str, tuple[int, ...]] = {
    "net.0.weight": (64, 11, 5),
    "net.0.bias": (64,),
    "net.1.weight": (64,),
    "net.1.bias": (64,),
    "net.1.running_mean": (64,),
    "net.1.running_var": (64,),
    "net.1.num_batches_tracked": (),
    "net.4.weight": (128, 64, 5),
    "net.4.bias": (128,),
    "net.5.weight": (128,),
    "net.5.bias": (128,),
    "net.5.running_mean": (128,),
    "net.5.running_var": (128,),
    "net.5.num_batches_tracked": (),
    "net.8.weight": (128, 128, 3),
    "net.8.bias": (128,),
    "net.9.weight": (128,),
    "net.9.bias": (128,),
    "net.9.running_mean": (128,),
    "net.9.running_var": (128,),
    "net.9.num_batches_tracked": (),
    "head.2.weight": (5, 128),
    "head.2.bias": (5,),
}


class E14TrainingError(RuntimeError):
    """A preregistered E14 training gate failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def command_sha256(argv: Sequence[str]) -> str:
    encoded = b"\0".join(os.fsencode(argument) for argument in argv)
    return hashlib.sha256(encoded).hexdigest()


def repo_relative(path: Path, repo: Path = REPO) -> str:
    try:
        return path.absolute().relative_to(repo.absolute()).as_posix()
    except ValueError as exc:
        raise E14TrainingError(f"path escapes repository: {path}") from exc


def path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def checkpoint_path(seed: int, repo: Path = REPO) -> Path:
    return (
        repo
        / "journal"
        / "models"
        / "generator_extension"
        / f"cnn_real_only_{MODEL_TAG}_seed{seed}.pt"
    )


def training_log_path(seed: int, repo: Path = REPO) -> Path:
    return (
        repo
        / "journal"
        / "results"
        / "logs"
        / f"train_cnn_real_only_{MODEL_TAG}_seed{seed}.log"
    )


def training_artifact_targets(repo: Path = REPO) -> tuple[Path, ...]:
    models = tuple(checkpoint_path(seed, repo) for seed in SEEDS)
    logs = tuple(training_log_path(seed, repo) for seed in SEEDS)
    targets = models + logs
    if len(targets) != 10 or len(set(targets)) != 10:
        raise AssertionError("E14 target definition must contain ten unique paths")
    return targets


def record_targets(repo: Path = REPO) -> tuple[Path, Path]:
    directory = (
        repo
        / "journal"
        / "experiments"
        / "e14_l4_counterfactual_factorial"
    )
    return (
        directory / "matched_real_training_preflight_v1.json",
        directory / "matched_real_training_run_v1.json",
    )


def failure_record_path(repo: Path = REPO) -> Path:
    return (
        repo
        / "journal"
        / "experiments"
        / "e14_l4_counterfactual_factorial"
        / "matched_real_training_failure_v1.json"
    )


def child_transcript_path(repo: Path = REPO) -> Path:
    return (
        repo
        / "journal"
        / "experiments"
        / "e14_l4_counterfactual_factorial"
        / "matched_real_training_child_output_v1.log"
    )


def assert_targets_absent(
    paths: Sequence[Path], *, label: str, repo: Path = REPO
) -> dict[str, Any]:
    collisions = [path for path in paths if path_lexists(path)]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"{label} collision(s): {rendered}")
    return {
        "label": label,
        "expected_count": len(paths),
        "verified_absent_count": len(paths),
        "all_absent": True,
        "paths": [repo_relative(path, repo) for path in paths],
    }


def verify_hashed_inputs(
    expected: Mapping[str, str],
    *,
    repo: Path = REPO,
    registration_scope: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for relative_path, expected_hash in expected.items():
        path = repo / relative_path
        if not path.is_file():
            failures.append(f"missing {relative_path}")
            continue
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            failures.append(
                f"{relative_path}: expected {expected_hash}, got {observed_hash}"
            )
        records[relative_path] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": observed_hash,
            "expected_sha256": expected_hash,
            "matches": observed_hash == expected_hash,
            "registration_scope": registration_scope,
        }
    if failures:
        raise E14TrainingError(
            "frozen input verification failed: " + "; ".join(failures)
        )
    return records


def verify_derived_standardizer(*, repo: Path = REPO) -> dict[str, Any]:
    """Recompute and hash the registered real-train-only standardizer."""
    train_path = repo / "journal" / "datasets" / "windows" / "train_windows.npz"
    try:
        with np.load(train_path, allow_pickle=True) as archive:
            train_x = archive["x"]
            if train_x.shape != (EXPECTED_REAL_TRAIN_WINDOWS, 128, 11):
                raise E14TrainingError(
                    "unexpected train window shape while fitting standardizer: "
                    f"{train_x.shape}"
                )
            mean = train_x.mean(axis=(0, 1), keepdims=True)
            standard_deviation = train_x.std(axis=(0, 1), keepdims=True)
            standard_deviation[standard_deviation < 1e-6] = 1.0
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, E14TrainingError):
            raise
        raise E14TrainingError(
            "could not recompute the frozen real-train standardizer"
        ) from exc

    mean = np.ascontiguousarray(mean, dtype="<f4")
    standard_deviation = np.ascontiguousarray(
        standard_deviation, dtype="<f4"
    )
    expected_shape = (1, 1, 11)
    if mean.shape != expected_shape or standard_deviation.shape != expected_shape:
        raise E14TrainingError("derived standardizer arrays have invalid shape")
    mean_bytes = mean.tobytes(order="C")
    standard_deviation_bytes = standard_deviation.tobytes(order="C")
    observed = {
        "mean": hashlib.sha256(mean_bytes).hexdigest(),
        "standard_deviation": hashlib.sha256(
            standard_deviation_bytes
        ).hexdigest(),
        "mean_then_standard_deviation": hashlib.sha256(
            mean_bytes + standard_deviation_bytes
        ).hexdigest(),
    }
    if observed != EXPECTED_STANDARDIZER_SHA256:
        raise E14TrainingError(
            "derived standardizer hash mismatch: "
            + json.dumps(
                {
                    "expected": EXPECTED_STANDARDIZER_SHA256,
                    "observed": observed,
                },
                sort_keys=True,
            )
        )
    return {
        "source_path": repo_relative(train_path, repo),
        "method": "lib_common.fit_standardizer-equivalent real-train reduction",
        "shape": list(expected_shape),
        "dtype": "<f4",
        "order": "C",
        "sha256": observed,
        "expected_sha256": dict(EXPECTED_STANDARDIZER_SHA256),
        "all_hashes_match": True,
    }


def run_git(
    repo: Path, arguments: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise E14TrainingError(
            f"Git provenance command failed: git {shlex.join(arguments)}"
        ) from exc


def git_status_records(repo: Path, pathspec: str | None = None) -> list[str]:
    arguments = [
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ]
    if pathspec is not None:
        arguments.extend(["--", pathspec])
    output = run_git(repo, arguments).stdout
    return [record for record in output.split("\0") if record]


def tracked_source_record(repo: Path, relative_path: str) -> dict[str, Any]:
    path = repo / relative_path
    if not path.is_file():
        raise E14TrainingError(f"required committed source is missing: {relative_path}")
    tracked = run_git(
        repo, ["ls-files", "--error-unmatch", "--", relative_path], check=False
    )
    if tracked.returncode != 0:
        raise E14TrainingError(
            f"required source is not committed/tracked: {relative_path}"
        )
    blob_result = run_git(
        repo, ["rev-parse", f"HEAD:{relative_path}"], check=False
    )
    if blob_result.returncode != 0:
        raise E14TrainingError(
            f"required source is absent from HEAD: {relative_path}"
        )
    blob_oid = blob_result.stdout.strip()
    try:
        committed_bytes = subprocess.run(
            ["git", "cat-file", "blob", blob_oid],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise E14TrainingError(
            f"could not read committed blob for {relative_path}"
        ) from exc
    worktree_hash = sha256_file(path)
    committed_hash = hashlib.sha256(committed_bytes).hexdigest()
    if worktree_hash != committed_hash:
        raise E14TrainingError(
            f"source differs from HEAD despite status check: {relative_path}"
        )
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": worktree_hash,
        "git_blob_oid": blob_oid,
        "committed_blob_sha256": committed_hash,
        "matches_head": True,
    }


def verify_clean_committed_source(
    *,
    repo: Path = REPO,
    required_paths: Sequence[str] = IMPLEMENTATION_SOURCE_PATHS,
) -> dict[str, Any]:
    head = run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    if len(head) != 40:
        raise E14TrainingError(f"could not resolve a full source commit: {head!r}")

    prereg_ancestor = run_git(
        repo,
        ["merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"],
        check=False,
    )
    if prereg_ancestor.returncode != 0:
        raise E14TrainingError(
            f"registered preregistration commit {PREREG_COMMIT} is not in HEAD"
        )

    status = git_status_records(repo)
    if status:
        raise E14TrainingError(
            "E14 training requires a completely clean pre-training worktree: "
            + json.dumps(status, ensure_ascii=True)
        )
    wisa_status = git_status_records(repo, "wisa")
    if wisa_status:
        raise E14TrainingError(
            "frozen wisa/ tree is not clean: "
            + json.dumps(wisa_status, ensure_ascii=True)
        )

    source_files = {
        relative_path: tracked_source_record(repo, relative_path)
        for relative_path in required_paths
    }
    if source_files[PREREG_PATH]["sha256"] != PREREG_SHA256:
        raise E14TrainingError("committed E14 preregistration hash changed")

    branch_result = run_git(
        repo, ["symbolic-ref", "--quiet", "--short", "HEAD"], check=False
    )
    commit_time = run_git(
        repo, ["show", "-s", "--format=%cI", "HEAD"]
    ).stdout.strip()
    return {
        "head_commit": head,
        "branch": (
            branch_result.stdout.strip()
            if branch_result.returncode == 0
            else None
        ),
        "commit_time": commit_time,
        "preregistration_commit": PREREG_COMMIT,
        "preregistration_commit_is_ancestor": True,
        "worktree_clean": True,
        "worktree_status_records": [],
        "wisa_clean": True,
        "wisa_status_records": [],
        "source_files": source_files,
    }


def nvidia_driver_state(cuda_available: bool) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        if cuda_available:
            raise E14TrainingError(
                "CUDA is available but nvidia-smi is unavailable; "
                "driver identity cannot be captured"
            )
        return {
            "nvidia_smi_executable": None,
            "query_returncode": None,
            "query_lines": [],
        }
    command = [
        executable,
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise E14TrainingError("failed to capture nvidia-smi state") from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if cuda_available and (result.returncode != 0 or not lines):
        raise E14TrainingError(
            "CUDA is available but nvidia-smi driver query failed: "
            + result.stderr.strip()
        )
    return {
        "nvidia_smi_executable": executable,
        "query_argv": command,
        "query_returncode": result.returncode,
        "query_lines": lines,
        "query_stderr": result.stderr.strip(),
    }


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def capture_environment(*, repo: Path = REPO) -> dict[str, Any]:
    expected_prefix = (repo / ".venv").resolve()
    observed_prefix = Path(sys.prefix).resolve()
    if observed_prefix != expected_prefix:
        raise E14TrainingError(
            "E14 must run inside the repository .venv: "
            f"expected {expected_prefix}, got {observed_prefix}"
        )

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count() if cuda_available else 0):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(properties.total_memory),
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )

    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None

    environment = {
        "python": {
            "version": platform.python_version(),
            "version_full": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "executable_realpath": str(Path(sys.executable).resolve()),
            "prefix": str(observed_prefix),
            "base_prefix": str(Path(sys.base_prefix).resolve()),
            "expected_venv_prefix": str(expected_prefix),
            "uses_expected_venv": True,
        },
        "packages": {
            "numpy": np.__version__,
            "numpy_distribution": package_version("numpy"),
            "torch": torch.__version__,
            "torch_distribution": package_version("torch"),
            "scikit_learn": sklearn.__version__,
            "scikit_learn_distribution": package_version("scikit-learn"),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "process": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "gid": os.getgid() if hasattr(os, "getgid") else None,
        },
        "cpu_and_threads": {
            "os_cpu_count": os.cpu_count(),
            "process_cpu_affinity": affinity,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "environment": {
                key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS
            },
            "trainer_dataloader_workers": 2,
            "trainer_train_batch_size": 512,
            "trainer_validation_batch_size": 1024,
        },
        "torch_build": {
            "git_version": getattr(torch.version, "git_version", None),
            "debug": getattr(torch.version, "debug", None),
            "cuda_build_version": torch.version.cuda,
            "hip_build_version": getattr(torch.version, "hip", None),
            "configuration": torch.__config__.show(),
        },
        "device": {
            "trainer_selector": "cuda if torch.cuda.is_available() else cpu",
            "selected_type": "cuda" if cuda_available else "cpu",
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_current_device": (
                torch.cuda.current_device() if cuda_available else None
            ),
            "devices": devices,
            "pin_memory": cuda_available,
            "cudnn_available": bool(torch.backends.cudnn.is_available()),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "driver": nvidia_driver_state(cuda_available),
        },
    }
    environment["identity_sha256"] = canonical_json_sha256(environment)
    return environment


def fixed_child_command() -> list[str]:
    return [
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


def build_preflight_record(
    *,
    repo: Path = REPO,
    mode: str,
    invocation_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    if mode not in {"dry-run", "execute"}:
        raise ValueError(f"invalid preflight mode: {mode}")

    artifacts = training_artifact_targets(repo)
    preflight_path, run_path = record_targets(repo)
    failure_path = failure_record_path(repo)
    transcript_path = child_transcript_path(repo)
    target_absence = assert_targets_absent(
        artifacts,
        label="E14 matched-real checkpoint/log target",
        repo=repo,
    )
    record_absence = assert_targets_absent(
        (preflight_path, run_path, failure_path, transcript_path),
        label="E14 matched-real record/transcript target",
        repo=repo,
    )
    frozen_registered = verify_hashed_inputs(
        FROZEN_REGISTERED_INPUTS,
        repo=repo,
        registration_scope="PREREG Section 3.1 plus PREREG itself",
    )
    additional_runtime = verify_hashed_inputs(
        ADDITIONAL_RUNTIME_INPUTS,
        repo=repo,
        registration_scope="implementation-commit freeze of actual validation input",
    )
    derived_standardizer = verify_derived_standardizer(repo=repo)
    source = verify_clean_committed_source(repo=repo)
    environment = capture_environment(repo=repo)
    child = fixed_child_command()
    invocation = list(invocation_argv if invocation_argv is not None else sys.argv)

    return {
        "schema_version": "e14.matched_real_training_preflight.v1",
        "record_type": "matched_real_training_preflight",
        "status": "T-PASS-PREFLIGHT",
        "mode": mode,
        "created_utc": utc_now(),
        "pre_result": True,
        "factorial_outcomes_computed": False,
        "source_provenance": source,
        "frozen_inputs": {
            "registered": frozen_registered,
            "additional_actual_trainer_inputs": additional_runtime,
            "derived_standardizer": derived_standardizer,
            "all_hashes_match": True,
        },
        "environment": environment,
        "invocation": {
            "argv": invocation,
            "display": shlex.join(invocation),
            "argv_sha256": command_sha256(invocation),
            "cwd": str(Path.cwd()),
            "wrapper_path": repo_relative(Path(__file__), repo),
            "wrapper_sha256": sha256_file(Path(__file__)),
        },
        "child_command": {
            "argv": child,
            "display": shlex.join(child),
            "argv_sha256": command_sha256(child),
            "cwd": str(repo),
            "environment_policy": "inherit wrapper environment without overrides",
            "sampling_policy": "legacy (frozen trainer default; no override flag)",
        },
        "fixed_protocol": {
            "setting": SETTING,
            "seeds": list(SEEDS),
            "budget_mode": "matched_steps",
            "max_optimizer_steps": MAX_STEPS,
            "validation_cadence_optimizer_steps": VALIDATION_CADENCE,
            "validation_checkpoints": EXPECTED_VALIDATION_CHECKPOINTS,
            "model_tag": MODEL_TAG,
            "sampling_policy": "legacy",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "loss": "balanced cross-entropy",
            "standardization": "complete real train only",
            "checkpoint_selection": "highest validation multiclass macro-F1",
        },
        "target_absence": target_absence,
        "record_target_absence": record_absence,
    }


def atomic_publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON using a same-directory, no-clobber hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path_lexists(path):
        raise FileExistsError(f"refusing to overwrite existing record: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"record appeared during atomic publication: {path}"
            ) from exc
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_publish_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes using a same-directory, no-clobber hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path_lexists(path):
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"output appeared during atomic publication: {path}"
            ) from exc
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def captured_output_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


def replay_captured_output(stream: Any, payload: bytes) -> None:
    if not payload:
        return
    binary_stream = getattr(stream, "buffer", None)
    if binary_stream is not None:
        binary_stream.write(payload)
        binary_stream.flush()
        return
    stream.write(payload.decode("utf-8", errors="replace"))
    stream.flush()


def persist_child_transcript(
    completed: subprocess.CompletedProcess[Any],
    path: Path,
    *,
    repo: Path = REPO,
) -> dict[str, Any]:
    stdout = captured_output_bytes(completed.stdout)
    stderr = captured_output_bytes(completed.stderr)
    skip_lines = [
        line
        for line in stdout.decode("utf-8", errors="replace").splitlines()
        if line.startswith("skip existing ")
    ]
    replay_captured_output(sys.stdout, stdout)
    replay_captured_output(sys.stderr, stderr)
    payload = b"".join(
        (
            b"E14 matched-real frozen trainer transcript v1\n",
            f"returncode={completed.returncode}\n".encode("ascii"),
            f"stdout_bytes={len(stdout)}\n".encode("ascii"),
            f"stderr_bytes={len(stderr)}\n".encode("ascii"),
            b"\n--- stdout ---\n",
            stdout,
            b"\n--- stderr ---\n",
            stderr,
        )
    )
    atomic_publish_bytes(path, payload)
    return {
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": "e14.matched_real_training_child_output.v1",
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "streams_preserved_separately": True,
        "existing_target_skip_lines": skip_lines,
    }


def path_state(path: Path, *, repo: Path = REPO) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": repo_relative(path, repo),
        "lexists": path_lexists(path),
    }
    if not record["lexists"]:
        return record
    try:
        status = path.lstat()
        record.update(
            {
                "is_symlink": path.is_symlink(),
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
                "mode": oct(status.st_mode & 0o7777),
            }
        )
        if record["is_file"] and not record["is_symlink"]:
            record["bytes"] = status.st_size
            record["sha256"] = sha256_file(path)
    except OSError as exc:
        record["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return record


def publish_failure_record(
    *,
    path: Path,
    preflight: Mapping[str, Any],
    preflight_path: Path,
    preflight_hash: str,
    stage: str,
    error: BaseException,
    execute_start_utc: str,
    failure_utc: str,
    duration_seconds: float,
    child_exit_status: int | None,
    transcript: Mapping[str, Any] | None,
    repo: Path = REPO,
) -> None:
    payload = {
        "schema_version": "e14.matched_real_training_failure.v1",
        "record_type": "matched_real_training_failure",
        "status": "T-FAIL-TRAINING",
        "factorial_outcomes_computed": False,
        "failure_stage": stage,
        "execute_start_utc": execute_start_utc,
        "failure_utc": failure_utc,
        "duration_seconds": duration_seconds,
        "source_commit": preflight["source_provenance"]["head_commit"],
        "child_command": preflight["child_command"],
        "child_exit_status": child_exit_status,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "preflight": {
            "path": repo_relative(preflight_path, repo),
            "bytes": preflight_path.stat().st_size,
            "sha256": preflight_hash,
            "schema_version": preflight["schema_version"],
        },
        "child_transcript": transcript,
        "training_artifact_states": [
            path_state(target, repo=repo)
            for target in training_artifact_targets(repo)
        ],
        "record_target_states_before_failure_publication": {
            "run": path_state(record_targets(repo)[1], repo=repo),
            "transcript": path_state(child_transcript_path(repo), repo=repo),
        },
        "preservation_policy": (
            "no overwrite; preflight, transcript, and any partial "
            "no-clobber trainer artifacts are preserved"
        ),
    }
    atomic_publish_json(path, payload)


def assert_execution_snapshot_unchanged(
    preflight: Mapping[str, Any], *, repo: Path = REPO
) -> None:
    observed_head = run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    expected_head = preflight["source_provenance"]["head_commit"]
    if observed_head != expected_head:
        raise E14TrainingError(
            f"source commit changed after preflight: {expected_head} -> {observed_head}"
        )
    for record in preflight["source_provenance"]["source_files"].values():
        path = repo / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise E14TrainingError(
                f"source changed after preflight: {record['path']}"
            )
    for group in (
        preflight["frozen_inputs"]["registered"],
        preflight["frozen_inputs"]["additional_actual_trainer_inputs"],
    ):
        for record in group.values():
            path = repo / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise E14TrainingError(
                    f"frozen input changed after preflight: {record['path']}"
                )


def allowed_post_training_status_paths(repo: Path = REPO) -> set[str]:
    preflight_path, _ = record_targets(repo)
    return {
        repo_relative(preflight_path, repo),
        repo_relative(child_transcript_path(repo), repo),
        *(repo_relative(training_log_path(seed, repo), repo) for seed in SEEDS),
    }


def assert_prelaunch_source_clean(
    preflight: Mapping[str, Any], *, repo: Path = REPO
) -> None:
    assert_execution_snapshot_unchanged(preflight, repo=repo)
    preflight_path, _ = record_targets(repo)
    allowed = {repo_relative(preflight_path, repo)}
    records = git_status_records(repo)
    blocking = [
        record
        for record in records
        if not (record.startswith("?? ") and record[3:] in allowed)
    ]
    if blocking:
        raise E14TrainingError(
            "source/worktree changed between preflight and child launch: "
            + json.dumps(blocking, ensure_ascii=True)
        )
    wisa_status = git_status_records(repo, "wisa")
    if wisa_status:
        raise E14TrainingError(
            "wisa/ changed between preflight and child launch: "
            + json.dumps(wisa_status, ensure_ascii=True)
        )


def assert_post_training_source_clean(
    preflight: Mapping[str, Any], *, repo: Path = REPO
) -> dict[str, Any]:
    assert_execution_snapshot_unchanged(preflight, repo=repo)
    allowed = allowed_post_training_status_paths(repo)
    records = git_status_records(repo)
    blocking: list[str] = []
    accepted: list[str] = []
    for record in records:
        if record.startswith("?? ") and record[3:] in allowed:
            accepted.append(record)
        else:
            blocking.append(record)
    if blocking:
        raise E14TrainingError(
            "source/worktree changed during training: "
            + json.dumps(blocking, ensure_ascii=True)
        )
    wisa_status = git_status_records(repo, "wisa")
    if wisa_status:
        raise E14TrainingError(
            "wisa/ changed during training: "
            + json.dumps(wisa_status, ensure_ascii=True)
        )
    return {
        "head_commit_unchanged": True,
        "source_hashes_unchanged": True,
        "frozen_input_hashes_unchanged": True,
        "wisa_clean": True,
        "accepted_generated_status_records": sorted(accepted),
        "blocking_status_records": [],
    }


def require_exact(record: Mapping[str, Any], key: str, expected: Any) -> None:
    observed = record.get(key)
    if observed != expected:
        raise E14TrainingError(
            f"training log field {key!r}: expected {expected!r}, got {observed!r}"
        )


def finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E14TrainingError(f"{field} is not numeric: {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise E14TrainingError(f"{field} is not finite: {value!r}")
    return converted


def validate_training_log(payload: Mapping[str, Any], seed: int) -> dict[str, Any]:
    require_exact(payload, "family", "cnn")
    require_exact(payload, "setting", SETTING)
    require_exact(payload, "seed", seed)
    require_exact(payload, "budget_mode", "matched_steps")
    require_exact(payload, "model_tag", MODEL_TAG)
    require_exact(payload, "sampling_policy", "legacy")
    require_exact(payload, "sampling_audit", None)
    require_exact(payload, "optimizer_steps", MAX_STEPS)
    require_exact(payload, "real_train_windows", EXPECTED_REAL_TRAIN_WINDOWS)
    require_exact(payload, "train_windows", EXPECTED_REAL_TRAIN_WINDOWS)
    require_exact(
        payload, "validation_checkpoints", EXPECTED_VALIDATION_CHECKPOINTS
    )
    require_exact(payload, "epochs_run", EXPECTED_VALIDATION_CHECKPOINTS)
    require_exact(
        payload,
        "epochs_run_semantics",
        "validation_checkpoints_not_augmented_data_epochs",
    )
    require_exact(
        payload, "selected_epoch_semantics", "validation_checkpoint"
    )

    history = payload.get("history")
    if not isinstance(history, list) or len(history) != EXPECTED_VALIDATION_CHECKPOINTS:
        raise E14TrainingError(
            "training log must contain exactly 12 validation history rows"
        )
    scores: list[float] = []
    for position, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            raise E14TrainingError(f"history row {position} is not an object")
        require_exact(row, "epoch", position)
        require_exact(row, "optimizer_step", position * VALIDATION_CADENCE)
        loss = finite_number(row.get("train_loss"), field=f"history[{position}].train_loss")
        score = finite_number(
            row.get("val_macro_f1"),
            field=f"history[{position}].val_macro_f1",
        )
        if loss < 0:
            raise E14TrainingError(f"history row {position} has negative loss")
        if not 0 <= score <= 1:
            raise E14TrainingError(
                f"history row {position} has invalid validation macro-F1"
            )
        scores.append(score)

    expected_selected = max(range(len(scores)), key=scores.__getitem__) + 1
    require_exact(payload, "selected_epoch", expected_selected)
    best_score = finite_number(
        payload.get("best_val_macro_f1"), field="best_val_macro_f1"
    )
    if best_score != scores[expected_selected - 1]:
        raise E14TrainingError(
            "best_val_macro_f1 does not match the selected history row"
        )
    elapsed = finite_number(payload.get("elapsed_seconds"), field="elapsed_seconds")
    if elapsed < 0:
        raise E14TrainingError("elapsed_seconds must be non-negative")
    return {
        "selected_validation_checkpoint": expected_selected,
        "selected_optimizer_step": expected_selected * VALIDATION_CADENCE,
        "best_val_macro_f1": best_score,
        "optimizer_steps": MAX_STEPS,
        "validation_checkpoints": EXPECTED_VALIDATION_CHECKPOINTS,
        "history_rows_valid": True,
    }


def validate_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise E14TrainingError(f"checkpoint is not a regular file: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise E14TrainingError(
            f"checkpoint failed weights_only load: {path}"
        ) from exc
    if not isinstance(state, Mapping):
        raise E14TrainingError(f"checkpoint is not a state dict: {path}")
    if set(state) != set(EXPECTED_STATE_SHAPES):
        missing = sorted(set(EXPECTED_STATE_SHAPES) - set(state))
        unexpected = sorted(set(state) - set(EXPECTED_STATE_SHAPES))
        raise E14TrainingError(
            f"checkpoint key mismatch for {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    tensor_records: dict[str, dict[str, Any]] = {}
    for key, expected_shape in EXPECTED_STATE_SHAPES.items():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise E14TrainingError(f"checkpoint value is not a tensor: {key}")
        observed_shape = tuple(tensor.shape)
        if observed_shape != expected_shape:
            raise E14TrainingError(
                f"checkpoint shape mismatch for {key}: "
                f"expected {expected_shape}, got {observed_shape}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise E14TrainingError(f"checkpoint contains nonfinite tensor: {key}")
        tensor_records[key] = {
            "shape": list(observed_shape),
            "dtype": str(tensor.dtype),
            "finite": True,
        }
    return {
        "weights_only_load": True,
        "state_dict_keys": len(state),
        "expected_state_dict_keys": len(EXPECTED_STATE_SHAPES),
        "architecture_compatible": True,
        "all_tensors_finite": True,
        "tensors": tensor_records,
    }


def read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E14TrainingError(f"could not read JSON object: {path}") from exc
    if not isinstance(payload, Mapping):
        raise E14TrainingError(f"JSON root is not an object: {path}")
    return payload


def validate_training_bundle(
    *, repo: Path = REPO
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        checkpoint = checkpoint_path(seed, repo)
        log = training_log_path(seed, repo)
        if (
            not checkpoint.is_file()
            or checkpoint.is_symlink()
            or not log.is_file()
            or log.is_symlink()
        ):
            raise E14TrainingError(
                "incomplete or non-regular matched-real training bundle "
                f"for seed {seed}"
            )
        log_validation = validate_training_log(read_json_object(log), seed)
        checkpoint_validation = validate_checkpoint(checkpoint)
        records.append(
            {
                "seed": seed,
                "checkpoint": {
                    "path": repo_relative(checkpoint, repo),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": sha256_file(checkpoint),
                    "validation": checkpoint_validation,
                },
                "log": {
                    "path": repo_relative(log, repo),
                    "bytes": log.stat().st_size,
                    "sha256": sha256_file(log),
                    "validation": log_validation,
                },
                **log_validation,
            }
        )
    if len(records) != len(SEEDS):
        raise AssertionError("five matched-real bundles are required")
    return records


def execute_training(preflight: dict[str, Any], *, repo: Path = REPO) -> dict[str, Any]:
    if preflight.get("mode") != "execute":
        raise E14TrainingError(
            "training launch requires an explicit execute-mode preflight"
        )
    preflight_path, run_path = record_targets(repo)
    failure_path = failure_record_path(repo)
    transcript_path = child_transcript_path(repo)
    atomic_publish_json(preflight_path, preflight)
    preflight_hash = sha256_file(preflight_path)
    execute_start_utc = utc_now()
    execute_start_monotonic = time.monotonic()
    stage = "prelaunch_validation"
    child_exit_status: int | None = None
    transcript: dict[str, Any] | None = None

    try:
        # Close the interval between the original checks and child launch.  The
        # preflight record itself is now the single allowed generated
        # source-tree path; every exact output target must still be absent.
        assert_prelaunch_source_clean(preflight, repo=repo)
        assert_targets_absent(
            training_artifact_targets(repo),
            label=(
                "E14 matched-real checkpoint/log target immediately "
                "before launch"
            ),
            repo=repo,
        )
        assert_targets_absent(
            (run_path, failure_path, transcript_path),
            label=(
                "E14 matched-real run/failure/transcript target immediately "
                "before launch"
            ),
            repo=repo,
        )

        child = list(preflight["child_command"]["argv"])
        stage = "child_execution"
        start_utc = utc_now()
        start_monotonic = time.monotonic()
        completed = subprocess.run(
            child,
            cwd=repo,
            check=False,
            capture_output=True,
        )
        end_utc = utc_now()
        duration = time.monotonic() - start_monotonic
        child_exit_status = completed.returncode

        stage = "child_transcript_publication"
        transcript = persist_child_transcript(
            completed,
            transcript_path,
            repo=repo,
        )
        if transcript["existing_target_skip_lines"]:
            stage = "child_output_validation"
            raise E14TrainingError(
                "frozen trainer reported an existing-target skip after the "
                "wrapper's exact-target absence gate: "
                + json.dumps(
                    transcript["existing_target_skip_lines"],
                    ensure_ascii=True,
                )
            )
        if completed.returncode != 0:
            stage = "child_execution"
            raise E14TrainingError(
                "frozen trainer failed with exit status "
                f"{completed.returncode}; preflight, transcript, and any "
                "partial no-clobber artifacts are preserved"
            )

        stage = "post_validation"
        source_post = assert_post_training_source_clean(preflight, repo=repo)
        environment_post = capture_environment(repo=repo)
        environment_matches = environment_post == preflight["environment"]
        if not environment_matches:
            raise E14TrainingError(
                "post-training environment/device identity differs from preflight"
            )
        artifacts = validate_training_bundle(repo=repo)

        run_record = {
            "schema_version": "e14.matched_real_training_run.v1",
            "record_type": "matched_real_training_run",
            "status": "T-PASS-TRAINING",
            "factorial_outcomes_computed": False,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "duration_seconds": duration,
            "source_commit": preflight["source_provenance"]["head_commit"],
            "source_post_validation": source_post,
            "child_command": preflight["child_command"],
            "child_exit_status": completed.returncode,
            "child_transcript": transcript,
            "preflight": {
                "path": repo_relative(preflight_path, repo),
                "bytes": preflight_path.stat().st_size,
                "sha256": preflight_hash,
                "schema_version": preflight["schema_version"],
            },
            "environment_post": environment_post,
            "environment_pre_identity_sha256": preflight["environment"][
                "identity_sha256"
            ],
            "environment_post_identity_sha256": environment_post[
                "identity_sha256"
            ],
            "environment_matches_preflight": True,
            "artifacts": artifacts,
            "validation": {
                "expected_seed_count": len(SEEDS),
                "validated_seed_count": len(artifacts),
                "seeds": list(SEEDS),
                "all_logs_valid": True,
                "all_checkpoints_weights_only_load": True,
                "all_checkpoint_architectures_compatible": True,
                "all_checkpoint_tensors_finite": True,
                "all_artifact_hashes_recorded": True,
                "preflight_cross_linked": True,
                "child_output_persisted": True,
                "child_reported_existing_target_skips": False,
                "no_e14_factorial_inference": True,
            },
        }
        stage = "run_record_publication"
        atomic_publish_json(run_path, run_record)
        return run_record
    except BaseException as exc:
        failure_utc = utc_now()
        failure_duration = time.monotonic() - execute_start_monotonic
        try:
            publish_failure_record(
                path=failure_path,
                preflight=preflight,
                preflight_path=preflight_path,
                preflight_hash=preflight_hash,
                stage=stage,
                error=exc,
                execute_start_utc=execute_start_utc,
                failure_utc=failure_utc,
                duration_seconds=failure_duration,
                child_exit_status=child_exit_status,
                transcript=transcript,
                repo=repo,
            )
        except BaseException as publication_error:
            raise E14TrainingError(
                f"{exc}; additionally failed to publish the versioned failure "
                f"record without overwrite: {publication_error}"
            ) from exc
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or explicitly execute the preregistered E14 matched-real "
            "training bundle. Default: read-only dry run."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every preflight gate without writing or training (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="publish preflight, run the frozen child command, and validate outputs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete validation/run record",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "execute" if args.execute else "dry-run"
    invocation_tail = list(argv) if argv is not None else sys.argv[1:]
    invocation = [sys.argv[0], *invocation_tail]
    try:
        preflight = build_preflight_record(
            repo=REPO, mode=mode, invocation_argv=invocation
        )
        if mode == "dry-run":
            if args.json:
                print(json.dumps(preflight, indent=2, sort_keys=True))
            else:
                print(
                    "E14 matched-real dry run passed: "
                    "10 checkpoint/log targets absent; frozen hashes, clean "
                    "committed source, .venv, environment, and device captured. "
                    "No files were written and no training was started."
                )
            return 0

        run_record = execute_training(preflight, repo=REPO)
        if args.json:
            print(json.dumps(run_record, indent=2, sort_keys=True))
        else:
            print(
                "E14 matched-real training passed: five logs/checkpoints "
                "validated and the post-training record was atomically published."
            )
        return 0
    except (E14TrainingError, FileExistsError, OSError, ValueError) as exc:
        print(f"E14 matched-real training gate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
