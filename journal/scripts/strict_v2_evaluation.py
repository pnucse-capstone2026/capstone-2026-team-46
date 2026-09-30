"""Strict-v2 evaluator scope, source, and input-provenance helpers.

Evaluators remain responsible for calling
``lib_common.validate_strict_v2_training_bundle`` with their exact trainer
contract before inference.  This module turns that validated bundle into a
compact record and hard-gates canonical evaluation on the committed evaluator
source plus the frozen, gitignored inputs that the requested scope consumes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lib_common as lc
import numpy as np


STRICT_V2_EVALUATION_INPUTS = {
    "train_windows": {
        "path": lc.WINDOWS / "train_windows.npz",
        "sha256": (
            "44b3bedeebe113e4850ebb2e38314b413f140f350712e0e9"
            "eda81a34faec7cb4"
        ),
    },
    "val_windows": {
        "path": lc.WINDOWS / "val_windows.npz",
        "sha256": (
            "b4920e7f60fbf38b38e12976129f120ebff160a21a4c770"
            "93f3746f3c1aa218e"
        ),
    },
    "test_windows": {
        "path": lc.WINDOWS / "test_windows.npz",
        "sha256": (
            "4ac72a497f5bfc7e59ce8baf08b78c5aebbf5e060fb4f2d"
            "73629a591c56b54b7"
        ),
    },
    "variant_test_windows": {
        "path": lc.WINDOWS / "variant_test_windows.npz",
        "sha256": (
            "91bd4d5dca2bd73907fad11bb94ea6dec075f60bb3b24e3c"
            "517617bf2d50b2ba"
        ),
    },
    "variant_sensitivity_windows": {
        "path": lc.WINDOWS / "variant_sensitivity_windows.npz",
        "sha256": (
            "4266fb3bbb3e248ef1ab53aaca19cb4534a3815a18c9134d"
            "c3db446cdfcf5674"
        ),
    },
    "target_id_shift_stress_windows": {
        "path": lc.WINDOWS / "target_id_shift_stress_windows.npz",
        "sha256": (
            "9f685e629b4c8acaf3312d95dd1d94145c502b2c07fedd8d"
            "a9a938526ed92bc1"
        ),
    },
    "out_of_generator_stress_windows": {
        "path": lc.WINDOWS / "out_of_generator_stress_windows.npz",
        "sha256": (
            "c91e5a026f9f0b771e9551a471efeb1faca66bda21e7035b"
            "c5b3620927dfa346"
        ),
    },
    "otids_cross_windows": {
        "path": lc.WINDOWS / "otids_cross_windows.npz",
        "sha256": (
            "b521f3546f633c6220e2696234abb6d7c37a03d64f19e71e"
            "c37bf36a100ed596"
        ),
    },
    "can_train_and_test_archive": {
        "path": lc.ROOT / "datasets" / "can-train-and-test.zip",
        "sha256": (
            "a9c607b38bd28f1768021ad01c29ffbfe4e82bb0ae5815ac"
            "3ce7ad74751ae061"
        ),
    },
    "road_archive": {
        "path": lc.ROOT / "datasets" / "road.zip",
        "sha256": (
            "0e4fe6ed7f99b5cdabf6b772c91a2025c614a071ba8579c"
            "532904d0ef7aea5f6"
        ),
    },
    "road_dataset_profile": {
        "path": lc.WISA_TABLES / "road_dataset_profile.csv",
        "sha256": (
            "0d6202ae75e7e8e883ef5fb40fd205c205803b25307932c2"
            "b2bd9cf0eef7df83"
        ),
    },
}
STRICT_V2_ROAD_FRAMES_DIRECTORY = (
    lc.ROOT / "datasets" / "processed" / "road_frames"
)
STRICT_V2_ROAD_FRAMES_COUNT = 41
STRICT_V2_ROAD_FRAMES_MANIFEST_SHA256 = (
    "1d434f4043ec7fdfddba636f2c479eff35ef246a331586a72f20236a271a5c64"
)


def _file_record(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"strict-v2 evaluation input is missing: {path}")
    return {
        "path": portable_path(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": lc.sha256_file(path),
    }


def _file_manifest(paths) -> dict:
    records = [_file_record(path) for path in sorted(
        (Path(path) for path in paths),
        key=lambda path: portable_path(path),
    )]
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "files": records,
        "file_count": len(records),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest_format": (
            "sha256(canonical-json sorted records[path,size_bytes,sha256])"
        ),
    }


def evaluation_model_input_provenance(artifacts: dict[str, Path]) -> dict:
    """Hash every checkpoint/scaler actually requested by a mixed evaluator."""
    if not artifacts:
        raise ValueError("strict-v2 evaluation model-input set is empty")
    records = {
        str(name): _file_record(path)
        for name, path in sorted(artifacts.items())
    }
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "policy": (
            "all strict-tagged and retained-untagged evaluator model inputs "
            "hashed before inference; no missing-input fallback"
        ),
        "artifacts": records,
        "artifact_count": len(records),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest_format": "sha256(canonical-json artifact mapping)",
    }


def evaluator_source_and_input_provenance(
    entrypoint: str | Path,
    input_keys,
    *,
    include_road_frames: bool = False,
    training_bundles=None,
) -> dict:
    """Hard-gate and record source plus inputs for a strict-v2 evaluator.

    The source check must run before inference or output creation.  Known
    gitignored arrays and raw archives are compared with frozen SHA-256
    anchors.  ROAD's processed capture directory is additionally bound by a
    sorted per-file manifest because those parquet files, not the zip bytes,
    are consumed by the evaluators.
    """
    source = lc._source_provenance()
    training_source_commits = []
    if training_bundles is not None:
        training_source_commits = sorted({
            str(bundle.get("sampling_audit", {}).get("source_commit") or "")
            for bundle in training_bundles
        })
        if training_source_commits != [source["source_commit"]]:
            raise RuntimeError(
                "strict-v2 evaluator/training source-commit mismatch; "
                "canonical evaluation requires one shared clean commit: "
                + json.dumps({
                    "evaluation_source_commit": source["source_commit"],
                    "training_source_commits": training_source_commits,
                }, sort_keys=True)
            )
    entrypoint_record = _file_record(Path(entrypoint))
    helper_records = {
        "strict_v2_evaluation": _file_record(Path(__file__)),
        "lib_common": _file_record(Path(lc.__file__)),
    }

    requested = [str(key) for key in input_keys]
    if len(requested) != len(set(requested)):
        raise ValueError("strict-v2 evaluation input keys contain duplicates")
    unknown = sorted(set(requested) - set(STRICT_V2_EVALUATION_INPUTS))
    if unknown:
        raise ValueError(f"unknown strict-v2 evaluation input key(s): {unknown}")
    inputs = {}
    mismatches = {}
    for key in requested:
        expected = STRICT_V2_EVALUATION_INPUTS[key]
        observed = _file_record(expected["path"])
        observed["expected_sha256"] = expected["sha256"]
        observed["verified_frozen_sha256"] = (
            observed["sha256"] == expected["sha256"]
        )
        inputs[key] = observed
        if not observed["verified_frozen_sha256"]:
            mismatches[key] = {
                "expected": expected["sha256"],
                "observed": observed["sha256"],
            }

    road_frames = None
    if include_road_frames:
        road_paths = sorted(
            STRICT_V2_ROAD_FRAMES_DIRECTORY.glob("*.parquet")
        )
        road_frames = _file_manifest(road_paths)
        road_frames.update({
            "directory": portable_path(STRICT_V2_ROAD_FRAMES_DIRECTORY),
            "expected_file_count": STRICT_V2_ROAD_FRAMES_COUNT,
            "expected_manifest_sha256": (
                STRICT_V2_ROAD_FRAMES_MANIFEST_SHA256
            ),
        })
        if road_frames["file_count"] != STRICT_V2_ROAD_FRAMES_COUNT:
            mismatches["road_frames.file_count"] = {
                "expected": STRICT_V2_ROAD_FRAMES_COUNT,
                "observed": road_frames["file_count"],
            }
        if (
            road_frames["manifest_sha256"]
            != STRICT_V2_ROAD_FRAMES_MANIFEST_SHA256
        ):
            mismatches["road_frames.manifest_sha256"] = {
                "expected": STRICT_V2_ROAD_FRAMES_MANIFEST_SHA256,
                "observed": road_frames["manifest_sha256"],
            }
        road_frames["verified_frozen_manifest"] = not any(
            key.startswith("road_frames.") for key in mismatches
        )

    if mismatches:
        raise RuntimeError(
            "strict-v2 evaluation input provenance mismatch; do not publish "
            f"canonical outputs: {json.dumps(mismatches, sort_keys=True)}"
        )

    record = {
        "schema_version": "strict_v2_evaluator_provenance_v1",
        "source": source,
        "evaluation_entrypoint": entrypoint_record,
        "source_helpers": helper_records,
        "evaluation_inputs": inputs,
    }
    if training_bundles is not None:
        record["verified_training_source_commits"] = (
            training_source_commits
        )
    if road_frames is not None:
        record["road_frames"] = road_frames
    return record


def _array_record(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(
            header + b"\0" + contiguous.tobytes(order="C")
        ).hexdigest(),
    }


def load_identical_saved_standardizer(paths) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the exact persisted scaler shared by strict recurrent models."""
    normalized = sorted(
        {Path(path).resolve() for path in paths},
        key=portable_path,
    )
    if not normalized:
        raise ValueError("strict-v2 saved-standardizer set is empty")

    baseline_mean = None
    baseline_std = None
    artifacts = []
    for path in normalized:
        artifact = _file_record(path)
        with np.load(path, allow_pickle=False) as saved:
            missing = sorted({"mean", "std"} - set(saved.files))
            if missing:
                raise RuntimeError(
                    f"strict-v2 standardizer {path} misses arrays: {missing}"
                )
            mean = np.asarray(saved["mean"]).copy()
            std = np.asarray(saved["std"]).copy()
        if (
            mean.shape != std.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise RuntimeError(
                f"strict-v2 standardizer arrays are invalid: {path}"
            )
        if baseline_mean is None:
            baseline_mean = mean
            baseline_std = std
        elif (
            _array_record(mean) != _array_record(baseline_mean)
            or _array_record(std) != _array_record(baseline_std)
        ):
            raise RuntimeError(
                "strict-v2 saved standardizers are not byte-exact equal; "
                f"first={normalized[0]} mismatch={path}"
            )
        artifact.update({
            "mean": _array_record(mean),
            "std": _array_record(std),
        })
        artifacts.append(artifact)

    assert baseline_mean is not None and baseline_std is not None
    return baseline_mean, baseline_std, {
        "policy": (
            "exact persisted training standardizer; all requested saved "
            "mean/std arrays must be byte-exact equal"
        ),
        "artifacts": artifacts,
        "shared_mean": _array_record(baseline_mean),
        "shared_std": _array_record(baseline_std),
    }


def validate_strict_v2_evaluation_scope(
    model_tag: str,
    output_tag: str,
    seeds: list[int] | tuple[int, ...] | None = None,
    *,
    reduced_scope: bool = False,
) -> None:
    """Bind strict-labelled outputs to strict models and a full seed set.

    Canonical strict-v2 summaries are five-pipeline-seed artifacts. A partial
    seed set is permitted only for an explicitly non-canonical smoke run whose
    model and output tags both say ``smoke``.
    """
    model_tag = str(model_tag or "")
    output_tag = str(output_tag or "")
    strict_model = lc.is_strict_v2_tag(model_tag)
    strict_output = lc.is_strict_v2_tag(output_tag)
    if strict_output and not strict_model:
        raise ValueError(
            "a strict-v2 output tag requires an explicit strict-v2 model tag"
        )
    if not strict_model:
        return
    if not strict_output:
        raise ValueError(
            "a strict-v2 model tag requires a strict-v2 output tag"
        )
    if seeds is not None:
        requested = [int(seed) for seed in seeds]
        full_seed_set = (
            len(requested) == len(lc.SEEDS)
            and len(requested) == len(set(requested))
            and set(requested) == set(lc.SEEDS)
        )
        if not full_seed_set and (
            "smoke" not in model_tag.lower()
            or "smoke" not in output_tag.lower()
        ):
            raise ValueError(
                "partial strict-v2 seed sets require both model and output "
                "tags to contain 'smoke'"
            )
    normalized_model_tag = model_tag.strip("_")
    if normalized_model_tag not in output_tag:
        raise ValueError(
            "a strict-v2 output tag must contain the complete model tag"
        )
    if (
        reduced_scope
        and output_tag.strip("_") == normalized_model_tag
    ):
        raise ValueError(
            "a reduced strict-v2 evaluation scope requires an additional "
            "output scope token beyond the model tag"
        )


def cnn_budget_mode_from_tag(tag: str) -> str:
    """Map the trainer's explicit matched-step tag to its budget contract."""
    return (
        "matched_steps"
        if "matchedsteps" in str(tag).lower()
        else "legacy"
    )


def transformer_expected_extras(tag: str) -> dict:
    """Canonical strict transformer evaluations must use the full train set."""
    return {} if "smoke" in str(tag).lower() else {"train_subset": None}


def portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(lc.ROOT.parent.resolve()))
    except ValueError:
        return str(resolved)


def source_standardizer_provenance(path: Path | None = None) -> dict:
    source = Path(path) if path is not None else lc.WINDOWS / "train_windows.npz"
    return {
        "source": portable_path(source),
        "source_sha256": lc.sha256_file(source),
        "derivation": (
            "feature mean/std over source-train windows and time; "
            "std<1e-6 replaced by 1.0"
        ),
    }


def bundle_provenance(
    *,
    family: str,
    setting: str,
    seed: int,
    tag: str,
    log_path: Path,
    artifacts: dict[str, Path],
    validated_record: dict,
) -> dict:
    audit = validated_record.get("sampling_audit", {})
    audit_keys = [
        "sampling_policy",
        "pool_identifier",
        "pool_path",
        "pool_sha256",
        "config_sha256",
        "real_data_provenance",
        "source_commit",
    ]
    sampling = audit.get("sampling")
    if isinstance(sampling, dict):
        indices_sha256 = sampling.get("indices_sha256")
    else:
        indices_sha256 = None
    compact_audit = {
        key: audit[key]
        for key in audit_keys
        if key in audit
    }
    if indices_sha256 is not None:
        compact_audit["indices_sha256"] = indices_sha256
    contract_keys = [
        "family",
        "setting",
        "seed",
        "model_tag",
        "tag",
        "sampling_policy",
        "budget_mode",
        "train_subset",
    ]
    return {
        "family": family,
        "setting": setting,
        "seed": int(seed),
        "tag": tag,
        "sampling_policy": "strict-v2",
        "validated_training_contract": {
            key: validated_record[key]
            for key in contract_keys
            if key in validated_record
        },
        "training_log": {
            "path": portable_path(log_path),
            "sha256": lc.sha256_file(log_path),
        },
        "artifacts": {
            name: {
                "path": portable_path(path),
                "sha256": lc.sha256_file(path),
            }
            for name, path in sorted(artifacts.items())
        },
        "sampling_audit": compact_audit,
    }
