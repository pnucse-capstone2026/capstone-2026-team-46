#!/usr/bin/env python3
"""Fit the 245 preregistered E17 Random-Forest arms.

Registered by ``journal/experiments/e17_rf_relational_replication/PREREG.md``
sections 5, 6, 14.2, and 15, as amended by
``AMENDMENT_2026-08-07_V1_BRIDGE_POOL_AUDIT_SOURCE.md`` (gate G1i reference
table for the four bridge Rule pools).

Grid (PREREG section 6)
-----------------------
* 20 generator realizations x 2 arms (Rule, placebo) x 5 pipeline seeds = 200
* 4 bridge realizations x 2 arms x 5 pipeline seeds                    =  40
* 5 shared Random-Forest real-only references                          =   5

E16 fit only the placebo arm for the bridge realizations because the frozen E15
CNN Rule checkpoints could be reused.  No frozen E15 Random-Forest checkpoint
exists, so E17 fits both bridge arms.  They remain a sensitivity and never enter
the primary ``n = 20``.

The matching contract (PREREG section 5.2)
------------------------------------------
A Random Forest has no optimizer-update budget, no early-stopping trajectory and
no minibatch order, so the CNN's 6,156-update matched budget has no analogue and
is not simulated.  It is replaced by the **identical-training-row contract**: the
E17 ``sampling_index_sha256`` for every ``(realization, arm, pipeline seed)``
must equal the frozen E16 CNN value.  Gate G10c-ii is blocking.  The 20 bridge
Rule fits have no E16 CNN counterpart and are recorded as explicitly exempt.

Stages
------
``(default)``       read-only preflight; performs no fitting.
``--verify-pools``  stage S2; hashes the 48 consumed pool slots, inherits the
                    frozen manipulation checks, publishes the audit.  No fit.
``--execute``       stages S3 and S4; 5 reference fits then 240 paired fits.

Every fit's synthetic pool is opened read-only.  Features are extracted once per
pool and reused across that pool's five pipeline seeds, which is exact: the
feature map is row-wise.  Importing this module performs no fitting.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib_common as lc  # noqa: E402
import generate_e16_rule_placebo_pairs as e16gen  # noqa: E402
import train_generator_extension_rf as rfmod  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _csv_bytes,
    _environment_record,
    _stage_bytes,
    _stage_json,
    atomic_publish_bundle,
)
from train_rule_construction_seed_crossing import (  # noqa: E402
    SAMPLING_SEED_OFFSET,
    SYNTHETIC_TOTAL,
    _load_real_train_val,
    sample_indices_without_replacement,
    sampling_indices_sha256,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "journal" / "experiments" / "e17_rf_relational_replication"
SYNTHETIC_DIR = REPO / "journal" / "datasets" / "synthetic"
MODEL_DIR = REPO / "journal" / "models" / "generator_extension"
LOG_DIR = REPO / "journal" / "results" / "logs"
TABLE_DIR = REPO / "journal" / "results" / "tables"

# --- registration provenance (PREREG header + freeze commit) ----------------

PREREG_PATH = "journal/experiments/e17_rf_relational_replication/PREREG.md"
#: Commit that froze the E17 registration.  The PREREG header names 42ec54c,
#: which was HEAD when the document was written; both are recorded and neither
#: is edited.
PREREG_COMMIT = "d18e027"
PREREG_SOURCE_COMMIT = "42ec54c"
AMENDMENT_PATH = ("journal/experiments/e17_rf_relational_replication/"
                  "AMENDMENT_2026-08-07_V1_BRIDGE_POOL_AUDIT_SOURCE.md")
AMENDMENT_COMMIT = "42ae5c5"

# --- frozen design constants (PREREG sections 3, 5.1, 6) --------------------

#: PREREG section 3.1, transcribed from the registration text.  G2i requires the
#: module's list to equal this literal; the gate fires if either drifts.
REGISTERED_CONSTRUCTION_SEEDS = (
    415637, 748203, 560657, 459336, 457191,
    999191, 715822, 298581, 922089, 864178,
    583909, 467232, 550427, 485791, 820632,
    837943, 222537, 685012, 192753, 739263,
)
#: PREREG section 3.2.
REGISTERED_BRIDGE_SEEDS = (271828, 161803, 141421, 173205)

CONSTRUCTION_SEEDS = e16gen.CONSTRUCTION_SEEDS
BRIDGE_SEEDS = e16gen.BRIDGE_SEEDS
PIPELINE_SEEDS = (7, 42, 123, 2026, 3407)
ARMS = ("rule", "placebo")

OUTPUT_VERSION = "v1"
MODEL_TAG = "e17_v1"
PER_ATTACK_CAP = 65_000
FEATURE_DIMENSION = 55

#: PREREG section 5.1, verbatim.  Gate G14 additionally requires these to equal
#: the keywords of the ``RandomForestClassifier`` call inside the frozen
#: ``train_generator_extension_rf.train_one``.
ESTIMATOR = {
    "n_estimators": 160,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": "seed",
}
#: ``random_state`` and ``n_jobs`` are excluded: the former is the pipeline seed
#: and the latter is a resource knob, not a model choice.
ESTIMATOR_FIDELITY_KEYS = (
    "n_estimators", "max_depth", "min_samples_leaf", "class_weight")

E16_POOL_AUDIT = TABLE_DIR / "e16_pool_audit_v2.csv"
E16_MANIPULATION_CHECKS = TABLE_DIR / "e16_manipulation_checks_v2.csv"
E16_TRAINING_MANIFEST = TABLE_DIR / "e16_training_manifest_v2.csv"
E15_POOL_AUDIT = TABLE_DIR / "e15_rule_construction_pool_audit_v2.csv"

POOL_INHERITANCE_SCHEMA = "e17.pool_inheritance.v1"
TRAINING_PREFLIGHT_SCHEMA = "e17.training_preflight.v1"
TRAINING_RUN_SCHEMA = "e17.training_run.v1"
TRAINING_LOG_SCHEMA = "e17.rf_training_log.v1"

EXPECTED_PRIMARY_FITS = len(CONSTRUCTION_SEEDS) * len(ARMS) * len(PIPELINE_SEEDS)
EXPECTED_BRIDGE_FITS = len(BRIDGE_SEEDS) * len(ARMS) * len(PIPELINE_SEEDS)
EXPECTED_REFERENCE_FITS = len(PIPELINE_SEEDS)
EXPECTED_FIT_COUNT = (
    EXPECTED_PRIMARY_FITS + EXPECTED_BRIDGE_FITS + EXPECTED_REFERENCE_FITS)


class E17TrainingError(RuntimeError):
    """Raised on any registered E17 training gate violation."""


def _stop(message: str) -> E17TrainingError:
    return E17TrainingError(f"T-STOP-E17-TRAIN: {message}")


def _incomplete(message: str) -> E17TrainingError:
    return E17TrainingError(f"T-INCOMPLETE-E17-TRAIN: {message}")


def utc_now() -> str:
    return e16gen.utc_now()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_and_cleanup(staged: Sequence[tuple[Path, Path]]) -> None:
    """Publish atomically, then remove the staged hard-link sources."""
    atomic_publish_bundle(staged)
    for source, _ in staged:
        try:
            Path(source).unlink()
        except FileNotFoundError:
            pass


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True, capture_output=True, text=True).stdout.strip()


def assert_registration_committed() -> dict[str, Any]:
    """S0/G0 — the registration and its amendment are committed ancestors."""
    record: dict[str, Any] = {
        "gate": "registration_committed",
        "passed": True,
        "preregistration_path": PREREG_PATH,
        "registration_source_commit_in_header": PREREG_SOURCE_COMMIT,
    }
    for label, rel_path, commit in (
            ("registration", PREREG_PATH, PREREG_COMMIT),
            ("amendment", AMENDMENT_PATH, AMENDMENT_COMMIT),
    ):
        path = REPO / rel_path
        if not path.exists():
            raise _stop(f"missing {label} document: {rel_path}")
        if _git("status", "--porcelain", "--", rel_path):
            raise _stop(f"{label} has uncommitted changes: {rel_path}")
        try:
            resolved = _git("rev-parse", commit)
            subprocess.run(
                ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                 commit, "HEAD"], check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise _stop(
                f"{label} commit {commit} is not an ancestor of HEAD") from exc
        record[f"{label}_commit"] = resolved
        record[f"{label}_commit_is_ancestor"] = True
        record[f"{label}_last_modified_commit"] = _git(
            "log", "-1", "--format=%H", "--", rel_path)
        record[f"{label}_sha256"] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    frozen = _git("status", "--porcelain", "--", "wisa")
    if frozen:
        raise _stop(f"the frozen wisa/ tree is dirty: {frozen.splitlines()[:3]}")
    record["wisa_tree_clean"] = True
    return record


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E17Job:
    construction_seed: int
    arm: str
    pipeline_seed: int
    role: str

    @property
    def setting(self) -> str:
        if self.arm == "real_only":
            return "real_only"
        return f"{self.arm}_0p30_cseed{self.construction_seed}"

    @property
    def pool_path(self) -> Path | None:
        """Read-only pool consumed by this fit; ``None`` for the reference."""
        if self.arm == "real_only":
            return None
        if self.arm == "placebo":
            return e16gen.placebo_pool_path(self.construction_seed, REPO)
        if self.role == "bridge_sensitivity":
            # The bridge Rule pools are the frozen E15 v2 artifacts; E16 never
            # published an e16_rule_ twin for them (amendment v1 section 2).
            return SYNTHETIC_DIR / (
                f"rule_cseed{self.construction_seed}_windows_v2.npz")
        return e16gen.rule_pool_path(self.construction_seed, REPO)

    @property
    def checkpoint_path(self) -> Path:
        return rfmod.checkpoint_path(self.setting, self.pipeline_seed, MODEL_TAG)

    @property
    def log_path(self) -> Path:
        return rfmod.training_log_path(
            self.setting, self.pipeline_seed, MODEL_TAG)

    @property
    def key(self) -> str:
        return f"{self.construction_seed}:{self.arm}:{self.pipeline_seed}"


def build_training_grid() -> tuple[E17Job, ...]:
    """Deterministic registered job order: references first, then the grid."""
    jobs: list[E17Job] = [
        E17Job(0, "real_only", pipeline_seed, "shared_reference")
        for pipeline_seed in PIPELINE_SEEDS
    ]
    for construction_seed in CONSTRUCTION_SEEDS:
        for arm in ARMS:
            for pipeline_seed in PIPELINE_SEEDS:
                jobs.append(E17Job(
                    construction_seed, arm, pipeline_seed, "primary"))
    for construction_seed in BRIDGE_SEEDS:
        for arm in ARMS:
            for pipeline_seed in PIPELINE_SEEDS:
                jobs.append(E17Job(
                    construction_seed, arm, pipeline_seed,
                    "bridge_sensitivity"))
    if len(jobs) != EXPECTED_FIT_COUNT:
        raise _stop(f"grid size {len(jobs)} != registered {EXPECTED_FIT_COUNT}")
    if len({job.key for job in jobs}) != len(jobs):
        raise _stop("duplicate job key in the training grid")
    return tuple(jobs)


def build_pool_groups() -> tuple[tuple[int, str, str], ...]:
    """One child interpreter per pool: (realization, arm, role)."""
    groups: list[tuple[int, str, str]] = [(0, "real_only", "shared_reference")]
    groups += [(g, a, "primary") for g in CONSTRUCTION_SEEDS for a in ARMS]
    groups += [(g, a, "bridge_sensitivity") for g in BRIDGE_SEEDS for a in ARMS]
    return tuple(groups)


def record_paths() -> dict[str, Path]:
    return {
        "pool_inheritance": EXPERIMENT_DIR
        / f"pool_inheritance_{OUTPUT_VERSION}.json",
        "pool_audit": TABLE_DIR
        / f"e17_pool_inheritance_audit_{OUTPUT_VERSION}.csv",
        "preflight": EXPERIMENT_DIR
        / f"training_preflight_{OUTPUT_VERSION}.json",
        "run": EXPERIMENT_DIR / f"training_run_{OUTPUT_VERSION}.json",
        "manifest": TABLE_DIR / f"e17_training_manifest_{OUTPUT_VERSION}.csv",
        "sampling_audit": TABLE_DIR
        / f"e17_sampling_audit_{OUTPUT_VERSION}.csv",
    }


def training_target_paths(jobs: Sequence[E17Job]) -> list[Path]:
    records = record_paths()
    paths = [job.checkpoint_path for job in jobs]
    paths += [job.log_path for job in jobs]
    paths += [records[key] for key in
              ("preflight", "run", "manifest", "sampling_audit")]
    return paths


def assert_targets_absent(paths: Sequence[Path]) -> dict[str, Any]:
    """Gate G10a — no-clobber preflight, mirroring the frozen RF harness."""
    present = [str(p.relative_to(REPO)) for p in paths
               if p.exists() or p.is_symlink()]
    if present:
        raise _stop(f"refusing to overwrite existing targets: {present[:8]}"
                    + (f" (+{len(present) - 8} more)" if len(present) > 8
                       else ""))
    return {"gate": "G10a_no_clobber", "passed": True, "checked": len(paths)}


# ---------------------------------------------------------------------------
# Gate G14 — Random-Forest contract fidelity
# ---------------------------------------------------------------------------


#: Imported, never re-implemented (PREREG section 5.3 harness note, gate G14).
rf_features = rfmod.rf_features


def _upstream_estimator_keywords() -> dict[str, Any]:
    """Parse the RandomForestClassifier keywords out of the frozen harness.

    ``train_generator_extension_rf.train_one`` builds the estimator inline, so
    there is no object to import.  Parsing its source is what makes G14 a gate
    rather than a restatement: if the frozen harness's configuration is ever
    edited, the three-way agreement between PREREG section 5.1, the harness, and
    this runner breaks and the stage halts.
    """
    tree = ast.parse(inspect.getsource(rfmod.train_one))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RandomForestClassifier"
    ]
    if len(calls) != 1:
        raise _stop(
            f"expected exactly one RandomForestClassifier construction in the "
            f"frozen RF harness, found {len(calls)}")
    keywords: dict[str, Any] = {}
    for keyword in calls[0].keywords:
        if keyword.arg is None:
            raise _stop("frozen RF harness uses **kwargs for the estimator")
        try:
            keywords[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            # ``random_state=seed`` is a name, not a literal.
            keywords[keyword.arg] = (
                keyword.value.id if isinstance(keyword.value, ast.Name)
                else ast.dump(keyword.value))
    return keywords


def verify_rf_contract() -> dict[str, Any]:
    """Gate G14 — features, estimator, and the imported-not-reimplemented rule."""
    if rf_features is not rfmod.rf_features:
        raise _stop(
            "the runner's feature function is not the imported "
            "train_generator_extension_rf.rf_features object")
    probe = np.zeros((3, 128, 11), dtype=np.float32)
    probe[1] = 1.0
    features = rf_features(probe)
    if features.shape != (3, FEATURE_DIMENSION):
        raise _stop(
            f"realized feature dimension {features.shape[1]} != "
            f"{FEATURE_DIMENSION}")
    if features.dtype != np.float32:
        raise _stop(f"feature dtype {features.dtype} is not float32")

    upstream = _upstream_estimator_keywords()
    mismatched = {
        key: (ESTIMATOR[key], upstream.get(key, "<absent>"))
        for key in ESTIMATOR_FIDELITY_KEYS
        if upstream.get(key, "<absent>") != ESTIMATOR[key]
    }
    if mismatched:
        raise _stop(
            f"frozen RF harness estimator differs from PREREG section 5.1: "
            f"{mismatched}")
    if upstream.get("random_state") != "seed":
        raise _stop(
            "frozen RF harness does not seed random_state from the pipeline "
            f"seed: random_state={upstream.get('random_state')!r}")
    return {
        "gate": "G14_rf_contract_fidelity",
        "passed": True,
        "feature_function": f"{rfmod.__name__}.rf_features",
        "feature_function_is_imported_object": True,
        "feature_dimension": FEATURE_DIMENSION,
        "features_computed_on_raw_windows": True,
        "standardizer_constructed": False,
        "estimator": {k: ESTIMATOR[k] for k in ESTIMATOR_FIDELITY_KEYS},
        "estimator_matches_frozen_harness_source": True,
        "upstream_keywords": upstream,
    }


def _forbidden_standardizer(*_args: Any, **_kwargs: Any) -> Any:
    raise _stop(
        "gate G14: a standardizer was constructed during an E17 Random-Forest "
        "fit; the registered arms fit on raw window statistics")


# ---------------------------------------------------------------------------
# Stage S2 — pool inheritance verification
# ---------------------------------------------------------------------------


def verify_seed_axis() -> dict[str, Any]:
    """Gate G2i — the seed lists equal the registered literals, in order."""
    if tuple(CONSTRUCTION_SEEDS) != REGISTERED_CONSTRUCTION_SEEDS:
        raise _stop(
            "generator-realization seed list differs from PREREG section 3.1")
    if tuple(BRIDGE_SEEDS) != REGISTERED_BRIDGE_SEEDS:
        raise _stop("bridge seed list differs from PREREG section 3.2")
    if len(set(CONSTRUCTION_SEEDS)) != 20 or len(set(BRIDGE_SEEDS)) != 4:
        raise _stop("seed lists contain duplicates")
    if set(CONSTRUCTION_SEEDS) & set(BRIDGE_SEEDS):
        raise _stop("primary and bridge seed sets overlap")
    return {
        "gate": "G2i_seed_integrity",
        "passed": True,
        "source_module": "journal/scripts/generate_e16_rule_placebo_pairs.py",
        "generator_realizations": list(CONSTRUCTION_SEEDS),
        "bridge_realizations": list(BRIDGE_SEEDS),
    }


def consumed_pool_slots() -> list[dict[str, Any]]:
    """The 48 read-only pool slots consumed by the registered grid."""
    slots: list[dict[str, Any]] = []
    for construction_seed in CONSTRUCTION_SEEDS:
        for arm in ARMS:
            job = E17Job(construction_seed, arm, PIPELINE_SEEDS[0], "primary")
            slots.append({"construction_seed": construction_seed, "arm": arm,
                          "role": "primary", "path": job.pool_path})
    for construction_seed in BRIDGE_SEEDS:
        for arm in ARMS:
            job = E17Job(construction_seed, arm, PIPELINE_SEEDS[0],
                         "bridge_sensitivity")
            slots.append({"construction_seed": construction_seed, "arm": arm,
                          "role": "bridge_sensitivity", "path": job.pool_path})
    return slots


def verify_pool_inheritance() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate G1i — every consumed pool matches its frozen reference digest.

    Reference table per slot (PREREG section 4.1 as amended 2026-08-07 v1):
    the 44 pools E16 published are compared against ``e16_pool_audit_v2.csv``
    ``x_digest``; the four bridge Rule pools, which E16 never published because
    it fit no bridge Rule arm, are compared against the frozen E15 v2 audit's
    whole-file digest.  Coverage is 48/48 and no comparison is waived.
    """
    for path in (E16_POOL_AUDIT, E15_POOL_AUDIT, E16_MANIPULATION_CHECKS):
        if not path.exists():
            raise _stop(f"missing frozen reference table: {path}")
    e16_audit = pd.read_csv(E16_POOL_AUDIT)
    e15_audit = pd.read_csv(E15_POOL_AUDIT)
    e16_reference = {
        (int(row["construction_seed"]), str(row["arm"])): str(row["x_digest"])
        for row in e16_audit.to_dict(orient="records")
    }
    e15_reference = {
        int(row["construction_seed"]): str(row["pool_file_sha256"])
        for row in e15_audit.to_dict(orient="records")
        if str(row["construction_seed"]) not in ("nan", "")
        and not pd.isna(row["construction_seed"])
    }
    crosslink = _inherit_manipulation_checks()

    rows: list[dict[str, Any]] = []
    mismatched: list[str] = []
    for slot in consumed_pool_slots():
        path = slot["path"]
        construction_seed = slot["construction_seed"]
        arm = slot["arm"]
        if not path.exists():
            raise _stop(f"missing consumed pool: {path.relative_to(REPO)}")
        if path.is_symlink():
            raise _stop(f"consumed pool is a symlink: {path.relative_to(REPO)}")
        file_digest = _sha256_file(path)
        with np.load(path, allow_pickle=False) as pool:
            x_digest = e16gen._array_digest(np.asarray(pool["x"]))
            windows = int(len(pool["x"]))
            label_digest = hashlib.sha256(np.ascontiguousarray(
                np.asarray(pool["y_attack_type"],
                           dtype=np.int64)).tobytes()).hexdigest()

        bridge_rule = (slot["role"] == "bridge_sensitivity" and arm == "rule")
        if bridge_rule:
            reference_table = str(E15_POOL_AUDIT.relative_to(REPO))
            reference_field = "pool_file_sha256"
            expected = e15_reference.get(construction_seed)
            observed = file_digest
            lineage = (
                "e16_manipulation_checks_v2.csv "
                "G7_e15_lineage_reproduction=passed")
        else:
            reference_table = str(E16_POOL_AUDIT.relative_to(REPO))
            reference_field = "x_digest"
            expected = e16_reference.get((construction_seed, arm))
            observed = x_digest
            lineage = ""

        matched = expected is not None and expected == observed
        if not matched:
            mismatched.append(f"{construction_seed}:{arm}")
        rows.append({
            "construction_seed": construction_seed,
            "arm": arm,
            "role": slot["role"],
            "pool_path": str(path.relative_to(REPO)),
            "windows": windows,
            "pool_file_sha256": file_digest,
            "x_digest": x_digest,
            "label_vector_sha256": label_digest,
            "reference_table": reference_table,
            "reference_field": reference_field,
            "reference_value": expected if expected is not None else "",
            "observed_value": observed,
            "matched": matched,
            "lineage_crosslink": lineage,
            "amendment": ("AMENDMENT_2026-08-07_V1_BRIDGE_POOL_AUDIT_SOURCE.md"
                          if bridge_rule else ""),
        })

    if mismatched:
        raise _stop(
            f"G1i pool hash inheritance FAILED for {len(mismatched)} pool(s): "
            f"{mismatched[:6]}; S2 halts for all realizations")

    # Aligned-pair identity is what makes the registered draw identical across
    # arms; it is asserted, not assumed (PREREG section 5.2, gate G10c-i).
    unpaired = []
    by_seed: dict[int, dict[str, str]] = {}
    for row in rows:
        by_seed.setdefault(int(row["construction_seed"]), {})[
            str(row["arm"])] = str(row["label_vector_sha256"])
    for construction_seed, arms in by_seed.items():
        if arms.get("rule") != arms.get("placebo"):
            unpaired.append(construction_seed)
    if unpaired:
        raise _stop(
            f"paired arms have different label vectors, so the registered draw "
            f"cannot be identical: {unpaired}")

    gate = {
        "gate": "G1i_pool_hash_inheritance",
        "passed": True,
        "slots_verified": len(rows),
        "slots_against_e16_pool_audit": sum(
            1 for r in rows if r["reference_field"] == "x_digest"),
        "slots_against_e15_pool_audit": sum(
            1 for r in rows if r["reference_field"] == "pool_file_sha256"),
        "aligned_pair_label_identity": True,
        "reference_tables": {
            str(E16_POOL_AUDIT.relative_to(REPO)): _sha256_file(E16_POOL_AUDIT),
            str(E15_POOL_AUDIT.relative_to(REPO)): _sha256_file(E15_POOL_AUDIT),
        },
        "manipulation_check_inheritance": crosslink,
    }
    return rows, gate


def _inherit_manipulation_checks() -> dict[str, Any]:
    """Gates G3i--G8i — inherited from the frozen E16 record, never re-run."""
    checks = pd.read_csv(E16_MANIPULATION_CHECKS)
    abort = checks[checks["gate"].astype(str) == "abort"]
    failed = abort[abort["passed"] != True]  # noqa: E712 - pandas mask
    if len(failed):
        raise _stop(
            f"frozen E16 manipulation checks record {len(failed)} failed abort "
            f"gate(s): {failed['check'].tolist()[:5]}")
    lineage = checks[checks["check"].astype(str) == "G7_e15_lineage_reproduction"]
    lineage_seeds = sorted(int(s) for s in lineage["construction_seed"])
    if lineage_seeds != sorted(BRIDGE_SEEDS):
        raise _stop(
            f"E16 lineage reproduction covers {lineage_seeds}, expected the "
            f"four bridge realizations {sorted(BRIDGE_SEEDS)}")
    return {
        "gate": "G3i_G8i_manipulation_check_inheritance",
        "passed": True,
        "source": str(E16_MANIPULATION_CHECKS.relative_to(REPO)),
        "source_sha256": _sha256_file(E16_MANIPULATION_CHECKS),
        "abort_gates_confirmed_passed": int(len(abort)),
        "recomputed": False,
        "e15_lineage_reproduction_realizations": lineage_seeds,
    }


def run_pool_verification() -> dict[str, Any]:
    """Stage S2.  No fitting occurs."""
    started = time.monotonic()
    records = record_paths()
    assert_targets_absent([records["pool_inheritance"], records["pool_audit"]])
    record: dict[str, Any] = {
        "schema_version": POOL_INHERITANCE_SCHEMA,
        "stage": "e17_pool_inheritance_verification",
        "started_utc": utc_now(),
        "model_fitting_performed": False,
        "registration": assert_registration_committed(),
        "environment": _environment_record(REPO),
        "seed_axis": verify_seed_axis(),
    }
    rows, gate = verify_pool_inheritance()
    record["pool_hash_inheritance"] = gate
    record["completed_utc"] = utc_now()
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record["status"] = "T-PASS"
    publish_and_cleanup([
        (_stage_bytes(records["pool_audit"], _csv_bytes(rows)),
         records["pool_audit"]),
        (_stage_json(records["pool_inheritance"], record),
         records["pool_inheritance"]),
    ])
    return record


# ---------------------------------------------------------------------------
# Preflight (stages S3/S4)
# ---------------------------------------------------------------------------


def build_preflight() -> dict[str, Any]:
    jobs = build_training_grid()
    records = record_paths()
    if not records["pool_audit"].exists():
        raise _stop(
            "stage S2 has not run: missing "
            f"{records['pool_audit'].relative_to(REPO)}")
    record: dict[str, Any] = {
        "schema_version": TRAINING_PREFLIGHT_SCHEMA,
        "stage": "e17_rf_paired_training",
        "built_utc": utc_now(),
        "expected_fits": EXPECTED_FIT_COUNT,
        "grid": {
            "generator_realizations": len(CONSTRUCTION_SEEDS),
            "bridge_realizations": len(BRIDGE_SEEDS),
            "arms": list(ARMS),
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "primary_fits": EXPECTED_PRIMARY_FITS,
            "bridge_fits": EXPECTED_BRIDGE_FITS,
            "reference_fits": EXPECTED_REFERENCE_FITS,
        },
        "protocol": {
            "detector_family": "random_forest",
            "matching_contract": "identical_training_rows",
            "optimizer_update_budget": None,
            "early_stopping": None,
            "minibatch_order": None,
            "standardizer": None,
            "synthetic_total": SYNTHETIC_TOTAL,
            "sampling_seed_offset": SAMPLING_SEED_OFFSET,
            "per_attack_cap": PER_ATTACK_CAP,
            "estimator": ESTIMATOR,
            "feature_dimension": FEATURE_DIMENSION,
        },
        "registration": assert_registration_committed(),
        "environment": _environment_record(REPO),
        "seed_axis": verify_seed_axis(),
        "rf_contract": verify_rf_contract(),
        "pool_inheritance_record": {
            "path": str(records["pool_inheritance"].relative_to(REPO)),
            "sha256": _sha256_file(records["pool_inheritance"]),
            "audit_path": str(records["pool_audit"].relative_to(REPO)),
            "audit_sha256": _sha256_file(records["pool_audit"]),
        },
        "e16_training_manifest": {
            "path": str(E16_TRAINING_MANIFEST.relative_to(REPO)),
            "sha256": _sha256_file(E16_TRAINING_MANIFEST),
        },
        "target_absence": assert_targets_absent(training_target_paths(jobs)),
    }
    return record


# ---------------------------------------------------------------------------
# Child fits — one pool, five pipeline seeds
# ---------------------------------------------------------------------------


def _real_training_rows() -> tuple[np.ndarray, np.ndarray]:
    real_x, real_y, _val_x, _val_y = _load_real_train_val(REPO)
    features = rf_features(real_x)
    del real_x
    return features, real_y.astype(np.int64)


def _fit_and_publish(
        job: E17Job,
        train_features: np.ndarray,
        train_y: np.ndarray,
        *,
        pool_record: Mapping[str, Any],
        sampling: Mapping[str, Any] | None,
        index_digest: str,
) -> dict[str, Any]:
    """Fit one Random Forest and publish its bundle atomically."""
    from sklearn.ensemble import RandomForestClassifier

    assert_targets_absent([job.checkpoint_path, job.log_path])
    if train_features.shape[1] != FEATURE_DIMENSION:
        raise _stop(
            f"{job.key}: feature dimension {train_features.shape[1]} != "
            f"{FEATURE_DIMENSION}")
    if len(train_features) != len(train_y):
        raise _stop(f"{job.key}: feature/label row mismatch")

    started = utc_now()
    start_time = time.monotonic()
    order = np.random.default_rng(job.pipeline_seed).permutation(len(train_y))
    forest = RandomForestClassifier(
        n_estimators=ESTIMATOR["n_estimators"],
        max_depth=ESTIMATOR["max_depth"],
        min_samples_leaf=ESTIMATOR["min_samples_leaf"],
        class_weight=ESTIMATOR["class_weight"],
        n_jobs=ESTIMATOR["n_jobs"],
        random_state=job.pipeline_seed,
    )
    forest.fit(train_features[order], train_y[order])
    elapsed = round(time.monotonic() - start_time, 3)

    realized = {
        "n_estimators": int(forest.n_estimators),
        "max_depth": forest.max_depth,
        "min_samples_leaf": int(forest.min_samples_leaf),
        "class_weight": forest.class_weight,
        "random_state": int(forest.random_state),
        "n_features_in": int(forest.n_features_in_),
        "classes": [int(c) for c in forest.classes_],
    }
    for key in ESTIMATOR_FIDELITY_KEYS:
        if realized[key] != ESTIMATOR[key]:
            raise _stop(
                f"{job.key}: realized {key}={realized[key]!r} != registered "
                f"{ESTIMATOR[key]!r}")
    if realized["n_features_in"] != FEATURE_DIMENSION:
        raise _stop(f"{job.key}: fitted on {realized['n_features_in']} features")
    if realized["random_state"] != job.pipeline_seed:
        raise _stop(f"{job.key}: random_state is not the pipeline seed")

    log_record = {
        "schema_version": TRAINING_LOG_SCHEMA,
        "experiment": "e17_rf_relational_replication",
        "family": "rf",
        "setting": job.setting,
        "construction_seed": job.construction_seed,
        "arm": job.arm,
        "pipeline_seed": job.pipeline_seed,
        "role": job.role,
        "model_tag": MODEL_TAG,
        "pool": pool_record,
        "sampling": sampling,
        "sampling_index_sha256": index_digest,
        "permutation_seed": job.pipeline_seed,
        "permutation": "numpy.random.default_rng(seed).permutation(len(y))",
        "standardizer": None,
        "features_on_raw_windows": True,
        "feature_dimension": FEATURE_DIMENSION,
        "estimator": realized,
        "train_windows": int(len(train_y)),
        "device": "cpu",
        "started_utc": started,
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "checkpoint": str(job.checkpoint_path.relative_to(REPO)),
    }

    staged_checkpoint = _dump_forest(forest, job.checkpoint_path.parent,
                                     job.checkpoint_path.name)
    staged_log = _stage_json(job.log_path, log_record)
    publish_and_cleanup([
        (staged_checkpoint, job.checkpoint_path),
        (staged_log, job.log_path),
    ])
    log_record["checkpoint_sha256"] = _sha256_file(job.checkpoint_path)
    print(f"  fit {job.key} rows={len(train_y)} {elapsed:.0f}s", flush=True)
    return log_record


def _dump_forest(forest: Any, directory: Path, name: str) -> Path:
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        dir=directory, prefix=f".{name}.e17-stage-", suffix=".tmp")
    os.close(descriptor)
    staged = Path(staged_name)
    with open(staged, "wb") as handle:
        joblib.dump(forest, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return staged


def run_child_group(construction_seed: int, arm: str, role: str
                    ) -> list[dict[str, Any]]:
    """Fit one pool's five pipeline seeds, extracting features once."""
    lc.fit_standardizer = _forbidden_standardizer  # gate G14
    verify_rf_contract()

    real_features, real_y = _real_training_rows()
    jobs = [E17Job(construction_seed, arm, pipeline_seed, role)
            for pipeline_seed in PIPELINE_SEEDS]
    assert_targets_absent(
        [j.checkpoint_path for j in jobs] + [j.log_path for j in jobs])

    if arm == "real_only":
        pool_record = {"path": None, "sha256": None,
                       "policy": "real_only_no_synthetic_draw"}
        return [
            _fit_and_publish(job, real_features, real_y,
                             pool_record=pool_record, sampling=None,
                             index_digest="")
            for job in jobs
        ]

    pool_path = jobs[0].pool_path
    if pool_path is None or not pool_path.exists():
        raise _stop(f"missing pool for {construction_seed}:{arm}")
    pool_digest = _sha256_file(pool_path)
    with np.load(pool_path, allow_pickle=False) as pool:
        pool_construction = int(pool["construction_seed"])
        pool_y = np.asarray(pool["y_attack_type"], dtype=np.int64)
        # Extract once per pool: rf_features is row-wise, so selecting rows of
        # the feature matrix is identical to featurizing the selected rows.
        pool_features = rf_features(np.asarray(pool["x"], dtype=np.float32))
    if pool_construction != construction_seed:
        raise _stop(
            f"pool construction seed {pool_construction} != "
            f"{construction_seed}")
    pool_record = {
        "path": str(pool_path.relative_to(REPO)),
        "sha256": pool_digest,
        "windows": int(len(pool_y)),
        "read_only": True,
    }

    records: list[dict[str, Any]] = []
    for job in jobs:
        indices, sampling = sample_indices_without_replacement(
            pool_y, SYNTHETIC_TOTAL, job.pipeline_seed + SAMPLING_SEED_OFFSET)
        _assert_draw_integrity(job, sampling)
        train_features = np.concatenate(
            (real_features, pool_features[indices]), axis=0)
        train_y = np.concatenate((real_y, pool_y[indices]), axis=0)
        records.append(_fit_and_publish(
            job, train_features, train_y, pool_record=pool_record,
            sampling=sampling,
            index_digest=sampling_indices_sha256(indices)))
        del train_features, train_y
    return records


def _assert_draw_integrity(job: E17Job, sampling: Mapping[str, Any]) -> None:
    """Gate G9i — per fit, per class: requested = drawn = unique, repeated 0."""
    totals = sampling["total"]
    if (totals["repeated"] != 0
            or totals["drawn"] != totals["unique"]
            or totals["requested"] != totals["drawn"]
            or totals["requested"] != SYNTHETIC_TOTAL):
        raise _stop(f"{job.key}: draw is not the registered no-replacement "
                    f"draw: {totals}")
    for label, entry in sampling["per_class"].items():
        if (entry["repeated"] != 0
                or entry["drawn"] != entry["unique"]
                or entry["requested"] != entry["drawn"]):
            raise _stop(f"{job.key}: class {label} draw integrity failed")
        if entry["requested"] >= PER_ATTACK_CAP:
            raise _stop(
                f"{job.key}: class {label} request {entry['requested']} is not "
                f"below the {PER_ATTACK_CAP} per-class pool capacity")


# ---------------------------------------------------------------------------
# Gate G15 — fit determinism
# ---------------------------------------------------------------------------


def _evaluation_probe_features() -> np.ndarray:
    """The frozen E14 held-out normal base panel, as Random-Forest features."""
    import evaluate_l4_counterfactual_factorial as e14

    prepare_path = (REPO / "journal" / "experiments"
                    / "e14_l4_counterfactual_factorial" / "prepare_v1.json")
    _manifest, bases, _latents = e14.load_prepared_inputs(
        json.loads(prepare_path.read_text()))
    return rf_features(np.asarray(bases, dtype=np.float32))


def run_determinism_child(construction_seed: int, arm: str, role: str,
                          pipeline_seed: int, out_path: Path) -> dict[str, Any]:
    """Refit one job in this fresh interpreter and write it outside the tree."""
    from sklearn.ensemble import RandomForestClassifier

    lc.fit_standardizer = _forbidden_standardizer
    job = E17Job(construction_seed, arm, pipeline_seed, role)
    real_features, real_y = _real_training_rows()
    with np.load(job.pool_path, allow_pickle=False) as pool:
        pool_y = np.asarray(pool["y_attack_type"], dtype=np.int64)
        pool_features = rf_features(np.asarray(pool["x"], dtype=np.float32))
    indices, _sampling = sample_indices_without_replacement(
        pool_y, SYNTHETIC_TOTAL, pipeline_seed + SAMPLING_SEED_OFFSET)
    train_features = np.concatenate(
        (real_features, pool_features[indices]), axis=0)
    train_y = np.concatenate((real_y, pool_y[indices]), axis=0)
    order = np.random.default_rng(pipeline_seed).permutation(len(train_y))
    forest = RandomForestClassifier(
        n_estimators=ESTIMATOR["n_estimators"],
        max_depth=ESTIMATOR["max_depth"],
        min_samples_leaf=ESTIMATOR["min_samples_leaf"],
        class_weight=ESTIMATOR["class_weight"],
        n_jobs=ESTIMATOR["n_jobs"],
        random_state=pipeline_seed,
    )
    forest.fit(train_features[order], train_y[order])
    del train_features, train_y, pool_features, real_features

    staged = _dump_forest(forest, out_path.parent, out_path.name)
    Path(staged).replace(out_path)
    probe = _evaluation_probe_features()
    predictions = forest.predict(probe)
    return {
        "key": job.key,
        "repeat_checkpoint": str(out_path),
        "repeat_checkpoint_sha256": _sha256_file(out_path),
        "prediction_rows": int(len(predictions)),
        "prediction_sha256": hashlib.sha256(np.ascontiguousarray(
            predictions.astype("<i8")).tobytes()).hexdigest(),
    }


def verify_fit_determinism() -> dict[str, Any]:
    """Gate G15 — one Rule and one placebo fit repeat byte-identically."""
    import joblib

    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="e17-determinism-") as scratch:
        for arm in ARMS:
            job = E17Job(CONSTRUCTION_SEEDS[0], arm, PIPELINE_SEEDS[0],
                         "primary")
            if not job.checkpoint_path.exists():
                raise _incomplete(
                    f"G15 cannot run: missing published checkpoint for "
                    f"{job.key}")
            out_path = Path(scratch) / f"repeat_{arm}.joblib"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--determinism-child",
                "--construction-seed", str(job.construction_seed),
                "--arm", job.arm,
                "--pipeline-seed", str(job.pipeline_seed),
                "--role", job.role,
                "--out", str(out_path),
            ]
            completed = subprocess.run(command, check=False,
                                       capture_output=True, text=True)
            if completed.returncode != 0:
                raise _stop(
                    f"G15 determinism child for {job.key} exited "
                    f"{completed.returncode}: {completed.stderr[-800:]}")
            child = json.loads(completed.stdout.strip().splitlines()[-1])
            published_digest = _sha256_file(job.checkpoint_path)
            byte_identical = (
                child["repeat_checkpoint_sha256"] == published_digest)
            forest = joblib.load(job.checkpoint_path)
            probe = _evaluation_probe_features()
            predictions = forest.predict(probe)
            prediction_digest = hashlib.sha256(np.ascontiguousarray(
                predictions.astype("<i8")).tobytes()).hexdigest()
            predictions_identical = (
                prediction_digest == child["prediction_sha256"])
            del forest
            if not byte_identical or not predictions_identical:
                raise _stop(
                    f"G15 fit determinism FAILED for {job.key}: "
                    f"byte_identical={byte_identical} "
                    f"predictions_identical={predictions_identical}")
            checks.append({
                "key": job.key,
                "arm": arm,
                "published_checkpoint_sha256": published_digest,
                "repeated_checkpoint_sha256": child[
                    "repeat_checkpoint_sha256"],
                "joblib_payload_byte_identical": True,
                "evaluation_prediction_rows": child["prediction_rows"],
                "evaluation_predictions_identical": True,
                "prediction_sha256": prediction_digest,
            })
    return {
        "gate": "G15_fit_determinism",
        "passed": True,
        "repeated_in_fresh_child_interpreter": True,
        "checks": checks,
        "note": "replaces the CNN deterministic-runtime-flag record, which has "
                "no Random-Forest analogue",
    }


# ---------------------------------------------------------------------------
# Post-fit gates
# ---------------------------------------------------------------------------


def verify_internal_pairing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate G10c-i — paired arms drew identical synthetic rows.

    120 pairs: 24 realizations x 5 pipeline seeds.  The gate can fail: the two
    arms draw independently from their own pool's label vector, and the draws
    coincide only because the aligned twins share a label vector.  Any pool
    substitution, re-ordering, or arm mix-up breaks it.
    """
    by_pair: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if str(row["arm"]) == "real_only":
            continue
        key = (int(row["construction_seed"]), int(row["pipeline_seed"]))
        by_pair.setdefault(key, {})[str(row["arm"])] = str(
            row["sampling_index_sha256"])
    mismatched = [f"{c}:{p}" for (c, p), arms in by_pair.items()
                  if arms.get("rule") != arms.get("placebo")]
    incomplete = [f"{c}:{p}" for (c, p), arms in by_pair.items()
                  if len(arms) != 2]
    if incomplete:
        raise _incomplete(f"incomplete arm pairs: {incomplete[:5]}")
    if mismatched:
        raise _stop(f"paired arms drew different synthetic rows: "
                    f"{mismatched[:5]}")
    expected = (len(CONSTRUCTION_SEEDS) + len(BRIDGE_SEEDS)) * len(PIPELINE_SEEDS)
    if len(by_pair) != expected:
        raise _incomplete(
            f"{len(by_pair)} arm pairs checked, registered {expected}")
    return {"gate": "G10c-i_paired_draw_identity", "passed": True,
            "pairs_checked": len(by_pair)}


def verify_cross_family_inheritance(rows: Sequence[Mapping[str, Any]]
                                    ) -> dict[str, Any]:
    """Gate G10c-ii — E17 draws equal the frozen E16 CNN draws.

    220 fits are checked.  The 20 bridge Rule fits have no E16 CNN counterpart
    (E16 fit only the placebo arm for the bridge realizations) and are recorded
    as exempt with their reason, never silently waived.
    """
    e16 = pd.read_csv(E16_TRAINING_MANIFEST)
    frozen = {
        (int(r["construction_seed"]), str(r["arm"]), int(r["pipeline_seed"])):
            str(r["sampling_index_sha256"])
        for r in e16.to_dict(orient="records")
    }
    checked = 0
    mismatched: list[str] = []
    exempt: list[str] = []
    for row in rows:
        arm = str(row["arm"])
        if arm == "real_only":
            continue
        key = (int(row["construction_seed"]), arm, int(row["pipeline_seed"]))
        if key not in frozen:
            if row["role"] == "bridge_sensitivity" and arm == "rule":
                exempt.append(f"{key[0]}:{arm}:{key[2]}")
                continue
            raise _stop(
                f"no frozen E16 CNN draw for {key}, and it is not an eligible "
                "bridge Rule exemption")
        checked += 1
        if frozen[key] != str(row["sampling_index_sha256"]):
            mismatched.append(f"{key[0]}:{arm}:{key[2]}")
    if mismatched:
        raise _stop(
            f"identical-training-row contract VIOLATED for {len(mismatched)} "
            f"fit(s): {mismatched[:5]}")
    expected_checked = EXPECTED_PRIMARY_FITS + (
        len(BRIDGE_SEEDS) * len(PIPELINE_SEEDS))
    if checked != expected_checked or len(exempt) != len(BRIDGE_SEEDS) * len(
            PIPELINE_SEEDS):
        raise _incomplete(
            f"G10c-ii checked {checked} fits with {len(exempt)} exemptions; "
            f"registered {expected_checked} and "
            f"{len(BRIDGE_SEEDS) * len(PIPELINE_SEEDS)}")
    return {
        "gate": "G10c-ii_cross_family_draw_inheritance",
        "passed": True,
        "fits_checked": checked,
        "reference": str(E16_TRAINING_MANIFEST.relative_to(REPO)),
        "reference_sha256": _sha256_file(E16_TRAINING_MANIFEST),
        "exempt_fits": exempt,
        "exemption_reason": (
            "E16 fit only the placebo arm for the bridge realizations, so the "
            "20 bridge Rule fits have no E16 CNN counterpart (PREREG section "
            "5.2); they are covered by G10c-i alone"),
    }


def verify_completeness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate G10d — every registered fit is present with its provenance."""
    if len(rows) != EXPECTED_FIT_COUNT:
        raise _incomplete(
            f"{len(rows)} fits present, registered {EXPECTED_FIT_COUNT}")
    by_role: dict[str, int] = {}
    for row in rows:
        by_role[str(row["role"])] = by_role.get(str(row["role"]), 0) + 1
    expected_roles = {
        "primary": EXPECTED_PRIMARY_FITS,
        "bridge_sensitivity": EXPECTED_BRIDGE_FITS,
        "shared_reference": EXPECTED_REFERENCE_FITS,
    }
    if by_role != expected_roles:
        raise _incomplete(f"fit roles {by_role} != registered {expected_roles}")
    required = ("checkpoint", "checkpoint_sha256", "pool_sha256",
                "sampling_index_sha256", "feature_dimension",
                "train_windows", "elapsed_seconds", "device",
                "permutation_seed", "n_estimators", "min_samples_leaf",
                "class_weight")
    for row in rows:
        missing = [field for field in required if row.get(field) in (None, "")]
        if str(row["arm"]) == "real_only":
            missing = [f for f in missing
                       if f not in ("pool_sha256", "sampling_index_sha256")]
        if missing:
            raise _incomplete(
                f"fit {row['construction_seed']}:{row['arm']}:"
                f"{row['pipeline_seed']} lacks {missing}")
    return {
        "gate": "G10d_fit_completeness",
        "passed": True,
        "fits": len(rows),
        "by_role": by_role,
        "devices": sorted({str(r["device"]) for r in rows}),
    }


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------


def _child_command(construction_seed: int, arm: str, role: str) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--construction-seed", str(construction_seed),
        "--arm", arm,
        "--role", role,
    ]


def _manifest_row(record: Mapping[str, Any], job: E17Job) -> dict[str, Any]:
    estimator = record["estimator"]
    pool = record["pool"]
    return {
        "construction_seed": job.construction_seed,
        "arm": job.arm,
        "pipeline_seed": job.pipeline_seed,
        "role": job.role,
        "setting": record["setting"],
        "model_tag": MODEL_TAG,
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "pool_path": pool.get("path") or "",
        "pool_sha256": pool.get("sha256") or "",
        "sampling_index_sha256": record["sampling_index_sha256"],
        "feature_dimension": record["feature_dimension"],
        "n_estimators": estimator["n_estimators"],
        "max_depth": "" if estimator["max_depth"] is None
                     else estimator["max_depth"],
        "min_samples_leaf": estimator["min_samples_leaf"],
        "class_weight": estimator["class_weight"],
        "permutation_seed": record["permutation_seed"],
        "train_windows": record["train_windows"],
        "elapsed_seconds": record["elapsed_seconds"],
        "device": record["device"],
    }


def execute(*, limit: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    preflight = build_preflight()
    records = record_paths()
    publish_and_cleanup([(_stage_json(records["preflight"], preflight),
                          records["preflight"])])

    groups = build_pool_groups()
    if limit is not None:
        groups = groups[:limit]

    manifest_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    for position, (construction_seed, arm, role) in enumerate(groups, start=1):
        print(f"[{position}/{len(groups)}] {construction_seed}:{arm} ({role})",
              flush=True)
        completed = subprocess.run(
            _child_command(construction_seed, arm, role), check=False)
        if completed.returncode != 0:
            raise _stop(
                f"child group {construction_seed}:{arm} exited "
                f"{completed.returncode}; stage halted with no outcome "
                "interpretation")
        for pipeline_seed in PIPELINE_SEEDS:
            job = E17Job(construction_seed, arm, pipeline_seed, role)
            if not job.checkpoint_path.exists() or not job.log_path.exists():
                raise _incomplete(f"child fit {job.key} published no bundle")
            record = json.loads(job.log_path.read_text())
            record.setdefault("checkpoint_sha256",
                              _sha256_file(job.checkpoint_path))
            manifest_rows.append(_manifest_row(record, job))
            sampling = record.get("sampling")
            if sampling is None:
                continue
            totals = sampling["total"]
            sampling_rows.append({
                "construction_seed": job.construction_seed,
                "arm": job.arm,
                "pipeline_seed": job.pipeline_seed,
                "role": job.role,
                "requested": totals["requested"],
                "drawn": totals["drawn"],
                "unique": totals["unique"],
                "repeated": totals["repeated"],
                "per_class_max_requested": max(
                    int(e["requested"]) for e in sampling["per_class"].values()),
                "per_attack_cap": PER_ATTACK_CAP,
                "index_sha256": record["sampling_index_sha256"],
            })

    internal_pairing = verify_internal_pairing(manifest_rows)
    cross_family = verify_cross_family_inheritance(manifest_rows)
    completeness = verify_completeness(manifest_rows)
    determinism = verify_fit_determinism()

    run_record = {
        "schema_version": TRAINING_RUN_SCHEMA,
        "stage": "e17_rf_paired_training",
        "started_utc": preflight["built_utc"],
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "fit_count": len(manifest_rows),
        "expected_fit_count": EXPECTED_FIT_COUNT,
        "registration": preflight["registration"],
        "rf_contract_gate": preflight["rf_contract"],
        "draw_integrity_gate": {
            "gate": "G9i_draw_integrity",
            "passed": True,
            "fits_checked": len(sampling_rows),
            "requested_equals_drawn_equals_unique": True,
            "repeated": 0,
            "per_attack_cap": PER_ATTACK_CAP,
        },
        "internal_pairing_gate": internal_pairing,
        "cross_family_inheritance_gate": cross_family,
        "completeness_gate": completeness,
        "determinism_gate": determinism,
        "matching_contract": (
            "identical training rows; the CNN's 6,156-update matched budget has "
            "no Random-Forest analogue and is not simulated"),
        "preflight_path": str(records["preflight"].relative_to(REPO)),
        "status": "T-PASS",
        "technical_status": "T-PASS",
    }
    publish_and_cleanup([
        (_stage_bytes(records["manifest"], _csv_bytes(manifest_rows)),
         records["manifest"]),
        (_stage_bytes(records["sampling_audit"], _csv_bytes(sampling_rows)),
         records["sampling_audit"]),
        (_stage_json(records["run"], run_record), records["run"]),
    ])
    return run_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-pools", action="store_true",
                        help="stage S2: hash the consumed pools; no fitting")
    parser.add_argument("--execute", action="store_true",
                        help="stages S3/S4: run the canonical 245-fit grid")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--determinism-child", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--construction-seed", type=int)
    parser.add_argument("--arm", choices=(*ARMS, "real_only"))
    parser.add_argument("--pipeline-seed", type=int)
    parser.add_argument("--role", default="primary")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N pool groups (smoke only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.determinism_child:
        print(json.dumps(run_determinism_child(
            args.construction_seed, args.arm, args.role, args.pipeline_seed,
            args.out)))
        return 0
    if args.child:
        records = run_child_group(args.construction_seed, args.arm, args.role)
        print(json.dumps({"fits": len(records)}))
        return 0
    if args.verify_pools:
        print(json.dumps(run_pool_verification(), indent=2, sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps(build_preflight(), indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(limit=args.limit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
