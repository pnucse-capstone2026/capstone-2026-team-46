"""Outcome-free recovery-gate tests and exact scientific-function continuity."""
import ast
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_attribution_local_replication_v1 as baseline
import evaluate_attribution_local_replication_v2 as evaluation


@pytest.mark.parametrize("name", ["configure_cpu", "cell_key", "cells", "identities",
    "verify_prepared", "predictions", "counts", "validate_counts", "interval", "analyze", "summarize"])
def test_scientific_functions_are_exactly_unchanged(name):
    def function(module):
        tree = ast.parse(Path(module.__file__).read_text())
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(function(evaluation)) == ast.dump(function(baseline))
    assert evaluation.PREPARED == baseline.PREPARED
    assert evaluation.ROOT != baseline.ROOT


@pytest.fixture
def completed():
    return {"status": "complete", "fits": 1000, "jobs": 600, "inherited_fits": 24,
            "new_fits": 976, "inherited_jobs": 15, "new_jobs": 585,
            "test_scoring_performed": False, "overall_research_complete": False}


def test_complete_recovery_accounting(completed):
    evaluation.require_complete_recovery(completed)


@pytest.mark.parametrize("key,value", [("fits", 998), ("jobs", 599), ("status", "started"),
    ("inherited_fits", 26), ("new_fits", 974), ("inherited_jobs", 16), ("new_jobs", 584),
    ("test_scoring_performed", True), ("overall_research_complete", True)])
def test_partial_or_wrong_recovery_rejected(completed, key, value):
    completed[key] = value
    with pytest.raises((ValueError, RuntimeError)):
        evaluation.require_complete_recovery(completed)


def test_completion_failure_precedes_any_checkpoint_loading(monkeypatch):
    monkeypatch.setattr(evaluation, "verify_freeze", lambda: {})
    def fail():
        raise ValueError("Incomplete continuation")
    monkeypatch.setattr(evaluation, "verify_continuation", fail)
    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint loading before complete recovery gate")
    monkeypatch.setattr(evaluation.torch, "load", forbidden)
    with pytest.raises(ValueError, match="Incomplete continuation"):
        evaluation.preflight()


def test_refreeze_is_refused_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "verify_freeze", lambda: pytest.fail("Refreeze performed work"))
    with pytest.raises(FileExistsError):
        evaluation.freeze()


@pytest.fixture
def lineage(tmp_path, monkeypatch, completed):
    recovery, study = evaluation.recovery, evaluation.study
    out, old, audit_path, frozen = (tmp_path / p for p in ("resume", "original", "audit.json", "freeze.json"))
    monkeypatch.setattr(recovery, "OUT", out)
    monkeypatch.setattr(recovery, "OLD", old)
    monkeypatch.setattr(recovery, "AUDIT", audit_path)
    monkeypatch.setattr(study, "FREEZE", frozen)
    study.write_json(frozen, {"seeds": {"confirmation": list(range(100))}})
    jobs = recovery.original.grid(study.config())
    manifest = [{"job": recovery.name(j), "files": [{"sha256": str(i)}]} for i, j in enumerate(jobs)]
    audit = {"status": "pass", "code_inputs": [], "scientific_freeze_sha256": study.sha(frozen),
             "verified_completed": manifest[:15], "remaining_jobs": [list(j) for j in jobs[15:]],
             "inherited_jobs": 15, "inherited_fits": 24, "remaining_fits": 976}
    study.write_json(audit_path, audit)
    study.write_json(out / "full_grid_manifest.json", manifest)
    completed.update(recovery_audit_sha256=study.sha(audit_path),
                     full_grid_manifest_sha256=study.sha(out / "full_grid_manifest.json"))
    study.write_json(out / "run.json", completed)
    study.write_json(out / "started.json", {"recovery_audit_sha256": study.sha(audit_path),
        "scientific_freeze_sha256": study.sha(frozen), "inherited_fits": 24, "planned_new_fits": 976,
        "retry_enabled": False, "test_scoring_enabled": False})
    for root, selected in ((old, manifest[:15]), (out, manifest[15:])):
        root.mkdir(exist_ok=True)
        with (root / "progress.jsonl").open("x") as handle:
            for r in selected:
                handle.write(evaluation.json.dumps({"event": "complete", "job": r["job"],
                    "receipt_sha256": r["files"][0]["sha256"]}) + "\n")
    monkeypatch.setattr(recovery, "code_inputs", lambda: [])
    preserved, checked = [], []
    monkeypatch.setattr(recovery, "verify_preserved", lambda record, archived: preserved.append(archived))
    def verify_job(job, cfg, seeds):
        checked.append(job)
        return manifest[jobs.index(job)]
    monkeypatch.setattr(recovery, "verify_job", verify_job)
    return out, completed, preserved, checked


def test_complete_lineage_checks_all_jobs(lineage):
    _, completed, preserved, checked = lineage
    assert evaluation.verify_continuation() == completed
    assert preserved == [True]
    assert len(checked) == 600


@pytest.mark.parametrize("corruption", ["manifest_hash", "audit_hash", "failure", "original_complete", "artifact"])
def test_lineage_corruption_is_rejected(lineage, monkeypatch, corruption):
    out, completed, _, _ = lineage
    original_read = evaluation.read
    if corruption in ("manifest_hash", "audit_hash"):
        changed = copy.deepcopy(completed)
        changed["full_grid_manifest_sha256" if corruption == "manifest_hash" else "recovery_audit_sha256"] = "bad"
        monkeypatch.setattr(evaluation, "read", lambda p: changed if p == out / "run.json" else original_read(p))
    elif corruption == "failure":
        evaluation.study.write_json(out / "failure.json", {})
    elif corruption == "original_complete":
        evaluation.study.write_json(evaluation.recovery.OLD / "run.json", {})
    else:
        monkeypatch.setattr(evaluation.recovery, "verify_job", lambda *args: {})
    with pytest.raises(ValueError):
        evaluation.verify_continuation()
