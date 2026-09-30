import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_attribution_local_grid_v1 as resume


def pair():
    common = {"stage": "confirmation", "realization": 2, "construction_seed": 235483,
        "pipeline": 123, "steps": 4,
        "history": [{"step": 2, "validation_macro_f1": .7}, {"step": 4, "validation_macro_f1": .7}],
        "selected_step": 2, "best_validation_macro_f1": .7}
    common.update({k: "paired" for k in resume.PAIRED})
    return [{**copy.deepcopy(common), "arm": arm} for arm in ["rule", "placebo"]]


def test_fixed_remainder_has_no_completed_fit():
    jobs = resume.original.grid(resume.study.config())
    pending = resume.remaining_jobs(jobs, [resume.name(j) for j in jobs[:15]])
    assert pending[0] == ("fit-pair", 2, 123)
    assert len(pending) == 585
    assert sum(j[0] == "fit-pair" for j in pending) * 2 == 976
    assert not set(pending).intersection(jobs[:15])


@pytest.mark.parametrize("indices", [[0, 2], [0, 0], [1, 0]])
def test_nonprefix_missing_duplicate_or_reordered_completion_fails(indices):
    jobs = resume.original.grid(resume.study.config())
    with pytest.raises(ValueError, match="contiguous prefix"):
        resume.remaining_jobs(jobs, [resume.name(jobs[i]) for i in indices])


def test_valid_pair_keeps_earliest_tie():
    resume.validate_fit_records(pair(), {"steps": 4, "validation_cadence": 2}, 2, 235483, 123)


@pytest.mark.parametrize("field,value", [("pipeline", 42), ("steps", 3),
    ("selected_step", 4), ("sampling_index_sha256", "different")])
def test_identity_budget_selector_and_pairing_mismatches_fail(field, value):
    logs = pair()
    logs[1][field] = value
    with pytest.raises(ValueError):
        resume.validate_fit_records(logs, {"steps": 4, "validation_cadence": 2}, 2, 235483, 123)


def make_interrupted(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    monkeypatch.setattr(resume.study, "FREEZE", freeze)
    source = tmp_path / "original"
    source.mkdir()
    (source / "started.json").write_text(json.dumps({"stage": "confirmation", "construction_seed": 235483,
        "pipeline": 123, "freeze_sha256": resume.study.sha(freeze)}))
    return source, tmp_path / "archive/attempt"


def test_start_only_attempt_moves_without_changing_bytes(tmp_path, monkeypatch):
    source, target = make_interrupted(tmp_path, monkeypatch)
    before = (source / "started.json").read_bytes()
    resume.archive_start_only(source, target, 235483, 123)
    assert not source.exists()
    assert (target / "started.json").read_bytes() == before


def test_partial_checkpoint_blocks_archival(tmp_path, monkeypatch):
    source, target = make_interrupted(tmp_path, monkeypatch)
    (source / "rule.pt").write_bytes(b"keep this partial model")
    with pytest.raises(ValueError, match="start-only"):
        resume.archive_start_only(source, target, 235483, 123)
    assert (source / "rule.pt").read_bytes() == b"keep this partial model"
    assert not target.exists()


def test_existing_archive_is_not_overwritten(tmp_path, monkeypatch):
    source, target = make_interrupted(tmp_path, monkeypatch)
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        resume.archive_start_only(source, target, 235483, 123)
    assert (source / "started.json").exists()
