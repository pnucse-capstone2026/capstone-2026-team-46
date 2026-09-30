"""Small synthetic checks; never generates the pilot/confirmation streams."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_attribution_positive_control_v1 as audit


@pytest.mark.parametrize("realization", range(4))
def test_matching_and_exact_negative_control(realization):
    cfg = audit.config()
    data = audit.make_split(cfg, "unit-test", realization, "train", bases=16)
    record = audit.validate_split(data)
    assert record["opposite_label_control_identity"]
    assert np.array_equal(data["rule"], audit.make_split(
        cfg, "unit-test", realization, "train", bases=16)["rule"])
    prediction = (data["placebo"][:, 0, 3] > 128).astype(int)
    assert np.mean(prediction == data["y"]) == 0.5
    features = audit.e18_features(data["rule"])
    assert features.shape == (32, 83)
    assert np.all(np.any(features != audit.e18_features(data["placebo"]), axis=1))


def test_random_stream_separation():
    cfg = audit.config()
    keys = [audit.seed_for(cfg, role, realization, split)
            for role in ("unit-test", "pilot", "confirmation")
            for realization in range(20) for split in ("train", "val", "test")]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("rule_recall,expected", [(0.85, True), (0.6, False)])
def test_unit_aggregation_and_joint_criterion(rule_recall, expected):
    cfg = audit.config()
    rows = [{"family": family, "realization": g, "pipeline": p, "arm": arm,
             "recall_intact": rule_recall if arm == "rule" else 0.5,
             "recall_broken": 0.5, "oracle_accuracy": 0.85}
            for family in audit.FAMILIES for g in range(20)
            for p in cfg["pipelines"] for arm in audit.ARMS]
    _, units, result = audit.summarize(rows, cfg, "unit-test")
    assert len(units) == 40
    assert result["joint_recovery"] is expected
    assert np.allclose(units.U, units.L)
    assert all(item["n_realizations"] == 20 for item in result["families"])


def test_no_clobber_json(tmp_path):
    target = tmp_path / "result.json"
    audit.write_json(target, {"original": True})
    with pytest.raises(FileExistsError):
        audit.write_json(target, {"original": False})
    assert target.read_text().find('"original": true') >= 0


def test_control_identity_gate_rejects_change():
    data = audit.make_split(audit.config(), "unit-test", 91, "test", bases=8)
    data["placebo"][0, 0, 0] += 1
    with pytest.raises(AssertionError):
        audit.validate_split(data)
