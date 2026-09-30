"""Outcome-free tests for B transforms, coverage, aggregation and execution gates."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_attribution_local_replication_v1 as evaluation


def mock_counts():
    cfg = evaluation.study.config()
    seeds = [101, 103, 107, 109, 113]
    rows = []
    for identity in evaluation.identities(cfg, seeds):
        for cell in evaluation.cells():
            g, arm = identity["realization"], identity["arm"]
            if cell["kind"] == "normal":
                binary = 10 + g if arm == "rule" else 5
                exact = 2000 - binary
            else:
                exact = {("rule", "intact"): 1400 + 10 * g, ("rule", "broken"): 800,
                         ("placebo", "intact"): 1500, ("placebo", "broken"): 1300}[arm, cell["relation"]]
                binary = exact + 100
            rows.append({**identity, **{k: cell[k] for k in ("kind", *evaluation.CELL, "n")},
                         "exact_correct": exact, "binary_attack": binary})
    return pd.DataFrame(rows), cfg, seeds


@pytest.mark.parametrize("label", [3, 4])
@pytest.mark.parametrize("k", [2, 4, 8, 16, 32])
def test_strict_matching_predicates_and_k32_continuity(label, k):
    rng = np.random.default_rng(7701)
    base = rng.integers(0, 256, (128, 11)).astype(np.float32)
    base[:, 0], base[:, 1], base[:, 10] = 100, 8, 0.001
    latent = evaluation.old.e14.build_latent(base, block_id="unit_test", block_number=1,
        block_position=0, test_window_index=999999, attack_label=label)
    intact, _ = evaluation.old.apply_missing_cell_transform(base, latent, attack_label=label, k=k, relation="intact")
    broken, stats = evaluation.old.apply_missing_cell_transform(base, latent, attack_label=label, k=k, relation="broken")
    result = evaluation.check_pair(base, latent, label, k, intact, broken, stats)
    assert 0 <= result["residual"] <= k
    corrupted = broken.copy()
    corrupted[0, 10] += 1
    with pytest.raises(ValueError, match="Non-target"):
        evaluation.check_pair(base, latent, label, k, intact, corrupted, stats)


def test_opposite_utility_and_reliance_and_construction_unit_ci():
    frame, cfg, seeds = mock_counts()
    result = evaluation.analyze(frame, cfg, seeds)
    primary = result["summary"].query("analysis_role == 'co_primary'").set_index("endpoint")
    assert set(primary.index) == {"U", "L"}
    assert primary.loc["U", "mean"] == pytest.approx(-0.04)
    assert primary.loc["L", "mean"] == pytest.approx(0.21)
    assert primary.n_constructions.eq(5).all()  # Not 25 pipelines or thousands of windows.
    assert primary.confidence.eq(0.975).all()
    values = np.array([-0.05 + 0.005 * g for g in range(5)])
    half = t.ppf(0.9875, 4) * values.std(ddof=1) / np.sqrt(5)
    assert primary.loc["U", "ci_high"] == pytest.approx(values.mean() + half)
    assert primary.loc["U", "ci_high"] < 0 < primary.loc["L", "ci_low"]
    secondary = result["summary"].query("analysis_role != 'co_primary'")
    assert secondary.confidence.eq(0.95).all()
    assert set(result["construction_units"].k) == {2, 4, 8, 16, 32}
    assert set(result["pipeline_sensitivity"].pipeline) == set(cfg["pipelines"])


def test_normal_false_positive_denominator_and_paired_difference():
    frame, cfg, seeds = mock_counts()
    summary = evaluation.analyze(frame, cfg, seeds)["fpr_summary"].set_index("endpoint")
    assert summary.loc["rule", "mean"] == pytest.approx(12 / 2000)
    assert summary.loc["placebo", "mean"] == pytest.approx(5 / 2000)
    assert summary.loc["delta", "mean"] == pytest.approx(7 / 2000)
    assert summary.n_constructions.eq(5).all()


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "denominator", "fractional", "wrong_seed", "normal_counts", "binary_count"])
def test_grid_and_count_corruption_is_rejected(corruption):
    frame, cfg, seeds = mock_counts()
    if corruption == "missing":
        frame = frame.iloc[:-1]
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif corruption == "denominator":
        frame.loc[0, "n"] = 1999
    elif corruption == "fractional":
        frame["exact_correct"] = frame.exact_correct.astype(float)
        frame.loc[0, "exact_correct"] = 1990.5
    elif corruption == "wrong_seed":
        frame.loc[0, "construction_seed"] = 999
    elif corruption == "normal_counts":
        frame.loc[0, "exact_correct"] = 1991
    else:
        frame.loc[frame.kind == "attack", "binary_attack"] = 0
    with pytest.raises(ValueError):
        evaluation.validate_counts(frame, cfg, seeds)


@pytest.mark.parametrize("record", [{}, {"status": "started", "fits": 1000, "jobs": 600},
    {"status": "complete", "fits": 998, "jobs": 599}])
def test_partial_training_cannot_open_evaluation(record):
    with pytest.raises(RuntimeError):
        evaluation.require_complete_launch(record)


def test_full_training_receipt_is_required_not_a_successful_partial_sample():
    evaluation.require_complete_launch({"status": "complete", "fits": 1000, "jobs": 600})


def test_cache_round_trip_without_real_data(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "PREPARED", tmp_path)
    monkeypatch.setattr(evaluation.study, "REPO", tmp_path)
    x = np.arange(2 * 128 * 11, dtype=np.float32).reshape(2, 128, 11)
    meta = {"key": "synthetic", "n": 2}
    record = evaluation.save_cache(meta, x, np.zeros((1, 1, 11), dtype=np.float32), np.ones((1, 1, 11), dtype=np.float32))
    assert np.array_equal(np.load(tmp_path / record["path"]), x)
    with pytest.raises(FileExistsError):
        evaluation.save_cache(meta, x, 0, 1)


def test_count_predictions_for_attack_and_normal():
    pred = np.array([0, 1, 3, 4], dtype=np.uint8)
    assert evaluation.counts(pred, {"attack_label": 3, "n": 4}) == {"exact_correct": 1, "binary_attack": 3}
    assert evaluation.counts(pred, {"attack_label": 0, "n": 4}) == {"exact_correct": 1, "binary_attack": 3}
    with pytest.raises(ValueError):
        evaluation.counts(np.array([5]), {"attack_label": 3, "n": 1})
