"""Small construction/role checks, without running the new experimental grid."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_attribution_local_replication_v1 as study
import run_attribution_local_grid_v1 as launcher


def test_seed_grid_is_fresh_fixed_and_disjoint():
    seeds = study.construction_seeds(study.config())
    assert seeds["pilot"] == [210961]
    assert seeds["confirmation"][:3] == [510571, 363939, 235483]
    all_seeds = seeds["pilot"] + seeds["confirmation"]
    assert len(all_seeds) == len(set(all_seeds)) == 101
    assert not set(all_seeds).intersection(set(study.gen.CONSTRUCTION_SEEDS) | set(study.gen.BRIDGE_SEEDS) | {314159})


def test_launcher_grid_has_exact_fixed_coverage():
    jobs = launcher.grid(study.config())
    assert len(jobs) == len(set(jobs)) == 600
    for index in range(100):
        selected = [job for job in jobs if job[1] == index]
        assert selected[0] == ("prepare", index, None)
        assert selected[1:] == [("fit-pair", index, pipeline) for pipeline in [7, 42, 123, 2026, 3407]]


def test_evaluation_selection_stays_historical():
    roles = study.audit_roles()
    assert roles["bases"] == 6000
    assert roles["sealed_window_overlap"] == 0
    assert not roles["new_holdout"]


def test_safe_serialization_and_canonical_wrapper(tmp_path):
    # This seed/normal substrate is only a unit test, not a pilot/main pool.
    rng = np.random.default_rng(991)
    x = rng.integers(0, 256, (16, 128, 11)).astype(np.float32)
    x[:, :, 0], x[:, :, 1], x[:, :, 10] = 100, 8, 0.001
    pair = study.gen.generate_pair(x, np.arange(16), per_attack=32, construction_seed=991)
    canonical = study.gen.generate_pool(x, np.arange(16), per_attack=32, construction_seed=991)
    checks = study.gen.evaluate_pair_gates(pair, study.gen.arrays_digest(canonical["arrays"]))
    assert not study.gen.failed_gates(checks)
    for arm in study.ARMS:
        arrays = study.gen.pool_arrays(pair, arm)
        arrays = {key: (value.astype(str) if value.dtype.kind == "O" else value)
                  for key, value in arrays.items()}
        path = tmp_path / f"{arm}.npz"
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as saved:
            assert all(saved[key].dtype.kind != "O" for key in saved.files)
            assert np.array_equal(saved["x"], pair.rule["x"] if arm == "rule" else pair.placebo_x)


def test_json_refuses_overwrite(tmp_path):
    path = tmp_path / "record.json"
    study.write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        study.write_json(path, {"original": False})


def test_synthetic_draw_is_unique_and_paired():
    y = np.repeat(np.arange(1, 5), 65000)
    first, receipt = study.sample_indices_without_replacement(y, 78645, 307)
    second, _ = study.sample_indices_without_replacement(y, 78645, 307)
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 78645
    assert receipt["total"]["repeated"] == 0
