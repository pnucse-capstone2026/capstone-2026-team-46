#!/usr/bin/env python3
"""Reload the new B pilot pools and checkpoints without training or test scoring."""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import run_attribution_local_replication_v1 as study


def main():
    study.verify_freeze()
    root = study.ROOT / "pilot_v1/g000"
    target = root / "serialized_verification_v1.json"
    if target.exists():
        raise FileExistsError(target)
    training = json.loads((root / "pipeline_7/run.json").read_text())
    assert training["status"] == "complete"
    prepared = json.loads((root / "prepare.json").read_text())
    pools = {}
    for record in prepared["pools"]:
        path = study.REPO / record["path"]
        assert study.sha(path) == record["sha256"]
        with np.load(path, allow_pickle=False) as data:
            pools[record["arm"]] = {key: data[key] for key in data.files}
    rule, control = pools["rule"], pools["placebo"]
    assert rule.keys() == control.keys()
    for key in rule:
        if key not in ("x", "arm"):
            assert np.array_equal(rule[key], control[key]), key
    x, z = rule["x"], control["x"]
    assert x.shape == z.shape == (260000, 128, 11)
    assert np.isfinite(x).all() and np.isfinite(z).all()
    assert np.array_equal(np.bincount(rule["y_attack_type"]), [0, 65000, 65000, 65000, 65000])
    for column in range(11):
        if column != 3:
            assert np.array_equal(x[:, :, column], z[:, :, column])
    assert np.array_equal(np.sort(x[:, :, 3], axis=1), np.sort(z[:, :, 3], axis=1))
    spoof = np.isin(rule["y_attack_type"], [3, 4])
    assert np.array_equal(x[~spoof], z[~spoof])
    residuals = {}
    for label, name in ((3, "gear"), (4, "rpm")):
        frames = residual = minimum = 0
        for index in np.flatnonzero(rule["y_attack_type"] == label):
            positions = rule["positions"][index]
            positions = positions[positions >= 0]
            mask = np.ones(128, dtype=bool)
            mask[positions] = False
            assert np.array_equal(x[index, mask, 3], z[index, mask, 3])
            d0, intact, broken = x[index, positions, 2], x[index, positions, 3], z[index, positions, 3]
            if label == 3:
                assert np.all(np.abs(255 - d0 - intact) <= 8)
                residual += int(np.count_nonzero(np.abs(255 - d0 - broken) <= 8))
            else:
                assert np.all((intact.astype(int) - 8 * d0.astype(int)) % 256 < 8)
                observed = int(np.count_nonzero((broken.astype(int) - 8 * d0.astype(int)) % 256 < 8))
                optimum = study.gen.rpm_structural_minimum(d0)
                assert observed == optimum
                residual += observed
                minimum += optimum
            frames += len(positions)
        assert residual / frames < study.gen.GEAR_RESIDUAL_MAX_FRACTION
        residuals[name] = {"frames": frames, "residual": residual, "fraction": residual / frames,
                           "structural_minimum": minimum if label == 4 else None}
    del pools, rule, control, x, z
    with np.load(study.REPO / "journal/datasets/windows/val_windows.npz", allow_pickle=False) as data:
        val_x, val_y = data["x"], data["y_attack_type"].astype(np.int64)
    torch.set_num_threads(study.config()["torch_threads"])
    device = torch.device("cuda")
    replay = []
    for arm in study.ARMS:
        record = json.loads((root / f"pipeline_7/{arm}.json").read_text())
        checkpoint = study.REPO / record["checkpoint"]
        assert study.sha(checkpoint) == record["checkpoint_sha256"]
        mean, std = np.asarray(record["standardizer_mean"], dtype=np.float32), np.asarray(record["standardizer_std"], dtype=np.float32)
        loader = DataLoader(study.lc.WindowDataset(study.lc.standardize(val_x, mean, std), val_y),
                            batch_size=study.config()["validation_batch_size"], shuffle=False, num_workers=0)
        model = study.CNN1D(classes=5).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        actual_y, probabilities = study.predict(model, loader, device)
        score = float(f1_score(actual_y, probabilities.argmax(axis=1), average="macro", zero_division=0))
        assert score == record["best_validation_macro_f1"]
        replay.append({"arm": arm, "validation_macro_f1": score, "stored_score_exact_match": True})
    result = {"status": "pass", "utc": study.now(), "pool_hashes_checked": 2,
        "numeric_metadata_matching": True, "non_target_matching": True,
        "multiset_matching": True, "dos_fuzzy_identity": True, "residuals": residuals,
        "checkpoint_validation_replay": replay, "test_scoring_performed": False,
        "verifier_sha256": study.sha(__file__)}
    study.write_json(target, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
