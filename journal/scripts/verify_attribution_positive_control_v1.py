#!/usr/bin/env python3
"""Audit saved positive-control predictions and replay a fixed checkpoint subset."""
import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import t

import run_attribution_positive_control_v1 as experiment


def verify(stage):
    cfg = experiment.config()
    experiment.verify_freeze()
    root = experiment.ROOT / f"{stage}_v1"
    target = root / "verification_v1.json"
    if target.exists():
        raise FileExistsError(target)
    count = cfg[f"{stage}_realizations"]
    pipelines = cfg["pipelines"] if stage == "confirmation" else cfg["pipelines"][:1]
    expected = {(g, family, arm, p) for g in range(count)
                for family in ("cnn", "rf83") for arm in ("rule", "placebo") for p in pipelines}
    files = sorted((root / "fits").glob("*.json"))
    records = [json.loads(path.read_text()) for path in files]
    assert len(records) == len(expected)
    assert {(r["realization"], r["family"], r["arm"], r["pipeline"]) for r in records} == expected
    torch.set_num_threads(cfg["torch_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replayed, rows, metric_checks = [], [], 0
    for g in range(count):
        data = experiment.make_split(cfg, stage, g, "test")
        for path, record in zip(files, records):
            if record["realization"] != g:
                continue
            checkpoint = experiment.REPO / record["checkpoint"]
            assert experiment.digest_file(checkpoint) == record["checkpoint_sha256"]
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
                assert np.array_equal(saved["y"], data["y"])
                values = {key: saved[key].copy() for key in ("intact", "broken")}
            row = {key: record[key] for key in ("realization", "family", "arm", "pipeline")}
            for condition, prediction in values.items():
                # Balanced class counts make accuracy equal to macro recall here.
                recalls = [float(np.mean(prediction[data["y"] == label] == label)) for label in (0, 1)]
                metric = float(np.mean(recalls))
                assert metric == record[f"recall_{condition}"]
                row[condition] = metric
                metric_checks += 1
            assert row["broken"] == 0.5
            assert np.array_equal(values["broken"][::2], values["broken"][1::2])
            rows.append(row)
            # Fixed coverage: first/last realization and first/last pipeline;
            # both model families and arms. Not selected from performance.
            if g not in (0, count - 1) or record["pipeline"] not in (pipelines[0], pipelines[-1]):
                continue
            if record["family"] == "cnn":
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = experiment.CNN1D(classes=2).to(device)
                model.load_state_dict(state["state_dict"])
                for condition, arm in (("intact", "rule"), ("broken", "placebo")):
                    x = np.ascontiguousarray(((data[arm] - state["standardizer_mean"])
                        / state["standardizer_std"]).transpose(0, 2, 1), dtype=np.float32)
                    actual = experiment.cnn_predict(model, x, device, cfg["batch_size"])
                    assert np.array_equal(actual, values[condition]), (path, condition)
                del model
            else:
                with checkpoint.open("rb") as handle:
                    model = pickle.load(handle)  # Trusted checkpoints created by this experiment.
                for condition, arm in (("intact", "rule"), ("broken", "placebo")):
                    actual = model.predict(experiment.e18_features(data[arm]))
                    assert np.array_equal(actual, values[condition]), (path, condition)
            replayed.append(path.stem)
    frame = pd.DataFrame(rows)
    summaries = json.loads((root / "summary.json").read_text())["families"]
    intervals = []
    for family in ("cnn", "rf83"):
        selected = frame[frame.family == family]
        by_unit = selected.groupby(["realization", "arm"])[["intact", "broken"]].mean().unstack("arm")
        u = by_unit["intact"]["rule"] - by_unit["intact"]["placebo"]
        l = u - (by_unit["broken"]["rule"] - by_unit["broken"]["placebo"])
        assert np.array_equal(u, l)
        half = t.ppf(0.975, len(u) - 1) * u.std(ddof=1) / np.sqrt(len(u))
        interval = [float(u.mean() - half), float(u.mean() + half)]
        stored = next(s for s in summaries if s["family"] == family)
        np.testing.assert_allclose([u.mean(), *interval],
            [stored["U_mean"], stored["U_ci95_low"], stored["U_ci95_high"]], atol=1e-14, rtol=0)
        intervals.append({"family": family, "U_mean": float(u.mean()), "U_ci95": interval})
    result = {"status": "pass", "stage": stage, "verified_utc": experiment.utc_now(),
              "freeze_unchanged": True, "checkpoint_hashes_checked": len(records),
              "prediction_metric_checks": metric_checks, "replayed_fits": replayed,
              "replay_device": str(device), "summary_recomputed": intervals,
              "verifier_sha256": experiment.digest_file(__file__),
              "scope": "all saved metrics/hashes; fixed subset serialized-model replay; not independent retraining"}
    experiment.write_json(target, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "confirmation"))
    verify(parser.parse_args().stage)
