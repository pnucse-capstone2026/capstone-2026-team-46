#!/usr/bin/env python3
"""Frozen-law, synthetic-only positive control; never invokes E16/E18 runners."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import torch

import lib_common as lc
from train_generator_extension_cnn import CNN1D
from run_e18_relation_visible_rf_training import e18_features


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "journal/experiments/attribution_strengthening_20260906/a_positive_control"
CONFIG_PATH = ROOT / "config_v1.json"
PROTOCOL_PATH = ROOT / "PROTOCOL_v1.md"
FREEZE_PATH = ROOT / "freeze_v1.json"
FAMILIES = ("cnn", "rf83")
ARMS = ("rule", "placebo")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype}|{value.shape}|".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def config():
    return json.loads(CONFIG_PATH.read_text())


def input_records():
    paths = [CONFIG_PATH, PROTOCOL_PATH, Path(__file__).resolve()]
    paths += [REPO / "journal/scripts" / name for name in (
        "lib_common.py", "train_generator_extension_cnn.py",
        "train_generator_extension_rf.py", "run_e18_relation_visible_rf_training.py",
    )]
    return [{"path": str(path.relative_to(REPO)), "sha256": digest_file(path)}
            for path in paths]


def freeze():
    if any((ROOT / name).exists() for name in ("pilot_v1", "confirmation_v1")):
        raise RuntimeError("A run directory already exists; cannot retroactively freeze")
    record = {"schema": "vehcom.attribution.positive_control.freeze.v1",
              "created_utc": utc_now(), "public_preregistration": False,
              "outcomes_observed": False, "inputs": input_records(), "config": config()}
    write_json(FREEZE_PATH, record)
    print(json.dumps({"freeze": str(FREEZE_PATH.relative_to(REPO)),
                      "sha256": digest_file(FREEZE_PATH)}), flush=True)


def verify_freeze():
    record = json.loads(FREEZE_PATH.read_text())
    if record["inputs"] != input_records() or record["config"] != config():
        raise RuntimeError("Frozen protocol/configuration/implementation changed")
    return record


def seed_for(cfg, role, realization, split):
    key = f"{cfg['seed_namespace']}|{role}|{realization}|{split}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def make_split(cfg, role, realization, split, bases=None):
    n = cfg["bases"][split] if bases is None else bases
    rng = np.random.default_rng(seed_for(cfg, role, realization, split))
    frames = cfg["frames"]
    levels = np.array(cfg["levels"], dtype=np.float32)
    if frames % len(levels) or not np.array_equal(levels, 255 - levels[::-1]):
        raise ValueError("The protocol needs complement-symmetric, balanced levels")
    base = rng.integers(0, 256, size=(n, frames, 11)).astype(np.float32)
    base[:, :, 0] = rng.integers(1, 2048, size=(n, 1))
    base[:, :, 1] = 8
    base[:, :, 10] = rng.uniform(0.0005, 0.0015, size=(n, frames))
    balanced = np.repeat(levels, frames // len(levels))
    values = np.stack([rng.permutation(balanced) for _ in range(n)])
    independent = np.stack([rng.permutation(balanced) for _ in range(n)])
    flips = rng.random(n) < cfg["noise_probability"]
    y = np.tile(np.array([0, 1], dtype=np.int64), n)
    relation = y ^ np.repeat(flips.astype(np.int64), 2)
    rule = np.repeat(base, 2, axis=0)
    rule[:, :, 2] = np.repeat(values, 2, axis=0)
    rule[:, :, 3] = np.where(relation[:, None] == 1,
                             255 - rule[:, :, 2], rule[:, :, 2])
    placebo = rule.copy()
    placebo[:, :, 3] = np.repeat(independent, 2, axis=0)
    oracle = np.all(rule[:, :, 2] + rule[:, :, 3] == 255, axis=1).astype(np.int64)
    return {"rule": rule, "placebo": placebo, "y": y,
            "oracle_accuracy": float(np.mean(oracle == y)),
            "seed": seed_for(cfg, role, realization, split)}


def validate_split(data):
    rule, placebo, y = data["rule"], data["placebo"], data["y"]
    other = [i for i in range(11) if i != 3]
    assert np.array_equal(rule[:, :, other], placebo[:, :, other])
    assert np.array_equal(np.sort(rule[:, :, 3], axis=1),
                          np.sort(placebo[:, :, 3], axis=1))
    assert np.array_equal(placebo[::2], placebo[1::2])
    assert np.array_equal(y[::2], np.zeros(len(y) // 2, dtype=np.int64))
    assert np.array_equal(y[1::2], np.ones(len(y) // 2, dtype=np.int64))
    for channel in range(11):
        assert np.array_equal(np.sort(rule[::2, :, channel], axis=1),
                              np.sort(rule[1::2, :, channel], axis=1))
    assert np.isfinite(rule).all() and np.isfinite(placebo).all()
    return {"windows": len(y), "rng_seed": data["seed"],
            "rule_sha256": digest_array(rule), "placebo_sha256": digest_array(placebo),
            "label_sha256": digest_array(y), "oracle_accuracy": data["oracle_accuracy"],
            "non_target_matching": True, "target_multiset_matching": True,
            "class_pair_marginal_matching": True, "opposite_label_control_identity": True}


def cnn_predict(model, x, device, batch):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            values = torch.from_numpy(x[start:start + batch]).to(device)
            predictions.append(model(values).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def state_digest(state):
    return hashlib.sha256("|".join(
        f"{name}:{digest_array(tensor.detach().cpu().numpy())}"
        for name, tensor in state.items()).encode()).hexdigest()


def fit_cnn(cfg, x, y, val_x, val_y, pipeline, device):
    lc.set_seed(pipeline)
    model = CNN1D(in_channels=11, classes=2).to(device)
    initial_hash = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    train_x = torch.from_numpy(x).to(device)
    train_y = torch.from_numpy(y).to(device)
    order_rng = np.random.default_rng(pipeline)
    index_digest = hashlib.sha256()
    history, best_state = [], None
    best_score, selected_step = -1.0, 0
    order, cursor = np.empty(0, dtype=np.int64), 0
    for step in range(1, cfg["cnn_steps"] + 1):
        if cursor >= len(order):
            order, cursor = order_rng.permutation(len(y)), 0
        indices = order[cursor:cursor + cfg["batch_size"]]
        cursor += len(indices)
        index_digest.update(indices.tobytes())
        batch_indices = torch.from_numpy(indices).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(train_x[batch_indices]),
                                                train_y[batch_indices])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step % cfg["validation_cadence"] == 0:
            prediction = cnn_predict(model, val_x, device, cfg["batch_size"])
            score = float(f1_score(val_y, prediction, average="macro", zero_division=0))
            history.append({"step": step, "validation_macro_f1": score,
                            "train_loss": float(loss.detach().cpu())})
            if score > best_score:
                best_score, selected_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"initial_state_sha256": initial_hash,
                   "batch_indices_sha256": index_digest.hexdigest(),
                   "optimizer_steps": cfg["cnn_steps"], "selected_step": selected_step,
                   "history": history}


def mean_interval(values, confidence):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sd / np.sqrt(len(values)))
    return mean, sd, mean - half, mean + half


def summarize(rows, cfg, stage):
    frame = pd.DataFrame(rows)
    units = []
    for (family, realization), group in frame.groupby(["family", "realization"]):
        arms = group.groupby("arm")[["recall_intact", "recall_broken"]].mean()
        u = float(arms.loc["rule", "recall_intact"] - arms.loc["placebo", "recall_intact"])
        reliance = float((arms.loc["rule", "recall_intact"] - arms.loc["rule", "recall_broken"])
                         - (arms.loc["placebo", "recall_intact"] - arms.loc["placebo", "recall_broken"]))
        assert abs(u - reliance) < 1e-12
        units.append({"family": family, "realization": int(realization), "U": u, "L": reliance,
                      "rule_intact": float(arms.loc["rule", "recall_intact"]),
                      "placebo_intact": float(arms.loc["placebo", "recall_intact"]),
                      "rule_broken": float(arms.loc["rule", "recall_broken"]),
                      "placebo_broken": float(arms.loc["placebo", "recall_broken"]),
                      "oracle_accuracy": float(group.oracle_accuracy.mean())})
    unit_frame = pd.DataFrame(units)
    required_u = ((1 - cfg["noise_probability"]) - 0.5) * cfg["minimum_oracle_advantage_recovery"]
    summaries = []
    for family, group in unit_frame.groupby("family"):
        mean, sd, low, high = mean_interval(group.U, cfg["interval_confidence"])
        summaries.append({"family": family, "n_realizations": len(group), "U_mean": mean,
                          "U_sd": sd, "U_ci95_low": low, "U_ci95_high": high,
                          "U_positive": int((group.U > 0).sum()), "L_equals_U_by_design": True,
                          "rule_intact_mean": float(group.rule_intact.mean()),
                          "placebo_intact_mean": float(group.placebo_intact.mean()),
                          "oracle_accuracy_mean": float(group.oracle_accuracy.mean()),
                          "minimum_mean_U": required_u,
                          "recovery_criterion_met": bool(low > 0 and mean >= required_u)})
    return frame, unit_frame, {"stage": stage, "pilot_excluded_from_confirmation": True,
                              "families": summaries,
                              "joint_recovery": all(s["recovery_criterion_met"] for s in summaries),
                              "interpretation": "synthetic diagnostic only; not physical CAN utility"}


def run(stage):
    frozen = verify_freeze()
    cfg = frozen["config"]
    if stage == "confirmation":
        pilot = json.loads((ROOT / "pilot_v1/run.json").read_text())
        if pilot["status"] != "complete" or not pilot["technical_checks_passed"]:
            raise RuntimeError("Technical pilot did not complete")
    out = ROOT / f"{stage}_v1"
    out.mkdir(exist_ok=False)
    (out / "models").mkdir()
    (out / "fits").mkdir()
    (out / "data_checks").mkdir()
    started = time.monotonic()
    write_json(out / "started.json", {"started_utc": utc_now(), "pid": os.getpid(),
               "stage": stage, "freeze_sha256": digest_file(FREEZE_PATH)})
    rows = []
    try:
        torch.set_num_threads(cfg["torch_threads"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CUDA unavailable; do not silently change the training environment")
        n = cfg[f"{stage}_realizations"]
        pipelines = cfg["pipelines"][:1] if stage == "pilot" else cfg["pipelines"]
        for realization in range(n):
            data = {split: make_split(cfg, stage, realization, split) for split in cfg["bases"]}
            checks = {split: validate_split(value) for split, value in data.items()}
            assert len({value["seed"] for value in data.values()}) == 3
            raw_train = data["train"]["rule"]
            mean = raw_train.mean(axis=(0, 1), dtype=np.float64)
            std = raw_train.std(axis=(0, 1), dtype=np.float64)
            std = np.where(std < 1e-8, 1.0, std)
            assert np.allclose(mean, data["train"]["placebo"].mean(axis=(0, 1), dtype=np.float64), atol=1e-12)
            control_std = data["train"]["placebo"].std(axis=(0, 1), dtype=np.float64)
            control_std = np.where(control_std < 1e-8, 1.0, control_std)
            assert np.allclose(std, control_std, rtol=0, atol=1e-10)
            arrays, feats = {}, {}
            for split, value in data.items():
                arrays[split], feats[split] = {}, {}
                for arm in ARMS:
                    arrays[split][arm] = np.ascontiguousarray(((value[arm] - mean) / std).transpose(0, 2, 1), dtype=np.float32)
                    feats[split][arm] = e18_features(value[arm])
                    assert np.isfinite(feats[split][arm]).all()
            visible = float(np.mean(np.any(feats["train"]["rule"] != feats["train"]["placebo"], axis=1)))
            assert visible > 0.99
            checks["feature_visibility"] = visible
            checks["standardizer"] = {"mean": mean.tolist(), "std": std.tolist()}
            write_json(out / "data_checks" / f"g{realization:02d}.json", checks)
            for family in FAMILIES:
                for pipeline in pipelines:
                    paired_history = []
                    for arm in ARMS:
                        fit_start = time.monotonic()
                        key = f"g{realization:02d}_{family}_{arm}_p{pipeline}"
                        if family == "cnn":
                            model, history = fit_cnn(cfg, arrays["train"][arm], data["train"]["y"],
                                                    arrays["val"]["rule"], data["val"]["y"], pipeline, device)
                            predictions = {condition: cnn_predict(model, arrays["test"][test_arm], device, cfg["batch_size"])
                                           for condition, test_arm in (("intact", "rule"), ("broken", "placebo"))}
                            checkpoint = out / "models" / (key + ".pt")
                            with checkpoint.open("xb") as handle:
                                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                            "standardizer_mean": mean, "standardizer_std": std}, handle)
                            del model
                        else:
                            order = np.random.default_rng(pipeline).permutation(len(data["train"]["y"]))
                            model = RandomForestClassifier(n_estimators=cfg["rf_trees"], max_depth=None,
                                      min_samples_leaf=2, class_weight="balanced_subsample",
                                      n_jobs=cfg["rf_workers"], random_state=pipeline)
                            model.fit(feats["train"][arm][order], data["train"]["y"][order])
                            predictions = {condition: model.predict(feats["test"][test_arm])
                                           for condition, test_arm in (("intact", "rule"), ("broken", "placebo"))}
                            history = {"batch_indices_sha256": digest_array(order)}
                            checkpoint = out / "models" / (key + ".pkl")
                            with checkpoint.open("xb") as handle:
                                pickle.dump(model, handle)
                        assert np.array_equal(predictions["broken"][::2], predictions["broken"][1::2])
                        recall = {condition: float(np.mean(pred == data["test"]["y"]))
                                  for condition, pred in predictions.items()}
                        assert recall["broken"] == 0.5
                        row = {"stage": stage, "realization": realization, "family": family,
                               "arm": arm, "pipeline": pipeline, "recall_intact": recall["intact"],
                               "recall_broken": recall["broken"], "oracle_accuracy": data["test"]["oracle_accuracy"],
                               "elapsed_seconds": time.monotonic() - fit_start}
                        with (out / "fits" / (key + ".npz")).open("xb") as handle:
                            np.savez_compressed(handle, y=data["test"]["y"], **predictions)
                        write_json(out / "fits" / (key + ".json"), {**row, "training": history,
                                  "checkpoint": str(checkpoint.relative_to(REPO)),
                                  "checkpoint_sha256": digest_file(checkpoint)})
                        rows.append(row)
                        paired_history.append(history)
                        print(json.dumps({"fit": key, **row}), flush=True)
                    assert paired_history[0]["batch_indices_sha256"] == paired_history[1]["batch_indices_sha256"]
                    if family == "cnn":
                        assert paired_history[0]["initial_state_sha256"] == paired_history[1]["initial_state_sha256"]
        assert len(rows) == n * len(pipelines) * len(FAMILIES) * len(ARMS)
        verify_freeze()
        fits, units, summary = summarize(rows, cfg, stage)
        fits.to_csv(out / "fits.csv", index=False, mode="x")
        units.to_csv(out / "realization_summary.csv", index=False, mode="x")
        write_json(out / "summary.json", summary)
        result = {"status": "complete", "technical_checks_passed": True, "fits": len(rows),
                  "stage": stage, "completed_utc": utc_now(), "elapsed_seconds": time.monotonic() - started,
                  "freeze_sha256": digest_file(FREEZE_PATH), "joint_recovery": summary["joint_recovery"],
                  "environment": {"python": sys.version, "numpy": np.__version__,
                                  "torch": torch.__version__, "sklearn": sklearn.__version__,
                                  "scipy": scipy.__version__, "device": str(device),
                                  "gpu": torch.cuda.get_device_name(0)}}
        write_json(out / "run.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(out / "failure.json", {"status": "failed", "error": repr(exc),
                   "completed_fits": len(rows), "utc": utc_now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pilot", "confirmation"))
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else run(args.stage)


if __name__ == "__main__":
    main()
