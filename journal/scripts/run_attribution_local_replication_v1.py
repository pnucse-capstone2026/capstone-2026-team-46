#!/usr/bin/env python3
"""No-overwrite preparation and training for the new conditional replication."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import generate_e16_rule_placebo_pairs as gen
import lib_common as lc
from train_generator_extension_cnn import CNN1D, predict
from train_rule_construction_seed_crossing import (
    _load_real_train_val, sample_indices_without_replacement, sampling_indices_sha256)

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "journal/experiments/attribution_strengthening_20260906/b_local_replication"
CONFIG = ROOT / "config_v1.json"
PROTOCOL = ROOT / "PROTOCOL_v1.md"
FREEZE = ROOT / "freeze_v1.json"
BASE = REPO / "journal/results/tables/e14_l4_factorial_base_manifest_v1.csv"
ROLES = REPO / "journal/results/tables/cfot_crfe_evaluation_split_v1.csv"
ARMS = ("rule", "placebo")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def config():
    return json.loads(CONFIG.read_text())


def construction_seeds(cfg):
    seen = set(gen.CONSTRUCTION_SEEDS) | set(gen.BRIDGE_SEEDS) | {314159}
    result = []
    for index in range(cfg["confirmation_realizations"] + 1):
        attempt = 0
        while True:
            key = f"{cfg['seed_namespace']}|{index}|{attempt}"
            candidate = 100000 + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % 900000
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
                break
            attempt += 1
    return {"pilot": result[:1], "confirmation": result[1:]}


def audit_roles():
    bases, roles = pd.read_csv(BASE), pd.read_csv(ROLES)
    assert len(bases) == 6000 and bases.test_window_index.is_unique
    assert roles.test_window_index.is_unique
    selected = bases.merge(roles[["test_window_index", "study_role"]],
                           on="test_window_index", validate="one_to_one", how="left")
    assert selected.study_role.eq(config()["evaluation_role"]).all()
    assert bases.y_binary.eq(0).all() and bases.y_attack_type.eq(0).all()
    sealed = set(roles.loc[roles.study_role == "source_locked_new_method_test", "test_window_index"])
    assert len(sealed) == 24000
    assert not sealed.intersection(bases.test_window_index)
    return {"bases": len(bases), "selected_roles": selected.study_role.value_counts().to_dict(),
            "sealed_window_overlap": 0, "new_holdout": False, "test_scoring_performed": False}


def records():
    scripts = ("run_attribution_local_replication_v1.py", "lib_common.py",
        "generate_e16_rule_placebo_pairs.py", "generate_rule_based_synthetic_v2.py",
        "train_generator_extension_cnn.py", "train_rule_construction_seed_crossing.py",
        "evaluate_l4_counterfactual_factorial.py", "evaluate_e20_missing_cell_probe.py")
    paths = [CONFIG, PROTOCOL, BASE, ROLES,
        REPO / "journal/results/tables/e14_l4_factorial_latent_manifest_v1.csv",
        REPO / "journal/experiments/e14_l4_counterfactual_factorial/prepare_v1.json",
        REPO / "journal/datasets/windows/train_windows.npz",
        REPO / "journal/datasets/windows/val_windows.npz",
        REPO / "journal/datasets/windows/test_windows.npz"]
    paths += [REPO / "journal/scripts" / name for name in scripts]
    return [{"path": str(path.relative_to(REPO)), "sha256": sha(path)} for path in paths]


def freeze():
    if any((ROOT / f"{stage}_v1").exists() for stage in ("pilot", "confirmation")):
        raise RuntimeError("Cannot freeze after new data generation")
    cfg = config()
    result = {"schema": "vehcom.attribution.local_replication.freeze.v1", "utc": now(),
              "config": cfg, "seeds": construction_seeds(cfg), "roles": audit_roles(),
              "inputs": records(), "public_preregistration": False}
    write_json(FREEZE, result)
    print(json.dumps({"freeze": str(FREEZE.relative_to(REPO)), "sha256": sha(FREEZE),
                      "pilot_seed": result["seeds"]["pilot"], "main_realizations": 100}), flush=True)


def verify_freeze():
    frozen = json.loads(FREEZE.read_text())
    assert frozen["config"] == config()
    assert frozen["seeds"] == construction_seeds(config())
    assert frozen["inputs"] == records(), "Frozen implementation or input changed"
    assert frozen["roles"] == audit_roles()
    return frozen


def unit(stage, index):
    frozen = verify_freeze()
    seeds = frozen["seeds"][stage]
    if not 0 <= index < len(seeds):
        raise ValueError("Realization index outside fixed grid")
    if stage == "confirmation":
        pilot = json.loads((ROOT / "pilot_v1/g000/pilot_technical_receipt.json").read_text())
        assert pilot["status"] == "pass" and pilot["excluded_from_confirmation"]
    return seeds[index], ROOT / f"{stage}_v1" / f"g{index:03d}"


def prepare(stage, index):
    seed, out = unit(stage, index)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(out / "prepare_started.json", {"utc": now(), "stage": stage,
        "construction_seed": seed, "freeze_sha256": sha(FREEZE), "inference_performed": False})
    try:
        cfg = config()
        train_x, normal_indices = gen._load_train_source()
        pair = gen.generate_pair(train_x, normal_indices, per_attack=cfg["per_attack"], construction_seed=seed)
        canonical = gen.generate_pool(train_x, normal_indices, per_attack=cfg["per_attack"], construction_seed=seed)
        checks = gen.evaluate_pair_gates(pair, gen.arrays_digest(canonical["arrays"]))
        write_json(out / "construction_checks.json", {"checks": checks, "stats": pair.stats})
        failures = gen.failed_gates(checks)
        if failures:
            raise RuntimeError(f"Construction gates failed: {failures}")
        del canonical, train_x
        pools = []
        for arm in ARMS:
            arrays = gen.pool_arrays(pair, arm)
            arrays = {key: (value.astype(str) if value.dtype.kind == "O" else value)
                      for key, value in arrays.items()}
            arrays["pool_schema_version"] = np.asarray("attribution.local_replication.pool.v1")
            path = out / f"{arm}_pool.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            pools.append({"arm": arm, "path": str(path.relative_to(REPO)),
                          "sha256": sha(path), "bytes": path.stat().st_size})
        result = {"status": "complete", "utc": now(), "construction_seed": seed,
                  "abort_gates_passed": True, "pools": pools, "elapsed_seconds": time.monotonic() - start}
        write_json(out / "prepare.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(out / "prepare_failure.json", {"utc": now(), "error": repr(exc)})
        raise


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_pair(stage, index, pipeline):
    seed, out = unit(stage, index)
    cfg = config()
    allowed = cfg["pipelines"] if stage == "confirmation" else cfg["pipelines"][:1]
    if pipeline not in allowed:
        raise ValueError("Pipeline outside fixed stage grid")
    fit_dir = out / f"pipeline_{pipeline}"
    fit_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(fit_dir / "started.json", {"utc": now(), "construction_seed": seed,
        "pipeline": pipeline, "stage": stage, "freeze_sha256": sha(FREEZE)})
    try:
        prepared = json.loads((out / "prepare.json").read_text())
        assert prepared["status"] == "complete" and prepared["abort_gates_passed"]
        assert torch.cuda.is_available(), "Pilot budget assumes an available CUDA device"
        torch.set_num_threads(cfg["torch_threads"])
        device = torch.device("cuda")
        real_x, real_y, val_x, val_y = _load_real_train_val(REPO)
        mean, std = lc.fit_standardizer(real_x)
        val_standardized = lc.standardize(val_x, mean, std)
        del val_x
        rows = []
        for arm in ARMS:
            arm_start = time.monotonic()
            pool_record = next(r for r in prepared["pools"] if r["arm"] == arm)
            pool_path = REPO / pool_record["path"]
            assert sha(pool_path) == pool_record["sha256"]
            with np.load(pool_path, allow_pickle=False) as pool:
                assert int(pool["construction_seed"]) == seed
                indices, sampling = sample_indices_without_replacement(pool["y_attack_type"],
                    cfg["synthetic_total"], pipeline + cfg["sampling_seed_offset"])
                selected_x = pool["x"][indices]
                selected_y = pool["y_attack_type"][indices].astype(np.int64)
            lc.set_seed(pipeline)
            train_x = lc.standardize(np.concatenate((real_x, selected_x)), mean, std)
            train_y = np.concatenate((real_y, selected_y))
            del selected_x
            train_loader = DataLoader(lc.WindowDataset(train_x, train_y), batch_size=cfg["batch_size"],
                shuffle=True, num_workers=cfg["loader_workers"], pin_memory=True)
            val_loader = DataLoader(lc.WindowDataset(val_standardized, val_y),
                batch_size=cfg["validation_batch_size"], shuffle=False,
                num_workers=cfg["loader_workers"], pin_memory=True)
            model = CNN1D(in_channels=11, classes=5).to(device)
            initial = state_digest(model)
            weights = compute_class_weight(class_weight="balanced", classes=np.arange(5), y=train_y)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            best, best_state, selected_step, history = -float("inf"), None, None, []
            iterator = iter(train_loader)
            for step in range(1, cfg["steps"] + 1):
                try:
                    x, y = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    x, y = next(iterator)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
                loss.backward()
                optimizer.step()
                if step % cfg["validation_cadence"] == 0:
                    true, probs = predict(model, val_loader, device)
                    score = float(f1_score(true, probs.argmax(axis=1), average="macro", zero_division=0))
                    assert np.isfinite(score) and np.isfinite(float(loss.detach().cpu()))
                    history.append({"step": step, "validation_macro_f1": score})
                    print(json.dumps({"stage": stage, "g": index, "pipeline": pipeline,
                        "arm": arm, "step": step, "val_macro_f1": score}), flush=True)
                    if score > best:
                        best, selected_step = score, step
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            assert best_state is not None and len(history) == cfg["steps"] // cfg["validation_cadence"]
            checkpoint = fit_dir / f"{arm}.pt"
            with checkpoint.open("xb") as handle:
                torch.save(best_state, handle)
            record = {"stage": stage, "construction_seed": seed, "realization": index,
                "pipeline": pipeline, "arm": arm, "steps": cfg["steps"], "history": history,
                "selected_step": selected_step, "best_validation_macro_f1": best,
                "initial_state_sha256": initial, "sampling": sampling,
                "sampling_index_sha256": sampling_indices_sha256(indices),
                "final_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                "final_cuda_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                "standardizer_mean": mean.tolist(), "standardizer_std": std.tolist(),
                "checkpoint": str(checkpoint.relative_to(REPO)), "checkpoint_sha256": sha(checkpoint),
                "elapsed_seconds": time.monotonic() - arm_start, "utc": now()}
            write_json(fit_dir / f"{arm}.json", record)
            rows.append(record)
            del iterator, train_loader, val_loader, train_x, train_y, model, optimizer, criterion
        paired = ("initial_state_sha256", "sampling_index_sha256", "final_cpu_rng_sha256", "final_cuda_rng_sha256")
        assert all(rows[0][key] == rows[1][key] for key in paired)
        result = {"status": "complete", "fits": 2, "pairing_checked": list(paired),
            "elapsed_seconds": time.monotonic() - start, "utc": now(),
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "numpy": np.__version__}
        write_json(fit_dir / "run.json", result)
        if stage == "pilot":
            write_json(out / "pilot_technical_receipt.json", {**result, "status": "pass",
                "excluded_from_confirmation": True, "detector_test_scoring_performed": False,
                "configuration_tuned_on_pilot": False,
                "projected_1000_fit_training_seconds": result["elapsed_seconds"] * 500,
                "projection_caveat": "one pair including source loading; excludes pool generation and test scoring"})
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(fit_dir / "failure.json", {"utc": now(), "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "fit-pair"))
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--pipeline", type=int, default=7)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "prepare":
        prepare(args.stage, args.index)
    else:
        fit_pair(args.stage, args.index, args.pipeline)
