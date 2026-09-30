#!/usr/bin/env python3
"""Prepare and freeze B evaluation; score only after the complete training grid."""
import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import t
import torch

import evaluate_e20_missing_cell_probe as old
import run_attribution_local_replication_v1 as study

ROOT = study.ROOT / "evaluation_v1"
PREPARED = ROOT / "prepared"
FROZEN = ROOT / "freeze.json"
SCORED = ROOT / "scored"
ANALYSIS = ROOT / "analysis"
SPEC = study.ROOT / "EVALUATION_v1.md"
TESTS = study.REPO / "journal/tests/test_attribution_local_evaluation_v1.py"
BLOCKS = ("block_01", "block_02", "block_03")
KS = (2, 4, 8, 16, 32)
ARMS = ("rule", "placebo")
RELATIONS = ("intact", "broken")
GROUP = ["realization", "construction_seed", "pipeline", "arm"]
CELL = ["block", "attack_label", "k", "relation"]
PAIRED = ("initial_state_sha256", "sampling_index_sha256", "final_cpu_rng_sha256", "final_cuda_rng_sha256")


def read(path):
    return json.loads(Path(path).read_text())


def configure_cpu():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def cell_key(block, label=0, k=0, relation="normal"):
    return f"{block}_a{label}_k{k}_{relation}"


def cells():
    result = [{"key": cell_key(b), "kind": "normal", "block": b,
               "attack_label": 0, "k": 0, "relation": "normal", "n": 2000} for b in BLOCKS]
    result += [{"key": cell_key(b, a, k, r), "kind": "attack", "block": b,
                "attack_label": a, "k": k, "relation": r, "n": 2000}
               for b, a, k, r in product(BLOCKS, (3, 4), KS, RELATIONS)]
    return result


def identities(cfg, seeds):
    return [{"realization": g, "construction_seed": seed, "pipeline": p, "arm": arm}
            for g, seed in enumerate(seeds) for p in cfg["pipelines"] for arm in ARMS]


def check_pair(base, latent, label, k, intact, broken, stats):
    positions = latent.s0_positions[:k]
    other = [i for i in range(11) if i != 3]
    if not np.array_equal(intact[:, other], broken[:, other]):
        raise ValueError("Non-target channel mismatch")
    outside = np.ones(128, dtype=bool)
    outside[positions] = False
    if not np.array_equal(intact[outside], broken[outside]) or not np.array_equal(intact[outside], base[outside]):
        raise ValueError("Off-injection mismatch")
    if not np.array_equal(np.sort(intact[:, 3]), np.sort(broken[:, 3])):
        raise ValueError("Target multiset mismatch")
    d0, d1, replacement = intact[positions, 2], intact[positions, 3], broken[positions, 3]
    if label == 3:
        original = np.abs(255 - d0 - d1) <= 8
        residual = int(np.count_nonzero(np.abs(255 - d0 - replacement) <= 8))
        minimum = 0
    else:
        original = (d1.astype(int) - 8 * d0.astype(int)) % 256 < 8
        residual = int(np.count_nonzero((replacement.astype(int) - 8 * d0.astype(int)) % 256 < 8))
        minimum = study.gen.rpm_structural_minimum(d0)
        if residual != minimum:
            raise ValueError("RPM residual is not the per-window structural minimum")
    if not original.all() or stats != {"frames": k, "off_relation": k - residual, "structural_minimum": minimum}:
        raise ValueError("Predicate audit disagrees with transform metadata")
    if k == 32:
        reference = old.e14.apply_factorial_transform(base, latent, attack_label=label,
                    id_stratum="canonical", p=0, s=0, d=0)
        if not np.array_equal(intact, reference):
            raise ValueError("E14 intact continuity failure")
    return {"frames": k, "residual": residual, "structural_minimum": minimum,
            "unchanged": int(np.array_equal(intact, broken))}


def save_cache(meta, raw, mean, std):
    data = old.e14._standardize(raw, mean, std)
    path = PREPARED / (meta["key"] + ".npy")
    with path.open("xb") as handle:
        np.save(handle, data, allow_pickle=False)
    reloaded = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.array_equal(data, reloaded):
        raise ValueError("Cache round-trip mismatch")
    return {**meta, "path": str(path.relative_to(study.REPO)), "sha256": study.sha(path)}


def prepare():
    study.verify_freeze()
    PREPARED.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    study.write_json(PREPARED / "started.json", {"utc": study.now(), "new_model_inference": False,
        "training_freeze_sha256": study.sha(study.FREEZE), "evaluator_sha256": study.sha(__file__)})
    try:
        manifest, bases, latents = old.e14.load_prepared_inputs(read(old.e14.PREPARE_PATHS["prepare_record"]))
        mean, std, standardizer_hashes = old.e14.fit_registered_standardizer()
        lookup = {(l.block_id, l.block_position, l.test_window_index, l.attack_label): l for l in latents}
        inventory, manipulation = [], []
        for block in BLOCKS:
            selected = manifest.loc[manifest.block_id == block]
            raw_bases = bases[selected.index]
            if len(selected) != 2000:
                raise ValueError("Unexpected block size")
            meta = next(c for c in cells() if c["key"] == cell_key(block))
            inventory.append(save_cache(meta, raw_bases, mean, std))
            for label, k in product((3, 4), KS):
                intact, broken = np.empty_like(raw_bases), np.empty_like(raw_bases)
                totals = {"frames": 0, "residual": 0, "structural_minimum": 0, "unchanged": 0}
                for i, row in enumerate(selected.itertuples()):
                    latent = lookup[(block, row.block_position, row.test_window_index, label)]
                    intact[i], _ = old.apply_missing_cell_transform(raw_bases[i], latent,
                                    attack_label=label, k=k, relation="intact")
                    broken[i], stats = old.apply_missing_cell_transform(raw_bases[i], latent,
                                    attack_label=label, k=k, relation="broken")
                    checks = check_pair(raw_bases[i], latent, label, k, intact[i], broken[i], stats)
                    for name in totals:
                        totals[name] += checks[name]
                manipulation.append({"block": block, "attack_label": label, "k": k, **totals})
                for relation, data in (("intact", intact), ("broken", broken)):
                    meta = next(c for c in cells() if c["key"] == cell_key(block, label, k, relation))
                    inventory.append(save_cache(meta, data, mean, std))
            print(json.dumps({"prepared_block": block, "new_model_inference": False}), flush=True)
        inventory.sort(key=lambda c: c["key"])
        if {c["key"] for c in inventory} != {c["key"] for c in cells()}:
            raise ValueError("Cache coverage mismatch")
        pd.DataFrame(manipulation).to_csv(PREPARED / "manipulation.csv", index=False, mode="x")
        result = {"status": "complete", "utc": study.now(), "cells": inventory,
            "standardizer": {"mean": mean.tolist(), "std": std.tolist(), "hashes": standardizer_hashes},
            "roles": study.audit_roles(), "k32_intact_continuity_checks": 12000,
            "matched_pair_checks": 60000, "new_model_inference": False,
            "elapsed_seconds": time.monotonic() - start}
        study.write_json(PREPARED / "run.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("cells", "standardizer")}), flush=True)
    except BaseException as exc:
        study.write_json(PREPARED / "failure.json", {"utc": study.now(), "error": repr(exc)})
        raise


def verify_prepared():
    record = read(PREPARED / "run.json")
    if record["status"] != "complete" or len(record["cells"]) != 63:
        raise ValueError("Preparation incomplete")
    expected = {c["key"]: c for c in cells()}
    if {c["key"] for c in record["cells"]} != set(expected):
        raise ValueError("Prepared cell coverage mismatch")
    for cell in record["cells"]:
        if any(cell[k] != v for k, v in expected[cell["key"]].items()):
            raise ValueError("Prepared cell metadata mismatch")
        path = study.REPO / cell["path"]
        if path != PREPARED / (cell["key"] + ".npy") or study.sha(path) != cell["sha256"]:
            raise ValueError("Prepared cache hash or path mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (2000, 128, 11) or values.dtype != np.float32:
            raise ValueError("Prepared cache representation mismatch")
    return record


def predictions(model, values):
    logits = old.e14.predict_standardized_logits(torch, model, values, batch_size=4096)
    return logits.argmax(axis=1).astype(np.uint8)


def counts(prediction, cell):
    prediction = np.asarray(prediction)
    if prediction.shape != (cell["n"],) or not np.isin(prediction, np.arange(5)).all():
        raise ValueError("Prediction shape or class range mismatch")
    return {"exact_correct": int(np.count_nonzero(prediction == cell["attack_label"])),
            "binary_attack": int(np.count_nonzero(prediction != 0))}


def continuity():
    study.verify_freeze()
    prepared = verify_prepared()
    out = ROOT / "historical_continuity.json"
    if out.exists():
        raise FileExistsError(out)
    configure_cpu()
    path = old.e16train.shared_real_checkpoint(7)
    bundle = next(b for b in read(old.e14.PREPARE_PATHS["prepare_record"])["checkpoint_bundles"]
                  if b["arm"] == "real_ms" and b["seed"] == 7)
    expected_hash = bundle["checkpoint"]["sha256"]
    if study.REPO / bundle["checkpoint"]["path"] != path:
        raise ValueError("Historical reference path mismatch")
    if study.sha(path) != expected_hash:
        raise ValueError("Historical reference checkpoint changed")
    model = study.CNN1D(classes=5)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    historical = pd.read_csv(old.TABLE_DIR / "e20_missing_cell_by_scenario_v1.csv")
    reference = historical[(historical.arm == "real_ms") & (historical.pipeline_seed == 7)]
    if len(reference) != 60:
        raise ValueError("Historical reference grid mismatch")
    checked = []
    for cell in prepared["cells"]:
        if cell["kind"] != "attack":
            continue
        actual = counts(predictions(model, np.load(study.REPO / cell["path"], mmap_mode="r")), cell)
        row = reference[(reference.block_id == cell["block"]) & (reference.attack_label == cell["attack_label"])
                        & (reference.k == cell["k"]) & (reference.relation == cell["relation"])].iloc[0]
        if actual != {"exact_correct": int(row.exact_correct), "binary_attack": int(row.binary_correct)}:
            raise ValueError(f"Historical prediction-count continuity failed: {cell['key']}")
        checked.append({"cell": cell["key"], **actual})
    study.write_json(out, {"status": "pass", "utc": study.now(), "checked_cells": checked,
        "checkpoint": str(path.relative_to(study.REPO)), "checkpoint_sha256": expected_hash,
        "historical_table_sha256": study.sha(old.TABLE_DIR / "e20_missing_cell_by_scenario_v1.csv"),
        "new_model_inference": False, "evaluator_sha256": study.sha(__file__)})
    print(json.dumps({"historical_cells_verified": len(checked), "new_model_inference": False}), flush=True)


def evaluator_inputs():
    paths = [Path(__file__), SPEC, TESTS, study.FREEZE, PREPARED / "run.json",
             PREPARED / "manipulation.csv", ROOT / "historical_continuity.json"]
    return [{"path": str(p.relative_to(study.REPO)), "sha256": study.sha(p)} for p in paths]


def freeze():
    study.verify_freeze()
    verify_prepared()
    if SCORED.exists() or read(ROOT / "historical_continuity.json")["status"] != "pass":
        raise RuntimeError("Cannot freeze after new scoring or before continuity passes")
    if read(ROOT / "historical_continuity.json")["evaluator_sha256"] != study.sha(__file__):
        raise RuntimeError("Evaluator changed since historical continuity")
    study.write_json(FROZEN, {"utc": study.now(), "inputs": evaluator_inputs(),
        "new_model_inference": False, "public_preregistration": False})
    print(json.dumps({"evaluation_freeze": str(FROZEN.relative_to(study.REPO)), "sha256": study.sha(FROZEN)}))


def verify_freeze():
    study.verify_freeze()
    if read(FROZEN)["inputs"] != evaluator_inputs():
        raise ValueError("Frozen evaluator input changed")
    return verify_prepared()


def require_complete_launch(record):
    if record.get("status") != "complete" or record.get("fits") != 1000 or record.get("jobs") != 600:
        raise RuntimeError("The full main training grid must finish before evaluation")


def preflight():
    prepared = verify_freeze()
    launch = study.ROOT / "confirmation_launch_v1/run.json"
    if not launch.exists():
        raise RuntimeError("Main training is incomplete; no new checkpoint will be loaded")
    require_complete_launch(read(launch))
    cfg, seeds = study.config(), read(study.FREEZE)["seeds"]["confirmation"]
    result = []
    base = study.ROOT / "confirmation_v1"
    expected_paths = set()
    for g, seed in enumerate(seeds):
        pool = read(base / f"g{g:03d}/prepare.json")
        if pool["status"] != "complete" or not pool["abort_gates_passed"] or pool["construction_seed"] != seed:
            raise ValueError("Unverified main construction pair")
        for pipeline in cfg["pipelines"]:
            directory = base / f"g{g:03d}/pipeline_{pipeline}"
            pair = read(directory / "run.json")
            if pair["status"] != "complete" or pair["fits"] != 2:
                raise ValueError("Incomplete paired fit")
            records = []
            for arm in ARMS:
                identity = {"realization": g, "construction_seed": seed, "pipeline": pipeline, "arm": arm}
                log = read(directory / f"{arm}.json")
                path = directory / f"{arm}.pt"
                expected_paths.add(path)
                if any(log[k] != v for k, v in identity.items()) or log["stage"] != "confirmation":
                    raise ValueError("Fit identity mismatch")
                if log["steps"] != cfg["steps"] or [h["step"] for h in log["history"]] != list(range(513, 6157, 513)):
                    raise ValueError("Training budget mismatch")
                best = max(log["history"], key=lambda h: h["validation_macro_f1"])
                if log["selected_step"] != best["step"] or log["best_validation_macro_f1"] != best["validation_macro_f1"]:
                    raise ValueError("Earliest-best validation selection mismatch")
                for name in ("mean", "std"):
                    if not np.array_equal(np.asarray(log[f"standardizer_{name}"], dtype=np.float32),
                                          np.asarray(prepared["standardizer"][name], dtype=np.float32)):
                        raise ValueError("Standardizer mismatch")
                if study.REPO / log["checkpoint"] != path or study.sha(path) != log["checkpoint_sha256"]:
                    raise ValueError("Checkpoint path/hash mismatch")
                result.append({**identity, "checkpoint": log["checkpoint"], "sha256": log["checkpoint_sha256"],
                               "training_log_sha256": study.sha(directory / f"{arm}.json")})
                records.append(log)
            if any(records[0][key] != records[1][key] for key in PAIRED):
                raise ValueError("Paired training mismatch")
    if set(base.glob("g*/pipeline_*/*.pt")) != expected_paths or len(result) != 1000:
        raise ValueError("Unexpected/missing main checkpoint")
    return prepared, result


def score():
    prepared, models = preflight()
    configure_cpu()
    SCORED.mkdir(exist_ok=False)
    study.write_json(SCORED / "started.json", {"utc": study.now(), "models": models,
                     "evaluation_freeze_sha256": study.sha(FROZEN)})
    start = time.monotonic()
    try:
        caches = {c["key"]: np.load(study.REPO / c["path"], mmap_mode="r") for c in prepared["cells"]}
        for identity in models:
            key = f"g{identity['realization']:03d}_p{identity['pipeline']}_{identity['arm']}"
            model = study.CNN1D(classes=5)
            path = study.REPO / identity["checkpoint"]
            if study.sha(path) != identity["sha256"]:
                raise ValueError("Checkpoint changed after preflight")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            saved, rows = {}, []
            for cell in prepared["cells"]:
                pred = predictions(model, caches[cell["key"]])
                saved[cell["key"]] = pred
                rows.append({**{k: identity[k] for k in GROUP},
                             **{k: cell[k] for k in ("kind", *CELL, "n")}, **counts(pred, cell)})
            path = SCORED / (key + ".npz")
            with path.open("xb") as handle:
                np.savez_compressed(handle, **saved)
            study.write_json(SCORED / (key + ".json"), {"identity": identity, "rows": rows,
                "predictions_sha256": study.sha(path), "utc": study.now()})
            print(json.dumps({"scored": key}), flush=True)
        study.write_json(SCORED / "run.json", {"status": "complete", "fits": len(models),
            "utc": study.now(), "elapsed_seconds": time.monotonic() - start,
            "device": "cpu", "torch": torch.__version__, "intraop_threads": 8, "batch_size": 4096})
    except BaseException as exc:
        study.write_json(SCORED / "failure.json", {"utc": study.now(), "error": repr(exc)})
        raise


def validate_counts(frame, cfg, seeds):
    keys = ["realization", "pipeline", "arm", *CELL]
    expected = {(i["realization"], i["pipeline"], i["arm"], c["block"], c["attack_label"], c["k"], c["relation"])
                for i in identities(cfg, seeds) for c in cells()}
    if len(frame) != len(expected) or frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete, duplicate or undeclared evaluation cells")
    if not frame.construction_seed.eq(frame.realization.map(dict(enumerate(seeds)))).all():
        raise ValueError("Construction identity mismatch")
    if frame.isna().any().any() or not frame.n.eq(2000).all():
        raise ValueError("Missing counts or wrong cell denominators")
    for name in ("exact_correct", "binary_attack"):
        values = frame[name].to_numpy()
        if not (np.isfinite(values).all() and np.equal(values, np.floor(values)).all() and ((values >= 0) & (values <= 2000)).all()):
            raise ValueError("Invalid integer count")
    attack = frame.attack_label.ne(0)
    if not (frame.kind.eq("attack") == attack).all() or not frame.loc[attack, "binary_attack"].ge(frame.loc[attack, "exact_correct"]).all():
        raise ValueError("Exact/binary count or kind mismatch")
    if not (frame.loc[~attack, "binary_attack"] + frame.loc[~attack, "exact_correct"]).eq(2000).all():
        raise ValueError("Normal FP/TN counts mismatch")


def interval(values, confidence):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite construction units required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n_constructions": len(values), "mean": mean, "sd": sd, "confidence": confidence,
            "ci_low": mean - half, "ci_high": mean + half, "positive_units": int((values > 0).sum())}


def analyze(frame, cfg, seeds):
    validate_counts(frame, cfg, seeds)
    attack = frame[frame.kind == "attack"].copy()
    pipeline_rows = []
    for metric, column in (("exact", "exact_correct"), ("binary", "binary_attack")):
        attack["recall"] = attack[column] / attack.n
        for family, labels in (("macro", (3, 4)), ("Gear", (3,)), ("RPM", (4,))):
            averaged = attack[attack.attack_label.isin(labels)].groupby(
                ["realization", "construction_seed", "pipeline", "k", "arm", "relation"]).recall.mean().unstack(["arm", "relation"])
            for index, row in averaged.iterrows():
                r_i, r_b, p_i, p_b = [float(row[a, r]) for a, r in product(ARMS, RELATIONS)]
                pipeline_rows.append(dict(zip(("realization", "construction_seed", "pipeline", "k"), index)) |
                    {"metric": metric, "family": family, "rule_intact": r_i, "rule_broken": r_b,
                     "placebo_intact": p_i, "placebo_broken": p_b, "U": r_i - p_i,
                     "L": (r_i - r_b) - (p_i - p_b)})
    by_pipeline = pd.DataFrame(pipeline_rows)
    endpoints = ["rule_intact", "rule_broken", "placebo_intact", "placebo_broken", "U", "L"]
    unit_keys = ["realization", "construction_seed", "metric", "family", "k"]
    units = by_pipeline.groupby(unit_keys, as_index=False)[endpoints].mean()
    summary, sensitivity = [], []
    for (metric, family, k), group in units.groupby(["metric", "family", "k"]):
        for endpoint in endpoints:
            primary = metric == "exact" and family == "macro" and k == 8 and endpoint in ("U", "L")
            summary.append({"metric": metric, "family": family, "k": int(k), "endpoint": endpoint,
                "analysis_role": "co_primary" if primary else "descriptive_pointwise",
                **interval(group[endpoint], 0.975 if primary else 0.95)})
    for (pipeline, metric, family, k), group in by_pipeline.groupby(["pipeline", "metric", "family", "k"]):
        for endpoint in endpoints:
            sensitivity.append({"pipeline": int(pipeline), "metric": metric, "family": family, "k": int(k),
                "endpoint": endpoint, "analysis_role": "descriptive_fixed_pipeline", **interval(group[endpoint], 0.95)})
    normal = frame[frame.kind == "normal"].copy()
    normal["fpr"] = normal.binary_attack / normal.n
    fpr_pipeline = normal.groupby(["realization", "construction_seed", "pipeline", "arm"]).fpr.mean().unstack("arm").reset_index()
    fpr_pipeline["delta"] = fpr_pipeline.rule - fpr_pipeline.placebo
    fpr_units = fpr_pipeline.groupby(["realization", "construction_seed"], as_index=False)[["rule", "placebo", "delta"]].mean()
    fpr_summary = [{"endpoint": name, "analysis_role": "descriptive_pointwise", **interval(fpr_units[name], 0.95)}
                   for name in ("rule", "placebo", "delta")]
    return {"by_pipeline": by_pipeline, "construction_units": units, "summary": pd.DataFrame(summary),
            "pipeline_sensitivity": pd.DataFrame(sensitivity), "fpr_by_pipeline": fpr_pipeline,
            "fpr_units": fpr_units, "fpr_summary": pd.DataFrame(fpr_summary)}


def summarize():
    verify_freeze()
    if read(SCORED / "run.json").get("fits") != 1000 or read(SCORED / "run.json").get("status") != "complete":
        raise RuntimeError("Complete scoring required before summary")
    models = read(SCORED / "started.json")["models"]
    rows = []
    for identity in models:
        key = f"g{identity['realization']:03d}_p{identity['pipeline']}_{identity['arm']}"
        record = read(SCORED / (key + ".json"))
        path = SCORED / (key + ".npz")
        if record["identity"] != identity or study.sha(path) != record["predictions_sha256"]:
            raise ValueError("Saved fit identity/prediction hash mismatch")
        if any(any(row[k] != identity[k] for k in GROUP) for row in record["rows"]):
            raise ValueError("Saved count row belongs to another model")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {c["key"] for c in cells()}:
                raise ValueError("Saved prediction cell coverage mismatch")
            by_key = {cell_key(r["block"], r["attack_label"], r["k"], r["relation"]): r for r in record["rows"]}
            if len(by_key) != len(cells()) or len(record["rows"]) != len(cells()):
                raise ValueError("Saved count coverage mismatch")
            for cell in cells():
                actual = counts(saved[cell["key"]], cell)
                if any(by_key[cell["key"]][k] != v for k, v in actual.items()):
                    raise ValueError("Saved counts disagree with predictions")
        rows.extend(record["rows"])
    frame = pd.DataFrame(rows)
    result = analyze(frame, study.config(), read(study.FREEZE)["seeds"]["confirmation"])
    ANALYSIS.mkdir(exist_ok=False)
    frame.to_csv(ANALYSIS / "counts.csv", index=False, mode="x")
    for name, table in result.items():
        table.to_csv(ANALYSIS / f"{name}.csv", index=False, mode="x")
    primary = result["summary"].query("analysis_role == 'co_primary'").to_dict("records")
    study.write_json(ANALYSIS / "run.json", {"status": "complete", "utc": study.now(),
        "co_primary": primary, "n_constructions": 100, "evaluation_freeze_sha256": study.sha(FROZEN),
        "interpretation": "construction-conditional U/L; no equivalence or source-generalization claim"})
    print(json.dumps({"co_primary": primary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "continuity", "freeze", "preflight", "score", "summarize"))
    action = parser.parse_args().action
    if action == "preflight":
        _, models = preflight()
        print(json.dumps({"ready": True, "verified_models": len(models)}))
    else:
        globals()[action]()
