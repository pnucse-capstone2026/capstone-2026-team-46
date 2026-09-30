#!/usr/bin/env python3
"""One authorized continuation of B; preserve completed fits and the lost attempt."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import run_attribution_local_replication_v1 as study
import run_attribution_local_grid_v1 as original

MAIN = study.ROOT / "confirmation_v1"
OLD = study.ROOT / "confirmation_launch_v1"
OUT = study.ROOT / "confirmation_resume_v1"
AUDIT = study.ROOT / "recovery_preflight_v1.json"
SPEC = study.ROOT / "RECOVERY_v1.md"
TEST = study.REPO / "journal/tests/test_attribution_resume_v1.py"
INTERRUPTED = ("fit-pair", 2, 123)
PAIRED = ("initial_state_sha256", "sampling_index_sha256", "final_cpu_rng_sha256",
          "final_cuda_rng_sha256", "standardizer_mean", "standardizer_std")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def name(job):
    action, index, pipeline = job
    return f"g{index:03d}_{action}" + (f"_p{pipeline}" if pipeline is not None else "")


def directory(job):
    action, index, pipeline = job
    base = MAIN / f"g{index:03d}"
    return base if action == "prepare" else base / f"pipeline_{pipeline}"


def receipt(job):
    return directory(job) / ("prepare.json" if job[0] == "prepare" else "run.json")


def file_record(path):
    require(path.is_file() and not path.is_symlink(), f"Not a regular file: {path}")
    return {"path": str(path.relative_to(study.REPO)), "sha256": study.sha(path),
            "bytes": path.stat().st_size}


def validate_fit_records(records, cfg, index, seed, pipeline):
    for record, arm in zip(records, study.ARMS, strict=True):
        identity = {"stage": "confirmation", "realization": index,
                    "construction_seed": seed, "pipeline": pipeline, "arm": arm}
        require(all(record[k] == v for k, v in identity.items()), "Fit identity mismatch")
        require(record["steps"] == cfg["steps"], "Training budget mismatch")
        history = record["history"]
        require([h["step"] for h in history] == list(range(cfg["validation_cadence"],
                cfg["steps"] + 1, cfg["validation_cadence"])), "Validation schedule mismatch")
        require(all(math.isfinite(h["validation_macro_f1"]) for h in history), "Nonfinite validation")
        best = max(history, key=lambda h: h["validation_macro_f1"])
        require(record["selected_step"] == best["step"] and
                record["best_validation_macro_f1"] == best["validation_macro_f1"],
                "Earliest-best checkpoint mismatch")
    require(all(records[0][k] == records[1][k] for k in PAIRED), "Pairing mismatch")


def verify_job(job, cfg, seeds):
    action, index, pipeline = job
    path, seed = directory(job), seeds[index]
    record = read(receipt(job))
    require(record["status"] == "complete", "Incomplete receipt")
    files = [file_record(receipt(job))]
    if action == "prepare":
        require(record["construction_seed"] == seed and record["abort_gates_passed"], "Pool gate mismatch")
        require([r["arm"] for r in record["pools"]] == list(study.ARMS), "Pool arm mismatch")
        for pool in record["pools"]:
            expected = path / f"{pool['arm']}_pool.npz"
            require(study.REPO / pool["path"] == expected, "Pool path mismatch")
            actual = file_record(expected)
            require(actual["sha256"] == pool["sha256"] and actual["bytes"] == pool["bytes"], "Pool hash mismatch")
            files.append(actual)
    else:
        require(record["fits"] == 2, "Paired fit count mismatch")
        logs = [read(path / f"{arm}.json") for arm in study.ARMS]
        validate_fit_records(logs, cfg, index, seed, pipeline)
        for arm, log in zip(study.ARMS, logs):
            expected = path / f"{arm}.pt"
            require(study.REPO / log["checkpoint"] == expected, "Checkpoint path mismatch")
            actual = file_record(expected)
            require(actual["sha256"] == log["checkpoint_sha256"], "Checkpoint hash mismatch")
            files.extend([actual, file_record(path / f"{arm}.json")])
    return {"job": name(job), "files": files}


def remaining_jobs(jobs, completed_names):
    completed_names = list(completed_names)
    require(completed_names == [name(j) for j in jobs[:len(completed_names)]],
            "Completed jobs must be the original contiguous prefix")
    return jobs[len(completed_names):]


def validate_start_only(path, seed, pipeline):
    require(path.is_dir() and not path.is_symlink(), "Missing interrupted directory")
    require({p.name for p in path.iterdir()} == {"started.json"},
            "Only the documented start-only attempt may be archived")
    record = read(path / "started.json")
    require(record["stage"] == "confirmation" and record["construction_seed"] == seed
            and record["pipeline"] == pipeline and record["freeze_sha256"] == study.sha(study.FREEZE),
            "Interrupted identity mismatch")


def archive_start_only(source, destination, seed, pipeline):
    validate_start_only(source, seed, pipeline)
    require(not destination.exists(), "Archive already exists")
    before = study.sha(source / "started.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    require(study.sha(destination / "started.json") == before, "Archived bytes changed")


def ensure_no_original_worker():
    targets = {"run_attribution_local_grid_v1.py", "run_attribution_local_replication_v1.py"}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(arg.decode(errors="replace")).name in targets for arg in args):
            raise RuntimeError(f"An original B worker is already active (PID {entry.name})")


def code_inputs():
    return [file_record(p) for p in (Path(__file__), SPEC, TEST, Path(original.__file__), study.FREEZE)]


def audit():
    ensure_no_original_worker()
    frozen = study.verify_freeze()
    cfg, seeds = study.config(), frozen["seeds"]["confirmation"]
    require(not OUT.exists() and not AUDIT.exists(), "Recovery has existing outputs")
    require(not (OLD / "run.json").exists() and not (OLD / "failure.json").exists(), "Original state changed")
    require(read(OLD / "started.json")["launcher_sha256"] == study.sha(original.__file__), "Original launcher changed")
    jobs, completed = original.grid(cfg), []
    for job in jobs:
        if receipt(job).exists():
            completed.append(verify_job(job, cfg, seeds))
        elif directory(job).exists():
            require(job == INTERRUPTED, "Unexpected partial job")
            validate_start_only(directory(job), seeds[job[1]], job[2])
    pending = remaining_jobs(jobs, [r["job"] for r in completed])
    require(len(completed) == 15 and pending[0] == INTERRUPTED, "Not the approved 24-fit checkpoint")
    require(len(list(MAIN.glob("g*/pipeline_*/*.pt"))) == 24, "Unexpected checkpoint files")
    require({p.name for p in MAIN.iterdir()} == {"g000", "g001", "g002"}, "Unexpected construction output")
    expected_pairs = {directory(j) for j in jobs[:15] if j[0] == "fit-pair"} | {directory(INTERRUPTED)}
    require(set(MAIN.glob("g*/pipeline_*")) == expected_pairs, "Unexpected pipeline output")
    events = [json.loads(line) for line in (OLD / "progress.jsonl").read_text().splitlines()]
    done = [r for r in events if r["event"] == "complete"]
    require([r["job"] for r in done] == [r["job"] for r in completed], "Original progress mismatch")
    for event, result in zip(done, completed):
        require(event["receipt_sha256"] == result["files"][0]["sha256"], "Original receipt hash mismatch")
    require(events[-1]["event"] == "started" and events[-1]["job"] == name(INTERRUPTED), "Last event changed")
    inventory = [file_record(p) for root in (MAIN, OLD) for p in sorted(root.rglob("*")) if p.is_file()]
    result = {"status": "pass", "utc": study.now(), "scientific_freeze_sha256": study.sha(study.FREEZE),
              "code_inputs": code_inputs(), "inherited_jobs": 15, "inherited_fits": 24,
              "remaining_jobs": [list(j) for j in pending], "remaining_fits": 976,
              "verified_completed": completed, "original_inventory": inventory,
              "test_scoring_performed": False, "checkpoint_loading_performed": False}
    study.write_json(AUDIT, result)
    print(json.dumps({"status": "pass", "inherited_fits": 24, "remaining_fits": 976,
                      "remaining_jobs": len(pending), "preserved_files": len(inventory),
                      "audit_sha256": study.sha(AUDIT)}), flush=True)


def verify_preserved(record, archived=False):
    source = directory(INTERRUPTED) / "started.json"
    target = OUT / "interrupted_attempts/g002_pipeline_123/started.json"
    for item in record["original_inventory"]:
        path = study.REPO / item["path"]
        if archived and path == source:
            path = target
        require(path.is_file() and not path.is_symlink() and study.sha(path) == item["sha256"],
                f"Preserved file changed: {item['path']}")


def start():
    ensure_no_original_worker()
    frozen = study.verify_freeze()
    record, cfg = read(AUDIT), study.config()
    require(record["status"] == "pass" and record["code_inputs"] == code_inputs(), "Recovery freeze changed")
    verify_preserved(record)
    pending = [tuple(j) for j in record["remaining_jobs"]]
    require(pending == original.grid(cfg)[15:] and record["remaining_fits"] == 976, "Recovery grid changed")
    OUT.mkdir(exist_ok=False)
    study.write_json(OUT / "started.json", {"status": "started", "utc": study.now(), "pid": os.getpid(),
        "recovery_audit_sha256": study.sha(AUDIT), "scientific_freeze_sha256": study.sha(study.FREEZE),
        "inherited_fits": 24, "planned_new_fits": 976, "retry_enabled": False,
        "test_scoring_enabled": False, "execution": "user systemd service; no automatic restart"})
    started, new_jobs, new_fits, child = time.monotonic(), 0, 0, None
    try:
        target = OUT / "interrupted_attempts/g002_pipeline_123"
        archive_start_only(directory(INTERRUPTED), target, frozen["seeds"]["confirmation"][2], 123)
        study.write_json(OUT / "archived_attempt.json", {"utc": study.now(),
            "original": str(directory(INTERRUPTED).relative_to(study.REPO)),
            "preserved_at": str(target.relative_to(study.REPO)), "sha256": study.sha(target / "started.json"),
            "reason": "start-only attempt lost its processes; no fitted model or outcome existed"})
        verify_preserved(record, archived=True)
        with (OUT / "progress.jsonl").open("x", buffering=1) as progress:
            for job in pending:
                action, index, pipeline = job
                command = [sys.executable, str(Path(study.__file__).resolve()), action,
                           "--stage", "confirmation", "--index", str(index)]
                if pipeline is not None:
                    command.extend(["--pipeline", str(pipeline)])
                with (OUT / f"{name(job)}.log").open("x") as log:
                    child = subprocess.Popen(command, cwd=study.REPO, stdout=log, stderr=subprocess.STDOUT)
                    event = {"event": "started", "utc": study.now(), "job": name(job),
                             "child_pid": child.pid, "completed_fits": 24 + new_fits}
                    progress.write(json.dumps(event) + "\n")
                    print(json.dumps(event), flush=True)
                    code = child.wait()
                require(code == 0, f"{name(job)} exited {code}; no automatic retry")
                result = verify_job(job, cfg, frozen["seeds"]["confirmation"])
                new_jobs += 1
                new_fits += 2 if action == "fit-pair" else 0
                event = {"event": "complete", "utc": study.now(), "job": name(job),
                         "completed_fits": 24 + new_fits, "receipt_sha256": result["files"][0]["sha256"]}
                progress.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
        require(new_jobs == 585 and new_fits == 976, "Incomplete continuation")
        study.verify_freeze()
        require(record["code_inputs"] == code_inputs(), "Recovery code changed during execution")
        verify_preserved(record, archived=True)
        manifest = [verify_job(j, cfg, frozen["seeds"]["confirmation"]) for j in original.grid(cfg)]
        require(len(list(MAIN.glob("g*/pipeline_*/*.pt"))) == 1000, "Final checkpoint coverage mismatch")
        study.write_json(OUT / "full_grid_manifest.json", manifest)
        result = {"status": "complete", "utc": study.now(), "fits": 1000, "jobs": 600,
                  "inherited_fits": 24, "new_fits": 976, "inherited_jobs": 15, "new_jobs": 585,
                  "elapsed_seconds": time.monotonic() - started, "recovery_audit_sha256": study.sha(AUDIT),
                  "full_grid_manifest_sha256": study.sha(OUT / "full_grid_manifest.json"),
                  "test_scoring_performed": False, "overall_research_complete": False}
        study.write_json(OUT / "run.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        study.write_json(OUT / "failure.json", {"status": "failed", "utc": study.now(),
            "error": repr(exc), "completed_fits": 24 + new_fits, "new_jobs": new_jobs})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "start"))
    args = parser.parse_args()
    audit() if args.action == "audit" else start()
