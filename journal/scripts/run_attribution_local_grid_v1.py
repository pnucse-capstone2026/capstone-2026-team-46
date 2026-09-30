#!/usr/bin/env python3
"""Run the fixed B training grid sequentially; no test scoring or retry policy."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import run_attribution_local_replication_v1 as study


def grid(cfg):
    jobs = []
    for index in range(cfg["confirmation_realizations"]):
        jobs.append(("prepare", index, None))
        jobs.extend(("fit-pair", index, pipeline) for pipeline in cfg["pipelines"])
    return jobs


def main():
    frozen = study.verify_freeze()
    cfg = study.config()
    pilot = study.ROOT / "pilot_v1/g000"
    for name in ("pilot_technical_receipt.json", "serialized_verification_v1.json"):
        assert json.loads((pilot / name).read_text())["status"] == "pass"
    main_root = study.ROOT / "confirmation_v1"
    if main_root.exists():
        raise FileExistsError("Main grid has existing data; this launcher does not resume or overwrite it")
    jobs = grid(cfg)
    assert len(jobs) == 600 and sum(action == "fit-pair" for action, _, _ in jobs) == 500
    prepared = json.loads((pilot / "prepare.json").read_text())
    projected_pool_bytes = sum(r["bytes"] for r in prepared["pools"]) * cfg["confirmation_realizations"]
    assert shutil.disk_usage(study.ROOT).free > 1.25 * projected_pool_bytes + 5 * 1024**3
    launch = study.ROOT / "confirmation_launch_v1"
    launch.mkdir(exist_ok=False)
    start = time.monotonic()
    study.write_json(launch / "started.json", {"status": "started", "utc": study.now(),
        "pid": os.getpid(), "freeze_sha256": study.sha(study.FREEZE),
        "launcher_sha256": study.sha(__file__), "planned_fits": 1000, "planned_pool_pairs": 100,
        "construction_seeds": frozen["seeds"]["confirmation"], "pipelines": cfg["pipelines"],
        "projected_pool_bytes": projected_pool_bytes,
        "test_scoring_enabled": False, "retry_enabled": False,
        "pilot_receipts": {name: study.sha(pilot / name) for name in
            ("pilot_technical_receipt.json", "serialized_verification_v1.json")}})
    completed_jobs, completed_fits = 0, 0
    try:
        with (launch / "progress.jsonl").open("x", buffering=1) as progress:
            for action, index, pipeline in jobs:
                name = f"g{index:03d}_{action}" + (f"_p{pipeline}" if pipeline is not None else "")
                command = [sys.executable, str(study.REPO / "journal/scripts/run_attribution_local_replication_v1.py"),
                           action, "--stage", "confirmation", "--index", str(index)]
                if pipeline is not None:
                    command.extend(["--pipeline", str(pipeline)])
                with (launch / f"{name}.log").open("x") as log:
                    process = subprocess.Popen(command, cwd=study.REPO, stdout=log, stderr=subprocess.STDOUT)
                    event = {"event": "started", "utc": study.now(), "job": name,
                             "child_pid": process.pid, "completed_fits": completed_fits}
                    progress.write(json.dumps(event) + "\n")
                    print(json.dumps(event), flush=True)
                    code = process.wait()
                if code != 0:
                    raise RuntimeError(f"{name} failed with exit code {code}; see preserved child log")
                receipt = main_root / f"g{index:03d}" / (
                    "prepare.json" if action == "prepare" else f"pipeline_{pipeline}/run.json")
                assert json.loads(receipt.read_text())["status"] == "complete"
                completed_jobs += 1
                completed_fits += 2 if action == "fit-pair" else 0
                event = {"event": "complete", "utc": study.now(), "job": name,
                         "completed_fits": completed_fits, "receipt_sha256": study.sha(receipt)}
                progress.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
        assert completed_jobs == 600 and completed_fits == 1000
        study.verify_freeze()
        result = {"status": "complete", "utc": study.now(), "fits": completed_fits,
                  "jobs": completed_jobs, "elapsed_seconds": time.monotonic() - start,
                  "test_scoring_performed": False, "overall_research_complete": False}
        study.write_json(launch / "run.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        study.write_json(launch / "failure.json", {"status": "failed", "utc": study.now(),
            "error": repr(exc), "completed_jobs": completed_jobs, "completed_fits": completed_fits})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", required=True,
                        help="Start all 100 pool pairs and 1,000 fits after pilot checks")
    parser.parse_args()
    main()
