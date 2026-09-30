#!/usr/bin/env python3
"""Tier 2 ROAD preprocessing.

Parses ROAD candump logs from datasets/road.zip into per-capture parquet
frame tables using the same 11-feature schema as the Car-Hacking pipeline,
with frame-level attack labels derived from capture_metadata.json:

  attack frame := elapsed time in injection_interval
                  AND (injection_id == 'XXX' or can_id == injection_id)
                  AND payload matches injection_data_str ('X' nibble = wildcard)

Accelerator captures carry no injected frames (physical exploit state) and are
excluded from frame-labeled evaluation. ROAD is evaluation-only data.
"""
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "datasets" / "road.zip"
OUT_DIR = ROOT / "datasets" / "processed" / "road_frames"
RAW_DOC_DIR = ROOT / "datasets" / "raw" / "road"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

LINE_RE = re.compile(r"\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")

FAMILY_PATTERNS = [
    ("correlated_signal", "correlated_signal"),
    ("fuzzing", "fuzzing"),
    ("max_speedometer", "max_speedometer"),
    ("reverse_light_off", "reverse_light_off"),
    ("reverse_light_on", "reverse_light_on"),
    ("max_engine_coolant_temp", "max_engine_coolant_temp"),
    ("accelerator", "accelerator"),
    ("ambient", "ambient"),
]


def capture_family(name: str) -> str:
    for prefix, family in FAMILY_PATTERNS:
        if name.startswith(prefix):
            return family
    return "other"


def payload_matches(data_hex: str, pattern: str) -> np.ndarray:
    """Vectorized wildcard match of zero-padded 16-nibble payload strings."""
    pat = pattern.upper()
    checks = [(i, c) for i, c in enumerate(pat) if c != "X"]
    if not checks:
        return np.ones(len(data_hex), dtype=bool)
    arr = np.frombuffer("".join(data_hex).encode(), dtype="S1").reshape(len(data_hex), 16)
    mask = np.ones(len(data_hex), dtype=bool)
    for i, c in checks:
        mask &= arr[:, i] == c.encode()
    return mask


def parse_log(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    ts_list, id_list, data_list = [], [], []
    with zf.open(member) as f:
        for raw in f:
            m = LINE_RE.match(raw.decode("ascii", errors="replace"))
            if not m:
                continue
            ts_list.append(float(m.group(1)))
            id_list.append(int(m.group(2), 16))
            data_list.append(m.group(3).upper())
    df = pd.DataFrame({"timestamp": ts_list, "can_id": id_list, "data": data_list})
    df["dlc"] = (df["data"].str.len() // 2).astype(np.int8)
    padded = df["data"].str.pad(16, side="right", fillchar="0")
    bytes_arr = np.frombuffer("".join(padded).encode(), dtype="S2".replace("2", "1")).reshape(len(df), 16)
    nibbles = np.where(bytes_arr >= b"A", bytes_arr.view(np.uint8) - 55, bytes_arr.view(np.uint8) - 48)
    byte_vals = (nibbles[:, 0::2] * 16 + nibbles[:, 1::2]).astype(np.int16)
    for i in range(8):
        df[f"data{i}"] = byte_vals[:, i]
    df["delta_t"] = df["timestamp"].diff().fillna(0.0).clip(lower=0.0)
    df["elapsed"] = df["timestamp"] - df["timestamp"].iloc[0]
    df["_padded"] = padded
    return df


def label_frames(df: pd.DataFrame, meta: dict) -> np.ndarray:
    interval = meta.get("injection_interval")
    if not interval:
        return np.zeros(len(df), dtype=np.int8)
    in_interval = (df["elapsed"] >= interval[0]) & (df["elapsed"] <= interval[1])
    injection_id = meta.get("injection_id")
    if injection_id and injection_id != "XXX":
        id_ok = df["can_id"] == int(injection_id, 16)
    else:
        id_ok = np.ones(len(df), dtype=bool)
    pattern = meta.get("injection_data_str") or "X" * 16
    data_ok = payload_matches(df["_padded"].tolist(), pattern)
    return (in_interval & id_ok & data_ok).to_numpy().astype(np.int8)


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOC_DIR.mkdir(parents=True, exist_ok=True)
    profile_rows = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        attack_meta = json.loads(zf.read("road/attacks/capture_metadata.json"))
        ambient_meta = json.loads(zf.read("road/ambient/capture_metadata.json"))
        members = [
            n
            for n in zf.namelist()
            if n.endswith(".log") and "__MACOSX" not in n and ("/ambient/" in n or "/attacks/" in n)
        ]
        for idx, member in enumerate(sorted(members), start=1):
            capture = Path(member).stem
            family = capture_family(capture)
            role = "ambient" if "/ambient/" in member else "attack"
            if family == "accelerator":
                profile_rows.append(
                    {"capture": capture, "role": "excluded_accelerator", "family": family,
                     "frames": 0, "attack_frames": 0, "reason": "no injected frames; physical exploit state"}
                )
                print(f"road {idx}/{len(members)} {capture}: excluded (accelerator)", flush=True)
                continue
            meta = (attack_meta if role == "attack" else ambient_meta).get(capture, {})
            df = parse_log(zf, member)
            y = label_frames(df, meta) if role == "attack" else np.zeros(len(df), dtype=np.int8)
            df["attack"] = y
            df["masquerade"] = int(capture.endswith("_masquerade"))
            out_cols = ["timestamp", "can_id", "dlc"] + [f"data{i}" for i in range(8)] + ["delta_t", "attack", "masquerade"]
            df[out_cols].to_parquet(OUT_DIR / f"{capture}.parquet", index=False)
            profile_rows.append(
                {
                    "capture": capture,
                    "role": role,
                    "family": family,
                    "masquerade": int(capture.endswith("_masquerade")),
                    "frames": int(len(df)),
                    "attack_frames": int(y.sum()),
                    "unique_ids": int(df["can_id"].nunique()),
                    "elapsed_sec": float(df["elapsed"].iloc[-1]) if len(df) else 0.0,
                    "injection_id": meta.get("injection_id", ""),
                    "injection_data": meta.get("injection_data_str", ""),
                }
            )
            print(
                f"road {idx}/{len(members)} {capture}: frames={len(df)} attack={int(y.sum())}",
                flush=True,
            )

    profile = pd.DataFrame(profile_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    profile.to_csv(TABLES / "road_dataset_profile.csv", index=False)

    (RAW_DOC_DIR / "DOWNLOAD.md").write_text(
        "# ROAD dataset download record\n\n"
        "- Source: https://zenodo.org/records/10462796 (`road.zip?download=1`)\n"
        "- Local path: `datasets/road.zip`\n"
        "- MD5: `cab184cfc2fe12c0834bc46188c0f330` (verified)\n"
        "- License: CC-BY-4.0\n"
        "- Citation: Verma et al., ROAD: the Real ORNL Automotive Dynamometer CAN IDS dataset.\n"
        "- Role in this project: evaluation-only external real-attack dataset (Tier 2).\n"
        "- Accelerator captures are excluded from frame-labeled evaluation (no injected frames).\n"
    )

    log = {
        "zip": str(ZIP_PATH.relative_to(ROOT)),
        "captures_processed": int((profile["role"] != "excluded_accelerator").sum()),
        "captures_excluded": int((profile["role"] == "excluded_accelerator").sum()),
        "total_frames": int(profile["frames"].sum()),
        "total_attack_frames": int(profile["attack_frames"].sum()),
        "outputs": ["datasets/processed/road_frames/*.parquet", "results/tables/road_dataset_profile.csv"],
        "elapsed_seconds": time.time() - start,
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "road_preprocess.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
