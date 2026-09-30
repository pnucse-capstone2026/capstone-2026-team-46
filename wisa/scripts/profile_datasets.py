#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT_PROFILE = ROOT / "experiments" / "00_profile"
OUT_TABLES = ROOT / "results" / "tables"
OUT_FIGURES = ROOT / "results" / "figures"
OUT_LOGS = ROOT / "results" / "logs"

CAR_ZIP = DATASETS / "9) Car-Hacking Dataset.zip"
OTIDS_ZIP = DATASETS / "10) CAN-Intrusion Dataset.zip"

CAR_FILES = [
    ("DoS_dataset.csv", "DoS", "car_attack_csv"),
    ("Fuzzy_dataset.csv", "Fuzzy", "car_attack_csv"),
    ("gear_dataset.csv", "Gear", "car_attack_csv"),
    ("RPM_dataset.csv", "RPM", "car_attack_csv"),
    ("normal_run_data/normal_run_data.txt", "Normal", "txt_all_normal"),
]

OTIDS_FILES = [
    ("Attack_free_dataset.txt", "Normal", "txt_all_normal"),
    ("DoS_attack_dataset.txt", "DoS", "txt_scenario_attack"),
    ("Fuzzy_attack_dataset.txt", "Fuzzy", "txt_scenario_attack"),
    ("Impersonation_attack_dataset.txt", "Impersonation", "txt_scenario_attack"),
]


class Profile:
    def __init__(self, dataset, source_file, attack_type, label_mode):
        self.dataset = dataset
        self.source_file = source_file
        self.attack_type = attack_type
        self.label_mode = label_mode
        self.rows = 0
        self.parse_errors = 0
        self.invalid_dlc = 0
        self.payload_len_mismatch = 0
        self.label_counts = Counter()
        self.binary_counts = Counter()
        self.attack_type_counts = Counter()
        self.can_id_counts = Counter()
        self.dlc_counts = Counter()
        self.ts_min = None
        self.ts_max = None
        self.prev_ts = None
        self.dt_count = 0
        self.dt_sum = 0.0
        self.dt_min = None
        self.dt_max = None
        self.dt_neg = 0
        self.dt_zero = 0
        self.monotonic_violations = 0
        self.dt_hist = Counter()

    def add_frame(self, timestamp, can_id, dlc, payload_len, original_label, binary_label, attack_type):
        self.rows += 1
        self.label_counts[original_label] += 1
        self.binary_counts[binary_label] += 1
        self.attack_type_counts[attack_type] += 1
        self.can_id_counts[can_id] += 1
        self.dlc_counts[dlc] += 1

        if dlc < 0 or dlc > 8:
            self.invalid_dlc += 1
        if payload_len != dlc:
            self.payload_len_mismatch += 1

        if self.ts_min is None or timestamp < self.ts_min:
            self.ts_min = timestamp
        if self.ts_max is None or timestamp > self.ts_max:
            self.ts_max = timestamp

        if self.prev_ts is not None:
            dt = timestamp - self.prev_ts
            self.dt_count += 1
            self.dt_sum += dt
            self.dt_min = dt if self.dt_min is None else min(self.dt_min, dt)
            self.dt_max = dt if self.dt_max is None else max(self.dt_max, dt)
            if dt < 0:
                self.dt_neg += 1
                self.monotonic_violations += 1
            if dt == 0:
                self.dt_zero += 1
            self.dt_hist[dt_bin(dt)] += 1
        self.prev_ts = timestamp

    def row(self):
        normal = self.binary_counts.get("Normal", 0)
        attack = self.binary_counts.get("Attack", 0)
        return {
            "dataset": self.dataset,
            "source_file": self.source_file,
            "label_mode": self.label_mode,
            "rows": self.rows,
            "normal_rows": normal,
            "attack_rows": attack,
            "parse_errors": self.parse_errors,
            "invalid_dlc": self.invalid_dlc,
            "payload_len_mismatch": self.payload_len_mismatch,
            "unique_can_ids": len(self.can_id_counts),
            "dlc_distribution": compact_counter(self.dlc_counts),
            "label_distribution": compact_counter(self.label_counts),
            "attack_type_distribution": compact_counter(self.attack_type_counts),
            "top_can_ids": compact_counter(self.can_id_counts, top=15, key_fmt=lambda x: f"0x{x:03x}"),
            "timestamp_min": none_to_empty(self.ts_min),
            "timestamp_max": none_to_empty(self.ts_max),
            "monotonic_violations": self.monotonic_violations,
            "delta_t_count": self.dt_count,
            "delta_t_min": none_to_empty(self.dt_min),
            "delta_t_mean": self.dt_sum / self.dt_count if self.dt_count else "",
            "delta_t_max": none_to_empty(self.dt_max),
            "delta_t_negative": self.dt_neg,
            "delta_t_zero": self.dt_zero,
        }


def none_to_empty(value):
    return "" if value is None else value


def compact_counter(counter, top=None, key_fmt=str):
    items = counter.most_common(top) if top else sorted(counter.items(), key=lambda kv: str(kv[0]))
    return ";".join(f"{key_fmt(k)}:{v}" for k, v in items)


def dt_bin(dt):
    if dt < 0:
        return "neg"
    if dt == 0:
        return "0"
    if dt < 1e-4:
        return "<0.1ms"
    if dt < 5e-4:
        return "0.1-0.5ms"
    if dt < 1e-3:
        return "0.5-1ms"
    if dt < 5e-3:
        return "1-5ms"
    if dt < 1e-2:
        return "5-10ms"
    if dt < 5e-2:
        return "10-50ms"
    if dt < 1e-1:
        return "50-100ms"
    return ">=100ms"


def parse_car_csv_line(line):
    parts = line.rstrip(b"\r\n").split(b",")
    if len(parts) < 5:
        raise ValueError("too few csv fields")
    timestamp = float(parts[0])
    can_id = int(parts[1], 16)
    dlc = int(parts[2])
    payload = parts[3:-1]
    label = parts[-1].decode("ascii").strip()
    return timestamp, can_id, dlc, len(payload), label


def parse_txt_line(line):
    parts = line.decode("ascii", errors="strict").strip().split()
    # Timestamp: <ts> ID: <id> 000 DLC: <dlc> <payload...>
    timestamp = float(parts[1])
    can_id = int(parts[3], 16)
    dlc_idx = parts.index("DLC:")
    dlc = int(parts[dlc_idx + 1])
    payload = parts[dlc_idx + 2:]
    return timestamp, can_id, dlc, len(payload)


def profile_zip(zip_path, files, dataset_name):
    profiles = []
    with zipfile.ZipFile(zip_path) as zf:
        for name, attack_type, mode in files:
            profile = Profile(dataset_name, name, attack_type, mode)
            with zf.open(name) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        if mode == "car_attack_csv":
                            ts, can_id, dlc, payload_len, raw_label = parse_car_csv_line(line)
                            if raw_label == "T":
                                binary = "Attack"
                                atype = attack_type
                            elif raw_label == "R":
                                binary = "Normal"
                                atype = "Normal"
                            else:
                                binary = "Unknown"
                                atype = "Unknown"
                            profile.add_frame(ts, can_id, dlc, payload_len, raw_label, binary, atype)
                        elif mode == "txt_all_normal":
                            ts, can_id, dlc, payload_len = parse_txt_line(line)
                            profile.add_frame(ts, can_id, dlc, payload_len, "scenario_normal", "Normal", "Normal")
                        elif mode == "txt_scenario_attack":
                            ts, can_id, dlc, payload_len = parse_txt_line(line)
                            profile.add_frame(ts, can_id, dlc, payload_len, f"scenario_{attack_type}", "Attack", attack_type)
                        else:
                            raise ValueError(f"unknown mode {mode}")
                    except Exception:
                        profile.parse_errors += 1
            profiles.append(profile)
            print(f"profiled {dataset_name}: {name} rows={profile.rows} errors={profile.parse_errors}")
    return profiles


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_hashes(paths):
    lines = []
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        lines.append(f"{path.name}\tsha256\t{h.hexdigest()}\n")
    (OUT_LOGS / "dataset_hashes.txt").write_text("".join(lines))


def aggregate_counter(profiles, attr):
    total = Counter()
    for profile in profiles:
        total.update(getattr(profile, attr))
    return total


def draw_bar_chart(path, title, labels, values, width=1200, height=700):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 170
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    draw.text((margin_l, 25), title, fill="black", font=font)
    max_v = max(values) if values else 1
    bar_w = max(8, plot_w // max(1, len(labels)) - 6)
    for i, (label, value) in enumerate(zip(labels, values)):
        x0 = margin_l + i * (plot_w / max(1, len(labels)))
        x1 = x0 + bar_w
        bar_h = 0 if max_v == 0 else int((value / max_v) * plot_h)
        y0 = margin_t + plot_h - bar_h
        y1 = margin_t + plot_h
        draw.rectangle([x0, y0, x1, y1], fill="#4c78a8")
        draw.text((x0, y0 - 15), short_num(value), fill="black", font=font)
        draw.text((x0, y1 + 8), label, fill="black", font=font)
    draw.line([margin_l, margin_t, margin_l, margin_t + plot_h], fill="black")
    draw.line([margin_l, margin_t + plot_h, width - margin_r, margin_t + plot_h], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def short_num(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def main():
    OUT_PROFILE.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    car_profiles = profile_zip(CAR_ZIP, CAR_FILES, "Car-Hacking")
    otids_profiles = profile_zip(OTIDS_ZIP, OTIDS_FILES, "OTIDS")
    all_profiles = car_profiles + otids_profiles

    write_csv(OUT_PROFILE / "car_hacking_profile.csv", [p.row() for p in car_profiles])
    write_csv(OUT_PROFILE / "otids_profile.csv", [p.row() for p in otids_profiles])
    write_csv(OUT_TABLES / "dataset_statistics.csv", [p.row() for p in all_profiles])
    write_dataset_hashes([CAR_ZIP, OTIDS_ZIP])

    top_ids = aggregate_counter(all_profiles, "can_id_counts").most_common(25)
    write_csv(
        OUT_PROFILE / "top_can_ids.csv",
        [{"can_id": f"0x{k:03x}", "count": v} for k, v in top_ids],
    )

    dt_order = ["neg", "0", "<0.1ms", "0.1-0.5ms", "0.5-1ms", "1-5ms", "5-10ms", "10-50ms", "50-100ms", ">=100ms"]
    dt_total = aggregate_counter(all_profiles, "dt_hist")
    write_csv(
        OUT_PROFILE / "delta_t_histogram.csv",
        [{"bin": b, "count": dt_total.get(b, 0)} for b in dt_order],
    )

    draw_bar_chart(
        OUT_FIGURES / "can_id_distribution.png",
        "Top CAN IDs across profiled datasets",
        [f"0x{k:03x}" for k, _ in top_ids],
        [v for _, v in top_ids],
    )
    draw_bar_chart(
        OUT_FIGURES / "delta_t_distribution.png",
        "Delta-t histogram across profiled datasets",
        dt_order,
        [dt_total.get(b, 0) for b in dt_order],
    )

    summary = {
        "car_hacking_total_rows": sum(p.rows for p in car_profiles),
        "otids_total_rows": sum(p.rows for p in otids_profiles),
        "all_total_rows": sum(p.rows for p in all_profiles),
        "car_hacking_attack_rows": sum(p.binary_counts.get("Attack", 0) for p in car_profiles),
        "car_hacking_normal_rows": sum(p.binary_counts.get("Normal", 0) for p in car_profiles),
        "otids_weak_attack_rows": sum(p.binary_counts.get("Attack", 0) for p in otids_profiles),
        "otids_normal_rows": sum(p.binary_counts.get("Normal", 0) for p in otids_profiles),
        "notes": [
            "OTIDS attack files do not contain Car-Hacking-style per-frame R/T labels; counts are scenario-level weak labels.",
            "Duplicate repeated frames are not removed during profiling because they may be attack behavior.",
        ],
    }
    (OUT_PROFILE / "profile_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
