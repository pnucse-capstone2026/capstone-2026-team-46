#!/usr/bin/env python3
"""ROAD calibration probe with logit-space / OOD-aware scores (DIAGNOSTIC ONLY).

Closes the admitted gap of road_calibration_probe.py, which only tested the
softmax attack score 1 - p(normal). Max-logit and energy scores are the
standard post-hoc remedies for softmax saturation, so a reviewer can object
that "no separating threshold exists" was only shown for softmax.

Protocol is identical to road_calibration_probe.py (same models, same
capture-level calibration/held-out split, same 1% calibration-FPR threshold,
original Car-Hacking standardizer only). Three score functions side by side
(higher = more anomalous for all three):
  1) softmax : 1 - p(normal)      -- reference; must reproduce existing numbers
  2) maxlogit: -max_c z_c         -- max-logit OOD score
  3) energy  : -logsumexp_c z_c   -- energy OOD score

Per score x setting x seed we report held-out ambient FPR at the
1%-calibration-FPR threshold, attack-capture window recall at that threshold,
AUROC separating held-out ambient windows from attack-capture attack windows
(whether ANY threshold could work), and the fraction of held-out ambient
windows sitting exactly at the maximal observed score (saturation).

This probe never touches model weights or model selection, and does not
modify any existing output; it writes only new *_logit_* files.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from run_cantt_tier1 import FEATURE_NAMES, LOGS, TABLES, iter_windows, load_models
from train_channel_ablation import write_csv
from train_real_only_baselines import WindowDataset, standardize

ROOT = Path(__file__).resolve().parents[1]
ROAD_FRAMES = ROOT / "datasets" / "processed" / "road_frames"
SELECTED_SETTINGS = {"real_only", "rule_0p30"}
CALIBRATION_CAPTURES = ["ambient_dyno_drive_basic_long", "ambient_highway_street_driving_long"]
TARGET_FPR = 0.01
SCORE_FNS = ["softmax", "maxlogit", "energy"]


def capture_windows(path: Path):
    df = pd.read_parquet(path)
    features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    frame_y = df["attack"].to_numpy(dtype=np.int8)
    yield from iter_windows(features, frame_y)


def predict_cnn_logits(model, loader, device):
    """Like train_real_only_baselines.predict_cnn but also returns raw logits.

    Softmax is computed on-device exactly as predict_cnn does, so the softmax
    reference scores are bit-identical to the original probe. New helper here
    instead of changing predict_cnn so existing callers are untouched.
    """
    model.eval()
    probs, logits_all = [], []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            logits_all.append(logits.cpu().numpy())
    return np.concatenate(probs), np.concatenate(logits_all)


def model_score_triplet(info: dict, x_std: np.ndarray) -> dict[str, np.ndarray]:
    # Same batching as the original probe's model_scores (DataLoader bs=1024).
    loader = DataLoader(WindowDataset(x_std, np.zeros(len(x_std), dtype=np.int64)), batch_size=1024, shuffle=False, num_workers=0)
    probs, logits = predict_cnn_logits(info["model"], loader, info["device"])
    z = logits.astype(np.float64)
    return {
        "softmax": (1.0 - probs[:, 0]).astype(np.float64),
        "maxlogit": -z.max(axis=1),
        "energy": -logsumexp(z, axis=1),
    }


def main() -> None:
    start = time.time()
    models = [m for m in load_models() if m["setting"] in SELECTED_SETTINGS]
    profile = pd.read_csv(TABLES / "road_dataset_profile.csv").set_index("capture")
    captures = sorted(ROAD_FRAMES.glob("*.parquet"))
    calib_paths = [p for p in captures if p.stem in CALIBRATION_CAPTURES]
    heldout_paths = [p for p in captures if p.stem not in CALIBRATION_CAPTURES]
    assert len(calib_paths) == 2

    # Pass 1: calibration scores under the original standardizer -> thresholds.
    calib_scores: dict[tuple, list] = defaultdict(list)
    with torch.no_grad():
        for path in calib_paths:
            for x, _y, _s in capture_windows(path):
                for info in models:
                    triplet = model_score_triplet(info, standardize(x, info["mean"], info["std"]))
                    for fn in SCORE_FNS:
                        calib_scores[(fn, info["setting"], info["seed"])].append(triplet[fn])
    calib_scores = {k: np.concatenate(v) for k, v in calib_scores.items()}
    thresholds = {k: float(np.quantile(v, 1.0 - TARGET_FPR)) for k, v in calib_scores.items()}

    # Pass 2: held-out scores (kept in memory: scalar score per window).
    ambient_scores: dict[tuple, list] = defaultdict(list)
    attack_scores: dict[tuple, list] = defaultdict(list)
    attack_labels: list[np.ndarray] = []
    with torch.no_grad():
        for idx, path in enumerate(heldout_paths, start=1):
            capture = path.stem
            axis = "ambient_heldout" if profile.loc[capture]["role"] == "ambient" else "attack_captures"
            print(f"probe {idx}/{len(heldout_paths)} {capture} ({axis})", flush=True)
            for x, y, _s in capture_windows(path):
                if axis == "attack_captures":
                    attack_labels.append(y)
                for info in models:
                    triplet = model_score_triplet(info, standardize(x, info["mean"], info["std"]))
                    store = ambient_scores if axis == "ambient_heldout" else attack_scores
                    for fn in SCORE_FNS:
                        store[(fn, info["setting"], info["seed"])].append(triplet[fn])

    # attack_labels repeats once per capture batch but is model-independent:
    # it was appended once per window batch, aligned with each model's scores.
    attack_y = np.concatenate(attack_labels).astype(bool)

    seed_rows = []
    for fn in SCORE_FNS:
        for info in models:
            key = (fn, info["setting"], info["seed"])
            thr = thresholds[key]
            calib = calib_scores[key]
            amb = np.concatenate(ambient_scores[key])
            att = np.concatenate(attack_scores[key])
            assert len(att) == len(attack_y)
            att_attack = att[attack_y]
            att_normal = att[~attack_y]
            amb_max = float(amb.max())
            auroc_y = np.concatenate([np.zeros(len(amb), dtype=np.int8), np.ones(len(att_attack), dtype=np.int8)])
            auroc = float(roc_auc_score(auroc_y, np.concatenate([amb, att_attack])))
            seed_rows.append(
                {
                    "score": fn,
                    "setting": info["setting"],
                    "seed": info["seed"],
                    "threshold": thr,
                    "calib_windows": len(calib),
                    "calib_fpr_at_threshold": float((calib >= thr).mean()),
                    "ambient_heldout_windows": len(amb),
                    "ambient_heldout_fpr": float((amb >= thr).mean()),
                    "attack_capture_attack_windows": int(attack_y.sum()),
                    "attack_recall": float((att_attack >= thr).mean()),
                    "attack_capture_normal_windows": int((~attack_y).sum()),
                    "attack_capture_fpr": float((att_normal >= thr).mean()),
                    "auroc_ambient_vs_attack": auroc,
                    "ambient_max_score": amb_max,
                    "ambient_saturation_frac": float((amb == amb_max).mean()),
                }
            )
    write_csv(TABLES / "road_calibration_probe_logit_by_seed.csv", seed_rows)

    df = pd.DataFrame(seed_rows)
    summary = []
    agg_cols = [
        ("ambient_heldout_fpr", "fpr"),
        ("attack_recall", "attack_recall"),
        ("auroc_ambient_vs_attack", "auroc"),
        ("ambient_saturation_frac", "saturation_frac"),
        ("threshold", "threshold"),
    ]
    for (fn, setting), group in df.groupby(["score", "setting"], sort=False):
        row = {
            "score": fn,
            "setting": setting,
            "seeds": ";".join(str(int(s)) for s in sorted(group["seed"].unique())),
            "ambient_heldout_windows": int(group["ambient_heldout_windows"].iloc[0]),
            "attack_capture_attack_windows": int(group["attack_capture_attack_windows"].iloc[0]),
        }
        for col, name in agg_cols:
            row[f"{name}_mean"] = float(group[col].mean())
            row[f"{name}_std"] = float(group[col].std(ddof=1))
        summary.append(row)
    write_csv(TABLES / "road_calibration_probe_logit_summary.csv", summary)

    log = {
        "diagnostic": "road_calibration_probe_logit",
        "note": "diagnostic only; no model weights or model selection touched; protocol identical to road_calibration_probe.py with score functions softmax / max-logit / energy",
        "calibration_captures": CALIBRATION_CAPTURES,
        "target_fpr": TARGET_FPR,
        "settings": sorted(SELECTED_SETTINGS),
        "score_functions": SCORE_FNS,
        "thresholds": {f"{k[0]}_{k[1]}_seed{k[2]}": v for k, v in thresholds.items()},
        "outputs": [
            "results/tables/road_calibration_probe_logit_by_seed.csv",
            "results/tables/road_calibration_probe_logit_summary.csv",
        ],
        "elapsed_seconds": time.time() - start,
    }
    (LOGS / "road_calibration_probe_logit.log").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
