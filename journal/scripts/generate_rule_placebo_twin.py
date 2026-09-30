#!/usr/bin/env python3
"""E10a placebo twin generator (experiments/e10_background_grammar_crossover/PREREG.md §2).

Regenerates the aligned rule pool with the exact rng-consumption order of
wisa/scripts/generate_rule_based_synthetic.py @ SEED 314159 (injectors forked
verbatim, extended only to also return the injected positions), asserts the
aligned twin equals datasets/synthetic/rule_based_windows.npz array-for-array
(this assertion is what licenses reusing the existing rule_0p30 checkpoints),
and derives the marginal-matched placebo pool by constrained within-window
permutation of the data1 channel over injected frames:

  Gear: off-bin rotation — group injected frames by the 6 data0 level bins
        (bin = data0 // 40), randomize within-bin order, rotate by the largest
        bin size, so data1 values move across level bins and the inverse
        relation |255 - data0 - data1| <= 8 is deterministically violated
        for every off-bin assignment (PREREG margin >= 17).
  RPM:  rejection resampling (cap 1000) until no frame pair satisfies the
        coupling ((data1 - 8*data0) mod 256) in [0, 8).

Every other bit (positions, data0, data2-7, CAN ID, DLC, delta_t, labels,
row order) is identical to the aligned pool; DoS/Fuzzy windows are copied
bit-identically. Gates M1/M2/M4 and the M3 Leg-B relation-aware AUC are
computed here; the script aborts on any gate violation (PREREG §2) and never
overwrites existing outputs.
"""
import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import lib_common as lc

SEED = 314159            # aligned stream — must match the original generator
PLACEBO_SEED = 1618033   # independent stream for the constrained permutations
PER_ATTACK = 65_000
WINDOW_SIZE = 128
MAX_POSITIONS = 90       # max burst length (step 1) across all injectors
ATTACK_IDS = {"DoS": 1, "Fuzzy": 2, "Gear": 3, "RPM": 4}

OUT_NPZ = lc.SYNTHETIC / "rule_placebo_windows.npz"
ALIGNED_REF = lc.SYNTHETIC / "rule_based_windows.npz"
CHECKS_CSV = lc.TABLES / "e10a_v1_manipulation_checks.csv"
LOG_JSON = lc.LOGS / "generate_rule_placebo_twin.log"

DATA0, DATA1 = 2, 3      # feature columns [can_id, dlc, data0..data7, delta_t]


# ---------------------------------------------------------------------------
# Injectors forked verbatim from wisa/scripts/generate_rule_based_synthetic.py
# (only change: positions are returned alongside the count; the rng call
# sequence is untouched so the aligned twin reproduces the frozen pool).
# ---------------------------------------------------------------------------

def choose_positions(rng, min_len, max_len, step_options):
    burst_len = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, WINDOW_SIZE - burst_len + 1))
    step = int(rng.choice(step_options))
    return np.arange(start, start + burst_len, step)


def synth_dos(rng, x):
    pos = choose_positions(rng, 32, 90, [1, 2])
    x[pos, 0] = 0x000
    x[pos, 1] = 8
    x[pos, 2:10] = rng.choice([0, 255], size=(len(pos), 8), p=[0.92, 0.08])
    x[pos, 10] = rng.uniform(0.00003, 0.002, size=len(pos))
    return len(pos), pos


def synth_fuzzy(rng, x):
    pos = choose_positions(rng, 24, 88, [1, 2, 3, 4])
    x[pos, 0] = rng.integers(0, 2048, size=len(pos))
    dlc = rng.choice([2, 5, 8], size=len(pos), p=[0.08, 0.12, 0.80])
    x[pos, 1] = dlc
    payload = rng.integers(0, 256, size=(len(pos), 8))
    for i, d in enumerate(dlc):
        if d < 8:
            payload[i, d:] = 0
    x[pos, 2:10] = payload
    x[pos, 10] = rng.uniform(0.00005, 0.02, size=len(pos))
    return len(pos), pos


def synth_spoof(rng, x, can_id, mode):
    pos = choose_positions(rng, 32, 90, [1, 2, 3])
    x[pos, 0] = can_id
    x[pos, 1] = 8
    base = x[pos, 2:10].copy()
    if mode == "gear":
        levels = rng.choice([0, 1, 2, 3, 4, 5], size=len(pos))
        base[:, 0] = np.clip(levels * 40 + rng.integers(0, 16, size=len(pos)), 0, 255)
        base[:, 1] = np.clip(255 - base[:, 0] + rng.integers(-8, 9, size=len(pos)), 0, 255)
    else:
        rpm = rng.integers(0, 8000, size=len(pos))
        base[:, 0] = (rpm // 32) % 256
        base[:, 1] = (rpm // 4) % 256
        base[:, 2] = np.clip(base[:, 2] + rng.integers(-50, 51, size=len(pos)), 0, 255)
    x[pos, 2:10] = base
    x[pos, 10] = np.maximum(x[pos, 10], 1e-5)
    return len(pos), pos


# ---------------------------------------------------------------------------
# Constrained permutations (placebo stream only; aligned stream untouched)
# ---------------------------------------------------------------------------

def permute_gear_data1(prng, d0, d1):
    """Off-bin rotation. Returns (new_d1, offbin_count, n)."""
    n = len(d1)
    bins = d0.astype(np.int64) // 40
    order = np.lexsort((prng.random(n), bins))  # grouped by bin, random inside
    m = int(np.bincount(bins, minlength=6).max())
    shift = m if 2 * m <= n else n - m
    if shift == 0 and n > 1:
        shift = 1
    src = order[(np.arange(n) + shift) % n]
    new_d1 = np.empty_like(d1)
    new_d1[order] = d1[src]
    offbin = int((bins[order] != bins[src]).sum())
    return new_d1, offbin, n


def permute_rpm_data1(prng, d0, d1, cap=1000):
    """Rejection resampling against the mod-256 coupling. Returns
    (new_d1, residual_matches, attempts, n)."""
    n = len(d1)
    d0i = d0.astype(np.int64)
    d1i = d1.astype(np.int64)
    best_perm, best_viol, attempts = None, n + 1, 0
    for attempt in range(1, cap + 1):
        perm = prng.permutation(n)
        viol = int((((d1i[perm] - 8 * d0i) % 256) < 8).sum())
        if viol < best_viol:
            best_perm, best_viol = perm, viol
        attempts = attempt
        if viol == 0:
            break
    return d1[best_perm], best_viol, attempts, n


def rf_features64(x):
    """Float64 twin of train_generator_extension_rf.rf_features (M4 gate)."""
    x = x.astype(np.float64)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    minv = x.min(axis=1)
    maxv = x.max(axis=1)
    delta = x[:, -1, :] - x[:, 0, :]
    return np.concatenate([mean, std, minv, maxv, delta], axis=1)


def main():
    t0 = time.time()
    for path in [OUT_NPZ, CHECKS_CSV, LOG_JSON]:
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    if not ALIGNED_REF.exists():
        raise SystemExit(f"missing aligned reference pool: {ALIGNED_REF}")

    rng = np.random.default_rng(SEED)
    prng = np.random.default_rng(PLACEBO_SEED)

    train = np.load(lc.WINDOWS / "train_windows.npz", allow_pickle=True)
    normal_idx = np.where(train["y_binary"] == 0)[0]
    if len(normal_idx) == 0:
        raise RuntimeError("no train normal windows available")

    xs_aligned, xs_placebo = [], []
    y_binary, y_attack, synthetic_type, injection_counts = [], [], [], []
    positions_padded, position_counts = [], []

    gear_stats = {"offbin": 0, "frames": 0, "residual_le8": 0,
                  "aligned_scores": [], "placebo_scores": []}
    rpm_stats = {"residual_matches": 0, "frames": 0, "capped_windows": 0,
                 "attempts_total": 0, "aligned_scores": [], "placebo_scores": []}

    specs = [
        ("DoS", ATTACK_IDS["DoS"], lambda r, x: synth_dos(r, x)),
        ("Fuzzy", ATTACK_IDS["Fuzzy"], lambda r, x: synth_fuzzy(r, x)),
        ("Gear", ATTACK_IDS["Gear"], lambda r, x: synth_spoof(r, x, 0x43F, "gear")),
        ("RPM", ATTACK_IDS["RPM"], lambda r, x: synth_spoof(r, x, 0x316, "rpm")),
    ]
    for name, label, fn in specs:
        base = rng.choice(normal_idx, size=PER_ATTACK, replace=True)
        x_part = train["x"][base].astype(np.float32).copy()
        counts = []
        part_positions = np.full((PER_ATTACK, MAX_POSITIONS), -1, dtype=np.int16)
        part_pos_counts = np.zeros(PER_ATTACK, dtype=np.int16)
        pos_list = []
        for i in range(PER_ATTACK):
            count, pos = fn(rng, x_part[i])
            counts.append(count)
            part_positions[i, :len(pos)] = pos
            part_pos_counts[i] = len(pos)
            pos_list.append(pos)

        placebo_part = x_part.copy()
        if name in ("Gear", "RPM"):
            for i in range(PER_ATTACK):
                pos = pos_list[i]
                d0 = placebo_part[i][pos, DATA0]
                d1 = placebo_part[i][pos, DATA1]
                if name == "Gear":
                    new_d1, offbin, n = permute_gear_data1(prng, d0, d1)
                    gear_stats["offbin"] += offbin
                    gear_stats["frames"] += n
                    res = np.abs(255.0 - d0 - new_d1)
                    gear_stats["residual_le8"] += int((res <= 8).sum())
                    gear_stats["placebo_scores"].append(float(res.mean()))
                    gear_stats["aligned_scores"].append(
                        float(np.abs(255.0 - d0 - d1).mean()))
                else:
                    new_d1, viol, attempts, n = permute_rpm_data1(prng, d0, d1)
                    rpm_stats["residual_matches"] += viol
                    rpm_stats["frames"] += n
                    rpm_stats["attempts_total"] += attempts
                    if viol > 0:
                        rpm_stats["capped_windows"] += 1
                    d0i = d0.astype(np.int64)
                    match_aligned = float(
                        (((d1.astype(np.int64) - 8 * d0i) % 256) < 8).mean())
                    match_placebo = float(
                        (((new_d1.astype(np.int64) - 8 * d0i) % 256) < 8).mean())
                    rpm_stats["aligned_scores"].append(1.0 - match_aligned)
                    rpm_stats["placebo_scores"].append(1.0 - match_placebo)
                placebo_part[i][pos, DATA1] = new_d1

        xs_aligned.append(x_part)
        xs_placebo.append(placebo_part)
        y_binary.extend([1] * PER_ATTACK)
        y_attack.extend([label] * PER_ATTACK)
        synthetic_type.extend([name] * PER_ATTACK)
        injection_counts.extend(counts)
        positions_padded.append(part_positions)
        position_counts.append(part_pos_counts)
        print(f"{name}: generated aligned+placebo parts "
              f"({time.time()-t0:.0f}s)", flush=True)

    x_aligned = np.concatenate(xs_aligned, axis=0).astype(np.float32)
    x_placebo = np.concatenate(xs_placebo, axis=0).astype(np.float32)
    y_binary = np.asarray(y_binary, dtype=np.int8)
    y_attack = np.asarray(y_attack, dtype=np.int8)
    synthetic_type = np.asarray(synthetic_type, dtype=object)
    injection_counts = np.asarray(injection_counts, dtype=np.int16)
    positions_padded = np.concatenate(positions_padded, axis=0)
    position_counts = np.concatenate(position_counts, axis=0)
    order = rng.permutation(len(y_binary))

    # ---- aligned-twin identity assertion (PREREG §2) ----------------------
    ref = np.load(ALIGNED_REF, allow_pickle=True)
    checks = []
    twin_ok = (
        np.array_equal(x_aligned[order], ref["x"])
        and np.array_equal(y_binary[order], ref["y_binary"])
        and np.array_equal(y_attack[order], ref["y_attack_type"])
        and list(synthetic_type[order]) == list(ref["synthetic_type"])
        and np.array_equal(injection_counts[order], ref["injection_count"])
    )
    checks.append({"check": "aligned_twin_identity", "gate": "abort",
                   "value": float(twin_ok), "threshold": 1.0,
                   "passed": bool(twin_ok)})
    if not twin_ok:
        lc.write_csv(CHECKS_CSV, checks)
        raise SystemExit(
            "aligned twin does not reproduce rule_based_windows.npz — "
            "PREREG §2 fallback (direct transform of the existing pool via "
            "bitwise base-window recovery) must be used instead")
    print(f"aligned twin identity assertion passed ({time.time()-t0:.0f}s)",
          flush=True)

    x_aligned_o = x_aligned[order]
    x_placebo_o = x_placebo[order]
    y_attack_o = y_attack[order]
    positions_o = positions_padded[order]
    pos_count_o = position_counts[order]

    # ---- M1 marginal identity (assert) ------------------------------------
    spoof_mask = (y_attack_o == 3) | (y_attack_o == 4)
    other_mask = ~spoof_mask
    other_cols = [c for c in range(11) if c != DATA1]
    m1_sorted = np.array_equal(
        np.sort(x_aligned_o[spoof_mask][:, :, DATA1], axis=1),
        np.sort(x_placebo_o[spoof_mask][:, :, DATA1], axis=1))
    m1_other = np.array_equal(
        x_aligned_o[spoof_mask][:, :, other_cols],
        x_placebo_o[spoof_mask][:, :, other_cols])
    m1_dosfuzzy = np.array_equal(x_aligned_o[other_mask], x_placebo_o[other_mask])
    checks += [
        {"check": "M1_data1_sorted_multiset_equal", "gate": "abort",
         "value": float(m1_sorted), "threshold": 1.0, "passed": bool(m1_sorted)},
        {"check": "M1_non_data1_channels_equal", "gate": "abort",
         "value": float(m1_other), "threshold": 1.0, "passed": bool(m1_other)},
        {"check": "M1_dos_fuzzy_bit_identical", "gate": "abort",
         "value": float(m1_dosfuzzy), "threshold": 1.0, "passed": bool(m1_dosfuzzy)},
    ]

    # ---- M2 residual relational structure (gate) --------------------------
    gear_residual_frac = gear_stats["residual_le8"] / max(gear_stats["frames"], 1)
    rpm_residual_beyond_cap = rpm_stats["residual_matches"]  # cap remainder only
    checks += [
        {"check": "M2_gear_residual_le8_fraction", "gate": "halt_stage2",
         "value": gear_residual_frac, "threshold": 0.01,
         "passed": bool(gear_residual_frac < 0.01)},
        {"check": "M2_rpm_residual_matches", "gate": "halt_stage2",
         "value": float(rpm_residual_beyond_cap),
         "threshold": 0.0,
         "passed": bool(rpm_stats["capped_windows"] == 0
                        or rpm_residual_beyond_cap == 0)},
        {"check": "M2_gear_offbin_fraction", "gate": "report",
         "value": gear_stats["offbin"] / max(gear_stats["frames"], 1),
         "threshold": "", "passed": ""},
        {"check": "M2_rpm_capped_windows", "gate": "report",
         "value": float(rpm_stats["capped_windows"]), "threshold": "",
         "passed": ""},
    ]

    # ---- M4 matching integrity via float64 RF features (gate) -------------
    DELTA_DATA1_COL = 4 * 11 + DATA1  # delta block, data1 channel
    exception = np.zeros(spoof_mask.sum(), dtype=bool)
    spoof_pos = positions_o[spoof_mask]
    spoof_cnt = pos_count_o[spoof_mask]
    for i in range(len(exception)):
        p = spoof_pos[i, :spoof_cnt[i]]
        exception[i] = (p[0] == 0) or (p[-1] == WINDOW_SIZE - 1)
    xa = x_aligned_o[spoof_mask]
    xp = x_placebo_o[spoof_mask]
    max_nonexc, max_exc_other, max_exc_delta = 0.0, 0.0, 0.0
    chunk = 20_000
    for s in range(0, len(xa), chunk):
        fa = rf_features64(xa[s:s + chunk])
        fp = rf_features64(xp[s:s + chunk])
        diff = np.abs(fa - fp)
        exc = exception[s:s + chunk]
        if (~exc).any():
            max_nonexc = max(max_nonexc, float(diff[~exc].max()))
        if exc.any():
            d_exc = diff[exc]
            max_exc_delta = max(max_exc_delta,
                                float(d_exc[:, DELTA_DATA1_COL].max()))
            other = np.delete(d_exc, DELTA_DATA1_COL, axis=1)
            max_exc_other = max(max_exc_other, float(other.max()))
    checks += [
        {"check": "M4_nonexception_rf_feature_linf", "gate": "halt_stage2",
         "value": max_nonexc, "threshold": 1e-9,
         "passed": bool(max_nonexc <= 1e-9)},
        {"check": "M4_exception_non_data1delta_linf", "gate": "halt_stage2",
         "value": max_exc_other, "threshold": 1e-9,
         "passed": bool(max_exc_other <= 1e-9)},
        {"check": "M4_exception_window_fraction", "gate": "report",
         "value": float(exception.mean()), "threshold": "", "passed": ""},
        {"check": "M4_exception_data1delta_max_abs", "gate": "report",
         "value": max_exc_delta, "threshold": "", "passed": ""},
    ]

    # ---- M3 Leg B relation-aware discriminability (descriptive) -----------
    gear_auc = float(roc_auc_score(
        [0] * len(gear_stats["aligned_scores"]) + [1] * len(gear_stats["placebo_scores"]),
        gear_stats["aligned_scores"] + gear_stats["placebo_scores"]))
    rpm_auc = float(roc_auc_score(
        [0] * len(rpm_stats["aligned_scores"]) + [1] * len(rpm_stats["placebo_scores"]),
        rpm_stats["aligned_scores"] + rpm_stats["placebo_scores"]))
    checks += [
        {"check": "M3legB_gear_relation_auc", "gate": "report",
         "value": gear_auc, "threshold": "", "passed": ""},
        {"check": "M3legB_rpm_relation_auc", "gate": "report",
         "value": rpm_auc, "threshold": "", "passed": ""},
    ]

    failed = [c["check"] for c in checks
              if c["passed"] is not True and c["gate"] in ("abort", "halt_stage2")]
    lc.write_csv(CHECKS_CSV, checks)
    if failed:
        raise SystemExit(f"gate violation(s): {failed} — stage-2 is halted "
                         f"per PREREG §2 (checks written to {CHECKS_CSV})")

    np.savez_compressed(
        OUT_NPZ,
        x=x_placebo_o,
        y_binary=y_binary[order],
        y_attack_type=y_attack_o,
        synthetic_type=synthetic_type[order],
        injection_count=injection_counts[order],
        positions=positions_o,
        position_count=pos_count_o,
        feature_names=np.asarray(lc.FEATURE_NAMES, dtype=object),
        window_size=np.asarray(WINDOW_SIZE, dtype=np.int32),
        stride=np.asarray(32, dtype=np.int32),
        seed=np.asarray(SEED, dtype=np.int32),
        placebo_seed=np.asarray(PLACEBO_SEED, dtype=np.int32),
    )

    lc.LOGS.mkdir(parents=True, exist_ok=True)
    LOG_JSON.write_text(json.dumps({
        "output": str(OUT_NPZ), "aligned_reference": str(ALIGNED_REF),
        "seed": SEED, "placebo_seed": PLACEBO_SEED,
        "total_windows": int(len(y_binary)),
        "gear": {k: v for k, v in gear_stats.items()
                 if k not in ("aligned_scores", "placebo_scores")},
        "rpm": {k: v for k, v in rpm_stats.items()
                if k not in ("aligned_scores", "placebo_scores")},
        "m3_legB_auc": {"gear": gear_auc, "rpm": rpm_auc},
        "checks": checks,
        "elapsed_seconds": time.time() - t0,
    }, indent=2, default=float))
    print(f"done: {OUT_NPZ} ({time.time()-t0:.0f}s); all gates passed", flush=True)


if __name__ == "__main__":
    main()
