#!/usr/bin/env python3
"""E10a source-only crossover scorer
(experiments/e10_background_grammar_crossover/PREREG.md §3-§5).

Stage 1 (--stage 1): scores the CONTROL arms (real_only, over_0p30) on the
paired clean/aligned/placebo transforms, computes the per-seed
probability-of-superiority A, the paired null contrast D0 = A(over) -
A(real_only), and the frozen band delta = max(0.02, 2 x SD_seeds(D0)) with the
0.10 cap (-> SX-cal). Writes e10a_v1_band_calibration.csv and the transform
hash. rule/placebo arms are NOT scored in this stage (PREREG §4).

Stage 2 (--stage 2): requires the stage-1 output and BAND_AMENDMENT freeze;
re-generates the transforms deterministically, asserts the recorded hash,
scores rule_0p30 / placebo_0p30 (plus the matchedsteps sensitivity pair),
computes D2/D1, the E0 precondition, the saturation/degeneracy gates, and the
pre-committed verdict (E0/S+/S+asym/S0/S-/SX/SX-cal). Writes
e10a_v1_crossover_by_seed.csv and e10a_v1_crossover_summary.csv.

Evaluation transforms reuse the TRAINING-pool grammar verbatim (synth_spoof
value rules; declared on-grammar probe) with new eval seeds 20260721-23 on the
E8 frame-disjoint blocks (membership asserted against
evaluation_realization_blocks_e8_v2.csv). Scores are pre-softmax logit margins
s(x) = max_{c>=1} z_c - z_0. CNN only (RF is the M4 pool-level negative
control, not a crossover detector).
"""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, t as student_t

import lib_common as lc
from evaluate_evaluation_realization import (
    frame_disjoint_indices,
    partition_blocks,
)
from train_generator_extension_cnn import CNN1D, MODELS

TAG = "e10a_v1"
E10_EVAL_SEEDS = {"block_01": 20260721, "block_02": 20260722, "block_03": 20260723}
BLOCK_SIZE = 12_000
GEAR_HALF = 6_000
TIE_EPS = 1e-6
MANIFEST = lc.TABLES / "evaluation_realization_blocks_e8_v2.csv"
CHECKS_CSV = lc.TABLES / "e10a_v1_manipulation_checks.csv"
BAND_CSV = lc.TABLES / f"{TAG}_band_calibration.csv"
BY_SEED_CSV = lc.TABLES / f"{TAG}_crossover_by_seed.csv"
SUMMARY_CSV = lc.TABLES / f"{TAG}_crossover_summary.csv"
STAGE1_ARMS = ["real_only", "over_0p30"]
STAGE2_ARMS = ["rule_0p30", "placebo_0p30"]
MATCHEDSTEPS_ARMS = [("rule_0p30", "matchedsteps"), ("placebo_0p30", "matchedsteps")]
DELTA_FLOOR = 0.02
DELTA_CAP = 0.10
DATA0, DATA1 = 2, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# transforms — training-pool grammar verbatim (PREREG §3)
# ---------------------------------------------------------------------------

def gear_offbin_sigma(prng, d0):
    """sigma[i] = source frame whose data1 value frame i receives (off-bin)."""
    n = len(d0)
    bins = d0.astype(np.int64) // 40
    order = np.lexsort((prng.random(n), bins))
    m = int(np.bincount(bins, minlength=6).max())
    shift = m if 2 * m <= n else n - m
    if shift == 0 and n > 1:
        shift = 1
    src = order[(np.arange(n) + shift) % n]
    sigma = np.empty(n, dtype=np.int64)
    sigma[order] = src
    return sigma


def rpm_reject_sigma(prng, d0, d1, cap=1000):
    n = len(d1)
    d0i = d0.astype(np.int64)
    d1i = d1.astype(np.int64)
    best, best_viol = None, n + 1
    for _ in range(cap):
        perm = prng.permutation(n)
        viol = int((((d1i[perm] - 8 * d0i) % 256) < 8).sum())
        if viol < best_viol:
            best, best_viol = perm, viol
        if viol == 0:
            break
    return best, best_viol


def build_block_transforms(x_test, block_indices, eval_seed):
    """Returns dict of float32 arrays: clean/aligned/placebo (12k) and
    off-grammar aligned/placebo pairs for the Gear half (6k each)."""
    rng = np.random.default_rng(eval_seed)
    prng = np.random.default_rng(eval_seed + 500_000)
    assigned = rng.permutation(np.asarray(block_indices))
    clean = x_test[assigned].astype(np.float32, copy=True)
    aligned = clean.copy()
    placebo = clean.copy()
    off_aligned = clean[:GEAR_HALF].copy()
    off_placebo = clean[:GEAR_HALF].copy()

    for i in range(BLOCK_SIZE):
        burst = int(rng.integers(32, 91))
        start = int(rng.integers(0, 128 - burst + 1))
        step = int(rng.choice([1, 2, 3]))
        pos = np.arange(start, start + burst, step)
        n = len(pos)
        gear = i < GEAR_HALF
        can_id = 0x43F if gear else 0x316
        for arr in (aligned, placebo):
            arr[i][pos, 0] = can_id
            arr[i][pos, 1] = 8
            arr[i][pos, 10] = np.maximum(arr[i][pos, 10], 1e-5)
        if gear:
            levels = rng.choice([0, 1, 2, 3, 4, 5], size=n)
            n0 = rng.integers(0, 16, size=n)
            n1 = rng.integers(-8, 9, size=n)
            d0 = np.clip(levels * 40 + n0, 0, 255).astype(np.float32)
            d1 = np.clip(255 - d0 + n1, 0, 255).astype(np.float32)
            sigma = gear_offbin_sigma(prng, d0)
            aligned[i][pos, DATA0] = d0
            aligned[i][pos, DATA1] = d1
            placebo[i][pos, DATA0] = d0
            placebo[i][pos, DATA1] = d1[sigma]
            # off-grammar shifted-marginal variant, same relation & same sigma
            d0s = np.clip(20 + levels * 40 + n0, 0, 255).astype(np.float32)
            d1s = np.clip(255 - d0s + n1, 0, 255).astype(np.float32)
            for arr in (off_aligned, off_placebo):
                arr[i][pos, 0] = can_id
                arr[i][pos, 1] = 8
                arr[i][pos, 10] = np.maximum(arr[i][pos, 10], 1e-5)
                arr[i][pos, DATA0] = d0s
            off_aligned[i][pos, DATA1] = d1s
            off_placebo[i][pos, DATA1] = d1s[sigma]
        else:
            rpm = rng.integers(0, 8000, size=n)
            jit = rng.integers(-50, 51, size=n)
            d0 = ((rpm // 32) % 256).astype(np.float32)
            d1 = ((rpm // 4) % 256).astype(np.float32)
            sigma, _viol = rpm_reject_sigma(prng, d0, d1)
            for arr in (aligned, placebo):
                arr[i][pos, DATA0] = d0
                arr[i][pos, 4] = np.clip(arr[i][pos, 4] + jit, 0, 255)
            aligned[i][pos, DATA1] = d1
            placebo[i][pos, DATA1] = d1[sigma]
    return {"assigned": assigned, "clean": clean, "aligned": aligned,
            "placebo": placebo, "off_aligned": off_aligned,
            "off_placebo": off_placebo}


def transforms_hash(blocks):
    h = hashlib.sha256()
    for name in sorted(blocks):
        tr = blocks[name]
        for key in ["assigned", "clean", "aligned", "placebo",
                    "off_aligned", "off_placebo"]:
            h.update(np.ascontiguousarray(tr[key]).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def logit_margins(model, x_std, batch=4096):
    out = np.empty(len(x_std), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x_std), batch):
            b = torch.from_numpy(
                x_std[i:i + batch].transpose(0, 2, 1).copy()).to(DEVICE)
            z = model(b)
            m = z[:, 1:].max(dim=1).values - z[:, 0]
            out[i:i + len(m)] = m.cpu().numpy()
    return out


def prob_superiority(a, b):
    """P(a > b) + 0.5 P(|a-b| <= TIE_EPS)."""
    diff = a - b
    ties = np.abs(diff) <= TIE_EPS
    wins = (diff > 0) & ~ties
    return float(wins.mean() + 0.5 * ties.mean())


def t_interval(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        return mean, sd, mean - half, mean + half
    return mean, sd, mean, mean  # degenerate point interval (PREREG §5)


def score_arm(setting, model_tag, block_transforms, std_cache, mean, std,
              rows_by_seed):
    """Returns per-seed dict with A, e0_stat, gates, descriptives."""
    tag = f"_{model_tag}" if model_tag else ""
    per_seed = {}
    for seed in lc.SEEDS:
        path = MODELS / f"cnn_{setting}{tag}_seed{seed}.pt"
        if not path.exists():
            raise SystemExit(f"missing checkpoint: {path}")
        model = CNN1D()
        model.load_state_dict(torch.load(path, map_location=DEVICE,
                                         weights_only=True))
        model.to(DEVICE).eval()
        block_A, block_e0, block_mad_eff, block_off_A = [], [], [], []
        gear_A, rpm_A = [], []
        clean_all, pair_diffs = [], []
        for bname in sorted(block_transforms):
            cache = std_cache[bname]
            s_clean = logit_margins(model, cache["clean"])
            s_al = logit_margins(model, cache["aligned"])
            s_pl = logit_margins(model, cache["placebo"])
            da = s_al - s_clean
            dp = s_pl - s_clean
            block_A.append(prob_superiority(da, dp))
            gear_A.append(prob_superiority(da[:GEAR_HALF], dp[:GEAR_HALF]))
            rpm_A.append(prob_superiority(da[GEAR_HALF:], dp[GEAR_HALF:]))
            ties0 = np.abs(da) <= TIE_EPS
            block_e0.append(float(((da > 0) & ~ties0).mean() + 0.5 * ties0.mean()))
            mad = float(np.median(np.abs(s_clean - np.median(s_clean))))
            block_mad_eff.append(
                float((da.mean() - dp.mean()) / mad) if mad > 0 else float("nan"))
            s_offa = logit_margins(model, cache["off_aligned"])
            s_offp = logit_margins(model, cache["off_placebo"])
            off_da = s_offa - s_clean[:GEAR_HALF]
            off_dp = s_offp - s_clean[:GEAR_HALF]
            block_off_A.append(prob_superiority(off_da, off_dp))
            clean_all.append(s_clean)
            pair_diffs.append(da - dp)
            rows_by_seed.append({
                "arm": setting + (f"_{model_tag}" if model_tag else ""),
                "seed": seed, "block": bname,
                "A": block_A[-1], "A_gear": gear_A[-1], "A_rpm": rpm_A[-1],
                "e0_stat": block_e0[-1], "off_grammar_A": block_off_A[-1],
                "mad_standardized_effect": block_mad_eff[-1],
            })
        clean_all = np.concatenate(clean_all)
        pair_diffs = np.concatenate(pair_diffs)
        gate_max_tie = float((np.abs(clean_all - clean_all.max()) <= TIE_EPS).mean())
        gate_spread = float(np.percentile(clean_all, 90)
                            - np.percentile(clean_all, 10))
        gate_pair_degen = float((np.abs(pair_diffs) <= TIE_EPS).mean())
        gates_pass = (gate_max_tie < 0.01) and (gate_spread >= 1e-3) \
            and (gate_pair_degen < 0.5)
        per_seed[seed] = {
            "A": float(np.mean(block_A)),
            "e0_stat": float(np.mean(block_e0)),
            "off_grammar_A": float(np.mean(block_off_A)),
            "A_gear": float(np.mean(gear_A)),
            "A_rpm": float(np.mean(rpm_A)),
            "mad_standardized_effect": float(np.nanmean(block_mad_eff)),
            "gate_clean_max_tie_frac": gate_max_tie,
            "gate_clean_q90_q10": gate_spread,
            "gate_pair_degeneracy_frac": gate_pair_degen,
            "gates_pass": gates_pass,
        }
        rows_by_seed.append({
            "arm": setting + (f"_{model_tag}" if model_tag else ""),
            "seed": seed, "block": "seed_mean", **per_seed[seed]})
        print(f"{setting}{tag} seed={seed}: A={per_seed[seed]['A']:.4f} "
              f"(gear {per_seed[seed]['A_gear']:.4f} / rpm "
              f"{per_seed[seed]['A_rpm']:.4f}), gates_pass={gates_pass}",
              flush=True)
    return per_seed


def realized_steps(setting, seed, model_tag=""):
    tag = f"_{model_tag}" if model_tag else ""
    path = lc.LOGS / f"train_cnn_{setting}{tag}_seed{seed}.log"
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text()).get("optimizer_steps", "")
    except (json.JSONDecodeError, OSError):
        return ""


# ---------------------------------------------------------------------------
# blocks + shared setup
# ---------------------------------------------------------------------------

def setup():
    test = lc.load_npz("test")
    eligible = frame_disjoint_indices(
        test["y_binary"], test["source_file"], test["segment_id"],
        test["start_index"], test["end_index"])
    blocks = partition_blocks(eligible)
    manifest = pd.read_csv(MANIFEST)
    for bname, indices in blocks.items():
        expected = set(manifest[manifest["block_id"] == bname]
                       ["test_window_index"].astype(int))
        if expected != set(int(i) for i in indices):
            raise SystemExit(f"block membership mismatch vs E8 manifest: {bname}")
    print(f"E8 block membership verified against {MANIFEST.name}", flush=True)

    train = lc.load_npz("train")
    mean, std = lc.fit_standardizer(train["x"])
    del train
    block_transforms = {
        bname: build_block_transforms(test["x"], blocks[bname],
                                      E10_EVAL_SEEDS[bname])
        for bname in sorted(blocks)
    }
    thash = transforms_hash(block_transforms)
    std_cache = {}
    for bname, tr in block_transforms.items():
        std_cache[bname] = {
            key: lc.standardize(tr[key], mean, std)
            for key in ["clean", "aligned", "placebo", "off_aligned",
                        "off_placebo"]
        }
    return block_transforms, std_cache, mean, std, thash


def check_manipulation_gates():
    if not CHECKS_CSV.exists():
        raise SystemExit(f"missing manipulation checks: {CHECKS_CSV} — run "
                         "generate_rule_placebo_twin.py first")
    checks = pd.read_csv(CHECKS_CSV)
    gated = checks[checks["gate"].isin(["abort", "halt_stage2"])]
    failed = gated[gated["passed"].astype(str).str.lower() != "true"]
    if len(failed):
        raise SystemExit(
            f"manipulation gate(s) failed: {list(failed['check'])} — stage-2 "
            "is halted per PREREG §2")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage1():
    t0 = time.time()
    if BAND_CSV.exists():
        raise SystemExit(f"refusing to overwrite existing output: {BAND_CSV}")
    block_transforms, std_cache, mean, std, thash = setup()
    rows_by_seed = []
    arm_A = {}
    for setting in STAGE1_ARMS:
        arm_A[setting] = score_arm(setting, "", block_transforms, std_cache,
                                   mean, std, rows_by_seed)
    d0 = [arm_A["over_0p30"][s]["A"] - arm_A["real_only"][s]["A"]
          for s in lc.SEEDS]
    d0_mean, d0_sd, d0_lo, d0_hi = t_interval(d0)
    two_sd = 2.0 * d0_sd
    sx_cal = two_sd > DELTA_CAP
    delta = max(DELTA_FLOOR, two_sd)
    n = len(d0)
    if d0_sd > 0:
        sd_lo = math.sqrt((n - 1) * d0_sd ** 2 / chi2.ppf(0.975, n - 1))
        sd_hi = math.sqrt((n - 1) * d0_sd ** 2 / chi2.ppf(0.025, n - 1))
    else:
        sd_lo = sd_hi = 0.0

    rows = list(rows_by_seed)
    rows.append({"arm": "D0_over_minus_real", "seed": "", "block": "summary",
                 "A": d0_mean, "d0_sd": d0_sd, "d0_ci_lo": d0_lo,
                 "d0_ci_hi": d0_hi, "d0_values": ";".join(f"{v:.6f}" for v in d0),
                 "sd_chi2_ci_lo": sd_lo, "sd_chi2_ci_hi": sd_hi,
                 "two_sd": two_sd, "delta": delta, "delta_floor": DELTA_FLOOR,
                 "delta_cap": DELTA_CAP, "sx_cal": sx_cal,
                 "transforms_sha256": thash})
    lc.write_csv(BAND_CSV, rows)
    lc.LOGS.mkdir(parents=True, exist_ok=True)
    (lc.LOGS / f"{TAG}_stage1.log").write_text(json.dumps({
        "stage": 1, "arms": STAGE1_ARMS, "d0": d0, "d0_sd": d0_sd,
        "delta": delta, "sx_cal": sx_cal, "transforms_sha256": thash,
        "elapsed_seconds": time.time() - t0}, indent=2))
    print(f"stage 1 done: delta={delta:.4f} (2xSD={two_sd:.4f}, "
          f"sx_cal={sx_cal}); freeze via BAND_AMENDMENT.md before stage 2 "
          f"({time.time()-t0:.0f}s)", flush=True)


def band_verdict(d2_lo, d2_hi, delta, comp_lo):
    """PREREG §5 boundary conventions: S+/S- require strict exceedance, S0
    requires the interval strictly inside the band; touching or crossing a
    boundary (including degenerate point intervals on it) is SX."""
    if d2_lo > delta:
        return "S+" if comp_lo > 0 else "S+asym"
    if d2_hi < -delta:
        return "S-"
    if d2_lo > -delta and d2_hi < delta:
        return "S0"
    return "SX"


def stage2():
    t0 = time.time()
    for path in [BY_SEED_CSV, SUMMARY_CSV]:
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    if not BAND_CSV.exists():
        raise SystemExit("stage-1 band calibration missing — run --stage 1 "
                         "and commit BAND_AMENDMENT.md first")
    check_manipulation_gates()
    band = pd.read_csv(BAND_CSV)
    summary_row = band[band["block"] == "summary"].iloc[0]
    delta = float(summary_row["delta"])
    sx_cal = str(summary_row["sx_cal"]).lower() == "true"
    recorded_hash = str(summary_row["transforms_sha256"])
    ctrl = band[band["block"] == "seed_mean"]
    a_over = {int(r["seed"]): float(r["A"])
              for _, r in ctrl[ctrl["arm"] == "over_0p30"].iterrows()}

    block_transforms, std_cache, mean, std, thash = setup()
    if thash != recorded_hash:
        raise SystemExit("transform hash mismatch vs stage 1 — aborting "
                         f"(stage1={recorded_hash[:16]}…, now={thash[:16]}…)")
    print("transform hash verified against stage 1", flush=True)

    rows_by_seed = []
    arm_A = {}
    for setting in STAGE2_ARMS:
        arm_A[setting] = score_arm(setting, "", block_transforms, std_cache,
                                   mean, std, rows_by_seed)
    ms_A = {}
    for setting, mtag in MATCHEDSTEPS_ARMS:
        ms_A[setting] = score_arm(setting, mtag, block_transforms, std_cache,
                                  mean, std, rows_by_seed)

    seeds = lc.SEEDS
    d2 = [arm_A["rule_0p30"][s]["A"] - arm_A["placebo_0p30"][s]["A"]
          for s in seeds]
    d2_mean, d2_sd, d2_lo, d2_hi = t_interval(d2)
    comp_rule = [arm_A["rule_0p30"][s]["A"] - 0.5 for s in seeds]
    cr_mean, cr_sd, cr_lo, cr_hi = t_interval(comp_rule)
    comp_placebo = [arm_A["placebo_0p30"][s]["A"] - 0.5 for s in seeds]
    cp_mean, cp_sd, cp_lo, cp_hi = t_interval(comp_placebo)
    e0_vals = [arm_A["rule_0p30"][s]["e0_stat"] for s in seeds]
    e0_mean, e0_sd, e0_lo, e0_hi = t_interval(e0_vals)
    e0_pass = e0_lo > 0.5
    d1 = [arm_A["rule_0p30"][s]["A"] - a_over[s] for s in seeds]
    d1_mean, d1_sd, d1_lo, d1_hi = t_interval(d1)
    d2_ms = [ms_A["rule_0p30"][s]["A"] - ms_A["placebo_0p30"][s]["A"]
             for s in seeds]
    ms_mean, ms_sd, ms_lo, ms_hi = t_interval(d2_ms)
    comp_rule_ms = [ms_A["rule_0p30"][s]["A"] - 0.5 for s in seeds]
    _, _, cr_ms_lo, _ = t_interval(comp_rule_ms)

    gate_failures = [
        f"{arm}:{s}" for arm, seed_map in
        list(arm_A.items()) + [(f"{a}_matchedsteps", m) for a, m in ms_A.items()]
        for s, v in seed_map.items() if not v["gates_pass"]]

    if not e0_pass:
        verdict_primary = "E0"
    elif sx_cal:
        verdict_primary = "SX-cal"
    else:
        verdict_primary = band_verdict(d2_lo, d2_hi, delta, cr_lo)
    verdict_ms = band_verdict(ms_lo, ms_hi, delta, cr_ms_lo)
    budget_sensitive = verdict_ms != verdict_primary
    verdict_final = ("SX" if budget_sensitive and verdict_primary not in
                     ("E0", "SX-cal") else verdict_primary)

    summary = [{
        "tag": TAG, "delta": delta, "sx_cal": sx_cal, "e0_pass": e0_pass,
        "e0_mean": e0_mean, "e0_ci_lo": e0_lo, "e0_ci_hi": e0_hi,
        "d2_mean": d2_mean, "d2_sd": d2_sd, "d2_ci_lo": d2_lo,
        "d2_ci_hi": d2_hi,
        "d2_values": ";".join(f"{v:.6f}" for v in d2),
        "a_rule_minus_half_ci_lo": cr_lo, "a_rule_minus_half_ci_hi": cr_hi,
        "a_placebo_minus_half_ci_lo": cp_lo,
        "a_placebo_minus_half_ci_hi": cp_hi,
        "d1_mean": d1_mean, "d1_ci_lo": d1_lo, "d1_ci_hi": d1_hi,
        "d2_matchedsteps_mean": ms_mean, "d2_matchedsteps_ci_lo": ms_lo,
        "d2_matchedsteps_ci_hi": ms_hi,
        "verdict_primary": verdict_primary, "verdict_matchedsteps": verdict_ms,
        "budget_sensitive": budget_sensitive, "verdict_final": verdict_final,
        "gate_failures": ";".join(gate_failures),
        "estimand_d2": "marginal_matched_relational_structure_training_contrast",
        "estimand_d1": "structure_vs_duplication_did_contrast",
        "transforms_sha256": thash,
    }]
    for setting in STAGE2_ARMS + ["real_only", "over_0p30"]:
        for seed in seeds:
            summary.append({
                "tag": f"steps:{setting}", "seed": seed,
                "realized_optimizer_steps": realized_steps(setting, seed)})
    for setting, mtag in MATCHEDSTEPS_ARMS:
        for seed in seeds:
            summary.append({
                "tag": f"steps:{setting}_{mtag}", "seed": seed,
                "realized_optimizer_steps": realized_steps(setting, seed, mtag)})

    lc.write_csv(BY_SEED_CSV, rows_by_seed)
    lc.write_csv(SUMMARY_CSV, summary)
    (lc.LOGS / f"{TAG}_stage2.log").write_text(json.dumps({
        "stage": 2, "delta": delta, "d2": d2, "d1": d1, "d2_ms": d2_ms,
        "e0": e0_vals, "verdict_primary": verdict_primary,
        "verdict_matchedsteps": verdict_ms, "verdict_final": verdict_final,
        "gate_failures": gate_failures, "transforms_sha256": thash,
        "elapsed_seconds": time.time() - t0}, indent=2))
    print(f"stage 2 done: D2={d2_mean:.4f} [{d2_lo:.4f}, {d2_hi:.4f}] vs "
          f"band ±{delta:.4f} -> primary={verdict_primary}, "
          f"matchedsteps={verdict_ms}, FINAL={verdict_final} "
          f"({time.time()-t0:.0f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    torch.manual_seed(0)  # inference-only; no stochastic layers at eval
    if args.stage == 1:
        stage1()
    else:
        stage2()


if __name__ == "__main__":
    main()
