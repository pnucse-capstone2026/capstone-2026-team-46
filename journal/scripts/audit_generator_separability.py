#!/usr/bin/env python3
"""Protocol-valid, class-matched learned-generator audit.

The original audit mixed imbalanced real attack classes with balanced
synthetic classes and randomly split heavily overlapping real windows.  This
revision keeps the legacy helper API for old, frozen analyses, but its CLI
uses a new procedure:

* equal real/synthetic counts for every attack class;
* real train/test partitions made from non-crossing raw-frame blocks using
  ``source_file``, ``segment_id``, ``start_index``, and ``end_index``;
* one overall class-matched domain AUC plus four class-conditional AUCs;
* full ID/DLC/payload/timing and DLC/payload cross-field validity;
* saved-condition metadata checks and a separately labelled source-RF proxy
  for semantic condition fidelity.

There is only one real source capture per attack class in the Car-Hacking
train split, so a fully capture-held-out real-vs-real null domain is not
identifiable.  The audit reports this limitation instead of manufacturing a
random-overlap null.

Default inputs are the frozen rule pool plus NEW protocol-valid GAN/AR-LM
pools.  Outputs are NEW suffixed files and are never overwritten implicitly::

  results/tables/generator_separability_audit_protocol_valid.csv
  results/tables/generator_grammar_audit_protocol_valid.csv
  results/tables/generator_condition_fidelity_audit_protocol_valid.csv
  results/logs/audit_generator_separability_protocol_valid.log
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import lib_common as lc
from generator_protocol import (EXPECTED_CLASS_NAMES,
                                summarize_condition_metadata,
                                summarize_protocol_validity)

SEED = 42
DEFAULT_BLOCK_FRAMES = 4096
DEFAULT_MAX_PER_CLASS = 12_500
DEFAULT_POOLS = ["rule", "ganvalid", "arlmvalid"]


def rf_features(x):
    """55-dimensional summary used by the frozen WISA RF audits."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    minv = x.min(axis=1)
    maxv = x.max(axis=1)
    delta = x[:, -1, :] - x[:, 0, :]
    return np.concatenate([mean, std, minv, maxv, delta], axis=1).astype(np.float32)


def discriminator_auc(real_x, syn_x):
    """Legacy random-split helper retained for old scripts only.

    New audits must use :func:`grouped_class_matched_domain_audit` below.
    """
    rng = np.random.default_rng(SEED)
    n = min(50_000, len(real_x), len(syn_x))
    real_idx = rng.choice(len(real_x), size=n, replace=False)
    syn_idx = rng.choice(len(syn_x), size=n, replace=False)
    x = np.concatenate([rf_features(real_x[real_idx]), rf_features(syn_x[syn_idx])])
    y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=SEED)
    clf = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    clf.fit(x_train, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))


def js_divergence(p, q):
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def marginal_js(real_vals, syn_vals, bins=128):
    lo = min(real_vals.min(), syn_vals.min())
    hi = max(real_vals.max(), syn_vals.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(real_vals, bins=edges)
    q, _ = np.histogram(syn_vals, bins=edges)
    return js_divergence(p, q)


def grammar_row(name, x, vocab_normal, vocab_all, reference_delta_t_range=None):
    """Return support-profile and full frame-protocol measurements."""
    ids = x[:, :, 0].astype(np.int64).reshape(-1)
    dlc = x[:, :, 1].reshape(-1)
    payload = x[:, :, 2:10].reshape(-1)
    return {
        "pool": name,
        "windows": int(len(x)),
        "id_in_train_normal_vocab": float(np.isin(ids, vocab_normal).mean()),
        "id_in_train_all_vocab": float(np.isin(ids, vocab_all).mean()),
        "unique_ids": int(len(np.unique(ids))),
        "dlc_integer_frac": float((np.abs(dlc - np.rint(dlc)) < 1e-6).mean()),
        "payload_at_bounds_frac": float(((payload == 0) | (payload == 255)).mean()),
        "id_at_bounds_frac": float(((ids == 0) | (ids == 2047)).mean()),
        "delta_t_zero_frac": float((x[:, :, 10] == 0).mean()),
        "delta_t_median": float(np.median(x[:, :, 10])),
        **summarize_protocol_validity(
            x, reference_delta_t_range=reference_delta_t_range),
    }


def build_purged_real_partitions(train, block_frames=DEFAULT_BLOCK_FRAMES, seed=SEED):
    """Split each attack class by non-overlapping raw-frame blocks.

    Windows crossing a block boundary are discarded.  Consequently no raw
    frame can occur in both returned train and test partitions even though the
    source windows were originally extracted with stride 32.
    """
    y = np.asarray(train["y_attack_type"])
    required = {"source_file", "segment_id", "start_index", "end_index"}
    if not required.issubset(train.files):
        rng = np.random.default_rng(seed)
        parts, rows = {}, []
        for cls in EXPECTED_CLASS_NAMES:
            idx = np.flatnonzero(y == cls)
            idx = rng.permutation(idx)
            cut = max(1, min(len(idx) - 1, round(0.7 * len(idx))))
            parts[cls] = {"train": idx[:cut], "test": idx[cut:]}
            rows.append({"attack_type_id": cls, "groups": "",
                         "kept_windows": int(len(idx)), "discarded_cross_block": ""})
        return parts, {
            "split_protocol": "fallback random split; raw overlap could not be purged",
            "overlap_purged": False,
            "class_rows": rows,
        }

    rng = np.random.default_rng(seed)
    source = np.asarray(train["source_file"]).astype(str)
    segment = np.asarray(train["segment_id"])
    start = np.asarray(train["start_index"], dtype=np.int64)
    end = np.asarray(train["end_index"], dtype=np.int64)  # inclusive
    parts, rows = {}, []
    for cls in EXPECTED_CLASS_NAMES:
        idx = np.flatnonzero(y == cls)
        start_block = start[idx] // block_frames
        end_block = end[idx] // block_frames
        keep = start_block == end_block
        kept = idx[keep]
        groups = np.asarray([
            f"{source[i]}|{int(segment[i])}|{int(start[i] // block_frames)}"
            for i in kept
        ])
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise RuntimeError(f"class {cls} has fewer than two purged raw-frame blocks")
        unique_groups = rng.permutation(unique_groups)
        n_test_groups = max(1, min(len(unique_groups) - 1,
                                   round(0.3 * len(unique_groups))))
        test_groups = unique_groups[:n_test_groups]
        is_test = np.isin(groups, test_groups)
        parts[cls] = {"train": kept[~is_test], "test": kept[is_test]}
        rows.append({
            "attack_type_id": cls,
            "groups": int(len(unique_groups)),
            "train_groups": int(len(unique_groups) - n_test_groups),
            "test_groups": int(n_test_groups),
            "kept_windows": int(len(kept)),
            "discarded_cross_block": int((~keep).sum()),
            "train_windows": int((~is_test).sum()),
            "test_windows": int(is_test.sum()),
        })
    return parts, {
        "split_protocol": (
            f"class-stratified {block_frames}-raw-frame groups; crossing windows discarded; "
            "group-disjoint 70/30 split"),
        "overlap_purged": True,
        "block_frames": int(block_frames),
        "class_rows": rows,
    }


def _sample_domain_features(real_x, real_parts, syn_x, syn_y,
                            max_per_class=DEFAULT_MAX_PER_CLASS, seed=SEED):
    rng = np.random.default_rng(seed)
    wanted_train = round(max_per_class * 0.7)
    wanted_test = max_per_class - wanted_train
    syn_by_class = {c: rng.permutation(np.flatnonzero(syn_y == c))
                    for c in EXPECTED_CLASS_NAMES}
    train_per = min(
        wanted_train,
        *(len(real_parts[c]["train"]) for c in EXPECTED_CLASS_NAMES),
        *(max(1, round(0.7 * len(syn_by_class[c]))) for c in EXPECTED_CLASS_NAMES),
    )
    test_per = min(
        wanted_test,
        *(len(real_parts[c]["test"]) for c in EXPECTED_CLASS_NAMES),
        *(len(syn_by_class[c]) - train_per for c in EXPECTED_CLASS_NAMES),
    )
    if train_per < 2 or test_per < 2:
        raise RuntimeError("insufficient class-conditional windows for domain audit")

    features = {}
    for cls in EXPECTED_CLASS_NAMES:
        ridx_train = rng.choice(real_parts[cls]["train"], train_per, replace=False)
        ridx_test = rng.choice(real_parts[cls]["test"], test_per, replace=False)
        sidx = syn_by_class[cls]
        sidx_train = sidx[:train_per]
        sidx_test = sidx[train_per:train_per + test_per]
        features[cls] = {
            "real_train": rf_features(real_x[ridx_train]),
            "real_test": rf_features(real_x[ridx_test]),
            "syn_train": rf_features(syn_x[sidx_train]),
            "syn_test": rf_features(syn_x[sidx_test]),
            "real_marginal_idx": ridx_test,
            "syn_marginal_idx": sidx_test,
        }
    return features, int(train_per), int(test_per)


def _fit_domain_auc(real_train, syn_train, real_test, syn_test, seed):
    x_train = np.concatenate([real_train, syn_train])
    y_train = np.concatenate([
        np.zeros(len(real_train), dtype=np.int8),
        np.ones(len(syn_train), dtype=np.int8),
    ])
    x_test = np.concatenate([real_test, syn_test])
    y_test = np.concatenate([
        np.zeros(len(real_test), dtype=np.int8),
        np.ones(len(syn_test), dtype=np.int8),
    ])
    clf = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    clf.fit(x_train, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))


def grouped_class_matched_domain_audit(real_x, real_parts, syn_x, syn_y,
                                       max_per_class=DEFAULT_MAX_PER_CLASS,
                                       seed=SEED):
    """Domain AUC with class balance and raw-frame-disjoint real partitions."""
    features, train_per, test_per = _sample_domain_features(
        real_x, real_parts, syn_x, syn_y, max_per_class, seed)
    real_train = np.concatenate([features[c]["real_train"] for c in EXPECTED_CLASS_NAMES])
    syn_train = np.concatenate([features[c]["syn_train"] for c in EXPECTED_CLASS_NAMES])
    real_test = np.concatenate([features[c]["real_test"] for c in EXPECTED_CLASS_NAMES])
    syn_test = np.concatenate([features[c]["syn_test"] for c in EXPECTED_CLASS_NAMES])
    row = {
        "real_vs_synthetic_auc": _fit_domain_auc(
            real_train, syn_train, real_test, syn_test, seed),
        "class_mixture_matched": True,
        "real_raw_frame_overlap_purged": True,
        "train_windows_per_class_per_domain": train_per,
        "test_windows_per_class_per_domain": test_per,
        "real_vs_real_null_auc": "",
        "null_limitation": (
            "not identifiable capture-conditionally: one real train capture per attack class"),
    }
    for cls, name in EXPECTED_CLASS_NAMES.items():
        f = features[cls]
        row[f"auc_{name}"] = _fit_domain_auc(
            f["real_train"], f["syn_train"], f["real_test"], f["syn_test"],
            seed + cls)
    return row, features


def train_source_condition_classifier(real_x, real_y, real_parts,
                                      max_per_class=DEFAULT_MAX_PER_CLASS, seed=SEED):
    """Train a real-only exact-class RF and report its group-held-out anchor."""
    rng = np.random.default_rng(seed + 1000)
    n_train = min(round(max_per_class * 0.7),
                  *(len(real_parts[c]["train"]) for c in EXPECTED_CLASS_NAMES))
    n_test = min(max_per_class - round(max_per_class * 0.7),
                 *(len(real_parts[c]["test"]) for c in EXPECTED_CLASS_NAMES))
    train_idx = np.concatenate([
        rng.choice(real_parts[c]["train"], n_train, replace=False)
        for c in EXPECTED_CLASS_NAMES
    ])
    test_idx = np.concatenate([
        rng.choice(real_parts[c]["test"], n_test, replace=False)
        for c in EXPECTED_CLASS_NAMES
    ])
    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed + 1000)
    clf.fit(rf_features(real_x[train_idx]), real_y[train_idx])
    pred = clf.predict(rf_features(real_x[test_idx]))
    anchor = {
        "source_rf_grouped_test_balanced_accuracy": float(
            balanced_accuracy_score(real_y[test_idx], pred)),
        "source_rf_train_windows_per_class": int(n_train),
        "source_rf_test_windows_per_class": int(n_test),
    }
    for cls, name in EXPECTED_CLASS_NAMES.items():
        m = real_y[test_idx] == cls
        anchor[f"source_rf_grouped_test_recall_{name}"] = float((pred[m] == cls).mean())
    return clf, anchor


def condition_fidelity_row(pool, data, syn_x, syn_y, clf, anchor,
                           sample_per_class=5000, seed=SEED):
    """Metadata fidelity plus a clearly labelled source-classifier proxy."""
    metadata = summarize_condition_metadata(
        data["y_binary"], syn_y, data["synthetic_type"],
        condition_label=data["condition_label"] if "condition_label" in data.files else None,
        condition_name=data["condition_name"] if "condition_name" in data.files else None)
    rng = np.random.default_rng(seed + 2000)
    idx_parts = []
    for cls in EXPECTED_CLASS_NAMES:
        candidates = np.flatnonzero(syn_y == cls)
        n = min(sample_per_class, len(candidates))
        idx_parts.append(rng.choice(candidates, n, replace=False))
    idx = np.concatenate(idx_parts)
    pred = clf.predict(rf_features(syn_x[idx]))
    row = {"pool": pool, **metadata, **anchor,
           "semantic_measure_type": "source-trained RF label-agreement proxy",
           "source_rf_proxy_windows": int(len(idx)),
           "source_rf_condition_agreement": float((pred == syn_y[idx]).mean())}
    for cls, name in EXPECTED_CLASS_NAMES.items():
        m = syn_y[idx] == cls
        row[f"source_rf_condition_agreement_{name}"] = float((pred[m] == cls).mean())
    return row


def _resolve_pool(spec):
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = lc.ROOT / path
    else:
        name = spec
        if name not in lc.SYNTHETIC_POOLS:
            raise ValueError(f"unknown pool {name!r}; use NAME=PATH for an explicit pool")
        path = lc.SYNTHETIC_POOLS[name]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"invalid pool name: {name!r}")
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"pool does not exist: {path}")
    return name, path


def _balanced_marginal_indices(real_features, max_per_class=5000):
    real_idx = np.concatenate([
        f["real_marginal_idx"][:max_per_class] for f in real_features.values()])
    syn_idx = np.concatenate([
        f["syn_marginal_idx"][:max_per_class] for f in real_features.values()])
    return real_idx, syn_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", nargs="+", default=DEFAULT_POOLS,
                        help="pool keys or NAME=PATH specs (default: rule ganvalid arlmvalid)")
    parser.add_argument("--output-tag", default="protocol_valid")
    parser.add_argument("--block-frames", type=int, default=DEFAULT_BLOCK_FRAMES)
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="explicitly replace this tag's output tables/log")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag):
        parser.error("--output-tag must contain only letters, digits, '_' or '-'")
    if args.block_frames < 128 or args.max_per_class < 10:
        parser.error("--block-frames must be >=128 and --max-per-class >=10")

    sep_out = lc.TABLES / f"generator_separability_audit_{args.output_tag}.csv"
    gram_out = lc.TABLES / f"generator_grammar_audit_{args.output_tag}.csv"
    cond_out = lc.TABLES / f"generator_condition_fidelity_audit_{args.output_tag}.csv"
    log_out = lc.LOGS / f"audit_generator_separability_{args.output_tag}.log"
    existing = [str(p) for p in [sep_out, gram_out, cond_out, log_out] if p.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(
            "refusing to overwrite audit outputs; select a new --output-tag: " + str(existing))

    train = lc.load_npz("train")
    real_y = np.asarray(train["y_attack_type"])
    real_x = train["x"]
    attack_mask = real_y > 0
    real_attack_x, real_attack_y = real_x[attack_mask], real_y[attack_mask]
    # Partitions use indices into the complete train array; remap to a compact
    # attack-only array for lower-memory downstream indexing.
    full_parts, split_meta = build_purged_real_partitions(
        train, block_frames=args.block_frames)
    full_to_attack = np.full(len(real_y), -1, dtype=np.int64)
    full_to_attack[np.flatnonzero(attack_mask)] = np.arange(int(attack_mask.sum()))
    real_parts = {c: {k: full_to_attack[v] for k, v in p.items()}
                  for c, p in full_parts.items()}

    normal_x = real_x[real_y == 0]
    vocab_normal = np.unique(normal_x[:, :, 0].astype(np.int64))
    vocab_all = np.unique(real_x[:, :, 0].astype(np.int64))
    attack_dt = real_attack_x[:, :, 10]
    reference_dt_range = (float(attack_dt.min()), float(attack_dt.max()))
    source_clf, source_anchor = train_source_condition_classifier(
        real_attack_x, real_attack_y, real_parts, args.max_per_class)

    grammar_rows = [grammar_row(
        "real_train_attack", real_attack_x, vocab_normal, vocab_all,
        reference_dt_range)]
    sep_rows, condition_rows = [], []
    resolved = [_resolve_pool(spec) for spec in args.pools]
    for pool, path in resolved:
        data = np.load(path, allow_pickle=True)
        required = {"x", "y_binary", "y_attack_type", "synthetic_type"}
        if not required.issubset(data.files):
            raise ValueError(f"{path} lacks required keys: {sorted(required - set(data.files))}")
        syn_x = data["x"]
        syn_y = np.asarray(data["y_attack_type"])
        observed = set(np.unique(syn_y).tolist())
        expected = set(EXPECTED_CLASS_NAMES)
        if observed != expected:
            raise ValueError(
                f"{pool} must contain exactly attack labels {sorted(expected)}; "
                f"observed {sorted(observed)}")

        domain, selected = grouped_class_matched_domain_audit(
            real_attack_x, real_parts, syn_x, syn_y, args.max_per_class)
        ridx, sidx = _balanced_marginal_indices(selected)
        try:
            recorded_path = str(path.resolve().relative_to(lc.ROOT.parent.resolve()))
        except ValueError:
            recorded_path = path.name
        row = {"pool": pool, "path": recorded_path, "windows": int(len(syn_x)),
               **domain, "split_protocol": split_meta["split_protocol"]}
        js_all = []
        for ch, name in enumerate(lc.FEATURE_NAMES):
            js = marginal_js(real_attack_x[ridx][:, :, ch].reshape(-1),
                             syn_x[sidx][:, :, ch].reshape(-1))
            row[f"js_{name}"] = js
            js_all.append(js)
        row["js_mean"] = float(np.mean(js_all))
        row["wasserstein_delta_t"] = float(wasserstein_distance(
            real_attack_x[ridx][:, :, 10].reshape(-1),
            syn_x[sidx][:, :, 10].reshape(-1)))
        sep_rows.append(row)
        grammar_rows.append(grammar_row(
            pool, syn_x, vocab_normal, vocab_all, reference_dt_range))
        condition_rows.append(condition_fidelity_row(
            pool, data, syn_x, syn_y, source_clf, source_anchor))
        print(json.dumps({"separability": row,
                          "condition_fidelity": condition_rows[-1]}, indent=2), flush=True)
        data.close()

    lc.write_csv(sep_out, sep_rows)
    lc.write_csv(gram_out, grammar_rows)
    lc.write_csv(cond_out, condition_rows)
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.write_text(json.dumps({
        "seed": SEED,
        "pools": [{"name": n, "path": str(p)} for n, p in resolved],
        "split": split_meta,
        "separability_rows": sep_rows,
        "grammar_rows": grammar_rows,
        "condition_rows": condition_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
