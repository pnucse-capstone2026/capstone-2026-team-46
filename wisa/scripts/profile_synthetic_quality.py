#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train_real_only_baselines import rf_features


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows"
SYNTHETIC = ROOT / "datasets" / "synthetic"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
SEED = 42


def prob_hist(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(np.float64)
    if counts.sum() == 0:
        return np.ones(len(bins) - 1) / (len(bins) - 1)
    return counts / counts.sum()


def js_divergence(p, q):
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def shannon_entropy(values, bins):
    p = prob_hist(values, bins)
    return float(entropy(p + 1e-12, base=2))


def periodicity_mae(real_x, syn_x):
    real_ids = real_x[:, :, 0].astype(np.int32).reshape(-1)
    real_dt = real_x[:, :, 10].reshape(-1)
    syn_ids = syn_x[:, :, 0].astype(np.int32).reshape(-1)
    syn_dt = syn_x[:, :, 10].reshape(-1)
    real_sum = np.bincount(real_ids, weights=real_dt, minlength=2048)
    real_cnt = np.bincount(real_ids, minlength=2048)
    syn_sum = np.bincount(syn_ids, weights=syn_dt, minlength=2048)
    syn_cnt = np.bincount(syn_ids, minlength=2048)
    common = (real_cnt > 100) & (syn_cnt > 100)
    if not np.any(common):
        return float("nan")
    real_mean = real_sum[common] / real_cnt[common]
    syn_mean = syn_sum[common] / syn_cnt[common]
    return float(np.mean(np.abs(real_mean - syn_mean)))


def discriminator_auc(real_x, syn_x):
    rng = np.random.default_rng(SEED)
    n = min(50_000, len(real_x), len(syn_x))
    real_idx = rng.choice(len(real_x), size=n, replace=False)
    syn_idx = rng.choice(len(syn_x), size=n, replace=False)
    x = np.concatenate([rf_features(real_x[real_idx]), rf_features(syn_x[syn_idx])], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores))


def plot_distribution(path, title, labels, real_values, syn_values, bins, xlabel):
    real_p = prob_hist(real_values, bins)
    syn_p = prob_hist(syn_values, bins)
    centers = np.arange(len(real_p))
    width = 0.45
    plt.figure(figsize=(9, 4.5))
    plt.bar(centers - width / 2, real_p, width=width, label="Real train attack")
    plt.bar(centers + width / 2, syn_p, width=width, label="Rule synthetic")
    if labels is not None:
        plt.xticks(centers, labels, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_top_id_distribution(path, real_p, syn_p):
    top_ids = np.argsort(real_p + syn_p)[-20:]
    top_ids = top_ids[np.argsort(-(real_p[top_ids] + syn_p[top_ids]))]
    x = np.arange(len(top_ids))
    width = 0.45
    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2, real_p[top_ids], width=width, label="Real train attack")
    plt.bar(x + width / 2, syn_p[top_ids], width=width, label="Rule synthetic")
    plt.xticks(x, [f"0x{i:03x}" for i in top_ids], rotation=45, ha="right")
    plt.title("CAN ID distribution: real attack vs rule synthetic")
    plt.xlabel("Top CAN IDs")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    train = np.load(WINDOWS / "train_windows.npz", allow_pickle=True)
    syn = np.load(SYNTHETIC / "rule_based_windows.npz", allow_pickle=True)
    real_x = train["x"][train["y_binary"] == 1].astype(np.float32)
    syn_x = syn["x"].astype(np.float32)

    real_ids = real_x[:, :, 0].reshape(-1)
    syn_ids = syn_x[:, :, 0].reshape(-1)
    real_dt = real_x[:, :, 10].reshape(-1)
    syn_dt = syn_x[:, :, 10].reshape(-1)
    real_payload = real_x[:, :, 2:10].reshape(-1)
    syn_payload = syn_x[:, :, 2:10].reshape(-1)

    id_bins = np.arange(0, 2050)
    byte_bins = np.arange(0, 258)
    dt_bins = np.array([0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1.0])

    real_id_p = prob_hist(real_ids, id_bins)
    syn_id_p = prob_hist(syn_ids, id_bins)
    real_dt_p = prob_hist(real_dt, dt_bins)
    syn_dt_p = prob_hist(syn_dt, dt_bins)
    real_payload_p = prob_hist(real_payload, byte_bins)
    syn_payload_p = prob_hist(syn_payload, byte_bins)

    auc = discriminator_auc(real_x, syn_x)
    row = {
        "generator": "rule_based",
        "real_windows": len(real_x),
        "synthetic_windows": len(syn_x),
        "can_id_js": js_divergence(real_id_p, syn_id_p),
        "delta_t_js": js_divergence(real_dt_p, syn_dt_p),
        "payload_js": js_divergence(real_payload_p, syn_payload_p),
        "can_id_wasserstein": float(wasserstein_distance(real_ids, syn_ids)),
        "delta_t_wasserstein": float(wasserstein_distance(real_dt, syn_dt)),
        "payload_wasserstein": float(wasserstein_distance(real_payload, syn_payload)),
        "real_can_id_entropy": shannon_entropy(real_ids, id_bins),
        "synthetic_can_id_entropy": shannon_entropy(syn_ids, id_bins),
        "real_payload_entropy": shannon_entropy(real_payload, byte_bins),
        "synthetic_payload_entropy": shannon_entropy(syn_payload, byte_bins),
        "real_delta_t_mean": float(np.mean(real_dt)),
        "synthetic_delta_t_mean": float(np.mean(syn_dt)),
        "periodicity_mae": periodicity_mae(real_x, syn_x),
        "real_vs_synthetic_auc": auc,
    }
    with (TABLES / "synthetic_quality_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    plot_top_id_distribution(
        FIGURES / "real_vs_synthetic_can_id_dist.png",
        real_id_p,
        syn_id_p,
    )
    dt_labels = ["0", "<1e-6", "<1e-5", "<5e-5", "<1e-4", "<5e-4", "<1e-3", "<5e-3", "<1e-2", "<5e-2", "<1e-1"]
    plot_distribution(
        FIGURES / "real_vs_synthetic_delta_t_dist.png",
        "Delta-t distribution: real attack vs rule synthetic",
        dt_labels,
        real_dt,
        syn_dt,
        dt_bins,
        "Delta-t bin",
    )
    plt.figure(figsize=(4.5, 4))
    plt.bar(["Rule synthetic"], [auc], color="#4c78a8")
    plt.ylim(0.5, 1.0)
    plt.ylabel("Real-vs-Synthetic AUC")
    plt.title("Synthetic artifact detectability")
    plt.tight_layout()
    plt.savefig(FIGURES / "real_vs_synthetic_auc.png", dpi=180)
    plt.close()
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
