#!/usr/bin/env python3
"""Task E3: train small temporal self-attention (transformer encoder) 5-class
window classifiers for the family-extension arm.

Confirmatory detector-family axis extension (CAN-BERT lineage gap): the family
is the ONLY variable. Everything else mirrors train_family_extension_lstm.py
exactly — arms (real_only, rule_0p30, rule_1p00), seeds lc.SEEDS, window
caches, standardizer fit on real train only, class-weighted CE, AdamW
1e-3/1e-4, batch 512, val multiclass Macro-F1 selection; real_only max 20
epochs/patience 4, augmented arms max 12/patience 3.

Architecture: linear projection 11 -> d_model=128, learned positional
embedding (128 positions), 4 pre-norm TransformerEncoder layers (nhead=4,
dim_feedforward=256, dropout 0.1), mean-pool over time, 2-layer head.
565,253 parameters vs 573,189 for the BiLSTM (same ballpark by design).

Models go only to journal/models/family_extension/ as
transformer_{setting}_seed{seed}.pt.

Smoke-test-only flags (never used for real runs): --tag appends a suffix such
as _smoke to all output filenames so scratch checkpoints cannot be confused
with real results; --train-subset trains on a stratified fraction of the
training windows.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import lib_common as lc

SETTINGS = ["real_only", "rule_0p30", "rule_1p00"]


class TransformerClassifier(nn.Module):
    def __init__(self, in_features=11, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1, seq_len=128, classes=5):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def forward(self, x):  # x: (B, 128, 11)
        h = self.proj(x) + self.pos.weight.unsqueeze(0)
        h = self.norm(self.encoder(h))
        return self.head(h.mean(dim=1))


def predict(model, loader, device):
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(probs)


def stratified_subset(y, fraction, seed):
    """Per-class proportional subsample (smoke testing only)."""
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        n = max(1, round(len(idx) * fraction))
        keep.append(rng.choice(idx, size=n, replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def training_artifact_paths(setting, seed, tag=""):
    return (
        lc.MODELS / f"transformer_{setting}_seed{seed}{tag}.pt",
        lc.MODELS / f"transformer_standardizer_{setting}_seed{seed}{tag}.npz",
        lc.LOGS / f"train_transformer_{setting}_seed{seed}{tag}.log",
    )


def train_one(
        setting,
        seed,
        max_epochs=None,
        tag="",
        train_subset=None,
        sampling_policy="legacy"):
    model_path, scaler_path, training_log = training_artifact_paths(
        setting, seed, tag
    )
    collisions = [
        path for path in (model_path, scaler_path, training_log) if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing training artifact(s): "
            + ", ".join(str(path) for path in collisions)
        )
    t0 = time.time()
    lc.set_seed(seed)
    if sampling_policy == "strict-v2":
        train_x, train_y, val_x, val_y, sampling_audit = (
            lc.build_train_arrays_v2(setting, seed)
        )
    elif sampling_policy == "legacy":
        train_x, train_y, val_x, val_y = lc.build_train_arrays(setting, seed)
        sampling_audit = None
    else:
        raise ValueError(f"unknown sampling policy: {sampling_policy}")
    if train_subset is not None:  # smoke testing only
        idx = stratified_subset(train_y, train_subset, seed)
        train_x, train_y = train_x[idx], train_y[idx]
    real_train = lc.load_npz("train")["x"]
    mean, std = lc.fit_standardizer(real_train)  # fit on real train only (WISA convention)
    del real_train
    train_x = lc.standardize(train_x, mean, std)
    val_x = lc.standardize(val_x, mean, std)

    if max_epochs is None:
        max_epochs = 20 if setting == "real_only" else 12
    patience = 4 if setting == "real_only" else 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        lc.SequenceDataset(train_x, train_y), batch_size=512, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        lc.SequenceDataset(val_x, val_y), batch_size=1024, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )

    model = TransformerClassifier(in_features=train_x.shape[-1]).to(device)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(5), y=train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
        val_true, val_probs = predict(model, val_loader, device)
        val_f1 = f1_score(val_true, val_probs.argmax(axis=1), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y), "val_macro_f1": float(val_f1)})
        print(f"transformer{tag} {setting} seed={seed} epoch={epoch} loss={history[-1]['train_loss']:.5f} "
              f"val_f1={val_f1:.5f} elapsed={time.time()-t0:.0f}s", flush=True)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    lc.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=mean, std=std)

    lc.LOGS.mkdir(parents=True, exist_ok=True)
    log = {
        "family": "transformer", "setting": setting, "seed": seed,
        "train_windows": int(len(train_y)), "best_val_macro_f1": float(best_f1),
        "epochs_run": len(history), "selected_epoch": int(max(history, key=lambda h: h["val_macro_f1"])["epoch"]),
        "history": history, "elapsed_seconds": time.time() - t0,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "tag": tag, "train_subset": train_subset,
        "sampling_policy": sampling_policy,
        "sampling_audit": sampling_audit,
    }
    if sampling_policy == "strict-v2":
        log.update(lc.strict_v2_artifact_record({
            "checkpoint": model_path,
            "standardizer": scaler_path,
        }))
        lc.write_json_atomic(training_log, log)
    else:
        training_log.write_text(json.dumps(log, indent=2))
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=SETTINGS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--tag", type=str, default="",
                        help="filename suffix for scratch runs (e.g. _smoke); leave empty for real runs")
    parser.add_argument("--train-subset", type=float, default=None,
                        help="stratified training-subset fraction (smoke testing only)")
    parser.add_argument(
        "--sampling-policy",
        choices=["legacy", "strict-v2"],
        default="legacy",
        help="strict-v2 opts into versioned pools and no-replacement sampling",
    )
    args = parser.parse_args()

    jobs = ([(s, sd) for s in SETTINGS for sd in lc.SEEDS] if args.all
            else [(args.setting, args.seed)])
    if not args.all and (args.setting is None or args.seed is None):
        raise SystemExit("either --all or both --setting and --seed")
    if args.max_epochs is not None and args.max_epochs <= 0:
        parser.error("--max-epochs must be a positive integer")
    if args.train_subset is not None and not 0 < args.train_subset <= 1:
        parser.error("--train-subset must be in (0, 1]")
    tag = args.tag or (
        "_sampling_v2" if args.sampling_policy == "strict-v2" else ""
    )
    try:
        lc.validate_artifact_tag(
            tag, name="--tag", leading_underscore=True
        )
    except ValueError as exc:
        parser.error(str(exc))
    if tag and not tag.startswith("_"):
        parser.error("--tag must start with '_'")
    if lc.is_strict_v2_tag(tag) and args.sampling_policy != "strict-v2":
        parser.error(
            "a strict-v2 model tag requires --sampling-policy strict-v2"
        )
    if (
            args.sampling_policy == "strict-v2"
            and (args.max_epochs is not None or args.train_subset is not None)
            and "smoke" not in tag
    ):
        parser.error(
            "strict-v2 --max-epochs/--train-subset overrides require a tag "
            "containing 'smoke'"
        )
    if args.sampling_policy == "strict-v2":
        if "matchedsteps" in tag.lower():
            parser.error(
                "transformer strict-v2 has no matched-step policy; "
                "--tag may not contain 'matchedsteps'"
            )
        unsupported = sorted(
            {setting.split("_")[0] for setting, _seed in jobs}
            - set(lc.STRICT_V2_POOL_CONFIGS)
        )
        if unsupported:
            parser.error(
                "strict-v2 supports only declared versioned pools; "
                f"unsupported: {unsupported}"
            )
        if "sampling_v2" not in tag and "unique_pool_v2" not in tag:
            parser.error(
                "strict-v2 --tag must include 'sampling_v2' or "
                "'unique_pool_v2'"
            )
        try:
            lc.validate_strict_v2_training_scope(
                family="transformer",
                settings=[setting for setting, _seed in jobs],
                seeds=[seed for _setting, seed in jobs],
                tag=tag,
            )
        except ValueError as exc:
            parser.error(str(exc))
    for setting, seed in jobs:
        model_path = training_artifact_paths(setting, seed, tag)[0]
        if model_path.exists():
            if args.sampling_policy == "strict-v2":
                strict_paths = training_artifact_paths(setting, seed, tag)
                lc.validate_strict_v2_training_bundle(
                    strict_paths[2],
                    {
                        "checkpoint": strict_paths[0],
                        "standardizer": strict_paths[1],
                    },
                    {
                        "family": "transformer",
                        "setting": setting,
                        "seed": seed,
                        "tag": tag,
                        "sampling_policy": "strict-v2",
                    },
                )
            print(f"skip existing {model_path.name}", flush=True)
            continue
        train_one(
            setting,
            seed,
            args.max_epochs,
            tag,
            args.train_subset,
            args.sampling_policy,
        )


if __name__ == "__main__":
    main()
