#!/usr/bin/env python3
"""Task A1: train BiLSTM 5-class window classifiers for the family-extension arm.

Mirrors the WISA CNN training conventions (class-weighted CE, AdamW 1e-3/1e-4,
batch 512, val multiclass Macro-F1 selection; real_only max 20 epochs/patience 4,
augmented arms max 12/patience 3) so results are comparable across families.
Models go only to journal/models/family_extension/.
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


class LSTMClassifier(nn.Module):
    def __init__(self, in_features=11, hidden=128, classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, classes),
        )

    def forward(self, x):  # x: (B, 128, 11)
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


def predict(model, loader, device):
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(probs)


def checkpoint_path(setting, seed, model_tag=""):
    tag = f"_{model_tag}" if model_tag else ""
    return lc.MODELS / f"lstm_{setting}{tag}_seed{seed}.pt"


def standardizer_path(setting, seed, model_tag=""):
    tag = f"_{model_tag}" if model_tag else ""
    return lc.MODELS / f"lstm_standardizer_{setting}{tag}_seed{seed}.npz"


def log_path(setting, seed, model_tag=""):
    tag = f"_{model_tag}" if model_tag else ""
    return lc.LOGS / f"train_lstm_{setting}{tag}_seed{seed}.log"


def train_one(
        setting,
        seed,
        max_epochs=None,
        model_tag="",
        sampling_policy="legacy"):
    model_path = checkpoint_path(setting, seed, model_tag)
    scaler_path = standardizer_path(setting, seed, model_tag)
    training_log = log_path(setting, seed, model_tag)
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

    model = LSTMClassifier(in_features=train_x.shape[-1]).to(device)
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
        print(f"lstm {setting} seed={seed} epoch={epoch} loss={history[-1]['train_loss']:.5f} "
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
        "family": "lstm", "setting": setting, "seed": seed,
        "model_tag": model_tag, "sampling_policy": sampling_policy,
        "sampling_audit": sampling_audit,
        "train_windows": int(len(train_y)), "best_val_macro_f1": float(best_f1),
        "epochs_run": len(history), "selected_epoch": int(max(history, key=lambda h: h["val_macro_f1"])["epoch"]),
        "history": history, "elapsed_seconds": time.time() - t0,
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
    parser.add_argument("--model-tag", default="",
                        help="version suffix for checkpoints/logs (without leading '_')")
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
    model_tag = args.model_tag or (
        "sampling_v2" if args.sampling_policy == "strict-v2" else ""
    )
    try:
        lc.validate_artifact_tag(model_tag, name="--model-tag")
    except ValueError as exc:
        parser.error(str(exc))
    if lc.is_strict_v2_tag(model_tag) and args.sampling_policy != "strict-v2":
        parser.error(
            "a strict-v2 model tag requires --sampling-policy strict-v2"
        )
    if (
            args.sampling_policy == "strict-v2"
            and args.max_epochs is not None
            and "smoke" not in model_tag
    ):
        parser.error(
            "strict-v2 --max-epochs requires a model tag containing 'smoke'"
        )
    if args.sampling_policy == "strict-v2":
        if "matchedsteps" in model_tag.lower():
            parser.error(
                "BiLSTM strict-v2 has no matched-step policy; "
                "--model-tag may not contain 'matchedsteps'"
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
        if "sampling_v2" not in model_tag and "unique_pool_v2" not in model_tag:
            parser.error(
                "strict-v2 --model-tag must include 'sampling_v2' or "
                "'unique_pool_v2'"
            )
        try:
            lc.validate_strict_v2_training_scope(
                family="lstm",
                settings=[setting for setting, _seed in jobs],
                seeds=[seed for _setting, seed in jobs],
                tag=model_tag,
            )
        except ValueError as exc:
            parser.error(str(exc))
    for setting, seed in jobs:
        model_path = checkpoint_path(setting, seed, model_tag)
        if model_path.exists():
            if args.sampling_policy == "strict-v2":
                lc.validate_strict_v2_training_bundle(
                    log_path(setting, seed, model_tag),
                    {
                        "checkpoint": model_path,
                        "standardizer": standardizer_path(
                            setting, seed, model_tag
                        ),
                    },
                    {
                        "family": "lstm",
                        "setting": setting,
                        "seed": seed,
                        "model_tag": model_tag,
                        "sampling_policy": "strict-v2",
                    },
                )
            print(f"skip existing {model_path.name}", flush=True)
            continue
        train_one(
            setting,
            seed,
            args.max_epochs,
            model_tag,
            args.sampling_policy,
        )


if __name__ == "__main__":
    main()
