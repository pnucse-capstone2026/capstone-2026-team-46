#!/usr/bin/env python3
"""Track B/D: train 1D-CNN 5-class window classifiers for the generator-extension
and seed-unification arms.

CNN1D architecture and training conventions forked from
wisa/scripts/train_real_only_baselines.py @ eb131df (class-weighted CE,
AdamW 1e-3/1e-4, batch 512, val multiclass Macro-F1 selection; real_only max 20
epochs/patience 4, augmented arms max 12/patience 3) so journal CNN results are
directly comparable to the frozen WISA tables and to the A-track LSTM/AE arms.

Arms:
  real_only            (D: unifies the WISA 3-seed main CNN results at 5 seeds)
  rule_0p30, rule_1p00 (D: rule arms at 5 seeds, matching the A-track LSTM arms)
  gan_0p30,  gan_1p00  (B: learned-generator augmentation arms)

Models go only to journal/models/generator_extension/.
"""
import argparse
import json
import math
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import lib_common as lc

SETTINGS = ["real_only", "rule_0p30", "rule_1p00", "gan_0p10", "gan_0p30", "gan_1p00",
            "ganb_0p30", "ganc_0p30",  # generator-seed sensitivity (critique #2)
            "over_0p30", "over_1p00",  # equal-budget real oversampling null (critique #7)
            "gansnap_0p30",  # ID-marginal-calibrated contrast (tautology defense, B-plan §8)
            "arlm_0p30"]  # E1 on-grammar learned generator (e1_arlm_ongrammar/PREREG.md)
# Corrected learned pools are opt-in so legacy --all behavior and published
# checkpoints remain unchanged. They enforce DLC/payload protocol consistency.
VALID_SETTINGS = SETTINGS + ["ganvalid_0p10", "ganvalid_0p30", "ganvalid_1p00",
                             "ganvalidb_1p00", "ganvalidc_1p00",
                             "arlmvalid_0p30",
                             # E10 marginal-matched placebo arm
                             # (e10_background_grammar_crossover/PREREG.md §1)
                             "placebo_0p30",
                             # E12 AR-LM generator-seed sensitivity arms
                             "arlmvalidb_0p30", "arlmvalidc_0p30"]
MODELS = lc.ROOT / "models" / "generator_extension"


class CNN1D(nn.Module):
    """Forked from wisa/scripts/train_real_only_baselines.py @ eb131df."""

    def __init__(self, in_channels=11, classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, classes))

    def forward(self, x):  # x: (B, 11, 128)
        return self.head(self.net(x))


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
    return MODELS / f"cnn_{setting}{tag}_seed{seed}.pt"


def training_log_path(setting, seed, model_tag=""):
    tag = f"_{model_tag}" if model_tag else ""
    return lc.LOGS / f"train_cnn_{setting}{tag}_seed{seed}.log"


def path_lexists(path):
    """Return true for every occupied path, including dangling symlinks."""
    return os.path.lexists(os.fspath(path))


def _fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_staging_file(target):
    """Create a private staging file on the target's filesystem."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.staged-",
        suffix=".tmp",
    )
    return descriptor, target.with_name(os.path.basename(name))


def _unlink_if_staged_hardlink(target, staged):
    """Roll back only a final link that still names our staged inode."""
    try:
        target_stat = os.stat(target, follow_symlinks=False)
        staged_stat = os.stat(staged, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        target_stat.st_dev == staged_stat.st_dev
        and target_stat.st_ino == staged_stat.st_ino
    ):
        os.unlink(target)


def _publish_staged_training_bundle(staged_targets):
    """Exclusively publish a complete staged checkpoint/log bundle."""
    collisions = [
        target for _, target in staged_targets if path_lexists(target)
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing training artifact(s): "
            + ", ".join(str(path) for path in collisions)
        )

    published = []
    try:
        for staged, target in staged_targets:
            try:
                os.link(staged, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"training artifact appeared during publication: {target}"
                ) from exc
            published.append((staged, target))
        for directory in dict.fromkeys(
            target.parent for _, target in staged_targets
        ):
            _fsync_directory(directory)
    except BaseException:
        for staged, target in reversed(published):
            _unlink_if_staged_hardlink(target, staged)
        for directory in dict.fromkeys(
            target.parent for _, target in published
        ):
            _fsync_directory(directory)
        raise


def train_one(setting, seed, max_epochs=None, budget_mode="legacy",
              max_steps=None, model_tag="", sampling_policy="legacy"):
    model_path = checkpoint_path(setting, seed, model_tag)
    log_path = training_log_path(setting, seed, model_tag)
    collisions = [
        path for path in (model_path, log_path) if path_lexists(path)
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
    real_train_windows = len(real_train)
    mean, std = lc.fit_standardizer(real_train)  # fit on real train only (WISA convention)
    del real_train
    train_x = lc.standardize(train_x, mean, std)
    val_x = lc.standardize(val_x, mean, std)

    if max_epochs is None:
        max_epochs = 20 if setting == "real_only" else 12
    patience = 4 if setting == "real_only" else 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        lc.WindowDataset(train_x, train_y), batch_size=512, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        lc.WindowDataset(val_x, val_y), batch_size=1024, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )

    model = CNN1D(in_channels=train_x.shape[-1]).to(device)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(5), y=train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1, best_state, stale, history = -1.0, None, 0, []
    optimizer_steps = 0

    def validate(train_loss, epoch):
        nonlocal best_f1, best_state, stale
        val_true, val_probs = predict(model, val_loader, device)
        val_f1 = f1_score(val_true, val_probs.argmax(axis=1), average="macro", zero_division=0)
        history.append({"epoch": epoch, "optimizer_step": optimizer_steps,
                        "train_loss": float(train_loss), "val_macro_f1": float(val_f1)})
        print(f"cnn {setting} seed={seed} epoch={epoch} step={optimizer_steps} "
              f"loss={train_loss:.5f} val_f1={val_f1:.5f} elapsed={time.time()-t0:.0f}s", flush=True)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

    if budget_mode == "matched_steps":
        # Use twelve real-training epochs worth of optimizer updates for every
        # arm. This diagnostic separates augmentation content from extra update
        # opportunities while keeping best-validation checkpoint selection.
        real_steps = math.ceil(real_train_windows / 512)
        target_steps = max_steps or real_steps * 12
        loader_iter = iter(train_loader)
        loss_sum, examples, checkpoint = 0.0, 0, 0
        while optimizer_steps < target_steps:
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)
            model.train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach().cpu()) * len(y)
            examples += len(y)
            if optimizer_steps % real_steps == 0 or optimizer_steps == target_steps:
                checkpoint += 1
                validate(loss_sum / examples, checkpoint)
                loss_sum, examples = 0.0, 0
    else:
        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                total_loss += float(loss.detach().cpu()) * len(y)
            validate(total_loss / len(train_y), epoch)
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    MODELS.mkdir(parents=True, exist_ok=True)
    lc.LOGS.mkdir(parents=True, exist_ok=True)
    collisions = [
        path for path in (model_path, log_path) if path_lexists(path)
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing training artifact(s): "
            + ", ".join(str(path) for path in collisions)
        )

    model_descriptor, staged_model = _new_staging_file(model_path)
    staged_log = None
    try:
        with os.fdopen(model_descriptor, "wb") as handle:
            torch.save(model.state_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())

        log = {
            "family": "cnn", "setting": setting, "seed": seed,
            "budget_mode": budget_mode, "model_tag": model_tag,
            "sampling_policy": sampling_policy,
            "sampling_audit": sampling_audit,
            "optimizer_steps": optimizer_steps, "real_train_windows": real_train_windows,
            "train_windows": int(len(train_y)), "best_val_macro_f1": float(best_f1),
            "epochs_run": len(history), "selected_epoch": int(max(history, key=lambda h: h["val_macro_f1"])["epoch"]),
            "epochs_run_semantics": ("full_dataloader_epochs" if budget_mode == "legacy"
                                      else "validation_checkpoints_not_augmented_data_epochs"),
            "selected_epoch_semantics": ("data_epoch" if budget_mode == "legacy"
                                          else "validation_checkpoint"),
            "validation_checkpoints": len(history),
            "history": history, "elapsed_seconds": time.time() - t0,
        }
        if sampling_policy == "strict-v2":
            log.update({
                "artifact_paths": {
                    "checkpoint": lc._audit_path(model_path),
                },
                "artifact_sha256": {
                    "checkpoint": lc.sha256_file(staged_model),
                },
            })

        log_descriptor, staged_log = _new_staging_file(log_path)
        with os.fdopen(log_descriptor, "w", encoding="utf-8") as handle:
            json.dump(log, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_staged_training_bundle((
            (staged_model, model_path),
            (staged_log, log_path),
        ))
        return log
    finally:
        for staged in (staged_model, staged_log):
            if staged is not None:
                try:
                    staged.unlink()
                except FileNotFoundError:
                    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, help="comma-separated subset of settings")
    parser.add_argument("--seeds", type=str, help="comma-separated subset of seeds")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--budget-mode", choices=["legacy", "matched_steps"], default="legacy")
    parser.add_argument("--max-steps", type=int,
                        help="override matched optimizer-step budget")
    parser.add_argument("--model-tag", default="",
                        help="checkpoint suffix; matched_steps defaults to matchedsteps")
    parser.add_argument(
        "--sampling-policy",
        choices=["legacy", "strict-v2"],
        default="legacy",
        help="strict-v2 opts into versioned pools and no-replacement sampling",
    )
    args = parser.parse_args()

    settings = args.settings.split(",") if args.settings else SETTINGS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else lc.SEEDS
    default_tag = ("matchedsteps" if args.budget_mode == "matched_steps" else "")
    if args.sampling_policy == "strict-v2":
        default_tag = f"{default_tag + '_' if default_tag else ''}sampling_v2"
    model_tag = args.model_tag or default_tag
    try:
        lc.validate_artifact_tag(model_tag, name="--model-tag")
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")
    if args.max_epochs is not None and args.max_epochs <= 0:
        parser.error("--max-epochs must be a positive integer")
    if args.budget_mode == "legacy" and args.max_steps is not None:
        parser.error("--max-steps requires --budget-mode matched_steps")
    if args.budget_mode == "matched_steps" and args.max_epochs is not None:
        parser.error("--max-epochs is a legacy-budget option; use --max-steps for matched_steps")
    if lc.is_strict_v2_tag(model_tag) and args.sampling_policy != "strict-v2":
        parser.error(
            "a strict-v2 model tag requires --sampling-policy strict-v2"
        )
    if args.sampling_policy == "strict-v2":
        tag_is_matched = "matchedsteps" in model_tag
        budget_is_matched = args.budget_mode == "matched_steps"
        if tag_is_matched != budget_is_matched:
            parser.error(
                "strict-v2 budget/tag mismatch: matched_steps requires a "
                "'matchedsteps' tag and legacy budget forbids it"
            )
        if (
                args.max_epochs is not None or args.max_steps is not None
        ) and "smoke" not in model_tag:
            parser.error(
                "strict-v2 budget overrides require a model tag containing "
                "'smoke'"
            )
    for s in settings:
        if s not in VALID_SETTINGS:
            raise SystemExit(f"unknown setting: {s}")
    if args.sampling_policy == "strict-v2":
        unsupported = sorted(
            {setting.split("_")[0] for setting in settings}
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
                family="cnn",
                settings=settings,
                seeds=seeds,
                tag=model_tag,
                budget_mode=args.budget_mode,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if not args.all and not (args.settings or args.seeds):
        raise SystemExit("pass --all to train every arm, or restrict with --settings/--seeds")

    for setting in settings:
        for seed in seeds:
            model_path = checkpoint_path(setting, seed, model_tag)
            if path_lexists(model_path):
                if args.sampling_policy == "strict-v2":
                    lc.validate_strict_v2_training_bundle(
                        training_log_path(setting, seed, model_tag),
                        {"checkpoint": model_path},
                        {
                            "family": "cnn",
                            "setting": setting,
                            "seed": seed,
                            "model_tag": model_tag,
                            "sampling_policy": "strict-v2",
                            "budget_mode": args.budget_mode,
                        },
                    )
                print(f"skip existing {model_path.name}", flush=True)
                continue
            train_one(setting, seed, args.max_epochs, args.budget_mode,
                      args.max_steps, model_tag, args.sampling_policy)


if __name__ == "__main__":
    main()
