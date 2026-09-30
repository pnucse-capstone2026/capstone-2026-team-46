#!/usr/bin/env python3
"""Task A2: train Conv1D autoencoders on Car-Hacking train-NORMAL windows only.

The AE never sees attack labels. Synthetic augmentation enters only at
threshold calibration time (evaluate_family_extension.py):
  - percentile: 99.5th percentile of val-normal reconstruction error
  - synthetic_calibrated: F1-optimal threshold on val-normal + sampled
    synthetic attack windows (rule pool, WISA sampling seed rule)
Models go only to journal/models/family_extension/.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lib_common as lc


class ConvAE(nn.Module):
    def __init__(self, in_channels=11):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 32, kernel_size=3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):  # x: (B, 11, 128)
        return self.decoder(self.encoder(x))


def reconstruction_errors(model, x, device, batch_size=1024):
    """Mean squared reconstruction error per window; x is (N,128,11) standardized."""
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i : i + batch_size].transpose(0, 2, 1)).to(device)
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def train_one(seed, max_epochs=20):
    t0 = time.time()
    lc.set_seed(seed)
    train = lc.load_npz("train")
    val = lc.load_npz("val")
    mean, std = lc.fit_standardizer(train["x"])
    train_normal = lc.standardize(train["x"][train["y_attack_type"] == 0], mean, std)
    val_normal = lc.standardize(val["x"][val["y_attack_type"] == 0], mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        lc.WindowDataset(train_normal, np.zeros(len(train_normal))), batch_size=512,
        shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    model = ConvAE(in_channels=train_normal.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val, best_state, stale, history = float("inf"), None, 0, []
    patience = 4
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
        val_err = float(reconstruction_errors(model, val_normal, device).mean())
        history.append({"epoch": epoch, "train_mse": total / len(train_normal), "val_normal_mse": val_err})
        print(f"ae seed={seed} epoch={epoch} train_mse={history[-1]['train_mse']:.6f} "
              f"val_mse={val_err:.6f} elapsed={time.time()-t0:.0f}s", flush=True)
        if val_err < best_val:
            best_val, stale = val_err, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    lc.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), lc.MODELS / f"ae_seed{seed}.pt")
    np.savez(lc.MODELS / f"ae_standardizer_seed{seed}.npz", mean=mean, std=std)
    lc.LOGS.mkdir(parents=True, exist_ok=True)
    log = {
        "family": "ae", "seed": seed, "train_normal_windows": int(len(train_normal)),
        "best_val_normal_mse": best_val, "epochs_run": len(history),
        "history": history, "elapsed_seconds": time.time() - t0,
    }
    (lc.LOGS / f"train_ae_seed{seed}.log").write_text(json.dumps(log, indent=2))
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()
    seeds = lc.SEEDS if args.all else [args.seed]
    if not args.all and args.seed is None:
        raise SystemExit("either --all or --seed")
    for seed in seeds:
        if (lc.MODELS / f"ae_seed{seed}.pt").exists():
            print(f"skip existing ae_seed{seed}.pt", flush=True)
            continue
        train_one(seed, args.max_epochs)


if __name__ == "__main__":
    main()
