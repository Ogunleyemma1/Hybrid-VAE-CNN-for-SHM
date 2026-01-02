# experiments/openlab/scripts/03_train_vae.py
"""
Step 03: Train Temporal VAE on TRAIN runs, Normal windows only (N).

Inputs:
  - processed/X_clean.npy
  - processed/window_labels_augmented.csv
  - processed/run_split.json

Outputs (to outputs/):
  - outputs/models/vae_openlab.pt
  - outputs/models/vae_norm_stats.npz  (mean, std, channels)
  - outputs/metrics/vae_train_history.json
  - outputs/figures/vae/*.pdf/.svg (optional)

This is a training entrypoint. The VAE architecture is imported from src/models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.temporal_vae import VAE
from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root, set_seed


def fit_mu_sd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 1)).astype(np.float32)
    sd = X.std(axis=(0, 1)).astype(np.float32)
    sd = np.where(sd < 1e-12, 1.0, sd).astype(np.float32)
    return mu, sd


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Xn = (X - mu[None, None, :]) / sd[None, None, :]
    Xn = np.clip(Xn, -clip, clip)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn.astype(np.float32)


def kl_weight(epoch: int, n_epochs: int, ratio: float = 0.30) -> float:
    x = (epoch - int(n_epochs * ratio)) / max(int(n_epochs * ratio), 1)
    return float(1.0 / (1.0 + np.exp(-5.0 * x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip_z", type=float, default=10.0)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--use_only_gold_normals", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "03_train_vae.log")

    set_seed(args.seed, deterministic_torch=True)

    x_path = processed_dir / "X_clean.npy"
    meta_path = processed_dir / "window_labels_augmented.csv"
    split_path = processed_dir / "run_split.json"

    X = np.load(x_path).astype(np.float32)
    meta = pd.read_csv(meta_path)
    splits = json.loads(split_path.read_text(encoding="utf-8"))

    train_runs = set(splits["train_runs"])
    train_mask = (meta["run_id"].isin(train_runs)) & (meta["label"] == "N")
    if args.use_only_gold_normals:
        train_mask = train_mask & (meta["label_source"] == "gold")

    idx = train_mask.to_numpy()
    X_train_raw = X[idx]
    if X_train_raw.shape[0] < 10:
        raise RuntimeError("Too few training normal windows. Check splits/labels or relax gold-only normals.")

    mu, sd = fit_mu_sd(X_train_raw)
    X_train = apply_standardize(X_train_raw, mu, sd, clip=args.clip_z)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        input_dim=int(X_train.shape[2]),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    history: Dict[str, List[float]] = {"total": [], "recon": [], "kl": [], "kl_w": []}

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    logger.info(f"Train windows: {X_train.shape} | device={device}")

    for ep in range(args.epochs):
        model.train()
        k_w = kl_weight(ep, args.epochs, ratio=0.30)
        tot = rec = kld = 0.0
        nb = 0

        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu_t, logvar = model(xb)

            recon_loss = nn.functional.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu_t.pow(2) - logvar.exp())
            loss = recon_loss + k_w * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            tot += float(loss.item())
            rec += float(recon_loss.item())
            kld += float(kl.item())
            nb += 1

        tot /= max(nb, 1)
        rec /= max(nb, 1)
        kld /= max(nb, 1)

        history["total"].append(tot)
        history["recon"].append(rec)
        history["kl"].append(kld)
        history["kl_w"].append(k_w)

        logger.info(f"Epoch {ep+1:03}/{args.epochs} | total={tot:.6f} recon={rec:.6f} kl={kld:.6f} kl_w={k_w:.3f}")

    # Save artifacts
    vae_path = dirs["models"] / "vae_openlab.pt"
    torch.save(model.state_dict(), vae_path)

    norm_path = dirs["models"] / "vae_norm_stats.npz"
    np.savez_compressed(norm_path, mean=mu, std=sd)

    hist_path = dirs["metrics"] / "vae_train_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info(f"Saved VAE: {vae_path}")
    logger.info(f"Saved norm stats: {norm_path}")
    logger.info(f"Saved history: {hist_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
