# experiments/4dof/scripts/03_train_vae.py
"""
Step 03 (4DOF): Train VAE on TRAIN runs, label=N only.

Outputs:
  outputs/models/vae_4dof.pt
  outputs/models/vae_norm_stats.npz
  outputs/metrics/vae_train_history.json
"""

from __future__ import annotations

import argparse
import json

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


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Z = (X - mu[None, None, :]) / sd[None, None, :]
    Z = np.clip(Z, -clip, clip)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32)


def kl_weight(epoch: int, n_epochs: int, ratio: float = 0.30) -> float:
    x = (epoch - int(n_epochs * ratio)) / max(int(n_epochs * ratio), 1)
    return float(1.0 / (1.0 + np.exp(-5.0 * x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/4dof/datasets/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "03_train_vae.log")
    set_seed(args.seed, deterministic_torch=True)

    X = np.load(processed_dir / "X.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "meta_windows.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))

    train_runs = set(splits["train_runs"])
    mask = meta["run_id"].isin(train_runs) & (meta["label"] == "N")
    Xtr = X[mask.to_numpy()]
    if Xtr.shape[0] < 50:
        raise RuntimeError("Too few normal windows for VAE training. Generate more normal runs or adjust windowing.")

    mu, sd = fit_mu_sd(Xtr)
    Ztr = standardize(Xtr, mu, sd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        input_dim=int(Ztr.shape[2]),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Ztr, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    hist = {"total": [], "recon": [], "kl": [], "kl_w": []}

    for ep in range(args.epochs):
        model.train()
        k_w = kl_weight(ep, args.epochs)
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

        hist["total"].append(tot / nb)
        hist["recon"].append(rec / nb)
        hist["kl"].append(kld / nb)
        hist["kl_w"].append(float(k_w))

        logger.info(
            f"Epoch {ep+1:03}/{args.epochs} total={hist['total'][-1]:.6f} "
            f"recon={hist['recon'][-1]:.6f} kl={hist['kl'][-1]:.6f} kl_w={k_w:.3f}"
        )

    torch.save(model.state_dict(), dirs["models"] / "vae_4dof.pt")
    np.savez_compressed(dirs["models"] / "vae_norm_stats.npz", mean=mu, std=sd)
    (dirs["metrics"] / "vae_train_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    logger.info("Saved VAE + stats + history.")
    logger.info("Done.")


if __name__ == "__main__":
    main()
