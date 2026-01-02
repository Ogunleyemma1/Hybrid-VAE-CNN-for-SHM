"""
experiments/1dof/scripts/01_train_vae.py

Train the Temporal VAE on NORMAL windows only.
Saves:
- outputs/models/vae_1dof.pt
- outputs/models/vae_norm_stats.npz
- outputs/metrics/vae_train_history.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.seed import set_seed
from src.models.temporal_vae import VAE

from scripts._helpers.io_local import ensure_dir, save_npz
from scripts._helpers.viz_local import set_publication_style


def _fit_mu_sd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 1)).astype(np.float32)
    sd = X.std(axis=(0, 1)).astype(np.float32)
    sd = np.where(sd < 1e-12, 1.0, sd).astype(np.float32)
    return mu, sd


def _standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Z = (X - mu[None, None, :]) / sd[None, None, :]
    Z = np.clip(Z, -clip, clip)
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _kl_weight(epoch: int, n_epochs: int, ratio: float = 0.30) -> float:
    # smooth logistic ramp
    t0 = int(n_epochs * ratio)
    x = (epoch - t0) / max(t0, 1)
    return float(1.0 / (1.0 + np.exp(-5.0 * x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)

    # match your manuscript defaults (adjust if needed)
    ap.add_argument("--latent_dim", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    set_publication_style()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("experiments/1dof/datasets/processed")
    out_dir = Path("experiments/1dof/outputs")
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)

    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy", allow_pickle=True)

    mask = (y_train == "normal")
    Xn = X_train[mask]
    if Xn.shape[0] < 20:
        raise RuntimeError("Too few normal windows. Increase normal runs or adjust windowing parameters.")

    mu, sd = _fit_mu_sd(Xn)
    Zn = _standardize(Xn, mu, sd)

    # Save stats for exact reproducibility
    save_npz(models_dir / "vae_norm_stats.npz", mean=mu, std=sd)

    model = VAE(
        input_dim=int(Zn.shape[-1]),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = torch.utils.data.TensorDataset(torch.tensor(Zn, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    hist = []
    for ep in range(args.epochs):
        model.train()
        kw = _kl_weight(ep, args.epochs, ratio=0.30)

        total = recon_sum = kl_sum = 0.0
        nb = 0

        for (xb,) in dl:
            xb = xb.to(device)
            recon, mu_t, logvar = model(xb)

            recon_loss = nn.functional.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu_t.pow(2) - logvar.exp())
            loss = recon_loss + kw * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            total += float(loss.item())
            recon_sum += float(recon_loss.item())
            kl_sum += float(kl.item())
            nb += 1

        hist.append(
            {"epoch": ep + 1, "total": total / nb, "recon": recon_sum / nb, "kl": kl_sum / nb, "kl_weight": kw}
        )

    pd.DataFrame(hist).to_csv(metrics_dir / "vae_train_history.csv", index=False)
    torch.save(model.state_dict(), models_dir / "vae_1dof.pt")


if __name__ == "__main__":
    main()
