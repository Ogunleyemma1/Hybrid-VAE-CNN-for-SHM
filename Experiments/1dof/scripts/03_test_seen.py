"""
experiments/1dof/scripts/03_test_seen.py

Seen test evaluation:
- reconstruction MSE and latent embeddings (z := mu)
- PCA scatter plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.models.temporal_vae import VAE
from src.utils.seed import set_seed

from scripts._helpers.io_local import ensure_dir
from scripts._helpers.viz_local import set_publication_style, save_pdf_svg
from scripts._helpers.viz_local import save_pdf_svg as _save


def _standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Z = (X - mu[None, None, :]) / sd[None, None, :]
    Z = np.clip(Z, -clip, clip)
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def _mse_and_latent(model: VAE, Z: np.ndarray, device: torch.device, batch: int):
    model.eval()
    mse = np.zeros(Z.shape[0], dtype=np.float32)
    z_all = []
    for i in range(0, Z.shape[0], batch):
        xb = torch.tensor(Z[i : i + batch], dtype=torch.float32, device=device)
        recon, mu_t, logvar = model(xb)
        r = recon.detach().cpu().numpy()
        mse[i : i + batch] = ((Z[i : i + batch] - r) ** 2).mean(axis=(1, 2))
        z_all.append(mu_t.detach().cpu().numpy())
    return mse, np.vstack(z_all).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    set_publication_style()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("experiments/1dof/datasets/processed")
    out_dir = Path("experiments/1dof/outputs")
    fig_lat = out_dir / "figures" / "latent"
    fig_vae = out_dir / "figures" / "vae"
    metrics_dir = out_dir / "metrics"
    models_dir = out_dir / "models"
    ensure_dir(fig_lat)
    ensure_dir(fig_vae)
    ensure_dir(metrics_dir)

    Xt = np.load(data_dir / "X_test.npy").astype(np.float32)
    yt = np.load(data_dir / "y_test.npy", allow_pickle=True)
    meta = pd.read_csv(data_dir / "meta_test.csv")

    stats = np.load(models_dir / "vae_norm_stats.npz")
    mu, sd = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    Zt = _standardize(Xt, mu, sd)

    model = VAE(input_dim=int(Zt.shape[-1]), latent_dim=3, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(models_dir / "vae_1dof.pt", map_location=device))

    mse, z = _mse_and_latent(model, Zt, device=device, batch=args.batch_size)

    out = meta.copy()
    out["label"] = yt
    out["mse"] = mse
    for j in range(z.shape[1]):
        out[f"z{j+1}"] = z[:, j]
    out.to_csv(metrics_dir / "vae_test_seen_scores.csv", index=False)

    # MSE hist
    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(111)
    for lab in ["normal", "drifted", "noisy"]:
        vals = out.loc[out["label"] == lab, "mse"].to_numpy()
        if len(vals) > 0:
            ax.hist(vals, bins=60, histtype="step", density=True, label=lab)
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    save_pdf_svg(fig, fig_vae / "vae_test_seen_mse_hist.pdf")

    # PCA scatter
    pca = PCA(n_components=2, random_state=args.seed)
    z2 = pca.fit_transform(z)

    fig = plt.figure(figsize=(6.5, 5.0))
    ax = fig.add_subplot(111)
    for lab in ["normal", "drifted", "noisy"]:
        m = (out["label"].to_numpy() == lab)
        if np.any(m):
            ax.scatter(z2[m, 0], z2[m, 1], s=10, alpha=0.75, label=lab)
    ax.set_xlabel("PC1 (latent)")
    ax.set_ylabel("PC2 (latent)")
    ax.legend(frameon=True)
    save_pdf_svg(fig, fig_lat / "latent_pca_test_seen.pdf")


if __name__ == "__main__":
    main()
