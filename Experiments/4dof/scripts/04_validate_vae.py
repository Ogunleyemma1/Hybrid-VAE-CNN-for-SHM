# experiments/4dof/scripts/04_validate_vae.py
"""
Step 04 (4DOF): Calibrate VAE threshold on VAL normal windows.

Outputs:
  outputs/metrics/vae_threshold.json
  outputs/metrics/vae_val_mse.npy
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.models.temporal_vae import VAE
from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Z = (X - mu[None, None, :]) / sd[None, None, :]
    Z = np.clip(Z, -clip, clip)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32)


@torch.no_grad()
def mse_all(model: VAE, Z: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval()
    out = np.zeros(Z.shape[0], dtype=np.float32)
    for i in range(0, Z.shape[0], batch):
        xb = torch.tensor(Z[i:i+batch], dtype=torch.float32, device=device)
        recon, _, _ = model(xb)
        r = recon.detach().cpu().numpy()
        out[i:i+batch] = ((Z[i:i+batch] - r) ** 2).mean(axis=(1, 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/4dof/datasets/processed")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--quantile", type=float, default=0.99)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)
    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "04_validate_vae.log")

    X = np.load(processed_dir / "X.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "meta_windows.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))
    val_runs = set(splits["val_runs"])

    mask = meta["run_id"].isin(val_runs) & (meta["label"] == "N")
    Xv = X[mask.to_numpy()]
    if Xv.shape[0] == 0:
        raise RuntimeError("No VAL normal windows found. Check splits and normal dataset generation.")

    stats = np.load(dirs["models"] / "vae_norm_stats.npz")
    mu = stats["mean"].astype(np.float32)
    sd = stats["std"].astype(np.float32)

    Zv = standardize(Xv, mu, sd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=int(Zv.shape[2]), latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load(dirs["models"] / "vae_4dof.pt", map_location=device))

    mse = mse_all(model, Zv, device=device, batch=args.batch_size)
    thr = float(np.quantile(mse, args.quantile))

    out = {
        "split": "val",
        "label": "N",
        "threshold": thr,
        "quantile": float(args.quantile),
        "n_val": int(len(mse)),
        "mse_mean": float(np.mean(mse)),
        "mse_p95": float(np.quantile(mse, 0.95)),
        "mse_p99": float(np.quantile(mse, 0.99)),
    }

    np.save(dirs["metrics"] / "vae_val_mse.npy", mse.astype(np.float32))
    (dirs["metrics"] / "vae_threshold.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info(out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
