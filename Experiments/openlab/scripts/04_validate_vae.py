# experiments/openlab/scripts/04_validate_vae.py
"""
Step 04: Validate VAE gate and compute an MSE threshold on VAL(GOLD) Normal windows.

Inputs:
  - processed/X_clean.npy
  - processed/window_labels_augmented.csv
  - processed/run_split.json
  - outputs/models/vae_openlab.pt
  - outputs/models/vae_norm_stats.npz

Outputs:
  - outputs/metrics/vae_threshold.json
  - outputs/metrics/vae_val_mse.npy
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.models.temporal_vae import VAE
from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root, set_seed


@torch.no_grad()
def compute_mse(model: VAE, Xn: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    model.eval()
    N = Xn.shape[0]
    mse = np.zeros(N, dtype=np.float32)
    for i in range(0, N, batch):
        xb = torch.tensor(Xn[i:i+batch], dtype=torch.float32, device=device)
        recon, _, _ = model(xb)
        r = recon.detach().cpu().numpy()
        e = (Xn[i:i+batch] - r) ** 2
        mse[i:i+batch] = e.mean(axis=(1, 2))
    return mse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold_quantile", type=float, default=0.99)
    ap.add_argument("--clip_z", type=float, default=10.0)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "04_validate_vae.log")

    set_seed(args.seed, deterministic_torch=True)

    x_path = processed_dir / "X_clean.npy"
    meta_path = processed_dir / "window_labels_augmented.csv"
    split_path = processed_dir / "run_split.json"

    X = np.load(x_path).astype(np.float32)
    meta = pd.read_csv(meta_path)
    splits = json.loads(split_path.read_text(encoding="utf-8"))

    val_runs = set(splits["val_runs"])
    val_mask = (meta["run_id"].isin(val_runs)) & (meta["label"] == "N") & (meta["label_source"] == "gold")
    if int(val_mask.sum()) == 0:
        raise RuntimeError("No VAL(GOLD) normal windows available. Threshold calibration not defensible.")

    X_val = X[val_mask.to_numpy()]

    # Load norm stats + model
    norm = np.load(dirs["models"] / "vae_norm_stats.npz")
    mu = norm["mean"].astype(np.float32)
    sd = norm["std"].astype(np.float32)
    sd = np.where(sd < 1e-12, 1.0, sd).astype(np.float32)

    Xn = (X_val - mu[None, None, :]) / sd[None, None, :]
    Xn = np.clip(Xn, -args.clip_z, args.clip_z).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=int(X.shape[2]), latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load(dirs["models"] / "vae_openlab.pt", map_location=device))

    mse = compute_mse(model, Xn, device=device, batch=args.batch_size)

    thr = float(np.quantile(mse, args.threshold_quantile))
    out = {
        "split": "val",
        "label_source": "gold",
        "label": "N",
        "threshold": thr,
        "threshold_quantile": float(args.threshold_quantile),
        "n_val": int(len(mse)),
        "mse_mean": float(np.mean(mse)),
        "mse_std": float(np.std(mse)),
        "mse_p95": float(np.quantile(mse, 0.95)),
        "mse_p99": float(np.quantile(mse, 0.99)),
    }

    (dirs["metrics"] / "vae_val_mse.npy").write_bytes(np.asarray(mse, dtype=np.float32).tobytes())
    (dirs["metrics"] / "vae_threshold.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info(f"Saved VAE threshold: {dirs['metrics'] / 'vae_threshold.json'}")
    logger.info(out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
