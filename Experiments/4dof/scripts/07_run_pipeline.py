# experiments/4dof/scripts/07_run_pipeline.py
"""
Step 07 (4DOF): Full pipeline evaluation:
  1) VAE gate: Normal (N) vs anomaly (fault)
  2) CNN: classify anomaly as Sensor Fault (SF) vs Structural Fault (ST)

Outputs:
  outputs/metrics/pipeline_report_<split>.txt
  outputs/metrics/pipeline_summary_<split>.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.models.temporal_vae import VAE
from src.models.cnn_model import CNN
from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Z = (X - mu[None, None, :]) / sd[None, None, :]
    Z = np.clip(Z, -clip, clip)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32)


@torch.no_grad()
def vae_mse(model: VAE, Z: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval()
    out = np.zeros(Z.shape[0], dtype=np.float32)
    for i in range(0, Z.shape[0], batch):
        xb = torch.tensor(Z[i:i+batch], dtype=torch.float32, device=device)
        recon, _, _ = model(xb)
        r = recon.detach().cpu().numpy()
        out[i:i+batch] = ((Z[i:i+batch] - r) ** 2).mean(axis=(1, 2))
    return out


@torch.no_grad()
def cnn_pred(model: CNN, Z: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(Z[:, None, :, :], dtype=torch.float32)
    pred = np.zeros(Z.shape[0], dtype=np.int64)
    for i in range(0, Xt.shape[0], batch):
        xb = Xt[i:i+batch].to(device)
        pred[i:i+batch] = model(xb).argmax(1).detach().cpu().numpy()
    return pred


def encode_3(labels: pd.Series) -> np.ndarray:
    # N->0, SF->1, ST->2
    lab = labels.astype(str).to_numpy()
    y = np.zeros(len(lab), dtype=np.int64)
    y[lab == "SF"] = 1
    y[lab == "ST"] = 2
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/4dof/datasets/processed")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)
    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "07_run_pipeline.log")

    X = np.load(processed_dir / "X.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "meta_windows.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))
    eval_runs = set(splits["val_runs"] if args.split == "val" else splits["test_runs"])

    mask = meta["run_id"].isin(eval_runs) & meta["label"].isin(["N", "SF", "ST"])
    Xe = X[mask.to_numpy()]
    y_true = encode_3(meta.loc[mask, "label"])

    # --- VAE gate ---
    vae_stats = np.load(dirs["models"] / "vae_norm_stats.npz")
    vae_mu = vae_stats["mean"].astype(np.float32)
    vae_sd = vae_stats["std"].astype(np.float32)
    Zvae = standardize(Xe, vae_mu, vae_sd)

    thr = json.loads((dirs["metrics"] / "vae_threshold.json").read_text(encoding="utf-8"))["threshold"]
    vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=int(Zvae.shape[2]), latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3).to(vae_device)
    vae.load_state_dict(torch.load(dirs["models"] / "vae_4dof.pt", map_location=vae_device))

    mse = vae_mse(vae, Zvae, device=vae_device, batch=args.batch_size)
    is_fault = mse > float(thr)

    # --- CNN classification for faults (SF vs ST) ---
    cnn_stats = np.load(dirs["models"] / "cnn_norm_stats.npz")
    cnn_mu = cnn_stats["mean"].astype(np.float32)
    cnn_sd = cnn_stats["std"].astype(np.float32)
    Zcnn = standardize(Xe, cnn_mu, cnn_sd)

    cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN(input_channels=1, num_classes=2, dropout_rate=args.dropout).to(cnn_device)
    cnn.load_state_dict(torch.load(dirs["models"] / "cnn_4dof.pt", map_location=cnn_device))

    y_pred = np.zeros_like(y_true)
    y_pred[~is_fault] = 0
    if np.any(is_fault):
        pred_fault = cnn_pred(cnn, Zcnn[is_fault], device=cnn_device, batch=args.batch_size)  # 0=SF,1=ST
        y_pred[is_fault] = np.where(pred_fault == 0, 1, 2)

    names = ["Normal (N)", "Sensor Fault (SF)", "Structural Fault (ST)"]
    rep = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    (dirs["metrics"] / f"pipeline_report_{args.split}.txt").write_text(rep, encoding="utf-8")
    summary = {
        "split": args.split,
        "runs": sorted(list(eval_runs)),
        "n_eval": int(len(y_true)),
        "vae_threshold": float(thr),
        "pred_faults": int(is_fault.sum()),
        "confusion_matrix": cm.tolist(),
    }
    (dirs["metrics"] / f"pipeline_summary_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(rep)
    logger.info(f"CM:\n{cm}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
