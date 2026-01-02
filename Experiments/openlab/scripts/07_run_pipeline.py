# experiments/openlab/scripts/07_run_pipeline.py
"""
Step 07: Full pipeline evaluation on VAL(GOLD) or TEST(GOLD):
  VAE gate (N vs anomaly) -> CNN classifier (SF vs E) for anomalies.

Inputs:
  - processed/X_raw.npy
  - processed/window_labels_augmented.csv
  - processed/run_split.json
  - outputs/models/vae_openlab.pt + vae_norm_stats.npz
  - outputs/metrics/vae_threshold.json
  - outputs/models/cnn_openlab.pt + cnn_norm_stats.npz
  - outputs/metrics/cnn_threshold.npy

Outputs:
  - outputs/metrics/pipeline_summary_<split>_gold.json
  - outputs/metrics/pipeline_report_<split>_gold.txt
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


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    x = (X - mu[None, None, :]) / sd[None, None, :]
    x = np.clip(x, -clip, clip)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


@torch.no_grad()
def vae_mse(model: VAE, Xn: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
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


@torch.no_grad()
def cnn_pE(model: CNN, Xstd: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(Xstd[:, None, :, :], dtype=torch.float32)
    out = []
    for i in range(0, Xt.shape[0], batch):
        xb = Xt[i:i+batch].to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1]
        out.append(p.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def encode_labels_3(labels: pd.Series) -> np.ndarray:
    # N->0, SF->1, E->2
    lab = labels.astype(str).to_numpy()
    if not np.all(np.isin(lab, ["N", "SF", "E"])):
        bad = sorted(set(lab) - set(["N", "SF", "E"]))
        raise ValueError(f"Unexpected labels: {bad}")
    y = np.zeros(len(lab), dtype=np.int64)
    y[lab == "SF"] = 1
    y[lab == "E"] = 2
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--split", type=str, choices=["val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "07_run_pipeline.log")

    X_raw = np.load(processed_dir / "X_raw.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "window_labels_augmented.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))

    eval_runs = set(splits["val_runs"] if args.split == "val" else splits["test_runs"])
    mask = meta["run_id"].isin(eval_runs) & (meta["label_source"] == "gold") & (meta["label"].isin(["N", "SF", "E"]))
    if int(mask.sum()) == 0:
        raise RuntimeError(f"No {args.split.upper()}(GOLD) windows available for N/SF/E.")

    X_eval = X_raw[mask.to_numpy()]
    y_true = encode_labels_3(meta.loc[mask, "label"])

    # ----- VAE gate -----
    vae_thr_obj = json.loads((dirs["metrics"] / "vae_threshold.json").read_text(encoding="utf-8"))
    mse_thr = float(vae_thr_obj["threshold"])

    vae_norm = np.load(dirs["models"] / "vae_norm_stats.npz")
    vae_mu = vae_norm["mean"].astype(np.float32)
    vae_sd = vae_norm["std"].astype(np.float32)
    vae_sd = np.where(vae_sd < 1e-12, 1.0, vae_sd).astype(np.float32)

    # Gate uses first 3 channels (consistent with X_clean policy)
    X_gate = X_eval[:, :, : len(vae_mu)]
    X_gate_n = apply_standardize(X_gate, vae_mu, vae_sd, clip=10.0)

    vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=int(X_gate_n.shape[2]), latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3).to(vae_device)
    vae.load_state_dict(torch.load(dirs["models"] / "vae_openlab.pt", map_location=vae_device))

    mse = vae_mse(vae, X_gate_n, device=vae_device, batch=args.batch_size)
    is_anom = (mse > mse_thr)

    # ----- CNN classifier for anomalies -----
    cnn_norm = np.load(dirs["models"] / "cnn_norm_stats.npz")
    cnn_mu = cnn_norm["mean"].astype(np.float32)
    cnn_sd = cnn_norm["std"].astype(np.float32)
    thr_cnn = float(np.load(dirs["metrics"] / "cnn_threshold.npy").ravel()[0])

    cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN(input_channels=1, num_classes=2, dropout_rate=args.dropout).to(cnn_device)
    cnn.load_state_dict(torch.load(dirs["models"] / "cnn_openlab.pt", map_location=cnn_device))

    y_pred = np.zeros_like(y_true, dtype=np.int64)  # default normal
    pE_all = np.zeros(len(y_true), dtype=np.float32)

    if np.any(is_anom):
        X_an = X_eval[is_anom]
        X_an_std = apply_standardize(X_an, cnn_mu, cnn_sd, clip=10.0)
        pE = cnn_pE(cnn, X_an_std, device=cnn_device, batch=args.batch_size)
        pE_all[is_anom] = pE
        y_pred[is_anom] = np.where(pE >= thr_cnn, 2, 1)  # E if pE>=thr else SF

    names = ["Normal (N)", "Sensor Fault (SF)", "Exceedance (E)"]
    rep = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    report_path = dirs["metrics"] / f"pipeline_report_{args.split}_gold.txt"
    report_path.write_text(rep + "\n\nCM:\n" + np.array2string(cm), encoding="utf-8")

    summary = {
        "split": args.split,
        "runs": sorted(list(eval_runs)),
        "n_eval": int(len(y_true)),
        "n_N": int((y_true == 0).sum()),
        "n_SF": int((y_true == 1).sum()),
        "n_E": int((y_true == 2).sum()),
        "vae_mse_threshold": float(mse_thr),
        "pred_anomalies": int(is_anom.sum()),
        "cnn_threshold": float(thr_cnn),
        "confusion_matrix": cm.tolist(),
    }
    sum_path = dirs["metrics"] / f"pipeline_summary_{args.split}_gold.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(rep)
    logger.info(f"CM:\n{cm}")
    logger.info(f"Saved: {report_path}")
    logger.info(f"Saved: {sum_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
