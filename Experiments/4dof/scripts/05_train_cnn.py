# experiments/4dof/scripts/05_train_cnn.py
"""
Step 05 (4DOF): Train CNN classifier on fault windows:
  SF (sensor fault) vs ST (structural fault)

Outputs:
  outputs/models/cnn_4dof.pt
  outputs/models/cnn_norm_stats.npz
  outputs/metrics/cnn_training_info.json
  outputs/metrics/cnn_val_report.txt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from src.models.cnn_model import CNN
from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root, set_seed


def fit_mu_sd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 1)).astype(np.float32)
    sd = X.std(axis=(0, 1)).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return mu, sd


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Z = (X - mu[None, None, :]) / sd[None, None, :]
    Z = np.clip(Z, -clip, clip)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32)


def encode_sf_st(labels: pd.Series) -> np.ndarray:
    lab = labels.astype(str).to_numpy()
    if not np.all(np.isin(lab, ["SF", "ST"])):
        bad = sorted(set(lab) - set(["SF", "ST"]))
        raise ValueError(f"Unexpected labels for CNN (expected SF/ST): {bad}")
    return np.where(lab == "SF", 0, 1).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/4dof/datasets/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "05_train_cnn.log")
    set_seed(args.seed, deterministic_torch=True)

    X = np.load(processed_dir / "X.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "meta_windows.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))

    train_runs = set(splits["train_runs"])
    val_runs = set(splits["val_runs"])

    train_mask = meta["run_id"].isin(train_runs) & meta["label"].isin(["SF", "ST"])
    val_mask = meta["run_id"].isin(val_runs) & meta["label"].isin(["SF", "ST"])

    Xtr = X[train_mask.to_numpy()]
    ytr = encode_sf_st(meta.loc[train_mask, "label"])
    Xva = X[val_mask.to_numpy()]
    yva = encode_sf_st(meta.loc[val_mask, "label"])

    if Xtr.shape[0] == 0 or Xva.shape[0] == 0:
        raise RuntimeError("CNN requires fault windows in both train and val splits (SF/ST).")

    mu, sd = fit_mu_sd(Xtr)
    Ztr = standardize(Xtr, mu, sd)
    Zva = standardize(Xva, mu, sd)

    np.savez_compressed(dirs["models"] / "cnn_norm_stats.npz", mean=mu, std=sd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_channels=1, num_classes=2, dropout_rate=args.dropout).to(device)

    # class weights
    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    w = (counts.sum() / np.maximum(counts, 1.0))
    w = w / w.sum() * 2.0
    w_t = torch.tensor(w, dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=w_t)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Ztr[:, None, :, :]), torch.tensor(ytr)),
        batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Zva[:, None, :, :]), torch.tensor(yva)),
        batch_size=max(args.batch_size, 256), shuffle=False, drop_last=False
    )

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb.to(device))
                va_loss += float(loss.item())
                y_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())
                y_true.extend(yb.numpy().tolist())
        va_loss /= max(1, len(val_loader))

        logger.info(f"Epoch {ep:03}/{args.epochs} | train={tr_loss:.6f} | val={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), dirs["models"] / "cnn_4dof.pt")

    rep = classification_report(y_true, y_pred, target_names=["SF", "ST"], zero_division=0)
    (dirs["metrics"] / "cnn_val_report.txt").write_text(rep, encoding="utf-8")

    info = {
        "seed": int(args.seed),
        "best_val_loss": float(best_val),
        "train_runs": sorted(list(train_runs)),
        "val_runs": sorted(list(val_runs)),
    }
    (dirs["metrics"] / "cnn_training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    logger.info("Saved cnn_4dof.pt, cnn_norm_stats.npz, reports.")
    logger.info("Done.")


if __name__ == "__main__":
    main()
