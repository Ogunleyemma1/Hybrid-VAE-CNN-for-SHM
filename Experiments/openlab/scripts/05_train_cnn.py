# experiments/openlab/scripts/05_train_cnn.py
"""
Step 05: Train CNN on TRAIN runs using SF/E windows (weak supervision allowed),
and save best model + threshold tuned on VAL(GOLD) with SF objective.

Inputs:
  - processed/X_raw.npy
  - processed/window_labels_augmented.csv
  - processed/run_split.json

Outputs:
  - outputs/models/cnn_openlab.pt
  - outputs/models/cnn_norm_stats.npz
  - outputs/metrics/cnn_threshold.npy
  - outputs/metrics/cnn_training_info.json
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

from src.models.cnn_model import CNN
from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root, set_seed


def fit_mu_sd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 1)).astype(np.float32)
    sd = X.std(axis=(0, 1)).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return mu, sd


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    x = (X - mu[None, None, :]) / sd[None, None, :]
    x = np.clip(x, -clip, clip)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def augment_sf(x: np.ndarray) -> np.ndarray:
    """
    Conservative SF augmentation: shifts/noise/spikes/segment drop.
    Applied only to SF class during training.
    """
    T, C = x.shape
    y = x.copy()

    if np.random.rand() < 0.4:
        shift = np.random.randint(-8, 9)
        y = np.roll(y, shift, axis=0)

    if np.random.rand() < 0.6:
        y = y + np.random.normal(0.0, 0.15, size=y.shape).astype(np.float32)

    if np.random.rand() < 0.5:
        n_spikes = np.random.randint(1, 6)
        for _ in range(n_spikes):
            t0 = np.random.randint(0, T)
            c0 = np.random.randint(0, C)
            y[t0, c0] += np.random.uniform(2.0, 6.0) * np.random.choice([-1.0, 1.0])

    if np.random.rand() < 0.5:
        seg_len = np.random.randint(10, 50)
        t0 = np.random.randint(0, max(1, T - seg_len))
        y[t0:t0 + seg_len, :] = 0.0

    return y.astype(np.float32)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        yi = int(self.y[idx])

        # 0=SF, 1=E
        if self.augment and yi == 0:
            x = augment_sf(x)

        x = torch.tensor(x[None, :, :], dtype=torch.float32)  # (1,T,C)
        y = torch.tensor(yi, dtype=torch.long)
        return x, y


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()


def fbeta(prec: float, rec: float, beta: float = 2.0) -> float:
    b2 = beta * beta
    denom = (b2 * prec + rec)
    if denom <= 1e-12:
        return 0.0
    return (1.0 + b2) * (prec * rec) / denom


def select_threshold_on_gold(y_true: np.ndarray, probs_e: np.ndarray, pmin: float = 0.60, grid: int = 99):
    """
    Choose threshold t maximizing SF recall subject to SF precision >= pmin.
    Tie-break by SF-F2.
    """
    best = {"t": 0.5, "sf_rec": -1.0, "sf_prec": -1.0, "sf_f2": -1.0}
    fallback = {"t": 0.5, "sf_rec": -1.0, "sf_prec": -1.0, "sf_f2": -1.0}

    for t in np.linspace(0.01, 0.99, grid):
        y_pred = (probs_e >= t).astype(int)  # 1=E, 0=SF

        y_true_sf = (y_true == 0).astype(int)
        y_pred_sf = (y_pred == 0).astype(int)

        sf_prec = precision_score(y_true_sf, y_pred_sf, zero_division=0)
        sf_rec = recall_score(y_true_sf, y_pred_sf, zero_division=0)
        sf_f2 = fbeta(sf_prec, sf_rec, beta=2.0)

        if (sf_rec > fallback["sf_rec"]) or (sf_rec == fallback["sf_rec"] and sf_f2 > fallback["sf_f2"]):
            fallback = {"t": float(t), "sf_rec": float(sf_rec), "sf_prec": float(sf_prec), "sf_f2": float(sf_f2)}

        if sf_prec >= pmin:
            if (sf_rec > best["sf_rec"]) or (sf_rec == best["sf_rec"] and sf_f2 > best["sf_f2"]):
                best = {"t": float(t), "sf_rec": float(sf_rec), "sf_prec": float(sf_prec), "sf_f2": float(sf_f2)}

    used_fallback = (best["sf_rec"] < 0.0)
    chosen = fallback if used_fallback else best
    return chosen["t"], chosen["sf_rec"], chosen["sf_prec"], chosen["sf_f2"], used_fallback


def encode_sf_e(labels: pd.Series) -> np.ndarray:
    # SF -> 0, E -> 1
    lab = labels.astype(str).to_numpy()
    if not np.all(np.isin(lab, ["SF", "E"])):
        bad = sorted(set(lab) - set(["SF", "E"]))
        raise ValueError(f"Unexpected labels in SF/E subset: {bad}")
    return np.where(lab == "SF", 0, 1).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--p_min", type=float, default=0.60)
    ap.add_argument("--augment_sf", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "05_train_cnn.log")
    set_seed(args.seed, deterministic_torch=True)

    X = np.load(processed_dir / "X_raw.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "window_labels_augmented.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))

    train_runs = set(splits["train_runs"])
    val_runs = set(splits["val_runs"])

    is_se = meta["label"].isin(["SF", "E"])
    train_mask = meta["run_id"].isin(train_runs) & is_se
    val_gold_mask = meta["run_id"].isin(val_runs) & is_se & (meta["label_source"] == "gold")

    if int(val_gold_mask.sum()) == 0:
        raise RuntimeError("VAL(GOLD) SF/E is empty. Threshold calibration not defensible.")

    X_train_raw = X[train_mask.to_numpy()]
    X_val_raw = X[val_gold_mask.to_numpy()]

    y_train = encode_sf_e(meta.loc[train_mask, "label"])
    y_val = encode_sf_e(meta.loc[val_gold_mask, "label"])

    mu, sd = fit_mu_sd(X_train_raw)
    X_train = apply_standardize(X_train_raw, mu, sd)
    X_val = apply_standardize(X_val_raw, mu, sd)

    # Save norm stats for pipeline
    norm_path = dirs["models"] / "cnn_norm_stats.npz"
    np.savez_compressed(norm_path, mean=mu, std=sd)

    train_ds = WindowDataset(X_train, y_train, augment=args.augment_sf)
    val_ds = WindowDataset(X_val, y_val, augment=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=max(args.batch_size, 256), shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_channels=1, num_classes=2, dropout_rate=args.dropout).to(device)

    # class weights -> focal alpha
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    w = (counts.sum() / np.maximum(counts, 1.0))
    w = w / w.sum() * 2.0
    alpha = torch.tensor(w, dtype=torch.float32, device=device)

    loss_fn = WeightedFocalLoss(alpha=alpha, gamma=2.0)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_sf_rec = -1.0
    best_thr = 0.5
    best_f2 = 0.0
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_sum += float(loss.item())
        tr_loss = tr_sum / max(1, len(train_loader))

        model.eval()
        all_logits = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                all_logits.append(model(xb).detach().cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        probs_e = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()

        thr, sf_rec, sf_prec, sf_f2, used_fallback = select_threshold_on_gold(
            y_val, probs_e, pmin=args.p_min, grid=99
        )

        improved = (sf_rec > best_sf_rec + 1e-6) or (abs(sf_rec - best_sf_rec) <= 1e-6 and sf_f2 > best_f2 + 1e-6)
        if improved:
            best_sf_rec = sf_rec
            best_thr = thr
            best_f2 = sf_f2
            no_improve = 0
            torch.save(model.state_dict(), dirs["models"] / "cnn_openlab.pt")
            np.save(dirs["metrics"] / "cnn_threshold.npy", np.array([best_thr], dtype=np.float32))
        else:
            no_improve += 1

        sched.step()

        fb = " FALLBACK" if used_fallback else ""
        logger.info(
            f"Epoch {ep:03}/{args.epochs} | train_loss={tr_loss:.6f} | "
            f"thr={thr:.2f} | SF_rec={sf_rec:.3f} SF_prec={sf_prec:.3f} SF_F2={sf_f2:.3f}{fb} | "
            f"best_rec={best_sf_rec:.3f} | no_improve={no_improve}/{args.patience}"
        )

        if no_improve >= args.patience:
            logger.info("Early stopping.")
            break

    # Save reproducibility info
    info = {
        "seed": int(args.seed),
        "augment_sf": bool(args.augment_sf),
        "p_min": float(args.p_min),
        "best_threshold": float(best_thr),
        "best_sf_recall": float(best_sf_rec),
        "best_sf_f2": float(best_f2),
        "train_runs": sorted(list(train_runs)),
        "val_runs": sorted(list(val_runs)),
    }
    (dirs["metrics"] / "cnn_training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Final report on VAL(GOLD)
    model.load_state_dict(torch.load(dirs["models"] / "cnn_openlab.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        all_logits = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            all_logits.append(model(xb).detach().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    probs_e = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
    y_pred = (probs_e >= best_thr).astype(int)

    rep = classification_report(y_val, y_pred, target_names=["SF", "E"], zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    logger.info("VAL(GOLD) report:\n" + rep)
    logger.info(f"VAL(GOLD) CM:\n{cm}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
