"""
06_train_cnn.py

Train a CNN classifier to separate Sensor Fault (SF) vs Structural Fault (ST) using RAW windows.

ST-leaning policy (requested):
- Training still uses WeightedRandomSampler + class-weighted focal loss.
- Model selection is ST-first (minority class):
    * Tune threshold on VAL to maximize ST recall subject to ST precision >= P_MIN_ST
    * Tie-break by ST-F2 (beta=BETA_FOR_F2_ST)
- Best checkpoint is selected using ST-F2 on VAL at its tuned threshold.

Inputs (from Data/extracted)
- X_raw.npy
- window_labels.csv
- run_split.json

Outputs (to Output/CNN_Training)
- artifacts/: best model checkpoint, normalization stats, training info json
- plots/    : loss curves (pdf/svg/png)
"""

from __future__ import annotations

import inspect
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import config as C
from io_utils import ensure_dir, save_json
from Models.cnn_model import CNN, SEQ_LEN, NUM_FEATURES


# =============================================================================
# Training settings
# =============================================================================
SEED = getattr(C, "SEED", 42)

BATCH_SIZE = 128
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
PATIENCE = 25
MAX_GRAD_NORM = 2.0
CLIP_Z = 10.0

LABEL_SF = "Sensor Fault"       # class 0
LABEL_ST = "Structural Fault"   # class 1

# -------------------------------
# ST-leaning threshold policy
# -------------------------------
THRESH_GRID = 99

# Choose threshold to favor ST recovery (minority class)
P_MIN_ST = 0.25          # ST precision floor (avoid trivial "everything ST")
BETA_FOR_F2_ST = 2.0     # F2 emphasizes recall (ST recall is the priority)

# Optional (usually keep disabled). If you want to prevent SF collapsing completely, set e.g. 0.40
MIN_PREC_SF = 0.00       # SF precision floor (0 disables)


# =============================================================================
# Paths (robust to your current layout)
# =============================================================================
def _artifact_path(key: str, fallback_name: str) -> str:
    if hasattr(C, "ARTIFACTS") and key in C.ARTIFACTS:
        return os.path.join(C.OUT_DIR, C.ARTIFACTS[key])
    return os.path.join(C.OUT_DIR, fallback_name)


X_RAW_PATH = _artifact_path("windows_raw", "X_raw.npy")
META_PATH  = _artifact_path("meta", "window_labels.csv")
SPLIT_PATH = _artifact_path("splits", "run_split.json")

OUTPUT_ROOT = os.path.join(C.PROJECT_DIR, "Output")
EXP_DIR = os.path.join(OUTPUT_ROOT, "CNN_Training")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
ARTIFACTS_DIR = os.path.join(EXP_DIR, "artifacts")
ensure_dir(PLOTS_DIR)
ensure_dir(ARTIFACTS_DIR)

BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "cnn_model_openlab.pt")
NORM_STATS_PATH = os.path.join(ARTIFACTS_DIR, "cnn_raw_mu_sd.npy")
INFO_PATH = os.path.join(ARTIFACTS_DIR, "cnn_training_info.json")

PLOT_BASE = "cnn_train_val_loss"
PLOT_PDF = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.pdf")
PLOT_SVG = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.svg")
PLOT_PNG = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.png")


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Standardization (train-only)
# =============================================================================
def fit_mu_sd(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = X_train.astype(np.float32)
    mu = np.mean(x, axis=(0, 1)).astype(np.float32)
    sd = np.std(x, axis=(0, 1)).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return mu, sd


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float) -> np.ndarray:
    x = X.astype(np.float32)
    x = (x - mu[None, None, :]) / sd[None, None, :]
    x = np.clip(x, -float(clip), float(clip))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx][None, :, :], dtype=torch.float32)  # (1,T,C)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_meta_columns(meta: pd.DataFrame) -> None:
    need = ["run_id", "label"]
    miss = [c for c in need if c not in meta.columns]
    if miss:
        raise ValueError(f"Meta file missing columns {miss}. Available: {list(meta.columns)}")


def label_to_binary(label: str):
    s = str(label).strip().lower()
    if s == "sensor fault":
        return 0
    if s == "structural fault":
        return 1
    return None


def filter_split_sf_st(
    X_raw: np.ndarray, meta: pd.DataFrame, split: Dict, split_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    runs = split[f"{split_name}_runs"]
    m_run = meta["run_id"].astype(str).isin([str(r) for r in runs])

    y_bin = meta["label"].apply(label_to_binary)
    m_cls = y_bin.notna()

    idx = np.where((m_run & m_cls).to_numpy())[0]
    X = X_raw[idx]
    y = y_bin.iloc[idx].to_numpy(dtype=np.int64)

    return X, y


# =============================================================================
# Loss (Weighted Focal)
# =============================================================================
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        at = self.alpha[targets]
        loss = at * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def fbeta(prec: float, rec: float, beta: float) -> float:
    if prec <= 0 or rec <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


@torch.no_grad()
def predict_proba_st(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def _prec_rec_for_class(y_true: np.ndarray, yhat: np.ndarray, cls: int) -> Tuple[float, float]:
    yt = (y_true == cls).astype(int)
    yp = (yhat == cls).astype(int)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    return float(prec), float(rec)


def tune_threshold_val_st_first(p_st: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    Decision rule: predict ST if p(ST) >= t else SF.

    Objective (ST-first):
      1) enforce ST precision >= P_MIN_ST
      2) optional enforce SF precision >= MIN_PREC_SF
      3) maximize ST recall
      4) tie-break by ST-F2 (beta=BETA_FOR_F2_ST)
      5) secondary tie-break by macro-F1

    If no threshold meets ST precision floor, fall back to best ST-F2 overall.
    """
    ts = np.linspace(0.01, 0.99, THRESH_GRID)

    best = None
    fallback = None

    for t in ts:
        yhat = (p_st >= t).astype(int)  # 1=ST, 0=SF

        prec_sf, rec_sf = _prec_rec_for_class(y_true, yhat, cls=0)
        prec_st, rec_st = _prec_rec_for_class(y_true, yhat, cls=1)

        f2_st = fbeta(prec_st, rec_st, beta=BETA_FOR_F2_ST)
        macro_f1 = f1_score(y_true, yhat, average="macro", zero_division=0)

        cand = {
            "t": float(t),
            "prec_sf": float(prec_sf),
            "rec_sf": float(rec_sf),
            "prec_st": float(prec_st),
            "rec_st": float(rec_st),
            "f2_st": float(f2_st),
            "macro_f1": float(macro_f1),
            "meets_prec_st": bool(prec_st >= float(P_MIN_ST)),
            "meets_prec_sf": bool(prec_sf >= float(MIN_PREC_SF)) if MIN_PREC_SF > 0 else True,
        }

        # Fallback: best f2_st overall
        if (fallback is None) or (cand["f2_st"] > fallback["f2_st"]):
            fallback = cand

        # Constraint satisfaction
        ok = cand["meets_prec_st"] and cand["meets_prec_sf"]

        if best is None:
            best = cand
            best["meets_constraints"] = bool(ok)
            continue

        best_ok = best.get("meets_constraints", False)
        cand_ok = ok

        # Prefer thresholds that satisfy constraints
        if cand_ok and not best_ok:
            best = cand
            best["meets_constraints"] = True
            continue

        if cand_ok == best_ok:
            # Primary: maximize ST recall
            if cand["rec_st"] > best["rec_st"]:
                best = cand
                best["meets_constraints"] = bool(cand_ok)
                continue
            # Tie-break: maximize f2_st
            if cand["rec_st"] == best["rec_st"] and cand["f2_st"] > best["f2_st"]:
                best = cand
                best["meets_constraints"] = bool(cand_ok)
                continue
            # Secondary tie-break: macro F1
            if cand["rec_st"] == best["rec_st"] and cand["f2_st"] == best["f2_st"] and cand["macro_f1"] > best["macro_f1"]:
                best = cand
                best["meets_constraints"] = bool(cand_ok)
                continue

    # If best never met constraints, use fallback
    if not best.get("meets_constraints", False):
        out = dict(fallback)
        out["used_fallback"] = True
        out["meets_constraints"] = False
        return out

    best["used_fallback"] = False
    return best


def build_cnn_model(dropout: float, device: str) -> nn.Module:
    """
    Robust CNN construction:
    - If CNN accepts dropout/dropout_rate, pass it.
    - Otherwise instantiate with no kwargs.
    """
    sig = inspect.signature(CNN.__init__)
    kwargs = {}
    if "dropout" in sig.parameters:
        kwargs["dropout"] = dropout
    elif "dropout_rate" in sig.parameters:
        kwargs["dropout_rate"] = dropout
    return CNN(**kwargs).to(device)


def main() -> None:
    set_seed(SEED)

    if not os.path.isfile(X_RAW_PATH):
        raise FileNotFoundError(f"Missing X_raw.npy at: {X_RAW_PATH}")
    if not os.path.isfile(META_PATH):
        raise FileNotFoundError(f"Missing window_labels.csv at: {META_PATH}")
    if not os.path.isfile(SPLIT_PATH):
        raise FileNotFoundError(f"Missing run_split.json at: {SPLIT_PATH}")

    X_raw = np.load(X_RAW_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH)
    split = _read_json(SPLIT_PATH)

    require_meta_columns(meta)

    if X_raw.shape[1] != SEQ_LEN or X_raw.shape[2] != NUM_FEATURES:
        raise RuntimeError(
            f"Data shape mismatch:\n"
            f"  X_raw has (T,C)=({X_raw.shape[1]},{X_raw.shape[2]})\n"
            f"  CNN expects (SEQ_LEN,NUM_FEATURES)=({SEQ_LEN},{NUM_FEATURES})\n\n"
            f"Fix: update Models/cnn_model.py constants to SEQ_LEN=200 and NUM_FEATURES=4."
        )

    Xtr, ytr = filter_split_sf_st(X_raw, meta, split, "train")
    Xva, yva = filter_split_sf_st(X_raw, meta, split, "val")

    print(f"Train windows: {len(ytr)} (SF={(ytr==0).sum()}, ST={(ytr==1).sum()})")
    print(f"Val windows  : {len(yva)} (SF={(yva==0).sum()}, ST={(yva==1).sum()})")

    mu, sd = fit_mu_sd(Xtr)
    np.save(NORM_STATS_PATH, np.stack([mu, sd], axis=0))
    print(f"Saved mu/sd to: {NORM_STATS_PATH}")

    Xtr_s = apply_standardize(Xtr, mu, sd, clip=CLIP_Z)
    Xva_s = apply_standardize(Xva, mu, sd, clip=CLIP_Z)

    # class weights for focal alpha (inverse frequency)
    n_sf = int((ytr == 0).sum())
    n_st = int((ytr == 1).sum())
    w_sf = 1.0 / max(1, n_sf)
    w_st = 1.0 / max(1, n_st)
    alpha = torch.tensor([w_sf, w_st], dtype=torch.float32)
    alpha = alpha / alpha.mean()

    print(f"Focal alpha weights : alpha_SF={alpha[0].item():.4f}, alpha_ST={alpha[1].item():.4f}")
    print("Using: WeightedRandomSampler + WeightedFocalLoss(gamma=2.0)")

    # Weighted sampler
    weights = np.where(ytr == 0, float(alpha[0].item()), float(alpha[1].item()))
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(WindowDataset(Xtr_s, ytr), batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    val_loader = DataLoader(WindowDataset(Xva_s, yva), batch_size=256, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_cnn_model(dropout=DROPOUT, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = WeightedFocalLoss(alpha=alpha.to(device), gamma=2.0)

    best_state = None
    best_score = -1.0
    best_info = None
    patience = 0

    tr_losses, va_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            epoch_loss += float(loss.item()) * x.shape[0]

        epoch_loss /= max(1, len(train_loader.dataset))
        tr_losses.append(epoch_loss)

        # val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += float(loss.item()) * x.shape[0]
        val_loss /= max(1, len(val_loader.dataset))
        va_losses.append(val_loss)

        # threshold tuning on val (ST-first)
        p_st, y_true = predict_proba_st(model, val_loader, device)
        tuned = tune_threshold_val_st_first(p_st, y_true)

        # checkpoint selection metric: ST-F2 (tuned threshold)
        score = tuned["f2_st"]
        improved = score > best_score

        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_info = {
                "epoch": int(epoch),
                "val": tuned,
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss),
                "settings": {
                    "P_MIN_ST": float(P_MIN_ST),
                    "BETA_FOR_F2_ST": float(BETA_FOR_F2_ST),
                    "MIN_PREC_SF": float(MIN_PREC_SF),
                    "CLIP_Z": float(CLIP_Z),
                    "THRESH_GRID": int(THRESH_GRID),
                },
            }
            patience = 0
        else:
            patience += 1

        fb = " (fallback)" if tuned.get("used_fallback", False) else ""
        print(
            f"Epoch {epoch:03d} | train={epoch_loss:.4f} val={val_loss:.4f} "
            f"| bestF2(ST)={best_score:.4f} "
            f"| t={tuned['t']:.3f} precST={tuned['prec_st']:.3f} recST={tuned['rec_st']:.3f} f2ST={tuned['f2_st']:.3f}{fb} "
            f"| precSF={tuned['prec_sf']:.3f} recSF={tuned['rec_sf']:.3f} macroF1={tuned['macro_f1']:.3f}"
        )

        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE}).")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model selected.")

    torch.save(best_state, BEST_MODEL_PATH)
    save_json(INFO_PATH, best_info)

    # Plot loss
    plt.figure()
    plt.plot(tr_losses, label="train")
    plt.plot(va_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PDF)
    plt.savefig(PLOT_SVG)
    plt.savefig(PLOT_PNG)
    plt.close()

    print(f"Saved best model: {BEST_MODEL_PATH}")
    print(f"Saved info      : {INFO_PATH}")
    print(f"Saved plots     : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
