"""
04_train_vae.py

Train a VAE for exceedance detection using CLEAN windows.

Protocol
- Training data: CLEAN windows labeled "Normal" from TRAIN runs only.
- Normalization: mean/std fitted on TRAIN normals only (per channel).
- Scope: structural exceedance only; sensor faults are handled via deterministic integrity rules.

Inputs (from Data/extracted)
- X_clean.npy
- window_labels.csv
- run_split.json

Outputs (to Output/VAE_Training)
- artifacts: model checkpoint, normalization stats, manifest
- plots: loss curves (pdf/svg/png)
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import config as C
from io_utils import ensure_dir, save_json
from Models.temporal_vae_model import VAE


# =============================================================================
# Training settings
# =============================================================================
BATCH_SIZE = 64
N_EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 2.0

LABEL_NORMAL = "Normal"

# Clean channels: [DMS_1, LWA_2_clean, LWA_3_clean, LWA_4_clean]
# Exceedance detection uses displacement channels only to avoid load-channel drift.
CHANNELS_IDX = [1, 2, 3]

CLIP_Z = 10.0

# Model capacity (explicit for reproducibility)
LATENT_DIM = 8
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2

# Plot styling
COLOR_TOTAL = "#d73027"  # updated
COLOR_RECON = "#1b9e77"
COLOR_KL = "#9970ab"
LINEWIDTH = 1.5


# =============================================================================
# Output layout (non-data artifacts)
# =============================================================================
OUTPUT_ROOT = os.path.join(C.PROJECT_DIR, "Output")
EXP_NAME = "VAE_Training"
EXP_DIR = os.path.join(OUTPUT_ROOT, EXP_NAME)

PLOTS_DIR = os.path.join(EXP_DIR, "plots")
ARTIFACTS_DIR = os.path.join(EXP_DIR, "artifacts")
ensure_dir(PLOTS_DIR)
ensure_dir(ARTIFACTS_DIR)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "vae_exceedance_clean.pt")
MEAN_PATH = os.path.join(ARTIFACTS_DIR, "vae_clean_mean.npy")
STD_PATH = os.path.join(ARTIFACTS_DIR, "vae_clean_std.npy")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "vae_clean_manifest.json")

PLOT_BASE = "vae_training_loss_curves"
PLOT_PDF = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.pdf")
PLOT_SVG = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.svg")
PLOT_PNG = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.png")


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=(0, 1)).astype(np.float32)
    sd = np.nanstd(X, axis=(0, 1)).astype(np.float32)
    sd = np.where(sd < 1e-12, 1.0, sd).astype(np.float32)
    return mu, sd


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip_z: float) -> np.ndarray:
    Xn = (X - mu[None, None, :]) / sd[None, None, :]
    Xn = np.clip(Xn, -float(clip_z), float(clip_z))
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn.astype(np.float32)


def select_channels(X: np.ndarray, idx: List[int]) -> Tuple[np.ndarray, List[int]]:
    idx = list(idx)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N,T,C); got shape {X.shape}.")
    if max(idx) >= X.shape[2]:
        raise ValueError(f"CHANNELS_IDX out of range for C={X.shape[2]}.")
    return X[:, :, idx], idx


def kl_anneal(epoch: int, n_epochs: int, anneal_ratio: float = 0.30) -> float:
    x = (epoch - (n_epochs * anneal_ratio)) / max(n_epochs * anneal_ratio, 1e-12)
    return float(1.0 / (1.0 + np.exp(-x * 5.0)))


def save_loss_plot(
    epochs: np.ndarray,
    loss_total: List[float],
    loss_recon: List[float],
    loss_kl: List[float],
    pdf_path: str,
    svg_path: str,
    png_path: str,
) -> None:
    # Keep your preferred style but force white figure/axes and restore black bounding box.
    plt.style.use("fivethirtyeight")

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(epochs, loss_total, label="Total", color=COLOR_TOTAL, linewidth=LINEWIDTH)
    ax.plot(epochs, loss_recon, label="Reconstruction", color=COLOR_RECON, linewidth=LINEWIDTH, linestyle="--")
    ax.plot(epochs, loss_kl, label="KL", color=COLOR_KL, linewidth=LINEWIDTH, linestyle=":")

    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)

    ax.tick_params(axis="both", labelsize=20)
    ax.set_xlim(float(np.min(epochs)), float(np.max(epochs)))

    # No internal grid
    ax.grid(False)

    # Restore visible black axes box/spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    ax.legend(fontsize=15, loc="upper right", frameon=True)

    fig.tight_layout()

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    set_seed(int(C.SEED))

    x_clean_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["windows_clean"])
    meta_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["meta"])
    split_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["splits"])

    if not os.path.isfile(x_clean_path):
        raise FileNotFoundError(x_clean_path)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(meta_path)
    if not os.path.isfile(split_path):
        raise FileNotFoundError(split_path)

    X = np.load(x_clean_path).astype(np.float32)  # (N,T,C)
    meta = pd.read_csv(meta_path)

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    if X.ndim != 3 or X.shape[1] != int(C.SEQ_LEN):
        raise ValueError(f"Expected X shape (N,{int(C.SEQ_LEN)},C); got {X.shape}.")
    if len(meta) != X.shape[0]:
        raise ValueError("Meta rows must match window tensor (same N).")
    if "run_id" not in meta.columns or "label" not in meta.columns:
        raise ValueError("Meta must contain 'run_id' and 'label' columns.")

    train_runs = set(map(str, split["train_runs"]))
    train_mask = meta["run_id"].astype(str).isin(train_runs) & (meta["label"] == LABEL_NORMAL)

    X_train_raw = X[train_mask.to_numpy()]
    if X_train_raw.shape[0] < 200:
        raise ValueError(f"Too few TRAIN normal windows: {X_train_raw.shape[0]}.")

    X_train_raw, used_channels = select_channels(X_train_raw, CHANNELS_IDX)
    input_dim = int(X_train_raw.shape[2])

    mu, sd = fit_mean_std(X_train_raw)
    np.save(MEAN_PATH, mu)
    np.save(STD_PATH, sd)

    X_train = standardize(X_train_raw, mu, sd, clip_z=CLIP_Z)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    history: Dict[str, List[float]] = {"loss": [], "recon": [], "kl": [], "kl_w": []}

    for epoch in range(int(N_EPOCHS)):
        model.train()
        kl_w = kl_anneal(epoch, int(N_EPOCHS), anneal_ratio=0.30)

        sum_loss = 0.0
        sum_rec = 0.0
        sum_kl = 0.0
        nb = 0

        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu_t, logvar = model(xb)

            rec = nn.functional.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu_t.pow(2) - logvar.exp())
            loss = rec + kl_w * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            sum_loss += float(loss.item())
            sum_rec += float(rec.item())
            sum_kl += float(kl.item())
            nb += 1

        history["loss"].append(sum_loss / max(nb, 1))
        history["recon"].append(sum_rec / max(nb, 1))
        history["kl"].append(sum_kl / max(nb, 1))
        history["kl_w"].append(float(kl_w))

        print(
            f"[train] epoch {epoch:03d}/{int(N_EPOCHS)} "
            f"kl_w={kl_w:.4f} loss={history['loss'][-1]:.6f} "
            f"recon={history['recon'][-1]:.6f} kl={history['kl'][-1]:.6f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)

    manifest = {
        "seed": int(C.SEED),
        "seq_len": int(C.SEQ_LEN),
        "channels_idx": used_channels,
        "normalization": {
            "clip_z": float(CLIP_Z),
            "mean_path": MEAN_PATH,
            "std_path": STD_PATH,
        },
        "model": {
            "input_dim": int(input_dim),
            "latent_dim": int(LATENT_DIM),
            "hidden_dim": int(HIDDEN_DIM),
            "num_layers": int(NUM_LAYERS),
            "dropout": float(DROPOUT),
        },
        "optimizer": {
            "name": "Adam",
            "lr": float(LR),
            "weight_decay": float(WEIGHT_DECAY),
            "max_grad_norm": float(MAX_GRAD_NORM),
        },
        "train": {
            "batch_size": int(BATCH_SIZE),
            "epochs": int(N_EPOCHS),
            "train_normals": int(X_train.shape[0]),
            "label_normal": LABEL_NORMAL,
        },
        "inputs": {
            "x_clean": os.path.basename(x_clean_path),
            "meta": os.path.basename(meta_path),
            "split": os.path.basename(split_path),
        },
        "outputs": {
            "exp_dir": EXP_DIR,
            "model_path": MODEL_PATH,
            "plots_dir": PLOTS_DIR,
            "artifacts_dir": ARTIFACTS_DIR,
        },
    }
    save_json(MANIFEST_PATH, manifest)

    epochs = np.arange(1, int(N_EPOCHS) + 1)
    save_loss_plot(
        epochs=epochs,
        loss_total=history["loss"],
        loss_recon=history["recon"],
        loss_kl=history["kl"],
        pdf_path=PLOT_PDF,
        svg_path=PLOT_SVG,
        png_path=PLOT_PNG,
    )

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")
    print(f"Saved plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
