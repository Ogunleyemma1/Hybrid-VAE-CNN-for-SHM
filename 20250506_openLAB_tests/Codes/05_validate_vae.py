"""
05_validate_vae.py

Compute reconstruction error on validation windows for exceedance detection.

Protocol
- Validation set: windows from VAL runs.
- Threshold is selected from VAL Normal windows using a high percentile.
- The MSE plot includes Normal, Structural Fault, and Sensor Fault distributions.

Outputs (to Output/VAE_Validation_and_Thresholding)
- artifacts: threshold json
- plots: reconstruction MSE histogram (pdf/svg/png)
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import config as C
from io_utils import ensure_dir, save_json
from Models.temporal_vae_model import VAE


# =============================================================================
# Settings
# =============================================================================
BATCH_SIZE = 256
CLIP_Z = 10.0
THR_PERCENTILE = 95.0

LABEL_N = "Normal"
LABEL_SF = "Sensor Fault"
LABEL_E = "Structural Fault"

LINEWIDTH = 1.5
SPINE_WIDTH = 1.5
THRESH_LINE_COLOR = "red"

# Histogram colors (explicit)
COLOR_NORMAL = "#1f77b4"
COLOR_STRUCT = "#ff7f0e"
COLOR_SENSOR = "#9467bd"


# =============================================================================
# Output layout (non-data artifacts)
# =============================================================================
OUTPUT_ROOT = os.path.join(C.PROJECT_DIR, "Output")
EXP_NAME = "VAE_Validation_and_Thresholding"
EXP_DIR = os.path.join(OUTPUT_ROOT, EXP_NAME)

PLOTS_DIR = os.path.join(EXP_DIR, "plots")
ARTIFACTS_DIR = os.path.join(EXP_DIR, "artifacts")
ensure_dir(PLOTS_DIR)
ensure_dir(ARTIFACTS_DIR)

THRESH_PATH = os.path.join(ARTIFACTS_DIR, "vae_threshold.json")

PLOT_BASE = "vae_val_mse_histogram"
PLOT_PDF = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.pdf")
PLOT_SVG = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.svg")
PLOT_PNG = os.path.join(PLOTS_DIR, f"{PLOT_BASE}.png")


# =============================================================================
# Helpers
# =============================================================================
def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip_z: float) -> np.ndarray:
    Xn = (X - mu[None, None, :]) / sd[None, None, :]
    Xn = np.clip(Xn, -float(clip_z), float(clip_z))
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn.astype(np.float32)


@torch.no_grad()
def recon_mse_per_window(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for i in range(0, X.shape[0], int(batch_size)):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
        recon, _, _ = model(xb)
        mse = torch.mean((recon - xb) ** 2, dim=(1, 2)).detach().cpu().numpy()
        out.append(mse)
    return np.concatenate(out, axis=0).astype(np.float32)


def save_histogram(
    mse_normal: np.ndarray,
    mse_struct: np.ndarray,
    mse_sensor: np.ndarray,
    threshold: float,
    percentile: float,
    pdf_path: str,
    svg_path: str,
    png_path: str,
) -> None:
    plt.style.use("fivethirtyeight")

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bins = 60

    if mse_normal.size > 0:
        ax.hist(
            mse_normal,
            bins=bins,
            alpha=0.70,
            label="Normal (VAL)",
            color=COLOR_NORMAL,
        )
    if mse_struct.size > 0:
        ax.hist(
            mse_struct,
            bins=bins,
            alpha=0.55,
            label="Structural Fault (VAL)",
            color=COLOR_STRUCT,
        )
    if mse_sensor.size > 0:
        ax.hist(
            mse_sensor,
            bins=bins,
            alpha=0.55,
            label="Sensor Fault (VAL)",
            color=COLOR_SENSOR,
        )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=LINEWIDTH,
        color=THRESH_LINE_COLOR,
        label=f"P{percentile:.1f} thr={threshold:.6f}",
    )

    ax.set_xlabel("Reconstruction MSE (standardized, clean)", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
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
    x_clean_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["windows_clean"])
    meta_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["meta"])
    split_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["splits"])

    if not os.path.isfile(x_clean_path):
        raise FileNotFoundError(x_clean_path)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(meta_path)
    if not os.path.isfile(split_path):
        raise FileNotFoundError(split_path)

    # Training artifacts are produced by 04_train_vae.py (Output/VAE_Training/artifacts)
    train_exp_dir = os.path.join(C.PROJECT_DIR, "Output", "VAE_Training")
    train_artifacts_dir = os.path.join(train_exp_dir, "artifacts")

    manifest_path = os.path.join(train_artifacts_dir, "vae_clean_manifest.json")
    model_path = os.path.join(train_artifacts_dir, "vae_exceedance_clean.pt")
    mean_path = os.path.join(train_artifacts_dir, "vae_clean_mean.npy")
    std_path = os.path.join(train_artifacts_dir, "vae_clean_std.npy")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.isfile(mean_path):
        raise FileNotFoundError(f"Missing mean: {mean_path}")
    if not os.path.isfile(std_path):
        raise FileNotFoundError(f"Missing std: {std_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ch_idx = list(map(int, manifest["channels_idx"]))
    model_cfg = manifest["model"]

    X = np.load(x_clean_path).astype(np.float32)
    meta = pd.read_csv(meta_path)

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_runs = set(map(str, split["val_runs"]))
    val_mask = meta["run_id"].astype(str).isin(val_runs)

    meta_val = meta[val_mask].copy()
    X_val_all = X[val_mask.to_numpy()]

    if X_val_all.size == 0:
        raise RuntimeError("No validation windows found for the selected VAL runs.")

    # Select channels and normalize
    X_val_all = X_val_all[:, :, ch_idx]
    mu = np.load(mean_path).astype(np.float32)
    sd = np.load(std_path).astype(np.float32)
    X_val_std = standardize(X_val_all, mu, sd, clip_z=CLIP_Z)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        input_dim=len(ch_idx),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mse = recon_mse_per_window(model, X_val_std, device=device, batch_size=BATCH_SIZE)

    labels = meta_val["label"].astype(str).to_numpy()
    isN = labels == LABEL_N
    isE = labels == LABEL_E
    isSF = labels == LABEL_SF

    mseN = mse[isN]
    mseE = mse[isE]
    mseSF = mse[isSF]

    if mseN.size < 50:
        raise RuntimeError(f"Too few VAL normals: {mseN.size}")

    # Threshold from normals only
    threshold = float(np.percentile(mseN, float(THR_PERCENTILE)))
    fpr = float((mseN > threshold).mean())
    tpr = float((mseE > threshold).mean()) if mseE.size > 0 else float("nan")
    sf_rate = float((mseSF > threshold).mean()) if mseSF.size > 0 else float("nan")

    result = {
        "threshold": float(threshold),
        "threshold_source": f"P{THR_PERCENTILE} of VAL normals",
        "val_counts": {
            "normal": int(mseN.size),
            "structural_fault": int(mseE.size),
            "sensor_fault": int(mseSF.size),
        },
        "val_rates_above_threshold": {
            "normal_fpr": float(fpr),
            "structural_tpr": float(tpr),
            "sensor_fault_rate": float(sf_rate),
        },
        "channels_idx": ch_idx,
        "training_artifacts": {
            "model_path": model_path,
            "mean_path": mean_path,
            "std_path": std_path,
            "manifest_path": manifest_path,
        },
        "inputs": {
            "x_clean": os.path.basename(x_clean_path),
            "meta": os.path.basename(meta_path),
            "split": os.path.basename(split_path),
        },
    }

    save_json(THRESH_PATH, result)

    save_histogram(
        mse_normal=mseN,
        mse_struct=mseE,
        mse_sensor=mseSF,
        threshold=threshold,
        percentile=float(THR_PERCENTILE),
        pdf_path=PLOT_PDF,
        svg_path=PLOT_SVG,
        png_path=PLOT_PNG,
    )

    print(f"Saved threshold: {THRESH_PATH}")
    print(f"Saved plots: {PLOTS_DIR}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
