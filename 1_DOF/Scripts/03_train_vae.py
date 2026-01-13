from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Ensure local imports work regardless of current working directory
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from datasets import compute_standardizer, standardize, make_windows
from Models.temporal_vae import TemporalVAE


def kl_weight(epoch: int, n_epochs: int, anneal_ratio: float = 0.3) -> float:
    pivot = int(n_epochs * anneal_ratio)
    denom = max(pivot, 1)
    x = (epoch - pivot) / denom
    return float(1.0 / (1.0 + np.exp(-5.0 * x)))


def configure_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)

    # No internal gridlines
    ax.grid(False)

    # Border box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    # Transparent axes background
    ax.set_facecolor("none")


def save_figure(fig, out_dir: Path, file_stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{file_stem}.pdf"
    png_path = out_dir / f"{file_stem}.png"
    svg_path = out_dir / f"{file_stem}.svg"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(png_path, format="png", bbox_inches="tight", transparent=True, dpi=300)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)

    print(f"[OK] saved: {pdf_path.name}, {png_path.name}, {svg_path.name}")


def encode_latent_mu(model: torch.nn.Module, windows: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    mus: list[np.ndarray] = []

    loader = DataLoader(
        TensorDataset(torch.tensor(windows, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            _, mu, _ = model(xb)
            mus.append(mu.detach().cpu().numpy())

    return np.concatenate(mus, axis=0)


def build_variant_window_labels(windows: np.ndarray) -> np.ndarray:
    """
    One label per window for the 4 variants:
      0 = Original
      1 = Drifted
      2 = Upscaled Amplitude
      3 = Low-Frequency

    Assumes input_dim = 12 with ordering:
      [x_* (4), v_* (4), a_* (4)]
    Variant columns:
      Original: [0,4,8], Drifted: [1,5,9], Amplitude: [2,6,10], Lowfreq: [3,7,11]
    """
    input_dim = windows.shape[2]
    if input_dim < 12:
        raise ValueError(f"Expected at least 12 channels (x/v/a × 4 variants). Got {input_dim}.")

    variant_cols = [
        [0, 4, 8],    # Original
        [1, 5, 9],    # Drifted
        [2, 6, 10],   # Upscaled Amplitude
        [3, 7, 11],   # Low-Frequency
    ]

    energies = []
    for cols in variant_cols:
        e = np.sum(windows[:, :, cols] ** 2, axis=(1, 2))
        energies.append(e)

    E = np.stack(energies, axis=1)  # [N, 4]
    return np.argmax(E, axis=1).astype(np.int64)


def plot_latent_pca_by_type(mu: np.ndarray, y: np.ndarray, out_dir: Path, tab_dir: Path) -> None:
    """
    PCA on latent means mu and scatter colored by variant type.
    Legend is placed below the axes with larger markers (proxy handles).
    """
    try:
        from sklearn.decomposition import PCA
    except Exception:
        print("[WARN] scikit-learn not available; skipping PCA plot.")
        return

    names = ["Original", "Drifted", "Upscaled Amplitude", "Low-Frequency"]
    colors = ["C0", "C1", "C2", "C3"]

    pca = PCA(n_components=2)
    Zp = pca.fit_transform(mu)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter per class
    for k, (name, color) in enumerate(zip(names, colors)):
        idx = (y == k)
        if np.any(idx):
            ax.scatter(
                Zp[idx, 0],
                Zp[idx, 1],
                s=18,              # point size in plot
                alpha=0.85,
                linewidths=0.0,
                color=color,
            )

    configure_axis(ax, xlabel="Principal Component 1", ylabel="Principal Component 2")

    # --- IMPORTANT: use proxy handles so legend marker size is controllable ---
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=12,        # increase legend dot diameter here
            markerfacecolor=c,
            markeredgecolor="none",
            label=n,
        )
        for n, c in zip(names, colors)
    ]

    # Legend below plot (not blocking)
    fig.legend(
        handles=legend_handles,
        labels=names,
        loc="lower center",
        ncol=4,
        fontsize=18,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Make room for the legend
    fig.tight_layout(rect=(0.02, 0.07, 1.0, 1.0))

    save_figure(fig, out_dir, "latent_pca_by_signal_type")
    plt.close(fig)

    ev = {
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
    }
    (tab_dir / "latent_pca_explained_variance.json").write_text(json.dumps(ev, indent=2))
    np.save(tab_dir / "latent_pca_components.npy", pca.components_)
    print(f"[OK] wrote {tab_dir / 'latent_pca_explained_variance.json'} and latent_pca_components.npy")


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # 1_DOF

    csv_path = root / "Data" / "raw" / "1dof_seen_variants.csv"
    proc_dir = root / "Data" / "processed"
    model_dir = root / "Scripts" / "Models"
    fig_dir = root / "Output" / "figures" / "training"
    tab_dir = root / "Output" / "tables" / "training"

    proc_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    seq_len = 80
    stride = 1
    train_frac = 0.5
    batch_size = 64
    n_epochs = 100
    lr = 1e-3

    latent_dim = 5
    hidden_dim = 32
    num_layers = 2
    dropout = 0.2
    anneal_ratio = 0.3

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}. Run 01_generate_seen_variants.py first.")

    df = pd.read_csv(csv_path)
    data = df.drop(columns=["time"]).values.astype(np.float32)

    # Split by time index: first 50% for training
    T = data.shape[0]
    split = int(train_frac * T)
    train_data = data[:split]

    split_info = {"T": int(T), "split_index": int(split), "train_frac": float(train_frac)}
    (proc_dir / "split.json").write_text(json.dumps(split_info, indent=2))

    # Standardizer from training only
    mean, std = compute_standardizer(train_data)
    np.save(proc_dir / "vae_mean.npy", mean)
    np.save(proc_dir / "vae_std.npy", std)

    meta = {
        "seq_len": int(seq_len),
        "stride": int(stride),
        "train_frac": float(train_frac),
        "input_dim": int(train_data.shape[1]),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "n_epochs": int(n_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "anneal_ratio": float(anneal_ratio),
        "csv_path": str(csv_path),
    }
    (proc_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Windowing
    train_norm = standardize(train_data, mean, std)
    train_windows = make_windows(train_norm, seq_len=seq_len, stride=stride)

    # Torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalVAE(
        input_dim=train_windows.shape[2],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = {
        "epoch": [],
        "loss_total": [],
        "loss_recon": [],
        "loss_kl": [],
        "kl_weight": [],
    }

    for epoch in range(n_epochs):
        model.train()

        total = 0.0
        total_recon = 0.0
        total_kl = 0.0

        w = kl_weight(epoch, n_epochs, anneal_ratio=anneal_ratio)

        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb)

            recon_loss = F.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + w * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            total_recon += float(recon_loss.item())
            total_kl += float(kl.item())

        avg_loss = total / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)

        history["epoch"].append(epoch + 1)
        history["loss_total"].append(avg_loss)
        history["loss_recon"].append(avg_recon)
        history["loss_kl"].append(avg_kl)
        history["kl_weight"].append(w)

        if (epoch + 1) % 10 == 0:
            print(
                f"[epoch {epoch+1:3d}/{n_epochs}] "
                f"loss={avg_loss:.6f} recon={avg_recon:.6f} kl={avg_kl:.6f} w={w:.3f}"
            )

    # Save model
    model_path = model_dir / "temporal_vae_state_dict.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[OK] saved model -> {model_path}")

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_csv = tab_dir / "training_losses.csv"
    hist_df.to_csv(hist_csv, index=False)
    print(f"[OK] wrote {hist_csv}")

    # Plot training curves (your style)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(hist_df["epoch"], hist_df["loss_total"], linewidth=1.5, linestyle="-", label="Total")
    ax.plot(hist_df["epoch"], hist_df["loss_recon"], linewidth=1.5, linestyle="--", label="Reconstruction")
    ax.plot(hist_df["epoch"], hist_df["loss_kl"], linewidth=1.5, linestyle=":", label="KL")

    configure_axis(ax, xlabel="Epoch", ylabel="Loss")
    ax.legend(fontsize=18, loc="upper right", frameon=True)
    fig.tight_layout()
    save_figure(fig, fig_dir, file_stem="training_curves")
    plt.close(fig)

    # Latent PCA colored by variant type with clean legend below
    mu_train = encode_latent_mu(model, train_windows, device=device, batch_size=512)
    y_train = build_variant_window_labels(train_windows)
    plot_latent_pca_by_type(mu_train, y_train, out_dir=fig_dir, tab_dir=tab_dir)


if __name__ == "__main__":
    main()
