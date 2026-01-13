from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.lines import Line2D


# Ensure local imports work regardless of current working directory
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from datasets import standardize, destandardize, make_windows, stitch_windows, segment_rmse
from Models.temporal_vae import TemporalVAE


def configure_axis(ax, xlabel: str | None, ylabel: str) -> None:
    if xlabel is not None:
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


def build_variant_window_labels(windows: np.ndarray) -> np.ndarray:
    """
    One label per window for the 4 variants:
      0 = Original, 1 = Drifted, 2 = Amplitude, 3 = Low-Frequency

    Assumes input_dim = 12 ordered as:
      [x_* (4), v_* (4), a_* (4)]
    """
    input_dim = windows.shape[2]
    if input_dim < 12:
        raise ValueError(f"Expected at least 12 channels (x/v/a × 4 variants). Got {input_dim}.")

    variant_indices = [
        [0, 4, 8],    # Original
        [1, 5, 9],    # Drifted
        [2, 6, 10],   # Amplitude
        [3, 7, 11],   # Low-Frequency
    ]

    energies = []
    for cols in variant_indices:
        e = np.sum(windows[:, :, cols] ** 2, axis=(1, 2))
        energies.append(e)

    E = np.stack(energies, axis=1)  # [N, 4]
    return np.argmax(E, axis=1).astype(np.int64)


def encode_latent_mu_with_labels(
    model: torch.nn.Module,
    windows: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    mu_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    loader = DataLoader(
        TensorDataset(
            torch.tensor(windows, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _, mu, _ = model(xb)
            mu_list.append(mu.detach().cpu().numpy())
            y_list.append(yb.cpu().numpy())

    return np.concatenate(mu_list, axis=0), np.concatenate(y_list, axis=0)


def plot_latent_pca_by_type(mu: np.ndarray, y: np.ndarray, out_dir: Path, tab_dir: Path, stem: str) -> None:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        print("[WARN] scikit-learn not available; skipping PCA plot.")
        return

    class_names = ["Original", "Drifted", "Amplitude", "Low-Frequency"]
    class_colors = ["C0", "C1", "C2", "C3"]

    pca = PCA(n_components=2)
    Zp = pca.fit_transform(mu)

    fig, ax = plt.subplots(figsize=(12, 8))

    for k, (name, color) in enumerate(zip(class_names, class_colors)):
        idx = (y == k)
        if np.any(idx):
            ax.scatter(
                Zp[idx, 0],
                Zp[idx, 1],
                s=12,
                alpha=0.75,
                linewidths=0.0,
                color=color,
            )

    configure_axis(ax, xlabel="Principal Component 1", ylabel="Principal Component 2")

    # Legend BELOW with larger dots (proxy handles)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=14,
               markerfacecolor=class_colors[i], markeredgecolor="none", label=class_names[i])
        for i in range(len(class_names))
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=18,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    save_figure(fig, out_dir, stem)
    plt.close(fig)

    ev = {
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
    }
    (tab_dir / f"{stem}_explained_variance.json").write_text(json.dumps(ev, indent=2))
    np.save(tab_dir / f"{stem}_components.npy", pca.components_)
    print(f"[OK] wrote {tab_dir / f'{stem}_explained_variance.json'} and {stem}_components.npy")


def plot_stacked_reconstruction(
    time_s: np.ndarray,
    recon_df: pd.DataFrame,
    base_prefix: str,       # "x_" or "v_" or "a_"
    y_label: str,           # "x (m)" etc.
    out_dir: Path,
    file_stem: str,
) -> None:
    # Variant order + colors consistent with earlier figures
    variants = [
        ("Original",      f"{base_prefix}original",         "C0"),
        ("Drifted",       f"{base_prefix}drift",            "C1"),
        ("Amplitude",     f"{base_prefix}amplitude_scaled", "C2"),
        ("Low-Frequency", f"{base_prefix}lowfreq",          "C3"),
    ]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    for ax, (label, col, color) in zip(axes, variants):
        col_recon = f"{col}_recon"

        ax.plot(time_s, recon_df[col].values, linewidth=1.5, color=color, linestyle="-")
        ax.plot(time_s, recon_df[col_recon].values, linewidth=1.5, color=color, linestyle="--")

        # Non-intrusive label inside each subplot (no overlap with legend)
        ax.text(
            0.02, 0.90, label,
            transform=ax.transAxes,
            fontsize=18,
            va="top",
            ha="left",
        )

        configure_axis(ax, xlabel=None, ylabel=y_label)

    axes[-1].set_xlabel("Time (s)", fontsize=20)

    # Legend below: line styles only
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="-", label="Measured"),
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Reconstructed"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=18,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handletextpad=0.8,
        columnspacing=2.0,
    )

    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    save_figure(fig, out_dir, file_stem)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # 1_DOF

    csv_path = root / "Data" / "raw" / "1dof_seen_variants.csv"
    proc_dir = root / "Data" / "processed"
    model_path = root / "Scripts" / "Models" / "temporal_vae_state_dict.pt"

    mean = np.load(proc_dir / "vae_mean.npy")
    std = np.load(proc_dir / "vae_std.npy")

    out_fig = root / "Output" / "figures" / "reconstruction_seen"
    out_tab = root / "Output" / "tables" / "reconstruction_seen"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    # Parameters (match training)
    seq_len = 80
    stride = 1
    segment_len = 100
    test_frac_start = 0.5

    latent_dim = 5
    hidden_dim = 32
    num_layers = 2
    dropout = 0.2

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}.")

    df = pd.read_csv(csv_path)
    time = df["time"].values
    data = df.drop(columns=["time"]).values.astype(np.float32)

    T = data.shape[0]
    start = int(test_frac_start * T)

    time_t = time[start:]
    data_t = data[start:]

    data_norm = standardize(data_t, mean, std)
    windows = make_windows(data_norm, seq_len=seq_len, stride=stride)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalVAE(
        input_dim=windows.shape[2],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(windows, dtype=torch.float32).to(device)
        recon, mu, logvar = model(xb)
        recon_np = recon.detach().cpu().numpy()

    recon_series_norm = stitch_windows(recon_np, full_len=data_norm.shape[0], stride=stride)
    recon_series = destandardize(recon_series_norm, mean, std)

    # Save reconstruction table
    col_names = df.columns[1:].tolist()
    recon_df = pd.DataFrame({"time": time_t})
    for j, c in enumerate(col_names):
        recon_df[c] = data_t[:, j]
        recon_df[c + "_recon"] = recon_series[:, j]

    recon_csv = out_tab / "reconstruction_series.csv"
    recon_df.to_csv(recon_csv, index=False)
    print(f"[OK] wrote {recon_csv}")

    # Segment RMSE
    rmses = segment_rmse(data_t, recon_series, segment_len=segment_len)
    rmse_df = pd.DataFrame({"segment_index": np.arange(len(rmses)), "rmse": rmses})
    rmse_csv = out_tab / "segment_rmse.csv"
    rmse_df.to_csv(rmse_csv, index=False)
    print(f"[OK] wrote {rmse_csv}")

    # RMSE curve
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rmse_df["segment_index"], rmse_df["rmse"], linewidth=1.5)
    configure_axis(ax, xlabel="Segment index", ylabel="RMSE")
    fig.tight_layout()
    save_figure(fig, out_fig, "segment_rmse_curve")
    plt.close(fig)

    # Stacked reconstruction plots (separate, readable)
    plot_stacked_reconstruction(
        time_s=recon_df["time"].values,
        recon_df=recon_df,
        base_prefix="x_",
        y_label="x (m)",
        out_dir=out_fig,
        file_stem="reconstruction_seen_displacement_stacked",
    )

    plot_stacked_reconstruction(
        time_s=recon_df["time"].values,
        recon_df=recon_df,
        base_prefix="v_",
        y_label="v (m/s)",
        out_dir=out_fig,
        file_stem="reconstruction_seen_velocity_stacked",
    )

    plot_stacked_reconstruction(
        time_s=recon_df["time"].values,
        recon_df=recon_df,
        base_prefix="a_",
        y_label=r"a (m/s$^2$)",
        out_dir=out_fig,
        file_stem="reconstruction_seen_acceleration_stacked",
    )

    # Latent PCA on test windows (colored by signal type)
    y_windows = build_variant_window_labels(windows)
    mu_test, y_test = encode_latent_mu_with_labels(model, windows, y_windows, device=device, batch_size=512)

    plot_latent_pca_by_type(
        mu=mu_test,
        y=y_test,
        out_dir=out_fig,
        tab_dir=out_tab,
        stem="latent_pca_by_signal_type_seen_test",
    )


if __name__ == "__main__":
    main()
