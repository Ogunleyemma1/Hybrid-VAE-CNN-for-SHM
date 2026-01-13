from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.Models.temporal_vae import TemporalVAE
from Scripts.Models.cnn_model import SEQ_LEN, NUM_FEATURES


# ===================== CONFIG (OLD-MATCH) =====================
SEED = 42

WINDOW_LEN = SEQ_LEN
STRIDE = 1

# OLD concept: train/val are time fractions of EACH normal run (before windowing)
TRAIN_FRAC = (0.0, 0.4)
VAL_FRAC   = (0.4, 0.7)

EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5

LATENT_DIM = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

KL_WARMUP_RATIO = 0.30
LOG_EVERY = 1
# =============================================================

# We still read files from run_splits.json, but we DO NOT use its window_indices for VAE
RUN_SPLITS_PATH = _ROOT / "Data" / "processed" / "run_splits.json"
PROCESSED_DIR = _ROOT / "Data" / "processed"
MODELS_DIR = _ROOT / "models"
FIG_DIR = _ROOT / "Output" / "figures"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_abs(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (_ROOT / pp).resolve()
    return pp


def load_csv_numeric(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    X = np.loadtxt(str(path), delimiter=",", skiprows=1).astype(np.float32)
    if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
        raise ValueError(f"Bad CSV shape in {path}: {X.shape}")
    return X


def slice_frac(X: np.ndarray, frac_range: Tuple[float, float]) -> np.ndarray:
    n = X.shape[0]
    s = int(n * float(frac_range[0]))
    e = int(n * float(frac_range[1]))
    e = max(e, s)
    return X[s:e]


def make_windows(X: np.ndarray, win_len: int, stride: int) -> np.ndarray:
    if X.shape[0] < win_len:
        return np.zeros((0, win_len, X.shape[1]), dtype=np.float32)
    idx = range(0, X.shape[0] - win_len + 1, stride)
    return np.stack([X[i:i + win_len] for i in idx], axis=0).astype(np.float32)


def compute_mean_std_from_windows(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if W.shape[0] == 0:
        raise RuntimeError("No training windows available to compute mean/std.")
    Xflat = W.reshape(-1, W.shape[-1])
    mean = Xflat.mean(axis=0).astype(np.float32)
    std = Xflat.std(axis=0).astype(np.float32)
    std[std == 0] = 1e-6
    return mean, std


def normalize_windows(W: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (W - mean[None, None, :]) / std[None, None, :]
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def kl_anneal_sigmoid(
    epoch: int,
    n_epochs: int,
    start: float = 0.0,
    stop: float = 1.0,
    anneal_ratio: float = 0.3,
) -> float:
    """
    Sigmoid KL annealing (matches your old training script behavior).
    epoch is 1-based (1..n_epochs).
    anneal_ratio controls how fast KL ramps up.
    """
    e0 = epoch - 1  # convert to 0-based like the old script math
    warm = max(1, int(n_epochs * anneal_ratio))
    x = (e0 - warm) / float(max(warm, 1))
    return float(stop / (1.0 + np.exp(-x * 5.0)))




def kl_weight(epoch: int, n_epochs: int, warmup_ratio: float) -> float:
    # Keep the same signature, but use sigmoid annealing (old-style)
    return kl_anneal_sigmoid(epoch, n_epochs, start=0.0, stop=1.0, anneal_ratio=warmup_ratio)




def plot_training(hist: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(hist["epoch"], hist["train_total"], linewidth=1.5, label="Train")
    ax.plot(hist["epoch"], hist["val_total"], linewidth=1.5, label="Val")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(False)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"vae_training_curves.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("[OK] saved: Output/figures/vae_training_curves.(pdf/png/svg)")


def build_fraction_windows(file_list: list[str], frac: Tuple[float, float]) -> np.ndarray:
    allW = []
    for fp in file_list:
        X = load_csv_numeric(as_abs(fp))
        X = slice_frac(X, frac)
        W = make_windows(X, WINDOW_LEN, STRIDE)
        if W.shape[0]:
            allW.append(W)
    if not allW:
        return np.zeros((0, WINDOW_LEN, NUM_FEATURES), dtype=np.float32)
    return np.concatenate(allW, axis=0).astype(np.float32)


def main() -> None:
    set_seed(SEED)

    splits = load_json(RUN_SPLITS_PATH)
    if "normal" not in splits or "files" not in splits["normal"]:
        raise RuntimeError("run_splits.json missing splits['normal']['files'].")

    normal_files: list[str] = splits["normal"]["files"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # OLD-MATCH: slice by time fractions BEFORE windowing
    Wtr_raw = build_fraction_windows(normal_files, TRAIN_FRAC)
    Wva_raw = build_fraction_windows(normal_files, VAL_FRAC)

    print(
        f"[INFO] normal windows train/val = {Wtr_raw.shape[0]}/{Wva_raw.shape[0]} "
        f"(T={WINDOW_LEN}, D={NUM_FEATURES}) using fractions train={TRAIN_FRAC} val={VAL_FRAC}"
    )
    if Wtr_raw.shape[0] == 0:
        raise RuntimeError("No normal/train windows. Check normal files and TRAIN_FRAC.")
    if Wva_raw.shape[0] == 0:
        raise RuntimeError("No normal/val windows. Check normal files and VAL_FRAC.")

    mean, std = compute_mean_std_from_windows(Wtr_raw)
    np.save(PROCESSED_DIR / "vae_mean.npy", mean)
    np.save(PROCESSED_DIR / "vae_std.npy", std)
    np.savez(PROCESSED_DIR / "normal_stats.npz", mean=mean, std=std)
    print("[OK] wrote: Data/processed/vae_mean.npy and vae_std.npy")
    print("[OK] wrote: Data/processed/normal_stats.npz (mean/std from normal/train windows only)")

    Ztr = normalize_windows(Wtr_raw, mean, std)
    Zva = normalize_windows(Wva_raw, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = TemporalVAE(
        input_dim=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    opt = torch.optim.Adam(vae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    g = torch.Generator()
    g.manual_seed(SEED)

    dl_tr = DataLoader(
        TensorDataset(torch.tensor(Ztr)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=g,
        drop_last=False,
    )
    dl_va = DataLoader(
        TensorDataset(torch.tensor(Zva)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    best_val = float("inf")
    best_epoch = -1

    hist = {
        "epoch": [],
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total":   [], "val_recon":   [], "val_kl":   [],
        "kl_w": [],
    }


    for epoch in range(1, EPOCHS + 1):
        kl_w = kl_weight(epoch, EPOCHS, KL_WARMUP_RATIO)

        vae.train()
        tr_loss_sum, tr_recon_sum, tr_kl_sum, tr_n = 0.0, 0.0, 0.0, 0

        for (xb,) in dl_tr:
            xb = xb.to(device)
            xhat, mu, logvar = vae(xb)

            recon = F.mse_loss(xhat, xb, reduction="mean")
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + kl_w * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=2.0)
            opt.step()


            bsz = xb.size(0)
            tr_loss_sum += float(loss.item()) * bsz
            tr_recon_sum += float(recon.item()) * bsz
            tr_kl_sum += float(kl.item()) * bsz
            tr_n += bsz

        avg = tr_loss_sum / max(tr_n, 1)
        avg_r = tr_recon_sum / max(tr_n, 1)
        avg_kl = tr_kl_sum / max(tr_n, 1)

        vae.eval()
        va_loss_sum, va_recon_sum, va_kl_sum, va_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for (xb,) in dl_va:
                xb = xb.to(device)
                xhat, mu, logvar = vae(xb)

                recon = F.mse_loss(xhat, xb, reduction="mean")
                kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + kl_w * kl

                bsz = xb.size(0)
                va_loss_sum += float(loss.item()) * bsz
                va_recon_sum += float(recon.item()) * bsz
                va_kl_sum += float(kl.item()) * bsz
                va_n += bsz

        vavg = va_loss_sum / max(va_n, 1)
        vavg_r = va_recon_sum / max(va_n, 1)
        vavg_kl = va_kl_sum / max(va_n, 1)

        hist["epoch"].append(epoch)
        hist["kl_w"].append(float(kl_w))
        hist["train_total"].append(avg)
        hist["train_recon"].append(avg_r)
        hist["train_kl"].append(avg_kl)
        hist["val_total"].append(vavg)
        hist["val_recon"].append(vavg_r)
        hist["val_kl"].append(vavg_kl)


        if epoch % int(LOG_EVERY) == 0:
            print(
                f"[train] epoch {epoch:03d}/{EPOCHS} | kl_w={kl_w:.6f} | "
                f"total={avg:.6f} | recon={avg_r:.6f} | kl={avg_kl:.6f}",
                flush=True,
            )
            print(
                f"[val  ] epoch {epoch:03d}/{EPOCHS} | kl_w={kl_w:.6f} | "
                f"total={vavg:.6f} | recon={vavg_r:.6f} | kl={vavg_kl:.6f}",
                flush=True,
            )


        if vavg < best_val:
            best_val = vavg
            best_epoch = epoch
            torch.save(vae.state_dict(), MODELS_DIR / "temporal_vae_state_dict.pt")

    plot_training(hist)

    meta = {
        "seed": SEED,
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
        "train_frac": list(TRAIN_FRAC),
        "val_frac": list(VAL_FRAC),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "kl_warmup_ratio": KL_WARMUP_RATIO,
        "best_val_total": float(best_val),
        "best_epoch": int(best_epoch),
        "protocol": "OLD-MATCH: fraction slicing before windowing; mean/std from normal/train fraction only; VAE trained on normal/train fraction only.",
        "splits_path": str(RUN_SPLITS_PATH).replace("\\", "/"),
    }
    with (PROCESSED_DIR / "stage1_vae_train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] saved: models/temporal_vae_state_dict.pt")
    print("[OK] wrote: Data/processed/stage1_vae_train_meta.json")


if __name__ == "__main__":
    main()
