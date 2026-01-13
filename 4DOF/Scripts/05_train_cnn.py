from __future__ import annotations

import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.Models.temporal_vae import TemporalVAE
from Scripts.Models.cnn_model import CNN, SEQ_LEN, NUM_FEATURES

# =================== CONFIG (consistent) ===================
SEED = 42

CNN_EPOCHS = 50
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 15

# VAE params must match the trained VAE
LATENT_DIM = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
VAE_DROPOUT = 0.3

# Build recon batches
RECON_BATCH = 512
# ==========================================================

RUN_SPLITS_PATH = _ROOT / "Data" / "processed" / "run_splits.json"
PROCESSED_DIR = _ROOT / "Data" / "processed"
MODELS_DIR = _ROOT / "models"
FIG_DIR = _ROOT / "Output" / "figures"


def set_seed(seed: int) -> None:
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


def make_windows(X: np.ndarray, win_len: int, stride: int) -> np.ndarray:
    if X.shape[0] < win_len:
        return np.zeros((0, win_len, X.shape[1]), dtype=np.float32)
    idx = range(0, X.shape[0] - win_len + 1, stride)
    return np.stack([X[i:i + win_len] for i in idx], axis=0).astype(np.float32)


def select_windows(W: np.ndarray, win_ids: List[int]) -> np.ndarray:
    if W.shape[0] == 0 or len(win_ids) == 0:
        return np.zeros((0, SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    ids = np.asarray(win_ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < W.shape[0])]
    if ids.size == 0:
        return np.zeros((0, SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    return W[ids]


def load_stats() -> Tuple[np.ndarray, np.ndarray]:
    stats = PROCESSED_DIR / "normal_stats.npz"
    if not stats.exists():
        raise FileNotFoundError("Missing Data/processed/normal_stats.npz. Run python -m Scripts.03_train_vae")
    d = np.load(stats)
    mean = d["mean"].astype(np.float32)
    std = d["std"].astype(np.float32)
    std[std == 0] = 1e-6
    return mean, std


def normalize_windows(W: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (W - mean[None, None, :]) / std[None, None, :]
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def build_cnn_inputs_batched(
    vae: torch.nn.Module,
    Z: np.ndarray,              # (N,T,D) normalized
    device: torch.device,
    batch: int,
) -> np.ndarray:
    """
    OLD style: xin = stack([Z, (Z - Zhat)^2], axis=channel) -> (N,2,T,D)
    """
    if Z.shape[0] == 0:
        return np.zeros((0, 2, SEQ_LEN, NUM_FEATURES), dtype=np.float32)

    vae.eval()
    out = np.zeros((Z.shape[0], 2, SEQ_LEN, NUM_FEATURES), dtype=np.float32)

    for i in range(0, Z.shape[0], batch):
        zb = torch.tensor(Z[i:i+batch], dtype=torch.float32, device=device)  # (B,T,D)
        zhat, _, _ = vae(zb)
        resid = (zb - zhat) ** 2
        xin = torch.stack([zb, resid], dim=1)  # (B,2,T,D)
        out[i:i+batch] = xin.detach().cpu().numpy().astype(np.float32)

    return out


def plot_train_val(train_loss: List[float], val_loss: List[float]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.plot(train_loss, label="Train Loss")
    ax.plot(val_loss, label="Validation Loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CNN Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"cnn_train_val_loss.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


def build_split_windows(files: List[str], win_map: Dict[str, Dict[str, List[int]]], split: str) -> np.ndarray:
    allW = []
    for fp in files:
        if fp not in win_map:
            continue
        X = load_csv_numeric(as_abs(fp))
        W = make_windows(X, SEQ_LEN, stride=1)
        Ws = select_windows(W, win_map[fp][split])
        if Ws.shape[0]:
            allW.append(Ws)
    if not allW:
        return np.zeros((0, SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    return np.concatenate(allW, axis=0).astype(np.float32)


def main() -> None:
    set_seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_json(RUN_SPLITS_PATH)
    if splits.get("mode") != "window_level_per_file":
        raise RuntimeError("run_splits.json must be window_level_per_file. Re-run: python -m Scripts.00_make_run_splits")

    mean, std = load_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    vae = TemporalVAE(
        input_dim=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=VAE_DROPOUT,
    ).to(device)

    vae_path = MODELS_DIR / "temporal_vae_state_dict.pt"
    if not vae_path.exists():
        raise FileNotFoundError("Missing models/temporal_vae_state_dict.pt. Run python -m Scripts.03_train_vae")
    vae.load_state_dict(torch.load(str(vae_path), map_location=device))
    vae.eval()

    # Build RAW windows (faults) by split indices
    sensor_files: List[str] = splits["sensor_fault"]["files"]
    sensor_map: Dict[str, Dict[str, List[int]]] = splits["sensor_fault"]["window_indices"]

    struct_files: List[str] = splits["structural_fault"]["files"]
    struct_map: Dict[str, Dict[str, List[int]]] = splits["structural_fault"]["window_indices"]

    Wtr_s = build_split_windows(sensor_files, sensor_map, "train")
    Wva_s = build_split_windows(sensor_files, sensor_map, "val")

    Wtr_st = build_split_windows(struct_files, struct_map, "train")
    Wva_st = build_split_windows(struct_files, struct_map, "val")

    if Wtr_s.shape[0] == 0 or Wtr_st.shape[0] == 0:
        raise RuntimeError("No fault/train windows. Check run_splits.json and ensure faults CSVs have enough rows for SEQ_LEN.")

    # Normalize using normal stats
    Ztr_s = normalize_windows(Wtr_s, mean, std)
    Zva_s = normalize_windows(Wva_s, mean, std)
    Ztr_st = normalize_windows(Wtr_st, mean, std)
    Zva_st = normalize_windows(Wva_st, mean, std)

    # Labels: sensor=0, structural=1
    Ztr = np.concatenate([Ztr_s, Ztr_st], axis=0)
    ytr = np.concatenate([
        np.zeros((Ztr_s.shape[0],), dtype=np.int64),
        np.ones((Ztr_st.shape[0],), dtype=np.int64),
    ], axis=0)

    Zva = np.concatenate([Zva_s, Zva_st], axis=0)
    yva = np.concatenate([
        np.zeros((Zva_s.shape[0],), dtype=np.int64),
        np.ones((Zva_st.shape[0],), dtype=np.int64),
    ], axis=0)

    # Deterministic shuffle
    rng = np.random.default_rng(SEED)
    p_tr = rng.permutation(Ztr.shape[0])
    p_va = rng.permutation(Zva.shape[0])
    Ztr, ytr = Ztr[p_tr], ytr[p_tr]
    Zva, yva = Zva[p_va], yva[p_va]

    print(f"[data] train windows: N={Ztr.shape[0]} (sensor={(ytr==0).sum()}, structural={(ytr==1).sum()})")
    print(f"[data]   val windows: N={Zva.shape[0]} (sensor={(yva==0).sum()}, structural={(yva==1).sum()})")

    # Build CNN inputs
    Xtr = build_cnn_inputs_batched(vae, Ztr, device, batch=RECON_BATCH)
    Xva = build_cnn_inputs_batched(vae, Zva, device, batch=RECON_BATCH)

    dl_tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # CNN
    model = CNN(input_channels=2, num_classes=2, dropout_rate=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(1, CNN_EPOCHS + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            tr_sum += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        tr_avg = tr_sum / max(tr_n, 1)

        model.eval()
        va_sum = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_sum += float(loss.item()) * xb.size(0)
                va_n += xb.size(0)

        va_avg = va_sum / max(va_n, 1)

        train_loss_hist.append(tr_avg)
        val_loss_hist.append(va_avg)

        print(f"[cnn] epoch {ep:03d}/{CNN_EPOCHS} train={tr_avg:.6f} val={va_avg:.6f}", flush=True)

        if va_avg < best_val:
            best_val = va_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[cnn] early stopping at epoch {ep}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), str(MODELS_DIR / "cnn_state_dict.pt"))
    plot_train_val(train_loss_hist, val_loss_hist)

    meta = {
        "seed": SEED,
        "epochs": CNN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "best_val_loss": float(best_val),
        "split_source": "run_splits.json window_level_per_file",
        "input_tensor": "stack([Z, (Z-Zhat)^2], channel) -> (N,2,T,D)",
    }
    (PROCESSED_DIR / "stage2_cnn_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[OK] saved: models/cnn_state_dict.pt")
    print("[OK] wrote: Data/processed/stage2_cnn_train_meta.json")
    print("[OK] saved: Output/figures/cnn_train_val_loss.(pdf/png/svg)")


if __name__ == "__main__":
    main()
