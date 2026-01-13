from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.Models.temporal_vae import TemporalVAE
from Scripts.Models.cnn_model import SEQ_LEN, NUM_FEATURES


SEED = 42
WINDOW_LEN = SEQ_LEN
STRIDE = 1

# OLD-MATCH: threshold uses healthy middle chunk
HEALTHY_FRAC = (0.4, 0.7)

PCTL = 99.0
BINS = 70
SCORE_DEF = "full_window_mse"

LATENT_DIM = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH = 512

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


def load_stats() -> Tuple[np.ndarray, np.ndarray, Path]:
    stats = PROCESSED_DIR / "normal_stats.npz"
    if not stats.exists():
        raise FileNotFoundError("Missing Data/processed/normal_stats.npz. Run python -m Scripts.03_train_vae")
    d = np.load(stats)
    mean = d["mean"].astype(np.float32)
    std = d["std"].astype(np.float32)
    std[std == 0] = 1e-6
    return mean, std, stats


def normalize_windows(W: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (W - mean[None, None, :]) / std[None, None, :]
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def full_mse_scores_batched(vae: torch.nn.Module, Z: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    if Z.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    vae.eval()
    scores = np.zeros((Z.shape[0],), dtype=np.float32)
    for i in range(0, Z.shape[0], batch):
        xb = torch.tensor(Z[i:i + batch], dtype=torch.float32, device=device)
        xhat, _, _ = vae(xb)
        mse = ((xb - xhat) ** 2).mean(dim=(1, 2))
        scores[i:i + batch] = mse.detach().cpu().numpy().astype(np.float32)
    return scores


def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    if scores.size == 0:
        return {}
    return {
        "n": float(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
    }


def plot_hist(s_n: np.ndarray, s_s: np.ndarray, s_st: np.ndarray, thr: float, log_x: bool = False) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    # Use common bin edges across all distributions
    all_scores = np.concatenate([s_n, s_s, s_st]) if (s_s.size or s_st.size) else s_n
    lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
    if hi <= lo:
        hi = lo + 1e-6

    # If log-x, shift away from 0 safely
    if log_x:
        eps = 1e-12
        lo = max(lo, eps)
        bins = np.logspace(np.log10(lo), np.log10(hi + eps), BINS + 1)
    else:
        bins = np.linspace(lo, hi, BINS + 1)

    ax.hist(s_n, bins=bins, alpha=0.75, label="Normal (val)", histtype="stepfilled")
    if s_s.size:
        ax.hist(s_s, bins=bins, alpha=0.55, label="Sensor fault (val)", histtype="stepfilled")
    if s_st.size:
        ax.hist(s_st, bins=bins, alpha=0.55, label="Structural fault (val)", histtype="stepfilled")

    ax.axvline(thr, linestyle="--", linewidth=1.5, color="red", label=f"Threshold ({PCTL:.0f}th pct.)")
    ax.set_xlabel("VAE MSE reconstruction error", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(False)
    if log_x:
        ax.set_xscale("log")

    fig.tight_layout()

    suffix = "logx" if log_x else "linear"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"vae_val_mse_histogram_{suffix}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[OK] saved: Output/figures/vae_val_mse_histogram_{suffix}.(pdf/png/svg)")


def plot_gate_roc_pr(y: np.ndarray, s: np.ndarray) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(fpr, tpr, linewidth=1.5)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(f"VAE Gate ROC (val) | AUROC={roc_auc:.3f}", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(False)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / "vae_gate_roc_curve.{ext}".format(ext=ext), bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)

    p, r, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot(r, p, linewidth=1.5)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(f"VAE Gate PR (val) | AP={ap:.3f}", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(False)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / "vae_gate_pr_curve.{ext}".format(ext=ext), bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


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

    mean, std, stats_path = load_stats()

    vae_path = MODELS_DIR / "temporal_vae_state_dict.pt"
    if not vae_path.exists():
        raise FileNotFoundError("Missing models/temporal_vae_state_dict.pt. Run python -m Scripts.03_train_vae")

    print(f"[INFO] stats: {str(stats_path).replace('\\', '/')}")
    print(f"[INFO] model: {str(vae_path).replace('\\', '/')}")
    print(f"[INFO] threshold fit: HEALTHY_FRAC={HEALTHY_FRAC} | PCTL={PCTL} | SCORE={SCORE_DEF} | (normal only)")

    # threshold uses healthy(val) fraction only
    n_files: list[str] = splits["normal"]["files"]
    Wn = build_fraction_windows(n_files, HEALTHY_FRAC)
    if Wn.shape[0] == 0:
        raise RuntimeError("No normal windows found for HEALTHY_FRAC. Check HEALTHY_FRAC and files.")

    # Optional diagnostics: also score faults on same fraction (val-like)
    s_files: list[str] = splits.get("sensor_fault", {}).get("files", [])
    st_files: list[str] = splits.get("structural_fault", {}).get("files", [])

    Ws = build_fraction_windows(s_files, HEALTHY_FRAC) if s_files else np.zeros((0, WINDOW_LEN, NUM_FEATURES), dtype=np.float32)
    Wst = build_fraction_windows(st_files, HEALTHY_FRAC) if st_files else np.zeros((0, WINDOW_LEN, NUM_FEATURES), dtype=np.float32)

    Zn = normalize_windows(Wn, mean, std)
    Zs = normalize_windows(Ws, mean, std) if Ws.shape[0] else np.zeros((0, WINDOW_LEN, NUM_FEATURES), dtype=np.float32)
    Zst = normalize_windows(Wst, mean, std) if Wst.shape[0] else np.zeros((0, WINDOW_LEN, NUM_FEATURES), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = TemporalVAE(
        input_dim=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    vae.load_state_dict(torch.load(str(vae_path), map_location=device))
    vae.eval()

    s_n = full_mse_scores_batched(vae, Zn, device, BATCH)
    s_s = full_mse_scores_batched(vae, Zs, device, BATCH) if Zs.shape[0] else np.zeros((0,), dtype=np.float32)
    s_st = full_mse_scores_batched(vae, Zst, device, BATCH) if Zst.shape[0] else np.zeros((0,), dtype=np.float32)

    thr = float(np.percentile(s_n, PCTL))

    # Reviewer-friendly: summary stats you can cite
    summary = {
        "normal_val": summarize_scores(s_n),
        "sensor_val": summarize_scores(s_s),
        "structural_val": summarize_scores(s_st),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "threshold": thr,
        "percentile": PCTL,
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
        "fit_data": f"normal fraction {HEALTHY_FRAC} only",
        "score_def": SCORE_DEF,
        "healthy_frac": list(HEALTHY_FRAC),
        "n_val_windows_normal": int(s_n.size),
        "n_val_windows_sensor": int(s_s.size),
        "n_val_windows_structural": int(s_st.size),
        "seed": SEED,
        "splits_path": str(RUN_SPLITS_PATH).replace("\\", "/"),
        "stats_path": str(stats_path).replace("\\", "/"),
        "model_path": str(vae_path).replace("\\", "/"),
        "score_summary": summary,
    }
    with (PROCESSED_DIR / "vae_threshold.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Plots: linear + log-x (log-x often reveals separability in heavy-tailed errors)
    plot_hist(s_n, s_s, s_st, thr, log_x=False)
    plot_hist(s_n, s_s, s_st, thr, log_x=True)

    # Gate ROC/PR (val diagnostics): normal=0, (sensor+struct)=1
    if (s_s.size + s_st.size) > 0:
        y = np.concatenate([np.zeros_like(s_n), np.ones_like(np.concatenate([s_s, s_st]))]).astype(np.int64)
        s = np.concatenate([s_n, np.concatenate([s_s, s_st])]).astype(np.float32)
        plot_gate_roc_pr(y, s)

    print(f"[OK] Threshold saved: {thr:.6f}")
    print("[OK] wrote: Data/processed/vae_threshold.json")


if __name__ == "__main__":
    main()
