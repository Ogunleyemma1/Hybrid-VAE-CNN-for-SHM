from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.Models.temporal_vae import TemporalVAE
from Scripts.Models.cnn_model import CNN, SEQ_LEN, NUM_FEATURES


# ------------------------------ CONFIG (OLD-MATCH) ------------------------------
SEED = 42

# OLD-MATCH: full pipeline test uses last portion of each run (same as old default)
FRAC_RANGE = (0.7, 1.0)

WINDOW_LEN = SEQ_LEN
STRIDE = 1

LATENT_DIM = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
VAE_DROPOUT = 0.3

CNN_DROPOUT = 0.5
BATCH = 512

SCORE_DEF = "full_window_mse"
LINE_W = 1.5
# -------------------------------------------------------------------------------

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


def class_names() -> List[str]:
    # requested naming
    return ["Normal", "Sensor Fault", "Structural Fault"]


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


def _save_figure_multi(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(
            out,
            bbox_inches="tight",
            transparent=True,
            dpi=300 if ext == "png" else None,
        )


def plot_cm_row_norm(cm: np.ndarray, labels: List[str], stem: str) -> None:
    """
    Row-normalized CM, blue grade (requested), no title.
    """
    plt.style.use("fivethirtyeight")

    cm = cm.astype(np.float64)
    s = cm.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    cmn = cm / s

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(cmn, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=14, rotation=25, ha="right")
    ax.set_yticklabels(labels, fontsize=14)

    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("Ground Truth", fontsize=16)

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=13)

    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    _save_figure_multi(fig, stem)
    plt.close(fig)


def compute_roc(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(auc(fpr, tpr))


def plot_roc_two(
    y_gate: np.ndarray,
    s_gate: np.ndarray,
    y_hyb: np.ndarray,
    s_hyb: np.ndarray,
    stem: str,
) -> Dict[str, float]:
    """
    One ROC plot containing:
      - VAE gate ROC (Normal vs Fault)
      - Hybrid ROC (Structural Fault vs {Normal + Sensor Fault})
    Colors styled to resemble your reference image (cyan + magenta + dark blue).
    No title. Transparent. Saves png/pdf/svg.
    """
    plt.style.use("fivethirtyeight")

    fpr_g, tpr_g, auc_g = compute_roc(y_gate, s_gate)
    fpr_h, tpr_h, auc_h = compute_roc(y_hyb, s_hyb)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    # diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)

    # gate: magenta dotted (like ref image micro-avg)
    ax.plot(
        fpr_g,
        tpr_g,
        linestyle=":",
        linewidth=LINE_W,
        label=f"VAE Gate (Normal vs Fault) | AUROC={auc_g:.3f}",
    )

    # hybrid: cyan solid (like ref image class curve)
    ax.plot(
        fpr_h,
        tpr_h,
        linestyle="-",
        linewidth=LINE_W,
        label=f"Hybrid (Structural vs Rest) | AUROC={auc_h:.3f}",
    )

    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=12, loc="lower right")

    fig.tight_layout()
    _save_figure_multi(fig, stem)
    plt.close(fig)

    return {"gate_auroc": float(auc_g), "hybrid_auroc": float(auc_h)}


def plot_pr_curve(y: np.ndarray, s: np.ndarray, stem: str, label_txt: str) -> Dict[str, float]:
    """
    Publication PR curve (no title). Returns AP.
    """
    plt.style.use("fivethirtyeight")

    p, r, _ = precision_recall_curve(y, s)
    ap = float(average_precision_score(y, s))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(r, p, linewidth=LINE_W, label=f"{label_txt} | AP={ap:.3f}")

    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=12, loc="lower left")

    fig.tight_layout()
    _save_figure_multi(fig, stem)
    plt.close(fig)

    return {"ap": float(ap)}


def build_group_windows(files: List[str]) -> np.ndarray:
    allW = []
    for fp in files:
        X = slice_frac(load_csv_numeric(as_abs(fp)), FRAC_RANGE)
        W = make_windows(X, WINDOW_LEN, STRIDE)
        if W.shape[0]:
            allW.append(W)
    if not allW:
        return np.zeros((0, WINDOW_LEN, NUM_FEATURES), dtype=np.float32)
    return np.concatenate(allW, axis=0).astype(np.float32)


def main() -> None:
    set_seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_json(RUN_SPLITS_PATH)
    mean, std = load_stats()

    thr_path = PROCESSED_DIR / "vae_threshold.json"
    if not thr_path.exists():
        raise FileNotFoundError("Missing Data/processed/vae_threshold.json. Run python -m Scripts.04_vae_thresholding")
    thr_j = load_json(thr_path)
    mse_threshold = float(thr_j["threshold"])
    score_def = thr_j.get("score_def", "unknown")
    print(f"[INFO] Loaded threshold: {mse_threshold:.6f} | score_def={score_def}")
    print(f"[INFO] Test fraction: FRAC_RANGE={FRAC_RANGE} | window_len={WINDOW_LEN} | stride={STRIDE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load VAE ----
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

    # ---- Load CNN ----
    cnn = CNN(input_channels=2, num_classes=2, dropout_rate=CNN_DROPOUT).to(device)
    cnn_path = MODELS_DIR / "cnn_state_dict.pt"
    if not cnn_path.exists():
        raise FileNotFoundError("Missing models/cnn_state_dict.pt. Train Stage-2 CNN first.")
    cnn.load_state_dict(torch.load(str(cnn_path), map_location=device))
    cnn.eval()

    # gate ROC/PR data: normal=0, (sensor+struct)=1 using MSE score
    gate_scores_all: List[np.ndarray] = []
    gate_labels_all: List[np.ndarray] = []

    # hybrid ROC/PR data: Structural Fault positive (2) vs rest (0/1)
    # score = p_struct (CNN softmax for anomalies; 0 for non-anom windows)
    hyb_scores_all: List[np.ndarray] = []
    hyb_labels_all: List[np.ndarray] = []

    gate_stats: Dict[str, Dict[str, float]] = {}

    def eval_group(files: List[str], gt_label: int, tag: str) -> Tuple[List[int], List[int]]:
        W = build_group_windows(files)
        if W.shape[0] == 0:
            print(f"[WARN] {tag}: no test windows")
            return [], []

        Z = normalize_windows(W, mean, std)

        y_true: List[int] = [gt_label] * Z.shape[0]
        y_pred = np.zeros((Z.shape[0],), dtype=np.int64)

        mse_all = np.zeros((Z.shape[0],), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, Z.shape[0], BATCH):
                zb = torch.tensor(Z[i:i + BATCH], dtype=torch.float32, device=device)
                xhat, _, _ = vae(zb)
                mse = ((zb - xhat) ** 2).mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)
                mse_all[i:i + BATCH] = mse

        # ---- Gate labels/scores ----
        gate_scores_all.append(mse_all.copy())
        gate_labels_all.append(np.full_like(mse_all, 0 if gt_label == 0 else 1, dtype=np.int64))

        anom_mask = mse_all > mse_threshold
        idx_anom = np.where(anom_mask)[0]

        # Hybrid labels always defined for all windows
        # structural positive only for gt_label == 2
        hyb_labels_all.append(np.full((Z.shape[0],), 1 if gt_label == 2 else 0, dtype=np.int64))
        hyb_score_full = np.zeros((Z.shape[0],), dtype=np.float32)

        if idx_anom.size > 0:
            with torch.no_grad():
                for j in range(0, idx_anom.size, BATCH):
                    sel = idx_anom[j:j + BATCH]
                    zb = torch.tensor(Z[sel], dtype=torch.float32, device=device)
                    xhat, _, _ = vae(zb)
                    resid = (zb - xhat) ** 2
                    xin = torch.stack([zb, resid], dim=1)  # [B, 2, T, D]
                    logits = cnn(xin)  # [B, 2] => {sensor, structural}
                    cls01 = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
                    y_pred[sel] = cls01 + 1  # map {0,1} -> {1,2}

                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
                    p_struct = probs[:, 1]  # structural prob within anomalies
                    hyb_score_full[sel] = p_struct

        # store hybrid score for all windows (0 for non-anomalous)
        hyb_scores_all.append(hyb_score_full)

        total_w = int(Z.shape[0])
        total_anom = int(anom_mask.sum())
        rate = total_anom / total_w if total_w > 0 else 0.0
        gate_stats[tag] = {"anom": float(total_anom), "total": float(total_w), "anom_rate": float(rate)}
        print(f"[gate] {tag}: anom_rate={rate:.3f} (anom={total_anom}/{total_w})")

        return y_true, y_pred.tolist()

    yt0, yp0 = eval_group(splits["normal"]["files"], 0, "normal/test")
    yt1, yp1 = eval_group(splits["sensor_fault"]["files"], 1, "sensor/test")
    yt2, yp2 = eval_group(splits["structural_fault"]["files"], 2, "struct/test")

    y_true = yt0 + yt1 + yt2
    y_pred = yp0 + yp1 + yp2

    acc = float(accuracy_score(y_true, y_pred))
    print(f"[RESULT] 3-class window accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("[CM] rows=GT (Normal, Sensor Fault, Structural Fault); cols=Pred")
    print(cm)

    # ---- PRF in terminal (requested) ----
    labels = [0, 1, 2]
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n[PRF] Per-class (Precision / Recall / F1 / Support)")
    for name, p_, r_, f1_, s_ in zip(class_names(), p_c, r_c, f1_c, sup_c):
        print(f"  - {name:18s}: P={p_:.4f} | R={r_:.4f} | F1={f1_:.4f} | N={int(s_)}")
    print(f"[PRF] Macro avg        : P={p_macro:.4f} | R={r_macro:.4f} | F1={f1_macro:.4f}")
    print(f"[PRF] Weighted avg     : P={p_w:.4f} | R={r_w:.4f} | F1={f1_w:.4f}\n")

    report = classification_report(y_true, y_pred, target_names=class_names(), digits=4, zero_division=0)
    (FIG_DIR / "pipeline_classification_report.txt").write_text(report, encoding="utf-8")

    # ---- Plot: confusion matrix row-normalized (blue grade) ----
    plot_cm_row_norm(cm, class_names(), "pipeline_confusion_matrix_row_normalized")

    # ---- Gate ROC/PR ----
    gate_scores = np.concatenate(gate_scores_all).astype(np.float32) if gate_scores_all else np.zeros((0,), dtype=np.float32)
    gate_labels = np.concatenate(gate_labels_all).astype(np.int64) if gate_labels_all else np.zeros((0,), dtype=np.int64)

    gate_metrics: Dict[str, float] = {}
    if gate_scores.size > 0 and np.unique(gate_labels).size == 2:
        # keep PR curve as separate publication plot
        gate_pr = plot_pr_curve(gate_labels, gate_scores, "vae_gate_pr_curve", "VAE Gate (Normal vs Fault)")
        gate_metrics.update(gate_pr)

        # gate PRF at fixed threshold
        y_gate_pred = (gate_scores > mse_threshold).astype(np.int64)
        p_g, r_g, f1_g, _ = precision_recall_fscore_support(gate_labels, y_gate_pred, average="binary", zero_division=0)
        print("[GATE PRF] (positive = anomaly)")
        print(f"  P={p_g:.4f} | R={r_g:.4f} | F1={f1_g:.4f}\n")
        gate_metrics.update({"precision": float(p_g), "recall": float(r_g), "f1": float(f1_g)})

    # ---- Hybrid ROC/PR (Structural Fault vs Rest) ----
    hyb_scores = np.concatenate(hyb_scores_all).astype(np.float32) if hyb_scores_all else np.zeros((0,), dtype=np.float32)
    hyb_labels = np.concatenate(hyb_labels_all).astype(np.int64) if hyb_labels_all else np.zeros((0,), dtype=np.int64)

    hybrid_metrics: Dict[str, float] = {}
    if hyb_scores.size > 0 and np.unique(hyb_labels).size == 2:
        # keep PR curve as separate publication plot
        hyb_pr = plot_pr_curve(hyb_labels, hyb_scores, "hybrid_struct_vs_rest_pr_curve", "Hybrid (Structural vs Rest)")
        hybrid_metrics.update(hyb_pr)

        # Hybrid PRF at default 0.5
        hyb_pred = (hyb_scores >= 0.5).astype(np.int64)
        p_h, r_h, f1_h, _ = precision_recall_fscore_support(hyb_labels, hyb_pred, average="binary", zero_division=0)
        print("[HYBRID PRF] (positive = Structural Fault | score=p_struct on anomalies; 0 otherwise)")
        print(f"  P={p_h:.4f} | R={r_h:.4f} | F1={f1_h:.4f}\n")
        hybrid_metrics.update({"precision": float(p_h), "recall": float(r_h), "f1": float(f1_h)})

    # ---- Combined ROC plot (requested): Gate + Hybrid on one figure ----
    roc_both_metrics: Dict[str, float] = {}
    if gate_scores.size > 0 and hyb_scores.size > 0 and np.unique(gate_labels).size == 2 and np.unique(hyb_labels).size == 2:
        roc_both_metrics = plot_roc_two(
            y_gate=gate_labels,
            s_gate=gate_scores,
            y_hyb=hyb_labels,
            s_hyb=hyb_scores,
            stem="roc_gate_vs_hybrid",
        )

    metrics = {
        "accuracy": acc,
        "confusion_matrix_counts": cm.tolist(),
        "gate": {
            "threshold_mse": float(mse_threshold),
            "score_def": SCORE_DEF,
            "frac_range": list(FRAC_RANGE),
            "gate_stats": gate_stats,
            **roc_both_metrics,   # includes gate_auroc/hybrid_auroc if computed
            **gate_metrics,       # AP + PRF for gate
        },
        "hybrid_struct_vs_rest": {
            "definition": "Structural Fault (positive) vs {Normal, Sensor Fault} (negative)",
            "score": "p_struct (CNN softmax on anomalies; 0 for non-anomalous windows)",
            **hybrid_metrics,     # AP + PRF for hybrid
        },
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
        "seed": SEED,
    }

    with (FIG_DIR / "pipeline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (FIG_DIR / "vae_gate_binary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics["gate"], f, indent=2)

    with (FIG_DIR / "hybrid_struct_vs_rest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics["hybrid_struct_vs_rest"], f, indent=2)

    print("[OK] wrote: Output/figures/pipeline_metrics.json")
    print("[OK] wrote: Output/figures/vae_gate_binary_metrics.json")
    print("[OK] wrote: Output/figures/hybrid_struct_vs_rest_metrics.json")
    print("[OK] Plots saved to: Output/figures")
    print("     - pipeline_confusion_matrix_row_normalized.(png/pdf/svg)")
    print("     - roc_gate_vs_hybrid.(png/pdf/svg)   [Gate + Hybrid ROC in one plot]")
    print("     - vae_gate_pr_curve.(png/pdf/svg)")
    print("     - hybrid_struct_vs_rest_pr_curve.(png/pdf/svg)")


if __name__ == "__main__":
    main()
