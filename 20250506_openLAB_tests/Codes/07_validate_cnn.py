"""
07_validate_cnn.py

Validate CNN for Sensor Fault (SF) vs Structural Fault (ST) classification.

ST-leaning policy (requested):
- VAL: tunes threshold to maximize ST recall subject to ST precision >= P_MIN_ST.
- TEST: uses the frozen threshold saved from VAL.
- Uses train-only mu/sd saved during CNN training.
- Robust model constructor (works if CNN __init__ accepts dropout or not).

Outputs (to Output/CNN_Validation)
- reports/: classification report + json summary
- plots/  : row-normalized confusion matrix + p(ST) histogram (pdf/svg/png)
- artifacts/: saved best threshold (VAL only)
"""

from __future__ import annotations

import inspect
import json
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import config as C
from io_utils import ensure_dir, save_json
from Models.cnn_model import CNN, SEQ_LEN, NUM_FEATURES


# =============================================================================
# Configuration
# =============================================================================
BATCH_SIZE = 256
DROPOUT = 0.4

SPLIT_TO_EVAL = "val"  # "val" tunes threshold; "test" loads frozen threshold

THRESH_GRID = 99
P_MIN_ST = 0.25
BETA_FOR_F2_ST = 2.0

# Optional: keep SF from collapsing completely (usually keep 0.0 unless necessary)
MIN_PREC_SF = 0.00

CLIP_Z = 10.0

LABEL_SF = "Sensor Fault"       # class 0
LABEL_ST = "Structural Fault"   # class 1

# Confusion matrix rendering
SHOW_VALUES = True
VALUE_FMT = "{:.2f}"
CM_CMAP = "Blues"


# =============================================================================
# Paths (derived from config.py + robust fallbacks)
# =============================================================================
def _artifact_path(key: str, fallback_name: str) -> str:
    if hasattr(C, "ARTIFACTS") and key in C.ARTIFACTS:
        return os.path.join(C.OUT_DIR, C.ARTIFACTS[key])
    return os.path.join(C.OUT_DIR, fallback_name)


X_RAW_PATH = _artifact_path("windows_raw", "X_raw.npy")
META_PATH  = _artifact_path("meta", "window_labels.csv")
SPLIT_PATH = _artifact_path("splits", "run_split.json")

OUTPUT_ROOT = os.path.join(C.PROJECT_DIR, "Output")
EXP_DIR = os.path.join(OUTPUT_ROOT, "CNN_Validation")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")
ARTIFACTS_DIR = os.path.join(EXP_DIR, "artifacts")
ensure_dir(PLOTS_DIR)
ensure_dir(REPORTS_DIR)
ensure_dir(ARTIFACTS_DIR)

THRESH_PATH = os.path.join(ARTIFACTS_DIR, "cnn_best_threshold.npy")

# Model paths: support both new layout and older ones
MODEL_CANDIDATES = [
    os.path.join(OUTPUT_ROOT, "CNN_Training", "artifacts", "cnn_model_openlab.pt"),
    os.path.join(getattr(C, "CODES_DIR", os.path.dirname(os.path.abspath(__file__))), "cnn_model_openlab.pt"),
]

# Norm stats paths: support both new + old
NORM_STATS_CANDIDATES = [
    os.path.join(OUTPUT_ROOT, "CNN_Training", "artifacts", "cnn_raw_mu_sd.npy"),
    os.path.join(OUTPUT_ROOT, "CNN_Training_Plots", "cnn_raw_mu_sd.npy"),
]


# =============================================================================
# Utilities
# =============================================================================
def find_first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = CLIP_Z) -> np.ndarray:
    x = X.astype(np.float32)
    x = (x - mu[None, None, :]) / sd[None, None, :]
    x = np.clip(x, -float(clip), float(clip))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def require_meta_columns(meta: pd.DataFrame) -> None:
    need = ["run_id", "label"]
    miss = [c for c in need if c not in meta.columns]
    if miss:
        raise ValueError(f"Meta file missing columns {miss}. Available: {list(meta.columns)}")


def encode_labels(label_series: pd.Series) -> np.ndarray:
    lab = label_series.astype(str).to_numpy()
    ok = np.isin(lab, [LABEL_SF, LABEL_ST])
    if not np.all(ok):
        bad = sorted(set(lab[~ok]))
        raise ValueError(f"Found labels outside [{LABEL_SF}, {LABEL_ST}]: {bad}")
    return np.where(lab == LABEL_SF, 0, 1).astype(np.int64)


def fbeta(prec: float, rec: float, beta: float = 2.0) -> float:
    if prec <= 0 or rec <= 0:
        return 0.0
    b2 = float(beta * beta)
    return (1.0 + b2) * (prec * rec) / (b2 * prec + rec)


def _prec_rec_for_class(y_true: np.ndarray, yhat: np.ndarray, cls: int) -> Tuple[float, float]:
    yt = (y_true == cls).astype(int)
    yp = (yhat == cls).astype(int)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    return float(prec), float(rec)


def select_threshold_st_first(y_true: np.ndarray, prob_st: np.ndarray) -> Dict:
    """
    Decision: predict ST if p(ST) >= t else SF.

    Objective:
      - enforce ST precision >= P_MIN_ST
      - optional enforce SF precision >= MIN_PREC_SF
      - maximize ST recall
      - tie-break by ST-F2
      - secondary tie-break by macro-F1

    Fallback: if no threshold meets ST precision floor, pick threshold with best ST-F2 overall.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_st = np.asarray(prob_st).astype(float)

    best = None
    fallback = None

    for t in np.linspace(0.01, 0.99, int(THRESH_GRID)):
        yhat = (prob_st >= t).astype(int)

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

        # fallback = best f2_st overall
        if (fallback is None) or (cand["f2_st"] > fallback["f2_st"]):
            fallback = cand

        ok = cand["meets_prec_st"] and cand["meets_prec_sf"]

        if best is None:
            best = cand
            best["meets_constraints"] = bool(ok)
            continue

        best_ok = best.get("meets_constraints", False)
        cand_ok = ok

        if cand_ok and not best_ok:
            best = cand
            best["meets_constraints"] = True
            continue

        if cand_ok == best_ok:
            if cand["rec_st"] > best["rec_st"]:
                best = cand
                best["meets_constraints"] = bool(cand_ok)
                continue
            if cand["rec_st"] == best["rec_st"] and cand["f2_st"] > best["f2_st"]:
                best = cand
                best["meets_constraints"] = bool(cand_ok)
                continue
            if cand["rec_st"] == best["rec_st"] and cand["f2_st"] == best["f2_st"] and cand["macro_f1"] > best["macro_f1"]:
                best = cand
                best["meets_constraints"] = bool(cand_ok)
                continue

    if not best.get("meets_constraints", False):
        out = dict(fallback)
        out["used_fallback"] = True
        out["meets_constraints"] = False
        return out

    best["used_fallback"] = False
    return best


def build_cnn_model(dropout: float, device: torch.device) -> torch.nn.Module:
    sig = inspect.signature(CNN.__init__)
    kwargs = {}
    if "dropout" in sig.parameters:
        kwargs["dropout"] = dropout
    elif "dropout_rate" in sig.parameters:
        kwargs["dropout_rate"] = dropout
    return CNN(**kwargs).to(device)


def plot_confusion_row_normalized(cm_counts: np.ndarray, names: list, title: str, out_base: str) -> None:
    cm = cm_counts.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    cm_norm = cm / row_sums

    plt.style.use("default")
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()
    ax.set_facecolor("white")

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=CM_CMAP, vmin=0.0, vmax=1.0)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)

    ticks = np.arange(len(names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=0)
    ax.set_yticklabels(names)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    if SHOW_VALUES:
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                v = cm_norm[i, j]
                ax.text(
                    j, i, VALUE_FMT.format(v),
                    ha="center", va="center",
                    fontsize=14,
                    color=("white" if v >= 0.55 else "black")
                )

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prob_hist(prob_st: np.ndarray, y_true: np.ndarray, thr: float, out_base: str) -> None:
    prob_st = np.asarray(prob_st).astype(float)
    y_true = np.asarray(y_true).astype(int)

    plt.style.use("default")
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.grid(False)

    plt.hist(prob_st[y_true == 0], bins=30, alpha=0.7, label="True Sensor Fault")
    plt.hist(prob_st[y_true == 1], bins=30, alpha=0.7, label="True Structural Fault")
    plt.axvline(thr, linestyle="--", linewidth=2, color="red", label=f"Threshold = {thr:.3f}")

    plt.xlabel("p(Structural Fault)", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    plt.legend(fontsize=15, loc="upper right")
    plt.tight_layout()

    plt.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.savefig(out_base + ".svg", bbox_inches="tight")
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    if not os.path.isfile(X_RAW_PATH):
        raise FileNotFoundError(f"Missing: {X_RAW_PATH}")
    if not os.path.isfile(META_PATH):
        raise FileNotFoundError(f"Missing: {META_PATH}")
    if not os.path.isfile(SPLIT_PATH):
        raise FileNotFoundError(f"Missing: {SPLIT_PATH}")

    cnn_model_path = find_first_existing(MODEL_CANDIDATES)
    if cnn_model_path is None:
        raise FileNotFoundError("Missing CNN model. Tried:\n" + "\n".join(MODEL_CANDIDATES))

    norm_stats_path = find_first_existing(NORM_STATS_CANDIDATES)
    if norm_stats_path is None:
        raise FileNotFoundError("Missing norm stats. Tried:\n" + "\n".join(NORM_STATS_CANDIDATES))

    X_raw = np.load(X_RAW_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH)
    require_meta_columns(meta)

    if X_raw.ndim != 3 or X_raw.shape[1] != int(SEQ_LEN) or X_raw.shape[2] != int(NUM_FEATURES):
        raise ValueError(f"Expected X_raw shape (N,{SEQ_LEN},{NUM_FEATURES}), got {X_raw.shape}")
    if len(meta) != X_raw.shape[0]:
        raise ValueError("Meta rows must match X_raw windows (same N).")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    if SPLIT_TO_EVAL not in ("val", "test"):
        raise ValueError("SPLIT_TO_EVAL must be 'val' or 'test'.")

    eval_runs = set(split["val_runs"] if SPLIT_TO_EVAL == "val" else split["test_runs"])
    m_eval = meta["run_id"].astype(str).isin(eval_runs).to_numpy()

    # keep SF/ST only
    lab_eval = meta.loc[m_eval, "label"].astype(str)
    keep = lab_eval.isin([LABEL_SF, LABEL_ST]).to_numpy()
    idx = np.where(m_eval)[0][keep]

    X_eval_raw = X_raw[idx]
    y_eval = encode_labels(meta.loc[idx, "label"])

    mu_sd = np.load(norm_stats_path).astype(np.float32)
    if mu_sd.shape[0] != 2 or mu_sd.shape[1] != NUM_FEATURES:
        raise ValueError(f"Expected mu/sd array shape (2,{NUM_FEATURES}), got {mu_sd.shape}")
    mu, sd = mu_sd[0], mu_sd[1]

    X_eval = apply_standardize(X_eval_raw, mu, sd, clip=CLIP_Z)
    X_eval_t = torch.tensor(X_eval[:, None, :, :], dtype=torch.float32)  # (N,1,T,C)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn_model(dropout=DROPOUT, device=device)
    state = torch.load(cnn_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    probs_st = []
    with torch.no_grad():
        for i in range(0, X_eval_t.shape[0], int(BATCH_SIZE)):
            xb = X_eval_t[i:i + int(BATCH_SIZE)].to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs_st.append(p.detach().cpu().numpy())
    probs_st = np.concatenate(probs_st, axis=0)

    # thresholding
    if SPLIT_TO_EVAL == "val":
        tuned = select_threshold_st_first(y_eval, probs_st)
        thr = float(tuned["t"])
        np.save(THRESH_PATH, np.array([thr], dtype=np.float32))

        fb = " (fallback)" if tuned.get("used_fallback", False) else ""
        print(
            f"Selected VAL thr={thr:.3f} | precST={tuned['prec_st']:.3f} recST={tuned['rec_st']:.3f} f2ST={tuned['f2_st']:.3f}{fb} "
            f"| precSF={tuned['prec_sf']:.3f} recSF={tuned['rec_sf']:.3f} macroF1={tuned['macro_f1']:.3f}"
        )
        print(f"Saved threshold: {THRESH_PATH}")
    else:
        if not os.path.isfile(THRESH_PATH):
            raise FileNotFoundError(f"Missing threshold: {THRESH_PATH} (run with SPLIT_TO_EVAL='val' first)")
        thr = float(np.load(THRESH_PATH).ravel()[0])
        print(f"Loaded frozen threshold from VAL: thr={thr:.3f}")

    y_pred = (probs_st >= thr).astype(int)

    names = [LABEL_SF, LABEL_ST]
    report = classification_report(y_eval, y_pred, target_names=names, zero_division=0)
    cm = confusion_matrix(y_eval, y_pred)

    print(f"\nCNN Report ({SPLIT_TO_EVAL.upper()} runs; SF vs ST):")
    print(report)
    print("Confusion matrix (counts):\n", cm)

    rpt_path = os.path.join(REPORTS_DIR, f"cnn_report_{SPLIT_TO_EVAL}.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(f"Split: {SPLIT_TO_EVAL}\n")
        f.write(f"Runs: {sorted(list(eval_runs))}\n")
        f.write(f"Threshold: {thr:.6f}\n")
        f.write(f"P_MIN_ST: {P_MIN_ST:.3f}\n")
        f.write(f"MIN_PREC_SF: {MIN_PREC_SF:.3f}\n")
        f.write(f"Model path: {cnn_model_path}\n")
        f.write(f"Norm stats used: {norm_stats_path}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (counts):\n")
        f.write(np.array2string(cm))

    cm_base = os.path.join(PLOTS_DIR, f"cnn_confusion_{SPLIT_TO_EVAL}_row_normalized")
    plot_confusion_row_normalized(
        cm_counts=cm,
        names=names,
        title=f"Confusion Matrix (CNN {SPLIT_TO_EVAL.upper()}, Row-Normalized)",
        out_base=cm_base
    )

    hist_base = os.path.join(PLOTS_DIR, f"cnn_pST_hist_{SPLIT_TO_EVAL}")
    plot_prob_hist(probs_st, y_eval, thr, hist_base)

    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_row_norm = (cm / row_sums).tolist()

    summary = {
        "split": SPLIT_TO_EVAL,
        "runs": sorted(list(eval_runs)),
        "n_eval": int(len(y_eval)),
        "threshold": float(thr),
        "P_MIN_ST": float(P_MIN_ST),
        "MIN_PREC_SF": float(MIN_PREC_SF),
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_normalized": cm_row_norm,
        "model_path": cnn_model_path,
        "norm_stats_path": norm_stats_path,
        "threshold_path": THRESH_PATH,
        "plots_dir": PLOTS_DIR,
        "reports_dir": REPORTS_DIR,
    }
    save_json(os.path.join(REPORTS_DIR, f"cnn_eval_summary_{SPLIT_TO_EVAL}.json"), summary)

    print(f"\nSaved: {rpt_path}")
    print(f"Saved: {cm_base}.pdf/.svg/.png")
    print(f"Saved: {hist_base}.pdf/.svg/.png")
    print(f"Outputs in: {EXP_DIR}")


if __name__ == "__main__":
    main()
