"""
09_validate_ml_baselines.py

Validate classical ML baselines (feature-based) for SF vs ST classification.

ST-leaning update (requested):
- On VAL: tunes threshold to favor Structural Fault (ST, minority class):
    Constraints:
      - ST precision >= P_MIN_ST
      - optional SF precision >= MIN_PREC_SF (set 0.0 to disable)
    Objective:
      - maximize ST recall
      - tie-break: ST-F2 (beta=BETA_FOR_F2_ST)
      - secondary tie-break: macro-F1
    Fallback:
      - if constraints are impossible, pick best ST-F2 overall.

Outputs (to Output/ML_Baselines_Validation)
- reports: per-model text reports and json summaries
- plots  : row-normalized confusion matrices (pdf/svg/png), p(ST) histograms (pdf/svg/png)

Reviewer notes
- Confusion matrices are row-normalized for comparability under class imbalance.
- Threshold is tuned only on VAL and then frozen for TEST evaluation.
"""

from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import config as C
from io_utils import ensure_dir, save_json


# =============================================================================
# Config
# =============================================================================
SPLIT_TO_EVAL = "val"  # "val" tunes thresholds; "test" loads frozen thresholds

THRESH_GRID = 99

# -------------------------------
# ST-leaning threshold policy
# -------------------------------
P_MIN_ST = 0.25          # ST precision floor to avoid trivial "predict ST always"
BETA_FOR_F2_ST = 2.0     # emphasize ST recall (F2)

# Optional: prevent SF collapsing too much (typically leave disabled)
MIN_PREC_SF = 0.00       # e.g. 0.40 if you want to enforce SF precision too

LABEL_SF = "Sensor Fault"       # 0
LABEL_ST = "Structural Fault"   # 1

# Confusion-matrix rendering
SHOW_VALUES = True     # set False for pure heatmap (no numbers)
VALUE_FMT = "{:.2f}"
CM_CMAP = "Blues"


# =============================================================================
# Paths (robust)
# =============================================================================
CODES_DIR = getattr(C, "CODES_DIR", os.path.dirname(os.path.abspath(__file__)))

X_FEAT_PATH = os.path.join(CODES_DIR, "ML_Features", "X_feat.npy")

def _artifact_path(key: str, fallback_name: str) -> str:
    if hasattr(C, "ARTIFACTS") and key in C.ARTIFACTS:
        return os.path.join(C.OUT_DIR, C.ARTIFACTS[key])
    return os.path.join(C.OUT_DIR, fallback_name)

META_AUG_PATH = os.path.join(C.OUT_DIR, "window_labels_augmented.csv")
META_STD_PATH = _artifact_path("meta", "window_labels.csv")
SPLIT_PATH = _artifact_path("splits", "run_split.json")

OUTPUT_ROOT = os.path.join(C.PROJECT_DIR, "Output")

# Trained baselines live here (from 08_train_ml_baselines.py)
ML_EXP_DIR = os.path.join(OUTPUT_ROOT, "ML_Baselines")
MODEL_DIR = os.path.join(ML_EXP_DIR, "artifacts")

# Validation outputs
EXP_DIR = os.path.join(OUTPUT_ROOT, "ML_Baselines_Validation")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")
ensure_dir(PLOTS_DIR)
ensure_dir(REPORTS_DIR)


def load_meta() -> pd.DataFrame:
    meta_path = META_AUG_PATH if os.path.isfile(META_AUG_PATH) else META_STD_PATH
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta: {meta_path}")
    print(f"[meta] Using: {meta_path}")
    meta = pd.read_csv(meta_path)
    need = ["run_id", "label"]
    miss = [c for c in need if c not in meta.columns]
    if miss:
        raise ValueError(f"Meta missing columns {miss}. Available: {list(meta.columns)}")
    return meta


# =============================================================================
# Label mapping
# =============================================================================
def map_label_to_binary(label: str):
    s = str(label).strip().lower()
    if s in ["sensor fault", "sf"]:
        return 0
    if s in ["structural fault", "st", "structural"]:
        return 1
    return None


# =============================================================================
# Threshold objective (ST-first)
# =============================================================================
def fbeta(prec: float, rec: float, beta: float = 2.0) -> float:
    if prec <= 0 or rec <= 0:
        return 0.0
    b2 = float(beta * beta)
    return (1.0 + b2) * (prec * rec) / (b2 * prec + rec)


def _prec_rec_for_class(y_true: np.ndarray, yhat: np.ndarray, cls: int):
    yt = (y_true == cls).astype(int)
    yp = (yhat == cls).astype(int)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    return float(prec), float(rec)


def select_threshold_st_first(
    y_true: np.ndarray,
    prob_st: np.ndarray,
    pmin_st: float = P_MIN_ST,
    min_prec_sf: float = MIN_PREC_SF,
    grid: int = THRESH_GRID,
    beta_st: float = BETA_FOR_F2_ST,
):
    """
    Decision: predict ST if p(ST) >= t else SF.

    Constraints:
      - ST precision >= pmin_st
      - optional SF precision >= min_prec_sf (set 0.0 to disable)

    Objective:
      - maximize ST recall
      - tie-break: maximize ST-F2 (beta=beta_st)
      - secondary tie-break: macro-F1

    Fallback:
      - if no threshold meets constraints, choose best ST-F2 overall.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_st = np.asarray(prob_st).astype(float)

    ts = np.linspace(0.01, 0.99, int(grid))
    best = None
    fallback = None

    for t in ts:
        yhat = (prob_st >= t).astype(int)  # 1=ST, 0=SF

        prec_sf, rec_sf = _prec_rec_for_class(y_true, yhat, cls=0)
        prec_st, rec_st = _prec_rec_for_class(y_true, yhat, cls=1)

        f2_st = fbeta(prec_st, rec_st, beta=beta_st)
        macro_f1 = f1_score(y_true, yhat, average="macro", zero_division=0)

        meets = (prec_st >= float(pmin_st)) and ((prec_sf >= float(min_prec_sf)) if float(min_prec_sf) > 0 else True)

        cand = {
            "t": float(t),
            "prec_sf": float(prec_sf),
            "rec_sf": float(rec_sf),
            "prec_st": float(prec_st),
            "rec_st": float(rec_st),
            "f2_st": float(f2_st),
            "macro_f1": float(macro_f1),
            "meets_constraints": bool(meets),
        }

        # fallback: best ST-F2 overall
        if (fallback is None) or (cand["f2_st"] > fallback["f2_st"]):
            fallback = cand

        if best is None:
            best = cand
            continue

        if cand["meets_constraints"] and not best["meets_constraints"]:
            best = cand
            continue

        if cand["meets_constraints"] == best["meets_constraints"]:
            # primary: maximize ST recall
            if cand["rec_st"] > best["rec_st"]:
                best = cand
                continue
            # tie-break: ST-F2
            if cand["rec_st"] == best["rec_st"] and cand["f2_st"] > best["f2_st"]:
                best = cand
                continue
            # secondary tie-break: macro-F1
            if (cand["rec_st"] == best["rec_st"]) and (cand["f2_st"] == best["f2_st"]) and (cand["macro_f1"] > best["macro_f1"]):
                best = cand
                continue

    # If constraints impossible, return fallback (best ST-F2)
    if not best["meets_constraints"]:
        out = dict(fallback)
        out["used_fallback"] = True
        return out

    best["used_fallback"] = False
    return best


def get_prob_st(model, X: np.ndarray) -> np.ndarray:
    """
    Return p(ST) for a fitted sklearn model/pipeline.
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] != 2:
            raise ValueError(f"Expected predict_proba with 2 classes, got shape {p.shape}")
        return p[:, 1].astype(np.float64)

    if hasattr(model, "decision_function"):
        s = model.decision_function(X).astype(np.float64)
        s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)
        return s

    raise TypeError(f"Model {type(model)} has neither predict_proba nor decision_function.")


# =============================================================================
# Plots
# =============================================================================
def plot_confusion_row_normalized(
    cm_counts: np.ndarray,
    class_names: list,
    title: str,
    out_base_no_ext: str,
    show_values: bool = SHOW_VALUES,
    value_fmt: str = VALUE_FMT,
    cmap: str = CM_CMAP,
) -> None:
    cm = cm_counts.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    cm_norm = cm / row_sums

    plt.style.use("default")
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()
    ax.set_facecolor("white")

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticklabels(class_names)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    if show_values:
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                v = cm_norm[i, j]
                ax.text(
                    j, i, value_fmt.format(v),
                    ha="center", va="center",
                    fontsize=14,
                    color=("white" if v >= 0.55 else "black")
                )

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(out_base_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(out_base_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(out_base_no_ext + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prob_hist(prob_st: np.ndarray, y_true: np.ndarray, thr: float, out_base_no_ext: str) -> None:
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

    plt.savefig(out_base_no_ext + ".pdf", bbox_inches="tight")
    plt.savefig(out_base_no_ext + ".svg", bbox_inches="tight")
    plt.savefig(out_base_no_ext + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    if SPLIT_TO_EVAL not in ("val", "test"):
        raise ValueError("SPLIT_TO_EVAL must be 'val' or 'test'.")

    if not os.path.isfile(X_FEAT_PATH):
        raise FileNotFoundError(f"Missing features: {X_FEAT_PATH} (run 03_featurize_windows.py)")
    if not os.path.isfile(SPLIT_PATH):
        raise FileNotFoundError(f"Missing split: {SPLIT_PATH}")
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Missing model dir: {MODEL_DIR} (run 08_train_ml_baselines.py)")

    X_feat = np.load(X_FEAT_PATH).astype(np.float32)
    meta = load_meta()

    if len(meta) != X_feat.shape[0]:
        raise ValueError(f"Meta rows ({len(meta)}) must match X_feat N ({X_feat.shape[0]})")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    eval_runs = set(map(str, split["val_runs"] if SPLIT_TO_EVAL == "val" else split["test_runs"]))
    m_eval = meta["run_id"].astype(str).isin(eval_runs).to_numpy()

    labels = meta.loc[m_eval, "label"].astype(str)
    keep = labels.isin([LABEL_SF, LABEL_ST]).to_numpy()
    idx = np.where(m_eval)[0][keep]

    y_eval = meta.loc[idx, "label"].apply(map_label_to_binary).to_numpy(dtype=int)
    X_eval = X_feat[idx]

    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")])
    if len(model_files) == 0:
        raise FileNotFoundError(f"No *.joblib models found in: {MODEL_DIR}.")

    target_names = [LABEL_SF, LABEL_ST]

    for mfile in model_files:
        name = os.path.splitext(mfile)[0]
        model_path = os.path.join(MODEL_DIR, mfile)
        thr_path = os.path.join(MODEL_DIR, f"{name}_threshold.npy")

        model = joblib.load(model_path)
        prob_st = get_prob_st(model, X_eval)

        used_fallback = False
        met_constraints = None

        if SPLIT_TO_EVAL == "val":
            tuned = select_threshold_st_first(
                y_true=y_eval,
                prob_st=prob_st,
                pmin_st=P_MIN_ST,
                min_prec_sf=MIN_PREC_SF,
                grid=THRESH_GRID,
                beta_st=BETA_FOR_F2_ST,
            )
            thr = float(tuned["t"])
            np.save(thr_path, np.array([thr], dtype=np.float32))

            used_fallback = bool(tuned.get("used_fallback", False))
            met_constraints = bool(tuned.get("meets_constraints", False))
            fb_note = " (fallback)" if used_fallback else ""

            print(
                f"[{name}] tuned thr={thr:.3f} | "
                f"ST_prec={tuned['prec_st']:.3f} ST_rec={tuned['rec_st']:.3f} ST_F2={tuned['f2_st']:.3f}{fb_note} | "
                f"SF_prec={tuned['prec_sf']:.3f} SF_rec={tuned['rec_sf']:.3f} macroF1={tuned['macro_f1']:.3f}"
            )
        else:
            if not os.path.isfile(thr_path):
                raise FileNotFoundError(
                    f"Missing threshold for {name}: {thr_path} "
                    f"(run with SPLIT_TO_EVAL='val' first)"
                )
            thr = float(np.load(thr_path).ravel()[0])
            print(f"[{name}] loaded frozen thr={thr:.3f}")

        y_pred = (prob_st >= thr).astype(int)

        report = classification_report(y_eval, y_pred, target_names=target_names, zero_division=0)
        cm = confusion_matrix(y_eval, y_pred)

        print(f"\n[{name}] Report ({SPLIT_TO_EVAL.upper()}):")
        print(report)
        print("Confusion matrix (counts):\n", cm)

        # Save report
        rpt_path = os.path.join(REPORTS_DIR, f"{name}_report_{SPLIT_TO_EVAL}.txt")
        with open(rpt_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Split: {SPLIT_TO_EVAL}\n")
            f.write(f"Runs: {sorted(list(eval_runs))}\n")
            f.write(f"Threshold: {thr:.6f}\n")
            f.write(f"P_MIN_ST: {P_MIN_ST:.3f}\n")
            f.write(f"MIN_PREC_SF: {MIN_PREC_SF:.3f}\n")
            if SPLIT_TO_EVAL == "val":
                f.write(f"Met constraints: {bool(met_constraints)}\n")
                f.write(f"Used fallback: {bool(used_fallback)}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Threshold path: {thr_path}\n\n")
            f.write(report)
            f.write("\nConfusion matrix (counts):\n")
            f.write(np.array2string(cm))

        # Save plots
        cm_base = os.path.join(PLOTS_DIR, f"{name}_confusion_{SPLIT_TO_EVAL}_row_normalized")
        plot_confusion_row_normalized(
            cm_counts=cm,
            class_names=target_names,
            title=f"Confusion Matrix ({name.upper()} {SPLIT_TO_EVAL.upper()}, Row-Normalized)",
            out_base_no_ext=cm_base,
            show_values=SHOW_VALUES,
            value_fmt=VALUE_FMT,
            cmap=CM_CMAP,
        )

        hist_base = os.path.join(PLOTS_DIR, f"{name}_pST_hist_{SPLIT_TO_EVAL}")
        plot_prob_hist(prob_st, y_eval, thr, hist_base)

        # JSON summary
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_row_norm = (cm / row_sums).tolist()

        summary = {
            "model": name,
            "split": SPLIT_TO_EVAL,
            "runs": sorted(list(eval_runs)),
            "n_eval": int(len(y_eval)),
            "threshold": float(thr),
            "P_MIN_ST": float(P_MIN_ST),
            "MIN_PREC_SF": float(MIN_PREC_SF),
            "used_fallback": bool(used_fallback) if SPLIT_TO_EVAL == "val" else None,
            "met_constraints": bool(met_constraints) if SPLIT_TO_EVAL == "val" else None,
            "confusion_matrix_counts": cm.tolist(),
            "confusion_matrix_row_normalized": cm_row_norm,
            "report_path": rpt_path,
            "model_path": model_path,
            "threshold_path": thr_path,
            "plots_dir": PLOTS_DIR,
        }
        save_json(os.path.join(REPORTS_DIR, f"{name}_summary_{SPLIT_TO_EVAL}.json"), summary)

        print(f"Saved: {rpt_path}")
        print(f"Saved: {cm_base}.pdf/.svg/.png")
        print(f"Saved: {hist_base}.pdf/.svg/.png")
        print(f"Saved threshold: {thr_path}")

    print(f"\nDone. Outputs in: {EXP_DIR}")


if __name__ == "__main__":
    main()
