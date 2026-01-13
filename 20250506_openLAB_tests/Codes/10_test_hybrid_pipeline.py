"""
10_test_hybrid_pipeline.py

Goal
----
Compare the FULL 3-class pipeline performance for multiple Stage-2 supervised models,
using the SAME Stage-1 VAE gate, and produce publication-ready plots:

(1) A grid of ROW-NORMALIZED 3-class confusion matrices
    - Labels: Normal, Sensor Fault, Structural Fault
    - Subplot titles: (a) VAE + CNN, (b) VAE + RF, ...
    - Different colormaps per subplot (as in your Image 1 style)
    - Values shown are normalized (row-wise)

(2) A "metrics bar plot" (as in your Image 2 style) for Stage-2 models only
    - Metrics: Accuracy, Precision, Recall, F1, AUROC
    - Computed on routed anomaly windows where GT ∈ {Sensor Fault, Structural Fault}
    - Precision/Recall/F1 are for Structural Fault as the positive class (ST = 1)

Protocol / reviewer-safe
------------------------
- Run-based splits (from run_split.json)
- VAE threshold fixed from VAE validation artifact (VAL-derived threshold JSON)
- CNN threshold fixed from CNN validation artifact (VAL-derived .npy)
- ML thresholds fixed from ML baseline artifacts (VAL-derived .npy)
- Normalization stats are train-only (loaded from artifacts)
- Confusion matrices are row-normalized

Outputs
-------
Output/VAE_CNN_Pipeline_Comparisons/<split>/
  plots/
    cm_grid_row_normalized.pdf/.svg/.png
    stage2_metrics_bar.pdf/.svg/.png
  reports/
    comparison_summary.json
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import config as C
from Models.temporal_vae_model import VAE


# =============================================================================
# USER SETTINGS
# =============================================================================
SPLIT_TO_EVAL = "test"   # "val" or "test"

STAGE2_MODELS = [
    ("cnn", None),
    ("ml", "cart"),
    ("ml", "rf"),
    ("ml", "gb"),
    ("ml", "hgb"),
    ("ml", "svm_rbf"),
]

BATCH_SIZE = 256
CLIP_Z = 10.0


# =============================================================================
# Labels
# =============================================================================
LABEL_N  = "Normal"
LABEL_SF = "Sensor Fault"
LABEL_ST = "Structural Fault"
LABELS_3 = [LABEL_N, LABEL_SF, LABEL_ST]


# =============================================================================
# Input paths (from extractor)
# =============================================================================
X_CLEAN_PATH = os.path.join(C.OUT_DIR, C.ARTIFACTS["windows_clean"])
X_RAW_PATH   = os.path.join(C.OUT_DIR, C.ARTIFACTS["windows_raw"])
META_PATH    = os.path.join(C.OUT_DIR, C.ARTIFACTS["meta"])
SPLIT_PATH   = os.path.join(C.OUT_DIR, C.ARTIFACTS["splits"])


# =============================================================================
# Output layout
# =============================================================================
PROJECT_DIR = getattr(C, "PROJECT_DIR", os.path.dirname(getattr(C, "CODES_DIR", os.getcwd())))
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "Output")

EXP_DIR = os.path.join(OUTPUT_ROOT, "Full_Pipeline_Test", SPLIT_TO_EVAL)
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_npy(path: str) -> np.ndarray:
    return np.load(path)


def resolve_vae_artifacts() -> Dict[str, str]:
    codes_dir = getattr(C, "CODES_DIR", os.getcwd())
    vae_train_art_dir = os.path.join(OUTPUT_ROOT, "VAE_Training", "artifacts")
    vae_val_dir = os.path.join(OUTPUT_ROOT, "VAE_Validation_and_Thresholding")

    candidates = {
        "manifest": [
            os.path.join(vae_train_art_dir, "vae_clean_manifest.json"),
            os.path.join(codes_dir, "vae_clean_manifest.json"),
        ],
        "model": [
            os.path.join(vae_train_art_dir, "vae_exceedance_clean.pt"),
            os.path.join(codes_dir, "vae_exceedance_clean.pt"),
        ],
        "mean": [
            os.path.join(vae_train_art_dir, "vae_clean_mean.npy"),
            os.path.join(codes_dir, "vae_clean_mean.npy"),
        ],
        "std": [
            os.path.join(vae_train_art_dir, "vae_clean_std.npy"),
            os.path.join(codes_dir, "vae_clean_std.npy"),
        ],
        "thr_json": [
            os.path.join(vae_val_dir, "vae_clean_threshold.json"),
            os.path.join(vae_val_dir, "artifacts", "vae_clean_threshold.json"),
            os.path.join(vae_val_dir, "reports", "vae_clean_threshold.json"),
            os.path.join(vae_val_dir, "plots", "vae_clean_threshold.json"),
            os.path.join(codes_dir, "VAE_CLEAN_Validation_Plots", "vae_clean_threshold.json"),
        ],
    }

    out = {}
    for k, plist in candidates.items():
        p = _first_existing(plist)
        if p is None:
            raise FileNotFoundError(f"Missing VAE artifact '{k}'. Tried:\n" + "\n".join(plist))
        out[k] = p
    return out


def resolve_cnn_artifacts() -> Dict[str, str]:
    codes_dir = getattr(C, "CODES_DIR", os.getcwd())

    cand_model = [
        os.path.join(OUTPUT_ROOT, "CNN_Training", "artifacts", "cnn_model_openlab.pt"),
        os.path.join(codes_dir, "cnn_model_openlab.pt"),
    ]
    cnn_model = _first_existing(cand_model)
    if cnn_model is None:
        raise FileNotFoundError("Missing CNN model. Tried:\n" + "\n".join(cand_model))

    cand_norm = [
        os.path.join(OUTPUT_ROOT, "CNN_Training", "artifacts", "cnn_raw_mu_sd.npy"),
        os.path.join(codes_dir, "CNN_Training_Plots", "cnn_raw_mu_sd.npy"),
        os.path.join(codes_dir, "CNN_Training", "artifacts", "cnn_raw_mu_sd.npy"),
    ]
    norm_stats = _first_existing(cand_norm)
    if norm_stats is None:
        raise FileNotFoundError("Missing CNN norm stats. Tried:\n" + "\n".join(cand_norm))

    cand_thr = [
        os.path.join(OUTPUT_ROOT, "CNN_Validation", "artifacts", "cnn_best_threshold.npy"),
        os.path.join(codes_dir, "CNN_Training_Plots", "cnn_best_threshold.npy"),
        os.path.join(codes_dir, "CNN_Training", "artifacts", "cnn_best_threshold.npy"),
    ]
    thr_path = _first_existing(cand_thr)
    if thr_path is None:
        raise FileNotFoundError("Missing CNN threshold. Tried:\n" + "\n".join(cand_thr))

    return {"model": cnn_model, "norm_stats": norm_stats, "thr": thr_path}


def resolve_ml_artifacts(model_name: str) -> Dict[str, str]:
    codes_dir = getattr(C, "CODES_DIR", os.getcwd())

    cand_xfeat = [
        os.path.join(codes_dir, "ML_Features", "X_feat.npy"),
        os.path.join(OUTPUT_ROOT, "ML_Features", "X_feat.npy"),
    ]
    x_feat = _first_existing(cand_xfeat)
    if x_feat is None:
        raise FileNotFoundError("Missing ML features X_feat.npy. Tried:\n" + "\n".join(cand_xfeat))

    cand_model = [
        os.path.join(OUTPUT_ROOT, "ML_Baselines", "artifacts", f"{model_name}.joblib"),
        os.path.join(codes_dir, "ML_Baselines", f"{model_name}.joblib"),
    ]
    model_path = _first_existing(cand_model)
    if model_path is None:
        raise FileNotFoundError("Missing ML model. Tried:\n" + "\n".join(cand_model))

    cand_thr = [
        os.path.join(OUTPUT_ROOT, "ML_Baselines", "artifacts", f"{model_name}_threshold.npy"),
        os.path.join(codes_dir, "ML_Baselines", f"{model_name}_threshold.npy"),
    ]
    thr_path = _first_existing(cand_thr)
    if thr_path is None:
        raise FileNotFoundError("Missing ML threshold. Tried:\n" + "\n".join(cand_thr))

    return {"x_feat": x_feat, "model": model_path, "thr": thr_path}


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = CLIP_Z) -> np.ndarray:
    Xn = (X - mu[None, None, :]) / sd[None, None, :]
    Xn = np.clip(Xn, -float(clip), float(clip))
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn.astype(np.float32)


@torch.no_grad()
def recon_mse_per_window(
    model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = BATCH_SIZE
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], int(batch_size)):
        xb = torch.tensor(X[i:i + int(batch_size)], dtype=torch.float32, device=device)
        recon, _, _ = model(xb)
        mse = torch.mean((recon - xb) ** 2, dim=(1, 2)).detach().cpu().numpy()
        out.append(mse)
    return np.concatenate(out, axis=0).astype(np.float32)


def load_eval_mask(meta_df: pd.DataFrame, split_obj: dict, split_name: str) -> Tuple[np.ndarray, List[str]]:
    if split_name == "val":
        runs = set(map(str, split_obj["val_runs"]))
    elif split_name == "test":
        runs = set(map(str, split_obj["test_runs"]))
    else:
        raise ValueError("SPLIT_TO_EVAL must be 'val' or 'test'")
    mask = meta_df["run_id"].astype(str).isin(runs).to_numpy()
    return mask, sorted(list(runs))


@torch.no_grad()
def stage2_predict_cnn(
    X_raw_eval: np.ndarray, anomaly_mask: np.ndarray, cnn_art: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, float]:
    from Models.cnn_model import CNN
    import inspect

    mu_sd = _load_npy(cnn_art["norm_stats"]).astype(np.float32)
    mu, sd = mu_sd[0], mu_sd[1]
    thr = float(_load_npy(cnn_art["thr"]).ravel()[0])

    Xa = X_raw_eval[anomaly_mask].astype(np.float32)
    Xa = standardize(Xa, mu, sd, clip=CLIP_Z)
    Xa = torch.tensor(Xa[:, None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sig = inspect.signature(CNN.__init__)
    kwargs = {}
    if "dropout" in sig.parameters:
        kwargs["dropout"] = 0.4
    elif "dropout_rate" in sig.parameters:
        kwargs["dropout_rate"] = 0.4

    model = CNN(**kwargs).to(device)
    model.load_state_dict(torch.load(cnn_art["model"], map_location=device))
    model.eval()

    probs_st = []
    for i in range(0, Xa.shape[0], int(BATCH_SIZE)):
        xb = Xa[i:i + int(BATCH_SIZE)].to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs_st.append(p.detach().cpu().numpy())

    prob_st = np.concatenate(probs_st, axis=0).astype(np.float64)
    pred_bin = (prob_st >= thr).astype(np.int64)
    return pred_bin, prob_st, thr


def stage2_predict_ml(
    X_feat_eval: np.ndarray, anomaly_mask: np.ndarray, ml_art: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, float]:
    import joblib

    model = joblib.load(ml_art["model"])
    thr = float(_load_npy(ml_art["thr"]).ravel()[0])

    Xa = X_feat_eval[anomaly_mask].astype(np.float32)
    prob_st = model.predict_proba(Xa)[:, 1].astype(np.float64)
    pred_bin = (prob_st >= thr).astype(np.int64)
    return pred_bin, prob_st, thr


def main() -> None:
    for p in [X_CLEAN_PATH, X_RAW_PATH, META_PATH, SPLIT_PATH]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")

    X_clean_all = np.load(X_CLEAN_PATH).astype(np.float32)
    X_raw_all = np.load(X_RAW_PATH).astype(np.float32)
    meta_all = pd.read_csv(META_PATH)

    split = _read_json(SPLIT_PATH)

    if len(meta_all) != X_clean_all.shape[0] or len(meta_all) != X_raw_all.shape[0]:
        raise ValueError("Meta rows must match both X_clean and X_raw windows.")

    eval_mask, eval_runs = load_eval_mask(meta_all, split, SPLIT_TO_EVAL)

    X_clean = X_clean_all[eval_mask]
    X_raw = X_raw_all[eval_mask]
    meta = meta_all.loc[eval_mask].reset_index(drop=True)
    y_true = meta["label"].astype(str).to_numpy()

    vae_art = resolve_vae_artifacts()
    man = _read_json(vae_art["manifest"])
    channels_idx = list(map(int, man["channels_idx"]))
    model_cfg = man["model"]

    thr_obj = _read_json(vae_art["thr_json"])
    vae_thr = float(thr_obj["threshold"])

    mu = _load_npy(vae_art["mean"]).astype(np.float32)
    sd = _load_npy(vae_art["std"]).astype(np.float32)

    X_gate = X_clean[:, :, channels_idx]
    X_gate_std = standardize(X_gate, mu, sd, clip=CLIP_Z)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(
        input_dim=len(channels_idx),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    vae.load_state_dict(torch.load(vae_art["model"], map_location=device))
    vae.eval()

    mse = recon_mse_per_window(vae, X_gate_std, device=device, batch_size=BATCH_SIZE)
    anomaly_mask = (mse > vae_thr)

    needs_ml = any(mode == "ml" for mode, _ in STAGE2_MODELS)
    X_feat_eval = None
    if needs_ml:
        any_ml_name = next(m for (mode, m) in STAGE2_MODELS if mode == "ml")
        ml_any = resolve_ml_artifacts(any_ml_name)
        X_feat_all = np.load(ml_any["x_feat"]).astype(np.float32)
        X_feat_eval = X_feat_all[eval_mask]

    cm_list: List[np.ndarray] = []
    titles: List[str] = []
    model_display_names: List[str] = []

    metrics_acc: List[float] = []
    metrics_prec: List[float] = []
    metrics_rec: List[float] = []
    metrics_f1: List[float] = []
    metrics_auc: List[float] = []

    for idx_model, (mode, ml_name) in enumerate(STAGE2_MODELS):
        y_pred = np.full(len(meta), LABEL_N, dtype=object)

        diag_thr = None
        prob_st = None

        if anomaly_mask.any():
            if mode == "cnn":
                cnn_art = resolve_cnn_artifacts()
                pred_bin, prob_st, diag_thr = stage2_predict_cnn(X_raw, anomaly_mask, cnn_art)
                stage2_name = "CNN"
            elif mode == "ml":
                if ml_name is None:
                    raise ValueError("ML model name is None for mode='ml'.")
                ml_art = resolve_ml_artifacts(ml_name)
                pred_bin, prob_st, diag_thr = stage2_predict_ml(X_feat_eval, anomaly_mask, ml_art)
                stage2_name = ml_name.upper()
            else:
                raise ValueError("Unknown mode in STAGE2_MODELS.")

            diag_labels = np.where(pred_bin == 0, LABEL_SF, LABEL_ST).astype(object)
            y_pred[anomaly_mask] = diag_labels
        else:
            stage2_name = "CNN" if mode == "cnn" else (ml_name.upper() if ml_name else "ML")

        cm3 = confusion_matrix(y_true, y_pred, labels=LABELS_3)
        cm_list.append(cm3)

        letter = chr(ord("a") + idx_model)
        pretty = f"VAE + {stage2_name}"
        titles.append(f"({letter}) {pretty}")
        model_display_names.append(stage2_name if stage2_name != "CNN" else "CNN")

        corr_sf = np.array([], dtype=float)
        corr_st = np.array([], dtype=float)

        if anomaly_mask.any() and (prob_st is not None):
            y_true_a = y_true[anomaly_mask]
            keep = np.isin(y_true_a, [LABEL_SF, LABEL_ST])
            if keep.any():
                y_true_bin = np.where(y_true_a[keep] == LABEL_ST, 1, 0).astype(int)
                y_pred_bin = np.where(y_pred[anomaly_mask][keep] == LABEL_ST, 1, 0).astype(int)
                prob_st_kept = prob_st[keep].astype(np.float64)

                acc = float(accuracy_score(y_true_bin, y_pred_bin))
                prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
                rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
                f1v = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

                if len(np.unique(y_true_bin)) == 2:
                    auc = float(roc_auc_score(y_true_bin, prob_st_kept))
                else:
                    auc = float("nan")

                corr = (y_pred_bin == y_true_bin).astype(float)
                corr_sf = corr[y_true_bin == 0]
                corr_st = corr[y_true_bin == 1]
            else:
                acc = prec = rec = f1v = auc = float("nan")
        else:
            acc = prec = rec = f1v = auc = float("nan")

        metrics_acc.append(acc)
        metrics_prec.append(prec)
        metrics_rec.append(rec)
        metrics_f1.append(f1v)
        metrics_auc.append(auc)

        sf_path = os.path.join(REPORTS_DIR, f"correctness_sf_{stage2_name}.npy")
        st_path = os.path.join(REPORTS_DIR, f"correctness_st_{stage2_name}.npy")
        np.save(sf_path, corr_sf.astype(np.float32))
        np.save(st_path, corr_st.astype(np.float32))

        print("\n" + "=" * 80)
        print(f"{pretty} | Split={SPLIT_TO_EVAL} | VAE_thr={vae_thr:.6f} | anomaly_rate={anomaly_mask.mean():.4f}")
        if diag_thr is not None:
            print(f"Stage-2 threshold: {diag_thr:.6f}")
        print("3-class report:")
        print(classification_report(y_true, y_pred, labels=LABELS_3, zero_division=0))
        print("CM counts [Normal, SF, ST]:\n", cm3)

    metrics_pack = {
        "model_names": model_display_names,
        "Accuracy": np.array(metrics_acc, dtype=np.float64),
        "Precision": np.array(metrics_prec, dtype=np.float64),
        "Recall": np.array(metrics_rec, dtype=np.float64),
        "F1": np.array(metrics_f1, dtype=np.float64),
        "AUROC": np.array(metrics_auc, dtype=np.float64),
    }
    np.save(os.path.join(REPORTS_DIR, "stage2_metrics.npy"), metrics_pack, allow_pickle=True)

    summary = {
        "split": SPLIT_TO_EVAL,
        "runs": eval_runs,
        "vae_threshold": float(vae_thr),
        "anomaly_rate": float(anomaly_mask.mean()),
        "labels_order": LABELS_3,
        "models": [],
    }

    for i, name in enumerate(model_display_names):
        summary["models"].append({
            "name": name,
            "stage2_metrics_on_routed_anomalies": {
                "accuracy": float(metrics_acc[i]) if np.isfinite(metrics_acc[i]) else None,
                "precision_ST": float(metrics_prec[i]) if np.isfinite(metrics_prec[i]) else None,
                "recall_ST": float(metrics_rec[i]) if np.isfinite(metrics_rec[i]) else None,
                "f1_ST": float(metrics_f1[i]) if np.isfinite(metrics_f1[i]) else None,
                "auroc_ST": float(metrics_auc[i]) if np.isfinite(metrics_auc[i]) else None,
            },
            "confusion_matrix_counts_3class": cm_list[i].tolist(),
        })

    out_json = os.path.join(REPORTS_DIR, "comparison_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {os.path.join(REPORTS_DIR, 'stage2_metrics.npy')}")
    print(f"Saved correctness arrays in: {REPORTS_DIR}")
    print(f"\nOutputs directory: {EXP_DIR}")


if __name__ == "__main__":
    main()
