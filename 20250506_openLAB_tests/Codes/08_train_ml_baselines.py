"""
08_train_ml_baselines.py

Train classical ML baselines (feature-based) for SF vs ST classification.

ST-leaning update (requested):
- Threshold selection is now ST-first (minority class):
    1) enforce ST precision >= P_MIN_ST
    2) optional enforce SF precision >= MIN_PREC_SF (default disabled)
    3) maximize ST recall
    4) tie-break by ST-F2 (beta=BETA_FOR_F2_ST)
    5) secondary tie-break by macro-F1
- If no threshold meets constraints, fallback picks best ST-F2 overall.

Outputs (to Output/ML_Baselines)
- artifacts/: trained models (*.joblib) and thresholds (*_threshold.npy)
- reports/  : consolidated validation summary (csv/json) and run metadata (json)
"""

from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score

import config as C
from io_utils import ensure_dir, save_json


# =============================================================================
# Configuration
# =============================================================================
SEED = getattr(C, "SEED", 42)

THRESH_GRID = 99

# -------------------------------
# ST-leaning threshold policy
# -------------------------------
P_MIN_ST = 0.25          # ST precision floor (avoid trivial "everything ST")
BETA_FOR_F2_ST = 2.0     # emphasize ST recall (F2)

# Optional: prevent SF collapsing too much (usually leave disabled = 0.0)
MIN_PREC_SF = 0.00       # e.g., 0.40 if you want to enforce SF precision too


# =============================================================================
# Paths (portable: derived from config.py with safe fallbacks)
# =============================================================================
CODES_DIR = getattr(C, "CODES_DIR", os.path.dirname(os.path.abspath(__file__)))

# Features produced by: 03_featurize_windows.py (your file name) / featurize_windows.py (older)
FEAT_DIR = os.path.join(CODES_DIR, "ML_Features")
X_FEAT_PATH = os.path.join(FEAT_DIR, "X_feat.npy")
FEAT_NAMES_PATH = os.path.join(FEAT_DIR, "feat_names.json")

def _artifact_path(key: str, fallback_name: str) -> str:
    if hasattr(C, "ARTIFACTS") and key in C.ARTIFACTS:
        return os.path.join(C.OUT_DIR, C.ARTIFACTS[key])
    return os.path.join(C.OUT_DIR, fallback_name)

# Meta file produced by extractor (augmented is optional)
META_AUG_PATH = os.path.join(C.OUT_DIR, "window_labels_augmented.csv")
META_STD_PATH = _artifact_path("meta", "window_labels.csv")

# Split file produced by 02_make_splits.py
SPLIT_PATH = _artifact_path("splits", "run_split.json")

# Non-data outputs
OUTPUT_ROOT = os.path.join(C.PROJECT_DIR, "Output")
EXP_DIR = os.path.join(OUTPUT_ROOT, "ML_Baselines")
ARTIFACTS_DIR = os.path.join(EXP_DIR, "artifacts")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")
ensure_dir(ARTIFACTS_DIR)
ensure_dir(REPORTS_DIR)


# =============================================================================
# Label mapping
# =============================================================================
def map_label_to_binary(label: str):
    """
    0 = Sensor Fault (SF)
    1 = Structural Fault (ST)
    None = ignore (e.g., Normal)
    """
    s = str(label).strip().lower()
    if s in ["sensor fault", "sf"]:
        return 0
    if s in ["structural fault", "st", "structural"]:
        return 1
    return None


def load_meta():
    meta_path = META_AUG_PATH if os.path.isfile(META_AUG_PATH) else META_STD_PATH
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    print(f"[meta] Using: {meta_path}")
    meta = pd.read_csv(meta_path)
    need = ["run_id", "label"]
    miss = [c for c in need if c not in meta.columns]
    if miss:
        raise ValueError(f"Meta missing columns {miss}. Available: {list(meta.columns)}")
    return meta, meta_path


def finite_report(X, tag):
    n = X.shape[0]
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    n_bad_rows = int((~np.isfinite(X)).any(axis=1).sum())
    print(f"[sanity:{tag}] NaNs={n_nan} | Infs={n_inf} | rows_with_nonfinite={n_bad_rows} / {n}")


# =============================================================================
# Threshold objective (ST-first)
# =============================================================================
def fbeta(prec: float, rec: float, beta: float = 2.0) -> float:
    if prec <= 0 or rec <= 0:
        return 0.0
    b2 = float(beta * beta)
    return (1.0 + b2) * prec * rec / (b2 * prec + rec)


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
        yhat = (prob_st >= t).astype(int)

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
            # primary: ST recall
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
        # map to (0,1) monotonically for thresholding
        s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)
        return s

    raise TypeError(f"Model {type(model)} has neither predict_proba nor decision_function.")


# =============================================================================
# Main
# =============================================================================
def main():
    # Input checks
    if not os.path.isfile(X_FEAT_PATH):
        raise FileNotFoundError(f"Missing: {X_FEAT_PATH} (run 03_featurize_windows.py first)")
    if not os.path.isfile(SPLIT_PATH):
        raise FileNotFoundError(f"Missing: {SPLIT_PATH} (run 02_make_splits.py first)")

    X_feat = np.load(X_FEAT_PATH).astype(np.float32)
    meta, meta_path = load_meta()

    if len(meta) != X_feat.shape[0]:
        raise ValueError(f"Meta rows ({len(meta)}) must match X_feat N ({X_feat.shape[0]})")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Map labels -> binary and filter SF/ST only
    y_bin = meta["label"].apply(map_label_to_binary)
    mask_se = y_bin.notna()
    meta_se = meta.loc[mask_se].copy()
    y = y_bin.loc[mask_se].astype(int).to_numpy()
    X = X_feat[mask_se.to_numpy()]

    train_runs = set(map(str, splits["train_runs"]))
    val_runs = set(map(str, splits["val_runs"]))
    test_runs = set(map(str, splits.get("test_runs", [])))

    train_mask = meta_se["run_id"].astype(str).isin(train_runs).to_numpy()
    val_mask = meta_se["run_id"].astype(str).isin(val_runs).to_numpy()
    test_mask = meta_se["run_id"].astype(str).isin(test_runs).to_numpy() if len(test_runs) else np.zeros(len(meta_se), dtype=bool)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train SF/ST: {X_train.shape[0]} (SF={(y_train==0).sum()}, ST={(y_train==1).sum()})")
    print(f"Val   SF/ST: {X_val.shape[0]}   (SF={(y_val==0).sum()}, ST={(y_val==1).sum()})")
    if X_test.shape[0] > 0:
        print(f"Test  SF/ST: {X_test.shape[0]}  (SF={(y_test==0).sum()}, ST={(y_test==1).sum()})")

    if X_train.shape[0] < 10 or len(np.unique(y_train)) < 2:
        raise RuntimeError("Training set is too small or missing a class. Check run_split.json and labels.")

    finite_report(X_train, "train_raw")
    finite_report(X_val, "val_raw")

    # Consistent NaN-safe preprocessing across models
    imputer = SimpleImputer(strategy="median")

    models = {
        "cart": Pipeline([
            ("imputer", imputer),
            ("clf", DecisionTreeClassifier(
                random_state=SEED,
                class_weight="balanced"
            ))
        ]),
        "rf": Pipeline([
            ("imputer", imputer),
            ("clf", RandomForestClassifier(
                random_state=SEED,
                n_estimators=400,
                class_weight="balanced_subsample",
                n_jobs=-1
            ))
        ]),
        "svm_rbf": Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=SEED
            ))
        ]),
        "gb": Pipeline([
            ("imputer", imputer),
            ("clf", GradientBoostingClassifier(random_state=SEED))
        ]),
        "hgb": HistGradientBoostingClassifier(
            random_state=SEED,
            max_depth=None,
            learning_rate=0.05,
            max_iter=400
        )
    }

    # Experiment metadata (reviewer-facing)
    run_info = {
        "seed": int(SEED),
        "P_MIN_ST": float(P_MIN_ST),
        "BETA_FOR_F2_ST": float(BETA_FOR_F2_ST),
        "MIN_PREC_SF": float(MIN_PREC_SF),
        "thresh_grid": int(THRESH_GRID),
        "x_feat_path": X_FEAT_PATH,
        "feat_names_path": FEAT_NAMES_PATH if os.path.isfile(FEAT_NAMES_PATH) else None,
        "meta_used": os.path.basename(meta_path),
        "split_path": SPLIT_PATH,
        "train_runs": sorted(list(train_runs)),
        "val_runs": sorted(list(val_runs)),
        "test_runs": sorted(list(test_runs)),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    }
    save_json(os.path.join(REPORTS_DIR, "ml_training_info.json"), run_info)

    # Train each model + tune threshold on VAL (ST-first)
    results = []
    for name, model in models.items():
        print(f"\n[train] {name}")
        row = {
            "model": name,
            "status": "ok",
            "val_threshold": None,
            # ST-first metrics
            "val_prec_st": None,
            "val_rec_st": None,
            "val_f2_st": None,
            # still report SF
            "val_prec_sf": None,
            "val_rec_sf": None,
            "val_macro_f1": None,
            "met_constraints": None,
            "used_fallback": None,
            "model_path": None,
            "threshold_path": None,
            "error": None,
        }

        try:
            model.fit(X_train, y_train)
            prob_val_st = get_prob_st(model, X_val)

            tuned = select_threshold_st_first(
                y_true=y_val,
                prob_st=prob_val_st,
                pmin_st=P_MIN_ST,
                min_prec_sf=MIN_PREC_SF,
                grid=THRESH_GRID,
                beta_st=BETA_FOR_F2_ST,
            )

            thr = float(tuned["t"])

            model_path = os.path.join(ARTIFACTS_DIR, f"{name}.joblib")
            thr_path = os.path.join(ARTIFACTS_DIR, f"{name}_threshold.npy")

            joblib.dump(model, model_path)
            np.save(thr_path, np.array([thr], dtype=np.float32))

            fb = " (fallback)" if tuned.get("used_fallback", False) else ""
            print(
                f"[val-thr] {name}: thr={thr:.3f} | "
                f"ST_prec={tuned['prec_st']:.3f} ST_rec={tuned['rec_st']:.3f} ST_F2={tuned['f2_st']:.3f}{fb} | "
                f"SF_prec={tuned['prec_sf']:.3f} SF_rec={tuned['rec_sf']:.3f} macroF1={tuned['macro_f1']:.3f}"
            )
            print(f"Saved: {model_path}")
            print(f"Saved: {thr_path}")

            row.update({
                "val_threshold": thr,
                "val_prec_st": float(tuned["prec_st"]),
                "val_rec_st": float(tuned["rec_st"]),
                "val_f2_st": float(tuned["f2_st"]),
                "val_prec_sf": float(tuned["prec_sf"]),
                "val_rec_sf": float(tuned["rec_sf"]),
                "val_macro_f1": float(tuned["macro_f1"]),
                "met_constraints": bool(tuned["meets_constraints"]),
                "used_fallback": bool(tuned.get("used_fallback", False)),
                "model_path": model_path,
                "threshold_path": thr_path,
            })

        except Exception as e:
            print(f"[ERROR] {name} failed: {repr(e)}")
            row["status"] = "failed"
            row["error"] = repr(e)

        results.append(row)

    # Consolidated summary
    res_df = pd.DataFrame(results)
    res_csv = os.path.join(REPORTS_DIR, "ml_val_summary.csv")
    res_json = os.path.join(REPORTS_DIR, "ml_val_summary.json")
    res_df.to_csv(res_csv, index=False)
    save_json(res_json, {"results": results})

    print(f"\nSaved summary: {res_csv}")
    print(f"Saved summary: {res_json}")
    print(f"Artifacts in: {ARTIFACTS_DIR}")
    print(f"Reports in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
