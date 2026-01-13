"""
01_extract_windows_and_labels.py

Goal:
- Sensor Fault: flagged by manufacturer-style integrity rules / obstructions.
- Structural Fault: clean displacement exceedance > ALLOW_MAX (e.g., 20 mm).
- Normal: otherwise (0..ALLOW_MAX).

Artifacts:
- X_raw.npy   : [DMS_1, LWA_2_raw, LWA_3_raw, LWA_4_raw] (raw with NaNs for obstruction sentinel)
- X_clean.npy : [DMS_1, LWA_2_clean, LWA_3_clean, LWA_4_clean] (cleaned for structural envelope)
- window_labels.csv : metadata and labels per window
- run_diagnostics.csv : per-run sanity checks

Key fixes in this version:
1) Provider outlier rule is implemented as AND (doc-consistent):
      (|Δu| >= 1 mm) AND (|u| >= 65 mm)
   plus invalid samples (NaN/inf). This prevents SF contamination at low displacement.
2) Structural exceedance uses selected CLEAN channel(s) only (default: LWA_3),
   so obstruction spikes on LWA_4 do not generate Structural Fault.
3) Label precedence enforced strictly: Sensor Fault > Structural Fault > Normal.
4) Windows with ill-defined structural envelope (all-NaN in chosen channels) are forced to Sensor Fault.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import config as C
from io_utils import ensure_dir, save_npy, save_csv
from openlab_import import import_catman_file, run_id_from_path
from feature_utils import (
    clean_openlab_and_rule,
    windowize_2d,
    windowize_1d,
)

LABEL_NORMAL = "Normal"
LABEL_SENSOR_FAULT = "Sensor Fault"
LABEL_STRUCT_FAULT = "Structural Fault"

# -----------------------------
# Choose which CLEAN channels define structural exceedance
# -----------------------------
# Recommended for your dataset: use LWA_3 only to avoid LWA_4 obstruction spikes creating ST.
STRUCT_CLEAN_CHANNELS = ["LWA_3"]  # recommended
# If later desired:
# STRUCT_CLEAN_CHANNELS = ["LWA_2", "LWA_3"]


def _require_columns(df: pd.DataFrame, cols, run_id: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Run {run_id}: missing columns {missing}. Available: {list(df.columns)[:30]}...")


def _to_float(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)


def provider_raw_outlier_mask_AND(u_raw: np.ndarray, diff_th: float = 1.0, abs_th: float = 65.0) -> np.ndarray:
    """
    Provider doc rule (AND):
      outlier at sample i if (|u[i]-u[i-1]| >= diff_th) AND (|u[i]| >= abs_th),
    plus invalid samples (NaN/inf) are outliers.
    Returns float mask in {0,1}.
    """
    u = np.asarray(u_raw, dtype=np.float32)
    n = u.size
    m = np.zeros((n,), dtype=bool)

    # invalid always flagged
    m |= ~np.isfinite(u)

    if n > 1:
        du = np.abs(np.diff(u))
        m[1:] |= (du >= float(diff_th)) & (np.abs(u[1:]) >= float(abs_th))

    return m.astype(np.float32)


def main() -> None:
    ensure_dir(C.OUT_DIR)

    paths = sorted(glob.glob(os.path.join(C.RAW_DIR, "MD_*.txt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No MD_*.txt found in RAW_DIR: {C.RAW_DIR}")

    X_clean_all: list[np.ndarray] = []
    X_raw_all: list[np.ndarray] = []
    meta_all: list[pd.DataFrame] = []
    run_diag: list[dict] = []

    # Map clean tensor channel indices
    # Xc channels: 0=DMS_1, 1=LWA_2_clean, 2=LWA_3_clean, 3=LWA_4_clean
    name_to_idx = {"LWA_2": 1, "LWA_3": 2, "LWA_4": 3}
    struct_idxs = [name_to_idx[c] for c in STRUCT_CLEAN_CHANNELS]

    for p in paths:
        run_id = run_id_from_path(p)
        df = import_catman_file(p)

        required = ["DMS_1", "LWA_2", "LWA_3", "LWA_4"]
        _require_columns(df, required, run_id)

        dms = _to_float(df, "DMS_1")
        u2_raw = _to_float(df, "LWA_2")
        u3_raw = _to_float(df, "LWA_3")
        u4_raw = _to_float(df, "LWA_4")

        # Manufacturer obstruction sentinel -> NaN (raw integrity)
        u2_raw[u2_raw <= C.OBSTRUCTION_SENTINEL] = np.nan
        u3_raw[u3_raw <= C.OBSTRUCTION_SENTINEL] = np.nan
        u4_raw[u4_raw <= C.OBSTRUCTION_SENTINEL] = np.nan

        # Provider AND-rule outlier masks (doc-consistent)
        out2 = provider_raw_outlier_mask_AND(u2_raw, diff_th=C.RAW_DIFF_TH_MM, abs_th=C.RAW_ABS_TH_MM)
        out3 = provider_raw_outlier_mask_AND(u3_raw, diff_th=C.RAW_DIFF_TH_MM, abs_th=C.RAW_ABS_TH_MM)
        out4 = provider_raw_outlier_mask_AND(u4_raw, diff_th=C.RAW_DIFF_TH_MM, abs_th=C.RAW_ABS_TH_MM)

        inv2 = (~np.isfinite(u2_raw)).astype(np.float32)
        inv3 = (~np.isfinite(u3_raw)).astype(np.float32)
        inv4 = (~np.isfinite(u4_raw)).astype(np.float32)

        raw_out_mask = np.maximum.reduce([out2, out3, out4]).astype(np.float32)
        raw_inv_mask = np.maximum.reduce([inv2, inv3, inv4]).astype(np.float32)

        # Clean for structural envelope (keep "removed" mask)
        u2_clean, u2_removed = clean_openlab_and_rule(
            u2_raw, max_jump=C.CLEAN_MAX_JUMP_MM, max_abs=C.CLEAN_MAX_ABS_MM, ma_window=C.MOVING_AVG_WINDOW
        )
        u3_clean, u3_removed = clean_openlab_and_rule(
            u3_raw, max_jump=C.CLEAN_MAX_JUMP_MM, max_abs=C.CLEAN_MAX_ABS_MM, ma_window=C.MOVING_AVG_WINDOW
        )
        u4_clean, u4_removed = clean_openlab_and_rule(
            u4_raw, max_jump=C.CLEAN_MAX_JUMP_MM, max_abs=C.CLEAN_MAX_ABS_MM, ma_window=C.MOVING_AVG_WINDOW
        )

        removed_mask = np.maximum.reduce([u2_removed, u3_removed, u4_removed]).astype(np.float32)

        # Assemble tensors
        A_clean = np.stack([dms, u2_clean, u3_clean, u4_clean], axis=1).astype(np.float32)
        A_raw = np.stack([dms, u2_raw, u3_raw, u4_raw], axis=1).astype(np.float32)

        # Keep only rows where DMS is finite
        keep = np.isfinite(dms)
        A_clean = A_clean[keep]
        A_raw = A_raw[keep]
        raw_out_mask = raw_out_mask[keep]
        raw_inv_mask = raw_inv_mask[keep]
        removed_mask = removed_mask[keep]

        # Windowize
        Xc, idx0 = windowize_2d(A_clean, seq_len=C.SEQ_LEN, stride=C.STRIDE)
        Xr, idx0r = windowize_2d(A_raw, seq_len=C.SEQ_LEN, stride=C.STRIDE)

        if Xc.shape[0] == 0:
            continue
        if not np.array_equal(idx0, idx0r):
            raise RuntimeError(f"Run {run_id}: mismatch in window starts between raw and clean.")

        outW, _ = windowize_1d(raw_out_mask, seq_len=C.SEQ_LEN, stride=C.STRIDE)
        invW, _ = windowize_1d(raw_inv_mask, seq_len=C.SEQ_LEN, stride=C.STRIDE)
        remW, _ = windowize_1d(removed_mask, seq_len=C.SEQ_LEN, stride=C.STRIDE)

        raw_out_ratio = outW.mean(axis=1).astype(np.float32)
        raw_inv_ratio = invW.mean(axis=1).astype(np.float32)
        removed_ratio = remW.mean(axis=1).astype(np.float32)

        # -----------------------------
        # Structural envelope from chosen CLEAN channels only
        # -----------------------------
        U = np.stack([Xc[:, :, j] for j in struct_idxs], axis=2)  # (nW, T, nStructCh)

        # Use nanmin/nanmax; but guard for all-NaN windows
        u_min = np.nanmin(U, axis=(1, 2)).astype(np.float32)
        u_max = np.nanmax(U, axis=(1, 2)).astype(np.float32)
        all_nan_struct = (~np.isfinite(u_min)) | (~np.isfinite(u_max))

        # Load-aware flatline proxy (DMS range)
        dms_win = Xc[:, :, 0]
        dms_rng = (np.nanmax(dms_win, axis=1) - np.nanmin(dms_win, axis=1)).astype(np.float32)
        u_var = np.nanvar(U, axis=(1, 2)).astype(np.float32)

        flatline_loadaware = ((u_var < C.FLAT_VAR_EPS) & (dms_rng > C.FORCE_RANGE_FOR_FLATLINE)).astype(int)

        # Sensor Fault (integrity first)
        # - raw_inv_ratio: obstruction sentinel -> NaN
        # - raw_out_ratio: provider AND-rule outliers
        # - removed_ratio: points removed by cleaning rule
        # - flatline_loadaware: suspicious constant displacement under changing load
        sensor_fault = (
            (raw_inv_ratio >= float(C.RAW_INVALID_RATIO_FAULT))
            | (raw_out_ratio > 0.0)
            | (removed_ratio > 0.0)
            | (flatline_loadaware == 1)
            | (all_nan_struct)  # if we cannot define u_min/u_max -> treat as SF
        )

        # Structural Fault (exceedance) from CLEAN u_max
        structural_fault = (u_max > float(C.ALLOW_MAX))

        # STRICT precedence: SF > ST > Normal
        label = np.full((len(u_max),), LABEL_NORMAL, dtype=object)
        label[structural_fault & (~sensor_fault)] = LABEL_STRUCT_FAULT
        label[sensor_fault] = LABEL_SENSOR_FAULT

        meta = pd.DataFrame(
            {
                "run_id": run_id,
                "win_start_idx": idx0.astype(int),
                "label": label,
                "u_min": u_min,
                "u_max": u_max,
                "dms_range": dms_rng,
                "raw_invalid_ratio": raw_inv_ratio,
                "raw_outlier_ratio": raw_out_ratio,
                "removed_ratio": removed_ratio,
                "flatline_loadaware": flatline_loadaware,
                "struct_channels_for_u_max": ",".join(STRUCT_CLEAN_CHANNELS),
                "all_nan_struct": all_nan_struct.astype(int),
            }
        )

        # Run diagnostics (raw maxima and % > 65mm) for sanity checking
        def pct_abs_gt(x: np.ndarray, thr: float) -> float:
            x = np.asarray(x, dtype=np.float32)
            m = np.isfinite(x)
            if m.sum() == 0:
                return 0.0
            return float((np.abs(x[m]) > float(thr)).mean())

        run_diag.append(
            {
                "run_id": run_id,
                "n_samples": int(A_raw.shape[0]),
                "u2_max_raw": float(np.nanmax(u2_raw)),
                "u3_max_raw": float(np.nanmax(u3_raw)),
                "u4_max_raw": float(np.nanmax(u4_raw)),
                "u2_pct_abs_gt65_raw": pct_abs_gt(u2_raw, 65.0),
                "u3_pct_abs_gt65_raw": pct_abs_gt(u3_raw, 65.0),
                "u4_pct_abs_gt65_raw": pct_abs_gt(u4_raw, 65.0),
                "struct_channels_for_u_max": ",".join(STRUCT_CLEAN_CHANNELS),
            }
        )

        X_clean_all.append(Xc.astype(np.float32))
        X_raw_all.append(Xr.astype(np.float32))
        meta_all.append(meta)

    if len(X_clean_all) == 0:
        raise RuntimeError("No windows extracted. Check RAW_DIR, SEQ_LEN, STRIDE.")

    X_clean = np.concatenate(X_clean_all, axis=0).astype(np.float32)
    X_raw = np.concatenate(X_raw_all, axis=0).astype(np.float32)
    meta_df = pd.concat(meta_all, ignore_index=True)

    save_npy(os.path.join(C.OUT_DIR, C.ARTIFACTS["windows_clean"]), X_clean)
    save_npy(os.path.join(C.OUT_DIR, C.ARTIFACTS["windows_raw"]), X_raw)
    save_csv(os.path.join(C.OUT_DIR, C.ARTIFACTS["meta"]), meta_df)
    save_csv(os.path.join(C.OUT_DIR, "run_diagnostics.csv"), pd.DataFrame(run_diag))

    print(f"Saved: {os.path.abspath(C.OUT_DIR)}")
    print(f"X_clean: {X_clean.shape}  X_raw: {X_raw.shape}  meta: {meta_df.shape}")
    print("\nLABEL COUNTS")
    print(meta_df["label"].value_counts())


if __name__ == "__main__":
    main()
