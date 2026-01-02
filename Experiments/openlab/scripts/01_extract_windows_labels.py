# experiments/openlab/scripts/01_extract_windows_labels.py
"""
Step 01: Windowing + weak-supervision label construction (silver) + optional gold merge.

Inputs:
  - experiments/openlab/datasets/processed/runs_manifest.json
  - experiments/openlab/datasets/processed/runs/<run_id>.npz

Outputs (to processed/):
  - X_clean.npy                 (N, T, 3) for VAE gate logic
  - X_raw.npy                   (N, T, 4) for CNN SF/E classifier
  - window_labels_silver.csv
  - window_labels_augmented.csv (gold overrides silver if provided)

Label policy:
  - silver labels are deterministic and auditable, using window-level sensor-quality rules.
  - gold labels (if a file exists) override silver on matching (run_id, win_start_idx).

Notes:
  - You can keep your VAE gate on the first 3 displacement channels (X_clean),
    while CNN uses 4 channels (X_raw).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


# ---------------------------
# Windowing
# ---------------------------
def windowize_2d(A: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    A: (M, C) -> windows: (N, seq_len, C), start_idx: (N,)
    """
    M = A.shape[0]
    if M < seq_len:
        return np.zeros((0, seq_len, A.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    idx0 = np.arange(0, M - seq_len + 1, stride, dtype=np.int64)
    X = np.stack([A[i : i + seq_len] for i in idx0], axis=0).astype(np.float32)
    return X, idx0


# ---------------------------
# Silver sensor-fault rules (window-level)
# ---------------------------
def invalid_ratio_1d(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.mean(~np.isfinite(x)))


def jump_ratio_1d(x: np.ndarray, jump_th: float = 1.0) -> float:
    x = np.asarray(x, dtype=np.float32)
    m = np.isfinite(x)
    if m.sum() < 2:
        return 1.0
    xx = x[m]
    d = np.abs(np.diff(xx))
    return float(np.mean(d > jump_th))


def is_stuck_1d(x: np.ndarray, var_eps: float = 1e-6) -> bool:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return True
    return bool(np.var(x) < var_eps)


def sensor_fault_silver(u_raw: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    Return (is_fault, diagnostics).
    Conservative and interpretable.
    """
    inv = invalid_ratio_1d(u_raw)
    jr = jump_ratio_1d(u_raw, jump_th=1.0)
    stuck = 1.0 if is_stuck_1d(u_raw, var_eps=1e-6) else 0.0

    # Thresholds chosen to be defensible rather than overly tuned.
    # You can report these in reproducibility.md.
    is_fault = (inv > 0.05) or (jr > 0.10) or (stuck > 0.5)

    diag = {"invalid_ratio": inv, "jump_ratio": jr, "stuck_flag": stuck}
    return bool(is_fault), diag


def load_gold_labels(gold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_csv)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    if "run_id" not in cols:
        raise ValueError(f"Gold label file missing 'run_id': {gold_csv}")

    if "win_start_idx" in cols:
        start_col = cols["win_start_idx"]
    elif "win_start" in cols:
        start_col = cols["win_start"]
    elif "start_idx" in cols:
        start_col = cols["start_idx"]
    else:
        raise ValueError(f"Gold label file missing win_start column: {gold_csv}")

    if "label" not in cols:
        raise ValueError(f"Gold label file missing 'label': {gold_csv}")

    out = pd.DataFrame({
        "run_id": df[cols["run_id"]].astype(str),
        "win_start_idx": pd.to_numeric(df[start_col], errors="coerce").astype("Int64"),
        "label_gold": df[cols["label"]].astype(str),
    })
    out = out.dropna(subset=["win_start_idx"]).copy()
    out["win_start_idx"] = out["win_start_idx"].astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--clean_channels", type=int, nargs="*", default=[0, 1, 2])  # first 3 displacements
    ap.add_argument("--raw_channels", type=int, nargs="*", default=[0, 1, 2, 3])   # first 4 displacements
    ap.add_argument("--gold_labels", type=str, default="")  # optional path to CSV
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "01_extract_windows_labels.log")

    manifest_path = processed_dir / "runs_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing runs_manifest.json. Run Step 00 first. Expected: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = sorted(manifest["runs"].keys())
    if not runs:
        raise RuntimeError("runs_manifest.json has no runs.")

    logger.info(f"Windowing runs: {len(runs)} | seq_len={args.seq_len} stride={args.stride}")

    X_clean_all = []
    X_raw_all = []
    rows = []

    for run_id in runs:
        npz_rel = manifest["runs"][run_id]["npz"]
        npz_path = processed_dir / npz_rel
        data = np.load(npz_path, allow_pickle=True)

        disp = data["disp_mm"].astype(np.float32)  # (M,K)
        time_s = data["time_s"].astype(np.float32) if "time_s" in data else np.arange(len(disp), dtype=np.float32)

        if disp.ndim != 2:
            raise ValueError(f"{run_id}: disp_mm must be 2D, got {disp.shape}")

        # select channels
        if max(args.raw_channels) >= disp.shape[1]:
            raise ValueError(f"{run_id}: raw_channels {args.raw_channels} invalid for disp shape {disp.shape}")
        if max(args.clean_channels) >= disp.shape[1]:
            raise ValueError(f"{run_id}: clean_channels {args.clean_channels} invalid for disp shape {disp.shape}")

        disp_raw = disp[:, args.raw_channels]
        disp_clean = disp[:, args.clean_channels]

        # windowize
        Xr, idx0 = windowize_2d(disp_raw, args.seq_len, args.stride)
        Xc, idx0c = windowize_2d(disp_clean, args.seq_len, args.stride)
        if len(idx0) != len(idx0c):
            raise RuntimeError(f"{run_id}: window index mismatch between raw and clean.")

        # silver labels: based on the “integrity” channel (conservative)
        # Default policy: use the last raw channel (often LWA_4) as primary SF signal.
        integrity_channel = Xr[:, :, -1]  # (N,T)
        for i, s in enumerate(idx0):
            u_int = integrity_channel[i]
            is_sf, diag = sensor_fault_silver(u_int)

            # Silver label logic:
            # - SF if sensor-fault flags
            # - otherwise "N" by default here. (Exceedance "E" is typically a gold label or comes from separate logic.)
            # You can later extend E_silver using force thresholds if you want.
            label_silver = "SF" if is_sf else "N"

            rows.append({
                "run_id": run_id,
                "win_start_idx": int(s),
                "t_start_s": float(time_s[s]) if s < len(time_s) else float("nan"),
                "label_silver": label_silver,
                "invalid_ratio": diag["invalid_ratio"],
                "jump_ratio": diag["jump_ratio"],
                "stuck_flag": diag["stuck_flag"],
            })

        X_clean_all.append(Xc)
        X_raw_all.append(Xr)

        logger.info(f"{run_id}: windows={Xr.shape[0]}")

    X_clean = np.concatenate(X_clean_all, axis=0).astype(np.float32)
    X_raw = np.concatenate(X_raw_all, axis=0).astype(np.float32)
    meta_silver = pd.DataFrame(rows)

    out_x_clean = processed_dir / "X_clean.npy"
    out_x_raw = processed_dir / "X_raw.npy"
    out_silver = processed_dir / "window_labels_silver.csv"
    out_aug = processed_dir / "window_labels_augmented.csv"

    np.save(out_x_clean, X_clean)
    np.save(out_x_raw, X_raw)
    meta_silver.to_csv(out_silver, index=False)

    # Augment with gold if provided
    meta_aug = meta_silver.copy()
    meta_aug["label"] = meta_aug["label_silver"]
    meta_aug["label_source"] = "silver"

    if args.gold_labels:
        gold_path = resolve_under_root(args.gold_labels, root=root)
        gold = load_gold_labels(gold_path)

        meta_aug = meta_aug.merge(gold, on=["run_id", "win_start_idx"], how="left")
        has_gold = meta_aug["label_gold"].notna()
        meta_aug.loc[has_gold, "label"] = meta_aug.loc[has_gold, "label_gold"]
        meta_aug.loc[has_gold, "label_source"] = "gold"

        meta_aug = meta_aug.drop(columns=["label_gold"])

    meta_aug.to_csv(out_aug, index=False)

    logger.info(f"Saved: {out_x_clean}")
    logger.info(f"Saved: {out_x_raw}")
    logger.info(f"Saved: {out_silver}")
    logger.info(f"Saved: {out_aug}")
    logger.info(f"Shapes: X_clean={X_clean.shape}, X_raw={X_raw.shape}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
