# experiments/4dof/scripts/01_extract_windows_labels.py
"""
Step 01 (4DOF): Window extraction + labels + manifest.

Input:
  experiments/4dof/datasets/raw/
    normal/*.csv
    faults/sensor_faults/**/*.csv
    faults/structural_faults/**/*.csv

Output:
  experiments/4dof/datasets/processed/
    X.npy
    meta_windows.csv

meta_windows.csv columns:
  - run_id
  - label   (N / SF / ST)
  - source_file
  - start_idx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


def windowize(A: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    M = A.shape[0]
    if M < seq_len:
        return np.zeros((0, seq_len, A.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    idx0 = np.arange(0, M - seq_len + 1, stride, dtype=np.int64)
    X = np.stack([A[i : i + seq_len] for i in idx0], axis=0).astype(np.float32)
    return X, idx0


def load_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    X = df.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def run_id_from_path(p: Path) -> str:
    # stable run id: folder + stem
    return f"{p.parent.name}__{p.stem}"


def gather_files(raw_dir: Path) -> List[Tuple[Path, str]]:
    normal = sorted((raw_dir / "normal").glob("*.csv"))
    sensor = sorted((raw_dir / "faults" / "sensor_faults").rglob("*.csv"))
    struct = sorted((raw_dir / "faults" / "structural_faults").rglob("*.csv"))

    out: List[Tuple[Path, str]] = []
    out += [(p, "N") for p in normal]
    out += [(p, "SF") for p in sensor]
    out += [(p, "ST") for p in struct]  # Structural Fault
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="experiments/4dof/datasets/raw")
    ap.add_argument("--processed_dir", type=str, default="experiments/4dof/datasets/processed")
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    root = find_repo_root()
    raw_dir = resolve_under_root(args.raw_dir, root=root)
    processed_dir = resolve_under_root(args.processed_dir, root=root)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "01_extract_windows_labels.log")

    pairs = gather_files(raw_dir)
    if not pairs:
        raise FileNotFoundError(f"No CSV files found under: {raw_dir}. Run Step 00 first.")

    X_all = []
    rows = []

    for fp, lab in pairs:
        A = load_csv(fp)
        Xw, idx0 = windowize(A, args.seq_len, args.stride)
        rid = run_id_from_path(fp)

        for s in idx0:
            rows.append(
                {
                    "run_id": rid,
                    "label": lab,  # N / SF / ST
                    "source_file": str(fp.relative_to(raw_dir)),
                    "start_idx": int(s),
                }
            )

        if Xw.shape[0] > 0:
            X_all.append(Xw)

        logger.info(f"{lab} | {fp.name} | windows={Xw.shape[0]}")

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    meta = pd.DataFrame(rows)

    np.save(processed_dir / "X.npy", X)
    meta.to_csv(processed_dir / "meta_windows.csv", index=False)

    logger.info(f"Saved X: {processed_dir / 'X.npy'} | shape={X.shape}")
    logger.info(f"Saved meta: {processed_dir / 'meta_windows.csv'} | rows={len(meta)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
