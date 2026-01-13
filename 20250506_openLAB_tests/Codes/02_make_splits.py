"""
02_make_splits.py

Creates run-based splits (train/val/test) and saves run_split.json.

Design:
- Run-level splitting prevents leakage across windows from the same run.
- Ensures each split has sufficient Normal windows for stable training/validation.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

import config as C
from io_utils import save_json

LABEL_NORMAL = "Normal"
MIN_NORMAL_WINDOWS = 200  # adjust if needed


def main() -> None:
    random.seed(C.SEED)
    np.random.seed(C.SEED)

    meta_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["meta"])
    out_path = os.path.join(C.OUT_DIR, C.ARTIFACTS["splits"])

    meta = pd.read_csv(meta_path)
    if "run_id" not in meta.columns or "label" not in meta.columns:
        raise ValueError("window_labels.csv must include 'run_id' and 'label' columns.")

    runs = sorted(meta["run_id"].astype(str).unique().tolist())
    if len(runs) < 3:
        raise ValueError("Need at least 3 runs to create train/val/test run splits.")

    random.shuffle(runs)

    n = len(runs)
    n_train = max(1, int(round(float(C.TRAIN_FRAC) * n)))
    n_val = max(1, int(round(float(C.VAL_FRAC) * n)))
    n_test = max(1, n - n_train - n_val)

    # Adjust to sum exactly
    while n_train + n_val + n_test > n:
        n_test = max(1, n_test - 1)
    while n_train + n_val + n_test < n:
        n_test += 1

    train_runs = runs[:n_train]
    val_runs = runs[n_train : n_train + n_val]
    test_runs = runs[n_train + n_val :]

    def count_normals(run_list: list[str]) -> int:
        m = meta["run_id"].astype(str).isin(run_list) & (meta["label"] == LABEL_NORMAL)
        return int(m.sum())

    nN_train = count_normals(train_runs)
    nN_val = count_normals(val_runs)
    nN_test = count_normals(test_runs)

    if nN_train < MIN_NORMAL_WINDOWS or nN_val < max(50, MIN_NORMAL_WINDOWS // 4):
        raise RuntimeError(
            "Not enough Normal windows in train/val under this run split.\n"
            f"Normals: train={nN_train}, val={nN_val}, test={nN_test}\n"
            "Fix: change TRAIN_FRAC/VAL_FRAC in config.py or reduce MIN_NORMAL_WINDOWS."
        )

    out = {
        "seed": int(C.SEED),
        "fractions": {
            "train_frac": float(C.TRAIN_FRAC),
            "val_frac": float(C.VAL_FRAC),
            "test_frac": float(C.TEST_FRAC),
        },
        "train_runs": train_runs,
        "val_runs": val_runs,
        "test_runs": test_runs,
        "counts": {
            "n_runs": int(n),
            "n_train_runs": int(len(train_runs)),
            "n_val_runs": int(len(val_runs)),
            "n_test_runs": int(len(test_runs)),
            "n_normal_train": int(nN_train),
            "n_normal_val": int(nN_val),
            "n_normal_test": int(nN_test),
        },
    }

    save_json(out_path, out)

    print(f"Saved split: {out_path}")
    print(pd.Series(out["counts"]).to_string())
    print("train_runs:", train_runs)
    print("val_runs:", val_runs)
    print("test_runs:", test_runs)


if __name__ == "__main__":
    main()
