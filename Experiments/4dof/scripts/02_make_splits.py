# experiments/4dof/scripts/02_make_splits.py
"""
Step 02 (4DOF): Deterministic run-level split (train/val/test).

Input:
  processed/meta_windows.csv

Output:
  processed/run_split.json

Note:
- Run-level splits prevent window leakage across splits.
"""

from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/4dof/datasets/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.40)
    ap.add_argument("--val_frac", type=float, default=0.30)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "02_make_splits.log")

    meta = pd.read_csv(processed_dir / "meta_windows.csv")
    runs = sorted(meta["run_id"].unique().tolist())
    if len(runs) < 3:
        raise RuntimeError("Need at least 3 runs for train/val/test splits.")

    rng = random.Random(args.seed)
    rng.shuffle(runs)

    n = len(runs)
    n_train = max(1, int(round(args.train_frac * n)))
    n_val = max(1, int(round(args.val_frac * n)))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    train = sorted(runs[:n_train])
    val = sorted(runs[n_train:n_train + n_val])
    test = sorted(runs[n_train + n_val:])

    out = {"seed": int(args.seed), "train_runs": train, "val_runs": val, "test_runs": test}
    (processed_dir / "run_split.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info(out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
