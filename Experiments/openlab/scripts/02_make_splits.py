# experiments/openlab/scripts/02_make_splits.py
"""
Step 02: Deterministic RUN-level splits (train/val/test).

Input:
  - experiments/openlab/datasets/processed/runs_manifest.json

Output:
  - experiments/openlab/datasets/processed/run_split.json

Design:
  - run-level splits prevent temporal leakage across windows.
  - seed-controlled shuffling.
  - optional forced runs for val/test to stabilize paper figures.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


def split_runs(runs: List[str], train_frac: float, val_frac: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("Fractions must satisfy: train>0, val>0, train+val<1.0")

    rng = random.Random(seed)
    runs = list(runs)
    rng.shuffle(runs)

    n = len(runs)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise RuntimeError("Not enough runs for a non-empty test split. Add more runs or adjust fractions.")

    train = runs[:n_train]
    val = runs[n_train : n_train + n_val]
    test = runs[n_train + n_val :]

    return sorted(train), sorted(val), sorted(test)


def apply_forced(train: List[str], val: List[str], test: List[str],
                 force_val: List[str], force_test: List[str]) -> Tuple[List[str], List[str], List[str]]:
    all_runs = set(train) | set(val) | set(test)

    fv = [r for r in force_val if r in all_runs]
    ft = [r for r in force_test if r in all_runs]

    # remove forced runs from wherever they are
    train = [r for r in train if r not in fv and r not in ft]
    val = [r for r in val if r not in fv and r not in ft]
    test = [r for r in test if r not in fv and r not in ft]

    # reassign
    val = sorted(set(val) | set(fv))
    test = sorted(set(test) | set(ft))

    # remaining stay train
    train = sorted(set(train))

    # sanity
    if len(set(train) & set(val)) or len(set(train) & set(test)) or len(set(val) & set(test)):
        raise RuntimeError("Forced run assignment produced overlapping splits.")

    if not train or not val or not test:
        raise RuntimeError("Forced run assignment produced an empty split. Adjust forced lists or fractions.")

    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.40)
    ap.add_argument("--val_frac", type=float, default=0.30)
    ap.add_argument("--force_val", type=str, nargs="*", default=[])
    ap.add_argument("--force_test", type=str, nargs="*", default=[])
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "02_make_splits.log")

    manifest_path = processed_dir / "runs_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing: {manifest_path} (run Step 00 first)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = sorted(manifest["runs"].keys())
    if len(runs) < 3:
        raise RuntimeError("Need at least 3 runs for train/val/test splits.")

    train, val, test = split_runs(runs, args.train_frac, args.val_frac, args.seed)

    if args.force_val or args.force_test:
        train, val, test = apply_forced(train, val, test, args.force_val, args.force_test)

    out = {"seed": int(args.seed), "train_runs": train, "val_runs": val, "test_runs": test}
    out_path = processed_dir / "run_split.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info(f"Saved split: {out_path}")
    logger.info(out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
