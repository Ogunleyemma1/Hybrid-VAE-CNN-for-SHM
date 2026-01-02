# src/data/splits.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def make_run_split(
    run_ids: Sequence[str],
    seed: int = 42,
    train_frac: float = 0.4,
    val_frac: float = 0.3,
    test_frac: float = 0.3,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Deterministic RUN-level split (prevents temporal leakage across windows).

    This is the library version of your current script logic. :contentReference[oaicite:2]{index=2}

    Behavior:
    - Shuffle with a fixed RNG seed.
    - Allocate n_train, n_val via floor, remainder to test.
    - Enforce at least 1 run per split if possible.

    Returns:
        train_runs, val_runs, test_runs
    """
    rng = np.random.default_rng(seed)
    run_ids = list(map(str, run_ids))
    rng.shuffle(run_ids)

    n = len(run_ids)
    if n < 3:
        raise ValueError(f"Need at least 3 runs for train/val/test split. Found n={n}.")

    s = float(train_frac + val_frac + test_frac)
    train_frac, val_frac, test_frac = float(train_frac) / s, float(val_frac) / s, float(test_frac) / s

    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))

    if n_train == 0:
        n_train = 1
    if n_val == 0:
        n_val = 1
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    train = run_ids[:n_train]
    val = run_ids[n_train : n_train + n_val]
    test = run_ids[n_train + n_val :]

    return train, val, test


def apply_forced_runs(
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
    force_val: Optional[Iterable[str]] = None,
    force_test: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Force specific run_ids into val/test (useful for consistent reporting/figures).

    Forced run_ids are removed from other splits.

    Returns:
        updated_train, updated_val, updated_test
    """
    force_val = set(map(str, force_val or []))
    force_test = set(map(str, force_test or []))

    train = [r for r in map(str, train) if r not in force_val and r not in force_test]
    val = [r for r in map(str, val) if r not in force_val and r not in force_test]
    test = [r for r in map(str, test) if r not in force_val and r not in force_test]

    val = sorted(set(val).union(force_val))
    test = sorted(set(test).union(force_test))
    train = sorted(set(train))

    # Sanity: no overlap
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Forced-run application created overlapping splits. Check inputs.")

    return train, val, test
