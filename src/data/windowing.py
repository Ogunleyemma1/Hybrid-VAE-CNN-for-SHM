# src/data/windowing.py
from __future__ import annotations

from typing import Tuple

import numpy as np


def windowize_2d(A: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a continuous multivariate signal A into overlapping windows.

    Args:
        A: array of shape (T, C)
        seq_len: window length L
        stride: hop size

    Returns:
        X: windows of shape (N, L, C)
        idx0: start indices of each window, shape (N,)

    Notes:
        - Deterministic.
        - Returns empty arrays if T < seq_len.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"Expected A with shape (T,C), got {A.shape}")
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive integers")

    n = A.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, A.shape[1]), dtype=np.float32), np.empty((0,), dtype=int)

    X, idx0 = [], []
    for i in range(0, n - seq_len + 1, stride):
        X.append(A[i : i + seq_len])
        idx0.append(i)

    return np.asarray(X, dtype=np.float32), np.asarray(idx0, dtype=int)


def windowize_1d(x: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a continuous 1D signal into overlapping windows.

    Args:
        x: array of shape (T,)
        seq_len: window length L
        stride: hop size

    Returns:
        W: windows of shape (N, L)
        idx0: start indices of each window, shape (N,)
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected x with shape (T,), got {x.shape}")
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive integers")

    n = x.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len), dtype=np.float32), np.empty((0,), dtype=int)

    W, idx0 = [], []
    for i in range(0, n - seq_len + 1, stride):
        W.append(x[i : i + seq_len])
        idx0.append(i)

    return np.asarray(W, dtype=np.float32), np.asarray(idx0, dtype=int)
