from __future__ import annotations

from typing import Tuple
import numpy as np


def windowize_1d(x: np.ndarray, window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D signal into sliding windows.

    Parameters
    ----------
    x : (N,) array
    window : window length
    stride : step size between windows

    Returns
    -------
    X : (n_windows, window, 1) windows
    starts : (n_windows,) start indices
    """
    x = np.asarray(x)
    n = int(x.shape[0])
    if n < window:
        return np.zeros((0, window, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    starts = np.arange(0, n - window + 1, stride, dtype=np.int64)
    X = np.stack([x[s : s + window] for s in starts], axis=0).astype(np.float32)
    X = X[:, :, None]  # channel dim
    return X, starts
