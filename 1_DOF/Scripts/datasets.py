from __future__ import annotations

import numpy as np


def compute_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    x: (T, F)
    returns mean, std (F,)
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0.0, 1e-6, std)
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def destandardize(xn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return xn * std + mean


def make_windows(x: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    """
    x: (T, F)
    returns windows: (N, seq_len, F)
    """
    T = x.shape[0]
    if T < seq_len:
        raise ValueError(f"Time series too short: T={T} < seq_len={seq_len}")
    idx = range(0, T - seq_len + 1, stride)
    windows = np.stack([x[i:i + seq_len] for i in idx], axis=0)
    return windows


def stitch_windows(windows: np.ndarray, full_len: int, stride: int = 1) -> np.ndarray:
    """
    windows: (N, seq_len, F) from sliding window reconstruction
    returns stitched series: (full_len, F)
    """
    N, seq_len, F = windows.shape
    out = np.zeros((full_len, F), dtype=float)
    cnt = np.zeros((full_len, 1), dtype=float)

    for n in range(N):
        start = n * stride
        end = start + seq_len
        out[start:end] += windows[n]
        cnt[start:end] += 1.0

    cnt[cnt == 0.0] = 1.0
    return out / cnt


def segment_rmse(y_true: np.ndarray, y_pred: np.ndarray, segment_len: int) -> np.ndarray:
    """
    y_true, y_pred: (T, F)
    returns rmse per segment: (S,)
    """
    T = y_true.shape[0]
    S = int(np.ceil(T / segment_len))
    rmses = []
    for s in range(S):
        i0 = s * segment_len
        i1 = min((s + 1) * segment_len, T)
        e = y_pred[i0:i1] - y_true[i0:i1]
        rmse = float(np.sqrt(np.mean(e**2)))
        rmses.append(rmse)
    return np.array(rmses, dtype=float)
