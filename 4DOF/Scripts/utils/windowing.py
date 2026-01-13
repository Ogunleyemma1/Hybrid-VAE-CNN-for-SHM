# Scripts/utils/windowing.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch


@dataclass
class WindowConfig:
    window_len: int
    stride: int


def make_windows(X: np.ndarray, win: WindowConfig) -> np.ndarray:
    """
    X: [T, D] -> windows: [N, window_len, D]
    """
    T, D = X.shape
    w = win.window_len
    s = win.stride
    if T < w:
        raise ValueError(f"Time length T={T} < window_len={w}")

    starts = list(range(0, T - w + 1, s))
    out = np.stack([X[i : i + w, :] for i in starts], axis=0)
    return out


def fit_normal_stats(windows: np.ndarray) -> Dict[str, np.ndarray]:
    """
    windows: [N, W, D], fit mean/std over (N,W).
    """
    mu = windows.reshape(-1, windows.shape[-1]).mean(axis=0)
    sd = windows.reshape(-1, windows.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return {"mean": mu, "std": sd}


def apply_normalization(windows: np.ndarray, stats: Dict[str, np.ndarray], clip: float | None = None) -> np.ndarray:
    mu = stats["mean"]
    sd = stats["std"]
    Z = (windows - mu[None, None, :]) / sd[None, None, :]
    if clip is not None:
        Z = np.clip(Z, -clip, clip)
    return Z


def save_stats_npz(path: str, stats: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=stats["mean"], std=stats["std"])


def load_stats_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {"mean": d["mean"], "std": d["std"]}


def to_torch(windows: np.ndarray) -> torch.Tensor:
    return torch.tensor(windows, dtype=torch.float32)
