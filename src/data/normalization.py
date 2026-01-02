# src/data/normalization.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class NormStats:
    """
    Normalization statistics for windowed time-series.

    mu/sigma are per-channel parameters applied to the last dimension.
    """
    method: Literal["standard", "robust"]
    mu: np.ndarray
    sigma: np.ndarray

    def as_dict(self) -> Dict:
        return {
            "method": self.method,
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }

    @staticmethod
    def from_dict(d: Dict) -> "NormStats":
        return NormStats(
            method=d["method"],
            mu=np.asarray(d["mu"], dtype=np.float32),
            sigma=np.asarray(d["sigma"], dtype=np.float32),
        )


def _validate_windows(X: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(f"Expected window tensor X with shape (N,L,C), got {X.shape}")
    if X.shape[-1] < 1:
        raise ValueError("Expected at least one channel")


def fit_standard_scaler(X_train: np.ndarray, eps: float = 1e-8) -> NormStats:
    """
    Fit per-channel mean/std on training windows only.

    Args:
        X_train: (N, L, C)
        eps: small constant to avoid division by zero

    Returns:
        NormStats(method="standard", mu, sigma)
    """
    _validate_windows(X_train)
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0).astype(np.float32)
    sigma = X_train.reshape(-1, X_train.shape[-1]).std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, eps).astype(np.float32)
    return NormStats(method="standard", mu=mu, sigma=sigma)


def apply_standard_scaler(X: np.ndarray, stats: NormStats) -> np.ndarray:
    """
    Apply per-channel standard scaling: (x - mu) / sigma.
    """
    _validate_windows(X)
    if stats.method != "standard":
        raise ValueError(f"Stats method is {stats.method}, expected 'standard'.")
    return ((X - stats.mu) / stats.sigma).astype(np.float32)


def fit_robust_scaler(X_train: np.ndarray, eps: float = 1e-8) -> NormStats:
    """
    Fit per-channel robust scaling parameters (median/MAD).

    Uses:
        mu := median
        sigma := 1.4826 * MAD

    This matches the style of robust stats you already compute in your CNN scripts. :contentReference[oaicite:3]{index=3}
    """
    _validate_windows(X_train)
    flat = X_train.reshape(-1, X_train.shape[-1])
    med = np.median(flat, axis=0).astype(np.float32)
    mad = np.median(np.abs(flat - med), axis=0).astype(np.float32)
    sigma = (1.4826 * mad).astype(np.float32)
    sigma = np.maximum(sigma, eps).astype(np.float32)
    return NormStats(method="robust", mu=med, sigma=sigma)


def apply_robust_scaler(X: np.ndarray, stats: NormStats) -> np.ndarray:
    """
    Apply per-channel robust scaling: (x - median) / (1.4826*MAD).
    """
    _validate_windows(X)
    if stats.method != "robust":
        raise ValueError(f"Stats method is {stats.method}, expected 'robust'.")
    return ((X - stats.mu) / stats.sigma).astype(np.float32)


def save_norm_stats(path: PathLike, stats: NormStats) -> None:
    """Save normalization stats to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(stats.as_dict(), f, indent=2)


def load_norm_stats(path: PathLike) -> NormStats:
    """Load normalization stats from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        d = json.load(f)
    return NormStats.from_dict(d)
