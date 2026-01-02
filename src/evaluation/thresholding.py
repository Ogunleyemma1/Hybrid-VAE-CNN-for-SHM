# src/evaluation/thresholding.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def mse_per_window(x: np.ndarray, xhat: np.ndarray) -> np.ndarray:
    """
    Compute per-window MSE for windowed time series.

    Shapes:
        x, xhat: (N, L, C)

    Returns:
        mse: (N,)  mean over (L,C)
    """
    x = np.asarray(x)
    xhat = np.asarray(xhat)
    if x.shape != xhat.shape:
        raise ValueError(f"Shape mismatch: x={x.shape} vs xhat={xhat.shape}")
    if x.ndim != 3:
        raise ValueError("Expected (N,L,C) arrays")

    return np.mean((x - xhat) ** 2, axis=(1, 2))


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    details: Dict[str, float]


def select_vae_mse_threshold(
    mse_normals: Sequence[float],
    percentile: float = 99.0,
) -> ThresholdResult:
    """
    Select VAE gate threshold from reconstruction MSE on NORMAL validation windows.

    Common policy:
        threshold = Pxx percentile of MSE on validation normals
    (This matches your current approach.)

    Args:
        mse_normals: iterable of MSE values from NORMAL windows
        percentile: e.g. 99 for P99

    Returns:
        ThresholdResult(threshold, details) where details include mean/std/min/max/pXX.
    """
    mse_normals = np.asarray(list(mse_normals), dtype=float)
    if mse_normals.size == 0:
        raise ValueError("mse_normals is empty; cannot select threshold.")

    thr = float(np.percentile(mse_normals, percentile))
    details = {
        "mean": float(mse_normals.mean()),
        "std": float(mse_normals.std()),
        "min": float(mse_normals.min()),
        "max": float(mse_normals.max()),
        f"p{int(percentile)}": thr,
        "n": float(mse_normals.size),
    }
    return ThresholdResult(threshold=thr, details=details)


def _binary_pr(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Precision/recall for a chosen positive class in a binary setting.
    """
    tp = float(((y_true == pos_label) & (y_pred == pos_label)).sum())
    fp = float(((y_true != pos_label) & (y_pred == pos_label)).sum())
    fn = float(((y_true == pos_label) & (y_pred != pos_label)).sum())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision, recall


def select_cnn_threshold_precision_floor(
    y_true: Sequence[int],
    p_pos: Sequence[float],
    precision_floor: float = 0.15,
    thresholds: Optional[Sequence[float]] = None,
    pos_label: int = 1,
) -> ThresholdResult:
    """
    Select a CNN decision threshold for the positive class (e.g., "E") using:

        maximize recall(pos_label)
        subject to precision(pos_label) >= precision_floor

    This implements your current validation logic in a reusable, testable form.

    Args:
        y_true: ground truth labels (binary or mapped to {0,1})
        p_pos: predicted probabilities for pos_label, shape (N,)
        precision_floor: minimum acceptable precision
        thresholds: optional iterable of candidate thresholds; if None, use np.linspace
        pos_label: label treated as positive (default 1)

    Returns:
        ThresholdResult with chosen threshold and summary metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)
    if y_true.shape != p_pos.shape:
        raise ValueError("y_true and p_pos must have the same shape.")
    if y_true.size == 0:
        raise ValueError("Empty arrays; cannot select threshold.")

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_thr = None
    best_recall = -1.0
    best_precision = 0.0

    for thr in thresholds:
        y_pred = (p_pos >= float(thr)).astype(int)
        precision, recall = _binary_pr(y_true, y_pred, pos_label=pos_label)

        if precision >= precision_floor and recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_thr = float(thr)

    # Fallback: if no threshold satisfies precision floor, pick threshold with best F1
    if best_thr is None:
        best_f1 = -1.0
        for thr in thresholds:
            y_pred = (p_pos >= float(thr)).astype(int)
            precision, recall = _binary_pr(y_true, y_pred, pos_label=pos_label)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
                best_precision = precision
                best_recall = recall

        details = {
            "policy": "fallback_best_f1",
            "precision_floor": float(precision_floor),
            "precision": float(best_precision),
            "recall": float(best_recall),
            "n": float(y_true.size),
        }
        return ThresholdResult(threshold=float(best_thr), details=details)

    details = {
        "policy": "max_recall_subject_to_precision_floor",
        "precision_floor": float(precision_floor),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "n": float(y_true.size),
    }
    return ThresholdResult(threshold=float(best_thr), details=details)
