# src/evaluation/confusion.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Sequence[int]]:
    """
    Compute a confusion matrix for multiclass classification.

    Convention:
        rows = true labels
        cols = predicted labels

    Args:
        y_true, y_pred: integer label sequences
        labels: explicit label order; if None, inferred from union.

    Returns:
        cm: (K, K) ndarray
        labels_used: list of labels in the order used
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels_used = list(map(int, labels))

    k = len(labels_used)
    idx = {lab: i for i, lab in enumerate(labels_used)}
    cm = np.zeros((k, k), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[idx[int(t)], idx[int(p)]] += 1

    return cm, labels_used


def normalize_confusion(cm: np.ndarray, mode: str = "true", eps: float = 1e-12) -> np.ndarray:
    """
    Normalize confusion matrix.

    Args:
        cm: (K, K)
        mode:
            - "true": row-normalized (each row sums to 1) => per-true-class distribution
            - "pred": col-normalized (each col sums to 1) => per-pred-class distribution
            - "all": global normalization (sum to 1)
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square (K,K) matrix.")

    if mode == "true":
        denom = cm.sum(axis=1, keepdims=True) + eps
        return cm / denom
    if mode == "pred":
        denom = cm.sum(axis=0, keepdims=True) + eps
        return cm / denom
    if mode == "all":
        return cm / (cm.sum() + eps)

    raise ValueError("mode must be one of: 'true', 'pred', 'all'.")


def format_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[int],
    label_names: Optional[Dict[int, str]] = None,
) -> str:
    """
    Create a simple aligned string representation for logs.

    Args:
        cm: (K,K)
        labels: list of label ids (length K)
        label_names: optional mapping label -> string
    """
    labels = list(labels)
    names = [label_names.get(l, str(l)) if label_names else str(l) for l in labels]

    # Column widths
    max_name = max(len(n) for n in names)
    max_val = max(len(str(int(v))) for v in cm.flatten()) if cm.size else 1
    cell_w = max(max_val, 5)

    header = " " * (max_name + 2) + "".join([f"{n:>{cell_w}s}" for n in names])
    rows = [header]
    for i, n in enumerate(names):
        row = f"{n:<{max_name}s}  " + "".join([f"{int(cm[i, j]):>{cell_w}d}" for j in range(len(names))])
        rows.append(row)
    return "\n".join(rows)
