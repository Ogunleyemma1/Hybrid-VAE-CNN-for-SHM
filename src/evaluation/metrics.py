# src/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PRF1:
    """Precision/Recall/F1 container."""
    precision: float
    recall: float
    f1: float
    support: int


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Classification accuracy.

    Args:
        y_true, y_pred: sequences of integer labels of equal length.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def precision_recall_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
    eps: float = 1e-12,
) -> Dict[int, PRF1]:
    """
    Compute per-class precision/recall/F1.

    Definitions:
        precision_k = TP_k / (TP_k + FP_k)
        recall_k    = TP_k / (TP_k + FN_k)
        f1_k        = 2 * precision_k * recall_k / (precision_k + recall_k)

    Args:
        y_true, y_pred: sequences of integer labels.
        labels: explicit label order; if None, inferred from union of labels.
        eps: numerical stability constant.

    Returns:
        dict: label -> PRF1
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = list(map(int, labels))

    out: Dict[int, PRF1] = {}
    for k in labels:
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        support = int((y_true == k).sum())

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2.0 * prec * rec / (prec + rec + eps)
        out[k] = PRF1(float(prec), float(rec), float(f1), support)

    return out


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Optional[Dict[int, str]] = None,
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute a lightweight report similar to sklearn's classification_report.

    Returns:
        dict with keys per class and aggregate keys:
            - "<class name>" : precision/recall/f1/support
            - "macro avg"    : mean of class metrics
            - "weighted avg" : support-weighted mean of class metrics
            - "accuracy"     : scalar accuracy
    """
    prf = precision_recall_f1(y_true, y_pred, labels=labels)
    labels_used = list(prf.keys())

    report: Dict[str, Dict[str, float]] = {}
    total_support = sum(prf[k].support for k in labels_used)

    # Per-class
    for k in labels_used:
        name = label_names[k] if (label_names and k in label_names) else str(k)
        report[name] = {
            "precision": prf[k].precision,
            "recall": prf[k].recall,
            "f1-score": prf[k].f1,
            "support": float(prf[k].support),
        }

    # Macro avg
    report["macro avg"] = {
        "precision": float(np.mean([prf[k].precision for k in labels_used])),
        "recall": float(np.mean([prf[k].recall for k in labels_used])),
        "f1-score": float(np.mean([prf[k].f1 for k in labels_used])),
        "support": float(total_support),
    }

    # Weighted avg
    if total_support > 0:
        w = np.asarray([prf[k].support for k in labels_used], dtype=float) / float(total_support)
        report["weighted avg"] = {
            "precision": float(np.sum([w[i] * prf[labels_used[i]].precision for i in range(len(labels_used))])),
            "recall": float(np.sum([w[i] * prf[labels_used[i]].recall for i in range(len(labels_used))])),
            "f1-score": float(np.sum([w[i] * prf[labels_used[i]].f1 for i in range(len(labels_used))])),
            "support": float(total_support),
        }
    else:
        report["weighted avg"] = {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1-score": float("nan"),
            "support": 0.0,
        }

    # Accuracy
    report["accuracy"] = {"accuracy": accuracy(y_true, y_pred), "support": float(total_support)}
    return report
