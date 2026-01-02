# src/evaluation/__init__.py
"""
Evaluation subpackage for the OpenLAB hybrid SHM pipeline.

Contains:
- metrics: lightweight metrics (precision/recall/F1, accuracy) without sklearn dependency
- confusion: confusion matrix utilities for binary and multiclass classification
- thresholding: threshold selection for VAE gating and CNN decision thresholds
"""

from .metrics import (
    accuracy,
    precision_recall_f1,
    classification_report,
)
from .confusion import (
    confusion_matrix,
    normalize_confusion,
    format_confusion_matrix,
)
from .thresholding import (
    mse_per_window,
    select_vae_mse_threshold,
    select_cnn_threshold_precision_floor,
)

__all__ = [
    # metrics
    "accuracy",
    "precision_recall_f1",
    "classification_report",
    # confusion
    "confusion_matrix",
    "normalize_confusion",
    "format_confusion_matrix",
    # thresholding
    "mse_per_window",
    "select_vae_mse_threshold",
    "select_cnn_threshold_precision_floor",
]
