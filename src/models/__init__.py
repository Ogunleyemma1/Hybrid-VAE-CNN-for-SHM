# src/models/__init__.py
"""
Model package for the OpenLAB hybrid SHM pipeline.

This package contains:
- TemporalVAE: sequence-to-sequence variational autoencoder for multivariate time series
- CNNClassifier: 2D CNN classifier for window-level fault discrimination
- Loss utilities and checkpoint helpers
"""

from .temporal_vae import TemporalVAE
from .cnn_model import CNNClassifier
from .losses import beta_vae_loss, kl_divergence, reconstruction_loss
from .checkpoints import save_checkpoint, load_checkpoint, find_latest_checkpoint

__all__ = [
    "TemporalVAE",
    "CNNClassifier",
    "beta_vae_loss",
    "kl_divergence",
    "reconstruction_loss",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
]
