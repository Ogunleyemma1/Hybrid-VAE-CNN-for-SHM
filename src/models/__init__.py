# src/models/__init__.py
"""
Models and training utilities for the hybrid VAE-CNN SHM workflow.

Public API
----------
- TemporalVAE: sequence-to-sequence variational autoencoder for multivariate time series
- CNNClassifier: CNN classifier for window-level discrimination (e.g., SF vs E)
- Loss utilities: reconstruction_loss, kl_divergence, beta_vae_loss, WeightedFocalLoss
- Checkpoint helpers: save_checkpoint, load_checkpoint, find_latest_checkpoint
"""

from .temporal_vae import TemporalVAE
from .cnn_model import CNNClassifier

from .losses import (
    reconstruction_loss,
    vae_kl_divergence,
    kl_divergence,      # alias for vae_kl_divergence
    beta_vae_loss,
    WeightedFocalLoss,
)

from .checkpoints import save_checkpoint, load_checkpoint, find_latest_checkpoint

__all__ = [
    "TemporalVAE",
    "CNNClassifier",
    "reconstruction_loss",
    "vae_kl_divergence",
    "kl_divergence",
    "beta_vae_loss",
    "WeightedFocalLoss",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
]
