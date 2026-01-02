# src/models/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# VAE losses
# -------------------------

def vae_reconstruction_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean-squared reconstruction error for continuous sensor time-series.

    Shapes:
        recon, target: (B, T, C)

    Returns:
        Scalar tensor: mean over all elements.
    """
    if recon.shape != target.shape:
        raise ValueError(f"Shape mismatch: recon={tuple(recon.shape)} vs target={tuple(target.shape)}")
    return F.mse_loss(recon, target, reduction="mean")


def vae_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    KL divergence KL(q(z|x) || p(z)) with q = N(mu, diag(exp(logvar))), p = N(0, I).

    We compute:
        KL = 0.5 * sum_j (exp(logvar_j) + mu_j^2 - 1 - logvar_j)

    Scaling:
        - reduction="mean": mean over batch (standard for stable training).
        - reduction="sum": sum over batch.

    Shapes:
        mu, logvar: (B, Z)
    """
    if mu.shape != logvar.shape:
        raise ValueError(f"Shape mismatch: mu={tuple(mu.shape)} vs logvar={tuple(logvar.shape)}")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")

    kl_per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)  # (B, Z)
    kl_per_sample = kl_per_dim.sum(dim=1)  # sum over latent dims => (B,)

    return kl_per_sample.mean() if reduction == "mean" else kl_per_sample.sum()


def beta_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    kl_reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Beta-VAE objective:
        L = recon_mse + beta * KL

    Returns:
        total, recon_loss, kl_loss
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    r = vae_reconstruction_mse(recon, target)
    k = vae_kl_divergence(mu, logvar, reduction=kl_reduction)
    total = r + float(beta) * k
    return total, r, k


# -------------------------
# CNN loss (Focal)
# -------------------------

class WeightedFocalLoss(nn.Module):
    """
    Weighted focal loss for classification with class imbalance.

    Args:
        alpha: Optional tensor of shape (num_classes,) giving per-class weights.
               If provided, alpha[target] rescales the loss per sample.
        gamma: Focusing parameter (>= 0). Typical: 1-3.

    Notes:
        - Expects logits (unnormalized scores).
        - Targets are integer class indices.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")  # (B,)
        pt = torch.exp(-ce)                                      # (B,)
        loss = ((1.0 - pt) ** self.gamma) * ce                   # (B,)

        if self.alpha is not None:
            if self.alpha.ndim != 1:
                raise ValueError("alpha must be a 1D tensor of shape (num_classes,)")
            loss = self.alpha[targets] * loss

        return loss.mean()
