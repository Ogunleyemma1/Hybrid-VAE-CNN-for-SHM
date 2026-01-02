# src/models/losses.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# VAE losses
# -----------------------------------------------------------------------------

def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss for continuous, multivariate time-series.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstructed signal of shape (B, T, C).
    target : torch.Tensor
        Ground-truth signal of shape (B, T, C).

    Returns
    -------
    torch.Tensor
        Scalar tensor: mean squared error over all elements.
    """
    if recon.shape != target.shape:
        raise ValueError(f"Shape mismatch: recon={tuple(recon.shape)} vs target={tuple(target.shape)}")
    return F.mse_loss(recon, target, reduction="mean")


def vae_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    KL divergence KL(q(z|x) || p(z)) for a diagonal Gaussian posterior.

    q(z|x) = N(mu, diag(exp(logvar)))
    p(z)   = N(0, I)

    KL per sample:
        0.5 * sum_j (exp(logvar_j) + mu_j^2 - 1 - logvar_j)

    Parameters
    ----------
    mu : torch.Tensor
        Mean of q(z|x), shape (B, Z).
    logvar : torch.Tensor
        Log-variance of q(z|x), shape (B, Z).
    reduction : str
        "mean" (default) averages over batch; "sum" sums over batch.

    Returns
    -------
    torch.Tensor
        Scalar tensor: KL divergence reduced over batch.
    """
    if mu.shape != logvar.shape:
        raise ValueError(f"Shape mismatch: mu={tuple(mu.shape)} vs logvar={tuple(logvar.shape)}")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")

    kl_per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)  # (B, Z)
    kl_per_sample = kl_per_dim.sum(dim=1)  # (B,)

    return kl_per_sample.mean() if reduction == "mean" else kl_per_sample.sum()


# Public alias for stable API naming
kl_divergence = vae_kl_divergence


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
        L = reconstruction_loss + beta * KL

    Parameters
    ----------
    recon, target : torch.Tensor
        Reconstruction and ground-truth, shape (B, T, C).
    mu, logvar : torch.Tensor
        Latent posterior parameters, shape (B, Z).
    beta : float
        Non-negative scaling factor for KL term.
    kl_reduction : str
        "mean" or "sum" reduction for KL.

    Returns
    -------
    total : torch.Tensor
    recon_loss : torch.Tensor
    kl_loss : torch.Tensor
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    r = reconstruction_loss(recon, target)
    k = vae_kl_divergence(mu, logvar, reduction=kl_reduction)
    total = r + float(beta) * k
    return total, r, k


# -----------------------------------------------------------------------------
# CNN loss (Weighted focal loss)
# -----------------------------------------------------------------------------

class WeightedFocalLoss(nn.Module):
    """
    Weighted focal loss for multi-class classification under class imbalance.

    This loss expects:
    - logits: shape (B, num_classes) (unnormalized scores)
    - targets: shape (B,) integer class indices

    Parameters
    ----------
    alpha : Optional[torch.Tensor]
        1D tensor of shape (num_classes,) giving per-class weights.
        When provided, alpha[targets] scales per-sample loss.
    gamma : float
        Focusing parameter (>= 0). Typical values: 1 to 3.
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
