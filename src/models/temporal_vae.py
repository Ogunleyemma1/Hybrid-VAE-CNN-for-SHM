# src/models/temporal_vae.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VAEOutput:
    """Structured output for TemporalVAE forward pass."""
    recon: torch.Tensor          # (B, T, C)
    mu: torch.Tensor             # (B, Z)
    logvar: torch.Tensor         # (B, Z)
    z: Optional[torch.Tensor]    # (B, Z) if requested


class TemporalVAE(nn.Module):
    """
    Temporal (LSTM-based) Variational Autoencoder for multivariate time-series windows.

    Expected input shape:
        x: (B, T, C)
            B: batch size
            T: sequence length (window length)
            C: number of channels / features

    Encoder:
        LSTM over time -> last hidden state -> (mu, logvar)

    Reparameterization:
        z = mu + eps * exp(0.5 * logvar)

    Decoder:
        z -> hidden vector -> repeated across time -> LSTM -> linear projection -> recon (B, T, C)

    Notes for reproducibility / publication:
    - The model is intentionally simple and shape-explicit.
    - LayerNorm stabilizes the encoder hidden representation.
    - Decoder input is a repeated latent-conditioned hidden vector (as in your current logic).
    """

    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim, latent_dim, and hidden_dim must be positive integers.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.encoder_norm = nn.LayerNorm(self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder
        self.fc_latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.input_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize weights with common, stable defaults for sequence models.
        This improves training stability and makes initialization explicit.
        """
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Linear layers
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

        nn.init.xavier_uniform_(self.fc_latent_to_hidden.weight)
        nn.init.zeros_(self.fc_latent_to_hidden.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of sequences into Gaussian parameters.

        Args:
            x: (B, T, C)

        Returns:
            mu: (B, Z)
            logvar: (B, Z)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,C), got {tuple(x.shape)}.")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got C={x.size(-1)}.")

        _, (h_n, _) = self.encoder_lstm(x)   # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]                     # (B, hidden_dim)
        h_last = self.encoder_norm(h_last)

        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick:
            z = mu + eps * std,  eps ~ N(0, I), std = exp(0.5*logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent vectors into sequences.

        Args:
            z: (B, Z)
            seq_len: T (window length)

        Returns:
            recon: (B, T, C)
        """
        if z.ndim != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(f"Expected z with shape (B,{self.latent_dim}), got {tuple(z.shape)}.")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")

        # (B,Z) -> (B,H) -> (B,1,H) -> (B,T,H)
        h0 = torch.tanh(self.fc_latent_to_hidden(z)).unsqueeze(1).repeat(1, seq_len, 1)

        decoded_seq, _ = self.decoder_lstm(h0)      # (B, T, H)
        recon = self.output_layer(decoded_seq)      # (B, T, C)
        return recon

    def forward(self, x: torch.Tensor, return_z: bool = False) -> VAEOutput:
        """
        Full VAE forward pass.

        Args:
            x: (B, T, C)
            return_z: if True, include sampled z in output.

        Returns:
            VAEOutput(recon, mu, logvar, z or None)
        """
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar, z=z if return_z else None)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns reconstructions only."""
        return self.forward(x, return_z=False).recon

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample sequences from the prior z ~ N(0, I).

        Returns:
            samples: (B, T, C)
        """
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive.")
        device = device or next(self.parameters()).device
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z, seq_len)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
            f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}"
        )
