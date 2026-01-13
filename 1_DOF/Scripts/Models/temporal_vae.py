from __future__ import annotations

import torch
import torch.nn as nn


class TemporalVAE(nn.Module):
    """
    LSTM encoder -> latent (mu, logvar) -> LSTM decoder reconstruction.
    """
    def __init__(self, input_dim: int = 12, latent_dim: int = 5, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # (B, H)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        h0 = torch.tanh(self.fc_latent_to_hidden(z)).unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(h0)
        return self.output_layer(decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar
