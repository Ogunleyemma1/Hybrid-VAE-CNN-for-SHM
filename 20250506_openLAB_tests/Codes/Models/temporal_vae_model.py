import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder (same as your framework)
        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x: (B, T, C)
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]                # (B, hidden_dim)
        h_last = self.layer_norm(h_last)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """
        Keep your exact logic:
        - map z -> hidden vector
        - repeat across time as the decoder input
        """
        # (B, hidden_dim) -> (B, 1, hidden_dim) -> (B, T, hidden_dim)
        h0 = torch.tanh(self.fc_latent_to_hidden(z)).unsqueeze(1).repeat(1, seq_len, 1)

        decoded_seq, _ = self.decoder_lstm(h0)   # (B, T, hidden_dim)
        return self.output_layer(decoded_seq)    # (B, T, input_dim)

    def forward(self, x):
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar
