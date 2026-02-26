# anomaly/autoencoder.py
"""
LSTM Autoencoder for unsupervised anomaly detection.

Architecture:
    Encoder:  Input (batch, seq_len, n_features) → LSTM → latent (batch, latent_dim)
    Decoder:  latent → LSTM → reconstructed (batch, seq_len, n_features)

Anomaly score = reconstruction error (MSE per sample).
High reconstruction error → likely anomalous (defective wafer).
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.

    Args:
        n_features:   Number of input sensor features
        latent_dim:   Dimension of latent representation (default 32)
        hidden_size:  LSTM hidden dimension (default 64)
        n_layers:     Number of LSTM layers (default 1)
        dropout:      Dropout rate (default 0.1)
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 32,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # ── Encoder ──
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

        # ── Decoder ──
        self.decoder_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, n_features)

    def encode(self, x):
        """
        Encode input sequence to latent representation.

        Args:
            x: (batch, seq_len, n_features)
        Returns:
            latent: (batch, latent_dim)
        """
        _, (hn, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(hn[-1])  # use last layer's hidden state
        return latent

    def decode(self, latent, seq_len: int):
        """
        Decode latent representation back to sequence.

        Args:
            latent:  (batch, latent_dim)
            seq_len: length of output sequence
        Returns:
            reconstructed: (batch, seq_len, n_features)
        """
        # Expand latent to sequence
        hidden = self.decoder_fc(latent)                    # (batch, hidden)
        hidden_seq = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden)

        decoded, _ = self.decoder_lstm(hidden_seq)          # (batch, seq_len, hidden)
        reconstructed = self.output_proj(decoded)           # (batch, seq_len, n_features)
        return reconstructed

    def forward(self, x):
        """
        Full forward pass: encode → decode.

        Args:
            x: (batch, seq_len, n_features)
        Returns:
            reconstructed: (batch, seq_len, n_features)
            latent:        (batch, latent_dim)
        """
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed, latent

    @staticmethod
    def reconstruction_error(original, reconstructed):
        """
        Compute per-sample MSE reconstruction error.

        Args:
            original:      (batch, seq_len, n_features)
            reconstructed: (batch, seq_len, n_features)
        Returns:
            errors: (batch,) — MSE per sample
        """
        mse = (original - reconstructed) ** 2
        return mse.mean(dim=(1, 2))  # mean over seq_len and features
