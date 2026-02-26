# models/transformer_model.py
"""
Temporal Transformer Classifier for time-series binary classification.

Architecture:
    Input (batch, seq_len, n_features)
      → Linear projection to d_model
      → Positional Encoding (sinusoidal)
      → Transformer Encoder (multi-head self-attention × N layers)
      → Mean pooling over sequence
      → FC classifier → Sigmoid

Advantages over LSTM:
  - Captures long-range dependencies via self-attention
  - Parallelizable (no sequential bottleneck)
  - Interpretable attention weights
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-based time-series classifier.

    Args:
        n_features:   Number of input sensor features
        hidden_size:  d_model dimension (default 128)
        n_layers:     Number of Transformer encoder layers (default 2)
        n_heads:      Number of attention heads (default 4)
        ff_dim:       Feed-forward dimension (default 256)
        dropout:      Dropout rate (default 0.2)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.d_model = hidden_size

        # Project raw features to d_model
        self.input_proj = nn.Linear(n_features, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            probabilities: (batch,)
        """
        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Mean pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        return self.classifier(x).squeeze(-1)
