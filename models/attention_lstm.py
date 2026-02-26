# models/attention_lstm.py
"""
Attention-Enhanced LSTM Classifier.

Architecture:
    Input (batch, seq_len, n_features)
      → LSTM (multi-layer)
      → Attention layer over all LSTM hidden states
      → Weighted sum → context vector
      → FC classifier → Sigmoid

Key advantage: The attention weights reveal WHICH TIMESTEPS are most
important for the prediction — providing interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Additive attention (Bahdanau-style) over LSTM hidden states.

    Learns: a_t = softmax(v^T * tanh(W * h_t + b))
    Output: context = Σ a_t * h_t
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs):
        """
        Args:
            lstm_outputs: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Score each timestep
        energy = torch.tanh(self.W(lstm_outputs))   # (batch, seq_len, hidden)
        scores = self.v(energy).squeeze(-1)          # (batch, seq_len)

        # Normalize via softmax
        attention_weights = F.softmax(scores, dim=1) # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),   # (batch, 1, seq_len)
            lstm_outputs,                      # (batch, seq_len, hidden)
        ).squeeze(1)                           # (batch, hidden)

        return context, attention_weights


class AttentionLSTM(nn.Module):
    """
    LSTM + Attention classifier.

    Args:
        n_features:   Number of input sensor features
        hidden_size:  LSTM hidden dimension (default 128)
        n_layers:     Number of LSTM layers (default 2)
        dropout:      Dropout rate (default 0.2)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.attention = AttentionLayer(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Store attention weights for interpretability
        self._last_attention_weights = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            probabilities: (batch,)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        context, attn_weights = self.attention(lstm_out)
        self._last_attention_weights = attn_weights.detach()

        return self.classifier(context).squeeze(-1)

    def get_attention_weights(self):
        """Return attention weights from the last forward pass."""
        return self._last_attention_weights
