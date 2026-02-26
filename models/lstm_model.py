# models/lstm_model.py
"""
Standard LSTM Classifier for time-series binary classification.

Architecture:
    Input (batch, seq_len, n_features)
      → LSTM (multi-layer, bidirectional-optional)
      → FC layers with ReLU + Dropout
      → Sigmoid → probability of defect
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,                    # absorb extra config keys gracefully
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

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
        # out: (batch, seq_len, hidden)  |  hn: (n_layers, batch, hidden)
        out, (hn, _) = self.lstm(x)
        h_last = hn[-1]                 # final layer hidden state
        return self.classifier(h_last).squeeze(-1)
