# model.py
"""
Backward-compatible import — redirects to models/lstm_model.py

Legacy scripts (centralized_train.py, federated_train.py, compare_models.py)
import from here. New code should use `from models import get_model`.
"""

from models.lstm_model import LSTMClassifier

__all__ = ["LSTMClassifier"]
