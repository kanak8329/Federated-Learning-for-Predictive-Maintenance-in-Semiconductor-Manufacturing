# utils.py
"""
Backward-compatible import — redirects to utils/metrics.py

Legacy scripts import `device`, `train_one_epoch`, `evaluate` from here.
New code should use `from utils.metrics import ...`.
"""

from utils.metrics import get_device, train_one_epoch, evaluate, batch_iter

device = get_device("auto")

__all__ = ["device", "train_one_epoch", "evaluate", "batch_iter"]
