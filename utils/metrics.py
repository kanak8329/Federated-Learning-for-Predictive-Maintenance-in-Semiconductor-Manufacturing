# utils/metrics.py
"""
Metric computation and training utilities.

Provides:
  - batch_iter()       : mini-batch iterator with shuffling
  - train_one_epoch()  : single training epoch
  - evaluate()         : full evaluation returning metrics dict
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def batch_iter(X, y, batch_size: int = 64, shuffle: bool = True):
    """Yield mini-batches from numpy arrays."""
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        b = idx[i : i + batch_size]
        yield X[b], y[b]


def train_one_epoch(model, optimizer, loss_fn, X, y, device=None, batch_size=64):
    """
    Train for one epoch. Returns average loss.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    total_loss = 0.0

    for xb, yb in batch_iter(X, y, batch_size):
        xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
        yb_t = torch.tensor(yb, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        out = model(xb_t)
        loss = loss_fn(out, yb_t)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)

    return total_loss / len(X)


def evaluate(model, X, y, device=None):
    """
    Evaluate model. Returns dict with accuracy, f1, precision, recall,
    roc_auc, and confusion_matrix.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        preds_prob = model(xb).cpu().numpy()
        preds_bin = (preds_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(y, preds_bin)),
        "f1":        float(f1_score(y, preds_bin, zero_division=0)),
        "precision": float(precision_score(y, preds_bin, zero_division=0)),
        "recall":    float(recall_score(y, preds_bin, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y, preds_prob))
    except Exception:
        metrics["roc_auc"] = None

    try:
        cm = confusion_matrix(y, preds_bin).tolist()
        metrics["confusion_matrix"] = cm
    except Exception:
        metrics["confusion_matrix"] = None

    return metrics
