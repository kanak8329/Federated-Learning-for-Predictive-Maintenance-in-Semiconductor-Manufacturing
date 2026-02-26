# models/__init__.py
"""
Model Registry — Factory for all model architectures.

Supported models:
  - lstm            : Standard LSTM classifier (Phase 1)
  - attention_lstm  : LSTM + learned attention (Phase 4)
  - transformer     : Temporal Transformer encoder (Phase 4)
"""

from models.lstm_model import LSTMClassifier


def get_model(name: str, n_features: int, **kwargs):
    """
    Instantiate a model by name.

    Args:
        name:       One of 'lstm', 'attention_lstm', 'transformer'
        n_features: Number of input sensor features
        **kwargs:   Passed through to the model constructor
    """
    registry = {
        "lstm": LSTMClassifier,
    }

    # Lazy imports for Phase 4 models (avoids import errors before they exist)
    try:
        from models.attention_lstm import AttentionLSTM
        registry["attention_lstm"] = AttentionLSTM
    except ImportError:
        pass

    try:
        from models.transformer_model import TransformerClassifier
        registry["transformer"] = TransformerClassifier
    except ImportError:
        pass

    if name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    return registry[name](n_features=n_features, **kwargs)
