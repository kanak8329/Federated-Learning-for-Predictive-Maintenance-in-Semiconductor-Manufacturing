# fl_strategies/base.py
"""
Abstract base class for all federated learning strategies.
"""

from abc import ABC, abstractmethod


class FederatedStrategy(ABC):
    """
    Interface that all FL strategies must implement.

    Methods:
        aggregate():    Combine local model updates into a global model
        local_train():  Train a local model on client data
        get_name():     Human-readable strategy name
    """

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.device = device

    @abstractmethod
    def aggregate(self, local_state_dicts: list, global_state_dict: dict) -> dict:
        """
        Aggregate local model weights into a new global state dict.

        Args:
            local_state_dicts: List of state dicts from participating clients
            global_state_dict: Current global model state dict

        Returns:
            Aggregated state dict
        """
        pass

    @abstractmethod
    def local_train(self, local_model, global_model, X_train, y_train,
                    local_epochs, lr, batch_size):
        """
        Perform local training on a single client.

        Args:
            local_model:  Client's local model (already loaded with global weights)
            global_model: Reference to global model (for proximal term, etc.)
            X_train:      Client training features (numpy)
            y_train:      Client training labels (numpy)
            local_epochs: Number of local SGD epochs
            lr:           Learning rate
            batch_size:   Mini-batch size
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable name of this strategy."""
        pass
