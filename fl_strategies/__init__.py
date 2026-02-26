# fl_strategies/__init__.py
"""
Federated Learning Strategy Registry.

Supported strategies:
  - fedavg   : Federated Averaging
  - fedprox  : FedProx (proximal regularization)
  - fednova  : FedNova (normalized averaging)
"""

from fl_strategies.fedavg import FedAvgStrategy
from fl_strategies.fedprox import FedProxStrategy
from fl_strategies.fednova import FedNovaStrategy


_REGISTRY = {
    "fedavg": FedAvgStrategy,
    "fedprox": FedProxStrategy,
    "fednova": FedNovaStrategy,
}


def get_strategy(name: str, cfg: dict, device):
    """Instantiate a federated strategy by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](cfg, device)
