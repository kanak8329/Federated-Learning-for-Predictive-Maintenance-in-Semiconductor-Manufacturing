# fl_strategies/fedavg.py
"""
FedAvg — Federated Averaging (McMahan et al., 2017)

The simplest FL aggregation: element-wise mean of all client model weights.
"""

import torch
from fl_strategies.base import FederatedStrategy
from utils.metrics import train_one_epoch


class FedAvgStrategy(FederatedStrategy):

    def aggregate(self, local_state_dicts: list, global_state_dict: dict) -> dict:
        """Simple element-wise mean of all client state dicts."""
        avg = {}
        n = len(local_state_dicts)
        for key in local_state_dicts[0]:
            avg[key] = sum(sd[key] for sd in local_state_dicts) / n
        return avg

    def local_train(self, local_model, global_model, X_train, y_train,
                    local_epochs, lr, batch_size):
        """Standard local SGD training."""
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        for _ in range(local_epochs):
            train_one_epoch(
                local_model, optimizer, loss_fn,
                X_train, y_train,
                device=self.device, batch_size=batch_size,
            )

    def get_name(self) -> str:
        return "FedAvg"
