# fl_strategies/fedprox.py
"""
FedProx — Federated Proximal (Li et al., 2020)

Adds a proximal regularization term to the local objective:

    L_local = L_task + (μ / 2) * ‖w - w_global‖²

This prevents local models from drifting too far from the global model,
which is critical when client data is non-IID.
"""

import torch
import numpy as np
from fl_strategies.base import FederatedStrategy
from utils.metrics import batch_iter


class FedProxStrategy(FederatedStrategy):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.mu = cfg["federated"].get("mu", 0.01)

    def aggregate(self, local_state_dicts: list, global_state_dict: dict) -> dict:
        """Same as FedAvg — element-wise mean."""
        avg = {}
        n = len(local_state_dicts)
        for key in local_state_dicts[0]:
            avg[key] = sum(sd[key] for sd in local_state_dicts) / n
        return avg

    def local_train(self, local_model, global_model, X_train, y_train,
                    local_epochs, lr, batch_size):
        """
        Local training with proximal term.

        Loss = BCE(pred, target) + (μ/2) * Σ ‖w_local - w_global‖²
        """
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        # Freeze a copy of global weights for proximal term
        global_params = {
            name: param.clone().detach()
            for name, param in global_model.named_parameters()
        }

        for epoch in range(local_epochs):
            local_model.train()
            for xb, yb in batch_iter(X_train, y_train, batch_size):
                xb_t = torch.tensor(xb, dtype=torch.float32, device=self.device)
                yb_t = torch.tensor(yb, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                pred = local_model(xb_t)
                task_loss = loss_fn(pred, yb_t)

                # Proximal term: (μ/2) * ‖w_local - w_global‖²
                prox_term = 0.0
                for name, param in local_model.named_parameters():
                    if name in global_params:
                        prox_term += ((param - global_params[name]) ** 2).sum()
                prox_term = (self.mu / 2.0) * prox_term

                loss = task_loss + prox_term
                loss.backward()
                optimizer.step()

    def get_name(self) -> str:
        return f"FedProx (mu={self.mu})"
