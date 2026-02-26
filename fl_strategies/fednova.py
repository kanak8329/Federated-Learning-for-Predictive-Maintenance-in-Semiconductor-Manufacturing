# fl_strategies/fednova.py
"""
FedNova — Federated Normalized Averaging (Wang et al., 2020)

Standard FedAvg can suffer from objective inconsistency when clients
perform different numbers of local SGD steps. FedNova fixes this by
normalizing each client's update by its number of local steps before
aggregating.

    Δw_global = Σ (τ_eff / τ_i) * Δw_i / N

where τ_i = number of local steps for client i,
      τ_eff = effective number of steps (e.g., mean of τ_i).
"""

import torch
from fl_strategies.base import FederatedStrategy
from utils.metrics import train_one_epoch


class FedNovaStrategy(FederatedStrategy):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.local_epochs = cfg["federated"]["local_epochs"]

    def aggregate(self, local_state_dicts: list, global_state_dict: dict) -> dict:
        """
        Normalized averaging: scale each client's update inversely
        by its number of local steps.
        """
        n = len(local_state_dicts)

        # Compute per-client deltas: Δw_i = w_local_i - w_global
        deltas = []
        for sd in local_state_dicts:
            delta = {}
            for key in sd:
                delta[key] = sd[key] - global_state_dict[key].cpu()
            deltas.append(delta)

        # For now, assume all clients do the same number of local steps
        # τ_i = local_epochs for all i  → normalized averaging = simple mean
        # In a heterogeneous setup, τ_i would vary per client.
        tau_i = [self.local_epochs] * n
        tau_eff = sum(tau_i) / n   # effective step count

        # Aggregate: w_global += τ_eff * Σ (Δw_i / τ_i) / N
        new_state = {}
        for key in global_state_dict:
            weighted_sum = sum(
                (tau_eff / tau_i[i]) * deltas[i][key]
                for i in range(n)
            ) / n
            new_state[key] = global_state_dict[key].cpu() + weighted_sum

        return new_state

    def local_train(self, local_model, global_model, X_train, y_train,
                    local_epochs, lr, batch_size):
        """Standard local SGD (same as FedAvg)."""
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        for _ in range(local_epochs):
            train_one_epoch(
                local_model, optimizer, loss_fn,
                X_train, y_train,
                device=self.device, batch_size=batch_size,
            )

    def get_name(self) -> str:
        return "FedNova"
