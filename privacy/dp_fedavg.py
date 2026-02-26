# privacy/dp_fedavg.py
"""
DP-FedAvg — Differentially Private Federated Averaging

Implements the core DP mechanisms:
  1. Per-client update clipping (bound sensitivity)
  2. Gaussian noise injection (calibrated to privacy budget)

Reference: McMahan et al., "Learning Differentially Private Recurrent
           Language Models" (ICLR 2018)
"""

import torch
import numpy as np


class DPFedAvg:
    """
    Differential Privacy handler for FedAvg aggregation.

    Args:
        noise_multiplier:  σ — ratio of noise std to clipping norm
        max_grad_norm:     C — maximum L2 norm for per-client updates
    """

    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def clip_update(self, local_state: dict, global_state: dict) -> dict:
        """
        Clip the L2 norm of the client update (Δw = w_local - w_global).

        If ‖Δw‖₂ > C, scale down to: Δw * (C / ‖Δw‖₂)
        """
        # Compute delta
        delta = {}
        for key in local_state:
            delta[key] = local_state[key] - global_state[key].cpu()

        # Compute global L2 norm of the update
        total_norm = 0.0
        for key in delta:
            total_norm += delta[key].float().norm(2).item() ** 2
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-10))
        if clip_factor < 1.0:
            for key in delta:
                delta[key] = delta[key] * clip_factor

        return delta

    def add_noise(self, aggregated_delta: dict, n_clients: int) -> dict:
        """
        Add calibrated Gaussian noise to the aggregated update.

        noise ~ N(0, σ² * C² / n²)  for each parameter
        """
        noise_std = (self.noise_multiplier * self.max_grad_norm) / n_clients

        noised_delta = {}
        for key in aggregated_delta:
            noise = torch.randn_like(aggregated_delta[key].float()) * noise_std
            noised_delta[key] = aggregated_delta[key] + noise

        return noised_delta

    def aggregate(self, local_state_dicts: list, global_state_dict: dict) -> dict:
        """
        Full DP-FedAvg aggregation pipeline:
          1. Compute per-client updates
          2. Clip each update to max_grad_norm
          3. Average clipped updates
          4. Add Gaussian noise
          5. Apply to global model
        """
        n = len(local_state_dicts)

        # Step 1 & 2: Clip each client's update
        clipped_deltas = []
        for sd in local_state_dicts:
            delta = self.clip_update(sd, global_state_dict)
            clipped_deltas.append(delta)

        # Step 3: Average
        avg_delta = {}
        for key in clipped_deltas[0]:
            avg_delta[key] = sum(d[key] for d in clipped_deltas) / n

        # Step 4: Add noise
        noised_delta = self.add_noise(avg_delta, n)

        # Step 5: Apply to global weights
        new_state = {}
        for key in global_state_dict:
            new_state[key] = global_state_dict[key].cpu() + noised_delta[key]

        return new_state
