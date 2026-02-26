# privacy/privacy_accountant.py
"""
Privacy Accountant — Tracks cumulative privacy budget using
Rényi Differential Privacy (RDP).

Simplified implementation that computes ε given σ (noise multiplier),
sampling rate q, number of steps T, and target δ.

Reference:
  Mironov, "Rényi Differential Privacy" (CSF 2017)
  Mironov et al., "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (2019)
"""

import numpy as np
from typing import List


class PrivacyAccountant:
    """
    Tracks cumulative (ε, δ)-differential privacy budget
    using the moments accountant (RDP-based).

    Args:
        noise_multiplier:  σ — Gaussian noise multiplier
        sample_rate:       q — fraction of clients sampled per round
        target_delta:      δ — target failure probability
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        sample_rate: float = 1.0,
        target_delta: float = 1e-5,
    ):
        self.sigma = noise_multiplier
        self.q = sample_rate
        self.delta = target_delta
        self.steps = 0
        self._rdp_orders = list(range(2, 129))  # RDP orders α ∈ [2, 128]
        self._rdp_cumulative = np.zeros(len(self._rdp_orders))

    def _compute_rdp_single_step(self, alpha: int) -> float:
        """
        Compute RDP of a single step of the Sampled Gaussian Mechanism.

        For the fully-participated case (q=1):
            RDP_α = α / (2σ²)

        For the subsampled case:
            RDP_α ≤ (1/(α-1)) * log(1 + C(α,2)*q²*exp((2-1)/σ²) + ...)
            (simplified upper bound)
        """
        if self.q >= 1.0:
            # Full participation
            return alpha / (2.0 * self.sigma ** 2)
        else:
            # Subsampled Gaussian mechanism (tight bound approximation)
            # Using the simple bound: RDP_α ≤ log(1 + q²·(α choose 2)·exp((α-1)/σ²)) / (α-1)
            log_term = np.log(1 + self.q ** 2 * alpha * (alpha - 1) / (2 * self.sigma ** 2))
            return log_term / (alpha - 1)

    def step(self) -> float:
        """
        Account for one additional FL round. Returns current ε.
        """
        self.steps += 1

        for i, alpha in enumerate(self._rdp_orders):
            self._rdp_cumulative[i] += self._compute_rdp_single_step(alpha)

        return self.get_epsilon()

    def get_epsilon(self) -> float:
        """
        Convert accumulated RDP guarantees to (ε, δ)-DP.

        ε = min over α of: RDP_α - log(δ)/(α-1)
        """
        eps_candidates = []
        for i, alpha in enumerate(self._rdp_orders):
            eps = self._rdp_cumulative[i] - np.log(self.delta) / (alpha - 1)
            eps_candidates.append(eps)

        return float(min(eps_candidates))

    def get_summary(self) -> dict:
        """Return summary of current privacy state."""
        return {
            "steps": self.steps,
            "epsilon": round(self.get_epsilon(), 4),
            "delta": self.delta,
            "noise_multiplier": self.sigma,
            "sample_rate": self.q,
        }

    def __repr__(self):
        eps = self.get_epsilon() if self.steps > 0 else "N/A"
        return f"PrivacyAccountant(steps={self.steps}, ε={eps}, δ={self.delta})"
