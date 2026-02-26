# anomaly/anomaly_detector.py
"""
Federated Anomaly Detection Pipeline.

Trains an LSTM autoencoder per client using federated learning,
then uses reconstruction error to flag anomalous samples.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anomaly.autoencoder import LSTMAutoencoder
from utils.metrics import get_device, batch_iter
from utils.data_loader import load_client_data, get_n_features


class FederatedAnomalyDetector:
    """
    Orchestrates federated training of LSTM autoencoders for anomaly detection.

    Args:
        cfg:    Full experiment config
        device: torch.device
    """

    def __init__(self, cfg: dict, device=None):
        self.cfg = cfg
        self.device = device or get_device(cfg["experiment"]["device"])

        anomaly_cfg = cfg.get("anomaly", {})
        self.latent_dim = anomaly_cfg.get("latent_dim", 32)
        self.threshold_percentile = anomaly_cfg.get("threshold_percentile", 95)
        self.epochs = anomaly_cfg.get("epochs", 20)
        self.threshold = None

    def train(self):
        """Full federated anomaly detection pipeline."""
        client_dir = self.cfg["data"]["client_dir"]
        n_features = get_n_features(client_dir)
        n_clients = self.cfg["data"]["n_clients"]
        lr = self.cfg["training"]["learning_rate"]
        batch_size = self.cfg["training"]["batch_size"]
        rounds = self.cfg["federated"]["rounds"]

        # Initialize global autoencoder
        global_ae = LSTMAutoencoder(
            n_features=n_features,
            latent_dim=self.latent_dim,
        ).to(self.device)

        print("\n" + "=" * 60)
        print("  🔍  FEDERATED ANOMALY DETECTION")
        print("=" * 60)

        for rnd in range(1, rounds + 1):
            print(f"\n  🔄 AE Round {rnd}/{rounds}")
            local_states = []

            for cid in range(1, n_clients + 1):
                data = load_client_data(client_dir, cid)
                X_train = data["X_train"]

                # Local autoencoder
                local_ae = LSTMAutoencoder(
                    n_features=n_features,
                    latent_dim=self.latent_dim,
                ).to(self.device)
                local_ae.load_state_dict(global_ae.state_dict())

                optimizer = torch.optim.Adam(local_ae.parameters(), lr=lr)
                loss_fn = torch.nn.MSELoss()

                local_ae.train()
                for ep in range(self.epochs):
                    total_loss = 0.0
                    count = 0
                    for xb, _ in batch_iter(X_train, np.zeros(len(X_train)), batch_size):
                        xb_t = torch.tensor(xb, dtype=torch.float32, device=self.device)
                        optimizer.zero_grad()
                        reconstructed, _ = local_ae(xb_t)
                        loss = loss_fn(reconstructed, xb_t)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(xb)
                        count += len(xb)

                local_states.append(
                    {k: v.cpu() for k, v in local_ae.state_dict().items()}
                )

            # FedAvg aggregation for autoencoder
            avg_state = {}
            for key in local_states[0]:
                avg_state[key] = sum(sd[key] for sd in local_states) / len(local_states)
            global_ae.load_state_dict(avg_state)

        self.model = global_ae
        print("  ✅ Autoencoder training complete")
        return global_ae

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each sample.

        Args:
            X: (N, seq_len, n_features)
        Returns:
            scores: (N,) — MSE reconstruction error per sample
        """
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            reconstructed, _ = self.model(xb)
            errors = LSTMAutoencoder.reconstruction_error(xb, reconstructed)
        return errors.cpu().numpy()

    def fit_threshold(self, X_normal: np.ndarray):
        """
        Set anomaly threshold based on reconstruction error percentile
        on normal (non-defective) data.
        """
        scores = self.compute_anomaly_scores(X_normal)
        self.threshold = float(np.percentile(scores, self.threshold_percentile))
        print(f"  🎯 Anomaly threshold (p{self.threshold_percentile}): {self.threshold:.6f}")
        return self.threshold

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict anomalies.

        Returns:
            dict with 'scores', 'predictions', 'threshold'
        """
        scores = self.compute_anomaly_scores(X)
        predictions = (scores > self.threshold).astype(int) if self.threshold else None

        return {
            "scores": scores,
            "predictions": predictions,
            "threshold": self.threshold,
        }

    def plot_scores(self, scores: np.ndarray, labels: np.ndarray = None,
                    save_path: str = None):
        """Plot reconstruction error distribution with optional true labels."""
        fig, ax = plt.subplots(figsize=(10, 5))

        if labels is not None:
            normal = scores[labels == 0]
            anomalous = scores[labels == 1]
            ax.hist(normal, bins=50, alpha=0.7, color="#2ecc71", label="Normal (Pass)")
            ax.hist(anomalous, bins=50, alpha=0.7, color="#e74c3c", label="Anomalous (Fail)")
        else:
            ax.hist(scores, bins=50, alpha=0.7, color="#3498db")

        if self.threshold:
            ax.axvline(self.threshold, color="#e67e22", linestyle="--", linewidth=2,
                       label=f"Threshold={self.threshold:.4f}")

        ax.set_title("Reconstruction Error Distribution")
        ax.set_xlabel("MSE Reconstruction Error")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  📊 Anomaly scores plot → {save_path}")
        plt.close()
