# data_utils/noniid_partition.py
"""
Non-IID Data Partitioning for Federated Learning.

Simulates realistic data heterogeneity across manufacturing fabs (clients):
  - Dirichlet: Controls label distribution skew via α parameter
  - Label Skew: Each client gets a dominant label class
  - Quantity Skew: Unequal dataset sizes across clients

Lower α → more heterogeneous → harder for FL algorithms.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def dirichlet_partition(X, y, n_clients=3, alpha=0.5, seed=42):
    """
    Partition data using Dirichlet distribution over labels.

    Args:
        X:          Features array (N, seq_len, n_features) or (N, ...)
        y:          Labels array (N,)
        n_clients:  Number of clients
        alpha:      Dirichlet concentration param (lower = more skewed)
        seed:       Random seed

    Returns:
        dict[int, dict]: client_id → {'X_train', 'y_train', 'X_test', 'y_test'}
    """
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(y))

    # Sort indices by class
    class_indices = {c: np.where(y == c)[0] for c in range(n_classes)}

    # Sample proportions from Dirichlet for each class
    client_indices = {i: [] for i in range(n_clients)}

    for c in range(n_classes):
        idx = class_indices[c]
        rng.shuffle(idx)

        # Dirichlet proportions
        proportions = rng.dirichlet([alpha] * n_clients)
        proportions = (proportions * len(idx)).astype(int)

        # Fix rounding
        proportions[-1] = len(idx) - proportions[:-1].sum()

        start = 0
        for i in range(n_clients):
            client_indices[i].extend(idx[start : start + proportions[i]])
            start += proportions[i]

    # Build client datasets
    client_data = {}
    for i in range(n_clients):
        cid = i + 1
        indices = np.array(client_indices[i])
        rng.shuffle(indices)

        X_c, y_c = X[indices], y[indices]
        X_train, X_test, y_train, y_test = train_test_split(
            X_c, y_c, test_size=0.2, random_state=seed,
        )

        client_data[cid] = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
        }

    return client_data


def label_skew_partition(X, y, n_clients=3, dominant_ratio=0.8, seed=42):
    """
    Each client gets a dominant label class (e.g., 80% of one class).

    For binary classification:
      - Client 1: 80% class-0, 20% class-1
      - Client 2: 80% class-1, 20% class-0
      - Client 3: 50/50 mix
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)

    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    split_0 = int(len(idx_0) * dominant_ratio)
    split_1 = int(len(idx_1) * dominant_ratio)

    assignments = {
        1: np.concatenate([idx_0[:split_0], idx_1[split_1:]]),       # mostly class 0
        2: np.concatenate([idx_1[:split_1], idx_0[split_0:]]),       # mostly class 1
    }

    # If more clients, distribute remaining evenly
    if n_clients > 2:
        remaining = np.concatenate([idx_0[split_0:], idx_1[split_1:]])
        rng.shuffle(remaining)
        extra_shards = np.array_split(remaining, n_clients - 2)
        for j, shard in enumerate(extra_shards, start=3):
            assignments[j] = shard

    client_data = {}
    for cid, indices in assignments.items():
        rng.shuffle(indices)
        X_c, y_c = X[indices], y[indices]
        X_train, X_test, y_train, y_test = train_test_split(
            X_c, y_c, test_size=0.2, random_state=seed,
        )
        client_data[cid] = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
        }

    return client_data


def quantity_skew_partition(X, y, n_clients=3, min_ratio=0.1, seed=42):
    """
    Unequal dataset sizes: client 1 gets the most data, client N gets the least.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)

    # Generate skewed sizes using exponential decay
    raw_sizes = np.array([1.0 / (i + 1) for i in range(n_clients)])
    raw_sizes = raw_sizes / raw_sizes.sum()
    sizes = (raw_sizes * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()  # fix rounding

    client_data = {}
    start = 0
    for i in range(n_clients):
        cid = i + 1
        end = start + sizes[i]
        idx = indices[start:end]

        X_c, y_c = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X_c, y_c, test_size=0.2, random_state=seed,
        )
        client_data[cid] = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
        }
        start = end

    return client_data


def plot_client_distributions(client_data: dict, save_path: str = None):
    """Visualize label distribution per client."""
    n = len(client_data)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = ["#e74c3c", "#2ecc71"]
    for ax, (cid, data) in zip(axes, sorted(client_data.items())):
        y = np.concatenate([data["y_train"], data["y_test"]])
        counts = [np.sum(y == 0), np.sum(y == 1)]
        ax.bar(["Pass (0)", "Fail (1)"], counts, color=colors, alpha=0.85)
        ax.set_title(f"Client {cid}\n(n={len(y)})")
        ax.set_ylabel("Count" if cid == 1 else "")
        for i, v in enumerate(counts):
            ax.text(i, v + 2, str(v), ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Client Label Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Client distributions plot → {save_path}")
    plt.close()


# ── Convenience function ──────────────────────────────────

def create_noniid_clients(X, y, cfg: dict):
    """
    Create non-IID client partitions based on config.

    Args:
        X: Windowed features
        y: Labels
        cfg: Full experiment config dict

    Returns:
        dict of client data
    """
    noniid_cfg = cfg["data"]["noniid"]
    strategy = noniid_cfg["strategy"]
    n_clients = cfg["data"]["n_clients"]
    alpha = noniid_cfg.get("alpha", 0.5)
    seed = cfg["experiment"]["seed"]

    strategies = {
        "dirichlet": lambda: dirichlet_partition(X, y, n_clients, alpha, seed),
        "label_skew": lambda: label_skew_partition(X, y, n_clients, seed=seed),
        "quantity_skew": lambda: quantity_skew_partition(X, y, n_clients, seed=seed),
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown non-IID strategy: {strategy}")

    print(f"  📦 Creating non-IID partitions: {strategy} (α={alpha})")
    return strategies[strategy]()
