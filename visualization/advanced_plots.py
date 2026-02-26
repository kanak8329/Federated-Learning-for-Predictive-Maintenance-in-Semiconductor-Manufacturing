# visualization/advanced_plots.py
"""
Advanced Visualization Suite for Federated Learning experiments.

Provides:
  - Confusion matrix heatmaps
  - ROC curves with AUC comparison
  - t-SNE embedding visualization
  - Client-level radar charts
  - Privacy-utility tradeoff curves
  - Convergence comparison across FL strategies
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          save_path=None, labels=None):
    """Plot a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = ["Pass (0)", "Fail (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_comparison(results: dict, save_path=None):
    """
    Plot ROC curves for multiple models on the same figure.

    Args:
        results: dict of model_name → {'y_true': array, 'y_prob': array}
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_tsne_embeddings(embeddings, labels, title="t-SNE Visualization",
                         save_path=None):
    """
    Plot t-SNE 2D embeddings colored by label.

    Args:
        embeddings: (N, 2) array from sklearn t-SNE
        labels:     (N,) array of class labels
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {0: "#2ecc71", 1: "#e74c3c"}
    names = {0: "Pass", 1: "Fail"}

    for label in [0, 1]:
        mask = labels == label
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[label], label=names[label],
            alpha=0.6, s=20, edgecolors="none",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_client_radar(client_metrics: dict, save_path=None):
    """
    Radar chart comparing per-client metrics.

    Args:
        client_metrics: dict of client_id → {'accuracy', 'f1', 'precision', 'recall'}
    """
    categories = ["Accuracy", "F1", "Precision", "Recall"]
    n_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (cid, metrics) in enumerate(sorted(client_metrics.items())):
        values = [metrics[k] for k in ["accuracy", "f1", "precision", "recall"]]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=f"Client {cid}", color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Client Performance", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_privacy_utility_tradeoff(epsilons, accuracies, f1_scores=None,
                                   save_path=None):
    """
    Plot the privacy-utility tradeoff: accuracy vs epsilon.

    Args:
        epsilons:    list of ε values
        accuracies:  list of corresponding accuracies
        f1_scores:   optional list of F1 scores
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epsilons, accuracies, "o-", color="#3498db", linewidth=2,
            markersize=8, label="Accuracy")

    if f1_scores:
        ax.plot(epsilons, f1_scores, "s-", color="#e74c3c", linewidth=2,
                markersize=8, label="F1 Score")

    ax.set_xlabel("Privacy Budget ε (lower = more private)", fontsize=12)
    ax.set_ylabel("Performance", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate regions
    ax.axvspan(0, 1, alpha=0.05, color="green", label="_Strong Privacy")
    ax.axvspan(1, 10, alpha=0.05, color="yellow", label="_Moderate Privacy")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_convergence_comparison(strategy_results: dict, save_path=None):
    """
    Compare convergence of different FL strategies.

    Args:
        strategy_results: dict of strategy_name → list of per-round accuracies
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    markers = ["o", "s", "^", "D"]

    for i, (name, accs) in enumerate(strategy_results.items()):
        rounds = range(1, len(accs) + 1)
        ax.plot(rounds, accs, f"{markers[i % len(markers)]}-",
                color=colors[i % len(colors)],
                linewidth=2, markersize=8, label=name)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Model Accuracy", fontsize=12)
    ax.set_title("FL Strategy Convergence Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()
