# run_experiment.py
"""
Config-driven experiment runner.

Usage:
    python run_experiment.py                           # uses configs/default.yaml
    python run_experiment.py --config configs/dp.yaml  # custom config
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import get_model
from utils.metrics import get_device, train_one_epoch, evaluate
from utils.data_loader import (
    load_client_data, load_all_clients_train, get_n_features,
)
from utils.logger import ExperimentLogger


# ── Config Loader ──────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Centralized Training ──────────────────────────────────

def run_centralized(cfg, device, logger):
    print("\n" + "=" * 60)
    print("  📊  CENTRALIZED TRAINING")
    print("=" * 60)

    client_dir = cfg["data"]["client_dir"]
    X_train, y_train = load_all_clients_train(client_dir)
    test_data = load_client_data(client_dir, 1)
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    n_features = X_train.shape[2]
    model = get_model(
        cfg["model"]["type"], n_features,
        hidden_size=cfg["model"]["hidden_size"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    loss_fn = torch.nn.BCELoss()

    history_loss, history_acc = [], []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss = train_one_epoch(
            model, optimizer, loss_fn, X_train, y_train,
            device=device, batch_size=cfg["training"]["batch_size"],
        )
        metrics = evaluate(model, X_test, y_test, device=device)
        history_loss.append(loss)
        history_acc.append(metrics["accuracy"])

        logger.log_round(epoch, {"loss": loss, **metrics})
        print(f"  Epoch {epoch:>3d} │ Loss={loss:.4f} │ Acc={metrics['accuracy']:.4f} │ F1={metrics['f1']:.4f}")

    # Save model
    models_dir = cfg["output"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "centralized_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save plots
    plots_dir = cfg["output"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(history_loss, "o-", color="#e74c3c")
    plt.title("Centralized — Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "centralized_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history_acc, "o-", color="#2ecc71")
    plt.title("Centralized — Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "centralized_accuracy.png"), dpi=150)
    plt.close()

    final = evaluate(model, X_test, y_test, device=device)
    print(f"\n  ✅ Centralized done │ Final Acc={final['accuracy']:.4f} F1={final['f1']:.4f}")
    return model, final


# ── Federated Training ────────────────────────────────────

def average_weights(state_dicts):
    """Simple FedAvg: element-wise mean of state dicts."""
    avg = {}
    for key in state_dicts[0]:
        avg[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return avg


def run_federated(cfg, device, logger):
    print("\n" + "=" * 60)
    print("  🌐  FEDERATED TRAINING")
    print("=" * 60)

    client_dir = cfg["data"]["client_dir"]
    n_features = get_n_features(client_dir)
    n_clients = cfg["data"]["n_clients"]
    strategy_name = cfg["federated"]["strategy"]

    # Try to use advanced strategy if available
    strategy = None
    if strategy_name != "fedavg":
        try:
            from fl_strategies import get_strategy
            strategy = get_strategy(strategy_name, cfg, device)
            print(f"  Using strategy: {strategy.get_name()}")
        except (ImportError, Exception) as e:
            print(f"  ⚠  Strategy '{strategy_name}' not available, falling back to FedAvg")
            strategy_name = "fedavg"

    global_model = get_model(
        cfg["model"]["type"], n_features,
        hidden_size=cfg["model"]["hidden_size"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    loss_fn = torch.nn.BCELoss()
    rounds = cfg["federated"]["rounds"]
    local_epochs = cfg["federated"]["local_epochs"]
    lr = cfg["training"]["learning_rate"]

    global_acc, global_f1 = [], []

    # Check for DP
    dp_enabled = cfg["federated"].get("dp", {}).get("enabled", False)
    privacy_accountant = None
    if dp_enabled:
        try:
            from privacy.dp_fedavg import DPFedAvg
            from privacy.privacy_accountant import PrivacyAccountant
            dp_config = cfg["federated"]["dp"]
            dp_handler = DPFedAvg(
                noise_multiplier=dp_config["noise_multiplier"],
                max_grad_norm=dp_config["max_grad_norm"],
            )
            privacy_accountant = PrivacyAccountant(
                noise_multiplier=dp_config["noise_multiplier"],
                sample_rate=1.0 / n_clients,
                target_delta=dp_config["target_delta"],
            )
            print(f"  🔒 Differential Privacy ENABLED (σ={dp_config['noise_multiplier']}, C={dp_config['max_grad_norm']})")
        except ImportError:
            print("  ⚠  DP modules not found, running without DP")
            dp_enabled = False

    for rnd in range(1, rounds + 1):
        print(f"\n  🔄 Round {rnd}/{rounds}")
        local_states = []

        for cid in range(1, n_clients + 1):
            data = load_client_data(client_dir, cid)

            local_model = get_model(
                cfg["model"]["type"], n_features,
                hidden_size=cfg["model"]["hidden_size"],
                n_layers=cfg["model"]["n_layers"],
                dropout=cfg["model"]["dropout"],
            ).to(device)
            local_model.load_state_dict(global_model.state_dict())

            if strategy and strategy_name == "fedprox":
                # FedProx: use strategy's local train
                strategy.local_train(
                    local_model, global_model,
                    data["X_train"], data["y_train"],
                    local_epochs, lr, cfg["training"]["batch_size"],
                )
            else:
                optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
                for _ in range(local_epochs):
                    train_one_epoch(
                        local_model, optimizer, loss_fn,
                        data["X_train"], data["y_train"],
                        device=device, batch_size=cfg["training"]["batch_size"],
                    )

            state = {k: v.cpu() for k, v in local_model.state_dict().items()}
            local_states.append(state)

        # Aggregation
        if dp_enabled:
            avg_state = dp_handler.aggregate(
                local_states, global_model.state_dict()
            )
            eps = privacy_accountant.step()
            logger.log_event("dp_round", {"round": rnd, "epsilon": eps})
        elif strategy and strategy_name == "fednova":
            avg_state = strategy.aggregate(local_states, global_model.state_dict())
        else:
            avg_state = average_weights(local_states)

        global_model.load_state_dict(avg_state)

        # Evaluate on client 1 test set
        test_data = load_client_data(client_dir, 1)
        metrics = evaluate(global_model, test_data["X_test"], test_data["y_test"], device=device)
        global_acc.append(metrics["accuracy"])
        global_f1.append(metrics["f1"])

        round_info = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        if dp_enabled and privacy_accountant:
            round_info["epsilon"] = privacy_accountant.get_epsilon()
        logger.log_round(rnd, round_info)

        print(f"     Global │ Acc={metrics['accuracy']:.4f} │ F1={metrics['f1']:.4f}")

    # Save model
    models_dir = cfg["output"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    torch.save(
        global_model.state_dict(),
        os.path.join(models_dir, f"federated_{strategy_name}_model.pt"),
    )

    # Plot
    plots_dir = cfg["output"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, rounds + 1), global_acc, "o-", color="#3498db")
    ax1.set_title(f"Federated ({strategy_name}) — Accuracy per Round")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, rounds + 1), global_f1, "s-", color="#9b59b6")
    ax2.set_title(f"Federated ({strategy_name}) — F1 per Round")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("F1 Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"federated_{strategy_name}_convergence.png"), dpi=150)
    plt.close()

    final = evaluate(global_model, test_data["X_test"], test_data["y_test"], device=device)
    print(f"\n  ✅ Federated ({strategy_name}) done │ Final Acc={final['accuracy']:.4f} F1={final['f1']:.4f}")
    return global_model, final


# ── Comparison Plot ───────────────────────────────────────

def plot_comparison(cent_metrics, fed_metrics, strategy_name, plots_dir):
    """Generate comparison bar chart between centralized and federated."""
    labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    cent_vals = [cent_metrics[k] for k in ["accuracy", "f1", "precision", "recall"]]
    fed_vals = [fed_metrics[k] for k in ["accuracy", "f1", "precision", "recall"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, cent_vals, width, label="Centralized", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, fed_vals, width, label=f"Federated ({strategy_name})", color="#3498db", alpha=0.85)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Centralized vs Federated — Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "centralized_vs_federated.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  📊 Comparison plot → {out_path}")


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FL Semiconductor Experiment Runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Fix seeds
    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(cfg["experiment"]["device"])
    print(f"\n🔧 Device: {device}")
    print(f"📋 Config: {args.config}")
    print(f"🧪 Experiment: {cfg['experiment']['name']}")

    # Logger
    logger = ExperimentLogger(
        log_dir=cfg["output"]["logs_dir"],
        experiment_name=cfg["experiment"]["name"],
    )
    logger.log_config(cfg)
    logger.log_event("experiment_started")

    # Ensure output dirs
    for d in [cfg["output"]["results_dir"], cfg["output"]["plots_dir"],
              cfg["output"]["models_dir"], cfg["output"]["logs_dir"]]:
        os.makedirs(d, exist_ok=True)

    # ── Run centralized ──
    logger.log_event("centralized_training_started")
    cent_model, cent_metrics = run_centralized(cfg, device, logger)
    logger.log_event("centralized_training_finished", cent_metrics)

    # ── Run federated ──
    logger.log_event("federated_training_started", {"strategy": cfg["federated"]["strategy"]})
    fed_model, fed_metrics = run_federated(cfg, device, logger)
    logger.log_event("federated_training_finished", fed_metrics)

    # ── Comparison ──
    plot_comparison(cent_metrics, fed_metrics, cfg["federated"]["strategy"], cfg["output"]["plots_dir"])

    # ── Final log ──
    logger.log_final({
        "centralized": cent_metrics,
        "federated": fed_metrics,
    })
    logger.save()

    print("\n" + "=" * 60)
    print("  🏁  EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
