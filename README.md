![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-green)
![Differential Privacy](https://img.shields.io/badge/Differential-Privacy-purple)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-orange?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)

# 🔬 Federated Learning for Semiconductor Predictive Maintenance

A **research-grade**, privacy-preserving federated learning framework for predictive maintenance in semiconductor manufacturing. Uses advanced LSTM, Attention-LSTM, and Transformer models on the [SECOM dataset](https://archive.ics.uci.edu/ml/datasets/SECOM) — with differential privacy, multiple FL strategies, anomaly detection, and an interactive Streamlit dashboard.

---

## 🚀 Key Results

| Model / Strategy | Accuracy | F1-Score | Privacy | Data Sharing |
|-----------------|----------|----------|---------|--------------|
| Centralized     | ~0.76    | ~0.66    | ❌ None  | ❌ Raw data shared |
| FedAvg          | ~0.73    | ~0.63    | ❌ None  | ✅ No sharing |
| FedProx         | ~0.74    | ~0.64    | ❌ None  | ✅ No sharing |
| FedNova         | ~0.73    | ~0.63    | ❌ None  | ✅ No sharing |
| DP-FedAvg       | ~0.70    | ~0.58    | ✅ (ε,δ)-DP | ✅ No sharing |

> **Key Insight**: Federated models achieve **~96% of centralized performance** while keeping all sensor data local. With DP, we add formal privacy guarantees at a modest accuracy cost.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    🔬 FL Semiconductor Framework             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │  Fab A   │  │  Fab B   │  │  Fab C   │  ← Clients      │
│  │ (Client) │  │ (Client) │  │ (Client) │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│       │              │              │                       │
│       └──────┬───────┴──────┬───────┘                      │
│              │              │                               │
│         ┌────▼────┐   ┌────▼─────┐                         │
│         │ Gradient │   │DP Noise  │ ← Privacy Layer        │
│         │ Clipping │   │Injection │                         │
│         └────┬────┘   └────┬─────┘                         │
│              │              │                               │
│         ┌────▼──────────────▼────┐                          │
│         │  FL Aggregation Server │                          │
│         │  FedAvg / FedProx /    │ ← Strategy Layer        │
│         │  FedNova              │                           │
│         └───────────┬───────────┘                           │
│                     │                                       │
│              ┌──────▼───────┐                               │
│              │ Global Model │                               │
│              │ LSTM / Attn  │ ← Model Layer                │
│              │ Transformer  │                               │
│              └──────┬───────┘                               │
│                     │                                       │
│         ┌───────────▼───────────┐                           │
│         │  📊 Dashboard &       │                           │
│         │  Visualization        │ ← Monitoring Layer       │
│         └───────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
federated-learning-semiconductor-predictive-maintenance/
│
├── configs/
│   └── default.yaml              # Central experiment configuration
│
├── models/
│   ├── __init__.py               # Model registry & factory
│   ├── lstm_model.py             # Standard LSTM classifier
│   ├── attention_lstm.py         # LSTM + Bahdanau attention
│   └── transformer_model.py      # Temporal Transformer encoder
│
├── fl_strategies/
│   ├── __init__.py               # Strategy registry
│   ├── base.py                   # Abstract FederatedStrategy
│   ├── fedavg.py                 # FedAvg (McMahan et al., 2017)
│   ├── fedprox.py                # FedProx with proximal term
│   └── fednova.py                # FedNova normalized averaging
│
├── privacy/
│   ├── __init__.py
│   ├── dp_fedavg.py              # DP-FedAvg (clipping + noise)
│   └── privacy_accountant.py     # RDP-based (ε,δ) tracker
│
├── anomaly/
│   ├── __init__.py
│   ├── autoencoder.py            # LSTM autoencoder
│   └── anomaly_detector.py       # Federated anomaly pipeline
│
├── data_utils/
│   ├── __init__.py
│   └── noniid_partition.py       # Dirichlet/label/quantity skew
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── metrics.py                # Training & evaluation utilities
│   └── logger.py                 # Structured JSON experiment logger
│
├── visualization/
│   ├── __init__.py
│   └── advanced_plots.py         # ROC, t-SNE, confusion matrix, etc.
│
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
│
├── data/                         # SECOM dataset (not in repo)
├── results/                      # Models, plots, logs
│
├── run_experiment.py             # Config-driven experiment runner
├── secom_preprocess.py           # SECOM data preprocessing
├── make_windows.py               # Sliding window creation
├── centralized_train.py          # Centralized baseline (legacy)
├── federated_train.py            # Federated training (legacy)
├── compare_models.py             # Model comparison (legacy)
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/federated-learning-semiconductor-predictive-maintenance.git
cd federated-learning-semiconductor-predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Download the [SECOM dataset](https://archive.ics.uci.edu/ml/datasets/SECOM) and place `secom.data` and `secom_labels.data` in the `data/` directory.

```bash
# Step 1: Preprocess
python secom_preprocess.py

# Step 2: Create windows & client splits
python make_windows.py
```

---

## 🚀 Usage

### Run a Full Experiment

```bash
# Default configuration (FedAvg + LSTM)
python run_experiment.py

# Custom configuration
python run_experiment.py --config configs/default.yaml
```

### Configuration Options

Edit `configs/default.yaml` to customize:

```yaml
model:
  type: "lstm"          # lstm | attention_lstm | transformer

federated:
  strategy: "fedavg"    # fedavg | fedprox | fednova
  rounds: 5
  dp:
    enabled: false      # Enable differential privacy
    noise_multiplier: 1.0
    max_grad_norm: 1.0
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🧠 Phase-by-Phase Capabilities

### Phase 1 — Foundation
- Clean config-driven architecture
- Model registry with factory pattern
- Structured JSON experiment logging

### Phase 2 — Advanced FL Strategies
- **FedProx**: Proximal regularization for non-IID resilience
- **FedNova**: Normalized averaging for heterogeneous updates
- **Non-IID Simulation**: Dirichlet, label-skew, quantity-skew partitioning

### Phase 3 — Privacy
- **DP-FedAvg**: Per-client gradient clipping + Gaussian noise
- **RDP Accountant**: Formal (ε,δ)-differential privacy tracking

### Phase 4 — Advanced Models
- **Attention-LSTM**: Interpretable attention over timesteps
- **Transformer**: Multi-head self-attention for long-range dependencies
- **Anomaly Detection**: Federated LSTM autoencoder

### Phase 5 — Dashboard
- **Streamlit Dashboard**: Interactive training monitor, model comparison, privacy tracker
- **Advanced Plots**: ROC curves, confusion matrices, t-SNE, radar charts

---

## 📊 Key Visualizations

The framework generates:
- Training loss & accuracy curves
- Centralized vs Federated comparison bar charts
- FL strategy convergence comparisons
- Privacy budget (ε) accumulation plots
- Client data distribution visualizations
- Confusion matrix heatmaps
- ROC curves with AUC comparison
- t-SNE embedding plots
- Per-client radar charts

---

## 📚 References

| Paper | Topic |
|-------|-------|
| [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) | FedAvg |
| [Li et al., 2020](https://arxiv.org/abs/1812.06127) | FedProx |
| [Wang et al., 2020](https://arxiv.org/abs/2007.07481) | FedNova |
| [McMahan et al., 2018](https://arxiv.org/abs/1710.06963) | DP-FL |
| [Mironov, 2017](https://arxiv.org/abs/1702.07476) | Rényi DP |

---

## 🔁 Reproducibility

All experiments are fully reproducible:
- Random seeds fixed via config (`seed: 42`)
- Identical train/test splits across runs
- Full experiment configs logged to JSON
- Environment: Python 3.9+, PyTorch 2.0+

---

## 📄 License

This project is for research and educational purposes.

---

## 📖 Project Guide

For a comprehensive, easy-to-understand breakdown of the entire project — including architecture explanations, interview Q&As, and talking points — see **[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.
