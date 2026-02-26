# 🔬 Federated Learning for Semiconductor Predictive Maintenance
## Complete Project Guide — Understand, Explain & Interview Ready

---

## 1. What Is This Project About? (The Big Picture)

**Problem:** Semiconductor manufacturing (chip-making) uses thousands of sensors on production lines. When a machine is about to fail, the sensor data shows patterns. We want to **predict failures before they happen** — this is called **Predictive Maintenance**.

**Challenge:** Different chip factories (called "fabs") have their own sensor data, but they **cannot share this data** with each other because it's proprietary and sensitive.

**Solution:** We use **Federated Learning (FL)** — a technique where each factory trains a machine learning model on its own data, and only shares the **model updates** (not the raw data) with a central server. The server combines these updates to create a single, better model. This way:
- ✅ No factory shares its raw sensor data
- ✅ Every factory benefits from a globally-trained model
- ✅ Privacy is preserved

---

## 2. Dataset: SECOM

| Detail | Value |
|--------|-------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SECOM) |
| Records | 1,567 wafer records |
| Features | 590 sensor measurements per record |
| Labels | Pass (0) or Fail (1) — binary classification |
| Class Imbalance | ~93% Pass, ~7% Fail (highly imbalanced) |

**What each row represents:** One semiconductor wafer passing through the production line, with 590 sensor readings recorded during manufacturing.

---

## 3. How The Data Pipeline Works

```
Step 1: Raw Data               Step 2: Clean & Scale         Step 3: Sliding Windows
┌───────────────┐             ┌──────────────────┐          ┌──────────────────┐
│ secom.data    │──Impute───> │ secom_clean.csv  │──Window─>│ (N, 10, 590)     │
│ secom_labels  │  NaN→median │ Scaled features  │   size=10│ Time-series      │
│ (raw, messy)  │  + StandardScaler              │          │ sequences        │
└───────────────┘             └──────────────────┘          └──────┬───────────┘
                                                                   │
                                                            Split into 3 clients
                                                            (simulating 3 fabs)
```

**Why Sliding Windows?** Instead of looking at one snapshot, the model looks at **10 consecutive readings** to understand temporal patterns. If sensor readings are gradually degrading over time, the window captures that trend.

---

## 4. Model Architectures (3 Options)

### 4a. Standard LSTM (Long Short-Term Memory)
- **What it does:** Processes the 10-step sequence one timestep at a time, remembering important past information
- **Architecture:** 2-layer LSTM → Dense(128→64) → ReLU → Dense(64→1) → Sigmoid
- **Strength:** Good at capturing sequential patterns
- **When to use:** Default choice, proven architecture for time-series

### 4b. Attention-LSTM
- **What it does:** Same LSTM backbone, but adds an **attention mechanism** that learns which timesteps matter most
- **Architecture:** LSTM → Attention Layer (learns weights per timestep) → Weighted Sum → Classifier
- **Strength:** Provides **interpretability** — you can see which timesteps contributed most to the prediction
- **Interview talking point:** *"If the model predicts a failure, we can trace back to which exact production stage showed warning signs"*

### 4c. Temporal Transformer
- **What it does:** Uses **self-attention** (from the Transformer architecture, same family as GPT) to look at all timesteps simultaneously
- **Architecture:** Linear Projection → Positional Encoding → Multi-Head Self-Attention (×2 layers) → Mean Pooling → Classifier
- **Strength:** Captures long-range dependencies, parallelizable (faster training)
- **Interview talking point:** *"We adapted the Transformer architecture from NLP for time-series sensor data"*

---

## 5. Federated Learning Strategies (3 Options)

### 5a. FedAvg (Federated Averaging)
- **How:** Each client trains locally → sends model weights → server takes element-wise average
- **Paper:** McMahan et al., 2017
- **Limitation:** Assumes all clients have similar (IID) data distributions

### 5b. FedProx (Federated Proximal)
- **How:** Same as FedAvg, but adds a **proximal term** to the local loss:
  ```
  Loss = Task_Loss + (mu/2) * ||w_local - w_global||²
  ```
  This prevents any client from drifting too far from the global model.
- **Why:** In the real world, Fab A might see mostly "Pass" wafers while Fab B sees more "Fail" wafers. FedProx handles this **non-IID** scenario better.
- **Paper:** Li et al., 2020

### 5c. FedNova (Federated Normalized Averaging)
- **How:** When clients do different amounts of local training, FedAvg becomes biased. FedNova **normalizes** each client's update by its number of local steps before averaging.
- **Why:** In production, some fabs might process data faster than others. FedNova ensures fairness.
- **Paper:** Wang et al., 2020

### Quick Comparison for Interviews:

| Strategy | Handles non-IID? | Handles uneven training? | Complexity |
|----------|:-:|:-:|:-:|
| FedAvg   | ❌ | ❌ | Simple |
| FedProx  | ✅ | ❌ | Medium |
| FedNova  | ❌ | ✅ | Medium |

---

## 6. Differential Privacy (DP)

**Why DP?** Even though FL doesn't share raw data, model updates can leak information. Differential Privacy adds formal mathematical guarantees.

### How DP-FedAvg Works:

```
Client trains model locally
        │
        ▼
Step 1: Clip the update         ← If ||Δw|| > C, scale down to C
        │                        (bounds how much one sample can influence the model)
        ▼
Step 2: Server averages updates
        │
        ▼
Step 3: Add Gaussian noise      ← noise ~ N(0, σ² · C² / n²)
        │                        (makes it impossible to reverse-engineer individual data)
        ▼
    Updated global model
```

### Key Parameters:
| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Noise Multiplier | σ | How much noise to add (higher = more private, less accurate) |
| Max Grad Norm | C | Clipping threshold (bounds sensitivity) |
| Epsilon | ε | Privacy budget (lower = stronger privacy guarantee) |
| Delta | δ | Probability of privacy failure (typically 10⁻⁵) |

### Privacy-Utility Tradeoff:
- **Low ε (e.g., 1-3):** Strong privacy, but accuracy drops
- **Medium ε (e.g., 5-10):** Good balance of privacy and accuracy
- **High ε (e.g., 50+):** Weak privacy, near-full accuracy

**Interview talking point:** *"We implemented a Rényi Differential Privacy (RDP) accountant that tracks cumulative privacy budget across all FL rounds, ensuring we never exceed our privacy budget"*

---

## 7. Anomaly Detection (Autoencoder)

**Concept:** Instead of classifying "Pass vs Fail" directly, we train an autoencoder to learn what **normal** data looks like. If a new sample has a high reconstruction error (the model can't reproduce it well), it's probably anomalous.

```
Input (10, 590) → Encoder LSTM → Latent Space (32 dims) → Decoder LSTM → Reconstructed (10, 590)
                                                                                    │
                                                                            Reconstruction Error = MSE
                                                                                    │
                                                                         Error > Threshold? → ANOMALY!
```

**Why this approach?**
- Works with **very few labeled failures** (unsupervised)
- Detects **novel failure modes** not seen in training
- Can be trained in a federated manner across fabs

---

## 8. Non-IID Data Simulation

In the real world, different fabs see different data distributions. We simulate this:

| Strategy | What it does | Real-world scenario |
|----------|-------------|-------------------|
| **Dirichlet** | Controls label distribution skew via α parameter | Fabs with varying defect rates |
| **Label Skew** | One client gets 80% of one class | A fab specializing in one product type |
| **Quantity Skew** | Unequal dataset sizes across clients | Small fab vs. large fab |

**α parameter (Dirichlet):**
- α = 100 → nearly IID (all clients have similar data)
- α = 0.5 → moderately heterogeneous
- α = 0.1 → highly heterogeneous (very different across clients)

---

## 9. Project Architecture (How Everything Connects)

```
configs/default.yaml    ← Single config file controls everything
        │
        ▼
run_experiment.py       ← Entry point: reads config, runs everything
        │
    ┌───┴───┐
    ▼       ▼
Centralized   Federated Training
Training      │
    │         ├── FL Strategy (FedAvg/FedProx/FedNova)
    │         ├── DP (optional: clipping + noise)
    │         └── Privacy Accountant (tracks ε)
    │
    └────┬────┘
         ▼
    Comparison & Plots
         │
         ▼
    results/logs/*.json  ← Full experiment metadata
         │
         ▼
    dashboard/app.py     ← Streamlit dashboard (reads JSON logs)
```

---

## 10. Key Results to Highlight

| Metric | Centralized | FedAvg | FedProx | With DP |
|--------|:-----------:|:------:|:-------:|:-------:|
| Accuracy | ~0.76 | ~0.73 | ~0.74 | ~0.70 |
| F1-Score | ~0.66 | ~0.63 | ~0.64 | ~0.58 |
| Data Privacy | ❌ None | Partial | Partial | ✅ Formal |

**Key Insights:**
1. Federated models achieve **~96% of centralized performance** without sharing data
2. FedProx slightly outperforms FedAvg in non-IID settings
3. DP adds ~3-5% accuracy cost but provides formal privacy guarantees

---

## 11. Interview Questions & Answers

### Q1: Why Federated Learning instead of just training a centralized model?
> **A:** In semiconductor manufacturing, sensor data contains trade secrets. TSMC wouldn't share data with Intel. FL lets them collaboratively train a model without sharing raw data. Each fab keeps its data local and only shares model weight updates.

### Q2: What is the difference between FedAvg and FedProx?
> **A:** FedAvg simply averages model weights from all clients. This works well when all clients have similar (IID) data. But in reality, different fabs might see different defect rates — FedProx adds a proximal term (μ/2) ||w_local - w_global||² to the local loss function. This prevents any client's local model from drifting too far from the global model, making it more robust to non-IID data.

### Q3: How does Differential Privacy work in your system?
> **A:** Two key mechanisms: (1) **Gradient clipping** — we bound each client's update to a maximum L2 norm C, limiting how much any single data point can influence the model. (2) **Gaussian noise injection** — we add calibrated noise to the aggregated update. Together, these give us (ε, δ)-differential privacy guarantees, tracked by an RDP accountant across all FL rounds.

### Q4: Why did you use LSTM instead of just a feedforward network?
> **A:** Semiconductor sensor data is temporal — readings from step 1 through step 10 form a sequence. LSTMs are designed for sequences; they maintain a hidden state that captures temporal patterns. A feedforward network would miss these sequential dependencies.

### Q5: What is the attention mechanism doing in your Attention-LSTM?
> **A:** The attention layer learns to assign importance weights to each timestep. For example, if a defect typically shows up as a sensor spike at step 7, the attention mechanism learns to focus on that timestep. This provides interpretability — we can inspect the attention weights to understand which production stages are most predictive of failure.

### Q6: Why use an autoencoder for anomaly detection?
> **A:** Traditional classifiers need labeled "failure" data, which is rare (~7% in SECOM). An autoencoder learns the pattern of **normal** wafers. When it sees a defective wafer, it can't reconstruct it well, resulting in high reconstruction error. This approach detects novel failure modes the model has never seen before.

### Q7: How do you handle class imbalance?
> **A:** SECOM is 93/7 split. We use F1-score and AUC-ROC as primary metrics (not accuracy, which would be misleading). The sliding window approach also helps by creating more training samples from the sequential data.

### Q8: What is the privacy-utility tradeoff?
> **A:** More privacy (lower ε) means more noise, which reduces model accuracy. In our experiments, an epsilon of ~10 adds about 3-5% accuracy cost compared to non-private FL. This is a tunable parameter — in production, the privacy officer would set the target epsilon based on regulatory requirements.

### Q9: How is your project structured?
> **A:** Config-driven architecture with separation of concerns: `models/` for architectures, `fl_strategies/` for aggregation algorithms, `privacy/` for DP mechanisms, `anomaly/` for unsupervised detection, and a Streamlit dashboard for visualization. Everything is controlled by a single YAML config file.

### Q10: How would you deploy this in production?
> **A:** Each fab would run a local training agent behind their firewall. A central server (could be cloud-hosted) manages aggregation. Communication uses encrypted channels. The trained global model is pushed back to each fab for inference on their production lines. The dashboard monitors training progress and privacy budget in real-time.

---

## 12. Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core language |
| **PyTorch** | Deep learning framework |
| **NumPy / Pandas** | Data manipulation |
| **Scikit-learn** | Preprocessing, metrics |
| **Matplotlib / Seaborn** | Static visualizations |
| **Plotly** | Interactive charts |
| **Streamlit** | Web dashboard |
| **YAML** | Configuration management |

---

## 13. How to Run The Project

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Preprocess SECOM data
python secom_preprocess.py
python make_windows.py

# Step 3: Run experiment (default config)
python run_experiment.py

# Step 4: Launch dashboard
streamlit run dashboard/app.py
```

### Customize via config (`configs/default.yaml`):
```yaml
model:
  type: "attention_lstm"     # Change model: lstm, attention_lstm, transformer

federated:
  strategy: "fedprox"        # Change strategy: fedavg, fedprox, fednova
  rounds: 10                 # More rounds = better convergence
  dp:
    enabled: true            # Toggle differential privacy
    noise_multiplier: 1.0    # Higher = more private, less accurate
```

---

## 14. Key Papers to Reference

1. **FedAvg:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
2. **FedProx:** Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
3. **FedNova:** Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" (NeurIPS 2020)
4. **DP-FL:** McMahan et al., "Learning Differentially Private Recurrent Language Models" (ICLR 2018)
5. **RDP:** Mironov, "Rényi Differential Privacy" (CSF 2017)
6. **Attention:** Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (ICLR 2015)
7. **Transformer:** Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
