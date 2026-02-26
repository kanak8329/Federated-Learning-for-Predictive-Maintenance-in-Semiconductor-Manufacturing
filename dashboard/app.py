# dashboard/app.py
"""
🚀 Federated Learning Dashboard — Semiconductor Predictive Maintenance

Interactive Streamlit dashboard for monitoring FL experiments.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
import glob
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Page Config ──────────────────────────────────────────

st.set_page_config(
    page_title="FL Semiconductor Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────

st.markdown('<p class="main-header">🔬 Federated Learning — Semiconductor Predictive Maintenance</p>', unsafe_allow_html=True)
st.markdown("*Privacy-preserving predictive maintenance using advanced federated learning strategies*")
st.divider()


# ── Sidebar ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=60)
    st.title("⚙️ Configuration")

    # Load experiment logs
    log_dir = "results/logs"
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.json"))) if os.path.exists(log_dir) else []

    if log_files:
        selected_log = st.selectbox(
            "📁 Select Experiment",
            log_files,
            format_func=lambda x: os.path.basename(x),
        )
    else:
        selected_log = None
        st.warning("No experiment logs found. Run an experiment first!")
        st.code("python run_experiment.py", language="bash")

    st.divider()
    st.markdown("### 📋 Quick Links")
    st.markdown("- 📊 [SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)")
    st.markdown("- 📄 [FedAvg Paper](https://arxiv.org/abs/1602.05629)")
    st.markdown("- 🔒 [DP-FL Paper](https://arxiv.org/abs/1710.06963)")


# ── Load Data ───────────────────────────────────────────

def load_experiment(path):
    with open(path, "r") as f:
        return json.load(f)

# ── Main Content ────────────────────────────────────────

if selected_log:
    exp = load_experiment(selected_log)

    # ── Overview Metrics ────────────────────────────────
    st.subheader("📊 Experiment Overview")

    col1, col2, col3, col4 = st.columns(4)

    final = exp.get("final_metrics", {})
    cent = final.get("centralized", {})
    fed = final.get("federated", {})

    with col1:
        st.metric("🎯 Centralized Acc", f"{cent.get('accuracy', 0):.4f}")
    with col2:
        st.metric("🌐 Federated Acc", f"{fed.get('accuracy', 0):.4f}",
                   delta=f"{(fed.get('accuracy', 0) - cent.get('accuracy', 0)):.4f}")
    with col3:
        st.metric("⏱️ Duration",
                   f"{exp.get('duration_seconds', 0):.1f}s")
    with col4:
        strategy = exp.get("config", {}).get("federated", {}).get("strategy", "fedavg")
        st.metric("🔄 Strategy", strategy.upper())

    st.divider()

    # ── Tabs ────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Training Curves",
        "🏆 Model Comparison",
        "👥 Client Analysis",
        "🔒 Privacy",
        "📋 Configuration",
    ])

    # ── Tab 1: Training Curves ──────────────────────────
    with tab1:
        st.subheader("Training Progress")

        rounds_data = exp.get("rounds", [])
        if rounds_data:
            round_nums = [r["round"] for r in rounds_data]
            accs = [r["metrics"].get("accuracy", 0) for r in rounds_data]
            f1s = [r["metrics"].get("f1", 0) for r in rounds_data]
            losses = [r["metrics"].get("loss") for r in rounds_data]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Accuracy Over Rounds/Epochs", "F1 Score Over Rounds/Epochs"),
            )

            fig.add_trace(
                go.Scatter(x=round_nums, y=accs, mode="lines+markers",
                           name="Accuracy", line=dict(color="#3498db", width=3)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=round_nums, y=f1s, mode="lines+markers",
                           name="F1 Score", line=dict(color="#e74c3c", width=3)),
                row=1, col=2,
            )

            fig.update_layout(
                height=400,
                showlegend=True,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Loss curve if available
            if any(l is not None for l in losses):
                fig_loss = go.Figure()
                valid_losses = [(r, l) for r, l in zip(round_nums, losses) if l is not None]
                if valid_losses:
                    fig_loss.add_trace(go.Scatter(
                        x=[r for r, _ in valid_losses],
                        y=[l for _, l in valid_losses],
                        mode="lines+markers",
                        name="Training Loss",
                        line=dict(color="#f39c12", width=3),
                    ))
                    fig_loss.update_layout(
                        title="Training Loss",
                        template="plotly_white",
                        height=350,
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("No round data available yet.")

    # ── Tab 2: Model Comparison ─────────────────────────
    with tab2:
        st.subheader("Centralized vs Federated Performance")

        if cent and fed:
            metrics_names = ["accuracy", "f1", "precision", "recall"]
            cent_vals = [cent.get(m, 0) for m in metrics_names]
            fed_vals = [fed.get(m, 0) for m in metrics_names]

            fig = go.Figure(data=[
                go.Bar(name="Centralized", x=metrics_names, y=cent_vals,
                       marker_color="#e74c3c", text=[f"{v:.3f}" for v in cent_vals],
                       textposition="outside"),
                go.Bar(name="Federated", x=metrics_names, y=fed_vals,
                       marker_color="#3498db", text=[f"{v:.3f}" for v in fed_vals],
                       textposition="outside"),
            ])

            fig.update_layout(
                barmode="group",
                yaxis=dict(range=[0, 1.15]),
                template="plotly_white",
                height=500,
                title="Performance Metrics Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.markdown("### 📊 Detailed Metrics Table")
            import pandas as pd
            df = pd.DataFrame({
                "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"],
                "Centralized": [cent.get(m, "N/A") for m in ["accuracy", "f1", "precision", "recall", "roc_auc"]],
                "Federated": [fed.get(m, "N/A") for m in ["accuracy", "f1", "precision", "recall", "roc_auc"]],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tab 3: Client Analysis ──────────────────────────
    with tab3:
        st.subheader("Per-Client Data Distribution")

        client_dir = exp.get("config", {}).get("data", {}).get("client_dir", "data/windows/clients")
        n_clients = exp.get("config", {}).get("data", {}).get("n_clients", 3)

        if os.path.exists(client_dir):
            client_stats = []
            for cid in range(1, n_clients + 1):
                train_path = os.path.join(client_dir, f"client{cid}_y_train.npy")
                test_path = os.path.join(client_dir, f"client{cid}_y_test.npy")
                if os.path.exists(train_path) and os.path.exists(test_path):
                    y_train = np.load(train_path)
                    y_test = np.load(test_path)
                    total = len(y_train) + len(y_test)
                    pass_pct = (np.sum(np.concatenate([y_train, y_test]) == 0) / total) * 100
                    client_stats.append({
                        "Client": f"Client {cid}",
                        "Train Samples": len(y_train),
                        "Test Samples": len(y_test),
                        "Total": total,
                        "Pass %": f"{pass_pct:.1f}%",
                        "Fail %": f"{100 - pass_pct:.1f}%",
                    })

            if client_stats:
                st.dataframe(client_stats, use_container_width=True, hide_index=True)

                # Distribution chart
                fig = go.Figure()
                for stat in client_stats:
                    cname = stat["Client"]
                    fig.add_trace(go.Bar(
                        name=cname,
                        x=["Train", "Test"],
                        y=[stat["Train Samples"], stat["Test Samples"]],
                    ))

                fig.update_layout(
                    barmode="group",
                    title="Client Dataset Sizes",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Client data directory not found.")

    # ── Tab 4: Privacy ──────────────────────────────────
    with tab4:
        st.subheader("🔒 Differential Privacy Budget")

        dp_config = exp.get("config", {}).get("federated", {}).get("dp", {})

        if dp_config.get("enabled", False):
            # Find DP events
            dp_events = [e for e in exp.get("events", []) if e.get("event") == "dp_round"]

            if dp_events:
                eps_vals = [e["details"]["epsilon"] for e in dp_events]
                rounds = list(range(1, len(eps_vals) + 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rounds, y=eps_vals,
                    mode="lines+markers",
                    name="ε (Privacy Budget)",
                    line=dict(color="#e74c3c", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(231, 76, 60, 0.1)",
                ))

                target_eps = dp_config.get("target_epsilon", 10)
                fig.add_hline(y=target_eps, line_dash="dash",
                              annotation_text=f"Target ε = {target_eps}",
                              line_color="#2ecc71")

                fig.update_layout(
                    title="Privacy Budget Accumulation (ε over Rounds)",
                    xaxis_title="Communication Round",
                    yaxis_title="ε (Epsilon)",
                    template="plotly_white",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Privacy info cards
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("σ (Noise Multiplier)", dp_config.get("noise_multiplier", "N/A"))
                with c2:
                    st.metric("C (Max Grad Norm)", dp_config.get("max_grad_norm", "N/A"))
                with c3:
                    st.metric("Current ε", f"{eps_vals[-1]:.4f}" if eps_vals else "N/A")
            else:
                st.info("DP is enabled but no round data recorded yet.")
        else:
            st.info("Differential Privacy is **not enabled** in this experiment.")
            st.markdown("""
            To enable DP, set in your config:
            ```yaml
            federated:
              dp:
                enabled: true
                noise_multiplier: 1.0
                max_grad_norm: 1.0
            ```
            """)

    # ── Tab 5: Configuration ────────────────────────────
    with tab5:
        st.subheader("📋 Experiment Configuration")

        config = exp.get("config", {})
        if config:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🧪 Experiment")
                st.json(config.get("experiment", {}))

                st.markdown("### 🤖 Model")
                st.json(config.get("model", {}))

            with col2:
                st.markdown("### 🌐 Federated")
                st.json(config.get("federated", {}))

                st.markdown("### 📂 Data")
                st.json(config.get("data", {}))

        # Events timeline
        events = exp.get("events", [])
        if events:
            st.markdown("### 📜 Event Timeline")
            for evt in events:
                time_str = evt.get("time", "")
                event_name = evt.get("event", "")
                details = evt.get("details", "")
                st.markdown(f"- `{time_str}` — **{event_name}** {details if details else ''}")

else:
    # ── Welcome Screen ──────────────────────────────────
    st.markdown("""
    ## 🚀 Welcome!

    This dashboard visualizes federated learning experiments for semiconductor
    predictive maintenance.

    ### Getting Started

    1. **Preprocess data**:
       ```bash
       python secom_preprocess.py
       python make_windows.py
       ```

    2. **Run an experiment**:
       ```bash
       python run_experiment.py --config configs/default.yaml
       ```

    3. **Launch this dashboard**:
       ```bash
       streamlit run dashboard/app.py
       ```

    ### 🏗️ Architecture

    | Component | Description |
    |-----------|------------|
    | **Models** | LSTM, Attention-LSTM, Transformer |
    | **FL Strategies** | FedAvg, FedProx, FedNova |
    | **Privacy** | DP-FedAvg with RDP accountant |
    | **Anomaly Detection** | LSTM Autoencoder |
    """)

# ── Footer ──────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85rem;'>"
    "🔬 Federated Learning for Semiconductor Predictive Maintenance | "
    "Built with Streamlit & PyTorch"
    "</div>",
    unsafe_allow_html=True,
)
