# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Dashboard Page Module
# =============================================================================
"""
Dashboard view rendering the main KPIs, recent prediction activity summaries,
live engine statuses, and quick analytics overview.
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from components.cards import explanation_card, metric_card
from components.detector import get_detector
from components.history import clear_history, history_dataframe
from components.theme import get_current_theme_colors
from src.config import DATASET_STATS_PATH, MODEL_RESULTS_PATH


def _load_stats() -> dict:
    """Load saved dataset/model stats, falling back to defaults."""
    stats = {}
    if os.path.exists(DATASET_STATS_PATH):
        try:
            with open(DATASET_STATS_PATH, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception:  # noqa: BLE001
            pass

    defaults = {
        "total_samples": stats.get("total_samples"),
        "models_trained": stats.get("models_trained"),
        "best_model": stats.get("best_model"),
        "best_accuracy": stats.get("best_accuracy"),
        "best_f1": stats.get("best_f1"),
        "phishing_count": stats.get("phishing_count"),
        "legitimate_count": stats.get("legitimate_count"),
    }
    return {k: v for k, v in defaults.items() if v is not None}


def _load_model_results() -> dict:
    """Load per-model evaluation results for the metrics table."""
    if not os.path.exists(MODEL_RESULTS_PATH):
        return {}
    try:
        with open(MODEL_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def show_dashboard():
    """Renders the main Dashboard view of the Streamlit app."""
    colors = get_current_theme_colors()
    stats = _load_stats()
    model_results = _load_model_results()

    st.markdown(
        f"""
        <h1 style="margin-bottom: 5px;">🛡️ <span class="gradient-text">Security Dashboard</span></h1>
        <p style="color: {colors['text']}; opacity: 0.8; font-size: 1.1rem; margin-bottom: 25px;">
            Real-time status, execution KPIs, and threat monitoring overview.
        </p>
        """,
        unsafe_allow_html=True,
    )

    detector = get_detector()
    model_online = detector.is_loaded

    # Best model from live artifacts (fall back to saved stats)
    best_model_name = (
        (detector.model_info or {}).get("model_name")
        if model_online and detector.model_info
        else stats.get("best_model", "—")
    )

    total_samples = stats.get("total_samples", 0)
    best_accuracy = stats.get("best_accuracy")
    models_trained = stats.get("models_trained", 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Dataset Size", f"{total_samples:,}" if total_samples else "N/A", "📂")
    with col2:
        metric_card("Models Compared", f"{models_trained}" if models_trained else "N/A", "🤖")
    with col3:
        metric_card(
            "Detection Accuracy",
            f"{best_accuracy}%" if best_accuracy is not None else "N/A",
            "🎯",
        )
    with col4:
        metric_card("Best Model", best_model_name or "N/A", "🏆")

    st.markdown("---")

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("### 📊 Threat Distribution Overview")

        phish_count = stats.get("phishing_count")
        legit_count = stats.get("legitimate_count")

        if phish_count is not None and legit_count is not None and (phish_count + legit_count) > 0:
            phish_pct = round(phish_count / (phish_count + legit_count) * 100, 1)
            legit_pct = round(100 - phish_pct, 1)
        else:
            phish_pct, legit_pct = None, None

        dist_df = pd.DataFrame({
            "Classification": ["Legitimate", "Phishing"],
            "Distribution (%)": [legit_pct, phish_pct],
        })

        fig = px.pie(
            dist_df,
            values="Distribution (%)",
            names="Classification",
            hole=0.5,
            color="Classification",
            color_discrete_map={
                "Legitimate": colors["legitimate"],
                "Phishing": colors["phishing"],
            },
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=250,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width="stretch")

        if model_results:
            st.markdown("### 🏆 Model Leaderboard (F1 Score)")
            leaderboard = pd.DataFrame([
                {
                    "Model": name,
                    "Accuracy": round(m.get("accuracy", 0) * 100, 2),
                    "F1 Score": round(m.get("f1_score", 0) * 100, 2),
                }
                for name, m in model_results.items()
            ]).sort_values("F1 Score", ascending=False)
            st.dataframe(leaderboard, width="stretch", hide_index=True)

    with right_col:
        st.markdown("### 🛡️ Core Detection Engine Info")

        status_color = colors["success"] if model_online else colors["danger"]
        status_text = "ACTIVE / PROTECTED" if model_online else "OFFLINE — run main.py"

        engine_status_content = f"""
        <div style="font-size: 0.95rem; line-height: 1.6;">
            <strong>Fitted Model:</strong> <code>{best_model_name}</code><br>
            <strong>NLP Vocabulary size:</strong> 10,000 words (TF-IDF bigrams)<br>
            <strong>Metadata features:</strong> 15 engineered features<br>
            <strong>Status:</strong>
            <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
        </div>
        """
        explanation_card(engine_status_content, type="info", title="Status Diagnostics")

        st.markdown("---")
        st.markdown("### 📝 Latest Activity Logs")

        history_df = history_dataframe()
        if history_df.empty:
            st.info("No prediction activity recorded in this session. Go to the 'Email Detector' page to test.")
        else:
            st.dataframe(history_df.head(5), width="stretch")

            if st.button("Clear Prediction Logs", key="clear_history_btn"):
                clear_history()
                st.rerun()
