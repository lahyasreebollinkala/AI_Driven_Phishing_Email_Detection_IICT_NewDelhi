# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Prediction History Component
# =============================================================================
"""
Helpers for storing and rendering the session prediction history.

History is kept in ``st.session_state["prediction_history"]`` (a list of
prediction result dicts) and is capped at ``MAX_PREDICTION_HISTORY`` entries.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import MAX_PREDICTION_HISTORY


def ensure_history() -> list[dict]:
    """Initialise the prediction history in session state."""
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    return st.session_state["prediction_history"]


def add_prediction(result: dict) -> None:
    """Append a prediction to history, trimming to the configured cap."""
    ensure_history().append(result)
    history = st.session_state["prediction_history"]
    if len(history) > MAX_PREDICTION_HISTORY:
        st.session_state["prediction_history"] = history[-MAX_PREDICTION_HISTORY:]


def clear_history() -> None:
    """Remove all saved predictions."""
    st.session_state["prediction_history"] = []


def history_dataframe() -> pd.DataFrame:
    """Return the history as a formatted DataFrame (empty if none)."""
    history = ensure_history()
    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    columns = [
        "timestamp", "verdict", "confidence", "risk_score",
        "risk_level", "model_name", "suspicious_keywords", "detected_urls",
    ]
    df = df[[c for c in columns if c in df.columns]].copy()
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].apply(lambda x: f"{float(x) * 100:.1f}%")
    if "risk_score" in df.columns:
        df["risk_score"] = df["risk_score"].apply(lambda x: f"{x}/100")
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    return df[::-1].reset_index(drop=True)
