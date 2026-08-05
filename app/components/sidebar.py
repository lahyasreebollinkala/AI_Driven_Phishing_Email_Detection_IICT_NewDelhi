# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Sidebar Navigation Component
# =============================================================================
"""
Renders a custom sidebar navigation for the cyber dashboard.
Provides links to different pages, displays system operational status indicators,
and includes a theme toggle switch.
"""

import streamlit as st

from components.cards import status_card
from components.detector import get_detector
from components.theme import get_current_theme_colors, toggle_theme
from src.config import APP_SUBTITLE, APP_TITLE, APP_VERSION


def render_sidebar() -> str:
    """
    Renders the sidebar navigation and utility panels.

    Returns:
        str: Selected page name.
    """
    colors = get_current_theme_colors()

    # App Branding
    st.sidebar.markdown(
        f"""
        <div class="sidebar-logo">
            <h2 style="margin: 0; color: {colors['primary']}; font-weight: 800;">{APP_TITLE}</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: {colors['text']}; opacity: 0.8;">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Navigation")

    pages = {
        "🛡️ Dashboard": "Dashboard",
        "🔍 Email Detector": "Email Detector",
        "📊 Dataset Analytics": "Dataset Analytics",
        "🏆 Model Performance": "Model Performance",
        "ℹ️ About Project": "About Project",
    }

    # Selected page session state handling
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dashboard"

    for display_name, page_id in pages.items():
        if st.sidebar.button(
            display_name,
            key=f"nav_btn_{page_id}",
            width="stretch",
            type="secondary" if st.session_state["current_page"] != page_id else "primary",
        ):
            st.session_state["current_page"] = page_id
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Security Status")

    # Detector status — the cached provider avoids re-loading artifacts per rerun
    detector = get_detector()
    model_online = detector.is_loaded

    status_card("Core Detection Engine", "ONLINE" if model_online else "OFFLINE", ok=model_online)
    status_card("NLP Preprocessor", "READY" if model_online else "ERROR", ok=model_online)
    status_card("Feature Vectorizer", "ACTIVE" if model_online else "INACTIVE", ok=model_online)

    st.sidebar.markdown("---")

    # Theme Toggle & System Info
    theme_label = "☀️ Light Mode" if st.session_state.get("theme") == "dark" else "🌙 Dark Mode"
    if st.sidebar.button(theme_label, key="theme_toggle_btn", width="stretch"):
        toggle_theme()
        st.rerun()

    st.sidebar.markdown(
        f"""
        <div style="text-align: center; font-size: 0.75rem; color: {colors['text']}; opacity: 0.6; margin-top: 15px;">
            <span>Version {APP_VERSION}</span><br>
            <span>IICT B.Tech Project © 2026</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st.session_state["current_page"]
