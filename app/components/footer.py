# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Footer Component
# =============================================================================
"""
Renders a consistent footer across all application pages.
"""

import streamlit as st

from src.config import APP_VERSION
from components.theme import get_current_theme_colors


def render_footer():
    """Render the page footer with branding and version info."""
    colors = get_current_theme_colors()
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 0.78rem; color: {colors['text']}; opacity: 0.55; padding: 10px 0;">
            🛡️ <strong>AI Phishing Shield</strong> v{APP_VERSION} — AI-Driven Phishing Email Detection Using NLP
            &nbsp;·&nbsp; IICT B.Tech Project © 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
