# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Streamlit Web Application Entry Point
# =============================================================================
"""
Main application router for the AI Phishing Shield dashboard.
Responsible for initializing pages, configuring layouts, injecting style assets,
and routing sidebar selection to corresponding sub-page views.
"""

import streamlit as st
import os

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Set Streamlit Page Settings (Must be the very first Streamlit command)
st.set_page_config(
    page_title="AI Phishing Shield — NLP Email Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.theme import init_theme_state, apply_theme_css
from components.sidebar import render_sidebar
from components.footer import render_footer
from components.history import ensure_history

# Import Page Views
from pages.dashboard import show_dashboard
from pages.email_detector import show_email_detector
from pages.dataset_analytics import show_dataset_analytics
from pages.model_performance import show_model_performance
from pages.about import show_about_project


def main():
    """Main routing and styling coordinator for the dashboard app."""

    # 1. Initialize session variables (theme and prediction history)
    init_theme_state()
    ensure_history()

    # 2. Inject Static style.css sheet
    style_css_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "static", "style.css"
    )
    if os.path.exists(style_css_path):
        try:
            with open(style_css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading style.css: {e}")

    # 3. Apply active theme overlays (Light vs Dark mode)
    apply_theme_css()

    # 4. Render Sidebar Navigation & get selected page ID
    selected_page = render_sidebar()

    # 5. Page Router
    if selected_page == "Dashboard":
        show_dashboard()
    elif selected_page == "Email Detector":
        show_email_detector()
    elif selected_page == "Dataset Analytics":
        show_dataset_analytics()
    elif selected_page == "Model Performance":
        show_model_performance()
    elif selected_page == "About Project":
        show_about_project()
    else:
        st.error(f"Unknown page selection: '{selected_page}'")

    # 6. Footer
    render_footer()


if __name__ == "__main__":
    main()
