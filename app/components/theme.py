# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Theme Management Component
# =============================================================================
"""
Manages application themes (Dark/Light) and provides customized HTML/CSS
injection wrappers. Also configures Plotly templates to match the active theme.
"""

import streamlit as st
import plotly.io as pio
from src.config import COLORS

# Theme Constants
THEME_DARK = "dark"
THEME_LIGHT = "light"


def init_theme_state():
    """Initialize theme in Streamlit session state if not already set."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = THEME_DARK


def toggle_theme():
    """Toggle between Dark and Light theme."""
    if st.session_state.get("theme") == THEME_LIGHT:
        st.session_state["theme"] = THEME_DARK
    else:
        st.session_state["theme"] = THEME_LIGHT


def get_current_theme_colors():
    """
    Get color palette dictionary matching the current theme.

    Returns:
        dict: Theme color hex codes.
    """
    init_theme_state()
    theme = st.session_state["theme"]

    if theme == THEME_LIGHT:
        return {
            "primary": "#0284c7",        # Sky blue
            "secondary": "#7c3aed",      # Purple
            "success": "#16a34a",        # Green
            "danger": "#dc2626",         # Red
            "warning": "#d97706",        # Amber
            "info": "#2563eb",           # Blue
            "background": "#f8fafc",     # Light slate
            "card": "#ffffff",           # White card
            "text": "#0f172a",           # Dark slate text
            "phishing": "#dc2626",
            "legitimate": "#16a34a",
        }
    else:
        # Return default dark colors from config
        return COLORS


def apply_theme_css():
    """Inject CSS variables and styles based on the active theme."""
    colors = get_current_theme_colors()
    theme = st.session_state["theme"]

    # Inject base overrides for Light vs Dark theme
    if theme == THEME_LIGHT:
        theme_override_css = f"""
        <style>
            html, body, [data-testid="stAppViewContainer"] {{
                background: #f8fafc !important;
                color: #0f172a !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: #f1f5f9 !important;
                border-right: 1px solid rgba(2, 132, 199, 0.15) !important;
            }}
            .cyber-card {{
                background: #ffffff !important;
                border: 1px solid rgba(2, 132, 199, 0.15) !important;
                color: #0f172a !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
            }}
            .cyber-card:hover {{
                border-color: rgba(2, 132, 199, 0.4) !important;
                box-shadow: 0 10px 20px rgba(2, 132, 199, 0.1) !important;
            }}
            .metric-val {{
                color: #0f172a !important;
            }}
            .metric-label {{
                color: #64748b !important;
            }}
            .sidebar-status-box {{
                background: #e2e8f0 !important;
                border: 1px solid rgba(2, 132, 199, 0.15) !important;
                color: #0f172a !important;
            }}
            .stTextArea>div>div>textarea {{
                background-color: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid rgba(2, 132, 199, 0.2) !important;
            }}
            .stTextArea>div>div>textarea:focus {{
                border-color: rgba(2, 132, 199, 0.6) !important;
            }}
            code {{
                background-color: #f1f5f9 !important;
                color: #0284c7 !important;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                background-color: #e2e8f0 !important;
                border: 1px solid rgba(2, 132, 199, 0.1) !important;
            }}
            .stTabs [data-baseweb="tab"] {{
                color: #64748b !important;
            }}
            .stTabs [aria-selected="true"] {{
                background-color: rgba(2, 132, 199, 0.15) !important;
                color: #0284c7 !important;
            }}
            hr {{
                background: linear-gradient(90deg, rgba(2, 132, 199, 0) 0%, rgba(2, 132, 199, 0.3) 50%, rgba(2, 132, 199, 0) 100%) !important;
            }}
        </style>
        """
    else:
        theme_override_css = """
        <style>
            /* Default styles already handled by style.css */
        </style>
        """

    st.markdown(theme_override_css, unsafe_allow_html=True)
    setup_plotly_theme(theme, colors)


def setup_plotly_theme(theme, colors):
    """
    Set default Plotly template based on the current theme.

    Args:
        theme (str): 'dark' or 'light'.
        colors (dict): Theme colors.
    """
    if theme == THEME_LIGHT:
        pio.templates.default = "plotly_white"
    else:
        pio.templates.default = "plotly_dark"

    # Set background colors globally for Plotly layouts
    template = pio.templates[pio.templates.default]
    template.layout.paper_bgcolor = "rgba(0,0,0,0)"
    template.layout.plot_bgcolor = "rgba(0,0,0,0)"
    template.layout.font.color = colors["text"]
    template.layout.font.family = "Inter, sans-serif"
