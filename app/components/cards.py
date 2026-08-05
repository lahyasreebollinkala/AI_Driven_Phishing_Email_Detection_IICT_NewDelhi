# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Reusable Cards & UI Components
# =============================================================================
"""
Defines reusable, styled glassmorphism components for Streamlit.
Includes:
    - metric_card: KPI numeric display with custom styling
    - status_card: Pulsing system status displays
    - explanation_card: Custom container with color borders matching threat level
"""

import streamlit as st
from .theme import get_current_theme_colors


def metric_card(title, value, icon="📈", delta=None, delta_higher_is_better=True):
    """
    Renders an animated glassmorphic card for exhibiting KPI metrics.

    Args:
        title (str): The metric label.
        value (str|int|float): The metric value to display.
        icon (str): Emoji or icon prefix.
        delta (str|int|float): Variance delta indicator.
        delta_higher_is_better (bool): If True, positive delta is green.
    """
    colors = get_current_theme_colors()

    delta_html = ""
    if delta is not None:
        val_is_positive = str(delta).startswith("+") or (isinstance(delta, (int, float)) and delta > 0)
        
        if val_is_positive:
            color_class = "delta-up" if delta_higher_is_better else "delta-down"
            icon_arrow = "▲"
        else:
            color_class = "delta-down" if delta_higher_is_better else "delta-up"
            icon_arrow = "▼"
            
        delta_html = f'<div class="metric-delta {color_class}">{icon_arrow} {delta}</div>'

    card_html = f"""
    <div class="cyber-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="metric-label">{title}</div>
            <div style="font-size: 1.5rem;">{icon}</div>
        </div>
        <div class="metric-val">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def status_card(label, value, ok=True):
    """
    Renders a system status indicator with a pulsing status dot.

    Args:
        label (str): Service or component name.
        value (str): Value status text (e.g. 'ONLINE').
        ok (bool): If True, pulse is green; otherwise red.
    """
    colors = get_current_theme_colors()
    pulse_color = colors["success"] if ok else colors["danger"]
    
    pulse_css = f"""
    <style>
        .pulse-{label.replace(' ', '-')} {{
            height: 10px;
            width: 10px;
            background-color: {pulse_color};
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 {pulse_color}77;
            animation: pulse-{label.replace(' ', '-')} 1.8s infinite;
        }}
        @keyframes pulse-{label.replace(' ', '-')} {{
            0% {{
                transform: scale(0.95);
                box-shadow: 0 0 0 0 {pulse_color}77;
            }}
            70% {{
                transform: scale(1);
                box-shadow: 0 0 0 6px {pulse_color}00;
            }}
            100% {{
                transform: scale(0.95);
                box-shadow: 0 0 0 0 {pulse_color}00;
            }}
        }}
    </style>
    """
    
    card_html = f"""
    {pulse_css}
    <div class="sidebar-status-box" style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.85rem; font-weight: 500; color: {colors['text']};">{label}</span>
        <span style="display: flex; align-items: center; gap: 8px;">
            <span class="pulse-{label.replace(' ', '-')}"></span>
            <strong style="font-size: 0.85rem; color: {pulse_color};">{value}</strong>
        </span>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def explanation_card(content, type="info", title=None):
    """
    Display a styled block with a left-accent border matching a specific type.

    Args:
        content (str): Text or HTML contents.
        type (str): Border style ('info', 'phishing', 'legitimate', 'warning').
        title (str): Optional card header title.
    """
    card_class = f"cyber-card cyber-card-{type}"
    title_html = f"<h4 style='margin-top: 0; margin-bottom: 10px;'>{title}</h4>" if title else ""
    
    html = f"""
    <div class="{card_class}">
        {title_html}
        <div>{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
