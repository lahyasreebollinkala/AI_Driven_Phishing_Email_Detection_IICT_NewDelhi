# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# About Page Module
# =============================================================================
"""
About Project Page rendering implementation methodologies,
architecture maps, development libraries, academic credits,
and licensing info.
"""

from __future__ import annotations

import textwrap

import streamlit as st

from components.theme import get_current_theme_colors


def show_about_project():
    """Renders the About Project Page."""
    colors = get_current_theme_colors()

    st.markdown(
        textwrap.dedent(
            f"""
            <h1>ℹ️ <span class="gradient-text">About Project</span></h1>
            <p style="color: {colors['text']}; opacity: 0.8; font-size: 1.1rem; margin-bottom: 25px;">
                Academic specifications and design architectures for the final-year B.Tech project demonstration.
            </p>
            """
        ),
        unsafe_allow_html=True,
    )

    # Grid columns
    col_details, col_credits = st.columns([3, 2])

    with col_details:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="cyber-card">
                    <h3 style="margin-top:0; color: {colors['primary']};">Project Objective</h3>
                    <p>
                        Phishing attacks remain one of the most prevalent and damaging cybersecurity threats.
                        This project presents an end-to-end Machine Learning pipeline utilizing Natural Language Processing (NLP)
                        and engineered metadata features to classify emails as <strong>Phishing</strong> or <strong>Legitimate</strong> with high precision.
                    </p>
                    <p>
                        By combining lexical patterns (TF-IDF Vectorization) with structural characteristics
                        (metadata features like URL density, uppercase character distribution, and urgency keyword matching),
                        the system establishes a robust defense perimeter capable of neutralizing social-engineering vectors.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="cyber-card">
                    <h3 style="margin-top:0; color: {colors['primary']};">Methodology Workflow</h3>
                    <p>
                        Our pipeline processes email text through a modular, reproducible workflow:
                    </p>
                    <ol style="line-height: 1.8;">
                        <li><strong>Ingestion:</strong> Merging 7 distinct public and academic datasets (~165k samples).</li>
                        <li><strong>NLP Preprocessing:</strong> Cleaning HTML/URLs, case folding, tokenization, stopword removal, and WordNet lemmatization.</li>
                        <li><strong>Feature Extraction:</strong> Compiling TF-IDF matrices (10,000 top n-grams) and hstacking 15 scaled structural metadata features.</li>
                        <li><strong>Model Comparison:</strong> Benchmarking 6 distinct classifiers: Logistic Regression, Naive Bayes, Random Forest, XGBoost, LightGBM, and a Neural Network (MLP).</li>
                        <li><strong>Inference Deployment:</strong> Streamlit-powered Web Application exposing the best-performing model.</li>
                    </ol>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with col_credits:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="cyber-card">
                    <h3 style="margin-top:0; color: {colors['primary']};">System Architecture</h3>
                    <pre style="background: rgba(0,0,0,0.2); padding:10px; border-radius:6px; font-family: monospace; font-size:0.8rem; overflow-x:auto; border: 1px solid rgba(0,212,255,0.05);">
dataset/ -> Ingest 7 raw datasets
  ↓
src/preprocessing.py -> NLP cleaning (HTML, URLs, lemmatize)
  ↓
src/feature_engineering.py -> TF-IDF (10k) + 15 metadata features
  ↓
src/train_models.py -> 6 classifiers compared
  ↓
src/evaluate_models.py -> Metrics + best model selection
  ↓
models/ -> joblib artifacts (best_model, vectorizer, scaler)
  ↓
app/app.py -> Streamlit Web App
src/api.py  -> FastAPI REST API (/docs Swagger)
                    </pre>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="cyber-card">
                    <h3 style="margin-top:0; color: {colors['primary']};">Technologies Used</h3>
                    <span style="display: inline-flex; flex-wrap: wrap; gap: 8px; font-size:0.85rem; font-weight:600;">
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">Python 3.14</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">NLTK</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">Scikit-Learn</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">XGBoost</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">LightGBM</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">Streamlit</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">FastAPI</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">Docker</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">Plotly</span>
                        <span style="background: rgba(0,212,255,0.15); color: {colors['primary']}; padding: 4px 8px; border-radius: 4px;">pytest</span>
                    </span>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="cyber-card">
                    <h3 style="margin-top:0; color: {colors['primary']};">Academic Credits</h3>
                    <p style="margin-bottom:5px; font-size:0.9rem;">
                        <strong>Institution:</strong> Indian Institute of Computing and Technology (IICT)
                    </p>
                    <p style="margin-bottom:5px; font-size:0.9rem;">
                        <strong>Project Title:</strong> AI-Driven Phishing Email Detection Using NLP
                    </p>
                    <p style="font-size:0.9rem;">
                        <strong>Academic Year:</strong> 2025 - 2026
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
