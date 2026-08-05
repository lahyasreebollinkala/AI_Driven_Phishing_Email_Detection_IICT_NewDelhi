# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Model Performance Page Module
# =============================================================================
"""
Model Performance and Validation View.
Loads evaluation reports and displays:
    - Interactive Metrics Table
    - Accuracy / F1 grouped comparisons (Plotly bar)
    - ROC Curves
    - Precision-Recall comparison curves
    - Confusion Matrices (Grid image)
    - Training vs Prediction Time scatter chart
    - Feature Importance details (for tree models)
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from components.theme import get_current_theme_colors
from src.config import COMPARISON_TABLE_PATH, IMAGES_DIR
from src.utils import load_json
from src.evaluate_models import CLASSIFICATION_REPORTS_PATH


def show_model_performance():
    """Renders the Model Performance comparison Page."""
    colors = get_current_theme_colors()

    st.markdown(
        f"""
        <h1>🏆 <span class="gradient-text">Model Performance Comparison</span></h1>
        <p style="color: {colors['text']}; opacity: 0.8; font-size: 1.1rem; margin-bottom: 25px;">
            Comparative analysis of the trained machine-learning classifiers.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Check for comparison table
    if not os.path.exists(COMPARISON_TABLE_PATH):
        st.warning(
            "⚠️ Model performance comparison data not found. Please run the training "
            "pipeline (main.py) to generate validation reports."
        )
        return

    metrics_df = pd.read_csv(COMPARISON_TABLE_PATH)

    # 1. Metric Summary Table
    st.markdown("### 📊 Overall Evaluation Metrics")
    highlight_cols = [
        c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]
        if c in metrics_df.columns
    ]
    min_cols = [c for c in ["Training Time (s)", "Prediction Time (s)"] if c in metrics_df.columns]

    styled = metrics_df.style.highlight_max(subset=highlight_cols, color="rgba(16, 185, 129, 0.25)")
    if min_cols:
        styled = styled.highlight_min(subset=min_cols, color="rgba(16, 185, 129, 0.25)")

    st.dataframe(styled, width="stretch")

    # CSV Export button
    col_csv, _ = st.columns([1, 4])
    with col_csv:
        csv_data = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Metrics CSV 📥",
            data=csv_data,
            file_name="model_performance_comparison.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown("---")

    tab_metrics, tab_curves, tab_matrices, tab_importance, tab_report = st.tabs([
        "📊 Bar Chart Comparisons",
        "📈 ROC & PR Curves",
        "🧩 Confusion Matrices",
        "📐 Feature Importance",
        "📋 Classification Reports",
    ])

    with tab_metrics:
        st.markdown("### Interactive Metric Comparison")

        value_cols = [
            c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]
            if c in metrics_df.columns
        ]
        melted_df = metrics_df.melt(
            id_vars="Model",
            value_vars=value_cols,
            var_name="Metric",
            value_name="Score (%)",
        )

        fig_bar = px.bar(
            melted_df,
            x="Metric",
            y="Score (%)",
            color="Model",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[70, 102]))
        st.plotly_chart(fig_bar, width="stretch")

        st.markdown("---")

        st.markdown("### Efficiency vs Accuracy Trade-off")
        fig_scatter = px.scatter(
            metrics_df,
            x="Training Time (s)",
            y="Accuracy",
            size="F1 Score",
            color="Model",
            text="Model",
            labels={
                "Training Time (s)": "Training Duration (Seconds)",
                "Accuracy": "Accuracy (%)",
            },
            title="Accuracy vs Training Time (Bubble Size = F1 Score)",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, width="stretch")

    with tab_curves:
        col_roc, col_pr = st.columns(2)

        with col_roc:
            st.markdown("### ROC Curves Overlay")
            roc_path = os.path.join(IMAGES_DIR, "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(Image.open(roc_path), width="stretch")
            else:
                st.info("ROC Curve overlay chart not found. Run main.py to generate.")

        with col_pr:
            st.markdown("### Precision-Recall Curves Overlay")
            pr_path = os.path.join(IMAGES_DIR, "precision_recall_curves.png")
            if os.path.exists(pr_path):
                st.image(Image.open(pr_path), width="stretch")
            else:
                st.info("Precision-Recall Curve overlay chart not found. Run main.py to generate.")

    with tab_matrices:
        st.markdown("### Confusion Matrices Grid")
        cm_path = os.path.join(IMAGES_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), width="stretch")
        else:
            st.info("Confusion Matrices grid chart not found. Run main.py to generate.")

    with tab_importance:
        st.markdown("### Top Engineered Features Importance")

        selected_model = st.selectbox(
            "Select Model for Feature Importance:",
            ["Random Forest", "XGBoost", "LightGBM"],
        )

        img_name = f"feature_importance_{selected_model.lower().replace(' ', '_')}.png"
        fi_path = os.path.join(IMAGES_DIR, img_name)

        if os.path.exists(fi_path):
            st.image(Image.open(fi_path), width="stretch")
        else:
            st.info(
                f"Feature importance chart for {selected_model} not found or model was skipped during training."
            )

    with tab_report:
        st.markdown("### Detailed Classification Reports")
        if not os.path.exists(CLASSIFICATION_REPORTS_PATH):
            st.info("Classification reports not found. Run main.py to generate.")
        else:
            reports = load_json(CLASSIFICATION_REPORTS_PATH)
            model_choice = st.selectbox("Select Model:", list(reports.keys()))
            report = reports.get(model_choice, {})
            if report:
                metrics_keys = ["accuracy"]
                acc_row = report.get("accuracy")
                per_class = {k: v for k, v in report.items() if k not in metrics_keys}
                rows = []
                for label_name, m in per_class.items():
                    if isinstance(m, dict):
                        rows.append({
                            "Class": label_name,
                            "Precision": round(m.get("precision", 0), 4),
                            "Recall": round(m.get("recall", 0), 4),
                            "F1": round(m.get("f1-score", 0), 4),
                            "Support": m.get("support", 0),
                        })
                rep_df = pd.DataFrame(rows)
                st.dataframe(rep_df, width="stretch", hide_index=True)
                if acc_row is not None:
                    st.metric("Overall Accuracy", f"{acc_row:.4f}")
            else:
                st.info("No report data for this model.")
