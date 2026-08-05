# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Email Detector Page Module
# =============================================================================
"""
Interactive Phishing Email Detection module.

Features:
    - Model selection among all saved classifiers
    - Paste email content (subject + body) or upload .txt/.eml files
    - Batch CSV upload with results export
    - Real-time prediction output, risk gauge, detected URLs,
      highlighted suspicious terms, recommendations, copy + report export
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.cards import explanation_card
from components.detector import get_detector
from components.history import add_prediction
from components.theme import get_current_theme_colors
from src.predict import PhishingDetector


def highlight_phish_keywords(text: str, keywords: list[str]) -> str:
    """
    Highlights found phishing keywords inside the text with inline CSS classes.

    Args:
        text (str): Raw email text.
        keywords (list[str]): List of suspicious keywords.

    Returns:
        str: HTML string with highlighted keywords.
    """
    if not keywords:
        return text.replace("\n", "<br>")

    # Sort keywords by length descending to match longer keywords first
    sorted_kws = sorted(set(keywords), key=len, reverse=True)
    html_text = text

    for kw in sorted_kws:
        pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
        html_text = pattern.sub(r'<span class="highlight-phish-keyword">\1</span>', html_text)

    return html_text.replace("\n", "<br>")


def _render_gauge(risk_score: int, verdict: str, colors: dict):
    """Render the interactive cyber-risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cyber Risk Factor", 'font': {'size': 16, 'color': colors['text']}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': colors['text']},
            'bar': {'color': colors['danger'] if verdict == "Phishing" else colors['success']},
            'bgcolor': colors['card'],
            'borderwidth': 2,
            'bordercolor': "rgba(0, 212, 255, 0.1)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.15)'},
            ],
        },
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': colors['text'], 'family': "Inter"},
    )
    st.plotly_chart(fig, width="stretch")


def _copy_button(text: str, key: str):
    """Render a button that copies text to the clipboard via JS."""
    html = f"""
    <script>
    function copyText_{key}() {{
        navigator.clipboard.writeText({text!r}).then(() => {{
            const el = document.getElementById('copy_ok_{key}');
            if (el) {{ el.style.display = 'inline'; setTimeout(() => el.style.display = 'none', 1500); }}
        }});
    }}
    </script>
    <button onclick="copyText_{key}()" style="...">Copy</button>
    <span id="copy_ok_{key}" style="display:none;color:#10b981;"> Copied!</span>
    """
    st.components.v1.html(html, height=60)


def show_email_detector():
    """Renders the Email Detector Page."""
    colors = get_current_theme_colors()

    st.markdown(
        f"""
        <h1>🔍 <span class="gradient-text">Email Detector</span></h1>
        <p style="color: {colors['text']}; opacity: 0.8; font-size: 1.1rem; margin-bottom: 25px;">
            Input email details to scan for phishing attempts, analyze risk thresholds, and inspect credentials.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Model selection (uses the cached detector pool)
    available = PhishingDetector.available_models()
    options = ["Best Model (recommended)"] + available
    selected = st.selectbox("🤖 Detection Model", options, index=0)

    model_name = None if selected.startswith("Best") else selected.split(" (")[0]

    detector = get_detector(model_name)
    if not detector.is_loaded:
        st.error(
            "⚠️ The Detection Model artifacts could not be loaded. "
            "Please ensure main.py has run successfully."
        )
        return

    # Input method: Single vs Batch
    input_mode = st.radio(
        "Input Mode",
        ["Single Email", "Batch CSV Upload"],
        horizontal=True,
        help="Analyse one email interactively, or bulk-scan a CSV of emails.",
    )

    email_text = ""
    email_subject = ""

    if input_mode == "Single Email":
        tab_text, tab_file = st.columns(2)

        with tab_text:
            st.markdown("### Paste Email Content")
            email_subject = st.text_input(
                "Subject (optional)",
                key="detector_subject",
                placeholder="Email subject line...",
            )
            email_text = st.text_area(
                "Copy & Paste the email header + body here:",
                height=250,
                placeholder="Type or paste the email contents you wish to audit...",
                key="detector_text_area",
            ) or ""

        with tab_file:
            st.markdown("### Upload Document File")
            uploaded_file = st.file_uploader(
                "Upload an email file (.txt, .eml)",
                type=["txt", "eml"],
                key="detector_file_uploader",
            )
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                with st.spinner("Parsing uploaded file..."):
                    file_text_res = detector.predict_from_text_bytes(file_bytes, uploaded_file.name)
                    if "error" not in file_text_res:
                        email_text = file_bytes.decode("utf-8", errors="ignore")
                        st.success(f"Successfully uploaded: {uploaded_file.name}")
                    else:
                        st.error(f"Error reading file: {file_text_res['error']}")

        col_pred, col_reset = st.columns([1, 4])
        predict_clicked = False
        with col_pred:
            if st.button("Run Scan 🛡️", width="stretch"):
                predict_clicked = True
        with col_reset:
            if st.button("Reset Scanner 🔄", width="stretch"):
                st.session_state["detector_text_area"] = ""
                st.session_state["detector_subject"] = ""
                st.session_state["detector_file_uploader"] = None
                st.rerun()

        if predict_clicked:
            if not email_text.strip():
                st.warning("⚠️ Scanner warning: Please insert or upload email content before running.")
                return

            with st.spinner("Analyzing email patterns and searching metadata signatures..."):
                result = detector.predict(email_text, subject=email_subject or None)

            if result.get("error"):
                st.error(result["error"])
                return

            add_prediction(result)
            st.toast(f"Scan complete — {result['verdict']} detected", icon="🛡️")

            st.markdown("---")
            st.markdown("## 🛡️ Threat Assessment Report")

            col_res, col_gauge = st.columns([3, 2])

            with col_res:
                verdict = result["verdict"]
                risk_level = result["risk_level"]
                risk_score = result["risk_score"]
                confidence = result["confidence"]

                verdict_style = "phishing" if verdict == "Phishing" else "legitimate"
                st.markdown(
                    f"""
                    <div class="cyber-card cyber-card-{verdict_style}">
                        <h3 style="margin-top:0; color: {colors[verdict_style]};">
                            Verdict: {verdict.upper()}
                        </h3>
                        <p style="font-size:1.1rem; margin-bottom: 5px;">
                            <strong>Threat Risk Rating:</strong> {risk_level} ({risk_score}/100)
                        </p>
                        <p style="font-size:1.1rem; margin-bottom: 5px;">
                            <strong>Model Confidence:</strong> {confidence * 100:.2f}%
                        </p>
                        <p style="font-size:0.9rem; margin-bottom: 0; opacity: 0.8;">
                            <strong>Model:</strong> {result.get('model_name', 'N/A')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                explanation_card(result["explanation"], type=verdict_style, title="Security Analysis Explanation")
                explanation_card(
                    result["recommendation"],
                    type="warning" if verdict == "Phishing" else "legitimate",
                    title="Actionable Recommendation",
                )

            with col_gauge:
                _render_gauge(risk_score, verdict, colors)

            st.markdown("---")

            tab_kw, tab_links, tab_report = st.tabs([
                "📝 Highlighted Text", "🔗 Detected URLs", "📥 Export Report",
            ])

            with tab_kw:
                st.markdown("### Suspicious Terms Highlighted")
                keywords = result["suspicious_keywords"]
                if keywords:
                    st.info(
                        f"The system flagged {len(keywords)} suspicious term(s) "
                        f"commonly associated with social engineering tactics."
                    )
                    highlighted_html = highlight_phish_keywords(email_text, keywords)
                    st.markdown(
                        f"""
                        <div style="background-color: {colors['card']}; border: 1px solid rgba(0, 212, 255, 0.1);
                                    padding: 15px; border-radius: 8px; font-family: monospace;
                                    max-height: 400px; overflow-y: auto;">
                            {highlighted_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("No suspicious social-engineering keywords detected in the message corpus.")

            with tab_links:
                st.markdown("### Extracted Links Summary")
                urls = result["detected_urls"]
                if urls:
                    st.warning(
                        f"Audited email contains {len(urls)} live URL link(s). "
                        f"Avoid clicking unverified links."
                    )
                    for idx, url in enumerate(urls):
                        st.code(f"[{idx + 1}] {url}", language="html")
                else:
                    st.success("No hyperlink URLs detected in this email content.")

            with tab_report:
                st.markdown("### Download Prediction Log")
                report_text = detector.generate_report(result)
                st.code(report_text, language="text")

                st.download_button(
                    label="Download Security Scan Report (.txt)",
                    data=report_text,
                    file_name=f"phishing_audit_report_{result['timestamp'].replace(' ', '_').replace(':', '-')}.txt",
                    mime="text/plain",
                    width="stretch",
                )

    else:
        st.markdown("### 📄 Batch Scan from CSV")
        st.caption(
            "Upload a CSV with an email-text column (default: `text_combined`). "
            "Each row is scanned and the results are appended."
        )

        uploaded_csv = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="detector_csv_uploader",
        )
        text_column = st.text_input("Text column", value="text_combined", key="detector_csv_col")
        subject_column = st.text_input(
            "Subject column (optional)", value="", key="detector_csv_subject_col"
        )

        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not parse CSV: {e}")
                return

            if text_column not in df.columns:
                st.warning(
                    f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
                )
                return

            if st.button("Run Batch Scan 🛡️", key="batch_run", width="stretch"):
                with st.spinner(f"Scanning {len(df):,} emails..."):
                    result_df = detector.predict_dataframe(
                        df,
                        text_col=text_column,
                        subject_col=subject_column if subject_column in df.columns else None,
                    )

                st.success(f"Batch scan complete — {len(result_df):,} emails analysed.")
                st.dataframe(result_df, width="stretch")

                phish_count = int(result_df["label"].sum())
                st.metric("Phishing Detected", f"{phish_count:,}", f"{phish_count / len(result_df):.1%}")

                st.download_button(
                    "Download Results CSV 📥",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_scan_results.csv",
                    mime="text/csv",
                    width="stretch",
                )
