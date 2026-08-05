# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Prediction Engine Module
# =============================================================================
"""
Production inference engine for phishing email detection.

Loads a trained model, TF-IDF vectorizer, and metadata scaler from disk and
provides a simple API for classifying new emails.

Features:
    - Single email prediction with confidence scores
    - Batch prediction from a DataFrame / list of emails
    - Model selection among all saved classifiers (default: best model)
    - Suspicious keyword highlighting
    - URL detection
    - Risk score calculation (0-100)
    - Natural-language explanation of why an email is phishing
    - File upload support (.txt, .eml)
    - Prediction report generation

Usage:
    detector = PhishingDetector()
    result = detector.predict("Click here to verify your account...")
    print(result["verdict"])       # "Phishing"
    print(result["confidence"])    # 0.94
    print(result["risk_score"])    # 87
"""

from __future__ import annotations

import email
import glob
import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.config import (
    APP_VERSION,
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    FEATURE_NAMES_PATH,
    LABEL_MAP,
    METADATA_SCALER_PATH,
    MODELS_DIR,
    RISK_THRESHOLDS,
    SUSPICIOUS_KEYWORDS,
    TFIDF_VECTORIZER_PATH,
    model_file_path,
)
from src.feature_engineering import FeatureEngineer
from src.preprocessing import TextPreprocessor
from src.utils import load_artifact, load_json, setup_logger

# Precompiled regex for URL detection in raw text
_URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)


class PhishingDetector:
    """
    Production-ready phishing email detection engine.

    Loads pre-trained artifacts on initialization and provides
    methods for classifying emails with detailed explanations.

    Attributes:
        model: The loaded classifier.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        scaler (StandardScaler): Fitted metadata scaler.
        preprocessor (TextPreprocessor): NLP cleaning pipeline.
        model_info (dict): Metadata about the loaded model.
        is_loaded (bool): Whether all artifacts loaded successfully.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.logger = setup_logger("PhishingDetector")
        self.preprocessor = None
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.model_info: dict | None = None
        self.is_loaded = False
        self.model_name = model_name

        # The NLP preprocessor is a hard dependency for prediction. If NLTK
        # data is unavailable on a fresh deployment it must degrade gracefully
        # instead of crashing the app at construction time.
        try:
            self.preprocessor = TextPreprocessor()
        except Exception as e:  # noqa: BLE001
            self.logger.error("Failed to initialize TextPreprocessor: %s", e)

        self._load_artifacts(model_name)

    # =========================================================================
    # ARTIFACT LOADING
    # =========================================================================

    @staticmethod
    def available_models() -> list[str]:
        """
        List display names of all classifiers saved in models/ (excluding
        vectorizer/scaler/feature artifacts).

        Returns:
            list[str]: Sorted model display names.
        """
        pattern = os.path.join(MODELS_DIR, "*.joblib")
        excluded = {"tfidf_vectorizer", "metadata_scaler", "feature_names"}
        names = []
        for filepath in glob.glob(pattern):
            base = os.path.splitext(os.path.basename(filepath))[0]
            if base in excluded:
                continue
            names.append(base.replace("_", " ").title())
        return sorted(names)

    def _load_artifacts(self, model_name: str | None = None) -> None:
        """Load all required model artifacts from disk."""
        try:
            if model_name:
                model_path = model_file_path(model_name)
            else:
                model_path = BEST_MODEL_PATH

            self.model = load_artifact(model_path)
            self.vectorizer = load_artifact(TFIDF_VECTORIZER_PATH)
            self.scaler = load_artifact(METADATA_SCALER_PATH)

            try:
                self.model_info = load_json(BEST_MODEL_INFO_PATH)
            except FileNotFoundError:
                self.model_info = {"model_name": model_name or "Best Model"}

            self.is_loaded = True
            self.logger.info(
                "PhishingDetector loaded successfully. Model: %s",
                self.model_info.get("model_name", "Unknown"),
            )
        except FileNotFoundError as e:
            self.logger.error(
                "Failed to load model artifacts: %s. "
                "Ensure the trained artifacts exist in %s "
                "(run main.py to train them). Model path tried: %s",
                e, MODELS_DIR, model_path,
            )
            self.is_loaded = False
        except Exception as e:  # noqa: BLE001
            self.logger.error(
                "Unexpected error loading artifacts (model=%s, vectorizer=%s, scaler=%s): %s",
                model_path, TFIDF_VECTORIZER_PATH, METADATA_SCALER_PATH, e,
            )
            self.is_loaded = False

    def _extract_features(self, raw_text: str, cleaned_text: str, subject: str = "") -> csr_matrix:
        """
        Build the feature vector for a single email.

        Combines TF-IDF features from cleaned text with metadata features from
        raw text. ``subject`` is passed through so metadata features such as
        ``has_subject`` and ``subject_length`` are computed correctly.

        Args:
            raw_text (str): Original email text.
            cleaned_text (str): NLP-cleaned text.
            subject (str, optional): Original email subject.

        Returns:
            scipy.sparse.csr_matrix: Feature vector.
        """
        # TF-IDF features
        tfidf_vector = self.vectorizer.transform([cleaned_text])

        # Metadata features — same vectorized extractor used at training time
        metadata = FeatureEngineer._extract_row_features(raw_text, subject).reshape(1, -1)
        metadata_scaled = self.scaler.transform(metadata)

        combined = hstack([tfidf_vector, csr_matrix(metadata_scaled)])

        # Handle Naive Bayes (clip negative values)
        model_name = (self.model_info or {}).get("model_name", "") or self.model_name or ""
        if "Naive Bayes" in model_name:
            combined.data = np.maximum(combined.data, 0)

        return combined

    # =========================================================================
    # ANALYSIS HELPERS
    # =========================================================================

    def get_suspicious_keywords(self, text: str) -> list[str]:
        """
        Find which suspicious keywords appear in the email text.

        Args:
            text (str): Raw email text.

        Returns:
            list[str]: Matched suspicious keywords.
        """
        text_lower = text.lower()
        return [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]

    def get_detected_urls(self, text: str) -> list[str]:
        """
        Extract all URLs from the email text.

        Args:
            text (str): Raw email text.

        Returns:
            list[str]: List of detected URLs.
        """
        return _URL_RE.findall(text)

    def calculate_risk_score(self, confidence: float, url_count: int, keyword_count: int) -> int:
        """
        Calculate a risk score from 0 to 100.

        The risk score is a weighted combination of:
            - Model confidence (60% weight)
            - Number of suspicious keywords (25% weight)
            - Number of detected URLs (15% weight)

        Args:
            confidence (float): Model's phishing probability (0-1).
            url_count (int): Number of URLs found.
            keyword_count (int): Number of suspicious keywords found.

        Returns:
            int: Risk score from 0 to 100.
        """
        confidence_score = confidence * 60
        keyword_score = min(keyword_count * 3, 25)
        url_score = min(url_count * 5, 15)

        total = confidence_score + keyword_score + url_score
        return min(int(round(total)), 100)

    def get_risk_level(self, risk_score: int) -> str:
        """
        Convert a numeric risk score to a human-readable level.

        Args:
            risk_score (int): Risk score (0-100).

        Returns:
            str: Risk level ('Low', 'Medium', 'High', or 'Critical').
        """
        if risk_score <= RISK_THRESHOLDS["low"]:
            return "Low"
        if risk_score <= RISK_THRESHOLDS["medium"]:
            return "Medium"
        if risk_score <= RISK_THRESHOLDS["high"]:
            return "High"
        return "Critical"

    def generate_explanation(self, is_phishing: bool, keywords: list[str], urls: list[str], confidence: float) -> str:
        """
        Generate a natural-language explanation for the prediction.

        Args:
            is_phishing (bool): Whether the email was classified as phishing.
            keywords (list[str]): Suspicious keywords found.
            urls (list[str]): URLs detected.
            confidence (float): Model confidence.

        Returns:
            str: Human-readable explanation.
        """
        if not is_phishing:
            explanation = (
                "This email appears to be **legitimate**. "
                "The content does not exhibit typical phishing characteristics. "
            )
            if keywords:
                explanation += (
                    f"Although {len(keywords)} keyword(s) were flagged "
                    f"({', '.join(keywords[:3])}), the overall text pattern "
                    f"is consistent with genuine communication."
                )
            else:
                explanation += "No suspicious keywords or deceptive patterns were detected."
            return explanation

        reasons = []

        if confidence > 0.9:
            reasons.append(
                f"The model is **highly confident** ({confidence:.1%}) that this "
                f"is a phishing email."
            )
        elif confidence > 0.7:
            reasons.append(
                f"The model is **moderately confident** ({confidence:.1%}) that this "
                f"is a phishing attempt."
            )
        else:
            reasons.append(
                f"The model detected phishing indicators with **{confidence:.1%}** confidence."
            )

        if keywords:
            kw_display = ", ".join(f"'{kw}'" for kw in keywords[:5])
            reasons.append(
                f"**Suspicious keywords** detected: {kw_display}. "
                f"These terms are commonly used in social engineering attacks."
            )

        if urls:
            reasons.append(
                f"**{len(urls)} URL(s)** were found. Phishing emails often contain "
                f"links to fake websites designed to steal credentials."
            )

        if not keywords and not urls:
            reasons.append(
                "The text patterns and linguistic structure match known "
                "phishing email templates."
            )

        return " ".join(reasons)

    def generate_recommendation(self, is_phishing: bool, risk_level: str) -> str:
        """
        Generate safety recommendations based on the prediction.

        Args:
            is_phishing (bool): Whether the email is phishing.
            risk_level (str): Risk level string.

        Returns:
            str: Actionable recommendation.
        """
        if not is_phishing:
            return (
                "This email appears safe. However, always exercise caution "
                "with unexpected emails, especially those containing links or "
                "requesting personal information."
            )

        recommendations = {
            "Low": (
                "This email shows minor phishing indicators. "
                "Verify the sender's address and avoid clicking any links "
                "until you confirm the email's authenticity."
            ),
            "Medium": (
                "This email has several phishing characteristics. "
                "Do NOT click any links or download attachments. "
                "Contact the supposed sender through official channels to verify."
            ),
            "High": (
                "This email is likely a phishing attempt. "
                "Do NOT interact with it. Report it to your IT security team. "
                "Delete the email and block the sender."
            ),
            "Critical": (
                "CRITICAL: This email is almost certainly a phishing attack. "
                "Do NOT click any links, open attachments, or reply. "
                "Report immediately to your security team. "
                "If you've already clicked any links, change your passwords immediately."
            ),
        }
        return recommendations.get(risk_level, recommendations["Medium"])

    # =========================================================================
    # SINGLE PREDICTION
    # =========================================================================

    def predict(self, email_text: str, subject: str | None = None) -> dict:
        """
        Classify an email as phishing or legitimate with full analysis.

        Args:
            email_text (str): Raw email text content.
            subject (str, optional): Email subject line.

        Returns:
            dict: Complete prediction result.
        """
        if not self.is_loaded:
            return {
                "error": "Model not loaded. Run main.py to train models first.",
                "verdict": "Unknown",
                "label": -1,
                "confidence": 0.0,
                "risk_score": 0,
                "risk_level": "Unknown",
                "suspicious_keywords": [],
                "detected_urls": [],
                "recommendation": "Model not available.",
                "explanation": "The model artifacts are not loaded.",
                "model_name": "None",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        if self.preprocessor is None:
            return {
                "error": "NLP preprocessor unavailable (NLTK data could not be loaded).",
                "verdict": "Unknown",
                "label": -1,
                "confidence": 0.0,
                "risk_score": 0,
                "risk_level": "Unknown",
                "suspicious_keywords": [],
                "detected_urls": [],
                "recommendation": "Prediction not available.",
                "explanation": "The NLP cleaning pipeline could not be initialised.",
                "model_name": "None",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        # Clean the text
        cleaned = self.preprocessor.clean_text(email_text)

        # Extract features
        features = self._extract_features(email_text, cleaned, subject or "")

        # Get prediction
        prediction = self.model.predict(features)[0]

        # Get confidence
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            confidence = float(proba[1])  # Probability of being phishing
        elif hasattr(self.model, "decision_function"):
            score = self.model.decision_function(features)[0]
            confidence = float(1 / (1 + np.exp(-score)))  # sigmoid
        else:
            confidence = float(prediction)

        is_phishing = int(prediction) == 1

        # If classified as legitimate, show confidence in the legitimate class
        display_confidence = confidence if is_phishing else (1 - confidence)

        keywords = self.get_suspicious_keywords(email_text)
        urls = self.get_detected_urls(email_text)

        risk_score = self.calculate_risk_score(confidence, len(urls), len(keywords))
        risk_level = self.get_risk_level(risk_score)

        explanation = self.generate_explanation(is_phishing, keywords, urls, confidence)
        recommendation = self.generate_recommendation(is_phishing, risk_level)

        result = {
            "verdict": LABEL_MAP[int(prediction)],
            "label": int(prediction),
            "confidence": round(display_confidence, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "suspicious_keywords": keywords,
            "detected_urls": urls,
            "recommendation": recommendation,
            "explanation": explanation,
            "model_name": (self.model_info or {}).get("model_name", "Unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.logger.info(
            "Prediction: %s (confidence=%.2f%%, risk=%d)",
            result["verdict"], result["confidence"] * 100, result["risk_score"],
        )

        return result

    # =========================================================================
    # BATCH PREDICTION
    # =========================================================================

    def predict_batch(self, texts: list[str] | pd.Series, subjects: list[str] | None = None) -> list[dict]:
        """
        Classify multiple emails at once.

        Args:
            texts (list[str] | pd.Series): Raw email texts.
            subjects (list[str] | None, optional): Parallel list of subjects.

        Returns:
            list[dict]: One prediction result per input email.
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        texts = [t if isinstance(t, str) else str(t) for t in texts]

        subjects_list: list[str]
        if subjects is None:
            subjects_list = [""] * len(texts)
        elif isinstance(subjects, pd.Series):
            subjects_list = subjects.tolist()
        else:
            subjects_list = list(subjects)

        return [
            self.predict(text, subject)
            for text, subject in zip(texts, subjects_list)
        ]

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "text_combined", subject_col: str | None = "subject") -> pd.DataFrame:
        """
        Predict on every row of a DataFrame and return a results DataFrame.

        Args:
            df (pd.DataFrame): Input emails.
            text_col (str): Column containing email text.
            subject_col (str | None): Optional column containing subjects.

        Returns:
            pd.DataFrame: Input DataFrame plus verdict/confidence/risk columns.
        """
        subjects = df[subject_col] if subject_col and subject_col in df.columns else None
        results = self.predict_batch(df[text_col].tolist(), subjects.tolist() if subjects is not None else None)

        out = df.copy()
        out["verdict"] = [r["verdict"] for r in results]
        out["label"] = [r["label"] for r in results]
        out["confidence"] = [r["confidence"] for r in results]
        out["risk_score"] = [r["risk_score"] for r in results]
        out["risk_level"] = [r["risk_level"] for r in results]
        return out

    # =========================================================================
    # FILE PREDICTION
    # =========================================================================

    @staticmethod
    def _parse_eml_text(text: str) -> str:
        """
        Parse .eml content (headers + body) and return combined subject/body.

        Args:
            text (str): Raw .eml source.

        Returns:
            str: Extracted email text (subject + body).
        """
        msg = email.message_from_string(text)
        parts = []

        subject = msg.get("Subject", "")
        if subject:
            parts.append(f"Subject: {subject}")

        def _decode_payload(payload) -> str:
            if payload is None:
                return ""
            try:
                return payload.decode("utf-8", errors="ignore")
            except AttributeError:
                return str(payload)

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    parts.append(_decode_payload(part.get_payload(decode=True)))
                elif content_type == "text/html":
                    from bs4 import BeautifulSoup
                    html_text = _decode_payload(part.get_payload(decode=True))
                    try:
                        soup = BeautifulSoup(html_text, "lxml")
                        parts.append(soup.get_text(separator=" "))
                    except Exception:  # noqa: BLE001
                        parts.append(re.sub(r"<[^>]+>", " ", html_text))
        else:
            parts.append(_decode_payload(msg.get_payload(decode=True)))

        return " ".join(parts)

    def predict_from_file(self, filepath: str) -> dict:
        """
        Read an email from a .txt or .eml file and classify it.

        Args:
            filepath (str): Path to the email file.

        Returns:
            dict: Prediction result (same format as predict()).
        """
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == ".eml":
                text = self._parse_eml_text(self._read_text(filepath))
            else:
                text = self._read_text(filepath)

            if not text.strip():
                return {"error": "File is empty.", "verdict": "Unknown"}

            return self.predict(text)

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error reading file %s: %s", filepath, e)
            return {"error": str(e), "verdict": "Unknown"}

    @staticmethod
    def _read_text(filepath: str) -> str:
        """Read a text file, tolerating encoding errors."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def predict_from_text_bytes(self, content_bytes: bytes, filename: str = "uploaded") -> dict:
        """
        Predict from uploaded file content bytes.

        Used by the Streamlit app for file upload handling.

        Args:
            content_bytes (bytes): Raw file content.
            filename (str): Original filename (for extension detection).

        Returns:
            dict: Prediction result.
        """
        ext = os.path.splitext(filename)[1].lower()

        try:
            text = content_bytes.decode("utf-8", errors="ignore")

            if ext == ".eml":
                text = self._parse_eml_text(text)

            if not text.strip():
                return {"error": "File content is empty.", "verdict": "Unknown"}

            return self.predict(text)

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error processing uploaded file: %s", e)
            return {"error": str(e), "verdict": "Unknown"}

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_report(self, result: dict) -> str:
        """
        Generate a downloadable text report of the prediction.

        Args:
            result (dict): Prediction result from predict().

        Returns:
            str: Formatted report text.
        """
        report = []
        report.append("=" * 60)
        report.append("   AI PHISHING SHIELD — PREDICTION REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"  Date/Time    : {result.get('timestamp', 'N/A')}")
        report.append(f"  Model Used   : {result.get('model_name', 'N/A')}")
        report.append("")
        report.append("-" * 60)
        report.append("  VERDICT")
        report.append("-" * 60)
        report.append(f"  Classification : {result.get('verdict', 'N/A')}")
        report.append(f"  Confidence     : {result.get('confidence', 0):.2%}")
        report.append(f"  Risk Score     : {result.get('risk_score', 0)}/100")
        report.append(f"  Risk Level     : {result.get('risk_level', 'N/A')}")
        report.append("")
        report.append("-" * 60)
        report.append("  ANALYSIS DETAILS")
        report.append("-" * 60)

        keywords = result.get("suspicious_keywords", [])
        report.append(f"  Suspicious Keywords ({len(keywords)}):")
        if keywords:
            for kw in keywords:
                report.append(f"    • {kw}")
        else:
            report.append("    None detected")

        urls = result.get("detected_urls", [])
        report.append(f"\n  Detected URLs ({len(urls)}):")
        if urls:
            for url in urls:
                report.append(f"    • {url}")
        else:
            report.append("    None detected")

        report.append("")
        report.append("-" * 60)
        report.append("  EXPLANATION")
        report.append("-" * 60)
        explanation = result.get("explanation", "N/A")
        # Remove markdown bold markers for plain text
        explanation = explanation.replace("**", "")
        report.append(f"  {explanation}")

        report.append("")
        report.append("-" * 60)
        report.append("  RECOMMENDATION")
        report.append("-" * 60)
        recommendation = result.get("recommendation", "N/A")
        report.append(f"  {recommendation}")

        report.append("")
        report.append("=" * 60)
        report.append(f"  Generated by AI Phishing Shield v{APP_VERSION}")
        report.append("=" * 60)

        return "\n".join(report)
