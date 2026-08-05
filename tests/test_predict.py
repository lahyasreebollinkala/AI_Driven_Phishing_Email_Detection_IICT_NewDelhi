"""Tests for src/predict.py — PhishingDetector analysis helpers and inference."""

import os

import numpy as np
import pandas as pd
import pytest

from src.config import BEST_MODEL_PATH
from src.predict import PhishingDetector


@pytest.fixture(scope="module")
def detector():
    return PhishingDetector()


def test_available_models_returns_list(detector):
    models = detector.available_models()
    assert isinstance(models, list)
    assert all(isinstance(m, str) for m in models)


def test_available_models_excludes_artifacts():
    models = PhishingDetector.available_models()
    assert "Tfidf Vectorizer" not in models
    assert "Metadata Scaler" not in models


def test_get_suspicious_keywords(detector):
    found = detector.get_suspicious_keywords(
        "URGENT: verify your account or your password will expire"
    )
    assert "urgent" in found
    assert "verify" in found
    assert "password" in found


def test_get_suspicious_keywords_case_insensitive(detector):
    assert detector.get_suspicious_keywords("LOGIN NOW") == ["login"]


def test_get_detected_urls(detector):
    urls = detector.get_detected_urls("Go to https://evil.com/steal now or www.fake.net")
    assert len(urls) == 2


def test_calculate_risk_score_bounds(detector):
    assert detector.calculate_risk_score(1.0, 10, 20) == 100
    assert detector.calculate_risk_score(0.0, 0, 0) == 0


def test_calculate_risk_score_monotonic(detector):
    low = detector.calculate_risk_score(0.2, 0, 0)
    high = detector.calculate_risk_score(0.9, 5, 10)
    assert high > low


def test_get_risk_level_thresholds(detector):
    assert detector.get_risk_level(0) == "Low"
    assert detector.get_risk_level(45) == "Medium"
    assert detector.get_risk_level(70) == "High"
    assert detector.get_risk_level(95) == "Critical"


def test_generate_explanation_legitimate(detector):
    explanation = detector.generate_explanation(False, [], [], 0.05)
    assert "legitimate" in explanation.lower()


def test_generate_explanation_phishing_with_indicators(detector):
    explanation = detector.generate_explanation(True, ["urgent"], ["http://x.io"], 0.95)
    assert "urgent" in explanation
    assert "URL" in explanation


def test_generate_recommendation_critical(detector):
    rec = detector.generate_recommendation(True, "Critical")
    assert "CRITICAL" in rec
    assert "password" in rec.lower()


def test_generate_report_contains_fields(detector):
    result = {
        "timestamp": "2026-01-01 00:00:00",
        "model_name": "Test Model",
        "verdict": "Phishing",
        "confidence": 0.94,
        "risk_score": 85,
        "risk_level": "Critical",
        "suspicious_keywords": ["urgent"],
        "detected_urls": ["http://evil.com"],
        "explanation": "**Highly confident** this is phishing.",
        "recommendation": "Do not click.",
    }
    report = detector.generate_report(result)
    assert "Phishing" in report
    assert "http://evil.com" in report
    assert "**" not in report


def test_parse_eml_multipart():
    eml = (
        "Subject: Confirm your account\n"
        "From: attacker@evil.com\n"
        "Content-Type: multipart/alternative; boundary=b\n"
        "\n"
        "--b\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        "Plain body here\n"
        "--b\n"
        "Content-Type: text/html\n"
        "\n"
        "<html><body><b>HTML</b> body</body></html>\n"
        "--b--\n"
    )
    parsed = PhishingDetector._parse_eml_text(eml)
    assert "Subject: Confirm your account" in parsed
    assert "HTML" in parsed
    assert "Plain body here" in parsed


def test_predict_from_file_empty_error(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    result = PhishingDetector().predict_from_file(str(empty))
    assert result["verdict"] == "Unknown"


@pytest.mark.skipif(not os.path.exists(BEST_MODEL_PATH), reason="No trained model artifact")
def test_end_to_end_predict_detector(detector):
    assert detector.is_loaded, "Best model should be loadable"
    result = detector.predict(
        "URGENT: Your account has been suspended. Verify your password immediately "
        "by clicking http://evil.example.com/login or your account will be closed."
    )
    assert result["verdict"] in ("Phishing", "Legitimate")
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in ("Low", "Medium", "High", "Critical")


@pytest.mark.skipif(not os.path.exists(BEST_MODEL_PATH), reason="No trained model artifact")
def test_predict_dataframe_columns(detector):
    df = pd.DataFrame(
        {
            "text_combined": [
                "URGENT verify your account",
                "Meeting at 3pm to discuss the quarterly report.",
            ],
            "subject": ["Action required", "Internal note"],
        }
    )
    out = detector.predict_dataframe(df)
    for col in ("verdict", "label", "confidence", "risk_score", "risk_level"):
        assert col in out.columns
    assert len(out) == 2


@pytest.mark.skipif(not os.path.exists(BEST_MODEL_PATH), reason="No trained model artifact")
def test_predict_batch_length_matches(detector):
    texts = ["hello", "urgent verify now", "plain email body"]
    results = detector.predict_batch(texts)
    assert len(results) == len(texts)
    assert all("verdict" in r for r in results)
