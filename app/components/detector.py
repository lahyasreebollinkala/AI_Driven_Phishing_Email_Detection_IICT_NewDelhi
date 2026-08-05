# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Cached Detector Provider
# =============================================================================
"""
Provides a cached, app-wide PhishingDetector instance.

Instantiating a PhishingDetector loads several large joblib artifacts and
constructs an NLTK preprocessor, which is wasteful to repeat on every
Streamlit rerun. ``@st.cache_resource`` ensures the detector (or a small
pool keyed by model name) is created only once per process.
"""

import streamlit as st

from src.predict import PhishingDetector


@st.cache_resource(show_spinner="Loading detection engine...")
def _get_detector(model_name: str | None) -> PhishingDetector:
    """Create and cache a PhishingDetector for a given model name."""
    return PhishingDetector(model_name=model_name)


def get_detector(model_name: str | None = None) -> PhishingDetector:
    """
    Return a cached PhishingDetector instance.

    Args:
        model_name (str | None, optional): Saved model to load. ``None`` uses
            the best model.

    Returns:
        PhishingDetector: Loaded detector (``is_loaded`` may be False if
            artifacts are missing).
    """
    return _get_detector(model_name)


def clear_detector_cache() -> None:
    """Drop the cached detectors (call after retraining the pipeline)."""
    _get_detector.clear()
