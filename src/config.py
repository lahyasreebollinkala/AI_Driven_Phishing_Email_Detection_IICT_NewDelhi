# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Configuration Module
# =============================================================================
"""
Central configuration for the entire phishing detection pipeline.

Contains all file paths, model hyperparameters, NLP constants, feature
engineering settings, and application-level constants. Every other module
imports from here — no magic numbers or hardcoded paths elsewhere.

Environment variables from a root-level ``.env`` file are honoured (see
``.env.example``). Values are read once at import time with sane defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# =============================================================================
# 0. ENVIRONMENT CONFIGURATION
# =============================================================================

# Project root is one level up from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from the project root (no-op if the file does not exist).
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# =============================================================================
# 1. PROJECT ROOT & DIRECTORY PATHS
# =============================================================================

# Dataset directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")

# Output directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# =============================================================================
# 2. RAW DATASET FILE PATHS
# =============================================================================

RAW_DATASETS = {
    "phishing_email": os.path.join(RAW_DATA_DIR, "phishing_email.csv"),
    "spamassassin": os.path.join(RAW_DATA_DIR, "SpamAssasin.csv"),
    "enron": os.path.join(RAW_DATA_DIR, "Enron.csv"),
    "ceas_08": os.path.join(RAW_DATA_DIR, "CEAS_08.csv"),
    "ling": os.path.join(RAW_DATA_DIR, "Ling.csv"),
    "nazario": os.path.join(RAW_DATA_DIR, "Nazario.csv"),
    "nigerian_fraud": os.path.join(RAW_DATA_DIR, "Nigerian_Fraud.csv"),
}

# =============================================================================
# 3. PROCESSED DATA FILE PATHS
# =============================================================================

MERGED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "merged_dataset.csv")
CLEANED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.csv")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train_data.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_data.csv")

# =============================================================================
# 4. MODEL ARTIFACT PATHS
# =============================================================================

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METADATA_SCALER_PATH = os.path.join(MODELS_DIR, "metadata_scaler.joblib")
MODEL_RESULTS_PATH = os.path.join(REPORTS_DIR, "model_results.json")
COMPARISON_TABLE_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")
BEST_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "best_model_info.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")
DATASET_STATS_PATH = os.path.join(REPORTS_DIR, "dataset_stats.json")

# =============================================================================
# 5. LABEL MAPPING
# =============================================================================

LABEL_MAP = {
    0: "Legitimate",
    1: "Phishing",
}

LABEL_REVERSE_MAP = {
    "Legitimate": 0,
    "Phishing": 1,
}

# =============================================================================
# 6. TRAIN/TEST SPLIT SETTINGS
# =============================================================================

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))     # 80/20 train-test split
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))  # Reproducibility seed
STRATIFY = os.getenv("STRATIFY", "true").lower() in ("1", "true", "yes")

# =============================================================================
# 7. NLP PREPROCESSING SETTINGS
# =============================================================================

# Maximum text length to process (characters) — truncate extremely long emails
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 50000))

# Minimum text length — discard emails shorter than this
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", 10))

# Reuse the previously-generated cleaned dataset instead of re-cleaning raw
# text. Speeds up repeated training runs dramatically.
USE_CACHED_CLEANED_DATA = os.getenv("USE_CACHED_CLEANED_DATA", "true").lower() in ("1", "true", "yes")

# =============================================================================
# 8. TF-IDF VECTORIZER SETTINGS
# =============================================================================

TFIDF_CONFIG = {
    "max_features": int(os.getenv("TFIDF_MAX_FEATURES", 10000)),  # Top-N terms
    "ngram_range": (1, 2),                                        # Uni + bigrams
    "min_df": 3,                # Ignore terms in fewer than 3 documents
    "max_df": 0.95,             # Ignore terms in >95% of documents
    "sublinear_tf": True,       # Apply sublinear TF scaling (1 + log(tf))
    "strip_accents": "unicode",  # Strip accents during preprocessing
}

# =============================================================================
# 9. SUSPICIOUS KEYWORDS LIST
# =============================================================================
# Common phishing keywords used for feature engineering. Curated from
# cybersecurity research on social engineering tactics.

SUSPICIOUS_KEYWORDS = [
    # Urgency & Action
    "urgent", "immediately", "act now", "action required", "expire",
    "suspended", "limited time", "deadline", "asap", "hurry",
    "time sensitive", "final warning", "last chance",

    # Account & Security
    "verify", "verify your account", "confirm your identity",
    "update your information", "password", "login", "credentials",
    "security alert", "unauthorized", "suspicious activity",
    "account suspended", "account locked", "unusual activity",
    "security update", "reset password",

    # Financial
    "bank", "credit card", "payment", "invoice", "transaction",
    "refund", "wire transfer", "billing", "prize", "winner",
    "lottery", "inheritance", "million dollars", "beneficiary",
    "claim your", "reward",

    # Call to Action
    "click here", "click below", "click the link", "open attachment",
    "download", "sign in", "log in", "submit", "apply now",

    # Authority Impersonation
    "official", "government", "irs", "fbi", "microsoft",
    "apple", "paypal", "amazon", "netflix", "google",
    "tech support", "customer service", "helpdesk",

    # Emotional Manipulation
    "congratulations", "you have been selected", "dear customer",
    "dear user", "dear friend", "valued customer",
    "important notice", "attention", "warning",

    # Nigerian/419 Scam
    "barrister", "diplomat", "consignment", "trunk box",
    "next of kin", "dying", "widow", "orphan",
    "confidential", "business proposal", "partnership",
]

# =============================================================================
# 10. MODEL HYPERPARAMETERS
# =============================================================================
# ``class_weight="balanced"``/``scale_pos_weight`` are used to counter the
# mild class imbalance in the merged corpus.

# Fallback ratio used if imbalance cannot be measured at runtime.
DEFAULT_POSITIVE_RATIO = 1.0

MODEL_CONFIGS = {
    "Logistic Regression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "class_weight": "balanced",
        },
    },
    "Naive Bayes": {
        "class": "sklearn.naive_bayes.MultinomialNB",
        "params": {
            "alpha": 0.1,
        },
    },
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_estimators": 200,
            "max_depth": 50,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "class_weight": "balanced",
        },
    },
    "XGBoost": {
        "class": "xgboost.XGBClassifier",
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "n_jobs": -1,
        },
    },
    "LightGBM": {
        "class": "lightgbm.LGBMClassifier",
        "params": {
            "n_estimators": 200,
            "max_depth": 15,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        },
    },
    "Neural Network": {
        "class": "sklearn.neural_network.MLPClassifier",
        "params": {
            "hidden_layer_sizes": (256, 128, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": RANDOM_STATE,
            "batch_size": 256,
        },
    },
}

# =============================================================================
# 11. VISUALIZATION SETTINGS
# =============================================================================

# Color palette for the cybersecurity theme
COLORS = {
    "primary": "#00d4ff",        # Cyan
    "secondary": "#7c3aed",      # Purple
    "success": "#10b981",        # Green
    "danger": "#ef4444",         # Red
    "warning": "#f59e0b",        # Amber
    "info": "#3b82f6",           # Blue
    "background": "#0a0e27",     # Deep navy
    "card": "#1a1f3a",           # Card background
    "text": "#e2e8f0",           # Light gray text
    "phishing": "#ef4444",       # Red for phishing
    "legitimate": "#10b981",     # Green for legitimate
}

# Plotly color sequence for model comparison charts
MODEL_COLORS = [
    "#00d4ff",  # Cyan
    "#7c3aed",  # Purple
    "#10b981",  # Green
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#3b82f6",  # Blue
]

# =============================================================================
# 12. APPLICATION SETTINGS
# =============================================================================

APP_TITLE = "AI Phishing Shield"
APP_SUBTITLE = "AI-Driven Phishing Email Detection Using NLP"
APP_ICON = "🛡️"
APP_VERSION = "2.0.0"

# Maximum number of predictions to keep in history
MAX_PREDICTION_HISTORY = int(os.getenv("MAX_PREDICTION_HISTORY", 50))

# Risk score thresholds
RISK_THRESHOLDS = {
    "low": 30,       # 0-30: Low risk
    "medium": 60,    # 31-60: Medium risk
    "high": 80,      # 61-80: High risk
    "critical": 100, # 81-100: Critical risk
}

# =============================================================================
# 13. LOGGING SETTINGS
# =============================================================================

LOG_FILE = os.path.join(LOGS_DIR, "phishing_detection.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# =============================================================================
# 14. API SETTINGS
# =============================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# =============================================================================
# 15. SERVER / STREAMLIT SETTINGS
# =============================================================================

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))


def model_file_path(model_name: str) -> str:
    """
    Return the on-disk path for a saved model by display name.

    Args:
        model_name (str): Display name (e.g. "Random Forest").

    Returns:
        str: Absolute path to the ``.joblib`` artifact.
    """
    safe_name = model_name.lower().replace(" ", "_")
    return os.path.join(MODELS_DIR, f"{safe_name}.joblib")
