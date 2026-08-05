# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Feature Engineering Module
# =============================================================================
"""
Extracts and combines two types of features for model training:

1. **TF-IDF Features**: Bag-of-words + bigram representation of cleaned text
   using scikit-learn's TfidfVectorizer.

2. **Metadata Features**: Numeric features extracted from the *original*
   (uncleaned) text — email length, word count, URL count, suspicious
   keywords, uppercase word ratio, etc.

The final feature matrix is a horizontal stack (hstack) of the sparse TF-IDF
matrix and the scaled metadata feature matrix.

Usage:
    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(df_train)
    X_test  = engineer.transform(df_test)

    # Production reload:
    engineer = FeatureEngineer.from_disk()
    X = engineer.transform(df)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_NAMES_PATH, METADATA_SCALER_PATH, SUSPICIOUS_KEYWORDS, TFIDF_CONFIG, TFIDF_VECTORIZER_PATH
from src.utils import load_artifact, save_artifact, setup_logger

# Precompiled regex patterns for metadata extraction
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_DIGIT_RE = re.compile(r"\d")
_SPECIAL_CHAR_RE = re.compile(r"[!@#$%^&*()_+=\[\]{};:'\",.<>?/\\|`~\-]")

# Keywords are lowered once at import time
_LOWERED_KEYWORDS = [kw.lower() for kw in SUSPICIOUS_KEYWORDS]


class FeatureEngineer:
    """
    Feature engineering pipeline that produces the final feature matrix
    for model training by combining TF-IDF and metadata features.

    Attributes:
        tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF transformer.
        scaler (StandardScaler): Fitted scaler for metadata features.
        metadata_feature_names (list[str]): Names of metadata columns.
    """

    # Names of all metadata features, in order
    METADATA_FEATURES = [
        "email_length",
        "word_count",
        "char_count",
        "url_count",
        "digit_count",
        "uppercase_word_count",
        "special_char_count",
        "suspicious_keyword_score",
        "has_subject",
        "subject_length",
        "exclamation_count",
        "question_mark_count",
        "avg_word_length",
        "unique_word_ratio",
        "uppercase_ratio",
    ]

    def __init__(self) -> None:
        self.logger = setup_logger("FeatureEngineer")
        self.tfidf_vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
        self.scaler = StandardScaler()
        self.is_fitted = False

    # =========================================================================
    # METADATA FEATURE EXTRACTION (vectorized — one pass per row)
    # =========================================================================

    @staticmethod
    def _extract_row_features(text: str, subject: str) -> np.ndarray:
        """
        Compute all 15 metadata features for a single (text, subject) pair.

        This runs in a single list-comprehension pass per row, which is
        dramatically faster than 15 separate ``Series.apply`` calls.

        Args:
            text (str): Original (uncleaned) email text.
            subject (str): Original email subject.

        Returns:
            np.ndarray: 15-element feature vector (order matches
                ``METADATA_FEATURES``).
        """
        text = "" if not isinstance(text, str) else text
        subject = "" if not isinstance(subject, str) else subject

        words = text.split()
        n_words = len(words)

        alpha_chars = [c for c in text if c.isalpha()]
        n_alpha = len(alpha_chars)
        n_upper_alpha = sum(1 for c in alpha_chars if c.isupper())

        lower_text = text.lower()
        lower_words = lower_text.split()
        n_unique = len(set(lower_words))

        return np.array(
            [
                len(text),                                  # email_length
                n_words,                                    # word_count
                len(text.replace(" ", "").replace("\t", "").replace("\n", "")),  # char_count
                len(_URL_RE.findall(text)),                 # url_count
                len(_DIGIT_RE.findall(text)),               # digit_count
                sum(1 for w in words if w.isupper() and len(w) >= 2),  # uppercase_word_count
                len(_SPECIAL_CHAR_RE.findall(text)),        # special_char_count
                sum(1 for kw in _LOWERED_KEYWORDS if kw in lower_text),  # suspicious_keyword_score
                1 if subject.strip() else 0,                # has_subject
                len(subject),                               # subject_length
                text.count("!"),                            # exclamation_count
                text.count("?"),                            # question_mark_count
                (sum(len(w) for w in words) / n_words) if n_words else 0.0,  # avg_word_length
                (n_unique / n_words) if n_words else 0.0,   # unique_word_ratio
                (n_upper_alpha / n_alpha) if n_alpha else 0.0,  # uppercase_ratio
            ],
            dtype=np.float64,
        )

    def extract_metadata_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract all metadata features from the DataFrame.

        Features are computed from 'text_combined' (original text) and
        'subject' columns. The result is a NumPy array with one column
        per metadata feature.

        Args:
            df (pd.DataFrame): Must contain 'text_combined' column.
                               Optionally contains 'subject'.

        Returns:
            np.ndarray: Shape (n_samples, n_metadata_features).
        """
        self.logger.info("Extracting metadata features...")

        texts = df["text_combined"].fillna("").astype(str)
        subjects = df.get("subject", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

        rows = [
            self._extract_row_features(text, subject)
            for text, subject in zip(texts, subjects)
        ]
        features = np.vstack(rows) if rows else np.zeros((0, len(self.METADATA_FEATURES)))

        self.logger.info("Extracted %d metadata features", features.shape[1])
        return features

    # =========================================================================
    # TF-IDF FEATURES
    # =========================================================================

    def fit_tfidf(self, texts: pd.Series) -> csr_matrix:
        """
        Fit the TF-IDF vectorizer on training texts.

        Args:
            texts (pd.Series): Cleaned text column from training data.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix.
        """
        config = dict(TFIDF_CONFIG)

        # Guard against tiny corpora where min_df is unsatisfiable
        # (sklearn raises "max_df corresponds to < documents than min_df").
        min_df = config.get("min_df", 3)
        if isinstance(min_df, int) and len(texts) < min_df:
            config["min_df"] = 1

        self.logger.info(
            "Fitting TF-IDF vectorizer (max_features=%s, ngram_range=%s)...",
            config["max_features"], config["ngram_range"],
        )
        vectorizer = TfidfVectorizer(**config)
        tfidf_matrix = vectorizer.fit_transform(texts)
        self.tfidf_vectorizer = vectorizer
        self.logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)
        return tfidf_matrix

    def transform_tfidf(self, texts: pd.Series) -> csr_matrix:
        """
        Transform texts using the already-fitted TF-IDF vectorizer.

        Args:
            texts (pd.Series): Cleaned text column.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix.
        """
        return self.tfidf_vectorizer.transform(texts)

    # =========================================================================
    # COMBINED FEATURE PIPELINE
    # =========================================================================

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Fit on training data and return the combined feature matrix.

        Combines TF-IDF features (sparse) with scaled metadata features.

        Args:
            df (pd.DataFrame): Training DataFrame with 'cleaned_text' and
                               'text_combined' columns.

        Returns:
            scipy.sparse.csr_matrix: Combined feature matrix.
        """
        self.logger.info("Feature Engineering — fit_transform (training)...")

        tfidf_matrix = self.fit_tfidf(df["cleaned_text"])
        metadata = self.extract_metadata_features(df)
        metadata_scaled = self.scaler.fit_transform(metadata)

        combined = hstack([tfidf_matrix, csr_matrix(metadata_scaled)])
        self.is_fitted = True

        self.logger.info("Combined feature matrix shape: %s", combined.shape)

        # Build feature names list for importance analysis
        tfidf_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        all_feature_names = tfidf_names + self.METADATA_FEATURES
        save_artifact(all_feature_names, FEATURE_NAMES_PATH)

        return combined

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Transform new data using the already-fitted vectorizer and scaler.

        Args:
            df (pd.DataFrame): New DataFrame with 'cleaned_text' and
                               'text_combined' columns.

        Returns:
            scipy.sparse.csr_matrix: Combined feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fit before transform.")

        self.logger.info("Feature Engineering — transform (test/inference)...")

        tfidf_matrix = self.transform_tfidf(df["cleaned_text"])
        metadata = self.extract_metadata_features(df)
        metadata_scaled = self.scaler.transform(metadata)

        combined = hstack([tfidf_matrix, csr_matrix(metadata_scaled)])
        self.logger.info("Combined feature matrix shape: %s", combined.shape)

        return combined

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self) -> None:
        """Save the fitted TF-IDF vectorizer and metadata scaler to disk."""
        save_artifact(self.tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
        save_artifact(self.scaler, METADATA_SCALER_PATH)
        self.logger.info("Saved vectorizer and scaler.")

    @classmethod
    def from_disk(cls) -> "FeatureEngineer":
        """
        Reconstruct a fitted FeatureEngineer from saved artifacts.

        Useful for inference contexts (Streamlit / API) where re-fitting
        would be wasteful.

        Returns:
            FeatureEngineer: Instance with fitted vectorizer and scaler.
        """
        instance = cls()
        instance.tfidf_vectorizer = load_artifact(TFIDF_VECTORIZER_PATH)
        instance.scaler = load_artifact(METADATA_SCALER_PATH)
        instance.is_fitted = True
        return instance

    def get_tfidf_feature_names(self) -> np.ndarray:
        """Return the feature names from the fitted TF-IDF vectorizer."""
        return self.tfidf_vectorizer.get_feature_names_out()
