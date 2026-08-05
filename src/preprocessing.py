# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Data Preprocessing Module
# =============================================================================
"""
Handles all data loading, merging, and NLP text preprocessing.

Two main classes:
    DatasetLoader     : Loads all 7 raw CSV datasets, unifies their schemas,
                        and merges them into a single clean DataFrame.
    TextPreprocessor  : Full NLP cleaning pipeline — HTML removal, URL stripping,
                        lowercasing, lemmatization, stopword removal, etc.

Usage:
    loader = DatasetLoader()
    df = loader.load_and_merge()

    preprocessor = TextPreprocessor()
    df = preprocessor.process_dataframe(df)
"""

from __future__ import annotations

import logging
import re

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from src.config import (
    CLEANED_DATASET_PATH,
    MAX_TEXT_LENGTH,
    MERGED_DATASET_PATH,
    MIN_TEXT_LENGTH,
    RAW_DATASETS,
)
from src.utils import ensure_dirs, setup_logger

# ---------------------------------------------------------------------------
# NLTK data bootstrap — tolerant of both legacy `punkt` and modern `punkt_tab`
# ---------------------------------------------------------------------------

# Resources to verify/download. ``punkt_tab`` is required by NLTK >= 3.9 for
# ``word_tokenize``; older installs only ship ``punkt``.
_NLTK_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}


def _ensure_nltk_data() -> None:
    """Download any missing NLTK resources (no-op when everything exists).

    This is best-effort: a missing resource or a failed/blocked download must
    never take the application down. Callers degrade gracefully (e.g. fall back
    to whitespace tokenization / no stopword filtering).
    """
    for resource_id, package in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_id)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:  # noqa: BLE001 — network errors, SSL, proxies
                pass
        except Exception:  # noqa: BLE001 — corrupt index files, etc.
            pass


# =============================================================================
# 1. DATASET LOADER
# =============================================================================

# Common schema columns for every unified dataset
_SCHEMA_COLUMNS = ["text_combined", "subject", "body", "sender", "label"]


class DatasetLoader:
    """
    Loads all raw email datasets, normalizes column names, and merges
    them into a single unified DataFrame.

    Each dataset has a different schema. This class maps every dataset
    to a common schema:
        - text_combined (str) : full email text (subject + body)
        - subject (str)       : email subject line (if available)
        - body (str)          : email body (if available)
        - sender (str)        : sender address (if available)
        - label (int)         : 0 = Legitimate, 1 = Phishing
        - source_dataset (str): name of the originating dataset
    """

    def __init__(self) -> None:
        self.logger = setup_logger("DatasetLoader")
        self.datasets: dict[str, pd.DataFrame] = {}

    def load_all_datasets(self) -> dict[str, pd.DataFrame]:
        """
        Load all CSV files from the raw data directory.

        Returns:
            dict[str, pd.DataFrame]: Mapping of dataset name → loaded DataFrame.
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading all raw datasets...")
        self.logger.info("=" * 60)

        for name, filepath in RAW_DATASETS.items():
            try:
                df = pd.read_csv(filepath, low_memory=False)
                self.datasets[name] = df
                self.logger.info(
                    "%s: %s rows, columns=%s",
                    name, f"{len(df):,}", list(df.columns),
                )
            except FileNotFoundError:
                self.logger.warning("%s: File not found at %s", name, filepath)
            except Exception as e:  # noqa: BLE001
                self.logger.error("%s: Error loading - %s", name, e)

        self.logger.info("Loaded %d datasets successfully.", len(self.datasets))
        return self.datasets

    @staticmethod
    def _normalize_labels(series: pd.Series) -> pd.Series:
        """
        Coerce heterogeneous label values into binary 0/1 ints.

        Accepts already-numeric labels and common string forms such as
        'phishing'/'legitimate' or '1'/'0'.
        """
        if pd.api.types.is_integer_dtype(series):
            return series.astype(int)

        def _to_int(value) -> int:
            if pd.isna(value):
                return 0
            if isinstance(value, (int, float)):
                return int(value)
            lowered = str(value).strip().lower()
            if lowered in ("1", "phishing", "spam", "true", "yes"):
                return 1
            return 0

        return series.map(_to_int).astype(int)

    def _unify_phishing_email(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize phishing_email.csv (has text_combined + label)."""
        result = pd.DataFrame()
        result["text_combined"] = df["text_combined"].astype(str)
        result["subject"] = ""
        result["body"] = df["text_combined"].astype(str)
        result["sender"] = ""
        result["label"] = self._normalize_labels(df["label"])
        return result

    def _unify_subject_body(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datasets with subject + body + label (Enron, Ling)."""
        result = pd.DataFrame()
        subject = df["subject"].fillna("").astype(str)
        body = df["body"].fillna("").astype(str)
        result["text_combined"] = (subject + " " + body).str.strip()
        result["subject"] = subject
        result["body"] = body
        result["sender"] = ""
        result["label"] = self._normalize_labels(df["label"])
        return result

    def _unify_full_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize datasets with sender, receiver, date, subject, body,
        label, urls (SpamAssassin, CEAS_08, Nazario, Nigerian_Fraud).
        """
        result = pd.DataFrame()
        subject = df["subject"].fillna("").astype(str)
        body = df["body"].fillna("").astype(str)
        result["text_combined"] = (subject + " " + body).str.strip()
        result["subject"] = subject
        result["body"] = body
        result["sender"] = df["sender"].fillna("").astype(str)
        result["label"] = self._normalize_labels(df["label"])
        return result

    def unify_schemas(self) -> None:
        """
        Apply dataset-specific schema normalization to all loaded datasets.

        Each dataset is mapped to the common schema using the appropriate
        helper method based on its column structure.
        """
        self.logger.info("Unifying dataset schemas...")

        unified: dict[str, pd.DataFrame] = {}
        for name, df in self.datasets.items():
            if name == "phishing_email":
                unified[name] = self._unify_phishing_email(df)
            elif name in ("enron", "ling"):
                unified[name] = self._unify_subject_body(df)
            else:
                unified[name] = self._unify_full_schema(df)

            self.logger.info("%s: unified -> %s rows", name, f"{len(unified[name]):,}")

        self.datasets = unified

    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all unified datasets into a single DataFrame.

        Adds a 'source_dataset' column to track the origin of each row.

        Returns:
            pd.DataFrame: Merged DataFrame with all datasets combined.
        """
        self.logger.info("Merging all datasets...")

        frames = []
        for name, df in self.datasets.items():
            frame = df.copy()
            frame["source_dataset"] = name
            frames.append(frame)

        merged = pd.concat(frames, ignore_index=True)

        self.logger.info("Merged dataset: %s total rows", f"{len(merged):,}")
        self.logger.info("Label distribution:\n%s", merged["label"].value_counts().to_string())
        return merged

    def load_and_merge(self) -> pd.DataFrame:
        """
        Full pipeline: load -> unify schemas -> merge -> save.

        Returns:
            pd.DataFrame: The merged, schema-unified dataset.
        """
        ensure_dirs()
        self.load_all_datasets()
        self.unify_schemas()
        merged_df = self.merge_datasets()

        merged_df.to_csv(MERGED_DATASET_PATH, index=False)
        self.logger.info("Saved merged dataset to: %s", MERGED_DATASET_PATH)

        return merged_df


# =============================================================================
# 2. TEXT PREPROCESSOR
# =============================================================================

# Common email signature markers — content after these is removed
_SIGNATURE_MARKERS = re.compile(
    r"(?:"
    r"^--\s*$|"                              # classic '--' signature separator
    r"^----+$|"                              # dashed separator lines
    r"sent from my (?:iphone|android|mobile)|"
    r"get outlook for (?:ios|android)|"
    r"^regards[,:]?$|^best regards[,:]?$|"
    r"^kind regards[,:]?$|^thanks[,:]?$|"
    r"^thank you[,:]?$|^sincerely[,:]?$|"
    r"^with warm regards[,:]?$|^yours truly[,:]?$|"
    r"^________________________________________________*$"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


class TextPreprocessor:
    """
    NLP text cleaning pipeline for email content.

    Applies the following transformations in order:
        1.  Lowercase
        2.  Remove HTML tags
        3.  Remove URLs
        4.  Remove email addresses
        5.  Remove signature/quote blocks
        6.  Remove punctuation
        7.  Remove numbers
        8.  Remove emojis
        9.  Tokenize
        10. Remove stopwords
        11. Lemmatize
        12. Rejoin tokens

    Each step is a standalone method that can be used independently.
    """

    # Precompiled regex patterns for performance
    _URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE)
    _EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    _PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
    _NUMBER_PATTERN = re.compile(r"\d+")
    _EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE,
    )
    _WHITESPACE_PATTERN = re.compile(r"\s+")
    _HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

    def __init__(self) -> None:
        self.logger = setup_logger("TextPreprocessor")
        _ensure_nltk_data()

        # NLTK corpus resources are loaded defensively. On a fresh deployment
        # (e.g. Streamlit Cloud) the data may not be present yet and the
        # download can be slow or blocked — the pipeline still works without
        # stopword removal / lemmatization, it is just slightly noisier.
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception as e:  # noqa: BLE001
            self.logger.warning("NLTK stopwords unavailable (%s); skipping stopword removal.", e)
            self.stop_words = set()

        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:  # noqa: BLE001
            self.logger.warning("NLTK WordNet unavailable (%s); skipping lemmatization.", e)
            self.lemmatizer = None

    # --- Individual Cleaning Steps ---

    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def remove_html(self, text: str) -> str:
        """Strip HTML tags using BeautifulSoup (regex fallback)."""
        try:
            soup = BeautifulSoup(text, "lxml")
            return soup.get_text(separator=" ")
        except Exception:  # noqa: BLE001 — lxml may be absent
            return self._HTML_TAG_PATTERN.sub(" ", text)

    def remove_urls(self, text: str) -> str:
        """Remove all URLs (http, https, www, ftp)."""
        return self._URL_PATTERN.sub(" ", text)

    def remove_email_addresses(self, text: str) -> str:
        """Remove all email addresses."""
        return self._EMAIL_PATTERN.sub(" ", text)

    def remove_signatures(self, text: str) -> str:
        """
        Strip common email signature blocks and quote separators.

        Everything from the first signature marker onwards is discarded.
        """
        match = _SIGNATURE_MARKERS.search(text)
        if match:
            text = text[: match.start()]
        return text

    def remove_punctuation(self, text: str) -> str:
        """Remove all punctuation characters."""
        return self._PUNCTUATION_PATTERN.sub(" ", text)

    def remove_numbers(self, text: str) -> str:
        """Remove all numeric characters."""
        return self._NUMBER_PATTERN.sub(" ", text)

    def remove_emojis(self, text: str) -> str:
        """Remove emoji and special Unicode characters."""
        return self._EMOJI_PATTERN.sub(" ", text)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into individual words using NLTK word_tokenize.

        Args:
            text (str): Input text.

        Returns:
            list[str]: List of word tokens.
        """
        try:
            return word_tokenize(text)
        except Exception:  # noqa: BLE001
            return text.split()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Remove English stopwords from token list.

        Args:
            tokens (list[str]): List of word tokens.

        Returns:
            list[str]: Filtered list without stopwords.
        """
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Apply lemmatization to reduce words to their base form.

        Args:
            tokens (list[str]): List of word tokens.

        Returns:
            list[str]: Lemmatized tokens.
        """
        if self.lemmatizer is None:
            return tokens
        # WordNet data is loaded lazily by NLTK on first use; if it is missing
        # (fresh deployment, blocked download) the call raises LookupError.
        result = []
        for token in tokens:
            try:
                result.append(self.lemmatizer.lemmatize(token))
            except Exception:  # noqa: BLE001 — corpus data unavailable
                result.append(token)
        return result

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into a single space."""
        return self._WHITESPACE_PATTERN.sub(" ", text).strip()

    # --- Full Pipeline ---

    def clean_text(self, text: str) -> str:
        """
        Apply the complete NLP preprocessing pipeline to a single text string.

        Pipeline order:
            lowercase -> remove HTML -> remove URLs -> remove emails ->
            remove signatures -> remove punctuation -> remove numbers ->
            remove emojis -> tokenize -> remove stopwords -> lemmatize -> rejoin

        Args:
            text (str): Raw email text.

        Returns:
            str: Fully cleaned text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Truncate extremely long texts for performance
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        text = self.to_lowercase(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_email_addresses(text)
        text = self.remove_signatures(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_emojis(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)

        cleaned = self.normalize_whitespace(" ".join(tokens))
        return cleaned

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full cleaning pipeline to an entire DataFrame.

        Steps:
            1. Handle missing values (fill NaN with empty string)
            2. Apply clean_text to text_combined column
            3. Remove duplicates
            4. Remove rows with empty cleaned text
            5. Remove rows with text shorter than MIN_TEXT_LENGTH

        Args:
            df (pd.DataFrame): DataFrame with 'text_combined' and 'label' columns.

        Returns:
            pd.DataFrame: Cleaned DataFrame ready for feature engineering.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting text preprocessing pipeline...")
        self.logger.info("=" * 60)

        initial_count = len(df)
        self.logger.info("Initial rows: %s", f"{initial_count:,}")

        # Step 1: Handle missing values
        df = df.copy()
        df["text_combined"] = df["text_combined"].fillna("")

        # Step 2: Apply NLP cleaning pipeline with progress bar
        self.logger.info("Applying NLP cleaning (this may take a few minutes)...")
        tqdm.pandas(desc="Cleaning emails")
        df["cleaned_text"] = df["text_combined"].progress_apply(self.clean_text)

        # Step 3: Remove duplicates based on cleaned text
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["cleaned_text"], keep="first")
        self.logger.info("Removed %s duplicate rows", f"{before_dedup - len(df):,}")

        # Step 4: Remove empty rows
        before_empty = len(df)
        df = df[df["cleaned_text"].str.strip().str.len() > 0]
        self.logger.info("Removed %s empty rows", f"{before_empty - len(df):,}")

        # Step 5: Remove very short texts
        before_short = len(df)
        df = df[df["cleaned_text"].str.len() >= MIN_TEXT_LENGTH]
        self.logger.info(
            "Removed %s short rows (< %s chars)", f"{before_short - len(df):,}", MIN_TEXT_LENGTH
        )

        df = df.reset_index(drop=True)

        final_count = len(df)
        self.logger.info(
            "Final rows: %s (removed %s total)",
            f"{final_count:,}",
            f"{initial_count - final_count:,}",
        )
        self.logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())

        # Save cleaned dataset
        ensure_dirs()
        df.to_csv(CLEANED_DATASET_PATH, index=False)
        self.logger.info("Saved cleaned dataset to: %s", CLEANED_DATASET_PATH)

        return df
