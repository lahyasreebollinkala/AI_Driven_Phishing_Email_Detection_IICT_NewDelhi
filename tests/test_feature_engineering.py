"""Tests for src/feature_engineering.py — metadata feature extraction."""

import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer


def test_extract_row_features_returns_15_values():
    feats = FeatureEngineer._extract_row_features("hello world", "subject")
    assert feats.shape == (15,)


def test_extract_row_features_length_and_words():
    text = "Click here to verify now!"
    feats = FeatureEngineer._extract_row_features(text, "")
    assert feats[0] == len(text)  # email_length
    assert feats[1] == 5          # word_count


def test_extract_row_features_url_count():
    text = "Visit https://evil.com now and http://more.example.org too"
    feats = FeatureEngineer._extract_row_features(text, "")
    assert feats[3] == 2  # url_count


def test_extract_row_features_has_subject():
    assert FeatureEngineer._extract_row_features("body", "Subject!")[8] == 1.0
    assert FeatureEngineer._extract_row_features("body", "")[8] == 0.0
    assert FeatureEngineer._extract_row_features("body", "   ")[8] == 0.0


def test_extract_row_features_subject_length():
    feats = FeatureEngineer._extract_row_features("body", "abcdef")
    assert feats[9] == 6


def test_extract_row_features_exclamations():
    feats = FeatureEngineer._extract_row_features("Act now!!!", "")
    assert feats[10] == 3


def test_extract_row_features_suspicious_keywords():
    feats = FeatureEngineer._extract_row_features("URGENT verify your account", "")
    assert feats[7] >= 3  # urgent, verify, account


def test_extract_row_features_empty_text_is_safe():
    feats = FeatureEngineer._extract_row_features("", "")
    assert np.all(np.isfinite(feats))
    assert feats[1] == 0  # no words


def test_extract_metadata_features_shape():
    df = pd.DataFrame(
        {
            "text_combined": ["one two three", "four"],
            "subject": ["s1", ""],
        }
    )
    engineer = FeatureEngineer()
    feats = engineer.extract_metadata_features(df)
    assert feats.shape == (2, len(FeatureEngineer.METADATA_FEATURES))


def test_metadata_feature_names_count_matches_vector_length():
    vector = FeatureEngineer._extract_row_features("a b c", "sub")
    assert len(FeatureEngineer.METADATA_FEATURES) == vector.shape[0]
