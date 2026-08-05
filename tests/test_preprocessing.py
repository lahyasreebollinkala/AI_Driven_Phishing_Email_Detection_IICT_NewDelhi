"""Tests for src/preprocessing.py — TextPreprocessor and DatasetLoader."""

import pandas as pd
import pytest

from src.preprocessing import DatasetLoader, TextPreprocessor


@pytest.fixture(scope="module")
def preprocessor():
    return TextPreprocessor()


def test_lowercase(preprocessor):
    assert preprocessor.to_lowercase("Hello WORLD") == "hello world"


def test_remove_urls(preprocessor):
    cleaned = preprocessor.remove_urls("Visit https://evil.example.com/steal now")
    assert "http" not in cleaned


def test_remove_email_addresses(preprocessor):
    cleaned = preprocessor.remove_email_addresses("mail attacker@evil.com now")
    assert "@" not in cleaned


def test_remove_punctuation(preprocessor):
    assert preprocessor.remove_punctuation("Hello, world!") == "Hello  world "


def test_remove_numbers(preprocessor):
    assert preprocessor.remove_numbers("win 1,000,000 dollars") == "win  , ,  dollars"


def test_remove_emojis(preprocessor):
    assert "🚨" not in preprocessor.remove_emojis("great news 🚨")


def test_remove_signatures_strips_from_marker(preprocessor):
    text = "This is the body.\n--\nJohn Smith\nCEO"
    assert preprocessor.remove_signatures(text) == "This is the body.\n"


def test_remove_stopwords(preprocessor):
    tokens = preprocessor.tokenize("the quick fox jumps")
    filtered = preprocessor.remove_stopwords(tokens)
    assert "the" not in filtered


def test_clean_text_empty_input(preprocessor):
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""


def test_clean_text_full_pipeline(preprocessor):
    cleaned = preprocessor.clean_text(
        "URGENT: Click http://evil.com to verify your PayPal account now!"
    )
    assert "http" not in cleaned
    assert "paypal" in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_is_whitespace_normalized(preprocessor):
    cleaned = preprocessor.clean_text("Double   spaces   everywhere")
    assert "  " not in cleaned


def test_clean_text_lemmatizes(preprocessor):
    cleaned = preprocessor.clean_text("The runners are running quickly")
    assert "running" in cleaned or "runner" in cleaned


def test_process_dataframe_adds_cleaned_text(tmp_path, monkeypatch):
    from src import preprocessing as preprocessing_module

    monkeypatch.setattr(
        preprocessing_module, "CLEANED_DATASET_PATH", str(tmp_path / "cleaned_test.csv")
    )
    pre = TextPreprocessor()
    df = pd.DataFrame(
        {
            "text_combined": [
                "Confirm your account <b>now</b>",
                "Please review the attached report.",
                "",
                "hi",
            ],
            "label": [1, 0, 0, 0],
        }
    )
    out = pre.process_dataframe(df)
    assert "cleaned_text" in out.columns
    assert len(out) >= 1
    assert all(out["cleaned_text"].str.len() > 0)


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series([0, 1, 0]), [0, 1, 0]),
        (pd.Series(["phishing", "legitimate", "spam"]), [1, 0, 1]),
        (pd.Series(["1", "0", "true"]), [1, 0, 1]),
        (pd.Series([None, "yes", "no"]), [0, 1, 0]),
    ],
)
def test_normalize_labels(series, expected):
    assert list(DatasetLoader._normalize_labels(series)) == expected
