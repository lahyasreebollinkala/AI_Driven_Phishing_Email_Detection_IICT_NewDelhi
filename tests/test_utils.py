"""Tests for src/utils.py — generic helper functions."""

import pytest

from src.utils import format_number, pluralize, safe_filename, truncate_text


@pytest.mark.parametrize(
    "text,length,expected",
    [
        ("short", 100, "short"),
        ("abcdef", 3, "abc..."),
        (None, 10, None),
        ("", 5, ""),
    ],
)
def test_truncate_text(text, length, expected):
    assert truncate_text(text, length) == expected


@pytest.mark.parametrize(
    "count,singular,expected",
    [
        (1, "email", "email"),
        (2, "email", "emails"),
        (0, "email", "emails"),
        (2, "box", "boxs"),
    ],
)
def test_pluralize(count, singular, expected):
    assert pluralize(count, singular) == expected


def test_pluralize_with_explicit_plural():
    assert pluralize(2, "box", "boxes") == "boxes"


def test_format_number_int():
    assert format_number(1234567) == "1,234,567"


def test_format_number_float():
    assert format_number(1234.5) == "1,234.50"


def test_safe_filename_replaces_unsafe_chars():
    assert safe_filename("a/b:c*?.txt") == "a_b_c___txt"


def test_safe_filename_keeps_safe_chars():
    assert safe_filename("my-model_v2") == "my-model_v2"
