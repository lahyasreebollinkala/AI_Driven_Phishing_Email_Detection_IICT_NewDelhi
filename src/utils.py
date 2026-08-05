# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Utility Module
# =============================================================================
"""
Shared utility functions used across the entire pipeline.

Provides:
    - Logging setup with file and console handlers
    - Timer decorator for measuring execution time
    - Artifact save/load wrappers (joblib serialization)
    - Directory creation helpers
    - JSON I/O helpers
    - Small display/number formatting helpers
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any

import joblib

from src.config import (
    IMAGES_DIR,
    LOG_DATE_FORMAT,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)


# =============================================================================
# 1. DIRECTORY MANAGEMENT
# =============================================================================

def ensure_dirs() -> None:
    """
    Create all required project directories if they don't already exist.

    Directories created:
        - dataset/processed/
        - models/
        - reports/
        - images/
        - logs/
    """
    directories = [
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        IMAGES_DIR,
        LOGS_DIR,
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)


# =============================================================================
# 2. LOGGING SETUP
# =============================================================================

def setup_logger(name: str = "phishing_detection", level: str | None = None) -> logging.Logger:
    """
    Configure and return a logger with both file and console handlers.

    Args:
        name (str): Logger name. Defaults to 'phishing_detection'.
        level (str, optional): Logging level override. Uses ``config.LOG_LEVEL`` if None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure log directory exists (best-effort; a read-only FS must not crash)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    except OSError:
        pass

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # File handler — writes to logs/phishing_detection.log (optional).
    # If the file cannot be opened (permissions, read-only FS), fall back to
    # console-only logging rather than failing at import/startup time.
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    # Console handler — prints to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# 3. TIMER DECORATOR
# =============================================================================

def timer(func):
    """
    Decorator that measures and logs function execution time.

    Usage::

        @timer
        def my_function():
            ...

    Note:
        For transparency, the wrapper returns ``(result, elapsed_seconds)``
        so callers can also capture the elapsed time programmatically.

    Returns:
        tuple: ``(original_return_value, elapsed_seconds)``
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("phishing_detection")
        logger.info("Starting: %s", func.__name__)

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        logger.info("Completed: %s in %s", func.__name__, _format_elapsed(elapsed))
        return result, elapsed

    return wrapper


def _format_elapsed(elapsed: float) -> str:
    """Format an elapsed-seconds value as a human readable string."""
    if elapsed < 60:
        return f"{elapsed:.2f} seconds"
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    return f"{minutes}m {seconds:.2f}s"


# =============================================================================
# 4. ARTIFACT SERIALIZATION
# =============================================================================

def save_artifact(obj: Any, filepath: str) -> None:
    """
    Save a Python object to disk using joblib serialization.

    Args:
        obj: Any picklable Python object (model, vectorizer, scaler, etc.).
        filepath (str): Absolute path to save the artifact.

    Raises:
        IOError: If the file cannot be written.
    """
    logger = logging.getLogger("phishing_detection")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        joblib.dump(obj, filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info("Saved artifact: %s (%.2f MB)", os.path.basename(filepath), size_mb)
    except Exception as e:  # noqa: BLE001 — re-raise for the caller to handle
        logger.error("Failed to save artifact %s: %s", filepath, e)
        raise


def load_artifact(filepath: str) -> Any:
    """
    Load a Python object from a joblib file.

    Args:
        filepath (str): Absolute path to the artifact file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the artifact file doesn't exist.
    """
    logger = logging.getLogger("phishing_detection")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found: {filepath}")

    try:
        obj = joblib.load(filepath)
        logger.info("Loaded artifact: %s", os.path.basename(filepath))
        return obj
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to load artifact %s: %s", filepath, e)
        raise


# =============================================================================
# 5. JSON I/O HELPERS
# =============================================================================

def save_json(data: dict | list, filepath: str) -> None:
    """
    Save a dictionary/list to a JSON file with pretty printing.

    Args:
        data (dict | list): Data to serialize.
        filepath (str): Absolute path to the JSON file.
    """
    logger = logging.getLogger("phishing_detection")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logger.info("Saved JSON: %s", os.path.basename(filepath))
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to save JSON %s: %s", filepath, e)
        raise


def load_json(filepath: str) -> dict | list:
    """
    Load data from a JSON file.

    Args:
        filepath (str): Absolute path to the JSON file.

    Returns:
        dict | list: Deserialized JSON data.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
    """
    logger = logging.getLogger("phishing_detection")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON: %s", os.path.basename(filepath))
        return data
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to load JSON %s: %s", filepath, e)
        raise


# =============================================================================
# 6. GENERAL HELPERS
# =============================================================================

def get_timestamp() -> str:
    """
    Get the current timestamp formatted for display.

    Returns:
        str: Formatted timestamp string (e.g., '2026-08-04 18:30:00').
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_number(n: int | float) -> str:
    """
    Format a large number with comma separators for display.

    Args:
        n (int | float): The number to format.

    Returns:
        str: Formatted number string (e.g., '164,972').
    """
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def truncate_text(text: str | None, max_length: int = 200) -> str | None:
    """
    Truncate text to a maximum length with an ellipsis.

    Args:
        text (str | None): The text to truncate.
        max_length (int): Maximum allowed length. Defaults to 200.

    Returns:
        str | None: Truncated text with '...' appended if it exceeds max_length.
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Return ``singular`` or ``plural`` depending on ``count`` (default plural = singular + 's')."""
    if count == 1:
        return singular
    return plural if plural is not None else singular + "s"


def safe_filename(name: str) -> str:
    """Sanitize a string for safe use as a filename."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
