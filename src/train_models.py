# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Model Training Module
# =============================================================================
"""
Trains and times multiple ML classifiers for phishing email detection.

Supports 6 models:
    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Random Forest
    4. XGBoost   (optional — gracefully skipped if not installed)
    5. LightGBM  (optional — gracefully skipped if not installed)
    6. Neural Network (scikit-learn MLPClassifier)

Class imbalance is handled automatically: scikit-learn models use
``class_weight="balanced"`` (set in config) while gradient-boosted models
receive an estimated ``scale_pos_weight``.

Each model is timed during training. Results are stored as a dict of
``{model_name: (fitted_model, training_time_seconds)}``.

Usage:
    trainer = ModelTrainer(positive_ratio=0.62)
    results = trainer.train_all(X_train, y_train)
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from src.config import MODEL_CONFIGS, MODELS_DIR, RANDOM_STATE
from src.utils import save_artifact, setup_logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"


def estimate_positive_ratio(y) -> float:
    """
    Estimate the ratio of positive (phishing) class in the training labels.

    Used to weight gradient-boosted models when classes are imbalanced.

    Args:
        y: Array-like of binary labels.

    Returns:
        float: ``max(y) / min(counts)`` style weight, or 1.0 when balanced.
    """
    y_arr = np.asarray(y)
    counts = np.bincount(y_arr.astype(int))
    if len(counts) < 2 or counts[0] == 0 or counts[1] == 0:
        return 1.0
    return counts[0] / counts[1]


class ModelTrainer:
    """
    Trains multiple ML classifiers with timing and logging.

    Handles optional dependencies (XGBoost, LightGBM) gracefully —
    if the library isn't installed, the model is skipped with a warning.

    Attributes:
        models (dict): {model_name: fitted_model_instance}
        training_times (dict): {model_name: training_time_seconds}
    """

    def __init__(self, positive_ratio: float = 1.0) -> None:
        self.logger = setup_logger("ModelTrainer")
        self.positive_ratio = positive_ratio
        self.models: dict[str, object] = {}
        self.training_times: dict[str, float] = {}

    def _create_model(self, name: str) -> object | None:
        """
        Instantiate a model by name using the config-driven hyperparameters.

        Gradient-boosted models receive ``scale_pos_weight`` derived from the
        measured class imbalance.

        Args:
            name (str): Model name as defined in config.MODEL_CONFIGS.

        Returns:
            object | None: Instantiated model, or None if the library is missing.
        """
        config = MODEL_CONFIGS[name]
        params = dict(config["params"])

        if name == "Logistic Regression":
            return LogisticRegression(**params)

        if name == "Naive Bayes":
            return MultinomialNB(**params)

        if name == "Random Forest":
            return RandomForestClassifier(**params)

        if name == "XGBoost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                self.logger.warning("XGBoost not installed — skipping. Install with: pip install xgboost")
                return None
            params["scale_pos_weight"] = self.positive_ratio
            return XGBClassifier(**params)

        if name == "LightGBM":
            try:
                from lightgbm import LGBMClassifier
            except ImportError:
                self.logger.warning("LightGBM not installed — skipping. Install with: pip install lightgbm")
                return None
            params["scale_pos_weight"] = self.positive_ratio
            return LGBMClassifier(**params)

        if name == "Neural Network":
            return MLPClassifier(**params)

        self.logger.error("Unknown model name: %s", name)
        return None

    @staticmethod
    def _prepare_data(X, name: str):
        """
        Prepare the feature matrix for a specific model.

        Multinomial Naive Bayes requires non-negative values, so negative
        entries are clipped to zero for that model only.

        Args:
            X: Feature matrix (sparse or dense).
            name (str): Model name.

        Returns:
            Prepared feature matrix.
        """
        if name == "Naive Bayes":
            if issparse(X):
                X_prepared = X.copy()
                X_prepared.data = np.maximum(X_prepared.data, 0)
                return X_prepared
            return np.maximum(X, 0)

        # MLP and the rest accept sparse matrices directly.
        return X

    def train_single(self, name: str, X_train, y_train) -> tuple[object | None, float]:
        """
        Train a single model with timing.

        Args:
            name (str): Model name from config.
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            tuple: (fitted_model, training_time_seconds) or (None, 0.0) if skipped.
        """
        self.logger.info("Training: %s...", name)

        model = self._create_model(name)
        if model is None:
            return None, 0.0

        X_prepared = self._prepare_data(X_train, name)

        try:
            start_time = time.time()
            model.fit(X_prepared, y_train)
            training_time = time.time() - start_time

            self.logger.info("%s trained in %s", name, _format_duration(training_time))
            return model, training_time

        except Exception as e:  # noqa: BLE001 — one bad model must not sink the pipeline
            self.logger.error("%s training failed: %s", name, e)
            return None, 0.0

    def train_all(
        self,
        X_train,
        y_train: ArrayLike,
        subset: list[str] | None = None,
    ) -> dict[str, tuple[object, float]]:
        """
        Train all configured models sequentially.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            subset (list[str] | None, optional): Restrict training to these
                model display names. Defaults to all configured models.

        Returns:
            dict: {model_name: (fitted_model, training_time_seconds)}
                  Only includes models that trained successfully.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting model training pipeline...")
        self.logger.info(
            "Training set: %s samples, %s features",
            f"{X_train.shape[0]:,}", f"{X_train.shape[1]:,}",
        )
        self.logger.info("Estimated positive_ratio (phishing): %.3f", self.positive_ratio)
        self.logger.info("=" * 60)

        model_names = subset or list(MODEL_CONFIGS)
        results: dict[str, tuple[object, float]] = {}

        for name in model_names:
            model, train_time = self.train_single(name, X_train, y_train)
            if model is not None:
                results[name] = (model, train_time)
                self.models[name] = model
                self.training_times[name] = train_time

        self.logger.info("=" * 60)
        self.logger.info(
            "Training complete. %d/%d models trained successfully.",
            len(results), len(model_names),
        )

        self.logger.info("Training Times Summary:")
        for name, (_, elapsed) in sorted(results.items(), key=lambda item: item[1][1]):
            self.logger.info("  %-25s : %s", name, _format_duration(elapsed))

        return results

    def save_models(self, results: dict[str, tuple[object, float]]) -> None:
        """
        Save all trained models to the models/ directory.

        Each model is saved as a separate ``.joblib`` file.

        Args:
            results (dict): {model_name: (fitted_model, training_time)}.
        """
        self.logger.info("Saving trained models...")

        for name, (model, _) in results.items():
            safe_name = name.lower().replace(" ", "_")
            filepath = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
            save_artifact(model, filepath)

        self.logger.info("Saved %d models to %s", len(results), MODELS_DIR)
