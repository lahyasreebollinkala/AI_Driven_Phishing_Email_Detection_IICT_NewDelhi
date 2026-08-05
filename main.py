# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Main Pipeline Orchestrator
# =============================================================================
"""
Entry point that runs the complete ML pipeline end-to-end:

    1. Load & merge all 7 datasets (skipped when cached cleaned data exists)
    2. Preprocess text (NLP cleaning)
    3. Split into train/test sets
    4. Engineer features (TF-IDF + metadata)
    5. Train all models
    6. Evaluate & compare models
    7. Save best model + artifacts
    8. Generate all visualizations

Usage:
    python main.py                        # full pipeline
    python main.py --no-clean             # force re-cleaning of raw data
    python main.py --models LR,RF,XGB     # train a subset of models
    python main.py --save-train-test      # persist train/test CSVs

The pipeline typically takes 10-20 minutes depending on hardware. All outputs
are saved to models/, reports/, and images/. Afterwards launch the app:
    streamlit run app/app.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split

from src.config import (
    CLEANED_DATASET_PATH,
    FEATURE_NAMES_PATH,
    MODEL_CONFIGS,
    RANDOM_STATE,
    STRATIFY,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
    USE_CACHED_CLEANED_DATA,
)
from src.evaluate_models import ModelEvaluator
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DatasetLoader, TextPreprocessor
from src.train_models import ModelTrainer, estimate_positive_ratio
from src.utils import ensure_dirs, load_artifact, setup_logger

VALID_MODEL_NAMES = list(MODEL_CONFIGS.keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="AI-Driven Phishing Email Detection pipeline")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Force re-cleaning of the raw datasets (ignore cached cleaned data).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(VALID_MODEL_NAMES),
        help=f"Comma-separated subset of models to train. Options: {', '.join(VALID_MODEL_NAMES)}",
    )
    parser.add_argument(
        "--save-train-test",
        action="store_true",
        help="Persist the train/test CSVs to dataset/processed/.",
    )
    return parser.parse_args(argv)


def _format_elapsed(seconds: float) -> str:
    """Format a duration in seconds as a human readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    return f"{int(seconds // 60)}m {seconds % 60:.1f}s"


def main(argv: list[str] | None = None) -> None:
    """
    Execute the full phishing detection pipeline.

    Steps:
        1. Directory setup
        2. Dataset loading & merging (or cached cleaned-data reuse)
        3. Text preprocessing
        4. Train/test split
        5. Feature engineering
        6. Model training
        7. Model evaluation
        8. Save best model
        9. Generate visualizations
    """
    args = parse_args(argv)

    logger = setup_logger("main")
    ensure_dirs()

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    requested_models = [m for m in requested_models if m in VALID_MODEL_NAMES]
    if not requested_models:
        logger.error("No valid models selected. Options: %s", ", ".join(VALID_MODEL_NAMES))
        return

    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("AI-DRIVEN PHISHING EMAIL DETECTION USING NLP")
    logger.info("   Full Training Pipeline")
    logger.info("=" * 70)

    # =========================================================================
    # STEP 1: LOAD & PREPROCESS (reuse cached cleaned data when possible)
    # =========================================================================
    use_cached = USE_CACHED_CLEANED_DATA and not args.no_clean
    if use_cached and os.path.exists(CLEANED_DATASET_PATH):
        logger.info("Step 1/8: Reusing cached cleaned dataset (%s)...", CLEANED_DATASET_PATH)
        import pandas as pd

        df = pd.read_csv(CLEANED_DATASET_PATH, low_memory=False)
        logger.info("Loaded cleaned dataset: %s rows", f"{len(df):,}")
    else:
        if args.no_clean:
            logger.info("Step 1/8: Re-cleaning raw datasets (--no-clean)...")
        else:
            logger.info("Step 1/8: Loading and merging raw datasets...")

        loader = DatasetLoader()
        df = loader.load_and_merge()
        logger.info("Merged dataset: %s rows", f"{len(df):,}")

        preprocessor = TextPreprocessor()
        df = preprocessor.process_dataframe(df)
        logger.info("Cleaned dataset: %s rows", f"{len(df):,}")

    # =========================================================================
    # STEP 2: TRAIN/TEST SPLIT (before feature fitting to avoid leakage)
    # =========================================================================
    logger.info("Step 2/8: Train/test split...")
    y = df["label"].values

    train_indices, test_indices = train_test_split(
        range(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None,
    )

    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_test = df.iloc[test_indices].reset_index(drop=True)
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    logger.info("Training set: %s samples", f"{len(df_train):,}")
    logger.info("Test set:     %s samples", f"{len(df_test):,}")
    logger.info(
        "Train labels: Legit=%s, Phishing=%s",
        f"{sum(y_train == 0):,}", f"{sum(y_train == 1):,}",
    )
    logger.info(
        "Test labels:  Legit=%s, Phishing=%s",
        f"{sum(y_test == 0):,}", f"{sum(y_test == 1):,}",
    )

    if args.save_train_test:
        df_train.to_csv(TRAIN_DATA_PATH, index=False)
        df_test.to_csv(TEST_DATA_PATH, index=False)
        logger.info("Saved train/test CSVs to %s.", os.path.dirname(TRAIN_DATA_PATH))

    # =========================================================================
    # STEP 3: FEATURE ENGINEERING (fit on train only — no data leakage)
    # =========================================================================
    logger.info("Step 3/8: Feature engineering...")
    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(df_train)
    X_test = engineer.transform(df_test)
    engineer.save()

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("X_test shape:  %s", X_test.shape)

    # =========================================================================
    # STEP 4: MODEL TRAINING
    # =========================================================================
    logger.info("Step 4/8: Training %d model(s)...", len(requested_models))
    positive_ratio = estimate_positive_ratio(y_train)
    trainer = ModelTrainer(positive_ratio=positive_ratio)
    trained_models = trainer.train_all(X_train, y_train, subset=requested_models)

    if not trained_models:
        logger.error("No models were trained successfully. Aborting.")
        return

    trainer.save_models(trained_models)

    # =========================================================================
    # STEP 5: MODEL EVALUATION
    # =========================================================================
    logger.info("Step 5/8: Evaluating models...")
    evaluator = ModelEvaluator()
    evaluator.evaluate_all(trained_models, X_test, y_test)

    # =========================================================================
    # STEP 6: SAVE BEST MODEL
    # =========================================================================
    logger.info("Step 6/8: Saving best model...")
    evaluator.save_best_model(trained_models)
    evaluator.save_all_results()

    # =========================================================================
    # STEP 7: GENERATE VISUALIZATIONS
    # =========================================================================
    logger.info("Step 7/8: Generating visualizations...")
    feature_names = load_artifact(FEATURE_NAMES_PATH)
    evaluator.generate_all_visualizations(df, y_test, trained_models, feature_names)

    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    total_time = time.time() - pipeline_start

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("   Total time: %s", _format_elapsed(total_time))
    logger.info("   Best model: %s", evaluator.best_model_name)
    best_metrics = evaluator.results[evaluator.best_model_name]
    logger.info("   Accuracy:   %.4f", best_metrics["accuracy"])
    logger.info("   F1 Score:   %.4f", best_metrics["f1_score"])
    logger.info("   ROC AUC:    %.4f", best_metrics["roc_auc"])
    logger.info("=" * 70)
    logger.info("\nTo launch the web app, run:")
    logger.info("   streamlit run app/app.py\n")
    logger.info("To launch the REST API, run:")
    logger.info("   uvicorn src.api:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
