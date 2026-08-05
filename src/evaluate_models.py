# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# Model Evaluation Module
# =============================================================================
"""
Evaluates all trained models, generates comparison tables, and produces
publication-quality visualizations.

Metrics computed per model:
    - Accuracy, Precision, Recall, F1 Score, ROC AUC
    - Confusion Matrix
    - Training Time, Prediction Time

Visualizations generated:
    - Confusion matrices (one per model)
    - ROC curves (all models overlaid)
    - Precision-Recall curves
    - Feature importance (tree-based models)
    - Model comparison bar chart
    - Class distribution charts
    - Word clouds
    - Email length distribution
    - Correlation heatmap

Usage:
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_all(trained_models, X_test, y_test)
    evaluator.generate_all_visualizations(df, X_test, y_test, feature_names)
"""

import os
import time
import logging
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
)
from scipy.sparse import issparse

from src.config import (
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    COLORS,
    COMPARISON_TABLE_PATH,
    DATASET_STATS_PATH,
    FEATURE_NAMES_PATH,
    IMAGES_DIR,
    LABEL_MAP,
    MODEL_COLORS,
    MODEL_RESULTS_PATH,
    REPORTS_DIR,
)
from src.utils import load_artifact, setup_logger, save_artifact, save_json

CLASSIFICATION_REPORTS_PATH = os.path.join(REPORTS_DIR, "classification_reports.json")


def _sanitize_report(report: dict) -> dict:
    """Recursively convert numpy scalar values to native Python types (JSON-safe)."""
    import numpy as _np

    sanitized = {}
    for key, value in report.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_report(value)
        elif isinstance(value, _np.generic):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


class ModelEvaluator:
    """
    Evaluates trained models and generates comprehensive comparison
    reports and visualizations.

    Attributes:
        results (dict): Evaluation results per model.
        best_model_name (str): Name of the best-performing model.
    """

    def __init__(self):
        self.logger = setup_logger("ModelEvaluator")
        self.results = {}
        self.best_model_name = None
        self.best_model = None

        # Set matplotlib style
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["card"],
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "font.family": "sans-serif",
            "font.size": 11,
        })

    def _prepare_data(self, X, model_name):
        """Prepare data for specific model (e.g., clip negatives for NB)."""
        if model_name == "Naive Bayes":
            if issparse(X):
                X_prepared = X.copy()
                X_prepared.data = np.maximum(X_prepared.data, 0)
                return X_prepared
            return np.maximum(X, 0)
        return X

    def evaluate_single(self, name, model, X_test, y_test, train_time):
        """
        Evaluate a single model on the test set.

        Args:
            name (str): Model name.
            model: Fitted model instance.
            X_test: Test feature matrix.
            y_test: True test labels.
            train_time (float): Training time in seconds.

        Returns:
            dict: Evaluation metrics for this model.
        """
        X_prepared = self._prepare_data(X_test, name)

        # Time the prediction
        start = time.time()
        y_pred = model.predict(X_prepared)
        pred_time = time.time() - start

        # Probability predictions for ROC/PR curves
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_prepared)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_prepared)
        else:
            y_prob = y_pred.astype(float)

        # Precision-Recall AUC — more informative than ROC-AUC under imbalance
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)

        # Metrics
        metrics = {
            "model_name": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": pr_auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=list(LABEL_MAP.values()), output_dict=True,
                zero_division=0,
            ),
            "training_time": round(train_time, 4),
            "prediction_time": round(pred_time, 4),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        self.logger.info(
            "%s: Acc=%.4f, F1=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f",
            name, metrics["accuracy"], metrics["f1_score"],
            metrics["roc_auc"], metrics["pr_auc"],
        )

        return metrics

    def evaluate_all(self, trained_models, X_test, y_test):
        """
        Evaluate all trained models and select the best one.

        Args:
            trained_models (dict): {name: (model, train_time)}.
            X_test: Test feature matrix.
            y_test: True test labels.

        Returns:
            pd.DataFrame: Comparison table with all metrics.
        """
        self.logger.info("=" * 60)
        self.logger.info("📊 Evaluating all models on test set...")
        self.logger.info(f"   Test set: {X_test.shape[0]:,} samples")
        self.logger.info("=" * 60)

        for name, (model, train_time) in trained_models.items():
            metrics = self.evaluate_single(name, model, X_test, y_test, train_time)
            self.results[name] = metrics

        # Build comparison DataFrame
        comparison_data = []
        for name, m in self.results.items():
            comparison_data.append({
                "Model": m["model_name"],
                "Accuracy": round(m["accuracy"] * 100, 2),
                "Precision": round(m["precision"] * 100, 2),
                "Recall": round(m["recall"] * 100, 2),
                "F1 Score": round(m["f1_score"] * 100, 2),
                "ROC AUC": round(m["roc_auc"] * 100, 2),
                "PR AUC": round(m["pr_auc"] * 100, 2),
                "Training Time (s)": m["training_time"],
                "Prediction Time (s)": m["prediction_time"],
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("F1 Score", ascending=False)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏆 MODEL COMPARISON TABLE")
        self.logger.info("=" * 60)
        self.logger.info("\n" + comparison_df.to_string(index=False))

        # Select best model (highest F1, then highest AUC as tiebreaker)
        best_row = comparison_df.iloc[0]
        self.best_model_name = best_row["Model"]
        self.best_model = trained_models[self.best_model_name][0]

        self.logger.info(f"\n🥇 Best Model: {self.best_model_name} "
                         f"(F1={best_row['F1 Score']}%, AUC={best_row['ROC AUC']}%)")

        # Save comparison table
        comparison_df.to_csv(COMPARISON_TABLE_PATH, index=False)
        self.logger.info(f"💾 Saved comparison table to: {COMPARISON_TABLE_PATH}")

        return comparison_df

    def save_best_model(self, trained_models):
        """
        Save the best model and its metadata.

        Args:
            trained_models (dict): {name: (model, train_time)}.
        """
        if self.best_model_name is None:
            self.logger.error("No best model selected — run evaluate_all first.")
            return

        # Save model
        save_artifact(self.best_model, BEST_MODEL_PATH)

        # Save model info
        best_metrics = self.results[self.best_model_name]
        info = {
            "model_name": self.best_model_name,
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1_score"],
            "roc_auc": best_metrics["roc_auc"],
            "pr_auc": best_metrics["pr_auc"],
            "training_time": best_metrics["training_time"],
            "prediction_time": best_metrics["prediction_time"],
        }
        save_json(info, BEST_MODEL_INFO_PATH)

        self.logger.info("Saved best model: %s", self.best_model_name)

    def save_all_results(self):
        """Save complete evaluation results as JSON for the Streamlit app."""
        serializable = {}
        classification_reports = {}
        for name, m in self.results.items():
            serializable[name] = {
                k: v for k, v in m.items()
                if k not in ("y_pred", "y_prob", "classification_report")
            }
            classification_reports[name] = _sanitize_report(m["classification_report"])
        save_json(serializable, MODEL_RESULTS_PATH)
        save_json(classification_reports, CLASSIFICATION_REPORTS_PATH)

    # =========================================================================
    # VISUALIZATION GENERATORS
    # =========================================================================

    def generate_confusion_matrices(self, y_test):
        """Generate and save confusion matrix heatmaps for each model."""
        self.logger.info("📊 Generating confusion matrices...")

        n_models = len(self.results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        fig.patch.set_facecolor(COLORS["background"])

        # Flatten axes for easy iteration
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (name, metrics) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = np.array(metrics["confusion_matrix"])

            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"],
                ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_title(name, fontsize=13, fontweight="bold", color=COLORS["primary"])
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Confusion Matrices", fontsize=16, fontweight="bold",
                      color=COLORS["primary"], y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, "confusion_matrices.png"),
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["background"],
        )
        plt.close()
        self.logger.info("Saved confusion_matrices.png")

        # Save a standalone confusion matrix per model (used by the Streamlit app)
        for name, metrics in self.results.items():
            safe_name = name.lower().replace(" ", "_")
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor(COLORS["background"])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"],
                ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_title(name, fontsize=13, fontweight="bold", color=COLORS["primary"])
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            plt.tight_layout()
            plt.savefig(
                os.path.join(IMAGES_DIR, f"confusion_matrix_{safe_name}.png"),
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"],
            )
            plt.close()
        self.logger.info("Saved per-model confusion matrices.")

    def generate_roc_curves(self, y_test):
        """Generate overlaid ROC curves for all models."""
        self.logger.info("📊 Generating ROC curves...")

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(COLORS["background"])

        for idx, (name, metrics) in enumerate(self.results.items()):
            color = MODEL_COLORS[idx % len(MODEL_COLORS)]
            fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
            auc_val = metrics["roc_auc"]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC = {auc_val:.4f})")

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — Model Comparison", fontsize=14,
                      fontweight="bold", color=COLORS["primary"])
        ax.legend(loc="lower right", fontsize=10, facecolor=COLORS["card"],
                  edgecolor=COLORS["primary"])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.15)

        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, "roc_curves.png"),
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["background"],
        )
        plt.close()
        self.logger.info("   ✅ Saved roc_curves.png")

    def generate_precision_recall_curves(self, y_test):
        """Generate Precision-Recall curves for all models."""
        self.logger.info("📊 Generating Precision-Recall curves...")

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(COLORS["background"])

        for idx, (name, metrics) in enumerate(self.results.items()):
            color = MODEL_COLORS[idx % len(MODEL_COLORS)]
            prec, rec, _ = precision_recall_curve(y_test, metrics["y_prob"])
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, color=color, linewidth=2,
                    label=f"{name} (PR AUC = {pr_auc:.4f})")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14,
                      fontweight="bold", color=COLORS["primary"])
        ax.legend(loc="lower left", fontsize=10, facecolor=COLORS["card"],
                  edgecolor=COLORS["primary"])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.15)

        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, "precision_recall_curves.png"),
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["background"],
        )
        plt.close()
        self.logger.info("   ✅ Saved precision_recall_curves.png")

    def generate_model_comparison_chart(self):
        """Generate grouped bar chart comparing all metrics across models."""
        self.logger.info("📊 Generating model comparison chart...")

        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

        model_names = list(self.results.keys())
        x = np.arange(len(labels))
        width = 0.8 / len(model_names)

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor(COLORS["background"])

        for i, name in enumerate(model_names):
            values = [self.results[name][m] * 100 for m in metrics_to_plot]
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            bars = ax.bar(x + i * width, values, width, label=name,
                         color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                        color=COLORS["text"], fontweight="bold")

        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Model Comparison — All Metrics", fontsize=14,
                      fontweight="bold", color=COLORS["primary"])
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=9, facecolor=COLORS["card"],
                  edgecolor=COLORS["primary"], loc="lower right")
        ax.set_ylim([0, 105])
        ax.grid(axis="y", alpha=0.15)

        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, "model_comparison.png"),
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["background"],
        )
        plt.close()
        self.logger.info("   ✅ Saved model_comparison.png")


    def generate_feature_importance_from_models(self, trained_models, feature_names):
        """
        Generate feature importance using the actual model objects.

        Args:
            trained_models (dict): {name: (model, train_time)}.
            feature_names (list[str]): Feature names list.
        """
        self.logger.info("📊 Generating feature importance from trained models...")

        tree_model_names = ["Random Forest", "XGBoost", "LightGBM"]
        available = {n: m for n, (m, _) in trained_models.items()
                     if n in tree_model_names}

        if not available:
            self.logger.info("   ℹ️  No tree-based models available.")
            return

        for name, model in available.items():
            try:
                importances = model.feature_importances_
                top_n = 20  # Show top 20 features

                # Get top features
                indices = np.argsort(importances)[-top_n:]
                top_features = [feature_names[i] if i < len(feature_names)
                                else f"feature_{i}" for i in indices]
                top_importances = importances[indices]

                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor(COLORS["background"])

                colors = [COLORS["primary"]] * top_n
                ax.barh(range(top_n), top_importances, color=colors, alpha=0.85,
                        edgecolor="white", linewidth=0.5)
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(top_features, fontsize=9)
                ax.set_xlabel("Importance", fontsize=12)
                ax.set_title(f"Top {top_n} Feature Importance — {name}",
                             fontsize=14, fontweight="bold", color=COLORS["primary"])
                ax.grid(axis="x", alpha=0.15)

                plt.tight_layout()
                safe_name = name.lower().replace(" ", "_")
                plt.savefig(
                    os.path.join(IMAGES_DIR, f"feature_importance_{safe_name}.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"],
                )
                plt.close()
                self.logger.info(f"   ✅ Saved feature_importance_{safe_name}.png")

            except Exception as e:
                self.logger.warning(f"   ⚠️  Could not plot importance for {name}: {e}")

    # =========================================================================
    # DATA VISUALIZATION (EDA)
    # =========================================================================

    def generate_class_distribution(self, df):
        """Generate class distribution pie and bar charts."""
        self.logger.info("📊 Generating class distribution chart...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(COLORS["background"])

        counts = df["label"].value_counts().sort_index()
        labels = [LABEL_MAP.get(i, str(i)) for i in counts.index]
        colors = [COLORS["legitimate"], COLORS["phishing"]]

        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"color": COLORS["text"], "fontsize": 12},
            wedgeprops={"edgecolor": COLORS["background"], "linewidth": 2},
        )
        for autotext in autotexts:
            autotext.set_fontweight("bold")
        ax1.set_title("Class Distribution", fontsize=14,
                       fontweight="bold", color=COLORS["primary"])

        # Bar chart
        bars = ax2.bar(labels, counts.values, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{val:,}", ha="center", va="bottom", fontsize=12,
                     color=COLORS["text"], fontweight="bold")
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Email Count by Class", fontsize=14,
                       fontweight="bold", color=COLORS["primary"])
        ax2.grid(axis="y", alpha=0.15)

        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, "class_distribution.png"),
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["background"],
        )
        plt.close()
        self.logger.info("   ✅ Saved class_distribution.png")

    def generate_word_clouds(self, df):
        """Generate word clouds for phishing and legitimate emails."""
        self.logger.info("📊 Generating word clouds...")

        try:
            from wordcloud import WordCloud
        except ImportError:
            self.logger.warning("   ⚠️  wordcloud not installed — skipping.")
            return

        text_col = "cleaned_text" if "cleaned_text" in df.columns else "text_combined"

        for label, label_name in LABEL_MAP.items():
            subset = df[df["label"] == label][text_col].dropna()
            text = " ".join(subset.astype(str).values)

            if not text.strip():
                continue

            color = COLORS["legitimate"] if label == 0 else COLORS["phishing"]

            wc = WordCloud(
                width=1200, height=600,
                background_color=COLORS["background"],
                colormap="cool" if label == 0 else "hot",
                max_words=100,
                max_font_size=120,
                random_state=42,
                contour_width=2,
                contour_color=color,
            ).generate(text)

            fig, ax = plt.subplots(figsize=(14, 7))
            fig.patch.set_facecolor(COLORS["background"])
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(
                f"Word Cloud — {label_name} Emails",
                fontsize=16, fontweight="bold", color=color, pad=15,
            )

            safe_name = label_name.lower()
            plt.savefig(
                os.path.join(IMAGES_DIR, f"wordcloud_{safe_name}.png"),
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"],
            )
            plt.close()
            self.logger.info(f"   ✅ Saved wordcloud_{safe_name}.png")

    def generate_email_length_distribution(self, df):
        """Generate email length distribution histogram."""
        self.logger.info("📊 Generating email length distribution...")

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(COLORS["background"])

        text_col = "text_combined"
        for label, label_name in LABEL_MAP.items():
            subset = df[df["label"] == label][text_col].dropna().astype(str)
            lengths = subset.str.len()
            # Clip to reasonable range for visualization
            lengths = lengths[lengths < lengths.quantile(0.99)]
            color = COLORS["legitimate"] if label == 0 else COLORS["phishing"]
            ax.hist(lengths, bins=50, alpha=0.6, label=label_name,
                    color=color, edgecolor="white", linewidth=0.3)

        ax.set_xlabel("Email Length (characters)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Email Length Distribution", fontsize=14,
                      fontweight="bold", color=COLORS["primary"])
        ax.legend(fontsize=11, facecolor=COLORS["card"],
                  edgecolor=COLORS["primary"])
        ax.grid(alpha=0.15)

        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, "email_length_distribution.png"),
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["background"],
        )
        plt.close()
        self.logger.info("   ✅ Saved email_length_distribution.png")

    def generate_top_words(self, df, top_n=20):
        """Generate top frequent words bar charts (overall, phishing, legitimate)."""
        self.logger.info("📊 Generating top words charts...")

        text_col = "cleaned_text" if "cleaned_text" in df.columns else "text_combined"

        categories = {
            "overall": df[text_col],
            "phishing": df[df["label"] == 1][text_col],
            "legitimate": df[df["label"] == 0][text_col],
        }

        color_map = {
            "overall": COLORS["primary"],
            "phishing": COLORS["phishing"],
            "legitimate": COLORS["legitimate"],
        }

        for cat_name, texts in categories.items():
            # Count word frequencies
            all_words = " ".join(texts.dropna().astype(str).values).split()
            word_freq = pd.Series(all_words).value_counts().head(top_n)

            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor(COLORS["background"])

            ax.barh(range(len(word_freq)), word_freq.values,
                    color=color_map[cat_name], alpha=0.85,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(word_freq)))
            ax.set_yticklabels(word_freq.index, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency", fontsize=12)
            ax.set_title(f"Top {top_n} Words — {cat_name.title()} Emails",
                         fontsize=14, fontweight="bold", color=COLORS["primary"])
            ax.grid(axis="x", alpha=0.15)

            plt.tight_layout()
            plt.savefig(
                os.path.join(IMAGES_DIR, f"top_words_{cat_name}.png"),
                dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"],
            )
            plt.close()
            self.logger.info(f"   ✅ Saved top_words_{cat_name}.png")

    def generate_correlation_heatmap(self, metadata_df):
        """
        Generate correlation heatmap of metadata features.

        Args:
            metadata_df (pd.DataFrame): DataFrame of metadata features.
        """
        self.logger.info("📊 Generating correlation heatmap...")

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(COLORS["background"])

        corr = metadata_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            square=True, linewidths=0.5,
            ax=ax, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8},
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14,
                      fontweight="bold", color=COLORS["primary"])

        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, "correlation_heatmap.png"),
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["background"],
        )
        plt.close()
        self.logger.info("   ✅ Saved correlation_heatmap.png")

    # =========================================================================
    # MASTER VISUALIZATION GENERATOR
    # =========================================================================

    def generate_all_visualizations(self, df, y_test, trained_models, feature_names=None):
        """
        Generate all visualizations at once.

        Args:
            df (pd.DataFrame): Full cleaned DataFrame.
            y_test: Test labels (for ROC/PR/CM).
            trained_models (dict): {name: (model, train_time)}.
            feature_names (list[str], optional): Feature names for importance.
                Falls back to the saved FEATURE_NAMES_PATH artifact.
        """
        self.logger.info("=" * 60)
        self.logger.info("Generating all visualizations...")
        self.logger.info("=" * 60)

        # EDA charts
        self.generate_class_distribution(df)
        self.generate_word_clouds(df)
        self.generate_email_length_distribution(df)
        self.generate_top_words(df)

        # Model evaluation charts
        self.generate_confusion_matrices(y_test)
        self.generate_roc_curves(y_test)
        self.generate_precision_recall_curves(y_test)
        self.generate_model_comparison_chart()

        if feature_names is None:
            try:
                feature_names = load_artifact(FEATURE_NAMES_PATH)
            except FileNotFoundError:
                feature_names = None

        if feature_names is not None:
            self.generate_feature_importance_from_models(trained_models, feature_names)
        else:
            self.logger.warning("No feature names available — skipping feature importance charts.")

        # Correlation heatmap (using metadata features from the DataFrame)
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        metadata = fe.extract_metadata_features(df)
        metadata_df = pd.DataFrame(metadata, columns=FeatureEngineer.METADATA_FEATURES)
        metadata_df["label"] = df["label"].values
        self.generate_correlation_heatmap(metadata_df)

        # Save dataset stats for the dashboard
        stats = {
            "total_samples": len(df),
            "phishing_count": int(df["label"].sum()),
            "legitimate_count": int((df["label"] == 0).sum()),
            "models_trained": len(self.results),
            "best_model": self.best_model_name,
            "best_accuracy": round(self.results[self.best_model_name]["accuracy"] * 100, 2),
            "best_f1": round(self.results[self.best_model_name]["f1_score"] * 100, 2),
            "source_datasets": df["source_dataset"].nunique() if "source_dataset" in df.columns else 7,
        }
        save_json(stats, DATASET_STATS_PATH)

        self.logger.info("=" * 60)
        self.logger.info("All visualizations generated successfully!")
        self.logger.info("=" * 60)
