import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates a trained classifier and generates reports and plots."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        baseline: bool = True,
    ) -> dict:
        """Runs full evaluation and prints a report. Returns metrics dict."""
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }

        print("\n" + "=" * 45)
        print("  Model evaluation")
        print("=" * 45)
        print(classification_report(y_test, y_pred, target_names=["no rain", "rain"]))

        if baseline:
            self._print_baseline(y_test)

        logger.info("Evaluation complete: %s", metrics)
        return metrics

    def plot_confusion_matrix(
        self, model, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Path:
        """Saves a confusion matrix heatmap as PNG."""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["no rain", "rain"],
            yticklabels=["no rain", "rain"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        plt.tight_layout()

        path = self._output_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s.", path)
        return path

    def plot_feature_importance(
        self, model, feature_columns: list[str]
    ) -> Path:
        """Saves a feature importance bar chart as PNG."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_columns[i] for i in indices]
        sorted_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(sorted_features[::-1], sorted_importances[::-1], color="#378ADD")
        ax.set_xlabel("Importance")
        ax.set_title("Feature importance")
        plt.tight_layout()

        path = self._output_dir / "feature_importance.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Feature importance plot saved to %s.", path)
        return path

    def _print_baseline(self, y_test: pd.Series) -> None:
        """Prints metrics for a naive baseline that always predicts no rain."""
        y_baseline = pd.Series([0] * len(y_test))
        baseline_acc = round(accuracy_score(y_test, y_baseline), 4)
        baseline_f1 = round(f1_score(y_test, y_baseline, zero_division=0), 4)
        print("-" * 45)
        print(f"  Baseline (always 'no rain')")
        print(f"  Accuracy: {baseline_acc}   F1: {baseline_f1}")
        print("=" * 45 + "\n")