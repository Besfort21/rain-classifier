import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains a Random Forest classifier on weather features."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )

    def split(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Chronological train/test split — no shuffle.

        Shuffling time-series data leaks future information into training.
        """
        X = df[feature_columns]
        y = df["will_rain"]

        split_index = int(len(df) * 0.8)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        logger.info(
            "Split: %d train rows, %d test rows.", len(X_train), len(X_test)
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits the model on training data."""
        self._model.fit(X_train, y_train)
        logger.info("Model trained on %d rows.", len(X_train))

    def save(self, filename: str = "rain_classifier.pkl") -> Path:
        """Saves the trained model to disk."""
        path = self._model_dir / filename
        joblib.dump(self._model, path)
        logger.info("Model saved to %s.", path)
        return path

    def get_model(self) -> RandomForestClassifier:
        return self._model