import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class RainPredictor:
    """Loads a saved model and predicts rain for new data."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train.py first."
            )
        self._model = joblib.load(model_path)
        logger.info("Model loaded from %s.", model_path)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Returns binary predictions (0 = no rain, 1 = rain)."""
        return pd.Series(self._model.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Returns probability of rain (0.0 to 1.0)."""
        return pd.Series(
            self._model.predict_proba(X)[:, 1], index=X.index
        ).round(2)