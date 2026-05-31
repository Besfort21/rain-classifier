from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.model.trainer import ModelTrainer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal feature DataFrame for training tests."""
    return pd.DataFrame({
        "temperature_c":    [15.0, 16.0, 17.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0],
        "feels_like_c":     [13.0, 14.0, 15.0, 12.0, 11.0, 10.0, 9.0,  8.0,  7.0, 6.0],
        "humidity_pct":     [70,   68,   65,   72,   75,   78,   80,   82,   85,  88],
        "wind_speed_kmh":   [10.0, 12.0, 8.0,  9.0,  11.0, 13.0, 7.0,  6.0,  5.0, 4.0],
        "precipitation_mm": [0.0,  0.0,  0.5,  0.0,  0.0,  0.2,  0.0,  0.0,  0.1, 0.0],
        "weather_code":     [2,    1,    0,    3,    2,    1,    0,    3,    2,   1],
        "hour":             [10,   11,   12,   13,   14,   15,   16,   17,   18,  19],
        "day_of_week":      [0,    0,    0,    0,    0,    0,    0,    0,    0,   0],
        "temp_lag_1h":      [14.0, 15.0, 16.0, 17.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0],
        "humidity_lag_1h":  [72,   70,   68,   65,   72,   75,   78,   80,   82,  85],
        "precip_lag_1h":    [0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.2,  0.0,  0.0, 0.1],
        "will_rain":        [0,    1,    0,    1,    0,    1,    0,    1,    0,   1],
    })


@pytest.fixture
def feature_columns() -> list[str]:
    return [
        "temperature_c", "feels_like_c", "humidity_pct", "wind_speed_kmh",
        "precipitation_mm", "weather_code", "hour", "day_of_week",
        "temp_lag_1h", "humidity_lag_1h", "precip_lag_1h",
    ]


@pytest.fixture
def trainer(tmp_path: Path) -> ModelTrainer:
    return ModelTrainer(tmp_path / "models")


class TestSplit:
    def test_split_sizes(self, trainer, sample_df, feature_columns) -> None:
        X_train, X_test, y_train, y_test = trainer.split(sample_df, feature_columns)
        assert len(X_train) == 8
        assert len(X_test) == 2

    def test_split_is_chronological(self, trainer, sample_df, feature_columns) -> None:
        X_train, X_test, _, _ = trainer.split(sample_df, feature_columns)
        assert X_train.index.tolist() == list(range(8))
        assert X_test.index.tolist() == [8, 9]

    def test_no_overlap(self, trainer, sample_df, feature_columns) -> None:
        X_train, X_test, _, _ = trainer.split(sample_df, feature_columns)
        assert len(set(X_train.index) & set(X_test.index)) == 0


class TestTrain:
    def test_train_returns_none(self, trainer, sample_df, feature_columns) -> None:
        X_train, _, y_train, _ = trainer.split(sample_df, feature_columns)
        result = trainer.train(X_train, y_train)
        assert result is None

    def test_model_is_fitted(self, trainer, sample_df, feature_columns) -> None:
        X_train, _, y_train, _ = trainer.split(sample_df, feature_columns)
        trainer.train(X_train, y_train)
        model = trainer.get_model()
        assert hasattr(model, "estimators_")

    def test_get_model_returns_random_forest(self, trainer) -> None:
        assert isinstance(trainer.get_model(), RandomForestClassifier)


class TestSave:
    def test_save_creates_file(self, trainer, sample_df, feature_columns, tmp_path) -> None:
        X_train, _, y_train, _ = trainer.split(sample_df, feature_columns)
        trainer.train(X_train, y_train)
        path = trainer.save()
        assert path.exists()

    def test_save_returns_path(self, trainer, sample_df, feature_columns) -> None:
        X_train, _, y_train, _ = trainer.split(sample_df, feature_columns)
        trainer.train(X_train, y_train)
        path = trainer.save()
        assert isinstance(path, Path)