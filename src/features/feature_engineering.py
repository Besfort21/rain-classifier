import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Loads weather data and builds features for rain classification."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def load_and_build(self) -> pd.DataFrame:
        """Loads raw data from SQLite and returns a feature DataFrame."""
        df = self._load_data()
        df = self._parse_types(df)
        df = self._sort(df)
        df = self._add_time_features(df)
        df = self._add_lag_features(df)
        df = self._add_target(df)
        df = self._drop_nulls(df)
        logger.info("Feature engineering complete: %d rows, %d columns.", len(df), len(df.columns))
        return df

    def get_feature_columns(self) -> list[str]:
        """Returns the list of feature column names used for training."""
        return [
            "temperature_c",
            "feels_like_c",
            "humidity_pct",
            "wind_speed_kmh",
            "precipitation_mm",
            "weather_code",
            "hour",
            "day_of_week",
            "temp_lag_1h",
            "humidity_lag_1h",
            "precip_lag_1h",
        ]

    def _load_data(self) -> pd.DataFrame:
        """Reads all weather data from SQLite into a DataFrame."""
        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM weather ORDER BY city, timestamp", conn
            )
        logger.info("Loaded %d rows from %s.", len(df), self._db_path)
        return df

    def _parse_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses timestamp column to datetime."""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sorts by city and timestamp — required for lag features."""
        return df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts hour of day and day of week from timestamp."""
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 1-hour lag features grouped by city.

        Lag features give the model information about the previous hour,
        which is a strong signal for weather prediction.
        """
        df["temp_lag_1h"] = df.groupby("city")["temperature_c"].shift(1)
        df["humidity_lag_1h"] = df.groupby("city")["humidity_pct"].shift(1)
        df["precip_lag_1h"] = df.groupby("city")["precipitation_mm"].shift(1)
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates the binary target label: will it rain in the next hour?

        will_rain = 1 if precipitation_mm in the next row > 0, else 0.
        Grouped by city so we don't leak across city boundaries.
        """
        df["will_rain"] = (
            df.groupby("city")["precipitation_mm"]
            .shift(-1)
            .gt(0)
            .astype(int)
        )
        return df

    def _drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops rows with NaN — created by lag and shift operations."""
        before = len(df)
        df = df.dropna(subset=self.get_feature_columns() + ["will_rain"])
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Dropped %d rows with NaN (from lag/shift).", dropped)
        return df.reset_index(drop=True)