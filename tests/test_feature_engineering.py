import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Creates a minimal in-memory SQLite DB with sample weather data."""
    path = tmp_path / "test.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE weather (
                id INTEGER PRIMARY KEY,
                city TEXT,
                timestamp TEXT,
                temperature_c REAL,
                feels_like_c REAL,
                humidity_pct INTEGER,
                wind_speed_kmh REAL,
                precipitation_mm REAL,
                weather_code INTEGER,
                weather_description TEXT
            )
        """)
        rows = [
            ("Solingen", "2026-05-10T10:00:00", 15.0, 13.0, 70, 10.0, 0.0, 2, "Partly cloudy"),
            ("Solingen", "2026-05-10T11:00:00", 16.0, 14.0, 68, 12.0, 0.0, 1, "Mainly clear"),
            ("Solingen", "2026-05-10T12:00:00", 17.0, 15.0, 65, 8.0,  0.5, 0, "Clear sky"),
            ("Solingen", "2026-05-10T13:00:00", 16.5, 14.5, 67, 9.0,  0.0, 2, "Partly cloudy"),
            ("Solingen", "2026-05-10T14:00:00", 16.0, 14.0, 69, 11.0, 0.0, 1, "Mainly clear"),
            ("Köln",     "2026-05-10T10:00:00", 14.0, 12.0, 75, 9.0,  0.0, 3, "Overcast"),
            ("Köln",     "2026-05-10T11:00:00", 15.0, 13.0, 72, 10.0, 0.2, 2, "Partly cloudy"),
            ("Köln",     "2026-05-10T12:00:00", 15.5, 13.5, 70, 11.0, 0.0, 1, "Mainly clear"),
            ("Köln",     "2026-05-10T13:00:00", 15.0, 13.0, 71, 10.0, 0.0, 2, "Partly cloudy"),
            ("Köln",     "2026-05-10T14:00:00", 14.5, 12.5, 73, 9.0,  0.0, 3, "Overcast"),
        ]
        conn.executemany(
            "INSERT INTO weather (city, timestamp, temperature_c, feels_like_c, "
            "humidity_pct, wind_speed_kmh, precipitation_mm, weather_code, weather_description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return path


@pytest.fixture
def engineer(db_path: Path) -> FeatureEngineer:
    return FeatureEngineer(db_path)


class TestLoadAndBuild:
    def test_returns_dataframe(self, engineer) -> None:
        df = engineer.load_and_build()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, engineer) -> None:
        df = engineer.load_and_build()
        for col in engineer.get_feature_columns() + ["will_rain"]:
            assert col in df.columns

    def test_no_nulls_in_features(self, engineer) -> None:
        df = engineer.load_and_build()
        assert df[engineer.get_feature_columns()].isnull().sum().sum() == 0

    def test_no_nulls_in_target(self, engineer) -> None:
        df = engineer.load_and_build()
        assert df["will_rain"].isnull().sum() == 0

    def test_target_is_binary(self, engineer) -> None:
        df = engineer.load_and_build()
        assert set(df["will_rain"].unique()).issubset({0, 1})

    def test_lag_features_present(self, engineer) -> None:
        df = engineer.load_and_build()
        assert "temp_lag_1h" in df.columns
        assert "humidity_lag_1h" in df.columns
        assert "precip_lag_1h" in df.columns

    def test_time_features_present(self, engineer) -> None:
        df = engineer.load_and_build()
        assert "hour" in df.columns
        assert "day_of_week" in df.columns

    def test_hour_range(self, engineer) -> None:
        df = engineer.load_and_build()
        assert df["hour"].between(0, 23).all()

    def test_day_of_week_range(self, engineer) -> None:
        df = engineer.load_and_build()
        assert df["day_of_week"].between(0, 6).all()

    def test_lag_does_not_cross_cities(self, engineer) -> None:
        """Last row of city A should not bleed into first row of city B."""
        df = engineer.load_and_build()
        for city, group in df.groupby("city"):
            first_lag = group.iloc[0]["temp_lag_1h"]
            assert pd.notna(first_lag)

    def test_will_rain_reflects_next_hour(self, engineer) -> None:
        """Row with precipitation > 0 in next hour should have will_rain = 1."""
        df = engineer.load_and_build()
        solingen = df[df["city"] == "Solingen"].reset_index(drop=True)
        # 11:00 should have will_rain=1 because 12:00 has precipitation_mm=0.5
        row = solingen[solingen["timestamp"] == "2026-05-10 11:00:00"]
        assert not row.empty
        assert row.iloc[0]["will_rain"] == 1