import logging
from pathlib import Path

from src.features.feature_engineering import FeatureEngineer
from src.model.predictor import RainPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

DB_PATH = Path("data/weather.db")
MODEL_PATH = Path("models/rain_classifier.pkl")


def main() -> None:
    logger.info("Loading latest data for prediction.")

    engineer = FeatureEngineer(DB_PATH)
    df = engineer.load_and_build()
    feature_columns = engineer.get_feature_columns()

    predictor = RainPredictor(MODEL_PATH)

    latest = df.groupby("city").tail(3).copy()
    X_latest = latest[feature_columns]

    latest["predicted_rain"] = predictor.predict(X_latest)
    latest["rain_probability"] = predictor.predict_proba(X_latest)

    print("\n" + "=" * 65)
    print("  Rain predictions — latest 3 hours per city")
    print("=" * 65)

    for city, group in latest.groupby("city"):
        print(f"\n  {city}")
        print(f"  {'Timestamp':<22} {'Temp':>6} {'Humidity':>9} {'Prediction':<12} {'Probability'}")
        print(f"  {'-'*60}")
        for _, row in group.iterrows():
            prediction = "🌧 rain" if row["predicted_rain"] == 1 else "☀ no rain"
            print(
                f"  {str(row['timestamp']):<22} "
                f"{row['temperature_c']:>5.1f}°C "
                f"{row['humidity_pct']:>8}% "
                f"  {prediction:<14} "
                f"  {row['rain_probability']:.0%}"
            )

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()