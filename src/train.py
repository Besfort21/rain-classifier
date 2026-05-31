import logging
from pathlib import Path

from src.evaluation.evaluator import ModelEvaluator
from src.features.feature_engineering import FeatureEngineer
from src.model.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

DB_PATH = Path("data/weather.db")
MODEL_DIR = Path("models/")
OUTPUT_DIR = Path("outputs/")


def main() -> None:
    logger.info("Starting training pipeline.")

    engineer = FeatureEngineer(DB_PATH)
    df = engineer.load_and_build()
    feature_columns = engineer.get_feature_columns()

    trainer = ModelTrainer(MODEL_DIR)
    X_train, X_test, y_train, y_test = trainer.split(df, feature_columns)
    trainer.train(X_train, y_train)

    evaluator = ModelEvaluator(OUTPUT_DIR)
    evaluator.evaluate(trainer.get_model(), X_test, y_test)
    evaluator.plot_confusion_matrix(trainer.get_model(), X_test, y_test)
    evaluator.plot_feature_importance(trainer.get_model(), feature_columns)

    path = trainer.save()
    logger.info("Training complete. Model saved to %s.", path)


if __name__ == "__main__":
    main()