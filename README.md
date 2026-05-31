# 🌧️ Rain Prediction Classifier

[![CI](https://github.com/Besfort21/rain-classifier/actions/workflows/ci.yml/badge.svg)]
![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)

A machine learning pipeline that predicts whether it will rain in the next hour,
trained on real weather data collected by the
[Weather ETL Pipeline](https://github.com/Besfort21/weather-etl).

---

## How it works

1. **Data** — loads hourly weather data (temperature, humidity, wind, precipitation)
   from the Weather ETL Pipeline's SQLite database
2. **Features** — engineers lag features (previous hour's values) and time features
   (hour of day, day of week) to give the model temporal context
3. **Training** — trains a Random Forest classifier with `class_weight="balanced"`
   to handle the natural imbalance between rainy and dry hours
4. **Evaluation** — compares against a naive baseline and reports precision, recall,
   F1 and a confusion matrix
5. **Prediction** — loads the saved model and predicts rain probability for the
   latest hours in the database

---

## Results

| Metric | Model | Baseline (always "no rain") |
|---|---|---|
| Accuracy | 92.7% | 90.1% |
| Precision (rain) | 75% | — |
| Recall (rain) | 40% | 0% |
| F1 (rain) | 0.52 | 0.00 |

The low recall reflects limited training data — the model improves as more hourly
data is collected by the ETL pipeline.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy |
| ML | scikit-learn (Random Forest) |
| Model persistence | joblib |
| Visualisation | matplotlib, seaborn |
| Exploration | Jupyter Notebook |
| Testing | pytest + pytest-cov |
| Linting | ruff |
| CI | GitHub Actions |

---

## Getting started

```bash
git clone https://github.com/USERNAME/rain-classifier.git
cd rain-classifier

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate.bat

pip install -r requirements.txt
```

Copy the SQLite database from the Weather ETL Pipeline:

```bash
cp ../weather-etl/data/weather.db data/weather.db
```

---

## Usage

Train the model:

```bash
python -m src.train
```

Predict rain for the latest hours:

```bash
python -m src.predict
```

---

## Running tests

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

---

## Design decisions

**Chronological train/test split** — the test set is the last 20% of data by time,
not a random sample. Shuffling time-series data leaks future information into
training and produces artificially high scores.

**Lag features** — the previous hour's temperature, humidity and precipitation are
strong predictors for the next hour's weather. Lag features are computed per city
to avoid bleeding values across city boundaries.

**`class_weight="balanced"`** — rainy hours make up only ~14% of the data. Without
balancing, the model would learn to always predict "no rain" and still achieve high
accuracy. Balanced weights force it to learn from the minority class.

**Separate feature engineering module** — `FeatureEngineer` is independent of the
model. The same feature logic is used for training and prediction, with a single
`get_feature_columns()` method as the source of truth.