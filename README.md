# Market Direction Prediction

Predict the **next-day direction** (up / down) of a market index from historical
OHLCV data stored in an Excel file.

The sample data (`data/ShKol.xlsx`) contains daily `Open/High/Low/Close/Volume`
rows for the total market index (شاخص کل), ordered oldest-first.

## How it works

1. **`src/data_loader.py`** — reads and validates the Excel OHLCV data.
2. **`src/features.py`** — engineers technical-indicator features (returns over
   multiple horizons, moving-average ratios, rolling volatility, volume
   momentum, RSI) and builds the next-day direction target.
3. **`src/model.py`** — a scikit-learn pipeline (`StandardScaler` +
   `GradientBoostingClassifier`) evaluated with a forward-chaining
   `TimeSeriesSplit` and compared against a majority-class baseline.
4. **`src/predict.py`** — produces tomorrow's direction and the probability of
   an up move from the most recent session.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --input data/ShKol.xlsx
```

Example output:

```
Loaded 4229 trading sessions from data/ShKol.xlsx
Built training frame with 4160 labelled samples

=== Time-series cross-validation ===
Folds:              5
Accuracy:           0.6139
F1 (up class):      0.6778
Baseline accuracy:  0.5711

=== Next-day prediction ===
Last close:         5,311,523.7
Direction:          UP (صعودی)
P(up):              0.7234
```

### Using your own file

Any `.xlsx` file with `Open`, `High`, `Low`, `Close`, `Volume` columns (rows
ordered oldest-first) works:

```bash
python main.py --input path/to/your.xlsx --sheet 0 --splits 5
```

## Tests

```bash
python -m pytest -q
```

## Disclaimer

This is a statistical model for educational purposes only. Market movements are
noisy and past performance does not guarantee future results — do **not** use
this as financial advice.
