# Market Direction Prediction

Predict the **next-day direction** (up / down) of a market index from historical
OHLCV data stored in an Excel file.

The sample data (`data/ShKol.xlsx`) contains daily `Open/High/Low/Close/Volume`
rows for the total market index (شاخص کل), ordered oldest-first.

## How it works

1. **`src/data_loader.py`** — reads and validates the Excel OHLCV data.
2. **`src/features.py`** — engineers technical-indicator features (returns over
   multiple horizons, moving-average ratios, distance from recent highs/lows,
   rolling volatility, volume momentum, RSI, Stochastic %K, Bollinger position,
   MACD, and up/down streaks) and builds the next-day direction target.
3. **`src/model.py`** — scikit-learn pipelines (`StandardScaler` +
   `GradientBoostingClassifier` / `RandomForestClassifier` / `LogisticRegression`).
   `select_model` picks the best classifier by forward-chaining `TimeSeriesSplit`
   accuracy, `tune_threshold` tunes the P(up) decision threshold to counter the
   upward-drift bias of a long-only index, and `walk_forward_accuracy` runs an
   expanding-window backtest. All are compared against a majority-class baseline.
4. **`src/predict.py`** — produces tomorrow's direction, the probability of an
   up move (using the tuned threshold) and, via an optional
   `GradientBoostingRegressor` (`train_return_model`), the predicted next-day
   **percentage change** and projected close.

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
Built training frame with 4129 labelled samples

Selected model:      logistic (threshold=0.43)

=== Time-series cross-validation ===
Folds:              5
Accuracy:           0.6573
F1 (up class):      0.6935
Baseline accuracy:  0.5727
Decision threshold: 0.43

Walk-forward accuracy (last 250d): 0.6960

=== Next-day prediction ===
Last close:         5,311,523.7
Direction:          UP (صعودی)
P(up):              0.6809
Predicted change:   +1.30%
Projected close:    5,380,536.4
```

Choose a specific classifier or backtest horizon with `--model` and
`--backtest`:

```bash
python main.py --model forest --backtest 500
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
