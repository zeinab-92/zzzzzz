"""Technical-indicator feature engineering for market-direction prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Column produced as the classification target: 1 if next close > current close.
TARGET_COLUMN = "target_up"


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical-indicator feature columns to an OHLCV frame.

    Args:
        df: DataFrame with Open/High/Low/Close/Volume, oldest row first.

    Returns:
        A copy of ``df`` with additional feature columns.
    """
    out = df.copy()
    close = out["Close"]

    # Returns over several horizons.
    out["return_1"] = close.pct_change(1)
    out["return_3"] = close.pct_change(3)
    out["return_5"] = close.pct_change(5)

    # Intraday range and body relative to close.
    out["range_pct"] = (out["High"] - out["Low"]) / close
    out["body_pct"] = (out["Close"] - out["Open"]) / close

    # Moving averages and their ratio to price.
    for window in (5, 10, 20):
        ma = close.rolling(window=window, min_periods=window).mean()
        out[f"ma_ratio_{window}"] = close / ma - 1.0

    # Rolling volatility of daily returns.
    out["volatility_5"] = out["return_1"].rolling(window=5, min_periods=5).std()
    out["volatility_10"] = out["return_1"].rolling(window=10, min_periods=10).std()

    # Volume momentum.
    vol_ma = out["Volume"].rolling(window=5, min_periods=5).mean()
    out["volume_ratio_5"] = out["Volume"] / vol_ma.replace(0.0, np.nan) - 1.0

    # Momentum oscillator.
    out["rsi_14"] = _rsi(close, period=14)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of engineered feature column names present in ``df``."""
    candidates = [
        "return_1",
        "return_3",
        "return_5",
        "range_pct",
        "body_pct",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_20",
        "volatility_5",
        "volatility_10",
        "volume_ratio_5",
        "rsi_14",
    ]
    return [c for c in candidates if c in df.columns]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the next-day direction target (1 = up, 0 = down/flat)."""
    out = df.copy()
    next_close = out["Close"].shift(-1)
    target = (next_close > out["Close"]).astype("Int64")
    # Rows without a known next-day close have an undefined target.
    target[next_close.isna()] = pd.NA
    out[TARGET_COLUMN] = target
    return out


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build features + target and drop rows with missing values.

    The final row (which has no known next-day close) is intentionally kept out
    of the training frame; use :func:`build_features` on the raw data to obtain
    the latest feature row for inference.
    """
    featured = build_features(df)
    with_target = add_target(featured)
    cols = feature_columns(with_target) + [TARGET_COLUMN]
    trimmed = with_target.dropna(subset=cols).reset_index(drop=True)
    trimmed[TARGET_COLUMN] = trimmed[TARGET_COLUMN].astype(int)
    return trimmed
