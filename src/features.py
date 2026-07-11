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


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line and histogram (normalised by close)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line / close, signal_line / close, hist / close


def _stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic oscillator %K in [0, 100]."""
    low_min = df["Low"].rolling(window=period, min_periods=period).min()
    high_max = df["High"].rolling(window=period, min_periods=period).max()
    denom = (high_max - low_min).replace(0.0, np.nan)
    k = 100.0 * (df["Close"] - low_min) / denom
    return k.clip(0.0, 100.0).fillna(50.0)


def _streak(close: pd.Series) -> pd.Series:
    """Signed length of the current consecutive up/down close streak."""
    sign = np.sign(close.diff().fillna(0.0))
    out = np.zeros(len(sign), dtype=float)
    run = 0.0
    prev = 0.0
    for i, s in enumerate(sign.to_numpy()):
        if s == 0:
            run = 0.0
        elif s == prev:
            run += s
        else:
            run = s
        out[i] = run
        prev = s
    return pd.Series(out, index=close.index)


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
    for h in (1, 3, 5, 10, 20):
        out[f"return_{h}"] = close.pct_change(h)

    # Intraday range and body relative to close.
    out["range_pct"] = (out["High"] - out["Low"]) / close
    out["body_pct"] = (out["Close"] - out["Open"]) / close
    out["close_loc"] = (out["Close"] - out["Low"]) / (
        (out["High"] - out["Low"]).replace(0.0, np.nan)
    )
    out["close_loc"] = out["close_loc"].fillna(0.5)

    # Moving averages and their ratio to price.
    for window in (5, 10, 20, 50):
        ma = close.rolling(window=window, min_periods=window).mean()
        out[f"ma_ratio_{window}"] = close / ma - 1.0

    # Distance from recent extremes.
    for window in (10, 20):
        hi = out["High"].rolling(window=window, min_periods=window).max()
        lo = out["Low"].rolling(window=window, min_periods=window).min()
        out[f"dist_high_{window}"] = close / hi - 1.0
        out[f"dist_low_{window}"] = close / lo - 1.0

    # Rolling volatility of daily returns.
    out["volatility_5"] = out["return_1"].rolling(window=5, min_periods=5).std()
    out["volatility_10"] = out["return_1"].rolling(window=10, min_periods=10).std()

    # Volume momentum.
    vol_ma = out["Volume"].rolling(window=5, min_periods=5).mean()
    out["volume_ratio_5"] = out["Volume"] / vol_ma.replace(0.0, np.nan) - 1.0
    vol_ma20 = out["Volume"].rolling(window=20, min_periods=20).mean()
    out["volume_ratio_20"] = out["Volume"] / vol_ma20.replace(0.0, np.nan) - 1.0

    # Momentum oscillators.
    out["rsi_14"] = _rsi(close, period=14)
    out["stoch_k_14"] = _stochastic(out, period=14)

    # Bollinger band position (z-score of close vs 20d mean).
    ma20 = close.rolling(window=20, min_periods=20).mean()
    sd20 = close.rolling(window=20, min_periods=20).std()
    out["bollinger_pos"] = (close - ma20) / sd20.replace(0.0, np.nan)

    # MACD.
    macd_line, macd_signal, macd_hist = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    # Consecutive up/down streak.
    out["streak"] = _streak(close)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of engineered feature column names present in ``df``."""
    candidates = [
        "return_1",
        "return_3",
        "return_5",
        "return_10",
        "return_20",
        "range_pct",
        "body_pct",
        "close_loc",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_20",
        "ma_ratio_50",
        "dist_high_10",
        "dist_low_10",
        "dist_high_20",
        "dist_low_20",
        "volatility_5",
        "volatility_10",
        "volume_ratio_5",
        "volume_ratio_20",
        "rsi_14",
        "stoch_k_14",
        "bollinger_pos",
        "macd",
        "macd_signal",
        "macd_hist",
        "streak",
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
