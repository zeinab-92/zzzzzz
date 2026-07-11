"""Tests for src.features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    TARGET_COLUMN,
    add_target,
    build_features,
    build_training_frame,
    feature_columns,
)


def _ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    open_ = close + rng.normal(0, 0.5, size=n)
    high = np.maximum(open_, close) + rng.uniform(0, 1, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, size=n)
    volume = rng.uniform(1_000, 5_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def test_build_features_adds_columns() -> None:
    result = build_features(_ohlcv())
    for col in feature_columns(result):
        assert col in result.columns
    assert "rsi_14" in result.columns


def test_rsi_within_bounds() -> None:
    result = build_features(_ohlcv())
    assert result["rsi_14"].between(0, 100).all()


def test_add_target_is_binary_and_shifted() -> None:
    df = pd.DataFrame(
        {
            "Open": [1, 1, 1, 1],
            "High": [1, 1, 1, 1],
            "Low": [1, 1, 1, 1],
            "Close": [10.0, 12.0, 11.0, 15.0],
            "Volume": [1, 1, 1, 1],
        }
    )
    result = add_target(df)
    # up, down, up, unknown(last -> NA)
    assert result[TARGET_COLUMN].tolist()[:3] == [1, 0, 1]
    assert pd.isna(result[TARGET_COLUMN].iloc[-1])


def test_build_training_frame_has_no_nulls_and_int_target() -> None:
    frame = build_training_frame(_ohlcv(80))
    cols = feature_columns(frame) + [TARGET_COLUMN]
    assert not frame[cols].isnull().any(axis=None)
    assert frame[TARGET_COLUMN].dtype == int
    assert set(frame[TARGET_COLUMN].unique()).issubset({0, 1})


def test_feature_columns_subset_of_frame() -> None:
    frame = build_training_frame(_ohlcv(80))
    assert set(feature_columns(frame)).issubset(set(frame.columns))
