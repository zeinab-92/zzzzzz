"""Tests for src.data_loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import clean_market_data, load_market_data


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["idx"] * 3,
            "Open": [10.0, 11.0, 12.0],
            "High": [11, 12, 13],
            "Low": [9, 10, 11],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [100.0, 110.0, 120.0],
        }
    )


def test_clean_keeps_required_columns() -> None:
    result = clean_market_data(_valid_frame())
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(result) == 3


def test_clean_missing_column_raises() -> None:
    df = _valid_frame().drop(columns=["Close"])
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_market_data(df)


def test_clean_drops_non_numeric_rows() -> None:
    df = _valid_frame()
    df["Close"] = df["Close"].astype(object)
    df.loc[1, "Close"] = "not-a-number"
    result = clean_market_data(df)
    assert len(result) == 2


def test_clean_all_invalid_raises() -> None:
    df = _valid_frame()
    df["Close"] = "x"
    with pytest.raises(ValueError, match="No valid OHLCV rows"):
        clean_market_data(df)


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_market_data(Path("does-not-exist.xlsx"))


def test_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sample.xlsx"
    _valid_frame().to_excel(path, index=False)
    result = load_market_data(path)
    assert len(result) == 3
    assert result["Close"].dtype == float
