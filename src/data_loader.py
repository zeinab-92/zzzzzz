"""Load and validate OHLCV market data from an Excel file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_market_data(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read an Excel file with OHLCV columns and return a clean DataFrame.

    The rows are assumed to be ordered chronologically (oldest first). The
    returned frame is indexed by a synthetic ``step`` column so that the last
    row always corresponds to the most recent trading session.

    Args:
        path: Path to the ``.xlsx`` file.
        sheet_name: Sheet index or name to read.

    Returns:
        A DataFrame containing the required OHLCV columns as floats.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required columns are missing or no rows remain.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    return clean_market_data(df)


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, coerce numeric types and drop invalid rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    clean = df.loc[:, REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if clean.empty:
        raise ValueError("No valid OHLCV rows after cleaning the data.")

    return clean
