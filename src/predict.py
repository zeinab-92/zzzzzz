"""Predict the next-day direction of the market index."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from .features import build_features, feature_columns


@dataclass
class Prediction:
    """Next-day direction prediction for the most recent session."""

    direction: str  # "UP" or "DOWN"
    probability_up: float
    last_close: float

    @property
    def label_fa(self) -> str:
        """Persian label for the predicted direction."""
        return "صعودی" if self.direction == "UP" else "نزولی"


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return the engineered feature row for the most recent session."""
    featured = build_features(df)
    features = feature_columns(featured)
    last = featured.iloc[[-1]][features]
    if last.isnull().any(axis=None):
        raise ValueError(
            "The latest row has missing feature values; provide more history."
        )
    return last


def predict_next_day(model: Pipeline, df: pd.DataFrame) -> Prediction:
    """Predict tomorrow's direction from the most recent OHLCV history."""
    last = latest_feature_row(df)
    x = last.to_numpy()

    proba_up = float(model.predict_proba(x)[0][1])
    direction = "UP" if proba_up >= 0.5 else "DOWN"
    last_close = float(df["Close"].iloc[-1])
    return Prediction(
        direction=direction,
        probability_up=proba_up,
        last_close=last_close,
    )
