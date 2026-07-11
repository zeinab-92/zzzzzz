"""Predict the next-day direction of the market index."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from .features import build_features, feature_columns


@dataclass
class Prediction:
    """Next-day prediction for the most recent session."""

    direction: str  # "UP" or "DOWN"
    probability_up: float
    last_close: float
    predicted_return_pct: float | None = None

    @property
    def label_fa(self) -> str:
        """Persian label for the predicted direction."""
        return "صعودی" if self.direction == "UP" else "نزولی"

    @property
    def predicted_close(self) -> float | None:
        """Projected next-day close implied by the predicted return."""
        if self.predicted_return_pct is None:
            return None
        return self.last_close * (1.0 + self.predicted_return_pct / 100.0)


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


def predict_next_day(
    model: Pipeline,
    df: pd.DataFrame,
    threshold: float = 0.5,
    return_model: Pipeline | None = None,
) -> Prediction:
    """Predict tomorrow's direction and (optionally) percentage change.

    Args:
        model: A fitted classification pipeline.
        df: Raw OHLCV history (oldest row first).
        threshold: Decision threshold on P(up); values above map to "UP".
        return_model: Optional fitted regression pipeline predicting the
            next-day percentage change of close.
    """
    last = latest_feature_row(df)
    x = last.to_numpy()

    proba_up = float(model.predict_proba(x)[0][1])
    direction = "UP" if proba_up >= threshold else "DOWN"
    last_close = float(df["Close"].iloc[-1])

    predicted_return_pct: float | None = None
    if return_model is not None:
        predicted_return_pct = float(return_model.predict(x)[0])

    return Prediction(
        direction=direction,
        probability_up=proba_up,
        last_close=last_close,
        predicted_return_pct=predicted_return_pct,
    )
