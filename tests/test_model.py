"""Tests for src.model and src.predict."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_training_frame
from src.model import build_model, evaluate_model, train_model
from src.predict import predict_next_day


def _ohlcv(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    open_ = close + rng.normal(0, 0.5, size=n)
    high = np.maximum(open_, close) + rng.uniform(0, 1, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, size=n)
    volume = rng.uniform(1_000, 5_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def test_build_model_has_scaler_and_classifier() -> None:
    model = build_model()
    assert "scaler" in model.named_steps
    assert "clf" in model.named_steps


def test_evaluate_model_returns_valid_metrics() -> None:
    frame = build_training_frame(_ohlcv())
    result = evaluate_model(frame, n_splits=4)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.n_splits == 4
    assert 0.0 <= result.baseline_accuracy <= 1.0


def test_train_model_is_fitted() -> None:
    from src.features import feature_columns

    frame = build_training_frame(_ohlcv())
    model = train_model(frame)
    # A fitted pipeline can produce probabilities of the right shape.
    proba = model.predict_proba(frame[feature_columns(frame)].to_numpy())
    assert proba.shape[1] == 2


def test_predict_next_day_shape_and_range() -> None:
    df = _ohlcv()
    frame = build_training_frame(df)
    model = train_model(frame)
    prediction = predict_next_day(model, df)
    assert prediction.direction in {"UP", "DOWN"}
    assert 0.0 <= prediction.probability_up <= 1.0
    assert prediction.last_close == float(df["Close"].iloc[-1])
    assert prediction.label_fa in {"صعودی", "نزولی"}
