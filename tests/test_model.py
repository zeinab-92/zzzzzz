"""Tests for src.model and src.predict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_training_frame, feature_columns
from src.model import (
    build_model,
    evaluate_model,
    select_model,
    train_model,
    tune_threshold,
    walk_forward_accuracy,
)
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


@pytest.mark.parametrize("name", ["gboost", "forest", "logistic"])
def test_build_model_has_scaler_and_classifier(name: str) -> None:
    model = build_model(name)
    assert "scaler" in model.named_steps
    assert "clf" in model.named_steps


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model_name"):
        build_model("not-a-model")


def test_evaluate_model_returns_valid_metrics() -> None:
    frame = build_training_frame(_ohlcv())
    result = evaluate_model(frame, n_splits=4)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.n_splits == 4
    assert 0.0 <= result.baseline_accuracy <= 1.0


def test_tune_threshold_in_range() -> None:
    frame = build_training_frame(_ohlcv(250))
    thr = tune_threshold(frame, n_splits=4)
    assert 0.35 <= thr <= 0.65


def test_evaluate_respects_threshold() -> None:
    frame = build_training_frame(_ohlcv(250))
    result = evaluate_model(frame, n_splits=4, threshold=0.4)
    assert result.threshold == 0.4


def test_walk_forward_accuracy_in_range() -> None:
    frame = build_training_frame(_ohlcv(120))
    acc = walk_forward_accuracy(frame, horizon=15, model_name="logistic")
    assert 0.0 <= acc <= 1.0


def test_select_model_returns_known_model() -> None:
    frame = build_training_frame(_ohlcv(250))
    name, thr, result = select_model(frame, n_splits=4)
    assert name in {"gboost", "forest", "logistic"}
    assert 0.35 <= thr <= 0.65
    assert 0.0 <= result.accuracy <= 1.0


def test_train_model_is_fitted() -> None:
    frame = build_training_frame(_ohlcv())
    model = train_model(frame)
    proba = model.predict_proba(frame[feature_columns(frame)].to_numpy())
    assert proba.shape[1] == 2


def test_predict_next_day_shape_and_range() -> None:
    df = _ohlcv()
    frame = build_training_frame(df)
    model = train_model(frame)
    prediction = predict_next_day(model, df, threshold=0.45)
    assert prediction.direction in {"UP", "DOWN"}
    assert 0.0 <= prediction.probability_up <= 1.0
    assert prediction.last_close == float(df["Close"].iloc[-1])
    assert prediction.label_fa in {"صعودی", "نزولی"}
