"""Train and evaluate a next-day market-direction classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGET_COLUMN, feature_columns


@dataclass
class EvaluationResult:
    """Cross-validated performance metrics."""

    accuracy: float
    f1: float
    n_splits: int
    baseline_accuracy: float


def build_model(random_state: int = 42) -> Pipeline:
    """Create the scaling + gradient-boosting classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(random_state=random_state),
            ),
        ]
    )


def evaluate_model(frame: pd.DataFrame, n_splits: int = 5) -> EvaluationResult:
    """Evaluate the model with a forward-chaining time-series split.

    Args:
        frame: Training frame produced by ``build_training_frame``.
        n_splits: Number of forward-chaining folds.

    Returns:
        An :class:`EvaluationResult` with mean accuracy and F1 across folds.
    """
    features = feature_columns(frame)
    x = frame[features].to_numpy()
    y = frame[TARGET_COLUMN].to_numpy()

    n_splits = max(2, min(n_splits, len(frame) - 1))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    accuracies: list[float] = []
    f1s: list[float] = []
    baselines: list[float] = []
    for train_idx, test_idx in splitter.split(x):
        model = build_model()
        model.fit(x[train_idx], y[train_idx])
        preds = model.predict(x[test_idx])
        accuracies.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, zero_division=0))
        # Majority-class baseline learned from the training fold.
        majority = int(round(y[train_idx].mean()))
        baselines.append(accuracy_score(y[test_idx], np.full_like(preds, majority)))

    return EvaluationResult(
        accuracy=float(np.mean(accuracies)),
        f1=float(np.mean(f1s)),
        n_splits=n_splits,
        baseline_accuracy=float(np.mean(baselines)),
    )


def train_model(frame: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Fit the model on the full training frame."""
    features = feature_columns(frame)
    x = frame[features].to_numpy()
    y = frame[TARGET_COLUMN].to_numpy()
    model = build_model(random_state=random_state)
    model.fit(x, y)
    return model
